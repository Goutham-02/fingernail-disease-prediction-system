import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import cv2
import json
import os
import argparse
from PIL import Image
from torchvision import transforms
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import math


# ===================== CONFIGURATION =====================
class Config:
    """Centralized configuration"""
    MODEL_PATH = "17classes_resnet_97.pth"
    DISEASE_DATA_PATH = "data1.json"
    OUTPUT_DIR = "output"
    GRADCAM_DIR = "output/gradcam"
    IMAGE_SIZE = (192, 192)
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]


# ===================== UTILITIES =====================
def normalize_key(k: str) -> str:
    """Normalize keys to canonical form"""
    if not isinstance(k, str):
        return k
    k = k.strip().lower().replace(" ", "_").replace("-", "_")
    k = k.replace("'", "").replace(".", "")
    k = k.replace("__", "_")
    return k


def ensure_dir(path: str):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)


def convert_to_serializable(obj):
    """Convert numpy/torch types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, 'item'):  # torch tensors
        return obj.item()
    return obj


# ===================== FEATURE EXTRACTION MODULE =====================
class GenericFeatureExtractor:
    """
    Extracts generic nail features that can be mapped to multiple diseases:
    - Color analysis (white, yellow, blue, pale, red)
    - Texture patterns (pitting, ridges, roughness)
    - Structural changes (lines, bands, separation)
    - Shape abnormalities (clubbing, spooning)
    """
    
    def __init__(self):
        self.feature_detectors = {
            'nail_pitting': self._detect_pitting,
            'horizontal_lines': self._detect_horizontal_lines,
            'vertical_lines': self._detect_vertical_lines,
            'white_discoloration': self._detect_white_color,
            'yellow_discoloration': self._detect_yellow_color,
            'blue_discoloration': self._detect_blue_color,
            'pale_appearance': self._detect_pale,
            'red_areas': self._detect_red,
            'nail_thickening': self._detect_thickening,
            'nail_separation': self._detect_separation,
            'surface_roughness': self._detect_roughness,
            'nail_clubbing': self._detect_clubbing,
            'spoon_shape': self._detect_spooning,
            'v_shaped_notch': self._detect_v_notch
        }
    
    def extract_features(self, image: np.ndarray, heatmap: np.ndarray = None) -> Dict[str, Any]:
        """
        Extract all generic features from nail image
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            heatmap: Optional attention heatmap to focus on relevant regions
            
        Returns:
            Dictionary of feature names to detection results
        """
        if image is None or image.size == 0:
            return {}
        
        # Apply heatmap mask if provided
        if heatmap is not None:
            roi_mask = self._create_roi_mask(image, heatmap)
        else:
            roi_mask = np.ones(image.shape[:2], dtype=np.uint8)
        
        features = {}
        for feature_name, detector_func in self.feature_detectors.items():
            try:
                features[feature_name] = detector_func(image, roi_mask)
            except Exception as e:
                features[feature_name] = {"present": False, "error": str(e)}
        
        return features
    
    def _create_roi_mask(self, image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        """Create ROI mask from heatmap"""
        if heatmap.shape != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        threshold = np.percentile(heatmap, 60)
        return (heatmap > threshold).astype(np.uint8)
    
    def _get_masked_region(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get masked image and gray version"""
        if mask.ndim == 3:
            mask = mask.squeeze()
        masked = image * mask[:, :, np.newaxis]
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray_masked = gray * mask
        return masked, gray_masked
    
    # ===== COLOR DETECTORS =====
    def _detect_white_color(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Detect white discoloration"""
        masked, _ = self._get_masked_region(image, mask)
        hsv = cv2.cvtColor(masked.astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        # White: low saturation, high value
        white_mask = (hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 200) & (mask > 0)
        white_frac = np.sum(white_mask) / (np.sum(mask) + 1e-8)
        
        return {
            "present": white_frac > 0.15,
            "white_frac": float(white_frac),
            "score": float(min(1.0, white_frac * 3))
        }
    
    def _detect_yellow_color(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Detect yellow discoloration"""
        masked, _ = self._get_masked_region(image, mask)
        hsv = cv2.cvtColor(masked.astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        # Yellow: hue 15-35, moderate saturation
        yellow_mask = (hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 35) & \
                      (hsv[:, :, 1] > 50) & (mask > 0)
        yellow_frac = np.sum(yellow_mask) / (np.sum(mask) + 1e-8)
        
        return {
            "present": yellow_frac > 0.1,
            "yellow_frac": float(yellow_frac),
            "score": float(min(1.0, yellow_frac * 5))
        }
    
    def _detect_blue_color(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Detect bluish discoloration"""
        masked, _ = self._get_masked_region(image, mask)
        hsv = cv2.cvtColor(masked.astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        # Blue: hue 90-130
        blue_mask = (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 130) & \
                    (hsv[:, :, 1] > 40) & (mask > 0)
        blue_frac = np.sum(blue_mask) / (np.sum(mask) + 1e-8)
        
        return {
            "present": blue_frac > 0.08,
            "blue_frac": float(blue_frac),
            "score": float(min(1.0, blue_frac * 6))
        }
    
    def _detect_pale(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Detect pale appearance"""
        masked, _ = self._get_masked_region(image, mask)
        hsv = cv2.cvtColor(masked.astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        valid = mask > 0
        if np.sum(valid) == 0:
            return {"present": False, "score": 0.0}
        
        mean_val = np.mean(hsv[valid, 2])
        mean_sat = np.mean(hsv[valid, 1])
        
        # Pale: high value, very low saturation
        is_pale = mean_val > 180 and mean_sat < 40
        
        return {
            "present": is_pale,
            "mean_value": float(mean_val),
            "mean_saturation": float(mean_sat),
            "score": float(1.0 if is_pale else 0.0)
        }
    
    def _detect_red(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Detect reddish areas"""
        masked, _ = self._get_masked_region(image, mask)
        hsv = cv2.cvtColor(masked.astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        # Red: hue 0-10 or 170-180
        red_mask = ((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170)) & \
                   (hsv[:, :, 1] > 60) & (mask > 0)
        red_frac = np.sum(red_mask) / (np.sum(mask) + 1e-8)
        
        return {
            "present": red_frac > 0.05,
            "red_frac": float(red_frac),
            "score": float(min(1.0, red_frac * 8))
        }
    
    # ===== TEXTURE & PATTERN DETECTORS =====
    def _detect_pitting(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Detect nail pitting (small dents)"""
        _, gray_masked = self._get_masked_region(image, mask)
        
        if gray_masked.max() < 10:
            return {"present": False, "count": 0, "score": 0.0}
        
        # Use blob detection for pits
        blur = cv2.GaussianBlur(gray_masked, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find small dark regions
        contours, _ = cv2.findContours(255 - thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pit_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 5 < area < 200:  # Small pits
                pit_count += 1
        
        return {
            "present": pit_count > 3,
            "count": int(pit_count),
            "score": float(min(1.0, pit_count / 10.0))
        }
    
    def _detect_horizontal_lines(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Detect horizontal lines/ridges"""
        _, gray_masked = self._get_masked_region(image, mask)
        
        if gray_masked.max() < 10:
            return {"present": False, "count": 0, "score": 0.0}
        
        edges = cv2.Canny(gray_masked, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15, minLineLength=20, maxLineGap=5)
        
        horizontal_count = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle < 20 or angle > 160:  # Horizontal
                    horizontal_count += 1
        
        return {
            "present": horizontal_count > 1,
            "count": int(horizontal_count),
            "score": float(min(1.0, horizontal_count / 5.0))
        }
    
    def _detect_vertical_lines(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Detect vertical lines/ridges"""
        _, gray_masked = self._get_masked_region(image, mask)
        
        if gray_masked.max() < 10:
            return {"present": False, "count": 0, "score": 0.0}
        
        edges = cv2.Canny(gray_masked, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15, minLineLength=20, maxLineGap=5)
        
        vertical_count = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if 70 < angle < 110:  # Vertical
                    vertical_count += 1
        
        return {
            "present": vertical_count > 1,
            "count": int(vertical_count),
            "score": float(min(1.0, vertical_count / 5.0))
        }
    
    def _detect_roughness(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Detect surface roughness"""
        _, gray_masked = self._get_masked_region(image, mask)
        
        if gray_masked.max() < 10:
            return {"present": False, "score": 0.0}
        
        # Compute texture variance
        valid_pixels = gray_masked[mask > 0]
        if len(valid_pixels) < 100:
            return {"present": False, "score": 0.0}
        
        # Sobel for texture
        sobelx = cv2.Sobel(gray_masked, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_masked, cv2.CV_64F, 0, 1, ksize=3)
        
        texture_strength = np.sqrt(sobelx**2 + sobely**2)
        texture_mean = np.mean(texture_strength[mask > 0])
        
        is_rough = texture_mean > 15
        
        return {
            "present": is_rough,
            "texture_strength": float(texture_mean),
            "score": float(min(1.0, texture_mean / 30.0))
        }
    
    # ===== STRUCTURAL DETECTORS =====
    def _detect_thickening(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Detect nail thickening (proxy via edge thickness)"""
        _, gray_masked = self._get_masked_region(image, mask)
        
        if gray_masked.max() < 10:
            return {"present": False, "score": 0.0}
        
        edges = cv2.Canny(gray_masked, 50, 150)
        edge_density = np.sum(edges > 0) / (np.sum(mask) + 1e-8)
        
        return {
            "present": edge_density > 0.1,
            "edge_density": float(edge_density),
            "score": float(min(1.0, edge_density * 5))
        }
    
    def _detect_separation(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Detect nail separation (onycholysis)"""
        _, gray_masked = self._get_masked_region(image, mask)
        
        if gray_masked.max() < 10:
            return {"present": False, "score": 0.0}
        
        # Look for gaps (very bright or dark regions)
        _, thresh = cv2.threshold(gray_masked, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {"present": False, "score": 0.0}
        
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        
        solidity = area / (hull_area + 1e-8)
        
        return {
            "present": solidity < 0.75,
            "solidity": float(solidity),
            "score": float(max(0.0, 1.0 - solidity))
        }
    
    def _detect_clubbing(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Detect nail clubbing (curved, bulbous)"""
        _, gray_masked = self._get_masked_region(image, mask)
        
        if gray_masked.max() < 10:
            return {"present": False, "score": 0.0}
        
        # Detect curvature via contour analysis
        _, thresh = cv2.threshold(gray_masked, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours or len(contours[0]) < 5:
            return {"present": False, "score": 0.0}
        
        cnt = max(contours, key=cv2.contourArea)
        
        # Fit ellipse to check curvature
        if len(cnt) >= 5:
            try:
                ellipse = cv2.fitEllipse(cnt)
                _, (MA, ma), _ = ellipse
                ratio = MA / (ma + 1e-8)
                
                is_clubbed = ratio > 1.3
                
                return {
                    "present": is_clubbed,
                    "aspect_ratio": float(ratio),
                    "score": float(min(1.0, max(0.0, ratio - 1.0)))
                }
            except:
                pass
        
        return {"present": False, "score": 0.0}
    
    def _detect_spooning(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Detect spoon-shaped nails (koilonychia)"""
        _, gray_masked = self._get_masked_region(image, mask)
        
        if gray_masked.max() < 10:
            return {"present": False, "score": 0.0}
        
        # Check for concave center (darker in middle)
        h, w = gray_masked.shape
        center_region = gray_masked[h//3:2*h//3, w//3:2*w//3]
        edge_region = gray_masked.copy()
        edge_region[h//3:2*h//3, w//3:2*w//3] = 0
        
        if center_region.size == 0 or edge_region.max() == 0:
            return {"present": False, "score": 0.0}
        
        center_mean = np.mean(center_region[center_region > 0]) if np.sum(center_region > 0) > 0 else 0
        edge_mean = np.mean(edge_region[edge_region > 0]) if np.sum(edge_region > 0) > 0 else 0
        
        center_depth = edge_mean - center_mean
        
        return {
            "present": center_depth > 20,
            "center_depth": float(center_depth),
            "score": float(min(1.0, center_depth / 40.0))
        }
    
    def _detect_v_notch(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Detect V-shaped notches at nail edge"""
        _, gray_masked = self._get_masked_region(image, mask)
        
        if gray_masked.max() < 10:
            return {"present": False, "score": 0.0}
        
        edges = cv2.Canny(gray_masked, 30, 100)
        
        # Look for V-patterns using Hough lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=10, maxLineGap=3)
        
        v_count = 0
        if lines is not None and len(lines) > 3:
            # Simple heuristic: converging lines
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
            
            # Check for pairs of opposite angles (V-shape)
            angles = sorted(angles)
            for i in range(len(angles) - 1):
                if abs(angles[i] - angles[i+1]) > 30:
                    v_count += 1
        
        return {
            "present": v_count > 1,
            "v_strength": int(v_count),
            "score": float(min(1.0, v_count / 3.0))
        }


# ===================== SYMPTOM PARSER =====================
class SymptomParser:
    """Parse user symptoms and match against disease database"""
    
    def __init__(self, disease_data: List[Dict]):
        self.disease_data = disease_data
        self.symptom_vocabulary = self._build_vocabulary()
    
    def _build_vocabulary(self) -> Dict[str, List[str]]:
        """Build symptom vocabulary from disease data"""
        vocab = {}
        for disease in self.disease_data:
            disease_name = normalize_key(disease.get("name", ""))
            symptoms = disease.get("associated_symptoms", [])
            vocab[disease_name] = [normalize_key(s) for s in symptoms]
        return vocab
    
    def parse_symptoms(self, symptom_text: str) -> Dict[str, Any]:
        """
        Parse comma-separated symptom text
        
        Returns:
            Dict with normalized symptom flags
        """
        if not symptom_text or not symptom_text.strip():
            return {}
        
        user_symptoms = [s.strip().lower() for s in symptom_text.split(",")]
        user_symptoms_norm = [normalize_key(s) for s in user_symptoms if s]
        
        # Create boolean flags
        symptom_flags = {}
        for sym in user_symptoms_norm:
            symptom_flags[sym] = True
        
        return symptom_flags
    
    def match_symptoms(self, user_symptoms: Dict[str, bool], disease_name: str) -> float:
        """
        Calculate symptom match score for a disease
        
        Returns:
            Score between 0 and 1
        """
        if not user_symptoms:
            return 0.0
        
        disease_norm = normalize_key(disease_name)
        known_symptoms = self.symptom_vocabulary.get(disease_norm, [])
        
        if not known_symptoms:
            return 0.0
        
        # Count matches
        matches = 0
        for user_sym in user_symptoms.keys():
            for known_sym in known_symptoms:
                # Fuzzy match: check if either contains the other
                if user_sym in known_sym or known_sym in user_sym:
                    matches += 1
                    break
        
        # Score: matches / known_symptoms (but cap at 1.0)
        score = matches / len(known_symptoms)
        return float(min(1.0, score))


# ===================== GRAD-CAM =====================
class GradCAM:
    """Gradient-weighted Class Activation Mapping"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activations)
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self.target_layer.register_full_backward_hook(self.save_gradients)
        else:
            self.target_layer.register_backward_hook(self.save_gradients)
    
    def save_activations(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(self, input_tensor):
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probabilities, dim=1)
        confidence, pred_idx = confidence.item(), pred_idx.item()
        
        score = output[:, pred_idx]
        score.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            return None, pred_idx, confidence
        
        # HiResCAM approach: Element-wise multiplication for better spatial preservation
        weighted_activations = self.gradients * self.activations
        
        # Sum over channels
        heatmap = torch.sum(weighted_activations, dim=1, keepdim=True)
        
        heatmap = F.relu(heatmap)
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Sharpening: Raise to power to suppress background noise and focus on peaks
        heatmap = torch.pow(heatmap, 3)
        
        return heatmap.squeeze(0), pred_idx, confidence


# ===================== FUSION ENGINE =====================
class InferenceFusion:
    """Combines model predictions, visual features, and symptoms"""
    
    def __init__(self, disease_data: List[Dict]):
        self.disease_data = {normalize_key(d["name"]): d for d in disease_data}
    
    def compute_visual_score(self, disease_name: str, features: Dict[str, Any]) -> float:
        """Compute visual evidence score for a disease"""
        disease_norm = normalize_key(disease_name)
        disease_info = self.disease_data.get(disease_norm, {})
        
        # Get expected visual features for this disease
        visual_features = disease_info.get("visual_features", [])
        if not visual_features:
            return 0.0
        
        # Normalize expected features
        expected = [normalize_key(vf) for vf in visual_features]
        
        # Score based on detected features
        scores = []
        for exp_feature in expected:
            # Try to find matching detected feature
            best_match_score = 0.0
            for det_feature, det_value in features.items():
                det_norm = normalize_key(det_feature)
                
                # Check if features are related
                if exp_feature in det_norm or det_norm in exp_feature:
                    # Extract numeric score
                    score = self._extract_score(det_value)
                    best_match_score = max(best_match_score, score)
            
            scores.append(best_match_score)
        
        if not scores:
            return 0.0
        
        # Combine: 70% max, 30% average
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        combined = 0.7 * max_score + 0.3 * avg_score
        
        return float(min(1.0, max(0.0, combined)))
    
    def _extract_score(self, value: Any) -> float:
        """Extract numeric score from detector output"""
        if value is None:
            return 0.0
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return float(min(1.0, max(0.0, value)))
        if isinstance(value, dict):
            if "score" in value:
                return float(min(1.0, max(0.0, value["score"])))
            if "present" in value:
                return 1.0 if value["present"] else 0.0
        return 0.0
    
    def combine_scores(self, model_prob: float, visual_score: float, 
                      symptom_score: float, disease_name: str) -> float:
        """Weighted combination of all scores"""
        disease_norm = normalize_key(disease_name)
        disease_info = self.disease_data.get(disease_norm, {})
        
        weights = disease_info.get("weights", {
            "model_prob": 0.5,
            "visual": 0.35,
            "symptom": 0.15
        })
        
        w_model = weights.get("model_prob", 0.5)
        w_visual = weights.get("visual", 0.35)
        w_symptom = weights.get("symptom", 0.15)
        
        # Normalize weights
        total = w_model + w_visual + w_symptom
        if total > 0:
            w_model /= total
            w_visual /= total
            w_symptom /= total
        
        combined = w_model * model_prob + w_visual * visual_score + w_symptom * symptom_score
        return float(min(1.0, max(0.0, combined)))
    
    def rank_predictions(self, model_probs: np.ndarray, class_names: List[str],
                        features: Dict[str, Any], symptom_scores: Dict[str, float]) -> List[Dict]:
        """
        Rank all disease predictions
        
        Returns:
            List of dicts with class, scores, and breakdown
        """
        results = []
        
        for idx, class_name in enumerate(class_names):
            model_prob = float(model_probs[idx])
            visual_score = self.compute_visual_score(class_name, features)
            symptom_score = symptom_scores.get(normalize_key(class_name), 0.0)
            
            combined_score = self.combine_scores(model_prob, visual_score, 
                                                 symptom_score, class_name)
            
            results.append({
                "disease": class_name,
                "combined_score": combined_score,
                "model_probability": model_prob,
                "visual_score": visual_score,
                "symptom_score": symptom_score
            })
        
        # Sort by combined score
        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results


# ===================== IMAGE PROCESSING =====================
def denormalize_image(tensor):
    """Denormalize image tensor for visualization"""
    mean = torch.tensor(Config.NORMALIZE_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(Config.NORMALIZE_STD, device=tensor.device).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


def apply_heatmap_overlay(img_tensor, heatmap):
    """Apply Grad-CAM heatmap overlay to image"""
    img = denormalize_image(img_tensor).cpu().permute(1, 2, 0).numpy()
    img = np.uint8(img * 255)
    
    heatmap_np = heatmap.cpu().numpy().squeeze()
    heatmap_resized = cv2.resize(heatmap_np, (img.shape[1], img.shape[0]))
    
    # Edge-Guided Refinement:
    # Multiply heatmap by image gradient to focus on structural details (lines, ridges)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize gradient
    if grad_mag.max() > 0:
        grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)
    
    # Combine: Heatmap * (0.3 + 0.7 * Gradient)
    # This keeps some base heatmap but heavily emphasizes edges within the hot region
    refined_heatmap = heatmap_resized * (0.3 + 0.7 * grad_mag)
    
    # Re-normalize
    if refined_heatmap.max() > 0:
        refined_heatmap = (refined_heatmap - refined_heatmap.min()) / (refined_heatmap.max() - refined_heatmap.min() + 1e-8)
    
    # Apply colormap to refined heatmap
    heatmap_color = cv2.applyColorMap(np.uint8(255 * refined_heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Smart Overlay with higher threshold
    threshold = 0.25
    mask = refined_heatmap > threshold
    
    superimposed = img.copy()
    
    if mask.any():
        # Create a blended version
        blended = cv2.addWeighted(img, 0.4, heatmap_color, 0.6, 0)
        
        mask_3d = np.stack([mask] * 3, axis=2)
        np.copyto(superimposed, blended, where=mask_3d)
    
    return superimposed, img


def save_gradcam_visualization(original: np.ndarray, overlay: np.ndarray, 
                               output_path: str, pred_label: str, confidence: float):
    """Save Grad-CAM visualization"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM: {pred_label}\nConfidence: {confidence*100:.1f}%", 
                     fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ===================== MAIN DIAGNOSIS SYSTEM =====================
class NailDiagnosisSystem:
    """Main integrated diagnosis system"""
    
    def __init__(self, model_path: str, disease_data_path: str):
        print("Initializing Nail Diagnosis System...")
        
        # Load disease database
        with open(disease_data_path, 'r') as f:
            self.disease_data = json.load(f)
        
        # Extract class names from disease data
        self.class_names = [d["name"] for d in self.disease_data]
        self.num_classes = len(self.class_names)
        
        print(f"Loaded {self.num_classes} disease classes")
        
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model("resnet18d", pretrained=False, num_classes=self.num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")
        
        # Initialize components
        self.grad_cam = GradCAM(self.model, self.model.layer4[-1].conv2)
        self.feature_extractor = GenericFeatureExtractor()
        self.symptom_parser = SymptomParser(self.disease_data)
        self.fusion_engine = InferenceFusion(self.disease_data)
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(Config.NORMALIZE_MEAN, Config.NORMALIZE_STD)
        ])
        
        print("System ready!\n")
    
    def process_single_image(self, image_path: str, image_id: int) -> Dict[str, Any]:
        """Process a single nail image"""
        print(f"Processing image {image_id}: {os.path.basename(image_path)}")
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get model prediction with Grad-CAM
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
        
        # Generate Grad-CAM
        heatmap, pred_idx, confidence = self.grad_cam(image_tensor)
        
        if heatmap is None:
            print(f"  Warning: Grad-CAM failed for image {image_id}")
            heatmap = torch.zeros((1, *Config.IMAGE_SIZE))
        
        # Create visualization
        overlay, original = apply_heatmap_overlay(image_tensor.squeeze(0), heatmap)
        
        # Save Grad-CAM
        ensure_dir(Config.GRADCAM_DIR)
        gradcam_filename = f"gradcam_img{image_id}_{self.class_names[pred_idx].replace(' ', '_')}.png"
        gradcam_path = os.path.join(Config.GRADCAM_DIR, gradcam_filename)
        save_gradcam_visualization(original, overlay, gradcam_path, 
                                  self.class_names[pred_idx], confidence)
        
        # Extract visual features
        heatmap_np = heatmap.cpu().numpy().squeeze()
        visual_features = self.feature_extractor.extract_features(original, heatmap_np)
        
        print(f"  Predicted: {self.class_names[pred_idx]} ({confidence*100:.1f}%)")
        print(f"  Grad-CAM saved: {gradcam_filename}")
        
        return {
            "image_id": image_id,
            "image_path": image_path,
            "gradcam_path": gradcam_path,
            "predicted_class": self.class_names[pred_idx],
            "confidence": float(confidence),
            "probabilities": {self.class_names[i]: float(probabilities[i]) 
                            for i in range(self.num_classes)},
            "visual_features": visual_features
        }
    
    def diagnose(self, image_paths: List[str], symptoms: str = "") -> Dict[str, Any]:
        """
        Main diagnosis function
        
        Args:
            image_paths: List of paths to nail images (1-5 images)
            symptoms: Comma-separated symptom text
            
        Returns:
            Comprehensive diagnosis results in JSON format
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Validate inputs
        if not image_paths or len(image_paths) == 0:
            raise ValueError("At least one image is required")
        if len(image_paths) > 5:
            raise ValueError("Maximum 5 images allowed")
        
        print("="*70)
        print("NAIL DISEASE DIAGNOSIS SYSTEM")
        print("="*70)
        print(f"Images to process: {len(image_paths)}")
        print(f"Symptoms provided: {'Yes' if symptoms.strip() else 'No'}")
        print("="*70 + "\n")
        
        # Parse symptoms
        parsed_symptoms = self.symptom_parser.parse_symptoms(symptoms)
        symptom_scores = {}
        for class_name in self.class_names:
            symptom_scores[normalize_key(class_name)] = \
                self.symptom_parser.match_symptoms(parsed_symptoms, class_name)
        
        # Process each image
        individual_results = []
        for idx, img_path in enumerate(image_paths, 1):
            try:
                result = self.process_single_image(img_path, idx)
                individual_results.append(result)
            except Exception as e:
                print(f"  Error processing image {idx}: {e}")
                continue
        
        if not individual_results:
            raise RuntimeError("Failed to process any images")
        
        print("\n" + "="*70)
        print("GENERATING INTEGRATED PREDICTIONS")
        print("="*70 + "\n")
        
        # Generate fused predictions for each image
        fused_results = []
        for result in individual_results:
            probs = np.array([result["probabilities"][cn] for cn in self.class_names])
            
            # Get ranked predictions
            ranked = self.fusion_engine.rank_predictions(
                probs, self.class_names, result["visual_features"], symptom_scores
            )
            
            # Add top 3 to result
            result["top_predictions"] = ranked[:3]
            result["all_ranked_predictions"] = ranked
            
            fused_results.append(result)
            
            print(f"Image {result['image_id']} - Top 3 Predictions:")
            for i, pred in enumerate(ranked[:3], 1):
                print(f"  {i}. {pred['disease']}: {pred['combined_score']*100:.1f}% "
                      f"(Model: {pred['model_probability']*100:.1f}%, "
                      f"Visual: {pred['visual_score']*100:.1f}%, "
                      f"Symptom: {pred['symptom_score']*100:.1f}%)")
            print()
        
        # Aggregate predictions across all images
        print("="*70)
        print("AGGREGATING MULTI-IMAGE RESULTS")
        print("="*70 + "\n")
        
        aggregated = self._aggregate_predictions(fused_results)
        
        # Build final output
        output = {
            "metadata": {
                "timestamp": timestamp,
                "num_images": len(image_paths),
                "symptoms_provided": symptoms.strip(),
                "parsed_symptoms": parsed_symptoms
            },
            "individual_predictions": fused_results,
            "aggregated_prediction": aggregated,
            "disease_information": self._get_disease_info(aggregated["predicted_disease"])
        }
        
        # Save output
        ensure_dir(Config.OUTPUT_DIR)
        output_path = os.path.join(Config.OUTPUT_DIR, f"diagnosis_{timestamp}.json")
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=convert_to_serializable)
        
        print(f"Results saved to: {output_path}")
        print("="*70 + "\n")
        
        return output
    
    def _aggregate_predictions(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate predictions across multiple images using weighted voting"""
        # Collect all predictions weighted by confidence
        disease_scores = {}
        
        for result in results:
            for pred in result["all_ranked_predictions"]:
                disease = pred["disease"]
                score = pred["combined_score"]
                confidence = result["confidence"]
                
                # Weight by individual image confidence
                weighted_score = score * confidence
                
                if disease not in disease_scores:
                    disease_scores[disease] = {
                        "total_weighted_score": 0.0,
                        "total_weight": 0.0,
                        "model_probs": [],
                        "visual_scores": [],
                        "symptom_scores": []
                    }
                
                disease_scores[disease]["total_weighted_score"] += weighted_score
                disease_scores[disease]["total_weight"] += confidence
                disease_scores[disease]["model_probs"].append(pred["model_probability"])
                disease_scores[disease]["visual_scores"].append(pred["visual_score"])
                disease_scores[disease]["symptom_scores"].append(pred["symptom_score"])
        
        # Compute final scores
        final_scores = []
        for disease, data in disease_scores.items():
            avg_score = data["total_weighted_score"] / (data["total_weight"] + 1e-8)
            avg_model = np.mean(data["model_probs"])
            avg_visual = np.mean(data["visual_scores"])
            avg_symptom = np.mean(data["symptom_scores"])
            
            final_scores.append({
                "disease": disease,
                "aggregated_score": float(avg_score),
                "avg_model_probability": float(avg_model),
                "avg_visual_score": float(avg_visual),
                "avg_symptom_score": float(avg_symptom)
            })
        
        # Sort by aggregated score
        final_scores.sort(key=lambda x: x["aggregated_score"], reverse=True)
        
        best = final_scores[0]
        
        print(f"Final Diagnosis: {best['disease']}")
        print(f"  Aggregated Score: {best['aggregated_score']*100:.1f}%")
        print(f"  Avg Model Prob: {best['avg_model_probability']*100:.1f}%")
        print(f"  Avg Visual Score: {best['avg_visual_score']*100:.1f}%")
        print(f"  Avg Symptom Score: {best['avg_symptom_score']*100:.1f}%")
        print()
        
        return {
            "predicted_disease": best["disease"],
            "aggregated_score": best["aggregated_score"],
            "score_breakdown": {
                "model_probability": best["avg_model_probability"],
                "visual_evidence": best["avg_visual_score"],
                "symptom_match": best["avg_symptom_score"]
            },
            "top_3_candidates": final_scores[:3],
            "all_candidates": final_scores
        }
    
    def _get_disease_info(self, disease_name: str) -> Dict[str, Any]:
        """Get detailed information about predicted disease"""
        disease_norm = normalize_key(disease_name)
        
        for disease in self.disease_data:
            if normalize_key(disease["name"]) == disease_norm:
                return {
                    "name": disease.get("name", ""),
                    "category": disease.get("category", ""),
                    "definition": disease.get("definition", ""),
                    "causes": disease.get("causes", []),
                    "visual_features": disease.get("visual_features", []),
                    "associated_symptoms": disease.get("associated_symptoms", []),
                    "related_conditions": disease.get("related_conditions", []),
                    "medical_treatments": disease.get("medical_treatments", []),
                    "when_to_see_doctor": disease.get("when_to_see_doctor", []),
                    "prognosis": disease.get("prognosis", ""),
                    "references": disease.get("references", [])
                }
        
        return {}


# ===================== CLI INTERFACE =====================
def main():
    parser = argparse.ArgumentParser(description="Integrated Nail Disease Diagnosis System")
    parser.add_argument("--images", nargs="+", required=True, 
                       help="Paths to 1-5 nail images")
    parser.add_argument("--symptoms", type=str, default="",
                       help="Comma-separated symptoms (e.g., 'pain,swelling,discoloration')")
    parser.add_argument("--model", type=str, default=Config.MODEL_PATH,
                       help="Path to model weights file")
    parser.add_argument("--disease-data", type=str, default=Config.DISEASE_DATA_PATH,
                       help="Path to disease database JSON")
    parser.add_argument("--output-dir", type=str, default=Config.OUTPUT_DIR,
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Update config
    Config.MODEL_PATH = args.model
    Config.DISEASE_DATA_PATH = args.disease_data
    Config.OUTPUT_DIR = args.output_dir
    Config.GRADCAM_DIR = os.path.join(args.output_dir, "gradcam")
    
    # Validate inputs
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.disease_data):
        print(f"Error: Disease data file not found: {args.disease_data}")
        return
    
    for img_path in args.images:
        if not os.path.exists(img_path):
            print(f"Error: Image not found: {img_path}")
            return
    
    # Initialize system
    system = NailDiagnosisSystem(args.model, args.disease_data)
    
    # Run diagnosis
    try:
        results = system.diagnose(args.images, args.symptoms)
        
        print("\n" + "="*70)
        print("DIAGNOSIS COMPLETE")
        print("="*70)
        print(f"\nPredicted Disease: {results['aggregated_prediction']['predicted_disease']}")
        print(f"Confidence: {results['aggregated_prediction']['aggregated_score']*100:.1f}%")
        print(f"\nFull results saved to: {Config.OUTPUT_DIR}/")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\nError during diagnosis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()