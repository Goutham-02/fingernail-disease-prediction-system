from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
from typing import List, Optional
import os
import uuid
import json
import logging
from pathlib import Path
import shutil

from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
import google.generativeai as genai

from integrated_nail_diagnosis import NailDiagnosisSystem, convert_to_serializable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Cloudinary with validation
try:
    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    )
    logger.info("Cloudinary configured successfully")
except Exception as e:
    logger.warning(f"Cloudinary configuration failed: {e}")

# Configure Gemini with validation
try:
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        genai.configure(api_key=gemini_key)
        logger.info("Gemini API configured successfully")
    else:
        logger.warning("GEMINI_API_KEY not found in environment")
except Exception as e:
    logger.warning(f"Gemini configuration failed: {e}")

UPLOAD_FOLDER = os.getenv("CLOUDINARY_UPLOAD_FOLDER", "nail-diagnosis/gradcam")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "10")) * 1024 * 1024  # Default 10MB
TEMP_UPLOAD_DIR = os.getenv("TEMP_UPLOAD_DIR", "temp_uploads")

# Create and validate temp directory at startup
try:
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    # Test write permission
    test_file = os.path.join(TEMP_UPLOAD_DIR, ".write_test")
    with open(test_file, "w") as f:
        f.write("test")
    os.remove(test_file)
    logger.info(f"Temp directory ready: {TEMP_UPLOAD_DIR}")
except Exception as e:
    logger.error(f"Cannot write to temp directory {TEMP_UPLOAD_DIR}: {e}")
    logger.error("File uploads will fail. Check directory permissions.")

# Preload disease DB for RAG
DISEASE_DB = []
try:
    disease_db_path = Path("data1.json")
    if disease_db_path.exists():
        with open(disease_db_path, "r", encoding="utf-8") as f:
            DISEASE_DB = json.load(f)
        if not isinstance(DISEASE_DB, list):
            logger.error("Disease DB is not a list, resetting to empty")
            DISEASE_DB = []
        logger.info(f"Loaded {len(DISEASE_DB)} disease entries")
    else:
        logger.warning("data1.json not found, starting with empty disease DB")
except json.JSONDecodeError as e:
    logger.error(f"Invalid JSON in data1.json: {e}")
    DISEASE_DB = []
except Exception as e:
    logger.error(f"Failed to load disease DB: {e}")
    DISEASE_DB = []

app = FastAPI(title="Nail Disease Diagnosis API", version="1.0.0")

# Templates and static setup
templates = Jinja2Templates(directory="templates")
try:
    os.makedirs("static", exist_ok=True)
except Exception:
    pass
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the diagnosis system with error handling
system = None
try:
    model_path = os.getenv("MODEL_PATH", "17classes_resnet_97.pth")
    disease_path = os.getenv("DISEASE_DATA_PATH", "data1.json")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    system = NailDiagnosisSystem(
        model_path=model_path,
        disease_data_path=disease_path,
    )
    logger.info("Diagnosis system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize diagnosis system: {e}")
    logger.error("API will return 503 for diagnosis requests")


def _is_supported_image(content_type: Optional[str], filename: str = "") -> bool:
    """Check if file is a supported image type."""
    if not content_type:
        # Fallback to filename extension
        if filename:
            ext = Path(filename).suffix.lower()
            return ext in [".png", ".jpg", ".jpeg", ".webp"]
        return False
    
    content_type = content_type.lower()
    return any(x in content_type for x in ["png", "jpg", "jpeg", "webp", "image"])


def _upload_to_cloudinary(local_path: str) -> Optional[str]:
    """Upload file to Cloudinary with comprehensive error handling."""
    try:
        if not Path(local_path).exists():
            logger.error(f"File not found for upload: {local_path}")
            return None
            
        config = cloudinary.config()
        if not (config.cloud_name and config.api_key and config.api_secret):
            logger.warning("Cloudinary not configured, skipping upload")
            return None
            
        result = cloudinary.uploader.upload(
            local_path,
            folder=UPLOAD_FOLDER,
            resource_type="image",
            timeout=30
        )
        url = result.get("secure_url")
        if url:
            logger.info(f"Successfully uploaded to Cloudinary: {url}")
        return url
    except cloudinary.exceptions.Error as e:
        logger.error(f"Cloudinary upload error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during Cloudinary upload: {e}")
        return None


def _extract_gemini_text(resp) -> str:
    """Extract text from Gemini response with fallbacks."""
    try:
        # Primary method
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()
        
        # Fallback for structured responses
        if hasattr(resp, "candidates") and resp.candidates:
            candidates = resp.candidates
            if len(candidates) > 0:
                candidate = candidates[0]
                if hasattr(candidate, "content"):
                    content = candidate.content
                    if hasattr(content, "parts") and content.parts:
                        parts_text = []
                        for part in content.parts:
                            if hasattr(part, "text"):
                                parts_text.append(part.text)
                        if parts_text:
                            return "".join(parts_text).strip()
                # Dict-based fallback
                elif isinstance(candidate, dict):
                    parts = candidate.get("content", {}).get("parts", [])
                    if parts:
                        return "".join(p.get("text", "") for p in parts).strip()
    except Exception as e:
        logger.error(f"Error extracting Gemini text: {e}")
    
    return ""


def build_llm_description(results: dict, disease_db: list, symptoms: str) -> str:
    """Generate LLM description using Gemini with RAG."""
    try:
        # Validate inputs
        if not results or not isinstance(results, dict):
            logger.warning("Invalid results for LLM description")
            return ""
        
        # Extract predicted disease
        predicted = ""
        try:
            predicted = results.get("aggregated_prediction", {}).get("predicted_disease", "")
        except Exception as e:
            logger.error(f"Error extracting predicted disease: {e}")
            return ""
        
        if not predicted:
            logger.warning("No predicted disease found")
            return ""

        # Retrieve matching disease context
        ctx = {}
        try:
            for disease in disease_db:
                if not isinstance(disease, dict):
                    continue
                disease_name = str(disease.get("name", "")).lower().strip()
                if disease_name == predicted.lower().strip():
                    ctx = disease
                    break
        except Exception as e:
            logger.error(f"Error searching disease DB: {e}")

        # Check Gemini configuration
        if not os.getenv("GEMINI_API_KEY"):
            logger.warning("Gemini API key not configured")
            return ""

        # Compose prompt
        breakdown = results.get("aggregated_prediction", {}).get("score_breakdown", {})
        symptoms_text = symptoms if symptoms else "None provided"
        
        prompt = (
            "You are a medical information assistant (not a doctor). Using the context below, "
            "explain the likely nail condition in plain English. Be concise (120-200 words), include: "
            "what it is, common visual signs, related symptoms, when to see a doctor, and general "
            "(non-prescriptive) care guidance. Mention model confidence at the end. Avoid giving "
            "diagnoses; use tentative language like 'may indicate', 'could suggest', etc.\n\n"
            f"User symptoms: {symptoms_text}\n"
            f"Predicted condition: {predicted}\n"
            f"Model confidence scores: {json.dumps(breakdown)}\n\n"
        )
        
        if ctx:
            # Truncate context to avoid token limits
            ctx_str = json.dumps(ctx, ensure_ascii=False)[:6000]
            prompt += f"Medical context:\n{ctx_str}\n"
        else:
            prompt += "Note: No additional medical context available for this condition.\n"

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=300,
                temperature=0.7,
            )
        )
        
        extracted_text = _extract_gemini_text(response)
        if extracted_text:
            logger.info("Successfully generated LLM description")
            return extracted_text
        else:
            logger.warning("Empty response from Gemini")
            return ""
            
    except Exception as e:
        logger.error(f"Error generating LLM description: {e}")
        return ""


@app.post("/diagnose")
async def diagnose(
    images: List[UploadFile] = File(...),
    symptoms: Optional[str] = Form("")
):
    """
    Diagnose nail condition from uploaded images.
    
    Args:
        images: 1-5 nail images (PNG, JPG, JPEG, WEBP)
        symptoms: Optional text description of symptoms
    
    Returns:
        JSON with diagnosis results, Grad-CAM visualizations, and AI explanation
    """
    # ONLY USE IMAGES for prediction. Do not use symptoms.
    if not images or not (1 <= len(images) <= 5):
        raise HTTPException(status_code=400, detail="Provide 1-5 images")

    os.makedirs("temp_uploads", exist_ok=True)
    saved_paths: List[str] = []

    try:
        # Save all uploaded images
        for f in images:
            if not _is_supported_image(f.content_type):
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {f.content_type}")
            dest = os.path.join("temp_uploads", f"{uuid.uuid4()}_{os.path.basename(f.filename)}")
            with open(dest, "wb") as out:
                shutil.copyfileobj(f.file, out)
            saved_paths.append(dest)

        results = system.diagnose(saved_paths, "")

        # Aggregate across images: choose final class by highest average confidence
        gradcam_urls: List[str] = []
        class_to_scores = {}
        for imgres in results.get("individual_predictions", []):
            name = imgres.get("predicted_class")
            score = float(imgres.get("confidence") or 0.0)
            class_to_scores.setdefault(name, []).append(score)
            # Upload Grad-CAM and collect URL list
            local_path = imgres.get("gradcam_path")
            if local_path and os.path.exists(local_path):
                url = _upload_to_cloudinary(local_path)
                if url:
                    gradcam_urls.append(url)

        if not class_to_scores:
            raise HTTPException(status_code=500, detail="No predictions produced")

        # Compute average confidence per class
        best_class = None
        best_avg = -1.0
        for cls, scores in class_to_scores.items():
            avg = sum(scores) / max(1, len(scores))
            if avg > best_avg:
                best_avg = avg
                best_class = cls

        # Lookup disease info for final class
        info = None
        for d in DISEASE_DB:
            if str(d.get("name", "")).lower() == str(best_class).lower():
                info = d
                break

        # Generate one description with Gemini for the final class
        desc = ""
        if info:
            try:
                prompt = (
                    "You are a medical information assistant (not a doctor). Using the context below, explain the nail disease in simple, plain language. Ignore any user symptoms. Be concise (120-200 words), include: what it is, common visual signs, related symptoms, when to see a doctor, and general (non-prescriptive) care guidance. Mention aggregated model confidence at the end. Avoid diagnosis claims; use tentative language like 'may indicate', 'could suggest'.\n\n"
                    f"Predicted condition: {best_class}\n"
                    f"Aggregated model confidence: {best_avg:.2f}\n"
                    f"Medical context:\n{json.dumps(info)[:6000]}\n"
                )
                model = genai.GenerativeModel("gemini-1.5-flash")
                resp = model.generate_content(prompt)
                desc = _extract_gemini_text(resp)
            except Exception:
                desc = "AI-generated description unavailable."

        final_payload = {
            "predicted_class": best_class,
            "aggregated_score": best_avg,
            "gradcam_urls": gradcam_urls,
            "description": desc,
            "disease_info": info,
        }

        return JSONResponse(content=convert_to_serializable(final_payload))
    finally:
        for p in saved_paths:
            try: os.remove(p)
            except: pass


@app.get("/health")
async def health():
    """
    Health check endpoint returning system readiness information.
    """
    try:
        # Check model status
        model_loaded = False
        try:
            if system is not None:
                # Common patterns for checking model readiness
                if hasattr(system, "model") and getattr(system, "model", None) is not None:
                    model_loaded = True
                elif hasattr(system, "is_ready") and callable(getattr(system, "is_ready")):
                    model_loaded = bool(system.is_ready())
                elif hasattr(system, "ready") and callable(getattr(system, "ready")):
                    model_loaded = bool(system.ready())
        except Exception as e:
            logger.error(f"Error checking model status: {e}")
            model_loaded = False

        # Check Cloudinary configuration
        cloudinary_configured = False
        try:
            config = cloudinary.config()
            cloudinary_configured = bool(
                config and 
                config.cloud_name and 
                config.api_key and 
                config.api_secret
            )
        except Exception as e:
            logger.error(f"Error checking Cloudinary config: {e}")

        # Check Gemini configuration
        gemini_configured = bool(os.getenv("GEMINI_API_KEY"))

        # Check temp directory
        temp_dir_exists = os.path.isdir(TEMP_UPLOAD_DIR)
        temp_dir_writable = False
        try:
            if temp_dir_exists:
                temp_dir_writable = os.access(TEMP_UPLOAD_DIR, os.W_OK)
        except Exception:
            pass

        payload = {
            "status": "healthy" if model_loaded else "degraded",
            "model_loaded": model_loaded,
            "disease_db_count": len(DISEASE_DB) if isinstance(DISEASE_DB, list) else 0,
            "cloudinary_configured": cloudinary_configured,
            "gemini_configured": gemini_configured,
            "temp_dir_exists": temp_dir_exists,
            "temp_dir_writable": temp_dir_writable,
            "temp_dir_path": TEMP_UPLOAD_DIR,
            "max_upload_images": 5,
            "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
        }

        # Return 503 if critical components unavailable
        if not model_loaded or not temp_dir_writable:
            payload["status"] = "unhealthy"
            payload["issues"] = []
            if not model_loaded:
                payload["issues"].append("Model not loaded")
            if not temp_dir_writable:
                payload["issues"].append(f"Temp directory not writable: {TEMP_UPLOAD_DIR}")
            status_code = 503
        else:
            status_code = 200
        return JSONResponse(content=payload, status_code=status_code)
        
    except Exception as exc:
        logger.error(f"Health check failed: {exc}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(exc)}
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Nail Disease Diagnosis API",
        "version": "1.0.0",
        "endpoints": {
            "POST /diagnose": "Upload images for diagnosis",
            "GET /health": "System health check",
        }
    }

# -------- UI ROUTES --------
@app.get("/ui", response_class=HTMLResponse)
async def ui_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
    

@app.get("/ui/diagnose", response_class=HTMLResponse)
async def ui_diagnose(request: Request):
    return templates.TemplateResponse("diagnose.html", {"request": request})


@app.get("/ui/health", response_class=HTMLResponse)
async def ui_health(request: Request):
    return templates.TemplateResponse("health.html", {"request": request})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)