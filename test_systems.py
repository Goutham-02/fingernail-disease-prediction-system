"""
Test script to verify the integrated system works correctly
"""

import os
import sys
import json

def test_imports():
    """Test if all required packages are installed"""
    print("Testing imports...")
    try:
        import torch
        import torchvision
        import timm
        import cv2
        import numpy
        from PIL import Image
        print("✓ All required packages installed")
        return True
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        return False

def test_files():
    """Test if required files exist"""
    print("\nTesting required files...")
    required_files = [
        "integrated_nail_diagnosis.py",
        "data1.json",
        "17classes_resnet_97.pth"
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ Found: {file}")
        else:
            print(f"✗ Missing: {file}")
            all_exist = False
    
    return all_exist

def test_disease_data():
    """Test if disease data is valid"""
    print("\nTesting disease data...")
    try:
        with open("data1.json", 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("✗ Disease data should be a list")
            return False
        
        if len(data) == 0:
            print("✗ Disease data is empty")
            return False
        
        print(f"✓ Found {len(data)} diseases in database")
        
        # Check required fields
        required_fields = ["name", "weights", "visual_features", "associated_symptoms"]
        sample = data[0]
        missing = [f for f in required_fields if f not in sample]
        
        if missing:
            print(f"✗ Missing fields in disease data: {missing}")
            return False
        
        print("✓ Disease data structure is valid")
        return True
        
    except Exception as e:
        print(f"✗ Error reading disease data: {e}")
        return False

def test_system_initialization():
    """Test if system can be initialized"""
    print("\nTesting system initialization...")
    try:
        from integrated_nail_diagnosis import NailDiagnosisSystem
        
        system = NailDiagnosisSystem(
            model_path="17classes_resnet_97.pth",
            disease_data_path="data1.json"
        )
        
        print("✓ System initialized successfully")
        print(f"  - Loaded {system.num_classes} disease classes")
        print(f"  - Model on device: {system.device}")
        return True, system
        
    except Exception as e:
        print(f"✗ Failed to initialize system: {e}")
        return False, None

def test_feature_extraction():
    """Test feature extraction on dummy image"""
    print("\nTesting feature extraction...")
    try:
        from integrated_nail_diagnosis import GenericFeatureExtractor
        import numpy as np
        
        extractor = GenericFeatureExtractor()
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        features = extractor.extract_features(dummy_image)
        
        if not features:
            print("✗ No features extracted")
            return False
        
        print(f"✓ Extracted {len(features)} feature types")
        print("  Feature detectors:")
        for feature_name in list(features.keys())[:5]:
            print(f"    - {feature_name}")
        if len(features) > 5:
            print(f"    ... and {len(features) - 5} more")
        
        return True
        
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        return False

def test_symptom_parsing():
    """Test symptom parsing"""
    print("\nTesting symptom parsing...")
    try:
        from integrated_nail_diagnosis import SymptomParser
        
        with open("data1.json", 'r') as f:
            disease_data = json.load(f)
        
        parser = SymptomParser(disease_data)
        
        test_symptoms = "patchy hair loss, itching, pain"
        parsed = parser.parse_symptoms(test_symptoms)
        
        if not parsed:
            print("✗ Failed to parse symptoms")
            return False
        
        print(f"✓ Parsed symptoms: {list(parsed.keys())}")
        
        # Test matching
        score = parser.match_symptoms(parsed, "aloperia_areata")
        print(f"  Match score for 'aloperia_areata': {score:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Symptom parsing failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("="*70)
    print("INTEGRATED NAIL DIAGNOSIS SYSTEM - TEST SUITE")
    print("="*70)
    
    tests = [
        ("Package Imports", test_imports),
        ("Required Files", test_files),
        ("Disease Data", test_disease_data),
        ("Feature Extraction", test_feature_extraction),
        ("Symptom Parsing", test_symptom_parsing),
    ]
    
    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))
    
    # System initialization test (requires all previous to pass)
    if all(r[1] for r in results):
        result, system = test_system_initialization()
        results.append(("System Initialization", result))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {name}")
    
    print("="*70)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nQuick start:")
        print("  python integrated_nail_diagnosis.py --images image.jpg --symptoms 'pain,swelling'")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
    
    print("="*70)
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)