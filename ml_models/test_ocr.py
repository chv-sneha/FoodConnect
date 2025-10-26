"""
Test OCR Service - Run this to check if everything works
"""

from src.ocr.ocr_service import OCRService
import os

def test_ocr():
    print("🔍 Testing OCR Service...")
    
    # Initialize OCR service
    try:
        ocr = OCRService()
        print("✅ OCR Service initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing OCR: {e}")
        return
    
    # Test with a sample image (you need to provide this)
    test_image_path = "test_images/sample_food_label.jpg"
    
    if os.path.exists(test_image_path):
        print(f"📸 Testing with image: {test_image_path}")
        
        result = ocr.extract_text(test_image_path)
        
        print(f"📝 Extracted Text: {result['text']}")
        print(f"🎯 Confidence: {result['confidence']}")
        print(f"🔧 Method: {result['method']}")
        print(f"✅ Success: {result['success']}")
    else:
        print(f"⚠️  Test image not found at: {test_image_path}")
        print("📋 Please add a food label image to test with")
        print("💡 You can still use the OCR service with any image path")

if __name__ == "__main__":
    test_ocr()