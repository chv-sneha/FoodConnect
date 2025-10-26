"""
Test Simple OCR - No OpenCV required
"""

from src.ocr.simple_ocr import SimpleOCR
import os

def test_simple_ocr():
    print("🔍 Testing Simple OCR Service...")
    
    try:
        ocr = SimpleOCR()
        print("✅ Simple OCR initialized successfully!")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Test with image if available
    test_image_path = "test_images/sample_food_label.jpg"
    
    if os.path.exists(test_image_path):
        print(f"📸 Testing with: {test_image_path}")
        result = ocr.extract_text(test_image_path)
        
        print(f"📝 Text: {result['text'][:200]}...")  # First 200 chars
        print(f"🎯 Confidence: {result['confidence']:.1f}")
        print(f"✅ Success: {result['success']}")
    else:
        print("⚠️  No test image found")
        print("📋 Add image to test_images/sample_food_label.jpg")

if __name__ == "__main__":
    test_simple_ocr()