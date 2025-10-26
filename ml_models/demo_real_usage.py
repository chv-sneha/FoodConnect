"""
Real Usage Demo - How OCR works with user uploaded images
"""

from src.ocr.simple_ocr import SimpleOCR

def analyze_user_image(image_path: str):
    """
    This is how your app will use OCR when user uploads ANY image
    """
    print(f"📸 User uploaded: {image_path}")
    
    # Initialize OCR
    ocr = SimpleOCR()
    
    # Extract text from user's image
    result = ocr.extract_text(image_path)
    
    if result['success']:
        print("✅ OCR SUCCESS!")
        print(f"📝 Extracted Text:\n{result['text']}")
        print(f"🎯 Confidence: {result['confidence']:.1f}%")
        
        # This text goes to Step 2 (Parser) to extract:
        # - Nutrition facts
        # - Ingredients list  
        # - FSSAI number
        
        return result['text']
    else:
        print("❌ OCR FAILED!")
        print(f"Error: {result.get('error', 'Unknown error')}")
        return None

# Example: User uploads image from phone/computer
if __name__ == "__main__":
    # This is what happens when user uploads ANY image:
    
    # Example 1: User uploads from phone
    user_image = "/path/to/user/uploaded/food_label.jpg"
    
    # Example 2: User takes photo with camera
    camera_image = "/uploads/camera_capture_123.jpg"
    
    # Example 3: User drags and drops image
    dropped_image = "/tmp/user_dropped_image.png"
    
    print("🔄 This is how OCR works with ANY user image:")
    print("1. User uploads image → Your app saves it")
    print("2. Your app calls: analyze_user_image(saved_image_path)")
    print("3. OCR extracts text from that image")
    print("4. Text goes to parser for nutrition/ingredients/FSSAI")
    
    # Test with any image path you have
    test_path = input("Enter path to any food label image (or press Enter to skip): ")
    if test_path and test_path.strip():
        analyze_user_image(test_path.strip())