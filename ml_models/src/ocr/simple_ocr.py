"""
Simple OCR Service - No OpenCV dependency
"""

from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import requests
import base64
from typing import Dict

class SimpleOCR:
    def __init__(self):
        self.confidence_threshold = 60
    
    def preprocess_image_simple(self, image_path: str) -> Image.Image:
        """Simple image preprocessing using PIL only"""
        # Open image
        img = Image.open(image_path)
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)
        
        return img
    
    def extract_text(self, image_path: str) -> Dict:
        """Extract text using simple Tesseract"""
        try:
            # Preprocess image
            processed_img = self.preprocess_image_simple(image_path)
            
            # Configure Tesseract
            custom_config = r'--oem 3 --psm 6'
            
            # Extract text
            text = pytesseract.image_to_string(processed_img, config=custom_config)
            
            # Get confidence data
            data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': text.strip(),
                'confidence': avg_confidence,
                'method': 'simple_tesseract',
                'success': True
            }
            
        except Exception as e:
            return {
                'text': '',
                'confidence': 0,
                'method': 'simple_tesseract',
                'success': False,
                'error': str(e)
            }

# Test function
if __name__ == "__main__":
    ocr = SimpleOCR()
    print("✅ Simple OCR initialized successfully!")