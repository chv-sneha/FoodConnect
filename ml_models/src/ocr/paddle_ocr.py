#!/usr/bin/env python3
"""
PaddleOCR for Food Labels - Much better than Tesseract
"""

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    
import cv2
import numpy as np
from typing import Dict, List
import re

class FoodLabelOCR:
    def __init__(self):
        if PADDLE_AVAILABLE:
            # Initialize PaddleOCR with English and Hindi
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            self.ocr_hindi = PaddleOCR(use_angle_cls=True, lang='hi', show_log=False)
        else:
            self.ocr = None
            print("⚠️ PaddleOCR not available, install with: pip install paddlepaddle paddleocr")
    
    def extract_text(self, image_path: str) -> Dict:
        """Extract text using PaddleOCR"""
        if not PADDLE_AVAILABLE or not self.ocr:
            return self._fallback_ocr(image_path)
        
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return {'success': False, 'error': 'Could not load image'}
            
            # Try English first
            result_en = self.ocr.ocr(img, cls=True)
            
            # Try Hindi if English confidence is low
            result_hi = self.ocr_hindi.ocr(img, cls=True)
            
            # Combine results
            all_text_blocks = []
            full_text = ""
            
            # Process English results
            if result_en and result_en[0]:
                for line in result_en[0]:
                    if len(line) >= 2:
                        bbox, (text, confidence) = line
                        if confidence > 0.5:  # Filter low confidence
                            all_text_blocks.append({
                                'text': text,
                                'confidence': confidence * 100,
                                'bbox': bbox,
                                'language': 'en'
                            })
                            full_text += text + " "
            
            # Process Hindi results (add if not duplicate)
            if result_hi and result_hi[0]:
                for line in result_hi[0]:
                    if len(line) >= 2:
                        bbox, (text, confidence) = line
                        if confidence > 0.5 and text not in full_text:
                            all_text_blocks.append({
                                'text': text,
                                'confidence': confidence * 100,
                                'bbox': bbox,
                                'language': 'hi'
                            })
                            full_text += text + " "
            
            # Calculate average confidence
            avg_confidence = np.mean([block['confidence'] for block in all_text_blocks]) if all_text_blocks else 0
            
            return {
                'success': True,
                'text_blocks': all_text_blocks,
                'full_text': full_text.strip(),
                'confidence': avg_confidence,
                'method': 'paddleocr'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _fallback_ocr(self, image_path: str) -> Dict:
        """Fallback to basic OCR if PaddleOCR not available"""
        try:
            import pytesseract
            
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Basic preprocessing
            gray = cv2.medianBlur(gray, 3)
            
            # Extract text
            text = pytesseract.image_to_string(gray, config='--oem 3 --psm 6')
            
            return {
                'success': True,
                'text_blocks': [{'text': text, 'confidence': 70, 'language': 'en'}],
                'full_text': text,
                'confidence': 70,
                'method': 'tesseract_fallback'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def extract_structured_data(self, image_path: str) -> Dict:
        """Extract structured data (nutrition table, ingredients, FSSAI)"""
        ocr_result = self.extract_text(image_path)
        
        if not ocr_result['success']:
            return ocr_result
        
        full_text = ocr_result['full_text']
        
        # Extract structured information
        structured_data = {
            'product_name': self._extract_product_name(full_text),
            'ingredients': self._extract_ingredients(full_text),
            'nutrition_facts': self._extract_nutrition_table(full_text),
            'fssai_number': self._extract_fssai(full_text),
            'brand': self._extract_brand(full_text)
        }
        
        return {
            'success': True,
            'structured_data': structured_data,
            'raw_ocr': ocr_result,
            'confidence': ocr_result['confidence']
        }
    
    def _extract_product_name(self, text: str) -> str:
        """Extract product name from text"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Look for product name patterns
        for line in lines[:5]:  # Check first 5 lines
            # Skip lines with numbers, ingredients, nutrition
            if not re.search(r'\d+|ingredient|nutrition|calorie|protein', line.lower()):
                if len(line) > 3 and len(line) < 50:
                    return line
        
        return lines[0] if lines else "Unknown Product"
    
    def _extract_ingredients(self, text: str) -> List[str]:
        """Extract ingredients list"""
        # Multiple patterns for ingredients
        patterns = [
            r'ingredients?[:\s]+(.*?)(?:nutrition|allergen|net|weight|mfg|exp|best)',
            r'ingredients?[:\s]+(.*?)(?:\n\n|\.|$)',
            r'contains?[:\s]+(.*?)(?:\n\n|\.|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                ingredients_text = match.group(1).strip()
                
                # Clean and split ingredients
                ingredients = []
                for item in re.split(r'[,;]', ingredients_text):
                    item = re.sub(r'[()[\]{}]', '', item).strip()
                    item = re.sub(r'\d+%?', '', item).strip()
                    
                    if len(item) > 2 and not item.isdigit():
                        ingredients.append(item)
                
                return ingredients[:15]  # Limit to 15 ingredients
        
        return []
    
    def _extract_nutrition_table(self, text: str) -> Dict:
        """Extract nutrition facts from text"""
        nutrition = {}
        
        # Enhanced patterns for nutrition extraction
        patterns = {
            'energy': r'(?:energy|calories?)[:\s]*(\d+(?:\.\d+)?)\s*(?:kcal|cal|kj)',
            'protein': r'protein[:\s]*(\d+(?:\.\d+)?)\s*g',
            'carbohydrates': r'(?:carbohydrate|carbs?)[:\s]*(\d+(?:\.\d+)?)\s*g',
            'fat': r'(?:total\s+fat|fat)[:\s]*(\d+(?:\.\d+)?)\s*g',
            'fiber': r'(?:dietary\s+fiber|fiber)[:\s]*(\d+(?:\.\d+)?)\s*g',
            'sugar': r'(?:total\s+sugar|sugar)[:\s]*(\d+(?:\.\d+)?)\s*g',
            'sodium': r'sodium[:\s]*(\d+(?:\.\d+)?)\s*(?:mg|g)'
        }
        
        text_lower = text.lower()
        
        for nutrient, pattern in patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                value = float(match.group(1))
                
                # Convert sodium from mg to g if needed
                if nutrient == 'sodium' and value > 10:
                    value = value / 1000
                
                nutrition[nutrient] = value
        
        return nutrition
    
    def _extract_fssai(self, text: str) -> str:
        """Extract FSSAI license number"""
        patterns = [
            r'fssai[:\s#-]*(\d{14})',
            r'lic[:\s#-]*no[:\s#-]*(\d{14})',
            r'license[:\s#-]*(\d{14})',
            r'(\d{14})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                number = match.group(1)
                if len(number) == 14:
                    return number
        
        return ""
    
    def _extract_brand(self, text: str) -> str:
        """Extract brand name"""
        lines = text.split('\n')
        
        # Look for brand patterns in first few lines
        for line in lines[:3]:
            line = line.strip()
            if len(line) > 2 and len(line) < 30:
                # Check if it looks like a brand name
                if re.match(r'^[A-Z][a-zA-Z\s]+$', line):
                    return line
        
        return ""

# Test the OCR
if __name__ == "__main__":
    ocr = FoodLabelOCR()
    print("✅ Food Label OCR initialized")
    
    if PADDLE_AVAILABLE:
        print("🚀 PaddleOCR available - high accuracy mode")
    else:
        print("⚠️ Using fallback OCR - install PaddleOCR for better results")
        print("   pip install paddlepaddle paddleocr")