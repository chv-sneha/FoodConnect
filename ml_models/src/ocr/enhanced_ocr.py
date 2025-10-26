#!/usr/bin/env python3
"""
Enhanced OCR for Food Labels - Multi-model approach
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple
import re
import easyocr
from PIL import Image, ImageEnhance, ImageFilter
import logging

class EnhancedFoodOCR:
    def __init__(self):
        # Initialize multiple OCR engines
        self.easyocr_reader = easyocr.Reader(['en', 'hi'], gpu=False)
        
        # Try to import PaddleOCR
        try:
            from paddleocr import PaddleOCR
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            self.paddle_available = True
        except ImportError:
            self.paddle_available = False
            logging.warning("PaddleOCR not available")
    
    def preprocess_image(self, image_path: str) -> List[np.ndarray]:
        """Advanced preprocessing for better OCR accuracy"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not load image")
        
        processed_images = []
        
        # 1. Original image
        processed_images.append(img)
        
        # 2. Grayscale with contrast enhancement
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        processed_images.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))
        
        # 3. Denoised version
        denoised = cv2.fastNlMeansDenoising(gray)
        processed_images.append(cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR))
        
        # 4. Sharpened version
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        processed_images.append(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR))
        
        # 5. Adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        processed_images.append(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR))
        
        return processed_images
    
    def extract_with_ensemble(self, image_path: str) -> Dict:
        """Extract text using ensemble of OCR models"""
        try:
            processed_images = self.preprocess_image(image_path)
            all_results = []
            
            # Run EasyOCR on all preprocessed images
            for i, img in enumerate(processed_images):
                try:
                    results = self.easyocr_reader.readtext(img)
                    for (bbox, text, confidence) in results:
                        if confidence > 0.3:  # Lower threshold for more text
                            all_results.append({
                                'text': text.strip(),
                                'confidence': confidence * 100,
                                'bbox': bbox,
                                'source': f'easyocr_v{i}',
                                'method': 'easyocr'
                            })
                except Exception as e:
                    logging.warning(f"EasyOCR failed on variant {i}: {e}")
            
            # Run PaddleOCR if available
            if self.paddle_available:
                try:
                    paddle_results = self.paddle_ocr.ocr(processed_images[0], cls=True)
                    if paddle_results and paddle_results[0]:
                        for line in paddle_results[0]:
                            if len(line) >= 2:
                                bbox, (text, confidence) = line
                                if confidence > 0.3:
                                    all_results.append({
                                        'text': text.strip(),
                                        'confidence': confidence * 100,
                                        'bbox': bbox,
                                        'source': 'paddle',
                                        'method': 'paddleocr'
                                    })
                except Exception as e:
                    logging.warning(f"PaddleOCR failed: {e}")
            
            # Deduplicate and merge results
            merged_results = self._merge_duplicate_texts(all_results)
            
            # Combine all text
            full_text = " ".join([r['text'] for r in merged_results])
            avg_confidence = np.mean([r['confidence'] for r in merged_results]) if merged_results else 0
            
            return {
                'success': True,
                'text_blocks': merged_results,
                'full_text': full_text,
                'confidence': avg_confidence,
                'total_blocks': len(merged_results)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _merge_duplicate_texts(self, results: List[Dict]) -> List[Dict]:
        """Merge duplicate text detections"""
        merged = {}
        
        for result in results:
            text = result['text'].lower().strip()
            if len(text) < 2:
                continue
                
            if text in merged:
                # Keep the one with higher confidence
                if result['confidence'] > merged[text]['confidence']:
                    merged[text] = result
            else:
                merged[text] = result
        
        return list(merged.values())
    
    def extract_structured_data(self, image_path: str) -> Dict:
        """Extract structured food label data"""
        ocr_result = self.extract_with_ensemble(image_path)
        
        if not ocr_result['success']:
            return ocr_result
        
        full_text = ocr_result['full_text']
        
        # Extract structured information
        structured_data = {
            'product_name': self._extract_product_name(full_text, ocr_result['text_blocks']),
            'ingredients': self._extract_ingredients(full_text),
            'nutrition_facts': self._extract_nutrition_facts(full_text),
            'fssai_number': self._extract_fssai(full_text),
            'brand': self._extract_brand(full_text),
            'allergens': self._extract_allergens(full_text),
            'net_weight': self._extract_net_weight(full_text),
            'mfg_date': self._extract_dates(full_text, 'mfg'),
            'exp_date': self._extract_dates(full_text, 'exp')
        }
        
        # Calculate Nutri-Score
        nutri_score = self._calculate_nutri_score(structured_data['nutrition_facts'])
        
        return {
            'success': True,
            'structured_data': structured_data,
            'nutri_score': nutri_score,
            'raw_ocr': ocr_result,
            'confidence': ocr_result['confidence']
        }
    
    def _extract_product_name(self, text: str, text_blocks: List[Dict]) -> str:
        """Extract product name using position and text analysis"""
        lines = text.split('\n')
        
        # Look for largest text (usually product name)
        if text_blocks:
            # Sort by confidence and position (top of image)
            sorted_blocks = sorted(text_blocks, key=lambda x: (-x['confidence'], -len(x['text'])))
            
            for block in sorted_blocks[:3]:
                text_content = block['text'].strip()
                if (len(text_content) > 3 and len(text_content) < 50 and 
                    not re.search(r'\d+\s*(g|kg|ml|l|%)|ingredient|nutrition|fssai', text_content.lower())):
                    return text_content
        
        # Fallback to first meaningful line
        for line in lines[:5]:
            line = line.strip()
            if len(line) > 3 and len(line) < 50:
                return line
        
        return "Unknown Product"
    
    def _extract_ingredients(self, text: str) -> List[str]:
        """Enhanced ingredient extraction"""
        # Multiple patterns for different label formats
        patterns = [
            r'ingredients?[:\s]+(.*?)(?:nutrition|allergen|contains|net\s+wt|weight|mfg|exp|best|fssai)',
            r'ingredients?[:\s]+(.*?)(?:\n\s*\n|\.|$)',
            r'contains?[:\s]+(.*?)(?:allergen|nutrition|net\s+wt)',
        ]
        
        ingredients = []
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                ingredients_text = match.group(1).strip()
                
                # Clean and split
                items = re.split(r'[,;]', ingredients_text)
                
                for item in items:
                    # Clean item
                    item = re.sub(r'[()[\]{}]', '', item).strip()
                    item = re.sub(r'\d+\.?\d*\s*%?', '', item).strip()
                    item = re.sub(r'\s+', ' ', item)
                    
                    if len(item) > 2 and not item.isdigit() and len(item) < 50:
                        ingredients.append(item.title())
                
                if ingredients:
                    return ingredients[:20]  # Limit to 20 ingredients
        
        return []
    
    def _extract_nutrition_facts(self, text: str) -> Dict:
        """Enhanced nutrition facts extraction"""
        nutrition = {}
        text_lower = text.lower()
        
        # Comprehensive nutrition patterns
        patterns = {
            'energy_kcal': r'(?:energy|calories?)[:\s]*(\d+(?:\.\d+)?)\s*(?:kcal|cal)',
            'energy_kj': r'energy[:\s]*(\d+(?:\.\d+)?)\s*kj',
            'protein': r'protein[:\s]*(\d+(?:\.\d+)?)\s*g',
            'carbohydrates': r'(?:carbohydrate|carbs?|total\s+carb)[:\s]*(\d+(?:\.\d+)?)\s*g',
            'total_fat': r'(?:total\s+fat|fat)[:\s]*(\d+(?:\.\d+)?)\s*g',
            'saturated_fat': r'saturated\s+fat[:\s]*(\d+(?:\.\d+)?)\s*g',
            'trans_fat': r'trans\s+fat[:\s]*(\d+(?:\.\d+)?)\s*g',
            'fiber': r'(?:dietary\s+fiber|fiber)[:\s]*(\d+(?:\.\d+)?)\s*g',
            'sugar': r'(?:total\s+sugar|sugar)[:\s]*(\d+(?:\.\d+)?)\s*g',
            'sodium': r'sodium[:\s]*(\d+(?:\.\d+)?)\s*(?:mg|g)',
            'cholesterol': r'cholesterol[:\s]*(\d+(?:\.\d+)?)\s*mg'
        }
        
        for nutrient, pattern in patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                value = float(match.group(1))
                
                # Unit conversions
                if nutrient == 'sodium' and value > 10:
                    value = value / 1000  # mg to g
                elif nutrient == 'energy_kj':
                    nutrition['energy_kcal'] = value / 4.184  # kJ to kcal
                    continue
                
                nutrition[nutrient] = value
        
        return nutrition
    
    def _extract_fssai(self, text: str) -> str:
        """Extract FSSAI license number"""
        patterns = [
            r'fssai[:\s#-]*(\d{14})',
            r'lic[:\s#-]*no[:\s#-]*(\d{14})',
            r'license[:\s#-]*no[:\s#-]*(\d{14})',
            r'(\d{14})'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) == 14:
                    return match
        
        return ""
    
    def _extract_allergens(self, text: str) -> List[str]:
        """Extract allergen information"""
        allergen_patterns = [
            r'allergen[:\s]+(.*?)(?:\n|$)',
            r'contains[:\s]+(.*?)(?:\n|$)',
            r'may\s+contain[:\s]+(.*?)(?:\n|$)'
        ]
        
        common_allergens = ['milk', 'eggs', 'fish', 'shellfish', 'nuts', 'peanuts', 'wheat', 'soy', 'sesame']
        found_allergens = []
        
        text_lower = text.lower()
        
        # Check for explicit allergen statements
        for pattern in allergen_patterns:
            match = re.search(pattern, text_lower)
            if match:
                allergen_text = match.group(1)
                for allergen in common_allergens:
                    if allergen in allergen_text:
                        found_allergens.append(allergen.title())
        
        # Check for allergens in ingredients
        for allergen in common_allergens:
            if allergen in text_lower:
                found_allergens.append(allergen.title())
        
        return list(set(found_allergens))
    
    def _extract_net_weight(self, text: str) -> str:
        """Extract net weight/quantity"""
        patterns = [
            r'net\s+(?:wt|weight)[:\s]*(\d+(?:\.\d+)?)\s*(g|kg|ml|l)',
            r'quantity[:\s]*(\d+(?:\.\d+)?)\s*(g|kg|ml|l)',
            r'(\d+(?:\.\d+)?)\s*(g|kg|ml|l)(?:\s+net)?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return f"{match.group(1)} {match.group(2)}"
        
        return ""
    
    def _extract_dates(self, text: str, date_type: str) -> str:
        """Extract manufacturing or expiry dates"""
        if date_type == 'mfg':
            patterns = [r'mfg[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', r'manufactured[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})']
        else:
            patterns = [r'exp[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', r'best\s+before[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})']
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return ""
    
    def _extract_brand(self, text: str) -> str:
        """Extract brand name"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        for line in lines[:3]:
            if (len(line) > 2 and len(line) < 30 and 
                not re.search(r'\d+|ingredient|nutrition|fssai', line.lower())):
                return line
        
        return ""
    
    def _calculate_nutri_score(self, nutrition: Dict) -> Dict:
        """Calculate Nutri-Score (A-E rating)"""
        if not nutrition:
            return {'grade': 'Unknown', 'score': 0, 'explanation': 'No nutrition data available'}
        
        # Nutri-Score calculation (simplified)
        negative_points = 0
        positive_points = 0
        
        # Negative points (per 100g)
        if 'energy_kcal' in nutrition:
            energy = nutrition['energy_kcal']
            if energy <= 335: negative_points += 0
            elif energy <= 670: negative_points += 1
            elif energy <= 1005: negative_points += 2
            elif energy <= 1340: negative_points += 3
            elif energy <= 1675: negative_points += 4
            elif energy <= 2010: negative_points += 5
            elif energy <= 2345: negative_points += 6
            elif energy <= 2680: negative_points += 7
            elif energy <= 3015: negative_points += 8
            elif energy <= 3350: negative_points += 9
            else: negative_points += 10
        
        if 'saturated_fat' in nutrition:
            sat_fat = nutrition['saturated_fat']
            if sat_fat <= 1: negative_points += 0
            elif sat_fat <= 2: negative_points += 1
            elif sat_fat <= 3: negative_points += 2
            elif sat_fat <= 4: negative_points += 3
            elif sat_fat <= 5: negative_points += 4
            elif sat_fat <= 6: negative_points += 5
            elif sat_fat <= 7: negative_points += 6
            elif sat_fat <= 8: negative_points += 7
            elif sat_fat <= 9: negative_points += 8
            elif sat_fat <= 10: negative_points += 9
            else: negative_points += 10
        
        if 'sugar' in nutrition:
            sugar = nutrition['sugar']
            if sugar <= 4.5: negative_points += 0
            elif sugar <= 9: negative_points += 1
            elif sugar <= 13.5: negative_points += 2
            elif sugar <= 18: negative_points += 3
            elif sugar <= 22.5: negative_points += 4
            elif sugar <= 27: negative_points += 5
            elif sugar <= 31: negative_points += 6
            elif sugar <= 36: negative_points += 7
            elif sugar <= 40: negative_points += 8
            elif sugar <= 45: negative_points += 9
            else: negative_points += 10
        
        if 'sodium' in nutrition:
            sodium_mg = nutrition['sodium'] * 1000  # Convert g to mg
            if sodium_mg <= 90: negative_points += 0
            elif sodium_mg <= 180: negative_points += 1
            elif sodium_mg <= 270: negative_points += 2
            elif sodium_mg <= 360: negative_points += 3
            elif sodium_mg <= 450: negative_points += 4
            elif sodium_mg <= 540: negative_points += 5
            elif sodium_mg <= 630: negative_points += 6
            elif sodium_mg <= 720: negative_points += 7
            elif sodium_mg <= 810: negative_points += 8
            elif sodium_mg <= 900: negative_points += 9
            else: negative_points += 10
        
        # Positive points
        if 'fiber' in nutrition:
            fiber = nutrition['fiber']
            if fiber <= 0.9: positive_points += 0
            elif fiber <= 1.9: positive_points += 1
            elif fiber <= 2.8: positive_points += 2
            elif fiber <= 3.7: positive_points += 3
            elif fiber <= 4.7: positive_points += 4
            else: positive_points += 5
        
        if 'protein' in nutrition:
            protein = nutrition['protein']
            if protein <= 1.6: positive_points += 0
            elif protein <= 3.2: positive_points += 1
            elif protein <= 4.8: positive_points += 2
            elif protein <= 6.4: positive_points += 3
            elif protein <= 8.0: positive_points += 4
            else: positive_points += 5
        
        # Calculate final score
        final_score = negative_points - positive_points
        
        # Determine grade
        if final_score <= -1: grade = 'A'
        elif final_score <= 2: grade = 'B'
        elif final_score <= 10: grade = 'C'
        elif final_score <= 18: grade = 'D'
        else: grade = 'E'
        
        return {
            'grade': grade,
            'score': final_score,
            'negative_points': negative_points,
            'positive_points': positive_points,
            'explanation': f'Nutri-Score {grade} (Score: {final_score})'
        }

# Usage example
if __name__ == "__main__":
    ocr = EnhancedFoodOCR()
    result = ocr.extract_structured_data("path/to/food/label.jpg")
    print(result)