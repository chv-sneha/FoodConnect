#!/usr/bin/env python3
"""
Production Nutrition Service - OpenFoodFacts + USDA FoodData Central
Official Nutri-Score calculation with 2023 algorithm
"""

import requests
import re
from typing import Dict, List, Optional
import logging

class ProductionNutritionService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # API endpoints
        self.openfoodfacts_api = "https://world.openfoodfacts.org/api/v0/product"
        self.usda_api = "https://api.nal.usda.gov/fdc/v1"
        self.usda_api_key = os.getenv('USDA_API_KEY', 'DEMO_KEY')
        
        # Nutri-Score thresholds (official 2023 algorithm)
        self.nutri_score_thresholds = {
            'solid': {'A': -1, 'B': 2, 'C': 10, 'D': 18},
            'beverage': {'A': 1, 'B': 5, 'C': 9, 'D': 13},
            'cheese': {'A': -1, 'B': 2, 'C': 10, 'D': 18},
            'fat': {'A': -1, 'B': 2, 'C': 10, 'D': 18}
        }
    
    def lookup_product_by_barcode(self, barcode: str) -> Optional[Dict]:
        """Lookup product in OpenFoodFacts by barcode"""
        try:
            url = f"{self.openfoodfacts_api}/{barcode}.json"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 1:  # Product found
                    product = data.get('product', {})
                    
                    return {
                        'source': 'openfoodfacts',
                        'product_name': product.get('product_name', ''),
                        'ingredients_text': product.get('ingredients_text', ''),
                        'nutrition_per_100g': self._extract_nutrition_openfoodfacts(product),
                        'nutriscore_grade': product.get('nutriscore_grade', ''),
                        'categories': product.get('categories', ''),
                        'brands': product.get('brands', ''),
                        'confidence': 0.95
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"OpenFoodFacts lookup error: {e}")
            return None
    
    def _extract_nutrition_openfoodfacts(self, product: Dict) -> Dict:
        """Extract nutrition facts from OpenFoodFacts product"""
        nutrition = {}
        
        # Map OpenFoodFacts fields to our standard format
        nutrient_map = {
            'energy-kcal_100g': 'energy',
            'proteins_100g': 'protein',
            'carbohydrates_100g': 'carbohydrates',
            'fat_100g': 'total_fat',
            'saturated-fat_100g': 'saturated_fat',
            'trans-fat_100g': 'trans_fat',
            'fiber_100g': 'fiber',
            'sugars_100g': 'sugars',
            'salt_100g': 'salt',
            'sodium_100g': 'sodium'
        }
        
        nutriments = product.get('nutriments', {})
        
        for off_key, our_key in nutrient_map.items():
            value = nutriments.get(off_key)
            if value is not None:
                # Convert salt to sodium if needed
                if our_key == 'sodium' and off_key == 'salt_100g':
                    value = value * 0.4  # Salt to sodium conversion
                nutrition[our_key] = float(value)
        
        return nutrition
    
    def extract_nutrition_from_text(self, text: str) -> Dict:
        """Extract nutrition facts from OCR text using improved patterns"""
        nutrition = {}
        
        # Enhanced patterns for Indian food labels
        patterns = {
            'energy': [
                r'energy[:\s]*(\d+(?:\.\d+)?)\s*(?:kcal|cal|kj)',
                r'calories?[:\s]*(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*kcal'
            ],
            'protein': [
                r'protein[:\s]*(\d+(?:\.\d+)?)\s*g',
                r'proteins?[:\s]*(\d+(?:\.\d+)?)\s*g'
            ],
            'carbohydrates': [
                r'(?:carbohydrate|carbs?|total carbohydrate)[:\s]*(\d+(?:\.\d+)?)\s*g',
                r'carbs[:\s]*(\d+(?:\.\d+)?)\s*g'
            ],
            'total_fat': [
                r'(?:total\s+fat|fat)[:\s]*(\d+(?:\.\d+)?)\s*g',
                r'fats?[:\s]*(\d+(?:\.\d+)?)\s*g'
            ],
            'saturated_fat': [
                r'saturated\s+fat[:\s]*(\d+(?:\.\d+)?)\s*g',
                r'sat\.?\s*fat[:\s]*(\d+(?:\.\d+)?)\s*g'
            ],
            'trans_fat': [
                r'trans\s+fat[:\s]*(\d+(?:\.\d+)?)\s*g',
                r'trans[:\s]*(\d+(?:\.\d+)?)\s*g'
            ],
            'fiber': [
                r'(?:dietary\s+fiber|fiber|fibre)[:\s]*(\d+(?:\.\d+)?)\s*g'
            ],
            'sugars': [
                r'(?:total\s+sugars?|sugars?)[:\s]*(\d+(?:\.\d+)?)\s*g',
                r'sugar[:\s]*(\d+(?:\.\d+)?)\s*g'
            ],
            'sodium': [
                r'sodium[:\s]*(\d+(?:\.\d+)?)\s*(?:mg|g)',
                r'salt[:\s]*(\d+(?:\.\d+)?)\s*(?:mg|g)'
            ]
        }
        
        text_lower = text.lower()
        
        for nutrient, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, text_lower)
                if match:
                    value = float(match.group(1))
                    
                    # Convert units
                    if nutrient == 'sodium':
                        if 'salt' in pattern and value > 10:  # Likely salt in mg, convert to sodium
                            value = value * 0.4 / 1000  # Salt mg to sodium g
                        elif value > 10:  # Sodium in mg, convert to g
                            value = value / 1000
                    
                    nutrition[nutrient] = value
                    break  # Use first match for each nutrient
        
        return nutrition
    
    def calculate_nutri_score_official(self, nutrition: Dict, category: str = 'solid') -> Dict:
        """
        Calculate Nutri-Score using official 2023 algorithm
        Based on French ANSES specifications
        """
        
        # Get nutrition values (per 100g)
        energy = nutrition.get('energy', 0)
        sugars = nutrition.get('sugars', 0)
        saturated_fat = nutrition.get('saturated_fat', 0)
        sodium = nutrition.get('sodium', 0)  # in grams
        fiber = nutrition.get('fiber', 0)
        protein = nutrition.get('protein', 0)
        fruits_vegetables = nutrition.get('fruits_vegetables_nuts', 0)  # percentage
        
        # NEGATIVE POINTS
        negative_points = 0
        
        # Energy points (kJ per 100g) - convert kcal to kJ if needed
        energy_kj = energy * 4.184 if energy < 1000 else energy
        
        if energy_kj >= 3350: negative_points += 10
        elif energy_kj >= 2720: negative_points += 9
        elif energy_kj >= 2090: negative_points += 8
        elif energy_kj >= 1460: negative_points += 7
        elif energy_kj >= 830: negative_points += 6
        elif energy_kj >= 670: negative_points += 5
        elif energy_kj >= 510: negative_points += 4
        elif energy_kj >= 350: negative_points += 3
        elif energy_kj >= 190: negative_points += 2
        elif energy_kj >= 30: negative_points += 1
        
        # Saturated fat points (g per 100g)
        if saturated_fat >= 10: negative_points += 10
        elif saturated_fat >= 9: negative_points += 9
        elif saturated_fat >= 8: negative_points += 8
        elif saturated_fat >= 7: negative_points += 7
        elif saturated_fat >= 6: negative_points += 6
        elif saturated_fat >= 5: negative_points += 5
        elif saturated_fat >= 4: negative_points += 4
        elif saturated_fat >= 3: negative_points += 3
        elif saturated_fat >= 2: negative_points += 2
        elif saturated_fat >= 1: negative_points += 1
        
        # Sugars points (g per 100g)
        if sugars >= 45: negative_points += 10
        elif sugars >= 40: negative_points += 9
        elif sugars >= 36: negative_points += 8
        elif sugars >= 31: negative_points += 7
        elif sugars >= 27: negative_points += 6
        elif sugars >= 22.5: negative_points += 5
        elif sugars >= 18: negative_points += 4
        elif sugars >= 13.5: negative_points += 3
        elif sugars >= 9: negative_points += 2
        elif sugars >= 4.5: negative_points += 1
        
        # Sodium points (g per 100g)
        if sodium >= 1.8: negative_points += 10
        elif sodium >= 1.62: negative_points += 9
        elif sodium >= 1.44: negative_points += 8
        elif sodium >= 1.26: negative_points += 7
        elif sodium >= 1.08: negative_points += 6
        elif sodium >= 0.9: negative_points += 5
        elif sodium >= 0.72: negative_points += 4
        elif sodium >= 0.54: negative_points += 3
        elif sodium >= 0.36: negative_points += 2
        elif sodium >= 0.18: negative_points += 1
        
        # POSITIVE POINTS
        positive_points = 0
        
        # Fruits, vegetables, legumes, nuts points (%)
        if fruits_vegetables >= 80: positive_points += 5
        elif fruits_vegetables >= 60: positive_points += 2
        elif fruits_vegetables >= 40: positive_points += 1
        
        # Fiber points (g per 100g)
        if fiber >= 4.7: positive_points += 5
        elif fiber >= 3.7: positive_points += 4
        elif fiber >= 2.8: positive_points += 3
        elif fiber >= 1.9: positive_points += 2
        elif fiber >= 0.9: positive_points += 1
        
        # Protein points (g per 100g) - only if negative points < 11
        if negative_points < 11:
            if protein >= 8: positive_points += 5
            elif protein >= 6.4: positive_points += 4
            elif protein >= 4.8: positive_points += 3
            elif protein >= 3.2: positive_points += 2
            elif protein >= 1.6: positive_points += 1
        
        # Calculate final score
        final_score = negative_points - positive_points
        
        # Determine grade based on category
        thresholds = self.nutri_score_thresholds.get(category, self.nutri_score_thresholds['solid'])
        
        if final_score <= thresholds['A']:
            grade = 'A'
        elif final_score <= thresholds['B']:
            grade = 'B'
        elif final_score <= thresholds['C']:
            grade = 'C'
        elif final_score <= thresholds['D']:
            grade = 'D'
        else:
            grade = 'E'
        
        # Convert to 0-100 scale for display
        score_100 = max(0, min(100, 100 - (final_score + 15) * 3))
        
        return {
            'grade': grade,
            'score': int(score_100),
            'negative_points': negative_points,
            'positive_points': positive_points,
            'final_score': final_score,
            'category': category,
            'algorithm_version': '2023',
            'breakdown': {
                'energy_points': min(10, max(0, int(energy_kj / 335))),
                'sugar_points': min(10, max(0, int(sugars / 4.5))),
                'saturated_fat_points': min(10, int(saturated_fat)),
                'sodium_points': min(10, max(0, int(sodium / 0.18))),
                'fiber_points': min(5, max(0, int(fiber / 0.9))),
                'protein_points': min(5, max(0, int(protein / 1.6))) if negative_points < 11 else 0,
                'fruits_vegetables_points': positive_points - min(5, max(0, int(fiber / 0.9))) - (min(5, max(0, int(protein / 1.6))) if negative_points < 11 else 0)
            }
        }
    
    def detect_product_category(self, ingredients: List[str], product_name: str = "") -> str:
        """Detect product category for Nutri-Score calculation"""
        
        text = (product_name + " " + " ".join(ingredients)).lower()
        
        # Category detection rules
        if any(word in text for word in ['cheese', 'paneer', 'cheddar', 'mozzarella']):
            return 'cheese'
        
        if any(word in text for word in ['oil', 'butter', 'ghee', 'margarine']):
            return 'fat'
        
        if any(word in text for word in ['juice', 'drink', 'beverage', 'soda', 'cola']):
            return 'beverage'
        
        return 'solid'  # Default category