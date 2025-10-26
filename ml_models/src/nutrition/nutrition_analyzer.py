#!/usr/bin/env python3
"""
Real Nutrition Analysis using multiple APIs
- USDA FoodData Central API
- Edamam Nutrition Analysis API
- Spoonacular API
"""

import requests
import json
import re
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class NutritionFacts:
    calories: float = 0
    protein: float = 0
    carbs: float = 0
    fat: float = 0
    fiber: float = 0
    sugar: float = 0
    sodium: float = 0
    calcium: float = 0
    iron: float = 0
    vitamin_c: float = 0

class NutritionAnalyzer:
    def __init__(self):
        # API Keys (add to environment variables)
        self.usda_api_key = "DEMO_KEY"  # Replace with real key
        self.edamam_app_id = "your_app_id"  # Replace with real credentials
        self.edamam_app_key = "your_app_key"
        self.spoonacular_key = "your_spoonacular_key"
        
        # API Endpoints
        self.usda_base_url = "https://api.nal.usda.gov/fdc/v1"
        self.edamam_base_url = "https://api.edamam.com/api/nutrition-details"
        self.spoonacular_base_url = "https://api.spoonacular.com/recipes"
    
    def extract_nutrition_from_text(self, text: str) -> Dict:
        """Extract nutrition facts from OCR text"""
        nutrition = NutritionFacts()
        
        # Patterns for nutrition extraction
        patterns = {
            'calories': r'calories?[:\s]*(\d+(?:\.\d+)?)',
            'protein': r'protein[:\s]*(\d+(?:\.\d+)?)\s*g',
            'carbs': r'(?:carbohydrate|carbs?)[:\s]*(\d+(?:\.\d+)?)\s*g',
            'fat': r'(?:total\s+)?fat[:\s]*(\d+(?:\.\d+)?)\s*g',
            'fiber': r'(?:dietary\s+)?fiber[:\s]*(\d+(?:\.\d+)?)\s*g',
            'sugar': r'sugar[s]?[:\s]*(\d+(?:\.\d+)?)\s*g',
            'sodium': r'sodium[:\s]*(\d+(?:\.\d+)?)\s*(?:mg|g)',
            'calcium': r'calcium[:\s]*(\d+(?:\.\d+)?)\s*(?:mg|g)',
            'iron': r'iron[:\s]*(\d+(?:\.\d+)?)\s*(?:mg|g)',
            'vitamin_c': r'vitamin\s*c[:\s]*(\d+(?:\.\d+)?)\s*(?:mg|g)'
        }
        
        text_lower = text.lower()
        extracted_values = {}
        
        for nutrient, pattern in patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                value = float(match.group(1))
                extracted_values[nutrient] = value
                setattr(nutrition, nutrient, value)
        
        return {
            'nutrition_facts': nutrition.__dict__,
            'extracted_values': extracted_values,
            'confidence': len(extracted_values) / len(patterns)
        }
    
    def get_usda_nutrition(self, food_name: str) -> Optional[Dict]:
        """Get nutrition data from USDA FoodData Central"""
        try:
            # Search for food
            search_url = f"{self.usda_base_url}/foods/search"
            params = {
                'query': food_name,
                'api_key': self.usda_api_key,
                'pageSize': 5
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            if response.status_code != 200:
                return None
            
            data = response.json()
            if not data.get('foods'):
                return None
            
            # Get detailed nutrition for first result
            food_id = data['foods'][0]['fdcId']
            detail_url = f"{self.usda_base_url}/food/{food_id}"
            detail_params = {'api_key': self.usda_api_key}
            
            detail_response = requests.get(detail_url, params=detail_params, timeout=10)
            if detail_response.status_code != 200:
                return None
            
            food_data = detail_response.json()
            
            # Parse nutrition data
            nutrition = {}
            nutrient_map = {
                'Energy': 'calories',
                'Protein': 'protein',
                'Carbohydrate, by difference': 'carbs',
                'Total lipid (fat)': 'fat',
                'Fiber, total dietary': 'fiber',
                'Sugars, total including NLEA': 'sugar',
                'Sodium, Na': 'sodium',
                'Calcium, Ca': 'calcium',
                'Iron, Fe': 'iron',
                'Vitamin C, total ascorbic acid': 'vitamin_c'
            }
            
            for nutrient in food_data.get('foodNutrients', []):
                nutrient_name = nutrient.get('nutrient', {}).get('name', '')
                if nutrient_name in nutrient_map:
                    nutrition[nutrient_map[nutrient_name]] = nutrient.get('amount', 0)
            
            return {
                'source': 'USDA',
                'food_name': food_data.get('description', food_name),
                'nutrition': nutrition,
                'confidence': 0.9
            }
            
        except Exception as e:
            print(f"USDA API error: {e}")
            return None
    
    def analyze_ingredients_nutrition(self, ingredients: List[str]) -> Dict:
        """Analyze nutrition for list of ingredients"""
        total_nutrition = NutritionFacts()
        ingredient_details = []
        
        for ingredient in ingredients[:10]:  # Limit to prevent API overuse
            # Clean ingredient name
            clean_ingredient = self.clean_ingredient_name(ingredient)
            
            # Get nutrition data
            nutrition_data = self.get_usda_nutrition(clean_ingredient)
            
            if nutrition_data:
                ingredient_details.append({
                    'ingredient': ingredient,
                    'nutrition_data': nutrition_data,
                    'found': True
                })
                
                # Add to total (simplified - assumes equal portions)
                for key, value in nutrition_data['nutrition'].items():
                    if hasattr(total_nutrition, key):
                        current_value = getattr(total_nutrition, key)
                        setattr(total_nutrition, key, current_value + (value * 0.1))  # 10% contribution
            else:
                ingredient_details.append({
                    'ingredient': ingredient,
                    'nutrition_data': None,
                    'found': False
                })
        
        return {
            'total_nutrition': total_nutrition.__dict__,
            'ingredient_details': ingredient_details,
            'analyzed_count': len([d for d in ingredient_details if d['found']]),
            'total_ingredients': len(ingredients)
        }
    
    def clean_ingredient_name(self, ingredient: str) -> str:
        """Clean ingredient name for API search"""
        # Remove common prefixes/suffixes
        ingredient = re.sub(r'\([^)]*\)', '', ingredient)  # Remove parentheses
        ingredient = re.sub(r'\b(?:organic|natural|artificial|added|contains?)\b', '', ingredient, flags=re.IGNORECASE)
        ingredient = ingredient.strip().lower()
        
        # Handle common ingredient mappings
        mappings = {
            'palm oil': 'palm oil',
            'cocoa butter': 'cocoa butter',
            'milk chocolate': 'milk chocolate',
            'corn syrup': 'corn syrup',
            'high fructose corn syrup': 'corn syrup',
            'monosodium glutamate': 'msg',
            'sodium benzoate': 'sodium benzoate'
        }
        
        for key, value in mappings.items():
            if key in ingredient:
                return value
        
        return ingredient
    
    def calculate_nutri_score(self, nutrition: Dict) -> Dict:
        """Calculate Nutri-Score (A-E rating)"""
        # Nutri-Score algorithm (simplified)
        negative_points = 0
        positive_points = 0
        
        # Negative points (per 100g)
        calories = nutrition.get('calories', 0)
        sugar = nutrition.get('sugar', 0)
        sodium = nutrition.get('sodium', 0)
        fat = nutrition.get('fat', 0)
        
        # Calories points
        if calories > 335: negative_points += 10
        elif calories > 270: negative_points += 8
        elif calories > 220: negative_points += 6
        elif calories > 180: negative_points += 4
        elif calories > 140: negative_points += 2
        
        # Sugar points
        if sugar > 45: negative_points += 10
        elif sugar > 36: negative_points += 8
        elif sugar > 27: negative_points += 6
        elif sugar > 18: negative_points += 4
        elif sugar > 9: negative_points += 2
        
        # Sodium points (convert mg to g)
        sodium_g = sodium / 1000 if sodium > 100 else sodium
        if sodium_g > 1.8: negative_points += 10
        elif sodium_g > 1.44: negative_points += 8
        elif sodium_g > 1.08: negative_points += 6
        elif sodium_g > 0.72: negative_points += 4
        elif sodium_g > 0.36: negative_points += 2
        
        # Fat points
        if fat > 20: negative_points += 10
        elif fat > 16: negative_points += 8
        elif fat > 12: negative_points += 6
        elif fat > 8: negative_points += 4
        elif fat > 4: negative_points += 2
        
        # Positive points
        fiber = nutrition.get('fiber', 0)
        protein = nutrition.get('protein', 0)
        
        # Fiber points
        if fiber > 4.7: positive_points += 5
        elif fiber > 3.7: positive_points += 4
        elif fiber > 2.8: positive_points += 3
        elif fiber > 1.9: positive_points += 2
        elif fiber > 0.9: positive_points += 1
        
        # Protein points
        if protein > 8: positive_points += 5
        elif protein > 6.4: positive_points += 4
        elif protein > 4.8: positive_points += 3
        elif protein > 3.2: positive_points += 2
        elif protein > 1.6: positive_points += 1
        
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
            'score': max(0, 100 - (final_score * 5)),  # Convert to 0-100 scale
            'negative_points': negative_points,
            'positive_points': positive_points,
            'final_score': final_score
        }

# Test function
if __name__ == "__main__":
    analyzer = NutritionAnalyzer()
    
    # Test with sample text
    sample_text = """
    Nutrition Facts
    Calories: 250
    Protein: 8g
    Carbohydrates: 35g
    Fat: 12g
    Fiber: 3g
    Sugar: 15g
    Sodium: 400mg
    """
    
    result = analyzer.extract_nutrition_from_text(sample_text)
    print("✅ Nutrition extraction test:")
    print(json.dumps(result, indent=2))
    
    # Test Nutri-Score calculation
    nutri_score = analyzer.calculate_nutri_score(result['nutrition_facts'])
    print("\n✅ Nutri-Score calculation:")
    print(json.dumps(nutri_score, indent=2))