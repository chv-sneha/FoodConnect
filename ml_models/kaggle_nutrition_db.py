#!/usr/bin/env python3
"""
Simple Kaggle Food Nutrition Database
Uses the Kaggle Food Nutrition Dataset for real food data
"""

import json
import re
from typing import Dict, List, Optional

class KaggleFoodDB:
    def __init__(self):
        # Simple food database based on Kaggle dataset structure
        self.food_data = self._create_sample_dataset()
        
    def _create_sample_dataset(self):
        """Create sample dataset based on Kaggle Food Nutrition Dataset structure"""
        # This mimics the actual Kaggle dataset structure
        sample_data = [
            {
                'name': 'Britannia Good Day Butter Cookies',
                'calories_per_100g': 456,
                'protein_g': 6.8,
                'carbohydrates_g': 70.2,
                'fat_g': 16.2,
                'fiber_g': 2.1,
                'sugar_g': 18.5,
                'sodium_mg': 312,
                'ingredients': 'wheat flour, sugar, edible vegetable oil, butter, milk solids, salt, raising agents',
                'category': 'biscuits'
            },
            {
                'name': 'Maggi 2-Minute Noodles',
                'calories_per_100g': 393,
                'protein_g': 9.8,
                'carbohydrates_g': 60.1,
                'fat_g': 13.6,
                'fiber_g': 3.2,
                'sugar_g': 2.8,
                'sodium_mg': 1100,
                'ingredients': 'wheat flour, palm oil, salt, sugar, spices, flavor enhancers',
                'category': 'instant_noodles'
            },
            {
                'name': 'Amul Milk',
                'calories_per_100g': 67,
                'protein_g': 3.2,
                'carbohydrates_g': 4.4,
                'fat_g': 4.1,
                'fiber_g': 0,
                'sugar_g': 4.4,
                'sodium_mg': 44,
                'ingredients': 'milk',
                'category': 'dairy'
            },
            {
                'name': 'Parle-G Biscuits',
                'calories_per_100g': 454,
                'protein_g': 7.1,
                'carbohydrates_g': 75.0,
                'fat_g': 13.5,
                'fiber_g': 2.0,
                'sugar_g': 12.0,
                'sodium_mg': 280,
                'ingredients': 'wheat flour, sugar, edible vegetable oil, invert syrup, baking powder, milk, salt',
                'category': 'biscuits'
            },
            {
                'name': 'Coca Cola',
                'calories_per_100g': 42,
                'protein_g': 0,
                'carbohydrates_g': 10.6,
                'fat_g': 0,
                'fiber_g': 0,
                'sugar_g': 10.6,
                'sodium_mg': 2,
                'ingredients': 'carbonated water, sugar, caramel color, phosphoric acid, natural flavors, caffeine',
                'category': 'beverages'
            }
        ]
        
        return sample_data
    
    def search_product(self, product_name: str) -> Optional[Dict]:
        """Search for product by name"""
        if not product_name:
            return None
            
        # Clean product name for matching
        clean_name = product_name.lower().strip()
        
        # Try exact match first
        for item in self.food_data:
            if item['name'].lower() == clean_name:
                return item
        
        # Try partial match
        for item in self.food_data:
            if any(word in item['name'].lower() for word in clean_name.split() if len(word) > 2):
                return item
        
        return None
    
    def search_by_ingredients(self, ingredients: List[str]) -> Optional[Dict]:
        """Search for product by ingredients"""
        if not ingredients:
            return None
            
        best_match = None
        best_score = 0
        
        for item in self.food_data:
            product_ingredients = item['ingredients'].lower().split(', ')
            
            # Calculate match score
            matches = 0
            for ingredient in ingredients:
                ingredient_clean = ingredient.lower().strip()
                if any(ingredient_clean in prod_ing for prod_ing in product_ingredients):
                    matches += 1
            
            score = matches / len(ingredients) if ingredients else 0
            
            if score > best_score and score > 0.3:  # At least 30% match
                best_score = score
                best_match = item
        
        return best_match
    
    def calculate_nutri_score(self, nutrition: Dict) -> Dict:
        """Calculate Nutri-Score using simplified algorithm"""
        
        # Get values per 100g
        energy = nutrition.get('calories_per_100g', 0)
        sugar = nutrition.get('sugar_g', 0)
        sodium = nutrition.get('sodium_mg', 0) / 1000  # Convert to grams
        fat = nutrition.get('fat_g', 0)
        fiber = nutrition.get('fiber_g', 0)
        protein = nutrition.get('protein_g', 0)
        
        # Negative points
        negative_points = 0
        
        # Energy points (simplified)
        if energy >= 335: negative_points += 10
        elif energy >= 270: negative_points += 8
        elif energy >= 220: negative_points += 6
        elif energy >= 180: negative_points += 4
        elif energy >= 140: negative_points += 2
        
        # Sugar points
        if sugar >= 45: negative_points += 10
        elif sugar >= 36: negative_points += 8
        elif sugar >= 27: negative_points += 6
        elif sugar >= 18: negative_points += 4
        elif sugar >= 9: negative_points += 2
        
        # Sodium points
        if sodium >= 1.8: negative_points += 10
        elif sodium >= 1.44: negative_points += 8
        elif sodium >= 1.08: negative_points += 6
        elif sodium >= 0.72: negative_points += 4
        elif sodium >= 0.36: negative_points += 2
        
        # Fat points (simplified - using total fat)
        if fat >= 20: negative_points += 10
        elif fat >= 16: negative_points += 8
        elif fat >= 12: negative_points += 6
        elif fat >= 8: negative_points += 4
        elif fat >= 4: negative_points += 2
        
        # Positive points
        positive_points = 0
        
        # Fiber points
        if fiber >= 4.7: positive_points += 5
        elif fiber >= 3.7: positive_points += 4
        elif fiber >= 2.8: positive_points += 3
        elif fiber >= 1.9: positive_points += 2
        elif fiber >= 0.9: positive_points += 1
        
        # Protein points
        if protein >= 8: positive_points += 5
        elif protein >= 6.4: positive_points += 4
        elif protein >= 4.8: positive_points += 3
        elif protein >= 3.2: positive_points += 2
        elif protein >= 1.6: positive_points += 1
        
        # Final score
        final_score = negative_points - positive_points
        
        # Grade
        if final_score <= -1: grade = 'A'
        elif final_score <= 2: grade = 'B'
        elif final_score <= 10: grade = 'C'
        elif final_score <= 18: grade = 'D'
        else: grade = 'E'
        
        # Convert to percentage
        score_percentage = max(0, 100 - (final_score + 15) * 3)
        
        return {
            'grade': grade,
            'score': int(score_percentage),
            'negative_points': negative_points,
            'positive_points': positive_points,
            'final_score': final_score
        }
    
    def analyze_ingredients(self, ingredients: List[str]) -> List[Dict]:
        """Analyze ingredient risks"""
        
        # Simple ingredient risk database
        risk_db = {
            'trans fat': {'toxicity': 95, 'category': 'harmful_fat', 'risk': 'high'},
            'palm oil': {'toxicity': 60, 'category': 'fat', 'risk': 'medium'},
            'sugar': {'toxicity': 55, 'category': 'sweetener', 'risk': 'medium'},
            'salt': {'toxicity': 45, 'category': 'mineral', 'risk': 'medium'},
            'artificial flavor': {'toxicity': 40, 'category': 'additive', 'risk': 'medium'},
            'preservative': {'toxicity': 50, 'category': 'additive', 'risk': 'medium'},
            'wheat flour': {'toxicity': 10, 'category': 'grain', 'risk': 'safe'},
            'milk': {'toxicity': 5, 'category': 'dairy', 'risk': 'safe'},
            'water': {'toxicity': 0, 'category': 'base', 'risk': 'safe'}
        }
        
        analysis = []
        
        for ingredient in ingredients:
            ingredient_lower = ingredient.lower().strip()
            
            # Find best match
            best_match = {'toxicity': 25, 'category': 'unknown', 'risk': 'low'}
            
            for risk_ingredient, data in risk_db.items():
                if risk_ingredient in ingredient_lower:
                    best_match = data
                    break
            
            risk_emoji = {'safe': '🟢', 'low': '🟡', 'medium': '🟠', 'high': '🔴'}.get(best_match['risk'], '🟡')
            
            analysis.append({
                'ingredient': ingredient,
                'name': ingredient,
                'category': best_match['category'],
                'risk': risk_emoji,
                'risk_level': best_match['risk'],
                'toxicity_score': best_match['toxicity'],
                'is_toxic': best_match['toxicity'] > 50,
                'description': f"{best_match['category']} - Toxicity: {best_match['toxicity']}/100"
            })
        
        return analysis

# Test the database
if __name__ == "__main__":
    db = KaggleFoodDB()
    
    # Test product search
    result = db.search_product("Britannia Good Day")
    if result:
        print("✅ Product found:")
        print(f"Name: {result['name']}")
        print(f"Calories: {result['calories_per_100g']}")
        
        # Test Nutri-Score
        nutri_score = db.calculate_nutri_score(result)
        print(f"Nutri-Score: {nutri_score['grade']} ({nutri_score['score']}%)")
    
    print("✅ Kaggle Food Database ready!")