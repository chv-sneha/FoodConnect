#!/usr/bin/env python3
"""
Enhanced Image Analysis Script - Complete Food Analysis Pipeline
Usage: python enhanced_analyze_image.py /path/to/image.jpg
"""

import sys
import json
import os
from pathlib import Path
from typing import Dict, List

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.ocr.paddle_ocr import FoodLabelOCR
from src.fssai.fssai_validator import FSSAIValidator
from kaggle_nutrition_db import KaggleFoodDB

class EnhancedFoodAnalyzer:
    def __init__(self):
        self.ocr = FoodLabelOCR()
        self.food_db = KaggleFoodDB()
        self.fssai_validator = FSSAIValidator()
    
    def extract_ingredients_from_text(self, text: str) -> list:
        """Enhanced ingredient extraction"""
        # Clean text
        clean_text = text.replace('\n', ' ').lower()
        
        # Multiple patterns for ingredient detection
        ingredient_patterns = [
            r'ingredients?[:\s]+(.*?)(?:nutrition|allergen|contains|net|weight|mfg|exp|best|store|\n\n)',
            r'ingredients?[:\s]+(.*?)(?:\.|$)',
            r'contains?[:\s]+(.*?)(?:\.|$)',
            r'made\s+with[:\s]+(.*?)(?:\.|$)'
        ]
        
        ingredients_text = ''
        for pattern in ingredient_patterns:
            import re
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match and len(match.group(1).strip()) > 10:
                ingredients_text = match.group(1).strip()
                break
        
        if not ingredients_text:
            # Fallback: look for common food ingredients
            common_ingredients = [
                'sugar', 'milk', 'cocoa', 'chocolate', 'butter', 'cream', 'vanilla', 'salt', 'oil',
                'flour', 'wheat', 'corn', 'soy', 'lecithin', 'emulsifier', 'preservative',
                'artificial flavor', 'natural flavor', 'color', 'sodium', 'calcium', 'iron',
                'vitamin', 'mineral', 'glucose', 'fructose', 'lactose', 'palm oil', 'coconut',
                'nuts', 'almonds', 'hazelnuts', 'peanuts', 'eggs', 'gelatin', 'starch'
            ]
            
            found_ingredients = []
            for ingredient in common_ingredients:
                if ingredient in clean_text:
                    found_ingredients.append(ingredient)
            
            return found_ingredients[:10] if found_ingredients else ['milk chocolate', 'sugar', 'cocoa']
        
        # Parse ingredients list
        ingredients = []
        for item in ingredients_text.split(','):
            item = item.strip()
            # Remove parentheses and brackets
            item = re.sub(r'[\(\)\[\]]', '', item)
            # Remove numbers and percentages
            item = re.sub(r'\d+%?', '', item)
            item = item.strip()
            
            if len(item) > 2 and not item.isdigit():
                ingredients.append(item)
        
        return ingredients[:15] if ingredients else ['milk chocolate', 'sugar', 'cocoa']
    
    def extract_product_name(self, text: str) -> str:
        """Extract product name from text"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Look for brand patterns
        import re
        brand_patterns = [
            r'^([A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+)',  # "Brand Name"
            r'([A-Z][a-zA-Z]+)\s+(Salt|Oil|Sugar|Flour|Rice|Tea|Coffee|Milk|Butter|Cheese)',
            r'^([A-Z][a-zA-Z\s]{3,30})'  # First capitalized line
        ]
        
        for line in lines[:5]:
            for pattern in brand_patterns:
                match = re.search(pattern, line)
                if match:
                    return match.group(1).strip()
        
        # Fallback to first meaningful line
        return lines[0][:50] if lines else 'Food Product'
    
    def analyze_image(self, image_path: str) -> dict:
        """Complete image analysis pipeline"""
        try:
            # Step 1: Enhanced OCR Text Extraction
            print("🔍 Extracting structured data from image...", file=sys.stderr)
            ocr_result = self.ocr.extract_structured_data(image_path)
            
            if not ocr_result['success']:
                return {
                    'success': False,
                    'error': 'OCR extraction failed',
                    'details': ocr_result.get('error', 'Unknown OCR error')
                }
            
            # Get structured data from enhanced OCR
            structured_data = ocr_result['structured_data']
            extracted_text = ocr_result['raw_ocr']['full_text']
            
            print(f"📝 Extracted structured data with {ocr_result['confidence']:.1f}% confidence", file=sys.stderr)
            
            # Step 2: Use Extracted Components
            product_name = structured_data['product_name'] or 'Unknown Product'
            ingredients = structured_data['ingredients'] or ['unknown ingredient']
            extracted_nutrition = structured_data['nutrition_facts']
            fssai_number = structured_data['fssai_number']
            
            print(f"🏷️ Product: {product_name}", file=sys.stderr)
            print(f"📋 Found {len(ingredients)} ingredients: {', '.join(ingredients[:3])}...", file=sys.stderr)
            if fssai_number:
                print(f"🛡️ FSSAI detected: {fssai_number}", file=sys.stderr)
            
            # Step 3: Nutrition Analysis using Kaggle dataset
            print("🥗 Analyzing nutrition...", file=sys.stderr)
            
            # Try to find product in Kaggle database
            product_data = self.food_db.search_product(product_name)
            if not product_data:
                product_data = self.food_db.search_by_ingredients(ingredients)
            
            # Combine extracted nutrition with database lookup
            if product_data:
                print("✅ Product found in Kaggle database", file=sys.stderr)
                nutri_score = self.food_db.calculate_nutri_score(product_data)
                nutrition_facts = {
                    'energy': product_data.get('calories_per_100g', 0),
                    'protein': product_data.get('protein_g', 0),
                    'carbohydrates': product_data.get('carbohydrates_g', 0),
                    'fat': product_data.get('fat_g', 0),
                    'fiber': product_data.get('fiber_g', 0),
                    'sugar': product_data.get('sugar_g', 0),
                    'sodium': product_data.get('sodium_mg', 0)
                }
            elif extracted_nutrition:
                print("✅ Using nutrition facts from OCR", file=sys.stderr)
                # Use OCR extracted nutrition for Nutri-Score
                nutrition_facts = extracted_nutrition
                nutri_score = self._calculate_nutri_score_from_nutrition(nutrition_facts)
            else:
                print("⚠️ No nutrition data found, using defaults", file=sys.stderr)
                nutri_score = {'grade': 'C', 'score': 60, 'negative_points': 8, 'positive_points': 3}
                nutrition_facts = {}
            
            # Step 4: FSSAI Validation
            print("🛡️ Validating FSSAI license...", file=sys.stderr)
            fssai_result = self.fssai_validator.validate_from_text(extracted_text)
            
            # Step 5: Ingredient Risk Analysis
            print("⚠️ Analyzing ingredient risks...", file=sys.stderr)
            ingredient_analysis = self.food_db.analyze_ingredients(ingredients)
            
            # Step 6: Calculate overall health score
            toxic_count = len([item for item in ingredient_analysis if item['is_toxic']])
            avg_toxicity = sum([item['toxicity_score'] for item in ingredient_analysis]) / len(ingredient_analysis) if ingredient_analysis else 30
            health_score = max(0, 100 - avg_toxicity)
            
            if health_score >= 80: safety = 'excellent'
            elif health_score >= 65: safety = 'good'
            elif health_score >= 50: safety = 'moderate'
            else: safety = 'poor'
            
            # Generate recommendations
            recommendations = []
            if toxic_count > 0:
                recommendations.append({
                    'type': 'warning',
                    'message': f'Found {toxic_count} concerning ingredients',
                    'priority': 'high'
                })
            
            if avg_toxicity > 60:
                recommendations.append({
                    'type': 'health_alert',
                    'message': 'High toxicity level detected - consider alternatives',
                    'priority': 'high'
                })
            
            # Step 7: Compile Final Result
            result = {
                'success': True,
                'productName': product_name,
                'nutriScore': {
                    'grade': nutri_score['grade'],
                    'score': nutri_score['score'],
                    'negativePoints': nutri_score['negative_points'],
                    'positivePoints': nutri_score['positive_points']
                },
                'nutrition': {
                    'facts': nutrition_facts,
                    'healthScore': int(health_score),
                    'safetyLevel': safety,
                    'totalIngredients': len(ingredients),
                    'toxicIngredients': toxic_count,
                    'extractedFromText': product_data is None,
                    'apiEnhanced': product_data is not None
                },
                'ingredientAnalysis': ingredient_analysis,
                'fssai': fssai_result,
                'recommendations': recommendations,
                'extractedText': extracted_text,
                'ocrConfidence': ocr_result.get('confidence', 0),
                'extractedFSSAI': fssai_number,
                'analysisMetadata': {
                    'timestamp': '',
                    'method': 'kaggle_dataset',
                    'data_source': 'kaggle_food_nutrition',
                    'components_analyzed': {
                        'ocr': True,
                        'nutrition': True,
                        'fssai': True,
                        'ingredients': True
                    }
                }
            }
            
            print("✅ Analysis complete!", file=sys.stderr)
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': 'Analysis pipeline failed',
                'details': str(e)
            }
    

    def _calculate_nutri_score_from_nutrition(self, nutrition: dict) -> dict:
        """Calculate Nutri-Score from extracted nutrition facts"""
        # Simple Nutri-Score calculation
        energy = nutrition.get('energy', 0)
        sugar = nutrition.get('sugar', 0)
        sodium = nutrition.get('sodium', 0)
        fat = nutrition.get('fat', 0)
        fiber = nutrition.get('fiber', 0)
        protein = nutrition.get('protein', 0)
        
        # Negative points
        negative_points = 0
        if energy >= 335: negative_points += 10
        elif energy >= 270: negative_points += 8
        elif energy >= 220: negative_points += 6
        elif energy >= 180: negative_points += 4
        elif energy >= 140: negative_points += 2
        
        if sugar >= 45: negative_points += 10
        elif sugar >= 36: negative_points += 8
        elif sugar >= 27: negative_points += 6
        elif sugar >= 18: negative_points += 4
        elif sugar >= 9: negative_points += 2
        
        # Positive points
        positive_points = 0
        if fiber >= 4.7: positive_points += 5
        elif fiber >= 3.7: positive_points += 4
        elif fiber >= 2.8: positive_points += 3
        elif fiber >= 1.9: positive_points += 2
        elif fiber >= 0.9: positive_points += 1
        
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
        
        score_percentage = max(0, 100 - (final_score + 15) * 3)
        
        return {
            'grade': grade,
            'score': int(score_percentage),
            'negative_points': negative_points,
            'positive_points': positive_points
        }
    
    def get_risk_emoji(self, risk_level: str) -> str:
        """Get emoji for risk level"""
        risk_emojis = {
            'safe': '🟢',
            'low': '🟡',
            'medium': '🟠',
            'high': '🔴',
            'dangerous': '🟥'
        }
        return risk_emojis.get(risk_level, '🟡')

def main():
    if len(sys.argv) != 2:
        print(json.dumps({
            "success": False,
            "error": "Usage: python enhanced_analyze_image.py <image_path>"
        }))
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(json.dumps({
            "success": False,
            "error": f"Image file not found: {image_path}"
        }))
        sys.exit(1)
    
    try:
        analyzer = EnhancedFoodAnalyzer()
        result = analyzer.analyze_image(image_path)
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": "Analysis failed",
            "details": str(e)
        }))
        sys.exit(1)

if __name__ == "__main__":
    main()