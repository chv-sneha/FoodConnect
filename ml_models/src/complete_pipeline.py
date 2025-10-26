#!/usr/bin/env python3
"""
Complete Pipeline: OCR -> Text Classification -> Nutrition Analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_ocr import EnhancedFoodOCR
from ingredient_classifier import IngredientTextClassifier
from nutrition_analyzer import NutritionAnalyzer
import re
from typing import Dict, List

class FoodAnalysisPipeline:
    def __init__(self, kaggle_dataset_path: str = None):
        # Initialize components
        self.ocr = EnhancedFoodOCR()
        self.classifier = IngredientTextClassifier()
        self.analyzer = NutritionAnalyzer(kaggle_dataset_path)
        
        # Train classifier if not already trained
        if not self.classifier.is_trained:
            print("Training ingredient classifier...")
            self.classifier.train()
        
        # Load nutrition dataset
        print("Loading nutrition dataset...")
        self.analyzer.load_kaggle_dataset()
    
    def analyze_food_image(self, image_path: str) -> Dict:
        """Complete analysis pipeline for food label image"""
        
        results = {
            'success': False,
            'ocr_results': {},
            'ingredient_extraction': {},
            'nutrition_analysis': {},
            'final_score': 0,
            'recommendations': []
        }
        
        try:
            # Step 1: OCR Text Extraction
            print("Step 1: Extracting text from image...")
            ocr_results = self.ocr.extract_with_ensemble(image_path)
            results['ocr_results'] = ocr_results
            
            if not ocr_results['success']:
                results['error'] = "OCR failed: " + ocr_results.get('error', 'Unknown error')
                return results
            
            # Step 2: Classify and Extract Ingredient Text
            print("Step 2: Identifying ingredient text...")
            ingredient_text = self.classifier.extract_ingredient_text(ocr_results['text_blocks'])
            
            if not ingredient_text:
                results['error'] = "No ingredient text found in image"
                return results
            
            # Step 3: Parse Individual Ingredients
            print("Step 3: Parsing ingredients...")
            ingredients = self._parse_ingredients(ingredient_text)
            
            results['ingredient_extraction'] = {
                'raw_text': ingredient_text,
                'parsed_ingredients': ingredients,
                'count': len(ingredients)
            }
            
            # Step 4: Nutrition Analysis
            print("Step 4: Analyzing nutrition...")
            nutrition_analysis = self.analyzer.analyze_ingredients(ingredients)
            results['nutrition_analysis'] = nutrition_analysis
            
            # Step 5: Generate Final Score and Recommendations
            print("Step 5: Generating recommendations...")
            final_score, recommendations = self._generate_final_assessment(nutrition_analysis)
            
            results['final_score'] = final_score
            results['recommendations'] = recommendations
            results['success'] = True
            
            return results
            
        except Exception as e:
            results['error'] = f"Pipeline error: {str(e)}"
            return results
    
    def _parse_ingredients(self, ingredient_text: str) -> List[str]:
        """Parse ingredient text into individual ingredients"""
        
        # Remove "Ingredients:" prefix
        text = re.sub(r'^ingredients?[:\s]*', '', ingredient_text, flags=re.IGNORECASE)
        
        # Split by common separators
        ingredients = re.split(r'[,;]', text)
        
        # Clean each ingredient
        cleaned_ingredients = []
        for ingredient in ingredients:
            # Remove percentages and parentheses
            ingredient = re.sub(r'\([^)]*\)', '', ingredient)
            ingredient = re.sub(r'\d+\.?\d*\s*%', '', ingredient)
            
            # Clean whitespace and special characters
            ingredient = re.sub(r'[^\w\s]', '', ingredient)
            ingredient = ingredient.strip()
            
            # Filter out very short or empty ingredients
            if len(ingredient) > 2 and not ingredient.isdigit():
                cleaned_ingredients.append(ingredient.title())
        
        return cleaned_ingredients[:15]  # Limit to 15 ingredients
    
    def _generate_final_assessment(self, nutrition_analysis: Dict) -> tuple:
        """Generate final health score and recommendations"""
        
        base_score = nutrition_analysis['health_score']
        recommendations = []
        
        # Adjust score based on match coverage
        match_coverage = nutrition_analysis.get('match_coverage', 0)
        if match_coverage < 50:
            base_score *= 0.8  # Reduce score if many ingredients unmatched
            recommendations.append("⚠️ Many ingredients could not be analyzed - consider products with clearer labels")
        
        # Generate specific recommendations based on nutrition
        nutrition = nutrition_analysis['total_nutrition']
        
        if nutrition['calories_per_100g'] > 400:
            recommendations.append("🔥 High calorie product - consume in moderation")
        
        if nutrition['sugar_g'] > 15:
            recommendations.append("🍯 High sugar content - limit intake, especially for diabetics")
        
        if nutrition['sodium_mg'] > 800:
            recommendations.append("🧂 High sodium - not suitable for high blood pressure")
        
        if nutrition['fat_g'] > 25:
            recommendations.append("🥑 High fat content - consider low-fat alternatives")
        
        if nutrition['fiber_g'] < 2:
            recommendations.append("🌾 Low fiber - add fruits/vegetables to your meal")
        
        # Positive recommendations
        if nutrition['protein_g'] > 8:
            recommendations.append("💪 Good protein source")
        
        if nutrition['fiber_g'] > 5:
            recommendations.append("🌾 High fiber - good for digestion")
        
        # Score-based recommendations
        if base_score >= 80:
            recommendations.append("✅ Healthy choice - good nutritional profile")
        elif base_score >= 60:
            recommendations.append("⚠️ Moderate choice - consume occasionally")
        else:
            recommendations.append("❌ Poor nutritional profile - consider healthier alternatives")
        
        return round(base_score, 1), recommendations
    
    def get_nutri_score_grade(self, nutrition: Dict) -> str:
        """Calculate Nutri-Score grade (A-E)"""
        
        # Simplified Nutri-Score calculation
        negative_points = 0
        positive_points = 0
        
        # Negative points
        calories = nutrition.get('calories_per_100g', 0)
        if calories > 335: negative_points += min(10, int((calories - 335) / 335 * 10))
        
        sugar = nutrition.get('sugar_g', 0)
        if sugar > 4.5: negative_points += min(10, int((sugar - 4.5) / 4.5 * 10))
        
        sodium = nutrition.get('sodium_mg', 0)
        if sodium > 90: negative_points += min(10, int((sodium - 90) / 90 * 10))
        
        fat = nutrition.get('fat_g', 0)
        if fat > 10: negative_points += min(10, int((fat - 10) / 10 * 10))
        
        # Positive points
        fiber = nutrition.get('fiber_g', 0)
        if fiber > 0.9: positive_points += min(5, int(fiber / 0.9))
        
        protein = nutrition.get('protein_g', 0)
        if protein > 1.6: positive_points += min(5, int(protein / 1.6))
        
        # Calculate final score
        final_score = negative_points - positive_points
        
        # Determine grade
        if final_score <= -1: return 'A'
        elif final_score <= 2: return 'B'
        elif final_score <= 10: return 'C'
        elif final_score <= 18: return 'D'
        else: return 'E'

# Usage example
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = FoodAnalysisPipeline()
    
    # Analyze food image
    image_path = "path/to/food/label.jpg"
    results = pipeline.analyze_food_image(image_path)
    
    if results['success']:
        print(f"✅ Analysis completed!")
        print(f"📊 Health Score: {results['final_score']}/100")
        print(f"🥘 Ingredients found: {results['ingredient_extraction']['count']}")
        print(f"📋 Recommendations:")
        for rec in results['recommendations']:
            print(f"  • {rec}")
    else:
        print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")