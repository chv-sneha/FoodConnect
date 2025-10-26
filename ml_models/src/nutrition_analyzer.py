#!/usr/bin/env python3
"""
Nutrition Analysis using Kaggle Food Dataset
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz, process
import re
from typing import List, Dict, Tuple
import requests
import os

class NutritionAnalyzer:
    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path
        self.nutrition_df = None
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.ingredient_vectors = None
        
    def load_kaggle_dataset(self):
        """Load Kaggle food nutrition dataset"""
        
        if self.dataset_path and os.path.exists(self.dataset_path):
            self.nutrition_df = pd.read_csv(self.dataset_path)
        else:
            # Create sample dataset if Kaggle dataset not available
            self.nutrition_df = self._create_sample_dataset()
        
        # Clean and prepare data
        self._prepare_dataset()
        
        # Create ingredient vectors for similarity matching
        self._create_ingredient_vectors()
        
        print(f"Loaded {len(self.nutrition_df)} food items")
    
    def _create_sample_dataset(self) -> pd.DataFrame:
        """Create sample nutrition dataset"""
        
        sample_data = [
            {
                'name': 'Wheat Flour',
                'calories_per_100g': 364,
                'protein_g': 10.3,
                'carbs_g': 76.3,
                'fat_g': 1.5,
                'fiber_g': 2.7,
                'sugar_g': 0.4,
                'sodium_mg': 2,
                'category': 'grains'
            },
            {
                'name': 'Sugar',
                'calories_per_100g': 387,
                'protein_g': 0,
                'carbs_g': 99.8,
                'fat_g': 0,
                'fiber_g': 0,
                'sugar_g': 99.8,
                'sodium_mg': 1,
                'category': 'sweeteners'
            },
            {
                'name': 'Palm Oil',
                'calories_per_100g': 884,
                'protein_g': 0,
                'carbs_g': 0,
                'fat_g': 100,
                'fiber_g': 0,
                'sugar_g': 0,
                'sodium_mg': 0,
                'category': 'oils'
            },
            {
                'name': 'Salt',
                'calories_per_100g': 0,
                'protein_g': 0,
                'carbs_g': 0,
                'fat_g': 0,
                'fiber_g': 0,
                'sugar_g': 0,
                'sodium_mg': 38758,
                'category': 'seasonings'
            },
            {
                'name': 'Rice',
                'calories_per_100g': 130,
                'protein_g': 2.7,
                'carbs_g': 28,
                'fat_g': 0.3,
                'fiber_g': 0.4,
                'sugar_g': 0.1,
                'sodium_mg': 5,
                'category': 'grains'
            }
        ]
        
        return pd.DataFrame(sample_data)
    
    def _prepare_dataset(self):
        """Clean and prepare the dataset"""
        
        # Standardize column names
        column_mapping = {
            'Food': 'name',
            'Calories': 'calories_per_100g',
            'Protein (g)': 'protein_g',
            'Carbohydrate (g)': 'carbs_g',
            'Fat (g)': 'fat_g',
            'Fiber (g)': 'fiber_g',
            'Sugar (g)': 'sugar_g',
            'Sodium (mg)': 'sodium_mg'
        }
        
        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in self.nutrition_df.columns:
                self.nutrition_df.rename(columns={old_name: new_name}, inplace=True)
        
        # Clean food names
        self.nutrition_df['name'] = self.nutrition_df['name'].str.lower().str.strip()
        
        # Fill missing values
        numeric_columns = ['calories_per_100g', 'protein_g', 'carbs_g', 'fat_g', 'fiber_g', 'sugar_g', 'sodium_mg']
        for col in numeric_columns:
            if col in self.nutrition_df.columns:
                self.nutrition_df[col] = pd.to_numeric(self.nutrition_df[col], errors='coerce').fillna(0)
    
    def _create_ingredient_vectors(self):
        """Create TF-IDF vectors for ingredient matching"""
        
        food_names = self.nutrition_df['name'].tolist()
        self.ingredient_vectors = self.vectorizer.fit_transform(food_names)
    
    def clean_ingredient_name(self, ingredient: str) -> str:
        """Clean ingredient name for matching"""
        
        # Convert to lowercase
        ingredient = ingredient.lower().strip()
        
        # Remove common prefixes/suffixes
        prefixes = ['refined', 'edible', 'natural', 'artificial', 'organic']
        suffixes = ['powder', 'extract', 'oil', 'flour']
        
        for prefix in prefixes:
            if ingredient.startswith(prefix):
                ingredient = ingredient.replace(prefix, '').strip()
        
        # Remove parentheses and percentages
        ingredient = re.sub(r'\([^)]*\)', '', ingredient)
        ingredient = re.sub(r'\d+%?', '', ingredient)
        
        # Remove extra whitespace
        ingredient = re.sub(r'\s+', ' ', ingredient).strip()
        
        return ingredient
    
    def find_best_match(self, ingredient: str, threshold: float = 0.6) -> Tuple[str, float, Dict]:
        """Find best matching food item for ingredient"""
        
        cleaned_ingredient = self.clean_ingredient_name(ingredient)
        
        # Method 1: Exact fuzzy matching
        food_names = self.nutrition_df['name'].tolist()
        fuzzy_match = process.extractOne(cleaned_ingredient, food_names, scorer=fuzz.ratio)
        
        if fuzzy_match and fuzzy_match[1] >= (threshold * 100):
            matched_food = self.nutrition_df[self.nutrition_df['name'] == fuzzy_match[0]].iloc[0]
            return fuzzy_match[0], fuzzy_match[1] / 100, matched_food.to_dict()
        
        # Method 2: TF-IDF similarity
        ingredient_vector = self.vectorizer.transform([cleaned_ingredient])
        similarities = cosine_similarity(ingredient_vector, self.ingredient_vectors).flatten()
        
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        if best_similarity >= threshold:
            matched_food = self.nutrition_df.iloc[best_idx]
            return matched_food['name'], best_similarity, matched_food.to_dict()
        
        return None, 0.0, {}
    
    def analyze_ingredients(self, ingredients: List[str]) -> Dict:
        """Analyze nutrition for list of ingredients"""
        
        if not self.nutrition_df is not None:
            self.load_kaggle_dataset()
        
        analysis_results = {
            'matched_ingredients': [],
            'unmatched_ingredients': [],
            'total_nutrition': {
                'calories_per_100g': 0,
                'protein_g': 0,
                'carbs_g': 0,
                'fat_g': 0,
                'fiber_g': 0,
                'sugar_g': 0,
                'sodium_mg': 0
            },
            'health_score': 0,
            'warnings': []
        }
        
        total_matched_weight = 0
        
        for ingredient in ingredients:
            if len(ingredient.strip()) < 2:
                continue
                
            match_name, confidence, nutrition_data = self.find_best_match(ingredient)
            
            if match_name:
                # Estimate ingredient weight (simplified)
                estimated_weight = self._estimate_ingredient_weight(ingredient, len(ingredients))
                
                matched_info = {
                    'original': ingredient,
                    'matched': match_name,
                    'confidence': confidence,
                    'estimated_weight_percent': estimated_weight,
                    'nutrition': nutrition_data
                }
                
                analysis_results['matched_ingredients'].append(matched_info)
                
                # Add to total nutrition (weighted)
                weight_factor = estimated_weight / 100
                for nutrient in analysis_results['total_nutrition']:
                    if nutrient in nutrition_data:
                        analysis_results['total_nutrition'][nutrient] += nutrition_data[nutrient] * weight_factor
                
                total_matched_weight += estimated_weight
                
            else:
                analysis_results['unmatched_ingredients'].append(ingredient)
        
        # Calculate health score and warnings
        analysis_results['health_score'] = self._calculate_health_score(analysis_results['total_nutrition'])
        analysis_results['warnings'] = self._generate_warnings(analysis_results['total_nutrition'])
        analysis_results['match_coverage'] = total_matched_weight
        
        return analysis_results
    
    def _estimate_ingredient_weight(self, ingredient: str, total_ingredients: int) -> float:
        """Estimate ingredient weight percentage (simplified)"""
        
        # Basic heuristic: first ingredients are usually higher percentage
        base_weight = 100 / total_ingredients
        
        # Adjust based on ingredient type
        ingredient_lower = ingredient.lower()
        
        if any(word in ingredient_lower for word in ['flour', 'rice', 'wheat', 'oats']):
            return min(base_weight * 2, 60)  # Main ingredients
        elif any(word in ingredient_lower for word in ['sugar', 'oil', 'fat']):
            return min(base_weight * 1.5, 30)  # Secondary ingredients
        elif any(word in ingredient_lower for word in ['salt', 'spice', 'flavor', 'color']):
            return min(base_weight * 0.5, 5)   # Minor ingredients
        
        return base_weight
    
    def _calculate_health_score(self, nutrition: Dict) -> float:
        """Calculate health score (0-100, higher is better)"""
        
        score = 100
        
        # Penalize high calories
        if nutrition['calories_per_100g'] > 400:
            score -= 20
        elif nutrition['calories_per_100g'] > 300:
            score -= 10
        
        # Penalize high sugar
        if nutrition['sugar_g'] > 20:
            score -= 25
        elif nutrition['sugar_g'] > 10:
            score -= 15
        
        # Penalize high sodium
        if nutrition['sodium_mg'] > 1000:
            score -= 20
        elif nutrition['sodium_mg'] > 500:
            score -= 10
        
        # Penalize high fat
        if nutrition['fat_g'] > 30:
            score -= 15
        elif nutrition['fat_g'] > 20:
            score -= 10
        
        # Reward fiber
        if nutrition['fiber_g'] > 5:
            score += 10
        elif nutrition['fiber_g'] > 3:
            score += 5
        
        # Reward protein
        if nutrition['protein_g'] > 10:
            score += 10
        elif nutrition['protein_g'] > 5:
            score += 5
        
        return max(0, min(100, score))
    
    def _generate_warnings(self, nutrition: Dict) -> List[str]:
        """Generate health warnings based on nutrition"""
        
        warnings = []
        
        if nutrition['calories_per_100g'] > 400:
            warnings.append("High calorie content")
        
        if nutrition['sugar_g'] > 15:
            warnings.append("High sugar content")
        
        if nutrition['sodium_mg'] > 800:
            warnings.append("High sodium content")
        
        if nutrition['fat_g'] > 25:
            warnings.append("High fat content")
        
        if nutrition['fiber_g'] < 2:
            warnings.append("Low fiber content")
        
        return warnings

# Usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = NutritionAnalyzer()
    analyzer.load_kaggle_dataset()
    
    # Test with sample ingredients
    ingredients = ["wheat flour", "sugar", "palm oil", "salt"]
    results = analyzer.analyze_ingredients(ingredients)
    
    print("Nutrition Analysis Results:")
    print(f"Health Score: {results['health_score']}/100")
    print(f"Matched: {len(results['matched_ingredients'])}/{len(ingredients)} ingredients")
    print(f"Warnings: {results['warnings']}")