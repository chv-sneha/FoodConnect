#!/usr/bin/env python3
import sys
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

class FoodMLPredictor:
    def __init__(self):
        self.models_path = Path(__file__).parent / 'models'
        self.load_models()
        self.ingredient_db = self.load_ingredient_database()
    
    def load_models(self):
        try:
            # Load the production model
            self.model = joblib.load(self.models_path / 'production_model.pkl')
            
            # Load preprocessing objects
            self.scaler = joblib.load(self.models_path / 'scaler.pkl')
            self.label_encoders = joblib.load(self.models_path / 'label_encoders.pkl')
            
            print("✅ Models loaded successfully", file=sys.stderr)
        except Exception as e:
            print(f"❌ Error loading models: {e}", file=sys.stderr)
            self.model = None
    
    def load_ingredient_database(self):
        try:
            db_path = self.models_path.parent / 'data' / 'processed' / 'ingredient_toxicity_db.csv'
            return pd.read_csv(db_path)
        except Exception as e:
            print(f"⚠️ Could not load ingredient database: {e}", file=sys.stderr)
            return None
    
    def preprocess_ingredient(self, ingredient_name):
        """Convert ingredient name to model features"""
        if self.ingredient_db is None:
            return self.create_default_features(ingredient_name)
        
        # Try to find ingredient in database
        ingredient_lower = ingredient_name.lower().strip()
        
        # Exact match
        match = self.ingredient_db[
            self.ingredient_db['ingredient_name'].str.lower() == ingredient_lower
        ]
        
        # Partial match if no exact match
        if match.empty:
            match = self.ingredient_db[
                self.ingredient_db['ingredient_name'].str.lower().str.contains(ingredient_lower, na=False)
            ]
        
        # Reverse partial match
        if match.empty:
            for _, row in self.ingredient_db.iterrows():
                if ingredient_lower in row['ingredient_name'].lower():
                    match = pd.DataFrame([row])
                    break
        
        if not match.empty:
            row = match.iloc[0]
            return self.create_features_from_row(row)
        else:
            return self.create_default_features(ingredient_name)
    
    def create_features_from_row(self, row):
        """Create feature vector from database row"""
        features = {
            'toxicity_score': row['toxicity_score'],
            'category_encoded': self.encode_category(row['category']),
            'health_impact_encoded': self.encode_health_impact(row['health_impact']),
            'is_toxic': int(row['toxicity_score'] > 50),
            'toxicity_level': self.get_toxicity_level(row['toxicity_score']),
            'high_toxicity_flag': int(row['toxicity_score'] > 70),
            'toxicity_score_squared': row['toxicity_score'] ** 2,
            'toxicity_score_normalized': row['toxicity_score'] / 100
        }
        return features
    
    def create_default_features(self, ingredient_name):
        """Create default features for unknown ingredients"""
        # Default to moderate toxicity for unknown ingredients
        toxicity_score = 40
        
        features = {
            'toxicity_score': toxicity_score,
            'category_encoded': 0,  # Unknown category
            'health_impact_encoded': 2,  # Medium impact
            'is_toxic': 0,
            'toxicity_level': 1,  # Low risk
            'high_toxicity_flag': 0,
            'toxicity_score_squared': toxicity_score ** 2,
            'toxicity_score_normalized': toxicity_score / 100
        }
        return features
    
    def encode_category(self, category):
        """Encode category using label encoder"""
        try:
            if 'category' in self.label_encoders:
                encoder = self.label_encoders['category']
                if category in encoder.classes_:
                    return encoder.transform([category])[0]
        except:
            pass
        return 0  # Default encoding
    
    def encode_health_impact(self, health_impact):
        """Encode health impact using label encoder"""
        try:
            if 'health_impact' in self.label_encoders:
                encoder = self.label_encoders['health_impact']
                if health_impact in encoder.classes_:
                    return encoder.transform([health_impact])[0]
        except:
            pass
        return 2  # Default encoding (medium)
    
    def get_toxicity_level(self, toxicity_score):
        """Convert toxicity score to level"""
        if toxicity_score <= 20:
            return 0  # Safe
        elif toxicity_score <= 40:
            return 1  # Low risk
        elif toxicity_score <= 70:
            return 2  # Medium risk
        else:
            return 3  # High risk
    
    def predict_ingredients(self, ingredients):
        """Predict toxicity for list of ingredients"""
        results = []
        
        for ingredient in ingredients:
            try:
                # Get features for ingredient
                features = self.preprocess_ingredient(ingredient)
                
                # Create feature vector for model
                feature_vector = [
                    features['toxicity_score'],
                    features['category_encoded'],
                    features['health_impact_encoded'],
                    features['toxicity_level'],
                    features['high_toxicity_flag'],
                    features['toxicity_score_squared'],
                    features['toxicity_score_normalized']
                ]
                
                # Make prediction if model is available
                if self.model is not None:
                    try:
                        # Scale features
                        feature_vector_scaled = self.scaler.transform([feature_vector])
                        
                        # Predict
                        prediction = self.model.predict(feature_vector_scaled)[0]
                        probability = self.model.predict_proba(feature_vector_scaled)[0]
                        
                        confidence = max(probability)
                    except Exception as e:
                        print(f"Model prediction error for {ingredient}: {e}", file=sys.stderr)
                        prediction = features['is_toxic']
                        confidence = 0.7
                else:
                    prediction = features['is_toxic']
                    confidence = 0.6
                
                # Create result
                result = {
                    'ingredient': ingredient,
                    'toxicity_score': features['toxicity_score'],
                    'is_toxic': bool(prediction),
                    'confidence': float(confidence),
                    'category': self.get_category_name(features['category_encoded']),
                    'risk_level': self.get_risk_level_name(features['toxicity_level']),
                    'health_impact': self.get_health_impact_name(features['health_impact_encoded'])
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing ingredient {ingredient}: {e}", file=sys.stderr)
                # Fallback result
                results.append({
                    'ingredient': ingredient,
                    'toxicity_score': 40,
                    'is_toxic': False,
                    'confidence': 0.3,
                    'category': 'unknown',
                    'risk_level': 'low',
                    'health_impact': 'medium'
                })
        
        return results
    
    def get_category_name(self, encoded_value):
        """Convert encoded category back to name"""
        categories = ['unknown', 'sweetener', 'preservative', 'flavor_enhancer', 'additive', 'fat', 'vitamin', 'mineral', 'macronutrient', 'fiber']
        if 0 <= encoded_value < len(categories):
            return categories[encoded_value]
        return 'unknown'
    
    def get_risk_level_name(self, level):
        """Convert risk level to name"""
        levels = ['safe', 'low', 'medium', 'high']
        if 0 <= level < len(levels):
            return levels[level]
        return 'medium'
    
    def get_health_impact_name(self, encoded_value):
        """Convert encoded health impact back to name"""
        impacts = ['beneficial', 'very_low', 'low', 'medium', 'high', 'very_high']
        if 0 <= encoded_value < len(impacts):
            return impacts[encoded_value]
        return 'medium'

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py '<ingredients_json>'", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Parse input
        ingredients_json = sys.argv[1]
        ingredients = json.loads(ingredients_json)
        
        # Create predictor
        predictor = FoodMLPredictor()
        
        # Make predictions
        results = predictor.predict_ingredients(ingredients)
        
        # Output results as JSON
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()