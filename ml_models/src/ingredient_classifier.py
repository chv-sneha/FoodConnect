#!/usr/bin/env python3
"""
Ingredient Text Classification - Identifies ingredient text from OCR output
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import re
from typing import List, Dict

class IngredientTextClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
    
    def create_training_data(self) -> pd.DataFrame:
        """Create training data for text classification"""
        
        # Ingredient text examples
        ingredient_texts = [
            "Ingredients: Wheat flour, Sugar, Palm oil, Salt",
            "Contains: Rice, Lentils, Turmeric, Cumin",
            "Wheat flour (60%), Sugar (20%), Vegetable oil",
            "Rice flour, Salt, Spices, Preservatives",
            "Milk solids, Sugar, Cocoa powder, Vanilla",
            "Tomatoes, Onions, Garlic, Ginger, Spices",
            "Refined wheat flour, Edible vegetable oil",
            "Basmati rice, Natural flavoring",
            "Whole wheat, Jaggery, Ghee, Cardamom",
            "Soybean oil, Mustard seeds, Fenugreek"
        ]
        
        # Non-ingredient text examples  
        non_ingredient_texts = [
            "Best before 12 months from manufacturing",
            "Net Weight: 500g",
            "FSSAI License No: 12345678901234",
            "Manufactured by ABC Foods Ltd",
            "Energy: 450 kcal per 100g",
            "Protein: 8g, Carbohydrates: 65g",
            "Store in cool dry place",
            "MRP Rs. 150 inclusive of all taxes",
            "Customer care: 1800-123-4567",
            "Nutritional Information per serving"
        ]
        
        # Create DataFrame
        data = []
        
        # Add ingredient texts (label = 1)
        for text in ingredient_texts:
            data.append({'text': text, 'is_ingredient': 1})
        
        # Add non-ingredient texts (label = 0)
        for text in non_ingredient_texts:
            data.append({'text': text, 'is_ingredient': 0})
        
        return pd.DataFrame(data)
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for classification"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep commas and colons
        text = re.sub(r'[^\w\s,:()]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def train(self, df: pd.DataFrame = None):
        """Train the ingredient text classifier"""
        
        if df is None:
            df = self.create_training_data()
        
        # Preprocess texts
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Split data
        X = df['processed_text']
        y = df['is_ingredient']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Vectorize text
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train classifier
        self.classifier.fit(X_train_vec, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Training Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.is_trained = True
        
        return accuracy
    
    def predict(self, text: str) -> Dict:
        """Predict if text contains ingredients"""
        
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        processed_text = self.preprocess_text(text)
        text_vec = self.vectorizer.transform([processed_text])
        
        prediction = self.classifier.predict(text_vec)[0]
        probability = self.classifier.predict_proba(text_vec)[0]
        
        return {
            'is_ingredient': bool(prediction),
            'confidence': float(max(probability)),
            'probabilities': {
                'not_ingredient': float(probability[0]),
                'ingredient': float(probability[1])
            }
        }
    
    def classify_ocr_blocks(self, text_blocks: List[Dict]) -> List[Dict]:
        """Classify multiple OCR text blocks"""
        
        results = []
        
        for block in text_blocks:
            text = block.get('text', '')
            prediction = self.predict(text)
            
            block_result = block.copy()
            block_result.update(prediction)
            results.append(block_result)
        
        return results
    
    def extract_ingredient_text(self, text_blocks: List[Dict]) -> str:
        """Extract only ingredient text from OCR blocks"""
        
        classified_blocks = self.classify_ocr_blocks(text_blocks)
        
        ingredient_texts = []
        for block in classified_blocks:
            if block['is_ingredient'] and block['confidence'] > 0.7:
                ingredient_texts.append(block['text'])
        
        return ' '.join(ingredient_texts)
    
    def save_model(self, filepath: str):
        """Save trained model"""
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.is_trained = model_data['is_trained']

# Usage
if __name__ == "__main__":
    # Train the classifier
    classifier = IngredientTextClassifier()
    classifier.train()
    
    # Save model
    classifier.save_model('models/ingredient_classifier.pkl')
    
    # Test prediction
    test_text = "Ingredients: Wheat flour, Sugar, Salt, Baking powder"
    result = classifier.predict(test_text)
    print(f"\nTest: {test_text}")
    print(f"Result: {result}")