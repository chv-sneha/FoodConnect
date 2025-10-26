#!/usr/bin/env python3
"""
Multi-Class Text Classification for Food Labels
Classifies text into: product_name, ingredients, nutrition, brand, other
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import pickle
import re
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

class FoodLabelTextClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'svm': SVC(kernel='rbf', random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42),
            'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
        }
        self.best_model = None
        self.best_score = 0
        
    def create_training_data(self) -> pd.DataFrame:
        """Create comprehensive training dataset"""
        
        training_data = [
            # Product Names
            {'text': 'Maggi 2-Minute Noodles', 'label': 'product_name'},
            {'text': 'Parle-G Original Biscuits', 'label': 'product_name'},
            {'text': 'Amul Fresh Milk', 'label': 'product_name'},
            {'text': 'Britannia Good Day Cookies', 'label': 'product_name'},
            {'text': 'Tata Tea Premium', 'label': 'product_name'},
            {'text': 'Haldirams Bhujia', 'label': 'product_name'},
            {'text': 'Nestle Cerelac', 'label': 'product_name'},
            {'text': 'Cadbury Dairy Milk', 'label': 'product_name'},
            
            # Ingredients
            {'text': 'Ingredients: Wheat flour, Sugar, Palm oil, Salt, Baking powder', 'label': 'ingredients'},
            {'text': 'Contains: Rice, Lentils, Turmeric, Cumin seeds, Coriander', 'label': 'ingredients'},
            {'text': 'Wheat flour (60%), Sugar (20%), Vegetable oil (15%), Salt (3%)', 'label': 'ingredients'},
            {'text': 'Rice flour, Gram flour, Spices and condiments, Edible oil', 'label': 'ingredients'},
            {'text': 'Milk solids, Sugar, Cocoa solids, Vanilla flavoring', 'label': 'ingredients'},
            {'text': 'Refined wheat flour, Sugar, Edible vegetable oil, Milk solids', 'label': 'ingredients'},
            {'text': 'Tea, Natural flavoring agents', 'label': 'ingredients'},
            {'text': 'Potato, Gram flour, Spices, Salt, Edible oil', 'label': 'ingredients'},
            
            # Nutrition Facts
            {'text': 'Energy: 450 kcal per 100g', 'label': 'nutrition'},
            {'text': 'Protein: 8g, Carbohydrates: 65g, Fat: 18g', 'label': 'nutrition'},
            {'text': 'Nutritional Information per serving (30g)', 'label': 'nutrition'},
            {'text': 'Calories: 150, Total Fat: 6g, Sodium: 200mg', 'label': 'nutrition'},
            {'text': 'Per 100g: Energy 520 kcal, Protein 12g, Carbohydrate 60g', 'label': 'nutrition'},
            {'text': 'Dietary Fiber: 3g, Sugar: 15g, Cholesterol: 0mg', 'label': 'nutrition'},
            {'text': 'Vitamin A: 10%, Vitamin C: 25%, Calcium: 8%', 'label': 'nutrition'},
            {'text': 'Trans Fat: 0g, Saturated Fat: 8g', 'label': 'nutrition'},
            
            # Brand Information
            {'text': 'Nestle India Ltd', 'label': 'brand'},
            {'text': 'Parle Products Pvt Ltd', 'label': 'brand'},
            {'text': 'Britannia Industries Limited', 'label': 'brand'},
            {'text': 'Amul - The Taste of India', 'label': 'brand'},
            {'text': 'Tata Consumer Products', 'label': 'brand'},
            {'text': 'Haldirams Manufacturing Company', 'label': 'brand'},
            {'text': 'Cadbury India Limited', 'label': 'brand'},
            {'text': 'ITC Limited', 'label': 'brand'},
            
            # Other Information
            {'text': 'Best before 12 months from manufacturing', 'label': 'other'},
            {'text': 'Net Weight: 500g', 'label': 'other'},
            {'text': 'FSSAI License No: 12345678901234', 'label': 'other'},
            {'text': 'MRP Rs. 150 inclusive of all taxes', 'label': 'other'},
            {'text': 'Store in cool dry place', 'label': 'other'},
            {'text': 'Customer care: 1800-123-4567', 'label': 'other'},
            {'text': 'Manufactured by ABC Foods Ltd', 'label': 'other'},
            {'text': 'Batch No: AB123CD456', 'label': 'other'},
            {'text': 'Mfg Date: 15/03/2024', 'label': 'other'},
            {'text': 'Exp Date: 15/03/2025', 'label': 'other'}
        ]
        
        return pd.DataFrame(training_data)
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for classification"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s:,%()-]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def train_all_models(self, df: pd.DataFrame = None) -> Dict:
        """Train and compare all models"""
        
        if df is None:
            df = self.create_training_data()
        
        # Preprocess
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Prepare data
        X = df['processed_text']
        y = df['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Vectorize
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        results = {}
        
        # Train each model
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train_vec, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_vec)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5)
            
            results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred),
                'model': model
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Track best model
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = model
        
        return results
    
    def plot_model_comparison(self, results: Dict):
        """Plot model performance comparison"""
        
        models = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in models]
        cv_scores = [results[model]['cv_mean'] for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        ax1.bar(models, accuracies, color='skyblue', alpha=0.7)
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        plt.setp(ax1.get_xticklabels(), rotation=45)
        
        # CV Score comparison
        ax2.bar(models, cv_scores, color='lightcoral', alpha=0.7)
        ax2.set_title('Cross-Validation Score Comparison')
        ax2.set_ylabel('CV Score')
        ax2.set_ylim(0, 1)
        plt.setp(ax2.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('ml_models/results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, labels):
        """Plot confusion matrix"""
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix - Best Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('ml_models/results/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict(self, text: str) -> Dict:
        """Predict text category"""
        
        if self.best_model is None:
            raise ValueError("No trained model available")
        
        processed_text = self.preprocess_text(text)
        text_vec = self.vectorizer.transform([processed_text])
        
        prediction = self.best_model.predict(text_vec)[0]
        
        # Get probabilities if available
        if hasattr(self.best_model, 'predict_proba'):
            probabilities = self.best_model.predict_proba(text_vec)[0]
            prob_dict = dict(zip(self.best_model.classes_, probabilities))
        else:
            prob_dict = {prediction: 1.0}
        
        return {
            'predicted_category': prediction,
            'confidence': max(prob_dict.values()),
            'probabilities': prob_dict
        }
    
    def classify_text_blocks(self, text_blocks: List[str]) -> List[Dict]:
        """Classify multiple text blocks"""
        
        results = []
        for text in text_blocks:
            prediction = self.predict(text)
            results.append({
                'text': text,
                'category': prediction['predicted_category'],
                'confidence': prediction['confidence']
            })
        
        return results
    
    def save_model(self, filepath: str):
        """Save trained model and vectorizer"""
        
        model_data = {
            'vectorizer': self.vectorizer,
            'best_model': self.best_model,
            'best_score': self.best_score
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load trained model"""
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.best_model = model_data['best_model']
        self.best_score = model_data['best_score']

def main():
    """Main training and evaluation function"""
    
    # Initialize classifier
    classifier = FoodLabelTextClassifier()
    
    # Train all models
    print("Training multiple models for comparison...")
    results = classifier.train_all_models()
    
    # Plot comparisons
    classifier.plot_model_comparison(results)
    
    # Print detailed results
    print("\n" + "="*50)
    print("DETAILED RESULTS")
    print("="*50)
    
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"CV Score: {result['cv_mean']:.4f} (+/- {result['cv_std']*2:.4f})")
        print("\nClassification Report:")
        print(result['classification_report'])
    
    # Save best model
    classifier.save_model('ml_models/models/text_classifier.pkl')
    
    # Test predictions
    test_texts = [
        "Maggi 2-Minute Noodles Masala",
        "Ingredients: Wheat flour, Sugar, Salt",
        "Energy: 450 kcal per 100g",
        "Nestle India Limited"
    ]
    
    print("\n" + "="*50)
    print("TEST PREDICTIONS")
    print("="*50)
    
    for text in test_texts:
        result = classifier.predict(text)
        print(f"\nText: '{text}'")
        print(f"Category: {result['predicted_category']}")
        print(f"Confidence: {result['confidence']:.3f}")

if __name__ == "__main__":
    main()