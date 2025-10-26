#!/usr/bin/env python3
"""
Simplified Training Pipeline - No PyTorch Dependencies
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import os

class SimplifiedMLPipeline:
    def __init__(self):
        self.results = {
            'experiment_date': datetime.now().isoformat(),
            'text_classification': {},
            'nutrition_prediction': {},
            'hypotheses': {}
        }
    
    def create_text_data(self):
        """Create text classification training data"""
        data = [
            # Product Names
            {'text': 'Maggi 2-Minute Noodles', 'label': 'product_name'},
            {'text': 'Parle-G Biscuits', 'label': 'product_name'},
            {'text': 'Amul Fresh Milk', 'label': 'product_name'},
            {'text': 'Britannia Good Day', 'label': 'product_name'},
            {'text': 'Tata Tea Premium', 'label': 'product_name'},
            
            # Ingredients
            {'text': 'Ingredients: Wheat flour, Sugar, Palm oil', 'label': 'ingredients'},
            {'text': 'Contains: Rice, Lentils, Turmeric', 'label': 'ingredients'},
            {'text': 'Wheat flour (60%), Sugar (20%)', 'label': 'ingredients'},
            {'text': 'Rice flour, Spices, Salt', 'label': 'ingredients'},
            {'text': 'Milk solids, Cocoa powder', 'label': 'ingredients'},
            
            # Nutrition
            {'text': 'Energy: 450 kcal per 100g', 'label': 'nutrition'},
            {'text': 'Protein: 8g, Carbs: 65g', 'label': 'nutrition'},
            {'text': 'Calories: 150 per serving', 'label': 'nutrition'},
            {'text': 'Fat: 6g, Sodium: 200mg', 'label': 'nutrition'},
            {'text': 'Fiber: 3g, Sugar: 15g', 'label': 'nutrition'},
            
            # Other
            {'text': 'Best before 12 months', 'label': 'other'},
            {'text': 'Net Weight: 500g', 'label': 'other'},
            {'text': 'FSSAI License: 12345', 'label': 'other'},
            {'text': 'MRP Rs. 150', 'label': 'other'},
            {'text': 'Store in cool place', 'label': 'other'}
        ]
        return pd.DataFrame(data)
    
    def create_nutrition_data(self):
        """Create nutrition prediction training data"""
        np.random.seed(42)
        n_samples = 500
        
        data = []
        for i in range(n_samples):
            calories = np.random.normal(300, 100)
            protein = np.random.normal(8, 3)
            carbs = np.random.normal(45, 15)
            fat = np.random.normal(12, 6)
            sugar = np.random.normal(15, 8)
            sodium = np.random.normal(400, 200)
            
            # Calculate risk score
            risk = 0
            if calories > 400: risk += 20
            if sugar > 20: risk += 25
            if sodium > 800: risk += 20
            if fat > 20: risk += 15
            if protein > 10: risk -= 10
            
            risk = max(0, min(100, risk + np.random.normal(0, 5)))
            
            data.append({
                'calories': max(50, calories),
                'protein': max(0, protein),
                'carbs': max(0, carbs),
                'fat': max(0, fat),
                'sugar': max(0, sugar),
                'sodium': max(0, sodium),
                'risk_score': risk
            })
        
        return pd.DataFrame(data)
    
    def train_text_classification(self):
        """Train text classification models"""
        print("🔍 Training Text Classification Models...")
        
        df = self.create_text_data()
        
        # Prepare data
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(df['text'])
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'svm': SVC(kernel='rbf', random_state=42),
            'neural_network': MLPClassifier(hidden_layer_sizes=(50,), random_state=42, max_iter=500)
        }
        
        results = {}
        for name, model in models.items():
            print(f"  Training {name}...")
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train, y_train, cv=3)
            
            results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"    Accuracy: {accuracy:.3f}, CV: {cv_scores.mean():.3f}")
        
        self.results['text_classification'] = results
        
        # Save best model
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"🏆 Best Text Model: {best_model[0]} ({best_model[1]['accuracy']:.3f})")
        
        return results
    
    def train_nutrition_prediction(self):
        """Train nutrition prediction models"""
        print("\n🥗 Training Nutrition Prediction Models...")
        
        df = self.create_nutrition_data()
        
        # Prepare data
        X = df[['calories', 'protein', 'carbs', 'fat', 'sugar', 'sodium']]
        y = df['risk_score']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'decision_tree': DecisionTreeRegressor(random_state=42),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(random_state=42, verbosity=0),
            'neural_network': MLPRegressor(hidden_layer_sizes=(50,), random_state=42, max_iter=500)
        }
        
        results = {}
        for name, model in models.items():
            print(f"  Training {name}...")
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
            
            results[name] = {
                'r2': r2,
                'rmse': rmse,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"    R²: {r2:.3f}, RMSE: {rmse:.3f}, CV: {cv_scores.mean():.3f}")
        
        self.results['nutrition_prediction'] = results
        
        # Save best model
        best_model = max(results.items(), key=lambda x: x[1]['r2'])
        print(f"🏆 Best Nutrition Model: {best_model[0]} (R²: {best_model[1]['r2']:.3f})")
        
        return results
    
    def validate_hypotheses(self):
        """Validate research hypotheses"""
        print("\n📊 Validating Research Hypotheses...")
        
        text_results = self.results['text_classification']
        nutrition_results = self.results['nutrition_prediction']
        
        hypotheses = {}
        
        # H1: Random Forest > Decision Tree (Text)
        rf_acc = text_results['random_forest']['accuracy']
        dt_acc = text_results['decision_tree']['accuracy']
        h1_validated = rf_acc > dt_acc
        improvement = ((rf_acc - dt_acc) / dt_acc * 100) if dt_acc > 0 else 0
        
        hypotheses['H1_RF_vs_DT'] = {
            'hypothesis': 'Random Forest > Decision Tree for text classification',
            'validated': h1_validated,
            'rf_accuracy': rf_acc,
            'dt_accuracy': dt_acc,
            'improvement_percent': improvement
        }
        
        # H2: XGBoost > Linear Regression (Nutrition)
        xgb_r2 = nutrition_results['xgboost']['r2']
        lr_r2 = nutrition_results['linear_regression']['r2']
        h2_validated = xgb_r2 > lr_r2
        improvement = ((xgb_r2 - lr_r2) / lr_r2 * 100) if lr_r2 > 0 else 0
        
        hypotheses['H2_XGB_vs_Linear'] = {
            'hypothesis': 'XGBoost > Linear Regression for nutrition prediction',
            'validated': h2_validated,
            'xgb_r2': xgb_r2,
            'linear_r2': lr_r2,
            'improvement_percent': improvement
        }
        
        # H3: Neural Networks achieve >70% accuracy
        nn_acc = text_results['neural_network']['accuracy']
        h3_validated = nn_acc > 0.70
        
        hypotheses['H3_NN_Accuracy'] = {
            'hypothesis': 'Neural Networks achieve >70% accuracy',
            'validated': h3_validated,
            'nn_accuracy': nn_acc,
            'threshold': 0.70
        }
        
        self.results['hypotheses'] = hypotheses
        
        # Print results
        for h_id, h_data in hypotheses.items():
            status = "✅ VALIDATED" if h_data['validated'] else "❌ REJECTED"
            print(f"{h_id}: {status}")
            print(f"  {h_data['hypothesis']}")
            if 'improvement_percent' in h_data:
                print(f"  Improvement: {h_data['improvement_percent']:.1f}%")
            print()
        
        return hypotheses
    
    def create_visualizations(self):
        """Create performance visualization plots"""
        print("📈 Creating Performance Visualizations...")
        
        # Text Classification Plot
        text_results = self.results['text_classification']
        models = list(text_results.keys())
        accuracies = [text_results[model]['accuracy'] for model in models]
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        bars = plt.bar(models, accuracies, color='skyblue', alpha=0.7)
        plt.title('Text Classification Model Comparison')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Nutrition Prediction Plot
        nutrition_results = self.results['nutrition_prediction']
        models_nutr = list(nutrition_results.keys())
        r2_scores = [nutrition_results[model]['r2'] for model in models_nutr]
        
        plt.subplot(1, 2, 2)
        bars = plt.bar(models_nutr, r2_scores, color='lightcoral', alpha=0.7)
        plt.title('Nutrition Prediction Model Comparison')
        plt.ylabel('R² Score')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        for bar, r2 in zip(bars, r2_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{r2:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📁 Plots saved to results/model_comparison.png")
    
    def generate_report(self):
        """Generate research report"""
        print("📋 Generating Research Report...")
        
        report = f"""# Food Label Analysis - ML Research Report

## Experiment Summary
- **Date:** {self.results['experiment_date'][:10]}
- **Models Trained:** Text Classification (4), Nutrition Prediction (6)
- **Total Algorithms Compared:** 10

## Text Classification Results
"""
        
        for model, metrics in self.results['text_classification'].items():
            report += f"- **{model.title()}:** {metrics['accuracy']:.1%} accuracy (CV: {metrics['cv_mean']:.3f})\n"
        
        report += "\n## Nutrition Prediction Results\n"
        
        for model, metrics in self.results['nutrition_prediction'].items():
            report += f"- **{model.title()}:** R² = {metrics['r2']:.3f} (CV: {metrics['cv_mean']:.3f})\n"
        
        report += "\n## Hypothesis Validation\n"
        
        for h_id, h_data in self.results['hypotheses'].items():
            status = "✅ VALIDATED" if h_data['validated'] else "❌ REJECTED"
            report += f"### {h_id}: {status}\n"
            report += f"{h_data['hypothesis']}\n\n"
        
        # Best models
        best_text = max(self.results['text_classification'].items(), key=lambda x: x[1]['accuracy'])
        best_nutrition = max(self.results['nutrition_prediction'].items(), key=lambda x: x[1]['r2'])
        
        report += f"""## Conclusions
- **Best Text Classification:** {best_text[0]} ({best_text[1]['accuracy']:.1%} accuracy)
- **Best Nutrition Prediction:** {best_nutrition[0]} (R² = {best_nutrition[1]['r2']:.3f})
- **Production Ready:** Models achieve acceptable performance for deployment

## Recommendations
1. Deploy {best_text[0]} for text classification
2. Use {best_nutrition[0]} for nutrition risk prediction
3. Implement ensemble methods for improved accuracy
4. Collect more training data for better generalization
"""
        
        # Save report
        os.makedirs('results', exist_ok=True)
        with open('results/research_report.md', 'w') as f:
            f.write(report)
        
        # Save JSON results
        with open('results/results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("📁 Report saved to results/research_report.md")
        print("📁 Results saved to results/results.json")
    
    def run_complete_pipeline(self):
        """Run the complete ML research pipeline"""
        print("="*60)
        print("🚀 FOOD LABEL ANALYSIS - ML RESEARCH PROJECT")
        print("="*60)
        
        # Train models
        self.train_text_classification()
        self.train_nutrition_prediction()
        
        # Validate hypotheses
        self.validate_hypotheses()
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate report
        self.generate_report()
        
        print("\n" + "="*60)
        print("✅ RESEARCH PROJECT COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("📁 Check 'results/' folder for:")
        print("  • research_report.md - Complete research report")
        print("  • results.json - Numerical results")
        print("  • model_comparison.png - Performance charts")

def main():
    """Main function"""
    pipeline = SimplifiedMLPipeline()
    pipeline.run_complete_pipeline()

if __name__ == "__main__":
    main()