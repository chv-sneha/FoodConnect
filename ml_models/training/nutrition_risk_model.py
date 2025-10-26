#!/usr/bin/env python3
"""
Nutrition Risk Prediction Model - Ensemble Approach
Uses Kaggle dataset to predict health risks from ingredients
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

class NutritionRiskPredictor:
    def __init__(self, kaggle_dataset_path: str = None):
        self.dataset_path = kaggle_dataset_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Initialize models
        self.models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'decision_tree': DecisionTreeRegressor(random_state=42),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(random_state=42),
            'neural_network': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
            'svm': SVR(kernel='rbf')
        }
        
        self.ensemble_model = None
        self.best_model = None
        self.best_score = float('-inf')
        
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load Kaggle dataset and prepare for training"""
        
        if self.dataset_path:
            try:
                df = pd.read_csv(self.dataset_path)
            except:
                df = self._create_synthetic_dataset()
        else:
            df = self._create_synthetic_dataset()
        
        # Clean and prepare data
        df = self._clean_dataset(df)
        df = self._engineer_features(df)
        
        return df
    
    def _create_synthetic_dataset(self) -> pd.DataFrame:
        """Create synthetic nutrition dataset for training"""
        
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic food data
        foods = []
        for i in range(n_samples):
            # Basic nutrition values
            calories = np.random.normal(300, 150)
            protein = np.random.normal(8, 4)
            carbs = np.random.normal(45, 20)
            fat = np.random.normal(12, 8)
            fiber = np.random.normal(3, 2)
            sugar = np.random.normal(15, 10)
            sodium = np.random.normal(400, 300)
            
            # Ensure positive values
            calories = max(50, calories)
            protein = max(0, protein)
            carbs = max(0, carbs)
            fat = max(0, fat)
            fiber = max(0, fiber)
            sugar = max(0, sugar)
            sodium = max(0, sodium)
            
            # Calculate health risk score (0-100, higher = more risky)
            risk_score = 0
            
            # High calorie penalty
            if calories > 400: risk_score += 20
            elif calories > 300: risk_score += 10
            
            # High sugar penalty
            if sugar > 20: risk_score += 25
            elif sugar > 10: risk_score += 15
            
            # High sodium penalty
            if sodium > 800: risk_score += 20
            elif sodium > 500: risk_score += 10
            
            # High fat penalty
            if fat > 20: risk_score += 15
            elif fat > 15: risk_score += 10
            
            # Low fiber penalty
            if fiber < 2: risk_score += 10
            
            # High protein bonus
            if protein > 10: risk_score -= 10
            elif protein > 5: risk_score -= 5
            
            # High fiber bonus
            if fiber > 5: risk_score -= 10
            elif fiber > 3: risk_score -= 5
            
            # Normalize to 0-100
            risk_score = max(0, min(100, risk_score + np.random.normal(0, 5)))
            
            foods.append({
                'food_name': f'Food_{i}',
                'calories_per_100g': calories,
                'protein_g': protein,
                'carbohydrates_g': carbs,
                'fat_g': fat,
                'fiber_g': fiber,
                'sugar_g': sugar,
                'sodium_mg': sodium,
                'health_risk_score': risk_score,
                'category': np.random.choice(['grains', 'dairy', 'meat', 'vegetables', 'fruits', 'snacks'])
            })
        
        return pd.DataFrame(foods)
    
    def _clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataset"""
        
        # Standardize column names
        column_mapping = {
            'Food': 'food_name',
            'Calories': 'calories_per_100g',
            'Protein (g)': 'protein_g',
            'Carbohydrate (g)': 'carbohydrates_g',
            'Fat (g)': 'fat_g',
            'Fiber (g)': 'fiber_g',
            'Sugar (g)': 'sugar_g',
            'Sodium (mg)': 'sodium_mg'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        # Convert to numeric
        numeric_columns = ['calories_per_100g', 'protein_g', 'carbohydrates_g', 
                          'fat_g', 'fiber_g', 'sugar_g', 'sodium_mg']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values
        df = df.fillna(df.median(numeric_only=True))
        
        # Remove outliers (beyond 3 standard deviations)
        for col in numeric_columns:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                df = df[abs(df[col] - mean) <= 3 * std]
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features"""
        
        # Calculate derived features
        df['calorie_density'] = df['calories_per_100g'] / 100
        df['protein_ratio'] = df['protein_g'] / (df['calories_per_100g'] / 4)  # protein calories / total calories
        df['carb_ratio'] = df['carbohydrates_g'] / (df['calories_per_100g'] / 4)
        df['fat_ratio'] = df['fat_g'] / (df['calories_per_100g'] / 9)
        df['sugar_to_carb_ratio'] = df['sugar_g'] / (df['carbohydrates_g'] + 1e-6)
        df['sodium_per_calorie'] = df['sodium_mg'] / (df['calories_per_100g'] + 1e-6)
        
        # Create health risk score if not present
        if 'health_risk_score' not in df.columns:
            df['health_risk_score'] = self._calculate_risk_score(df)
        
        return df
    
    def _calculate_risk_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate health risk score based on nutrition values"""
        
        risk_scores = []
        
        for _, row in df.iterrows():
            score = 0
            
            # High calorie penalty
            if row['calories_per_100g'] > 400: score += 20
            elif row['calories_per_100g'] > 300: score += 10
            
            # High sugar penalty
            if row['sugar_g'] > 20: score += 25
            elif row['sugar_g'] > 10: score += 15
            
            # High sodium penalty
            if row['sodium_mg'] > 800: score += 20
            elif row['sodium_mg'] > 500: score += 10
            
            # High fat penalty
            if row['fat_g'] > 20: score += 15
            elif row['fat_g'] > 15: score += 10
            
            # Low fiber penalty
            if row['fiber_g'] < 2: score += 10
            
            # High protein bonus
            if row['protein_g'] > 10: score -= 10
            elif row['protein_g'] > 5: score -= 5
            
            # High fiber bonus
            if row['fiber_g'] > 5: score -= 10
            elif row['fiber_g'] > 3: score -= 5
            
            risk_scores.append(max(0, min(100, score)))
        
        return pd.Series(risk_scores)
    
    def prepare_features(self, df: pd.DataFrame):
        """Prepare features for training"""
        
        # Select feature columns
        feature_columns = [
            'calories_per_100g', 'protein_g', 'carbohydrates_g', 'fat_g',
            'fiber_g', 'sugar_g', 'sodium_mg', 'calorie_density',
            'protein_ratio', 'carb_ratio', 'fat_ratio', 'sugar_to_carb_ratio',
            'sodium_per_calorie'
        ]
        
        # Handle categorical features
        if 'category' in df.columns:
            if 'category' not in self.label_encoders:
                self.label_encoders['category'] = LabelEncoder()
                df['category_encoded'] = self.label_encoders['category'].fit_transform(df['category'])
            else:
                df['category_encoded'] = self.label_encoders['category'].transform(df['category'])
            feature_columns.append('category_encoded')
        
        X = df[feature_columns].fillna(0)
        y = df['health_risk_score']
        
        return X, y
    
    def train_all_models(self, df: pd.DataFrame = None) -> Dict:
        """Train and compare all models"""
        
        if df is None:
            df = self.load_and_prepare_data()
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        # Train each model
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for models that need it
            if name in ['neural_network', 'svm']:
                X_train_model = X_train_scaled
                X_test_model = X_test_scaled
            else:
                X_train_model = X_train
                X_test_model = X_test
            
            # Train model
            model.fit(X_train_model, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_model)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_model, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model': model
            }
            
            print(f"RMSE: {rmse:.4f}")
            print(f"R²: {r2:.4f}")
            print(f"CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Track best model
            if r2 > self.best_score:
                self.best_score = r2
                self.best_model = model
        
        # Create ensemble model
        self._create_ensemble_model(results, X_train, y_train)
        
        return results
    
    def _create_ensemble_model(self, results: Dict, X_train, y_train):
        """Create ensemble model from best performers"""
        
        # Select top 3 models based on R² score
        sorted_models = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)
        top_models = sorted_models[:3]
        
        ensemble_estimators = []
        for name, result in top_models:
            ensemble_estimators.append((name, result['model']))
        
        # Create voting regressor
        self.ensemble_model = VotingRegressor(estimators=ensemble_estimators)
        self.ensemble_model.fit(X_train, y_train)
        
        print(f"\nEnsemble created with: {[name for name, _ in top_models]}")
    
    def plot_model_comparison(self, results: Dict):
        """Plot model performance comparison"""
        
        models = list(results.keys())
        r2_scores = [results[model]['r2'] for model in models]
        rmse_scores = [results[model]['rmse'] for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # R² comparison
        ax1.bar(models, r2_scores, color='skyblue', alpha=0.7)
        ax1.set_title('Model R² Score Comparison')
        ax1.set_ylabel('R² Score')
        ax1.set_ylim(0, 1)
        plt.setp(ax1.get_xticklabels(), rotation=45)
        
        # RMSE comparison
        ax2.bar(models, rmse_scores, color='lightcoral', alpha=0.7)
        ax2.set_title('Model RMSE Comparison')
        ax2.set_ylabel('RMSE')
        plt.setp(ax2.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('ml_models/results/nutrition_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_risk(self, nutrition_data: Dict) -> Dict:
        """Predict health risk from nutrition data"""
        
        if self.ensemble_model is None:
            raise ValueError("No trained model available")
        
        # Convert to DataFrame
        df = pd.DataFrame([nutrition_data])
        df = self._engineer_features(df)
        
        # Prepare features
        X, _ = self.prepare_features(df)
        
        # Predict
        risk_score = self.ensemble_model.predict(X)[0]
        
        # Generate risk category
        if risk_score < 30:
            risk_category = "Low Risk"
            color = "green"
        elif risk_score < 60:
            risk_category = "Moderate Risk"
            color = "orange"
        else:
            risk_category = "High Risk"
            color = "red"
        
        return {
            'risk_score': round(risk_score, 1),
            'risk_category': risk_category,
            'color': color,
            'recommendations': self._generate_recommendations(nutrition_data, risk_score)
        }
    
    def _generate_recommendations(self, nutrition_data: Dict, risk_score: float) -> List[str]:
        """Generate health recommendations"""
        
        recommendations = []
        
        if nutrition_data.get('calories_per_100g', 0) > 400:
            recommendations.append("High calorie content - consume in smaller portions")
        
        if nutrition_data.get('sugar_g', 0) > 15:
            recommendations.append("High sugar - limit intake, especially for diabetics")
        
        if nutrition_data.get('sodium_mg', 0) > 800:
            recommendations.append("High sodium - not suitable for hypertension")
        
        if nutrition_data.get('fat_g', 0) > 20:
            recommendations.append("High fat content - consider low-fat alternatives")
        
        if nutrition_data.get('fiber_g', 0) < 3:
            recommendations.append("Low fiber - add fruits/vegetables to meal")
        
        if risk_score > 70:
            recommendations.append("Consider healthier alternatives")
        elif risk_score < 30:
            recommendations.append("Good nutritional choice")
        
        return recommendations
    
    def save_model(self, filepath: str):
        """Save trained models"""
        
        model_data = {
            'ensemble_model': self.ensemble_model,
            'best_model': self.best_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'best_score': self.best_score
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

def main():
    """Main training function"""
    
    # Initialize predictor
    predictor = NutritionRiskPredictor()
    
    # Load and prepare data
    print("Loading and preparing dataset...")
    df = predictor.load_and_prepare_data()
    print(f"Dataset shape: {df.shape}")
    
    # Train models
    print("\nTraining multiple models...")
    results = predictor.train_all_models(df)
    
    # Plot results
    predictor.plot_model_comparison(results)
    
    # Print results
    print("\n" + "="*60)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*60)
    
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  RMSE: {result['rmse']:.4f}")
        print(f"  R²: {result['r2']:.4f}")
        print(f"  MAE: {result['mae']:.4f}")
        print(f"  CV R²: {result['cv_mean']:.4f} (+/- {result['cv_std']*2:.4f})")
    
    # Save model
    predictor.save_model('ml_models/models/nutrition_risk_model.pkl')
    
    # Test prediction
    test_nutrition = {
        'calories_per_100g': 450,
        'protein_g': 8,
        'carbohydrates_g': 65,
        'fat_g': 18,
        'fiber_g': 2,
        'sugar_g': 20,
        'sodium_mg': 600,
        'category': 'snacks'
    }
    
    result = predictor.predict_risk(test_nutrition)
    
    print("\n" + "="*60)
    print("TEST PREDICTION")
    print("="*60)
    print(f"Risk Score: {result['risk_score']}/100")
    print(f"Risk Category: {result['risk_category']}")
    print("Recommendations:")
    for rec in result['recommendations']:
        print(f"  • {rec}")

if __name__ == "__main__":
    main()