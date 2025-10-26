# Complete Implementation Guide: Food Safety ML Project

## 🚀 Phase 1: Setup & Environment (Week 1)

### 1.1 Development Environment Setup

#### Install Required Software
```bash
# 1. Install Python 3.8+
# Download from python.org

# 2. Install Anaconda (recommended)
# Download from anaconda.com

# 3. Create virtual environment
conda create -n foodsense python=3.8
conda activate foodsense

# 4. Install Jupyter Lab
conda install jupyterlab

# 5. Install Git
# Download from git-scm.com
```

#### Create Project Structure
```bash
# Navigate to your project
cd SmartConsumerGuide

# Create ML directories
mkdir -p ml_models/{data/{raw,processed,external},notebooks,src/{data,features,models,visualization},models/{trained_models,model_configs},reports/figures}

# Create requirements file
touch ml_models/requirements.txt
```

#### Install Python Libraries
```bash
# Create requirements.txt
cat > ml_models/requirements.txt << EOF
# Core Data Science
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.1
seaborn==0.12.2
plotly==5.15.0

# Machine Learning
xgboost==1.7.6
tensorflow==2.13.0
keras==2.13.1
lightgbm==4.0.0

# Data Processing
scipy==1.11.1
imbalanced-learn==0.11.0
feature-engine==1.6.2

# Model Evaluation
yellowbrick==1.5
shap==0.42.1
lime==0.2.0.1

# Utilities
joblib==1.3.1
pickle5==0.0.12
tqdm==4.65.0

# Jupyter Extensions
ipywidgets==8.0.7
jupyter-dash==0.4.2
EOF

# Install all packages
pip install -r ml_models/requirements.txt
```

### 1.2 Platform Accounts Setup

#### Kaggle Setup
```bash
# 1. Create account at kaggle.com
# 2. Go to Account → API → Create New API Token
# 3. Download kaggle.json

# Install Kaggle API
pip install kaggle

# Setup credentials (Linux/Mac)
mkdir ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Test connection
kaggle datasets list
```

#### Google Colab Setup (Alternative)
```python
# If using Colab, create this setup cell:
!pip install kaggle
from google.colab import files
files.upload()  # Upload kaggle.json
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

---

## 📊 Phase 2: Data Collection & Exploration (Week 2)

### 2.1 Dataset Collection

#### Primary Datasets from Kaggle
```bash
# Navigate to ML directory
cd ml_models/data/raw

# Download key datasets
kaggle datasets download -d shrutisaxena/food-nutrition-dataset
kaggle datasets download -d thedevastator/the-ultimate-food-composition-dataset
kaggle datasets download -d shuyangli94/food-com-recipes-and-user-interactions
kaggle datasets download -d irkaal/foodcom-recipes-and-reviews

# Extract datasets
unzip "*.zip"
rm *.zip
```

#### Create Data Collection Notebook
```python
# Create: ml_models/notebooks/01_data_collection.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up paths
DATA_RAW = Path('../data/raw')
DATA_PROCESSED = Path('../data/processed')
DATA_PROCESSED.mkdir(exist_ok=True)

# Load datasets
nutrition_df = pd.read_csv(DATA_RAW / 'nutrition.csv')
recipes_df = pd.read_csv(DATA_RAW / 'recipes.csv')

# Basic exploration
print("Dataset shapes:")
print(f"Nutrition: {nutrition_df.shape}")
print(f"Recipes: {recipes_df.shape}")

# Display info
nutrition_df.info()
nutrition_df.head()
```

### 2.2 Custom Data Creation

#### Create Ingredient Toxicity Database
```python
# Create: ml_models/src/data/create_ingredient_db.py

import pandas as pd
import numpy as np

def create_ingredient_toxicity_db():
    """Create synthetic ingredient toxicity database"""
    
    # Common food ingredients with toxicity scores (0-100)
    ingredients_data = {
        'ingredient_name': [
            'sugar', 'salt', 'msg', 'aspartame', 'sodium_benzoate',
            'high_fructose_corn_syrup', 'trans_fat', 'artificial_colors',
            'bht', 'bha', 'sodium_nitrite', 'potassium_sorbate',
            'natural_flavors', 'citric_acid', 'vitamin_c', 'calcium',
            'iron', 'protein', 'fiber', 'whole_wheat'
        ],
        'toxicity_score': [
            75, 70, 85, 80, 65, 90, 95, 75, 85, 85, 90, 45,
            30, 20, 10, 5, 5, 0, 0, 0
        ],
        'category': [
            'sweetener', 'preservative', 'flavor_enhancer', 'sweetener', 'preservative',
            'sweetener', 'fat', 'additive', 'preservative', 'preservative',
            'preservative', 'preservative', 'flavor', 'preservative', 'vitamin',
            'mineral', 'mineral', 'macronutrient', 'fiber', 'grain'
        ],
        'health_impact': [
            'high', 'medium', 'high', 'high', 'medium', 'high', 'very_high',
            'medium', 'high', 'high', 'high', 'low', 'low', 'very_low',
            'beneficial', 'beneficial', 'beneficial', 'beneficial', 'beneficial', 'beneficial'
        ]
    }
    
    df = pd.DataFrame(ingredients_data)
    df.to_csv('../data/processed/ingredient_toxicity_db.csv', index=False)
    return df

# Run the function
ingredient_db = create_ingredient_toxicity_db()
print("Ingredient database created!")
```

---

## 🔍 Phase 3: Data Preprocessing & EDA (Week 3)

### 3.1 Exploratory Data Analysis

#### Create EDA Notebook
```python
# Create: ml_models/notebooks/02_exploratory_data_analysis.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load processed data
ingredient_db = pd.read_csv('../data/processed/ingredient_toxicity_db.csv')

# 1. Distribution Analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Toxicity score distribution
axes[0,0].hist(ingredient_db['toxicity_score'], bins=20, alpha=0.7, color='red')
axes[0,0].set_title('Distribution of Toxicity Scores')
axes[0,0].set_xlabel('Toxicity Score')
axes[0,0].set_ylabel('Frequency')

# Category distribution
ingredient_db['category'].value_counts().plot(kind='bar', ax=axes[0,1])
axes[0,1].set_title('Ingredient Categories')
axes[0,1].tick_params(axis='x', rotation=45)

# Health impact distribution
ingredient_db['health_impact'].value_counts().plot(kind='pie', ax=axes[1,0], autopct='%1.1f%%')
axes[1,0].set_title('Health Impact Distribution')

# Toxicity by category
sns.boxplot(data=ingredient_db, x='category', y='toxicity_score', ax=axes[1,1])
axes[1,1].set_title('Toxicity Score by Category')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('../reports/figures/eda_basic_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Interactive Plotly Visualizations
fig = px.scatter(ingredient_db, 
                x='ingredient_name', 
                y='toxicity_score',
                color='category',
                size='toxicity_score',
                hover_data=['health_impact'],
                title='Interactive Ingredient Toxicity Analysis')

fig.update_layout(xaxis_tickangle=-45)
fig.write_html('../reports/figures/interactive_toxicity_analysis.html')
fig.show()
```

### 3.2 Data Preprocessing Pipeline

#### Create Preprocessing Module
```python
# Create: ml_models/src/data/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib

class FoodDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        
    def preprocess_ingredients(self, df):
        """Preprocess ingredient data"""
        df_processed = df.copy()
        
        # Handle missing values
        df_processed = df_processed.fillna({
            'toxicity_score': df_processed['toxicity_score'].median(),
            'category': 'unknown',
            'health_impact': 'unknown'
        })
        
        # Create binary toxicity classification
        df_processed['is_toxic'] = (df_processed['toxicity_score'] > 50).astype(int)
        
        # Encode categorical variables
        for col in ['category', 'health_impact']:
            le = LabelEncoder()
            df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
            self.label_encoders[col] = le
        
        # Create feature matrix
        feature_cols = ['toxicity_score', 'category_encoded', 'health_impact_encoded']
        X = df_processed[feature_cols]
        y = df_processed['is_toxic']
        
        return X, y, df_processed
    
    def create_user_features(self, user_data):
        """Create features from user profile"""
        features = {
            'age_group': self._categorize_age(user_data.get('age', 30)),
            'allergy_count': len(user_data.get('allergies', [])),
            'health_condition_count': len(user_data.get('health_conditions', [])),
            'risk_tolerance': self._calculate_risk_tolerance(user_data)
        }
        return features
    
    def _categorize_age(self, age):
        if age < 18: return 0  # child
        elif age < 65: return 1  # adult
        else: return 2  # senior
    
    def _calculate_risk_tolerance(self, user_data):
        # Lower tolerance if more health conditions
        base_tolerance = 0.5
        health_conditions = len(user_data.get('health_conditions', []))
        return max(0.1, base_tolerance - (health_conditions * 0.1))

# Usage example
preprocessor = FoodDataPreprocessor()
```

---

## 🤖 Phase 4: Model Development (Weeks 4-6)

### 4.1 Baseline Models Implementation

#### Create Model Training Notebook
```python
# Create: ml_models/notebooks/03_model_development.ipynb

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import joblib

# Load and preprocess data
from src.data.preprocessing import FoodDataPreprocessor

preprocessor = FoodDataPreprocessor()
ingredient_db = pd.read_csv('../data/processed/ingredient_toxicity_db.csv')
X, y, df_processed = preprocessor.preprocess_ingredients(ingredient_db)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"Class distribution: {np.bincount(y_train)}")
```

#### Model 1: Decision Tree
```python
# Decision Tree Implementation
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_model.fit(X_train, y_train)

# Predictions
dt_pred = dt_model.predict(X_test)
dt_prob = dt_model.predict_proba(X_test)[:, 1]

# Evaluation
print("Decision Tree Results:")
print(classification_report(y_test, dt_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, dt_prob):.3f}")

# Save model
joblib.dump(dt_model, '../models/trained_models/decision_tree_model.pkl')
```

#### Model 2: Random Forest
```python
# Random Forest with hyperparameter tuning
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

rf_model = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf_model, rf_params, cv=5, scoring='roc_auc', n_jobs=-1)
rf_grid.fit(X_train, y_train)

# Best model
best_rf = rf_grid.best_estimator_
rf_pred = best_rf.predict(X_test)
rf_prob = best_rf.predict_proba(X_test)[:, 1]

print("Random Forest Results:")
print(f"Best parameters: {rf_grid.best_params_}")
print(classification_report(y_test, rf_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, rf_prob):.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Save model
joblib.dump(best_rf, '../models/trained_models/random_forest_model.pkl')
```

#### Model 3: XGBoost
```python
# XGBoost implementation
xgb_params = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 0.9, 1.0]
}

xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=5, scoring='roc_auc', n_jobs=-1)
xgb_grid.fit(X_train, y_train)

# Best model
best_xgb = xgb_grid.best_estimator_
xgb_pred = best_xgb.predict(X_test)
xgb_prob = best_xgb.predict_proba(X_test)[:, 1]

print("XGBoost Results:")
print(f"Best parameters: {xgb_grid.best_params_}")
print(classification_report(y_test, xgb_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, xgb_prob):.3f}")

# Save model
joblib.dump(best_xgb, '../models/trained_models/xgboost_model.pkl')
```

### 4.2 Neural Network Implementation

#### Create Deep Learning Model
```python
# Neural Network with TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# Prepare data for neural network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build neural network
def create_neural_network(input_dim):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

# Train neural network
nn_model = create_neural_network(X_train_scaled.shape[1])

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)

# Train model
history = nn_model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate
nn_prob = nn_model.predict(X_test_scaled).flatten()
nn_pred = (nn_prob > 0.5).astype(int)

print("Neural Network Results:")
print(classification_report(y_test, nn_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, nn_prob):.3f}")

# Save model
nn_model.save('../models/trained_models/neural_network_model.h5')
joblib.dump(scaler, '../models/trained_models/neural_network_scaler.pkl')
```

---

## 🎯 **Quick Start Commands Summary**

```bash
# 1. Setup Environment
conda create -n foodsense python=3.8
conda activate foodsense
cd SmartConsumerGuide

# 2. Install ML Dependencies
pip install -r ml_models/requirements.txt

# 3. Setup Kaggle
pip install kaggle
# Upload kaggle.json to ~/.kaggle/

# 4. Download Data
cd ml_models/data/raw
kaggle datasets download -d shrutisaxena/food-nutrition-dataset

# 5. Start Development
jupyter lab  # For ML development
npm run dev  # For web app (separate terminal)

# 6. Run ML API
cd ml_models
python src/api/ml_service.py

# 7. Run Tests
python -m pytest tests/ -v

# 8. Generate Results
python src/generate_results.py
```

This comprehensive guide gives you everything needed to implement your ML project from scratch, including exact code, tools, and step-by-step instructions. Each phase builds upon the previous one, ensuring you meet all academic requirements while creating a functional food safety platform.