# Week 4: Data Preprocessing & Feature Engineering Guide

## ✅ Completed Steps

### Step 1: Data Preprocessing Pipeline
**File Created:** `ml_models/notebooks/03_data_preprocessing.ipynb`

**Preprocessing Steps:**
1. ✅ Data cleaning (no missing values, no duplicates)
2. ✅ Data type optimization (categorical → category type)
3. ✅ Feature encoding (Label Encoding for categorical variables)
4. ✅ Feature engineering (4 new features created)
5. ✅ Feature scaling (StandardScaler)
6. ✅ Feature selection (SelectKBest)
7. ✅ Dimensionality reduction (PCA)
8. ✅ Train/test splitting (80/20 split)

### Step 2: Feature Engineering Results
**Original Features:** 5 features
**Engineered Features:** 9 features total

**New Features Created:**
1. `toxicity_level` - Categorical toxicity levels (0-3)
2. `high_toxicity_flag` - Binary flag for high toxicity (>70)
3. `toxicity_score_squared` - Non-linear toxicity relationship
4. `toxicity_score_normalized` - Normalized toxicity (0-1 scale)

**Issue Fixed:** 
- **Problem:** `ValueError: Cannot convert float NaN to integer`
- **Solution:** Added `.fillna(1)` before `.astype(int)` conversion

### Step 3: Feature Selection & Dimensionality Reduction
**Feature Selection Results:**
- **Total Features:** 9
- **Selected Features:** 8 (top performing features)
- **Selection Method:** SelectKBest with f_classif scoring

**PCA Results:**
- **PCA Components:** 4
- **Variance Explained:** 95.3%
- **Original Dimensions:** 9 → **Reduced Dimensions:** 4

### Step 4: Data Splitting & Scaling
**Train/Test Split:**
- **Training Samples:** 32 (80%)
- **Test Samples:** 8 (20%)
- **Stratified Split:** Maintains class distribution
- **Random State:** 42 (reproducible results)

**Feature Scaling:**
- **Method:** StandardScaler (mean=0, std=1)
- **Applied to:** All numeric features
- **Fit on:** Training data only
- **Transform:** Both training and test data

### Step 5: Datasets Created & Saved

**Four Dataset Versions:**
1. **Original:** Raw features without scaling
2. **Scaled:** StandardScaler applied features
3. **Selected Features:** Top 8 features only
4. **PCA:** 4 principal components (95.3% variance)

**Files Saved:**
```
ml_models/data/processed/
├── original/
│   ├── X_train.csv, X_test.csv
│   └── y_train.csv, y_test.csv
├── scaled/
│   ├── X_train.csv, X_test.csv
│   └── y_train.csv, y_test.csv
├── selected_features/
│   ├── X_train.csv, X_test.csv
│   └── y_train.csv, y_test.csv
└── pca/
    ├── X_train.csv, X_test.csv
    └── y_train.csv, y_test.csv
```

**Preprocessing Objects Saved:**
```
ml_models/models/
├── scaler.pkl
├── label_encoders.pkl
├── pca.pkl
├── feature_selector.pkl
└── top_features.pkl
```

## 📊 Preprocessing Summary

### Original Data
- **Samples:** 40 ingredients
- **Features:** 5 original features
- **Missing Values:** 0 (clean dataset)
- **Duplicates:** 0 (no duplicates)

### Feature Engineering
- **Original Features:** 5
- **Engineered Features:** 9 total
- **New Features Created:** 4 additional features

### Data Splitting
- **Training Samples:** 32
- **Test Samples:** 8
- **Training Positive Class:** Balanced distribution
- **Test Positive Class:** Balanced distribution

### Feature Selection
- **Total Features:** 9
- **Selected Features:** 8
- **PCA Components:** 4
- **PCA Variance Explained:** 95.3%

## 🔄 To Rerun Week 4

```bash
# 1. Activate environment
conda activate foodsense

# 2. Navigate to ML directory
cd /Users/mac/Library/CloudStorage/OneDrive-Personal/PROJECTS/FOOD/SmartConsumerGuide/ml_models

# 3. Start Jupyter Lab
jupyter lab

# 4. Open and run: notebooks/03_data_preprocessing.ipynb
```

## 🛠️ Troubleshooting

### NaN Conversion Error
```python
# If you get "Cannot convert float NaN to integer" error:
toxicity_bins = pd.cut(data['toxicity_score'], bins=[0,20,40,70,100], labels=[0,1,2,3])
data['toxicity_level'] = toxicity_bins.fillna(1).astype(int)  # Fill NaN first
```

### Missing Preprocessed Data
```bash
# Check if data exists
ls -la data/processed/

# Rerun preprocessing if needed
jupyter lab notebooks/03_data_preprocessing.ipynb
```

## ✅ Week 4 Status: COMPLETE

**Completed:**
- ✅ Comprehensive data preprocessing pipeline
- ✅ Feature engineering (4 new features)
- ✅ Feature encoding and scaling
- ✅ Feature selection (top 8 features)
- ✅ PCA dimensionality reduction (95.3% variance)
- ✅ Train/test data splitting
- ✅ Four different dataset versions created
- ✅ All preprocessing objects saved for reuse
- ✅ Preprocessing summary report generated

**Key Outputs:**
- **9 total features** (5 original + 4 engineered)
- **4 dataset versions** for different ML approaches
- **95.3% variance** retained with PCA
- **Clean, ML-ready data** for model training

**Ready for Week 5: Model Development & Training**

## 📋 Next Steps (Week 5)

1. **Baseline Model Development**
   - Simple logistic regression baseline
   - Performance benchmarking
   - Evaluation metrics setup

2. **Decision Tree Implementation**
   - Decision tree classifier
   - Hyperparameter tuning
   - Feature importance analysis

3. **Random Forest Training**
   - Ensemble method implementation
   - Cross-validation
   - Model comparison

4. **Model Evaluation & Comparison**
   - Performance metrics comparison
   - ROC curves and confusion matrices
   - Best model selection

**Week 4 preprocessing provides the foundation for all ML model development in Week 5.**