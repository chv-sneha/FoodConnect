# Week 2: Data Collection Guide

## ✅ Completed Steps

### Step 1: Kaggle API Setup
**Platform Setup:**
1. Went to kaggle.com → Account → API → Create New API Token
2. Downloaded kaggle.json file

**Terminal Commands:**
```bash
# Activate environment
conda activate foodsense

# Create kaggle directory
mkdir -p ~/.kaggle

# Move downloaded file
mv ~/Downloads/kaggle.json ~/.kaggle/

# Set permissions
chmod 600 ~/.kaggle/kaggle.json

# Test connection
kaggle datasets list
```

**Result:** ✅ Kaggle API working - saw list of datasets

### Step 2: Download Food Datasets
**Navigate to raw data directory:**
```bash
cd /Users/mac/Library/CloudStorage/OneDrive-Personal/PROJECTS/FOOD/SmartConsumerGuide/ml_models/data/raw
```

**Download datasets:**
```bash
# Successfully downloaded
kaggle datasets download -d shrutisaxena/food-nutrition-dataset

# This one was restricted (403 Forbidden)
kaggle datasets download -d thedevastator/the-ultimate-food-composition-dataset

# Extract files
unzip "*.zip"

# Clean up
rm *.zip

# Check files
ls -la
```

**Result:** ✅ Downloaded food-nutrition-dataset
- `food.csv` (2.6MB)
- `food1.csv` (2.6MB)

### Step 3: Create Custom Ingredient Database
**Navigate to source directory:**
```bash
cd ../../src/data
```

**Run Python script:**
```bash
python create_ingredient_db.py
```

**Result:** ✅ Created ingredient database
- **40 ingredients** with toxicity scores
- **Risk distribution:**
  - High Risk: 9 ingredients
  - Medium Risk: 8 ingredients  
  - Low Risk: 8 ingredients
  - Safe: 7 ingredients
- **Saved to:** `data/processed/ingredient_toxicity_db.csv`

### Step 4: Data Collection Notebook
**Start Jupyter Lab:**
```bash
cd ../..
jupyter lab
```

**Notebook:** `notebooks/01_data_collection.ipynb`

**Key modifications made:**
- **Cell 3 Issue:** Original code tried to recreate database but path was wrong
- **Solution:** Replaced with loading existing database:
```python
# Load the existing database we already created
ingredient_db = pd.read_csv('../data/processed/ingredient_toxicity_db.csv')
print(f"✅ Loaded ingredient database with {len(ingredient_db)} ingredients")
ingredient_db.head(10)
```

**Notebook Results:**
- ✅ Loaded ingredient database (40 ingredients)
- ✅ Analyzed data quality (no missing values, no duplicates)
- ✅ Created visualizations
- ✅ Saved processed data

## 📊 Data Summary

### Ingredient Toxicity Database
- **Total ingredients:** 40
- **Features:** ingredient_name, toxicity_score, category, health_impact, allergen_risk, is_toxic, risk_level
- **Categories:** sweetener, preservative, flavor_enhancer, additive, etc.
- **Toxicity range:** 0-95 (lower is safer)

### Kaggle Food Dataset
- **Files:** food.csv, food1.csv
- **Size:** 2.6MB each
- **Content:** Food nutrition information

### Risk Level Distribution
```
High Risk      9 ingredients (70-100 toxicity)
Medium Risk    8 ingredients (40-69 toxicity)  
Low Risk       8 ingredients (20-39 toxicity)
Safe           7 ingredients (0-19 toxicity)
```

## 📁 Files Created

### Data Files
- `ml_models/data/raw/food.csv`
- `ml_models/data/raw/food1.csv`
- `ml_models/data/processed/ingredient_toxicity_db.csv`

### Code Files
- `ml_models/src/data/create_ingredient_db.py`
- `ml_models/notebooks/01_data_collection.ipynb`

### Visualizations
- `ml_models/reports/figures/week2_data_exploration.png`

## 🔄 To Rerun Week 2

```bash
# 1. Activate environment
conda activate foodsense

# 2. Navigate to ML directory
cd /Users/mac/Library/CloudStorage/OneDrive-Personal/PROJECTS/FOOD/SmartConsumerGuide/ml_models

# 3. Create ingredient database (if needed)
cd src/data
python create_ingredient_db.py
cd ../..

# 4. Start Jupyter Lab
jupyter lab

# 5. Open and run: notebooks/01_data_collection.ipynb
```

## 🛠️ Troubleshooting

### Kaggle API Issues
```bash
# Check if kaggle.json exists
ls -la ~/.kaggle/

# Test connection
kaggle datasets list

# Re-download token if needed from kaggle.com
```

### File Path Issues
```bash
# Check current directory
pwd

# Navigate to correct directory
cd /Users/mac/Library/CloudStorage/OneDrive-Personal/PROJECTS/FOOD/SmartConsumerGuide/ml_models
```

### Jupyter Notebook Issues
- **FileNotFoundError:** Use `pd.read_csv('../data/processed/ingredient_toxicity_db.csv')` instead of recreating
- **Import errors:** Make sure you're in the right directory and environment is activated

## ✅ Week 2 Status: COMPLETE

**Completed:**
- ✅ Kaggle API setup and authentication
- ✅ Downloaded food nutrition dataset (2 files)
- ✅ Created custom ingredient toxicity database (40 ingredients)
- ✅ Data collection notebook executed successfully
- ✅ Data quality assessment completed
- ✅ Visualizations created and saved
- ✅ Processed data saved for Week 3

**Ready for Week 3: Data Preprocessing & EDA**

## 📋 Next Steps (Week 3)

1. **Exploratory Data Analysis (EDA)**
   - Statistical analysis of food datasets
   - Correlation analysis
   - Distribution plots

2. **Data Preprocessing**
   - Handle missing values
   - Feature scaling/normalization
   - Categorical encoding

3. **Feature Engineering**
   - Create new features from existing data
   - Feature selection
   - Dimensionality reduction

4. **Data Splitting**
   - Train/test split
   - Validation set creation
   - Stratified sampling for balanced datasets