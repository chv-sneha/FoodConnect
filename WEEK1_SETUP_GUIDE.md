# Week 1: Environment Setup Guide

## ✅ Completed Steps

### Step 1: Install Anaconda
- Downloaded from https://www.anaconda.com/download
- Installed with default settings
- Verified with: `conda --version`

### Step 2: Create Python Environment
```bash
# Create environment
conda create -n foodsense python=3.8 -y

# Activate environment
conda activate foodsense

# Verify Python version
python --version
```

### Step 3: Create Project Structure
```bash
# Navigate to project
cd /Users/mac/Library/CloudStorage/OneDrive-Personal/PROJECTS/FOOD/SmartConsumerGuide

# Create ML directories
mkdir -p ml_models/data/{raw,processed,external}
mkdir -p ml_models/notebooks
mkdir -p ml_models/src/{data,features,models,visualization}
mkdir -p ml_models/models/{trained_models,model_configs}
mkdir -p ml_models/reports/figures
```

### Step 4: Install Python Packages
Created `ml_models/requirements.txt` with Python 3.8 compatible versions:
```
# Core Data Science
pandas==1.3.5
numpy==1.21.6
scikit-learn==1.0.2
matplotlib==3.5.3
seaborn==0.11.2

# Machine Learning
xgboost==1.6.2
tensorflow==2.8.4

# Data Processing
scipy==1.7.3

# Utilities
joblib==1.1.1
tqdm==4.64.1

# Jupyter Extensions
ipywidgets==7.7.2

# Kaggle API
kaggle==1.5.12
```

Installed with:
```bash
cd ml_models
pip install -r requirements.txt
```

### Step 5: Test Environment
Created test notebook: `ml_models/notebooks/00_environment_test.ipynb`

## 🔄 To Reactivate Environment Later

```bash
# Navigate to project
cd /Users/mac/Library/CloudStorage/OneDrive-Personal/PROJECTS/FOOD/SmartConsumerGuide/ml_models

# Activate environment
conda activate foodsense

# Start Jupyter Lab
jupyter lab
```

## 📋 Next Steps (Week 2)

### Kaggle Setup (Still Needed)
1. Go to kaggle.com → Account → API → Create New API Token
2. Download kaggle.json
3. Run:
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets list  # Test connection
```

### Data Collection
- Download food datasets from Kaggle
- Create ingredient toxicity database
- Start exploratory data analysis

## 🛠️ Troubleshooting

### If environment doesn't activate:
```bash
conda info --envs  # List all environments
conda activate foodsense
```

### If packages missing:
```bash
conda activate foodsense
pip install -r requirements.txt
```

### If Jupyter doesn't start:
```bash
conda activate foodsense
conda install jupyterlab -y
jupyter lab
```

## ✅ Week 1 Status: COMPLETE
- ✅ Anaconda installed
- ✅ Python 3.8 environment created
- ✅ Project structure created
- ✅ All ML libraries installed
- ✅ Test notebook created
- 🔄 Kaggle API setup (pending)

**Ready for Week 2: Data Collection & Exploration**