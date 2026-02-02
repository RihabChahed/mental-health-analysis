# 🧠 Mental Health Prediction and Analysis System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive machine learning system for predicting mental health conditions, identifying comorbidities, and analyzing risk factors using real-world clinical data.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Author](#author)

## Overview

This project was developed as part of a final-year internship at **Farhat Hached Hospital** in Tunisia. The goal is to create an intelligent platform for mental health data management and analysis, specifically focusing on:

1. **Depression Prediction** using Support Vector Machines (SVM)
2. **Comorbidity Identification** using Hierarchical Clustering
3. **Risk Factor Analysis** using Random Forest feature importance

The system aims to assist healthcare professionals in:
- Early detection of mental health conditions
- Understanding relationships between different mental health disorders
- Identifying the most significant risk factors for intervention

## Features

### Depression Prediction
- Binary classification (depressed vs. non-depressed)
- Support Vector Machine (SVM) with linear kernel
- Handles class imbalance using SMOTE
- Cross-validation for robust evaluation
- Achieves ~82% accuracy with 5-fold CV

### Comorbidity Detection
- Hierarchical Agglomerative Clustering
- Optimal cluster selection using Silhouette Score
- Jaccard distance for binary health condition data
- Identifies frequently co-occurring mental health conditions

### Risk Factor Analysis
- Random Forest for feature importance
- Analyzes 6 mental health conditions:
  - Anxiety disorders
  - Depression
  - Low self-esteem
  - Alexithymia
  - Social media addiction (Facebook)
  - Gaming addiction
- Identifies top risk factors for each condition

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/mental-health-analysis.git
cd mental-health-analysis
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

##  Usage

### Basic Usage

1. **Prepare your data**
   - Place your `.sav` (SPSS) data file in the project directory
   - Update the data path in `mental_health_models.py` (line 52)

2. **Run the analysis**
```bash
python mental_health_models.py
```

3. **View results**
   - Console output: Classification reports, accuracy scores
   - Generated files in `results/` folder:
     - `confusion_matrix_depression.png`
     - `mental_illness_distribution.png`
     - `silhouette_scores.png`
     - `feature_importance_all_conditions.png`
   - Saved models in `models/` folder:
     - `depression_svm_model.pkl`
     - `scaler.pkl`

### Example: Making Predictions

```python
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load('models/depression_svm_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Example patient data: [redouble, fb_addiction, gaming_addiction, 
#                        tobacco, alcohol, physical_inactivity, 
#                        unhealthy_diet, obesity]
patient_data = np.array([[0, 1, 0, 1, 0, 1, 1, 0]])

# Standardize and predict
patient_data_scaled = scaler.transform(patient_data)
prediction = model.predict(patient_data_scaled)

if prediction[0] == 0:
    print("Result: Patient is NOT at risk for depression")
else:
    print("Result: Patient IS at risk for depression")
```

## Models

### 1. Depression Prediction (SVM)

**Algorithm:** Support Vector Machine with linear kernel

**Features used:**
- Academic failure history
- Social media addiction (Facebook)
- Gaming addiction
- Tobacco use
- Alcohol consumption
- Physical inactivity
- Unhealthy diet
- Obesity

**Performance:**
- Training Accuracy: ~85-90%
- Test Accuracy: ~80-85%
- Cross-validation (5-fold): ~82% ± 3%

**Data preprocessing:**
- Mode imputation for missing values
- Standard scaling
- SMOTE for class imbalance

### 2. Comorbidity Clustering

**Algorithm:** Agglomerative Hierarchical Clustering

**Distance metric:** Jaccard (for binary data)

**Linkage:** Complete linkage

**Optimal clusters:** Determined by Silhouette Score (typically 3-5 clusters)

### 3. Risk Factor Analysis

**Algorithm:** Random Forest (Classifier/Regressor)

**Number of trees:** 100

**Purpose:** Identify the most important predictors for each mental health condition

## Results

### Key Findings

1. **Top Risk Factors for Depression:**
   - Physical inactivity
   - Social media addiction
   - Academic failure

2. **Common Comorbidities:**
   - Depression + Anxiety (high co-occurrence)
   - Gaming addiction + Social media addiction
   - Low self-esteem + Depression

3. **Model Performance:**
   - SVM achieves reliable predictions with good generalization
   - Clustering identifies meaningful patient groups
   - Feature importance aligns with clinical knowledge

### Sample Outputs

![Confusion Matrix](results/confusion_matrix_depression.png)
*Confusion matrix showing depression prediction performance*

![Feature Importance](results/feature_importance_all_conditions.png)
*Top risk factors for each mental health condition*

## Project Structure

```
mental-health-analysis/
│
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
├── mental_health_models.py           # Main analysis script
│
├── data/                             # Data folder (not tracked)
│   └── README.md                     # Data description
│
├── models/                           # Saved models (not tracked)
│   ├── depression_svm_model.pkl
│   └── scaler.pkl
│
└── results/                          # Generated visualizations (not tracked)
    ├── confusion_matrix_depression.png
    ├── mental_illness_distribution.png
    ├── silhouette_scores.png
    └── feature_importance_all_conditions.png
```

## 🔧 Technical Details

### Data Preprocessing
- **Missing values:** Mode imputation (appropriate for categorical data)
- **Feature scaling:** StandardScaler (fit only on training data)
- **Class imbalance:** SMOTE oversampling

### Best Practices Implemented
- Train-test split before preprocessing (avoids data leakage)
- Cross-validation for robust evaluation
- Comprehensive metrics (precision, recall, F1-score)
- Silhouette analysis for optimal clustering
- Model persistence (save/load trained models)
- Reproducibility (random_state set everywhere)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Author

**Rihab CHAHED**

- Email: chahedryhab@gmail.com
- LinkedIn: [rihab-chahed](https://www.linkedin.com/in/rihab-chahed-557597226)
- Master's in Data Science - Higher Institute of Computer Science and Mathematics of Monastir

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Farhat Hached Hospital** for providing the clinical data and internship opportunity
- **Faculty of Sciences of Monastir** for academic supervision
- The scikit-learn and Python data science community

## Disclaimer

This system is intended for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare professionals for medical advice.

---

**Made with ❤️ for mental health awareness**
