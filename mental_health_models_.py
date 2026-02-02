# -*- coding: utf-8 -*-
"""Mental Health Models - Corrected Version

This script performs three main tasks:
1. Prediction of depression using SVM
2. Clustering of mental health conditions
3. Classification of important factors for each illness

Author: Rihab CHAHED
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os

# Imports for modeling
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, silhouette_score, pairwise_distances)
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

# For handling imbalanced data (install if needed: pip install imbalanced-learn)
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    print("Warning: SMOTE not available. Install with: pip install imbalanced-learn")
    SMOTE_AVAILABLE = False

# Install and import pyreadstat
try:
    import pyreadstat
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'pyreadstat'])
    import pyreadstat

# Ensure output directory exists
OUTPUT_DIR = '/mnt/user-data/outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 80)
print("LOADING DATA")
print("=" * 80)

# Update this path to your actual file location
mental_health_data = pyreadstat.read_sav("/content/base_projetIAfinal_scolaire_2019_HTA.sav")
mental_health_data = pd.DataFrame(mental_health_data[0])

print(f"Data shape: {mental_health_data.shape}")
print(f"\n First rows:\n{mental_health_data.head()}")

# =============================================================================
# 1. DEPRESSION PREDICTION MODEL (SVM)
# =============================================================================
print("\n" + "=" * 80)
print("1. DEPRESSION PREDICTION MODEL (SVM)")
print("=" * 80)

# Select relevant features
predict_data = mental_health_data[['d_atcd_redoublemet',
                                    'addiction_FB', 'copie_addictjv21items',
                                    'TABACado', 'ALCOOLado', 'INACTIVITEPHYSIQUEado',
                                    'ALIMENTATIONMALSAINEado', 'OBESITEado', 'severité_depression']].copy()

# Rename columns for clarity
predict_data.columns = ['redouble', 'addict_fb', 'addict_vidg', 'tabac',
                        'alcool', 'activ_phy', 'aliment_saine', 'obesite', 'sever_dep']

print(f"\nMissing values before imputation:\n{predict_data.isnull().sum()}")

#CORRECTION 1: Use mode imputation for categorical/binary data (NOT linear interpolation)
imputer = SimpleImputer(strategy='most_frequent')
predict_data_imputed = pd.DataFrame(
    imputer.fit_transform(predict_data),
    columns=predict_data.columns
)

print(f"\n Missing values after imputation:\n{predict_data_imputed.isnull().sum()}")

# Binary transformation: 0,1 → 0 (no depression), others → 1 (depression)
predict_data_imputed['sever_dep'] = predict_data_imputed['sever_dep'].apply(
    lambda x: 0 if x == 0 else 1
)

print(f"\n Class distribution:\n{predict_data_imputed['sever_dep'].value_counts()}")
print(f"Class balance: {predict_data_imputed['sever_dep'].value_counts(normalize=True)}")

# Separate features and target
X = predict_data_imputed.drop(columns='sever_dep')
y = predict_data_imputed['sever_dep']

#CORRECTION 2: Split BEFORE scaling to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

#CORRECTION 3: Fit scaler ONLY on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform, no fit!

# Handle class imbalance with SMOTE (if available)
if SMOTE_AVAILABLE and y_train.value_counts()[0] / y_train.value_counts()[1] > 2:
    print("\n  Class imbalance detected. Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(f"After SMOTE: {pd.Series(y_train_resampled).value_counts()}")
else:
    X_train_resampled = X_train_scaled
    y_train_resampled = y_train

# Train SVM model
print("\n Training SVM model...")
classifier = svm.SVC(kernel='linear', random_state=42)
classifier.fit(X_train_resampled, y_train_resampled)

#CORRECTION 4: Evaluate on BOTH train and test sets
y_train_pred = classifier.predict(X_train_scaled)
y_test_pred = classifier.predict(X_test_scaled)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\n Results:")
print(f"Training Accuracy: {train_accuracy:.3f}")
print(f"Test Accuracy: {test_accuracy:.3f}")

#CORRECTION 5: Add detailed metrics
print(f"\n Classification Report (Test Set):")
print(classification_report(y_test, y_test_pred,
                          target_names=['No Depression', 'Depression']))

print(f"\n Confusion Matrix (Test Set):")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Depression', 'Depression'],
            yticklabels=['No Depression', 'Depression'])
plt.title('Confusion Matrix - Depression Prediction')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_depression.png'), dpi=300, bbox_inches='tight')
plt.show()

#CORRECTION 6: Cross-validation for robustness
print("\n Cross-validation (5-fold):")
cv_scores = cross_val_score(classifier, X_train_scaled, y_train, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# Save model and scaler
print("\n Saving model and scaler...")
joblib.dump(classifier, os.path.join(OUTPUT_DIR, 'depression_svm_model.pkl'))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))
print("Model saved successfully!")

# Example prediction with new data
print("\n Example prediction:")
input_data = np.array([[0, 1, 0, 1, 0, 1, 1, 0]])  # Example patient data
input_data_scaled = scaler.transform(input_data)
prediction = classifier.predict(input_data_scaled)

if prediction[0] == 0:
    print("Result: The person is NOT depressed")
else:
    print("Result: The person IS depressed")

# =============================================================================
# 2. CLUSTERING MODEL - COMORBIDITIES
# =============================================================================
print("\n" + "=" * 80)
print("2. CLUSTERING MODEL - IDENTIFYING COMORBIDITIES")
print("=" * 80)

# Select mental health conditions
necessary_features = ['f_trouble_anxieux', 'g_estime_soi', 'severité_depression',
                     'alexithymie_stades', 'addiction_FB', 'copie_addictjv21items']

mental_illness = mental_health_data[necessary_features].copy()

print(f"\n Missing values before imputation:\n{mental_illness.isnull().sum()}")

# Impute missing values
imputer_cluster = SimpleImputer(strategy='most_frequent')
mental_illness_imputed = pd.DataFrame(
    imputer_cluster.fit_transform(mental_illness),
    columns=mental_illness.columns
)

# Convert to binary (0 or 1)
def binarize(x):
    return 1 if x > 1 else x

mental_illness_binary = mental_illness_imputed.applymap(binarize)

# Visualize distribution of mental illnesses
counts = mental_illness_binary.sum()
plt.figure(figsize=(8, 8))
plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Distribution of Mental Health Conditions')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'mental_illness_distribution.png'), dpi=300, bbox_inches='tight')
plt.show()

# Calculate pairwise distances using Jaccard (appropriate for binary data)
print("\n Calculating pairwise distances (Jaccard metric)...")
distances = pairwise_distances(mental_illness_binary.values, metric='jaccard')

#CORRECTION 7: Find optimal number of clusters using silhouette score
print("\n Finding optimal number of clusters...")
silhouette_scores = []
cluster_range = range(2, 8)

for n_clusters in cluster_range:
    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='complete'
    )
    cluster_labels = clusterer.fit_predict(distances)
    silhouette_avg = silhouette_score(distances, cluster_labels, metric='precomputed')
    silhouette_scores.append(silhouette_avg)
    print(f"n_clusters = {n_clusters}, silhouette score = {silhouette_avg:.3f}")

# Plot silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(cluster_range, silhouette_scores, marker='o', linewidth=2)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Optimal Number of Clusters (Silhouette Method)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'silhouette_scores.png'), dpi=300, bbox_inches='tight')
plt.show()

# Use optimal number of clusters (highest silhouette score)
optimal_n_clusters = cluster_range[np.argmax(silhouette_scores)]
print(f"\n Optimal number of clusters: {optimal_n_clusters}")

# Perform final clustering
cluster = AgglomerativeClustering(
    n_clusters=optimal_n_clusters,
    metric='precomputed',
    linkage='complete'
)
mental_illness_binary['cluster'] = cluster.fit_predict(distances)

# Identify common conditions in each cluster
print(f"\n Common mental health conditions per cluster:")
for cluster_label in sorted(mental_illness_binary['cluster'].unique()):
    cluster_data = mental_illness_binary[mental_illness_binary['cluster'] == cluster_label]
    cluster_data = cluster_data.drop(columns='cluster')
    common_conditions = cluster_data.sum().sort_values(ascending=False).head(3)
    print(f"\n🔹 Cluster {cluster_label} ({len(cluster_data)} individuals):")
    for condition, count in common_conditions.items():
        percentage = (count / len(cluster_data)) * 100
        print(f"   - {condition}: {count} ({percentage:.1f}%)")

# =============================================================================
# 3. FEATURE IMPORTANCE - IDENTIFYING RISK FACTORS
# =============================================================================
print("\n" + "=" * 80)
print("3. FEATURE IMPORTANCE - IDENTIFYING RISK FACTORS")
print("=" * 80)

# Select features for classification
f_needed = ['d_atcd_redoublemet', 'f_trouble_anxieux', 'g_estime_soi',
           'severité_depression', 'alexithymie_stades', 'argent_poche_semaine',
           'addiction_FB', 'copie_addictjv21items',
           'TABACado', 'ALCOOLado', 'INACTIVITEPHYSIQUEado',
           'ALIMENTATIONMALSAINEado', 'OBESITEado']

classification = mental_health_data[f_needed].copy()

# Impute missing values
imputer_class = SimpleImputer(strategy='most_frequent')
classification_imputed = pd.DataFrame(
    imputer_class.fit_transform(classification),
    columns=classification.columns
)

# Define features and targets
features = classification_imputed.drop(
    ['f_trouble_anxieux', 'g_estime_soi', 'severité_depression',
     'alexithymie_stades', 'addiction_FB', 'copie_addictjv21items'],
    axis=1
)

targets = {
    'Anxiety': classification_imputed['f_trouble_anxieux'],
    'Self-Esteem': classification_imputed['g_estime_soi'],
    'Depression': classification_imputed['severité_depression'],
    'Alexithymia': classification_imputed['alexithymie_stades'],
    'Facebook Addiction': classification_imputed['addiction_FB'],
    'Gaming Addiction': classification_imputed['copie_addictjv21items']
}

#CORRECTION 8: Fix copy-paste errors - use correct targets
print("\n Analyzing feature importance for each condition...")

fig, axes = plt.subplots(3, 2, figsize=(16, 18))
axes = axes.ravel()

for idx, (condition_name, target) in enumerate(targets.items()):
    print(f"\n Analyzing: {condition_name}")

    # Determine if regression or classification
    if target.nunique() > 10:  # Continuous variable
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:  # Categorical variable
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Fit model
    model.fit(features, target)

    # Get feature importance
    importance = pd.Series(
        model.feature_importances_,
        index=features.columns
    ).sort_values(ascending=False)

    # Plot
    sns.barplot(x=importance.values, y=importance.index, ax=axes[idx], palette='viridis')
    axes[idx].set_title(f'Feature Importance for {condition_name}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Importance Score', fontsize=10)
    axes[idx].set_ylabel('')

    # Print top 3 factors
    print(f"  Top 3 factors: {importance.head(3).to_dict()}")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance_all_conditions.png'), dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\n Generated files:")
print("  - depression_svm_model.pkl (trained model)")
print("  - scaler.pkl (data scaler)")
print("  - confusion_matrix_depression.png")
print("  - mental_illness_distribution.png")
print("  - silhouette_scores.png")
print("  - feature_importance_all_conditions.png")
print("\n" + "=" * 80)
