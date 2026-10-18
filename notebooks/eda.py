# ============================================================
# AIMLCZG523 - Machine Learning Operations
# Assignment 1 - Exploratory Data Analysis
# Dataset: Heart Disease UCI Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Create screenshots directory
os.makedirs('screenshots', exist_ok=True)
os.makedirs('data', exist_ok=True)

print("=" * 60)
print("HEART DISEASE UCI - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ============================================================
# SECTION 1: LOAD DATASET
# ============================================================

print("\n--- Loading Dataset ---")
df = pd.read_csv('data/heart.csv')

print(f"Dataset Shape: {df.shape}")
print(f"Total Records: {df.shape[0]}")
print(f"Total Features: {df.shape[1]}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df.head())

# ============================================================
# SECTION 2: DATA OVERVIEW
# ============================================================

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Summary Statistics ---")
print(df.describe())

# ============================================================
# SECTION 3: MISSING VALUES
# ============================================================

print("\n--- Missing Values ---")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct
})
print(missing_df[missing_df['Missing Count'] > 0])

if missing.sum() == 0:
    print("No missing values found!")
else:
    # Fill numeric missing values with median
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"Filled '{col}' with median: {df[col].median()}")

# ============================================================
# SECTION 4: FEATURE ENCODING
# ============================================================

print("\n--- Feature Information ---")
# Identify categorical and numeric columns
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

print(f"Categorical features: {categorical_cols}")
print(f"Numeric features: {numeric_cols}")
print(f"Target: target (0=No Disease, 1=Disease)")

# ============================================================
# SECTION 5: CLASS BALANCE
# ============================================================

print("\n--- Class Balance ---")
class_counts = df['target'].value_counts()
print(class_counts)
print(f"\nClass Distribution:")
print(f"No Heart Disease (0): {class_counts[0]} ({class_counts[0]/len(df)*100:.1f}%)")
print(f"Heart Disease (1): {class_counts[1]} ({class_counts[1]/len(df)*100:.1f}%)")

# Plot class balance
plt.figure(figsize=(8, 5))
colors = ['#2ecc71', '#e74c3c']
bars = plt.bar(['No Heart Disease', 'Heart Disease'],
               class_counts.values,
               color=colors, edgecolor='black', linewidth=0.5)
for bar, count in zip(bars, class_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 2,
             str(count), ha='center', va='bottom', fontweight='bold')
plt.title('Class Balance — Heart Disease Dataset', fontsize=14, fontweight='bold')
plt.xlabel('Class', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig('screenshots/class_balance.png', dpi=150)
plt.show()
print("Class balance chart saved.")

# ============================================================
# SECTION 6: HISTOGRAMS
# ============================================================

print("\n--- Generating Histograms ---")

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Feature Distributions — Heart Disease Dataset',
             fontsize=16, fontweight='bold')

all_features = numeric_cols + ['sex', 'cp', 'fbs', 'restecg']
colors_hist = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12',
               '#9b59b6', '#1abc9c', '#e67e22', '#34495e', '#e91e63']

for idx, (feature, color) in enumerate(zip(all_features, colors_hist)):
    row, col = idx // 3, idx % 3
    axes[row][col].hist(df[feature], bins=20, color=color,
                        alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[row][col].set_title(f'{feature}', fontsize=12, fontweight='bold')
    axes[row][col].set_xlabel('Value')
    axes[row][col].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('screenshots/histograms.png', dpi=150)
plt.show()
print("Histograms saved.")

# ============================================================
# SECTION 7: CORRELATION HEATMAP
# ============================================================

print("\n--- Generating Correlation Heatmap ---")

plt.figure(figsize=(12, 10))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='RdYlGn',
    center=0,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)
plt.title('Correlation Heatmap — Heart Disease Features',
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('screenshots/correlation_heatmap.png', dpi=150)
plt.show()
print("Correlation heatmap saved.")

# ============================================================
# SECTION 8: FEATURE vs TARGET ANALYSIS
# ============================================================

print("\n--- Feature vs Target Analysis ---")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Key Features vs Heart Disease Target',
             fontsize=16, fontweight='bold')

key_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'cp']

for idx, feature in enumerate(key_features):
    row, col = idx // 3, idx % 3
    df.boxplot(column=feature, by='target', ax=axes[row][col],
               patch_artist=True)
    axes[row][col].set_title(f'{feature} by Target')
    axes[row][col].set_xlabel('Target (0=No Disease, 1=Disease)')
    axes[row][col].set_ylabel(feature)

plt.tight_layout()
plt.savefig('screenshots/feature_vs_target.png', dpi=150)
plt.show()
print("Feature vs target chart saved.")

# ============================================================
# SECTION 9: AGE DISTRIBUTION BY TARGET
# ============================================================

plt.figure(figsize=(10, 6))
df[df['target'] == 0]['age'].hist(bins=20, alpha=0.7,
                                   label='No Disease', color='#2ecc71')
df[df['target'] == 1]['age'].hist(bins=20, alpha=0.7,
                                   label='Heart Disease', color='#e74c3c')
plt.title('Age Distribution by Heart Disease Status',
          fontsize=14, fontweight='bold')
plt.xlabel('Age', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('screenshots/age_distribution.png', dpi=150)
plt.show()
print("Age distribution chart saved.")

# ============================================================
# SECTION 10: SAVE CLEANED DATA
# ============================================================

df.to_csv('data/heart_cleaned.csv', index=False)
print("\nCleaned dataset saved to data/heart_cleaned.csv")

print("\n" + "=" * 60)
print("EDA COMPLETE!")
print("Screenshots saved in screenshots/ folder")
print("=" * 60)
