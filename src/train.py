# ============================================================
# AIMLCZG523 - Machine Learning Operations
# Assignment 1 - Model Training + MLflow Experiment Tracking
# Dataset: Heart Disease UCI Dataset
# Models: Logistic Regression + Random Forest
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)
from sklearn.pipeline import Pipeline

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/heart_cleaned.csv"
MODEL_DIR = "models/"
SCREENSHOTS_DIR = "screenshots/"
MLFLOW_EXPERIMENT = "Heart Disease Classification"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

print("=" * 60)
print("HEART DISEASE - MODEL TRAINING + EXPERIMENT TRACKING")
print("=" * 60)

# ============================================================
# SECTION 1: LOAD & PREPARE DATA
# ============================================================

print("\n--- Loading Dataset ---")
df = pd.read_csv(DATA_PATH)
print(f"Dataset Shape: {df.shape}")

# Features and target
X = df.drop('target', axis=1)
y = df['target']

print(f"Features shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# ============================================================
# SECTION 2: FEATURE ENGINEERING
# ============================================================

print("\n--- Feature Engineering ---")

# Identify categorical and numeric columns
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# One-hot encode categorical features
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
print(f"Shape after encoding: {X_encoded.shape}")
print(f"Features after encoding: {X_encoded.columns.tolist()}")

# Save feature names for reproducibility
feature_names = X_encoded.columns.tolist()
with open(f"{MODEL_DIR}feature_names.pkl", 'wb') as f:
    pickle.dump(feature_names, f)
print(f"Feature names saved to {MODEL_DIR}feature_names.pkl")

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}")
print(f"Test size: {X_test.shape[0]}")

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# Scale only numeric columns that exist after encoding
numeric_present = [col for col in numeric_cols if col in X_train.columns]
X_train_scaled[numeric_present] = scaler.fit_transform(X_train[numeric_present])
X_test_scaled[numeric_present] = scaler.transform(X_test[numeric_present])

# Save scaler for reproducibility
with open(f"{MODEL_DIR}scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to {MODEL_DIR}scaler.pkl")

# ============================================================
# HELPER FUNCTION: Evaluate Model
# ============================================================

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate model and return metrics dictionary."""

    # Cross validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

    print(f"\n{model_name} Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  CV Score:  {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred,
          target_names=['No Disease', 'Heart Disease']))

    return metrics, y_pred, y_prob


# ============================================================
# HELPER FUNCTION: Save Plots
# ============================================================

def save_confusion_matrix(y_test, y_pred, model_name, filename):
    """Save confusion matrix plot."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Disease', 'Heart Disease'],
                yticklabels=['No Disease', 'Heart Disease'])
    plt.title(f'Confusion Matrix — {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    return filename


def save_roc_curve(y_test, y_prob, model_name, filename):
    """Save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#e74c3c', lw=2,
             label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve — {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    return filename


# ============================================================
# SECTION 3: MLFLOW SETUP
# ============================================================

print("\n--- Setting up MLflow ---")
db_path = os.path.abspath('mlflow.db').replace('\\', '/')
mlflow.set_tracking_uri(f"sqlite:///{db_path}")
mlflow.set_experiment(MLFLOW_EXPERIMENT)
print(f"MLflow experiment: {MLFLOW_EXPERIMENT}")
print(f"MLflow tracking URI: mlruns/")

# ============================================================
# SECTION 4: MODEL 1 — LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 60)

lr_params = {
    'C': 1.0,
    'max_iter': 1000,
    'solver': 'lbfgs',
    'random_state': RANDOM_STATE
}

with mlflow.start_run(run_name="Logistic_Regression"):

    # Train model
    lr_model = LogisticRegression(**lr_params)
    lr_model.fit(X_train_scaled, y_train)

    # Evaluate
    lr_metrics, lr_pred, lr_prob = evaluate_model(
        lr_model, X_train_scaled, X_test_scaled,
        y_train, y_test, "Logistic Regression"
    )

    # Save plots
    lr_cm_path = save_confusion_matrix(
        y_test, lr_pred, "Logistic Regression",
        f"{SCREENSHOTS_DIR}lr_confusion_matrix.png"
    )
    lr_roc_path = save_roc_curve(
        y_test, lr_prob, "Logistic Regression",
        f"{SCREENSHOTS_DIR}lr_roc_curve.png"
    )

    # Log to MLflow
    mlflow.log_params(lr_params)
    mlflow.log_params({
        'model_type': 'Logistic Regression',
        'test_size': TEST_SIZE,
        'cv_folds': CV_FOLDS,
        'random_state': RANDOM_STATE,
        'n_features': X_train_scaled.shape[1],
        'train_samples': X_train_scaled.shape[0]
    })
    mlflow.log_metrics({
        'accuracy': lr_metrics['accuracy'],
        'precision': lr_metrics['precision'],
        'recall': lr_metrics['recall'],
        'roc_auc': lr_metrics['roc_auc'],
        'cv_mean': lr_metrics['cv_mean'],
        'cv_std': lr_metrics['cv_std']
    })
    mlflow.log_artifact(lr_cm_path)
    mlflow.log_artifact(lr_roc_path)
    mlflow.sklearn.log_model(lr_model, "logistic_regression_model")

    print("\nLogistic Regression logged to MLflow!")

# Save model locally
with open(f"{MODEL_DIR}logistic_regression.pkl", 'wb') as f:
    pickle.dump(lr_model, f)
print(f"Logistic Regression saved to {MODEL_DIR}logistic_regression.pkl")

# ============================================================
# SECTION 5: MODEL 2 — RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("MODEL 2: RANDOM FOREST")
print("=" * 60)

rf_params = {
    'n_estimators': 100,
    'max_depth': 6,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE
}

with mlflow.start_run(run_name="Random_Forest"):

    # Train model
    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(X_train_scaled, y_train)

    # Evaluate
    rf_metrics, rf_pred, rf_prob = evaluate_model(
        rf_model, X_train_scaled, X_test_scaled,
        y_train, y_test, "Random Forest"
    )

    # Feature importance plot
    feature_importance = pd.DataFrame({
        'Feature': X_train_scaled.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='Importance', y='Feature',
                palette='viridis')
    plt.title('Top 10 Feature Importances — Random Forest',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    rf_fi_path = f"{SCREENSHOTS_DIR}rf_feature_importance.png"
    plt.savefig(rf_fi_path, dpi=150)
    plt.close()

    # Save plots
    rf_cm_path = save_confusion_matrix(
        y_test, rf_pred, "Random Forest",
        f"{SCREENSHOTS_DIR}rf_confusion_matrix.png"
    )
    rf_roc_path = save_roc_curve(
        y_test, rf_prob, "Random Forest",
        f"{SCREENSHOTS_DIR}rf_roc_curve.png"
    )

    # Log to MLflow
    mlflow.log_params(rf_params)
    mlflow.log_params({
        'model_type': 'Random Forest',
        'test_size': TEST_SIZE,
        'cv_folds': CV_FOLDS,
        'random_state': RANDOM_STATE,
        'n_features': X_train_scaled.shape[1],
        'train_samples': X_train_scaled.shape[0]
    })
    mlflow.log_metrics({
        'accuracy': rf_metrics['accuracy'],
        'precision': rf_metrics['precision'],
        'recall': rf_metrics['recall'],
        'roc_auc': rf_metrics['roc_auc'],
        'cv_mean': rf_metrics['cv_mean'],
        'cv_std': rf_metrics['cv_std']
    })
    mlflow.log_artifact(rf_cm_path)
    mlflow.log_artifact(rf_roc_path)
    mlflow.log_artifact(rf_fi_path)
    mlflow.sklearn.log_model(
        rf_model,
        "random_forest_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    print("\nRandom Forest logged to MLflow!")

# Save model locally
with open(f"{MODEL_DIR}random_forest.pkl", 'wb') as f:
    pickle.dump(rf_model, f)
print(f"Random Forest saved to {MODEL_DIR}random_forest.pkl")

# ============================================================
# SECTION 6: MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [lr_metrics['accuracy'], rf_metrics['accuracy']],
    'Precision': [lr_metrics['precision'], rf_metrics['precision']],
    'Recall': [lr_metrics['recall'], rf_metrics['recall']],
    'ROC-AUC': [lr_metrics['roc_auc'], rf_metrics['roc_auc']],
    'CV Mean': [lr_metrics['cv_mean'], rf_metrics['cv_mean']]
})

print(comparison_df.to_string(index=False))

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Model Comparison — Heart Disease Classification',
             fontsize=14, fontweight='bold')

metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'ROC-AUC']
x = np.arange(len(metrics_to_plot))
width = 0.35

lr_values = [lr_metrics['accuracy'], lr_metrics['precision'],
             lr_metrics['recall'], lr_metrics['roc_auc']]
rf_values = [rf_metrics['accuracy'], rf_metrics['precision'],
             rf_metrics['recall'], rf_metrics['roc_auc']]

bars1 = axes[0].bar(x - width/2, lr_values, width,
                     label='Logistic Regression', color='#3498db', alpha=0.8)
bars2 = axes[0].bar(x + width/2, rf_values, width,
                     label='Random Forest', color='#e74c3c', alpha=0.8)
axes[0].set_xlabel('Metric')
axes[0].set_ylabel('Score')
axes[0].set_title('Metrics Comparison')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics_to_plot)
axes[0].legend()
axes[0].set_ylim([0, 1.1])

# ROC curves comparison
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
axes[1].plot(fpr_lr, tpr_lr, color='#3498db', lw=2,
             label=f'LR (AUC={lr_metrics["roc_auc"]:.3f})')
axes[1].plot(fpr_rf, tpr_rf, color='#e74c3c', lw=2,
             label=f'RF (AUC={rf_metrics["roc_auc"]:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curves Comparison')
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{SCREENSHOTS_DIR}model_comparison.png", dpi=150)
plt.show()
print("Model comparison chart saved.")

# ============================================================
# SECTION 7: SELECT & SAVE BEST MODEL
# ============================================================

print("\n--- Selecting Best Model ---")

if rf_metrics['roc_auc'] >= lr_metrics['roc_auc']:
    best_model = rf_model
    best_model_name = "Random Forest"
    best_metrics = rf_metrics
else:
    best_model = lr_model
    best_model_name = "Logistic Regression"
    best_metrics = lr_metrics

print(f"Best Model: {best_model_name}")
print(f"Best ROC-AUC: {best_metrics['roc_auc']:.4f}")

# Save best model
with open(f"{MODEL_DIR}best_model.pkl", 'wb') as f:
    pickle.dump(best_model, f)
print(f"Best model saved to {MODEL_DIR}best_model.pkl")

# Log best model to MLflow
with mlflow.start_run(run_name="Best_Model"):
    mlflow.log_param("best_model", best_model_name)
    mlflow.log_metrics({
        'best_accuracy': best_metrics['accuracy'],
        'best_precision': best_metrics['precision'],
        'best_recall': best_metrics['recall'],
        'best_roc_auc': best_metrics['roc_auc']
    })
    mlflow.sklearn.log_model(
        best_model,
        "best_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )
    mlflow.log_artifact(f"{SCREENSHOTS_DIR}model_comparison.png")
    print("Best model logged to MLflow!")

# ============================================================
# SECTION 8: SAVE PREPROCESSING PIPELINE
# ============================================================

print("\n--- Saving Preprocessing Pipeline ---")

preprocessing_pipeline = {
    'scaler': scaler,
    'feature_names': feature_names,
    'categorical_cols': categorical_cols,
    'numeric_cols': numeric_cols,
    'numeric_present': numeric_present
}

with open(f"{MODEL_DIR}preprocessing_pipeline.pkl", 'wb') as f:
    pickle.dump(preprocessing_pipeline, f)
print(f"Preprocessing pipeline saved to {MODEL_DIR}preprocessing_pipeline.pkl")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print(f"Best Model: {best_model_name}")
print(f"Best ROC-AUC: {best_metrics['roc_auc']:.4f}")
print(f"Models saved in: {MODEL_DIR}")
print(f"Charts saved in: {SCREENSHOTS_DIR}")
print(f"MLflow UI: run 'mlflow ui' to view experiments")
print("=" * 60)
