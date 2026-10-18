# ============================================================
# AIMLCZG523 - Machine Learning Operations
# Assignment 1 - Preprocessing Pipeline
# Ensures full reproducibility for inference
# ============================================================

import pandas as pd
import numpy as np
import pickle
import os


def load_preprocessing_pipeline(model_dir: str = "models/"):
    """Load the saved preprocessing pipeline."""
    pipeline_path = os.path.join(model_dir, "preprocessing_pipeline.pkl")
    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline


def preprocess_input(data: dict, model_dir: str = "models/") -> np.ndarray:
    """
    Preprocess a single input record for prediction.

    Args:
        data: Dictionary with feature values
        model_dir: Path to models directory

    Returns:
        Preprocessed numpy array ready for prediction
    """
    # Load pipeline
    pipeline = load_preprocessing_pipeline(model_dir)
    scaler = pipeline['scaler']
    feature_names = pipeline['feature_names']
    categorical_cols = pipeline['categorical_cols']
    numeric_cols = pipeline['numeric_cols']
    numeric_present = pipeline['numeric_present']

    # Convert input to DataFrame
    df = pd.DataFrame([data])

    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Align columns with training features
    df_aligned = df_encoded.reindex(columns=feature_names, fill_value=0)

    # Scale numeric features
    df_aligned[numeric_present] = scaler.transform(df_aligned[numeric_present])

    return df_aligned.values


def preprocess_dataframe(df: pd.DataFrame,
                         model_dir: str = "models/") -> np.ndarray:
    """
    Preprocess a full DataFrame for batch prediction.

    Args:
        df: Input DataFrame
        model_dir: Path to models directory

    Returns:
        Preprocessed numpy array ready for prediction
    """
    # Load pipeline
    pipeline = load_preprocessing_pipeline(model_dir)
    scaler = pipeline['scaler']
    feature_names = pipeline['feature_names']
    categorical_cols = pipeline['categorical_cols']
    numeric_present = pipeline['numeric_present']

    # Drop target if present
    if 'target' in df.columns:
        df = df.drop('target', axis=1)

    # One-hot encode
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Align columns
    df_aligned = df_encoded.reindex(columns=feature_names, fill_value=0)

    # Scale
    df_aligned[numeric_present] = scaler.transform(df_aligned[numeric_present])

    return df_aligned.values


if __name__ == "__main__":
    # Test preprocessing with sample input
    sample = {
        'age': 52,
        'sex': 1,
        'cp': 0,
        'trestbps': 125,
        'chol': 212,
        'fbs': 0,
        'restecg': 1,
        'thalach': 168,
        'exang': 0,
        'oldpeak': 1.0,
        'slope': 2,
        'ca': 2,
        'thal': 3
    }

    print("Testing preprocessing pipeline...")
    result = preprocess_input(sample)
    print(f"Input shape: {result.shape}")
    print(f"Preprocessing successful!")
