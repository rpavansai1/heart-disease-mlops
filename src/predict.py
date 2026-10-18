# ============================================================
# AIMLCZG523 - Machine Learning Operations
# Assignment 1 - Inference Script
# ============================================================

import pickle
import numpy as np
import os
from preprocess import preprocess_input


def load_model(model_path: str = "models/best_model.pkl"):
    """Load the saved model."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def predict(data: dict, model_dir: str = "models/") -> dict:
    """
    Make a prediction for a single input.

    Args:
        data: Dictionary with patient features
        model_dir: Path to models directory

    Returns:
        Dictionary with prediction and confidence
    """
    # Load model
    model = load_model(os.path.join(model_dir, "best_model.pkl"))

    # Preprocess input
    X = preprocess_input(data, model_dir)

    # Predict
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    confidence = float(max(probability))

    result = {
        'prediction': int(prediction),
        'prediction_label': 'Heart Disease' if prediction == 1 else 'No Heart Disease',
        'confidence': round(confidence, 4),
        'probability_no_disease': round(float(probability[0]), 4),
        'probability_disease': round(float(probability[1]), 4)
    }

    return result


if __name__ == "__main__":
    # Sample patient data for testing
    sample_patient = {
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

    print("Testing inference pipeline...")
    print(f"Input: {sample_patient}")

    result = predict(sample_patient)
    print(f"\nPrediction: {result['prediction_label']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Probability (No Disease): {result['probability_no_disease']}")
    print(f"Probability (Disease): {result['probability_disease']}")
