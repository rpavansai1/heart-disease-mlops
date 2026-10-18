# ============================================================
# AIMLCZG523 - Machine Learning Operations
# Assignment 1 - Unit Tests for Model
# ============================================================

import pytest
import pandas as pd
import numpy as np
import pickle
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def sample_input():
    """Sample patient input for prediction."""
    return {
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


@pytest.fixture
def loaded_model():
    """Load the best model for testing."""
    model_path = 'models/best_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


@pytest.fixture
def loaded_pipeline():
    """Load the preprocessing pipeline for testing."""
    pipeline_path = 'models/preprocessing_pipeline.pkl'
    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline


@pytest.fixture
def preprocessed_input(sample_input, loaded_pipeline):
    """Preprocess sample input for model testing."""
    scaler = loaded_pipeline['scaler']
    feature_names = loaded_pipeline['feature_names']
    categorical_cols = loaded_pipeline['categorical_cols']
    numeric_present = loaded_pipeline['numeric_present']

    df = pd.DataFrame([sample_input])
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df_aligned = df_encoded.reindex(columns=feature_names, fill_value=0)
    df_aligned[numeric_present] = scaler.transform(df_aligned[numeric_present])

    return df_aligned.values


# ============================================================
# TEST 1: Model Files
# ============================================================

class TestModelFiles:

    def test_best_model_exists(self):
        """Test that best model file exists."""
        assert os.path.exists('models/best_model.pkl'), \
            "best_model.pkl not found"

    def test_logistic_regression_exists(self):
        """Test that logistic regression model file exists."""
        assert os.path.exists('models/logistic_regression.pkl'), \
            "logistic_regression.pkl not found"

    def test_random_forest_exists(self):
        """Test that random forest model file exists."""
        assert os.path.exists('models/random_forest.pkl'), \
            "random_forest.pkl not found"

    def test_preprocessing_pipeline_exists(self):
        """Test that preprocessing pipeline exists."""
        assert os.path.exists('models/preprocessing_pipeline.pkl'), \
            "preprocessing_pipeline.pkl not found"

    def test_model_loadable(self):
        """Test that best model can be loaded successfully."""
        with open('models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        assert model is not None, "Model loaded as None"

    def test_pipeline_loadable(self):
        """Test that preprocessing pipeline can be loaded."""
        with open('models/preprocessing_pipeline.pkl', 'rb') as f:
            pipeline = pickle.load(f)
        assert pipeline is not None, "Pipeline loaded as None"
        assert 'scaler' in pipeline, "Scaler missing from pipeline"
        assert 'feature_names' in pipeline, \
            "Feature names missing from pipeline"


# ============================================================
# TEST 2: Model Prediction
# ============================================================

class TestModelPrediction:

    def test_prediction_returns_binary(self, loaded_model,
                                       preprocessed_input):
        """Test that prediction returns binary value (0 or 1)."""
        prediction = loaded_model.predict(preprocessed_input)[0]
        assert prediction in [0, 1], \
            f"Prediction not binary: {prediction}"

    def test_prediction_probability_sum(self, loaded_model,
                                        preprocessed_input):
        """Test that prediction probabilities sum to 1."""
        probabilities = loaded_model.predict_proba(preprocessed_input)[0]
        assert abs(sum(probabilities) - 1.0) < 1e-6, \
            f"Probabilities don't sum to 1: {sum(probabilities)}"

    def test_prediction_probability_range(self, loaded_model,
                                          preprocessed_input):
        """Test that probabilities are between 0 and 1."""
        probabilities = loaded_model.predict_proba(preprocessed_input)[0]
        for prob in probabilities:
            assert 0 <= prob <= 1, \
                f"Probability out of range: {prob}"

    def test_model_has_predict_method(self, loaded_model):
        """Test that model has predict method."""
        assert hasattr(loaded_model, 'predict'), \
            "Model missing predict method"

    def test_model_has_predict_proba_method(self, loaded_model):
        """Test that model has predict_proba method."""
        assert hasattr(loaded_model, 'predict_proba'), \
            "Model missing predict_proba method"

    def test_batch_prediction(self, loaded_model, loaded_pipeline):
        """Test batch prediction with multiple inputs."""
        df = pd.read_csv('data/heart_cleaned.csv')
        X = df.drop('target', axis=1)

        scaler = loaded_pipeline['scaler']
        feature_names = loaded_pipeline['feature_names']
        categorical_cols = loaded_pipeline['categorical_cols']
        numeric_present = loaded_pipeline['numeric_present']

        X_encoded = pd.get_dummies(X, columns=categorical_cols,
                                   drop_first=True)
        X_aligned = X_encoded.reindex(columns=feature_names, fill_value=0)
        X_aligned[numeric_present] = scaler.transform(
            X_aligned[numeric_present])

        predictions = loaded_model.predict(X_aligned.values)
        assert len(predictions) == len(df), \
            "Batch prediction length mismatch"


# ============================================================
# TEST 3: Model Performance
# ============================================================

class TestModelPerformance:

    def test_model_accuracy_above_threshold(self, loaded_model,
                                             loaded_pipeline):
        """Test that model accuracy is above minimum threshold."""
        from sklearn.metrics import accuracy_score

        df = pd.read_csv('data/heart_cleaned.csv')
        X = df.drop('target', axis=1)
        y = df['target']

        scaler = loaded_pipeline['scaler']
        feature_names = loaded_pipeline['feature_names']
        categorical_cols = loaded_pipeline['categorical_cols']
        numeric_present = loaded_pipeline['numeric_present']

        X_encoded = pd.get_dummies(X, columns=categorical_cols,
                                   drop_first=True)
        X_aligned = X_encoded.reindex(columns=feature_names, fill_value=0)
        X_aligned[numeric_present] = scaler.transform(
            X_aligned[numeric_present])

        predictions = loaded_model.predict(X_aligned.values)
        accuracy = accuracy_score(y, predictions)

        assert accuracy >= 0.70, \
            f"Model accuracy below threshold: {accuracy:.4f}"

    def test_model_roc_auc_above_threshold(self, loaded_model,
                                            loaded_pipeline):
        """Test that model ROC-AUC is above minimum threshold."""
        from sklearn.metrics import roc_auc_score

        df = pd.read_csv('data/heart_cleaned.csv')
        X = df.drop('target', axis=1)
        y = df['target']

        scaler = loaded_pipeline['scaler']
        feature_names = loaded_pipeline['feature_names']
        categorical_cols = loaded_pipeline['categorical_cols']
        numeric_present = loaded_pipeline['numeric_present']

        X_encoded = pd.get_dummies(X, columns=categorical_cols,
                                   drop_first=True)
        X_aligned = X_encoded.reindex(columns=feature_names, fill_value=0)
        X_aligned[numeric_present] = scaler.transform(
            X_aligned[numeric_present])

        probabilities = loaded_model.predict_proba(X_aligned.values)[:, 1]
        roc_auc = roc_auc_score(y, probabilities)

        assert roc_auc >= 0.75, \
            f"Model ROC-AUC below threshold: {roc_auc:.4f}"

    def test_prediction_output_shape(self, loaded_model,
                                     preprocessed_input):
        """Test that prediction output has correct shape."""
        predictions = loaded_model.predict(preprocessed_input)
        assert predictions.shape == (1,), \
            f"Unexpected prediction shape: {predictions.shape}"

    def test_probability_output_shape(self, loaded_model,
                                      preprocessed_input):
        """Test that probability output has correct shape."""
        probabilities = loaded_model.predict_proba(preprocessed_input)
        assert probabilities.shape == (1, 2), \
            f"Unexpected probability shape: {probabilities.shape}"


# ============================================================
# TEST 4: MLflow Tracking
# ============================================================

class TestMLflowTracking:

    def test_mlflow_db_exists(self):
        """Test that MLflow database exists."""
        assert os.path.exists('mlflow.db'), \
            "mlflow.db not found — MLflow tracking not set up"

    def test_mlflow_experiments_exist(self):
        """Test that MLflow experiments were created."""
        import mlflow
        db_path = os.path.abspath('mlflow.db').replace('\\', '/')
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        assert len(experiments) > 0, \
            "No MLflow experiments found"

    def test_mlflow_runs_exist(self):
        """Test that MLflow runs were logged."""
        import mlflow
        db_path = os.path.abspath('mlflow.db').replace('\\', '/')
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        total_runs = 0
        for exp in experiments:
            runs = client.search_runs(exp.experiment_id)
            total_runs += len(runs)
        assert total_runs > 0, "No MLflow runs found"
