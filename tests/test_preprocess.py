# ============================================================
# AIMLCZG523 - Machine Learning Operations
# Assignment 1 - Unit Tests for Preprocessing
# ============================================================

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing."""
    return pd.DataFrame({
        'age': [52, 63, 37, 41, 56],
        'sex': [1, 1, 1, 0, 1],
        'cp': [0, 4, 3, 2, 2],
        'trestbps': [125, 130, 130, 130, 120],
        'chol': [212, 254, 250, 204, 236],
        'fbs': [0, 0, 0, 0, 0],
        'restecg': [1, 2, 0, 2, 0],
        'thalach': [168, 147, 187, 172, 178],
        'exang': [0, 0, 0, 0, 0],
        'oldpeak': [1.0, 1.4, 3.5, 1.4, 0.8],
        'slope': [2, 2, 3, 1, 2],
        'ca': [2, 1, 0, 0, 0],
        'thal': [3, 2, 3, 2, 2],
        'target': [0, 0, 0, 1, 1]
    })


@pytest.fixture
def sample_input():
    """Create a sample input for single prediction testing."""
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


# ============================================================
# TEST 1: Dataset Loading
# ============================================================

class TestDataLoading:

    def test_dataset_exists(self):
        """Test that dataset file exists."""
        assert os.path.exists('data/heart.csv'), \
            "heart.csv not found in data/ folder"

    def test_cleaned_dataset_exists(self):
        """Test that cleaned dataset file exists."""
        assert os.path.exists('data/heart_cleaned.csv'), \
            "heart_cleaned.csv not found in data/ folder"

    def test_dataset_not_empty(self):
        """Test that dataset is not empty."""
        df = pd.read_csv('data/heart.csv')
        assert len(df) > 0, "Dataset is empty"

    def test_dataset_has_correct_columns(self):
        """Test that dataset has required columns."""
        df = pd.read_csv('data/heart.csv')
        required_columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol',
            'fbs', 'restecg', 'thalach', 'exang',
            'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_dataset_sufficient_records(self):
        """Test that dataset has sufficient records."""
        df = pd.read_csv('data/heart.csv')
        assert len(df) >= 100, \
            f"Dataset has insufficient records: {len(df)}"

    def test_target_column_binary(self):
        """Test that target column is binary (0 or 1)."""
        df = pd.read_csv('data/heart.csv')
        unique_values = set(df['target'].unique())
        assert unique_values.issubset({0, 1}), \
            f"Target column has non-binary values: {unique_values}"


# ============================================================
# TEST 2: Data Quality
# ============================================================

class TestDataQuality:

    def test_no_missing_values_after_cleaning(self, sample_dataframe):
        """Test that there are no missing values after cleaning."""
        df = sample_dataframe.copy()
        # Impute missing values
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col].fillna(df[col].median(), inplace=True)
        assert df.isnull().sum().sum() == 0, \
            "Missing values found after cleaning"

    def test_age_range_valid(self, sample_dataframe):
        """Test that age values are within valid range."""
        df = sample_dataframe
        assert df['age'].min() >= 0, "Age has negative values"
        assert df['age'].max() <= 120, "Age exceeds maximum valid value"

    def test_target_distribution(self, sample_dataframe):
        """Test that target has both classes."""
        df = sample_dataframe
        assert 0 in df['target'].values, "No negative class in target"
        assert 1 in df['target'].values, "No positive class in target"

    def test_numeric_columns_dtype(self, sample_dataframe):
        """Test that numeric columns have correct dtype."""
        df = sample_dataframe
        numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(df[col]), \
                f"Column {col} is not numeric"

    def test_no_duplicate_rows(self):
        """Test that cleaned dataset has no duplicate rows."""
        df = pd.read_csv('data/heart_cleaned.csv')
        duplicates = df.duplicated().sum()
        duplicate_pct = duplicates / len(df) * 100
        assert duplicate_pct < 5, f"Dataset has too many duplicate rows: {duplicates} ({duplicate_pct:.1f}%)"


# ============================================================
# TEST 3: Feature Engineering
# ============================================================

class TestFeatureEngineering:

    def test_encoding_produces_correct_shape(self, sample_dataframe):
        """Test that encoding increases number of columns."""
        df = sample_dataframe.drop('target', axis=1)
        original_cols = df.shape[1]
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg',
                           'exang', 'slope', 'ca', 'thal']
        df_encoded = pd.get_dummies(df, columns=categorical_cols,
                                    drop_first=True)
        assert df_encoded.shape[1] >= original_cols, \
            "Encoding did not increase number of features"

    def test_normalization_range(self, sample_dataframe):
        """Test that normalized values are between 0 and 1."""
        from sklearn.preprocessing import MinMaxScaler
        df = sample_dataframe.copy()
        numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        for col in numeric_cols:
            assert df[col].min() >= 0, f"Column {col} has values below 0 after normalization"
            assert df[col].max() <= 1+1e-10, f"Column {col} has values above 1 after normalization"

    def test_standard_scaler_mean(self, sample_dataframe):
        """Test that StandardScaler produces zero mean."""
        from sklearn.preprocessing import StandardScaler
        df = sample_dataframe.copy()
        numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[numeric_cols])
        means = scaled.mean(axis=0)
        for mean in means:
            assert abs(mean) < 1e-10, \
                f"StandardScaler mean not zero: {mean}"

    def test_feature_names_saved(self):
        """Test that feature names file exists."""
        assert os.path.exists('models/feature_names.pkl'), \
            "feature_names.pkl not found in models/ folder"

    def test_scaler_saved(self):
        """Test that scaler file exists."""
        assert os.path.exists('models/scaler.pkl'), \
            "scaler.pkl not found in models/ folder"

    def test_preprocessing_pipeline_saved(self):
        """Test that preprocessing pipeline file exists."""
        assert os.path.exists('models/preprocessing_pipeline.pkl'), \
            "preprocessing_pipeline.pkl not found in models/ folder"


# ============================================================
# TEST 4: Input Validation
# ============================================================

class TestInputValidation:

    def test_sample_input_has_required_keys(self, sample_input):
        """Test that sample input has all required keys."""
        required_keys = [
            'age', 'sex', 'cp', 'trestbps', 'chol',
            'fbs', 'restecg', 'thalach', 'exang',
            'oldpeak', 'slope', 'ca', 'thal'
        ]
        for key in required_keys:
            assert key in sample_input, \
                f"Missing required key: {key}"

    def test_input_values_are_numeric(self, sample_input):
        """Test that all input values are numeric."""
        for key, value in sample_input.items():
            assert isinstance(value, (int, float)), \
                f"Input key {key} has non-numeric value: {value}"

    def test_age_in_valid_range(self, sample_input):
        """Test that age is in valid range."""
        assert 0 <= sample_input['age'] <= 120, \
            f"Age out of valid range: {sample_input['age']}"

    def test_sex_is_binary(self, sample_input):
        """Test that sex is binary (0 or 1)."""
        assert sample_input['sex'] in [0, 1], \
            f"Sex must be 0 or 1, got: {sample_input['sex']}"

    def test_fbs_is_binary(self, sample_input):
        """Test that fbs is binary (0 or 1)."""
        assert sample_input['fbs'] in [0, 1], \
            f"FBS must be 0 or 1, got: {sample_input['fbs']}"
