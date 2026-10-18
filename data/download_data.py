# ============================================================
# AIMLCZG523 - Machine Learning Operations
# Assignment 1 - Dataset Download Script
# Dataset: Heart Disease UCI Dataset
# ============================================================

import pandas as pd
import os

def download_heart_disease_data(output_path: str = "data/heart.csv"):
    """
    Downloads the Heart Disease UCI dataset from a public URL
    and saves it to the specified path.
    """
    print("Downloading Heart Disease UCI Dataset...")

    # Public URL for Heart Disease dataset
    url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart_disease.csv"

    try:
        # Download dataset
        df = pd.read_csv(url)

        # Rename columns to standard UCI names if needed
        standard_columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol',
            'fbs', 'restecg', 'thalach', 'exang',
            'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]

        # Assign standard column names
        if len(df.columns) == 14:
            df.columns = standard_columns

        # Create output directory if not exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV
        df.to_csv(output_path, index=False)

        print(f"Dataset downloaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Saved to: {output_path}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        print(f"\nColumns: {df.columns.tolist()}")

        return df

    except Exception as e:
        print(f"Download failed: {e}")
        print("Trying alternative source...")

        try:
            # Alternative URL
            alt_url = "https://raw.githubusercontent.com/education454/heart_disease_dataset/main/heart.csv"
            df = pd.read_csv(alt_url)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)

            print(f"Dataset downloaded from alternative source!")
            print(f"Shape: {df.shape}")
            print(f"Saved to: {output_path}")

            return df

        except Exception as e2:
            print(f"Alternative download also failed: {e2}")
            print("\nManual Download Instructions:")
            print("1. Go to: https://www.kaggle.com/datasets/ronitf/heart-disease-uci")
            print("2. Download heart.csv")
            print("3. Place it in the data/ folder")
            return None


if __name__ == "__main__":
    df = download_heart_disease_data()
