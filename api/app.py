# ============================================================
# AIMLCZG523 - Machine Learning Operations
# Assignment 1 - FastAPI Prediction API
# Endpoints:
#   GET  /          -> Health check
#   GET  /health    -> Detailed health check
#   POST /predict   -> Heart disease prediction
#   GET  /metrics   -> Prometheus metrics
# ============================================================

import pickle
import os
import logging
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

# ============================================================
# LOGGING SETUP
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api_requests.log')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# FASTAPI APP SETUP
# ============================================================

app = FastAPI(
    title="Heart Disease Prediction API",
    description="""
    ## AIMLCZG523 - MLOps Assignment 1

    A cloud-ready API for predicting heart disease risk
    based on patient health data.

    ### Endpoints
    - **GET /**        - Root health check
    - **GET /health**  - Detailed health status
    - **POST /predict** - Predict heart disease risk
    - **GET /metrics** - Prometheus metrics
    """,
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

# ============================================================
# REQUEST LOGGING MIDDLEWARE
# ============================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all API requests with timing."""
    start_time = time.time()
    logger.info(f"Request: {request.method} {request.url.path}")

    response = await call_next(request)

    duration = time.time() - start_time
    logger.info(
        f"Response: {request.method} {request.url.path} "
        f"Status: {response.status_code} "
        f"Duration: {duration:.4f}s"
    )
    return response


# ============================================================
# LOAD MODEL & PIPELINE
# ============================================================

MODEL_DIR = "models/"

def load_model_and_pipeline():
    """Load model and preprocessing pipeline."""
    try:
        # Load best model
        with open(os.path.join(MODEL_DIR, "best_model.pkl"), 'rb') as f:
            model = pickle.load(f)

        # Load preprocessing pipeline
        with open(os.path.join(MODEL_DIR, "preprocessing_pipeline.pkl"), 'rb') as f:
            pipeline = pickle.load(f)

        logger.info(f"Model loaded: {type(model).__name__}")
        logger.info("Preprocessing pipeline loaded successfully")
        return model, pipeline

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, None


# Load at startup
model, pipeline = load_model_and_pipeline()

# ============================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================

class PatientData(BaseModel):
    """Input schema for patient health data."""
    age: float = Field(..., description="Age of patient", example=52)
    sex: float = Field(..., description="Sex (1=male, 0=female)", example=1)
    cp: float = Field(..., description="Chest pain type (0-3)", example=0)
    trestbps: float = Field(..., description="Resting blood pressure", example=125)
    chol: float = Field(..., description="Serum cholesterol in mg/dl", example=212)
    fbs: float = Field(..., description="Fasting blood sugar > 120 mg/dl (1=true)", example=0)
    restecg: float = Field(..., description="Resting ECG results (0-2)", example=1)
    thalach: float = Field(..., description="Maximum heart rate achieved", example=168)
    exang: float = Field(..., description="Exercise induced angina (1=yes)", example=0)
    oldpeak: float = Field(..., description="ST depression induced by exercise", example=1.0)
    slope: float = Field(..., description="Slope of peak exercise ST segment", example=2)
    ca: float = Field(..., description="Number of major vessels (0-3)", example=2)
    thal: float = Field(..., description="Thal (1=normal, 2=fixed defect, 3=reversable defect)", example=3)

    class Config:
        json_schema_extra = {
            "example": {
                "age": 52,
                "sex": 1,
                "cp": 0,
                "trestbps": 125,
                "chol": 212,
                "fbs": 0,
                "restecg": 1,
                "thalach": 168,
                "exang": 0,
                "oldpeak": 1.0,
                "slope": 2,
                "ca": 2,
                "thal": 3
            }
        }


class PredictionResponse(BaseModel):
    """Output schema for prediction response."""
    prediction: int
    prediction_label: str
    confidence: float
    probability_no_disease: float
    probability_disease: float
    model_type: str
    timestamp: str


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    """Root endpoint — health check."""
    logger.info("Root endpoint called")
    return {
        "status": "healthy",
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "endpoints": [
            "GET  /        - Root health check",
            "GET  /health  - Detailed health check",
            "POST /predict - Predict heart disease",
            "GET  /metrics - Prometheus metrics",
            "GET  /docs    - API documentation"
        ],
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    logger.info("Health check endpoint called")

    model_status = "loaded" if model is not None else "not loaded"
    pipeline_status = "loaded" if pipeline is not None else "not loaded"

    if model is None or pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model or pipeline not loaded"
        )

    return {
        "status": "healthy",
        "model_status": model_status,
        "model_type": type(model).__name__,
        "pipeline_status": pipeline_status,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientData):
    """
    Predict heart disease risk for a patient.

    Accepts patient health data and returns:
    - prediction (0 or 1)
    - prediction label
    - confidence score
    - probability scores
    """
    logger.info(f"Prediction request received: age={patient.age}, sex={patient.sex}")

    if model is None or pipeline is None:
        logger.error("Model not loaded — cannot make prediction")
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )

    try:
        # Convert input to dict
        input_data = patient.dict()
        logger.info(f"Input data: {input_data}")

        # Extract pipeline components
        scaler = pipeline['scaler']
        feature_names = pipeline['feature_names']
        categorical_cols = pipeline['categorical_cols']
        numeric_present = pipeline['numeric_present']

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # One-hot encode categorical features
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # Align with training features
        df_aligned = df_encoded.reindex(columns=feature_names, fill_value=0)

        # Scale numeric features
        df_aligned[numeric_present] = scaler.transform(df_aligned[numeric_present])

        # Make prediction
        prediction = int(model.predict(df_aligned.values)[0])
        probabilities = model.predict_proba(df_aligned.values)[0]
        confidence = float(max(probabilities))

        result = {
            "prediction": prediction,
            "prediction_label": "Heart Disease" if prediction == 1 else "No Heart Disease",
            "confidence": round(confidence, 4),
            "probability_no_disease": round(float(probabilities[0]), 4),
            "probability_disease": round(float(probabilities[1]), 4),
            "model_type": type(model).__name__,
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"Prediction result: {result['prediction_label']} "
                   f"(confidence: {result['confidence']})")

        return result

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
