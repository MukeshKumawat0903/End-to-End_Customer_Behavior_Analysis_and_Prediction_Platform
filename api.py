"""
FastAPI application for model inference.
"""
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import get_config
from src.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthCheckResponse,
    ModelMetadata
)
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Initialize FastAPI app
config = get_config()
app = FastAPI(
    title="Customer Behavior Prediction API",
    description="API for predicting customer conversion probability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and metadata
model = None
model_metadata = None


def load_model(model_path: Optional[str] = None) -> tuple:
    """
    Load the trained model and metadata.
    
    Args:
        model_path: Path to model file. If None, uses latest from config.
        
    Returns:
        Tuple of (model, metadata)
        
    Raises:
        FileNotFoundError: If model file not found.
    """
    try:
        if model_path is None:
            models_dir = Path(config.storage.models_dir)
            # Find the latest model
            model_files = sorted(models_dir.glob("model_*/model.pkl"))
            if not model_files:
                raise FileNotFoundError(f"No model found in {models_dir}")
            model_path = model_files[-1]
        
        logger.info(f"Loading model from: {model_path}")
        artifacts = joblib.load(model_path)
        
        loaded_model = artifacts['model']
        metadata = artifacts.get('metadata', {})
        
        logger.info("Model loaded successfully")
        return loaded_model, metadata
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model, model_metadata
    try:
        model, model_metadata = load_model()
        logger.info("API started successfully")
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        model = None
        model_metadata = None


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Customer Behavior Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    return HealthCheckResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=model_metadata.get('timestamp') if model_metadata else None,
        timestamp=datetime.now()
    )


@app.get("/model/metadata", response_model=dict)
async def get_model_metadata():
    """Get model metadata."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "metadata": model_metadata,
        "status": "loaded"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a single prediction.
    
    Args:
        request: Prediction request with feature values.
        
    Returns:
        Prediction response with probability and classification.
        
    Raises:
        HTTPException: If model not loaded or prediction fails.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Convert request to DataFrame
        data = request.dict()
        
        # Convert is_weekend to int
        data['is_weekend'] = int(data['is_weekend'])
        
        # Create DataFrame with correct feature order
        features = config.features.numerical + config.features.categorical
        df = pd.DataFrame([data])
        
        # Ensure correct column order
        df = df[features]
        
        # Make prediction
        probability = float(model.predict_proba(df)[0, 1])
        
        # Get threshold
        threshold = config.threshold.default
        if config.threshold.use_optimal and model_metadata:
            threshold = model_metadata.get('optimal_threshold', threshold)
        
        prediction = int(probability >= threshold)
        
        logger.info(f"Prediction made: probability={probability:.4f}, prediction={prediction}")
        
        return PredictionResponse(
            probability=probability,
            prediction=prediction,
            confidence="",  # Will be set by validator
            threshold=threshold
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions.
    
    Args:
        request: Batch prediction request with list of feature values.
        
    Returns:
        Batch prediction response with list of predictions.
        
    Raises:
        HTTPException: If model not loaded or prediction fails.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Convert requests to DataFrame
        data_list = [req.dict() for req in request.data]
        
        # Convert is_weekend to int
        for data in data_list:
            data['is_weekend'] = int(data['is_weekend'])
        
        df = pd.DataFrame(data_list)
        
        # Ensure correct column order
        features = config.features.numerical + config.features.categorical
        df = df[features]
        
        # Make predictions
        probabilities = model.predict_proba(df)[:, 1]
        
        # Get threshold
        threshold = config.threshold.default
        if config.threshold.use_optimal and model_metadata:
            threshold = model_metadata.get('optimal_threshold', threshold)
        
        predictions = (probabilities >= threshold).astype(int)
        
        # Create response
        responses = [
            PredictionResponse(
                probability=float(prob),
                prediction=int(pred),
                confidence="",
                threshold=threshold
            )
            for prob, pred in zip(probabilities, predictions)
        ]
        
        logger.info(f"Batch prediction made: {len(responses)} predictions")
        
        return BatchPredictionResponse(
            predictions=responses,
            count=len(responses)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.post("/model/reload")
async def reload_model():
    """Reload the model."""
    global model, model_metadata
    
    try:
        model, model_metadata = load_model()
        logger.info("Model reloaded successfully")
        return {"status": "success", "message": "Model reloaded"}
    except Exception as e:
        logger.error(f"Failed to reload model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload model: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
        log_level="info"
    )
