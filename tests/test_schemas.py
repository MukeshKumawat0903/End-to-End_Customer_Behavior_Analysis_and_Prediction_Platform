"""
Unit tests for data schemas and validation.
"""
import pytest
from datetime import datetime
from pydantic import ValidationError

from src.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    HealthCheckResponse
)


def test_prediction_request_valid():
    """Test valid prediction request."""
    request = PredictionRequest(
        price=100.0,
        hour=14,
        is_weekend=False,
        total_clicks=10,
        avg_session_time=120.5,
        purchase_freq=0.3,
        products_viewed=5,
        price_range=50.0,
        session_duration=300.0,
        recency=10,
        frequency=3,
        monetary=500.0,
        main_category="electronics",
        rfm_segment=2
    )
    
    assert request.price == 100.0
    assert request.hour == 14
    assert request.is_weekend == False


def test_prediction_request_invalid_hour():
    """Test prediction request with invalid hour."""
    with pytest.raises(ValidationError):
        PredictionRequest(
            price=100.0,
            hour=25,  # Invalid: > 23
            is_weekend=False,
            total_clicks=10,
            avg_session_time=120.5,
            purchase_freq=0.3,
            products_viewed=5,
            price_range=50.0,
            session_duration=300.0,
            recency=10,
            frequency=3,
            monetary=500.0,
            main_category="electronics",
            rfm_segment=2
        )


def test_prediction_request_invalid_price():
    """Test prediction request with invalid price."""
    with pytest.raises(ValidationError):
        PredictionRequest(
            price=-10.0,  # Invalid: negative
            hour=14,
            is_weekend=False,
            total_clicks=10,
            avg_session_time=120.5,
            purchase_freq=0.3,
            products_viewed=5,
            price_range=50.0,
            session_duration=300.0,
            recency=10,
            frequency=3,
            monetary=500.0,
            main_category="electronics",
            rfm_segment=2
        )


def test_prediction_request_invalid_purchase_freq():
    """Test prediction request with invalid purchase frequency."""
    with pytest.raises(ValidationError):
        PredictionRequest(
            price=100.0,
            hour=14,
            is_weekend=False,
            total_clicks=10,
            avg_session_time=120.5,
            purchase_freq=1.5,  # Invalid: > 1
            products_viewed=5,
            price_range=50.0,
            session_duration=300.0,
            recency=10,
            frequency=3,
            monetary=500.0,
            main_category="electronics",
            rfm_segment=2
        )


def test_prediction_response_confidence():
    """Test prediction response confidence calculation."""
    # High confidence (probability far from threshold)
    response = PredictionResponse(
        probability=0.9,
        prediction=1,
        confidence="",
        threshold=0.5
    )
    assert response.confidence == "high"
    
    # Medium confidence
    response = PredictionResponse(
        probability=0.65,
        prediction=1,
        confidence="",
        threshold=0.5
    )
    assert response.confidence == "medium"
    
    # Low confidence (probability close to threshold)
    response = PredictionResponse(
        probability=0.52,
        prediction=1,
        confidence="",
        threshold=0.5
    )
    assert response.confidence == "low"


def test_batch_prediction_request_valid():
    """Test valid batch prediction request."""
    requests = [
        PredictionRequest(
            price=100.0,
            hour=14,
            is_weekend=False,
            total_clicks=10,
            avg_session_time=120.5,
            purchase_freq=0.3,
            products_viewed=5,
            price_range=50.0,
            session_duration=300.0,
            recency=10,
            frequency=3,
            monetary=500.0,
            main_category="electronics",
            rfm_segment=2
        )
        for _ in range(5)
    ]
    
    batch_request = BatchPredictionRequest(data=requests)
    assert len(batch_request.data) == 5


def test_batch_prediction_request_empty():
    """Test batch prediction request with empty data."""
    with pytest.raises(ValidationError):
        BatchPredictionRequest(data=[])


def test_batch_prediction_request_too_large():
    """Test batch prediction request exceeding size limit."""
    requests = [
        PredictionRequest(
            price=100.0,
            hour=14,
            is_weekend=False,
            total_clicks=10,
            avg_session_time=120.5,
            purchase_freq=0.3,
            products_viewed=5,
            price_range=50.0,
            session_duration=300.0,
            recency=10,
            frequency=3,
            monetary=500.0,
            main_category="electronics",
            rfm_segment=2
        )
        for _ in range(1001)  # Exceeds limit of 1000
    ]
    
    with pytest.raises(ValidationError):
        BatchPredictionRequest(data=requests)


def test_health_check_response():
    """Test health check response."""
    response = HealthCheckResponse(
        status="healthy",
        model_loaded=True,
        model_version="v1.0"
    )
    
    assert response.status == "healthy"
    assert response.model_loaded == True
    assert isinstance(response.timestamp, datetime)
