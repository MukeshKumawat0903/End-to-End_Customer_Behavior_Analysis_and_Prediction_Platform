"""
Data validation schemas using Pydantic for ensuring data quality.
"""
from datetime import datetime
from typing import Optional, List, Annotated

from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic.types import StringConstraints


class RawDataSchema(BaseModel):
    """Schema for raw e-commerce data."""
    event_time: datetime
    event_type: Annotated[str, StringConstraints(pattern=r'^(view|cart|purchase|remove_from_cart)$')]
    product_id: int = Field(gt=0)
    category_id: int = Field(ge=0)
    category_code: Optional[str] = None
    brand: Optional[str] = None
    price: float = Field(gt=0, description="Price must be positive")
    user_id: int = Field(gt=0)
    user_session: str
    
    @field_validator('price')
    @classmethod
    def validate_price(cls, v):
        """Ensure price is reasonable."""
        if v > 100000:  # Sanity check
            raise ValueError(f"Price {v} seems unreasonably high")
        return v


class ProcessedDataSchema(BaseModel):
    """Schema for processed/engineered features."""
    price: float = Field(gt=0)
    hour: int = Field(ge=0, le=23)
    is_weekend: int = Field(ge=0, le=1)
    total_clicks: int = Field(ge=0)
    avg_session_time: float = Field(ge=0)
    purchase_freq: float = Field(ge=0, le=1)
    products_viewed: int = Field(ge=0)
    price_range: float = Field(ge=0)
    session_duration: float = Field(ge=0)
    recency: int = Field(ge=0)
    frequency: int = Field(ge=0)
    monetary: float = Field(ge=0)
    main_category: int = Field(ge=0)
    rfm_segment: int = Field(ge=0)
    target: int = Field(ge=0, le=1)
    
    model_config = ConfigDict(validate_assignment=True)


class PredictionRequest(BaseModel):
    """Schema for API prediction requests."""
    price: float = Field(gt=0, description="Product price")
    hour: int = Field(ge=0, le=23, description="Hour of day (0-23)")
    is_weekend: bool = Field(description="Whether event occurred on weekend")
    total_clicks: int = Field(ge=0, description="Total user clicks")
    avg_session_time: float = Field(ge=0, description="Average session time in seconds")
    purchase_freq: float = Field(ge=0, le=1, description="Historical purchase frequency")
    products_viewed: int = Field(ge=0, description="Number of products viewed in session")
    price_range: float = Field(ge=0, description="Price range in session")
    session_duration: float = Field(ge=0, description="Session duration in seconds")
    recency: int = Field(ge=0, description="Days since last purchase")
    frequency: int = Field(ge=0, description="Number of past purchases")
    monetary: float = Field(ge=0, description="Total monetary value of purchases")
    main_category: str = Field(description="Main product category")
    rfm_segment: int = Field(ge=0, le=3, description="RFM segment (0-3)")
    
    @field_validator('purchase_freq')
    @classmethod
    def validate_purchase_freq(cls, v):
        """Ensure purchase frequency is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Purchase frequency must be between 0 and 1")
        return v


class PredictionResponse(BaseModel):
    """Schema for API prediction responses."""
    probability: float = Field(ge=0, le=1, description="Probability of conversion")
    prediction: int = Field(ge=0, le=1, description="Binary prediction (0 or 1)")
    confidence: str = Field(description="Confidence level (low, medium, high)")
    threshold: float = Field(ge=0, le=1, description="Threshold used for classification")
    
    @field_validator('confidence', mode='before')
    @classmethod
    def set_confidence(cls, v, info):
        """Calculate confidence level based on probability."""
        if v and v != '':
            return v
        
        # Get probability from other fields
        data = info.data if hasattr(info, 'data') else {}
        if 'probability' in data:
            prob = data['probability']
            if prob < 0.3 or prob > 0.7:
                return "high"
            elif prob < 0.4 or prob > 0.6:
                return "medium"
            else:
                return "low"
        return "medium"


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction requests."""
    data: List[PredictionRequest]
    
    @field_validator('data')
    @classmethod
    def validate_batch_size(cls, v):
        """Ensure batch size is reasonable."""
        if len(v) == 0:
            raise ValueError("Batch cannot be empty")
        if len(v) > 1000:
            raise ValueError("Batch size exceeds maximum of 1000")
        return v


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction responses."""
    predictions: List[PredictionResponse]
    count: int = Field(ge=0, description="Number of predictions")
    
    @field_validator('count', mode='before')
    @classmethod
    def set_count(cls, v, info):
        """Set count based on predictions list."""
        data = info.data if hasattr(info, 'data') else {}
        if 'predictions' in data:
            return len(data['predictions'])
        return v if v is not None else 0


class ModelMetadata(BaseModel):
    """Schema for model metadata."""
    model_name: str
    model_version: str
    training_date: datetime
    metrics: dict
    features: List[str]
    hyperparameters: dict
    
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )


class HealthCheckResponse(BaseModel):
    """Schema for API health check response."""
    status: str = Field(description="Service status")
    model_loaded: bool = Field(description="Whether model is loaded")
    model_version: Optional[str] = Field(None, description="Model version")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )
