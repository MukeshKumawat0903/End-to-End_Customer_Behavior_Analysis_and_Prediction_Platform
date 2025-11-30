"""
Unit tests for data preprocessing functions.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.preprocess import (
    clean_data,
    create_time_features,
    create_behavior_features,
    create_session_features,
    calculate_rfm
)


@pytest.fixture
def sample_raw_data():
    """Create sample raw data for testing."""
    n_samples = 100
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    
    data = {
        'event_time': [base_time + timedelta(hours=i) for i in range(n_samples)],
        'event_type': np.random.choice(['view', 'cart', 'purchase'], n_samples),
        'product_id': np.random.randint(1, 100, n_samples),
        'category_code': np.random.choice(['electronics.smartphone', 'apparel.shoes', 'home.furniture'], n_samples),
        'brand': np.random.choice(['brand_a', 'brand_b', 'brand_c'], n_samples),
        'price': np.random.uniform(10, 1000, n_samples),
        'user_id': np.random.randint(1, 20, n_samples),
        'user_session': np.random.choice([f'session_{i}' for i in range(30)], n_samples)
    }
    
    df = pd.DataFrame(data)
    return df


def test_clean_data(sample_raw_data):
    """Test data cleaning."""
    # Add some duplicates and nulls
    df = sample_raw_data.copy()
    df.loc[0, 'price'] = None
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    
    cleaned = clean_data(df)
    
    # Check that nulls and duplicates are removed
    assert cleaned.isnull().sum().sum() == 0
    assert len(cleaned) < len(df)


def test_create_time_features(sample_raw_data):
    """Test time feature creation."""
    df = create_time_features(sample_raw_data.copy())
    
    # Check that time features exist
    assert 'hour' in df.columns
    assert 'day_of_week' in df.columns
    assert 'is_weekend' in df.columns
    
    # Validate ranges
    assert df['hour'].min() >= 0
    assert df['hour'].max() <= 23
    assert df['day_of_week'].min() >= 0
    assert df['day_of_week'].max() <= 6
    assert df['is_weekend'].isin([0, 1]).all()


def test_create_behavior_features(sample_raw_data):
    """Test behavior feature creation."""
    df = create_behavior_features(sample_raw_data.copy())
    
    # Check that behavior features exist
    assert 'total_clicks' in df.columns
    assert 'avg_session_time' in df.columns
    assert 'purchase_freq' in df.columns
    
    # Validate types and ranges
    assert df['total_clicks'].min() >= 0
    assert (df['purchase_freq'] >= 0).all()
    assert (df['purchase_freq'] <= 1).all()


def test_create_session_features(sample_raw_data):
    """Test session feature creation."""
    df = create_session_features(sample_raw_data.copy())
    
    # Check that session features exist
    assert 'products_viewed' in df.columns
    assert 'price_range' in df.columns
    assert 'session_duration' in df.columns
    assert 'main_category' in df.columns
    
    # Validate types and ranges
    assert df['products_viewed'].min() >= 0
    assert df['price_range'].min() >= 0
    assert df['session_duration'].min() >= 0


def test_calculate_rfm(sample_raw_data):
    """Test RFM calculation."""
    # Add time features first
    df = create_time_features(sample_raw_data.copy())
    df = calculate_rfm(df)
    
    # Check that RFM features exist
    assert 'recency' in df.columns
    assert 'frequency' in df.columns
    assert 'monetary' in df.columns
    assert 'rfm_segment' in df.columns
    
    # Validate types and ranges
    assert df['recency'].min() >= 0
    assert df['frequency'].min() >= 0
    assert df['monetary'].min() >= 0
    assert df['rfm_segment'].min() >= 0


def test_full_preprocessing_pipeline(sample_raw_data):
    """Test the full preprocessing pipeline."""
    df = sample_raw_data.copy()
    
    # Apply all transformations
    df = clean_data(df)
    df = create_time_features(df)
    df = create_behavior_features(df)
    df = create_session_features(df)
    df = calculate_rfm(df)
    
    # Check that all expected features exist
    expected_features = [
        'price', 'hour', 'is_weekend', 'total_clicks',
        'avg_session_time', 'purchase_freq', 'products_viewed',
        'price_range', 'session_duration', 'recency', 'frequency',
        'monetary', 'main_category', 'rfm_segment'
    ]
    
    for feature in expected_features:
        assert feature in df.columns, f"Missing feature: {feature}"
    
    # Check data types
    assert pd.api.types.is_numeric_dtype(df['price'])
    assert pd.api.types.is_integer_dtype(df['hour'])
    assert pd.api.types.is_integer_dtype(df['is_weekend'])
