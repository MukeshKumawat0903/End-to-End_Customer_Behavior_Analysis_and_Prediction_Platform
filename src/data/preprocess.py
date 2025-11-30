import gc
from typing import Tuple, Optional

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from pydantic import ValidationError

from src.visualization.visualize import run_eda_visualization, plot_rfm_segments
from src.utils.logger import get_logger
from src.schemas import RawDataSchema, ProcessedDataSchema

logger = get_logger(__name__)

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, tuple, tuple]:
    """
    Full preprocessing pipeline with logging and error handling.
    
    Args:
        df: Raw input DataFrame.
        
    Returns:
        Tuple of (processed_df, eda_results, rfm_results).
        
    Raises:
        ValueError: If required columns are missing.
    """
    logger.info(f"Starting preprocessing pipeline with {len(df)} rows")
    
    # Validate required columns
    required_cols = ['event_time', 'event_type', 'product_id', 'price', 'user_id', 'user_session']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df = clean_data(df)
    logger.info(f"After cleaning: {len(df)} rows")
    
    df = create_time_features(df)
    logger.info("Created time features")
    
    df = create_behavior_features(df)
    logger.info("Created behavior features")
    
    df = create_session_features(df)
    logger.info("Created session features")
    
    results = run_eda_visualization(df)
    logger.info("Generated EDA visualizations")
    
    df = encode_categoricals(df)
    logger.info("Encoded categorical features")
    
    df = calculate_rfm(df)
    logger.info("Calculated RFM features")
    
    rmf_results = plot_rfm_segments(df)
    logger.info("Generated RFM visualizations")
    
    df = required_data_filter_for_model_training(df)
    logger.info(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
    
    return df, results, rmf_results

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data by removing nulls and duplicates with schema validation.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Cleaned DataFrame.
    """
    initial_rows = len(df)
    
    # Remove nulls
    df = df.dropna()
    nulls_removed = initial_rows - len(df)
    if nulls_removed > 0:
        logger.info(f"Removed {nulls_removed} rows with null values")
    
    # Remove duplicates
    df = df.drop_duplicates(keep='last')
    duplicates_removed = initial_rows - nulls_removed - len(df)
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate rows")
    
    # Validate sample rows using RawDataSchema
    validate_raw_data_sample(df, sample_size=min(100, len(df)))
    
    return df

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from event_time.
    
    Args:
        df: Input DataFrame with 'event_time' column.
        
    Returns:
        DataFrame with additional time features.
    """
    # Ensure event_time is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['event_time']):
        df['event_time'] = pd.to_datetime(df['event_time'])
    
    # Sort by time for proper session calculations
    df = df.sort_values(['user_id', 'user_session', 'event_time']).reset_index(drop=True)
    
    df['hour'] = df['event_time'].dt.hour
    df['day_of_week'] = df['event_time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    return df

def create_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create user behavior features with proper session-aware calculations (optimized).
    
    Args:
        df: Input DataFrame with user and event data.
        
    Returns:
        DataFrame with additional behavior features.
    """
    # Sort data by user and time for proper diff calculations
    df_sorted = df.sort_values(['user_id', 'user_session', 'event_time']).copy()
    
    # User-level features (vectorized)
    user_stats = df_sorted.groupby('user_id').agg(
        total_clicks=('event_type', lambda x: (x == 'view').sum()),
        purchase_freq=('event_type', lambda x: (x == 'purchase').mean())
    ).reset_index()
    
    # Calculate average session time per user (vectorized approach)
    # Add time differences within each session
    df_sorted['time_diff'] = df_sorted.groupby(['user_id', 'user_session'])['event_time'].diff().dt.total_seconds()
    
    # Calculate average session time per user (only non-null diffs within sessions)
    avg_session_time = df_sorted.groupby('user_id')['time_diff'].mean().fillna(0).reset_index()
    avg_session_time.columns = ['user_id', 'avg_session_time']
    
    # Merge user stats
    user_stats = user_stats.merge(avg_session_time, on='user_id', how='left')
    user_stats['avg_session_time'] = user_stats['avg_session_time'].fillna(0)
    
    # Merge back to original dataframe
    df = df.merge(user_stats, on='user_id', how='left')
    
    return df

def create_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create session-level features"""
    session_stats = df.groupby('user_session').agg(
        session_start=('event_time', 'min'),
        session_end=('event_time', 'max'),
        products_viewed=('product_id', 'nunique'),
        price_range=('price', lambda x: x.max() - x.min())
    ).reset_index()
    
    session_stats['session_duration'] = (
        session_stats['session_end'] - session_stats['session_start']
    ).dt.total_seconds()
    
    df = df.merge(session_stats, on='user_session', how='left')
    
    # Extract main_category from original strings
    df['category_code'] = df['category_code'].fillna('unknown')
    df['main_category'] = df['category_code'].str.split('.', n=1, expand=True)[0]
    return df

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables as numeric codes.
    
    Args:
        df: Input DataFrame with categorical columns.
        
    Returns:
        DataFrame with encoded categorical features.
    """
    # Fill missing values before encoding
    if 'category_code' in df.columns:
        df['category_code'] = df['category_code'].fillna('unknown')
        df['category_code'] = df['category_code'].astype('category').cat.codes
    
    if 'brand' in df.columns:
        df['brand'] = df['brand'].fillna('unknown')
        df['brand'] = df['brand'].astype('category').cat.codes
    
    if 'main_category' in df.columns:
        df['main_category'] = df['main_category'].fillna('unknown')
        df['main_category'] = df['main_category'].astype('category').cat.codes
    
    return df

def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate RFM features and clusters"""
    current_date = df['event_time'].max() + pd.Timedelta(days=1)
    
    # Create RFM features
    rfm = df.groupby('user_id').agg(
        recency=('event_time', lambda x: (current_date - x.max()).days),
        frequency=('user_session', 'nunique'),
        monetary=('price', 'sum')
    ).reset_index()
    
    # Normalize and cluster
    scaler = preprocessing.StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['rfm_segment'] = kmeans.fit_predict(rfm_scaled)

    df_processed = df.merge(rfm, on='user_id', how='left', suffixes=('', '_rfm'))

    # Merge ALL RFM features with prefix
    return df_processed

def required_data_filter_for_model_training(df: pd.DataFrame) -> pd.DataFrame:
    """Filter data for model training with schema validation"""
    # Prepare modeling dataset
    model_df = df[df['event_type'] == 'cart'].copy()
    model_df['target'] = model_df['user_session'].isin(
        df[df['event_type'] == 'purchase']['user_session']
    ).astype(int)

    # To memory management
    # Step 1: Delete the large DataFrame explicitly
    del df
    # Step 2: Run garbage collection to reclaim memory
    gc.collect()

    required_features_cols = [
    'price', 'hour', 'is_weekend', 'total_clicks',
    'avg_session_time', 'purchase_freq', 'products_viewed',
    'price_range', 'session_duration', 'recency', 'frequency', 'monetary',
    'main_category', 'rfm_segment', 'target'
    ]
    cols_to_drop = [col for col in model_df.columns if col not in required_features_cols]
    model_df.drop(columns=cols_to_drop, inplace=True)

    print("Dropped Columns:", cols_to_drop)
    print("\nmodel_df Shape:", model_df.shape)
    print("\nmodel_df Columns:", model_df.columns)
    
    # Validate processed data using ProcessedDataSchema
    validate_processed_data_sample(model_df, sample_size=min(100, len(model_df)))
    
    return model_df


def validate_raw_data_sample(df: pd.DataFrame, sample_size: int = 100) -> None:
    """
    Validate a sample of raw data using RawDataSchema.
    
    Args:
        df: Raw DataFrame to validate.
        sample_size: Number of rows to validate.
        
    Raises:
        ValidationError: If validation fails.
    """
    logger.info(f"Validating {sample_size} rows of raw data...")
    sample_df = df.head(sample_size)
    
    validation_errors = 0
    for idx, row in sample_df.iterrows():
        try:
            RawDataSchema(**row.to_dict())
        except ValidationError as e:
            validation_errors += 1
            if validation_errors <= 5:  # Log first 5 errors
                logger.warning(f"Validation error at row {idx}: {e}")
    
    if validation_errors > 0:
        logger.warning(f"Found {validation_errors} validation errors in {sample_size} rows")
    else:
        logger.info(f"All {sample_size} raw data rows passed validation")


def validate_processed_data_sample(df: pd.DataFrame, sample_size: int = 100) -> None:
    """
    Validate a sample of processed data using ProcessedDataSchema.
    
    Args:
        df: Processed DataFrame to validate.
        sample_size: Number of rows to validate.
        
    Raises:
        ValidationError: If validation fails.
    """
    logger.info(f"Validating {sample_size} rows of processed data...")
    sample_df = df.head(sample_size)
    
    validation_errors = 0
    for idx, row in sample_df.iterrows():
        try:
            ProcessedDataSchema(**row.to_dict())
        except ValidationError as e:
            validation_errors += 1
            if validation_errors <= 5:  # Log first 5 errors
                logger.warning(f"Validation error at row {idx}: {e}")
    
    if validation_errors > 0:
        logger.warning(f"Found {validation_errors} validation errors in {sample_size} rows")
    else:
        logger.info(f"All {sample_size} processed data rows passed validation")