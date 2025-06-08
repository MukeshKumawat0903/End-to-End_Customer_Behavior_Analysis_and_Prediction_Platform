import gc
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from src.visualization.visualize import run_eda_visualization, plot_rfm_segments

def preprocess_data(df: pd.DataFrame):
    """Full preprocessing pipeline"""
    df = clean_data(df)
    df = create_time_features(df)
    df = create_behavior_features(df)
    df = create_session_features(df)
    results = run_eda_visualization(df)
    df = encode_categoricals(df)
    df = calculate_rfm(df)
    rmf_results = plot_rfm_segments(df)
    df = required_data_filter_for_model_training(df)
    return df, results, rmf_results

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Data cleaning steps"""
    return df.dropna().drop_duplicates(keep='last')

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features"""
    df['hour'] = df['event_time'].dt.hour
    df['day_of_week'] = df['event_time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    return df

def create_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create user behavior features"""
    # User-level features
    user_stats = df.groupby('user_id').agg(
        total_clicks=('event_type', lambda x: (x == 'view').sum()),
        avg_session_time=('event_time', lambda x: x.diff().mean().total_seconds()),
        purchase_freq=('event_type', lambda x: (x == 'purchase').mean())
    ).reset_index()
    
    return df.merge(user_stats, on='user_id', how='left')

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
    """Encode categorical variables"""
    # Convert to categorical codes
    df['category_code'] = df['category_code'].astype('category').cat.codes
    df['brand'] = df['brand'].astype('category').cat.codes
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
    """Filter data for model training"""
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
    
    return model_df