"""
Configuration module for loading and managing application settings.
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, ConfigDict


class DataConfig(BaseModel):
    """Data configuration settings."""
    raw_data_path: str
    processed_data_path: str
    sample_size: Optional[int] = None
    test_size: float = 0.20
    validation_size: float = 0.25
    random_state: int = 42

    @field_validator('test_size', 'validation_size')
    @classmethod
    def validate_split_size(cls, v):
        if not 0 < v < 1:
            raise ValueError("Split size must be between 0 and 1")
        return v


class FeaturesConfig(BaseModel):
    """Feature configuration settings."""
    numerical: List[str]
    categorical: List[str]
    target: str


class HyperparametersConfig(BaseModel):
    """Hyperparameter configuration for model tuning."""
    learning_rate: List[float] = [0.01, 0.05, 0.1]
    max_depth: List[int] = [3, 5, 7]
    min_child_weight: List[int] = [1, 3, 5]
    subsample: List[float] = [0.8, 1.0]
    colsample_bytree: List[float] = [0.8, 1.0]
    n_estimators: List[int] = [100, 200]


class ModelConfig(BaseModel):
    """Model configuration settings."""
    type: str = "xgboost"
    objective: str = "binary:logistic"
    eval_metric: str = "auc"
    tree_method: str = "hist"
    device: str = "cpu"
    early_stopping_rounds: int = 50
    hyperparameters: HyperparametersConfig
    scale_pos_weight: Optional[float] = None
    search_method: str = "grid"
    cv_folds: int = 3
    n_iter: int = 20


class CrossValidationConfig(BaseModel):
    """Cross-validation configuration settings."""
    n_splits: int = 5
    stratified: bool = True
    shuffle: bool = True
    random_state: int = 42


class ThresholdConfig(BaseModel):
    """Threshold configuration settings."""
    default: float = 0.5
    optimization_metric: str = "f1"
    use_optimal: bool = True


class EvaluationConfig(BaseModel):
    """Evaluation configuration settings."""
    metrics: List[str]
    calibration_bins: int = 10
    percentiles: List[float] = [0.05, 0.1, 0.2, 0.3, 0.5]


class SHAPConfig(BaseModel):
    """SHAP configuration settings."""
    max_samples: int = 1000
    plot_samples: int = 50
    background_samples: int = 100


class RFMConfig(BaseModel):
    """RFM configuration settings."""
    n_clusters: int = 4
    features: List[str] = ["recency", "frequency", "monetary"]


class LoggingConfig(BaseModel):
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"
    file: str = "logs/app.log"
    max_bytes: int = 10485760
    backup_count: int = 5


class MLflowConfig(BaseModel):
    """MLflow configuration settings."""
    enabled: bool = True
    tracking_uri: str = "mlruns"
    experiment_name: str = "customer_behavior_prediction"
    artifact_location: str = "mlruns/artifacts"


class StorageConfig(BaseModel):
    """Storage configuration settings."""
    models_dir: str = "models"
    save_preprocessor: bool = True
    save_metadata: bool = True
    versioning: bool = True


class DashboardConfig(BaseModel):
    """Dashboard configuration settings."""
    title: str = "Customer Conversion Prediction Dashboard"
    page_layout: str = "wide"
    theme: str = "light"
    cache_data: bool = True


class APIConfig(BaseModel):
    """API configuration settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    workers: int = 4
    timeout: int = 60
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8501"]


class Config(BaseModel):
    """Main configuration class."""
    data: DataConfig
    features: FeaturesConfig
    model: ModelConfig
    cross_validation: CrossValidationConfig
    threshold: ThresholdConfig
    evaluation: EvaluationConfig
    shap: SHAPConfig
    rfm: RFMConfig
    logging: LoggingConfig
    mlflow: MLflowConfig
    storage: StorageConfig
    dashboard: DashboardConfig
    api: APIConfig


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file. If None, uses default path.
        
    Returns:
        Config object with all settings.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config file is invalid.
    """
    if config_path is None:
        # Get project root directory
        current_file = Path(__file__)
        project_root = current_file.parent.parent
        config_path = project_root / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    try:
        config = Config(**config_dict)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}")
    
    return config


def get_project_root() -> Path:
    """Get the project root directory."""
    current_file = Path(__file__)
    return current_file.parent.parent


def ensure_directories(config: Config) -> None:
    """
    Ensure all required directories exist.
    
    Args:
        config: Configuration object.
    """
    project_root = get_project_root()
    
    # Create directories
    dirs_to_create = [
        Path(config.storage.models_dir),
        Path(config.logging.file).parent,
        Path(config.mlflow.tracking_uri),
        Path(config.data.raw_data_path).parent,
        Path(config.data.processed_data_path).parent,
    ]
    
    for dir_path in dirs_to_create:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)


# Global config instance (lazy loaded)
_config: Optional[Config] = None


def get_config(reload: bool = False) -> Config:
    """
    Get the global configuration instance.
    
    Args:
        reload: If True, reload configuration from file.
        
    Returns:
        Config object.
    """
    global _config
    
    if _config is None or reload:
        _config = load_config()
        ensure_directories(_config)
    
    return _config


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"Model type: {config.model.type}")
    print(f"Data path: {config.data.raw_data_path}")
    print(f"Features: {len(config.features.numerical)} numerical, {len(config.features.categorical)} categorical")
