"""
MLflow integration for experiment tracking.
"""
import os
from typing import Dict, Any, Optional
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd

from src.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLflowTracker:
    """MLflow experiment tracker for model training and evaluation."""
    
    def __init__(self, experiment_name: Optional[str] = None):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment. If None, uses config.
        """
        self.config = get_config()
        
        if not self.config.mlflow.enabled:
            logger.info("MLflow tracking is disabled in config")
            self.enabled = False
            return
        
        self.enabled = True
        
        # Set tracking URI
        tracking_uri = self.config.mlflow.tracking_uri
        if not tracking_uri.startswith(('http://', 'https://', 'file://')):
            # Convert relative path to absolute
            project_root = Path(__file__).parent.parent.parent
            tracking_uri = str(project_root / tracking_uri)
        
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI: {tracking_uri}")
        
        # Set experiment
        if experiment_name is None:
            experiment_name = self.config.mlflow.experiment_name
        
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment: {experiment_name}")
        
        self.run_id = None
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        Start an MLflow run.
        
        Args:
            run_name: Optional name for the run.
            tags: Optional tags for the run.
        """
        if not self.enabled:
            return
        
        mlflow.start_run(run_name=run_name, tags=tags)
        self.run_id = mlflow.active_run().info.run_id
        logger.info(f"Started MLflow run: {self.run_id}")
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters to log.
        """
        if not self.enabled:
            return
        
        for key, value in params.items():
            try:
                # MLflow doesn't accept complex types
                if isinstance(value, (list, dict, tuple)):
                    value = str(value)
                mlflow.log_param(key, value)
            except Exception as e:
                logger.warning(f"Failed to log parameter {key}: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics to log.
            step: Optional step number for metric tracking.
        """
        if not self.enabled:
            return
        
        for key, value in metrics.items():
            try:
                # Convert numpy types to Python types
                if isinstance(value, (np.integer, np.floating)):
                    value = float(value)
                elif isinstance(value, np.ndarray):
                    value = float(value.item()) if value.size == 1 else float(np.mean(value))
                
                mlflow.log_metric(key, value, step=step)
            except Exception as e:
                logger.warning(f"Failed to log metric {key}: {e}")
    
    def log_model(self, model: Any, artifact_path: str = "model", **kwargs):
        """
        Log model to MLflow.
        
        Args:
            model: Model to log.
            artifact_path: Path within the run's artifact directory.
            **kwargs: Additional arguments for model logging.
        """
        if not self.enabled:
            return
        
        try:
            # Detect model type and log accordingly
            model_type = str(type(model)).lower()
            
            if 'xgboost' in model_type or 'xgb' in model_type:
                mlflow.xgboost.log_model(model, artifact_path, **kwargs)
            elif 'sklearn' in model_type or 'pipeline' in model_type:
                mlflow.sklearn.log_model(model, artifact_path, **kwargs)
            else:
                # Fallback to sklearn
                mlflow.sklearn.log_model(model, artifact_path, **kwargs)
            
            logger.info(f"Logged model to MLflow: {artifact_path}")
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact to MLflow.
        
        Args:
            local_path: Local path to the artifact.
            artifact_path: Optional path within the run's artifact directory.
        """
        if not self.enabled:
            return
        
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.info(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")
    
    def log_figure(self, figure, artifact_file: str):
        """
        Log a matplotlib/plotly figure to MLflow.
        
        Args:
            figure: Matplotlib or Plotly figure object.
            artifact_file: Filename for the artifact.
        """
        if not self.enabled:
            return
        
        try:
            mlflow.log_figure(figure, artifact_file)
            logger.info(f"Logged figure: {artifact_file}")
        except Exception as e:
            logger.error(f"Failed to log figure: {e}")
    
    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str):
        """
        Log a dictionary as JSON artifact.
        
        Args:
            dictionary: Dictionary to log.
            artifact_file: Filename for the artifact.
        """
        if not self.enabled:
            return
        
        try:
            mlflow.log_dict(dictionary, artifact_file)
            logger.info(f"Logged dictionary: {artifact_file}")
        except Exception as e:
            logger.error(f"Failed to log dictionary: {e}")
    
    def log_dataframe(self, df: pd.DataFrame, artifact_file: str):
        """
        Log a pandas DataFrame as artifact.
        
        Args:
            df: DataFrame to log.
            artifact_file: Filename for the artifact.
        """
        if not self.enabled:
            return
        
        try:
            # Save temporarily and log
            temp_path = f"/tmp/{artifact_file}"
            df.to_csv(temp_path, index=False)
            mlflow.log_artifact(temp_path)
            os.remove(temp_path)
            logger.info(f"Logged dataframe: {artifact_file}")
        except Exception as e:
            logger.error(f"Failed to log dataframe: {e}")
    
    def set_tags(self, tags: Dict[str, str]):
        """
        Set tags for the current run.
        
        Args:
            tags: Dictionary of tags.
        """
        if not self.enabled:
            return
        
        for key, value in tags.items():
            try:
                mlflow.set_tag(key, value)
            except Exception as e:
                logger.warning(f"Failed to set tag {key}: {e}")
    
    def end_run(self, status: str = "FINISHED"):
        """
        End the current MLflow run.
        
        Args:
            status: Run status (FINISHED, FAILED, KILLED).
        """
        if not self.enabled:
            return
        
        try:
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run: {self.run_id}")
        except Exception as e:
            logger.error(f"Failed to end run: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start_run()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.end_run(status="FAILED")
        else:
            self.end_run(status="FINISHED")


def log_training_run(
    model: Any,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    artifacts: Optional[Dict[str, str]] = None,
    tags: Optional[Dict[str, str]] = None,
    run_name: Optional[str] = None
):
    """
    Convenience function to log a complete training run.
    
    Args:
        model: Trained model.
        params: Model parameters.
        metrics: Evaluation metrics.
        artifacts: Optional dictionary of artifact paths.
        tags: Optional tags for the run.
        run_name: Optional name for the run.
    """
    tracker = MLflowTracker()
    
    if not tracker.enabled:
        logger.info("MLflow tracking disabled, skipping logging")
        return
    
    with tracker:
        # Set tags
        if tags:
            tracker.set_tags(tags)
        
        # Log parameters
        tracker.log_params(params)
        
        # Log metrics
        tracker.log_metrics(metrics)
        
        # Log model
        tracker.log_model(model)
        
        # Log artifacts
        if artifacts:
            for artifact_path in artifacts.values():
                if os.path.exists(artifact_path):
                    tracker.log_artifact(artifact_path)
        
        logger.info("Training run logged to MLflow successfully")
