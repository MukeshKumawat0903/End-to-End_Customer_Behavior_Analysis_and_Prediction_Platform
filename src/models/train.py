from typing import Optional, Dict, Any, Tuple, List

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from src.utils.logger import get_logger
from src.config import get_config

logger = get_logger(__name__)


class XGBoostEarlyStoppingWrapper(Pipeline):
    """Custom Pipeline wrapper that supports XGBoost early stopping."""
    
    def __init__(self, steps, *, memory=None, verbose=False):
        """
        Initialize the wrapper.
        
        Args:
            steps: List of (name, transform) tuples.
            memory: Memory caching parameter.
            verbose: Verbosity flag.
        """
        super().__init__(steps, memory=memory, verbose=verbose)
        
    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Internal transformation method.
        
        Args:
            X: Input features.
            
        Returns:
            Transformed features.
        """
        for _, step in self.steps[:-1]:
            X = step.transform(X)
        return X
    
    def fit(self, X: pd.DataFrame, y: pd.Series, xgb_eval_set: Optional[Tuple] = None) -> 'XGBoostEarlyStoppingWrapper':
        """
        Fit the pipeline with early stopping support.
        
        Args:
            X: Training features.
            y: Training labels.
            xgb_eval_set: Optional tuple of (X_val, y_val) for early stopping.
            
        Returns:
            Self.
        """
        # Fit all preprocessing steps
        Xt = X
        for name, step in self.steps[:-1]:
            if hasattr(step, "fit_transform"):
                Xt = step.fit_transform(Xt, y)
            else:
                Xt = step.fit(Xt, y).transform(Xt)
        
        # Process evaluation set
        if xgb_eval_set:
            X_val, y_val = xgb_eval_set
            X_val_t = self._transform(X_val)
            eval_set = [(X_val_t, y_val)]
        else:
            eval_set = None
            
        # Fit XGBoost model
        self.steps[-1][1].fit(Xt, y, eval_set=eval_set)
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        Xt = self._transform(X)
        return self.steps[-1][1].predict_proba(Xt)
    
    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Calculate score."""
        Xt = self._transform(X)
        return self.steps[-1][1].score(Xt, y)


def calculate_scale_pos_weight(y: pd.Series) -> float:
    """
    Calculate scale_pos_weight for imbalanced datasets.
    
    Args:
        y: Target labels.
        
    Returns:
        Calculated scale_pos_weight.
    """
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f} (neg: {neg_count}, pos: {pos_count})")
    return scale_pos_weight


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    threshold: float,
    numerical_features: List[str],
    categorical_features: List[str],
    class_labels: Optional[List[str]] = None,
    use_config: bool = True
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    Train XGBoost classifier with hyperparameter tuning and class imbalance handling.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        X_test: Test features.
        threshold: Classification threshold.
        numerical_features: List of numerical feature names.
        categorical_features: List of categorical feature names.
        class_labels: Optional class labels for display.
        use_config: Whether to use config file parameters.
        
    Returns:
        Tuple of (trained_model, test_probabilities, test_predictions).
    """
    logger.info("Starting model training...")
    
    # Load config if requested
    if use_config:
        config = get_config()
        model_config = config.model
    else:
        model_config = None
    
    # Create preprocessor
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    
    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = calculate_scale_pos_weight(y_train)
    if model_config and model_config.scale_pos_weight is not None:
        scale_pos_weight = model_config.scale_pos_weight
    
    # Get model parameters
    if model_config:
        objective = model_config.objective
        eval_metric = model_config.eval_metric
        tree_method = model_config.tree_method
        device = model_config.device
        early_stopping_rounds = model_config.early_stopping_rounds
    else:
        objective = 'binary:logistic'
        eval_metric = 'auc'
        tree_method = 'hist'
        device = 'cpu'
        early_stopping_rounds = 50
    
    logger.info(f"Model config: objective={objective}, eval_metric={eval_metric}, device={device}")
    
    # Create pipeline
    xgb_pipe = XGBoostEarlyStoppingWrapper([
        ('preprocessor', preprocessor),
        ('xgb', xgb.XGBClassifier(
            objective=objective,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            tree_method=tree_method,
            device=device,
            scale_pos_weight=scale_pos_weight,
            random_state=42
        ))
    ])
    
    # Configure parameter grid
    if model_config and model_config.hyperparameters:
        hp = model_config.hyperparameters
        param_grid = {
            'xgb__learning_rate': hp.learning_rate,
            'xgb__max_depth': hp.max_depth,
            'xgb__min_child_weight': hp.min_child_weight,
            'xgb__subsample': hp.subsample,
            'xgb__colsample_bytree': hp.colsample_bytree,
            'xgb__n_estimators': hp.n_estimators
        }
        logger.info(f"Using parameter grid from config with {np.prod([len(v) for v in param_grid.values()])} combinations")
    else:
        param_grid = {
            'xgb__learning_rate': [0.05, 0.1],
            'xgb__max_depth': [3, 5],
            'xgb__subsample': [0.8, 1.0],
            'xgb__colsample_bytree': [0.8, 1.0]
        }
        logger.info("Using default parameter grid")
    
    # Determine search method
    if model_config and model_config.search_method == 'random':
        logger.info(f"Using RandomizedSearchCV with {model_config.n_iter} iterations")
        search = RandomizedSearchCV(
            xgb_pipe,
            param_grid,
            n_iter=model_config.n_iter,
            cv=model_config.cv_folds,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
    else:
        logger.info("Using GridSearchCV")
        cv_folds = model_config.cv_folds if model_config else 3
        search = GridSearchCV(
            xgb_pipe,
            param_grid,
            cv=cv_folds,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
    
    # Train with proper evaluation handling
    logger.info("Fitting model...")
    search.fit(X_train, y_train, xgb_eval_set=(X_val, y_val))
    
    # Get the best model
    model = search.best_estimator_
    logger.info(f"Best parameters: {search.best_params_}")
    logger.info(f"Best CV score: {search.best_score_:.4f}")
    
    # Make predictions on the test set
    logger.info("Making predictions on test set...")
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    logger.info("Model training completed successfully")
    return model, y_proba, y_pred