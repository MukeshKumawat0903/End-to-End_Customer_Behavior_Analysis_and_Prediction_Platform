import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
import cupy as cp

class XGBoostEarlyStoppingWrapper(Pipeline):
    def __init__(self, steps, *, memory=None, verbose=False):
        super().__init__(steps, memory=memory, verbose=verbose)
        
    def _transform(self, X):
        """Internal transformation method"""
        for _, step in self.steps[:-1]:
            X = step.transform(X)
        return X
    
    def fit(self, X, y, xgb_eval_set=None):
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
    
    def predict_proba(self, X):
        Xt = self._transform(X)
        return self.steps[-1][1].predict_proba(Xt)
    
    def score(self, X, y):
        Xt = self._transform(X)
        return self.steps[-1][1].score(Xt, y)

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    threshold: float,
    numerical_features: list,
    categorical_features: list,
    class_labels: list = None
    ):

    """Train XGBoost classifier with hyperparameter tuning"""
    # preprocessor = ColumnTransformer([
    #     ('num', StandardScaler(), list(range(len(numerical_features)))),
    #     ('cat', OneHotEncoder(handle_unknown='ignore'), list(range(len(numerical_features), len(numerical_features + categorical_features))))
    # ])

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    xgb_pipe = XGBoostEarlyStoppingWrapper([
        ('preprocessor', preprocessor),
        ('xgb', xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            early_stopping_rounds=50,
            tree_method='hist',
            device='cpu',
            # device='cuda'  # ✅ This replaces gpu_hist

        ))
    ])

    # Configure parameter grid
    param_grid = {
        'xgb__learning_rate': [0.05, 0.1],
        'xgb__max_depth': [3, 5],
        'xgb__subsample': [0.8, 1.0],
        'xgb__colsample_bytree': [0.8, 1.0]
    }

    # Set up grid search
    grid_search = GridSearchCV(
        xgb_pipe,
        param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    # Train with proper evaluation handling
    grid_search.fit(X_train, y_train, 
                xgb_eval_set=(X_val, y_val))
    
    # Get the best model and make predictions on the test set
    model = grid_search.best_estimator_
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return model, y_proba, y_pred