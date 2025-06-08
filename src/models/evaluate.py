import logging
import numpy as np
import pandas as pd
from sklearn.base import clone
from typing import Any, Dict, List, Optional, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, brier_score_loss,
    log_loss, classification_report, average_precision_score
)
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

DataType = Any

class MetricsHelper:
    @staticmethod
    def compute_metrics(y_true, y_pred, y_proba):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "brier_score": brier_score_loss(y_true, y_proba)
        }

class CrossValidator:
    def __init__(self, model: Any, model_name: str = "Model"):
        """
        Args:
            model: The scikit-learn compatible model to evaluate.
            model_name: A name for the model, used in plot titles.
        """
        self.model = model
        self.model_name = model_name
        self.cv_results: Dict[str, Any] = {}
        self.oof_predictions: Optional[np.ndarray] = None
        self.y_train_oof: Optional[DataType] = None # To store y_train corresponding to OOF

    def detect_booster_type(self, model):
        """
        Detects the type of gradient boosting model (XGBoost or LightGBM) used within a given model or pipeline.

        Args:
            model: The model instance to inspect. Can be a direct estimator, a scikit-learn Pipeline, or a custom wrapper.

        Returns:
            str or None: Returns 'xgb' if an XGBoost model is detected, 'lgbm' if a LightGBM model is detected, or None if neither is found.

        Notes:
            - Handles direct estimators, scikit-learn Pipelines (by inspecting named_steps), 
            and custom wrappers with 'model' or 'estimator' attributes.
            - Useful for enabling framework-specific logic (e.g., early stopping) in evaluation or training utilities.
        """
        model_str = str(type(model)).lower()
        if 'xgboost' in model_str or 'xgb' in model_str:
            return 'xgb'
        if 'lightgbm' in model_str or 'lgbm' in model_str:
            return 'lgbm'
        
        # Handle pipeline or custom wrapper
        if hasattr(model, 'named_steps'):
            for name, step in model.named_steps.items():
                step_str = str(type(step)).lower()
                if 'xgboost' in step_str or 'xgb' in step_str:
                    return 'xgb'
                if 'lightgbm' in step_str or 'lgbm' in step_str:
                    return 'lgbm'

        # Try accessing nested model directly if custom
        if hasattr(model, 'model') or hasattr(model, 'estimator'):
            inner = getattr(model, 'model', None) or getattr(model, 'estimator', None)
            return self.detect_booster_type(inner)

        return None

    def _fit_model_with_early_stopping(
        self,
        fold_model: Any,
        X_train_fold: DataType,
        y_train_fold: DataType,
        X_val_fold: Optional[DataType] = None,
        y_val_fold: Optional[DataType] = None,
        early_stopping_rounds: Optional[int] = None,
        model_fit_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Fits the model with optional early stopping if supported."""

        if model_fit_params is None:
            model_fit_params = {}

        model_type_str = str(type(fold_model)).lower()
        booster_type = self.detect_booster_type(fold_model)
        is_xgb = booster_type == 'xgb'
        is_lgbm = booster_type == 'lgbm'

        use_early_stopping = early_stopping_rounds is not None and (is_xgb or is_lgbm)

        # If early stopping is requested but no validation data provided
        if use_early_stopping and (X_val_fold is None or y_val_fold is None):
            raise ValueError("Validation data is required for early stopping.")

        try:
            if is_xgb and use_early_stopping:
                if hasattr(fold_model, "fit"):
                    try:
                        # Try passing the eval_set normally
                        fold_model.fit(
                            X_train_fold, y_train_fold,
                            eval_set=[(X_val_fold, y_val_fold)],
                            early_stopping_rounds=early_stopping_rounds,
                            verbose=False,
                            **model_fit_params
                        )
                    except TypeError as e:
                        # Pipeline case - try alternate fit_param
                        if hasattr(fold_model, 'fit'):
                            fit_params = {'xgb_eval_set': (X_val_fold, y_val_fold)}
                            fold_model.fit(X_train_fold, y_train_fold, **fit_params)
            elif is_lgbm and use_early_stopping:
                fold_model.fit(
                    X_train_fold,
                    y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False,
                    **model_fit_params
                )
            else:
                # No early stopping
                fold_model.fit(X_train_fold, y_train_fold, **model_fit_params)

        except Exception as e:
            print(f"⚠️ Early stopping failed: {e}\n➡️ Retrying without early stopping.")
            fold_model.fit(X_train_fold, y_train_fold, **model_fit_params)

        return fold_model

    def perform_cross_validation(
        self,
        X: DataType,
        y: DataType,
        n_splits: int = 5,
        random_state: int = 42,
        early_stopping_rounds: Optional[int] = 20,
        model_fit_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Performs stratified K-fold cross-validation.

        Args:
            X: Feature matrix.
            y: Target vector.
            n_splits: Number of folds.
            random_state: Seed for reproducibility.
            early_stopping_rounds: Number of rounds for early stopping (for XGBoost/LightGBM).
                                   Set to None to disable.
            model_fit_params: Additional parameters to pass to the model's fit method.

        Returns:
            A dictionary with CV results (AUC scores, best iterations, OOF predictions).
        """
        # Set up StratifiedKFold
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # Initialize lists to store results
        fold_auc_scores: List[float] = []
        fold_pr_auc_scores: List[float] = []
        fold_f1_scores: List[float] = [] # F1 for positive class
        best_iterations: List[int] = []
        
        # Initialize OOF predictions array
        oof_preds_array = np.zeros(len(y))
        self.y_train_oof = y # Store y for OOF evaluation

        for fold_num, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            # Clone the fresh model for each fold
            fold_model = clone(self.model) 

            # Handle pandas DataFrame/Series indexing
            X_train_fold, X_val_fold = (X.iloc[train_idx], X.iloc[val_idx]) if hasattr(X, 'iloc') else (X[train_idx], X[val_idx])
            y_train_fold, y_val_fold = (y.iloc[train_idx], y.iloc[val_idx]) if hasattr(y, 'iloc') else (y[train_idx], y[val_idx])

            X_train_fold = pd.DataFrame(X.iloc[train_idx]) if hasattr(X, 'iloc') else pd.DataFrame(X[train_idx])
            X_val_fold = pd.DataFrame(X.iloc[val_idx]) if hasattr(X, 'iloc') else pd.DataFrame(X[val_idx])
            y_train_fold = pd.Series(y.iloc[train_idx]) if hasattr(y, 'iloc') else pd.Series(y[train_idx])
            y_val_fold = pd.Series(y.iloc[val_idx]) if hasattr(y, 'iloc') else pd.Series(y[val_idx])

            # Fit the model with early stopping if applicable
            fold_model = self._fit_model_with_early_stopping(
                fold_model, X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                early_stopping_rounds, model_fit_params
            )

            # Make predictions
            y_pred_proba_fold = fold_model.predict_proba(X_val_fold)[:, 1]
            oof_preds_array[val_idx] = y_pred_proba_fold
            y_pred_binary_fold = (y_pred_proba_fold >= 0.5).astype(int) # For F1

            # Calculate metrics - AUC, PR AUC, F1
            fold_auc = roc_auc_score(y_val_fold, y_pred_proba_fold)
            fold_pr_auc = average_precision_score(y_val_fold, y_pred_proba_fold)
            report_dict = classification_report(
                y_val_fold,
                y_pred_binary_fold,
                output_dict=True,
                zero_division=0
            )

            # Get the positive class label for F1 score extraction.
            unique_labels_in_y = np.unique(y)
            if len(unique_labels_in_y) > 1:
                # Sort to ensure consistent positive class selection (e.g., 0/1 or string labels).
                # Positive class is usually the higher value or second in sorted order.
                positive_class_label = str(sorted(unique_labels_in_y)[1])
            else:
                # Fallback if only one class is present (should not occur with stratified CV).
                positive_class_label = str(unique_labels_in_y[0]) if len(unique_labels_in_y) > 0 else '1'

            fold_f1 = report_dict.get(positive_class_label, {}).get('f1-score', 0)

            # Append metrics to lists
            fold_auc_scores.append(fold_auc)
            fold_pr_auc_scores.append(fold_pr_auc)
            fold_f1_scores.append(fold_f1)

            print(f"  Fold AUC: {fold_auc:.4f} | PR AUC: {fold_pr_auc:.4f} | F1 (0.5 thr): {fold_f1:.4f}")

            # Capture best iteration if available (XGBoost/LightGBM)
            current_best_iter = None
            try:
                if hasattr(fold_model, 'best_iteration_') and fold_model.best_iteration_ is not None:  # LightGBM sklearn API
                    current_best_iter = fold_model.best_iteration_
                elif hasattr(fold_model, 'best_iteration') and fold_model.best_iteration is not None:  # XGBoost sklearn API
                    current_best_iter = fold_model.best_iteration
                elif hasattr(fold_model, 'get_booster') and hasattr(fold_model.get_booster(), 'best_iteration'):  # Native XGBoost
                    current_best_iter = fold_model.get_booster().best_iteration
                elif hasattr(fold_model, '_Booster') and hasattr(fold_model._Booster, 'best_iteration'):  # LightGBM native
                    current_best_iter = fold_model._Booster.best_iteration

                if current_best_iter is not None:
                    best_iterations.append(current_best_iter)
                    print(f"  Best Iteration: {current_best_iter}")
            except AttributeError:
                pass  # Model doesn't have this attribute

        # Store OOF predictions
        self.oof_predictions = oof_preds_array
        self.cv_results = {
            'mean_auc': np.mean(fold_auc_scores), 'std_auc': np.std(fold_auc_scores),
            'auc_scores': fold_auc_scores,
            'mean_pr_auc': np.mean(fold_pr_auc_scores), 'std_pr_auc': np.std(fold_pr_auc_scores),
            'pr_auc_scores': fold_pr_auc_scores,
            'mean_f1_score_thresh05': np.mean(fold_f1_scores), 'std_f1_score_thresh05': np.std(fold_f1_scores),
            'f1_scores_thresh05': fold_f1_scores,
            'best_iterations': best_iterations,
            'oof_predictions': self.oof_predictions
        }

        print("\n📈 Cross-Validation Summary:")
        print(f"  Mean ROC AUC: {self.cv_results['mean_auc']:.4f} ± {self.cv_results['std_auc']:.4f}")
        print(f"  Mean PR AUC:  {self.cv_results['mean_pr_auc']:.4f} ± {self.cv_results['std_pr_auc']:.4f}")
        print(f"  Mean F1 (0.5 thr): {self.cv_results['mean_f1_score_thresh05']:.4f} ± {self.cv_results['std_f1_score_thresh05']:.4f}")
        if best_iterations:
            print(f"  Average Best Iteration: {np.mean(best_iterations):.0f}")
        return self.cv_results

class ModelEvaluator:
    def __init__(self, model: Any, model_name: str = "Model"):
        self.model = model
        self.model_name = model_name
        self.oof_predictions = None
        self.y_train_oof = None
        self.trained_models = []

    def perform_cross_validation(
        self, X: DataType, y: DataType,
        n_splits: int = 5,
        early_stopping_rounds: Optional[int] = 20,
        model_fit_params: Optional[Dict[str, Any]] = None
    ):
        logger.info("🚀 Starting cross-validation using CrossValidator...")
        cross_validator = CrossValidator(self.model)
        self.cv_results = cross_validator.perform_cross_validation(
            X, y,
            n_splits=n_splits,
            early_stopping_rounds=early_stopping_rounds,
            model_fit_params=model_fit_params
        )
        self.oof_predictions = cross_validator.oof_predictions
        self.y_train_oof = cross_validator.y_train_oof
        logger.info("🎯 Cross-validation completed and stored in evaluator.")

        return self.oof_predictions, self.y_train_oof, self.cv_results

    def find_optimal_threshold(self, y_true, y_scores, metric='f1') -> float:
        thresholds = np.linspace(0.01, 0.99, 99)
        scores = []
        for thresh in thresholds:
            y_pred = (y_scores >= thresh).astype(int)
            if metric == 'f1':
                score = f1_score(y_true, y_pred)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred)
            scores.append(score)
        best_threshold = thresholds[np.argmax(scores)]
        print(f"Optimal Threshold for {metric}: {best_threshold:.2f}")
        return best_threshold
    
def run_full_evaluation(
    trained_model: Any,
    X_train: DataType,
    y_train: DataType,
    X_test: DataType,
    y_test: DataType,
    class_labels: List[str],
    n_splits: int = 5,
    early_stopping_rounds: Optional[int] = 20,
    model_fit_params: Optional[Dict[str, Any]] = None
):
    """
    Runs a full evaluation pipeline including cross-validation, OOF evaluation, test set evaluation,
    threshold optimization, and various plots.

    Args:
        trained_model: The trained model to evaluate.
        X_train: Training feature matrix.
        y_train: Training target vector.
        X_test: Test feature matrix.
        y_test: Test target vector.
        class_labels: List of class labels for confusion matrix and plots.
        n_splits: Number of splits for cross-validation.
        early_stopping_rounds: Rounds for early stopping (if applicable).
        model_fit_params: Additional parameters to pass to the model's fit method.

    Returns:
        None
    """
    logger.info("🔍 Starting full evaluation...Perform cross-validation")

    # Clone the trained model to avoid modifying the original
    evaluator = ModelEvaluator(
        model=clone(trained_model),
        model_name=trained_model.__class__.__name__
    )

    # Perform cross-validation
    oof_predictions, y_train_oof, cv_results  = evaluator.perform_cross_validation(
        X_train, y_train,
        n_splits=n_splits,
        early_stopping_rounds=early_stopping_rounds,
        model_fit_params=model_fit_params
    )
    # Set evaluator's model to the fully trained model (if needed for test set)
    evaluator.model = trained_model

    # Find optimal threshold based on F1 score
    optimal_threshold = evaluator.find_optimal_threshold(
        evaluator.y_train_oof, evaluator.oof_predictions, metric='f1'
    )
    
    logger.info("📊 Cross-Validation Summary Geberated")
    return oof_predictions, y_train_oof, optimal_threshold, cv_results
