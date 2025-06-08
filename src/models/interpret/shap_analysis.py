import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Any

def calculate_shap_values(
    model: Any,
    preprocessor: Any,
    data: pd.DataFrame,
    model_type: str = 'xgb',
    # feature_input_names: list = None
) -> Tuple[shap.Explanation, np.ndarray, list]:
    """Calculate SHAP values for model explanations."""
    # Extract processed features and names
    processed_data = preprocessor.transform(data)
    feature_names = preprocessor.get_feature_names_out()

    # Initialize appropriate explainer
    if model_type.lower() == 'xgb':
        explainer = shap.TreeExplainer(model)
    else:  # Fallback to generic explainer
        explainer = shap.Explainer(model)
    
    shap_values = explainer(processed_data)
    return shap_values, processed_data, feature_names

def save_top_features(shap_values: shap.Explanation, feature_names: list, output_path: str):
    """Save top 10 important features to CSV."""
    shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
    top_features = shap_df.abs().mean().sort_values(ascending=False).head(10).index
    shap_df[top_features].to_csv(output_path, index=False)