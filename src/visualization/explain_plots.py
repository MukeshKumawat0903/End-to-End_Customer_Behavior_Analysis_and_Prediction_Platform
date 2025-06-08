import shap
import matplotlib.pyplot as plt

def plot_shap_global(shap_values: shap.Explanation, feature_names: list, plot_type: str = 'bar', output_path: str = None):
    """Generate global SHAP explanation plots."""
    plt.figure(figsize=(12, 6))
    
    if plot_type == 'bar':
        shap.plots.bar(shap_values, show=False)
        plt.title("Global Feature Importance (SHAP)")
    elif plot_type == 'dot':
        shap.summary_plot(shap_values.values, features=shap_values.data, 
                         feature_names=feature_names, show=False)
        plt.title("SHAP Summary (Dot Plot)")
    
    plt.tight_layout()

    return plt

def save_force_plot(shap_values: shap.Explanation, sample_idx: int, feature_names: list, output_path: str):
    """Save interactive force plot for individual prediction."""
    sample_explanation = shap_values[sample_idx]
    
    force_plot = shap.force_plot(
        shap_values.base_values,
        sample_explanation.values,
        features=shap_values.data[sample_idx],
        feature_names=feature_names
    )
 
    return force_plot