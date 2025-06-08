import os
import json
import joblib
from datetime import datetime

import numpy as np
import pandas as pd

def save_model_artifacts(model, preprocessor, features, metrics, save_dir="models"):
    """Save model artifacts with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(save_dir, f"model_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    # Ensure JSON-serializable formats
    safe_metrics = {
        k: float(v) if isinstance(v, (np.floating, np.ndarray)) else v
        for k, v in metrics.items()
    }
    safe_features = list(features) if isinstance(features, (np.ndarray, pd.Index)) else features

    artifacts = {
        'model': model,
        'preprocessor': preprocessor,
        'metadata': {
            'timestamp': timestamp,
            'metrics': safe_metrics,
            'features': safe_features
        }
    }

    joblib.dump(artifacts, os.path.join(model_dir, "model.pkl"))
    with open(os.path.join(model_dir, "metadata.json"), 'w') as f:
        json.dump(artifacts['metadata'], f)

    return model_dir
