"""
Unit tests for model utilities.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import tempfile
import shutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.models.utils import save_model_artifacts


@pytest.fixture
def mock_model():
    """Create a real scikit-learn model for testing."""
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    # Fit with dummy data so it's a trained model
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    model.fit(X, y)
    return model


@pytest.fixture
def mock_preprocessor():
    """Create a real scikit-learn preprocessor for testing."""
    preprocessor = StandardScaler()
    # Fit with dummy data
    X = np.array([[1, 2], [3, 4], [5, 6]])
    preprocessor.fit(X)
    return preprocessor


@pytest.fixture
def temp_models_dir():
    """Create temporary directory for models."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_save_model_artifacts(mock_model, mock_preprocessor, temp_models_dir):
    """Test saving model artifacts."""
    features = ['feature1', 'feature2', 'feature3']
    metrics = {
        'accuracy': 0.85,
        'roc_auc': 0.90,
        'f1_score': 0.82
    }
    
    model_dir = save_model_artifacts(
        model=mock_model,
        preprocessor=mock_preprocessor,
        features=features,
        metrics=metrics,
        save_dir=temp_models_dir
    )
    
    # Check that directory was created
    assert Path(model_dir).exists()
    
    # Check that files exist
    assert (Path(model_dir) / "model.pkl").exists()
    assert (Path(model_dir) / "metadata.json").exists()


def test_save_model_artifacts_with_numpy_metrics(mock_model, mock_preprocessor, temp_models_dir):
    """Test saving with numpy types in metrics."""
    features = ['feature1', 'feature2']
    metrics = {
        'accuracy': np.float64(0.85),
        'roc_auc': np.float64(0.90),
        'f1_score': 0.82
    }
    
    model_dir = save_model_artifacts(
        model=mock_model,
        preprocessor=mock_preprocessor,
        features=features,
        metrics=metrics,
        save_dir=temp_models_dir
    )
    
    # Should not raise error
    assert Path(model_dir).exists()


def test_save_model_artifacts_with_index_features(mock_model, mock_preprocessor, temp_models_dir):
    """Test saving with pandas Index as features."""
    features = pd.Index(['feature1', 'feature2', 'feature3'])
    metrics = {'accuracy': 0.85}
    
    model_dir = save_model_artifacts(
        model=mock_model,
        preprocessor=mock_preprocessor,
        features=features,
        metrics=metrics,
        save_dir=temp_models_dir
    )
    
    assert Path(model_dir).exists()
