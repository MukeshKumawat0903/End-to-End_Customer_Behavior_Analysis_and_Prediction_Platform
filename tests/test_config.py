"""
Unit tests for configuration module.
"""
import pytest
from pathlib import Path
from src.config import load_config, get_config, Config


def test_load_config():
    """Test configuration loading."""
    config = load_config()
    assert isinstance(config, Config)
    assert config.data.test_size > 0
    assert config.data.test_size < 1


def test_config_validation():
    """Test configuration validation."""
    config = load_config()
    
    # Test data config
    assert 0 < config.data.test_size < 1
    assert 0 < config.data.validation_size < 1
    
    # Test features config
    assert len(config.features.numerical) > 0
    assert len(config.features.categorical) > 0
    assert config.features.target == "target"
    
    # Test model config
    assert config.model.type in ["xgboost", "lightgbm"]
    assert config.model.device in ["cpu", "cuda"]
    
    # Test CV config
    assert config.cross_validation.n_splits > 1
    assert isinstance(config.cross_validation.stratified, bool)


def test_get_config_singleton():
    """Test that get_config returns the same instance."""
    config1 = get_config()
    config2 = get_config()
    assert config1 is config2


def test_config_reload():
    """Test configuration reload."""
    config1 = get_config()
    config2 = get_config(reload=True)
    assert isinstance(config2, Config)
