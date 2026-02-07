"""
Configuration loader for trust parameters.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


def load_trust_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load trust configuration from JSON file.
    
    Args:
        config_path: Path to config file (default: config/trust_config.json)
        
    Returns:
        Dictionary with configuration parameters
    """
    if config_path is None:
        # Default path
        project_root = Path(__file__).parent.parent
        config_path = project_root / 'config' / 'trust_config.json'
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        # Return default configuration
        return get_default_config()
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Warning: Failed to load config from {config_path}: {e}")
        print("Using default configuration")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """
    Get default trust configuration.
    
    Returns:
        Dictionary with default configuration
    """
    return {
        "trust_manager": {
            "alpha": 0.7,
            "decay_rate": 0.95,
            "anomaly_threshold": 0.2,
            "initial_trust": 0.5,
            "storage_dir": "results/trust_history"
        },
        "consistency": {
            "window_size": 5
        },
        "trend_analysis": {
            "window_size": 5,
            "improving_threshold": 0.01,
            "declining_threshold": -0.01
        },
        "logging": {
            "level": "INFO",
            "log_file": "results/trust_updates.log"
        }
    }


def apply_trust_config(trust_manager: Any, config: Dict[str, Any]) -> None:
    """
    Apply configuration to TrustManager instance.
    
    Args:
        trust_manager: TrustManager instance
        config: Configuration dictionary
    """
    trust_config = config.get('trust_manager', {})
    
    if 'alpha' in trust_config:
        trust_manager.alpha = trust_config['alpha']
    if 'decay_rate' in trust_config:
        trust_manager.decay_rate = trust_config['decay_rate']
    if 'anomaly_threshold' in trust_config:
        trust_manager.anomaly_threshold = trust_config['anomaly_threshold']
    if 'initial_trust' in trust_config:
        trust_manager.initial_trust = trust_config['initial_trust']
    if 'storage_dir' in trust_config:
        trust_manager.storage_dir = trust_config['storage_dir']
        if trust_manager.storage_dir:
            os.makedirs(trust_manager.storage_dir, exist_ok=True)
