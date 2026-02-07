#!/usr/bin/env python3
"""
Quick test script to verify all imports work correctly.
Run this before running the main experiment.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test all module imports."""
    print("Testing imports...")
    
    try:
        from preprocessing import load_client_data, prepare_labels, split_data, prepare_features
        print("✓ preprocessing module imported")
    except Exception as e:
        print(f"✗ preprocessing module failed: {e}")
        return False
    
    try:
        from local_training import train_local_model, evaluate_model, get_model_parameters
        print("✓ local_training module imported")
    except Exception as e:
        print(f"✗ local_training module failed: {e}")
        return False
    
    try:
        from federated_client import FederatedClient
        print("✓ federated_client module imported")
    except Exception as e:
        print(f"✗ federated_client module failed: {e}")
        return False
    
    try:
        from federated_server import FedAvgAggregator, TrustAwareAggregator, EnsembleAggregator
        print("✓ federated_server module imported")
    except Exception as e:
        print(f"✗ federated_server module failed: {e}")
        return False
    
    try:
        from evaluation import compute_metrics, evaluate_model_on_test, compare_approaches
        print("✓ evaluation module imported")
    except Exception as e:
        print(f"✗ evaluation module failed: {e}")
        return False
    
    try:
        from visualization import plot_trust_distribution, plot_performance_comparison
        print("✓ visualization module imported")
    except Exception as e:
        print(f"✗ visualization module failed: {e}")
        return False
    
    print("\nAll imports successful! ✓")
    return True

if __name__ == '__main__':
    success = test_imports()
    sys.exit(0 if success else 1)
