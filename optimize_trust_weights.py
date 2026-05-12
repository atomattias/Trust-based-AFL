#!/usr/bin/env python3
"""Optimize trust weights to improve Trust-Aware performance."""

import subprocess
import re
import json

def extract_accuracy(output):
    """Extract Trust-Aware accuracy from output."""
    match = re.search(r'Trust-Aware\s+(\d+\.\d+)', output)
    if match:
        return float(match.group(1))
    return None

def test_config(lambda1, lambda2, lambda3, lambda4, beta=0.8):
    """Test a specific configuration."""
    # Update config file
    config = {
        "trust_manager": {
            "alpha": 0.5,
            "decay_rate": 0.95,
            "anomaly_threshold": 0.2,
            "initial_trust": 0.5,
            "storage_dir": "results/trust_history"
        },
        "multi_signal": {
            "lambda1": lambda1,
            "lambda2": lambda2,
            "lambda3": lambda3,
            "lambda4": lambda4
        }
    }
    
    with open('config/trust_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run experiment
    result = subprocess.run(
        ['python3', 'experiment.py',
         '--data-dir', 'data/CSVs/ctu13_heterogeneous',
         '--test-csv', 'data/CSVs/ctu13_test_set.csv',
         '--model-type', 'logistic_regression',
         '--num-rounds', '10',
         '--trust-alpha', '0.5',
         '--multi-signal-trust',
         '--random-state', '42'],
        capture_output=True,
        text=True,
        timeout=300
    )
    
    accuracy = extract_accuracy(result.stdout)
    return accuracy

def main():
    print("="*60)
    print("OPTIMIZING MULTI-SIGNAL TRUST FUSION WEIGHTS")
    print("="*60)
    
    # Test different configurations
    configs = [
        # Current baseline
        {"name": "Baseline", "lambda1": 1.0, "lambda2": 0.3, "lambda3": 0.2, "lambda4": 0.2},
        
        # Increase accuracy weight (most important)
        {"name": "High Accuracy Weight", "lambda1": 1.5, "lambda2": 0.2, "lambda3": 0.15, "lambda4": 0.15},
        {"name": "Very High Accuracy", "lambda1": 2.0, "lambda2": 0.15, "lambda3": 0.1, "lambda4": 0.1},
        
        # Increase stability weight
        {"name": "High Stability", "lambda1": 1.0, "lambda2": 0.5, "lambda3": 0.2, "lambda4": 0.2},
        
        # Increase uncertainty weight
        {"name": "High Uncertainty", "lambda1": 1.0, "lambda2": 0.2, "lambda3": 0.2, "lambda4": 0.5},
        
        # Balanced but higher total
        {"name": "Balanced High", "lambda1": 1.2, "lambda2": 0.4, "lambda3": 0.3, "lambda4": 0.3},
    ]
    
    results = []
    for config in configs:
        print(f"\nTesting: {config['name']}")
        print(f"  Weights: λ1={config['lambda1']}, λ2={config['lambda2']}, λ3={config['lambda3']}, λ4={config['lambda4']}")
        acc = test_config(config['lambda1'], config['lambda2'], config['lambda3'], config['lambda4'])
        if acc:
            results.append((config['name'], acc, config))
            print(f"  Accuracy: {acc:.4f}")
        else:
            print("  Failed to extract accuracy")
    
    # Find best configuration
    if results:
        results.sort(key=lambda x: x[1], reverse=True)
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        for name, acc, config in results:
            print(f"{name:25s}: {acc:.4f} (λ1={config['lambda1']}, λ2={config['lambda2']}, λ3={config['lambda3']}, λ4={config['lambda4']})")
        
        best_name, best_acc, best_config = results[0]
        print(f"\n✅ Best Configuration: {best_name}")
        print(f"   Accuracy: {best_acc:.4f}")
        print(f"   Weights: λ1={best_config['lambda1']}, λ2={best_config['lambda2']}, λ3={best_config['lambda3']}, λ4={best_config['lambda4']}")

if __name__ == '__main__':
    main()
