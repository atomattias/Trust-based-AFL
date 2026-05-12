"""
Ablation study script for multi-signal trust fusion.

Tests different combinations of trust signals:
1. Accuracy only
2. Accuracy + Stability
3. Accuracy + Stability + Drift
4. Full multi-signal (all signals)
"""

import os
import sys
import json
import subprocess
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from experiment import ExperimentRunner


def run_ablation_experiment(
    variant_name: str,
    lambda_weights: Dict[str, float],
    data_dir: str,
    test_csv: str = None,
    num_rounds: int = 10,
    num_trials: int = 3,
    model_type: str = 'logistic_regression'
) -> Dict[str, float]:
    """
    Run a single ablation experiment with specified lambda weights.
    
    Returns:
        Dictionary with results (accuracy, f1_score, etc.)
    """
    print(f"\n{'='*60}")
    print(f"Running: {variant_name}")
    print(f"Lambda weights: {lambda_weights}")
    print(f"{'='*60}\n")
    
    # Run experiment with multiple trials
    all_results = []
    
    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}")
        
        # Create runner
        runner = ExperimentRunner(
            data_dir=data_dir,
            model_type=model_type,
            random_state=42 + trial,
            test_csv=test_csv,
            num_rounds=num_rounds,
            trust_alpha=0.5,
            trust_storage_dir=None,
            use_multi_signal=True
        )
        
        # Override lambda weights in trust manager after initialization
        if hasattr(runner, 'trust_manager') and runner.trust_manager is not None:
            runner.trust_manager.lambda_weights = {
                'lambda1': lambda_weights['lambda1'],
                'lambda2': lambda_weights['lambda2'],
                'lambda3': lambda_weights['lambda3'],
                'lambda4': lambda_weights['lambda4']
            }
        
        # Run experiment
        results = runner.run_experiment(num_clients=None)
        
        if 'trust_aware' in results:
            all_results.append(results['trust_aware'])
    
    # Compute statistics
    if not all_results:
        return {}
    
    metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'false_negative_rate']
    stats = {}
    
    for metric in metrics:
        values = [r.get(metric, 0) for r in all_results if metric in r]
        if values:
            stats[metric] = {
                'mean': sum(values) / len(values),
                'std': pd.Series(values).std() if len(values) > 1 else 0.0,
                'values': values
            }
    
    # Print summary
    print(f"\n{variant_name} Results:")
    print(f"  Accuracy: {stats.get('accuracy', {}).get('mean', 0):.4f} ± {stats.get('accuracy', {}).get('std', 0):.4f}")
    print(f"  F1-Score: {stats.get('f1_score', {}).get('mean', 0):.4f} ± {stats.get('f1_score', {}).get('std', 0):.4f}")
    
    return stats


def main():
    """Run ablation study."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ablation study for multi-signal trust fusion')
    parser.add_argument('--data-dir', type=str, default='data/CSVs',
                       help='Directory containing CSV files')
    parser.add_argument('--test-csv', type=str, default=None,
                       help='Path to test CSV file')
    parser.add_argument('--num-rounds', type=int, default=10,
                       help='Number of federated learning rounds')
    parser.add_argument('--num-trials', type=int, default=3,
                       help='Number of trials per variant')
    parser.add_argument('--model-type', type=str, default='logistic_regression',
                       choices=['random_forest', 'logistic_regression'],
                       help='Model type')
    parser.add_argument('--dataset', type=str, default='both',
                       choices=['ctu13', 'honeypot', 'both'],
                       help='Which dataset(s) to test')
    
    args = parser.parse_args()
    
    # Define ablation variants
    variants = [
        {
            'name': 'Accuracy only',
            'lambda': {'lambda1': 1.0, 'lambda2': 0.0, 'lambda3': 0.0, 'lambda4': 0.0}
        },
        {
            'name': 'Accuracy + Stability',
            'lambda': {'lambda1': 1.0, 'lambda2': 0.3, 'lambda3': 0.0, 'lambda4': 0.0}
        },
        {
            'name': 'Accuracy + Stability + Drift',
            'lambda': {'lambda1': 1.0, 'lambda2': 0.3, 'lambda3': 0.2, 'lambda4': 0.0}
        },
        {
            'name': 'Full (All signals)',
            'lambda': {'lambda1': 1.0, 'lambda2': 0.3, 'lambda3': 0.2, 'lambda4': 0.2}
        }
    ]
    
    # Determine datasets
    datasets = []
    if args.dataset in ['ctu13', 'both']:
        # CTU-13 dataset
        ctu13_dir = 'data/CSVs/ctu13_heterogeneous'
        ctu13_test = 'data/CSVs/ctu13_test_set.csv'
        if os.path.exists(ctu13_dir):
            datasets.append(('CTU-13', ctu13_dir, ctu13_test))
    
    if args.dataset in ['honeypot', 'both']:
        # Honeypot dataset
        honeypot_dir = 'data/CSVs/honeypot_heterogeneous'
        honeypot_test = 'data/CSVs/heterogeneous_test_set.csv'
        if os.path.exists(honeypot_dir):
            datasets.append(('Honeypot', honeypot_dir, honeypot_test))
    
    if not datasets:
        print("Error: No datasets found. Please check data directories.")
        return
    
    # Run ablation study for each dataset
    all_results = {}
    
    for dataset_name, data_dir, test_csv in datasets:
        print(f"\n{'='*80}")
        print(f"ABLATION STUDY: {dataset_name}")
        print(f"{'='*80}\n")
        
        dataset_results = {}
        
        for variant in variants:
            results = run_ablation_experiment(
                variant_name=f"{dataset_name} - {variant['name']}",
                lambda_weights=variant['lambda'],
                data_dir=data_dir,
                test_csv=test_csv if os.path.exists(test_csv) else None,
                num_rounds=args.num_rounds,
                num_trials=args.num_trials,
                model_type=args.model_type
            )
            
            dataset_results[variant['name']] = results
        
        all_results[dataset_name] = dataset_results
    
    # Print summary table
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    
    for dataset_name, dataset_results in all_results.items():
        print(f"\n{dataset_name}:")
        print(f"{'Variant':<40} {'Accuracy':<15} {'F1-Score':<15}")
        print("-" * 70)
        
        for variant_name, results in dataset_results.items():
            acc_mean = results.get('accuracy', {}).get('mean', 0)
            acc_std = results.get('accuracy', {}).get('std', 0)
            f1_mean = results.get('f1_score', {}).get('mean', 0)
            f1_std = results.get('f1_score', {}).get('std', 0)
            
            print(f"{variant_name:<40} {acc_mean:.4f}±{acc_std:.4f}  {f1_mean:.4f}±{f1_std:.4f}")
    
    # Save results
    os.makedirs('results/reports', exist_ok=True)
    output_file = 'results/reports/ablation_study_results.json'
    
    # Convert to JSON-serializable format
    json_results = {}
    for dataset_name, dataset_results in all_results.items():
        json_results[dataset_name] = {}
        for variant_name, results in dataset_results.items():
            json_results[dataset_name][variant_name] = {
                metric: {
                    'mean': stats['mean'],
                    'std': stats['std']
                }
                for metric, stats in results.items()
            }
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")


if __name__ == '__main__':
    main()
