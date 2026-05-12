"""
Beta (β) comparison script for trust exponent.

Tests different values of β (trust exponent) to empirically justify β=0.8.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from experiment import ExperimentRunner
from federated_server import TrustAwareAggregator, EnsembleAggregator


def run_beta_experiment(
    beta: float,
    data_dir: str,
    test_csv: str = None,
    num_rounds: int = 10,
    num_trials: int = 3,
    model_type: str = 'logistic_regression'
) -> Dict[str, float]:
    """
    Run experiment with specified beta value.
    
    Returns:
        Dictionary with results (accuracy, f1_score, etc.)
    """
    print(f"\n{'='*60}")
    print(f"Testing β = {beta}")
    print(f"{'='*60}\n")
    
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
        
        # Patch both TrustAwareAggregator and EnsembleAggregator to use custom beta
        # Store original __init__ methods
        original_trust_init = TrustAwareAggregator.__init__
        original_ensemble_init = EnsembleAggregator.__init__
        
        def patched_trust_init(self, *args, **kwargs):
            # Override beta with the captured beta value
            kwargs['beta'] = beta
            original_trust_init(self, *args, **kwargs)
        
        def patched_ensemble_init(self, *args, **kwargs):
            # Override beta with the captured beta value
            kwargs['beta'] = beta
            original_ensemble_init(self, *args, **kwargs)
        
        # Temporarily replace __init__ methods
        TrustAwareAggregator.__init__ = patched_trust_init
        EnsembleAggregator.__init__ = patched_ensemble_init
        
        try:
        
            # Run experiment
            results = runner.run_experiment(num_clients=None)
            
            if 'trust_aware' in results:
                all_results.append(results['trust_aware'])
        finally:
            # Restore original __init__ methods
            TrustAwareAggregator.__init__ = original_trust_init
            EnsembleAggregator.__init__ = original_ensemble_init
    
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
    print(f"\nβ = {beta} Results:")
    print(f"  Accuracy: {stats.get('accuracy', {}).get('mean', 0):.4f} ± {stats.get('accuracy', {}).get('std', 0):.4f}")
    print(f"  F1-Score: {stats.get('f1_score', {}).get('mean', 0):.4f} ± {stats.get('f1_score', {}).get('std', 0):.4f}")
    
    return stats


def main():
    """Run beta comparison study."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Beta (β) comparison for trust exponent')
    parser.add_argument('--data-dir', type=str, default='data/CSVs/ctu13_heterogeneous',
                       help='Directory containing CSV files (default: CTU-13)')
    parser.add_argument('--test-csv', type=str, default='data/CSVs/heterogeneous_test_set.csv',
                       help='Path to test CSV file')
    parser.add_argument('--num-rounds', type=int, default=10,
                       help='Number of federated learning rounds')
    parser.add_argument('--num-trials', type=int, default=3,
                       help='Number of trials per beta value')
    parser.add_argument('--model-type', type=str, default='logistic_regression',
                       choices=['random_forest', 'logistic_regression'],
                       help='Model type')
    parser.add_argument('--beta-values', type=float, nargs='+', default=[0.5, 0.8, 1.0, 1.2],
                       help='Beta values to test (default: 0.5 0.8 1.0 1.2)')
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        print("Available directories:")
        data_root = Path('data/CSVs')
        if data_root.exists():
            for d in sorted(data_root.iterdir()):
                if d.is_dir():
                    print(f"  - {d}")
        return
    
    # Check if test CSV exists
    if args.test_csv and not os.path.exists(args.test_csv):
        print(f"Warning: Test CSV not found: {args.test_csv}")
        args.test_csv = None
    
    print(f"\n{'='*80}")
    print(f"BETA (β) COMPARISON STUDY")
    print(f"{'='*80}")
    print(f"\nDataset: {args.data_dir}")
    print(f"Beta values to test: {args.beta_values}")
    print(f"Number of trials per beta: {args.num_trials}")
    print(f"Number of rounds: {args.num_rounds}\n")
    
    # Run experiments for each beta value
    all_results = {}
    
    for beta in args.beta_values:
        results = run_beta_experiment(
            beta=beta,
            data_dir=args.data_dir,
            test_csv=args.test_csv,
            num_rounds=args.num_rounds,
            num_trials=args.num_trials,
            model_type=args.model_type
        )
        all_results[beta] = results
    
    # Print summary table
    print("\n" + "="*80)
    print("BETA COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n{'β':<10} {'Accuracy':<20} {'F1-Score':<20} {'Description':<30}")
    print("-" * 80)
    
    descriptions = {
        0.5: "Strong compression",
        0.8: "Optimal (balanced)",
        1.0: "Linear weighting",
        1.2: "Convex amplification"
    }
    
    for beta in sorted(all_results.keys()):
        results = all_results[beta]
        acc_mean = results.get('accuracy', {}).get('mean', 0)
        acc_std = results.get('accuracy', {}).get('std', 0)
        f1_mean = results.get('f1_score', {}).get('mean', 0)
        f1_std = results.get('f1_score', {}).get('std', 0)
        desc = descriptions.get(beta, "")
        
        print(f"{beta:<10.1f} {acc_mean:.4f}±{acc_std:.4f}    {f1_mean:.4f}±{f1_std:.4f}    {desc}")
    
    # Find best beta
    best_beta = max(all_results.keys(), 
                   key=lambda b: all_results[b].get('f1_score', {}).get('mean', 0))
    best_f1 = all_results[best_beta].get('f1_score', {}).get('mean', 0)
    
    print(f"\nBest β: {best_beta} (F1-Score: {best_f1:.4f})")
    
    # Save results
    os.makedirs('results/reports', exist_ok=True)
    output_file = 'results/reports/beta_comparison_results.json'
    
    # Convert to JSON-serializable format
    json_results = {
        'beta_values': list(all_results.keys()),
        'results': {
            str(beta): {
                metric: {
                    'mean': stats['mean'],
                    'std': stats['std']
                }
                for metric, stats in results.items()
            }
            for beta, results in all_results.items()
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")


if __name__ == '__main__':
    main()
