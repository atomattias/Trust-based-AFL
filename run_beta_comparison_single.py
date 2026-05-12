"""
Beta (β) comparison script for trust exponent - SINGLE RUN VERSION.

Tests different values of β (trust exponent) with single runs (seed=42) to match paper methodology.
Paper states: "Random seed is set to 42 for reproducibility. While repeated trials were not performed"
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from experiment import ExperimentRunner
from federated_server import TrustAwareAggregator, EnsembleAggregator


def run_beta_experiment_single(
    beta: float,
    data_dir: str,
    test_csv: str = None,
    num_rounds: int = 10,
    model_type: str = 'logistic_regression',
    random_seed: int = 42
) -> Dict[str, float]:
    """
    Run experiment with specified beta value (single run, matching paper).
    
    Returns:
        Dictionary with results (accuracy, f1_score, etc.)
    """
    print(f"\n{'='*60}")
    print(f"Testing β = {beta} (seed={random_seed}, single run)")
    print(f"{'='*60}\n")
    
    # Create runner
    runner = ExperimentRunner(
        data_dir=data_dir,
        model_type=model_type,
        random_state=random_seed,
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
        # Run experiment (single run)
        results = runner.run_experiment(num_clients=None)
        
        if 'trust_aware' in results:
            return results['trust_aware']
        else:
            return {}
    finally:
        # Restore original __init__ methods
        TrustAwareAggregator.__init__ = original_trust_init
        EnsembleAggregator.__init__ = original_ensemble_init


def main():
    """Run beta comparison study (single runs, matching paper methodology)."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Beta (β) comparison for trust exponent (single runs)')
    parser.add_argument('--data-dir', type=str, default='data/CSVs/ctu13_heterogeneous',
                       help='Directory containing CSV files (default: CTU-13)')
    parser.add_argument('--test-csv', type=str, default='data/CSVs/ctu13_test_set.csv',
                       help='Path to test CSV file')
    parser.add_argument('--num-rounds', type=int, default=10,
                       help='Number of federated learning rounds')
    parser.add_argument('--model-type', type=str, default='logistic_regression',
                       choices=['random_forest', 'logistic_regression'],
                       help='Model type')
    parser.add_argument('--beta-values', type=float, nargs='+', default=[0.5, 0.8, 1.0, 1.2],
                       help='Beta values to test (default: 0.5 0.8 1.0 1.2)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed (default: 42, matching paper)')
    
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
    print(f"BETA (β) COMPARISON STUDY - SINGLE RUNS (MATCHING PAPER)")
    print(f"{'='*80}")
    print(f"Dataset: {args.data_dir}")
    print(f"Test set: {args.test_csv}")
    print(f"Rounds: {args.num_rounds}")
    print(f"Model: {args.model_type}")
    print(f"Random seed: {args.random_seed} (matching paper)")
    print(f"Beta values: {args.beta_values}")
    print(f"\nNote: Using single runs (not averages) to match paper methodology")
    print(f"Paper states: 'Random seed is set to 42 for reproducibility.'")
    print(f"Paper states: 'While repeated trials were not performed'")
    print(f"{'='*80}\n")
    
    # Run experiments for each beta value
    all_results = {}
    
    for beta in args.beta_values:
        result = run_beta_experiment_single(
            beta=beta,
            data_dir=args.data_dir,
            test_csv=args.test_csv,
            num_rounds=args.num_rounds,
            model_type=args.model_type,
            random_seed=args.random_seed
        )
        
        if result:
            all_results[str(beta)] = result
            print(f"\nβ = {beta} Results:")
            print(f"  Accuracy: {result.get('accuracy', 0):.4f} ({result.get('accuracy', 0)*100:.2f}%)")
            print(f"  F1-Score: {result.get('f1_score', 0):.4f} ({result.get('f1_score', 0)*100:.2f}%)")
    
    # Save results
    output = {
        'beta_values': args.beta_values,
        'random_seed': args.random_seed,
        'methodology': 'single_run',
        'results': all_results
    }
    
    output_file = Path('results/reports/beta_comparison_single_run_results.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"BETA COMPARISON SUMMARY (SINGLE RUNS)")
    print(f"{'='*80}")
    print(f"{'β':<6} {'Accuracy':<20} {'F1-Score':<20} {'Description':<30}")
    print("-" * 80)
    
    descriptions = {
        0.5: "Strong compression",
        0.8: "Optimal (claimed)",
        1.0: "Linear weighting",
        1.2: "Convex amplification"
    }
    
    for beta in sorted(args.beta_values):
        if str(beta) in all_results:
            r = all_results[str(beta)]
            acc = r.get('accuracy', 0) * 100
            f1 = r.get('f1_score', 0) * 100
            desc = descriptions.get(beta, "")
            print(f"{beta:<6.1f} {acc:>6.2f}%{'':<11} {f1:>6.2f}%{'':<11} {desc}")
    
    # Find best beta
    if all_results:
        best_beta = max(args.beta_values, 
                       key=lambda b: all_results.get(str(b), {}).get('f1_score', 0))
        best_result = all_results[str(best_beta)]
        print(f"\nBest β: {best_beta} (F1-Score: {best_result.get('f1_score', 0)*100:.2f}%)")
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Compare with paper claim
    if '0.8' in all_results:
        paper_acc = 71.51
        paper_f1 = 83.17
        exp_acc = all_results['0.8'].get('accuracy', 0) * 100
        exp_f1 = all_results['0.8'].get('f1_score', 0) * 100
        
        print(f"\n{'='*80}")
        print(f"COMPARISON WITH PAPER CLAIM (β=0.8)")
        print(f"{'='*80}")
        print(f"Paper claim:  {paper_acc:.2f}% accuracy, {paper_f1:.2f}% F1")
        print(f"Experiment:   {exp_acc:.2f}% accuracy, {exp_f1:.2f}% F1")
        print(f"Gap:          {exp_acc - paper_acc:+.2f} pp accuracy, {exp_f1 - paper_f1:+.2f} pp F1")
        
        if abs(exp_acc - paper_acc) < 2:
            print(f"\n✅ VERY CLOSE! Within 2 pp of paper claim")
        elif abs(exp_acc - paper_acc) < 5:
            print(f"\n✅ Close! Within 5 pp of paper claim")
        else:
            print(f"\n⚠️  Still some gap - may need further investigation")


if __name__ == '__main__':
    main()
