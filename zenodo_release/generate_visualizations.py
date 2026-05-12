#!/usr/bin/env python3
"""
Generate confusion matrix and ROC curve plots for the paper.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from experiment import ExperimentRunner
from src.visualization import plot_confusion_matrices, plot_roc_curves

def generate_plots_for_dataset(data_dir, test_csv, dataset_name, output_dir='results/plots/Figures'):
    """Generate confusion matrices and ROC curves for a dataset."""
    print(f"\n{'='*60}")
    print(f"Generating visualizations for {dataset_name}")
    print(f"{'='*60}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run experiment (single trial with seed 42 for consistency)
    print(f"Running experiment for {dataset_name}...")
    runner = ExperimentRunner(
        data_dir=data_dir,
        model_type='logistic_regression',
        random_state=42,
        test_csv=test_csv,
        num_rounds=10,
        trust_alpha=0.5,
        use_multi_signal=True
    )
    
    results = runner.run_experiment(num_clients=None)
    
    # Prepare results for plotting
    plot_results = {
        'Centralized': results['centralized'],
        'FedAvg': results['federated_equal_weight'],
        'TrustFed-Honeypot': results['trust_aware']
    }
    
    # Generate confusion matrix plot
    print(f"\nGenerating confusion matrix plot...")
    cm_path = os.path.join(output_dir, f'{dataset_name.lower()}_confusion_matrices.png')
    plot_confusion_matrices(plot_results, save_path=cm_path)
    print(f"✓ Saved: {cm_path}")
    
    # Generate ROC curve plot
    print(f"\nGenerating ROC curve plot...")
    # Extract y_true from results
    y_true = None
    for result in plot_results.values():
        if 'y_true' in result:
            y_true = np.array(result['y_true'])
            break
    
    if y_true is not None:
        roc_path = os.path.join(output_dir, f'{dataset_name.lower()}_roc_curves.png')
        plot_roc_curves(plot_results, y_true=y_true,
                       save_path=roc_path,
                       title=f'ROC Curves - {dataset_name} Dataset')
        print(f"✓ Saved: {roc_path}")
    else:
        print("⚠ Warning: y_true not found in results, skipping ROC curve")
    
    return results

def main():
    """Generate all visualizations."""
    print("="*60)
    print("Generating Confusion Matrix and ROC Curve Plots")
    print("="*60)
    
    # CTU-13 Dataset
    ctu13_results = generate_plots_for_dataset(
        data_dir='data/CSVs/ctu13_heterogeneous',
        test_csv='data/CSVs/ctu13_test_set.csv',
        dataset_name='ctu13'
    )
    
    # Honeypot Dataset
    honeypot_results = generate_plots_for_dataset(
        data_dir='data/CSVs/honeypot_heterogeneous',
        test_csv='data/CSVs/heterogeneous_test_set.csv',
        dataset_name='honeypot'
    )
    
    print("\n" + "="*60)
    print("All visualizations generated successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  - results/plots/Figures/ctu13_confusion_matrices.png")
    print("  - results/plots/Figures/ctu13_roc_curves.png")
    print("  - results/plots/Figures/honeypot_confusion_matrices.png")
    print("  - results/plots/Figures/honeypot_roc_curves.png")

if __name__ == '__main__':
    main()
