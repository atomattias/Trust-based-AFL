#!/usr/bin/env python3
"""
Create confusion matrix and ROC curve figures from existing experiment results.
This script loads results from JSON files and generates the plots.
"""

import json
import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.visualization import plot_confusion_matrices, plot_roc_curves
from experiment import ExperimentRunner

def create_figures_for_dataset(data_dir, test_csv, dataset_name, output_dir='results/plots/Figures'):
    """Create figures for a dataset by running experiment and saving plots."""
    print(f"\n{'='*60}")
    print(f"Creating figures for {dataset_name} dataset")
    print(f"{'='*60}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear trust history to get fresh results
    import subprocess
    subprocess.run(['rm', '-rf', 'results/trust_history/*.json'], 
                   stderr=subprocess.DEVNULL, shell=True)
    
    # Run experiment
    print(f"Running experiment (this may take a few minutes)...")
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
    
    # Prepare results with proper names
    plot_results = {
        'Centralized': results['centralized'],
        'FedAvg': results['federated_equal_weight'],
        'TrustFed-Honeypot': results['trust_aware']
    }
    
    # Generate confusion matrix
    print(f"\nGenerating confusion matrix plot...")
    cm_path = os.path.join(output_dir, f'{dataset_name.lower()}_confusion_matrices.png')
    try:
        plot_confusion_matrices(plot_results, save_path=cm_path)
        if os.path.exists(cm_path):
            print(f"✓ Successfully created: {cm_path}")
        else:
            print(f"✗ Failed to create: {cm_path}")
    except Exception as e:
        print(f"✗ Error creating confusion matrix: {e}")
    
    # Generate ROC curve
    print(f"\nGenerating ROC curve plot...")
    y_true = None
    for result in plot_results.values():
        if 'y_true' in result:
            y_true = np.array(result['y_true'])
            break
    
    if y_true is not None:
        roc_path = os.path.join(output_dir, f'{dataset_name.lower()}_roc_curves.png')
        try:
            plot_roc_curves(plot_results, y_true=y_true,
                           save_path=roc_path,
                           title=f'ROC Curves - {dataset_name.upper()} Dataset')
            if os.path.exists(roc_path):
                print(f"✓ Successfully created: {roc_path}")
            else:
                print(f"✗ Failed to create: {roc_path}")
        except Exception as e:
            print(f"✗ Error creating ROC curve: {e}")
    else:
        print("⚠ Warning: y_true not found, cannot generate ROC curve")
    
    return results

def main():
    """Generate all figures."""
    print("="*60)
    print("Generating Confusion Matrix and ROC Curve Figures")
    print("="*60)
    
    # CTU-13
    print("\n" + "="*60)
    print("CTU-13 Dataset")
    print("="*60)
    ctu13_results = create_figures_for_dataset(
        data_dir='data/CSVs/ctu13_heterogeneous',
        test_csv='data/CSVs/ctu13_test_set.csv',
        dataset_name='ctu13'
    )
    
    # Honeypot
    print("\n" + "="*60)
    print("Honeypot Dataset")
    print("="*60)
    honeypot_results = create_figures_for_dataset(
        data_dir='data/CSVs/honeypot_heterogeneous',
        test_csv='data/CSVs/heterogeneous_test_set.csv',
        dataset_name='honeypot'
    )
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("\nGenerated figures:")
    figure_files = [
        'results/plots/Figures/ctu13_confusion_matrices.png',
        'results/plots/Figures/ctu13_roc_curves.png',
        'results/plots/Figures/honeypot_confusion_matrices.png',
        'results/plots/Figures/honeypot_roc_curves.png'
    ]
    
    for fig_file in figure_files:
        if os.path.exists(fig_file):
            size = os.path.getsize(fig_file) / 1024  # KB
            print(f"  ✓ {fig_file} ({size:.1f} KB)")
        else:
            print(f"  ✗ {fig_file} (not found)")

if __name__ == '__main__':
    main()
