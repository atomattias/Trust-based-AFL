#!/usr/bin/env python3
"""Analyze consistency by reading results from JSON files."""

import json
import os
import statistics
from pathlib import Path

def read_result_file(filepath):
    """Read accuracy from result JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            if 'trust_aware' in data and 'accuracy' in data['trust_aware']:
                return data['trust_aware']['accuracy']
    except:
        pass
    return None

def main():
    results_dir = Path('results/reports')
    
    # Find all result files
    result_files = list(results_dir.glob('experiment_results*.json'))
    
    if not result_files:
        print("No result files found. Run experiments first.")
        return
    
    print("="*60)
    print("CONSISTENCY ANALYSIS FROM SAVED RESULTS")
    print("="*60)
    print(f"\nFound {len(result_files)} result files")
    
    accuracies = []
    for f in result_files:
        acc = read_result_file(f)
        if acc is not None:
            accuracies.append(acc)
    
    if accuracies:
        print(f"\nTrust-Aware Accuracy Results ({len(accuracies)} runs):")
        print(f"  Results: {[f'{x:.4f}' for x in accuracies]}")
        print(f"  Mean: {statistics.mean(accuracies):.4f}")
        if len(accuracies) > 1:
            print(f"  Std:  {statistics.stdev(accuracies):.4f}")
        print(f"  Min:  {min(accuracies):.4f}")
        print(f"  Max:  {max(accuracies):.4f}")
        print(f"  Range: {max(accuracies) - min(accuracies):.4f}")
        
        # Categorize results
        high_perf = [a for a in accuracies if a >= 0.70]
        medium_perf = [a for a in accuracies if 0.50 <= a < 0.70]
        low_perf = [a for a in accuracies if a < 0.50]
        
        print(f"\nPerformance Categories:")
        print(f"  High (≥0.70): {len(high_perf)} runs - {[f'{x:.4f}' for x in high_perf]}")
        print(f"  Medium (0.50-0.70): {len(medium_perf)} runs - {[f'{x:.4f}' for x in medium_perf]}")
        print(f"  Low (<0.50): {len(low_perf)} runs - {[f'{x:.4f}' for x in low_perf]}")
    else:
        print("No valid results found.")

if __name__ == '__main__':
    main()
