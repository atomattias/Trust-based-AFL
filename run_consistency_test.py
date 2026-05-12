#!/usr/bin/env python3
"""Run multiple experiments to test consistency of trust methods."""

import subprocess
import re
import json
import statistics

def extract_accuracy(output):
    """Extract Trust-Aware accuracy from experiment output."""
    # Look for the results comparison table
    lines = output.split('\n')
    for i, line in enumerate(lines):
        if 'RESULTS COMPARISON' in line or 'Metric' in line and 'Centralized' in line:
            # Look for Trust-Aware line in the next few lines
            for j in range(i+1, min(i+15, len(lines))):
                if 'Trust-Aware' in lines[j]:
                    # Extract all numbers from the line
                    numbers = re.findall(r'\d+\.\d+', lines[j])
                    if len(numbers) >= 3:
                        # Third number is Trust-Aware accuracy
                        return float(numbers[2])
                    elif len(numbers) == 1:
                        # Sometimes only one number
                        return float(numbers[0])
    # Alternative: look for "Test Accuracy:" in Trust-Aware section
    for i, line in enumerate(lines):
        if 'Approach 3' in line or 'Trust-Aware' in line:
            for j in range(i, min(i+50, len(lines))):
                if 'Test Accuracy:' in lines[j]:
                    match = re.search(r'Test Accuracy:\s+(\d+\.\d+)', lines[j])
                    if match:
                        return float(match.group(1))
    return None

def run_experiment(seed, use_multi_signal=False):
    """Run a single experiment."""
    cmd = [
        'python3', 'experiment.py',
        '--data-dir', 'data/CSVs/ctu13_heterogeneous',
        '--test-csv', 'data/CSVs/ctu13_test_set.csv',
        '--model-type', 'logistic_regression',
        '--num-rounds', '10',
        '--trust-alpha', '0.5',
        '--random-state', str(seed)
    ]
    if use_multi_signal:
        cmd.append('--multi-signal-trust')
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )
        accuracy = extract_accuracy(result.stdout)
        return accuracy
    except subprocess.TimeoutExpired:
        print(f"  Timeout for seed {seed}")
        return None
    except Exception as e:
        print(f"  Error for seed {seed}: {e}")
        return None

def main():
    seeds = [42, 123, 456, 789, 999]
    
    print("="*60)
    print("CONSISTENCY TEST: Simple Trust vs Multi-Signal Trust")
    print("="*60)
    
    # Test Simple Trust
    print("\n=== Simple Trust (5 runs) ===")
    simple_results = []
    for seed in seeds:
        print(f"Running seed {seed}...", end=' ', flush=True)
        acc = run_experiment(seed, use_multi_signal=False)
        if acc is not None:
            simple_results.append(acc)
            print(f"Accuracy: {acc:.4f}")
        else:
            print("Failed")
    
    # Test Multi-Signal Trust
    print("\n=== Multi-Signal Trust (5 runs) ===")
    multi_results = []
    for seed in seeds:
        print(f"Running seed {seed}...", end=' ', flush=True)
        acc = run_experiment(seed, use_multi_signal=True)
        if acc is not None:
            multi_results.append(acc)
            print(f"Accuracy: {acc:.4f}")
        else:
            print("Failed")
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    if simple_results:
        print(f"\nSimple Trust:")
        print(f"  Results: {[f'{x:.4f}' for x in simple_results]}")
        print(f"  Mean: {statistics.mean(simple_results):.4f}")
        print(f"  Std:  {statistics.stdev(simple_results):.4f}")
        print(f"  Min:  {min(simple_results):.4f}")
        print(f"  Max:  {max(simple_results):.4f}")
        print(f"  Range: {max(simple_results) - min(simple_results):.4f}")
    
    if multi_results:
        print(f"\nMulti-Signal Trust:")
        print(f"  Results: {[f'{x:.4f}' for x in multi_results]}")
        print(f"  Mean: {statistics.mean(multi_results):.4f}")
        print(f"  Std:  {statistics.stdev(multi_results):.4f}")
        print(f"  Min:  {min(multi_results):.4f}")
        print(f"  Max:  {max(multi_results):.4f}")
        print(f"  Range: {max(multi_results) - min(multi_results):.4f}")
    
    if simple_results and multi_results:
        print(f"\nComparison:")
        print(f"  Simple Trust Mean: {statistics.mean(simple_results):.4f}")
        print(f"  Multi-Signal Mean: {statistics.mean(multi_results):.4f}")
        print(f"  Difference: {statistics.mean(multi_results) - statistics.mean(simple_results):.4f}")
        print(f"  Simple Trust Std: {statistics.stdev(simple_results):.4f}")
        print(f"  Multi-Signal Std: {statistics.stdev(multi_results):.4f}")
        print(f"  Consistency: {'Multi-Signal' if statistics.stdev(multi_results) < statistics.stdev(simple_results) else 'Simple'} is more consistent")

if __name__ == '__main__':
    main()
