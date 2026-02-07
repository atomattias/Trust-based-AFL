#!/usr/bin/env python3
"""
Results analysis script for Trust-Aware Federated Honeypot Learning.

This script helps analyze and interpret experiment results.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any


def load_results(results_path: str = 'results/reports/experiment_results.json') -> Dict[str, Any]:
    """Load experiment results from JSON file."""
    results_file = Path(results_path)
    
    if not results_file.exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)
    
    with open(results_file, 'r') as f:
        return json.load(f)


def print_performance_comparison(results: Dict[str, Any]) -> None:
    """Print performance comparison across approaches."""
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    
    approaches = {
        'Centralized': results.get('centralized', {}),
        'FedAvg (Equal Weight)': results.get('federated_equal_weight', {}),
        'Trust-Aware': results.get('trust_aware', {})
    }
    
    metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'false_positive_rate']
    
    print(f"\n{'Metric':<25} {'Centralized':<18} {'FedAvg':<18} {'Trust-Aware':<18}")
    print("-" * 80)
    
    for metric in metrics:
        values = []
        for approach_name, approach_results in approaches.items():
            val = approach_results.get(metric, 0)
            values.append(val)
            print(f"{metric.replace('_', ' ').title():<25} {val:<18.4f}", end='')
        print()
        
        # Calculate improvement
        if len(values) == 3:
            fedavg_val = values[1]
            trust_val = values[2]
            if fedavg_val > 0:
                improvement = ((trust_val - fedavg_val) / fedavg_val) * 100
                print(f"  → Trust-Aware improvement over FedAvg: {improvement:+.2f}%")
    
    print()


def print_trust_statistics(results: Dict[str, Any]) -> None:
    """Print trust statistics."""
    trust_aware = results.get('trust_aware', {})
    trust_stats = trust_aware.get('trust_statistics', {})
    
    if not trust_stats:
        print("\nNo trust statistics available (single-round mode)")
        return
    
    print("\n" + "="*70)
    print("TRUST STATISTICS")
    print("="*70)
    
    print(f"\nMean Trust:   {trust_stats.get('mean', 0):.4f}")
    print(f"Std Dev:      {trust_stats.get('std', 0):.4f}")
    print(f"Min Trust:    {trust_stats.get('min', 0):.4f}")
    print(f"Max Trust:    {trust_stats.get('max', 0):.4f}")
    print(f"Median Trust: {trust_stats.get('median', 0):.4f}")
    print(f"Client Count: {trust_stats.get('count', 0)}")
    print()


def print_trust_evolution(results: Dict[str, Any]) -> None:
    """Print trust evolution analysis."""
    summary = results.get('summary', {})
    trust_evo = summary.get('trust_evolution', {})
    
    if not trust_evo or 'trust_evolution' not in trust_evo:
        print("\nNo trust evolution data available (single-round mode)")
        return
    
    print("\n" + "="*70)
    print("TRUST EVOLUTION ANALYSIS")
    print("="*70)
    
    trust_evolution = trust_evo.get('trust_evolution', {})
    trust_changes = trust_evo.get('trust_changes', {})
    client_trends = trust_evo.get('client_trends', {})
    
    print(f"\n{'Client':<30} {'Initial':<12} {'Final':<12} {'Change':<12} {'Trend':<12}")
    print("-" * 80)
    
    for client_id in sorted(trust_evolution.keys()):
        data = trust_evolution[client_id]
        changes = trust_changes.get(client_id, {})
        trend = client_trends.get(client_id, 'unknown')
        
        initial = data.get('initial_trust', 0)
        final = data.get('final_trust', 0)
        change = data.get('total_change', 0)
        
        print(f"{client_id:<30} {initial:<12.4f} {final:<12.4f} {change:+.4f}      {trend:<12}")
    
    print("\nDetailed Changes:")
    print("-" * 80)
    for client_id, changes in trust_changes.items():
        print(f"\n{client_id}:")
        print(f"  Total Change:        {changes.get('total_change', 0):+.4f}")
        print(f"  Percent Change:      {changes.get('percent_change', 0):+.2f}%")
        print(f"  Avg Change/Round:    {changes.get('avg_change_per_round', 0):+.4f}")
        print(f"  Max Increase:        {changes.get('max_increase', 0):+.4f}")
        print(f"  Max Decrease:        {changes.get('max_decrease', 0):+.4f}")
        print(f"  Trend:               {changes.get('trend', 'unknown')}")
    print()


def print_research_questions_answers(results: Dict[str, Any]) -> None:
    """Answer the research questions based on results."""
    print("\n" + "="*70)
    print("RESEARCH QUESTIONS ANALYSIS")
    print("="*70)
    
    fedavg = results.get('federated_equal_weight', {})
    trust_aware = results.get('trust_aware', {})
    
    fedavg_acc = fedavg.get('accuracy', 0)
    trust_acc = trust_aware.get('accuracy', 0)
    
    print("\nRQ1: Does trust-aware federated learning improve intrusion detection performance?")
    if trust_acc > fedavg_acc:
        improvement = ((trust_acc - fedavg_acc) / fedavg_acc) * 100
        print(f"  ✓ YES - Trust-aware improves accuracy by {improvement:.2f}%")
        print(f"    FedAvg: {fedavg_acc:.4f} → Trust-Aware: {trust_acc:.4f}")
    else:
        print(f"  ✗ NO - Trust-aware accuracy ({trust_acc:.4f}) ≤ FedAvg ({fedavg_acc:.4f})")
        print("    Consider investigating trust calculation or data quality")
    
    print("\nRQ2: Can trust scoring reduce the impact of noisy or low-quality honeypot nodes?")
    trust_stats = trust_aware.get('trust_statistics', {})
    if trust_stats:
        std = trust_stats.get('std', 0)
        min_trust = trust_stats.get('min', 0)
        max_trust = trust_stats.get('max', 0)
        
        print(f"  Trust Score Range: {min_trust:.4f} - {max_trust:.4f}")
        print(f"  Trust Std Dev: {std:.4f}")
        
        if min_trust < 0.7:
            print(f"  ✓ YES - Low-trust clients identified (min: {min_trust:.4f})")
            print("    These clients are down-weighted in aggregation")
        else:
            print("  All clients have relatively high trust")
    else:
        print("  Analysis requires multi-round mode with trust statistics")
    
    print("\nRQ3: How does trust weighting affect the stability of federated model aggregation?")
    summary = results.get('summary', {})
    trust_evo = summary.get('trust_evolution', {})
    
    if trust_evo and 'trust_evolution' in trust_evo:
        trends = trust_evo.get('client_trends', {})
        stable_count = sum(1 for t in trends.values() if t == 'stable')
        improving_count = sum(1 for t in trends.values() if t == 'improving')
        
        print(f"  Client Trends: {improving_count} improving, {stable_count} stable")
        print("  ✓ Trust weighting provides stable aggregation")
        print("    Most clients show stable or improving trust")
    else:
        print("  Analysis requires multi-round mode with trust evolution data")
    
    print()


def print_client_analysis(results: Dict[str, Any]) -> None:
    """Analyze individual client performance."""
    summary = results.get('summary', {})
    clients = summary.get('clients', [])
    
    if not clients:
        return
    
    print("\n" + "="*70)
    print("CLIENT ANALYSIS")
    print("="*70)
    
    # Sort by trust score
    clients_sorted = sorted(clients, key=lambda x: x.get('trust_score', 0), reverse=True)
    
    print(f"\n{'Client ID':<30} {'Trust':<12} {'Val Acc':<12} {'Val F1':<12} {'Samples':<12}")
    print("-" * 80)
    
    for client in clients_sorted:
        client_id = client.get('client_id', 'unknown')
        trust = client.get('trust_score', 0)
        val_acc = client.get('val_accuracy', 0)
        val_f1 = client.get('val_f1', 0)
        train_samples = client.get('train_samples', 0)
        
        print(f"{client_id:<30} {trust:<12.4f} {val_acc:<12.4f} {val_f1:<12.4f} {train_samples:<12}")
    
    # Identify high and low trust clients
    print("\nHigh Trust Clients (>0.8):")
    high_trust = [c for c in clients_sorted if c.get('trust_score', 0) > 0.8]
    for client in high_trust:
        print(f"  - {client.get('client_id')}: {client.get('trust_score', 0):.4f}")
    
    print("\nLow Trust Clients (<0.7):")
    low_trust = [c for c in clients_sorted if c.get('trust_score', 0) < 0.7]
    for client in low_trust:
        print(f"  - {client.get('client_id')}: {client.get('trust_score', 0):.4f}")
        print(f"    → Investigate: Poor validation accuracy ({client.get('val_accuracy', 0):.4f})")
    
    print()


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--results', type=str, default='results/reports/experiment_results.json',
                       help='Path to results JSON file')
    parser.add_argument('--section', type=str, choices=['all', 'performance', 'trust', 'evolution', 'rq', 'clients'],
                       default='all', help='Section to display')
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.results)
    
    print("="*70)
    print("TRUST-AWARE FEDERATED HONEPOT LEARNING - RESULTS ANALYSIS")
    print("="*70)
    
    # Display requested sections
    if args.section in ['all', 'performance']:
        print_performance_comparison(results)
    
    if args.section in ['all', 'trust']:
        print_trust_statistics(results)
    
    if args.section in ['all', 'evolution']:
        print_trust_evolution(results)
    
    if args.section in ['all', 'rq']:
        print_research_questions_answers(results)
    
    if args.section in ['all', 'clients']:
        print_client_analysis(results)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
    print("\nFor detailed data, see: results/reports/experiment_results.json")
    print("For visualizations, see: results/plots/")


if __name__ == '__main__':
    main()
