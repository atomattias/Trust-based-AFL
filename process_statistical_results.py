#!/usr/bin/env python3
"""
Process statistical validation results and compute confidence intervals.
"""

import re
import json
from scipy import stats
import numpy as np

def extract_results_from_log(log_file):
    """Extract statistical results from log file."""
    with open(log_file, 'r') as f:
        log = f.read()
    
    results = {}
    
    # Extract accuracy
    match = re.search(r'Accuracy\s+([\d.]+)±([\d.]+)\s+([\d.]+)±([\d.]+)\s+([\d.]+)±([\d.]+)', log, re.DOTALL)
    if match:
        results['centralized'] = {'accuracy': {'mean': float(match.group(1)), 'std': float(match.group(2))}}
        results['fedavg'] = {'accuracy': {'mean': float(match.group(3)), 'std': float(match.group(4))}}
        results['trust_aware'] = {'accuracy': {'mean': float(match.group(5)), 'std': float(match.group(6))}}
    
    # Extract F1
    match_f1 = re.search(r'F1 Score\s+([\d.]+)±([\d.]+)\s+([\d.]+)±([\d.]+)\s+([\d.]+)±([\d.]+)', log, re.DOTALL)
    if match_f1:
        results['centralized']['f1_score'] = {'mean': float(match_f1.group(1)), 'std': float(match_f1.group(2))}
        results['fedavg']['f1_score'] = {'mean': float(match_f1.group(3)), 'std': float(match_f1.group(4))}
        results['trust_aware']['f1_score'] = {'mean': float(match_f1.group(5)), 'std': float(match_f1.group(6))}
    
    # Extract precision
    match_prec = re.search(r'Precision\s+([\d.]+)±([\d.]+)\s+([\d.]+)±([\d.]+)\s+([\d.]+)±([\d.]+)', log, re.DOTALL)
    if match_prec:
        results['centralized']['precision'] = {'mean': float(match_prec.group(1)), 'std': float(match_prec.group(2))}
        results['fedavg']['precision'] = {'mean': float(match_prec.group(3)), 'std': float(match_prec.group(4))}
        results['trust_aware']['precision'] = {'mean': float(match_prec.group(5)), 'std': float(match_prec.group(6))}
    
    # Extract recall
    match_rec = re.search(r'Recall\s+([\d.]+)±([\d.]+)\s+([\d.]+)±([\d.]+)\s+([\d.]+)±([\d.]+)', log, re.DOTALL)
    if match_rec:
        results['centralized']['recall'] = {'mean': float(match_rec.group(1)), 'std': float(match_rec.group(2))}
        results['fedavg']['recall'] = {'mean': float(match_rec.group(3)), 'std': float(match_rec.group(4))}
        results['trust_aware']['recall'] = {'mean': float(match_rec.group(5)), 'std': float(match_rec.group(6))}
    
    return results

def compute_confidence_interval(mean, std, n=5, confidence=0.95):
    """Compute 95% confidence interval using t-distribution."""
    # t-value for 95% CI with n-1 degrees of freedom
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_value * std / np.sqrt(n)
    return mean - margin, mean + margin

def format_percentage(value, std=None, ci=None):
    """Format as percentage with std and/or CI."""
    if std is not None and ci is not None:
        return f"{value*100:.2f}% ± {std*100:.2f}% (95% CI: {ci[0]*100:.2f}% - {ci[1]*100:.2f}%)"
    elif std is not None:
        return f"{value*100:.2f}% ± {std*100:.2f}%"
    else:
        return f"{value*100:.2f}%"

def main():
    # Process CTU-13 results
    print("="*60)
    print("CTU-13 Statistical Results")
    print("="*60)
    ctu13_results = extract_results_from_log('ctu13_statistical_validation.log')
    
    ctu13_formatted = {}
    for approach in ['centralized', 'fedavg', 'trust_aware']:
        ctu13_formatted[approach] = {}
        for metric in ['accuracy', 'f1_score', 'precision', 'recall']:
            if metric in ctu13_results[approach]:
                mean = ctu13_results[approach][metric]['mean']
                std = ctu13_results[approach][metric]['std']
                ci = compute_confidence_interval(mean, std, n=5)
                ctu13_formatted[approach][metric] = {
                    'mean': mean,
                    'std': std,
                    'ci': ci,
                    'formatted': format_percentage(mean, std, ci)
                }
                print(f"{approach} {metric}: {format_percentage(mean, std, ci)}")
    
    # Process Honeypot results
    print("\n" + "="*60)
    print("Honeypot Statistical Results")
    print("="*60)
    honeypot_results = extract_results_from_log('honeypot_statistical_validation.log')
    
    honeypot_formatted = {}
    for approach in ['centralized', 'fedavg', 'trust_aware']:
        honeypot_formatted[approach] = {}
        for metric in ['accuracy', 'f1_score', 'precision', 'recall']:
            if metric in honeypot_results[approach]:
                mean = honeypot_results[approach][metric]['mean']
                std = honeypot_results[approach][metric]['std']
                ci = compute_confidence_interval(mean, std, n=5)
                honeypot_formatted[approach][metric] = {
                    'mean': mean,
                    'std': std,
                    'ci': ci,
                    'formatted': format_percentage(mean, std, ci)
                }
                print(f"{approach} {metric}: {format_percentage(mean, std, ci)}")
    
    # Save to JSON
    output = {
        'ctu13': ctu13_formatted,
        'honeypot': honeypot_formatted
    }
    
    with open('results/reports/statistical_results_formatted.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n✓ Results saved to results/reports/statistical_results_formatted.json")
    
    # Print LaTeX table format
    print("\n" + "="*60)
    print("LaTeX Table Format (CTU-13)")
    print("="*60)
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("\\textbf{Approach} & \\textbf{Accuracy} & \\textbf{F1-Score} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{FNR} \\\\")
    print("\\midrule")
    
    for approach_name, approach_key in [('Centralized', 'centralized'), ('FedAvg', 'fedavg'), ('TrustFed-Honeypot', 'trust_aware')]:
        acc = ctu13_formatted[approach_key]['accuracy']
        f1 = ctu13_formatted[approach_key]['f1_score']
        prec = ctu13_formatted[approach_key]['precision']
        rec = ctu13_formatted[approach_key]['recall']
        fnr = 1 - rec['mean']  # FNR = 1 - Recall
        
        print(f"{approach_name} & {acc['mean']*100:.2f}\\% $\\pm$ {acc['std']*100:.2f}\\% & {f1['mean']*100:.2f}\\% $\\pm$ {f1['std']*100:.2f}\\% & {prec['mean']*100:.2f}\\% $\\pm$ {prec['std']*100:.2f}\\% & {rec['mean']*100:.2f}\\% $\\pm$ {rec['std']*100:.2f}\\% & {fnr*100:.2f}\\% \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")

if __name__ == '__main__':
    main()
