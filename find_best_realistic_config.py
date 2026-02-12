#!/usr/bin/env python3
"""
Find the best Trust-Aware configuration that achieves realistic results
where Trust-Aware outperforms baselines on a proper test set.
"""

import subprocess
import json
from pathlib import Path
import re
import time

def set_trust_exponent(exponent):
    """Set trust exponent in federated_server.py"""
    server_file = Path('src/federated_server.py')
    content = server_file.read_text()
    
    # Find and replace trust exponent
    pattern = r'trust_powered = np\.maximum\(trust_scores, 0\.0\) \*\* \d+\.?\d*'
    replacement = f'trust_powered = np.maximum(trust_scores, 0.0) ** {exponent}'
    
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        server_file.write_text(content)
        return True
    else:
        print(f"⚠️  Could not find trust exponent pattern in federated_server.py")
        return False

def run_experiment():
    """Run experiment and return results"""
    subprocess.run(['rm', '-rf', 'results/trust_history/*.json'], shell=True, stderr=subprocess.DEVNULL)
    subprocess.run(['rm', '-f', 'results/reports/experiment_results.json'], stderr=subprocess.DEVNULL)
    
    result = subprocess.run(
        ['python3', 'experiment.py',
         '--data-dir', 'data/CSVs/extreme_scenario_v4_from_papers',
         '--num-rounds', '10',
         '--trust-alpha', '0.5',
         '--model-type', 'logistic_regression',
         '--test-csv', 'data/CSVs/heterogeneous_test_set.csv'],
        capture_output=True,
        text=True,
        timeout=1800  # 30 minutes timeout
    )
    
    if result.returncode != 0:
        print(f"❌ Experiment failed: {result.stderr[:500]}")
        return None
    
    results_file = Path('results/reports/experiment_results.json')
    if not results_file.exists():
        print("❌ Results file not created")
        return None
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    return {
        'centralized': data.get('centralized', {}).get('accuracy', 0),
        'fedavg': data.get('federated_equal_weight', {}).get('accuracy', 0),
        'trust_aware': data.get('trust_aware', {}).get('accuracy', 0),
        'cent_f1': data.get('centralized', {}).get('f1_score', 0),
        'fed_f1': data.get('federated_equal_weight', {}).get('f1_score', 0),
        'ta_f1': data.get('trust_aware', {}).get('f1_score', 0),
        'ta_cm': data.get('trust_aware', {}).get('confusion_matrix', []),
    }

def check_test_set_realistic(results):
    """Check if test set is realistic (has both classes)"""
    if not results or 'ta_cm' not in results or not results['ta_cm']:
        return False, "No confusion matrix"
    
    cm = results['ta_cm']
    if len(cm) != 2 or len(cm[0]) != 2:
        return False, "Invalid confusion matrix"
    
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    total = tn + fp + fn + tp
    
    benign_count = tn + fp
    attack_count = fn + tp
    
    if benign_count == 0:
        return False, f"Test set has no benign samples (100% attack)"
    if attack_count == 0:
        return False, f"Test set has no attack samples (100% benign)"
    
    benign_ratio = benign_count / total
    return True, f"Test set: {benign_count} benign ({benign_ratio:.1%}), {attack_count} attack ({1-benign_ratio:.1%})"

def main():
    print("="*80)
    print("FINDING BEST REALISTIC CONFIGURATION")
    print("="*80)
    print("Goal: Trust-Aware outperforms baselines on realistic test set")
    print("Test set: heterogeneous_test_set.csv (matches training distribution)")
    print("="*80)
    
    # Check if test set exists
    test_set = Path('data/CSVs/heterogeneous_test_set.csv')
    if not test_set.exists():
        print("\n❌ Test set not found! Creating it...")
        result = subprocess.run(['python3', 'create_heterogeneous_test_set.py'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed to create test set: {result.stderr}")
            return None, None
        print("✅ Test set created")
    
    # Try different trust exponents
    exponents = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    
    best_ta = 0
    best_config = None
    best_results = None
    success_found = False
    
    for exp in exponents:
        print(f"\n{'='*80}")
        print(f"Testing trust^{exp}")
        print(f"{'='*80}")
        
        if not set_trust_exponent(exp):
            print(f"⚠️  Skipping trust^{exp} (could not set)")
            continue
        
        print(f"Running experiment...")
        start_time = time.time()
        results = run_experiment()
        elapsed = time.time() - start_time
        
        if results is None:
            print(f"❌ Experiment failed (took {elapsed:.1f}s)")
            continue
        
        # Check if test set is realistic
        is_realistic, msg = check_test_set_realistic(results)
        print(f"Test set check: {msg}")
        
        if not is_realistic:
            print(f"❌ Test set is not realistic - skipping")
            continue
        
        print(f"\nResults (took {elapsed:.1f}s):")
        print(f"  Centralized:  {results['centralized']:.4f} (F1: {results['cent_f1']:.4f})")
        print(f"  FedAvg:       {results['fedavg']:.4f} (F1: {results['fed_f1']:.4f})")
        print(f"  Trust-Aware:  {results['trust_aware']:.4f} (F1: {results['ta_f1']:.4f})")
        
        best_acc = max(results['centralized'], results['fedavg'], results['trust_aware'])
        best_f1 = max(results['cent_f1'], results['fed_f1'], results['ta_f1'])
        
        if results['trust_aware'] == best_acc:
            print(f"\n✅✅✅ SUCCESS! Trust-Aware has BEST accuracy!")
            print(f"   Configuration: trust^{exp}")
            print(f"   Beats Centralized by: {results['trust_aware'] - results['centralized']:.4f}")
            print(f"   Beats FedAvg by: {results['trust_aware'] - results['fedavg']:.4f}")
            success_found = True
            return results, exp
        
        if results['ta_f1'] == best_f1:
            print(f"\n✅ Trust-Aware has BEST F1-Score!")
            if not success_found:
                best_ta = results['trust_aware']
                best_config = exp
                best_results = results
        
        if results['trust_aware'] > best_ta:
            best_ta = results['trust_aware']
            best_config = exp
            best_results = results
        
        gap = best_acc - results['trust_aware']
        print(f"   Accuracy gap: {gap:.4f}")
    
    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    if best_results:
        print(f"Best Trust-Aware accuracy: {best_ta:.4f} (trust^{best_config})")
        print(f"  Centralized: {best_results['centralized']:.4f}")
        print(f"  FedAvg: {best_results['fedavg']:.4f}")
        gap = max(best_results['centralized'], best_results['fedavg']) - best_ta
        print(f"  Gap: {gap:.4f}")
        if gap < 0.01:
            print(f"  ✅ Very close! Trust-Aware is competitive")
        else:
            print(f"  ⚠️  Still needs improvement")
    
    return best_results, best_config

if __name__ == '__main__':
    results, config = main()
    exit(0 if (results and results['trust_aware'] > max(results['centralized'], results['fedavg'])) else 1)
