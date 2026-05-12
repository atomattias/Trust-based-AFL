#!/usr/bin/env python3
"""
Create a balanced test set from heterogeneous client files.
This ensures the test set matches the training distribution.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from preprocessing import load_client_data, prepare_labels


def create_heterogeneous_test_set(
    data_dir: str = 'data/CSVs/extreme_scenario_v4_from_papers',
    output_path: str = 'data/CSVs/heterogeneous_test_set.csv',
    test_size: int = 10000,
    benign_ratio: float = 0.2,
    random_state: int = 42,
    exclude_clients: list = None
):
    """
    Create a balanced test set from heterogeneous client files.
    
    Args:
        data_dir: Directory containing heterogeneous client files
        output_path: Where to save the test set
        test_size: Total number of samples
        benign_ratio: Proportion of benign samples (0.2 = 20%)
        random_state: Random seed
        exclude_clients: List of client names to exclude (e.g., ['heartbleed'])
    """
    print("="*80)
    print("CREATING HETEROGENEOUS TEST SET")
    print("="*80)
    print(f"Test size: {test_size}")
    print(f"Benign ratio: {benign_ratio:.1%}")
    print()
    
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        return None
    
    # Find all heterogeneous client files
    all_files = sorted([f for f in data_dir.glob('*.csv') 
                       if 'benign' not in f.name.lower()])
    
    if exclude_clients:
        original_count = len(all_files)
        all_files = [f for f in all_files 
                    if not any(excluded in f.name.lower() for excluded in exclude_clients)]
        print(f"Excluded {original_count - len(all_files)} client(s): {exclude_clients}")
    
    print(f"Found {len(all_files)} client files")
    
    # Load and combine samples
    all_benign = []
    all_attack = []
    
    for client_file in all_files:
        try:
            df = load_client_data(str(client_file))
            df = prepare_labels(df)
            
            benign_df = df[df['label'] == 0].copy()
            attack_df = df[df['label'] == 1].copy()
            
            if len(benign_df) > 0:
                all_benign.append(benign_df)
            if len(attack_df) > 0:
                all_attack.append(attack_df)
            
            print(f"  {client_file.name}: {len(benign_df)} benign, {len(attack_df)} attack")
        except Exception as e:
            print(f"  ⚠️  Error loading {client_file.name}: {e}")
            continue
    
    if not all_benign and not all_attack:
        print("❌ No valid samples found!")
        return None
    
    # Combine
    if all_benign:
        combined_benign = pd.concat(all_benign, ignore_index=True)
    else:
        combined_benign = pd.DataFrame()
    
    if all_attack:
        combined_attack = pd.concat(all_attack, ignore_index=True)
    else:
        combined_attack = pd.DataFrame()
    
    print(f"\nTotal available:")
    print(f"  Benign: {len(combined_benign)}")
    print(f"  Attack: {len(combined_attack)}")
    
    # Calculate how many of each we need
    n_benign_needed = int(test_size * benign_ratio)
    n_attack_needed = test_size - n_benign_needed
    
    # Sample
    np.random.seed(random_state)
    
    if len(combined_benign) > 0:
        benign_sample = combined_benign.sample(
            n=min(n_benign_needed, len(combined_benign)), 
            random_state=random_state
        )
    else:
        print("⚠️  WARNING: No benign samples available!")
        benign_sample = pd.DataFrame()
        n_benign_needed = 0
        n_attack_needed = test_size
    
    if len(combined_attack) > 0:
        attack_sample = combined_attack.sample(
            n=min(n_attack_needed, len(combined_attack)), 
            random_state=random_state
        )
    else:
        print("⚠️  WARNING: No attack samples available!")
        attack_sample = pd.DataFrame()
    
    # Combine and shuffle
    if len(benign_sample) > 0 and len(attack_sample) > 0:
        test_df = pd.concat([benign_sample, attack_sample], ignore_index=True)
    elif len(benign_sample) > 0:
        test_df = benign_sample
    elif len(attack_sample) > 0:
        test_df = attack_sample
    else:
        print("❌ No samples to create test set!")
        return None
    
    test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Verify
    final_benign = (test_df['label'] == 0).sum()
    final_attack = (test_df['label'] == 1).sum()
    actual_ratio = final_benign / len(test_df) if len(test_df) > 0 else 0
    
    print(f"\n✅ Created heterogeneous test set:")
    print(f"   Total samples: {len(test_df)}")
    print(f"   Benign: {final_benign} ({actual_ratio:.1%})")
    print(f"   Attacks: {final_attack} ({1-actual_ratio:.1%})")
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved to: {output_path}")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description='Create heterogeneous test set')
    parser.add_argument('--data-dir', type=str, 
                       default='data/CSVs/extreme_scenario_v4_from_papers',
                       help='Directory containing heterogeneous client files')
    parser.add_argument('--output', type=str,
                       default='data/CSVs/heterogeneous_test_set.csv',
                       help='Output path for test set')
    parser.add_argument('--test-size', type=int, default=10000,
                       help='Total number of test samples')
    parser.add_argument('--benign-ratio', type=float, default=0.2,
                       help='Proportion of benign samples (0.2 = 20%%)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--exclude', type=str, nargs='+', default=['heartbleed'],
                       help='Client names to exclude')
    
    args = parser.parse_args()
    
    create_heterogeneous_test_set(
        data_dir=args.data_dir,
        output_path=args.output,
        test_size=args.test_size,
        benign_ratio=args.benign_ratio,
        random_state=args.seed,
        exclude_clients=args.exclude
    )


if __name__ == '__main__':
    main()
