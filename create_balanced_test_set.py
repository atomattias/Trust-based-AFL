#!/usr/bin/env python3
"""
Create a balanced test set from mixed files.

This script creates a dedicated test set file with both benign and attack samples.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def create_balanced_test_set(
    mixed_files: list,
    output_path: str = 'data/CSVs/balanced_test_set.csv',
    test_size: int = 5000,
    benign_ratio: float = 0.2,
    random_state: int = 42
):
    """
    Create a balanced test set from mixed files.
    
    Args:
        mixed_files: List of paths to mixed CSV files
        output_path: Where to save the test set
        test_size: Total number of samples
        benign_ratio: Proportion of benign samples (0.2 = 20%)
        random_state: Random seed
    """
    print("="*70)
    print("CREATING BALANCED TEST SET")
    print("="*70)
    
    all_samples = []
    
    for mixed_file in mixed_files:
        try:
            df = pd.read_csv(mixed_file)
            print(f"\nLoaded {Path(mixed_file).name}: {len(df)} samples")
            
            # Check label distribution
            if 'label' in df.columns:
                labels = df['label'].value_counts()
                benign_count = labels.get(0, 0)
                attack_count = labels.get(1, 0)
                print(f"  Benign: {benign_count}, Attacks: {attack_count}")
                all_samples.append(df)
            else:
                print(f"  ⚠️  No 'label' column found, skipping")
        except Exception as e:
            print(f"  ❌ Error loading {mixed_file}: {e}")
    
    if not all_samples:
        print("\n❌ No valid samples found!")
        return None
    
    # Combine all samples
    combined_df = pd.concat(all_samples, ignore_index=True)
    print(f"\nCombined dataset: {len(combined_df)} samples")
    
    # Separate by label
    benign_df = combined_df[combined_df['label'] == 0].copy()
    attack_df = combined_df[combined_df['label'] == 1].copy()
    
    print(f"Total benign: {len(benign_df)}")
    print(f"Total attacks: {len(attack_df)}")
    
    # Calculate how many of each we need
    n_benign_needed = int(test_size * benign_ratio)
    n_attack_needed = test_size - n_benign_needed
    
    # Sample
    np.random.seed(random_state)
    benign_sample = benign_df.sample(n=min(n_benign_needed, len(benign_df)), random_state=random_state)
    attack_sample = attack_df.sample(n=min(n_attack_needed, len(attack_df)), random_state=random_state)
    
    # Combine and shuffle
    test_df = pd.concat([benign_sample, attack_sample], ignore_index=True)
    test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Verify
    final_benign = (test_df['label'] == 0).sum()
    final_attack = (test_df['label'] == 1).sum()
    actual_ratio = final_benign / len(test_df)
    
    print(f"\n✅ Created balanced test set:")
    print(f"   Total samples: {len(test_df)}")
    print(f"   Benign: {final_benign} ({actual_ratio:.1%})")
    print(f"   Attacks: {final_attack} ({1-actual_ratio:.1%})")
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved to: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Create balanced test set from mixed files')
    parser.add_argument('--data-dir', type=str, default='data/CSVs',
                       help='Directory containing CSV files')
    parser.add_argument('--output', type=str, default='data/CSVs/balanced_test_set.csv',
                       help='Output path for test set')
    parser.add_argument('--test-size', type=int, default=5000,
                       help='Total number of test samples')
    parser.add_argument('--benign-ratio', type=float, default=0.2,
                       help='Proportion of benign samples (0.2 = 20%%)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    mixed_files = list(data_dir.glob('mixed_*.csv'))
    
    if not mixed_files:
        print("❌ No mixed files found!")
        print(f"   Looking in: {data_dir}")
        print("   Run: python3 prepare_realistic_data.py --benign-ratio 0.2")
        return
    
    print(f"Found {len(mixed_files)} mixed files")
    
    create_balanced_test_set(
        mixed_files,
        output_path=args.output,
        test_size=args.test_size,
        benign_ratio=args.benign_ratio
    )
    
    print("\n" + "="*70)
    print("To use this test set, run:")
    print(f"  python3 experiment.py --test-csv {args.output} --num-rounds 10")
    print("="*70)


if __name__ == '__main__':
    main()
