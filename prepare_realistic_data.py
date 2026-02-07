#!/usr/bin/env python3
"""
Data preparation script to create realistic datasets by mixing benign traffic.

This script mixes benign samples into attack files to create balanced datasets
that better represent real-world intrusion detection scenarios.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import List, Optional


def mix_benign_into_attack(
    attack_file: Path,
    benign_files: List[Path],
    benign_ratio: float = 0.2,
    output_dir: Optional[Path] = None,
    random_state: int = 42
) -> Optional[Path]:
    """
    Mix benign traffic into attack file to create realistic dataset.
    
    Args:
        attack_file: Path to attack CSV
        benign_files: List of benign CSV paths
        benign_ratio: Proportion of benign samples (0.2 = 20% benign, 80% attack)
        output_dir: Directory to save mixed files (None = same as attack_file)
        random_state: Random seed
        
    Returns:
        Path to created mixed file, or None if failed
    """
    print(f"\nProcessing: {attack_file.name}")
    
    # Load attack data
    try:
        attack_df = pd.read_csv(attack_file)
        print(f"  Loaded {len(attack_df)} attack samples")
    except Exception as e:
        print(f"  Error loading attack file: {e}")
        return None
    
    # Prepare labels
    if 'Label' in attack_df.columns:
        attack_df['label'] = attack_df['Label'].apply(
            lambda x: 0 if str(x).upper() == 'BENIGN' else 1
        )
    else:
        attack_df['label'] = 1  # Assume all attacks if no label
    
    # Count actual attacks (exclude any existing benign)
    attack_count = (attack_df['label'] == 1).sum()
    existing_benign = (attack_df['label'] == 0).sum()
    
    print(f"  Attack samples: {attack_count}, Existing benign: {existing_benign}")
    
    # Calculate how many benign samples we need
    # If we want benign_ratio of total, and we have attack_count attacks:
    # benign_count / (attack_count + benign_count) = benign_ratio
    # Solving: benign_count = attack_count * benign_ratio / (1 - benign_ratio)
    needed_benign = int(attack_count * benign_ratio / (1 - benign_ratio))
    needed_benign = max(0, needed_benign - existing_benign)  # Subtract existing
    
    if needed_benign <= 0:
        print(f"  Already has enough benign samples, skipping")
        return None
    
    # Load benign data
    benign_dfs = []
    for bf in benign_files:
        try:
            bdf = pd.read_csv(bf)
            if 'Label' in bdf.columns:
                bdf['label'] = bdf['Label'].apply(
                    lambda x: 0 if str(x).upper() == 'BENIGN' else 1
                )
            else:
                bdf['label'] = 0  # Assume benign if no label
            
            # Only take benign samples
            benign_only = bdf[bdf['label'] == 0].copy()
            if len(benign_only) > 0:
                benign_dfs.append(benign_only)
                print(f"  Found {len(benign_only)} benign samples in {bf.name}")
        except Exception as e:
            print(f"  Error loading {bf.name}: {e}")
            continue
    
    if not benign_dfs:
        print(f"  No benign data available")
        return None
    
    # Combine all benign data
    all_benign = pd.concat(benign_dfs, ignore_index=True)
    print(f"  Total benign samples available: {len(all_benign)}")
    
    # Sample needed amount
    sample_size = min(needed_benign, len(all_benign))
    benign_sample = all_benign.sample(n=sample_size, random_state=random_state)
    print(f"  Sampling {sample_size} benign samples")
    
    # Combine attack and benign
    mixed_df = pd.concat([attack_df, benign_sample], ignore_index=True)
    
    # Shuffle to mix classes
    mixed_df = mixed_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Verify final distribution
    final_benign = (mixed_df['label'] == 0).sum()
    final_attack = (mixed_df['label'] == 1).sum()
    actual_ratio = final_benign / len(mixed_df)
    
    print(f"  Final: {final_attack} attacks, {final_benign} benign ({actual_ratio:.1%})")
    
    # Save
    if output_dir is None:
        output_dir = attack_file.parent
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"mixed_{attack_file.name}"
    mixed_df.to_csv(output_path, index=False)
    print(f"  ✓ Saved to: {output_path}")
    
    return output_path


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Prepare realistic datasets by mixing benign traffic')
    parser.add_argument('--data-dir', type=str, default='data/CSVs',
                       help='Directory containing CSV files')
    parser.add_argument('--benign-ratio', type=float, default=0.2,
                       help='Proportion of benign samples (0.2 = 20%%)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: same as data-dir)')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of attack files to process')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    
    # Find attack and benign files
    all_files = list(data_dir.glob('*.csv'))
    attack_files = [f for f in all_files if 'benign' not in f.name.lower()]
    benign_files = [f for f in all_files if 'benign' in f.name.lower()]
    
    print("="*70)
    print("PREPARING REALISTIC DATASETS")
    print("="*70)
    print(f"\nFound {len(attack_files)} attack files")
    print(f"Found {len(benign_files)} benign files")
    print(f"Target benign ratio: {args.benign_ratio:.1%}")
    
    if not benign_files:
        print("\nERROR: No benign CSV files found!")
        print("Please ensure benign CSV files are in the data directory.")
        return
    
    if args.max_files:
        attack_files = attack_files[:args.max_files]
    
    # Process each attack file
    created_files = []
    for attack_file in attack_files:
        result = mix_benign_into_attack(
            attack_file,
            benign_files,
            benign_ratio=args.benign_ratio,
            output_dir=output_dir
        )
        if result:
            created_files.append(result)
    
    print("\n" + "="*70)
    print(f"SUMMARY: Created {len(created_files)} mixed dataset files")
    print("="*70)
    print("\nTo use these files, run:")
    print(f"  python3 experiment.py --data-dir {output_dir}")
    print("\nOr copy mixed files to data/CSVs/ and use default data-dir")


if __name__ == '__main__':
    main()
