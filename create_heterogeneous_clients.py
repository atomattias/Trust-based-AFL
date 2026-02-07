"""
Script to create heterogeneous client datasets with varying quality levels.

Creates clients with:
- High-quality: Clean data, good performance (trust: 0.95-0.99)
- Medium-quality: Some noise, moderate performance (trust: 0.70-0.85)
- Low-quality: Noisy data, poor performance (trust: 0.50-0.65)
- Compromised: Poisoned data, very poor performance (trust: 0.20-0.40)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from sklearn.utils import resample
import random


def add_label_noise(df: pd.DataFrame, noise_ratio: float, random_state: int = 42) -> pd.DataFrame:
    """
    Add label noise to a dataset by flipping labels.
    
    Args:
        df: DataFrame with 'label' column
        noise_ratio: Proportion of labels to flip (0.0 to 1.0)
        random_state: Random seed
        
    Returns:
        DataFrame with noisy labels
    """
    df = df.copy()
    np.random.seed(random_state)
    
    if 'label' not in df.columns:
        # Try to find label column
        label_col = None
        for col in df.columns:
            if col.lower() == 'label':
                label_col = col
                break
        
        if label_col is None:
            raise ValueError("No 'label' column found")
    else:
        label_col = 'label'
    
    # Get indices to flip
    n_samples = len(df)
    n_flip = int(n_samples * noise_ratio)
    flip_indices = np.random.choice(n_samples, size=n_flip, replace=False)
    
    # Flip labels (0 <-> 1, or BENIGN <-> attack)
    for idx in flip_indices:
        current_label = df.iloc[idx][label_col]
        
        # Handle binary labels (0/1)
        if current_label in [0, 1]:
            df.iloc[idx, df.columns.get_loc(label_col)] = 1 - current_label
        # Handle text labels (BENIGN/attack)
        elif str(current_label).upper() == 'BENIGN':
            df.iloc[idx, df.columns.get_loc(label_col)] = 'ATTACK'
        else:
            df.iloc[idx, df.columns.get_loc(label_col)] = 'BENIGN'
    
    return df


def corrupt_features(df: pd.DataFrame, corruption_ratio: float, random_state: int = 42) -> pd.DataFrame:
    """
    Corrupt features by adding noise or setting to random values.
    
    Args:
        df: DataFrame with features
        corruption_ratio: Proportion of feature values to corrupt
        random_state: Random seed
        
    Returns:
        DataFrame with corrupted features
    """
    df = df.copy()
    np.random.seed(random_state)
    
    # Exclude non-feature columns
    exclude_cols = ['flow_id', 'timestamp', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 
                   'protocol', 'label', 'Label']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    if not feature_cols:
        return df
    
    # Corrupt features
    n_samples = len(df)
    n_corrupt = int(n_samples * corruption_ratio)
    corrupt_indices = np.random.choice(n_samples, size=n_corrupt, replace=False)
    
    for idx in corrupt_indices:
        # Corrupt random features for this sample
        n_features_to_corrupt = max(1, int(len(feature_cols) * 0.1))  # Corrupt 10% of features
        corrupt_features = np.random.choice(feature_cols, size=n_features_to_corrupt, replace=False)
        
        for feat in corrupt_features:
            if df[feat].dtype in [np.int64, np.float64]:
                # Add noise or set to random value
                if np.random.random() < 0.5:
                    # Add noise (convert to float if int)
                    noise = np.random.normal(0, df[feat].std() * 2)
                    if df[feat].dtype == np.int64:
                        df[feat] = df[feat].astype(float)
                    df.iloc[idx, df.columns.get_loc(feat)] += noise
                else:
                    # Set to random value in range (convert to float if int)
                    min_val = float(df[feat].min())
                    max_val = float(df[feat].max())
                    if df[feat].dtype == np.int64:
                        df[feat] = df[feat].astype(float)
                    df.iloc[idx, df.columns.get_loc(feat)] = np.random.uniform(min_val, max_val)
    
    return df


def create_heterogeneous_client(
    source_file: str,
    output_file: str,
    quality_tier: str,
    target_size: int = None,
    random_state: int = 42
) -> None:
    """
    Create a heterogeneous client dataset with specified quality tier.
    
    Args:
        source_file: Path to source CSV file
        output_file: Path to output CSV file
        quality_tier: 'high', 'medium', 'low', or 'compromised'
        target_size: Target dataset size (None = use all data)
        random_state: Random seed
    """
    print(f"\nCreating {quality_tier}-quality client from {Path(source_file).name}")
    
    # Load source data
    df = pd.read_csv(source_file)
    
    # Prepare labels if needed
    if 'label' not in df.columns:
        label_col = None
        for col in df.columns:
            if col.lower() == 'label':
                label_col = col
                break
        if label_col:
            df = df.rename(columns={label_col: 'label'})
    
    # Apply quality modifications
    if quality_tier == 'high':
        # High-quality: Clean data, minimal modifications
        df_modified = df.copy()
        label_noise = 0.0
        feature_corruption = 0.0
        
    elif quality_tier == 'medium':
        # Medium-quality: Noticeable noise (should measurably degrade)
        label_noise = 0.25  # 25% label noise
        feature_corruption = 0.15  # 15% feature corruption
        df_modified = add_label_noise(df, label_noise, random_state)
        df_modified = corrupt_features(df_modified, feature_corruption, random_state)
        
    elif quality_tier == 'low':
        # Low-quality: Very heavy noise (should be weak clients)
        label_noise = 0.55  # 55% label noise
        feature_corruption = 0.35  # 35% feature corruption
        df_modified = add_label_noise(df, label_noise, random_state)
        df_modified = corrupt_features(df_modified, feature_corruption, random_state)
        
    elif quality_tier == 'compromised':
        # Compromised: Severely poisoned data (should be clearly untrustworthy)
        # Balance: Severe enough to hurt FedAvg/Centralized, but validation can still detect it
        label_noise = 0.65  # 65% label noise (poisoned) - enough to degrade but validation can detect
        feature_corruption = 0.55  # 55% feature corruption - severe but detectable
        df_modified = add_label_noise(df, label_noise, random_state)
        df_modified = corrupt_features(df_modified, feature_corruption, random_state)
        
    else:
        raise ValueError(f"Unknown quality tier: {quality_tier}")
    
    # Adjust dataset size if needed
    if target_size is not None and len(df_modified) > target_size:
        # Downsample while maintaining class balance if possible
        try:
            if 'label' in df_modified.columns:
                # Try stratified sampling
                df_modified = resample(
                    df_modified,
                    n_samples=target_size,
                    random_state=random_state,
                    stratify=df_modified['label']
                )
            else:
                # Random sampling
                df_modified = df_modified.sample(n=target_size, random_state=random_state)
        except:
            # Fallback to random sampling
            df_modified = df_modified.sample(n=target_size, random_state=random_state)
    elif target_size is not None and len(df_modified) < target_size:
        # Upsample (with replacement)
        df_modified = resample(
            df_modified,
            n_samples=target_size,
            random_state=random_state,
            replace=True
        )
    
    # Save modified dataset
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_modified.to_csv(output_path, index=False)
    
    # Report statistics
    if 'label' in df_modified.columns:
        benign_count = (df_modified['label'] == 0).sum() + (df_modified['label'] == 'BENIGN').sum()
        attack_count = len(df_modified) - benign_count
        print(f"  Created: {len(df_modified)} samples")
        print(f"    Benign: {benign_count}, Attack: {attack_count}")
        print(f"    Label noise: {label_noise*100:.0f}%, Feature corruption: {feature_corruption*100:.0f}%")
    else:
        print(f"  Created: {len(df_modified)} samples")


def main():
    parser = argparse.ArgumentParser(description='Create heterogeneous client datasets')
    parser.add_argument('--data-dir', type=str, default='data/CSVs',
                       help='Directory containing source CSV files')
    parser.add_argument('--output-dir', type=str, default='data/CSVs/heterogeneous',
                       help='Output directory for heterogeneous clients')
    parser.add_argument('--high-count', type=int, default=4,
                       help='Number of high-quality clients')
    parser.add_argument('--medium-count', type=int, default=4,
                       help='Number of medium-quality clients')
    parser.add_argument('--low-count', type=int, default=4,
                       help='Number of low-quality clients')
    parser.add_argument('--compromised-count', type=int, default=4,
                       help='Number of compromised clients')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find source files (prefer mixed files)
    source_files = sorted(list(data_dir.glob('mixed_*.csv')))
    if not source_files:
        source_files = sorted(list(data_dir.glob('*.csv')))
        source_files = [f for f in source_files if 'benign' not in f.name.lower()]
    
    if not source_files:
        print(f"Error: No source CSV files found in {data_dir}")
        return
    
    print("="*70)
    print("CREATING HETEROGENEOUS CLIENT DATASETS")
    print("="*70)
    print(f"\nSource directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"\nQuality distribution (HARSH SCENARIO):")
    print(f"  High-quality: {args.high_count} clients (trust: 0.90-0.95)")
    print(f"  Medium-quality: {args.medium_count} clients (trust: 0.60-0.75)")
    print(f"  Low-quality: {args.low_count} clients (trust: 0.40-0.55)")
    print(f"  Compromised: {args.compromised_count} clients (trust: 0.15-0.30)")
    
    # Create clients
    client_idx = 1
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # High-quality clients
    for i in range(args.high_count):
        if client_idx - 1 < len(source_files):
            source = source_files[(client_idx - 1) % len(source_files)]
            output = output_dir / f"client_{client_idx}_high_quality_{source.stem}.csv"
            create_heterogeneous_client(
                str(source),
                str(output),
                'high',
                random_state=args.seed + client_idx
            )
            client_idx += 1
    
    # Medium-quality clients
    for i in range(args.medium_count):
        if client_idx - 1 < len(source_files):
            source = source_files[(client_idx - 1) % len(source_files)]
            output = output_dir / f"client_{client_idx}_medium_quality_{source.stem}.csv"
            create_heterogeneous_client(
                str(source),
                str(output),
                'medium',
                target_size=5000,  # Smaller dataset for medium quality
                random_state=args.seed + client_idx
            )
            client_idx += 1
    
    # Low-quality clients
    for i in range(args.low_count):
        if client_idx - 1 < len(source_files):
            source = source_files[(client_idx - 1) % len(source_files)]
            output = output_dir / f"client_{client_idx}_low_quality_{source.stem}.csv"
            create_heterogeneous_client(
                str(source),
                str(output),
                'low',
                target_size=3000,  # Slightly larger: ensures low-quality clients still influence FedAvg/centralized
                random_state=args.seed + client_idx
            )
            client_idx += 1
    
    # Compromised clients
    for i in range(args.compromised_count):
        if client_idx - 1 < len(source_files):
            source = source_files[(client_idx - 1) % len(source_files)]
            output = output_dir / f"client_{client_idx}_compromised_{source.stem}.csv"
            create_heterogeneous_client(
                str(source),
                str(output),
                'compromised',
                target_size=20000,  # Large datasets to dominate FedAvg/Centralized if weighted equally
                random_state=args.seed + client_idx
            )
            client_idx += 1
    
    print("\n" + "="*70)
    print(f"✓ Created {client_idx - 1} heterogeneous clients")
    print(f"  Output directory: {output_dir}")
    print("="*70)
    print("\nNext steps:")
    print(f"  1. Run experiment with: python3 experiment.py --data-dir {output_dir}")
    print(f"  2. Use --num-rounds 10 to see dynamic trust evolution")


if __name__ == '__main__':
    main()
