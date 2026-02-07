"""
Main experiment script for Trust-Aware Federated Honeypot Learning.

This script orchestrates three approaches:
1. Centralized Learning (upper bound)
2. Standard Federated Learning (equal weights)
3. Trust-Aware Federated Learning (proposed method)
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from preprocessing import load_client_data, prepare_labels, prepare_features, split_data
from local_training import train_local_model
from federated_client import FederatedClient
from federated_server import FedAvgAggregator, TrustAwareAggregator, EnsembleAggregator, TrustManager
from evaluation import evaluate_model_on_test, generate_results_summary
from visualization import save_all_visualizations
from config_loader import load_trust_config, apply_trust_config


class ExperimentRunner:
    """Orchestrates the complete experiment."""
    
    def __init__(
        self,
        data_dir: str = 'data/CSVs',
        model_type: str = 'random_forest',
        random_state: int = 42,
        test_csv: Optional[str] = None,
        num_rounds: int = 1,
        trust_alpha: float = 0.7,
        trust_storage_dir: Optional[str] = None
    ):
        """
        Initialize the experiment runner.
        
        Args:
            data_dir: Directory containing CSV files
            model_type: Type of model ('random_forest' or 'logistic_regression')
            random_state: Random seed
            test_csv: Optional path to test CSV (if None, will use one of the client CSVs)
            num_rounds: Number of federated learning rounds (1 = single round, >1 = multi-round with adaptive trust)
            trust_alpha: History weight for adaptive trust (0.7 = 70% old, 30% new)
            trust_storage_dir: Directory to store trust history (None = in-memory only)
        """
        self.data_dir = Path(data_dir)
        self.model_type = model_type
        self.random_state = random_state
        self.test_csv = test_csv
        self.num_rounds = num_rounds
        
        self.clients = []
        self.client_updates = []
        self.test_data = None
        
        # Initialize TrustManager for adaptive trust (if multi-round)
        if num_rounds > 1:
            # Load configuration
            trust_config = load_trust_config()
            
            # Override with command-line arguments if provided
            if trust_storage_dir is None:
                trust_storage_dir = trust_config.get('trust_manager', {}).get('storage_dir', 'results/trust_history')
            
            # Create TrustManager with config or defaults
            trust_mgr_config = trust_config.get('trust_manager', {})
            self.trust_manager = TrustManager(
                alpha=trust_alpha if trust_alpha != 0.7 else trust_mgr_config.get('alpha', 0.7),
                decay_rate=trust_mgr_config.get('decay_rate', 0.95),
                anomaly_threshold=trust_mgr_config.get('anomaly_threshold', 0.2),
                initial_trust=trust_mgr_config.get('initial_trust', 0.5),
                storage_dir=trust_storage_dir
            )
            
            # Apply any additional config
            apply_trust_config(self.trust_manager, trust_config)
            
            # Load existing trust histories if available
            self.trust_manager.load_all_trust_histories()
        else:
            self.trust_manager = None
        
    def discover_client_files(self) -> List[str]:
        """
        Discover all CSV files in the data directory.
        Prefers heterogeneous clients if they exist, then mixed files, then original files.
        
        Returns:
            List of CSV file paths
        """
        csv_files = list(self.data_dir.glob('*.csv'))
        
        # Filter out benign files for client selection (we'll use them for mixing)
        all_attack_files = [f for f in csv_files if 'benign' not in f.name.lower()]
        benign_files = [f for f in csv_files if 'benign' in f.name.lower()]
        
        # Check for heterogeneous clients (created by create_heterogeneous_clients.py)
        heterogeneous_files = [f for f in all_attack_files if any(
            tier in f.name.lower() for tier in ['high_quality', 'medium_quality', 'low_quality', 'compromised']
        )]
        
        # Prefer mixed files if they exist (created by prepare_realistic_data.py)
        mixed_files = [f for f in all_attack_files if f.name.startswith('mixed_') and f not in heterogeneous_files]
        original_files = [f for f in all_attack_files if not f.name.startswith('mixed_') and f not in heterogeneous_files]
        
        if heterogeneous_files:
            # Use heterogeneous clients (they have varying quality)
            attack_files = heterogeneous_files
            print(f"Found {len(heterogeneous_files)} heterogeneous client files (varying quality)")
            print(f"  This will demonstrate weaknesses of Centralized and FedAvg!")
            print(f"  High-quality, medium-quality, low-quality, and compromised clients detected")
        elif mixed_files:
            # Use mixed files (they contain both benign and attack samples)
            attack_files = mixed_files
            print(f"Found {len(mixed_files)} mixed CSV files (with benign samples)")
            print(f"Found {len(original_files)} original attack CSV files (ignored)")
        else:
            # Fall back to original files
            attack_files = original_files
            print(f"Found {len(original_files)} attack CSV files")
            if len(mixed_files) == 0:
                print("  (Tip: Run prepare_realistic_data.py to create mixed datasets)")
                print("  (Tip: Run create_heterogeneous_clients.py to create heterogeneous clients)")
        
        print(f"Found {len(benign_files)} benign CSV files")
        
        return [str(f) for f in attack_files], [str(f) for f in benign_files]
    
    def prepare_test_data(self, test_csv_path: str) -> tuple:
        """
        Prepare test dataset.
        
        Args:
            test_csv_path: Path to test CSV file
            
        Returns:
            X_test, y_test
        """
        print(f"\nLoading test data from {test_csv_path}")
        df_test = load_client_data(test_csv_path)
        df_test = prepare_labels(df_test)
        X_test = prepare_features(df_test)
        y_test = df_test['label']
        
        print(f"Test set: {len(X_test)} samples, {X_test.shape[1]} features")
        print(f"Test set class distribution: {(y_test == 0).sum()} benign, {(y_test == 1).sum()} attack")
        
        return X_test, y_test
    
    def setup_clients(self, attack_files: List[str], benign_files: List[str], num_clients: Optional[int] = None) -> None:
        """
        Set up federated clients from CSV files.
        
        Args:
            attack_files: List of attack CSV file paths
            benign_files: List of benign CSV file paths (for mixing)
            num_clients: Number of clients to use (None = use all)
        """
        if num_clients is not None:
            attack_files = attack_files[:num_clients]
        
        print(f"\nSetting up {len(attack_files)} federated clients...")
        
        for idx, attack_file in enumerate(attack_files):
            client_id = f"client_{idx+1}_{Path(attack_file).stem}"
            print(f"  Setting up {client_id}...")
            
            # Extract quality from filename if present (e.g., 'high_quality_client_1.csv')
            import re
            quality_match = re.search(r'(high|medium|low|compromised)_quality', Path(attack_file).name)
            client_quality = quality_match.group(1) if quality_match else 'unknown'
            
            # Debug: Check if quality was extracted
            if 'compromised' in Path(attack_file).name.lower() and client_quality != 'compromised':
                # Try alternative pattern: client_X_compromised_...
                alt_match = re.search(r'_compromised_', Path(attack_file).name)
                if alt_match:
                    client_quality = 'compromised'
            
            # CRITICAL FIX: Find original clean source file for validation
            # For heterogeneous clients (compromised/low/medium), use clean validation set
            clean_validation_source = None
            if client_quality in ['compromised', 'low', 'medium']:
                # Extract source file name from heterogeneous client filename
                # Pattern: client_X_compromised_mixed_Y.csv -> mixed_Y.csv
                # Handle cases like: mixed_ssh_patator-new.csv (with hyphens)
                filename = Path(attack_file).name
                # Find "mixed_" followed by everything until ".csv"
                source_match = re.search(r'mixed_[^.]+\.csv', filename)
                if source_match:
                    source_filename = source_match.group(0)
                    # Look for original source in data/CSVs directory
                    original_source = Path('data/CSVs') / source_filename
                    if original_source.exists():
                        clean_validation_source = str(original_source)
                        print(f"\n    ✅ FOUND CLEAN VALIDATION SOURCE for {client_id}: {source_filename}")
                    else:
                        print(f"\n    ⚠️  Clean source not found: {source_filename} (looking in data/CSVs/)")
                else:
                    print(f"\n    ⚠️  Could not extract source filename from: {filename}")
            else:
                print(f"    (High-quality client - no clean validation needed)")
            
            client = FederatedClient(
                client_id=client_id,
                data_path=attack_file,
                model_type=self.model_type,
                random_state=self.random_state,
                client_quality=client_quality,
                clean_validation_source=clean_validation_source
            )
            
            # Load and prepare data
            client.load_data()
            
            # Optionally mix in some benign data (simulate realistic honeypot)
            # For simplicity, we'll use the attack data as-is
            # In a real scenario, you might sample benign data and mix it
            
            self.clients.append(client)
        
        print(f"Successfully set up {len(self.clients)} clients")
    
    def train_clients(self) -> None:
        """Train all client models."""
        print("\n" + "="*60)
        print("Training Local Models at Each Client")
        print("="*60)
        
        for client in self.clients:
            print(f"\nTraining {client.client_id}...")
            client.train()
            client.evaluate()
            client.compute_trust()
            
            info = client.get_info()
            print(f"  Train samples: {info['train_samples']}")
            print(f"  Val samples: {info['val_samples']}")
            print(f"  Val Accuracy: {info['val_accuracy']:.4f}")
            print(f"  Trust Score: {info['trust_score']:.4f}")
        
        # Collect model updates
        self.client_updates = [client.get_model_update() for client in self.clients]
        
        print(f"\n✓ All {len(self.clients)} clients trained successfully")
    
    def approach_1_centralized(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Approach 1: Centralized Learning.
        
        In heterogeneous client scenarios, this approach is weak because:
        - Cannot filter bad data from low-quality clients
        - All data (good + bad) is combined and trained together
        - Bad clients degrade the global model
        - No concept of dynamic trust (single round only)
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Results dictionary
        """
        print("\n" + "="*60)
        print("Approach 1: Centralized Learning")
        print("="*60)
        print("Note: In heterogeneous scenarios, Centralized is weak because")
        print("      it cannot filter bad data from low-quality clients.")
        print("      Also, Centralized has no dynamic trust concept (single round).")
        
        # Combine all training data from all clients
        print("\nCombining all client data (good + bad)...")
        all_X_train = []
        all_y_train = []
        client_qualities = []
        
        for client in self.clients:
            all_X_train.append(client.X_train)
            all_y_train.append(client.y_train)
            
            # Detect client quality from filename
            client_path = Path(client.data_path)
            if 'high_quality' in client_path.name.lower():
                client_qualities.append('high')
            elif 'medium_quality' in client_path.name.lower():
                client_qualities.append('medium')
            elif 'low_quality' in client_path.name.lower():
                client_qualities.append('low')
            elif 'compromised' in client_path.name.lower():
                client_qualities.append('compromised')
            else:
                client_qualities.append('unknown')
        
        X_combined = pd.concat(all_X_train, ignore_index=True)
        y_combined = pd.concat(all_y_train, ignore_index=True)
        
        print(f"Combined dataset: {len(X_combined)} samples")
        print(f"Class distribution: {(y_combined == 0).sum()} benign, {(y_combined == 1).sum()} attack")
        
        # Report client quality distribution
        if any(q != 'unknown' for q in client_qualities):
            quality_counts = {}
            for q in client_qualities:
                quality_counts[q] = quality_counts.get(q, 0) + 1
            print(f"\nClient quality distribution:")
            for quality, count in sorted(quality_counts.items()):
                print(f"  {quality}: {count} clients")
            print("  ⚠️  Centralized cannot filter bad data - all clients contribute equally!")
            print("  ⚠️  Centralized has no dynamic trust - single round only!")
        
        # Train centralized model
        print("\nTraining centralized model on combined data...")
        centralized_model, train_metrics = train_local_model(
            X_combined,
            y_combined,
            model_type=self.model_type,
            random_state=self.random_state
        )
        
        print(f"Training accuracy: {train_metrics['train_accuracy']:.4f}")
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        results = evaluate_model_on_test(centralized_model, X_test, y_test)
        
        print(f"\nTest Accuracy: {results['accuracy']:.4f}")
        print(f"Test F1-Score: {results['f1_score']:.4f}")
        print(f"Test Precision: {results['precision']:.4f}")
        print(f"Test Recall: {results['recall']:.4f}")
        print(f"False Positive Rate: {results['false_positive_rate']:.4f}")
        
        # Add note about weakness in heterogeneous scenarios
        if any(q in ['low', 'compromised'] for q in client_qualities):
            print("\n⚠️  Centralized weakness: Bad clients degrade model performance")
            print("    Cannot filter or weight clients by quality")
            print("    No dynamic trust - single round only (cannot adapt)")
        
        return results
    
    def approach_2_federated_equal_weight(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Approach 2: Standard Federated Learning (equal weights - FedAvg).
        
        In heterogeneous client scenarios, this approach is weak because:
        - All clients have equal weight (1/N)
        - Cannot differentiate good vs bad clients
        - Low-quality clients have same influence as high-quality
        - Weights stay constant (no adaptation)
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Results dictionary
        """
        print("\n" + "="*60)
        print("Approach 2: Standard Federated Learning (Equal Weights - FedAvg)")
        print("="*60)
        print("Note: In heterogeneous scenarios, FedAvg is weak because")
        print("      all clients have equal weight - cannot prioritize good clients.")
        
        # Report client trust scores to show heterogeneity
        trust_scores = [update.get('trust', 0) for update in self.client_updates]
        if trust_scores:
            print(f"\nClient trust scores (validation accuracy):")
            for i, (update, trust) in enumerate(zip(self.client_updates, trust_scores)):
                client_id = update.get('client_id', f'client_{i+1}')
                print(f"  {client_id}: {trust:.4f}")
            
            trust_range = max(trust_scores) - min(trust_scores)
            print(f"\nTrust score range: {min(trust_scores):.4f} - {max(trust_scores):.4f} (range: {trust_range:.4f})")
            if trust_range > 0.3:
                print("  ⚠️  High heterogeneity detected - FedAvg equal weighting is suboptimal!")
            print(f"  FedAvg weight per client: {1.0/len(self.client_updates):.4f} (equal for all)")
        
        # Use FedAvgAggregator with retraining (true FedAvg)
        print("\nAggregating client models with equal weights using true FedAvg (retraining)...")
        print("  All clients contribute equally (1/{})".format(len(self.client_updates)))
        aggregator = FedAvgAggregator(model_type=self.model_type)
        global_model = aggregator.aggregate(self.client_updates, use_retraining=True)
        
        # Evaluate global model
        results = evaluate_model_on_test(global_model, X_test, y_test)
        
        print(f"\nTest Accuracy: {results['accuracy']:.4f}")
        print(f"Test F1-Score: {results['f1_score']:.4f}")
        print(f"Test Precision: {results['precision']:.4f}")
        print(f"Test Recall: {results['recall']:.4f}")
        print(f"False Positive Rate: {results['false_positive_rate']:.4f}")
        
        # Add note about weakness
        if trust_scores and trust_range > 0.3:
            print("\n⚠️  FedAvg weakness: Equal weighting doesn't differentiate client quality")
            print("    Low-quality clients have same influence as high-quality clients")
            print("    Weights stay constant - cannot adapt to changing conditions")
        
        return results
    
    def approach_3_trust_aware(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Approach 3: Trust-Aware Federated Learning (proposed method).
        
        Supports both single-round (static trust) and multi-round (adaptive trust) modes.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Results dictionary
        """
        if self.num_rounds > 1:
            return self._approach_3_multi_round(X_test, y_test)
        else:
            return self._approach_3_single_round(X_test, y_test)
    
    def _approach_3_single_round(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Single-round trust-aware federated learning (static trust).
        
        In heterogeneous client scenarios, this approach is strong because:
        - Clients weighted by trust scores (quality-based)
        - High-trust clients contribute more
        - Low-trust clients have minimal influence
        """
        print("\n" + "="*60)
        print("Approach 3: Trust-Aware Federated Learning (Single Round)")
        print("="*60)
        print("Note: Trust-Aware is strong in heterogeneous scenarios because")
        print("      it weights clients by quality (trust scores).")
        
        # Display trust scores
        print("\nClient Trust Scores (validation accuracy):")
        trust_scores = []
        for update in self.client_updates:
            trust = update.get('trust', 0)
            trust_scores.append(trust)
            print(f"  {update['client_id']}: {trust:.4f}")
        
        if trust_scores:
            trust_range = max(trust_scores) - min(trust_scores)
            print(f"\nTrust score range: {min(trust_scores):.4f} - {max(trust_scores):.4f} (range: {trust_range:.4f})")
            if trust_range > 0.3:
                print("  ✅ High heterogeneity - Trust-Aware can leverage quality differences!")
        
        # Use TrustAwareAggregator with retraining (true trust-aware FedAvg)
        print("\nAggregating client models with trust-weighted retraining (true trust-aware FedAvg)...")
        print("  Clients weighted by trust scores (quality-based weighting)")
        aggregator = TrustAwareAggregator(model_type=self.model_type)
        global_model = aggregator.aggregate(self.client_updates, use_retraining=True)
        
        # Display client weights
        weights = aggregator.get_client_weights(self.client_updates)
        print("\nClient Weights in Aggregation (trust-weighted):")
        for client_id, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            # Find corresponding trust score
            trust = next((u.get('trust', 0) for u in self.client_updates if u.get('client_id') == client_id), 0)
            print(f"  {client_id}: {weight:.4f} (trust: {trust:.4f})")
        
        # Show comparison with equal weights
        equal_weight = 1.0 / len(self.client_updates)
        print(f"\nComparison:")
        print(f"  Equal weight (FedAvg): {equal_weight:.4f} per client")
        print(f"  Trust-weighted: High-trust clients get more, low-trust get less")
        
        # Evaluate global model
        results = evaluate_model_on_test(global_model, X_test, y_test)
        
        print(f"\nTest Accuracy: {results['accuracy']:.4f}")
        print(f"Test F1-Score: {results['f1_score']:.4f}")
        print(f"Test Precision: {results['precision']:.4f}")
        print(f"Test Recall: {results['recall']:.4f}")
        print(f"False Positive Rate: {results['false_positive_rate']:.4f}")
        
        # Add note about strength
        if trust_scores and trust_range > 0.3:
            print("\n✅ Trust-Aware strength: Quality-based weighting prioritizes good clients")
            print("    High-trust clients contribute more, low-trust clients contribute less")
        
        return results
    
    def _approach_3_multi_round(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Multi-round trust-aware federated learning with adaptive trust."""
        print("\n" + "="*60)
        print(f"Approach 3: Trust-Aware Federated Learning (Multi-Round: {self.num_rounds} rounds)")
        print("="*60)
        
        # Initialize clients in TrustManager
        for client in self.clients:
            self.trust_manager.initialize_client(client.client_id)
        
        global_model = None
        
        # Multi-round federated learning loop
        for round_num in range(1, self.num_rounds + 1):
            print(f"\n{'='*60}")
            print(f"Round {round_num}/{self.num_rounds}")
            print(f"{'='*60}")
            
            # DYNAMIC TRUST EVOLUTION: Inject changes to simulate compromise/improvement
            if round_num == 4:  # After round 3, some clients get compromised
                print("\n🔴 DYNAMIC EVENT: Some clients getting compromised (round 4)")
                # Select 2-3 compromised clients (low/compromised quality clients)
                compromised_clients = [
                    c for c in self.clients 
                    if 'low' in c.client_quality.lower() or 'compromised' in c.client_quality.lower()
                ][:3]  # Take up to 3
                for client in compromised_clients:
                    print(f"  ⚠️  {client.client_id}: Injecting additional corruption (simulating compromise)")
                    client.inject_dynamic_change(round_num, change_type='degrade', severity=0.4)
            
            elif round_num == 6:  # After round 5, some clients improve
                print("\n🟢 DYNAMIC EVENT: Some clients improving data quality (round 6)")
                # Select 2-3 medium-quality clients to improve
                improving_clients = [
                    c for c in self.clients 
                    if 'medium' in c.client_quality.lower()
                ][:2]  # Take up to 2
                for client in improving_clients:
                    print(f"  ✅ {client.client_id}: Restoring data quality (simulating improvement)")
                    client.inject_dynamic_change(round_num, change_type='improve', severity=0.5)
            
            # Phase 1: Local Training
            print("\nPhase 1: Local Training")
            for client in self.clients:
                client.train()
                client.evaluate()
                client.compute_trust()
            
            # Phase 2: Trust Update
            print("\nPhase 2: Trust Update")
            for client in self.clients:
                val_acc = client.val_metrics['accuracy']
                perf_metrics = {
                    'val_f1': client.val_metrics['f1_score'],
                    'val_precision': client.val_metrics['precision'],
                    'val_recall': client.val_metrics['recall']
                }
                
                updated_trust = self.trust_manager.update_trust(
                    client.client_id,
                    round_num,
                    val_acc,
                    perf_metrics
                )
                
                # Update client's trust score
                client.trust_score = updated_trust
                
                # Check for anomalies
                anomaly = self.trust_manager.detect_anomaly(client.client_id)
                if anomaly:
                    print(f"  ⚠️  Anomaly detected for {client.client_id}: {anomaly['type']} (drop: {anomaly['magnitude']:.4f})")
            
            # Display updated trust scores and evolution
            print(f"\nUpdated Trust Scores (Round {round_num}):")
            all_trust = self.trust_manager.get_all_trust_scores()
            trust_list = []
            for client_id, trust in sorted(all_trust.items(), key=lambda x: x[1], reverse=True):
                trust_list.append(trust)
                # Get trust history to show change
                if client_id in self.trust_manager.trust_histories:
                    history = self.trust_manager.trust_histories[client_id]
                    if history and len(history.trust_scores) > 1:
                        prev_trust = history.trust_scores[-2] if len(history.trust_scores) > 1 else trust
                        change = trust - prev_trust
                        change_str = f" ({change:+.4f})" if abs(change) > 0.001 else " (stable)"
                        print(f"  {client_id}: {trust:.4f}{change_str}")
                    else:
                        print(f"  {client_id}: {trust:.4f}")
                else:
                    print(f"  {client_id}: {trust:.4f}")
            
            # Show trust evolution summary
            if round_num > 1 and trust_list:
                trust_range = max(trust_list) - min(trust_list)
                print(f"\nTrust Statistics (Round {round_num}):")
                print(f"  Range: {min(trust_list):.4f} - {max(trust_list):.4f} (range: {trust_range:.4f})")
                print(f"  Mean: {np.mean(trust_list):.4f}, Std: {np.std(trust_list):.4f}")
                if trust_range > 0.3:
                    print("  ✅ High heterogeneity - Trust-Aware adapts dynamically!")
                print("  📊 Trust scores change over rounds (dynamic) - FedAvg weights stay constant!")
            
            # Phase 3: Get Model Updates
            self.client_updates = [
                client.get_model_update(round_num=round_num) 
                for client in self.clients
            ]
            
            # Phase 4: Aggregation (using EnsembleAggregator for proper aggregation)
            print("\nPhase 3: Aggregation")
            
            # Update trust scores in client updates for ensemble aggregator
            all_trust = self.trust_manager.get_all_trust_scores()
            for update in self.client_updates:
                update['trust'] = all_trust.get(update['client_id'], update.get('trust', 0.5))
            
            # Use EnsembleAggregator with trust-weighted voting
            ensemble = EnsembleAggregator(model_type=self.model_type, aggregation_method='weighted_voting')
            ensemble.aggregate(self.client_updates)
            global_model = ensemble  # Store ensemble for potential use
            
            # Display client weights
            weights = ensemble.get_client_weights(self.client_updates)
            print("\nClient Weights in Aggregation:")
            for client_id, weight in weights.items():
                print(f"  {client_id}: {weight:.4f}")
            
            # Save trust history periodically
            if round_num % 5 == 0 or round_num == self.num_rounds:
                self.trust_manager.save_trust_history()
        
        # Final evaluation
        print(f"\n{'='*60}")
        print("Final Evaluation")
        print(f"{'='*60}")
        
        # Use the global model from last round (already retrained with trust weights)
        # Or re-aggregate with final trust scores if needed
        all_trust = self.trust_manager.get_all_trust_scores()
        for update in self.client_updates:
            update['trust'] = all_trust.get(update['client_id'], update.get('trust', 0.5))
        
        # Re-aggregate with final trust scores for final evaluation
        final_aggregator = TrustAwareAggregator(
            model_type=self.model_type,
            trust_manager=self.trust_manager
        )
        final_global_model = final_aggregator.aggregate(self.client_updates, use_retraining=True)
        results = evaluate_model_on_test(final_global_model, X_test, y_test)
        
        # Add trust statistics to results
        trust_stats = self.trust_manager.get_statistics()
        results['trust_statistics'] = trust_stats
        
        print(f"\nTest Accuracy: {results['accuracy']:.4f}")
        print(f"Test F1-Score: {results['f1_score']:.4f}")
        print(f"Test Precision: {results['precision']:.4f}")
        print(f"Test Recall: {results['recall']:.4f}")
        print(f"False Positive Rate: {results['false_positive_rate']:.4f}")
        
        print(f"\nTrust Statistics:")
        print(f"  Mean: {trust_stats['mean']:.4f}")
        print(f"  Std: {trust_stats['std']:.4f}")
        print(f"  Min: {trust_stats['min']:.4f}")
        print(f"  Max: {trust_stats['max']:.4f}")
        
        return results
    
    def run_experiment(self, num_clients: Optional[int] = None, num_rounds: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete experiment.
        
        Args:
            num_clients: Number of clients to use (None = use all)
            num_rounds: Override num_rounds from initialization (optional)
            
        Returns:
            Complete results dictionary
        """
        if num_rounds is not None:
            self.num_rounds = num_rounds
        
        print("="*60)
        print("Trust-Aware Federated Honeypot Learning Experiment")
        if self.num_rounds > 1:
            print(f"Mode: Multi-Round ({self.num_rounds} rounds with adaptive trust)")
        else:
            print("Mode: Single-Round (static trust)")
        print("="*60)
        
        # Discover data files
        attack_files, benign_files = self.discover_client_files()
        
        if not attack_files:
            raise ValueError(f"No attack CSV files found in {self.data_dir}")
        
        # Set up clients
        self.setup_clients(attack_files, benign_files, num_clients=num_clients)
        
        # Train all clients
        self.train_clients()
        
        # Prepare test data
        if self.test_csv is None:
            # Strategy: Create a balanced test set from training data of mixed clients
            # This ensures we have both benign and attack samples
            mixed_clients = [c for c in self.clients if 'mixed' in c.client_id.lower()]
            
            if mixed_clients:
                # Use training data from first mixed client (has both benign and attacks)
                # Training data should have benign samples even if validation doesn't
                test_client = mixed_clients[0]
                X_test = test_client.X_train
                y_test = test_client.y_train
                
                # Sample a subset for testing (to avoid using all training data)
                from sklearn.model_selection import train_test_split
                X_test, _, y_test, _ = train_test_split(
                    X_test, y_test, 
                    test_size=0.5,  # Use 50% of training data as test
                    random_state=42,
                    stratify=y_test  # Maintain class balance
                )
                
                print(f"\nUsing training data subset from {test_client.client_id} as test set")
                print(f"  Test set: {len(X_test)} samples")
                print(f"  Class distribution: {(y_test == 0).sum()} benign, {(y_test == 1).sum()} attacks")
                print(f"  Benign ratio: {(y_test == 0).sum() / len(y_test) * 100:.1f}%")
                test_file = None
            else:
                # Fallback: Try to use a mixed file
                mixed_test_files = [f for f in attack_files if 'mixed_' in Path(f).name]
                
                if mixed_test_files and len(mixed_test_files) > len(self.clients):
                    test_file = mixed_test_files[len(self.clients)]
                    print(f"\nUsing mixed file for test set: {Path(test_file).name}")
                elif len(attack_files) > len(self.clients):
                    test_file = attack_files[len(self.clients)]
                else:
                    # Last resort: Use validation set from first client
                    test_file = None
                    X_test = self.clients[0].X_val
                    y_test = self.clients[0].y_val
                    print(f"\nUsing validation set from {self.clients[0].client_id} as test set")
                    print(f"  (This validation set: {(y_test == 0).sum()} benign, {(y_test == 1).sum()} attacks)")
        else:
            test_file = self.test_csv
            X_test, y_test = self.prepare_test_data(test_file)
        
        if test_file:
            X_test, y_test = self.prepare_test_data(test_file)
        
        # Run all three approaches
        centralized_results = self.approach_1_centralized(X_test, y_test)
        federated_results = self.approach_2_federated_equal_weight(X_test, y_test)
        trust_aware_results = self.approach_3_trust_aware(X_test, y_test)
        
        # Generate summary
        client_info = [client.get_info() for client in self.clients]
        summary = generate_results_summary(
            centralized_results,
            federated_results,
            trust_aware_results,
            client_info,
            trust_manager=self.trust_manager if hasattr(self, 'trust_manager') else None
        )
        
        # Print comparison
        print("\n" + "="*60)
        print("RESULTS COMPARISON")
        print("="*60)
        print(f"\n{'Metric':<25} {'Centralized':<15} {'FedAvg':<15} {'Trust-Aware':<15}")
        print("-" * 70)
        print(f"{'Accuracy':<25} {centralized_results['accuracy']:<15.4f} {federated_results['accuracy']:<15.4f} {trust_aware_results['accuracy']:<15.4f}")
        print(f"{'F1-Score':<25} {centralized_results['f1_score']:<15.4f} {federated_results['f1_score']:<15.4f} {trust_aware_results['f1_score']:<15.4f}")
        print(f"{'Precision':<25} {centralized_results['precision']:<15.4f} {federated_results['precision']:<15.4f} {trust_aware_results['precision']:<15.4f}")
        print(f"{'Recall':<25} {centralized_results['recall']:<15.4f} {federated_results['recall']:<15.4f} {trust_aware_results['recall']:<15.4f}")
        print(f"{'False Positive Rate':<25} {centralized_results['false_positive_rate']:<15.4f} {federated_results['false_positive_rate']:<15.4f} {trust_aware_results['false_positive_rate']:<15.4f}")
        
        # Save results
        results_dict = {
            'centralized': centralized_results,
            'federated_equal_weight': federated_results,
            'trust_aware': trust_aware_results,
            'summary': summary
        }
        
        # Save to JSON
        os.makedirs('results/reports', exist_ok=True)
        with open('results/reports/experiment_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        print(f"\n✓ Results saved to results/reports/experiment_results.json")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        save_all_visualizations(
            client_info,
            {
                'centralized': centralized_results,
                'federated_equal_weight': federated_results,
                'trust_aware': trust_aware_results
            },
            output_dir='results/plots',
            trust_manager=self.trust_manager if hasattr(self, 'trust_manager') else None
        )
        
        return results_dict


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Trust-Aware Federated Honeypot Learning Experiment')
    parser.add_argument('--data-dir', type=str, default='data/CSVs',
                       help='Directory containing CSV files')
    parser.add_argument('--model-type', type=str, default='random_forest',
                       choices=['random_forest', 'logistic_regression'],
                       help='Type of model to use')
    parser.add_argument('--num-clients', type=int, default=None,
                       help='Number of clients to use (None = use all)')
    parser.add_argument('--num-rounds', type=int, default=1,
                       help='Number of federated learning rounds (1 = single round, >1 = multi-round with adaptive trust)')
    parser.add_argument('--test-csv', type=str, default=None,
                       help='Path to test CSV file (optional)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--trust-alpha', type=float, default=0.7,
                       help='History weight for adaptive trust (0.7 = 70%% old, 30%% new)')
    parser.add_argument('--trust-storage-dir', type=str, default=None,
                       help='Directory to store trust history (default: results/trust_history)')
    
    args = parser.parse_args()
    
    # Run experiment
    runner = ExperimentRunner(
        data_dir=args.data_dir,
        model_type=args.model_type,
        random_state=args.random_state,
        test_csv=args.test_csv,
        num_rounds=args.num_rounds,
        trust_alpha=args.trust_alpha,
        trust_storage_dir=args.trust_storage_dir
    )
    
    results = runner.run_experiment(num_clients=args.num_clients)
    
    print("\n" + "="*60)
    print("Experiment completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
