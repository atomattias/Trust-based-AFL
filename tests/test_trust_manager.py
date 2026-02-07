"""
Unit tests for TrustManager class.
"""

import unittest
import numpy as np
import tempfile
import shutil
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from federated_server import TrustManager, TrustHistory


class TestTrustHistory(unittest.TestCase):
    """Test TrustHistory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client_id = "test_client"
        self.initial_trust = 0.5
        self.history = TrustHistory(self.client_id, self.initial_trust)
    
    def test_initialization(self):
        """Test TrustHistory initialization."""
        self.assertEqual(self.history.client_id, self.client_id)
        self.assertEqual(self.history.trust_scores, [self.initial_trust])
        self.assertEqual(len(self.history.timestamps), 1)
        self.assertEqual(self.history.round_numbers, [0])
    
    def test_add_round(self):
        """Test adding a round."""
        self.history.add_round(1, 0.75, {'val_acc': 0.75})
        self.assertEqual(len(self.history.trust_scores), 2)
        self.assertEqual(self.history.trust_scores[-1], 0.75)
        self.assertEqual(self.history.round_numbers[-1], 1)
    
    def test_get_latest_trust(self):
        """Test getting latest trust score."""
        self.history.add_round(1, 0.8, {})
        self.assertEqual(self.history.get_latest_trust(), 0.8)
    
    def test_get_consistency_score(self):
        """Test consistency score calculation."""
        # Add consistent trust scores
        for i in range(5):
            self.history.add_round(i+1, 0.75, {})
        
        consistency = self.history.get_consistency_score()
        self.assertGreater(consistency, 0.9)  # Should be highly consistent
    
    def test_get_trend(self):
        """Test trend analysis."""
        # Add improving trust scores
        for i, trust in enumerate([0.5, 0.6, 0.7, 0.8, 0.9]):
            self.history.add_round(i+1, trust, {})
        
        trend = self.history.get_trend()
        self.assertEqual(trend, 'improving')
    
    def test_to_dict_from_dict(self):
        """Test serialization."""
        self.history.add_round(1, 0.75, {'val_acc': 0.75})
        data = self.history.to_dict()
        
        new_history = TrustHistory.from_dict(data)
        self.assertEqual(new_history.client_id, self.client_id)
        self.assertEqual(new_history.trust_scores, self.history.trust_scores)


class TestTrustManager(unittest.TestCase):
    """Test TrustManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.trust_manager = TrustManager(
            alpha=0.7,
            decay_rate=0.95,
            anomaly_threshold=0.2,
            initial_trust=0.5,
            storage_dir=self.temp_dir
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test TrustManager initialization."""
        self.assertEqual(self.trust_manager.alpha, 0.7)
        self.assertEqual(self.trust_manager.decay_rate, 0.95)
        self.assertEqual(self.trust_manager.anomaly_threshold, 0.2)
        self.assertEqual(self.trust_manager.initial_trust, 0.5)
    
    def test_initialize_client(self):
        """Test client initialization."""
        client_id = "test_client"
        self.trust_manager.initialize_client(client_id)
        self.assertIn(client_id, self.trust_manager.trust_histories)
        self.assertEqual(self.trust_manager.get_trust(client_id), 0.5)
    
    def test_update_trust_first_round(self):
        """Test trust update in first round."""
        client_id = "test_client"
        val_acc = 0.8
        
        updated_trust = self.trust_manager.update_trust(client_id, 1, val_acc)
        
        # First round: trust = alpha * initial + (1-alpha) * val_acc
        expected = 0.7 * 0.5 + 0.3 * 0.8
        self.assertAlmostEqual(updated_trust, expected, places=4)
    
    def test_update_trust_multiple_rounds(self):
        """Test trust update over multiple rounds."""
        client_id = "test_client"
        
        # Round 1
        trust1 = self.trust_manager.update_trust(client_id, 1, 0.6)
        
        # Round 2
        trust2 = self.trust_manager.update_trust(client_id, 2, 0.8)
        
        # Round 2 should be: alpha * trust1 + (1-alpha) * 0.8
        expected = 0.7 * trust1 + 0.3 * 0.8
        self.assertAlmostEqual(trust2, expected, places=4)
    
    def test_trust_bounds_validation(self):
        """Test trust bounds validation."""
        client_id = "test_client"
        
        # Test upper bound
        trust_high = self.trust_manager.update_trust(client_id, 1, 1.5)  # > 1.0
        self.assertLessEqual(trust_high, 1.0)
        
        # Test lower bound
        trust_low = self.trust_manager.update_trust(client_id, 2, -0.5)  # < 0.0
        self.assertGreaterEqual(trust_low, 0.0)
    
    def test_apply_decay(self):
        """Test trust decay."""
        client_id = "test_client"
        initial_trust = 0.8
        
        # Set initial trust
        self.trust_manager.update_trust(client_id, 1, initial_trust)
        
        # Apply decay
        decayed = self.trust_manager.apply_decay(client_id, 2)
        
        expected = initial_trust * 0.95
        self.assertAlmostEqual(decayed, expected, places=4)
    
    def test_detect_anomaly(self):
        """Test anomaly detection."""
        client_id = "test_client"
        
        # Set high trust
        self.trust_manager.update_trust(client_id, 1, 0.9)
        
        # Sudden drop
        self.trust_manager.update_trust(client_id, 2, 0.5)  # Drop of 0.4 > threshold 0.2
        
        anomaly = self.trust_manager.detect_anomaly(client_id)
        self.assertIsNotNone(anomaly)
        self.assertEqual(anomaly['type'], 'sudden_drop')
        self.assertGreater(anomaly['magnitude'], 0.2)
    
    def test_get_statistics(self):
        """Test trust statistics."""
        # Add multiple clients
        for i, val_acc in enumerate([0.7, 0.8, 0.9]):
            client_id = f"client_{i}"
            self.trust_manager.update_trust(client_id, 1, val_acc)
        
        stats = self.trust_manager.get_statistics()
        
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('min', stats)
        self.assertIn('max', stats)
        self.assertIn('median', stats)
        self.assertEqual(stats['count'], 3)
    
    def test_save_and_load_trust_history(self):
        """Test saving and loading trust history."""
        client_id = "test_client"
        
        # Update trust a few times
        for round_num, val_acc in enumerate([0.6, 0.7, 0.8], 1):
            self.trust_manager.update_trust(client_id, round_num, val_acc)
        
        # Save
        self.trust_manager.save_trust_history(client_id)
        
        # Create new manager and load
        new_manager = TrustManager(storage_dir=self.temp_dir)
        loaded = new_manager.load_trust_history(client_id)
        
        self.assertTrue(loaded)
        self.assertIn(client_id, new_manager.trust_histories)
        self.assertEqual(
            new_manager.get_trust(client_id),
            self.trust_manager.get_trust(client_id)
        )


if __name__ == '__main__':
    unittest.main()
