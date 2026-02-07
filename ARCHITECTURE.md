# Trust-Aware Federated Honeypot Learning
## System Architecture Document

---

## 1. System Overview

The Trust-Aware Federated Honeypot Learning system enables multiple honeypots to collaboratively train an intrusion detection model without sharing raw network traffic data. The system uses adaptive trust scoring to weight client contributions based on their reliability and performance.

### Key Principles
- **Privacy Preservation**: Raw data never leaves each honeypot
- **Trust-Aware Aggregation**: Client contributions weighted by trust scores
- **Adaptive Trust**: Trust scores evolve dynamically over multiple rounds
- **Decentralized Learning**: Each client trains independently

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Federated Server                            │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │         Trust Manager                                      │ │
│  │  - Trust History Storage                                   │ │
│  │  - Adaptive Trust Updates                                  │ │
│  │  - Anomaly Detection                                       │ │
│  │  - Trust Decay Management                                  │ │
│  └──────────────────────────────────────────────────────────┘ │
│                          │                                      │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │         Aggregators                                        │ │
│  │  - FedAvgAggregator (equal weights)                       │ │
│  │  - TrustAwareAggregator (trust-weighted)                  │ │
│  │  - EnsembleAggregator (voting-based)                      │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
┌───────▼──────┐  ┌───────▼──────┐  ┌───────▼──────┐
│  Honeypot A  │  │  Honeypot B   │  │  Honeypot C   │
│              │  │               │  │               │
│ - Local      │  │ - Local       │  │ - Local       │
│   Training   │  │   Training    │  │   Training    │
│ - Validation │  │ - Validation  │  │ - Validation │
│ - Trust      │  │ - Trust       │  │ - Trust       │
│   Metrics    │  │   Metrics     │  │   Metrics     │
└──────────────┘  └───────────────┘  └───────────────┘
```

---

## 3. Component Architecture

### 3.1 Trust Manager

**Location**: `src/federated_server.py`

**Responsibilities**:
- Maintain trust history for all clients
- Update trust scores using adaptive formula
- Detect anomalies in trust changes
- Apply trust decay for inactive clients
- Persist trust history to disk

**Key Methods**:
- `update_trust()`: Update trust score using weighted moving average
- `get_trust()`: Retrieve current trust score for a client
- `detect_anomaly()`: Identify sudden trust drops
- `apply_decay()`: Apply decay for inactive clients
- `save_trust_history()`: Persist trust data
- `load_trust_history()`: Load persisted trust data

**Trust Update Formula**:
```
trust_new = α × trust_old + (1-α) × validation_accuracy
```
Where α (alpha) = history weight (default: 0.7)

### 3.2 Trust History

**Location**: `src/federated_server.py`

**Data Structure**:
```python
TrustHistory {
    client_id: str
    trust_scores: [0.85, 0.87, 0.82, ...]  # Historical trust
    timestamps: [t1, t2, t3, ...]
    round_numbers: [1, 2, 3, ...]
    performance_metrics: [
        {round: 1, val_acc: 0.85, consistency: 0.9},
        {round: 2, val_acc: 0.87, consistency: 0.92},
        ...
    ]
    metadata: {
        initial_trust: 0.5,
        last_updated: timestamp,
        update_count: 15
    }
}
```

**Features**:
- Tracks trust evolution over rounds
- Stores performance metrics per round
- Calculates consistency scores
- Analyzes trends (improving/declining/stable)

### 3.3 Federated Client

**Location**: `src/federated_client.py`

**Responsibilities**:
- Load and preprocess local data
- Train local intrusion detection model
- Evaluate model on validation set
- Compute trust score (validation accuracy)
- Track performance history across rounds
- Provide model updates for aggregation

**Key Attributes**:
- `performance_history`: List of round-by-round metrics
- `trust_score`: Current trust score
- `model`: Trained local model
- `val_metrics`: Validation metrics

**Key Methods**:
- `load_data()`: Load and preprocess CSV data
- `train()`: Train local model
- `evaluate()`: Evaluate on validation set
- `compute_trust()`: Calculate trust score
- `record_round_performance()`: Track round metrics
- `get_consistency_score()`: Calculate performance consistency
- `get_performance_trend()`: Analyze trend
- `get_model_update()`: Prepare update for server

### 3.4 Aggregators

**Location**: `src/federated_server.py`

#### FedAvgAggregator
- **Purpose**: Baseline federated learning
- **Method**: Equal-weight averaging
- **Use Case**: Comparison baseline

#### TrustAwareAggregator
- **Purpose**: Trust-weighted aggregation
- **Method**: Weighted averaging based on trust scores
- **Trust Source**: 
  - Static mode: From client updates
  - Adaptive mode: From TrustManager
- **Weight Formula**: `weight_i = trust_i / sum(all_trust_scores)`

#### EnsembleAggregator
- **Purpose**: Voting-based aggregation
- **Method**: Weighted voting from all client models
- **Use Case**: Random Forest aggregation

---

## 4. Data Flow Architecture

### 4.1 Single-Round Flow (Static Trust)

```
┌─────────────────────────────────────────────────────────┐
│ 1. Client Setup                                          │
│    - Load CSV data                                      │
│    - Preprocess features                                │
│    - Split train/validation (80/20)                    │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Local Training                                       │
│    - Train model on local data                          │
│    - Evaluate on validation set                        │
│    - Compute trust = validation accuracy                │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Model Updates                                        │
│    - Extract model parameters                           │
│    - Package with trust score                           │
│    - Send to server                                     │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Aggregation                                          │
│    - Weight by trust scores                             │
│    - Aggregate parameters                               │
│    - Create global model                                │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│ 5. Evaluation                                           │
│    - Test on held-out data                              │
│    - Compute metrics                                    │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Multi-Round Flow (Adaptive Trust)

```
For each round (1 to num_rounds):
┌─────────────────────────────────────────────────────────┐
│ Phase 1: Local Training                                │
│    - Each client trains locally                        │
│    - Evaluate on validation set                        │
│    - Compute validation accuracy                       │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 2: Trust Update                                  │
│    - TrustManager receives performance metrics         │
│    - Updates trust: trust_new = α×old + (1-α)×new     │
│    - Detects anomalies                                 │
│    - Applies decay if needed                           │
│    - Stores in trust history                           │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 3: Aggregation                                   │
│    - Get updated trust scores from TrustManager        │
│    - Weight client contributions                        │
│    - Aggregate model parameters                         │
│    - Create global model                               │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 4: Distribution (Optional)                       │
│    - Send global model to clients                      │
│    - Clients can use for next round                    │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Trust Calculation Pipeline

```
Input: New Performance Metrics (Round N)
  │
  ▼
┌─────────────────────────────────┐
│ 1. Retrieve Trust History       │
│    - Get previous trust score   │
│    - Get performance history    │
│    - Get metadata               │
└─────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────┐
│ 2. Compute Factors              │
│    - Current validation accuracy│
│    - Consistency score          │
│    - Trend analysis              │
└─────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────┐
│ 3. Apply Trust Update Formula   │
│    trust_new = α × trust_old +   │
│                (1-α) × accuracy  │
└─────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────┐
│ 4. Apply Decay (if needed)     │
│    - Time-based decay          │
│    - Participation decay       │
└─────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────┐
│ 5. Validate & Smooth           │
│    - Check bounds [0, 1]      │
│    - Detect anomalies          │
│    - Apply smoothing           │
└─────────────────────────────────┘
  │
  ▼
Output: Updated Trust Score
```

---

## 6. Storage Architecture

### 6.1 Trust History Storage

**Directory Structure**:
```
results/
└── trust_history/
    ├── client_1_trust_history.json
    ├── client_2_trust_history.json
    └── ...
```

**File Format** (JSON):
```json
{
  "client_id": "client_1",
  "trust_scores": [0.5, 0.75, 0.82, 0.85],
  "timestamps": ["2024-01-01T10:00:00", ...],
  "round_numbers": [0, 1, 2, 3],
  "performance_metrics": [
    {
      "round": 1,
      "validation_accuracy": 0.75,
      "consistency_score": 0.9,
      "trend": "improving"
    },
    ...
  ],
  "metadata": {
    "initial_trust": 0.5,
    "last_updated": "2024-01-01T10:30:00",
    "update_count": 3
  }
}
```

### 6.2 Results Storage

**Directory Structure**:
```
results/
├── models/           # Saved models (if any)
├── plots/           # Visualizations
│   ├── trust_distribution.png
│   ├── performance_comparison.png
│   ├── confusion_matrices.png
│   ├── trust_evolution.png
│   └── ...
└── reports/          # Results summaries
    └── experiment_results.json
```

---

## 7. Module Dependencies

```
experiment.py
    │
    ├── src/preprocessing.py
    │   └── Data loading and preparation
    │
    ├── src/local_training.py
    │   └── Model training and evaluation
    │
    ├── src/federated_client.py
    │   ├── Client management
    │   └── Performance tracking
    │
    ├── src/federated_server.py
    │   ├── TrustManager
    │   ├── TrustHistory
    │   ├── FedAvgAggregator
    │   ├── TrustAwareAggregator
    │   └── EnsembleAggregator
    │
    ├── src/evaluation.py
    │   └── Metrics computation
    │
    └── src/visualization.py
        └── Plotting functions
```

---

## 8. Configuration Parameters

### 8.1 Trust Manager Parameters

- **alpha** (default: 0.7): History weight in trust update
  - Higher = more weight to past trust
  - Lower = more responsive to current performance

- **decay_rate** (default: 0.95): Trust decay per round
  - Applied when client is inactive
  - 0.95 = 5% decay per round

- **anomaly_threshold** (default: 0.2): Threshold for anomaly detection
  - Detects sudden trust drops > threshold

- **initial_trust** (default: 0.5): Starting trust for new clients

### 8.2 Experiment Parameters

- **num_rounds** (default: 1): Number of federated learning rounds
  - 1 = single-round (static trust)
  - >1 = multi-round (adaptive trust)

- **model_type**: 'random_forest' or 'logistic_regression'

- **random_state**: Random seed for reproducibility

---

## 9. Security and Privacy

### 9.1 Privacy Preservation

- **No Raw Data Sharing**: Only model parameters are shared
- **Local Processing**: All data preprocessing happens locally
- **Encrypted Communication**: (Can be added for production)

### 9.2 Trust Security

- **Trust History Encryption**: (Can be added)
- **Anomaly Detection**: Identifies suspicious trust changes
- **Access Control**: (Can be added for multi-user scenarios)

---

## 10. Scalability Considerations

### 10.1 Horizontal Scaling

- Multiple Trust Managers (sharded by client ID)
- Distributed trust history storage
- Load balancing for client requests

### 10.2 Vertical Scaling

- Caching frequently accessed trust scores
- Batch processing of trust updates
- Asynchronous trust computation

---

## 11. Extension Points

### 11.1 Advanced Trust Models

- Multi-factor trust scoring
- Bayesian trust updates
- Context-aware trust

### 11.2 Additional Features

- Differential privacy integration
- Adversarial robustness testing
- Real-time trust monitoring
- Trust visualization dashboards

---

## 12. System Requirements

### 12.1 Software Dependencies

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0

### 12.2 Hardware Requirements

- Minimum: 4GB RAM, 2 CPU cores
- Recommended: 8GB+ RAM, 4+ CPU cores
- Storage: ~100MB for code + data

---

## Document Information

**Version**: 1.0  
**Last Updated**: 2024  
**Author**: Trust-Aware Federated Honeypot Learning Project  
**Status**: Active Development

---

*This architecture document describes the Trust-Aware Federated Honeypot Learning system for intrusion detection. The system enables collaborative learning across distributed honeypots while maintaining privacy and adapting trust scores dynamically.*
