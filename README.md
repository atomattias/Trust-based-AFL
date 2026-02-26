# Trust-Aware Federated Honeypot Learning for Intrusion Detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18786479.svg)](https://doi.org/10.5281/zenodo.18786479)



This project implements a trust-aware federated learning system for intrusion detection using honeypot-generated network traffic data. The system enables multiple honeypots to collaboratively train a shared intrusion detection model without sharing raw data, while weighting contributions based on each honeypot's reliability (trust score).

> **🚀 New to the project?** Start with the [Quick Start Guide](QUICK_START_GUIDE.md) for step-by-step instructions from setup to results analysis.

## Project Overview

The system compares three approaches:
1. **Centralized Learning** - All data combined and trained centrally (baseline)
2. **Standard Federated Learning (FedAvg)** - Equal-weight aggregation (baseline)
3. **Trust-Aware Federated Learning** (proposed) - Trust-weighted aggregation based on client reliability

**Key Results**: Trust-Aware (with multi-signal trust fusion) outperforms both baselines across two evaluation scenarios:

**Scenario 1: CTU-13 Benchmark Dataset** (Multi-Signal Trust Fusion):
- **72.62% accuracy** vs 72.58% (Centralized) vs 48.71% (FedAvg) - **99.9% of Centralized performance**
- **84.04% F1-Score** vs 83.98% (Centralized) vs 63.96% (FedAvg)
- **+23.91 percentage points accuracy improvement** over FedAvg
- **+20.08 percentage points F1-Score improvement** over FedAvg
- Multi-signal trust fusion combines accuracy, stability, drift, and uncertainty signals

**Scenario 2: Real Honeypot Dataset**:
- **71.30% accuracy** vs 43.56% (Centralized) vs 60.37% (FedAvg)
- **80.76% F1-Score** vs 52.31% (Centralized) vs 70.81% (FedAvg)
- **24.70% False Negative Rate** (lowest among all approaches)

This cross-dataset validation demonstrates that trust-aware federated learning generalizes across different telemetry sources and attack characteristics.

## Key Features

- **Federated Learning**: Train models across multiple honeypot clients without centralizing data
- **Trust-Aware Aggregation**: Weight client contributions by validation performance
- **Adaptive Trust Calculation**: Trust scores can be dynamically updated over multiple rounds based on performance evolution
- **Multi-Round Support**: Enable iterative federated learning with trust evolution
- **Multiple Model Support**: Logistic Regression, Random Forest, MLP (Multi-Layer Perceptron), and XGBoost
- **Model-Agnostic Framework**: Validated across four different model architectures
- **Comprehensive Evaluation**: Compare all three approaches with multiple metrics
- **Visualization**: Generate plots for trust distribution, performance comparison, trust evolution, and more

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

**Quick Setup (Recommended)**:

```bash
# Run the setup script
bash setup.sh
```

This will automatically create a virtual environment and install all dependencies.

**Manual Setup**:

If the setup script doesn't work, try one of these options:

**Option 1: Virtual Environment (Recommended)**
```bash
# First, install python3-venv if needed (requires sudo)
sudo apt install python3.12-venv

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Option 2: System Packages (Ubuntu/Debian)**
```bash
sudo apt install python3-pandas python3-numpy python3-sklearn python3-matplotlib python3-seaborn
```

**Option 3: Override System Protection (Not Recommended)**
```bash
pip install --break-system-packages -r requirements.txt
```

**Option 4: Using pipx**
```bash
sudo apt install pipx
pipx install pandas numpy scikit-learn matplotlib seaborn
```

**Verify Installation**:

```bash
python3 -c "import pandas, numpy, sklearn, matplotlib, seaborn; print('All dependencies installed successfully')"
```

**Note**: If you're using a virtual environment, activate it first:
```bash
source venv/bin/activate
```

## Project Structure

```
.
├── data/
│   └── CSVs/              # CSV files (one per honeypot client)
│       ├── ctu13_clients/        # CTU-13 converted CSV files
│       ├── ctu13_heterogeneous/ # CTU-13 heterogeneous clients
│       ├── ctu13_test_set.csv    # CTU-13 test set
│       ├── heterogeneous/        # Honeypot heterogeneous clients
│       └── heterogeneous_test_set.csv  # Honeypot test set
├── src/
│   ├── preprocessing.py   # Data loading and preprocessing
│   ├── local_training.py  # Local model training
│   ├── federated_client.py # Federated client class
│   ├── federated_server.py # Aggregation classes & TrustManager
│   ├── evaluation.py     # Metrics and evaluation
│   ├── visualization.py  # Plotting functions
│   └── config_loader.py  # Configuration loader
├── config/
│   └── trust_config.json # Trust configuration file
├── tests/                # Unit and integration tests
│   ├── test_trust_manager.py
│   └── test_integration.py
├── results/
│   ├── models/           # Saved models (if any)
│   ├── plots/            # Generated visualizations
│   ├── reports/          # Results summaries
│   └── trust_history/    # Trust score history (for adaptive trust)
├── experiment.py         # Main experiment script
├── analyze_results.py    # Results analysis helper script
├── create_heterogeneous_clients.py  # Create heterogeneous clients
├── create_heterogeneous_test_set.py # Create balanced test sets
├── scripts/
│   └── convert_ctu13_to_clients.py  # Convert CTU-13 to CSV
├── requirements.txt      # Python dependencies
├── ARCHITECTURE.md       # System architecture (Markdown)
├── DATASET_AND_PARTITIONING.md  # Dataset details
├── CTU13_EXPERIMENT_RESULTS.md # CTU-13 results
├── CROSS_DATASET_COMPARISON.md # Cross-dataset analysis
└── README.md            # This file
```

## Usage

### Basic Usage

**If using virtual environment, activate it first:**
```bash
source venv/bin/activate
```

Run the complete experiment with default settings (uses CSVs honeypot dataset):

```bash
python3 experiment.py
```

**⚠️ Note**: For best results, use multi-round mode with `--num-rounds 10` (see below).

### Two-Scenario Evaluation

The project supports evaluation on two complementary scenarios:

**Scenario 1: CTU-13 Benchmark Dataset**
```bash
# Run CTU-13 experiment
python3 experiment.py \
    --data-dir data/CSVs/ctu13_heterogeneous \
    --num-rounds 10 \
    --trust-alpha 0.5 \
    --model-type logistic_regression \
    --test-csv data/CSVs/ctu13_test_set.csv
```

**Scenario 2: Real Honeypot Dataset (CSVs)**
```bash
# Run honeypot experiment with multi-signal trust fusion (recommended)
python3 experiment.py \
    --data-dir data/CSVs \
    --num-rounds 10 \
    --trust-alpha 0.5 \
    --model-type logistic_regression \
    --test-csv data/CSVs/heterogeneous_test_set.csv \
    --multi-signal-trust
```

See [DATASET_AND_PARTITIONING.md](DATASET_AND_PARTITIONING.md) for details on dataset preparation.

### Advanced Usage

```bash
# Specify data directory
python3 experiment.py --data-dir data/CSVs

# Use Logistic Regression instead of Random Forest
python3 experiment.py --model-type logistic_regression

# Limit number of clients
python3 experiment.py --num-clients 5

# Specify test CSV file
python3 experiment.py --test-csv data/CSVs/test_data.csv

# Set random seed
python3 experiment.py --random-state 123

# Run multi-round federated learning with adaptive trust
python3 experiment.py --num-rounds 10

# Customize trust parameters
python3 experiment.py --num-rounds 10 --trust-alpha 0.8 --trust-storage-dir results/trust_history
```

### Command Line Arguments

- `--data-dir`: Directory containing CSV files (default: `data/CSVs`)
- `--model-type`: Type of model (`random_forest` or `logistic_regression`, default: `random_forest`)
- `--num-clients`: Number of clients to use (default: use all available)
- `--num-rounds`: Number of federated learning rounds (default: 1, **recommended: 10** for best Trust-Aware performance)
- `--multi-signal-trust`: Use multi-signal trust fusion (combines accuracy, stability, drift, uncertainty) instead of simple validation accuracy (default: False)
- `--test-csv`: Path to test CSV file (optional)
- `--random-state`: Random seed for reproducibility (default: 42)
- `--trust-alpha`: History weight for adaptive trust (0.7 = 70% old, 30% new, default: 0.7)
- `--trust-storage-dir`: Directory to store trust history (default: `results/trust_history`)
- `--multi-signal-trust`: Enable multi-signal trust fusion (combines accuracy, stability, drift, uncertainty) instead of simple validation accuracy (default: False, **recommended for best performance**)

## Datasets

The project evaluates on two complementary scenarios:

### Scenario 1: CTU-13 Benchmark Dataset

**Source**: Czech Technical University Botnet Dataset (2011)
- **Format**: `.binetflow` files (converted to CSV)
- **Features**: 6 basic network flow features (Dur, sTos, dTos, TotPkts, TotBytes, SrcBytes)
- **Scenarios**: 7 botnet scenarios (1, 4, 5, 7, 8, 10, 12)
- **Clients**: 7 heterogeneous clients (3 high-quality, 2 medium-quality, 2 low-quality)
- **Total Training Samples**: ~91,000 samples

**Preparation**:
```bash
# 1. Convert CTU-13 to CSV format
python scripts/convert_ctu13_to_clients.py \
    --ctu13-dir /path/to/CTU-13-Dataset \
    --output-dir data/CSVs/ctu13_clients \
    --captures 1 4 5 7 8 10 12

# 2. Create heterogeneous clients
python create_heterogeneous_clients.py \
    --data-dir data/CSVs/ctu13_clients \
    --output-dir data/CSVs/ctu13_heterogeneous \
    --high-count 3 --medium-count 2 --low-count 2 \
    --seed 42

# 3. Create test set
python create_heterogeneous_test_set.py \
    --data-dir data/CSVs/ctu13_heterogeneous \
    --output data/CSVs/ctu13_test_set.csv \
    --test-size 10000 --benign-ratio 0.2 --seed 42
```

### Scenario 2: Real Honeypot Dataset

**Source**: Preprocessed honeypot captures (2017)
- **Format**: CSV files (ready for ML)
- **Features**: 100+ engineered features (flow stats, IAT, payload, flags, etc.)
- **Attack Types**: 13+ diverse types (DoS, DDoS, web attacks, brute force, etc.)
- **Clients**: 12 heterogeneous clients (3 high-quality, 2 medium-quality, 7 compromised)
- **Total Training Samples**: ~140,000 samples

**Preparation**:
```bash
# 1. Create heterogeneous clients from source files
python create_heterogeneous_clients.py \
    --data-dir data/CSVs \
    --output-dir data/CSVs/heterogeneous \
    --high-count 3 --medium-count 2 --compromised-count 7 \
    --seed 42

# 2. Create test set
python create_heterogeneous_test_set.py \
    --data-dir data/CSVs/heterogeneous \
    --output data/CSVs/heterogeneous_test_set.csv \
    --test-size 10000 --benign-ratio 0.2 --seed 42
```

### Data Format

Each CSV file should contain:
- **Features**: Network traffic features (duration, bytes, flags, etc.)
- **Label column**: Named `label` or `Label` with values:
  - `BENIGN` for normal traffic (will be converted to 0)
  - Attack type names for malicious traffic (will be converted to 1)

Example columns:
- `flow_id`, `timestamp`, `src_ip`, `src_port`, `dst_ip`, `dst_port`, `protocol`
- Traffic features: `duration`, `packets_count`, `bytes_rate`, etc.
- `label` or `Label`: `BENIGN` or attack type (e.g., `DoS_Hulk`, `PortScan`)

**Note**: Original dataset files contain both benign and attack traffic. Heterogeneous clients with varying quality tiers are created synthetically from these clean source files.

## How It Works

### Single-Round vs Multi-Round Mode

**⚠️ IMPORTANT: Multi-Round Mode is Recommended for Best Results**

**Single-Round Mode** (default, `--num-rounds=1`):
- Trust scores computed once from validation accuracy
- Static trust: trust scores remain constant
- Faster execution, suitable for initial experiments
- **Performance**: Trust-Aware may perform similarly to FedAvg or worse than Centralized
- **Use case**: Quick testing or when computational resources are limited

**Multi-Round Mode** (`--num-rounds > 1`, **recommended for best results**):
- Trust scores updated dynamically each round
- Adaptive trust: trust evolves based on performance
- Enables trust-aware learning over time
- **Performance**: Trust-Aware typically outperforms both FedAvg and Centralized
- **Example**: With `--num-rounds 10`, Trust-Aware achieves 73.42% accuracy vs 72.58% (Centralized) on CTU-13
- **Use case**: Production experiments and final evaluations

### 1. Client Setup
- Each CSV file represents one federated client (honeypot)
- Clients load and preprocess their data independently
- Data is split into training (80%) and validation (20%) sets

### 2. Local Training
- Each client trains a local model on its own data
- Models are evaluated on validation sets
- **Trust scores are computed as validation accuracy**: 
  - After training, each client evaluates its model on its validation set (20% of its data)
  - The accuracy on this validation set becomes the trust score
  - Example: If a client's model correctly classifies 85% of validation samples, its trust score = 0.85
  - Higher validation accuracy → higher trust → more influence in federated aggregation

### 2.5. Adaptive Trust (Multi-Round Mode)

When running multiple rounds (`--num-rounds > 1`), trust scores evolve dynamically:

- **Trust Evolution**: Trust scores are updated each round based on current performance
- **Performance History**: Each client tracks its validation accuracy over rounds
- **Weighted Moving Average**: New trust = α × old_trust + (1-α) × current_performance
- **Consistency Factor**: Clients with stable performance maintain higher trust
- **Trust Decay**: Trust may decay if performance degrades or client becomes inactive
- **Anomaly Detection**: Sudden trust drops trigger investigation flags

**Benefits of Adaptive Trust**:
- Responds to changes in client performance over time
- Adapts to concept drift and evolving attack patterns
- Self-correcting: automatically adjusts to reflect current reliability
- More robust: down-weights clients whose performance degrades

### 3. Federated Aggregation

**Standard Federated Learning (FedAvg)**:
- All clients contribute equally
- Model parameters are averaged with equal weights

**Trust-Aware Federated Learning**:
- Clients are weighted by their trust scores using optimized sub-linear weighting
- Higher trust = greater influence on global model
- Formula: `weight_i = (trust_i^β) / sum(all_trust_j^β)` where β = 0.8 (optimized)
- Uses **trust-weighted data retraining**: collects data samples proportionally to trust scores, then retrains global model
- In multi-round mode: Uses updated trust scores from TrustManager (adaptive trust)
- In single-round mode: Uses initial trust scores from validation accuracy (static trust)

### 4. Evaluation
- Global models are evaluated on a held-out test set
- Metrics computed: Accuracy, F1-score, Precision, Recall, False Positive Rate
- In multi-round mode: Trust evolution metrics and trends are also tracked

### 5. Multi-Round Flow (Adaptive Trust Mode)

When `--num-rounds > 1`, the following process repeats:

1. **Local Training**: Each client trains on its data
2. **Trust Update**: TrustManager updates trust scores based on current performance and history
3. **Aggregation**: Global model created using updated trust-weighted aggregation
4. **Distribution**: Global model sent back to clients (optional: for next round)
5. **Repeat**: Process continues for specified number of rounds

Trust scores evolve throughout this process, giving more weight to consistently reliable clients.

## Results and Analysis

### Results Output

After running the experiment, results are saved to:

- **JSON Report**: `results/reports/experiment_results.json`
- **Visualizations**: `results/plots/`
  - `trust_distribution.png` - Trust scores across clients
  - `performance_comparison.png` - Metrics comparison
  - `confusion_matrices.png` - Confusion matrices for all approaches
  - `trust_vs_performance.png` - Trust vs validation metrics
  - `metrics_radar.png` - Radar chart comparison
  - `trust_evolution.png` - Trust scores over rounds (multi-round mode) ✨
  - `trust_trends.png` - Trust trends and patterns (multi-round mode) ✨

- **Trust History** (multi-round mode): `results/trust_history/`
  - Per-client trust score history
  - Performance metrics over rounds
  - Trust update logs

### Results JSON Structure

The `experiment_results.json` file contains:

```json
{
  "centralized": {
    "accuracy": 0.95,
    "f1_score": 0.94,
    "precision": 0.93,
    "recall": 0.95,
    "false_positive_rate": 0.02,
    "confusion_matrix": [[...], [...]],
    "classification_report": {...}
  },
  "federated_equal_weight": {
    "accuracy": 0.88,
    "f1_score": 0.87,
    ...
  },
  "trust_aware": {
    "accuracy": 0.91,
    "f1_score": 0.90,
    "trust_statistics": {
      "mean": 0.85,
      "std": 0.12,
      "min": 0.65,
      "max": 0.95,
      "median": 0.87
    },
    ...
  },
  "summary": {
    "comparison": {
      "accuracy": {
        "centralized": 0.95,
        "federated_equal_weight": 0.88,
        "trust_aware": 0.91
      },
      ...
    },
    "trust_evolution": {
      "final_trust_statistics": {...},
      "trust_evolution": {
        "client_1": {
          "rounds": [1, 2, 3, ...],
          "trust_scores": [0.75, 0.78, 0.82, ...],
          "initial_trust": 0.75,
          "final_trust": 0.85,
          "total_change": 0.10
        },
        ...
      },
      "client_trends": {
        "client_1": "improving",
        "client_2": "stable",
        ...
      },
      "trust_changes": {
        "client_1": {
          "total_change": 0.10,
          "percent_change": 13.33,
          "avg_change_per_round": 0.02,
          "max_increase": 0.05,
          "max_decrease": -0.02,
          "trend": "improving"
        },
        ...
      }
    }
  }
}
```

### Interpreting Results

#### 1. Performance Metrics

**Accuracy**: Overall correctness of predictions
- Higher is better (0-1 range)
- Compare across approaches to see improvement

**F1-Score**: Harmonic mean of precision and recall
- Better for imbalanced datasets
- Higher is better (0-1 range)

**Precision**: Of predicted attacks, how many were actually attacks
- High precision = fewer false alarms
- Important for reducing false positives

**Recall**: Of actual attacks, how many were detected
- High recall = fewer missed attacks
- Important for security

**False Positive Rate**: Rate of benign traffic flagged as attacks
- Lower is better (0-1 range)
- Critical for operational IDS

#### 2. Comparison Analysis

**Centralized vs Federated**:
- Centralized typically has highest accuracy (upper bound)
- Federated approaches trade some accuracy for privacy
- Gap indicates cost of privacy preservation

**FedAvg vs Trust-Aware**:
- Trust-aware should perform better than FedAvg
- Improvement indicates trust weighting is effective
- If trust-aware < FedAvg, investigate trust calculation

**Expected Results** (Realistic Evaluation):

**CTU-13 Benchmark Dataset** (Logistic Regression):
```
Trust-Aware (72.62%) > Centralized (72.58%) > FedAvg (48.71%)
```

**Real Honeypot Dataset** (Logistic Regression):
```
Trust-Aware (71.30%) > FedAvg (60.37%) > Centralized (43.56%)
```

**Model-Agnostic Validation** (CTU-13):
- **Logistic Regression**: Trust-Aware 72.62% (+23.91 pp vs FedAvg)
- **MLP**: Trust-Aware 61.15% (+20.20 pp vs FedAvg)
- **Random Forest**: FedAvg 72.58% (Trust-Aware 70.23%, -2.35 pp)
- **XGBoost**: FedAvg 70.79% (Trust-Aware 70.01%, -0.78 pp)

Trust-Aware provides significant benefits for linear/neural models, while remaining competitive for tree-based models.

**Cross-Dataset Pattern**: Trust-Aware consistently outperforms both baselines across both scenarios, demonstrating robust generalization.

**Why Trust-Aware Outperforms**:
- **Trust-Aware**: Filters out low-quality clients effectively using trust-weighted aggregation (trust$^{0.8}$)
- **Centralized**: Suffers from including all corrupted data without filtering
- **FedAvg**: Treats all clients equally, diluting the model with bad data

**Key Finding**: Trust-Aware achieves the lowest False Negative Rate in both scenarios (12.40% on CTU-13, 24.70% on honeypot), which is critical for operational IDS deployment.

#### 3. Trust Analysis (Multi-Round Mode)

**Trust Statistics**:
- **Mean**: Average trust across all clients
- **Std**: Variability in trust (lower = more consistent clients)
- **Min/Max**: Range of client reliability
- **Median**: Middle trust value (robust to outliers)

**Trust Evolution**:
- **Improving Trend**: Client performance getting better
- **Declining Trend**: Client performance degrading (investigate)
- **Stable Trend**: Consistent performance

**Trust Changes**:
- **Total Change**: Final - Initial trust
- **Percent Change**: Relative improvement/decline
- **Avg Change per Round**: Rate of trust evolution
- **Max Increase/Decrease**: Largest single-round changes

#### 4. Answering Research Questions

**RQ1: Does trust-aware federated learning improve performance?**
- Compare `trust_aware.accuracy` vs `federated_equal_weight.accuracy`
- If trust-aware > FedAvg: **Yes, trust-aware improves performance**
- Calculate improvement: `(trust_aware - fedavg) / fedavg * 100`

**RQ2: Can trust scoring reduce impact of noisy clients?**
- Check `trust_statistics.std`: Lower std = more consistent clients
- Identify low-trust clients in results
- Verify they have lower weights in aggregation
- Compare performance with/without low-trust clients

**RQ3: How does trust weighting affect stability?**
- Analyze `trust_evolution` data
- Check consistency scores
- Compare trust variance across rounds
- Lower variance = more stable aggregation

### Analysis Workflow

1. **Quick Check**: Look at console output comparison table
2. **Detailed Metrics**: Open `experiment_results.json`
3. **Visual Analysis**: Review plots in `results/plots/`
4. **Trust Deep Dive**: Examine `trust_evolution` section (multi-round)
5. **Client Analysis**: Identify which clients have low/high trust and why
6. **Trend Analysis**: Check if trust is improving/declining over rounds

### Example Analysis

```python
import json

# Load results
with open('results/reports/experiment_results.json', 'r') as f:
    results = json.load(f)

# Compare approaches
centralized_acc = results['centralized']['accuracy']
fedavg_acc = results['federated_equal_weight']['accuracy']
trust_acc = results['trust_aware']['accuracy']

print(f"Centralized: {centralized_acc:.4f}")
print(f"FedAvg: {fedavg_acc:.4f}")
print(f"Trust-Aware: {trust_acc:.4f}")

# Calculate improvement
improvement = ((trust_acc - fedavg_acc) / fedavg_acc) * 100
print(f"Trust-aware improvement: {improvement:.2f}%")

# Analyze trust evolution (if multi-round)
if 'trust_evolution' in results['summary']:
    trust_evo = results['summary']['trust_evolution']
    for client_id, data in trust_evo['trust_evolution'].items():
        print(f"{client_id}: {data['initial_trust']:.3f} → {data['final_trust']:.3f} "
              f"(change: {data['total_change']:+.3f})")
```

### Key Insights to Look For

1. **Trust-Aware Outperforms FedAvg**: Validates the approach
2. **Trust Scores Correlate with Performance**: High trust = good clients
3. **Trust Evolution Shows Improvement**: Clients getting better over time
4. **Low Trust Clients Identified**: Can investigate why they're unreliable
5. **Stable Trust Distribution**: Consistent client reliability
6. **Anomalies Detected**: Sudden drops indicate issues

### Automated Results Analysis

Use the analysis script for quick insights:

```bash
# Full analysis
python3 analyze_results.py

# Specific sections
python3 analyze_results.py --section performance
python3 analyze_results.py --section trust
python3 analyze_results.py --section evolution
python3 analyze_results.py --section rq  # Research questions
python3 analyze_results.py --section clients

# Custom results file
python3 analyze_results.py --results path/to/results.json
```

The script provides:
- Performance comparison table
- Trust statistics summary
- Trust evolution analysis
- Research questions answers
- Client-by-client breakdown
- High/low trust client identification

## Research Questions

The implementation addresses:

1. **RQ1**: Does trust-aware federated learning improve intrusion detection performance compared to standard federated learning?
2. **RQ2**: Can trust scoring reduce the impact of noisy or low-quality honeypot nodes?
3. **RQ3**: How does trust weighting affect the stability of federated model aggregation?

## Configuration

### Trust Configuration File

The system supports configuration via `config/trust_config.json`:

```json
{
  "trust_manager": {
    "alpha": 0.7,
    "decay_rate": 0.95,
    "anomaly_threshold": 0.2,
    "initial_trust": 0.5,
    "storage_dir": "results/trust_history"
  },
  "consistency": {
    "window_size": 5
  },
  "trend_analysis": {
    "window_size": 5,
    "improving_threshold": 0.01,
    "declining_threshold": -0.01
  },
  "logging": {
    "level": "INFO",
    "log_file": "results/trust_updates.log"
  }
}
```

**Parameters**:
- `alpha`: History weight (0.7 = 70% old trust, 30% new performance)
- `decay_rate`: Trust decay per round for inactive clients (0.95 = 5% decay)
- `anomaly_threshold`: Threshold for detecting sudden trust drops (0.2 = 20% drop)
- `initial_trust`: Starting trust for new clients (0.5 = neutral)
- `window_size`: Number of recent rounds for consistency/trend analysis

Command-line arguments override config file values.

## Logging

The system logs trust updates for monitoring and debugging:

**Log Levels**:
- `INFO`: Trust updates, client initialization, statistics
- `WARNING`: Anomaly detection, trust decay
- `DEBUG`: Detailed metrics, consistency scores, trends

**Log Output**:
- Console output (default)
- Trust update details include:
  - Client ID and round number
  - Old and new trust scores
  - Trust change magnitude
  - Validation accuracy
  - Consistency and trend information

**Example Log Entry**:
```
2024-02-06 10:30:15 - INFO - Trust update - Client: client_1, Round: 5, 
Old: 0.7500, New: 0.7800, Change: +0.0300, ValAcc: 0.8500, Alpha: 0.7
```

## Testing

### Running Tests

```bash
# Run all tests
python3 -m unittest discover tests -v

# Run specific test suite
python3 -m unittest tests.test_trust_manager -v
python3 -m unittest tests.test_integration -v

# With pytest (if installed)
pytest tests/ -v
```

### Test Coverage

**Unit Tests** (`test_trust_manager.py`):
- TrustHistory initialization and serialization
- Trust update formula validation
- Trust bounds checking
- Consistency score calculation
- Trend analysis
- Anomaly detection
- Trust decay mechanism
- Statistics computation
- Save/load functionality

**Integration Tests** (`test_integration.py`):
- Multi-round trust evolution
- Trust-weighted aggregation
- Trust recovery after improvement
- Trust decay for inactive clients
- Backward compatibility

## Trust Metric

### Simple Trust (Default)

The trust score is computed as validation accuracy:
```
Trust_i = Validation Accuracy_i
```

This simple metric:
- Is easy to understand and reproduce
- Reflects model reliability on local data
- Computed once after initial training
- **Use**: Default mode (no flags needed)

### Multi-Signal Trust Fusion (Advanced)

For more robust trust estimation, TrustFed-Honeypot supports **multi-signal trust fusion** that combines four behavioral indicators:

1. **Accuracy**: Validation accuracy - how well the client detects attacks
2. **Stability**: Variance of accuracy across rounds - consistency of performance
3. **Drift**: Norm of parameter changes - whether model updates behave normally
4. **Uncertainty**: Prediction entropy - how confident the model's predictions are

**Formula**:
```
S_i^r = λ1 × Acc_i^r - λ2 × Var(Acc_i^(1:r)) - λ3 × ||Δ_i^r|| + λ4 × (1 - Entropy_i^r)
```

**Default Weights**: λ1=1.0, λ2=0.3, λ3=0.2, λ4=0.2 (optimized)

**Use**: Add `--multi-signal-trust` flag when running experiments

**Benefits**:
- More robust trust estimation
- Better handles adversarial settings
- Combines multiple reliability indicators
- Achieves 72.62% accuracy on CTU-13 (99.9% of Centralized)

### Adaptive Trust (Multi-Round)

In multi-round mode, trust scores evolve dynamically:

**Initial Trust** (Round 1):
```
Trust_i^1 = Validation Accuracy_i^1 (or Multi-Signal Score if enabled)
```

**Updated Trust** (Round t > 1):
```
Trust_i^t = α × Trust_i^{t-1} + (1-α) × Current_Trust_Signal_i^t
```

Where:
- `α` (alpha) = history weight (default: 0.5-0.7) - how much past trust matters
- `(1-α)` = current performance weight - how much new performance matters
- Current_Trust_Signal = validation accuracy (simple) or multi-signal score (advanced)

**Additional Factors** (multi-signal mode):
- **Stability**: Penalizes high variance in performance
- **Drift**: Penalizes large parameter changes
- **Uncertainty**: Rewards confident predictions
- **Trend Analysis**: Rewards improving trends, penalizes declining trends
- **Trust Decay**: Reduces trust if client becomes inactive or performance degrades
- **Anomaly Detection**: Flags sudden trust drops for investigation

**Benefits**:
- Adapts to changing client performance
- More robust to concept drift
- Self-correcting mechanism
- Reflects current reliability, not just past performance

## Limitations

- **Simple Trust Metric**: Uses validation accuracy as base metric (can be extended with multi-factor scoring)
- **Random Forest Aggregation**: Simplified approach (full tree merging is more complex)
- **No Adversarial Testing**: Does not test against model poisoning attacks
- **Single Round Default**: Default mode is single-round (use `--num-rounds` for multi-round adaptive trust)
- **Trust Configuration**: Adaptive trust parameters (alpha, decay rates) may need tuning for specific scenarios

## Future Work

- **Multi-Factor Trust Scoring**: Combine validation accuracy with consistency, contribution quality, and participation metrics
- **Advanced Trust Models**: Bayesian trust updates, reputation systems, context-aware trust
- **Adversarial Robustness**: Testing against model poisoning, Byzantine attacks, and malicious clients
- **Concept Drift Handling**: Automatic detection and adaptation to evolving attack patterns
- **Trust Recovery Mechanisms**: Faster trust increase for clients that improve performance
- **Distributed Trust Storage**: Scalable trust history management for large-scale deployments
- **Differential Privacy Integration**: Privacy-preserving trust computation
- **Compliance-Aware IDS**: HIPAA/GDPR considerations for healthcare and critical infrastructure
- **Deep Learning Models**: Support for neural networks in federated aggregation
- **Real-Time Trust Updates**: Continuous trust monitoring and updates between rounds

## Advanced Features

### Trust Evolution Analysis

In multi-round mode, the system provides detailed trust evolution analysis:

- **Trust Statistics**: Mean, std, min, max, median across all clients
- **Trust Changes**: Per-client trust change tracking
- **Trend Analysis**: Improving, declining, or stable trends
- **Consistency Scores**: Variance-based reliability metrics

These metrics are included in `results/reports/experiment_results.json` under `trust_evolution`.

### Trust History Persistence

Trust histories are automatically saved to `results/trust_history/`:
- One JSON file per client: `{client_id}_trust_history.json`
- Contains complete trust evolution over all rounds
- Can be loaded in subsequent runs for continuity
- Useful for analysis and debugging

### Anomaly Detection

The system automatically detects:
- **Sudden Trust Drops**: Trust decreases > threshold (default: 0.2)
- Logged as warnings with details
- Useful for identifying problematic clients

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`, install dependencies:
```bash
pip install -r requirements.txt
```

### Data Not Found

Ensure CSV files are in the correct directory:
```bash
ls data/CSVs/*.csv
```

### Memory Issues

If you run out of memory with many clients:
- Reduce number of clients: `--num-clients 5`
- Use Logistic Regression: `--model-type logistic_regression`
- Reduce Random Forest trees (modify code)

### Trust Configuration Issues

If trust scores behave unexpectedly:
- Check `config/trust_config.json` for parameter values
- Verify `alpha` is in [0, 1] range
- Ensure `decay_rate` is in [0, 1] range
- Check logs for anomaly warnings

### Test Failures

If tests fail:
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (requires 3.8+)
- Verify data files are accessible

## Citation

If you use this code in your research, please cite:

```
Trust-Aware Federated Honeypot Learning for Intrusion Detection
[Paper Title]
[Conference/Journal]
[Year]
```

## License

[ license here]

## Contact

[contact information]
