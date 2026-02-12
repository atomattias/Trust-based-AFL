# Dataset and Data Partitioning Documentation

## 1. Dataset Overview

### 1.1 Source Dataset

The project uses the **CTU-13 Dataset** (also known as the CTU University Dataset), which is a well-known dataset for network intrusion detection research. The CTU-13 dataset contains:

- **Real network traffic captures** from the CTU University network
- **Multiple attack scenarios**: Botnet, DDoS, DoS, PortScan, SSH Brute Force, Web attacks (SQL Injection, XSS, Brute Force), FTP Patator, Heartbleed, and more
- **Benign traffic**: Normal network traffic from university network
- **Labeled data**: Each flow is labeled as either BENIGN or a specific attack type

### 1.2 Dataset Structure

The original CTU-13 dataset is organized by attack type:
- Each attack type has its own CSV file
- Files contain network flow features (duration, bytes, packets, flags, etc.)
- Labels indicate whether traffic is benign or a specific attack type

### 1.3 Data Preprocessing

The raw CTU-13 data is preprocessed to:
1. **Extract features**: Network flow characteristics (114 features)
2. **Prepare labels**: Convert attack labels to binary (0 = benign, 1 = attack)
3. **Normalize features**: Ensure consistent feature ranges
4. **Handle missing values**: Clean and impute missing data

---

## 2. Heterogeneous Client Creation

### 2.1 Motivation

To simulate realistic federated learning scenarios with heterogeneous client quality, we create clients with varying data quality levels:

- **High-Quality Clients**: Clean, reliable data (minimal or no corruption)
- **Medium-Quality Clients**: Moderate data quality (some noise)
- **Low-Quality Clients**: Lower data quality (more noise)
- **Compromised Clients**: Severely corrupted data (high label noise and feature corruption)

### 2.2 Client Creation Process

The `create_heterogeneous_clients.py` script creates heterogeneous clients from source files:

#### 2.2.1 Source Files

Source files are located in:
- `data/CSVs/` - Original or mixed CSV files
- `~/Papers/CSVs/` - Alternative source location

Source files are typically named:
- `mixed_*.csv` - Files containing both benign and attack samples (preferred)
- `*.csv` - Original attack files

#### 2.2.2 Quality Tiers

**High-Quality Clients**:
- **Label Noise**: 0% (no label corruption)
- **Feature Corruption**: 0% (no feature corruption)
- **Target Size**: 25,000 samples per client
- **Trust Score Range**: High (typically 0.95-1.00 in ideal scenarios)
- **Purpose**: Represent reliable honeypots with clean data

**Medium-Quality Clients**:
- **Label Noise**: 10-20% (moderate label corruption)
- **Feature Corruption**: 10-20% (moderate feature corruption)
- **Target Size**: 5,000 samples per client
- **Trust Score Range**: Medium (typically 0.60-0.75)
- **Purpose**: Represent honeypots with some data quality issues

**Low-Quality Clients**:
- **Label Noise**: 30-40% (higher label corruption)
- **Feature Corruption**: 30-40% (higher feature corruption)
- **Target Size**: 3,000 samples per client
- **Trust Score Range**: Low (typically 0.40-0.55)
- **Purpose**: Represent honeypots with significant data quality issues

**Compromised Clients**:
- **Label Noise**: 99% (extreme label corruption)
- **Feature Corruption**: 95% (extreme feature corruption)
- **Target Size**: 25,000 samples per client
- **Trust Score Range**: Very Low (typically 0.02-0.37)
- **Purpose**: Represent compromised or malicious honeypots that provide corrupted data

#### 2.2.3 Corruption Methods

**Label Noise**:
- Randomly flips labels with specified probability
- Example: 99% label noise means 99% of labels are randomly flipped
- Simulates mislabeling or adversarial data corruption

**Feature Corruption**:
- Randomly corrupts feature values with specified probability
- Methods include:
  - Random value injection
  - Feature scaling corruption
  - Feature replacement with noise
- Simulates sensor errors or data transmission corruption

#### 2.2.4 Client Naming Convention

Created clients follow this naming pattern:
```
client_{ID}_{quality_tier}_{source_name}.csv
```

Examples:
- `client_1_high_quality_botnet_ares.csv`
- `client_4_medium_quality_dos_hulk.csv`
- `client_6_compromised_dos_slowloris.csv`
- `client_10_compromised_ssh_patator-new.csv`

### 2.3 Client Distribution

In our experiments, we typically use:
- **3 High-Quality Clients**: Clean data, large datasets
- **2 Medium-Quality Clients**: Moderate quality, smaller datasets
- **7 Compromised Clients**: Severely corrupted data, large datasets

**Total**: 12 clients (excluding problematic clients like heartbleed)

---

## 3. Training and Test Set Partitioning

### 3.1 Training Set Creation

#### 3.1.1 Client-Level Splitting

Each client's data is split locally:

1. **Load Client Data**: Each client loads its CSV file
2. **Train/Validation Split**: 80% training, 20% validation
   - Training set: Used for local model training
   - Validation set: Used for trust score computation
3. **No Test Set at Client Level**: Test set is created separately at the global level

#### 3.1.2 Training Data Characteristics

- **Total Training Data**: ~140,000 samples across all clients
- **Distribution**: Varies by client quality
  - High-quality: ~60,000 samples (3 clients × 20,000)
  - Medium-quality: ~8,000 samples (2 clients × 4,000)
  - Compromised: ~72,000 samples (7 clients × ~10,000)
- **Class Distribution**: Varies by client
  - High-quality clients: Often 100% attack (single-class)
  - Medium-quality clients: ~25% benign, 75% attack
  - Compromised clients: ~96% benign, 4% attack (due to corruption)

### 3.2 Test Set Creation

#### 3.2.1 Test Set Requirements

A proper test set must:
1. **Be Completely Separate**: No overlap with training data
2. **Match Training Distribution**: Similar to training data distribution
3. **Include Both Classes**: Both benign and attack samples
4. **Be Representative**: Reflect real-world scenarios

#### 3.2.2 Heterogeneous Test Set Creation

The `create_heterogeneous_test_set.py` script creates a proper test set:

**Process**:
1. **Source**: Samples from heterogeneous client files (same distribution as training)
2. **Exclusion**: Excludes clients used for training
3. **Sampling**: 
   - Samples from all available client files
   - Combines benign and attack samples
   - Maintains desired class distribution
4. **Size**: 10,000 samples (configurable)
5. **Class Distribution**: 20% benign, 80% attack (realistic for honeypot scenarios)

**Output**: `data/CSVs/heterogeneous_test_set.csv`

#### 3.2.3 Test Set Characteristics

- **Total Samples**: 10,000
- **Benign Samples**: 2,000 (20%)
- **Attack Samples**: 8,000 (80%)
- **Source**: Heterogeneous client files (matches training distribution)
- **No Data Leakage**: Completely separate from training data
- **Representative**: Reflects realistic honeypot scenario

### 3.3 Data Leakage Prevention

#### 3.3.1 Critical Issue Identified

Initially, the test set was created incorrectly:
- **Problem**: Test set used training data from clients
- **Result**: Data leakage - models tested on data they had seen
- **Impact**: Unrealistic perfect accuracy (1.0000)

#### 3.3.2 Solution Implemented

1. **Reserve Test Set Before Training**:
   - Test set is created/reserved before any client training
   - Ensures complete separation

2. **Use Separate Files**:
   - Test set uses files not used for client training
   - Or uses pre-created heterogeneous test set

3. **Verify Separation**:
   - Check that test set files are not in client file list
   - Verify no overlap between training and test data

#### 3.3.3 Current Implementation

In `experiment.py`:
```python
# Reserve test file BEFORE setting up clients
if self.test_csv is None:
    # Prefer heterogeneous test set
    heterogeneous_test_file = Path('data/CSVs/heterogeneous_test_set.csv')
    if heterogeneous_test_file.exists():
        self.reserved_test_file = str(heterogeneous_test_file)
        # Use all heterogeneous files for clients
        attack_files_for_clients = attack_files
```

This ensures:
- Test set is completely separate
- No data leakage
- Realistic evaluation

---

## 4. Addressing Missing Corrupted Clients

### 4.1 Problem Statement

In realistic federated learning scenarios, we need to simulate:
- **Heterogeneous client quality**: Not all clients have the same data quality
- **Compromised clients**: Some clients may be compromised or provide corrupted data
- **Quality variation**: Clients with varying levels of data corruption

However, the original CTU-13 dataset does not include pre-corrupted clients. We need to create them synthetically.

### 4.2 Solution: Synthetic Corruption

#### 4.2.1 Corruption Strategy

We create compromised clients by applying synthetic corruption to clean source data:

**Label Noise**:
- Randomly flips labels with high probability (99% for compromised clients)
- Simulates mislabeling, adversarial attacks, or sensor errors
- Example: A benign sample (label=0) becomes attack (label=1) with 99% probability

**Feature Corruption**:
- Randomly corrupts feature values with high probability (95% for compromised clients)
- Methods:
  - Random value injection
  - Feature scaling corruption
  - Gaussian noise addition
- Simulates data transmission errors or malicious data manipulation

#### 4.2.2 Implementation Details

The `create_heterogeneous_clients.py` script implements corruption:

```python
def create_heterogeneous_client(
    source_file: str,
    output_file: str,
    quality_tier: str,
    target_size: int = None,
    random_state: int = 42
):
    # Load source data
    df = load_client_data(source_file)
    df = prepare_labels(df)
    
    # Apply corruption based on quality tier
    if quality_tier == 'compromised':
        label_noise = 0.99  # 99% label noise
        feature_corruption = 0.95  # 95% feature corruption
    elif quality_tier == 'low':
        label_noise = 0.40
        feature_corruption = 0.40
    elif quality_tier == 'medium':
        label_noise = 0.20
        feature_corruption = 0.20
    else:  # high
        label_noise = 0.0
        feature_corruption = 0.0
    
    # Apply label noise
    if label_noise > 0:
        mask = np.random.random(len(df)) < label_noise
        df.loc[mask, 'label'] = 1 - df.loc[mask, 'label']
    
    # Apply feature corruption
    if feature_corruption > 0:
        # Corrupt feature values
        # ... (implementation details)
    
    # Save corrupted client data
    df.to_csv(output_file, index=False)
```

#### 4.2.3 Corruption Levels

**Compromised Clients** (Extreme Corruption):
- Label Noise: 99%
- Feature Corruption: 95%
- Purpose: Simulate severely compromised honeypots
- Expected Trust: 0.02-0.37 (very low)

**Low-Quality Clients** (High Corruption):
- Label Noise: 40%
- Feature Corruption: 40%
- Purpose: Simulate honeypots with significant issues
- Expected Trust: 0.40-0.55

**Medium-Quality Clients** (Moderate Corruption):
- Label Noise: 20%
- Feature Corruption: 20%
- Purpose: Simulate honeypots with some issues
- Expected Trust: 0.60-0.75

**High-Quality Clients** (No Corruption):
- Label Noise: 0%
- Feature Corruption: 0%
- Purpose: Simulate reliable honeypots
- Expected Trust: 0.95-1.00 (in ideal scenarios)

### 4.3 Validation Set Strategy

#### 4.3.1 Clean vs Corrupted Validation

**For Compromised Clients**:
- **Training Data**: Corrupted (99% label noise, 95% feature corruption)
- **Validation Data**: Also corrupted (same corruption level)
- **Rationale**: 
  - Model trained on corrupted data learns wrong patterns
  - Validation on corrupted data shows poor performance
  - Trust score reflects the corrupted quality
  - Low trust score correctly identifies compromised client

**For Medium/Low-Quality Clients**:
- **Training Data**: Corrupted (moderate corruption)
- **Validation Data**: Clean (from original source file)
- **Rationale**:
  - Model trained on corrupted data may still learn some patterns
  - Validation on clean data shows actual model quality
  - Trust score reflects true client reliability
  - Enables differentiation between medium and low quality

**For High-Quality Clients**:
- **Training Data**: Clean
- **Validation Data**: Clean
- **Rationale**:
  - No corruption, so both are clean
  - Trust score reflects high quality

#### 4.3.2 Implementation

In `experiment.py`:
```python
if client_quality == 'compromised':
    # Use corrupted validation (no clean source)
    clean_validation_source = None
    print("COMPROMISED CLIENT: Using CORRUPTED validation")
elif client_quality in ['low', 'medium']:
    # Use clean validation source
    clean_validation_source = find_clean_source(attack_file)
    print("Using CLEAN validation source")
```

### 4.4 Trust Score Impact

#### 4.4.1 Trust Score Differentiation

The corruption strategy ensures clear trust score differentiation:

- **Compromised Clients**: Trust 0.02-0.37 (very low)
  - High corruption → poor model performance → low trust
  - Correctly identified as unreliable

- **Medium-Quality Clients**: Trust 0.45-0.76 (moderate)
  - Moderate corruption → moderate performance → medium trust
  - Correctly identified as moderately reliable

- **High-Quality Clients**: Trust varies (may be low due to single-class data)
  - No corruption → good performance → higher trust
  - However, single-class data (100% attack) can limit trust scores

#### 4.4.2 Trust-Aware Aggregation Benefit

The clear trust differentiation enables Trust-Aware to:
1. **Identify Compromised Clients**: Low trust scores (0.02-0.37)
2. **Down-Weight Bad Clients**: Minimal contribution to global model
3. **Up-Weight Good Clients**: Higher contribution from reliable clients
4. **Improve Global Model**: Better performance than equal-weight FedAvg

---

## 5. Data Flow Summary

### 5.1 Complete Pipeline

```
1. Source Data (CTU-13 Dataset)
   ↓
2. Create Heterogeneous Clients
   - High-quality: No corruption
   - Medium-quality: 20% corruption
   - Low-quality: 40% corruption
   - Compromised: 99% label noise, 95% feature corruption
   ↓
3. Client Training Data (80% split)
   - Each client splits its data: 80% train, 20% validation
   - Training data used for local model training
   ↓
4. Client Validation Data (20% split)
   - Used for trust score computation
   - Clean validation for medium/low clients
   - Corrupted validation for compromised clients
   ↓
5. Create Test Set
   - Sample from heterogeneous client files
   - 10,000 samples: 20% benign, 80% attack
   - Completely separate from training data
   ↓
6. Global Model Evaluation
   - Test on separate test set
   - No data leakage
   - Realistic performance metrics
```

### 5.2 Key Statistics

**Training Data**:
- Total: ~140,000 samples
- High-quality: ~60,000 samples
- Medium-quality: ~8,000 samples
- Compromised: ~72,000 samples

**Test Data**:
- Total: 10,000 samples
- Benign: 2,000 (20%)
- Attack: 8,000 (80%)
- Source: Heterogeneous clients (matches training distribution)

**Client Distribution**:
- 3 High-quality clients
- 2 Medium-quality clients
- 7 Compromised clients
- Total: 12 clients

---

## 6. Reproducibility

### 6.1 Creating Heterogeneous Clients

```bash
# Create heterogeneous clients from source files
python3 create_heterogeneous_clients.py \
    --data-dir data/CSVs \
    --output-dir data/CSVs/extreme_scenario_v4_from_papers \
    --high-count 3 \
    --medium-count 2 \
    --low-count 0 \
    --compromised-count 7 \
    --seed 42
```

### 6.2 Creating Test Set

```bash
# Create heterogeneous test set
python3 create_heterogeneous_test_set.py \
    --data-dir data/CSVs/extreme_scenario_v4_from_papers \
    --output data/CSVs/heterogeneous_test_set.csv \
    --test-size 10000 \
    --benign-ratio 0.2 \
    --seed 42 \
    --exclude heartbleed
```

### 6.3 Running Experiment

```bash
# Run experiment with proper test set
python3 experiment.py \
    --data-dir data/CSVs/extreme_scenario_v4_from_papers \
    --num-rounds 10 \
    --trust-alpha 0.5 \
    --model-type logistic_regression \
    --test-csv data/CSVs/heterogeneous_test_set.csv
```

---

## 7. Key Takeaways

1. **Dataset**: CTU-13 dataset with network traffic from various attack scenarios
2. **Client Creation**: Synthetic corruption creates heterogeneous client quality
3. **Training Split**: 80% train, 20% validation per client
4. **Test Set**: Separate heterogeneous test set (10,000 samples, 20% benign, 80% attack)
5. **No Data Leakage**: Test set completely separate from training data
6. **Corruption Strategy**: 
   - Compromised: 99% label noise, 95% feature corruption
   - Medium: 20% corruption
   - High: 0% corruption
7. **Validation Strategy**: 
   - Compromised clients use corrupted validation
   - Medium/low clients use clean validation
8. **Trust Differentiation**: Clear trust score ranges enable effective Trust-Aware aggregation

---

## 8. References

- **CTU-13 Dataset**: [CTU University Dataset for Botnet Detection](https://www.stratosphereips.org/datasets-ctu13)
- **Heterogeneous Client Creation**: `create_heterogeneous_clients.py`
- **Test Set Creation**: `create_heterogeneous_test_set.py`
- **Experiment Configuration**: `experiment.py`

---

*This document describes the dataset, data partitioning, and corruption strategy used in the Trust-Aware Federated Honeypot Learning system.*
