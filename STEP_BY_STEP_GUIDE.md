# Step-by-Step Guide: From Beginning to Results Analysis

This guide walks you through the complete process, step by step, with actual commands and expected outputs.

---

## 🎯 Overview

**Goal**: Run a complete experiment and analyze results  
**Time**: ~5-10 minutes for single-round, ~15-60 minutes for multi-round  
**Outcome**: Results, visualizations, and analysis

---

## Step 1: Verify Your Setup (2 minutes)

### 1.1 Check Python and Packages

```bash
# Check Python version
python3 --version
# Should show: Python 3.8 or higher

# Check if packages are installed
python3 -c "import pandas, numpy, sklearn, matplotlib, seaborn; print('✓ All packages ready')"
```

**If you get errors**, install packages:
```bash
# Option 1: System packages
sudo apt install python3-pandas python3-numpy python3-sklearn python3-matplotlib python3-seaborn

# Option 2: Virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 1.2 Verify Data Files

```bash
# Check data directory
ls data/CSVs/*.csv | head -5

# You should see files like:
# data/CSVs/dos_hulk.csv
# data/CSVs/portscan.csv
# etc.
```

---

## Step 2: Run Your First Experiment (5 minutes)

### 2.1 Quick Test with 3 Clients

```bash
# Navigate to project directory
cd "/home/mattias/Papers/Trust-Aware Federated Honeypot Learning"

# Run experiment with 3 clients (faster for testing)
python3 experiment.py --num-clients 3
```

### 2.2 What Happens During Execution

You'll see output like this:

```
============================================================
Trust-Aware Federated Honeypot Learning Experiment
Mode: Single-Round (static trust)
============================================================
Found 13 attack CSV files
Found 5 benign CSV files

Setting up 3 federated clients...
  Setting up client_1_ddos_loit...
  Setting up client_2_web_brute_force...
  Setting up client_3_ftp_patator...
Successfully set up 3 clients

============================================================
Training Local Models at Each Client
============================================================

Training client_1_ddos_loit...
  Train samples: 76586
  Val samples: 19147
  Val Accuracy: 1.0000
  Trust Score: 1.0000
...
```

### 2.3 Expected Output

At the end, you'll see:

```
============================================================
RESULTS COMPARISON
============================================================

Metric                    Centralized     FedAvg          Trust-Aware    
----------------------------------------------------------------------
Accuracy                  1.0000          0.8800          0.9100         
F1-Score                  0.9400          0.8700          0.9000         
...

✓ Results saved to results/reports/experiment_results.json
Generating visualizations...
All visualizations saved to results/plots

============================================================
Experiment completed successfully!
============================================================
```

---

## Step 3: Check Your Results (1 minute)

### 3.1 View Results Files

```bash
# Check what was created
ls -lh results/reports/
ls -lh results/plots/

# View JSON results (first 50 lines)
head -50 results/reports/experiment_results.json
```

### 3.2 Quick Results Check

```bash
# Extract key metrics
python3 -c "
import json
with open('results/reports/experiment_results.json', 'r') as f:
    r = json.load(f)
print('Accuracy:')
print(f\"  Centralized: {r['centralized']['accuracy']:.4f}\")
print(f\"  FedAvg: {r['federated_equal_weight']['accuracy']:.4f}\")
print(f\"  Trust-Aware: {r['trust_aware']['accuracy']:.4f}\")
"
```

---

## Step 4: Analyze Results (2 minutes)

### 4.1 Run Automated Analysis

```bash
# Full analysis
python3 analyze_results.py
```

**Output will show:**
```
======================================================================
PERFORMANCE COMPARISON
======================================================================

Metric                   Centralized     FedAvg          Trust-Aware    
--------------------------------------------------------------------------------
accuracy                 0.9500          0.8800          0.9100         
  → Trust-Aware improvement over FedAvg: +3.41%

f1_score                 0.9400          0.8700          0.9000         
  → Trust-Aware improvement over FedAvg: +3.45%
...

======================================================================
TRUST STATISTICS
======================================================================

Mean Trust:   0.8500
Std Dev:      0.1200
Min Trust:    0.6500
Max Trust:    0.9500
...
```

### 4.2 View Specific Sections

```bash
# Just performance comparison
python3 analyze_results.py --section performance

# Just research questions
python3 analyze_results.py --section rq

# Just client analysis
python3 analyze_results.py --section clients
```

### 4.3 View Visualizations

```bash
# List all plots
ls -lh results/plots/

# Open in image viewer (if available)
xdg-open results/plots/performance_comparison.png
xdg-open results/plots/trust_distribution.png
```

**Key plots to check:**
- `performance_comparison.png` - Quick overview
- `trust_distribution.png` - Client trust scores
- `confusion_matrices.png` - Detailed performance
- `metrics_radar.png` - Multi-metric comparison

---

## Step 5: Run Multi-Round Experiment (15-60 minutes)

### 5.1 Run with Adaptive Trust

```bash
# Run 10 rounds with adaptive trust
python3 experiment.py --num-rounds 10 --num-clients 5
```

### 5.2 What to Expect

You'll see output for each round:

```
============================================================
Round 1/10
============================================================

Phase 1: Local Training
Training client_1...
  Val Accuracy: 0.8500
  Trust Score: 0.8500

Phase 2: Trust Update
Trust update - Client: client_1, Round: 1, Old: 0.5000, New: 0.8500, Change: +0.3500

Updated Trust Scores:
  client_1: 0.8500
  client_2: 0.7500
  ...

Phase 3: Aggregation
Client Weights in Aggregation:
  client_1: 0.4000
  client_2: 0.3500
  ...

============================================================
Round 2/10
============================================================
...
```

### 5.3 Check Trust Evolution

After multi-round experiment:

```bash
# View trust history
ls -lh results/trust_history/

# View a client's trust evolution
cat results/trust_history/client_1_trust_history.json | python3 -m json.tool
```

---

## Step 6: Analyze Multi-Round Results

### 6.1 Full Analysis

```bash
python3 analyze_results.py
```

Now you'll see additional sections:

```
======================================================================
TRUST EVOLUTION ANALYSIS
======================================================================

Client                          Initial      Final        Change      Trend        
--------------------------------------------------------------------------------
client_1_ddos_loit              0.7500       0.8500       +0.1000     improving    
client_2_web_brute_force        0.6500       0.7200       +0.0700     improving    
...

Detailed Changes:
--------------------------------------------------------------------------------
client_1_ddos_loit:
  Total Change:        +0.1000
  Percent Change:      13.33%
  Avg Change/Round:    +0.0100
  Max Increase:        +0.0500
  Max Decrease:        -0.0200
  Trend:               improving
```

### 6.2 View Trust Evolution Plots

```bash
# Check for new plots
ls -lh results/plots/trust_evolution.png
ls -lh results/plots/trust_trends.png

# Open them
xdg-open results/plots/trust_evolution.png
xdg-open results/plots/trust_trends.png
```

---

## Step 7: Interpret Your Results

### 7.1 Answer Research Questions

**RQ1: Does trust-aware improve performance?**

Look at the comparison:
```
Accuracy: FedAvg: 0.8800, Trust-Aware: 0.9100
→ Improvement: +3.41%
→ Answer: YES, trust-aware improves performance
```

**RQ2: Can trust reduce impact of noisy clients?**

Check:
- Trust statistics show range (min to max)
- Low-trust clients identified
- They have lower weights in aggregation
→ Answer: YES, low-trust clients are down-weighted

**RQ3: How does trust affect stability?**

Check (multi-round):
- Trust evolution shows stable/improving trends
- Consistency scores are reasonable
→ Answer: Trust weighting provides stable aggregation

### 7.2 Key Findings to Document

1. **Performance Improvement**: Trust-aware vs FedAvg
2. **Trust Distribution**: Which clients are most/least reliable
3. **Trust Evolution**: How trust changes over rounds (if multi-round)
4. **Client Analysis**: Which clients need attention

---

## Step 8: Export Results for Paper

### 8.1 Extract Key Numbers

```bash
# Create a summary table
python3 analyze_results.py --section performance > results_summary.txt

# View it
cat results_summary.txt
```

### 8.2 Use Visualizations

The plots in `results/plots/` are ready for:
- Paper figures
- Presentations
- Reports

**Recommended for paper:**
- `performance_comparison.png` - Main results table
- `trust_evolution.png` - Trust over time (if multi-round)
- `confusion_matrices.png` - Detailed performance

### 8.3 JSON Data for Tables

```bash
# Pretty print for copying to paper
python3 -c "
import json
with open('results/reports/experiment_results.json', 'r') as f:
    r = json.load(f)
    
print('Table: Performance Comparison')
print('='*60)
print(f\"Centralized:  {r['centralized']['accuracy']:.4f}\")
print(f\"FedAvg:       {r['federated_equal_weight']['accuracy']:.4f}\")
print(f\"Trust-Aware:  {r['trust_aware']['accuracy']:.4f}\")
"
```

---

## Complete Workflow Example

Here's the complete sequence:

```bash
# 1. Setup (one time)
cd "/home/mattias/Papers/Trust-Aware Federated Honeypot Learning"
source venv/bin/activate  # If using venv

# 2. Quick test
python3 experiment.py --num-clients 3

# 3. Check results
python3 analyze_results.py --section performance

# 4. Full single-round experiment
python3 experiment.py

# 5. Full analysis
python3 analyze_results.py

# 6. Multi-round experiment (if needed)
python3 experiment.py --num-rounds 10

# 7. Multi-round analysis
python3 analyze_results.py

# 8. View all outputs
ls -lh results/reports/
ls -lh results/plots/
ls -lh results/trust_history/  # If multi-round
```

---

## Troubleshooting

### Issue: All accuracies are 1.0

**Cause**: Test set only has one class (all attacks, no benign)

**Solution**: 
- This is expected if your test CSV only contains attacks
- The system still works correctly
- For better evaluation, use a test set with both classes

### Issue: Trust scores all the same

**Cause**: All clients have same validation accuracy

**Solution**:
- This is fine for single-round
- In multi-round, trust will differentiate over time
- Or use clients with different data quality

### Issue: Visualization errors

**Solution**: Already fixed! The code now handles edge cases.

---

## Next Steps

1. **Document your findings**: Use the analysis output
2. **Create paper tables**: Extract from JSON results
3. **Use visualizations**: Include plots in your paper
4. **Run more experiments**: Try different parameters
5. **Compare scenarios**: Run with different client sets

---

## Quick Command Reference

```bash
# Single-round experiment
python3 experiment.py --num-clients 5

# Multi-round experiment
python3 experiment.py --num-rounds 10

# Analysis
python3 analyze_results.py

# View results
cat results/reports/experiment_results.json | python3 -m json.tool | less

# View plots
ls results/plots/
```

---

**You're all set!** Follow these steps and you'll have complete results and analysis.
