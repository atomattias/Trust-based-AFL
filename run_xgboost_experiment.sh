#!/bin/bash
# Script to run XGBoost experiment on CTU-13 dataset
# Make sure XGBoost is installed first!

set -e

echo "=========================================="
echo "XGBoost Experiment Runner"
echo "=========================================="

# Check if XGBoost is installed
echo "Checking for XGBoost..."
if ! python3 -c "import xgboost" 2>/dev/null; then
    echo "❌ ERROR: XGBoost is not installed!"
    echo ""
    echo "Please install XGBoost first:"
    echo "  pip install xgboost>=2.0.0"
    echo "  OR"
    echo "  pip install --user xgboost>=2.0.0"
    echo ""
    echo "See XGBOOST_INSTALLATION.md for more details."
    exit 1
fi

XGB_VERSION=$(python3 -c "import xgboost; print(xgboost.__version__)" 2>/dev/null)
echo "✅ XGBoost version: $XGB_VERSION"
echo ""

# Clear trust history
echo "Clearing trust history..."
rm -rf results/trust_history/*

# Run experiment
echo "Running XGBoost experiment..."
echo "Configuration:"
echo "  - Dataset: CTU-13 heterogeneous"
echo "  - Test set: CTU-13 test set"
echo "  - Model: XGBoost"
echo "  - Rounds: 10"
echo "  - Multi-signal trust: Yes"
echo "  - Random seed: 42"
echo ""

python3 experiment.py \
  --data-dir data/CSVs/ctu13_heterogeneous \
  --test-csv data/CSVs/ctu13_test_set.csv \
  --model-type xgboost \
  --num-rounds 10 \
  --multi-signal-trust \
  --random-state 42 \
  2>&1 | tee xgboost_experiment_ctu13.log

echo ""
echo "=========================================="
echo "Experiment completed!"
echo "Results saved to: xgboost_experiment_ctu13.log"
echo "=========================================="
