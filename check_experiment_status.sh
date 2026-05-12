#!/bin/bash
# Script to check the status of running experiments

echo "=== Experiment Status Check ==="
echo ""

# Check beta comparison
if [ -f "beta_comparison_output.log" ]; then
    echo "Beta Comparison:"
    echo "  Last 5 lines:"
    tail -5 beta_comparison_output.log | sed 's/^/    /'
    echo ""
    if grep -q "BETA COMPARISON SUMMARY" beta_comparison_output.log; then
        echo "  ✓ Beta comparison COMPLETED"
        if [ -f "results/reports/beta_comparison_results.json" ]; then
            echo "  ✓ Results file exists"
        fi
    else
        echo "  ⏳ Beta comparison IN PROGRESS"
    fi
else
    echo "Beta Comparison: Not started or log file not found"
fi

echo ""

# Check ablation study
if [ -f "ablation_study_output.log" ]; then
    echo "Ablation Study:"
    echo "  Last 5 lines:"
    tail -5 ablation_study_output.log | sed 's/^/    /'
    echo ""
    if grep -q "ABLATION STUDY SUMMARY" ablation_study_output.log; then
        echo "  ✓ Ablation study COMPLETED"
        if [ -f "results/reports/ablation_study_results.json" ]; then
            echo "  ✓ Results file exists"
        fi
    else
        echo "  ⏳ Ablation study IN PROGRESS"
    fi
else
    echo "Ablation Study: Not started or log file not found"
fi

echo ""

# Check running processes
RUNNING=$(ps aux | grep -E "(run_beta|run_ablation)" | grep -v grep | wc -l)
if [ "$RUNNING" -gt 0 ]; then
    echo "Running processes: $RUNNING"
    ps aux | grep -E "(run_beta|run_ablation)" | grep -v grep | awk '{print "  PID:", $2, "CMD:", $11, $12, $13}'
else
    echo "No experiment processes currently running"
fi

echo ""
echo "=== End Status Check ==="
