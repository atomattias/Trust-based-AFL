#!/bin/bash
# Alternative setup script that tries multiple methods

set -e

echo "=========================================="
echo "Trust-Aware Federated Honeypot Learning"
echo "Alternative Setup Script"
echo "=========================================="
echo ""

# Method 1: Try to use system packages
echo "Method 1: Checking for system packages..."
if python3 -c "import pandas, numpy, sklearn, matplotlib, seaborn" 2>/dev/null; then
    echo "✓ All required packages are already installed!"
    echo "You can proceed to run the experiment."
    exit 0
fi

# Method 2: Try virtual environment
echo ""
echo "Method 2: Attempting virtual environment..."
if python3 -m venv venv 2>/dev/null; then
    echo "✓ Virtual environment created"
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "✓ Setup complete with virtual environment!"
    exit 0
fi

# Method 3: Use --break-system-packages (with warning)
echo ""
echo "Method 3: Installing with --break-system-packages..."
echo "WARNING: This will install packages system-wide."
echo "Press Ctrl+C to cancel, or Enter to continue..."
read -r

pip install --break-system-packages -r requirements.txt

echo ""
echo "✓ Setup complete!"
echo ""
echo "To run the experiment:"
echo "  python3 experiment.py"
