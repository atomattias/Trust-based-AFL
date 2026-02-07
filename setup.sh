#!/bin/bash
# Setup script for Trust-Aware Federated Honeypot Learning

set -e

echo "=========================================="
echo "Trust-Aware Federated Honeypot Learning"
echo "Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Try to create virtual environment
echo ""
echo "Attempting to create virtual environment..."
if python3 -m venv venv 2>/dev/null; then
    echo "✓ Virtual environment created successfully"
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo "Upgrading pip..."
    pip install --upgrade pip
    echo "Installing dependencies..."
    pip install -r requirements.txt
    echo ""
    echo "✓ Setup complete!"
    echo ""
    echo "To activate the virtual environment in the future, run:"
    echo "  source venv/bin/activate"
    echo ""
    echo "To run the experiment:"
    echo "  python3 experiment.py"
    echo ""
else
    echo "✗ Failed to create virtual environment"
    echo ""
    echo "Your system requires python3-venv package."
    echo "Please install it using one of these methods:"
    echo ""
    echo "Option 1 (Recommended - requires sudo):"
    echo "  sudo apt install python3.12-venv"
    echo "  Then run this script again: bash setup.sh"
    echo ""
    echo "Option 2 (Use system packages):"
    echo "  sudo apt install python3-pandas python3-numpy python3-sklearn python3-matplotlib python3-seaborn"
    echo ""
    echo "Option 3 (Override system protection - not recommended):"
    echo "  pip install --break-system-packages -r requirements.txt"
    echo ""
    echo "Option 4 (Use pipx for isolated installation):"
    echo "  sudo apt install pipx"
    echo "  pipx install pandas numpy scikit-learn matplotlib seaborn"
    echo ""
    exit 1
fi
