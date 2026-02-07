# Installation Guide

## Quick Installation Steps

### Step 1: Install python3-venv (Required for virtual environments)

```bash
sudo apt install python3.12-venv
```

**Note**: You need to run this command yourself in your terminal (it requires your password).

### Step 2: Run Setup Script

After installing `python3-venv`, run:

```bash
bash setup.sh
```

## Alternative Installation Methods

If you cannot install `python3-venv`, use one of these alternatives:

### Option A: Use System Packages (Recommended Alternative)

```bash
sudo apt install python3-pandas python3-numpy python3-sklearn python3-matplotlib python3-seaborn
```

Then verify installation:
```bash
python3 -c "import pandas, numpy, sklearn, matplotlib, seaborn; print('All packages installed!')"
```

### Option B: Override System Protection

```bash
pip install --break-system-packages -r requirements.txt
```

**Warning**: This installs packages system-wide and may conflict with system packages.

### Option C: Use Alternative Setup Script

```bash
bash setup_alternative.sh
```

This script will try multiple methods automatically.

## Verify Installation

After installation, test that everything works:

```bash
python3 test_imports.py
```

If successful, you should see:
```
✓ preprocessing module imported
✓ local_training module imported
✓ federated_client module imported
✓ federated_server module imported
✓ evaluation module imported
✓ visualization module imported

All imports successful! ✓
```

## Run the Experiment

Once installation is complete:

```bash
# If using virtual environment, activate it first:
source venv/bin/activate

# Run the experiment:
python3 experiment.py
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'pandas'"

- Make sure you installed the packages (see options above)
- If using venv, make sure it's activated: `source venv/bin/activate`
- Try: `python3 -c "import sys; print(sys.path)"` to check Python path

### "externally-managed-environment" error

- Install `python3-venv`: `sudo apt install python3.12-venv`
- Or use system packages: `sudo apt install python3-pandas python3-numpy python3-sklearn python3-matplotlib python3-seaborn`
- Or use `--break-system-packages` flag (not recommended)

### Virtual environment issues

- Remove old venv: `rm -rf venv`
- Recreate: `python3 -m venv venv`
- Activate: `source venv/bin/activate`
- Install: `pip install -r requirements.txt`
