#!/bin/bash
# Script to prepare files for Zenodo release

echo "=========================================="
echo "Preparing Zenodo Release Package"
echo "=========================================="
echo ""

# Create release directory
RELEASE_DIR="zenodo_release"
rm -rf "$RELEASE_DIR"
mkdir -p "$RELEASE_DIR"

echo "Copying essential files..."

# Copy source code
cp -r src/ "$RELEASE_DIR/"
cp -r config/ "$RELEASE_DIR/"
cp experiment.py "$RELEASE_DIR/"
cp requirements.txt "$RELEASE_DIR/"

# Copy documentation
cp README.md "$RELEASE_DIR/"
cp EXECUTIVE_SUMMARY.md "$RELEASE_DIR/"
cp ARCHITECTURE.md "$RELEASE_DIR/"
cp ZENODO_DESCRIPTION.md "$RELEASE_DIR/"
cp .zenodo.json "$RELEASE_DIR/"

# Copy paper
mkdir -p "$RELEASE_DIR/results/plots"
cp results/plots/overleaf.tex "$RELEASE_DIR/results/plots/"

# Copy figures
cp -r results/plots/Figures/ "$RELEASE_DIR/results/plots/"

# Copy tests
if [ -d "tests" ]; then
    cp -r tests/ "$RELEASE_DIR/"
fi

# Copy scripts
if [ -d "scripts" ]; then
    cp -r scripts/ "$RELEASE_DIR/"
fi

# Copy utility scripts
cp create_heterogeneous_clients.py "$RELEASE_DIR/" 2>/dev/null || true
cp create_heterogeneous_test_set.py "$RELEASE_DIR/" 2>/dev/null || true
cp generate_visualizations.py "$RELEASE_DIR/" 2>/dev/null || true
cp create_combined_figures.py "$RELEASE_DIR/" 2>/dev/null || true

# Create data directory structure (without large CSV files)
mkdir -p "$RELEASE_DIR/data/CSVs"
touch "$RELEASE_DIR/data/CSVs/.gitkeep"
echo "# Data files should be downloaded separately or obtained from original sources" > "$RELEASE_DIR/data/CSVs/README.md"
echo "# CTU-13 dataset: https://www.stratosphereips.org/datasets-ctu13" >> "$RELEASE_DIR/data/CSVs/README.md"
echo "# Honeypot data: Contact authors for access" >> "$RELEASE_DIR/data/CSVs/README.md"

# Remove Python cache
find "$RELEASE_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$RELEASE_DIR" -name "*.pyc" -delete 2>/dev/null || true

echo ""
echo "✓ Release package prepared in: $RELEASE_DIR"
echo ""
echo "Creating ZIP archive..."
cd "$RELEASE_DIR"
zip -r ../zenodo_release.zip . -x "*.git*" "*.DS_Store" "*__pycache__*" "*.pyc"
cd ..

echo ""
echo "=========================================="
echo "✓ Release package ready!"
echo "=========================================="
echo ""
echo "Files created:"
echo "  - Directory: $RELEASE_DIR/"
echo "  - ZIP archive: zenodo_release.zip"
echo ""
echo "Next steps:"
echo "1. Review $RELEASE_DIR/ to ensure all files are included"
echo "2. Update .zenodo.json with your name and affiliation"
echo "3. Go to https://zenodo.org/deposit/new"
echo "4. Upload zenodo_release.zip"
echo "5. Zenodo will auto-fill metadata from .zenodo.json"
echo "6. Review and publish"
echo ""
