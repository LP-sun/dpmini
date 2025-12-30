#!/bin/bash
# Package optimized training script for deployment to another device
# This creates a tarball with all necessary files

set -e

PACKAGE_NAME="deepmd_optimized_training_$(date +%Y%m%d_%H%M%S).tar.gz"
TEMP_DIR="deepmd_optimized_package"

echo "========================================================================"
echo "  Creating Deployment Package"
echo "========================================================================"
echo ""

# Clean up any existing temp directory
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

echo "Copying files to package..."

# Core training files
cp run_training_optimized.sh "$TEMP_DIR/"
cp train_cuda.py "$TEMP_DIR/"
cp README_OPTIMIZED_TRAINING.md "$TEMP_DIR/"

# Configuration and data
cp -r se_e2_a "$TEMP_DIR/"
cp -r dpmini "$TEMP_DIR/"

# Optional: Include data (comment out if too large)
if [ -d "collect" ]; then
    echo "  Including collect/ directory (this may take a while)..."
    cp -r collect "$TEMP_DIR/"
fi

# Documentation and analysis
cp BOTTLENECK_ANALYSIS.md "$TEMP_DIR/" 2>/dev/null || true
cp test_batch_sizes.sh "$TEMP_DIR/" 2>/dev/null || true
cp quick_bottleneck_test.py "$TEMP_DIR/" 2>/dev/null || true

# Create a deployment guide
cat > "$TEMP_DIR/DEPLOY.md" << 'EOF'
# Deployment Instructions

## 1. Extract the package
```bash
tar -xzf deepmd_optimized_training_*.tar.gz
cd deepmd_optimized_package
```

## 2. Set up Python environment
```bash
# Option A: Use existing conda environment
conda activate your_env

# Option B: Create new environment
conda create -n deepmd python=3.10
conda activate deepmd
pip install torch numpy

# Option C: Specify Python path when running
export PYTHON="/path/to/your/python"
```

## 3. Verify GPU availability
```bash
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 4. Start optimized training
```bash
# Default configuration (batch_size=64, num_workers=0)
./run_training_optimized.sh

# With mixed precision (recommended)
./run_training_optimized.sh --batch-size 64 --mixed-precision

# Conservative (for smaller GPU)
./run_training_optimized.sh --batch-size 32
```

## 5. Monitor progress
```bash
tail -f training_optimized.log
```

## Expected Performance
- Speed: ~14 steps/s (vs 2.3 baseline)
- GPU utilization: 60-80% (vs 23-24% baseline)
- Training time: ~2 hours (vs 11.8 hours baseline)

See README_OPTIMIZED_TRAINING.md for detailed documentation.
EOF

# Create requirements.txt
cat > "$TEMP_DIR/requirements.txt" << 'EOF'
torch>=2.0.0
numpy>=1.20.0
psutil
EOF

# Create a quick start script
cat > "$TEMP_DIR/quick_start.sh" << 'EOF'
#!/bin/bash
# Quick start with automatic environment detection

echo "Quick Start - Optimized DeepMD Training"
echo "========================================"
echo ""

# Try to find Python with torch
PYTHON=""
for py_cmd in python3 python /opt/conda/bin/python ~/miniconda3/bin/python ~/anaconda3/bin/python; do
    if command -v "$py_cmd" &> /dev/null; then
        if "$py_cmd" -c "import torch" 2>/dev/null; then
            PYTHON="$py_cmd"
            echo "Found Python with PyTorch: $PYTHON"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Error: Could not find Python with PyTorch installed"
    echo "Please install PyTorch or set PYTHON environment variable"
    exit 1
fi

# Check CUDA
if ! "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "Warning: CUDA not available. Training will be slow on CPU."
fi

# Export for use by training script
export PYTHON

echo ""
echo "Starting training with optimized settings..."
./run_training_optimized.sh --batch-size 64
EOF

chmod +x "$TEMP_DIR/quick_start.sh"
chmod +x "$TEMP_DIR/run_training_optimized.sh"
chmod +x "$TEMP_DIR/test_batch_sizes.sh" 2>/dev/null || true

# Create the tarball
echo ""
echo "Creating package: $PACKAGE_NAME"
tar -czf "$PACKAGE_NAME" "$TEMP_DIR"

# Clean up temp directory
rm -rf "$TEMP_DIR"

# Calculate size
SIZE=$(du -h "$PACKAGE_NAME" | cut -f1)

echo ""
echo "========================================================================"
echo "  Package created successfully!"
echo "========================================================================"
echo ""
echo "Package: $PACKAGE_NAME"
echo "Size: $SIZE"
echo ""
echo "Transfer to another device:"
echo "  scp $PACKAGE_NAME user@remote-host:/path/to/destination/"
echo ""
echo "On the remote device:"
echo "  tar -xzf $PACKAGE_NAME"
echo "  cd deepmd_optimized_package"
echo "  ./quick_start.sh"
echo ""
echo "Or follow DEPLOY.md for detailed instructions"
echo ""
