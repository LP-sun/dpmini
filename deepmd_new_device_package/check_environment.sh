#!/bin/bash
# Quick environment check before training
# This is a lightweight version of diagnose_training_performance.sh

echo "========================================="
echo "  Environment Quick Check"
echo "========================================="
echo ""

# Check Python
PYTHON="${PYTHON:-python3}"
echo "Python executable: $PYTHON"
$PYTHON --version || { echo "❌ Python not found!"; exit 1; }
echo ""

# Check PyTorch
echo "Checking PyTorch..."
$PYTHON -c "import torch; print(f'✓ PyTorch {torch.__version__}')" 2>/dev/null || {
    echo "❌ PyTorch NOT installed!"
    echo ""
    echo "Fix:"
    echo "  1. Activate correct conda environment:"
    echo "     conda activate ai4m"
    echo ""
    echo "  2. Or specify Python path:"
    echo "     export PYTHON='/path/to/conda/envs/ai4m/bin/python'"
    echo ""
    echo "  3. Or install PyTorch:"
    echo "     pip install torch --index-url https://download.pytorch.org/whl/cu118"
    echo ""
    exit 1
}

# Check CUDA
echo "Checking CUDA..."
$PYTHON -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')" || exit 1

if ! $PYTHON -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo ""
    echo "⚠️  WARNING: CUDA not available!"
    echo "   Training will be VERY slow (CPU only)"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "⚠️  nvidia-smi not found"
fi

# Check data
echo ""
echo "Checking data..."
if [ -d "collect/data0/set.000" ]; then
    echo "✓ Data directory exists"
    COORD_SIZE=$(stat -f%z collect/data0/set.000/coord.npy 2>/dev/null || stat -c%s collect/data0/set.000/coord.npy 2>/dev/null || echo "0")
    if [ "$COORD_SIZE" -gt 1000000 ]; then
        echo "✓ Data files look good (coord.npy: $(($COORD_SIZE / 1024 / 1024)) MB)"
    else
        echo "⚠️  Data files seem too small"
    fi
else
    echo "❌ Data directory NOT found: collect/data0/set.000"
    exit 1
fi

# Check dpmini module
echo ""
echo "Checking dpmini module..."
$PYTHON -c "from dpmini import DeepMDModel, DeepMDDataset; print('✓ dpmini module OK')" 2>/dev/null || {
    echo "❌ dpmini module not found or has errors"
    exit 1
}

echo ""
echo "========================================="
echo "✓ Environment check PASSED!"
echo "========================================="
echo ""
echo "You can now run training:"
echo "  ./run_training_optimized.sh --batch-size 64"
echo ""
