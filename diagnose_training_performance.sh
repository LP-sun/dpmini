#!/bin/bash
# Comprehensive performance diagnosis script
# Run this on the new device to identify why training is slow

set -e

PYTHON="${PYTHON:-python3}"
LOG_FILE="diagnosis_$(date +%Y%m%d_%H%M%S).log"

echo "========================================================================"
echo "  DeepMD Training Performance Diagnosis"
echo "========================================================================"
echo ""
echo "This script will check:"
echo "  1. Hardware and CUDA availability"
echo "  2. Python environment and dependencies"
echo "  3. Data loading performance"
echo "  4. Model computation performance"
echo "  5. Current training configuration"
echo ""
echo "Diagnosis log will be saved to: $LOG_FILE"
echo ""

# Redirect all output to both console and log file
exec > >(tee -a "$LOG_FILE") 2>&1

# ============================================================================
# 1. Hardware Check
# ============================================================================

echo "========================================================================"
echo "1. Hardware Information"
echo "========================================================================"
echo ""

echo "CPU Information:"
lscpu | grep -E "Model name|CPU\(s\)|Thread|Core|MHz" || echo "lscpu not available"
echo ""

echo "Memory:"
free -h
echo ""

echo "GPU Information:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free,compute_cap --format=csv
    echo ""
    echo "GPU Utilization (current):"
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv
else
    echo "⚠️  nvidia-smi not found - GPU may not be available!"
fi
echo ""

# ============================================================================
# 2. Python Environment Check
# ============================================================================

echo "========================================================================"
echo "2. Python Environment"
echo "========================================================================"
echo ""

echo "Python executable: $PYTHON"
$PYTHON --version
echo ""

echo "Checking critical packages..."
$PYTHON << 'PYCHECK'
import sys
print(f"Python path: {sys.executable}")

packages = {
    'torch': ['__version__', 'cuda.is_available()', 'version.cuda', 'backends.cudnn.version()'],
    'numpy': ['__version__'],
}

for pkg, attrs in packages.items():
    try:
        mod = __import__(pkg)
        print(f"\n✓ {pkg}:")
        for attr in attrs:
            try:
                if '()' in attr:
                    val = eval(f'mod.{attr}')
                else:
                    val = eval(f'mod.{attr}')
                print(f"  - {attr}: {val}")
            except Exception as e:
                print(f"  - {attr}: Error - {e}")
    except ImportError:
        print(f"\n✗ {pkg}: NOT INSTALLED")

# Check CUDA details
try:
    import torch
    if torch.cuda.is_available():
        print(f"\n✓ CUDA Device:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  - GPU {i}: {props.name}")
            print(f"    Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
    else:
        print("\n⚠️  CUDA is NOT available in PyTorch!")
        print("    This will make training EXTREMELY slow (CPU only)")
except Exception as e:
    print(f"\n✗ Error checking CUDA: {e}")
PYCHECK

echo ""

# ============================================================================
# 3. Check if training is currently running
# ============================================================================

echo "========================================================================"
echo "3. Current Training Status"
echo "========================================================================"
echo ""

if [ -f "training_optimized.pid" ]; then
    PID=$(cat training_optimized.pid)
    echo "Found PID file: $PID"
    
    if ps -p $PID > /dev/null 2>&1; then
        echo "✓ Training process is running (PID: $PID)"
        echo ""
        echo "Process details:"
        ps -p $PID -o pid,etime,%cpu,%mem,cmd
        echo ""
        
        # Check the actual command line
        echo "Command line arguments:"
        cat /proc/$PID/cmdline 2>/dev/null | tr '\0' ' ' || echo "(unable to read)"
        echo ""
        
        if [ -f "training_optimized.log" ]; then
            echo "Latest log entries:"
            tail -20 training_optimized.log
            echo ""
            
            # Extract performance metrics
            echo "Performance metrics from log:"
            RECENT_SPEED=$(grep "speed" training_optimized.log | tail -5 | grep -oP 'speed \K[0-9.]+' | tail -1)
            RECENT_GPU=$(grep "util" training_optimized.log | tail -5 | grep -oP 'util \K[0-9]+' | tail -1)
            BATCH_SIZE=$(grep "Batch size:" training_optimized.log | head -1 | grep -oP 'Batch size: \K[0-9]+')
            NUM_WORKERS=$(grep "Number of workers:" training_optimized.log | head -1 | grep -oP 'Number of workers: \K[0-9]+')
            
            echo "  Current speed: ${RECENT_SPEED:-N/A} steps/s"
            echo "  GPU utilization: ${RECENT_GPU:-N/A}%"
            echo "  Batch size: ${BATCH_SIZE:-N/A}"
            echo "  Num workers: ${NUM_WORKERS:-N/A}"
            
            if [ -n "$RECENT_SPEED" ]; then
                if (( $(echo "$RECENT_SPEED < 5" | bc -l 2>/dev/null || echo 0) )); then
                    echo ""
                    echo "⚠️  WARNING: Speed is very slow (< 5 steps/s)!"
                fi
            fi
        fi
    else
        echo "✗ Process not running (stale PID file)"
    fi
else
    echo "No training process found (no PID file)"
fi
echo ""

# ============================================================================
# 4. Quick Performance Test
# ============================================================================

echo "========================================================================"
echo "4. Running Quick Performance Test"
echo "========================================================================"
echo ""

if [ ! -f "quick_bottleneck_test.py" ]; then
    echo "⚠️  quick_bottleneck_test.py not found, creating it..."
    cat > quick_bottleneck_test.py << 'PYEOF'
#!/usr/bin/env python3
import time
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from dpmini import DeepMDDataset

def quick_test():
    print("Testing data loading performance...\n")
    
    # Load config
    with open("se_e2_a/input_torch.json", 'r') as f:
        config = json.load(f)
    
    type_map = config['model']['type_map']
    system_dirs = ['collect/data0']
    
    # Test with num_workers=0
    print("Creating dataset...")
    dataset = DeepMDDataset(system_dirs, type_map=type_map)
    
    dataloader = DataLoader(
        dataset, batch_size=8, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    
    # Quick speed test
    print(f"Testing 20 batches...")
    start = time.time()
    for i, batch in enumerate(dataloader):
        if i >= 20:
            break
        if torch.cuda.is_available():
            positions, atom_types, box, energy, forces = batch
            positions = positions.cuda()
            atom_types = atom_types.cuda()
            torch.cuda.synchronize()
    
    elapsed = time.time() - start
    speed = 20 / elapsed
    
    print(f"\nResult: {speed:.2f} batches/s")
    
    if speed < 50:
        print("⚠️  Data loading is SLOW!")
        print("   Possible causes:")
        print("   - Data not in correct location")
        print("   - Slow disk I/O")
        print("   - Dataset not properly loaded")
    else:
        print("✓ Data loading is fast")
    
    return speed

if __name__ == '__main__':
    try:
        quick_test()
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
PYEOF
fi

echo "Running performance test..."
$PYTHON quick_bottleneck_test.py || echo "⚠️  Performance test failed"
echo ""

# ============================================================================
# 5. Check Data Integrity
# ============================================================================

echo "========================================================================"
echo "5. Data Integrity Check"
echo "========================================================================"
echo ""

echo "Checking data directories..."
if [ -d "collect/data0" ]; then
    echo "✓ collect/data0 exists"
    echo "  Contents:"
    ls -lh collect/data0/
    
    if [ -d "collect/data0/set.000" ]; then
        echo ""
        echo "  set.000 files:"
        ls -lh collect/data0/set.000/ | grep -E "coord|box|energy|force"
        
        # Check file sizes
        COORD_SIZE=$(stat -f%z collect/data0/set.000/coord.npy 2>/dev/null || stat -c%s collect/data0/set.000/coord.npy 2>/dev/null || echo "0")
        if [ "$COORD_SIZE" -lt 1000 ]; then
            echo ""
            echo "⚠️  WARNING: coord.npy is very small ($COORD_SIZE bytes)"
            echo "   Data may not be properly copied"
        fi
    else
        echo "✗ set.000 directory not found!"
    fi
else
    echo "✗ collect/data0 NOT FOUND!"
    echo "  This is critical - training cannot work without data"
fi
echo ""

# ============================================================================
# 6. Configuration File Check
# ============================================================================

echo "========================================================================"
echo "6. Configuration File"
echo "========================================================================"
echo ""

if [ -f "se_e2_a/input_torch.json" ]; then
    echo "✓ Configuration file exists"
    echo ""
    echo "Current batch_size setting:"
    grep -A 2 '"batch_size"' se_e2_a/input_torch.json || echo "Not found in config"
else
    echo "✗ Configuration file NOT FOUND: se_e2_a/input_torch.json"
fi
echo ""

# Check for optimized config
if ls se_e2_a/input_torch_optimized_*.json 1> /dev/null 2>&1; then
    echo "Optimized config files found:"
    ls -lh se_e2_a/input_torch_optimized_*.json
fi
echo ""

# ============================================================================
# Summary and Recommendations
# ============================================================================

echo "========================================================================"
echo "7. DIAGNOSIS SUMMARY"
echo "========================================================================"
echo ""

# Analyze and provide recommendations
HAS_GPU=0
if command -v nvidia-smi &> /dev/null; then
    if $PYTHON -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        HAS_GPU=1
    fi
fi

echo "Critical Issues Found:"
if [ $HAS_GPU -eq 0 ]; then
    echo "  ❌ CUDA is not available - This is the PRIMARY issue!"
    echo "     Training on CPU is 100x+ slower than GPU"
    echo "     Fix: Ensure CUDA-enabled PyTorch is installed"
    echo "          pip install torch --index-url https://download.pytorch.org/whl/cu118"
fi

if [ ! -d "collect/data0" ]; then
    echo "  ❌ Training data not found"
    echo "     Fix: Copy data directory to this location"
fi

if [ ! -f "dpmini/__init__.py" ]; then
    echo "  ❌ dpmini module not found"
    echo "     Fix: Copy dpmini directory to this location"
fi

echo ""
echo "Recommendations:"
echo ""

if [ $HAS_GPU -eq 1 ]; then
    echo "✓ GPU is available - Good!"
    echo ""
    echo "To improve performance:"
    echo "  1. Use larger batch size: --batch-size 64"
    echo "  2. Use num_workers=0: --num-workers 0"
    echo "  3. Enable mixed precision: --mixed-precision"
else
    echo "1. Install CUDA-enabled PyTorch (CRITICAL):"
    echo "   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    echo ""
    echo "2. Verify CUDA installation:"
    echo "   nvidia-smi"
    echo "   python -c 'import torch; print(torch.cuda.is_available())'"
fi

echo ""
echo "To re-run diagnosis:"
echo "  ./diagnose_training_performance.sh"
echo ""
echo "Full diagnosis saved to: $LOG_FILE"
echo ""
echo "========================================================================"
