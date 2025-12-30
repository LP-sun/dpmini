#!/bin/bash
# Test different batch sizes to find optimal performance

echo "=================================="
echo "Batch Size Optimization Test"
echo "=================================="
echo ""
echo "This will test training speed with different batch sizes"
echo "Each test runs for 50 steps to measure performance"
echo ""

PROJECT_DIR="/home/ubuntu/pj"
PYTHON="/home/ubuntu/miniforge3/envs/ai4m/bin/python"

cd "$PROJECT_DIR"

# Create test config files with different batch sizes
for BS in 8 16 32 64; do
    echo "--------------------------------------"
    echo "Testing batch_size=$BS"
    echo "--------------------------------------"
    
    # Copy config and modify batch size
    TEST_CONFIG="se_e2_a/input_torch_bs${BS}.json"
    cp se_e2_a/input_torch.json "$TEST_CONFIG"
    
    # Modify batch size in JSON (simple sed replacement)
    sed -i 's/"batch_size": [0-9]\+/"batch_size": '$BS'/g' "$TEST_CONFIG"
    
    # Run training for 50 steps
    echo "Running 50 steps with batch_size=$BS..."
    timeout 120 $PYTHON train_cuda.py \
        --config "$TEST_CONFIG" \
        --checkpoint-dir "test_bs${BS}" \
        --export-dir "test_bs${BS}_export" \
        --num-workers 0 \
        --max-steps 50 \
        2>&1 | tee "test_bs${BS}.log"
    
    # Extract speed from log
    SPEED=$(grep "speed" "test_bs${BS}.log" | tail -5 | grep -oP 'speed \K[0-9.]+' | tail -1)
    GPU_UTIL=$(grep "util" "test_bs${BS}.log" | tail -5 | grep -oP 'util \K[0-9]+' | tail -1)
    GPU_MEM=$(grep "GPU mem" "test_bs${BS}.log" | tail -5 | grep -oP 'GPU mem \K[0-9]+' | tail -1)
    
    echo ""
    echo "Results for batch_size=$BS:"
    echo "  Speed: $SPEED steps/s"
    echo "  GPU Util: ${GPU_UTIL}%"
    echo "  GPU Memory: ${GPU_MEM} MiB"
    echo ""
    
    # Cleanup test checkpoints
    rm -rf "test_bs${BS}" "test_bs${BS}_export"
done

echo ""
echo "=================================="
echo "Summary"
echo "=================================="
echo ""
echo "Batch Size | Speed (steps/s) | GPU Util | GPU Mem"
echo "-----------|-----------------|----------|--------"

for BS in 8 16 32 64; do
    if [ -f "test_bs${BS}.log" ]; then
        SPEED=$(grep "speed" "test_bs${BS}.log" | tail -5 | grep -oP 'speed \K[0-9.]+' | tail -1)
        GPU_UTIL=$(grep "util" "test_bs${BS}.log" | tail -5 | grep -oP 'util \K[0-9]+' | tail -1)
        GPU_MEM=$(grep "GPU mem" "test_bs${BS}.log" | tail -5 | grep -oP 'GPU mem \K[0-9]+' | tail -1)
        printf "%-10s | %-15s | %-8s | %s MiB\n" "$BS" "${SPEED:-N/A}" "${GPU_UTIL:-N/A}%" "${GPU_MEM:-N/A}"
    fi
done

echo ""
echo "Full logs saved to: test_bs*.log"
echo ""
echo "Recommendation: Use the batch size with highest steps/s"
