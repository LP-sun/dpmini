#!/bin/bash
# Quick test script: run model evaluation on all collect datasets

echo "============================================================"
echo "  DeepMD Model Test on collect/ datasets"
echo "============================================================"
echo ""

MODEL="${1:-exports_cuda_opt/model_cuda_20251228-121805.pth}"
PYTHON="${2:-conda run -n deepmd python}"

if [ ! -f "$MODEL" ]; then
    echo "❌ Model not found: $MODEL"
    exit 1
fi

echo "Model: $MODEL"
echo "Environment: deepmd"
echo ""

# Test configurations
declare -a DATASETS=("collect/data0" "collect/O64H128" "collect/data2" "collect/O128H256")
declare -a NUM_FRAMES=(1458 200 200 100)

for i in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$i]}"
    num_frames="${NUM_FRAMES[$i]}"
    
    if [ ! -d "$dataset/set.000" ]; then
        echo "⚠️  Skipping $dataset (not found)"
        continue
    fi
    
    echo "============================================================"
    echo "Testing: $dataset"
    echo "============================================================"
    
    $PYTHON test_model.py \
        --model "$MODEL" \
        --data-dir "$dataset" \
        --num-frames "$num_frames"
    
    echo ""
done

echo "============================================================"
echo "✓ All tests completed!"
echo "============================================================"
echo ""
echo "For more details, see: TEST_RESULTS_SUMMARY.md"
