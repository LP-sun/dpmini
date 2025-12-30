#!/bin/bash
# Simple training launcher
set -e
cd /home/ubuntu/pj

# Ensure conda is activated
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" = "base" ]; then
    echo "Activating conda environment..."
    . /home/ubuntu/miniforge3/bin/activate ai4m
fi

echo "Starting optimized CUDA training..."
echo "Config: se_e2_a/input_torch.json"
echo "Batch size: 8"
echo "Workers: 4"
echo ""

python3 train_cuda_optimized.py \
    --config se_e2_a/input_torch.json \
    --data-dir collect/O64H128 \
    --checkpoint-dir checkpoints_cuda_opt \
    --export-dir exports_cuda_opt \
    --num-workers 4 \
    --batch-size-multiplier 1 2>&1 | tee cuda_training_opt.log
