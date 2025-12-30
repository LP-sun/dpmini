#!/bin/bash
# Quick test training first, then full training
cd /home/ubuntu/pj

# Activate environment
source /home/ubuntu/miniforge3/bin/activate ai4m

echo "Starting quick validation test (1000 steps)..."
python3 train_cuda.py \
  --config se_e2_a/input_torch_quick_test.json \
  --data-dir collect/O64H128 \
  --checkpoint-dir checkpoints_cuda \
  --export-dir exports_cuda \
  --num-workers 2 2>&1 | tee quick_test.log

echo ""
echo "✅ Quick test completed!"
echo ""
echo "Now starting full training (100000 steps)..."
echo "Log: cuda_training_session.log"

python3 train_cuda.py \
  --config se_e2_a/input_torch.json \
  --data-dir collect/O64H128 \
  --checkpoint-dir checkpoints_cuda \
  --export-dir exports_cuda \
  --num-workers 2 2>&1 | tee cuda_training_session.log
