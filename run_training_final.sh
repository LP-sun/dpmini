#!/bin/bash
# Direct training launcher - no nohup, just run in background
cd /home/ubuntu/pj

# Run training and output to log
/home/ubuntu/miniforge3/bin/conda run -n ai4m python3 train_cuda.py \
    --config se_e2_a/input_torch.json \
    --data-dir collect/O64H128 \
    --checkpoint-dir checkpoints_cuda \
    --export-dir exports_cuda \
    --num-workers 4 >> cuda_training_final.log 2>&1
