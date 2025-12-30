#!/bin/bash
# High-performance CUDA training launcher with optimization defaults
# Batch size: 8 (from config, can override with --batch-size-multiplier)
# Workers: 4
# Mixed precision: optional (--mixed-precision flag)
# Pin memory: enabled in DataLoader

set -e
cd /home/ubuntu/pj

# Ensure conda environment is activated
if [[ -z "${CONDA_DEFAULT_ENV}" ]] || [[ "${CONDA_DEFAULT_ENV}" == "base" ]]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Activating conda environment..."
    source /home/ubuntu/miniforge3/bin/activate ai4m
fi

# Create timestamped log file
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="cuda_training_opt_${TIMESTAMP}.log"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting optimized CUDA training..."
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Config: se_e2_a/input_torch.json (batch_size: 8)"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Log file: ${LOG_FILE}"
echo ""

# Start training with nohup and tee for both console and file logging
nohup python3 train_cuda_optimized.py \
    --config se_e2_a/input_torch.json \
    --data-dir collect/O64H128 \
    --checkpoint-dir checkpoints_cuda_opt \
    --export-dir exports_cuda_opt \
    --num-workers 4 \
    --batch-size-multiplier 1 \
    "$@" 2>&1 | tee "${LOG_FILE}" &

PID=$!
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Training process started with PID ${PID}"
echo "${PID}" > cuda_training_opt.pid

# Create symlink to latest log
ln -sf "${LOG_FILE}" cuda_training_opt.log

echo "[$(date +'%Y-%m-%d %H:%M:%S')] To monitor progress, run: tail -f ${LOG_FILE}"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Or use: bash monitor_training.sh"
