#!/bin/bash
# Train DeepMD model on CUDA with timestamped logging

set -e

PROJECT_DIR="/home/ubuntu/pj"
cd "$PROJECT_DIR"

# Activate conda environment
source /home/ubuntu/miniforge3/bin/activate ai4m

# Prepare timestamped log file and symlink to latest
TS=$(date +%Y%m%d-%H%M%S)
LOG_FILE="cuda_training_${TS}.log"

echo "=========================================="
echo "Starting DeepMD CUDA training"
echo "------------------------------------------"
echo "Project dir   : $PROJECT_DIR"
echo "Conda env     : ${CONDA_DEFAULT_ENV:-unknown}"
echo "Config        : se_e2_a/input_torch.json"
echo "Data dir      : collect/O64H128"
echo "Checkpoints   : checkpoints_cuda"
echo "Exports       : exports_cuda"
echo "Log file      : $LOG_FILE"
echo "=========================================="

# Launch CUDA training with unbuffered output and nohup
PYTHONUNBUFFERED=1 nohup /home/ubuntu/miniforge3/envs/ai4m/bin/python -u train_cuda.py \
  --config se_e2_a/input_torch.json \
  --data-dir collect/O64H128 \
  --checkpoint-dir checkpoints_cuda \
  --export-dir exports_cuda \
  "$@" > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > cuda_training.pid
ln -sf "$LOG_FILE" cuda_training.log

echo "CUDA training started (PID: $PID)"
echo "Follow log: tail -f $PROJECT_DIR/$LOG_FILE"
echo "Latest log: tail -f $PROJECT_DIR/cuda_training.log"
echo "Stop      : kill $(cat cuda_training.pid)"
