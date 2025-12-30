#!/bin/bash
# Optimized training script based on bottleneck analysis
# Designed for maximum GPU utilization and training speed
#
# Expected performance: ~6x faster than default settings
# - Default: 2.3 steps/s, 11.8 hours
# - Optimized: ~14 steps/s, ~2 hours
#
# Usage:
#   ./run_training_optimized.sh [--batch-size 64] [--mixed-precision]

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# ============================================================================
# Configuration
# ============================================================================

# Python environment (modify if needed)
PYTHON="${PYTHON:-/home/ubuntu/miniforge3/envs/ai4m/bin/python}"

# Training configuration
CONFIG_FILE="${CONFIG_FILE:-se_e2_a/input_torch.json}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints_optimized}"
EXPORT_DIR="${EXPORT_DIR:-exports_optimized}"

# Optimized parameters (based on performance analysis)
BATCH_SIZE="${BATCH_SIZE:-64}"              # Increased from 8 (4x more GPU utilization)
NUM_WORKERS="${NUM_WORKERS:-0}"             # 0 is fastest for in-memory data
MIXED_PRECISION="${MIXED_PRECISION:-}"      # Add --mixed-precision to enable

# Log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="training_optimized_${TIMESTAMP}.log"
PID_FILE="training_optimized.pid"

# ============================================================================
# Parse command line arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --mixed-precision)
            MIXED_PRECISION="--mixed-precision"
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --export-dir)
            EXPORT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --batch-size N          Batch size (default: 64)"
            echo "  --num-workers N         DataLoader workers (default: 0)"
            echo "  --mixed-precision       Enable mixed precision training"
            echo "  --config FILE           Config file (default: se_e2_a/input_torch.json)"
            echo "  --checkpoint-dir DIR    Checkpoint directory"
            echo "  --export-dir DIR        Export directory"
            echo "  --help, -h              Show this help"
            echo ""
            echo "Environment variables:"
            echo "  PYTHON                  Python executable path"
            echo "  BATCH_SIZE              Override batch size"
            echo "  NUM_WORKERS             Override num_workers"
            echo "  MIXED_PRECISION         Set to '--mixed-precision' to enable"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# Prepare optimized config file
# ============================================================================

# Create a modified config with optimized batch size
OPTIMIZED_CONFIG="${CONFIG_FILE%.json}_optimized_bs${BATCH_SIZE}.json"

if [ -f "$CONFIG_FILE" ]; then
    echo "Creating optimized config: $OPTIMIZED_CONFIG"
    cp "$CONFIG_FILE" "$OPTIMIZED_CONFIG"
    
    # Update batch size in training data section
    if command -v jq &> /dev/null; then
        # Use jq if available (more reliable)
        jq ".training.training_data.batch_size = $BATCH_SIZE | .training.validation_data.batch_size = $BATCH_SIZE" \
            "$CONFIG_FILE" > "$OPTIMIZED_CONFIG"
    else
        # Fallback to sed
        sed -i.bak 's/"batch_size": [0-9]\+/"batch_size": '$BATCH_SIZE'/g' "$OPTIMIZED_CONFIG"
    fi
    
    CONFIG_TO_USE="$OPTIMIZED_CONFIG"
else
    echo "Warning: Config file $CONFIG_FILE not found, using as-is"
    CONFIG_TO_USE="$CONFIG_FILE"
fi

# ============================================================================
# Display configuration
# ============================================================================

echo "========================================================================"
echo "  Optimized DeepMD Training"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Config file       : $CONFIG_TO_USE"
echo "  Batch size        : $BATCH_SIZE (optimized from 8)"
echo "  Num workers       : $NUM_WORKERS (optimized from 4)"
echo "  Mixed precision   : ${MIXED_PRECISION:-disabled}"
echo "  Checkpoint dir    : $CHECKPOINT_DIR"
echo "  Export dir        : $EXPORT_DIR"
echo "  Log file          : $LOG_FILE"
echo ""
echo "Expected performance:"
echo "  Speed             : ~14 steps/s (vs 2.3 steps/s baseline)"
echo "  GPU utilization   : 60-80% (vs 23-24% baseline)"
echo "  Training time     : ~2 hours (vs 11.8 hours baseline)"
echo ""

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# ============================================================================
# Start training
# ============================================================================

echo "Starting training at $(date)..."
echo ""

# Build command
CMD="$PYTHON -u train_cuda.py \
    --config $CONFIG_TO_USE \
    --checkpoint-dir $CHECKPOINT_DIR \
    --export-dir $EXPORT_DIR \
    --num-workers $NUM_WORKERS \
    $MIXED_PRECISION"

echo "Command: $CMD"
echo ""
echo "========================================================================"
echo ""

# Run with nohup and log both stdout and stderr
nohup $CMD > "$LOG_FILE" 2>&1 &

PID=$!
echo $PID > "$PID_FILE"

echo "Training started with PID: $PID"
echo "Log file: $LOG_FILE"
echo "PID file: $PID_FILE"
echo ""

# Wait a moment and check if process is still running
sleep 3

if ps -p $PID > /dev/null; then
    echo "✓ Training process is running"
    echo ""
    echo "Monitor progress:"
    echo "  tail -f $LOG_FILE"
    echo ""
    echo "Check speed:"
    echo "  grep 'speed' $LOG_FILE | tail -20"
    echo ""
    echo "Stop training:"
    echo "  kill $PID"
    echo ""
    
    # Create symlink to latest log
    ln -sf "$LOG_FILE" training_optimized.log
    echo "Latest log symlink: training_optimized.log"
    echo ""
    
    # Show first few lines
    echo "Initial output (first 30 lines):"
    echo "------------------------------------------------------------------------"
    sleep 2
    head -30 "$LOG_FILE" 2>/dev/null || echo "(log file not ready yet)"
    
else
    echo "✗ Training process failed to start!"
    echo "Check log file for errors: $LOG_FILE"
    exit 1
fi

echo ""
echo "========================================================================"
echo "Training started successfully!"
echo "========================================================================"
