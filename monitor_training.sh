#!/bin/bash
# Monitor training progress (auto-detect CPU/CUDA and select corresponding log)

BASE_DIR="/home/ubuntu/pj"
cd "$BASE_DIR" 2>/dev/null || true

echo "========================================"
echo "DeepMD Training Monitor"
echo "========================================"
echo ""

# Detect CUDA training first
PID=$(ps aux | grep "train_deepmd_pytorch_cuda" | grep -v grep | awk '{print $2}' | head -1)
MODE="cuda"
LOG_FILE="$BASE_DIR/cuda_training_batch8.log"
CHECKPOINT_DIR="$BASE_DIR/checkpoints_cuda"
EXPORT_DIR="$BASE_DIR/exports_cuda"

# If no CUDA training, detect CPU training
if [ -z "$PID" ]; then
    PID=$(ps aux | grep "train_deepmd_pytorch" | grep -v grep | awk '{print $2}' | head -1)
    MODE="cpu"
    LOG_FILE="$BASE_DIR/training.log"
    CHECKPOINT_DIR="$BASE_DIR/checkpoints"
    EXPORT_DIR="$BASE_DIR/exports"
fi

# Display process info
if [ -z "$PID" ]; then
    echo "❌ Training process not found"
    echo "Log file hint: training.log / cuda_training.log"
    echo "========================================"
    exit 0
else
    echo "✅ Training running (PID: $PID | mode: $MODE)"
    echo "Log file: $LOG_FILE"
    echo ""
    ps aux | awk -v pid="$PID" '$2==pid {printf "   CPU: %s%% | Memory: %s MB | Runtime: %s\n", $3, int($6/1024), $10}'
    echo ""
fi

# Latest log lines and progress analysis (robust to binary logs)
echo "📊 Latest training progress:"
echo "---"
LATEST_STEPS=$(tail -n 50 "$LOG_FILE" 2>/dev/null | grep -a "Step" | tail -5)
if [ -z "$LATEST_STEPS" ]; then
    echo "   No training steps logged yet"
else
    echo "$LATEST_STEPS"
fi
echo ""

# Estimate completion time using process runtime and current step
if [ -n "$PID" ] && [ -n "$LATEST_STEPS" ]; then
    echo "⏱️  Time Estimation:"
    echo "---"
    CURRENT_STEP=$(echo "$LATEST_STEPS" | tail -1 | grep -oP 'Step\s+\K\d+')

    # Try to read total steps from the running command
    CMDLINE=$(ps -p "$PID" -o args= 2>/dev/null)
    CONFIG_PATH=$(echo "$CMDLINE" | sed -n 's/.*--config \([^ ]*\).*/\1/p')
    if [ -n "$CONFIG_PATH" ] && [ -f "$CONFIG_PATH" ]; then
        TOTAL_STEPS=$(grep -oP '"numb_steps":\s*\K\d+' "$CONFIG_PATH" 2>/dev/null)
    fi
    # Fallback defaults
    if [ -z "$TOTAL_STEPS" ]; then
        if [ "$MODE" = "cuda" ]; then TOTAL_STEPS=100000; else TOTAL_STEPS=10000; fi
    fi

    # Get process elapsed time string (HH:MM:SS or MM:SS or DD-HH:MM:SS)
    RUNTIME_STR=$(ps -p "$PID" -o etime= 2>/dev/null | tr -d ' ')
    if [ -n "$RUNTIME_STR" ] && [ -n "$CURRENT_STEP" ] && [ "$CURRENT_STEP" -gt 0 ]; then
        # Convert runtime string to seconds
        if [[ "$RUNTIME_STR" == *-* ]]; then
            DAYS=${RUNTIME_STR%%-*}
            HMS=${RUNTIME_STR#*-}
            IFS=':' read -r HOURS MINUTES SECONDS <<< "$HMS"
            RUNTIME_SEC=$((DAYS*86400 + HOURS*3600 + MINUTES*60 + SECONDS))
        else
            IFS=':' read -ra PARTS <<< "$RUNTIME_STR"
            if [ ${#PARTS[@]} -eq 3 ]; then
                RUNTIME_SEC=$((PARTS[0]*3600 + PARTS[1]*60 + PARTS[2]))
            else
                RUNTIME_SEC=$((PARTS[0]*60 + PARTS[1]))
            fi
        fi

        if [ "$RUNTIME_SEC" -gt 0 ]; then
            SPEED=$(echo "scale=4; $CURRENT_STEP / $RUNTIME_SEC" | bc)
            REMAINING_STEPS=$((TOTAL_STEPS - CURRENT_STEP))
            # Avoid division by zero
            if [ "$SPEED" = "0" ] || [ -z "$SPEED" ]; then
                echo "   Calculating... (insufficient speed data)"
            else
                REMAINING_SEC=$(echo "scale=0; $REMAINING_STEPS / $SPEED" | bc)
                RUNTIME_MIN=$((RUNTIME_SEC / 60))
                RUNTIME_HOURS=$(echo "scale=1; $RUNTIME_MIN / 60" | bc)
                REMAINING_MIN=$((REMAINING_SEC / 60))
                REMAINING_HOURS=$(echo "scale=1; $REMAINING_MIN / 60" | bc)
                PROGRESS=$(echo "scale=1; $CURRENT_STEP * 100 / $TOTAL_STEPS" | bc)
                ETA_TS=$(($(date +%s) + REMAINING_SEC))
                ETA=$(date -d "@$ETA_TS" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || echo "N/A")

                STEPS_PER_MIN=$(echo "scale=1; $SPEED * 60" | bc)
                if (( $(echo "$RUNTIME_HOURS < 1" | bc -l) )); then
                    RUNTIME_DISPLAY="${RUNTIME_MIN} min"
                else
                    RUNTIME_DISPLAY="${RUNTIME_MIN} min (${RUNTIME_HOURS}h)"
                fi
                if (( $(echo "$REMAINING_HOURS < 1" | bc -l) )); then
                    REMAINING_DISPLAY="~${REMAINING_MIN} min"
                else
                    REMAINING_DISPLAY="~${REMAINING_MIN} min (~${REMAINING_HOURS}h)"
                fi

                echo "   Current: Step $CURRENT_STEP / $TOTAL_STEPS ($PROGRESS%)"
                echo "   Speed: ${STEPS_PER_MIN} steps/min"
                echo "   Elapsed: ${RUNTIME_DISPLAY}"
                echo "   Remaining: ${REMAINING_DISPLAY}"
                echo "   Est. completion: $ETA"
            fi
        else
            echo "   Calculating... (runtime too short)"
        fi
    else
        echo "   Calculating... (need more data)"
    fi
    echo ""
fi

# Checkpoints
echo "💾 Saved checkpoints: ($MODE)"
ls -lh "$CHECKPOINT_DIR"/*.pt 2>/dev/null | awk '{print "   " $9 " (" $5 ")"}' || echo "   No checkpoints yet"
echo ""

# Export models
echo "📦 Exported models: ($MODE)"
ls -lh "$EXPORT_DIR"/*.pth 2>/dev/null | awk '{print "   " $9 " (" $5 ")"}' || echo "   No exports yet"
echo ""

echo "========================================"
