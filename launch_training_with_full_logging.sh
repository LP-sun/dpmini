#!/bin/bash

################################################################################
# 强化日志捕获启动脚本
# 
# 特性：
# 1. 完整捕获 stdout + stderr 到独立日志文件
# 2. 实时显示输出到终端（via tee）
# 3. 在进程死亡时记录退出码和死亡原因
# 4. 使用 -u 无缓冲 Python 执行
# 5. 记录启动参数和环境信息
#
# 用法：
#   ./launch_training_with_full_logging.sh [--config CONFIG] [--steps STEPS] [--grad-accum ACCUM]
#
# 示例：
#   ./launch_training_with_full_logging.sh --config config_formal_100k.json --grad-accum 4
################################################################################

set -o pipefail  # 捕获管道中任何命令的失败

# 默认参数
CONFIG="config_formal_100k.json"
CHECKPOINT_DIR="checkpoints_formal_long"
EXPORT_DIR="exports_formal_long"
FORCE_LOSS="mse"
GRAD_ACCUM=8
NUMB_STEPS=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
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
        --force-loss)
            FORCE_LOSS="$2"
            shift 2
            ;;
        --grad-accum)
            GRAD_ACCUM="$2"
            shift 2
            ;;
        --steps)
            NUMB_STEPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 设置工作目录
WORK_DIR="/home/ubuntu/pj"
cd "$WORK_DIR" || { echo "Failed to cd to $WORK_DIR"; exit 1; }

# 创建日志目录
LOG_DIR="$WORK_DIR/logs"
mkdir -p "$LOG_DIR"

# 生成时间戳和唯一日志文件名
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="$LOG_DIR/train_formal_${TIMESTAMP}.log"
METADATA_FILE="${LOG_FILE%.log}_metadata.txt"

# ============================================================================
# 阶段 1：记录启动信息与环境
# ============================================================================

{
    echo "================================================================================"
    echo "DeepMD Training Launch - Full Logging Mode"
    echo "================================================================================"
    echo ""
    echo "[LAUNCH TIME] $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "[WORKING DIR] $(pwd)"
    echo "[LOG FILE] $LOG_FILE"
    echo ""
    
    echo "--- STARTUP PARAMETERS ---"
    echo "CONFIG: $CONFIG"
    echo "CHECKPOINT_DIR: $CHECKPOINT_DIR"
    echo "EXPORT_DIR: $EXPORT_DIR"
    echo "FORCE_LOSS: $FORCE_LOSS"
    echo "GRAD_ACCUM_STEPS: $GRAD_ACCUM"
    if [ -n "$NUMB_STEPS" ]; then
        echo "NUMB_STEPS (override): $NUMB_STEPS"
    fi
    echo ""
    
    echo "--- ENVIRONMENT ---"
    echo "Shell: $(bash --version | head -1)"
    echo "Python PATH: $(which python 2>/dev/null || which python3 2>/dev/null)"
    
    # 检查 conda 环境
    if [ -n "$CONDA_DEFAULT_ENV" ]; then
        echo "Conda Env: $CONDA_DEFAULT_ENV"
    fi
    
    echo "Hostname: $(hostname)"
    echo "User: $(whoami)"
    echo ""
    
    echo "--- GPU INFO ---"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi failed"
    else
        echo "nvidia-smi not found"
    fi
    echo ""
    
    echo "--- CONFIG FILE CONTENT ---"
    if [ -f "$CONFIG" ]; then
        echo "File: $CONFIG"
        cat "$CONFIG"
    else
        echo "ERROR: Config file not found: $CONFIG"
    fi
    echo ""
    
    echo "================================================================================"
    echo "TRAINING START"
    echo "================================================================================"
    echo ""
    
} | tee "$METADATA_FILE"

# 同时将元数据写入主日志文件
cat "$METADATA_FILE" > "$LOG_FILE"

# ============================================================================
# 阶段 2：激活环境并启动训练，完整捕获输出
# ============================================================================

echo "" | tee -a "$LOG_FILE"
echo "[$(date '+%H:%M:%S')] Activating conda environment and launching training..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 启动子 shell，执行训练，同时捕获所有输出
(
    # 在子 shell 中激活环境
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate /home/ubuntu/miniforge3/envs/cuda_env 2>/dev/null || {
        echo "[ERROR] Failed to activate conda environment" >&2
        exit 1
    }
    
    # 验证 Python 可用
    python --version || exit 1
    
    echo "[$(date '+%H:%M:%S')] Environment activated successfully"
    echo ""
    
    # 构建命令行
    CMD="python -u train_cuda_optimized.py"
    CMD="$CMD --config $CONFIG"
    CMD="$CMD --checkpoint-dir $CHECKPOINT_DIR"
    CMD="$CMD --export-dir $EXPORT_DIR"
    CMD="$CMD --force-loss $FORCE_LOSS"
    CMD="$CMD --grad-accumulation-steps $GRAD_ACCUM"
    
    echo "[$(date '+%H:%M:%S')] Executing: $CMD"
    echo ""
    
    # 执行训练（stderr 和 stdout 合并）
    eval $CMD
    
    # 捕获退出码
    EXIT_CODE=$?
    
    exit $EXIT_CODE
    
) 2>&1 | tee -a "$LOG_FILE"

# 捕获子 shell 的退出码
TRAINING_EXIT_CODE=${PIPESTATUS[0]}

# ============================================================================
# 阶段 3：记录训练终止信息
# ============================================================================

echo "" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "TRAINING COMPLETION/FAILURE REPORT" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"

{
    echo ""
    echo "[END TIME] $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "[EXIT CODE] $TRAINING_EXIT_CODE"
    
    if [ $TRAINING_EXIT_CODE -eq 0 ]; then
        echo "[STATUS] ✓ Training completed successfully"
    elif [ $TRAINING_EXIT_CODE -eq 130 ]; then
        echo "[STATUS] ⚠ Training interrupted by user (Ctrl+C)"
    elif [ $TRAINING_EXIT_CODE -eq 137 ]; then
        echo "[STATUS] ✗ Training killed by system (OOM or signal)"
    elif [ $TRAINING_EXIT_CODE -eq 139 ]; then
        echo "[STATUS] ✗ Training crashed (Segmentation fault)"
    else
        echo "[STATUS] ✗ Training failed with exit code $TRAINING_EXIT_CODE"
    fi
    
    echo ""
    echo "--- FINAL OUTPUT FILE SIZES ---"
    
    if [ -d "$CHECKPOINT_DIR" ]; then
        COUNT=$(find "$CHECKPOINT_DIR" -type f | wc -l)
        SIZE=$(du -sh "$CHECKPOINT_DIR" 2>/dev/null | cut -f1)
        echo "Checkpoints ($CHECKPOINT_DIR): $COUNT files, ~$SIZE"
    else
        echo "Checkpoints ($CHECKPOINT_DIR): Not created"
    fi
    
    if [ -d "$EXPORT_DIR" ]; then
        COUNT=$(find "$EXPORT_DIR" -type f | wc -l)
        SIZE=$(du -sh "$EXPORT_DIR" 2>/dev/null | cut -f1)
        echo "Exports ($EXPORT_DIR): $COUNT files, ~$SIZE"
    else
        echo "Exports ($EXPORT_DIR): Not created"
    fi
    
    echo ""
    echo "--- INTERNAL LOSS LOG (if exists) ---"
    if [ -f "$WORK_DIR/cuda_training_opt.log" ]; then
        LINES=$(wc -l < "$WORK_DIR/cuda_training_opt.log")
        SIZE=$(du -sh "$WORK_DIR/cuda_training_opt.log" | cut -f1)
        echo "cuda_training_opt.log: $LINES lines, ~$SIZE"
        echo ""
        echo "Last 10 lines:"
        tail -n 10 "$WORK_DIR/cuda_training_opt.log" | sed 's/^/  /'
    else
        echo "cuda_training_opt.log: Not found"
    fi
    
    echo ""
    echo "--- SYSTEM RESOURCES AT END ---"
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Memory:"
        nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null | sed 's/^/  /' || echo "  (failed to query)"
    fi
    
    echo ""
    echo "--- MONITORING COMMANDS FOR NEXT RUN ---"
    echo ""
    echo "# Real-time log tail:"
    echo "tail -f '$LOG_FILE'"
    echo ""
    echo "# Extract metrics to CSV (when log accumulates):"
    echo "python -u extract_training_metrics.py --log '$LOG_FILE' --output metrics_${TIMESTAMP}.csv"
    echo ""
    echo "# Watch GPU/checkpoint progress:"
    echo "watch -n 10 'date; ls -lh $CHECKPOINT_DIR; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader'"
    echo ""
    
    echo "================================================================================"
    echo "Full log saved to: $LOG_FILE"
    echo "================================================================================"
    
} | tee -a "$LOG_FILE"

# 返回训练的退出码（这样脚本的退出码反映训练的结果）
exit $TRAINING_EXIT_CODE
