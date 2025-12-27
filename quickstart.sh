#!/bin/bash
# Quick start script for DeepMD PyTorch training (timestamped logging)

set -e

PROJECT_DIR="/home/ubuntu/pj"
cd "$PROJECT_DIR"

echo "=========================================="
echo "DeepMD PyTorch 快速启动"
echo "=========================================="
echo ""

# Avoid duplicate training processes
RUNNING_PIDS=$(ps aux | grep -E "(train_deepmd_pytorch|train_deepmd_pytorch_cuda)" | grep -v grep | awk '{print $2}')
if [ -n "$RUNNING_PIDS" ]; then
    echo "⚠️  检测到正在运行的训练进程: $RUNNING_PIDS"
    echo "   为避免日志冲突，请先停止现有训练或直接使用 monitor_training.sh 监控。"
    exit 1
fi

# Activate conda environment
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "ai4m" ]; then
    source /home/ubuntu/miniforge3/bin/activate ai4m
fi
echo "✓ Conda 环境: $CONDA_DEFAULT_ENV"

# Check data
if [ ! -d "collect/O64H128/set.000" ]; then
    echo "❌ 数据目录不存在: collect/O64H128/set.000"
    exit 1
fi
echo "✓ 数据目录存在"
echo ""

# Prepare timestamped log file and symlink
TS=$(date +%Y%m%d-%H%M%S)
LOG_FILE="quickstart_${TS}.log"
ln -sf "$LOG_FILE" training.log
echo "日志文件: $LOG_FILE (同时更新 training.log 指向最新)"
echo ""

# Run tests (logged)
echo "1. 运行单元测试..." | tee -a "$LOG_FILE"
python -m pytest tests/test_descriptor.py -v -q 2>&1 | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Quick training (500步) with timestamped logging
echo "2. 运行快速训练 (500步)..." | tee -a "$LOG_FILE"
PYTHONUNBUFFERED=1 /home/ubuntu/miniforge3/envs/ai4m/bin/python -u train_deepmd_pytorch.py \
    --config se_e2_a/input_torch_test.json \
    --data-dir collect/O64H128 \
    --checkpoint-dir checkpoints_test \
    --export-dir exports_test 2>&1 | tee -a "$LOG_FILE"

STATUS=${PIPESTATUS[0]}
if [ $STATUS -eq 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "✅ 快速测试完成!" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "输出文件:" | tee -a "$LOG_FILE"
    echo "  - 模型: exports_test/model_*.pth" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "下一步:" | tee -a "$LOG_FILE"
    echo "  1. 运行完整训练 (CPU):" | tee -a "$LOG_FILE"
    echo "     ./run_training.sh" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "  2. 运行完整训练 (CUDA):" | tee -a "$LOG_FILE"
    echo "     ./run_training_cuda.sh" | tee -a "$LOG_FILE"
else
    echo "❌ 训练失败" | tee -a "$LOG_FILE"
    exit 1
fi
