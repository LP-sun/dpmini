#!/bin/bash
# nohup启动完整训练，强制刷新输出

set -e

cd /home/ubuntu/pj

echo "=========================================================================="
echo "  启动完整数据集训练 (nohup后台运行)"
echo "=========================================================================="
echo ""
echo "配置: se_e2_a/input_torch.json"
echo "数据: ../collect/data0 (1823 frames)"
echo "目标: 100000 steps"
echo "优化: batch=8, disp_freq=100, numb_btch=1"
echo ""

# 备份原配置（已修改）
if [ ! -f se_e2_a/input_torch.json.backup_original ]; then
    echo "[1/3] 备份原配置..."
    cp se_e2_a/input_torch.json se_e2_a/input_torch.json.backup_optimized
fi

echo "[2/3] 验证配置..."
grep -E "batch_size|disp_freq|numb_btch" se_e2_a/input_torch.json

echo ""
echo "[3/3] 启动训练 (nohup后台)..."
echo ""

# 设置环境变量（强制刷新输出）
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate ai4m

# 使用nohup后台启动，强制刷新输出到文件
nohup python3 -u train_cuda.py \
    --config se_e2_a/input_torch.json \
    --checkpoint-dir checkpoints_cuda_opt \
    --export-dir exports_cuda_opt \
    --num-workers 4 \
    --verbose \
    > training_full_optimized.log 2>&1 &

TRAIN_PID=$!
echo "✓ 训练已启动"
echo "  PID: $TRAIN_PID"
echo "  日志: training_full_optimized.log"
echo ""

# 保存PID
echo $TRAIN_PID > training_full_optimized.pid

# 立即显示开始的日志
sleep 2
echo "初始日志输出:"
echo "---"
head -30 training_full_optimized.log
echo "..."
echo "---"
echo ""
echo "=========================================================================="
echo "监控命令:"
echo "  实时日志: tail -f training_full_optimized.log"
echo "  搜索步数: grep 'Step' training_full_optimized.log | tail -20"
echo "  检查进程: ps -p $TRAIN_PID"
echo "  GPU监控: watch -n 1 nvidia-smi"
echo "  停止训练: kill $TRAIN_PID"
echo "=========================================================================="
