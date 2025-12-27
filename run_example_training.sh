#!/bin/bash
# 快速启动examples/data训练的脚本

set -e

echo "================================================================"
echo "    启动 DeepMD 优化训练 (examples/data/water)"
echo "================================================================"
echo ""

# 激活conda环境
echo "[1] 激活conda环境: ai4m"
eval "$(conda shell.bash hook)"
conda activate ai4m

# 验证环境
echo "[2] 验证环境..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" || {
    echo "❌ PyTorch未安装，尝试cuda_env环境"
    conda activate cuda_env
    python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
}

# 设置环境变量
echo "[3] 设置环境变量..."
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS=$MKL_NUM_THREADS"

# 解析参数
BATCH_SIZE=${1:-12}
NUM_WORKERS=${2:-4}
NUM_EPOCHS=${3:-10}

echo ""
echo "[4] 训练配置:"
echo "  Batch size: $BATCH_SIZE"
echo "  Num workers: $NUM_WORKERS"
echo "  Epochs: $NUM_EPOCHS"
echo ""

# 启动训练
echo "[5] 启动训练..."
echo "================================================================"
echo ""

python3 train_example_optimized.py \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --num-epochs $NUM_EPOCHS

echo ""
echo "================================================================"
echo "训练完成!"
echo "================================================================"
