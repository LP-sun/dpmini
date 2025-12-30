#!/bin/bash
# 使用examples/data测试优化后的训练性能

set -e

echo "================================================================"
echo "  使用 examples/data/water 测试优化配置"
echo "================================================================"
echo ""

# 激活环境
eval "$(conda shell.bash hook)"
conda activate ai4m

# 验证
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"

# 设置环境变量
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUM_WORKERS=4

echo ""
echo "环境配置:"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS: $MKL_NUM_THREADS"
echo "  NUM_WORKERS: $NUM_WORKERS"
echo ""

# 配置文件
CONFIG="config_example_test.json"

# 运行训练
echo "启动训练..."
echo "  配置: $CONFIG"
echo "  数据: examples/data/water"
echo "  batch_size: 8"
echo "  步数: 500"
echo ""

python3 train_cuda.py \
    --config $CONFIG \
    --checkpoint-dir checkpoints_example_test \
    --export-dir exports_example_test \
    --num-workers 4 \
    --verbose

echo ""
echo "================================================================"
echo "训练完成!"
echo "================================================================"
