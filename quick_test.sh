#!/bin/bash
# 一键测试 - 使用examples/data验证优化效果

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║        DeepMD 性能优化 - 快速验证测试                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 激活环境
eval "$(conda shell.bash hook)"
conda activate ai4m

# 环境变量
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "[1/4] 环境检查..."
python3 -c "import torch; import numpy; print(f'  ✓ PyTorch {torch.__version__}')"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/^/  ✓ GPU: /'
echo ""

echo "[2/4] 配置说明..."
echo "  数据集: examples/data/water (400 frames)"
echo "  batch_size: 8"
echo "  num_workers: 4"
echo "  训练步数: 500"
echo "  预计时间: 5-10分钟"
echo ""

echo "[3/4] 启动训练..."
echo "  日志文件: quick_test.log"
echo ""

# 后台启动训练
python3 train_deepmd_pytorch_cuda.py \
    --config config_example_test.json \
    --checkpoint-dir checkpoints_quick_test \
    --export-dir exports_quick_test \
    --num-workers 4 \
    > quick_test.log 2>&1 &

TRAIN_PID=$!
echo "  训练PID: $TRAIN_PID"
echo $TRAIN_PID > quick_test.pid
echo ""

echo "[4/4] 监控训练 (按Ctrl+C停止监控，训练继续)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Time     | GPU% | VRAM(MB) | CPU%  | Status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 监控30秒
for i in {1..30}; do
    if ! kill -0 $TRAIN_PID 2>/dev/null; then
        echo "  训练进程已结束"
        break
    fi
    
    # GPU信息
    gpu_info=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null || echo "0, 0")
    gpu_util=$(echo $gpu_info | cut -d',' -f1 | tr -d ' ')
    gpu_mem=$(echo $gpu_info | cut -d',' -f2 | tr -d ' ')
    
    # CPU信息
    cpu_util=$(ps -p $TRAIN_PID -o %cpu= 2>/dev/null | tr -d ' ' || echo "0")
    
    # 状态
    if [ $i -lt 5 ]; then
        status="初始化..."
    else
        status="训练中"
    fi
    
    printf "  %02d:%02d:%02d | %3s%% | %7s  | %5s | %s\n" \
        $((i/3600)) $(((i%3600)/60)) $((i%60)) \
        "$gpu_util" "$gpu_mem" "$cpu_util" "$status"
    
    sleep 1
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 检查训练状态
if kill -0 $TRAIN_PID 2>/dev/null; then
    echo "✓ 训练正在后台运行 (PID: $TRAIN_PID)"
    echo ""
    echo "查看实时日志:"
    echo "  tail -f quick_test.log"
    echo ""
    echo "继续监控:"
    echo "  ./monitor_resources.sh"
    echo ""
    echo "停止训练:"
    echo "  kill $TRAIN_PID"
else
    echo "⚠ 训练进程已结束，检查日志:"
    echo ""
    tail -50 quick_test.log
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  测试启动完成 - 请等待训练进行并观察GPU利用率              ║"
echo "╚════════════════════════════════════════════════════════════╝"
