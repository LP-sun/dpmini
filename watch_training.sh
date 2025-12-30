#!/bin/bash
# 实时监控训练进度

clear
echo "════════════════════════════════════════════════════════"
echo "  Force Loss 优化训练监控"
echo "════════════════════════════════════════════════════════"
echo ""

# 检查进程
if ps aux | grep -q "[p]ython train_cuda_optimized"; then
    echo "✅ 训练进程运行中"
    PID=$(ps aux | grep "[p]ython train_cuda_optimized" | grep -v conda | awk '{print $2}')
    echo "   PID: $PID"
    
    # 运行时间和资源使用
    ps -p $PID -o %cpu,%mem,etime --no-headers
    echo ""
else
    echo "❌ 训练进程未运行"
    echo ""
    exit 1
fi

# 最新训练步数
echo "════════════════════════════════════════════════════════"
echo "  最新训练日志 (最近 15 步)"
echo "════════════════════════════════════════════════════════"
tail -15 cuda_training_opt.log | grep "Step"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Force RMSE 趋势（最近 5 步）"
echo "════════════════════════════════════════════════════════"
tail -5 cuda_training_opt.log | grep -oP 'f_rmse=\K[0-9.]+' | nl

echo ""
echo "════════════════════════════════════════════════════════"
echo "  快速命令"
echo "════════════════════════════════════════════════════════"
echo "  实时刷新: watch -n 5 ./watch_training.sh"
echo "  查看完整日志: tail -f cuda_training_opt.log"
echo "  停止训练: kill $PID"
echo ""
