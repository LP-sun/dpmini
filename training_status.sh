#!/bin/bash
# 快速查看训练状态

PID=10069

if ! ps -p $PID > /dev/null 2>&1; then
    echo "⚠️  训练进程已结束 (PID: $PID)"
    exit 1
fi

echo "✅ 训练运行中 (PID: $PID)"
echo ""

# 最新进度
echo "📊 最新进度:"
tail -1 training_full_optimized.log | sed 's/^/  /'
echo ""

# 速度统计
echo "⚡ 速度统计 (最近5个输出):"
grep 'Step' training_full_optimized.log | tail -5 | grep -oP 'Step\s+\K[0-9]+|speed\s+\K[0-9.]+' | paste - - | sed 's/^/  Step /' | sed 's/\t/ @ /' | sed 's/$/ steps\/s/'
echo ""

# GPU状态
echo "🖥️  GPU状态:"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | sed 's/^/  /'
echo ""

# 进程信息
echo "📈 进程信息:"
ps -p $PID -o pid,etime,%cpu,%mem,rss,vsz | tail -1 | awk '{printf "  PID: %s, 运行时间: %s, CPU: %s%%, MEM: %s%%\n", $1, $2, $3, $4}' 
echo ""

# 日志大小
lines=$(wc -l < training_full_optimized.log)
size=$(ls -lh training_full_optimized.log | awk '{print $5}')
echo "📝 日志: $size ($lines 行)"
