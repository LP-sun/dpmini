#!/bin/bash
# 监控长时间MD模拟进度

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║          🔬 长时间MD模拟监控 (10 ps)                             ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# 检查进程
echo "【进程状态】"
if ps aux | grep -q "md_simulation.py.*20000.*md_long_10ps"; then
    PID=$(ps aux | grep "md_simulation.py.*20000.*md_long_10ps" | grep -v grep | awk 'NR==2 {print $2}')
    CPU=$(ps aux | grep "$PID" | grep python | awk '{print $3}')
    MEM=$(ps aux | grep "$PID" | grep python | awk '{print $6}')
    TIME=$(ps aux | grep "$PID" | grep python | awk '{print $10}')
    echo "  ✓ 进程运行中 (PID: $PID)"
    echo "  CPU: ${CPU}%"
    echo "  内存: ${MEM} KB"
    echo "  运行时间: $TIME"
else
    echo "  ✗ 进程已停止或完成"
fi

echo ""
echo "【GPU状态】"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader | \
    awk -F, '{printf "  GPU利用率: %s\n  显存使用: %s / %s\n", $1, $2, $3}'

echo ""
echo "【检查点文件】"
ls -lh md_long_10ps_checkpoint_*.npz 2>/dev/null | tail -3 | awk '{print "  " $9 " (" $5 ")"}'
if [ ! -f "md_long_10ps_checkpoint_5000.npz" ]; then
    echo "  ⏳ 等待第一个检查点 (5000步)..."
fi

echo ""
echo "【日志内容】"
if [ -f "md_long_10ps.log" ] && [ -s "md_long_10ps.log" ]; then
    echo "  日志大小: $(ls -lh md_long_10ps.log | awk '{print $5}')"
    echo ""
    tail -15 md_long_10ps.log 2>/dev/null | grep -E "Step|✓|Running|Energy" | tail -10
else
    echo "  ⏳ 日志文件还在缓冲中..."
fi

echo ""
echo "【预计完成时间】"
# 简单估算：如果100步需要约10秒，20000步需要约2000秒 = 33分钟
echo "  总步数: 20000步"
echo "  预计总时长: ~30-40分钟"
echo "  物理模拟时间: 10 ps"

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "持续监控命令: watch -n 30 './monitor_long_md.sh'"
echo "查看完整日志: tail -f md_long_10ps.log"
echo ""
