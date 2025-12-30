#!/bin/bash
# 监控MD模拟进度

echo "=== MD 模拟监控 ==="
echo "启动时间: $(date)"
echo ""

# 查看日志进度
echo "📊 运行进度:"
if [ -f "md_formal.log" ]; then
    lines=$(wc -l < md_formal.log)
    if [ $lines -gt 0 ]; then
        tail -20 md_formal.log
    else
        echo "日志文件仍在初始化..."
    fi
else
    echo "等待日志文件生成..."
fi

echo ""
echo "🔍 进程状态:"
ps aux | grep "md_simulation.py" | grep -v grep || echo "未找到进程"

echo ""
echo "💾 输出文件:"
ls -lh md_simulation_formal.npz 2>/dev/null || echo "输出文件尚未完成"

echo ""
echo "💻 GPU 使用:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "GPU 信息不可用"
