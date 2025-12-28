#!/bin/bash
echo "================================"
echo "训练速度对比"
echo "================================"
echo ""

echo "优化前 (串行for循环):"
if [ -f "training_optimized_final.log" ]; then
    SPEED_OLD=$(grep "speed" training_optimized_final.log | tail -5 | grep -oP 'speed \K[0-9.]+' | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}')
    echo "  平均速度: ${SPEED_OLD} steps/s"
else
    echo "  无数据"
fi

echo ""
echo "优化后 (矢量化GPU并行):"
if [ -f "training_vectorized.log" ]; then
    SPEED_NEW=$(grep "speed" training_vectorized.log | tail -5 | grep -oP 'speed \K[0-9.]+' | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}')
    echo "  平均速度: ${SPEED_NEW} steps/s"
    
    if [ -n "$SPEED_OLD" ] && [ -n "$SPEED_NEW" ] && [ "$SPEED_OLD" != "N/A" ] && [ "$SPEED_NEW" != "N/A" ]; then
        SPEEDUP=$(echo "scale=2; $SPEED_NEW / $SPEED_OLD" | bc)
        PERCENT=$(echo "scale=1; ($SPEED_NEW - $SPEED_OLD) / $SPEED_OLD * 100" | bc)
        echo ""
        echo "性能提升: ${SPEEDUP}x (${PERCENT}%)"
    fi
else
    echo "  训练进行中..."
fi

echo ""
echo "最新进度:"
grep "Step" training_vectorized.log 2>/dev/null | tail -3 | sed 's/^/  /'

echo ""
echo "GPU状态:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader | sed 's/^/  /'
