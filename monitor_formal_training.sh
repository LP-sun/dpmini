#!/bin/bash
# 监控10万步正式训练进度

LOG_FILE="/home/ubuntu/pj/training_formal_100k.log"
CHECK_INTERVAL=60  # 每60秒检查一次

echo "=== 10万步正式训练监控 ==="
echo "日志文件: $LOG_FILE"
echo "更新频率: 每500步记录一次（共200行）"
echo ""

step_count=0
last_checked=0

while true; do
    if [ -f "$LOG_FILE" ]; then
        # 获取最新行数
        line_count=$(wc -l < "$LOG_FILE" 2>/dev/null || echo 0)
        
        if [ $line_count -gt $last_checked ]; then
            # 显示新增行
            tail -n $((line_count - last_checked)) "$LOG_FILE" | grep -E "(Step|f_rmse|f_loss|e_loss)" | tail -5
            last_checked=$line_count
            
            # 提取最后一步的步数和损失值
            latest=$(grep "Step" "$LOG_FILE" | tail -1)
            if [ ! -z "$latest" ]; then
                echo ">>> $latest"
            fi
        fi
    fi
    
    sleep $CHECK_INTERVAL
done
