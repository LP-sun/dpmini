#!/bin/bash
# 实时监控CPU和GPU资源使用情况

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "================================================================"
echo "    DeepMD Training Resource Monitor"
echo "================================================================"
echo ""
echo "监控中... (按 Ctrl+C 停止)"
echo ""

# 创建日志文件
log_file="monitor_$(date +%Y%m%d_%H%M%S).csv"
echo "Timestamp,PID,CPU%,MEM%,Threads,GPU_Util%,GPU_Mem_MB,GPU_Temp_C" > $log_file
echo "日志保存至: $log_file"
echo ""

# 显示表头
printf "${BLUE}%-19s ${GREEN}%-8s ${YELLOW}%-7s %-7s %-8s ${RED}%-9s %-11s %-8s${NC}\n" \
    "Time" "PID" "CPU%" "MEM%" "Threads" "GPU%" "GPU_MEM(MB)" "GPU_°C"
echo "----------------------------------------------------------------"

# 主循环
while true; do
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    
    # 查找训练进程
    pids=$(pgrep -f "train_deepmd|train.*pytorch" 2>/dev/null)
    
    if [ -z "$pids" ]; then
        printf "${RED}%-19s No training process found${NC}\n" "$timestamp"
    else
        # GPU信息
        gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
        gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
        
        # 处理每个进程
        for pid in $pids; do
            # CPU和内存
            cpu=$(ps -p $pid -o %cpu= 2>/dev/null | tr -d ' ')
            mem=$(ps -p $pid -o %mem= 2>/dev/null | tr -d ' ')
            threads=$(ls /proc/$pid/task 2>/dev/null | wc -l)
            
            if [ ! -z "$cpu" ]; then
                # 显示到终端 (彩色)
                printf "%-19s " "$(date +%H:%M:%S)"
                printf "${GREEN}%-8s${NC} " "$pid"
                printf "${YELLOW}%-7s %-7s %-8s${NC} " "$cpu" "$mem" "$threads"
                printf "${RED}%-9s %-11s %-8s${NC}\n" "$gpu_util" "$gpu_mem" "$gpu_temp"
                
                # 记录到CSV
                echo "$(date +%s),$pid,$cpu,$mem,$threads,$gpu_util,$gpu_mem,$gpu_temp" >> $log_file
            fi
        done
    fi
    
    sleep 1
done
