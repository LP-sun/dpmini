#!/bin/bash
# 列出所有DeepMD训练进程的详细信息

echo "================================================================"
echo "    DeepMD Training Process Inspector"
echo "================================================================"
echo ""

# 查找所有相关进程
pids=$(pgrep -af "python.*train_deepmd|python.*train.*pytorch|dp train" | awk '{print $1}')

if [ -z "$pids" ]; then
    echo "⚠️  No training processes found."
    echo ""
    echo "Search patterns:"
    echo "  - python.*train_deepmd"
    echo "  - python.*train.*pytorch"  
    echo "  - dp train"
    exit 0
fi

echo "Found $(echo $pids | wc -w) training process(es)"
echo ""

for pid in $pids; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "PID: $pid"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # 命令行
    cmdline=$(ps -p $pid -o args= 2>/dev/null)
    if [ ! -z "$cmdline" ]; then
        echo "Command:"
        echo "  $cmdline" | head -c 200
        echo ""
    fi
    
    # 工作目录
    cwd=$(readlink -f /proc/$pid/cwd 2>/dev/null)
    if [ ! -z "$cwd" ]; then
        echo "Working Directory:"
        echo "  $cwd"
    else
        echo "Working Directory: N/A"
    fi
    
    # CPU使用率
    cpu=$(ps -p $pid -o %cpu= 2>/dev/null | tr -d ' ')
    echo "CPU Usage: ${cpu}%"
    
    # 内存使用
    mem=$(ps -p $pid -o %mem= 2>/dev/null | tr -d ' ')
    mem_mb=$(ps -p $pid -o rss= 2>/dev/null | awk '{print $1/1024}')
    echo "Memory: ${mem}% (${mem_mb} MB)"
    
    # 线程数
    threads=$(ls /proc/$pid/task 2>/dev/null | wc -l)
    echo "Threads: $threads"
    
    # 运行时间
    etime=$(ps -p $pid -o etime= 2>/dev/null | tr -d ' ')
    echo "Running Time: $etime"
    
    # 状态
    state=$(ps -p $pid -o state= 2>/dev/null | tr -d ' ')
    echo "State: $state (R=running, S=sleeping, D=disk wait)"
    
    echo ""
done

echo "================================================================"
echo "Environment Variables (first process)"
echo "================================================================"

first_pid=$(echo $pids | awk '{print $1}')
if [ ! -z "$first_pid" ]; then
    echo ""
    cat /proc/$first_pid/environ 2>/dev/null | tr '\0' '\n' | grep -E "OMP_NUM_THREADS|MKL_NUM_THREADS|CUDA|NUM_WORKERS|PYTORCH" | head -20
    echo ""
fi

echo "================================================================"
echo "Quick Stats Summary"
echo "================================================================"
echo ""
total_cpu=$(ps -p $(echo $pids | tr ' ' ',') -o %cpu= 2>/dev/null | awk '{s+=$1} END {print s}')
echo "Total CPU: ${total_cpu}%"

total_mem=$(ps -p $(echo $pids | tr ' ' ',') -o %mem= 2>/dev/null | awk '{s+=$1} END {print s}')
echo "Total Memory: ${total_mem}%"

echo ""
echo "================================================================"
