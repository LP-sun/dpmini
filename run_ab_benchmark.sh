#!/bin/bash
# A/B 性能基准测试脚本
# 测试不同配置的性能影响

set -e

echo "================================================================"
echo "    DeepMD A/B Performance Benchmark"
echo "================================================================"
echo ""

# 确保脚本可执行
chmod +x list_training_processes.sh monitor_resources.sh gpu_monitor.sh

# 配置
DATA_DIR="examples/data/water"
NUM_STEPS=200  # 快速测试
SAVE_DIR="benchmark_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $SAVE_DIR

echo "测试配置:"
echo "  数据集: $DATA_DIR"
echo "  测试步数: $NUM_STEPS (每个配置)"
echo "  结果保存: $SAVE_DIR"
echo ""

# 定义测试配置
declare -a CONFIGS=(
    "A0_baseline:1:4:2:auto"
    "A1_batch8:8:4:2:auto"
    "A2_batch8_lowval:8:4:2:auto"
    "A3_batch8_workers4:8:4:4:8"
    "A4_batch16:16:4:4:8"
)

echo "测试矩阵:"
echo "  配置名称 | batch_size | num_workers | OMP_THREADS"
echo "  --------------------------------------------------------"
for config in "${CONFIGS[@]}"; do
    IFS=':' read -r name batch workers omp <<< "$config"
    printf "  %-20s | %-10s | %-11s | %-11s\n" "$name" "$batch" "$workers" "$omp"
done
echo ""

# 主测试循环
for config in "${CONFIGS[@]}"; do
    IFS=':' read -r name batch freq workers omp <<< "$config"
    
    echo "================================================================"
    echo "  测试: $name"
    echo "================================================================"
    echo "  batch_size=$batch, num_workers=$workers, OMP_THREADS=$omp"
    echo ""
    
    # 设置环境变量
    if [ "$omp" != "auto" ]; then
        export OMP_NUM_THREADS=$omp
        export MKL_NUM_THREADS=$omp
        echo "  环境变量: OMP_NUM_THREADS=$omp"
    else
        unset OMP_NUM_THREADS
        unset MKL_NUM_THREADS
        echo "  环境变量: OMP_NUM_THREADS=auto"
    fi
    
    # 运行训练
    log_file="$SAVE_DIR/${name}_train.log"
    gpu_log="$SAVE_DIR/${name}_gpu.log"
    
    echo "  开始训练... (日志: $log_file)"
    
    # 后台启动GPU监控
    nvidia-smi dmon -s um -c $NUM_STEPS -d 1 > $gpu_log 2>&1 &
    GPU_MON_PID=$!
    
    # 记录开始时间
    start_time=$(date +%s)
    
    # 运行训练
    python3 train_example_optimized.py \
        --batch-size $batch \
        --num-workers $workers \
        --num-epochs 5 \
        --data-dir $DATA_DIR \
        > $log_file 2>&1
    
    # 记录结束时间
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    
    # 停止GPU监控
    kill $GPU_MON_PID 2>/dev/null || true
    
    # 解析结果
    avg_speed=$(grep "平均速度" $log_file | tail -1 | grep -oP '\d+\.\d+' || echo "0.0")
    total_time=$(grep "总时间" $log_file | tail -1 | grep -oP '\d+\.\d+s' | grep -oP '\d+\.\d+' || echo "$elapsed")
    
    # GPU统计
    if [ -f $gpu_log ]; then
        avg_gpu=$(awk 'NR>2 {sum+=$3; count++} END {if(count>0) print sum/count; else print 0}' $gpu_log)
        avg_mem=$(awk 'NR>2 {sum+=$4; count++} END {if(count>0) print sum/count; else print 0}' $gpu_log)
    else
        avg_gpu="N/A"
        avg_mem="N/A"
    fi
    
    # 保存结果
    echo "$name,$batch,$workers,$omp,$avg_speed,$total_time,$avg_gpu,$avg_mem" >> $SAVE_DIR/results.csv
    
    echo ""
    echo "  ✓ 完成!"
    echo "    - 速度: $avg_speed steps/s"
    echo "    - 时间: $total_time s"
    echo "    - GPU利用率: $avg_gpu %"
    echo "    - GPU显存: $avg_mem MB"
    echo ""
    
    # 短暂休息
    sleep 2
done

# 生成报告
echo "================================================================"
echo "    测试完成 - 结果汇总"
echo "================================================================"
echo ""
echo "配置名称            | Speed(steps/s) | Time(s) | GPU_Util(%) | GPU_Mem(MB)"
echo "--------------------------------------------------------------------------------"

while IFS=',' read -r name batch workers omp speed time gpu_util gpu_mem; do
    printf "%-20s | %-14s | %-7s | %-11s | %-11s\n" \
        "$name" "$speed" "$time" "$gpu_util" "$gpu_mem"
done < $SAVE_DIR/results.csv

echo ""
echo "详细日志保存在: $SAVE_DIR/"
echo ""

# 找出最佳配置
best_config=$(sort -t',' -k5 -nr $SAVE_DIR/results.csv | head -1 | cut -d',' -f1)
best_speed=$(sort -t',' -k5 -nr $SAVE_DIR/results.csv | head -1 | cut -d',' -f5)

echo "================================================================"
echo "  推荐配置: $best_config"
echo "  最佳速度: $best_speed steps/s"
echo "================================================================"
