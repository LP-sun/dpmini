#!/bin/bash
# GPU详细监控 (nvidia-smi dmon模式)

echo "================================================================"
echo "    GPU Detailed Monitoring (nvidia-smi dmon)"
echo "================================================================"
echo ""
echo "监控指标:"
echo "  - sm: GPU利用率 (%)"
echo "  - mem: 显存利用率 (%)"
echo "  - enc: 编码器利用率 (%)"
echo "  - dec: 解码器利用率 (%)"
echo "  - mclk: 显存时钟 (MHz)"
echo "  - pclk: 处理器时钟 (MHz)"
echo ""
echo "按 Ctrl+C 停止监控"
echo "================================================================"
echo ""

# 创建日志文件
log_file="gpu_monitor_$(date +%Y%m%d_%H%M%S).log"
echo "日志保存至: $log_file"
echo ""

# 使用nvidia-smi dmon进行实时监控
# -s: 指标选择 (u=利用率, m=显存, c=时钟, t=温度)
# -c: 采样次数 (9999=持续监控)
# -d: 采样间隔(秒)
nvidia-smi dmon -s umct -c 9999 -d 1 | tee $log_file
