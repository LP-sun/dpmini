#!/bin/bash
# 等待MD模拟完成并监控进度

echo "等待MD模拟完成..."
echo ""

# 循环检查进程
count=0
while ps aux | grep -q "md_simulation.py.*500"; do
    count=$((count+1))
    elapsed=$((count * 10))
    printf "\r⏳ 已运行 %d 秒... CPU: $(ps aux | grep '84126' | awk '{print $3}')%% GPU: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | head -1)" "$elapsed"
    sleep 10
done

echo ""
echo "✅ MD模拟进程已完成！"
echo ""

# 检查输出文件
if [ -f "/home/ubuntu/pj/md_simulation_formal.npz" ]; then
    echo "📊 输出文件信息："
    ls -lh /home/ubuntu/pj/md_simulation_formal.npz
    
    # 显示文件中的数据信息
    echo ""
    echo "📈 轨迹数据统计："
    python3 << 'PYTHON'
import numpy as np
data = np.load('/home/ubuntu/pj/md_simulation_formal.npz')
print(f"  轨迹帧数: {len(data['trajectory'])}")
print(f"  原子数: {data['trajectory'][0].shape[0]}")
print(f"  模拟时间: {data['times'][-1]:.4f} ps")
print(f"  平均势能: {np.mean(data['energies'][:,0]):.4f} eV")
print(f"  平均动能: {np.mean(data['energies'][:,1]):.4f} eV")
print(f"  能量漂移: {(data['energies'][-1,2] - data['energies'][0,2]) / data['energies'][0,2] * 100:.2f}%")
PYTHON
else
    echo "❌ 输出文件未找到"
fi

