#!/usr/bin/env python3
"""分析MD模拟结果"""
import numpy as np

print("\n" + "="*70)
print("📊 MD 模拟完成报告 - md_simulation_formal.npz")
print("="*70 + "\n")

data = np.load('/home/ubuntu/pj/md_simulation_formal.npz')

print("【数据规模】")
print(f"  ✓ 轨迹帧数: {len(data['trajectory'])}")
print(f"  ✓ 原子总数: {data['trajectory'][0].shape[0]}")
print(f"  ✓ 模拟时间: {data['times'][-1]:.4f} ps")
print(f"  ✓ 总模拟步数: {len(data['times'])}")

energies = data['energies']
print("\n【能量统计】")
pe_mean = np.mean(energies[:,0])
ke_mean = np.mean(energies[:,1])
te_mean = np.mean(energies[:,2])
print(f"  平均势能 (PE): {pe_mean:10.4f} ± {np.std(energies[:,0]):6.4f} eV")
print(f"  平均动能 (KE): {ke_mean:10.4f} ± {np.std(energies[:,1]):6.4f} eV")
print(f"  平均总能 (TE): {te_mean:10.4f} ± {np.std(energies[:,2]):6.4f} eV")

energy_drift = (energies[-1,2] - energies[0,2]) / abs(energies[0,2]) * 100
print(f"  能量漂移: {energy_drift:8.3f}%")

print("\n【能量范围】")
print(f"  PE 范围: [{np.min(energies[:,0]):.4f}, {np.max(energies[:,0]):.4f}] eV")
print(f"  KE 范围: [{np.min(energies[:,1]):.4f}, {np.max(energies[:,1]):.4f}] eV")
print(f"  TE 范围: [{np.min(energies[:,2]):.4f}, {np.max(energies[:,2]):.4f}] eV")

print("\n【保存的数据】")
for key in data.keys():
    val = data[key]
    if isinstance(val, np.ndarray):
        print(f"  ✓ {key:20s} shape={str(val.shape):20s} dtype={val.dtype}")
    else:
        print(f"  ✓ {key:20s} {type(val)}")

print("\n" + "="*70)
print("✅ MD 模拟成功完成！数据已保存")
print("="*70 + "\n")
