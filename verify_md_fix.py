#!/usr/bin/env python3
"""
验证修复后的MD模拟：检查能量变化和物理正确性
"""
import numpy as np
import sys

def analyze_md_output(npz_file):
    """分析MD输出文件"""
    data = np.load(npz_file)
    
    print("\n" + "="*70)
    print(f"📊 MD 模拟验证报告: {npz_file}")
    print("="*70 + "\n")
    
    trajectory = data['trajectory']
    energies = data['energies']
    times = data['times']
    
    # 基本信息
    print("【基本信息】")
    print(f"  帧数: {len(trajectory)}")
    print(f"  原子数: {trajectory[0].shape[0]}")
    print(f"  时间步数: {len(times)}")
    print(f"  模拟时间: {times[-1]:.6f} ps")
    print(f"  时间步长: {times[1]-times[0] if len(times)>1 else 0:.6f} ps")
    
    # 位置变化
    print("\n【轨迹分析】")
    pos_diff = np.linalg.norm(trajectory[-1] - trajectory[0], axis=1)
    print(f"  最大原子位移: {pos_diff.max():.6f} Å")
    print(f"  平均原子位移: {pos_diff.mean():.6f} Å")
    print(f"  RMS位移: {np.sqrt(np.mean(pos_diff**2)):.6f} Å")
    
    # 逐帧位移
    frame_displacements = []
    for i in range(1, len(trajectory)):
        disp = np.linalg.norm(trajectory[i] - trajectory[i-1], axis=1).max()
        frame_displacements.append(disp)
    
    if frame_displacements:
        print(f"  每步最大位移: {np.max(frame_displacements):.6f} Å")
        print(f"  每步平均位移: {np.mean(frame_displacements):.6f} Å")
    
    # 能量分析
    print("\n【能量分析】")
    pe = energies[:, 0]
    ke = energies[:, 1]
    te = energies[:, 2]
    
    print(f"  势能 (PE):")
    print(f"    平均: {pe.mean():.6f} eV")
    print(f"    标准差: {pe.std():.6f} eV")
    print(f"    范围: [{pe.min():.6f}, {pe.max():.6f}]")
    print(f"    变化: {pe.max() - pe.min():.6f} eV ({(pe.max()-pe.min())/abs(pe.mean())*100:.3f}%)")
    
    print(f"  动能 (KE):")
    print(f"    平均: {ke.mean():.6f} eV")
    print(f"    标准差: {ke.std():.6f} eV")
    print(f"    范围: [{ke.min():.6f}, {ke.max():.6f}]")
    print(f"    变化: {ke.max() - ke.min():.6f} eV ({(ke.max()-ke.min())/abs(ke.mean())*100:.3f}%)")
    
    print(f"  总能 (TE):")
    print(f"    平均: {te.mean():.6f} eV")
    print(f"    标准差: {te.std():.6f} eV")
    print(f"    范围: [{te.min():.6f}, {te.max():.6f}]")
    
    if len(te) > 1:
        drift = (te[-1] - te[0]) / abs(te[0]) * 100
        print(f"    能量漂移: {drift:.6f}%")
    
    # 诊断
    print("\n【诊断】")
    
    if pos_diff.max() < 1e-6:
        print("  ⚠️  警告: 原子几乎没有移动！")
        print("     可能原因: 时间步太小、初速度太小、或力计算有问题")
    elif pos_diff.max() < 0.01:
        print("  ⚠️  注意: 原子位移很小")
        print("     建议: 增加时间步或模拟时间")
    else:
        print("  ✅ 原子正常运动")
    
    if pe.std() < 1e-6:
        print("  ⚠️  警告: 势能完全不变！")
        print("     可能原因: 模型输出固定值、梯度未计算、或数值问题")
    elif pe.std() < 0.01:
        print("  ⚠️  注意: 势能变化很小")
    else:
        print(f"  ✅ 势能有合理变化")
    
    if ke.std() < 1e-6:
        print("  ⚠️  警告: 动能完全不变！")
        print("     可能原因: 速度更新有问题")
    elif ke.std() < 0.01:
        print("  ⚠️  注意: 动能变化很小")
    else:
        print(f"  ✅ 动能有合理变化")
    
    if len(te) > 1 and abs(drift) > 10:
        print(f"  ⚠️  警告: 能量漂移过大 ({drift:.2f}%)")
        print("     可能原因: 时间步过大、数值不稳定")
    elif len(te) > 1 and abs(drift) < 0.001:
        print(f"  ⚠️  可疑: 能量守恒过于完美 ({drift:.6f}%)")
        print("     正常MD应有0.01-1%的小漂移")
    else:
        print(f"  ✅ 能量守恒合理")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        npz_file = sys.argv[1]
    else:
        # 查找最新的md_test_fixed文件
        import glob
        files = glob.glob("md_test_fixed_*.npz")
        if files:
            npz_file = max(files, key=lambda x: x.split('_')[-1])
        else:
            print("未找到MD输出文件")
            sys.exit(1)
    
    analyze_md_output(npz_file)
