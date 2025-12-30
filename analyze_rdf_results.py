#!/usr/bin/env python3
"""分析RDF计算结果"""
import numpy as np
import sys

def analyze_rdf(rdf_file):
    """分析RDF数据文件"""
    print("\n" + "="*70)
    print("📊 径向分布函数 (RDF) 分析报告")
    print("="*70 + "\n")
    
    # 读取数据
    data = np.loadtxt(rdf_file, skiprows=2)
    r = data[:, 0]
    g_oo = data[:, 1]
    g_oh = data[:, 2]
    g_hh = data[:, 3]
    
    print(f"【数据文件】{rdf_file}")
    print(f"  数据点数: {len(r)}")
    print(f"  距离范围: {r[0]:.3f} - {r[-1]:.3f} Å")
    print(f"  距离步长: {(r[1]-r[0]):.4f} Å")
    
    # O-O 配对分析
    print("\n【O-O 配对分析】")
    oo_nonzero = g_oo[g_oo > 0]
    if len(oo_nonzero) > 0:
        max_idx = np.argmax(g_oo)
        print(f"  第一峰位置: {r[max_idx]:.3f} Å")
        print(f"  第一峰高度: {g_oo[max_idx]:.2f}")
        print(f"  非零点数: {len(oo_nonzero)}")
        print(f"  平均值: {np.mean(oo_nonzero):.2f}")
        
        # 寻找第一配位壳层
        first_shell_mask = (r > 2.0) & (r < 3.5) & (g_oo > 0.5)
        if np.any(first_shell_mask):
            coord_num = np.trapz(g_oo[first_shell_mask] * r[first_shell_mask]**2, r[first_shell_mask])
            print(f"  第一配位数 (估计): {coord_num:.1f}")
    else:
        print("  ⚠️  无O-O配对数据")
    
    # O-H 配对分析
    print("\n【O-H 配对分析】")
    oh_nonzero = g_oh[g_oh > 0]
    if len(oh_nonzero) > 0:
        max_idx = np.argmax(g_oh)
        print(f"  第一峰位置: {r[max_idx]:.3f} Å")
        print(f"  第一峰高度: {g_oh[max_idx]:.2f}")
        print(f"  非零点数: {len(oh_nonzero)}")
        print(f"  平均值: {np.mean(oh_nonzero):.2f}")
        
        # 分析O-H键长
        oh_bond_mask = (r < 1.2) & (g_oh > 5)
        if np.any(oh_bond_mask):
            bond_peak = r[oh_bond_mask][np.argmax(g_oh[oh_bond_mask])]
            print(f"  O-H 键长 (峰值): {bond_peak:.3f} Å")
    else:
        print("  ⚠️  无O-H配对数据")
    
    # H-H 配对分析
    print("\n【H-H 配对分析】")
    hh_nonzero = g_hh[g_hh > 0]
    if len(hh_nonzero) > 0:
        max_idx = np.argmax(g_hh)
        print(f"  第一峰位置: {r[max_idx]:.3f} Å")
        print(f"  第一峰高度: {g_hh[max_idx]:.2f}")
        print(f"  非零点数: {len(hh_nonzero)}")
        print(f"  平均值: {np.mean(hh_nonzero):.2f}")
    else:
        print("  ⚠️  无H-H配对数据")
    
    # 总体统计
    print("\n【总体统计】")
    total_nonzero = len(oo_nonzero) + len(oh_nonzero) + len(hh_nonzero)
    print(f"  总非零点数: {total_nonzero}")
    print(f"  数据覆盖率: {total_nonzero/(len(r)*3)*100:.1f}%")
    
    # 物理意义
    print("\n【物理解释】")
    print("  ✓ O-H 峰 (~0.95-1.0 Å): 水分子内共价键")
    print("  ✓ O-O 峰 (~2.8 Å): 氢键作用的水分子间距")
    print("  ✓ H-H 峰 (~1.5 Å): 同一水分子内H-H距离")
    print("  ✓ 第二配位壳层: ~4.5-5.5 Å")
    
    print("\n" + "="*70)
    print("✅ RDF 分析完成")
    print("="*70 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        rdf_file = sys.argv[1]
    else:
        rdf_file = "/home/ubuntu/pj/pic/rdf_formal_data_20251229-022738.txt"
    
    analyze_rdf(rdf_file)
