#!/usr/bin/env python3
"""
分析DeepMD数据集的邻居统计信息
用于优化descriptor中的sel参数
"""

import argparse
import numpy as np
from pathlib import Path
import sys

def compute_distances_pbc(pos1, pos2, box):
    """计算周期性边界条件下的距离"""
    delta = pos2 - pos1
    # 最小镜像约定
    box_diag = np.diag(box)
    delta = delta - box_diag * np.round(delta / box_diag)
    dist = np.linalg.norm(delta, axis=-1)
    return dist

def analyze_system(system_dir: Path, rcut: float, type_map: list):
    """分析一个system的邻居统计"""
    print(f"\n分析 {system_dir.name}...")
    
    # 加载所有set
    all_neighbor_counts = {i: [] for i in range(len(type_map))}
    
    for set_dir in sorted(system_dir.glob('set.*')):
        coord = np.load(set_dir / 'coord.npy')
        box = np.load(set_dir / 'box.npy')
        
        nframe = coord.shape[0]
        natom = coord.shape[1] // 3
        
        coord = coord.reshape(nframe, natom, 3)
        box = box.reshape(nframe, 3, 3)
        
        # 推断原子类型 (假设水分子: 1/3 O, 2/3 H)
        if natom % 3 == 0:
            nO = natom // 3
            atom_types = np.array([0] * nO + [1] * (natom - nO))
        else:
            print(f"  警告: natom={natom} 不是3的倍数，跳过")
            continue
        
        # 随机采样部分frame (节省时间)
        sample_frames = min(100, nframe)
        frame_indices = np.random.choice(nframe, sample_frames, replace=False)
        
        for frame_idx in frame_indices:
            positions = coord[frame_idx]
            box_mat = box[frame_idx]
            
            # 计算每个原子的邻居数
            for i in range(natom):
                type_i = atom_types[i]
                neighbor_count = {j: 0 for j in range(len(type_map))}
                
                for j in range(natom):
                    if i == j:
                        continue
                    
                    dist = compute_distances_pbc(positions[i], positions[j], box_mat)
                    
                    if dist < rcut:
                        type_j = atom_types[j]
                        neighbor_count[type_j] += 1
                
                # 记录每种类型的邻居数
                for type_j in range(len(type_map)):
                    all_neighbor_counts[type_i].append(neighbor_count[type_j])
    
    # 统计结果
    print(f"\n  邻居统计 (rcut={rcut}):")
    print("  " + "="*60)
    
    recommended_sel = []
    
    for type_i in range(len(type_map)):
        counts = np.array(all_neighbor_counts[type_i])
        
        if len(counts) == 0:
            print(f"  {type_map[type_i]}: 无数据")
            recommended_sel.append(0)
            continue
        
        max_count = int(np.max(counts))
        mean_count = np.mean(counts)
        p99 = int(np.percentile(counts, 99))
        p95 = int(np.percentile(counts, 95))
        
        print(f"  {type_map[type_i]} 的邻居类型分布:")
        for type_j in range(len(type_map)):
            type_counts = [all_neighbor_counts[type_i][k*len(type_map) + type_j] 
                          for k in range(len(all_neighbor_counts[type_i])//len(type_map))]
            if type_counts:
                max_j = int(np.max(type_counts))
                mean_j = np.mean(type_counts)
                p99_j = int(np.percentile(type_counts, 99))
                print(f"    → {type_map[type_j]}: max={max_j:3d}, "
                      f"99%={p99_j:3d}, mean={mean_j:5.1f}")
    
    print("  " + "="*60)
    
    return all_neighbor_counts

def main():
    parser = argparse.ArgumentParser(description='分析DeepMD数据集的邻居统计')
    parser.add_argument('--data', type=str, default='examples/data/water',
                       help='数据目录路径')
    parser.add_argument('--rcut', type=float, default=6.0,
                       help='截断半径')
    parser.add_argument('--type-map', nargs='+', default=['O', 'H'],
                       help='原子类型列表')
    args = parser.parse_args()
    
    data_dir = Path(args.data)
    
    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        sys.exit(1)
    
    print("="*70)
    print("  DeepMD 邻居统计分析工具")
    print("="*70)
    print(f"数据目录: {data_dir}")
    print(f"截断半径: {args.rcut}")
    print(f"原子类型: {args.type_map}")
    
    # 分析
    stats = analyze_system(data_dir, args.rcut, args.type_map)
    
    # 推荐sel参数
    print("\n" + "="*70)
    print("  推荐的 sel 参数")
    print("="*70)
    
    recommended_sel = []
    for type_i in range(len(args.type_map)):
        total_counts = []
        # 汇总该类型原子的所有邻居数
        for type_j in range(len(args.type_map)):
            if type_j < len(stats[type_i]):
                counts = stats[type_i]
                if counts:
                    total_counts.extend(counts)
        
        if total_counts:
            max_val = int(np.max(total_counts))
            p99_val = int(np.percentile(total_counts, 99))
            # 推荐值: 99分位数 + 10%余量
            recommended = int(p99_val * 1.1)
            recommended_sel.append(recommended)
            
            print(f"{args.type_map[type_i]}:")
            print(f"  当前最大值: {max_val}")
            print(f"  99分位数: {p99_val}")
            print(f"  推荐值 (99%+10%): {recommended}")
        else:
            recommended_sel.append(0)
    
    print("\n配置建议:")
    print("  \"sel\": " + str(recommended_sel))
    print("\n说明:")
    print("  - 使用99分位数可覆盖99%的情况")
    print("  - 加10%余量确保极端情况下不会截断")
    print("  - 如果接受偶尔截断，可直接使用99分位数")
    print("="*70)

if __name__ == '__main__':
    main()
