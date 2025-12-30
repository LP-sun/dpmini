#!/usr/bin/env python3
"""
从训练日志中提取 f_rmse, e_loss, f_loss, lr 等指标并生成可视化。
"""

import sys
import re
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib 未安装，只生成 CSV，不生成图表")


def extract_metrics(log_file):
    """从日志文件中提取训练指标。
    
    支持两种格式：
    1. 原始训练（带pref_e/pref_f）：Step  84500 | lr=2.917e-07 | loss=63.1566 | e_rmse=0.0242 f_rmse=0.9285 | pref_e=0.8528 pref_f=73.25
    2. V3优化训练（无pref）：Step      1 | lr=9.999e-05 | loss=374.5789 | e_rmse=4.8750 f_rmse=1.3664 f_ema=1.3664
    """
    metrics = {
        'step': [],
        'lr': [],
        'loss': [],
        'e_rmse': [],
        'f_rmse': [],
        'pref_e': [],
        'pref_f': [],
    }
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    match_count = 0
    
    # 首先尝试匹配带pref的格式（原始训练）
    pattern_with_pref = r'Step\s+(\d+)\s+\(update=\d+\)\s+\|\s+lr=([0-9.e-]+)\s+\|\s+loss=([0-9.e+-]+)\s+\|\s+e_rmse=([0-9.e+-]+)\s+f_rmse=([0-9.e+-]+).*?\|\s+pref_e=([0-9.e+-]+)\s+pref_f=([0-9.e+-]+)'
    
    matches = list(re.finditer(pattern_with_pref, content))
    
    if matches:
        # 使用带pref的格式
        for match in matches:
            metrics['step'].append(int(match.group(1)))
            metrics['lr'].append(float(match.group(2)))
            metrics['loss'].append(float(match.group(3)))
            metrics['e_rmse'].append(float(match.group(4)))
            metrics['f_rmse'].append(float(match.group(5)))
            metrics['pref_e'].append(float(match.group(6)))
            metrics['pref_f'].append(float(match.group(7)))
            match_count += 1
    else:
        # 尝试匹配不带pref的格式（V3优化训练）
        pattern_without_pref = r'Step\s+(\d+)\s+\(update=\s*\d+\)\s+\|\s+lr=([0-9.e-]+)\s+\|\s+loss=([0-9.e+-]+)\s+\|\s+e_rmse=([0-9.e+-]+)\s+f_rmse=([0-9.e+-]+)'
        
        matches = list(re.finditer(pattern_without_pref, content))
        
        for match in matches:
            metrics['step'].append(int(match.group(1)))
            metrics['lr'].append(float(match.group(2)))
            metrics['loss'].append(float(match.group(3)))
            metrics['e_rmse'].append(float(match.group(4)))
            metrics['f_rmse'].append(float(match.group(5)))
            metrics['pref_e'].append(0.0)  # 填充零值
            metrics['pref_f'].append(0.0)  # 填充零值
            match_count += 1
    
    return metrics, match_count


def compute_statistics(metrics, metric_name):
    """计算单个指标的统计数据。"""
    if not metrics[metric_name]:
        return None
    
    data = np.array(metrics[metric_name])
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'last_10_mean': np.mean(data[-10:]) if len(data) >= 10 else np.mean(data),
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python extract_training_metrics_fixed.py <log_file>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    log_path = Path(log_file)
    
    if not log_path.exists():
        print(f"❌ 日志文件不存在: {log_file}")
        sys.exit(1)
    
    print(f"📖 读取日志: {log_file}")
    metrics, match_count = extract_metrics(log_file)
    
    if match_count == 0:
        print("❌ 未找到任何训练步骤数据")
        sys.exit(1)
    
    print(f"✓ 成功提取 {match_count} 个数据点")
    
    # 生成 CSV 文件
    csv_file = log_path.parent / f"{log_path.stem}_metrics.csv"
    with open(csv_file, 'w') as f:
        f.write("step,lr,loss,e_rmse,f_rmse,pref_e,pref_f\n")
        for i in range(len(metrics['step'])):
            f.write(f"{metrics['step'][i]},{metrics['lr'][i]:.6e},{metrics['loss'][i]:.6f},"
                   f"{metrics['e_rmse'][i]:.6f},{metrics['f_rmse'][i]:.6f},"
                   f"{metrics['pref_e'][i]:.6f},{metrics['pref_f'][i]:.6f}\n")
    print(f"💾 CSV 已保存: {csv_file}")
    
    # 统计数据
    print("\n" + "="*70)
    print("📊 指标统计")
    print("="*70)
    
    for metric_name in ['f_rmse', 'loss', 'e_rmse', 'lr']:
        stats = compute_statistics(metrics, metric_name)
        if stats:
            print(f"\n{metric_name.upper()}:")
            print(f"  Mean:       {stats['mean']:.6e}")
            print(f"  Std:        {stats['std']:.6e}")
            print(f"  Min:        {stats['min']:.6e}")
            print(f"  Max:        {stats['max']:.6e}")
            print(f"  Last 10 Mean: {stats['last_10_mean']:.6e}")
    
    # 生成可视化
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Training Metrics: {log_path.stem}', fontsize=14, fontweight='bold')
        
        # f_rmse
        axes[0, 0].plot(metrics['step'], metrics['f_rmse'], 'b-', linewidth=1.5, alpha=0.7)
        axes[0, 0].set_title('Force RMSE (f_rmse)', fontweight='bold')
        axes[0, 0].set_ylabel('f_rmse')
        axes[0, 0].grid(True, alpha=0.3)
        
        # loss
        axes[0, 1].plot(metrics['step'], metrics['loss'], 'r-', linewidth=1.5, alpha=0.7)
        axes[0, 1].set_title('Total Loss', fontweight='bold')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # e_rmse
        axes[1, 0].plot(metrics['step'], metrics['e_rmse'], 'g-', linewidth=1.5, alpha=0.7)
        axes[1, 0].set_title('Energy RMSE (e_rmse)', fontweight='bold')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('e_rmse')
        axes[1, 0].grid(True, alpha=0.3)
        
        # learning rate
        axes[1, 1].semilogy(metrics['step'], metrics['lr'], 'purple', linewidth=1.5, alpha=0.7)
        axes[1, 1].set_title('Learning Rate (log scale)', fontweight='bold')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        
        # 保存图表
        plot_file = log_path.parent / f"{log_path.stem}_metrics.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\n📊 可视化已保存: {plot_file}")
        plt.close()
    
    print("\n✓ 分析完成")


if __name__ == '__main__':
    main()
