#!/usr/bin/env python3
"""
从训练日志中提取所有训练步骤的指标（修复版本，支持跨行Step信息）
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


def extract_metrics_from_log(log_file):
    """从日志文件中提取训练指标（支持两种格式和跨行）。
    
    格式1（原始）: Step <n> (update=<u>) | lr=<v> | loss=<v> | e_rmse=<v> f_rmse=<v> | pref_e=<v> pref_f=<v>
    格式2（V3）: Step <n> (update=<u>) | lr=<v> | loss=<v> | e_rmse=<v> f_rmse=<v> f_ema=<v>
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
    
    # 先将多行Step信息合并为单行（处理换行问题）
    # 替换Step行中间的换行符
    content = re.sub(r'(Step\s+\d+[^\n]*)\n\s*([a-z_])', r'\1 \2', content)
    
    # 首先尝试匹配带pref_e和pref_f的格式（原始训练）
    pattern_with_pref = r'Step\s+(\d+)\s+\(update=\s*\d+\)\s+\|\s+lr=([0-9.e-]+)\s+\|\s+loss=([0-9.e+-]+)\s+\|\s+e_rmse=([0-9.e+-]+)\s+f_rmse=([0-9.e+-]+).*?\|\s+pref_e=([0-9.e+-]+)\s+pref_f=([0-9.e+-]+)'
    
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
    
    return metrics


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
        print("用法: python extract_all_metrics.py <log_file>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    log_path = Path(log_file)
    
    if not log_path.exists():
        print(f"❌ 日志文件不存在: {log_file}")
        sys.exit(1)
    
    print(f"📖 读取日志: {log_file}")
    metrics = extract_metrics_from_log(log_file)
    
    if not metrics['step']:
        print("❌ 未找到任何训练步骤数据")
        sys.exit(1)
    
    # 排序by step number
    sorted_indices = np.argsort(metrics['step'])
    for key in metrics:
        metrics[key] = [metrics[key][i] for i in sorted_indices]
    
    print(f"✓ 成功提取 {len(metrics['step'])} 个数据点")
    print(f"  范围: Step {metrics['step'][0]} 到 Step {metrics['step'][-1]}")
    
    # 生成 CSV 文件
    csv_file = log_path.parent / f"{log_path.stem}_metrics_full.csv"
    with open(csv_file, 'w') as f:
        f.write("step,lr,loss,e_rmse,f_rmse,pref_e,pref_f\n")
        for i in range(len(metrics['step'])):
            f.write(f"{metrics['step'][i]},{metrics['lr'][i]:.6e},{metrics['loss'][i]:.6f},"
                   f"{metrics['e_rmse'][i]:.6f},{metrics['f_rmse'][i]:.6f},"
                   f"{metrics['pref_e'][i]:.6f},{metrics['pref_f'][i]:.6f}\n")
    print(f"💾 CSV 已保存: {csv_file}")
    
    # 统计数据
    print("\n" + "="*70)
    print("📊 指标统计（完整训练过程）")
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
        fig.suptitle(f'Training Metrics (Full): {log_path.stem}', fontsize=14, fontweight='bold')
        
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
        plot_file = log_path.parent / f"{log_path.stem}_metrics_full.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\n📊 可视化已保存: {plot_file}")
        plt.close()
    
    print("\n✓ 完整分析完成")


if __name__ == '__main__':
    main()
