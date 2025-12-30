#!/usr/bin/env python3
"""
提取训练日志中的 f_rmse 和 e_loss 数据并生成可视化。

用法：
    python extract_training_metrics.py logs/train_short_20251229-182340.log
    
输出：
    - metrics.csv (可用于 Excel 或 Pandas 分析)
    - plot_metrics.png (f_rmse 和 e_loss 曲线)
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
    
    Args:
        log_file: 日志文件路径
        
    Returns:
        dict: 包含 step, f_rmse, e_loss, f_loss, lr, pref_f 的列表
    """
    metrics = {
        'step': [],
        'f_rmse': [],
        'e_loss': [],
        'f_loss': [],
        'lr': [],
        'pref_f': [],
    }
    
    # 匹配格式：Step    500 | lr=9.500e-04 | loss=... | e_loss=... f_loss=... f_rmse=... | pref_e=... pref_f=...
    pattern = r'Step\s+(\d+)\s+\|\s+lr=([0-9.e-]+)\s+\|\s+loss=[0-9.e+]+\s+\|\s+e_loss=([0-9.e+-]+)\s+f_loss=([0-9.e+-]+)\s+f_rmse=([0-9.e+-]+)\s+\|\s+pref_e=[0-9.e+-]+\s+pref_f=([0-9.e+-]+)'
    
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                step = int(match.group(1))
                lr = float(match.group(2))
                e_loss = float(match.group(3))
                f_loss = float(match.group(4))
                f_rmse = float(match.group(5))
                pref_f = float(match.group(6))
                
                metrics['step'].append(step)
                metrics['lr'].append(lr)
                metrics['e_loss'].append(e_loss)
                metrics['f_loss'].append(f_loss)
                metrics['f_rmse'].append(f_rmse)
                metrics['pref_f'].append(pref_f)
    
    return metrics


def save_csv(metrics, output_file):
    """保存为 CSV 文件。"""
    with open(output_file, 'w') as f:
        f.write('step,lr,e_loss,f_loss,f_rmse,pref_f\n')
        for i in range(len(metrics['step'])):
            f.write(f"{metrics['step'][i]},{metrics['lr'][i]:.6e},"
                   f"{metrics['e_loss'][i]:.6e},{metrics['f_loss'][i]:.6e},"
                   f"{metrics['f_rmse'][i]:.6e},{metrics['pref_f'][i]:.6e}\n")


def plot_metrics(metrics, output_file='metrics.png'):
    """绘制 f_rmse 和 e_loss 曲线。"""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DeepMD Training Metrics', fontsize=16, fontweight='bold')
    
    steps = np.array(metrics['step'])
    
    # F_RMSE 曲线
    ax = axes[0, 0]
    ax.plot(steps, metrics['f_rmse'], 'b-', linewidth=1.5, label='f_rmse')
    ax.set_xlabel('Step')
    ax.set_ylabel('F_RMSE', color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax.grid(True, alpha=0.3)
    ax.set_title('Force RMSE (Lower is Better)')
    if metrics['f_rmse']:
        ax.text(0.05, 0.95, f"Range: {min(metrics['f_rmse']):.4f} - {max(metrics['f_rmse']):.4f}",
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # E_LOSS 曲线
    ax = axes[0, 1]
    ax.semilogy(steps, metrics['e_loss'], 'r-', linewidth=1.5, label='e_loss')
    ax.set_xlabel('Step')
    ax.set_ylabel('E_LOSS (log scale)', color='r')
    ax.tick_params(axis='y', labelcolor='r')
    ax.grid(True, alpha=0.3)
    ax.set_title('Energy Loss (Log Scale)')
    
    # F_LOSS 曲线
    ax = axes[1, 0]
    ax.semilogy(steps, metrics['f_loss'], 'g-', linewidth=1.5, label='f_loss')
    ax.set_xlabel('Step')
    ax.set_ylabel('F_LOSS (log scale)', color='g')
    ax.tick_params(axis='y', labelcolor='g')
    ax.grid(True, alpha=0.3)
    ax.set_title('Force Loss (Log Scale)')
    
    # 学习率曲线
    ax = axes[1, 1]
    ax.semilogy(steps, metrics['lr'], 'orange', linewidth=1.5, label='learning_rate')
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate (log scale)', color='orange')
    ax.tick_params(axis='y', labelcolor='orange')
    ax.grid(True, alpha=0.3)
    ax.set_title('Learning Rate Schedule')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ 图表已保存: {output_file}")


def main():
    if len(sys.argv) < 2:
        print("用法: python extract_training_metrics.py <log_file>")
        print("示例: python extract_training_metrics.py logs/train_short_20251229-182340.log")
        sys.exit(1)
    
    log_file = Path(sys.argv[1])
    if not log_file.exists():
        print(f"❌ 文件不存在: {log_file}")
        sys.exit(1)
    
    print(f"📖 读取日志: {log_file}")
    metrics = extract_metrics(log_file)
    
    if not metrics['step']:
        print("❌ 未找到任何训练步骤数据")
        sys.exit(1)
    
    print(f"✓ 提取了 {len(metrics['step'])} 个数据点")
    
    # 保存 CSV
    csv_file = Path('metrics.csv')
    save_csv(metrics, csv_file)
    print(f"✓ CSV 已保存: {csv_file}")
    
    # 显示统计
    print("\n📊 F_RMSE 统计:")
    print(f"  最小值: {min(metrics['f_rmse']):.6f}")
    print(f"  最大值: {max(metrics['f_rmse']):.6f}")
    print(f"  平均值: {np.mean(metrics['f_rmse']):.6f}")
    print(f"  最终值: {metrics['f_rmse'][-1]:.6f}")
    
    print("\n📊 E_LOSS 统计:")
    print(f"  最小值: {min(metrics['e_loss']):.6e}")
    print(f"  最大值: {max(metrics['e_loss']):.6e}")
    print(f"  平均值: {np.mean(metrics['e_loss']):.6e}")
    
    print("\n📊 F_LOSS 统计:")
    print(f"  最小值: {min(metrics['f_loss']):.6e}")
    print(f"  最大值: {max(metrics['f_loss']):.6e}")
    print(f"  平均值: {np.mean(metrics['f_loss']):.6e}")
    
    # 绘制图表
    if HAS_MATPLOTLIB:
        plot_file = Path('metrics.png')
        plot_metrics(metrics, plot_file)
    else:
        print("\n⚠️  跳过图表生成（matplotlib 未安装）")
        print("安装: pip install matplotlib")
    
    print("\n✅ 完成！")


if __name__ == '__main__':
    main()
