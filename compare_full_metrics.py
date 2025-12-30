#!/usr/bin/env python3
"""为完整训练数据生成对比可视化和统计"""
import numpy as np
import matplotlib.pyplot as plt
import csv

def load_csv(file_path):
    metrics = {'step': [], 'lr': [], 'loss': [], 'e_rmse': [], 'f_rmse': []}
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics['step'].append(int(row['step']))
            metrics['lr'].append(float(row['lr']))
            metrics['loss'].append(float(row['loss']))
            metrics['e_rmse'].append(float(row['e_rmse']))
            metrics['f_rmse'].append(float(row['f_rmse']))
    return metrics

print("加载完整训练数据...")
original = load_csv('logs/train_fixed_20251230-015420_metrics_full.csv')
v3 = load_csv('logs/run_optimized_v3_20251230-030858_metrics_full.csv')

print(f"原始训练: {len(original['step'])} 个数据点 (Step 1-{max(original['step'])})")
print(f"V3训练:   {len(v3['step'])} 个数据点 (Step 1-{max(v3['step'])})")

# 生成对比图表
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Complete Training Comparison: Original vs V3 Optimized (All Steps)', 
             fontsize=16, fontweight='bold')

colors = {'original': '#1f77b4', 'v3': '#ff7f0e'}

# f_rmse
ax = axes[0, 0]
ax.plot(original['step'], original['f_rmse'], color=colors['original'], linewidth=2, label='Original', alpha=0.8)
ax.plot(v3['step'], v3['f_rmse'], color=colors['v3'], linewidth=2, label='V3 Optimized', alpha=0.8)
ax.set_title('Force RMSE (f_rmse) - Complete Training', fontweight='bold')
ax.set_ylabel('f_rmse', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
orig_f = np.mean(original['f_rmse'])
v3_f = np.mean(v3['f_rmse'])
ax.text(0.02, 0.98, f"Orig Mean: {orig_f:.4f}\nV3 Mean: {v3_f:.4f}\nDiff: {(v3_f-orig_f)/orig_f*100:.1f}%",
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

# e_rmse
ax = axes[0, 1]
ax.plot(original['step'], original['e_rmse'], color=colors['original'], linewidth=2, label='Original', alpha=0.8)
ax.plot(v3['step'], v3['e_rmse'], color=colors['v3'], linewidth=2, label='V3 Optimized', alpha=0.8)
ax.set_title('Energy RMSE (e_rmse) - Complete Training', fontweight='bold')
ax.set_ylabel('e_rmse', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
orig_e = np.mean(original['e_rmse'])
v3_e = np.mean(v3['e_rmse'])
ax.text(0.02, 0.98, f"Orig Mean: {orig_e:.4f}\nV3 Mean: {v3_e:.4f}\nDiff: {(v3_e-orig_e)/orig_e*100:.1f}%",
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))

# loss
ax = axes[1, 0]
ax.plot(original['step'], original['loss'], color=colors['original'], linewidth=2, label='Original', alpha=0.8)
ax.plot(v3['step'], v3['loss'], color=colors['v3'], linewidth=2, label='V3 Optimized', alpha=0.8)
ax.set_title('Total Loss - Complete Training', fontweight='bold')
ax.set_xlabel('Step', fontweight='bold')
ax.set_ylabel('Loss', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
orig_l = np.mean(original['loss'])
v3_l = np.mean(v3['loss'])
ax.text(0.02, 0.98, f"Orig Mean: {orig_l:.2f}\nV3 Mean: {v3_l:.2f}\nDiff: {(v3_l-orig_l)/orig_l*100:.1f}%",
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))

# lr
ax = axes[1, 1]
ax.semilogy(original['step'], original['lr'], color=colors['original'], linewidth=2, label='Original', alpha=0.8)
ax.semilogy(v3['step'], v3['lr'], color=colors['v3'], linewidth=2, label='V3 Optimized', alpha=0.8)
ax.set_title('Learning Rate (log scale)', fontweight='bold')
ax.set_xlabel('Step', fontweight='bold')
ax.set_ylabel('Learning Rate', fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('METRICS_COMPARISON_FULL_TRAINING.png', dpi=150, bbox_inches='tight')
print("\n✓ 完整训练对比图已保存: METRICS_COMPARISON_FULL_TRAINING.png")

# 详细统计
print("\n" + "="*80)
print("📊 完整训练过程详细对比统计")
print("="*80)

metrics_list = [('f_rmse', 'Force RMSE'), ('e_rmse', 'Energy RMSE'), ('loss', 'Loss'), ('lr', 'Learning Rate')]

for metric_key, metric_name in metrics_list:
    orig = np.array(original[metric_key])
    v3_data = np.array(v3[metric_key])
    
    print(f"\n【{metric_name}】")
    print(f"  原始训练:")
    print(f"    Mean = {np.mean(orig):.6e}  |  Std = {np.std(orig):.6e}  |  Range = [{np.min(orig):.6e}, {np.max(orig):.6e}]")
    print(f"  V3优化:")
    print(f"    Mean = {np.mean(v3_data):.6e}  |  Std = {np.std(v3_data):.6e}  |  Range = [{np.min(v3_data):.6e}, {np.max(v3_data):.6e}]")
    
    pct = (np.mean(v3_data) - np.mean(orig)) / np.mean(orig) * 100
    if pct < 0:
        print(f"  结论: V3更优 ↓ {abs(pct):.1f}%")
    else:
        print(f"  结论: 原始更优 ↑ {pct:.1f}%")

print("\n" + "="*80)
