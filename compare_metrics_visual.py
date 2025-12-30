#!/usr/bin/env python3
"""
生成两个训练版本的并排对比可视化。
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

def load_metrics_csv(csv_file):
    """Load metrics from CSV file."""
    metrics = {
        'step': [],
        'lr': [],
        'loss': [],
        'e_rmse': [],
        'f_rmse': [],
        'pref_e': [],
        'pref_f': [],
    }
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics['step'].append(int(row['step']))
            metrics['lr'].append(float(row['lr']))
            metrics['loss'].append(float(row['loss']))
            metrics['e_rmse'].append(float(row['e_rmse']))
            metrics['f_rmse'].append(float(row['f_rmse']))
            metrics['pref_e'].append(float(row['pref_e']))
            metrics['pref_f'].append(float(row['pref_f']))
    
    return metrics


# Load both training metrics
original_metrics = load_metrics_csv('logs/train_fixed_20251230-015420_metrics.csv')
v3_metrics = load_metrics_csv('logs/run_optimized_v3_20251230-030858_metrics.csv')

# Create side-by-side comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Training Metrics Comparison: Original vs V3 Optimized', 
             fontsize=16, fontweight='bold', y=0.995)

colors = {'original': '#1f77b4', 'v3': '#ff7f0e'}
labels = {'original': 'Original (Serial)', 'v3': 'V3 Optimized (DataLoader)'}

# Plot 1: Force RMSE (f_rmse)
ax = axes[0, 0]
ax.plot(original_metrics['step'], original_metrics['f_rmse'], 
        color=colors['original'], linewidth=2, label=labels['original'], alpha=0.8)
ax.plot(v3_metrics['step'], v3_metrics['f_rmse'], 
        color=colors['v3'], linewidth=2, label=labels['v3'], alpha=0.8)
ax.set_title('Force RMSE (f_rmse) Comparison', fontweight='bold', fontsize=12)
ax.set_ylabel('f_rmse', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=10)
ax.text(0.02, 0.98, f'Original Mean: {np.mean(original_metrics["f_rmse"]):.4f}\nV3 Mean: {np.mean(v3_metrics["f_rmse"]):.4f}',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: Energy RMSE (e_rmse)
ax = axes[0, 1]
ax.plot(original_metrics['step'], original_metrics['e_rmse'], 
        color=colors['original'], linewidth=2, label=labels['original'], alpha=0.8)
ax.plot(v3_metrics['step'], v3_metrics['e_rmse'], 
        color=colors['v3'], linewidth=2, label=labels['v3'], alpha=0.8)
ax.set_title('Energy RMSE (e_rmse) Comparison', fontweight='bold', fontsize=12)
ax.set_ylabel('e_rmse', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=10)
ax.text(0.02, 0.98, f'Original Mean: {np.mean(original_metrics["e_rmse"]):.4f}\nV3 Mean: {np.mean(v3_metrics["e_rmse"]):.4f}',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# Plot 3: Total Loss
ax = axes[1, 0]
ax.plot(original_metrics['step'], original_metrics['loss'], 
        color=colors['original'], linewidth=2, label=labels['original'], alpha=0.8)
ax.plot(v3_metrics['step'], v3_metrics['loss'], 
        color=colors['v3'], linewidth=2, label=labels['v3'], alpha=0.8)
ax.set_title('Total Loss Comparison', fontweight='bold', fontsize=12)
ax.set_xlabel('Step', fontweight='bold')
ax.set_ylabel('Loss', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=10)
ax.text(0.02, 0.98, f'Original Mean: {np.mean(original_metrics["loss"]):.2f}\nV3 Mean: {np.mean(v3_metrics["loss"]):.2f}',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Plot 4: Learning Rate (log scale)
ax = axes[1, 1]
ax.semilogy(original_metrics['step'], original_metrics['lr'], 
            color=colors['original'], linewidth=2, label=labels['original'], alpha=0.8)
ax.semilogy(v3_metrics['step'], v3_metrics['lr'], 
            color=colors['v3'], linewidth=2, label=labels['v3'], alpha=0.8)
ax.set_title('Learning Rate (log scale)', fontweight='bold', fontsize=12)
ax.set_xlabel('Step', fontweight='bold')
ax.set_ylabel('Learning Rate', fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(loc='best', fontsize=10)
ax.text(0.02, 0.98, f'Original Final LR: {original_metrics["lr"][-1]:.2e}\nV3 Final LR: {v3_metrics["lr"][-1]:.2e}',
        transform=ax.transAxes, fontsize=9, verticalalignment='top', 
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.tight_layout()
plt.savefig('METRICS_COMPARISON_SIDE_BY_SIDE.png', dpi=150, bbox_inches='tight')
print("✓ 并排对比图已保存: METRICS_COMPARISON_SIDE_BY_SIDE.png")

# Generate comparison summary table
print("\n" + "="*80)
print("📊 详细指标对比表")
print("="*80)

metrics_to_compare = ['f_rmse', 'e_rmse', 'loss', 'lr']
metric_names = ['Force RMSE', 'Energy RMSE', 'Loss', 'Learning Rate']

for metric_name, display_name in zip(metrics_to_compare, metric_names):
    orig = np.array(original_metrics[metric_name])
    v3 = np.array(v3_metrics[metric_name])
    
    print(f"\n{display_name} ({metric_name}):")
    print(f"  Original:  Mean={np.mean(orig):.4e}  Std={np.std(orig):.4e}  Min={np.min(orig):.4e}  Max={np.max(orig):.4e}")
    print(f"  V3:        Mean={np.mean(v3):.4e}  Std={np.std(v3):.4e}  Min={np.min(v3):.4e}  Max={np.max(v3):.4e}")
    
    pct_diff = (np.mean(v3) - np.mean(orig)) / np.mean(orig) * 100
    symbol = "↓" if pct_diff < 0 else "↑"
    print(f"  差异:     {symbol} {abs(pct_diff):.1f}% {'(V3更优)' if pct_diff < 0 else '(原始更优)'}")

print("\n" + "="*80)
