#!/usr/bin/env python3
"""
Plot training curves from cuda_training_opt.log
Validates the 3 health indicators:
  A. diff < 1e-5 (loss consistency)
  B. f_loss shows downward trend (rolling mean)
  C. No exploded values

Usage: python plot_training_curves_fixed_v1.py [--log cuda_training_opt.log]
"""
import os
import re
import argparse
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Plot training curves')
    parser.add_argument('--log', type=str, default='cuda_training_opt.log',
                       help='Path to training log file')
    args = parser.parse_args()
    
    LOG_PATH = args.log
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs("pic", exist_ok=True)
    
    # Resolve symlinks if needed
    LOG_PATH = os.path.realpath(LOG_PATH)
    
    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError(f"Log file not found: {LOG_PATH}")
    
    # Parse log format:
    # Step  100 | lr=... | loss=... | e_loss=... | f_loss=... | pref_e=... pref_f=... | diff=...
    pat = re.compile(
        r"Step\s+(?P<step>\d+).*?"
        r"lr=(?P<lr>[\deE\+\-\.]+).*?"
        r"loss=(?P<loss>[\deE\+\-\.]+).*?"
        r"e_loss=(?P<e_loss>[\deE\+\-\.]+).*?"
        r"f_loss=(?P<f_loss>[\deE\+\-\.]+).*?"
        r"pref_e=(?P<pref_e>[\deE\+\-\.]+)\s+pref_f=(?P<pref_f>[\deE\+\-\.]+).*?"
        r"diff=(?P<diff>[\deE\+\-\.]+)"
    )
    
    rows = []
    with open(LOG_PATH, "r", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                d = {k: float(v) for k, v in m.groupdict().items() if k != "step"}
                d["step"] = int(m.group("step"))
                rows.append(d)
    
    if not rows:
        raise RuntimeError(f"No matched lines found in {LOG_PATH}")
    
    df = pd.DataFrame(rows).sort_values("step")
    print(f"Loaded {len(df)} training steps from {LOG_PATH}")
    
    # Rolling mean for trend analysis (more robust for "health" judgment)
    win = max(5, len(df) // 50)
    df["f_loss_rm"] = df["f_loss"].rolling(win, min_periods=1).mean()
    df["e_loss_rm"] = df["e_loss"].rolling(win, min_periods=1).mean()
    
    # ========== Health Check ==========
    print("\n" + "=" * 80)
    print("HEALTH CHECK REPORT")
    print("=" * 80)
    
    # Indicator A: Loss consistency
    max_diff = df["diff"].max()
    mean_diff = df["diff"].mean()
    print(f"\n[A] Loss Consistency:")
    print(f"    max(diff) = {max_diff:.3e}")
    print(f"    mean(diff) = {mean_diff:.3e}")
    if max_diff < 1e-5:
        print("    ✅ PASS: diff < 1e-5 (loss consistency verified)")
    else:
        print(f"    ⚠️  WARN: diff={max_diff:.3e} > 1e-5 (check compute_loss)")
    
    # Indicator B: f_loss trend
    if len(df) >= 100:
        early_f = df.iloc[:min(500, len(df)//2)]["f_loss_rm"].mean()
        later_f = df.iloc[-min(500, len(df)//2):]["f_loss_rm"].mean()
        reduction = (1 - later_f / early_f) * 100
        
        print(f"\n[B] Force Loss Trend:")
        print(f"    Early f_loss (rolling mean): {early_f:.6f}")
        print(f"    Later f_loss (rolling mean): {later_f:.6f}")
        print(f"    Reduction: {reduction:.1f}%")
        
        if reduction > 10:
            print(f"    ✅ PASS: f_loss shows downward trend")
        elif reduction > 0:
            print(f"    ⚠️  MARGINAL: f_loss decreased but slowly")
        else:
            print(f"    ❌ FAIL: f_loss not decreasing (may need neighbor list fix)")
    else:
        print(f"\n[B] Force Loss Trend: Too few steps ({len(df)}) to evaluate")
    
    # Indicator C: Descriptor health (indirect check via loss values)
    print(f"\n[C] Numerical Stability:")
    print(f"    f_loss range: [{df['f_loss'].min():.6f}, {df['f_loss'].max():.6f}]")
    print(f"    e_loss range: [{df['e_loss'].min():.3e}, {df['e_loss'].max():.3e}]")
    
    if df["f_loss"].max() < 1e3 and df["e_loss"].max() < 1e9:
        print("    ✅ PASS: No exploded loss values")
    else:
        print("    ⚠️  WARN: Large loss values detected (check for NaN/Inf in training)")
    
    print("\n" + "=" * 80)
    
    # ========== Plots ==========
    
    # 1) f_loss with rolling mean
    plt.figure(figsize=(10, 6))
    plt.plot(df["step"], df["f_loss"], alpha=0.3, label="f_loss (raw)", color='C0')
    plt.plot(df["step"], df["f_loss_rm"], label=f"f_loss (rolling mean, win={win})", 
             color='C0', linewidth=2)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Force MSE (eV²/Å²)", fontsize=12)
    plt.title("Force Loss Curve", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"pic/f_loss_{timestamp}.png", dpi=300)
    print(f"Saved: pic/f_loss_{timestamp}.png")
    plt.close()
    
    # 2) e_loss with rolling mean
    plt.figure(figsize=(10, 6))
    plt.plot(df["step"], df["e_loss"], alpha=0.3, label="e_loss (raw)", color='C1')
    plt.plot(df["step"], df["e_loss_rm"], label=f"e_loss (rolling mean, win={win})", 
             color='C1', linewidth=2)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Energy MSE (eV²)", fontsize=12)
    plt.title("Energy Loss Curve", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"pic/e_loss_{timestamp}.png", dpi=300)
    print(f"Saved: pic/e_loss_{timestamp}.png")
    plt.close()
    
    # 3) Learning rate
    plt.figure(figsize=(10, 6))
    plt.plot(df["step"], df["lr"], label="Learning Rate", color='C2', linewidth=2)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Learning Rate", fontsize=12)
    plt.title("Learning Rate Schedule", fontsize=14)
    plt.yscale('log')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"pic/lr_{timestamp}.png", dpi=300)
    print(f"Saved: pic/lr_{timestamp}.png")
    plt.close()
    
    # 4) Loss consistency (diff)
    plt.figure(figsize=(10, 6))
    plt.plot(df["step"], df["diff"], label="|loss - (pref_e*e_loss + pref_f*f_loss)|", 
             color='C3', linewidth=1)
    plt.axhline(y=1e-5, color='r', linestyle='--', label='Threshold (1e-5)')
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Loss Consistency Error", fontsize=12)
    plt.title("Loss Consistency Check", fontsize=14)
    plt.yscale('log')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"pic/diff_{timestamp}.png", dpi=300)
    print(f"Saved: pic/diff_{timestamp}.png")
    plt.close()
    
    # 5) Combined plot: e_loss + f_loss on dual y-axis
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.set_xlabel("Step", fontsize=12)
    ax1.set_ylabel("Energy MSE (eV²)", color='C1', fontsize=12)
    ax1.plot(df["step"], df["e_loss_rm"], color='C1', linewidth=2, label='e_loss')
    ax1.tick_params(axis='y', labelcolor='C1')
    ax1.grid(alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Force MSE (eV²/Å²)", color='C0', fontsize=12)
    ax2.plot(df["step"], df["f_loss_rm"], color='C0', linewidth=2, label='f_loss')
    ax2.tick_params(axis='y', labelcolor='C0')
    
    plt.title("Energy & Force Loss (Rolling Mean)", fontsize=14)
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    fig.tight_layout()
    plt.savefig(f"pic/combined_losses_{timestamp}.png", dpi=300)
    print(f"Saved: pic/combined_losses_{timestamp}.png")
    plt.close()
    
    # Export data
    csv_path = f"pic/training_curves_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    print("\n✅ All plots and data exported to pic/")
    print(f"   Timestamp: {timestamp}")


if __name__ == "__main__":
    main()
