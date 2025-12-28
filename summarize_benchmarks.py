#!/usr/bin/env python3
"""
读取 benchmark_results.csv，输出终端表格、推荐参数，并可选绘图。
用法：
  python3 summarize_benchmarks.py [benchmark_results.csv]
环境：可选安装 matplotlib（若存在则自动画图到 pic/）。
"""
import csv
import sys
import os
from datetime import datetime
from typing import List, Dict

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.csv"

if not os.path.exists(CSV_PATH):
    sys.exit(f"未找到 CSV: {CSV_PATH}")

rows: List[Dict[str, str]] = []
with open(CSV_PATH, newline='') as f:
    reader = csv.DictReader(f)
    for r in reader:
        try:
            r["frames_per_s"] = float(r.get("frames_per_s", "nan"))
        except Exception:
            r["frames_per_s"] = float("nan")
        try:
            r["steps_per_s"] = float(r.get("steps_per_s", "nan"))
        except Exception:
            r["steps_per_s"] = float("nan")
        try:
            r["gpu_util_avg"] = float(r.get("gpu_util_avg", "nan"))
        except Exception:
            r["gpu_util_avg"] = float("nan")
        try:
            r["cpu_sum_avg_pct"] = float(r.get("cpu_sum_avg_pct", "nan"))
        except Exception:
            r["cpu_sum_avg_pct"] = float("nan")
        rows.append(r)

if not rows:
    sys.exit("CSV 为空，无数据可汇总")

rows.sort(key=lambda x: x.get("frames_per_s", float('-inf')), reverse=True)

cols = ["batch_spec", "batch_size", "num_workers", "steps_per_s", "frames_per_s", "gpu_util_avg", "cpu_sum_avg_pct", "gpu_shared"]
widths = {c: len(c) for c in cols}
for r in rows:
    for c in cols:
        widths[c] = max(widths[c], len(str(r.get(c, ""))))

def fmt(val, key):
    if isinstance(val, float):
        if val != val:  # NaN
            return "NA"
        if key in {"steps_per_s", "frames_per_s", "gpu_util_avg", "cpu_sum_avg_pct"}:
            return f"{val:.2f}"
    return str(val)

header = " | ".join(c.ljust(widths[c]) for c in cols)
sep = "-" * len(header)
print(header)
print(sep)
for r in rows:
    print(" | ".join(fmt(r.get(c, ""), c).ljust(widths[c]) for c in cols))

best = next((r for r in rows if str(r.get("gpu_shared", "NO")).upper().startswith("NO")), rows[0])
print("\n推荐组合：batch_spec=%s, num_workers=%s, frames/s=%.2f, gpu_util=%.2f, cpu_sum=%.2f%%, gpu_shared=%s" % (
    best.get("batch_spec"),
    best.get("num_workers"),
    best.get("frames_per_s", float('nan')),
    best.get("gpu_util_avg", float('nan')),
    best.get("cpu_sum_avg_pct", float('nan')),
    best.get("gpu_shared", "?")
))

# 绘图（可选）
try:
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit(0)

os.makedirs("pic", exist_ok=True)
ts = datetime.now().strftime("%Y%m%d-%H%M%S")

# frames/s vs batch_size
plt.figure(figsize=(6, 4))
plt.title("Frames/s vs batch_size")
x = [str(r.get("batch_spec", "")) for r in rows]
y = [r.get("frames_per_s", float('nan')) for r in rows]
plt.plot(x, y, marker='o')
plt.xlabel("batch_spec")
plt.ylabel("frames/s")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plot1 = os.path.join("pic", f"frames_vs_batch_{ts}.png")
plt.savefig(plot1, dpi=200)
print(f"保存图表: {plot1}")

# frames/s vs workers
plt.figure(figsize=(6, 4))
plt.title("Frames/s vs num_workers")
workers = [int(r.get("num_workers", 0)) for r in rows]
frames = [r.get("frames_per_s", float('nan')) for r in rows]
plt.scatter(workers, frames, c='tab:blue')
plt.xlabel("num_workers")
plt.ylabel("frames/s")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plot2 = os.path.join("pic", f"frames_vs_workers_{ts}.png")
plt.savefig(plot2, dpi=200)
print(f"保存图表: {plot2}")
