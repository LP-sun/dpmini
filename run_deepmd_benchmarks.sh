#!/usr/bin/env bash
set -euo pipefail

# 自动基准测试 DeepMD-kit (PyTorch backend) 在不同 batch_size / num-workers 下的吞吐
# 需求：bash + python3 + nvidia-smi（无 root 权限）
# 用法：
#   chmod +x run_deepmd_benchmarks.sh
#   ./run_deepmd_benchmarks.sh se_e2_a/input_torch.json 500
# 可选环境变量：
#   BENCH_TIME=300   # 每组最多运行秒数（>0 时使用 timeout）
#   BATCH_SPECS="8 auto:4096 auto:8192 auto:16384"
#   WORKERS="0 2 4"
#   NATOMS=192       # 每帧原子数，用于 auto:N → batch_size
#   DISP_FREQ=500    # 基准时的打印频率（取 min(DISP_FREQ, bench_steps)）
#   LOG_ROOT=bench_runs
#   CSV_PATH=benchmark_results.csv
#   OMP_NUM_THREADS/MKL_NUM_THREADS 覆盖 CPU 线程数

CONFIG=${1:-se_e2_a/input_torch.json}
BENCH_STEPS=${2:-500}
BENCH_TIME=${BENCH_TIME:-0}
BATCH_SPECS_STR=${BATCH_SPECS:-"8 auto:4096 auto:8192 auto:16384"}
WORKERS_STR=${WORKERS:-"0 2 4"}
NATOMS=${NATOMS:-192}
DISP_FREQ_BASE=${DISP_FREQ:-500}
LOG_ROOT=${LOG_ROOT:-bench_runs}
CSV_PATH=${CSV_PATH:-benchmark_results.csv}
OMP_THREADS=${OMP_NUM_THREADS:-8}
MKL_THREADS=${MKL_NUM_THREADS:-8}

mkdir -p "${LOG_ROOT}"

# CSV header
if [[ ! -f "${CSV_PATH}" ]]; then
  echo "batch_spec,batch_size,num_workers,steps,duration_s,steps_per_s,frames_per_s,gpu_util_avg,gpu_mem_avg_mib,cpu_sum_avg_pct,gpu_shared,notes,log_path" >"${CSV_PATH}"
fi

# prerequisite checks
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi 未找到" >&2; exit 1; fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3 未找到" >&2; exit 1; fi

calc_batch_size() {
  local spec="$1"
  if [[ "$spec" =~ ^auto: ]]; then
    local target=${spec#auto:}
    python3 - "$target" "$NATOMS" <<'PY'
import sys, math
target=float(sys.argv[1]); natoms=float(sys.argv[2])
print(max(1, int(math.ceil(target / natoms))))
PY
  else
    echo "$spec"
  fi
}

slugify() { echo "$1" | tr ':' '_' | tr ' ' '_' | tr -c 'A-Za-z0-9_-' '_'; }

check_gpu_sharing() {
  local out lines
  out=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)
  lines=$(echo "$out" | grep -E '^[0-9]' || true)
  if [[ -n "$lines" ]]; then
    echo "YES:${lines//$'\n'/; }"
  else
    echo "NO"
  fi
}

make_temp_config() {
  local base_config="$1" batch_size="$2" disp_freq="$3" out_path="$4"
  python3 - "$base_config" "$batch_size" "$disp_freq" "$BENCH_STEPS" "$out_path" <<'PY'
import json, sys, pathlib
base, bs, disp, steps, out_path = sys.argv[1:6]
bs = int(bs); disp = int(disp); steps = int(steps)
base_path = pathlib.Path(base).resolve()
base_dir = base_path.parent
with base_path.open() as f:
    cfg = json.load(f)
train = cfg.get("training", {})

def _abs_paths(paths):
    out = []
    for p in paths:
        out.append(str((base_dir / pathlib.Path(p)).resolve()))
    return out

td = train.setdefault("training_data", {})
td["batch_size"] = bs
if "systems" in td:
    td["systems"] = _abs_paths(td["systems"])

vd = train.setdefault("validation_data", {})
vd.setdefault("batch_size", bs)
vd["numb_btch"] = 1
if "systems" in vd:
    vd["systems"] = _abs_paths(vd["systems"])

train["disp_freq"] = disp
train["numb_steps"] = steps
cfg["training"] = train
pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(out_path).write_text(json.dumps(cfg, indent=2))
PY
}

summaries() {
  python3 - "$CSV_PATH" <<'PY'
import csv, sys
path = sys.argv[1]
rows = []
with open(path, newline='') as f:
    reader = csv.DictReader(f)
    for r in reader:
        try:
            r["frames_per_s"] = float(r.get("frames_per_s", "nan"))
        except Exception:
            r["frames_per_s"] = float("nan")
        rows.append(r)
rows.sort(key=lambda x: x.get("frames_per_s", float('-inf')), reverse=True)
if not rows:
    print("[汇总] CSV 无数据")
    sys.exit(0)
cols = ["batch_spec","batch_size","num_workers","steps_per_s","frames_per_s","gpu_util_avg","cpu_sum_avg_pct","gpu_shared"]
widths = {c: len(c) for c in cols}
for r in rows:
    for c in cols:
        widths[c] = max(widths[c], len(str(r.get(c, ""))))
header = " | ".join(c.ljust(widths[c]) for c in cols)
print(header)
print("-" * len(header))
for r in rows:
    print(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
best = next((r for r in rows if str(r.get("gpu_shared", "NO")).upper().startswith("NO")), rows[0])
print("\n[推荐] batch_spec=%s, workers=%s, frames/s=%.2f (gpu_shared=%s)" % (
    best.get("batch_spec"), best.get("num_workers"), float(best.get("frames_per_s", -1)), best.get("gpu_shared", "?")))
PY
}

BATCH_SPECS=($BATCH_SPECS_STR)
WORKERS=($WORKERS_STR)

for spec in "${BATCH_SPECS[@]}"; do
  bs=$(calc_batch_size "$spec")
  for w in "${WORKERS[@]}"; do
    label="$(slugify "${spec}_w${w}")"
    run_dir="${LOG_ROOT}/${label}"
    mkdir -p "$run_dir"
    log_file="${run_dir}/train.log"
    gpu_log="${run_dir}/gpu.log"
    cpu_log="${run_dir}/cpu.log"
    tmp_cfg="${run_dir}/config.json"
    ckpt_dir="${run_dir}/checkpoints"
    exp_dir="${run_dir}/exports"

    disp_freq=$(( BENCH_STEPS < DISP_FREQ_BASE ? BENCH_STEPS : DISP_FREQ_BASE ))
    make_temp_config "$CONFIG" "$bs" "$disp_freq" "$tmp_cfg"
    [[ -f "$tmp_cfg" ]] || { echo "[ERROR] 临时配置未生成: $tmp_cfg" >&2; exit 1; }

    gpu_shared=$(check_gpu_sharing)

    echo "[RUN] spec=${spec} (batch=${bs}), workers=${w}, disp_freq=${disp_freq}, bench_steps=${BENCH_STEPS}, gpu_shared=${gpu_shared}"
    echo "  logs: ${log_file}"

    start_ts=$(date +%s)

    # GPU sampler
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 >"${gpu_log}" 2>/dev/null &
    gpu_mon_pid=$!

    monitor_cpu() {
      local pid="$1"
      while kill -0 "$pid" 2>/dev/null; do
        P_CPU=$(ps -p "$pid" -o %cpu= | tr '\n' ' ')
        C_CPU=$(ps --ppid "$pid" -o %cpu= | tr '\n' ' ')
        P_CPU="$P_CPU" C_CPU="$C_CPU" python3 - <<'PY'
import os
def total(txt: str) -> float:
    return sum(float(x) for x in txt.split() if x.strip())
p = total(os.environ.get("P_CPU", ""))
c = total(os.environ.get("C_CPU", ""))
print(f"{p+c:.2f}")
PY
        sleep 1
      done
    }

    # Launch training (optionally with timeout)
    mkdir -p "$ckpt_dir" "$exp_dir"
    if [[ "$BENCH_TIME" -gt 0 ]]; then
      timeout --signal=INT "${BENCH_TIME}"s \
        env OMP_NUM_THREADS="$OMP_THREADS" MKL_NUM_THREADS="$MKL_THREADS" PYTHONUNBUFFERED=1 \
        python3 -u train_cuda.py --config "$tmp_cfg" --checkpoint-dir "$ckpt_dir" --export-dir "$exp_dir" --num-workers "$w" \
        >"${log_file}" 2>&1 &
    else
      OMP_NUM_THREADS="$OMP_THREADS" MKL_NUM_THREADS="$MKL_THREADS" PYTHONUNBUFFERED=1 \
        python3 -u train_cuda.py --config "$tmp_cfg" --checkpoint-dir "$ckpt_dir" --export-dir "$exp_dir" --num-workers "$w" \
        >"${log_file}" 2>&1 &
    fi
    train_pid=$!

    monitor_cpu "$train_pid" >"${cpu_log}" 2>/dev/null &
    cpu_mon_pid=$!

    wait "$train_pid" || true
    end_ts=$(date +%s)

    # stop samplers
    kill "$gpu_mon_pid" "$cpu_mon_pid" 2>/dev/null || true

    duration=$(( end_ts - start_ts ))
    [[ $duration -le 0 ]] && duration=1

    steps_run=$(python3 - "$log_file" <<'PY'
import sys, re
path = sys.argv[1]
s = 0
try:
    with open(path, errors="ignore") as f:
        for line in f:
            m = re.search(r"Step\s+(\d+)", line)
            if m:
                s = int(m.group(1))
except FileNotFoundError:
    s = 0
print(s)
PY
)
    if [[ -z "$steps_run" || "$steps_run" -le 0 ]]; then
      steps_run=$BENCH_STEPS
    fi

    steps_per_s=$(python3 - "$steps_run" "$duration" <<'PY'
import sys
steps=float(sys.argv[1]); dur=float(sys.argv[2]);
print(f"{steps/dur:.4f}")
PY
)
    frames_per_s=$(python3 - "$steps_per_s" "$bs" <<'PY'
import sys
s=float(sys.argv[1]); b=float(sys.argv[2]);
print(f"{s*b:.4f}")
PY
)

    gpu_util_avg=$(python3 - "$gpu_log" <<'PY'
import sys
vals = []
path = sys.argv[1]
try:
    with open(path) as f:
        for line in f:
            parts = line.strip().split(',')
            if not parts:
                continue
            try:
                vals.append(float(parts[0]))
            except Exception:
                pass
except FileNotFoundError:
    pass
print(f"{sum(vals)/len(vals):.2f}" if vals else "NA")
PY
)

    gpu_mem_avg=$(python3 - "$gpu_log" <<'PY'
import sys
vals = []
path = sys.argv[1]
try:
    with open(path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            try:
                vals.append(float(parts[1]))
            except Exception:
                pass
except FileNotFoundError:
    pass
print(f"{sum(vals)/len(vals):.2f}" if vals else "NA")
PY
)

    cpu_sum_avg=$(python3 - "$cpu_log" <<'PY'
import sys
vals = []
path = sys.argv[1]
try:
    with open(path) as f:
        for line in f:
            try:
                vals.append(float(line.strip()))
            except Exception:
                pass
except FileNotFoundError:
    pass
print(f"{sum(vals)/len(vals):.2f}" if vals else "NA")
PY
)

    echo "${spec},${bs},${w},${steps_run},${duration},${steps_per_s},${frames_per_s},${gpu_util_avg},${gpu_mem_avg},${cpu_sum_avg},${gpu_shared},,${log_file}" >>"${CSV_PATH}"

    echo "  -> steps=${steps_run}, duration=${duration}s, steps/s=${steps_per_s}, frames/s=${frames_per_s}, gpu_util_avg=${gpu_util_avg}, cpu_sum_avg=${cpu_sum_avg}"
  done
done

echo
summaries
