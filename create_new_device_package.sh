#!/bin/bash
# Create a clean, minimal package to run training on a new device
set -euo pipefail

PKG_DIR="deepmd_new_device_package"
PKG_TGZ="deepmd_new_device_package_$(date +%Y%m%d-%H%M%S).tar.gz"

echo "======================================================================"
echo "Building new-device package: $PKG_TGZ"
echo "======================================================================"

rm -rf "$PKG_DIR"
mkdir -p "$PKG_DIR"

echo "[1/6] Copy core training scripts"
cp train_cuda_optimized.py "$PKG_DIR/"
cp train_optimized_dataloader_v3.py "$PKG_DIR/"
cp run_optimized_training_v3.sh "$PKG_DIR/"

echo "[2/6] Copy configs"
cp config_formal_100k_optimized.json "$PKG_DIR/" 2>/dev/null || true
cp config_formal_100k_fixed.json "$PKG_DIR/" 2>/dev/null || true
cp config_minimal.json "$PKG_DIR/" 2>/dev/null || true

echo "[3/6] Copy library code (dpmini)"
cp -r dpmini "$PKG_DIR/"

echo "[4/6] Add helper scripts"
cp check_environment.sh "$PKG_DIR/" 2>/dev/null || true

echo "[5/6] Create requirements.txt (Torch installed separately)"
cat > "$PKG_DIR/requirements.txt" << 'EOF'
numpy>=1.24
EOF

echo "[6/6] Create DEPLOY.md"
cat > "$PKG_DIR/DEPLOY.md" << 'EOF'
# DeepMD Training (New Device) - Deployment Guide

This package contains the minimal files to run the original and the optimized (v3) training on a new device.

## Contents
- Scripts: `train_cuda_optimized.py` (原始), `train_optimized_dataloader_v3.py` (优化v3)
- Launcher: `run_optimized_training_v3.sh`
- Configs: `config_formal_100k_optimized.json`, `config_formal_100k_fixed.json`, `config_minimal.json`
- Library: `dpmini/`
- Utils: `check_environment.sh`, `requirements.txt`

## 1) Prepare Python environment
Use conda (recommended) or system Python. Install PyTorch based on your CUDA driver.

```bash
conda create -n deepmd python=3.10 -y
conda activate deepmd

# Install PyTorch (choose the right command for your CUDA)
# CUDA 11.8 example:
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio

# Then basic deps
pip install -r requirements.txt
```

Check CUDA:
```bash
python - <<'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
print('Torch:', torch.__version__)
print('CUDA build:', torch.version.cuda)
PY
```

## 2) Prepare data
Training expects DeepMD-style data under a directory, e.g. `collect/O64H128`.

- Option A: Sync from the source host (recommended)
```bash
# on target device, from the same user or with SSH configured
mkdir -p collect
rsync -avP user@source-host:/home/ubuntu/pj/collect/O64H128 collect/
```

- Option B: Use another compatible dataset and pass `--data-dir` flag to scripts.

## 3) Quick functional test (small run)
Run 200 steps to validate environment and I/O, without long training time.

```bash
# Optimized v3 (单机单卡)
python -u train_optimized_dataloader_v3.py \
  --config config_formal_100k_optimized.json \
  --data-dir collect/O64H128 \
  --checkpoint-dir checkpoints_v3_test \
  --export-dir exports_v3_test \
  --grad-accumulation-steps 4 \
  --num-workers 4 \
  --batch-size 4 \
  --test-steps 200

# 原始脚本（对比）
python -u train_cuda_optimized.py \
  --config config_formal_100k_fixed.json \
  --checkpoint-dir checkpoints_fixed_test \
  --export-dir exports_fixed_test \
  --grad-accumulation-steps 4 \
  --force-loss mse \
  --test-steps 200
```

## 4) Full training (单独运行，避免与其他任务争抢GPU)
使用提供的启动器（禁用stdout缓冲，日志更及时）。

```bash
bash run_optimized_training_v3.sh \
  --data-dir collect/O64H128 \
  --checkpoint-dir checkpoints_optimized_v3 \
  --export-dir exports_optimized_v3 \
  --batch-size 4 \
  --num-workers 4 \
  --grad-accumulation-steps 4
```

或直接调用脚本（同等参数）：

```bash
python -u train_optimized_dataloader_v3.py \
  --config config_formal_100k_optimized.json \
  --data-dir collect/O64H128 \
  --checkpoint-dir checkpoints_optimized_v3 \
  --export-dir exports_optimized_v3 \
  --force-loss mse \
  --grad-accumulation-steps 4 \
  --num-workers 4 \
  --batch-size 4
```

## 5) Monitoring
```bash
watch -n 2 nvidia-smi
tail -f logs/*.log  # 如将stdout重定向到日志
```

## Notes
- 请单独运行任务，避免与其他训练并发从而拖慢速度与误导ETA。
- 若日志不够及时，可确保使用 `python -u` 或设置 `PYTHONUNBUFFERED=1`。
- 若显存更充足，可尝试 `--batch-size 8` 并适当提升 `--num-workers`。
EOF

echo "Creating tarball: $PKG_TGZ"
tar -czf "$PKG_TGZ" "$PKG_DIR"
echo "Done. Package: $PKG_TGZ"
echo "You can transfer with: scp $PKG_TGZ <user>@<host>:/path/"
