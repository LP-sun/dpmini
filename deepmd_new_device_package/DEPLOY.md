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
