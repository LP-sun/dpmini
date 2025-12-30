#!/bin/bash

cd /home/ubuntu/pj

echo "======================================================================"
echo "启动优化版本V3的DeepMD训练 (DataLoader + Accum=4)"
echo "======================================================================"
echo ""
echo "优化配置:"
echo "  ✓ DataLoader: 启用 (4 workers, persistent_workers)"
echo "  ✓ Batch Size: 4 (真实批处理)"
echo "  ✓ Grad Accumulation: 4 (仅累积末尾step)"
echo "  ✓ pin_memory: True, prefetch_factor: 2"
echo "  ✓ non_blocking H2D"
echo ""

# 非交互shell中激活conda
if [ -f "/home/ubuntu/miniforge3/etc/profile.d/conda.sh" ]; then
  source "/home/ubuntu/miniforge3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null)"
  [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] && source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

set -e
conda activate ai4m

# 使用 -u 禁用stdout缓冲，便于实时查看日志
python -u train_optimized_dataloader_v3.py \
  --config config_formal_100k_optimized.json \
  --checkpoint-dir checkpoints_optimized_v3 \
  --export-dir exports_optimized_v3 \
  --force-loss mse \
  --grad-accumulation-steps 4 \
  --num-workers 4 \
  --batch-size 4 \
  "$@"

echo ""
echo "======================================================================"
echo "V3训练完成!"
echo "======================================================================"
