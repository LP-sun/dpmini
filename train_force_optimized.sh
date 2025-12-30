#!/bin/bash
# Force Loss 优化训练 - 推荐配置
# 适用于短期训练 (2k-20k steps)

set -e

CONFIG="${1:-config_short_test.json}"
DATA_DIR="${2:-collect/O64H128}"
STEPS="${3:-10000}"

echo "═══════════════════════════════════════════════════"
echo "  Force Loss 优化训练启动器"
echo "═══════════════════════════════════════════════════"
echo "配置: $CONFIG"
echo "数据: $DATA_DIR"
echo "步数: $STEPS"
echo ""
echo "优化特性:"
echo "  ✓ grad_accumulation_steps=8 (等效 batch=8)"
echo "  ✓ limit_pref_f=20 (后期仍重视 force)"
echo "  ✓ stop_lr=1e-6 (短训练不会学不动)"
echo "  ✓ f_rmse 实时打印 (直观力误差)"
echo "  ✓ mixed precision (FP16 加速)"
echo "═══════════════════════════════════════════════════"
echo ""

# 确保使用正确的环境
conda run -n ai4m python train_cuda_optimized.py \
  --config "$CONFIG" \
  --data-dir "$DATA_DIR" \
  --grad-accumulation-steps 8 \
  --mixed-precision \
  --force-loss mse

echo ""
echo "═══════════════════════════════════════════════════"
echo "  训练完成！"
echo "═══════════════════════════════════════════════════"
echo "检查点: checkpoints_cuda/"
echo "导出模型: exports_cuda/model_final.pth"
echo "训练日志: cuda_training_opt.log"
echo ""
echo "生成可视化报告："
echo "  conda run -n ai4m python plot_training_curves_fixed_v1.py"
echo ""
