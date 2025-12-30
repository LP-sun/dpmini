#!/bin/bash

echo "======================================================================"
echo "性能对比测试: 原始版本 vs 优化版本"
echo "======================================================================"
echo ""

# 测试步数
TEST_STEPS=200

echo "测试配置:"
echo "  - 训练步数: $TEST_STEPS"
echo "  - 配置: config_formal_100k_optimized.json"
echo "  - 数据: collect/O64H128 (1823帧)"
echo ""

# Test 1: 原始版本 (不使用DataLoader)
echo "========== 测试1: 原始版本 (batch_size=1, no DataLoader) =========="
echo "命令:"
echo "python train_cuda_optimized.py \\"
echo "  --config config_formal_100k_optimized.json \\"
echo "  --checkpoint-dir checkpoints_fixed \\"
echo "  --export-dir exports_fixed \\"
echo "  --force-loss mse \\"
echo "  --grad-accumulation-steps 4 \\"
echo "  --num-workers 0 \\"
echo "  --test-steps $TEST_STEPS"
echo ""

start_time=$(date +%s)
timeout 300 python train_cuda_optimized.py \
  --config config_formal_100k_optimized.json \
  --checkpoint-dir checkpoints_fixed \
  --export-dir exports_fixed \
  --force-loss mse \
  --grad-accumulation-steps 4 \
  --num-workers 0 \
  --test-steps $TEST_STEPS 2>&1 | grep -E "Step|completed" | head -20
end_time=$(date +%s)

original_time=$((end_time - start_time))
echo ""
echo "原始版本耗时: $original_time 秒"
echo ""

# Test 2: 优化版本 (使用DataLoader)
echo "========== 测试2: 优化版本 (batch_size=4, DataLoader, 4 workers) =========="
echo "命令:"
echo "python train_optimized_dataloader.py \\"
echo "  --config config_formal_100k_optimized.json \\"
echo "  --checkpoint-dir checkpoints_optimized \\"
echo "  --export-dir exports_optimized \\"
echo "  --force-loss mse \\"
echo "  --grad-accumulation-steps 1 \\"
echo "  --num-workers 4 \\"
echo "  --batch-size 4 \\"
echo "  --test-steps $TEST_STEPS"
echo ""

start_time=$(date +%s)
timeout 300 python train_optimized_dataloader.py \
  --config config_formal_100k_optimized.json \
  --checkpoint-dir checkpoints_optimized \
  --export-dir exports_optimized \
  --force-loss mse \
  --grad-accumulation-steps 1 \
  --num-workers 4 \
  --batch-size 4 \
  --test-steps $TEST_STEPS 2>&1 | grep -E "Step|completed" | head -20
end_time=$(date +%s)

optimized_time=$((end_time - start_time))
echo ""
echo "优化版本耗时: $optimized_time 秒"
echo ""

# 计算性能提升
if [ $original_time -gt 0 ]; then
    improvement=$((100 * (original_time - optimized_time) / original_time))
    echo "======================================================================"
    echo "性能提升: $improvement%"
    echo "加速比: $(echo "scale=2; $original_time / $optimized_time" | bc)x"
    echo "======================================================================"
else
    echo "原始版本测试失败"
fi

