#!/bin/bash
# 完整的验证集评估 + 决策流程
# 自动化评估 checkpoint、检查收敛、生成决策报告

set -e

PROJECT_DIR="/home/ubuntu/pj"
cd "$PROJECT_DIR"

echo "========================================================================="
echo "🔍 验证集评估 + 收敛性决策完整流程"
echo "========================================================================="
echo ""

# ============================================================================
# 第 1 步: 检查验证集是否存在
# ============================================================================
echo "✓ 第 1/5 步: 验证前置条件"
echo "-----------------------------------------------------------------------"

VAL_INDICES_FILE="logs/val_indices.npy"
if [ ! -f "$VAL_INDICES_FILE" ]; then
    echo "❌ 验证集索引文件不存在，创建一个..."
    python3 << 'EOF'
import numpy as np
np.random.seed(42)
val_indices = np.random.choice(1823, size=300, replace=False)
np.save('logs/val_indices.npy', val_indices)
print(f"✓ 创建验证集: {len(val_indices)} 帧，保存到 {VAL_INDICES_FILE}")
EOF
fi

echo "✓ 验证集索引: $VAL_INDICES_FILE"
echo ""

# ============================================================================
# 第 2 步: 评估原始训练的最新 checkpoint
# ============================================================================
echo "✓ 第 2/5 步: 评估原始训练 (Original)"
echo "-----------------------------------------------------------------------"

python3 batch_evaluate_checkpoints.py \
    --ckpt-dir checkpoints_fixed \
    --output logs/ckpt_eval_fixed.csv \
    --config config_formal_100k_fixed.json \
    --n-latest 5 \
    2>&1 | grep -E "📊|📝|✓|✗|Step|E_RMSE|F_RMSE"

echo ""

# ============================================================================
# 第 3 步: 评估 V3 训练的最新 checkpoint
# ============================================================================
echo "✓ 第 3/5 步: 评估 V3 训练 (Optimized)"
echo "-----------------------------------------------------------------------"

python3 batch_evaluate_checkpoints.py \
    --ckpt-dir checkpoints_optimized_v3 \
    --output logs/ckpt_eval_v3.csv \
    --config config_formal_100k_optimized.json \
    --n-latest 5 \
    2>&1 | grep -E "📊|📝|✓|✗|Step|E_RMSE|F_RMSE"

echo ""

# ============================================================================
# 第 4 步: 检查原始训练收敛
# ============================================================================
echo "✓ 第 4/5 步: 检查原始训练收敛情况"
echo "-----------------------------------------------------------------------"

ORIGINAL_DECISION=$(python3 check_convergence.py \
    --csv logs/ckpt_eval_fixed.csv \
    --metric f_rmse \
    --threshold 0.01 \
    --window 3 2>&1 | tail -1 | python3 -c "import sys, json; print(json.load(sys.stdin)['decision'])")

echo "决策: $ORIGINAL_DECISION"

if [ "$ORIGINAL_DECISION" = "STOP" ]; then
    echo "⊘ 原始训练已收敛，建议停止"
    ORIGINAL_CONTINUE=false
else
    echo "✓ 原始训练仍有改进空间，可继续"
    ORIGINAL_CONTINUE=true
fi
echo ""

# ============================================================================
# 第 5 步: 检查 V3 训练收敛
# ============================================================================
echo "✓ 第 5/5 步: 检查 V3 训练收敛情况"
echo "-----------------------------------------------------------------------"

V3_DECISION=$(python3 check_convergence.py \
    --csv logs/ckpt_eval_v3.csv \
    --metric f_rmse \
    --threshold 0.01 \
    --window 3 2>&1 | tail -1 | python3 -c "import sys, json; print(json.load(sys.stdin)['decision'])")

echo "决策: $V3_DECISION"

if [ "$V3_DECISION" = "STOP" ]; then
    echo "⊘ V3 训练已收敛，建议停止"
    V3_CONTINUE=false
else
    echo "✓ V3 训练仍有改进空间，可继续"
    V3_CONTINUE=true
fi
echo ""

# ============================================================================
# 最终建议
# ============================================================================
echo "========================================================================="
echo "📋 最终建议总结"
echo "========================================================================="
echo ""

# 获取最新的性能数据
ORIGINAL_F_RMSE=$(tail -1 logs/ckpt_eval_fixed.csv | cut -d',' -f3)
V3_F_RMSE=$(tail -1 logs/ckpt_eval_v3.csv | cut -d',' -f3)

echo "🏆 性能对比:"
echo "  - 原始训练: F_RMSE = $ORIGINAL_F_RMSE eV/Å"
echo "  - V3 训练:   F_RMSE = $V3_F_RMSE eV/Å"

if (( $(echo "$V3_F_RMSE < $ORIGINAL_F_RMSE" | bc -l) )); then
    IMPROVEMENT=$(echo "scale=1; ($ORIGINAL_F_RMSE - $V3_F_RMSE) / $ORIGINAL_F_RMSE * 100" | bc -l)
    echo "  ✓ V3 更优 (改进 $IMPROVEMENT%)"
else
    echo "  ✓ 原始训练更优"
fi

echo ""
echo "🎯 建议操作:"
echo ""

if [ "$ORIGINAL_CONTINUE" = "false" ]; then
    echo "  原始训练:"
    echo "    ⊘ 已收敛，可停止"
    echo "    命令: kill -INT 138895"
    echo ""
else
    echo "  原始训练:"
    echo "    ✓ 继续训练至 100k 步"
    echo ""
fi

if [ "$V3_CONTINUE" = "false" ]; then
    echo "  V3 训练:"
    echo "    ⊘ 已收敛，建议停止"
    echo "    命令: kill -INT 147981"
    echo "    推荐使用 checkpoint: checkpoints_optimized_v3/model_step80000.pt"
    echo ""
else
    echo "  V3 训练:"
    echo "    ✓ 继续训练至 100k 步"
    echo ""
fi

echo "========================================================================="
echo "✓ 流程完成！详见报告文件:"
echo "  - logs/ckpt_eval_fixed.csv (原始训练评估结果)"
echo "  - logs/ckpt_eval_v3.csv (V3 训练评估结果)"
echo "  - VALIDATION_CHECKPOINT_ANALYSIS.md (完整分析报告)"
echo "========================================================================="
