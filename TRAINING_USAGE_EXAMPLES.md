# 训练质量修复 - 使用示例

本文档展示如何使用修复后的训练系统并验证"健康"指标。

---

## 完整工作流

### 1️⃣ 验证修复 (必须)

```bash
# 运行自检脚本
python test_training_fixes.py
```

**预期输出**:
```
✅ ALL TESTS PASSED
Training fixes validated:
  ✓ Shape handling correct (batch_size=1)
  ✓ Loss consistency verified (diff < 1e-5)
  ✓ Early convergence confirmed
  ✓ No NaN/Inf in descriptor or forces
```

---

### 2️⃣ 短训练测试 (500-2000步)

```bash
# 启动训练
python train_cuda_optimized.py \
    --config se_e2_a/input_torch.json \
    --data-dir collect/O64H128 \
    --grad-accumulation-steps 4 \
    --num-workers 4 \
    2>&1 | tee quick_test_fixed.log

# 训练过程中会自动写入 cuda_training_opt.log
```

**实时监控**:
```bash
# 查看最新日志
tail -f cuda_training_opt.log

# 检查 diff 一致性
grep "diff=" cuda_training_opt.log | tail -20
```

---

### 3️⃣ 健康检查 (关键步骤)

训练 500-2000 步后（或随时）：

```bash
# 绘制曲线并生成健康报告
python plot_training_curves_fixed_v1.py --log cuda_training_opt.log
```

**健康报告示例**:
```
================================================================================
HEALTH CHECK REPORT
================================================================================

[A] Loss Consistency:
    max(diff) = 2.384e-07
    mean(diff) = 1.123e-07
    ✅ PASS: diff < 1e-5 (loss consistency verified)

[B] Force Loss Trend:
    Early f_loss (rolling mean): 1.423690
    Later f_loss (rolling mean): 0.534821
    Reduction: 62.4%
    ✅ PASS: f_loss shows downward trend

[C] Numerical Stability:
    f_loss range: [0.298765, 2.134567]
    e_loss range: [1.234e+01, 5.678e+05]
    ✅ PASS: No exploded loss values

================================================================================
```

**输出文件** (保存在 `pic/` 目录):
- `f_loss_YYYYMMDD-HHMMSS.png` - 力损失曲线 ⭐
- `e_loss_YYYYMMDD-HHMMSS.png` - 能量损失曲线
- `lr_YYYYMMDD-HHMMSS.png` - 学习率曲线
- `diff_YYYYMMDD-HHMMSS.png` - 一致性检查
- `combined_losses_YYYYMMDD-HHMMSS.png` - 双轴组合图
- `training_curves_YYYYMMDD-HHMMSS.csv` - 原始数据

---

### 4️⃣ 判定标准

#### ✅ 健康训练的特征

查看 `pic/f_loss_*.png`:
- **Rolling mean 曲线向下**（最重要）
- 典型模式：从 ~1.0 降到 ~0.4
- 允许波动，但趋势明确

查看健康报告:
- **指标A**: `max(diff) < 1e-5` ✅
- **指标B**: `Reduction > 10%` ✅
- **指标C**: `f_loss < 1e3` ✅

#### ⚠️ 如果 f_loss 平台

**症状**: 指标A/C通过，但指标B失败（f_loss无明显下降）

**解决方案**:
```bash
# 1. 应用严格 sel 语义修复
python fix_neighbor_list_strict_sel.py --apply

# 2. 重新验证
python test_training_fixes.py

# 3. 重新训练并对比
python train_cuda_optimized.py \
    --config se_e2_a/input_torch.json \
    --data-dir collect/O64H128 \
    --grad-accumulation-steps 4

# 4. 再次检查健康指标
python plot_training_curves_fixed_v1.py
```

**预期改善**:
- 更稳定的 f_loss 下降
- 更符合 DeepMD 参考结果
- 特别对类型分布不均的构型效果明显

---

### 5️⃣ 完整训练

确认健康后，启动完整训练：

```bash
python train_cuda_optimized.py \
    --config se_e2_a/input_torch.json \
    --data-dir collect/O64H128 \
    --grad-accumulation-steps 8 \
    --num-workers 4 \
    2>&1 | tee full_training_fixed.log
```

**定期检查** (每 10k 步):
```bash
python plot_training_curves_fixed_v1.py
```

---

## 常见场景

### 场景1: 从零开始

```bash
# 步骤1: 验证
python test_training_fixes.py

# 步骤2: 短训练
python train_cuda_optimized.py \
    --config se_e2_a/input_torch.json \
    --data-dir collect/O64H128 \
    --grad-accumulation-steps 4 &

# 步骤3: 等待500步后检查
sleep 300  # 根据速度调整
python plot_training_curves_fixed_v1.py

# 步骤4: 如健康，继续训练；如平台，应用 fix_neighbor_list
```

### 场景2: 已有训练但 f_loss 平台

```bash
# 检查当前状态
python plot_training_curves_fixed_v1.py

# 如果指标B失败，应用修复
python fix_neighbor_list_strict_sel.py --apply

# 从头重新训练
rm cuda_training_opt.log checkpoints_cuda/* exports_cuda/*
python train_cuda_optimized.py ...
```

### 场景3: 对比不同配置

```bash
# 训练1: 默认配置
python train_cuda_optimized.py --data-dir collect/O64H128 \
    --grad-accumulation-steps 4
mv cuda_training_opt.log cuda_training_default.log

# 训练2: 应用 strict sel
python fix_neighbor_list_strict_sel.py --apply
python train_cuda_optimized.py --data-dir collect/O64H128 \
    --grad-accumulation-steps 4
mv cuda_training_opt.log cuda_training_strict_sel.log

# 对比
python plot_training_curves_fixed_v1.py --log cuda_training_default.log
python plot_training_curves_fixed_v1.py --log cuda_training_strict_sel.log

# 比较 pic/ 目录中的曲线
```

---

## 故障排查

### Q: 健康报告显示 diff > 1e-5

**原因**: Loss 组成不一致

**检查**:
```bash
# 查看训练日志中的 diff 列
grep "diff=" cuda_training_opt.log | awk '{print $NF}'

# 如果持续大于 1e-5，说明 compute_loss 实现有问题
```

**解决**: 重新运行 `test_training_fixes.py`，TEST 2 应该能捕获这个问题

### Q: f_loss 一开始就很大 (>10)

**原因**: 可能是数据问题或模型初始化问题

**检查**:
```bash
# 验证数据加载
python -c "
from dpmini import DeepMDDataset
ds = DeepMDDataset(['collect/O64H128'], type_map=['O', 'H'])
pos, types, box, e, f = ds[0]
print(f'Energy: {e.item():.3f} eV')
print(f'Force range: [{f.min().item():.3f}, {f.max().item():.3f}] eV/Å')
"

# 正常范围：Energy ~数百eV，Force ~0.1-10 eV/Å
```

### Q: Descriptor 统计异常

**症状**: 健康报告显示 f_loss 或 e_loss 极大

**解决**:
```bash
# 重新初始化模型权重
rm checkpoints_cuda/* exports_cuda/*

# 确认 descriptor clipping 生效
grep "clamp" dpmini/descriptor.py
# 应该有: descriptor = torch.clamp(descriptor, min=-1e4, max=1e4)
```

---

## 文件组织

训练完成后，建议的文件组织：

```
pj/
├── pic/                              # 训练曲线（自动生成）
│   ├── f_loss_20251229-120000.png
│   ├── e_loss_20251229-120000.png
│   ├── combined_losses_*.png
│   └── training_curves_*.csv
├── checkpoints_cuda/                 # 训练断点
│   ├── model_step10000.pt
│   └── model_step20000.pt
├── exports_cuda/                     # 导出模型
│   └── model_final.pth
├── cuda_training_opt.log            # 训练日志
├── quick_test_fixed.log             # 短测试日志
└── full_training_fixed.log          # 完整训练日志
```

---

## 总结：最小健康判定流程

```bash
# 1. 训练 500-2000 步
python train_cuda_optimized.py ... &

# 2. 生成健康报告
python plot_training_curves_fixed_v1.py

# 3. 查看报告中的 3 个指标
#    - [A] diff < 1e-5 ✅
#    - [B] f_loss reduction > 10% ✅
#    - [C] f_loss < 1e3 ✅

# 4. 查看 pic/f_loss_*.png 中的 rolling mean 曲线
#    应向下且明显

# 5. 如果 [B] 失败：
python fix_neighbor_list_strict_sel.py --apply
# 然后重新训练
```

**一句话判定**: 
- ✅ **健康**: `diff < 1e-5` 且 `f_loss rolling mean 明显下降`
- ⚠️ **需优化**: `diff < 1e-5` 但 `f_loss 平台` → 应用 strict sel 修复
- ❌ **有问题**: `diff > 1e-5` → 检查 compute_loss 实现

---

**修复完成！所有工具就绪！** 🚀
