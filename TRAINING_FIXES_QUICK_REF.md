# 训练质量修复 - 快速参考

**状态**: ✅ 所有修复已完成并验证  
**日期**: 2025-12-29

---

## 🎯 核心修复（4个关键bug）

### 1. Loss 定义错误 ❌→✅
```python
# ❌ 修复前：f_loss 被二次归一化
total_loss = pref_e * e_loss + pref_f * f_loss / natoms

# ✅ 修复后：正确的 loss 组合
total_loss = pref_e * e_loss + pref_f * f_loss
```
**影响**: f_loss 梯度信号增强 ~200x，力场可被有效优化

### 2. 学习率过早衰减 ❌→✅
```python
# ❌ 修复前：hard-coded decay_steps=5000
decay_steps = lr_config.get('decay_steps', 5000)

# ✅ 修复后：对齐训练长度
decay_steps = lr_config.get('decay_steps', numb_steps)
```
**影响**: lr 在整个训练期内保持合理值

### 3. 梯度累积未实现 ❌→✅
```python
# ✅ 修复后：真正的梯度累积
if accum_step == 0:
    optimizer.zero_grad()
(loss / args.grad_accumulation_steps).backward()
accum_step += 1
if accum_step >= args.grad_accumulation_steps:
    optimizer.step()
    accum_step = 0
```
**影响**: 支持等效大 batch，训练更稳定

### 4. Descriptor 数值爆炸 ❌→✅
```python
# ✅ 修复后：小权重初始化 + clipping
nn.init.normal_(self.axis_projection.weight, mean=0.0, std=0.01)
descriptor = torch.clamp(descriptor, min=-1e4, max=1e4)
```
**影响**: 防止早期梯度爆炸

---

## 📝 快速验证

### 步骤 1: 运行自检脚本
```bash
python test_training_fixes.py
```

**预期输出**:
```
✅ TEST 1 PASSED: Shape validation & forward pass
✅ TEST 2 PASSED: Loss consistency verified (diff < 1e-5)
✅ TEST 3 PASSED: Training shows improvement
✅ ALL TESTS PASSED
```

### 步骤 2: 短训练测试（500-2000步）
```bash
python train_cuda_optimized.py \
    --config se_e2_a/input_torch.json \
    --data-dir collect/O64H128 \
    --grad-accumulation-steps 4 \
    2>&1 | tee quick_test_fixed.log
```

### 步骤 3: 绘制训练曲线并验证健康指标 ⭐
```bash
python plot_training_curves_fixed_v1.py --log cuda_training_opt.log
```

**健康判定：3个硬指标（必须全部满足）**:

#### 指标 A: Loss 组成一致性
- **标准**: `max(diff) < 1e-5`
- **含义**: total_loss 的打印口径与反传一致
- **检查**: 查看 `pic/diff_*.png` 和健康报告

#### 指标 B: Force Loss 趋势性下降 ⭐⭐⭐
- **标准**: 前500-2000步的 f_loss rolling mean 应明显下行
- **含义**: 模型能有效学习力场
- **检查**: 查看 `pic/f_loss_*.png`，rolling mean 应向下
- **参考**: 健康训练中 f_loss 从 ~1.0 降到 ~0.4

#### 指标 C: Descriptor 数值稳定
- **标准**: f_loss < 1e3, e_loss < 1e9, 无 NaN/Inf
- **含义**: 描述符不爆炸，梯度链路正常
- **检查**: 健康报告自动验证

---

## 📊 训练 Log 示例

**修复后的正常输出**:
```
Step 100 | lr=9.950e-04 | loss=1.23e+03 | e_loss=1.45e+02 | f_loss=1.08e+00 | pref_e=0.022 pref_f=982 | diff=0.000e+00
Step 200 | lr=9.900e-04 | loss=8.76e+02 | e_loss=9.87e+01 | f_loss=7.92e-01 | pref_e=0.024 pref_f=964 | diff=0.000e+00
Step 500 | lr=9.750e-04 | loss=4.32e+02 | e_loss=3.21e+01 | f_loss=3.98e-01 | pref_e=0.030 pref_f=900 | diff=0.000e+00
```

**关键指标**:
- `diff=0.000e+00` ✅ loss 一致性正确
- `f_loss` 单调下降 ✅ 力场在学习
- `lr` 保持在合理范围 ✅ 不过早衰减

---

## 🔧 常用命令

### 完整训练
```bash
python train_cuda_optimized.py \
    --config se_e2_a/input_torch.json \
    --data-dir collect/O64H128 \
    --grad-accumulation-steps 8 \
    --num-workers 4 \
    2>&1 | tee full_training_fixed.log
```

### 监控训练
```bash
# 实时查看 loss
tail -f cuda_training_opt.log | grep "Step"

# 检查 diff 一致性
grep "diff=" cuda_training_opt.log | tail -20
```

### 提取 loss 曲线
```bash
grep "Step" cuda_training_opt.log | \
    awk '{print $2, $8, $10}' | \
    sed 's/[|,]//g' > loss_curve.dat
```

---

## ⚠️ 故障排查

### 问题: `diff` 不为 0
```bash
# 检查 compute_loss 是否被正确调用
grep -n "compute_loss" train_cuda_optimized.py
# 应返回 tensor，不应有 .item()
```

### 问题: f_loss 仍然平台

**如果指标A/C通过，但指标B失败（f_loss平台）**:

优先级1：修复 neighbor list 的严格 sel 语义
```bash
# 查看修复说明
python fix_neighbor_list_strict_sel.py

# 应用修复（会备份原文件）
python fix_neighbor_list_strict_sel.py --apply

# 重新验证
python test_training_fixes.py
```

优先级2：检查其他因素
```bash
# 1. 检查 f_loss 数值是否合理（应比旧版大~200倍）
# 2. 检查 lr 是否过早衰减
grep "lr=" cuda_training_opt.log | tail -50
# 3. 检查数据质与工具

### 核心文档
- **完整文档**: [TRAINING_FIXES_DOCUMENTATION.md](TRAINING_FIXES_DOCUMENTATION.md)
- **快速参考**: 本文档

### 验证工具
- **自检脚本**: `python test_training_fixes.py`
- **曲线绘制**: `python plot_training_curves_fixed_v1.py`
- **健康判定**: 查看绘图脚本输出的 HEALTH CHECK REPORT

### 可选修复
- **严格 sel 语义**: `python fix_neighbor_list_strict_sel.py --apply`

### 训练脚本
- **标准训练**: `train_cuda_optimized.py`
- **输出日志**: `cuda_training_opt.log`（自动生成）
- 当前实现：全局排序 `(type*1000 + dist)`，取前 sum(sel) 个
- 严格实现：每种类型分别取 top-k，各取 sel[t] 个
- **影响**：当某些构型下类型分布不均时，严格实现更符合 DeepMD 预期

### 问题: Descriptor 爆炸
```bash
# 重新训练，应已修复
# 如仍出现，检查 clipping 是否生效
python -c "import torch; from dpmini import DeepMDModel; \
    m=DeepMDModel(['O','H'],6.0,0.5,[46,92],[25,50,100],16,[240,240,240]); \
    print('axis_projection weight std:', m.descriptor.axis_projection.weight.std().item())"
# 应输出 std ≈ 0.01
```（基于3个硬指标）

### 训练前验证
- [ ] `test_training_fixes.py` 全部通过
- [ ] 已备份旧版本代码（如需回退）

### 训练中验证（前 500-2000 步）⭐
运行 `python plot_training_curves_fixed_v1.py` 查看健康报告：

- [ ] **指标A**: `max(diff) < 1e-5` ✅ (Loss 一致性)
- [ ] **指标B**: `f_loss rolling mean` 明显下降 ✅ (可训练性)
  - 参考：从 ~1.0 降到 ~0.4 即为健康
  - 如平台：优先应用 `fix_neighbor_list_strict_sel.py`
- [ ] **指标C**: `f_loss < 1e3`, `e_loss < 1e9` ✅ (数值稳定)

### 训练后验证（最终模型）
- [ ] Force MAE < 1.0 eV/Å (目标 < 0.5)
- [ ] Energy MAE < 0.05 eV/atom
- [ ] 测试集上泛化良好

### 一句话判定标准
**如果 diff 长期 <1e-5 且 f_loss rolling mean 在前几千步明显下行**：✅ f_loss 是健康的aining_fixes.py` 全部通过
- [ ] 已备份旧版本代码（如需回退）

训练中（前 1000 步）：
- [ ] Log 中 `diff < 1e-5`
- [ ] `f_loss` 有下降趋势
- [ ] `lr` 在合理范围（不锁死）
- [ ] 无 NaN/Inf/异常值

训练后（最终模型）：
- [ ] Force MAE < 1.0 eV/Å (目标 < 0.5)
- [ ] Energy MAE < 0.05 eV/atom
- [ ] 测试集上泛化良好

---

**修复完成！Ready for production training!** 🚀
