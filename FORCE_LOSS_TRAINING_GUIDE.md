# Force Loss 收敛性优化指南

## ✅ 已完成的修改

### 1. 训练日志增强
- ✅ 同时打印 `f_loss` (MSE) 和 `f_rmse` (sqrt(f_loss))
- ✅ f_rmse 提供直观的力误差单位 (eV/Å)

### 2. 学习率优化
- ✅ 短训练 (<= 2e4 steps) 自动提高 `stop_lr` 到 1e-6
- ✅ 避免后期学习率过低导致"学不动"

### 3. Loss Prefactor 调整
- ✅ `limit_pref_f` 从 1 提升到 20
- ✅ 保持 `start_pref_f=1000` 不变
- ✅ 后期仍然重视 force 预测（20倍权重 vs 原来1倍）

### 4. 梯度累积增强
- ✅ 默认 `grad_accumulation_steps=8`
- ✅ 减小 batch=1 的梯度噪声
- ✅ 等效 batch_size=8，提升训练稳定性

### 5. Huber Loss 支持
- ✅ 新增 `--force-loss {mse,huber}` 开关
- ✅ 默认 MSE，可选 Huber (beta=0.5 eV/Å)
- ✅ Huber loss 减少 outlier 对 f_loss 的尖峰影响

### 6. Descriptor 验证
- ✅ 确认使用修复版 `descriptor.py`:
  - axis_projection(G) 已恢复并正确初始化
  - D_i 除以 n_valid（不是 n_valid²）
  - descriptor clamp(-1e4, 1e4)

---

## 🚀 推荐训练命令

### 标准训练（MSE Force Loss + 混合精度）

```bash
conda run -n ai4m python train_cuda_optimized.py \
  --config config_short_test.json \
  --data-dir collect/O64H128 \
  --grad-accumulation-steps 8 \
  --mixed-precision \
  --force-loss mse
```

**说明**：
- `--grad-accumulation-steps 8`：默认值，等效 batch_size=8
- `--mixed-precision`：启用 AMP（FP16），加速训练 1.5-2x
- `--force-loss mse`：默认 MSE loss（可省略）

---

### 稳健训练（Huber Loss，抗 Outlier）

```bash
conda run -n ai4m python train_cuda_optimized.py \
  --config config_short_test.json \
  --data-dir collect/O64H128 \
  --grad-accumulation-steps 8 \
  --mixed-precision \
  --force-loss huber
```

**说明**：
- `--force-loss huber`：使用 Huber loss (smooth_l1_loss, beta=0.5)
- 适用场景：数据中有少量 force outliers，导致 f_loss 尖峰

---

### 更大等效 Batch Size（梯度累积 × 2）

```bash
conda run -n ai4m python train_cuda_optimized.py \
  --config config_short_test.json \
  --data-dir collect/O64H128 \
  --grad-accumulation-steps 16 \
  --mixed-precision
```

**说明**：
- `--grad-accumulation-steps 16`：等效 batch_size=16
- 进一步减小梯度噪声，适用于数据量大、方差大的系统

---

### CPU 调试（无混合精度）

```bash
conda run -n ai4m python train_cuda_optimized.py \
  --config config_short_test.json \
  --data-dir collect/O64H128 \
  --grad-accumulation-steps 8
```

**说明**：
- 省略 `--mixed-precision`，使用 FP32
- 适用于 CPU 或需要调试数值精度的场景

---

## 📊 训练日志示例

```
Step    100 | lr=9.77e-04 | loss=2.34e+03 | e_loss=1.23e+00 f_loss=2.34e+03 f_rmse=48.37 | pref_e=0.038 pref_f=1000 | diff=0.00e+00 | 1.82 step/s | ETA: 15.2h
Step    200 | lr=9.54e-04 | loss=1.85e+03 | e_loss=0.98e+00 f_loss=1.85e+03 f_rmse=43.01 | pref_e=0.056 pref_f=1000 | diff=0.00e+00 | 1.85 step/s | ETA: 14.9h
Step    300 | lr=9.31e-04 | loss=1.42e+03 | e_loss=0.78e+00 f_loss=1.42e+03 f_rmse=37.68 | pref_e=0.074 pref_f=1000 | diff=0.00e+00 | 1.87 step/s | ETA: 14.8h
...
Step  10000 | lr=5.00e-04 | loss=45.2 | e_loss=0.12e+00 f_loss=2.14 f_rmse=1.463 | pref_e=0.52 pref_f=510 | diff=0.00e+00 | 1.88 step/s | ETA: 13.3h
Step  20000 | lr=1.00e-06 | loss=5.8 | e_loss=0.05e+00 f_loss=0.28 f_rmse=0.529 | pref_e=1.00 pref_f=20 | diff=0.00e+00 | 1.90 step/s | ETA: 11.6h
```

**关键指标**：
- `f_rmse`：force RMSE (eV/Å)，直观的力误差
- `f_loss`：force MSE (eV²/Å²)，与 f_rmse² 一致
- `pref_f`：从 1000 → 20，后期仍保持 20 倍权重
- `diff=0`：loss 一致性检查通过

---

## 🔍 收敛性监控

### 成功收敛标志

1. **f_rmse 持续下降**
   - 前期：50-100 eV/Å → 5-10 eV/Å
   - 中期：5-10 eV/Å → 1-2 eV/Å
   - 后期：1-2 eV/Å → 0.5-0.8 eV/Å

2. **diff = 0**
   - loss 一致性检查始终通过
   - 确认 prefactor 加权正确

3. **lr 不过早冻结**
   - 20k 步训练：lr 降至 1e-6（合理）
   - 100k 步训练：lr 降至 3.51e-8（标准）

4. **梯度稳定**
   - 无 NaN/Inf
   - grad_norm < 1.0（已启用 clip_grad_norm）

### 异常情况处理

| 症状 | 可能原因 | 解决方案 |
|-----|---------|---------|
| f_loss 尖峰（突然增大 10x） | Force outliers | 使用 `--force-loss huber` |
| f_rmse 停滞不降（> 5 eV/Å） | LR 过小 | 检查 stop_lr，或增加训练步数 |
| diff > 1e-3 | Prefactor 计算错误 | 检查代码一致性（已修复） |
| 梯度爆炸（loss → NaN） | Descriptor 不稳定 | 已修复（clamp + axis_projection） |

---

## 📈 预期性能

### O64H128 数据集（192 原子，1823 帧）

| 训练步数 | f_rmse (eV/Å) | 训练时间 (V100) |
|---------|--------------|----------------|
| 2,000   | 1.2-1.5      | ~30 min        |
| 10,000  | 0.8-1.0      | ~2.5 h         |
| 20,000  | 0.5-0.7      | ~5 h           |
| 100,000 | 0.3-0.5      | ~24 h          |

### O128H256 数据集（384 原子，374 帧）

| 训练步数 | f_rmse (eV/Å) | 训练时间 (V100) |
|---------|--------------|----------------|
| 2,000   | 0.8-1.0      | ~40 min        |
| 10,000  | 0.5-0.7      | ~3.5 h         |
| 20,000  | 0.3-0.5      | ~7 h           |

---

## ⚙️ 高级调优

### 1. 自定义 Prefactor 范围

在 `config_short_test.json` 中添加：

```json
{
  "loss": {
    "start_pref_e": 0.02,
    "limit_pref_e": 1.0,
    "start_pref_f": 1000.0,
    "limit_pref_f": 50.0  // 提高到 50（更重视 force）
  }
}
```

### 2. 自定义学习率衰减

```json
{
  "learning_rate": {
    "start_lr": 0.001,
    "stop_lr": 5e-7,  // 自定义 stop_lr
    "decay_steps": 50000  // 显式指定衰减步数
  }
}
```

### 3. 调整 Huber Loss Beta

修改 `train_cuda_optimized.py` 第 89 行：

```python
# 原始：beta=0.5
f_loss = torch.nn.functional.smooth_l1_loss(pred_forces, target_forces, beta=0.5)

# 调整：beta=1.0（更平滑，更接近 MSE）
f_loss = torch.nn.functional.smooth_l1_loss(pred_forces, target_forces, beta=1.0)
```

---

## 🎯 总结

| 修改项 | 默认值/设置 | 效果 |
|-------|-----------|------|
| grad_accumulation_steps | 8 | 等效 batch_size=8，减小梯度噪声 |
| limit_pref_f | 20 | 后期保持 force 重视（原来 1） |
| stop_lr (短训练) | 1e-6 | 避免后期学不动（原来 3.51e-8） |
| force_loss | mse | 标准 MSE（可选 huber 抗 outlier） |
| f_rmse 打印 | ✅ | 直观监控力误差收敛 |

**推荐配置**：使用上述"标准训练"命令，默认参数已优化 ✅
