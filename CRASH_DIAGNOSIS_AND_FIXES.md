# 训练进程崩溃诊断与修复方案

## 一、进程崩溃信息整理

### 进程与环境
| 项目 | 值 |
|------|-----|
| **PID** | 125735 |
| **启动时间** | 2025-12-29 19:02 |
| **崩溃时间** | 2025-12-29 19:18:23 |
| **运行时长** | ~16 分钟 |
| **完成步数** | 2000 步（相对于首轮）；日志中显示多轮重复数据 |
| **Checkpoint** | 0 个（save_freq=5000，未到达） |
| **Export** | 0 个 |

### 启动命令
```bash
source activate /home/ubuntu/miniforge3/envs/cuda_env
cd /home/ubuntu/pj
python -u train_cuda_optimized.py \
  --config config_formal_100k.json \
  --checkpoint-dir checkpoints_formal_long \
  --export-dir exports_formal_long \
  --force-loss mse \
  --grad-accumulation-steps 8
```

### 运行环境
- **Python**: 3.x（conda env cuda_env）
- **PyTorch**: 2.5.1+cu121（或 2.4.1）
- **CUDA**: 12.1
- **GPU**: NVIDIA L20-8Q（单卡）
- **数据**: collect/O64H128（1823 帧，192 原子，O:64, H:128）

---

## 二、日志异常分析

### 关键现象
1. **日志中包含多个 epoch/轮次**：同一个日志文件中有多套 step 1~500 的记录
   - 可能指示：进程重启、多进程竞争写入、或样本重采样

2. **Energy Loss 初始值异常高**
   ```
   Step   100 | e_loss=91507.7  | f_loss=1.58  | loss=7819.8
   Step   200 | e_loss=5297.79  | f_loss=1.06  | loss=1582.5
   Step   300 | e_loss=1302.31  | f_loss=2.59  | loss=2432.67
   ```
   - **问题**: e_loss 从 91k 跌到 1.3k，波动极大（不符合正常训练曲线）
   - **推断**: 初始化梯度或能量计算存在数值溢出；或损失权重不匹配

3. **Prefactor 权重与总 Loss 不匹配**
   ```
   Step 100: total_loss=7819.8  vs  0.069*91507.7 + 951*1.58 ≈ 6313 + 1503 ≈ 7816 ✓
   ```
   - 计算一致，但数字本身太大（通常 loss 应为 < 100）

4. **进程无 stderr 即死亡**：
   - 无 Python traceback 或 CUDA error message
   - 推测：CUDA 检测到 nan/inf 并强制终止，或内存不足

---

## 三、根本原因分析

### 假设 1：初始学习率过高
- **当前**: `start_lr = 0.001`
- **batch_size**: 1（无梯度累积时的有效批大小）
- **第一步梯度**: 初始化网络的梯度范数可能很大（特别是权重接近随机值）
- **冲击**: `lr * large_grad` 导致参数更新过度，loss 爆炸

### 假设 2：能量损失函数未归一化
- **当前配置**: 
  ```json
  "loss": {
    "start_pref_e": 0.02,
    "limit_pref_e": 1.0,
    "start_pref_f": 1000.0,
    "limit_pref_f": 20.0
  }
  ```
- **问题**: 初始时 `pref_e = 0.02`，但 `e_loss` 的绝对值高达 91k（未按原子数或能量范围归一化）
- **结果**: `total_loss = 0.02 * 91k + 1000 * 1.58 ≈ 7900`（与日志一致，但太大）

### 假设 3：梯度累积与优化器状态不同步
- **梯度累积步数**: 8
- **每步**: loss 缩放 1/8，backward 后不立即 step
- **风险**: 梯度缩放混乱、或中间梯度范数判断不准确导致提前终止

### 假设 4：CUDA 数值溢出
- **无日志终止** 通常指 CUDA 运行时错误（如 nan/inf）
- **触发**: 梯度爆炸 → nan 传播 → CUDA 内核失败

---

## 四、拟采纳修复方案（**不执行训练**）

### 修复 A：降低初始学习率

**文件**: `config_formal_100k.json`

**修改**:
```json
{
  "learning_rate": {
    "type": "exp",
    "start_lr": 0.0001,      // 从 0.001 → 0.0001（降低 10 倍）
    "stop_lr": 1e-6,
    "decay_steps": 20000
  }
}
```

**理由**: 降低初始梯度冲击，给优化器更温和的起点

---

### 修复 B：强化数值稳定性检测

**文件**: `train_cuda_optimized.py`

**位置**: `compute_loss()` 函数后 + 每个 backward 前

**修改内容**:

在 `compute_loss()` 后添加检查：
```python
def compute_loss(pred_energy, target_energy, pred_forces, target_forces, 
                pref_e, pref_f, force_loss_type='mse'):
    """..."""
    # 现有代码
    e_loss = torch.nn.functional.mse_loss(pred_energy, target_energy)
    if force_loss_type == 'huber':
        f_loss = torch.nn.functional.smooth_l1_loss(pred_forces, target_forces, beta=0.5)
    else:
        f_loss = torch.nn.functional.mse_loss(pred_forces, target_forces)
    
    total_loss = pref_e * e_loss + pref_f * f_loss
    
    # === 新增：数值检查 ===
    if not torch.isfinite(e_loss):
        print(f"[WARNING] e_loss is non-finite: {e_loss}")
        e_loss = torch.tensor(1e10, dtype=e_loss.dtype, device=e_loss.device)
    if not torch.isfinite(f_loss):
        print(f"[WARNING] f_loss is non-finite: {f_loss}")
        f_loss = torch.tensor(1e10, dtype=f_loss.dtype, device=f_loss.device)
    if not torch.isfinite(total_loss):
        print(f"[WARNING] total_loss is non-finite: {total_loss}")
        total_loss = torch.tensor(1e10, dtype=total_loss.dtype, device=total_loss.device)
    
    return total_loss, e_loss, f_loss
```

在 backward 前添加梯度裁剪：
```python
# Forward pass ...
loss, e_loss, f_loss = compute_loss(...)

# === 新增：强制梯度检查 ===
if accum_step == 0:
    optimizer.zero_grad()

# Backward
(loss / args.grad_accumulation_steps).backward()

# === 新增：梯度范数检查 ===
if accum_step == 0:  # 在第一步时检查
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.norm().item()
            if not (param_norm == param_norm):  # NaN check
                print(f"[WARNING] Gradient NaN detected in parameter {p.shape}")
            total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5
    if total_norm > 1000:
        print(f"[WARNING] Very high gradient norm: {total_norm:.2e}")
```

---

### 修复 C：简化梯度累积参数

**文件**: `config_formal_100k.json` 或启动命令

**修改**:
```bash
# 启动时改为：
--grad-accumulation-steps 4    # 从 8 → 4
```

**理由**: 减少中间状态数量，降低数值不稳定的来源

---

### 修复 D：调整 Loss 权重初值

**文件**: `config_formal_100k.json`

**修改**:
```json
{
  "loss": {
    "start_pref_e": 0.1,        // 从 0.02 → 0.1（增强能量权重）
    "limit_pref_e": 1.0,
    "start_pref_f": 100.0,      // 从 1000.0 → 100.0（降低初始力权重）
    "limit_pref_f": 20.0
  }
}
```

**理由**: 能量权重过小导致 e_loss 绝对值畸大；力权重过大在早期可能引起数值问题

---

## 五、快速检查清单（代码修改前）

- [ ] 确认 `config_formal_100k.json` 的 `start_lr` 当前值
- [ ] 检查 descriptor 初始化是否使用了特殊种子（seed=1）
- [ ] 验证 FittingNet 的初始权重范围（通常应为 [-0.1, 0.1]）
- [ ] 测试单步前向传播：`loss.item()` 是否为有限数

---

## 六、建议下一步行动

1. **应用修复 A + B + C**：
   - 修改 `config_formal_100k.json`：`start_lr=0.0001`，`start_pref_e=0.1`，`start_pref_f=100`
   - 修改 `train_cuda_optimized.py`：加入数值检查与梯度监控
   - 启动时改为 `--grad-accumulation-steps 4`

2. **运行一次短训练验证**（如 500 步）：
   ```bash
   python -u train_cuda_optimized.py \
     --config config_formal_100k.json \
     --numb-steps 500 \  # 覆盖 config 中的 numb_steps
     --checkpoint-dir checkpoints_test \
     --export-dir exports_test \
     --grad-accumulation-steps 4 \
     2>&1 | tee logs/test_stability_$(date +%Y%m%d-%H%M%S).log
   ```

3. **观察**：
   - Step 1-10 的 loss 是否呈下降趋势
   - Step 50 后是否稳定（无爆炸或 nan）
   - loss 绝对值是否合理（预期 < 100）

4. **若验证成功**，重新启动完整 100k 步训练

---

## 七、附录：相关配置文件当前值

### `config_formal_100k.json` (current)
```json
{
  "learning_rate": {
    "type": "exp",
    "start_lr": 0.001,
    "stop_lr": 1e-6,
    "decay_steps": 20000
  },
  "loss": {
    "start_pref_e": 0.02,
    "limit_pref_e": 1.0,
    "start_pref_f": 1000.0,
    "limit_pref_f": 20.0
  },
  "training": {
    "numb_steps": 100000,
    "batch_size": 1,
    "disp_freq": 500,
    "save_freq": 5000
  }
}
```

### 历史对标（短训练，已成功）
```json
{
  "learning_rate": {
    "start_lr": 0.001,
    "stop_lr": 1e-6,
    "decay_steps": 20
  },
  "loss": {
    "start_pref_e": 0.02,
    "limit_pref_e": 1.0,
    "start_pref_f": 1000.0,
    "limit_pref_f": 20.0
  },
  "training": {
    "numb_steps": 2000,
    "save_freq": 10000
  }
}
```

---

## 结论

进程崩溃最可能由**梯度爆炸**（高初始学习率 + 初期大梯度）或**数值溢出**（无 NaN 检测）导致。建议优先降低 `start_lr` 并加入防守性数值检查，然后在短训练中验证。
