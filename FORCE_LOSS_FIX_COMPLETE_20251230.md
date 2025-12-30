# Force Loss 修复完整报告 (2025-12-30)

## 问题诊断

**症状**: f_loss/f_rmse 长期平台、曲线像噪声墙、边际收益极小

**根本原因**:
1. **Loss 缩放错误**: force loss 被额外除以 natoms，导致权重过小
2. **梯度累积未生效**: 旧代码在每个 micro-step 都调用 optimizer.step()
3. **缺少趋势指标**: 只看瞬时 f_rmse 无法判断收敛趋势

---

## 修复方案

### 1) compute_loss 修复 (train_cuda_optimized.py:73-101)

**关键改动**:
```python
def compute_loss(pred_energy, target_energy, pred_forces, target_forces, 
                pref_e, pref_f, natoms, force_loss_type='mse'):
    # Energy: per-atom MSE (scale matching)
    e_err_per_atom = (pred_energy - target_energy) / natoms
    e_loss = e_err_per_atom ** 2
    
    # Force: MSE over all components (no extra /natoms scaling)
    f_loss = F.mse_loss(pred_forces, target_forces)
    
    # Canonical loss combination
    total_loss = pref_e * e_loss + pref_f * f_loss
    
    # RMSE for intuitive metrics
    e_rmse = torch.sqrt(e_loss)
    f_rmse = torch.sqrt(f_loss)
    
    return total_loss, e_loss, f_loss, e_rmse, f_rmse
```

**修复要点**:
- Energy 改为 **per-atom MSE** (除以 natoms)
- Force 保持 **mean-reduced MSE** (不再额外除 natoms)
- 返回 **e_rmse, f_rmse** 便于评估

---

### 2) 梯度累积修复 (train_cuda_optimized.py:280-340)

**关键改动**:
```python
# Zero grad only at start of accumulation cycle
if accum_step == 0:
    optimizer.zero_grad()

# Backward with scaled loss
loss_scaled = loss / args.grad_accumulation_steps
loss_scaled.backward()

# Step optimizer after accumulation
accum_step += 1
if accum_step >= args.grad_accumulation_steps:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()  # Zero for next cycle
    accum_step = 0
    update_step += 1  # Track actual optimizer steps
```

**修复要点**:
- 每周期开始时 **zero_grad()**
- backward 前 **loss 除以 K** (K=grad_accumulation_steps)
- 周期结束时 **step + zero_grad**
- 新增 **update_step** 计数实际优化步数

---

### 3) EMA 趋势跟踪

```python
# EMA tracking (exponential moving average, alpha=0.01)
f_rmse_ema = None
ema_alpha = 0.01

# Update after each step
f_rmse_val = f_rmse.item()
if f_rmse_ema is None:
    f_rmse_ema = f_rmse_val
else:
    f_rmse_ema = (1 - ema_alpha) * f_rmse_ema + ema_alpha * f_rmse_val
```

**日志输出格式**:
```
Step   2000 (update=  500) | lr=1.000e-06 | loss=207.769 | 
e_rmse=4.4212 f_rmse=1.0168 f_ema=1.1392 | 
pref_e=0.05 pref_f=200 | diff=0.000e+00
```

---

## 验证结果

### 快速测试 (2000 steps)

| 指标 | 值 | 状态 |
|------|-----|------|
| 初始 f_rmse_ema | 1.2764 | - |
| 最佳 f_rmse_ema (step 1000) | 1.0772 | ✅ |
| 最终 f_rmse_ema | 1.1392 | ✅ |
| **下降幅度 (初始→最终)** | **10.7%** | ✅ PASS |
| **下降幅度 (初始→最佳)** | **15.6%** | ✅ PASS |

**验证通过**:
- ✅ 梯度累积正常工作
- ✅ loss 一致性通过 (diff=0.000e+00)
- ✅ f_rmse 有明显下降趋势
- ✅ 无数值异常 (NaN/Inf/梯度爆炸)

---

## 正式训练配置

**文件**: `config_formal_100k_fixed.json`

```json
{
  "learning_rate": {
    "start_lr": 0.0001,
    "stop_lr": 1e-07,
    "decay_steps": 100000
  },
  "loss": {
    "start_pref_e": 0.05,
    "limit_pref_e": 1.0,
    "start_pref_f": 200.0,
    "limit_pref_f": 50.0
  },
  "training": {
    "numb_steps": 100000,
    "disp_freq": 500,
    "save_freq": 5000
  }
}
```

**关键改进**:
- decay_steps = 100k (lr 持续下降，不在 20k 冻结)
- pref_f = 200→50 (保持力权重，不过低)
- pref_e = 0.05→1.0 (能量权重逐步增加)

---

## 启动与监控命令

### 启动训练
```bash
cd /home/ubuntu/pj
conda activate ai4m

nohup python -u train_cuda_optimized.py \
  --config config_formal_100k_fixed.json \
  --checkpoint-dir checkpoints_fixed \
  --export-dir exports_fixed \
  --force-loss mse \
  --grad-accumulation-steps 4 \
  > logs/train_fixed_$(date +%Y%m%d-%H%M%S).log 2>&1 &

echo $! > logs/train_fixed.pid
```

### 监控命令
```bash
# 实时日志
tail -f logs/train_fixed_*.log

# GPU 监控
nvidia-smi -l 2

# 进程状态
ps -p $(cat logs/train_fixed.pid) -o pid,pcpu,pmem,cmd

# f_rmse_ema 趋势
grep "f_ema=" logs/train_fixed_*.log | tail -20
```

### 终止训练
```bash
# 优雅停止
kill -INT $(cat logs/train_fixed.pid)

# 若 60 秒内未退出
kill -TERM $(cat logs/train_fixed.pid)

# 强制停止
kill -KILL $(cat logs/train_fixed.pid)
```

---

## 当前状态

**训练 PID**: 138887  
**日志**: `logs/train_fixed_20251230-013944.log`  
**检查点**: `checkpoints_fixed/`  
**预计完成**: ~13 小时后

**初始输出** (step 1-5):
```
Step      1 (update=    0) | lr=9.999e-05 | loss=119.316 | 
  e_rmse=4.9337 f_rmse=0.7684 f_ema=0.7684

Step      5 (update=    1) | lr=9.997e-05 | loss=819.123 | 
  e_rmse=4.8259 f_rmse=2.0224 f_ema=0.7970
```

---

## 备份信息

**旧训练备份**: `backups/20251230-013618/`
- 旧日志: `train_formal_20251229-210839.log`
- 旧检查点: `checkpoints_fixed_neighbor/`

**新训练路径**:
- 日志: `logs/train_fixed_*.log`
- 检查点: `checkpoints_fixed/model_stepXXXXX.pt`
- 导出: `exports_fixed/model_final.pth`

---

## 作者 & 验证

**修复日期**: 2025-12-30  
**验证状态**: ✅ 快速测试通过 (f_rmse_ema 下降 10.7%)  
**生产状态**: 🚀 100k 训练运行中 (PID: 138887)
