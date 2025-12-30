# 完整日志捕获启动指南

## 概述

[launch_training_with_full_logging.sh](launch_training_with_full_logging.sh) 是一个强化的训练启动脚本，确保：

1. ✓ **完整捕获 stdout + stderr** → 单个独立日志文件
2. ✓ **实时显示到终端**（via `tee`）
3. ✓ **记录进程死亡原因**（退出码、资源使用等）
4. ✓ **无缓冲 Python 执行**（`-u` 标志）
5. ✓ **元数据与环境信息**（启动参数、GPU 状态、配置文件内容）

---

## 快速启动

### 方案 1：使用稳定配置（推荐，已降低学习率）

```bash
cd /home/ubuntu/pj
./launch_training_with_full_logging.sh \
  --config config_formal_100k_stable.json \
  --grad-accum 4
```

**输出日志**：`logs/train_formal_YYYYMMDD-HHMMSS.log`

### 方案 2：使用原始配置（对标之前的设置）

```bash
cd /home/ubuntu/pj
./launch_training_with_full_logging.sh \
  --config config_formal_100k.json \
  --grad-accum 8
```

### 方案 3：短验证运行（快速测试稳定性，~5 分钟）

```bash
cd /home/ubuntu/pj
./launch_training_with_full_logging.sh \
  --config config_formal_100k_stable.json \
  --steps 1000 \
  --grad-accum 4 \
  --checkpoint-dir checkpoints_test_stable \
  --export-dir exports_test_stable
```

---

## 日志文件说明

启动脚本会在 `logs/` 目录下生成：

### 主日志文件
```
logs/train_formal_20251229-193500.log
```

**包含内容**：
- ✓ 启动参数与环境信息（顶部）
- ✓ 配置文件完整内容
- ✓ Python stdout（实时 loss 输出）
- ✓ Python stderr（任何错误或异常）
- ✓ 训练终止报告（退出码、资源使用、最后 10 行 loss）
- ✓ 下一步操作建议（底部）

### 元数据文件（可选）
```
logs/train_formal_20251229-193500_metadata.txt
```

启动时的元数据快照（便于事后查询）。

---

## 查看日志

### 实时跟随输出
```bash
tail -f logs/train_formal_20251229-193500.log
```

### 查看最后 50 行（含死亡信息）
```bash
tail -n 50 logs/train_formal_20251229-193500.log
```

### 搜索错误关键字
```bash
grep -i "error\|warning\|nan\|inf" logs/train_formal_20251229-193500.log
```

### 统计训练步数与 loss 趋势
```bash
grep "^Step" logs/train_formal_20251229-193500.log | tail -20
```

---

## 脚本特性详解

### 1. 完整环境记录
```
[LAUNCH TIME] 2025-12-29 19:35:00 CST
[WORKING DIR] /home/ubuntu/pj
[LOG FILE] logs/train_formal_20251229-193500.log

--- STARTUP PARAMETERS ---
CONFIG: config_formal_100k_stable.json
CHECKPOINT_DIR: checkpoints_formal_long
FORCE_LOSS: mse
GRAD_ACCUM_STEPS: 4

--- ENVIRONMENT ---
Python: /home/ubuntu/miniforge3/envs/cuda_env/bin/python
Conda Env: cuda_env

--- GPU INFO ---
0, NVIDIA L20-8Q, 8192 MiB
```

### 2. 错误捕获与死亡报告
```
===============================================================================
TRAINING COMPLETION/FAILURE REPORT
===============================================================================

[END TIME] 2025-12-29 19:45:30 CST
[EXIT CODE] 1
[STATUS] ✗ Training failed with exit code 1

--- FINAL OUTPUT FILE SIZES ---
Checkpoints: 2 files, ~512M
Exports: 1 files, ~100M

--- INTERNAL LOSS LOG ---
cuda_training_opt.log: 45 lines, ~12K
Last 10 lines:
  Step   5000 | lr=5.988e-04 | loss=604.351 | e_loss=8.03493 | f_loss=0.634907
  Step   5500 | lr=5.688e-04 | loss=1386.77 | e_loss=32.4303 | f_loss=1.46324
  ...
```

### 3. 实时 tee 输出
```
[HH:MM:SS] Activating conda environment and launching training...
[HH:MM:SS] Environment activated successfully
[HH:MM:SS] Executing: python -u train_cuda_optimized.py --config ...

Starting high-performance DeepMD training with CUDA optimizations...
CUDA OPTIMIZATION CONFIGURATION
...
```

---

## 配置选项

### 稳定版配置：`config_formal_100k_stable.json`
```json
{
  "learning_rate": {
    "start_lr": 0.0001,    ← 降低 10 倍（从 0.001）
    "stop_lr": 1e-6,
    "decay_steps": 20000
  },
  "loss": {
    "start_pref_e": 0.1,   ← 升高（从 0.02）
    "start_pref_f": 100.0, ← 降低（从 1000）
    "limit_pref_f": 20.0
  }
}
```

**为什么稳定**？
- 初始学习率 0.0001 避免梯度爆炸
- 提高能量权重起点 → loss 更均衡
- 降低力权重起点 → 早期数值更稳定

---

## 常见用法示例

### 例 1：启动完整 100k 步训练（稳定配置）
```bash
./launch_training_with_full_logging.sh \
  --config config_formal_100k_stable.json \
  --grad-accum 4
```

**预期**：
- 日志文件：`logs/train_formal_YYYYMMDD-HHMMSS.log`
- 首个 checkpoint（step 5000）出现约 20-30 分钟后
- 整个训练耗时 ~13 小时

### 例 2：验证稳定性（500 步快速测试）
```bash
./launch_training_with_full_logging.sh \
  --config config_formal_100k_stable.json \
  --steps 500 \
  --grad-accum 4 \
  --checkpoint-dir checkpoints_quick_test \
  --export-dir exports_quick_test
```

**预期**：
- 耗时 ~2-3 分钟
- 观察前 10 步的 loss 是否合理（应为 < 100）
- 观察 step 500 时 loss 是否明显下降

### 例 3：原始配置（对标之前失败的运行）
```bash
./launch_training_with_full_logging.sh \
  --config config_formal_100k.json \
  --grad-accum 8
```

**警告**：此配置导致之前的进程崩溃，仅用于对比测试。

---

## 脚本流程图

```
┌─────────────────────────────────────┐
│  启动 shell 脚本                      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 阶段 1: 记录启动参数、环境、配置      │
│  → 元数据写入 metadata.txt           │
│  → 元数据追加到主日志                 │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 阶段 2: 激活 conda，启动 Python      │
│  $ python -u train_cuda_optimized.py │
│  → stdout/stderr via tee            │
│  → 实时写入主日志文件                 │
│  → 同时显示在终端                     │
└──────────────┬──────────────────────┘
               │
        ┌──────┴──────┐
        │             │
   成功完成      异常/崩溃
        │             │
        ▼             ▼
┌──────────────┐  ┌──────────────────┐
│ EXIT=0      │  │ EXIT≠0 (捕获)     │
└──────────────┘  └──────────────────┘
        │             │
        └──────┬──────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 阶段 3: 记录完成/失败报告            │
│  → 退出码解释                        │
│  → Checkpoint/Export 大小            │
│  → cuda_training_opt.log 最后 10 行  │
│  → GPU 资源使用                      │
│  → 下一步命令建议                     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 返回训练的退出码（脚本反映训练结果）  │
└─────────────────────────────────────┘
```

---

## 故障排除

### Q: 日志文件为空或不更新？
**A**: 检查脚本是否运行：
```bash
ps aux | grep launch_training
```

如果脚本卡住，可能是 conda 激活失败。手动检查：
```bash
source /home/ubuntu/miniforge3/bin/activate
conda activate cuda_env
python --version
```

### Q: 为什么日志中没有 Python 输出？
**A**: 可能 stderr 未正确混合。尝试：
```bash
python -u train_cuda_optimized.py ... 2>&1 | tee log.txt
```

### Q: 进程在运行但日志未写入？
**A**: 日志可能被缓冲。使用 `tail -f` 实时查看（强制刷新），或增加 `python -u` 的无缓冲标志（脚本已包含）。

---

## 下一步

### 推荐流程：
1. **短验证** (500-1000 步)：
   ```bash
   ./launch_training_with_full_logging.sh \
     --config config_formal_100k_stable.json \
     --steps 500 --grad-accum 4
   ```
   
2. **查看日志**：
   ```bash
   cat logs/train_formal_*.log
   ```
   
3. **检查稳定性**：
   - Step 1-10 loss 是否合理？
   - Step 500 loss 是否明显下降？
   - 有无 NaN/Inf 警告？
   
4. **若通过验证，启动完整训练**：
   ```bash
   ./launch_training_with_full_logging.sh \
     --config config_formal_100k_stable.json \
     --grad-accum 4
   ```

---

## 附录：日志文件位置速查

| 文件 | 位置 | 说明 |
|------|------|------|
| **主训练日志** | `logs/train_formal_YYYYMMDD-HHMMSS.log` | 完整输出（推荐查看） |
| **元数据快照** | `logs/train_formal_YYYYMMDD-HHMMSS_metadata.txt` | 启动时的环境记录 |
| **内部 loss 日志** | `cuda_training_opt.log` | train_cuda_optimized.py 生成（备用） |
| **Checkpoint** | `checkpoints_formal_long/*.pt` | 定期保存的模型权重 |
| **导出模型** | `exports_formal_long/*.pth` | 最终模型 |
