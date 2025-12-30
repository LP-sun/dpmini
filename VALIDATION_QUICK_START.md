# ⚡ 验证集评估系统 - 快速参考

## 📊 评估结果概览

| 指标 | 原始 (90k) | V3 (80k) | 优势 |
|------|-----------|---------|------|
| **F_RMSE** | 0.816 eV/Å | **0.715 eV/Å** | **↓12.4%** ✓ |
| **F_RMSE_tail** | 1.702 eV/Å | **1.288 eV/Å** | **↓24.3%** ✓ |
| **收敛状态** | ✓ STOP | ✓ STOP | V3 优先 |

## 🎯 立即行动

### 一键执行完整评估
```bash
bash run_validation_pipeline.sh
```

### 或分步执行

**Step 1: 评估原始训练**
```bash
python batch_evaluate_checkpoints.py \
  --ckpt-dir checkpoints_fixed \
  --output logs/ckpt_eval_fixed.csv \
  --n-latest 5
```

**Step 2: 评估 V3 训练**
```bash
python batch_evaluate_checkpoints.py \
  --ckpt-dir checkpoints_optimized_v3 \
  --output logs/ckpt_eval_v3.csv \
  --config config_formal_100k_optimized.json \
  --n-latest 5
```

**Step 3: 检查收敛**
```bash
# 原始训练
python check_convergence.py --csv logs/ckpt_eval_fixed.csv

# V3 训练  
python check_convergence.py --csv logs/ckpt_eval_v3.csv
```

## 🚀 推荐操作

### 若要停止 V3 训练（已收敛）
```bash
kill -INT 147981
# 注意: 需要在训练脚本中添加信号处理器以保存 checkpoint
```

### 若要评估特定 checkpoint
```bash
python evaluate_checkpoint_fast.py \
  --checkpoint checkpoints_optimized_v3/model_step80000.pt \
  --system /home/ubuntu/pj/collect/O64H128 \
  --val-indices logs/val_indices.npy \
  --config config_formal_100k_optimized.json
```

## 📋 CSV 输出格式

所有 CSV 文件包含:
```
step,e_rmse_per_atom,f_rmse,f_rmse_tail,n_val_frames
85000,4.982335,0.816451,1.701898,300
```

## 📖 详细文档

- **完整分析**: [VALIDATION_CHECKPOINT_ANALYSIS.md](VALIDATION_CHECKPOINT_ANALYSIS.md)
- **系统指南**: [VALIDATION_SYSTEM_GUIDE.md](VALIDATION_SYSTEM_GUIDE.md)
- **训练日志**: 
  - `logs/train_fixed_20251230-015420.log` (原始)
  - `logs/run_optimized_v3_20251230-030858.log` (V3)

## ✅ 关键决策

✓ **两个训练都已收敛** (改进 < 1%)
✓ **V3 性能更优** (12-24% 改进)
✓ **推荐使用**: `checkpoints_optimized_v3/model_step80000.pt`

## 🔧 当前训练状态

```
原始: Step 93,000 / 100,000 (93%) → ETA ~1.2h
V3:  Step 78,900 / 100,000 (78.9%) → ETA ~3.8h
```

## 📞 快速帮助

**Q: 如何评估 checkpoint?**  
A: `python evaluate_checkpoint_fast.py --checkpoint path/to/model.pt ...`

**Q: 如何检查是否收敛?**  
A: `python check_convergence.py --csv logs/ckpt_eval_xxx.csv`

**Q: 应该用哪个 checkpoint?**  
A: `checkpoints_optimized_v3/model_step80000.pt` (最优)

**Q: 能否立即停止训练?**  
A: 需要信号处理器 (见训练脚本)，或等待 ETA 完成

---

**系统就绪！查看详细文档或执行 `bash run_validation_pipeline.sh`**
