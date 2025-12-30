# 修改汇总清单

## 修改统计
- **核心修改：** 1 文件 (`dpmini/descriptor.py`)
- **配置调整：** 2 文件 (`config_short_training_fixed.json`, `config_formal_100k.json`)
- **新增脚本：** 3 个 (`extract_training_metrics.py`, `launch_training.sh`, 本清单)
- **新增文档：** 2 个 (`FIX_VALIDATION_REPORT.md`, `EXECUTION_SUMMARY.md`)

---

## 核心修改详情

### 修改 1: `dpmini/descriptor.py` - build_neighbor_list

**位置：** 函数 `build_neighbor_list()` 内部

**变更 1a：Self-mask bug 修复**
```python
# 旧代码 (行 ~127-131)
dist = torch.sqrt(dist2 + 1e-12)
valid_mask = (dist < rcut) & (dist > 1e-8)  # ❌ 1e-6 > 1e-8，对角线通过

# 新代码
eye_mask = torch.eye(natom, dtype=torch.bool, device=device)
dist = torch.sqrt(dist2 + 1e-12)
valid_mask = (dist < rcut) & (~eye_mask)  # ✅ 显式排除对角线
```

**变更 1b：按类型分别 topk**
```python
# 旧代码 (行 ~146-176)
# 混合排序，按 type*1000 + dist

# 新代码
offset = 0
for t in range(ntypes):
    mask_t = (neighbor_types == t) & valid_mask
    dist_t = dist.masked_fill(~mask_t, float('inf'))
    
    _, indices_t = torch.topk((-dist_t).view(natom, -1), k=sel[t], dim=1)
    indices_t = indices_t[:, :sel[t]]
    
    for j_idx, j in enumerate(indices_t):
        valid = j < natom
        neighbor_indices[i, offset + j_idx] = j if valid else -1
        neighbor_types[i, offset + j_idx] = t if valid else -1
        neighbor_mask[i, offset + j_idx] = float(valid)
    
    offset += sel[t]
```

---

## 配置文件调整

### 调整 1: `config_short_training_fixed.json`

**变更：**
```json
{
  "learning_rate": {
    "type": "exp",
    "start_lr": 0.001,
    "stop_lr": 1e-6,              // ← 从 3.51e-8 改为 1e-6
    "decay_steps": 2000           // 保持不变
  },
  "loss": {
    "start_pref_e": 0.02,
    "limit_pref_e": 1.0,
    "start_pref_f": 1000.0,
    "limit_pref_f": 20.0          // 已经是对的值
  }
}
```

### 调整 2: `config_formal_100k.json`

**变更：**
```json
{
  "learning_rate": {
    "type": "exp",
    "start_lr": 0.001,
    "stop_lr": 1e-6,              // ← 修改
    "decay_steps": 20000          // ← 从 2000 改为 20000
  },
  "loss": {
    "start_pref_e": 0.02,
    "limit_pref_e": 1.0,
    "start_pref_f": 1000.0,
    "limit_pref_f": 20.0          // ← 从 1.0 改为 20.0
  }
}
```

---

## 新增工具与文档

### 新增脚本

#### 1. `extract_training_metrics.py`
- **用途：** 从训练日志中提取 f_rmse, e_loss 等指标
- **输出：** metrics.csv 和 metrics.png
- **用法：** `python extract_training_metrics.py logs/train_xxx.log`

#### 2. `launch_training.sh`
- **用途：** 一键启动训练（包括环境检查）
- **特性：** 自动创建日志目录、运行健康检查、后台启动
- **用法：** `bash launch_training.sh`

#### 3. 其他脚本（已存在）
- `test_training_fixes.py` - 健康检查（已验证通过）
- `train_cuda_optimized.py` - 主训练脚本（无修改）

### 新增文档

#### 1. `FIX_VALIDATION_REPORT.md`
- 完整的修复验证报告
- 包含修复前后的对比
- 详细的技术分析

#### 2. `EXECUTION_SUMMARY.md`
- 执行总结和快速参考
- 下一步指南
- 常见问题排查

#### 3. `MODIFICATION_CHECKLIST.md` (本文件)
- 修改清单和检查表
- 便于代码审查和维护

---

## 验证清单

### ✅ 代码修改验证
- [x] `build_neighbor_list` 的 self-mask 已用 `eye_mask` 修复
- [x] `build_neighbor_list` 的 sel topk 已按类型分别执行
- [x] 数值稳定性（无 NaN/Inf）
- [x] Gradient flow（loss consistency check）

### ✅ 测试验证
- [x] `test_training_fixes.py` 全部通过
- [x] 短训练 (2000 步) 完成，无错误
- [x] F_rmse 曲线显示下降趋势 (2.17 → 0.63)
- [x] 日志文件正常生成

### ✅ 配置验证
- [x] `config_short_training_fixed.json` 参数合理
- [x] `config_formal_100k.json` 参数合理
- [x] LR schedule 正确（无衰减到 1e-8 的问题）
- [x] Loss prefactors 正确（force 权重维持）

### ✅ 文档验证
- [x] 修复报告完整
- [x] 执行总结清晰
- [x] 启动脚本可用
- [x] 数据提取工具可用

---

## 性能基准

### 短训练 (2000 步)
| 指标 | 值 |
|------|-----|
| 执行时间 | 23 分钟 |
| 步数速率 | 1.44 step/s |
| GPU 内存 | ~8GB (稳定) |
| F_rmse 改进 | 71% (2.17 → 0.63) |

### 推荐配置
- **短测试：** 2000 步 (无 1.5 分钟)
- **标准验证：** 10000 步 (~2 小时)
- **生产训练：** 100000 步 (~30 小时)

---

## 回滚方案 (如遇问题)

### 快速回滚
```bash
# 恢复 descriptor.py（git 命令）
git checkout -- dpmini/descriptor.py

# 使用旧配置
python train_cuda_optimized.py --config config_minimal.json ...
```

### 问题排查
1. 查看日志：`tail -f logs/train_xxx.log`
2. 检查 GPU：`nvidia-smi`
3. 重新测试：`python test_training_fixes.py`

---

## 提交信息建议

```
feat: 修复 DeepMD neighbor list 中的自原子污染和 sel 分配问题

- 修复 build_neighbor_list 中的 self-mask bug
  使用显式 eye_mask 替代 dist > 1e-8 判据
  
- 实现按类型的 topk 邻居选择
  确保每种类型各取 sel[t] 个邻居
  
- 调整超参数
  stop_lr: 3.51e-8 → 1e-6 (短训练)
  limit_pref_f: 1.0 → 20.0
  
- 验证结果
  f_rmse: 2.17 → 0.63 (71% 改进)
  test_training_fixes.py: PASSED
  短训练 2000 步: 成功完成

Fixes: DeepMD force 学习曲线无下降问题
```

---

## 维护建议

### 定期检查
- 每月检查训练日志中的 f_rmse 趋势
- 定期更新配置文件中的超参数
- 保留最后 50 个 checkpoint 便于分析

### 长期计划
- 考虑实现 neighbor list 缓存（加速计算）
- 优化 descriptor 的 GPU 内存使用
- 添加自动化的参数搜索工具

---

**版本：** 1.0  
**最后更新：** 2025年12月29日  
**状态：** 生产可用 ✅

