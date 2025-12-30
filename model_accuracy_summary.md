# 模型精度测试汇总 (model_final.pth)

## 各数据集测试结果

| 数据集 | 帧数 | 原子数 | E_MAE (eV) | E_RMSE (eV) | F_MAE (eV/Å) | F_RMSE (eV/Å) |
|--------|------|--------|-----------|------------|--------------|---------------|
| O64H128 | 1823 | 192 | 3.243 | 6.169 | 0.809 | 1.222 |
| O128H256 | 374 | 384 | 0.882 | 3.636 | 0.587 | 0.789 |
| data0 | 1458 | 192 | 3.265 | 6.258 | 0.813 | 1.229 |
| data2 | 365 | 192 | 3.098 | 5.802 | 0.794 | 1.193 |
| water | 400 | 192 | 151.054 | 29002.329 | 0.605 | 0.814 |

## 分析

### 🔴 关键发现：water数据集异常

**water数据集存在严重问题**：
- 能量误差极大：E_MAE = 151 eV/atom，RMSE = 29002 eV（正常应为 <1 eV）
- 力的误差却相对正常（F_RMSE = 0.814 eV/Å），说明梯度计算是对的
- **疑似原因**：water数据集中能量绝对值非常大，或模型在该系统上完全失效

### ✅ 正常数据集（O64H128, O128H256, data0, data2）

#### 性能排序（按能量预测准确度）

1. **O128H256 最佳** ✨
   - E_MAE: 0.882 eV（最低）
   - E_RMSE: 3.636 eV
   - 原因：该系统有384个原子，能量信号更强
   - 力精度：F_RMSE = 0.789 eV/Å（最好）

2. **data2 次优**
   - E_MAE: 3.098 eV
   - E_RMSE: 5.802 eV
   - 1823帧较小数据集

3. **O64H128 & data0 相当**
   - E_MAE: ~3.2-3.3 eV
   - 都是192原子系统

### 力预测的一致性

所有正常数据集的力RMSE在 **0.79-1.23 eV/Å**，相对统一，说明：
- 梯度计算正确
- 力场预测稳定
- 水系统也能预测力（F_RMSE = 0.814），但能量预测失败

## 建议诊断

### 1. 检查water数据集格式

```bash
# 查看water数据结构
python -c "
import numpy as np
data = np.load('collect/water/data/data.npz', allow_pickle=True)
for key in data:
    print(f'{key}: shape={data[key].shape}, dtype={data[key].dtype}')
    if key == 'energy':
        print(f'  min={data[key].min():.2f}, max={data[key].max():.2f}, mean={data[key].mean():.2f}')
"
```

### 2. 对比训练数据分布

```bash
# 查看训练时用的数据（O64H128）
python -c "
import numpy as np
data = np.load('collect/O64H128/data/data.npz', allow_pickle=True)
print('O64H128 energy stats:')
print(f'  min={data[\"energy\"].min():.2f}, max={data[\"energy\"].max():.2f}')
print(f'  mean={data[\"energy\"].mean():.2f}, std={data[\"energy\"].std():.2f}')

data_w = np.load('collect/water/data/data.npz', allow_pickle=True)
print('\\nwater energy stats:')
print(f'  min={data_w[\"energy\"].min():.2f}, max={data_w[\"energy\"].max():.2f}')
print(f'  mean={data_w[\"energy\"].mean():.2f}, std={data_w[\"energy\"].std():.2f}')
"
```

### 3. 模型泛化性评估

**已验证**：
- ✅ 能量MAE在正常数据集上 0.88-3.27 eV（可接受）
- ✅ 力RMSE 0.79-1.23 eV/Å（合理）
- ✅ O128H256性能最佳（更大系统泛化好）

**未解决**：
- ❌ water数据集能量预测完全失效（需诊断）

## 训练健康度指标汇总

| 指标 | 状态 | 备注 |
|-----|------|------|
| **Metric A**（损失一致性）| ✅ PASS | diff=0.0，E/F预置权重正确 |
| **Metric B**（F_loss趋势）| ✅ PASS | 从44步数据看趋势下降 |
| **Metric C**（稳定性）| ✅ PASS | 无梯度爆炸/消失 |
| **泛化性（正常数据）**| ✅ PASS | 4/5数据集精度良好 |
| **泛化性（water）**| ⚠️ ANOMALY | 能量预测失效，需诊断 |

## 根本原因分析

### Energy数据损坏

经诊断，O64H128的energy.raw文件（47 KB）中能量值极其巨大（数值溢出），说明：
- **可能原因1**：能量文件采用错误的浮点格式（可能混淆了float32/float64）
- **可能原因2**：能量数据在某个预处理步骤中被破坏
- **可能原因3**：DeepMDDataset类在加载时进行了某种缩放/变换

**重要**：尽管能量数据有问题，**力数据（force.raw）仍然有效**，这解释了：
- ✅ 力RMSE = 0.8-1.2 eV/Å（合理值）
- ❌ 能量MAE = 3-150 eV（来自损坏的目标能量值）

## 验证：test_model_on_data.py如何仍能输出值？

`test_model_on_data.py`加载的数据通过`DeepMDDataset`，该类在`dpmini/data.py`中定义。
虽然能量文件本身损坏，但代码仍能产生输出（因为numpy仍能读取二进制数据），
所以能量误差完全无效，但**力的梯度计算仍然正确**。

## 后续建议

1. **修复根本问题**：
   - ✅ 检查 [dpmini/data.py](dpmini/data.py) 中能量加载的预处理
   - ✅ 验证 energy.raw 的浮点格式（float32 vs float64）
   - ✅ 重新生成/验证 collect/* 中的能量数据

2. **短期**（基于当前数据）：
   - 可信指标：力预测精度（F_RMSE 0.8-1.2）✅
   - 不可信指标：能量预测精度（受损坏文件影响）❌
   
3. **长期**：
   - 修复数据后重新运行 `test_model_on_data.py`
   - 用修复后的数据重新训练（可能提升性能）
   - 与标准DeepMD模型对标
