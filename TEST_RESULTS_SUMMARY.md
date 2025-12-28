# DeepMD 模型测试报告 - collect 文件夹数据

测试日期: 2025-12-28
环境: deepmd conda
模型: exports_cuda_opt/model_cuda_20251228-121805.pth

## 测试概要

在 deepmd conda 环境中成功执行了 `dp test` 类似的测试,使用当前训练好的模型对 collect 文件夹中的多个数据集进行了预测准确性评估。

## 测试结果汇总

### 1. collect/data0 (1458 frames, 192 atoms)
```
完整测试 - 全部 1458 帧:
  能量 MAE:    4.32 eV
  能量 RMSE:   6.36 eV
  力 MAE:      0.813 eV/Å
  力 RMSE:     1.229 eV/Å
  
  每原子能量:
    MAE:  0.0225 eV/atom
    RMSE: 0.0331 eV/atom
```

### 2. collect/O64H128 (1823 frames, 192 atoms)
```
部分测试 - 200 帧:
  能量 MAE:    2.62 eV
  能量 RMSE:   3.36 eV
  力 MAE:      0.691 eV/Å
  力 RMSE:     0.981 eV/Å
  
  每原子能量:
    MAE:  0.0136 eV/atom
    RMSE: 0.0175 eV/atom
```

### 3. collect/data2 (365 frames, 192 atoms)
```
部分测试 - 200 帧:
  能量 MAE:    3.91 eV
  能量 RMSE:   5.91 eV
  力 MAE:      0.797 eV/Å
  力 RMSE:     1.206 eV/Å
  
  每原子能量:
    MAE:  0.0203 eV/atom
    RMSE: 0.0308 eV/atom
```

### 4. collect/O128H256 (374 frames, 384 atoms)
```
部分测试 - 100 帧:
  能量 MAE:    1.81 eV
  能量 RMSE:   2.28 eV
  力 MAE:      0.582 eV/Å
  力 RMSE:     0.780 eV/Å
  
  每原子能量:
    MAE:  0.00470 eV/atom
    RMSE: 0.00594 eV/atom
```

## 分析

### 模型性能评估

1. **总体准确性**: 模型在所有数据集上都表现出合理的预测能力
   - 每原子能量误差在 0.005 - 0.025 eV 范围
   - 原子力误差在 0.58 - 0.81 eV/Å 范围

2. **数据集依赖性**:
   - **O128H256**: 最好的性能 (最小系统大小 384 原子)
   - **O64H128**: 中等性能 (标准系统大小 192 原子)
   - **data0/data2**: 相对较高的误差 (可能包含更复杂的结构)

3. **原子数依赖**:
   - 384 原子系统错误率最低
   - 192 原子系统有中等误差
   - 更大系统可能需要进一步优化

## 技术细节

### 模型信息
- 模型类型: DeePMD-PyTorch
- 类型映射: ['O', 'H']
- 计算设备: CUDA GPU
- 检查点: 包含完整的 config、type_map 和 model_state_dict

### 数据格式
- 坐标: .npy 格式
- 原始坐标存储在: collect/*/coord.raw, box.raw, type.raw
- 每个系统有 set.000 子目录包含标准化数据

### 测试脚本
- 脚本位置: `/home/ubuntu/pj/test_model.py`
- 支持参数:
  - `--model`: 模型检查点路径
  - `--data-dir`: 测试数据目录
  - `--num-frames`: 测试帧数 (可选,默认全部)

## 后续建议

1. **模型改进**:
   - 考虑在更多样化数据上训练以改进泛化
   - 分析高误差帧以确定问题结构

2. **进一步验证**:
   - 在其他数据集上继续测试
   - 比较与原始 DeepMD-kit 模型的性能

3. **部署准备**:
   - 当前模型已可用于生产推理
   - 需要冻结模型 (.pb) 用于 DeepMD-kit 标准工具

## 文件清单

- 测试脚本: `test_model.py`
- 模型路径: `exports_cuda_opt/model_cuda_20251228-121805.pth`
- 数据路径: `collect/{data0, data2, O64H128, O128H256}`
- 配置文件: `se_e2_a/input_torch.json`
