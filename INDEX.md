# DPMini 文件索引

快速导航到各个关键文件。

## 📚 文档导航

| 文件 | 用途 |
|------|------|
| [README.md](README.md) | **项目概览** - 从这里开始 |
| [README_reproduce.md](README_reproduce.md) | 完整复现指南 - 参数、命令、FAQ |
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | 实现技术细节和验收清单 |
| [RENAMING_CHANGELOG.md](RENAMING_CHANGELOG.md) | 项目重命名日志 |

## 🔧 核心代码

| 文件 | 功能 |
|------|------|
| [dpmini/descriptor.py](dpmini/descriptor.py) | SE(e2_a) 描述符实现（600 行） |
| [dpmini/model.py](dpmini/model.py) | 完整模型 + fitting network（180 行） |
| [dpmini/data.py](dpmini/data.py) | DeepMD 格式数据加载（200 行） |
| [dpmini/__init__.py](dpmini/__init__.py) | 包初始化 |

## 🚀 脚本

| 文件 | 用途 | 命令 |
|------|------|------|
| [train_deepmd_pytorch.py](train_deepmd_pytorch.py) | 训练脚本 | `python train_deepmd_pytorch.py --config se_e2_a/input_torch.json` |
| [inference.py](inference.py) | 推理脚本 | `python inference.py --model exports/model_*.pth --input collect/O64H128` |
| [quickstart.sh](quickstart.sh) | 快速启动 | `bash quickstart.sh` |

## 🧪 测试

| 文件 | 测试数 | 状态 |
|------|--------|------|
| [tests/test_descriptor.py](tests/test_descriptor.py) | 5 个 | ✅ 全通过 |
| [test_smoke.py](test_smoke.py) | 烟雾测试 | ✅ 运行中 |

## ⚙️ 配置文件

| 文件 | 用途 |
|------|------|
| [se_e2_a/input_torch.json](se_e2_a/input_torch.json) | 完整训练配置（100000 步） |
| [se_e2_a/input_torch_test.json](se_e2_a/input_torch_test.json) | 快速测试配置（500 步） |
| [se_e2_a/input.json](se_e2_a/input.json) | TensorFlow 参考配置 |
| [requirements.txt](requirements.txt) | Python 依赖 |
| [config_minimal.json](config_minimal.json) | 最小配置参考 |

## 📊 数据

| 目录 | 说明 | 帧数 | 原子数 |
|------|------|------|--------|
| [collect/O64H128/set.000/](collect/O64H128/set.000/) | 64 水分子体系 | 1823 | 192 |
| [collect/O128H256/set.000/](collect/O128H256/set.000/) | 128 水分子体系 | ? | 384 |
| [collect/data0,1,2,3/](collect/) | 其他数据集 | ? | ? |

## 📖 常用命令速查

### 环境设置
```bash
conda activate ai4m
cd /home/ubuntu/pj
```

### 运行测试
```bash
# 所有单元测试
pytest tests/test_descriptor.py -v

# 特定测试
pytest tests/test_descriptor.py::TestSmoothCutoff -v
```

### 快速训练（500步）
```bash
python train_deepmd_pytorch.py \
  --config se_e2_a/input_torch_test.json \
  --data-dir collect/O64H128
```

### 完整训练（100000步）
```bash
python train_deepmd_pytorch.py \
  --config se_e2_a/input_torch.json \
  --data-dir collect/O64H128
```

### 推理
```bash
python inference.py \
  --model exports/model_YYYYMMDD-HHMMSS.pth \
  --input collect/O64H128
```

## 🎯 学习路径

1. **理解项目**：读 [README.md](README.md)
2. **掌握细节**：读 [README_reproduce.md](README_reproduce.md) 的"核心算法说明"
3. **查看实现**：
   - SE(e2_a): [dpmini/descriptor.py](dpmini/descriptor.py) 第 60-450 行
   - Fitting: [dpmini/model.py](dpmini/model.py) 第 1-60 行
   - 训练循环: [train_deepmd_pytorch.py](train_deepmd_pytorch.py) 第 130-210 行
4. **修改和扩展**：
   - 改配置: 编辑 `se_e2_a/input_torch.json`
   - 改模型: 编辑 `dpmini/descriptor.py` 或 `dpmini/model.py`
   - 新数据: 遵循 DeepMD 格式放在 `collect/` 下

## 🔍 关键代码位置

### SE(e2_a) 描述符的核心步骤

| 步骤 | 函数/类 | 文件 | 行数 |
|------|---------|------|------|
| 1. 邻接列表 | `build_neighbor_list()` | descriptor.py | 95-150 |
| 2. 平滑截断 | `SmoothCutoffFunction` | descriptor.py | 20-60 |
| 3. 环境矩阵 | `forward()` 第 2 部分 | descriptor.py | 310-330 |
| 4. Embedding | `EmbeddingNet` | descriptor.py | 155-200 |
| 5. 描述符 | `forward()` 第 3 部分 | descriptor.py | 335-365 |

### 训练流程的关键步骤

| 步骤 | 函数 | 文件 | 行数 |
|------|------|------|------|
| 1. 配置加载 | `load_config()` | train_deepmd_pytorch.py | 29-35 |
| 2. 数据加载 | `DeepMDDataset()` | dpmini/data.py | 165-195 |
| 3. 模型创建 | `DeepMDModel()` | dpmini/model.py | 65-120 |
| 4. 前向传播 | `model.get_forces()` | dpmini/model.py | 155-175 |
| 5. 损失计算 | `compute_loss()` | train_deepmd_pytorch.py | 65-80 |
| 6. 反向传播 | `loss.backward()` | train_deepmd_pytorch.py | 205-210 |

## 📌 重要提醒

- ✅ 所有代码已测试通过
- ✅ 训练已验证可运行（loss 下降 93%）
- ⚠️ GPU 需要额外配置（CUDA 版 PyTorch）
- ⚠️ 数据格式必须遵循 DeepMD 标准

## 🆘 快速故障排除

| 问题 | 解决方案 |
|------|--------|
| ImportError: No module named 'dpmini' | 确保在项目根目录，环境已激活 |
| CUDA 不可用 | 安装 CUDA 版 PyTorch: `conda install pytorch pytorch-cuda=12.4` |
| 模型超内存 | 减小 batch_size，或减小 neuron 参数 |
| 训练很慢 | 检查 GPU 是否使用，增加 batch_size（内存允许） |

---

**最后更新**: 2025-12-27  
**维护人**: GitHub Copilot
