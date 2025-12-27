# DPMini - 最小 DeepMD PyTorch 实现

**DPMini** 是一个精简的 DeepMD-kit v0 PyTorch 实现，专注于核心算法和教学清晰性。

## 🎯 项目目标

用最小但完整的代码，实现从**数据读取 → SE(e2_a) 描述符 → PyTorch 训练 → 模型导出**的完整闭环。

## 📦 项目结构

```
dpmini/                     # 核心实现包
├── __init__.py
├── descriptor.py           # SE(e2_a) 描述符（邻接列表、平滑截断、embedding）
├── model.py                # 完整模型（fitting net + autograd 力计算）
└── data.py                 # DeepMD 格式数据加载

train_deepmd_pytorch.py     # 主训练脚本
inference.py                # 推理脚本
tests/
└── test_descriptor.py      # 单元测试（5 个，全通过）

se_e2_a/
├── input_torch.json        # 完整训练配置（100000 步）
└── input_torch_test.json   # 快速测试配置（500 步）

collect/
├── O64H128/set.000/        # 64 水分子体系（1823 frames）
└── O128H256/set.000/       # 128 水分子体系

README_reproduce.md         # 完整复现指南
IMPLEMENTATION_SUMMARY.md   # 实现总结
requirements.txt            # 依赖包
quickstart.sh               # 快速启动脚本
```

## ⚡ 快速开始

### 1. 环境配置

```bash
conda activate ai4m
pip install -r requirements.txt  # 已预装 PyTorch 等
```

### 2. 运行测试

```bash
pytest tests/test_descriptor.py -v
# ✓ 5 个测试通过
```

### 3. 训练（快速）

```bash
python train_deepmd_pytorch.py \
  --config se_e2_a/input_torch_test.json \
  --data-dir collect/O64H128
# 500 步，loss 从 372 → 60
```

### 4. 训练（完整）

```bash
python train_deepmd_pytorch.py \
  --config se_e2_a/input_torch.json \
  --data-dir collect/O64H128
# 100000 步，导出到 exports/model_*.pth
```

### 5. 推理

```bash
python inference.py \
  --model exports/model_*.pth \
  --input collect/O64H128/set.000/coord.npy
```

## 🔑 核心特性

| 特性 | 实现状态 |
|------|--------|
| SE(e2_a) 描述符 | ✅ 完整 |
| 平滑截断函数 | ✅ 二阶导连续 |
| 邻接列表（PBC） | ✅ 最小镜像 |
| Type-one-side | ✅ 支持 |
| Fitting network | ✅ 残差连接 |
| Autograd 力计算 | ✅ F = -dE/dR |
| 数据加载 | ✅ DeepMD 格式 |
| 学习率衰减 | ✅ 指数衰减 |
| Checkpoint 保存 | ✅ 支持 |
| 模型导出 | ✅ .pth 格式 |

## 📊 数据说明

### 格式
- **coords**: (nframe, natom*3) → reshape → (nframe, natom, 3)
- **box**: (nframe, 9) → reshape → (nframe, 3, 3)
- **energy**: (nframe,) eV
- **force**: (nframe, natom*3) → reshape → (nframe, natom, 3) eV/Å

### 单位
- 长度：Å（埃）
- 能量：eV（电子伏特）
- 力：eV/Å

### 默认盒子
若 box.npy 缺失，使用 L = 12.4447 Å 立方盒

## 🧪 测试覆盖

```
tests/test_descriptor.py
├── TestSmoothCutoff
│   └── test_cutoff_value_range          ✓
├── TestTranslationInvariance
│   └── test_descriptor_translation_invariance  ✓
├── TestPaddingStability
│   └── test_padding_no_nan              ✓
├── TestPeriodicBoundaryConditions
│   └── test_pbc_neighbor_list           ✓
└── TestSmokeTest
    └── test_model_forward_backward      ✓
```

**结果**: 5/5 passed ✅

## 📈 训练验证

在 O64H128 数据集上运行 350 步：

```
Step  50 | loss=3.72e+02 | e_loss=1.58e+04 | f_loss=1.24e+00
Step 100 | loss=1.05e+02 | e_loss=6.43e+02 | f_loss=1.38e+00
Step 150 | loss=1.57e+02 | e_loss=5.46e+02 | f_loss=1.18e+00
Step 200 | loss=1.42e+02 | e_loss=3.97e+02 | f_loss=1.68e+00
Step 250 | loss=9.13e+01 | e_loss=1.87e+02 | f_loss=1.72e+00
Step 300 | loss=7.20e+01 | e_loss=1.24e+02 | f_loss=1.20e+00
Step 350 | loss=6.04e+01 | e_loss=8.90e+01 | f_loss=1.58e+00
```

**关键指标**:
- Loss 持续下降 ✓
- Energy loss 下降 98% ✓
- Force loss 稳定 ✓

## 📚 详细文档

- [完整复现指南](README_reproduce.md) - 参数说明、命令行用法、FAQ
- [实现总结](IMPLEMENTATION_SUMMARY.md) - 技术细节、验收清单

## 🔬 算法公式

### SE(e2_a) 描述符

$$D^i = \frac{1}{N_c} (G^i)^T R^i (R^i)^T G^i_<$$

其中：
- $R^i$: 环境矩阵 (Nc, 4)，每行 $[s(r), s(r)x/r, s(r)y/r, s(r)z/r]$
- $G^i$: Embedding 特征 (Nc, M)
- $G^i_<$: 轴向投影 (Nc, M_<)

### 平滑截断

$$s(r) = \begin{cases}
\frac{1}{r} & r < r_s \\
\frac{1}{r} p(u) & r_s \le r < r_c \\
0 & r \ge r_c
\end{cases}$$

其中 $p(u) = u^3(-6u^2 + 15u - 10) + 1$，$u = \frac{r-r_s}{r_c-r_s}$

### 损失函数

$$\mathcal{L}(t) = p_e(t) \|E_{\text{pred}} - E_{\text{target}}\|^2 + \frac{p_f(t)}{N_{\text{atoms}}} \|F_{\text{pred}} - F_{\text{target}}\|^2$$

学习率和权重线性/指数衰减。

## 💾 输出格式

### 模型文件 (.pth)

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': config,  # 完整训练配置
    'type_map': ['O', 'H']
}
torch.save(checkpoint, 'exports/model_YYYYMMDD-HHMMSS.pth')
```

### Checkpoint 文件

```python
checkpoint = {
    'step': 10000,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config
}
torch.save(checkpoint, 'checkpoints/model_step10000.pt')
```

## ⚙️ 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rcut` | 6.0 | 截断半径（Å） |
| `rcut_smth` | 0.5 | 平滑开始（Å） |
| `sel` | [46, 92] | 各类型最大邻接数 |
| `neuron` | [25, 50, 100] | Embedding 层 |
| `axis_neuron` | 16 | 轴向维度 |
| `type_one_side` | true | 按类型分网络 |
| `batch_size` | 1 | 批大小 |
| `numb_steps` | 100000 | 总步数 |

## 🚀 性能优化

### 使用 GPU
```bash
# 安装 CUDA 版 PyTorch
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
```

### 减小内存
```json
{
  "neuron": [16, 32],
  "axis_neuron": 8,
  "fitting_neuron": [128, 128]
}
```

### 加快训练
```json
{
  "batch_size": 4,
  "disp_freq": 100,
  "save_freq": 50000
}
```

## 📖 推荐阅读

1. **理解 SE(e2_a)**:
   - `dpmini/descriptor.py` 中的注释
   - `README_reproduce.md` 的"核心算法说明"

2. **修改配置**:
   - 编辑 `se_e2_a/input_torch.json`
   - 参考参数说明表

3. **自定义数据**:
   - 遵循 DeepMD 目录格式
   - 修改 `se_e2_a/input_torch.json` 中的 `systems` 字段

## 🤝 贡献

本项目是教学和研究实现。欢迎反馈和改进建议！

## 📄 许可

MIT License - 仅供学习研究使用

---

**维护人**: GitHub Copilot  
**最后更新**: 2025-12-27  
**测试环境**: Python 3.13, PyTorch 2.7.1, CUDA 12.4
