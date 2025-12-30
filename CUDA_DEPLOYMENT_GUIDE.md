# CUDA 加速训练环境部署指南

**创建时间**: 2025-12-27  
**目的**: 在独立环境中使用CUDA GPU加速训练，不影响现有CPU训练任务

## 📋 项目结构

```
/home/ubuntu/pj           # CPU训练环境（原项目，继续运行）
/home/ubuntu/pj_cuda      # CUDA GPU加速环境（新复制）

Conda环境:
- ai4m                    # 现有CPU环境（保持原样）
- cuda_env                # 新建CUDA GPU环境
```

---

## ✅ 已完成的设置

### 1. 创建独立的CUDA环境
```bash
conda create -n cuda_env python=3.10 -y
```

### 2. 安装PyTorch CUDA版本 (2.5.1+cu121)
```bash
source /home/ubuntu/miniforge3/bin/activate cuda_env
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. 安装依赖包
```bash
pip install numpy scipy h5py pytest
```

### 4. 复制项目文件
```bash
cp -r /home/ubuntu/pj /home/ubuntu/pj_cuda
# 大小: 363M
```

### 5. 创建CUDA优化训练脚本
- 文件: `/home/ubuntu/pj/train_cuda.py`
- 包含GPU加速优化:
  - CUDA设备检测和信息显示
  - pin_memory加速数据传输
  - 混合精度训练(fp16可选)
  - cuDNN auto-tuner自动优化
  - GPU内存管理监控
  - 进度估算和速度统计

---

## 🚀 使用方法

### 方案A: 在原项目目录启动CUDA训练（推荐）

```bash
# 激活CUDA环境
conda activate cuda_env

# 进入项目目录
cd /home/ubuntu/pj

# 运行CUDA优化版训练脚本
python train_cuda.py \
  --config se_e2_a/input_torch_quick_test.json \
  --data-dir collect/O64H128 \
  --checkpoint-dir checkpoints_cuda \
  --export-dir exports_cuda \
  --num-workers 4 \
  --mixed-precision

# 参数说明:
# --config              配置文件路径
# --data-dir            数据目录（使用原数据）
# --checkpoint-dir      保存检查点到别的目录（避免覆盖原检查点）
# --export-dir          导出模型到别的目录
# --num-workers         数据加载线程数（GPU训练建议4-8）
# --mixed-precision     启用混合精度(fp16)加速，可选
```

### 方案B: 在新复制的环境中训练（完全隔离）

```bash
# 激活CUDA环境
conda activate cuda_env

# 进入新复制的项目目录
cd /home/ubuntu/pj_cuda

# 运行CUDA优化版训练脚本
python train_cuda.py \
  --config se_e2_a/input_torch.json \
  --data-dir collect/O64H128 \
  --num-workers 4
```

---

## 📊 性能对比

### CPU vs GPU 预期性能差异

| 配置 | 设备 | 批处理大小 | 预期速度 |
|------|------|-----------|---------|
| 原环境 | CPU (700%) | 1 | ~10-15 steps/min (~0.17 steps/sec) |
| CUDA环境 | NVIDIA GPU | 1 | ~5-10x加速 (预计 1-2 steps/sec) |
| CUDA环境 | NVIDIA GPU | 4-8 | ~8-15x加速 (预计 1.5-3 steps/sec) |

**预计完成时间**:
- CPU训练 (10000步): ~10-12小时
- CUDA训练 (10000步): ~1-2小时

---

## 🔍 验证CUDA环境

### 检查PyTorch CUDA支持

```bash
conda activate cuda_env
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'CUDA版本: {torch.version.cuda}')
print(f'GPU数量: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        props = torch.cuda.get_device_properties(i)
        print(f'  内存: {props.total_memory / 1e9:.2f} GB')
"
```

### 运行快速测试

```bash
conda activate cuda_env
cd /home/ubuntu/pj

# 运行CUDA版本快速测试 (500步，30秒-1分钟)
python train_cuda.py \
  --config se_e2_a/input_torch_quick_test.json \
  --data-dir collect/O64H128 \
  --checkpoint-dir /tmp/test_ckpt \
  --export-dir /tmp/test_export
```

---

## 📝 CUDA优化版脚本特性

### 新增功能

1. **CUDA设备检查** (`check_cuda_info()`)
   - 自动检测GPU可用性
   - 显示GPU型号、内存、CUDA/cuDNN版本
   - 启用cuDNN自动调优

2. **数据加载优化**
   ```python
   DataLoader(
       dataset, 
       batch_size=batch_size,
       pin_memory=torch.cuda.is_available(),  # GPU内存预固定
       num_workers=4,                          # 多进程加载
       persistent_workers=True                # 保持加载进程
   )
   ```

3. **混合精度训练(可选)**
   ```bash
   # 启用 --mixed-precision 标志
   python train_cuda.py ... --mixed-precision
   
   # 优势:
   # - 降低显存占用 (~50%)
   # - 加快计算速度 (1.5-2x)
   # - 精度损失极小
   ```

4. **性能监测**
   - 每个显示间隔显示训练速度 (steps/sec)
   - 显示剩余时间估算 (ETA)
   - GPU内存使用统计

5. **并发训练**
   - CPU任务: ai4m环境继续运行
   - GPU任务: cuda_env环境独立运行
   - 相互不影响

---

## 💾 检查点和模型导出

### 文件位置约定

```
Original CPU Training:
  checkpoints/         # CPU版检查点
  exports/             # CPU版导出模型

CUDA GPU Training:
  checkpoints_cuda/    # GPU版检查点（推荐）
  exports_cuda/        # GPU版导出模型（推荐）
```

### 模型命名规则

```
CPU版本:  model_20251227-194512.pth
CUDA版本: model_cuda_20251227-194512.pth
```

### 恢复训练

```bash
# 从检查点恢复 (需要修改train_cuda.py添加加载逻辑)
# 当前版本: 从头开始训练
# 改进计划: 支持 --resume-from-checkpoint 参数
```

---

## 🛠️ 常见问题

### 问题1: CUDA不可用

```bash
# 检查
python -c "import torch; print(torch.cuda.is_available())"

# 如果返回False:
# 1. 检查NVIDIA GPU: lspci | grep NVIDIA
# 2. 检查驱动: nvidia-smi
# 3. 重新安装PyTorch CUDA版本
conda activate cuda_env
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 问题2: GPU内存不足 (CUDA out of memory)

```bash
# 解决方案:
# 1. 减少批处理大小 (修改config文件的batch_size)
# 2. 启用混合精度: --mixed-precision
# 3. 减少num_workers (数据加载线程)

python train_cuda.py \
  --config se_e2_a/input_torch.json \
  --data-dir collect/O64H128 \
  --num-workers 2 \
  --mixed-precision
```

### 问题3: CPU和GPU训练冲突

```bash
# 不会冲突!
# - CPU训练: conda activate ai4m + 使用cpu
# - GPU训练: conda activate cuda_env + 使用cuda
# 两个环境完全独立

# 验证当前训练进程
ps aux | grep train_deepmd_pytorch
# 应该看到两个进程:
#   python train_cpu.py      (ai4m环境)
#   python train_cuda.py (cuda_env环境)
```

### 问题4: 模型不兼容

```bash
# CPU版模型 (.pth) 可以在GPU加载，但需要修改device:
# 在inference.py或推理脚本中:
# 
# CPU加载: checkpoint = torch.load(path, map_location='cpu')
# GPU加载: checkpoint = torch.load(path)  # 默认使用cuda
```

---

## 📈 优化建议

### 1. 增加批处理大小 (如果GPU内存允许)

```json
// 在 se_e2_a/input_torch.json 中修改:
{
  "training": {
    "training_data": {
      "batch_size": 4  // 原为1，改为4-8
    }
  }
}
```

### 2. 增加数据加载线程数

```bash
python train_cuda.py \
  ... \
  --num-workers 8  # 根据CPU核心数调整
```

### 3. 启用混合精度 (推荐)

```bash
python train_cuda.py \
  ... \
  --mixed-precision
```

### 4. 监控GPU使用

```bash
# 在另一个终端运行
watch -n 1 nvidia-smi
```

---

## 🔄 后续改进

### 计划功能

- [ ] 从检查点恢复训练 (`--resume-from-checkpoint`)
- [ ] 分布式多GPU训练 (DistributedDataParallel)
- [ ] 学习率预热 (learning rate warmup)
- [ ] 梯度累积 (gradient accumulation)
- [ ] 张量并行 (tensor parallelism for large models)

---

## 📞 故障排除

### 运行诊断脚本

```bash
conda activate cuda_env
cd /home/ubuntu/pj

# 创建诊断脚本
python << 'EOF'
import sys
import torch
import numpy as np
from dpmini import DeepMDModel, DeepMDDataset

print("=" * 60)
print("DeepMD CUDA环境诊断")
print("=" * 60)

# 1. PyTorch检查
print("\n1. PyTorch信息:")
print(f"   版本: {torch.__version__}")
print(f"   CUDA可用: {torch.cuda.is_available()}")

# 2. 数据加载检查
print("\n2. 数据加载检查:")
try:
    dataset = DeepMDDataset(['collect/O64H128'], type_map=['O', 'H'])
    print(f"   数据集大小: {len(dataset)} frames")
    print(f"   ✓ 数据加载成功")
except Exception as e:
    print(f"   ✗ 数据加载失败: {e}")

# 3. 模型创建检查
print("\n3. 模型创建检查:")
try:
    model = DeepMDModel(
        type_map=['O', 'H'],
        rcut=6.0, rcut_smth=0.5, sel=[64, 128],
        descriptor_neuron=[25, 50, 100],
        axis_neuron=16,
        fitting_neuron=[240, 240, 240]
    )
    
    # 移到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"   参数数量: {params}")
    print(f"   设备: {device}")
    print(f"   ✓ 模型创建成功")
except Exception as e:
    print(f"   ✗ 模型创建失败: {e}")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)
EOF
```

---

## ✨ 总结

| 方面 | 状态 | 备注 |
|------|------|------|
| 环境隔离 | ✅ | 独立conda环境，不影响原任务 |
| CUDA支持 | ✅ | PyTorch CUDA 12.1 已安装 |
| 项目复制 | ✅ | /home/ubuntu/pj_cuda (363M) |
| 训练脚本 | ✅ | train_cuda.py |
| GPU加速 | ✅ | 预计5-15x加速 |
| 混合精度 | ✅ | 可选启用 |
| 即用状态 | ✅ | 可以立即启动训练 |

---

**下一步**: 运行快速测试验证GPU加速效果
```bash
conda activate cuda_env
cd /home/ubuntu/pj
python train_cuda.py \
  --config se_e2_a/input_torch_quick_test.json \
  --data-dir collect/O64H128 \
  --num-workers 4
```
