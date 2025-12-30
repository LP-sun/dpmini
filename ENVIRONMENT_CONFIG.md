# 环境配置总结

**部署日期**: 2025-12-27  
**部署人**: GitHub Copilot

## 系统环境

```
操作系统: Linux
CPU: 8核 (Intel)
总内存: 16+ GB
GPU: NVIDIA L20-8Q (8.36 GB VRAM)
CUDA能力: 8.9 (Hopper架构)
```

## Conda环境配置

### 环境1: ai4m (CPU训练 - 原有)
```bash
# 激活
conda activate ai4m

# 路径
/home/ubuntu/miniforge3/envs/ai4m

# PyTorch版本
torch==2.9.1+cpu  (CPU版本)

# 状态: ✓ 保持原样，CPU训练继续运行
```

### 环境2: cuda_env (GPU训练 - 新建) ⭐
```bash
# 激活
conda activate cuda_env

# 路径
/home/ubuntu/miniforge3/envs/cuda_env

# Python版本
Python 3.10.19 (conda-forge)

# 已安装包清单
- torch==2.5.1+cu121         ✓ PyTorch CUDA 12.1
- torchvision==0.20.1+cu121  ✓ 计算机视觉库
- torchaudio==2.5.1+cu121    ✓ 音频库
- numpy==2.2.6               ✓ 数值计算
- scipy==1.15.3              ✓ 科学计算
- h5py==3.15.1               ✓ HDF5数据格式
- pytest==9.0.2              ✓ 测试框架
- cuda-runtime==12.1         ✓ CUDA运行库
- cudnn==9.01                ✓ cuDNN深度学习库
```

## 项目目录结构

```
/home/ubuntu/
├── pj/                          # CPU训练项目 (保持原样)
│   ├── train_cpu.py         (原脚本)
│   ├── train_cuda.py    (新增: CUDA版本) ⭐
│   ├── inference.py
│   ├── dpmini/
│   ├── collect/O64H128/        # 共享数据目录 (只读)
│   ├── checkpoints/            # CPU检查点 (不改动)
│   ├── exports/                # CPU导出 (不改动)
│   ├── checkpoints_cuda/       # GPU检查点 (新增) ⭐
│   ├── exports_cuda/           # GPU导出 (新增) ⭐
│   ├── se_e2_a/
│   ├── CUDA_DEPLOYMENT_GUIDE.md  (新增) ⭐
│   ├── CUDA_SETUP_COMPLETE.md    (新增) ⭐
│   ├── README_CUDA_QUICK_START.txt (新增) ⭐
│   ├── cuda_quickstart.sh       (新增) ⭐
│   ├── verify_cuda_env.py       (新增) ⭐
│   └── ...
│
└── pj_cuda/                     # GPU训练项目 (新增副本) ⭐
    └── (所有文件副本, 363M)
```

## 关键配置文件

### 1. PyTorch CUDA配置

```python
# 在train_cuda.py中
import torch

# 自动检测GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# CUDA优化
torch.backends.cudnn.benchmark = True  # 启用自动优化
torch.cuda.empty_cache()                # 清空缓存

# 混合精度训练 (可选)
scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
```

### 2. DataLoader配置

```python
DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),  # GPU内存预固定
    persistent_workers=True                # 保持加载进程
)
```

### 3. 模型移到GPU

```python
model = DeepMDModel(...)
model.to(device)  # 移到CUDA设备

# 验证
print(next(model.parameters()).device)  # 应显示 cuda:0
```

### 4. 训练循环优化

```python
# 标准精度
optimizer.zero_grad()
pred = model(input)
loss = criterion(pred, target)
loss.backward()
optimizer.step()

# 混合精度
with torch.cuda.amp.autocast():
    pred = model(input)
    loss = criterion(pred, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## GPU管理

### NVIDIA GPU信息
```bash
# 查看GPU
nvidia-smi

# 输出示例:
# Name                         : NVIDIA L20-8Q
# Total Memory                 : 8352 MB (8.36 GB)
# Compute Capability          : 8.9
# Max Threads Per Block        : 1024
```

### 内存监控
```bash
# 实时监控 (每1秒更新)
watch -n 1 nvidia-smi

# 查看当前占用
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## 版本兼容性

```
Python          3.10.19      ✓
PyTorch         2.5.1        ✓
CUDA            12.1         ✓
cuDNN           9.01         ✓
numpy           2.2.6        ✓
scipy           1.15.3       ✓
```

## 性能配置

### 推荐设置

```bash
# 数据加载线程 (取决于CPU核心数)
--num-workers 4-8

# 启用混合精度 (强烈推荐)
--mixed-precision

# 批处理大小 (受显存限制)
batch_size: 1-8 (当前: 1)

# 梯度裁剪 (防止爆炸)
max_norm: 1.0
```

### 优化建议

1. **提高数据加载速度**
   ```bash
   --num-workers 8         # 增加加载线程
   pin_memory=True         # 已启用
   persistent_workers=True # 已启用
   ```

2. **降低显存占用**
   ```bash
   --mixed-precision       # 启用fp16混合精度
   # 可降低50%显存占用
   ```

3. **加速计算**
   ```bash
   torch.backends.cudnn.benchmark = True  # 自动调优
   ```

## 验证清单

- [x] CUDA驱动已安装 (nvidia-smi 可用)
- [x] CUDA工具包 12.1 已安装
- [x] cuDNN 9.01 已安装
- [x] PyTorch 2.5.1+cu121 已安装
- [x] GPU识别正常 (NVIDIA L20-8Q)
- [x] CUDA张量操作正常
- [x] 数据集加载正常 (1823 frames)
- [x] 模型创建正常 (3553参数)
- [x] 所有依赖包已安装
- [x] 所有脚本已验证

**验证状态**: 6/6 检查通过 ✅

## 并发训练配置

### CPU训练 (ai4m环境)
```bash
# 进程1
conda activate ai4m
python train_cpu.py \
  --config se_e2_a/input_torch_10k.json \
  --data-dir collect/O64H128 \
  --checkpoint-dir checkpoints \
  --export-dir exports
```

### GPU训练 (cuda_env环境)
```bash
# 进程2 (独立终端)
conda activate cuda_env
python train_cuda.py \
  --config se_e2_a/input_torch.json \
  --data-dir collect/O64H128 \
  --checkpoint-dir checkpoints_cuda \
  --export-dir exports_cuda \
  --num-workers 4 \
  --mixed-precision
```

**两个进程可同时运行，相互不影响**

## 问题诊断

### GPU检测失败

```bash
# 检查驱动
nvidia-smi

# 检查PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# 重新安装PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121 -U
```

### 显存不足

```bash
# 启用混合精度
--mixed-precision

# 或修改batch_size
# 在se_e2_a/input_torch.json中改为1
```

### 训练速度慢

```bash
# 检查GPU使用率
watch -n 1 nvidia-smi

# 增加workers
--num-workers 8

# 启用混合精度
--mixed-precision
```

## 后续维护

### 定期检查

```bash
# 每周检查一次
python verify_cuda_env.py

# 监控GPU健康
nvidia-smi --query-gpu=temperature.gpu --format=csv

# 显存泄漏检测
nvidia-smi --query-gpu=memory.used --format=csv
```

### 环境更新

```bash
# 更新PyTorch
pip install -U torch --index-url https://download.pytorch.org/whl/cu121

# 更新CUDA相关
pip install -U nvidia-cuda-runtime-cu121 nvidia-cudnn-cu12
```

## 预期性能

| 指标 | CPU (ai4m) | GPU (cuda_env) | 加速倍数 |
|------|-----------|----------------|---------|
| 设备 | Intel CPU | NVIDIA L20-8Q | - |
| 单步耗时 | ~6秒 | ~0.3-1秒 | 5-15x |
| 500步耗时 | ~50分钟 | ~5-10分钟 | 5-10x |
| 10000步耗时 | ~10-12小时 | ~1-2小时 | 5-10x |

---

**环境配置完成日期**: 2025-12-27  
**验证状态**: ✅ 全部通过  
**部署状态**: ✅ 可以启动
