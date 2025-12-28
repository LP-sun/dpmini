# ⚠️ 新设备训练速度慢？问题排查指南

## 问题症状
在新设备上运行优化脚本，但速度很慢（远低于预期的14 steps/s）

## 🔴 最常见原因：使用了错误的Python环境

### 问题诊断

**症状：**
- 速度 < 3 steps/s（应该是 ~14 steps/s）
- GPU利用率低（<30%）
- 或者直接报错找不到模块

**根本原因：**
使用了系统默认的 `python3`，而不是安装了PyTorch的conda环境

### 快速诊断（3步）

```bash
# 步骤1：运行环境检查
./check_environment.sh

# 如果显示 "❌ PyTorch NOT installed"，说明环境不对

# 步骤2：查看当前使用的Python
which python
# 如果显示 /usr/bin/python3，这就是问题所在！

# 步骤3：找到正确的Python环境
conda env list
# 找到包含PyTorch的环境，比如 ai4m
```

## ✅ 解决方案

### 方案1：激活正确的conda环境（推荐）

```bash
# 1. 激活包含PyTorch的环境
conda activate ai4m  # 替换为你的环境名

# 2. 验证环境正确
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# 应该显示: CUDA: True

# 3. 运行训练
./run_training_optimized.sh --batch-size 64
```

### 方案2：设置PYTHON环境变量

```bash
# 1. 找到正确的Python路径
conda env list
# 假设环境在 /home/ubuntu/miniforge3/envs/ai4m

# 2. 设置环境变量
export PYTHON="/home/ubuntu/miniforge3/envs/ai4m/bin/python"

# 3. 运行训练
./run_training_optimized.sh --batch-size 64
```

### 方案3：修改脚本默认Python路径

编辑 `run_training_optimized.sh`，在文件开头修改：

```bash
# 修改这一行（大约在第12行）
PYTHON="${PYTHON:-/home/ubuntu/miniforge3/envs/ai4m/bin/python}"
# 改为你的实际Python路径
```

## 📋 完整排查清单

使用诊断工具进行全面检查：

```bash
# 运行完整诊断（生成详细报告）
./diagnose_training_performance.sh
```

诊断脚本会检查：
- ✓ GPU硬件和驱动
- ✓ Python环境和PyTorch安装
- ✓ CUDA是否可用
- ✓ 数据文件完整性
- ✓ 配置文件正确性

## 🎯 验证环境正确的标准

运行这些命令，应该看到以下输出：

```bash
# 1. PyTorch已安装
python -c "import torch; print(torch.__version__)"
# 输出: 2.x.x

# 2. CUDA可用
python -c "import torch; print(torch.cuda.is_available())"
# 输出: True

# 3. GPU可见
nvidia-smi
# 应该看到你的GPU（如 NVIDIA L20-8Q）

# 4. 数据存在
ls collect/data0/set.000/
# 应该看到: coord.npy, box.npy, energy.npy, force.npy
```

## 🚀 环境正确后的预期性能

| 指标 | 预期值 |
|------|--------|
| 训练速度 | ~14 steps/s |
| GPU利用率 | 60-80% |
| 显存使用 | ~200 MB (batch_size=64) |

如果环境正确但速度仍然慢：

```bash
# 运行性能测试
./test_batch_sizes.sh

# 或快速测试
python quick_bottleneck_test.py
```

## 📦 新设备部署完整流程

```bash
# ===== 在源设备 =====
# 1. 打包所有文件
./create_deployment_package.sh

# 2. 传输到目标设备
scp deepmd_optimized_training_*.tar.gz user@new-host:/path/

# ===== 在目标设备 =====
# 3. 解压
tar -xzf deepmd_optimized_training_*.tar.gz
cd deepmd_optimized_package

# 4. 激活正确的conda环境（关键！）
conda activate ai4m  # 或你的环境名

# 5. 运行环境检查
./check_environment.sh

# 6. 如果检查通过，启动训练
./run_training_optimized.sh --batch-size 64

# 7. 监控训练
tail -f training_optimized.log
```

## 🔧 其他可能的问题

### 问题2：数据未正确复制

```bash
# 检查数据大小
du -sh collect/data0/set.000/
# 应该有几十MB

# 检查文件数量
ls collect/data0/set.000/
# 应该有 coord.npy, box.npy, energy.npy, force.npy
```

### 问题3：dpmini模块未复制

```bash
# 检查模块
ls -la dpmini/
# 应该看到 __init__.py, model.py, descriptor.py, data.py

# 测试导入
python -c "from dpmini import DeepMDModel"
```

### 问题4：CUDA驱动问题

```bash
# 检查CUDA驱动
nvidia-smi
# 应该显示GPU信息

# 检查PyTorch CUDA版本
python -c "import torch; print(torch.version.cuda)"
# 应该显示 CUDA版本（如 11.8）
```

## 💡 最佳实践

1. **永远先运行检查脚本**
   ```bash
   ./check_environment.sh  # 快速检查
   # 或
   ./diagnose_training_performance.sh  # 完整诊断
   ```

2. **确保在正确的环境中**
   ```bash
   # 在shell启动文件中添加
   echo 'conda activate ai4m' >> ~/.bashrc
   ```

3. **使用绝对路径**
   ```bash
   # 在脚本中使用绝对路径避免混淆
   export PYTHON="/home/ubuntu/miniforge3/envs/ai4m/bin/python"
   ```

## 📞 获取帮助

如果问题仍未解决，运行完整诊断并查看日志：

```bash
./diagnose_training_performance.sh > diagnosis.log 2>&1

# 查看诊断结果
less diagnosis.log

# 或发送给技术支持
```

诊断日志包含：
- 硬件信息
- Python环境详情
- CUDA状态
- 数据完整性
- 性能测试结果

---

**关键提醒：99%的"速度慢"问题都是因为使用了错误的Python环境！**

请先运行：
```bash
./check_environment.sh
```
