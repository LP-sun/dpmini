# 🚀 优化训练快速参考

## ⚠️ 首次使用必读

**在新设备上，先检查环境（避免速度慢问题）：**
```bash
./check_environment.sh
# 如果显示 "PyTorch NOT installed"，需要先激活正确环境：
conda activate ai4m  # 或你的环境名
```

## 一键启动（推荐配置）

```bash
./run_training_optimized.sh
```

默认使用：
- ✅ batch_size=64（GPU利用率提升4倍）
- ✅ num_workers=0（数据加载最快）
- ⚡ 预期速度：~14 steps/s（6倍提速）
- ⏱️ 预期时间：~2小时（原11.8小时）

---

## 其他常用配置

### 保守配置（小显存GPU）
```bash
./run_training_optimized.sh --batch-size 32
```

### 激进配置（启用混合精度）
```bash
./run_training_optimized.sh --batch-size 64 --mixed-precision
```

### 测试配置（快速验证）
```bash
# 只跑100步测试性能
./run_training_optimized.sh --batch-size 64 --config config_minimal.json
```

---

## 监控命令

```bash
# 实时日志
tail -f training_optimized.log

# 查看速度
grep 'speed' training_optimized.log | tail -10

# GPU监控
watch -n 1 nvidia-smi

# 训练进度
grep 'Step' training_optimized.log | tail -5
```

---

## 部署到其他设备

### 方式1：打包传输
```bash
# 在当前设备
./create_deployment_package.sh

# 传输到目标设备
scp deepmd_optimized_training_*.tar.gz user@remote:/path/

# 在目标设备解压并运行
tar -xzf deepmd_optimized_training_*.tar.gz
cd deepmd_optimized_package

# ⚠️ 重要：先激活正确的Python环境
conda activate ai4m  # 或你的环境名

# 检查环境
./check_environment.sh

# 如果检查通过，启动训练
./quick_start.sh
```

### 方式2：直接复制文件
```bash
# 复制以下文件到目标设备：
scp run_training_optimized.sh user@remote:/path/
scp train_deepmd_pytorch_cuda.py user@remote:/path/
scp -r se_e2_a dpmini collect user@remote:/path/

# 在目标设备运行
ssh user@remote
cd /path
./run_training_optimized.sh
```

---

## 性能对比

| 配置 | 速度 | GPU利用率 | 训练时间 |
|------|------|-----------|----------|
| 原配置 (bs=8, nw=4) | 2.3 steps/s | 24% | 11.8小时 |
| **优化配置 (bs=64, nw=0)** | **~14 steps/s** | **70%** | **~2小时** |
| 提升 | **6倍** | **3倍** | **6倍** |

---

## 故障排查

### 速度很慢（最常见）
```bash
# 原因：使用了错误的Python环境
# 解决：
conda activate ai4m  # 激活正确环境
./check_environment.sh  # 验证环境

# 详细排查指南：
# 参见 TROUBLESHOOTING_NEW_DEVICE.md
```

### 显存不足
```bash
# 减小batch size
./run_training_optimized.sh --batch-size 32
# 或更小
./run_training_optimized.sh --batch-size 16
```

### 检查环境
```bash
# 快速检查
./check_environment.sh

# 完check_environment.sh` - 环境快速检查 ⚡
- `diagnose_training_performance.sh` - 完整性能诊断
- `README_OPTIMIZED_TRAINING.md` - 详细使用文档
- `TROUBLESHOOTING_NEW_DEVICE.md` - 新设备问题排查 🔧
./diagnose_training_performance.sh
```

### 停止训练
```bash
kill $(cat training_optimized.pid)
```

---

## 文件说明

- `run_training_optimized.sh` - 优化版启动脚本 ⭐
- `README_OPTIMIZED_TRAINING.md` - 详细使用文档
- `BOTTLENECK_ANALYSIS.md` - 性能分析报告
- `create_deployment_package.sh` - 打包部署脚本
- `test_batch_sizes.sh` - 批量测试工具
- `QUICK_REFERENCE.md` - 本文件

---

**完整文档**：参见 [README_OPTIMIZED_TRAINING.md](README_OPTIMIZED_TRAINING.md)  
**性能分析**：参见 [BOTTLENECK_ANALYSIS.md](BOTTLENECK_ANALYSIS.md)
