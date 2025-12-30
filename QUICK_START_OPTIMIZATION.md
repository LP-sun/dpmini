# 性能优化 - 快速执行指南

## 🎯 立即可执行的命令

### 1️⃣ 查看当前运行的训练进程
```bash
./list_training_processes.sh
```

### 2️⃣ 实时监控资源使用 (CPU + GPU)
```bash
./monitor_resources.sh
```
按 Ctrl+C 停止，结果保存在 `monitor_YYYYMMDD_HHMMSS.csv`

### 3️⃣ GPU详细监控
```bash
./gpu_monitor.sh
```

### 4️⃣ 分析邻居统计并优化sel参数
```bash
python3 analyze_neighbor_statistics.py --data examples/data/water --rcut 6.0
```

### 5️⃣ 使用 examples/data 快速测试优化训练
```bash
# 默认配置 (batch=12, workers=4)
python3 train_example_optimized.py

# 自定义配置
python3 train_example_optimized.py \
    --batch-size 16 \
    --num-workers 4 \
    --num-epochs 10 \
    --learning-rate 0.001
```

### 6️⃣ 运行 A/B 性能基准测试
```bash
./run_ab_benchmark.sh
```
这将自动测试多个配置，找出最优组合（约需10-15分钟）

---

## 📊 快速诊断步骤

### 步骤1: 检查当前状态
```bash
# 查看训练进程
./list_training_processes.sh

# 同时监控1分钟
timeout 60 ./monitor_resources.sh
```

### 步骤2: 识别瓶颈
**GPU利用率低 (<30%) + CPU高 (>600%)**
→ batch_size太小，立即增大到8-16

**GPU利用率中等 (30-60%)**  
→ 数据加载慢，增加num_workers到4-8

**GPU利用率高 (>70%)**
→ 已经很好，考虑降低验证频率提升吞吐

### 步骤3: 应用优化
编辑配置文件 `se_e2_a/input_torch.json`:

```json
{
  "training": {
    "training_data": {
      "batch_size": 12      // 从1或8改为12-16
    },
    "validation_data": {
      "batch_size": 12,     // 匹配训练batch
      "numb_btch": 1        // 从3改为1
    },
    "disp_freq": 100,       // 从10改为100
    "numb_steps": 100000
  }
}
```

环境变量:
```bash
export NUM_WORKERS=4
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

### 步骤4: 重新启动训练
```bash
# 如果需要停止当前训练
kill <PID>  # 从 list_training_processes.sh 获取PID

# 启动优化后的训练
python3 train_cuda.py \
    --config se_e2_a/input_torch.json \
    --num-workers 4
```

---

## 🔥 最关键的3个优化

### ✅ 1. 增大 batch_size (最重要!)
```json
"batch_size": 12  // 或 16，从1改起
```
**预期提升**: 3-5倍

### ✅ 2. 降低验证频率
```json
"disp_freq": 100,    // 从10改为100
"numb_btch": 1       // 从3改为1
```
**预期提升**: 10-20%

### ✅ 3. 优化sel参数
```bash
python3 analyze_neighbor_statistics.py
# 根据输出调整 descriptor.sel
```
**预期提升**: 15-25%

---

## 📈 预期性能改进

| 阶段 | steps/s | 100k steps完成时间 | 改进倍数 |
|------|---------|------------------|---------|
| 当前 (batch=1) | 1.3 | 21h | 1.0x |
| batch=8 | 4.5 | 6h | 3.5x |
| batch=12 | 6.0 | 4.6h | 4.6x |
| batch=12 + 全优化 | 7-8 | 3.5-4h | **5-6x** |

---

## 🛠️ 故障排查

### 问题: "CUDA out of memory"
```bash
# 降低batch_size
python3 train_example_optimized.py --batch-size 8
```

### 问题: "Too many open files"
```bash
# 降低num_workers
python3 train_example_optimized.py --num-workers 2
```

### 问题: CPU争用严重
```bash
# 减少OMP线程
export OMP_NUM_THREADS=4
export NUM_WORKERS=2
```

---

## 📝 完整示例

### 示例1: 使用 examples/data 快速验证
```bash
# 1. 设置环境
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# 2. 运行优化训练
python3 train_example_optimized.py \
    --batch-size 12 \
    --num-workers 4 \
    --num-epochs 20

# 3. 同时监控 (新终端)
./monitor_resources.sh
```

### 示例2: 完整A/B测试
```bash
# 一键运行所有测试配置
./run_ab_benchmark.sh

# 查看结果
cat benchmark_results_*/results.csv
```

### 示例3: 生产训练 (完整数据集)
```bash
# 1. 先用小数据测试
python3 train_example_optimized.py --num-epochs 5

# 2. 确认没问题后，使用完整数据
python3 train_cuda.py \
    --config se_e2_a/input_torch.json \
    --num-workers 4 \
    > training.log 2>&1 &

# 3. 监控
./monitor_resources.sh
```

---

## 🎓 参数说明速查

| 参数 | 含义 | 推荐值 | 影响 |
|------|------|--------|------|
| batch_size | 每次训练的样本数 | 8-16 | GPU利用率 ↑ |
| num_workers | 数据加载进程数 | 4 | 数据加载速度 ↑ |
| OMP_NUM_THREADS | PyTorch计算线程 | 8 | CPU计算效率 |
| disp_freq | 验证频率(步) | 100 | 训练吞吐 ↑ |
| numb_btch | 验证批次数 | 1 | 验证速度 ↑ |
| sel | 邻居数上限 | 统计得出 | 内存和速度 |

---

**下一步**: 运行 `./list_training_processes.sh` 检查当前状态，然后执行上述优化！
