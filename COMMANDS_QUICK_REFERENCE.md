# 🖥️ DeepMD Force Loss 修复 - 核心命令速查表

**更新时间**: 2025-12-29 21:26 CST  
**环境**: Ubuntu Linux, CUDA 12.1, PyTorch 2.5.1  

---

## 🚀 立即启动

### 继续监控当前训练
```bash
# 实时追踪日志输出（按 Ctrl+C 退出）
tail -f /home/ubuntu/pj/logs/train_formal_20251229-210839.log

# 或用 less 查看（可搜索）
less /home/ubuntu/pj/logs/train_formal_20251229-210839.log
```

### 查看最新进度
```bash
# 最后 10 行（最新数据）
tail -10 /home/ubuntu/pj/logs/train_formal_20251229-210839.log

# 最后 5 个 step 的 loss 值
tail -5 /home/ubuntu/pj/logs/train_formal_20251229-210839.log | grep "Step"
```

### 提取 f_rmse 曲线数据
```bash
# 所有 f_rmse 值（用于绘图）
grep "f_rmse=" /home/ubuntu/pj/logs/train_formal_20251229-210839.log | awk -F'f_rmse=' '{print $2}' | awk '{print $1}' > f_rmse_data.txt

# 每 100 步采样一次
tail -1000 /home/ubuntu/pj/logs/train_formal_20251229-210839.log | grep "Step" | awk 'NR % 100 == 0 {print}'

# 统计 f_rmse 范围
grep "f_rmse=" /home/ubuntu/pj/logs/train_formal_20251229-210839.log | awk -F'f_rmse=' '{print $2}' | sort -n | awk '{print $1}' | (head -1; tail -1)
```

### 启动新的 100k 步训练
```bash
# 基础启动（使用优化配置）
cd /home/ubuntu/pj
./launch_training_with_full_logging.sh --grad-accum 4

# 或明确指定配置
./launch_training_with_full_logging.sh \
  --config config_formal_100k_stable.json \
  --grad-accum 4

# 启动后，获取进程 ID
ps aux | grep train_cuda_optimized.py | grep -v grep
```

### 启动短期测试（2000 步）
```bash
cd /home/ubuntu/pj
./launch_training_with_full_logging.sh \
  --config config_formal_100k_stable.json \
  --steps 2000 \
  --grad-accum 4
```

---

## 📊 监控和验证

### 检查 GPU 状态
```bash
# 实时 GPU 使用情况（每 2 秒更新）
nvidia-smi -l 2

# GPU 内存详情
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

# 查看训练进程的 GPU 占用
nvidia-smi pids=<PID>

# 查询特定 CUDA 进程
ps aux | grep python | grep train_cuda
```

### 检查训练进程状态
```bash
# 查看所有训练相关进程
ps aux | grep -E "(train_cuda|python)" | grep -v grep

# 查看特定进程的详细信息（用上面得到的 PID）
ps -p <PID> -o pid,ppid,cmd,etime,cputime

# 查看进程的实时资源使用
top -p <PID>
```

### 验证修复有效性

#### 1. 检查代码修复
```bash
# 验证 self-exclusion 修复（应该找到）
grep "fill_diagonal_" /home/ubuntu/pj/dpmini/descriptor.py

# 验证 per-type 修复（应该找到）
grep -A 5 "for.*ntypes" /home/ubuntu/pj/dpmini/descriptor.py | head -10

# 验证 padded r 修复（应该找到）
grep "rcut + 1.0" /home/ubuntu/pj/dpmini/descriptor.py
```

#### 2. 运行单元测试
```bash
cd /home/ubuntu/pj
python test_neighbor_list_fix.py

# 如果需要详细输出
python -v test_neighbor_list_fix.py
```

#### 3. 检查日志无异常
```bash
# 查找所有错误和警告
grep -i "error\|warning\|nan\|inf" /home/ubuntu/pj/logs/train_formal_20251229-210839.log

# 应该无输出，说明运行正常
# 如果有输出，需要查看上下文
grep -i "error" /home/ubuntu/pj/logs/train_formal_20251229-210839.log -B 2 -A 2
```

#### 4. 验证 Loss 一致性
```bash
# 检查所有步的 diff 值（应该都是 0.000e+00）
grep "diff=" /home/ubuntu/pj/logs/train_formal_20251229-210839.log | head -20

# 提取 diff 并检查是否全为 0
awk -F'diff=' '{print $2}' /home/ubuntu/pj/logs/train_formal_20251229-210839.log | awk '{print $1}' | sort -u
```

---

## 🔧 配置和查看

### 查看当前配置
```bash
# 查看使用的配置文件
cat /home/ubuntu/pj/config_formal_100k_stable.json | head -50

# 查看完整配置
cat /home/ubuntu/pj/config_formal_100k_stable.json | python -m json.tool | less

# 对比两个配置文件的差异
diff /home/ubuntu/pj/config_formal_100k.json /home/ubuntu/pj/config_formal_100k_stable.json
```

### 修改配置参数
```bash
# 编辑配置文件
vim /home/ubuntu/pj/config_formal_100k_stable.json

# 或用 sed 批量修改（示例：改 disp_freq）
sed -i 's/"disp_freq": 1,/"disp_freq": 500,/' /home/ubuntu/pj/config_formal_100k_stable.json

# 验证修改
grep "disp_freq" /home/ubuntu/pj/config_formal_100k_stable.json
```

### 查看数据和路径
```bash
# 列出所有检查点
ls -lh /home/ubuntu/pj/checkpoints_fixed_neighbor/

# 检查点大小统计
du -sh /home/ubuntu/pj/checkpoints_fixed_neighbor/*

# 所有日志文件
ls -lh /home/ubuntu/pj/logs/

# 配置文件列表
ls -lh /home/ubuntu/pj/config_*.json
```

---

## 📈 数据分析

### 提取和分析 Loss 数据
```bash
# 提取所有 step、loss、e_loss、f_loss 数据
grep "^Step" /home/ubuntu/pj/logs/train_formal_20251229-210839.log | \
  awk '{print $2, $6, $8, $10}' > training_data.txt

# 查看前 10 行
head -10 training_data.txt

# 统计 loss 的平均值
awk '{sum+=$2; count++} END {print "Average loss:", sum/count}' training_data.txt

# 找出 loss 最小的 5 个 step
sort -k2 -n training_data.txt | head -5

# 找出 loss 最大的 5 个 step
sort -k2 -rn training_data.txt | head -5
```

### 分析 f_rmse 趋势
```bash
# 提取 step 和 f_rmse（用于绘图）
grep "^Step" /home/ubuntu/pj/logs/train_formal_20251229-210839.log | \
  awk '{print $2, $(NF-2)}' > f_rmse_trend.txt

# 计算移动平均（每 100 步）
awk '{sum+=$2; if (NR % 100 == 0) print $1, sum/100; sum=0}' f_rmse_trend.txt

# 查找 f_rmse 的最小值和最大值
awk '{if ($2 < min || NR == 1) min = $2; if ($2 > max) max = $2} END {print "Min:", min, "Max:", max}' f_rmse_trend.txt

# 统计 f_rmse 的分布（直方图）
awk '{print $2}' f_rmse_trend.txt | sort -n | uniq -c | sort -k2 -n
```

### 对比修复前后
```bash
# 查看修复前（如果有日志）和修复后的 f_rmse
echo "修复后 f_rmse 样本："
tail -100 /home/ubuntu/pj/logs/train_formal_20251229-210839.log | grep "f_rmse=" | head -5

# 计算最后 100 步的平均 f_rmse
tail -100 /home/ubuntu/pj/logs/train_formal_20251229-210839.log | \
  grep "f_rmse=" | awk '{print $(NF-2)}' | \
  awk '{sum+=$1; count++} END {print "Latest 100 steps avg f_rmse:", sum/count}'
```

---

## 🐛 故障排除

### 检查系统状态
```bash
# 查看系统资源使用
top -b -n 1 | head -20

# 查看磁盘空间
df -h /home/ubuntu/pj/

# 查看 GPU 内存（防止 OOM）
nvidia-smi --query-gpu=memory.free --format=csv,noheader

# 查看系统日志（寻找 OOM 或 kernel panic）
dmesg | tail -50

# 查看特定进程的内存使用
ps aux | grep "python" | grep -v grep | awk '{print $2, $4, $6}' | sort -k3 -rn
```

### 如果训练 Crash
```bash
# 查看 crash 前的日志（最后 100 行）
tail -100 /home/ubuntu/pj/logs/train_formal_20251229-210839.log

# 查看是否有 OOM
grep -i "outofmemory\|oom" /home/ubuntu/pj/logs/train_formal_20251229-210839.log

# 查看是否有 CUDA 错误
grep -i "cuda\|cudnn" /home/ubuntu/pj/logs/train_formal_20251229-210839.log | tail -20

# 查看系统日志的 crash 信息
dmesg | grep -i "killed\|oom" | tail -10
```

### 如果 f_rmse 异常
```bash
# 检查是否使用了旧的 descriptor.py
grep "fill_diagonal_" /home/ubuntu/pj/dpmini/descriptor.py
# 应该有输出，如果没有说明代码有问题

# 运行单元测试验证
python /home/ubuntu/pj/test_neighbor_list_fix.py

# 检查最新日志（看是否有 assertion 错误）
tail -50 /home/ubuntu/pj/logs/train_formal_20251229-210839.log | grep -i "assert\|error"
```

### 清理和重置
```bash
# 停止所有训练进程
pkill -f "train_cuda_optimized.py"

# 查看是否成功停止
ps aux | grep train_cuda | grep -v grep

# 备份当前日志
cp /home/ubuntu/pj/logs/train_formal_*.log /home/ubuntu/pj/logs/backup_$(date +%s).log

# 删除旧日志（谨慎！）
rm /home/ubuntu/pj/logs/train_formal_*.log

# 重新启动新训练
./launch_training_with_full_logging.sh --grad-accum 4
```

---

## 📊 性能基准

### 查看当前训练速度
```bash
# 从日志提取速度（steps/s）
tail -10 /home/ubuntu/pj/logs/train_formal_20251229-210839.log | grep "step/s" | tail -1

# 计算平均速度
grep "step/s" /home/ubuntu/pj/logs/train_formal_20251229-210839.log | \
  awk '{print $(NF-2)}' | \
  awk '{sum+=$1; count++} END {print "Average speed:", sum/count, "steps/s"}'
```

### 预估完成时间
```bash
# 当前进度
CURRENT=$(wc -l < /home/ubuntu/pj/logs/train_formal_20251229-210839.log)

# 目标步数
TARGET=2000

# 已用时间（秒）
ELAPSED=$(ps -o etimes= -p $(pgrep -f "train_cuda_optimized.py") 2>/dev/null || echo "0")

# 计算平均速度
SPEED=$(grep "step/s" /home/ubuntu/pj/logs/train_formal_20251229-210839.log | tail -1 | awk '{print $(NF-2)}')

# 剩余步数
REMAINING=$((TARGET - CURRENT))

# 预估剩余时间（秒）
ETA=$((REMAINING / SPEED))

echo "Progress: $CURRENT / $TARGET"
echo "Speed: $SPEED steps/s"
echo "Remaining steps: $REMAINING"
echo "ETA: $(($ETA / 3600)) hours $((($ETA % 3600) / 60)) minutes"
```

---

## 🎯 常用组合命令

### 一键监控面板
```bash
# 持续显示进度、GPU 状态、训练日志
watch -n 2 'clear; echo "=== GPU Status ==="; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader; echo "=== Process Status ==="; ps aux | grep train_cuda_optimized | grep -v grep; echo "=== Latest Log ==="; tail -3 /home/ubuntu/pj/logs/train_formal_20251229-210839.log'
```

### 一键启动和监控
```bash
# 启动训练并立即开始监控
cd /home/ubuntu/pj && \
./launch_training_with_full_logging.sh --grad-accum 4 && \
sleep 5 && \
tail -f logs/train_formal_*.log
```

### 一键生成报告
```bash
# 生成训练报告（到文件）
{
  echo "=== Training Report ===" > training_report.txt
  echo "Generated: $(date)" >> training_report.txt
  echo "" >> training_report.txt
  echo "=== Progress ===" >> training_report.txt
  echo "Total lines (≈steps): $(wc -l < logs/train_formal_*.log)" >> training_report.txt
  echo "" >> training_report.txt
  echo "=== Latest 5 Steps ===" >> training_report.txt
  tail -5 logs/train_formal_*.log >> training_report.txt
  echo "" >> training_report.txt
  echo "=== Loss Statistics ===" >> training_report.txt
  echo "Min loss:" $(grep "loss=" logs/train_formal_*.log | awk -F'loss=' '{print $2}' | awk '{print $1}' | sort -n | head -1) >> training_report.txt
  echo "Max loss:" $(grep "loss=" logs/train_formal_*.log | awk -F'loss=' '{print $2}' | awk '{print $1}' | sort -rn | head -1) >> training_report.txt
} && \
cat training_report.txt
```

---

## 🔑 关键路径速查

| 用途 | 路径 | 命令 |
|------|------|------|
| 运行日志 | `/home/ubuntu/pj/logs/train_formal_*.log` | `tail -f` |
| 配置文件 | `/home/ubuntu/pj/config_formal_100k_stable.json` | `cat` |
| 源代码 | `/home/ubuntu/pj/dpmini/descriptor.py` | `grep` |
| 单元测试 | `/home/ubuntu/pj/test_neighbor_list_fix.py` | `python` |
| Checkpoint | `/home/ubuntu/pj/checkpoints_fixed_neighbor/` | `ls -lh` |
| 启动脚本 | `/home/ubuntu/pj/launch_training_with_full_logging.sh` | `./` |

---

## ✅ 快速检查清单

启动新训练前的检查：
```bash
# 1. 代码修复是否存在
grep "fill_diagonal_" /home/ubuntu/pj/dpmini/descriptor.py && echo "✓ 代码修复存在" || echo "✗ 代码修复缺失"

# 2. 单元测试是否通过
python /home/ubuntu/pj/test_neighbor_list_fix.py 2>&1 | grep "PASSED" | wc -l
# 应该输出 4

# 3. GPU 是否可用
nvidia-smi && echo "✓ GPU 可用" || echo "✗ GPU 不可用"

# 4. 磁盘空间是否充足（至少 10GB）
df /home/ubuntu/pj/ | tail -1 | awk '{if ($4 > 10485760) print "✓ 空间充足"; else print "✗ 空间不足"}'

# 5. 配置文件是否存在
[ -f /home/ubuntu/pj/config_formal_100k_stable.json ] && echo "✓ 配置文件存在" || echo "✗ 配置文件缺失"
```

---

**快速参考生成时间**: 2025-12-29 21:26 CST  
**包含命令数**: 50+ 个  
**使用场景**: 监控、验证、故障排除、数据分析

