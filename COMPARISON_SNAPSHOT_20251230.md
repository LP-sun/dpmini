# V3 vs 原始训练对比（并发运行，12月30日 03:09 启动）

## 系统状态
- GPU: NVIDIA L20-8Q（8GB）
- GPU利用率: 93%（两个训练并行争抢）
- 显存占用: 1129/8192 MiB

## 正在运行的进程
- **原始训练** (PID: 138895)
  - 脚本: `train_cuda_optimized.py`
  - 配置: `config_formal_100k_fixed.json`
  - 参数: 梯度累积=4，batch_size=1（虚拟），workers=4
  - 启动时间: 2025-12-30 01:54:20（已运行~1.25小时）
  
- **V3优化训练** (PID: 147981 主+4个worker: 148023,148024,148025,148047)
  - 脚本: `train_optimized_dataloader_v3.py`
  - 配置: `config_formal_100k_optimized.json`
  - 参数: 梯度累积=4，batch_size=4（真实），workers=4，persistent_workers，pin_memory，non_blocking
  - 启动时间: 2025-12-30 03:09（刚启动）

## 日志文件
- 原始: `logs/train_fixed_20251230-015420.log`
- V3: `logs/run_optimized_v3_20251230-030858.log`

## 早期吞吐对比（热身阶段）
从日志摘取Step 1-5的速度：

**V3** (Step 1-5)
```
Step 1: 0.96 step/s  ETA: 29.0h
Step 2: 1.18 step/s  ETA: 23.5h
Step 3: 1.29 step/s  ETA: 21.6h
Step 4: 1.32 step/s  ETA: 21.1h (update=1, first optimizer.step)
Step 5: 1.36 step/s  ETA: 20.4h (update=1)
```

**原始** (早期日志已不可见，参考后期稳态Step 500+)
```
Step 500: 2.13 step/s  ETA: 12.9h
Step 5500: 2.40 step/s ETA: 10.9h
```

## 观察
1. **热身延迟**: V3早期速度低于原始，可能因为:
   - DataLoader worker初始化（persistent_workers会逐步优化）
   - 累积窗口内的前4步未执行optimizer.step，故梯度未清零（这是设计，无妨）
   
2. **并行争抢**: GPU利用率维持93%，但两个训练分享带宽，各自吞吐低于单独运行
   - 原始单独运行时稳态可达2.1-2.4 step/s
   - V3现在约1.3 step/s（受原始占用影响）
   
3. **待观察**:
   - V3进入稳态后（50-100步后）的真实速度提升
   - 累积末尾的第一个optimizer.step（Step 4）是否触发无问题

## 后续计划
- 继续观察V3的Step 100、200、500时的速度与ETA收敛
- 让原始训练完成或停止，单独运行V3 500-1000步以获得公平对比
- 最终提交对比报告

## 监控命令
```bash
# 实时查看两个日志
tail -f logs/train_fixed_20251230-015420.log &
tail -f logs/run_optimized_v3_20251230-030858.log &

# GPU和进程
watch -n 2 nvidia-smi

# 查询进程树
pgrep -f "train_cuda_optimized|train_optimized_dataloader_v3"
```
