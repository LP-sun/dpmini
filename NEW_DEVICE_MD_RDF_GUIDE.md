# 新设备 MD + RDF 分析快速指南（Copilot 阅读版）

本指南面向已训练出模型、尚未安装 DeepMD 命令行工具（dp）的新设备。只需 Python + 依赖即可运行 MD 模拟、计算 RDF 并完成结构解读与轨迹统计。

## 包含内容与脚本
- MD 模拟: md_simulation.py
- RDF 计算: compute_rdf.py
- RDF 解读: interpret_rdf.py
- 轨迹统计: analyze_trajectory.py
- 一键工作流: run_md_rdf_analysis.sh
- 依赖清单: requirements.txt

可选辅助：test_model.py、test_smoke.py（快速验证环境），示例数据 collect/data0（结构参考）。

## 先决条件
- Python 3.10+（推荐使用 conda 虚拟环境 deepmd）
- GPU/CUDA（可选，加速用；CPU 也可运行）
- 已训练好的模型权重 .pth（例如 exports_cuda_opt/model_cuda_YYYYMMDD-HHMMSS.pth）

## 环境准备
```bash
# 创建并激活环境
conda create -n deepmd python=3.10 -y
conda activate deepmd

# 进入工程目录并安装依赖
cd /path/to/pj
pip install -r requirements.txt
```

## 一键运行工作流（MD → RDF → 解读 → 轨迹统计）
```bash
# 参数：<模型pth> <数据目录> <步数> <输出前缀>
./run_md_rdf_analysis.sh exports_cuda_opt/model_cuda_YYYYMMDD-HHMMSS.pth collect/data0 1000 my_analysis
```
输出产物：
- 轨迹：my_analysis_md.npz
- RDF 图：my_analysis_rdf_plot.png
- RDF 数据：my_analysis_rdf_data.txt
- 轨迹统计：my_analysis_summary.json

## 分步运行（便于排查与自定义）
1) MD 模拟（生成 my_md.npz）
```bash
conda run -n deepmd python -u md_simulation.py \
  --model exports_cuda_opt/model_cuda_YYYYMMDD-HHMMSS.pth \
  --system collect/data0 \
  --steps 1000 \
  --dt 0.001 \
  --output my_md
```
2) 计算 RDF（生成 PNG + TXT）
```bash
conda run -n deepmd python -u compute_rdf.py \
  my_md.npz \
  --pairs O-O O-H H-H \
  --rmax 8.0 \
  --nbins 200 \
  --output my_rdf
```
3) 解读 RDF 数据（峰位与配位数等）
```bash
conda run -n deepmd python -u interpret_rdf.py my_rdf_data.txt
```
4) 轨迹统计汇总（能量、温度等）
```bash
conda run -n deepmd python -u analyze_trajectory.py \
  my_md.npz \
  --save my_summary.json
```

## 日志与监控（可选）
- 长跑任务（nohup后台）
```bash
nohup ./run_md_rdf_analysis.sh exports_cuda_opt/model_cuda_YYYYMMDD-HHMMSS.pth collect/data0 5000 long_run > long_run.log 2>&1 & echo $!
# 监控
tail -f long_run.log
ps -p <PID> -o pid,etime,cmd
nvidia-smi
```
- 训练日志（如在同机继续训练）：training_vectorized.log（stdout/stderr重定向）

## 常见问题排查
- 无需安装 dp 命令行工具：上述分析流程纯 Python 脚本即可运行。
- 日志长期无进度：可能在 GPU 预热或 I/O；用 nvidia-smi/tail -f 观察，避免多进程争抢同一 GPU。
- GPU 未被使用：确认 PyTorch 能检测到 CUDA（日志中会打印），或强制使用 CPU 运行作对比。
- 输出文件找不到：检查输出前缀与当前工作目录，脚本会在当前目录生成产物。

## 参考与下一步
- RDF 峰位与液态水参考：interpret_rdf.py 输出中包含峰值与配位数评估
- 扩展分析（可后续添加）：角度分布函数（ADF）、氢键统计、MSD 与扩散系数、长时间模拟

---
如需在新设备上打包最小复现（仅上述脚本与 README），可进一步提供打包脚本与说明文件；也可将示例数据结构复制以快速验证。
