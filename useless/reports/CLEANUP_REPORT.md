# 代码清理与整理报告

日期: 2025-12-28

## 摘要
- 已将明显冗余/不再使用的代码备份到: `useless/`
  - `useless/deepmd-kit/`（上游 TensorFlow 版 DeepMD-kit 源与文档，当前 PyTorch 实现未直接依赖）
  - `useless/don't use/`（原型/旧脚本：`Ground Truth.py`, `Mini-DeepMD.py`）
- 保留并标记为核心的代码：
  - `dpmini/`（核心实现：`descriptor.py`, `model.py`, `data.py`, `__init__.py`）
  - 训练脚本：`train_cpu.py`, `train_cuda.py`, `train_cuda_optimized.py`, `train_example_optimized.py`
  - 推理与工作流：`inference.py`, `md_simulation.py`, `compute_rdf.py`, `interpret_rdf.py`, `analyze_trajectory.py`
  - 基准与诊断：`quick_bottleneck_test.py`, `profile_training_bottleneck.py`, `summarize_benchmarks.py`, `diagnose_performance.py`
  - 测试：`tests/test_descriptor.py`, `test_model.py`, `test_model_on_data.py`, `test_forward.py`, `test_neighbor.py`, `test_one_step.py`, `test_smoke.py`
  - 配置：`se_e2_a/`（训练配置 JSON）
  - 常用脚本：`run_training*.sh`, `run_md_rdf_analysis.sh`, `quickstart.sh`, `cuda_quickstart.sh`, `check_environment.sh` 等

## 识别原则
- 被训练/推理/工作流脚本显式调用的 Python 文件：保留
- 被测试覆盖（`tests/` 或 `test_*.py` 引用）的模块：保留
- 仅作为上游参考且未在当前脚本中被调用的旧版或第三方源：迁移至 `useless/`
- 含有“暂不使用”、“原型”等标注或目录名（如 `don't use/`）：迁移至 `useless/`

## 文件命名整理建议（未执行，待确认）
为提升可读性与一致性，建议将训练脚本统一为更短、规范化命名，并批量更新引用：
- `train_cpu.py` → `train_cpu.py`
- `train_cuda.py` → `train_cuda.py`
- `train_cuda_optimized.py` → `train_cuda_optimized.py`

说明：以上变更涉及多个 `.sh` 与文档引用（如 `run_training*.sh`, `QUICK_TEST_GUIDE.sh` 等）。若确认执行，我将：
1) 重命名文件
2) 全局更新脚本与文档中的引用
3) 运行最小验证（导入 `dpmini`、执行 `tests/test_descriptor.py`）

## 下一步
- 如需继续深入清理：
  - 审核 `examples/` 与 `tools/` 中是否存在重复/未用脚本，并按上述原则迁移
  - 压缩/清理大型日志与中间产物目录（仅数据，不属于“代码”），如 `checkpoints_*/`, `exports_*/`, `*.log`, `*.pid`（如需）
- 如确认执行“训练脚本重命名”，请回复“执行重命名”，我将批量更新并验证。

