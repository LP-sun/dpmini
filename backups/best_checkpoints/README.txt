================================================================================
最优 Checkpoint 备份目录
================================================================================

【备份时间】2025-12-30 18:31:55

【最优模型】
  文件: best_model_v3_step80000_20251230_183155.pt
  来源: checkpoints_optimized_v3/model_step80000.pt
  训练: V3 优化版本 (DataLoader + 4 workers)

【性能指标】(基于 300 帧验证集)
  F_RMSE:       0.715 eV/Å (比原始训练低 12.4%)
  F_RMSE_tail:  1.288 eV/Å (比原始训练低 24.3%)
  E_RMSE:       61.22 eV/atom
  训练步数:     80,000 / 100,000
  收敛状态:     已收敛 (改进率 0.89% < 1% 阈值)

【为何选择此 checkpoint】
  1. 力预测性能最优 (F_RMSE 全场景最低)
  2. 困难样本表现优异 (F_RMSE_tail 最低)
  3. 已达收敛，继续训练无显著改进
  4. 比原始训练 Step 90k 性能提升 12-24%

【使用方法】
  # 加载模型
  import torch
  checkpoint = torch.load('best_model_v3_step80000_20251230_183155.pt')
  model.load_state_dict(checkpoint['model'])
  
  # 或用于推理
  python inference.py --checkpoint backups/best_checkpoints/best_model_v3_step80000_20251230_183155.pt

【训练配置】
  config: config_formal_100k_optimized.json
  data: collect/O64H128
  batch_size: 4
  workers: 4
  gradient_accumulation: 4
  force_loss: mse

【相关文档】
  - 评估报告: VALIDATION_CHECKPOINT_ANALYSIS.md
  - 系统指南: VALIDATION_SYSTEM_GUIDE.md
  - 完整分析: VALIDATION_IMPLEMENTATION_REPORT.md

【停止训练决策】
  基于固定验证集的收敛性分析，两个训练都已达到收敛：
  - 原始训练 (Step 93k): F_RMSE 改进 0.10% < 1%
  - V3 训练 (Step 79k):  F_RMSE 改进 0.89% < 1%
  
  决定停止训练以节省计算资源（~5 小时），使用已达最优性能的 checkpoint。

================================================================================
备份创建: 2025-12-30
最后更新: 2025-12-30
================================================================================
