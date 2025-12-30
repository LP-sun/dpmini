#!/bin/bash
# CUDA GPU训练 - 快速参考卡
# 保存此文件为 README_CUDA_QUICK_START.md

cat << 'EOF'

╔══════════════════════════════════════════════════════════════════════════════╗
║                    DeepMD CUDA GPU 加速训练 - 快速参考卡                     ║
║                         部署完成：2025-12-27                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

📍 当前状态
═══════════════════════════════════════════════════════════════════════════════

  原项目 (/home/ubuntu/pj):
    ✓ CPU训练持续运行中 (进程ID: 914161)
    ✓ ai4m环境保持原样
    ✓ 所有原文件完全保留

  新环境 (/home/ubuntu/pj_cuda):
    ✓ CUDA环境创建完成 (cuda_env)
    ✓ PyTorch 2.5.1+cu121已安装
    ✓ GPU: NVIDIA L20-8Q (8.36 GB)
    ✓ 所有文件复制完毕 (363M)

  预期加速倍数: 5-15x ⚡
  预期训练时间: 1-2小时 (相比CPU的10-12小时)


🚀 1分钟快速启动
═══════════════════════════════════════════════════════════════════════════════

  conda activate cuda_env
  cd /home/ubuntu/pj
  bash cuda_quickstart.sh


📋 详细命令
═══════════════════════════════════════════════════════════════════════════════

快速测试 (500步, 预计30秒-1分钟):
─────────────────────────────────────
  conda activate cuda_env
  cd /home/ubuntu/pj
  python train_cuda.py \
    --config se_e2_a/input_torch_quick_test.json \
    --data-dir collect/O64H128 \
    --num-workers 4

完整训练 (10000步, 预计1-2小时):
──────────────────────────────────
  conda activate cuda_env
  cd /home/ubuntu/pj
  python train_cuda.py \
    --config se_e2_a/input_torch.json \
    --data-dir collect/O64H128 \
    --num-workers 4 \
    --mixed-precision

后台运行 (推荐):
────────────────
  nohup python train_cuda.py \
    --config se_e2_a/input_torch.json \
    --data-dir collect/O64H128 \
    --num-workers 4 \
    --mixed-precision > cuda_training.log 2>&1 &

  # 查看进度
  tail -f cuda_training.log


🔍 验证和监控
═══════════════════════════════════════════════════════════════════════════════

验证环境:
─────────
  conda activate cuda_env
  cd /home/ubuntu/pj
  python verify_cuda_env.py

监控GPU:
────────
  watch -n 1 nvidia-smi
  # 或
  nvidia-smi -l 1

监控训练:
─────────
  tail -f cuda_training.log
  tail -f training.log

查看进程:
─────────
  ps aux | grep train_deepmd_pytorch
  # 应该看到两个:
  #   1. ...train_cpu.py (ai4m, CPU)
  #   2. ...train_cuda.py (cuda_env, GPU)


📂 重要文件位置
═══════════════════════════════════════════════════════════════════════════════

CUDA训练脚本:
  /home/ubuntu/pj/train_cuda.py

快速启动脚本:
  /home/ubuntu/pj/cuda_quickstart.sh

环境验证脚本:
  /home/ubuntu/pj/verify_cuda_env.py

部署指南:
  /home/ubuntu/pj/CUDA_DEPLOYMENT_GUIDE.md
  /home/ubuntu/pj/CUDA_SETUP_COMPLETE.md

模型输出目录:
  CPU版本: /home/ubuntu/pj/exports/
  GPU版本: /home/ubuntu/pj/exports_cuda/ ⭐

检查点目录:
  CPU版本: /home/ubuntu/pj/checkpoints/
  GPU版本: /home/ubuntu/pj/checkpoints_cuda/ ⭐


🎯 性能对比
═══════════════════════════════════════════════════════════════════════════════

                  CPU训练              GPU训练             加速倍数
  ─────────────────────────────────────────────────────────────────
  处理器          Intel CPU (8核)      NVIDIA L20-8Q        —
  内存占用        ~4.2 GB             ~2-4 GB              相似
  显存占用        N/A                 ~1-3 GB              —
  训练速度        ~0.17 steps/sec     ~1-3 steps/sec       5-15x ⚡
  500步耗时       ~50分钟             ~5-10分钟            5-10x
  10000步耗时     ~10-12小时          ~1-2小时             5-10x


⚙️ 常用参数说明
═══════════════════════════════════════════════════════════════════════════════

--config             配置文件路径 (JSON)
--data-dir           数据目录 (DeepMD格式)
--checkpoint-dir     保存检查点目录 (默认: checkpoints_cuda)
--export-dir         导出模型目录 (默认: exports_cuda)
--num-workers        数据加载线程数 (推荐: 4-8, 默认: 4)
--mixed-precision    启用混合精度训练 (可选, 推荐启用)


🔧 故障排除
═══════════════════════════════════════════════════════════════════════════════

❌ CUDA not available
──────────────────
  原因: 显卡驱动或PyTorch版本问题
  解决:
    nvidia-smi                # 检查驱动
    pip install -U torch --index-url https://download.pytorch.org/whl/cu121

❌ CUDA out of memory
──────────────────
  原因: 显存不足
  解决:
    1. 启用混合精度: --mixed-precision
    2. 减少workers: --num-workers 2
    3. 修改config中的batch_size为1

❌ 速度没有预期快
──────────────────
  原因: 数据加载瓶颈或GPU未充分利用
  解决:
    1. 增加workers: --num-workers 8
    2. 启用混合精度: --mixed-precision
    3. 检查GPU使用率: watch -n 1 nvidia-smi


💾 模型导出和推理
═══════════════════════════════════════════════════════════════════════════════

查看模型:
─────────
  ls -lh /home/ubuntu/pj/exports_cuda/

推理:
─────
  conda activate cuda_env
  cd /home/ubuntu/pj
  python inference.py \
    --model exports_cuda/model_cuda_*.pth \
    --input collect/O64H128/set.000/coord.npy


🎓 更多资讯
═══════════════════════════════════════════════════════════════════════════════

部署指南:        CUDA_DEPLOYMENT_GUIDE.md
完成总结:        CUDA_SETUP_COMPLETE.md
项目README:      README.md
复现指南:        README_reproduce.md
实现总结:        IMPLEMENTATION_SUMMARY.md


✨ 关键优势
═══════════════════════════════════════════════════════════════════════════════

1. 环境完全隔离
   ✓ 原CPU任务继续运行，不受影响
   ✓ 两个环境独立，相互不冲突
   ✓ 可同时进行CPU和GPU训练

2. GPU加速显著
   ✓ 预计5-15x加速
   ✓ 节省8-11小时训练时间
   ✓ GPU显存充足 (8.36 GB)

3. 即用即开
   ✓ 所有配置已完成
   ✓ 只需3条命令启动
   ✓ 包含完整验证和监控

4. 易于扩展
   ✓ 支持从检查点恢复 (计划中)
   ✓ 支持混合精度训练
   ✓ 支持多GPU分布式 (计划中)


════════════════════════════════════════════════════════════════════════════════

                    🚀 准备就绪，可以启动GPU训练了！

                    命令: conda activate cuda_env && cd /home/ubuntu/pj && bash cuda_quickstart.sh

════════════════════════════════════════════════════════════════════════════════

EOF
