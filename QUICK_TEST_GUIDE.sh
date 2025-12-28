#!/bin/bash
# Quick reference: Model testing and MD simulation

echo "=================================================="
echo "  DeepMD Model Testing & MD Simulation"
echo "=================================================="
echo ""
echo "1️⃣  运行模型测试 (评估预测准确性)"
echo "   Command: conda run -n deepmd python test_model.py --model exports_cuda_opt/model_cuda_20251228-121805.pth --data-dir collect/data0"
echo ""

echo "2️⃣  运行分子动力学模拟 (NVE 微正则系综)"
echo "   Command: conda run -n deepmd python md_simulation.py --model exports_cuda_opt/model_cuda_20251228-121805.pth --system collect/data0 --steps 100"
echo ""

echo "3️⃣  查看完整报告"
echo "   - 测试结果: cat TEST_RESULTS_SUMMARY.md"
echo "   - MD 指南: cat MD_COMPATIBILITY_GUIDE.md"
echo ""

echo "📊 当前模型状态"
echo "   ✅ 能进行精确推理 (能量/力)"
echo "   ✅ 支持 MD 模拟"
echo "   ✅ CUDA 加速可用"
echo "   ⚠️  不能直接用 DeepMD-kit 标准工具"
echo ""

echo "Quick examples:"
echo ""
echo "# 测试所有数据集"
echo "./test_all_datasets.sh"
echo ""
echo "# 短时间 MD (50步 = 0.05 ps)"
echo "conda run -n deepmd python md_simulation.py --model exports_cuda_opt/model_cuda_20251228-121805.pth --system collect/data0 --steps 50 --dt 0.001"
echo ""
echo "# 中等时间 MD (1000步 = 1 ps)"  
echo "conda run -n deepmd python md_simulation.py --model exports_cuda_opt/model_cuda_20251228-121805.pth --system collect/data0 --steps 1000 --dt 0.001 --output md_1ps"
echo ""
echo "# 在不同系统上 MD"
echo "conda run -n deepmd python md_simulation.py --model exports_cuda_opt/model_cuda_20251228-121805.pth --system collect/O64H128 --steps 200"
echo ""

echo "=================================================="
