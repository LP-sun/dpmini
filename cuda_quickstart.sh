#!/bin/bash

# CUDA GPU加速训练快速启动脚本
# 使用: bash cuda_quickstart.sh [步数]

set -e

echo "=========================================="
echo "  DeepMD CUDA GPU 加速训练启动脚本"
echo "=========================================="
echo ""

# 参数: 步数 (默认: 10000)
STEPS=${1:-10000}

# 检查conda环境
echo "1. 检查conda环境..."
if ! command -v conda &> /dev/null; then
    echo "❌ 错误: 未找到conda，请先安装Miniconda/Anaconda"
    exit 1
fi

# 激活CUDA环境
echo "2. 激活cuda_env环境..."
source /home/ubuntu/miniforge3/bin/activate cuda_env || {
    echo "❌ 错误: 无法激活cuda_env环境"
    echo "   请先运行: conda create -n cuda_env python=3.10"
    exit 1
}

# 验证PyTorch
echo "3. 验证PyTorch CUDA支持..."
python << 'PYEOF'
import torch
if not torch.cuda.is_available():
    print("⚠️  警告: CUDA不可用，将使用CPU训练(速度很慢)")
    print("   GPU名称: 无")
    print("   GPU内存: 无")
else:
    print(f"✓ CUDA可用")
    print(f"  PyTorch版本: {torch.__version__}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    内存: {props.total_memory / 1e9:.2f} GB")
PYEOF

# 进入项目目录
echo ""
echo "4. 进入项目目录..."
cd /home/ubuntu/pj || {
    echo "❌ 错误: 无法进入 /home/ubuntu/pj"
    exit 1
}

# 检查训练脚本
if [ ! -f "train_cuda.py" ]; then
    echo "❌ 错误: 找不到 train_cuda.py"
    exit 1
fi

# 检查数据目录
if [ ! -d "collect/O64H128" ]; then
    echo "❌ 错误: 找不到数据目录 collect/O64H128"
    exit 1
fi

# 生成配置文件 (快速测试)
echo ""
echo "5. 生成快速测试配置..."
mkdir -p se_e2_a
if [ ! -f "se_e2_a/input_torch_quick_test.json" ]; then
    python << 'JSONEOF'
import json
config = {
    "model": {
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "se_e2_a",
            "sel": [64, 128],
            "rcut": 6.0,
            "rcut_smth": 0.5,
            "neuron": [25, 50, 100],
            "axis_neuron": 16,
            "type_one_side": True,
            "resnet_dt": True,
            "seed": 1
        },
        "fitting_net": {
            "neuron": [240, 240, 240],
            "resnet_dt": True,
            "seed": 1
        }
    },
    "training": {
        "numb_steps": 500,
        "disp_freq": 50,
        "save_freq": 500,
        "training_data": {
            "batch_size": 1,
            "systems": ["collect/O64H128"]
        }
    },
    "learning_rate": {
        "start_lr": 1e-3,
        "stop_lr": 3.51e-8,
        "decay_steps": 5000
    },
    "loss": {
        "start_pref_e": 0.02,
        "limit_pref_e": 1.0,
        "start_pref_f": 1000.0,
        "limit_pref_f": 1.0
    }
}
with open("se_e2_a/input_torch_quick_test.json", "w") as f:
    json.dump(config, f, indent=2)
print("✓ 快速测试配置已生成")
JSONEOF
else
    echo "✓ 快速测试配置已存在"
fi

# 创建输出目录
echo ""
echo "6. 创建输出目录..."
mkdir -p checkpoints_cuda exports_cuda

# 启动训练
echo ""
echo "=========================================="
echo "  开始GPU加速训练"
echo "=========================================="
echo "配置: se_e2_a/input_torch_quick_test.json"
echo "数据: collect/O64H128 (1823 frames)"
echo "步数: 500 (快速测试)"
echo "设备: CUDA GPU (如果可用)"
echo ""

python train_cuda.py \
    --config se_e2_a/input_torch_quick_test.json \
    --data-dir collect/O64H128 \
    --checkpoint-dir checkpoints_cuda \
    --export-dir exports_cuda \
    --num-workers 4 \
    --mixed-precision

echo ""
echo "=========================================="
echo "  训练完成!"
echo "=========================================="
echo ""
echo "✓ 检查点已保存到: checkpoints_cuda/"
echo "✓ 模型已导出到: exports_cuda/"
echo ""
echo "后续步骤:"
echo "1. 查看结果: ls -lh exports_cuda/"
echo "2. 运行推理: python inference.py --model exports_cuda/model_cuda_*.pth --input collect/O64H128/set.000/coord.npy"
echo "3. 完整训练: 修改--config参数为 se_e2_a/input_torch.json"
echo ""
