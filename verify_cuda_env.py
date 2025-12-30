#!/usr/bin/env python3
"""
验证CUDA GPU加速环境的脚本
运行此脚本确保所有依赖已正确安装
"""

import sys
import platform
from pathlib import Path

def print_header(title):
    """打印标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def check_python():
    """检查Python版本"""
    print_header("1. Python环境")
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    return True

def check_torch():
    """检查PyTorch安装"""
    print_header("2. PyTorch检查")
    try:
        import torch
        print(f"✓ PyTorch已安装: {torch.__version__}")
        
        # 检查CUDA
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA可用: {'✓ 是' if cuda_available else '✗ 否'}")
        
        if cuda_available:
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  cuDNN版本: {torch.backends.cudnn.version()}")
            print(f"  GPU数量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"\n  GPU {i}:")
                print(f"    名称: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    总内存: {props.total_memory / 1e9:.2f} GB")
                print(f"    已分配: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
                print(f"    已保留: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
                print(f"    计算能力: {props.major}.{props.minor}")
            
            # 测试CUDA张量操作
            print("\n  CUDA功能测试:")
            try:
                x = torch.randn(100).cuda()
                y = torch.randn(100).cuda()
                z = torch.matmul(x, y)
                print(f"    ✓ CUDA张量操作正常")
            except Exception as e:
                print(f"    ✗ CUDA张量操作失败: {e}")
        else:
            print("  ⚠️  CUDA不可用，将使用CPU训练(性能较低)")
        
        return cuda_available
    except ImportError:
        print("✗ PyTorch未安装")
        return False

def check_dependencies():
    """检查其他依赖"""
    print_header("3. 依赖包检查")
    
    packages = [
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('h5py', 'h5py'),
        ('pytest', 'pytest')
    ]
    
    all_ok = True
    for display_name, import_name in packages:
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {display_name}: {version}")
        except ImportError:
            print(f"✗ {display_name}: 未安装")
            all_ok = False
    
    return all_ok

def check_dpmini():
    """检查DPMini模块"""
    print_header("4. DPMini模块检查")
    
    try:
        from dpmini import DeepMDModel, DeepMDDataset
        print("✓ DPMini模块导入成功")
        
        # 检查数据
        try:
            dataset = DeepMDDataset(['collect/O64H128'], type_map=['O', 'H'])
            print(f"✓ 数据集加载成功: {len(dataset)} frames")
        except Exception as e:
            print(f"⚠️  数据集加载: {e}")
        
        return True
    except ImportError as e:
        print(f"✗ DPMini模块导入失败: {e}")
        return False

def check_files():
    """检查关键文件"""
    print_header("5. 关键文件检查")
    
    files = {
        "训练脚本 (CPU)": "train_cpu.py",
        "训练脚本 (CUDA)": "train_cuda.py",
        "推理脚本": "inference.py",
        "配置文件": "se_e2_a/input_torch.json",
        "快速配置": "se_e2_a/input_torch_quick_test.json",
        "数据目录": "collect/O64H128/set.000",
        "DPMini包": "dpmini/__init__.py",
    }
    
    all_ok = True
    for name, path in files.items():
        if Path(path).exists():
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name}: {path} (不存在)")
            all_ok = False
    
    return all_ok

def check_training_script():
    """检查训练脚本是否可执行"""
    print_header("6. 训练脚本检查")
    
    try:
        with open('train_cuda.py', 'r') as f:
            content = f.read()
        
        # 检查关键函数
        functions = ['check_cuda_info', 'get_learning_rate', 'compute_loss', 'main']
        for func in functions:
            if f'def {func}' in content:
                print(f"✓ 函数 {func} 存在")
            else:
                print(f"✗ 函数 {func} 缺失")
        
        return True
    except Exception as e:
        print(f"✗ 训练脚本检查失败: {e}")
        return False

def run_diagnostic():
    """运行诊断"""
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "DeepMD CUDA GPU 环境诊断工具" + " " * 20 + "║")
    print("╚" + "=" * 68 + "╝")
    
    results = {
        "Python环境": check_python(),
        "PyTorch (CUDA)": check_torch(),
        "依赖包": check_dependencies(),
        "DPMini模块": check_dpmini(),
        "关键文件": check_files(),
        "训练脚本": check_training_script()
    }
    
    # 总结
    print_header("诊断总结")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for name, status in results.items():
        status_str = "✓ 通过" if status else "✗ 失败"
        print(f"{status_str}: {name}")
    
    print(f"\n总体: {passed}/{total} 检查通过")
    
    if passed == total:
        print("\n" + "🎉 " * 15)
        print("所有检查都通过了! 可以开始CUDA加速训练\n")
        print("快速开始命令:")
        print("  bash cuda_quickstart.sh\n")
        return 0
    else:
        print("\n" + "⚠️  " * 15)
        print("存在未通过的检查，请解决上述问题后再继续\n")
        return 1

if __name__ == '__main__':
    sys.exit(run_diagnostic())
