"""Smoke test for the entire DeepMD training pipeline."""
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dpmini import DeepMDModel, DeepMDDataset

print("=" * 60)
print("DeepMD PyTorch 烟雾测试")
print("=" * 60)

# 1. 测试数据加载
print("\n[1/6] 测试数据加载...")
try:
    dataset = DeepMDDataset("collect/O64H128", type_map=["O", "H"])
    print(f"✅ 数据集加载成功: {len(dataset)} frames")
    
    # 获取一个样本
    sample = dataset[0]
    print(f"   - positions shape: {sample['positions'].shape}")
    print(f"   - forces shape: {sample['forces'].shape}")
    print(f"   - energy: {sample['energy'].item():.4f} eV")
    print(f"   - types: {sample['types'].shape}")
except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    sys.exit(1)

# 2. 测试模型创建
print("\n[2/6] 测试模型创建...")
try:
    config = {
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "se_e2_a",
            "rcut": 6.0,
            "rcut_smth": 5.5,
            "sel": [46, 92],  # max neighbors: O, H
            "neuron": [25, 50, 100],
            "axis_neuron": 16,
            "type_one_side": True
        },
        "fitting_net": {
            "neuron": [240, 240, 240],
            "resnet_dt": True
        }
    }
    
    model = DeepMDModel(config)
    print(f"✅ 模型创建成功")
    print(f"   - 总参数量: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"❌ 模型创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. 测试前向传播（能量）
print("\n[3/6] 测试前向传播（能量计算）...")
try:
    batch = {
        'positions': sample['positions'].unsqueeze(0),  # (1, natom, 3)
        'types': sample['types'].unsqueeze(0),  # (1, natom)
        'box': sample['box'].unsqueeze(0)  # (1, 3, 3)
    }
    
    with torch.no_grad():
        energy_pred = model(batch['positions'], batch['types'], batch['box'])
    
    print(f"✅ 前向传播成功")
    print(f"   - 预测能量: {energy_pred.item():.4f} eV")
    print(f"   - 真实能量: {sample['energy'].item():.4f} eV")
except Exception as e:
    print(f"❌ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. 测试力计算（autograd）
print("\n[4/6] 测试力计算（autograd）...")
try:
    positions = batch['positions'].clone().requires_grad_(True)
    energy = model(positions, batch['types'], batch['box'])
    forces = -torch.autograd.grad(energy.sum(), positions, create_graph=False)[0]
    
    print(f"✅ 力计算成功")
    print(f"   - forces shape: {forces.shape}")
    print(f"   - forces norm: {forces.norm().item():.4f} eV/Å")
    print(f"   - true forces norm: {sample['forces'].norm().item():.4f} eV/Å")
except Exception as e:
    print(f"❌ 力计算失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. 测试批处理
print("\n[5/6] 测试批处理...")
try:
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, 
                           collate_fn=dataset.collate_fn)
    batch = next(iter(dataloader))
    
    energy_pred = model(batch['positions'], batch['types'], batch['box'])
    
    print(f"✅ 批处理成功")
    print(f"   - batch size: {len(energy_pred)}")
    print(f"   - 预测能量: {energy_pred.detach().numpy()}")
    print(f"   - 真实能量: {batch['energy'].numpy()}")
except Exception as e:
    print(f"❌ 批处理失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 6. 测试训练一步
print("\n[6/6] 测试训练一步...")
try:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 前向
    positions = batch['positions'].requires_grad_(True)
    energy_pred = model(positions, batch['types'], batch['box'])
    forces_pred = -torch.autograd.grad(
        energy_pred.sum(), positions, create_graph=True
    )[0]
    
    # 损失
    loss_energy = torch.nn.functional.mse_loss(energy_pred, batch['energy'])
    loss_force = torch.nn.functional.mse_loss(forces_pred, batch['forces'])
    loss = loss_energy + 0.1 * loss_force
    
    # 反向
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"✅ 训练一步成功")
    print(f"   - loss_energy: {loss_energy.item():.6f}")
    print(f"   - loss_force: {loss_force.item():.6f}")
    print(f"   - loss_total: {loss.item():.6f}")
except Exception as e:
    print(f"❌ 训练失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ 所有烟雾测试通过！")
print("=" * 60)
print("\n下一步:")
print("  1. 运行完整训练: python train_cpu.py --config se_e2_a/input_torch.json")
print("  2. 运行单元测试: pytest tests/test_descriptor.py -v")
print("  3. 运行推理测试: python inference.py --model checkpoints/model_final.pth")
