#!/usr/bin/env python3
"""快速训练测试 - 验证3个指标"""
import sys
sys.path.insert(0, '/home/ubuntu/pj')

import torch
import json
from datetime import datetime
from dpmini.data import DeepMDDataset
from dpmini.model import DeepMDModel
from torch.optim import Adam

print("✓ 导入成功")

# 加载config
with open('config_short_test.json') as f:
    config = json.load(f)
print("✓ 配置加载成功")

# 加载数据集
dataset = DeepMDDataset(['collect/O64H128'], type_map=['O', 'H'])
print(f"✓ 数据集加载成功: {len(dataset)} frames")

# 构建模型
model = DeepMDModel(
    type_map=['O', 'H'],
    rcut=config['model']['descriptor']['rcut'],
    rcut_smth=config['model']['descriptor']['rcut_smth'],
    sel=config['model']['descriptor']['sel'],
    descriptor_neuron=config['model']['descriptor']['neuron'],
    axis_neuron=config['model']['descriptor']['axis_neuron'],
    fitting_neuron=config['model']['fitting_net']['neuron']
).cuda()
print(f"✓ 模型创建成功: {sum(p.numel() for p in model.parameters())} 参数")

# 创建日志文件
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_file = f"cuda_training_opt.log"
log = open(log_file, 'w')

# 简单训练循环（不使用DataLoader）
optimizer = Adam(model.parameters(), lr=0.001)
losses = []

for step in range(1, 401):  # 400步
    idx = step % len(dataset)
    data = dataset[idx]
    positions = data[0].cuda()  # (natom, 3)
    positions.requires_grad_(True)  # Must be True to compute forces
    atom_types = data[1].cuda()  # (natom,)
    # cell = data[2]  # (3, 3)
    target_energy = data[3].cuda()  # scalar
    target_forces = data[4].cuda()  # (natom, 3)
    
    optimizer.zero_grad()
    
    positions.requires_grad_(True)
    pred_energy, _, pred_forces = model.get_forces(positions, atom_types)
    
    # Energy loss: MSE(pred, target)
    e_loss = torch.nn.functional.mse_loss(pred_energy, target_energy)
    # Force loss: MSE over all atom forces
    f_loss = torch.nn.functional.mse_loss(pred_forces, target_forces)
    # Combined loss
    loss = 0.02 * e_loss + 1000 * f_loss
    
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if step % 50 == 0:
        lr = optimizer.param_groups[0]['lr']
        # Loss consistency check
        total_recon = 0.02 * e_loss + 1000 * f_loss
        diff = abs((loss - total_recon).item()) if isinstance(loss, torch.Tensor) else 0
        log_line = f"Step {step:4d} | lr={lr:.4e} | loss={loss.item():.6e} | e_loss={e_loss.item():.6e} | f_loss={f_loss.item():.6e} | pref_e=0.02 pref_f=1000 | diff={diff:.3e}\n"
        log.write(log_line)
        print(log_line.strip())
        log.flush()

log.close()
print(f"\n✅ 训练完成，日志保存到 {log_file}")
print(f"初始loss: {losses[0]:.4e}")
print(f"最后loss: {losses[-1]:.4e}")
print(f"下降比例: {(1 - losses[-1]/losses[0])*100:.1f}%")
