#!/usr/bin/env python3
"""
验证集评估工具
用固定的验证集评估任意 checkpoint，获取 E_RMSE、F_RMSE 和 tail F_RMSE（最差 10%）
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from dpmini import DeepMDModel, DeepMDDataset

def load_config(config_path):
    """加载 JSON 配置文件"""
    with open(config_path, 'r') as f:
        return json.load(f)

def evaluate_checkpoint(checkpoint_path, system_dir, val_indices_path, config_path, device='cuda'):
    """
    评估 checkpoint 在验证集上的性能
    
    Args:
        checkpoint_path: 模型 checkpoint 路径
        system_dir: 数据系统目录
        val_indices_path: 验证集索引文件
        config_path: 配置文件路径
        device: 计算设备
    
    Returns:
        dict: 包含 E_RMSE, F_RMSE, F_RMSE_tail 的字典
    """
    
    # 加载验证集索引
    if not Path(val_indices_path).exists():
        print(f"❌ 验证集索引不存在: {val_indices_path}")
        return None
    
    val_indices = np.load(val_indices_path)
    print(f"✓ 加载验证集索引: {len(val_indices)} 帧")
    
    # 加载数据集（DeepMDDataset 期望列表）
    dataset = DeepMDDataset([system_dir])
    print(f"✓ 加载数据系统: {system_dir} ({len(dataset)} 帧)")
    
    # 加载配置
    config = load_config(config_path)
    model_config = config.get('model', {})
    descriptor_config = model_config.get('descriptor', {})
    fitting_config = model_config.get('fitting_net', {})
    type_map = model_config.get('type_map', ['O', 'H'])
    
    # 加载模型
    model = DeepMDModel(
        type_map=type_map,
        rcut=descriptor_config['rcut'],
        rcut_smth=descriptor_config['rcut_smth'],
        sel=descriptor_config['sel'],
        descriptor_neuron=descriptor_config['neuron'],
        axis_neuron=descriptor_config['axis_neuron'],
        fitting_neuron=fitting_config['neuron'],
        type_one_side=descriptor_config.get('type_one_side', True),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print(f"✓ 加载 checkpoint: {checkpoint_path}")
    
    # 评估
    energy_errors = []
    force_errors = []
    
    with torch.no_grad():
        for val_idx in val_indices:
            data = dataset[int(val_idx)]
            positions = data[0].to(device)  # (natom, 3)
            atom_types = data[1].to(device)
            box = data[2].to(device)
            target_energy = data[3].to(device)
            target_forces = data[4].to(device)  # (natom, 3)
            
            # Forward pass
            positions.requires_grad_(True)
            pred_energy, pred_forces = model(positions, atom_types, box)
            positions.requires_grad_(False)
            
            # Energy RMSE per atom
            natom = positions.shape[0]
            energy_error = ((pred_energy - target_energy) ** 2).item()
            energy_errors.append(energy_error)
            
            # Force RMSE (全 3N 分量)
            force_diff = pred_forces - target_forces
            force_mse = (force_diff ** 2).mean().item()
            force_error = np.sqrt(force_mse)
            force_errors.append(force_error)
    
    # 统计
    force_errors = np.array(force_errors)
    energy_errors = np.array(energy_errors)
    
    # E_RMSE per atom (RMSE of energy errors)
    e_rmse_per_atom = np.sqrt(np.mean(energy_errors))
    
    # F_RMSE (所有力的 RMSE)
    f_rmse = np.mean(force_errors)
    
    # F_RMSE_tail: 最差 10% 帧的平均 F_RMSE
    tail_count = max(1, len(force_errors) // 10)
    f_rmse_tail = np.mean(np.sort(force_errors)[-tail_count:])
    
    return {
        'e_rmse_per_atom': float(e_rmse_per_atom),
        'f_rmse': float(f_rmse),
        'f_rmse_tail': float(f_rmse_tail),
        'n_val_frames': len(val_indices),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate checkpoint on validation set')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--system', type=str, default='collect/O64H128', help='System directory')
    parser.add_argument('--val-indices', type=str, default='logs/val_indices.npy', 
                       help='Validation indices file')
    parser.add_argument('--config', type=str, default='config_formal_100k_fixed.json',
                       help='Config file for model architecture')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    print(f"🔍 评估 checkpoint: {args.checkpoint}")
    result = evaluate_checkpoint(args.checkpoint, args.system, args.val_indices, args.config, args.device)
    
    if result:
        print("\n" + "="*70)
        print("📊 验证集评估结果")
        print("="*70)
        print(f"E_RMSE (per atom): {result['e_rmse_per_atom']:.6f}")
        print(f"F_RMSE (full):     {result['f_rmse']:.6f}")
        print(f"F_RMSE (tail 10%): {result['f_rmse_tail']:.6f}")
        print(f"验证帧数:         {result['n_val_frames']}")
        print("="*70)
        
        # 输出 JSON 格式以便脚本处理
        import json
        print(json.dumps(result))
