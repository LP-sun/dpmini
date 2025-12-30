#!/usr/bin/env python3
"""
快速验证集评估工具
仅加载验证集子集，避免加载全量数据集
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from dpmini import DeepMDModel, DeepMDDataset

def load_config(config_path):
    """加载 JSON 配置文件"""
    with open(config_path, 'r') as f:
        return json.load(f)

def evaluate_checkpoint_fast(checkpoint_path, system_dir, val_indices_path, config_path, device='cuda'):
    """
    快速评估 checkpoint 在验证集上的性能
    只加载验证集的样本，避免加载全量数据集
    
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
    
    # 加载数据集
    print("加载数据集...")
    dataset = DeepMDDataset([system_dir])
    print(f"✓ 加载数据系统: {system_dir} ({len(dataset)} 帧)")
    
    # 加载配置
    config = load_config(config_path)
    model_config = config.get('model', {})
    descriptor_config = model_config.get('descriptor', {})
    fitting_config = model_config.get('fitting_net', {})
    type_map = model_config.get('type_map', ['O', 'H'])
    
    # 初始化模型
    print("初始化模型...")
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
    
    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print(f"✓ 加载 checkpoint: {checkpoint_path}")
    
    # 评估验证集
    print(f"评估 {len(val_indices)} 个验证样本...")
    energy_errors = []
    force_errors = []
    
    for idx_in_dataset in val_indices:
        coord, atom_type, box, energy_ref, force_ref = dataset[idx_in_dataset]
        
        # 转移到设备
        coord = coord.to(device)  # (natom, 3)
        coord.requires_grad_(True)  # 需要梯度来计算力
        atom_type = atom_type.to(device)  # (natom,)
        box = box.to(device) if box is not None else None
        
        # 前向传播获取力
        pred_energy, _, pred_forces = model.get_forces(coord, atom_type, box)
        
        # 计算误差
        with torch.no_grad():
            energy_error = float((pred_energy - energy_ref.to(device)).abs().cpu().numpy())
            energy_errors.append(energy_error)
            
            # 力的逐分量 RMSE
            force_ref = force_ref.to(device)
            force_diff = (pred_forces - force_ref).cpu().numpy()  # (natom, 3)
            f_rmse_frame = np.sqrt(np.mean(force_diff ** 2))
            force_errors.append(f_rmse_frame)
    
    # 统计结果
    energy_errors = np.array(energy_errors)
    force_errors = np.array(force_errors)
    
    # E_RMSE per atom (RMSE of energy errors)
    e_rmse_per_atom = np.sqrt(np.mean(energy_errors**2))
    
    # F_RMSE (平均帧力 RMSE)
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
    parser = argparse.ArgumentParser(description='Fast evaluate checkpoint on validation set')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--system', type=str, default='collect/O64H128', help='System directory')
    parser.add_argument('--val-indices', type=str, default='logs/val_indices.npy', 
                       help='Validation indices file')
    parser.add_argument('--config', type=str, default='config_formal_100k_fixed.json',
                       help='Config file for model architecture')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    print(f"🔍 快速评估 checkpoint: {args.checkpoint}")
    result = evaluate_checkpoint_fast(args.checkpoint, args.system, args.val_indices, args.config, args.device)
    
    if result:
        print("\n" + "="*70)
        print("📊 验证集评估结果")
        print("="*70)
        print(f"E_RMSE (per atom): {result['e_rmse_per_atom']:.6f}")
        print(f"F_RMSE (mean):     {result['f_rmse']:.6f}")
        print(f"F_RMSE (tail 10%): {result['f_rmse_tail']:.6f}")
        print(f"验证帧数:         {result['n_val_frames']}")
        print("="*70)
        
        # 输出 JSON 格式以便脚本处理
        print(json.dumps(result))
