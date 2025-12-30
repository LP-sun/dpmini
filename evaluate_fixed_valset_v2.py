#!/usr/bin/env python3
"""
固定验证集评估脚本 (改进版)
- 从 collect/O64H128 随机固定抽取 200 帧作为验证集 (val_idx.npy)
- 用导出的 checkpoint 在这 200 帧上评估
- 保存结果到 logs/val_metrics.json
"""

import json
import argparse
from pathlib import Path
import numpy as np
import torch
from dpmini import DeepMDModel, DeepMDDataset

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return json.load(f)

def create_fixed_valset(data_dir, n_frames=200, output_path='val_idx.npy', seed=42):
    """
    从数据系统中随机固定抽取验证集索引
    """
    dataset = DeepMDDataset([data_dir])
    total_frames = len(dataset)
    
    print(f"\n📊 数据系统: {data_dir}")
    print(f"   总帧数: {total_frames}")
    print(f"   提取帧数: {n_frames}")
    
    # 固定随机种子以确保可重复性
    np.random.seed(seed)
    val_indices = np.random.choice(total_frames, size=min(n_frames, total_frames), replace=False)
    val_indices = np.sort(val_indices)
    
    # 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, val_indices)
    print(f"✓ 验证集索引已保存: {output_path}")
    print(f"  索引范围: {val_indices.min()} - {val_indices.max()}")
    print(f"  前 10 个: {val_indices[:10]}")
    
    return val_indices

def evaluate_on_valset(checkpoint_path, system_dir, val_indices, config_path, device='cuda'):
    """
    在固定验证集上评估模型性能
    
    Returns:
        dict: 包含性能指标的字典
    """
    print(f"\n🔄 加载模型: {checkpoint_path}")
    
    # 加载配置
    config = load_config(config_path)
    model_config = config.get('model', {})
    descriptor_config = model_config.get('descriptor', {})
    fitting_config = model_config.get('fitting_net', {})
    type_map = model_config.get('type_map', ['O', 'H'])
    
    # 重构并加载模型
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
    print(f"✓ 模型加载成功 (设备: {device})")
    
    # 加载数据集
    dataset = DeepMDDataset([system_dir])
    print(f"✓ 数据集加载: {len(dataset)} 帧")
    
    # 评估
    print(f"\n📈 在 {len(val_indices)} 帧验证集上评估...")
    
    energy_errors = []
    force_errors = []
    force_errors_all = []  # 所有原子的力误差，用于计算统计量
    
    for i, val_idx in enumerate(val_indices):
        if (i + 1) % 50 == 0:
            print(f"  进度: {i + 1}/{len(val_indices)}")
        
        data = dataset[int(val_idx)]
        positions = data[0].to(device)  # (natom, 3)
        atom_types = data[1].to(device)
        box = data[2].to(device)
        target_energy = data[3].to(device)
        target_forces = data[4].to(device)  # (natom, 3)
        
        # Forward pass - 需要计算梯度以得到力
        positions.requires_grad_(True)
        pred_energy, pred_atomic_energies = model(positions, atom_types, box)
        
        # 计算力: F = -dE/dr
        pred_forces = -torch.autograd.grad(pred_energy, positions, create_graph=False)[0]
        
        with torch.no_grad():
            # Energy 误差 (per atom RMSE)
            natom = positions.shape[0]
            energy_error = ((pred_energy - target_energy) ** 2).item()
            energy_errors.append(energy_error)
            
            # Force 误差 (per frame)
            force_diff = pred_forces - target_forces  # (natom, 3)
            force_mse_frame = (force_diff ** 2).mean().item()
            force_rmse_frame = np.sqrt(force_mse_frame)
            force_errors.append(force_rmse_frame)
            
            # 保存所有原子的力误差以计算全局统计
            force_errors_all.extend((force_diff ** 2).sqrt().cpu().numpy().flatten())
    
    # 统计
    force_errors = np.array(force_errors)
    force_errors_all = np.array(force_errors_all)
    energy_errors = np.array(energy_errors)
    
    # 能量误差 (per atom RMSE)
    e_rmse_per_atom = np.sqrt(np.mean(energy_errors))
    e_mae_per_atom = np.sqrt(np.mean(energy_errors))  # 当前使用 RMSE
    
    # 力误差统计
    f_rmse_mean = np.mean(force_errors)
    f_rmse_median = np.median(force_errors)
    f_rmse_95 = np.percentile(force_errors, 95)
    f_rmse_std = np.std(force_errors)
    
    # 按帧的最差 10% (tail)
    tail_count = max(1, len(force_errors) // 10)
    f_rmse_tail = np.mean(np.sort(force_errors)[-tail_count:])
    
    print(f"\n✓ 评估完成:")
    print(f"  F_RMSE_mean:     {f_rmse_mean:.6f} eV/Å")
    print(f"  F_RMSE_median:   {f_rmse_median:.6f} eV/Å")
    print(f"  F_RMSE_95%ile:   {f_rmse_95:.6f} eV/Å")
    print(f"  F_RMSE_tail10%:  {f_rmse_tail:.6f} eV/Å")
    print(f"  F_RMSE_std:      {f_rmse_std:.6f} eV/Å")
    print(f"  E_RMSE_per_atom: {e_rmse_per_atom:.6f} eV")
    
    return {
        'f_rmse_mean': float(f_rmse_mean),
        'f_rmse_median': float(f_rmse_median),
        'f_rmse_95': float(f_rmse_95),
        'f_rmse_tail': float(f_rmse_tail),
        'f_rmse_std': float(f_rmse_std),
        'e_rmse_per_atom': float(e_rmse_per_atom),
        'e_mae_per_atom': float(e_mae_per_atom),
        'n_frames': len(val_indices),
        'checkpoint': str(checkpoint_path),
        'config': str(config_path),
    }

def main():
    parser = argparse.ArgumentParser(description='固定验证集评估')
    parser.add_argument('--data-dir', default='collect/O64H128', help='数据目录')
    parser.add_argument('--checkpoint', default='checkpoints_optimized_v3/latest_checkpoint.pt', 
                        help='模型 checkpoint 路径')
    parser.add_argument('--config', default='config_formal_100k_optimized.json',
                        help='配置文件路径')
    parser.add_argument('--val-idx-path', default='val_idx.npy', help='验证集索引输出路径')
    parser.add_argument('--n-frames', type=int, default=200, help='验证集帧数')
    parser.add_argument('--output-json', default='logs/val_metrics.json', 
                        help='评估结果输出路径')
    parser.add_argument('--device', default='cuda', help='计算设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    try:
        # 1. 创建固定验证集
        val_indices = create_fixed_valset(
            args.data_dir, 
            n_frames=args.n_frames,
            output_path=args.val_idx_path,
            seed=args.seed
        )
        
        # 2. 评估
        metrics = evaluate_on_valset(
            args.checkpoint,
            args.data_dir,
            val_indices,
            args.config,
            device=args.device
        )
        
        # 3. 保存结果
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✅ 评估结果已保存: {output_path}")
        
        # 打印摘要
        print(f"\n{'='*70}")
        print(f"📋 固定验证集评估摘要")
        print(f"{'='*70}")
        print(f"验证集:        {len(val_indices)} 帧 (来自 {args.data_dir})")
        print(f"Checkpoint:    {args.checkpoint}")
        print(f"配置:         {args.config}")
        print(f"{'─'*70}")
        print(f"F_RMSE_mean:   {metrics['f_rmse_mean']:.6f} eV/Å")
        print(f"F_RMSE_median: {metrics['f_rmse_median']:.6f} eV/Å")
        print(f"F_RMSE_95%:    {metrics['f_rmse_95']:.6f} eV/Å")
        print(f"F_RMSE_tail:   {metrics['f_rmse_tail']:.6f} eV/Å (最差10%)")
        print(f"E_RMSE/atom:   {metrics['e_rmse_per_atom']:.6f} eV")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
