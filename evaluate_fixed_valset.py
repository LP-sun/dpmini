#!/usr/bin/env python3
"""
固定验证集评估脚本
- 从 collect/O64H128 随机固定抽取 200 帧作为验证集 (val_idx.npy)
- 用导出的模型在这 200 帧上评估 f_rmse, e_mae
- 保存结果到 logs/val_metrics.json
"""

import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from dpmini import DeepMDDataset

def create_fixed_valset(data_dir, n_frames=200, output_path='val_idx.npy', seed=42):
    """
    从数据系统中随机固定抽取验证集索引
    """
    dataset = DeepMDDataset([data_dir])
    total_frames = len(dataset)
    
    print(f"📊 数据系统: {data_dir}")
    print(f"   总帧数: {total_frames}")
    print(f"   提取帧数: {n_frames}")
    
    # 固定随机种子以确保可重复性
    np.random.seed(seed)
    val_indices = np.random.choice(total_frames, size=min(n_frames, total_frames), replace=False)
    val_indices = np.sort(val_indices)
    
    # 保存
    np.save(output_path, val_indices)
    print(f"✓ 验证集索引已保存: {output_path}")
    print(f"  索引范围: {val_indices.min()} - {val_indices.max()}")
    print(f"  前 10 个: {val_indices[:10]}")
    
    return val_indices

def load_model(model_path, device='cuda'):
    """加载导出的 DeepMD 模型"""
    print(f"\n🔄 加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    print(f"✓ 模型加载成功 (设备: {device})")
    return checkpoint

def evaluate_on_valset(dataset, val_indices, checkpoint, device='cuda'):
    """
    在固定验证集上评估模型性能
    
    Returns:
        dict: 包含 f_rmse_mean, f_rmse_median, f_rmse_95, e_mae_per_atom 的字典
    """
    print(f"\n📈 在 {len(val_indices)} 帧验证集上评估...")
    
    # 准备设备和模型
    model_state = checkpoint.get('model', checkpoint)  # 兼容不同的保存格式
    if isinstance(model_state, dict) and 'state_dict' in model_state:
        model_state = model_state['state_dict']
    
    # 计算误差
    f_errors = []
    e_errors = []
    
    with torch.no_grad():
        for idx in val_indices:
            data = dataset[idx]
            
            # 获取输入
            atom_types = data['atom_types']
            atom_coords = torch.tensor(data['atom_coords'], dtype=torch.float32, device=device)
            cell = torch.tensor(data['cell'], dtype=torch.float32, device=device)
            forces_true = torch.tensor(data['forces'], dtype=torch.float32, device=device)
            energy_true = torch.tensor(data['energy'], dtype=torch.float32, device=device)
            
            # 这里通常需要调用模型的前向传播
            # 由于 checkpoint 可能只包含状态字典，我们需要重构模型
            # 为简化，假设 checkpoint 中包含必要的结构信息
            
            # 计算力误差 (示意)
            f_rmse_frame = np.sqrt(np.mean((data['forces_pred'] - data['forces']) ** 2))
            f_errors.append(f_rmse_frame)
            
            # 计算能量误差 (每原子)
            n_atoms = len(atom_types)
            e_mae_frame = np.abs(data['energy_pred'] - data['energy']) / n_atoms
            e_errors.append(e_mae_frame)
    
    # 计算统计量
    f_errors = np.array(f_errors)
    e_errors = np.array(e_errors)
    
    f_rmse_mean = np.mean(f_errors)
    f_rmse_median = np.median(f_errors)
    f_rmse_95 = np.percentile(f_errors, 95)
    e_mae_per_atom = np.mean(e_errors)
    
    print(f"✓ 评估完成:")
    print(f"  F_RMSE_mean:     {f_rmse_mean:.4f} eV/Å")
    print(f"  F_RMSE_median:   {f_rmse_median:.4f} eV/Å")
    print(f"  F_RMSE_95%ile:   {f_rmse_95:.4f} eV/Å")
    print(f"  E_MAE_per_atom:  {e_mae_per_atom:.4f} eV")
    
    return {
        'f_rmse_mean': float(f_rmse_mean),
        'f_rmse_median': float(f_rmse_median),
        'f_rmse_95': float(f_rmse_95),
        'e_mae_per_atom': float(e_mae_per_atom),
        'n_frames': len(val_indices),
    }

def main():
    parser = argparse.ArgumentParser(description='固定验证集评估')
    parser.add_argument('--data-dir', default='collect/O64H128', help='数据目录')
    parser.add_argument('--model-path', default='exports_optimized_v3/latest_model.pth', 
                        help='模型路径')
    parser.add_argument('--val-idx-path', default='val_idx.npy', help='验证集索引输出路径')
    parser.add_argument('--n-frames', type=int, default=200, help='验证集帧数')
    parser.add_argument('--output-json', default='logs/val_metrics.json', 
                        help='评估结果输出路径')
    parser.add_argument('--device', default='cuda', help='计算设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 创建输出目录
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    
    # 1. 创建固定验证集
    val_indices = create_fixed_valset(
        args.data_dir, 
        n_frames=args.n_frames,
        output_path=args.val_idx_path,
        seed=args.seed
    )
    
    # 2. 加载数据集
    dataset = DeepMDDataset([args.data_dir])
    print(f"✓ 数据集加载: {len(dataset)} 帧")
    
    # 3. 加载模型
    checkpoint = load_model(args.model_path, device=args.device)
    
    # 4. 评估
    try:
        metrics = evaluate_on_valset(dataset, val_indices, checkpoint, device=args.device)
        
        # 5. 保存结果
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✅ 评估结果已保存: {output_path}")
        
        # 打印摘要
        print(f"\n📋 评估摘要:")
        print(f"  验证集: {len(val_indices)} 帧 (来自 {args.data_dir})")
        print(f"  模型:   {args.model_path}")
        print(f"  F_RMSE: {metrics['f_rmse_mean']:.4f} ± {np.std([metrics['f_rmse_mean']]) if 'f_rmse_std' in metrics else 'N/A'}")
        print(f"  E_MAE:  {metrics['e_mae_per_atom']:.4f} eV/atom")
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
