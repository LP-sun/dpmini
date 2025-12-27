#!/usr/bin/env python3
"""
使用 examples/data/water 进行快速优化训练测试
用于验证性能优化效果
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path
import time
import os
import sys
import argparse

# === 环境配置 ===
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('MKL_NUM_THREADS', '8')

from dpmini.data import DeepMDDataset
from dpmini.model import DeepMDModel


def setup_cuda_optimizations():
    """启用CUDA优化"""
    if torch.cuda.is_available():
        # cuDNN自动调优 - 选择最快的卷积算法
        torch.backends.cudnn.benchmark = True
        
        # TF32加速 (Ampere架构及以上)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        print("✓ CUDA优化已启用")
        print(f"  - cuDNN benchmark: True")
        print(f"  - TF32: True")
    else:
        print("⚠ CUDA不可用，将使用CPU训练")


def print_gpu_info():
    """打印GPU信息"""
    if not torch.cuda.is_available():
        return
    
    print("\n" + "="*70)
    print("GPU 信息")
    print("="*70)
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  显存: {props.total_memory / 1e9:.2f} GB")
        print(f"  计算能力: {props.major}.{props.minor}")
    print("="*70 + "\n")


def train(config):
    """训练主函数"""
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("\n[1] 加载数据...")
    dataset = DeepMDDataset(
        system_dirs=config['system_dirs'],
        type_map=config['type_map']
    )
    
    # 划分训练集和测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"训练集: {train_size} frames")
    print(f"测试集: {test_size} frames")
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(config['num_workers'] > 0)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(config['num_workers'] > 0)
    )
    
    print(f"Batch size: {config['batch_size']}")
    print(f"Num workers: {config['num_workers']}")
    
    # 创建模型
    print("\n[2] 创建模型...")
    model = DeepMDModel(
        type_map=config['type_map'],
        rcut=config['rcut'],
        rcut_smth=config['rcut_smth'],
        sel=config['sel'],
        descriptor_neuron=config['neuron'],
        axis_neuron=config['axis_neuron'],
        fitting_neuron=config['fitting_neuron']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,}")
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # 训练循环
    print("\n[3] 开始训练...")
    print("="*70)
    
    best_test_loss = float('inf')
    t_start = time.time()
    step_times = []
    
    for epoch in range(1, config['num_epochs'] + 1):
        # === 训练阶段 ===
        model.train()
        train_loss_sum = 0.0
        train_e_loss_sum = 0.0
        train_f_loss_sum = 0.0
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            t_batch_start = time.time()
            
            positions_batch, atom_types_batch, box_batch, energy_label_batch, force_label_batch = batch
            positions_batch = positions_batch.to(device, non_blocking=True)
            atom_types_batch = atom_types_batch.to(device, non_blocking=True)
            box_batch = box_batch.to(device, non_blocking=True)
            energy_label_batch = energy_label_batch.to(device, non_blocking=True)
            force_label_batch = force_label_batch.to(device, non_blocking=True)
            
            batch_size_actual = positions_batch.shape[0]
            
            # 累积batch中所有样本的损失
            optimizer.zero_grad()
            batch_loss = 0.0
            batch_e_loss = 0.0
            batch_f_loss = 0.0
            
            for b in range(batch_size_actual):
                positions = positions_batch[b]  # (natom, 3)
                atom_types = atom_types_batch[b]  # (natom,)
                box = box_batch[b]  # (3, 3)
                energy_label = energy_label_batch[b]  # scalar
                force_label = force_label_batch[b]  # (natom, 3)
                
                # 前向传播 - get_forces 返回 (energy, atomic_energies, forces)
                energy_pred, atomic_energies, force_pred = model.get_forces(positions, atom_types, box)
                
                # 计算损失
                natoms = positions.shape[0]
                e_loss = (energy_pred - energy_label) ** 2
                f_loss = torch.mean((force_pred - force_label) ** 2)
                
                pref_e = config['pref_e']
                pref_f = config['pref_f']
                sample_loss = pref_e * e_loss + pref_f * f_loss * natoms
                
                # 累积
                batch_loss += sample_loss
                batch_e_loss += e_loss.item()
                batch_f_loss += f_loss.item()
            
            # 平均并反向传播
            batch_loss = batch_loss / batch_size_actual
            batch_loss.backward()
            optimizer.step()
            
            train_loss_sum += batch_loss.item()
            train_e_loss_sum += batch_e_loss / batch_size_actual
            train_f_loss_sum += batch_f_loss / batch_size_actual
            train_batches += 1
            
            # 记录时间
            t_batch = time.time() - t_batch_start
            step_times.append(t_batch)
            
            # 限制记录数量
            if len(step_times) > 100:
                step_times.pop(0)
        
        train_loss_avg = train_loss_sum / train_batches
        train_e_loss_avg = train_e_loss_sum / train_batches
        train_f_loss_avg = train_f_loss_sum / train_batches
        
        # === 测试阶段 ===
        model.eval()
        test_loss_sum = 0.0
        test_e_loss_sum = 0.0
        test_f_loss_sum = 0.0
        test_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                positions_batch, atom_types_batch, box_batch, energy_label_batch, force_label_batch = batch
                positions_batch = positions_batch.to(device, non_blocking=True)
                atom_types_batch = atom_types_batch.to(device, non_blocking=True)
                box_batch = box_batch.to(device, non_blocking=True)
                energy_label_batch = energy_label_batch.to(device, non_blocking=True)
                force_label_batch = force_label_batch.to(device, non_blocking=True)
                
                batch_size_actual = positions_batch.shape[0]
                
                batch_loss = 0.0
                batch_e_loss = 0.0
                batch_f_loss = 0.0
                
                for b in range(batch_size_actual):
                    positions = positions_batch[b]
                    atom_types = atom_types_batch[b]
                    box = box_batch[b]
                    energy_label = energy_label_batch[b]
                    force_label = force_label_batch[b]
                    
                    energy_pred, atomic_energies, force_pred = model.get_forces(positions, atom_types, box)
                    
                    natoms = positions.shape[0]
                    e_loss = (energy_pred - energy_label) ** 2
                    f_loss = torch.mean((force_pred - force_label) ** 2)
                    loss = pref_e * e_loss + pref_f * f_loss * natoms
                    
                    batch_loss += loss.item()
                    batch_e_loss += e_loss.item()
                    batch_f_loss += f_loss.item()
                
                test_loss_sum += batch_loss / batch_size_actual
                test_e_loss_sum += batch_e_loss / batch_size_actual
                test_f_loss_sum += batch_f_loss / batch_size_actual
                test_batches += 1
        
        test_loss_avg = test_loss_sum / test_batches
        test_e_loss_avg = test_e_loss_sum / test_batches
        test_f_loss_avg = test_f_loss_sum / test_batches
        
        # 统计
        elapsed = time.time() - t_start
        avg_step_time = np.mean(step_times) if step_times else 0
        steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
        
        # GPU信息
        gpu_mem = ""
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated(0) / 1e6
            mem_reserved = torch.cuda.memory_reserved(0) / 1e6
            gpu_mem = f"GPU: {mem_allocated:.0f}/{mem_reserved:.0f}MB"
        
        # 打印
        print(f"Epoch {epoch:3d}/{config['num_epochs']} | "
              f"Train Loss: {train_loss_avg:8.4f} (E:{train_e_loss_avg:6.4f}, F:{train_f_loss_avg:6.4f}) | "
              f"Test Loss: {test_loss_avg:8.4f} (E:{test_e_loss_avg:6.4f}, F:{test_f_loss_avg:6.4f}) | "
              f"{steps_per_sec:.2f} steps/s | {gpu_mem}")
        
        # 保存最佳模型
        if test_loss_avg < best_test_loss:
            best_test_loss = test_loss_avg
            checkpoint_path = Path(config['save_dir']) / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': test_loss_avg,
            }, checkpoint_path)
    
    # 训练完成
    total_time = time.time() - t_start
    print("\n" + "="*70)
    print("训练完成!")
    print(f"总时间: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"平均速度: {len(train_loader)*config['num_epochs']/total_time:.2f} steps/s")
    print(f"最佳测试损失: {best_test_loss:.6f}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='DeepMD优化训练 - examples/data')
    parser.add_argument('--batch-size', type=int, default=12, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Data loader workers')
    parser.add_argument('--num-epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--data-dir', type=str, default='examples/data/water', help='Data directory')
    args = parser.parse_args()
    
    print("="*70)
    print("DeepMD 优化训练测试 - examples/data/water")
    print("="*70)
    
    # 设置CUDA优化
    setup_cuda_optimizations()
    print_gpu_info()
    
    # 配置
    config = {
        'system_dirs': [args.data_dir],
        'type_map': ['O', 'H'],
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'save_dir': 'checkpoints_example',
        
        # 模型配置
        'rcut': 6.0,
        'rcut_smth': 0.5,
        'sel': [46, 92],
        'neuron': [25, 50, 100],
        'axis_neuron': 16,
        'fitting_neuron': [240, 240, 240],
        
        # 损失权重
        'pref_e': 1.0,
        'pref_f': 1.0,
    }
    
    # 创建保存目录
    Path(config['save_dir']).mkdir(exist_ok=True)
    
    # 打印配置
    print("\n训练配置:")
    print(f"  数据: {config['system_dirs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Num workers: {config['num_workers']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Rcut: {config['rcut']}")
    print(f"  Sel: {config['sel']}")
    print("")
    
    # 开始训练
    train(config)


if __name__ == '__main__':
    main()
