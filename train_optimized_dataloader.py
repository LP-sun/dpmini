#!/usr/bin/env python3
"""
优化版本训练脚本 - 在原始脚本基础上启用DataLoader
"""
import sys
import os
sys.path.insert(0, '/home/ubuntu/pj')
os.chdir('/home/ubuntu/pj')

# 导入原始脚本的所有函数
from train_cuda_optimized import (
    load_config, get_learning_rate, get_loss_prefactors, compute_loss, check_cuda_info
)

import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dpmini import DeepMDModel, DeepMDDataset


def main():
    parser = argparse.ArgumentParser(description='Optimized DeepMD training with DataLoader')
    parser.add_argument('--config', type=str, default='config_formal_100k_optimized.json',
                       help='Path to config JSON file')
    parser.add_argument('--data-dir', type=str, default='collect/O64H128',
                       help='Path to data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_optimized',
                       help='Directory to save checkpoints')
    parser.add_argument('--export-dir', type=str, default='exports_optimized',
                       help='Directory to export final models')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Use mixed precision training')
    parser.add_argument('--grad-accumulation-steps', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--force-loss', type=str, default='mse',
                       help='Force loss type')
    parser.add_argument('--test-steps', type=int, default=0,
                       help='Test mode: run only N steps')
    args = parser.parse_args()
    
    print("Starting optimized DeepMD training with DataLoader...")
    check_cuda_info()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    config = load_config(config_path)
    
    # Extract model config
    model_config = config.get('model', {})
    descriptor_config = model_config.get('descriptor', {})
    fitting_config = model_config.get('fitting_net', {})
    type_map = model_config.get('type_map', ['O', 'H'])
    
    training_config = config.get('training', {})
    numb_steps = training_config.get('numb_steps', 100000)
    if args.test_steps > 0:
        numb_steps = args.test_steps
    
    system_dirs = [args.data_dir]
    batch_size = args.batch_size
    
    print(f"\n{'=' * 80}")
    print("TRAINING CONFIGURATION (OPTIMIZED)")
    print(f"{'=' * 80}")
    print(f"Training data: {system_dirs}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {args.num_workers}")
    print(f"Grad accumulation steps: {args.grad_accumulation_steps}")
    print(f"Total steps: {numb_steps}")
    print(f"pin_memory: True, non_blocking: True")
    print()
    
    # Create dataset and DataLoader
    dataset = DeepMDDataset(system_dirs, type_map=type_map)
    print(f"Dataset: {len(dataset)} frames, {dataset[0][0].shape[0]} atoms per frame")
    
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        prefetch_factor=2,
        pin_memory=True,
        shuffle=True,
        drop_last=False
    )
    print(f"DataLoader: {len(train_loader)} batches per epoch\n")
    
    # Create model
    print("Creating model...")
    model = DeepMDModel(
        type_map=type_map,
        rcut=descriptor_config['rcut'],
        rcut_smth=descriptor_config['rcut_smth'],
        sel=descriptor_config['sel'],
        descriptor_neuron=descriptor_config['neuron'],
        axis_neuron=descriptor_config['axis_neuron'],
        fitting_neuron=fitting_config['neuron'],
        type_one_side=descriptor_config.get('type_one_side', True),
        resnet_dt=fitting_config.get('resnet_dt', True)
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    model.to(device)
    
    # Scaler for mixed precision
    scaler = None
    if args.mixed_precision and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        print("Using mixed precision\n")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create directories
    Path(args.checkpoint_dir).mkdir(exist_ok=True)
    Path(args.export_dir).mkdir(exist_ok=True)
    
    print(f"{'=' * 80}")
    print(f"Training for {numb_steps} steps...")
    print(f"{'=' * 80}\n")
    
    step = 1
    training_start_time = datetime.now()
    f_rmse_ema = None
    ema_alpha = 0.01
    
    try:
        epoch = 0
        while step <= numb_steps:
            epoch += 1
            print(f"\n[Epoch {epoch}]")
            
            for batch_idx, batch in enumerate(train_loader):
                if step > numb_steps:
                    break
                
                # Unpack batch
                positions_batch, atom_types_batch, boxes_batch, energies_batch, forces_batch = batch
                batch_size_actual = positions_batch.shape[0]
                
                # Process each sample in batch
                for b_idx in range(batch_size_actual):
                    if step > numb_steps:
                        break
                    
                    # Get sample from batch
                    positions = positions_batch[b_idx].to(device, non_blocking=True).requires_grad_(True)
                    atom_types = atom_types_batch[b_idx].to(device, non_blocking=True)
                    box = boxes_batch[b_idx].to(device, non_blocking=True)
                    target_energy = energies_batch[b_idx].to(device, non_blocking=True)
                    target_forces = forces_batch[b_idx].to(device, non_blocking=True)
                    
                    natoms = positions.shape[0]
                    
                    # Get learning rate and prefactors
                    lr = get_learning_rate(step, config, numb_steps)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    
                    pref_e, pref_f = get_loss_prefactors(step, numb_steps, config)
                    
                    # Zero grad
                    optimizer.zero_grad()
                    
                    # Forward pass
                    if scaler is not None:
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            pred_energy, _, pred_forces = model.get_forces(positions, atom_types, box)
                            loss, e_loss, f_loss, e_rmse, f_rmse = compute_loss(
                                pred_energy, target_energy, pred_forces, target_forces,
                                pref_e, pref_f, natoms, force_loss_type=args.force_loss
                            )
                        
                        loss_scaled = loss / args.grad_accumulation_steps
                        scaler.scale(loss_scaled).backward()
                    else:
                        pred_energy, _, pred_forces = model.get_forces(positions, atom_types, box)
                        loss, e_loss, f_loss, e_rmse, f_rmse = compute_loss(
                            pred_energy, target_energy, pred_forces, target_forces,
                            pref_e, pref_f, natoms, force_loss_type=args.force_loss
                        )
                        
                        loss_scaled = loss / args.grad_accumulation_steps
                        loss_scaled.backward()
                    
                    # Optimizer step
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    # Update EMA
                    f_rmse_val = f_rmse.item()
                    if f_rmse_ema is None:
                        f_rmse_ema = f_rmse_val
                    else:
                        f_rmse_ema = (1 - ema_alpha) * f_rmse_ema + ema_alpha * f_rmse_val
                    
                    # Logging
                    if step % 100 == 0 or step <= 5:
                        elapsed = (datetime.now() - training_start_time).total_seconds()
                        speed = step / elapsed if elapsed > 0 else 0
                        remaining = numb_steps - step
                        eta = remaining / speed / 3600 if speed > 0 else 0
                        
                        print(
                            f"Step {step:6d} | lr={lr:.3e} | loss={loss.item():.4f} | "
                            f"e_rmse={e_rmse.item():.4f} f_rmse={f_rmse.item():.4f} f_ema={f_rmse_ema:.4f} | "
                            f"{speed:.2f} step/s | ETA: {eta:.1f}h"
                        )
                    
                    # Checkpoint
                    if step % 10000 == 0:
                        ckpt_path = Path(args.checkpoint_dir) / f'model_step{step}.pt'
                        torch.save(model.state_dict(), ckpt_path)
                        print(f"  Checkpoint saved: {ckpt_path}")
                    
                    step += 1
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    # Final export
    print(f"\nTraining completed at step {step}")
    print(f"Total time: {(datetime.now() - training_start_time).total_seconds() / 3600:.2f}h")
    
    export_path = Path(args.export_dir) / 'model_final.pth'
    torch.save(model.state_dict(), export_path)
    print(f"Model exported to {export_path}")
    print(f"Checkpoints: {args.checkpoint_dir}")
    print(f"Exports: {args.export_dir}")


if __name__ == '__main__':
    main()
