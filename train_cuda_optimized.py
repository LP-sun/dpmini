#!/usr/bin/env python3
"""
High-performance CUDA training with optimizations:
- Larger batch size (default 8, configurable)
- Multi-worker data loading (default 4 workers)
- pin_memory=True for GPU transfers
- Gradient accumulation option
- CuDNN auto-tuner enabled
- Optional mixed precision
"""
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import our DPMini implementation
from dpmini import DeepMDModel, DeepMDDataset


def load_config(config_path: Path) -> dict:
    """Load training configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def get_learning_rate(step: int, config: dict, numb_steps: int) -> float:
    """Compute learning rate with exponential decay.
    
    Fixed: 
    - decay_steps now defaults to numb_steps to avoid premature lr freeze.
    - For short training (<= 2e4 steps), stop_lr raised to 1e-6 for better convergence.
    """
    lr_config = config.get('learning_rate', {})
    start_lr = lr_config.get('start_lr', 1e-3)
    # Short training needs higher stop_lr to keep learning in later stages
    default_stop_lr = 1e-6 if numb_steps <= 20000 else 3.51e-8
    stop_lr = lr_config.get('stop_lr', default_stop_lr)
    # Critical fix: decay_steps should match training duration
    decay_steps = lr_config.get('decay_steps', numb_steps)
    
    if step >= decay_steps:
        return stop_lr
    
    ratio = (stop_lr / start_lr) ** (step / decay_steps)
    return start_lr * ratio


def get_loss_prefactors(step: int, numb_steps: int, config: dict) -> tuple:
    """Compute loss prefactors with linear schedule.
    
    Fixed: limit_pref_f raised from 1 to 20 to maintain force emphasis in later training.
    """
    loss_config = config.get('loss', {})
    start_pref_e = loss_config.get('start_pref_e', 0.02)
    limit_pref_e = loss_config.get('limit_pref_e', 1.0)
    start_pref_f = loss_config.get('start_pref_f', 1000.0)
    limit_pref_f = loss_config.get('limit_pref_f', 20.0)  # Raised from 1.0 to 20.0
    
    progress = min(1.0, step / numb_steps)
    pref_e = start_pref_e + (limit_pref_e - start_pref_e) * progress
    pref_f = start_pref_f + (limit_pref_f - start_pref_f) * progress
    
    return pref_e, pref_f


def compute_loss(pred_energy, target_energy, pred_forces, target_forces, 
                pref_e, pref_f, natoms, force_loss_type='mse'):
    """Compute energy and force loss.
    
    Fixed (2025-12-30):
    - Energy loss: per-atom MSE to match force scaling
    - Force loss: MSE over all 3N components (no /natoms, already mean-reduced)
    - Return rmse values for intuitive error metrics
    - total_loss = pref_e * e_loss + pref_f * f_loss (canonical form)
    """
    # Energy loss: per-atom MSE (normalize by natoms for scale matching)
    e_err_per_atom = (pred_energy - target_energy) / natoms
    e_loss = e_err_per_atom ** 2  # scalar MSE
    
    # Force loss: MSE over all force components (already mean-reduced by F.mse_loss)
    if force_loss_type == 'huber':
        # Huber loss (smooth L1) with beta=0.5 eV/Å to reduce outlier spikes
        f_loss = torch.nn.functional.smooth_l1_loss(pred_forces, target_forces, beta=0.5)
    else:  # 'mse'
        f_loss = torch.nn.functional.mse_loss(pred_forces, target_forces)
    
    # Canonical loss combination (force weight NOT divided by natoms)
    total_loss = pref_e * e_loss + pref_f * f_loss
    
    # RMSE for intuitive error metrics
    e_rmse = torch.sqrt(e_loss)
    f_rmse = torch.sqrt(f_loss)
    
    return total_loss, e_loss, f_loss, e_rmse, f_rmse


def check_cuda_info():
    """Display CUDA device information."""
    print("=" * 80)
    print("CUDA OPTIMIZATION CONFIGURATION")
    print("=" * 80)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"PyTorch Version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        # Enable cuDNN auto-tuner for better performance
        torch.backends.cudnn.benchmark = True
        print(f"cuDNN benchmark: ENABLED (faster conv/attn kernels)")
        
        # Allow TF32 precision for potential speedup (still safe for training)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"TF32 tensor operations: ENABLED (safe for training)")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
            
        print()


def main():
    parser = argparse.ArgumentParser(description='High-performance DeepMD training')
    parser.add_argument('--config', type=str, default='se_e2_a/input_torch.json',
                       help='Path to config JSON file')
    parser.add_argument('--data-dir', type=str, default='collect/O64H128',
                       help='Path to data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_cuda',
                       help='Directory to save checkpoints')
    parser.add_argument('--export-dir', type=str, default='exports_cuda',
                       help='Directory to export final models')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    parser.add_argument('--batch-size-multiplier', type=int, default=1,
                       help='Multiply batch_size from config by this factor')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Use mixed precision training')
    parser.add_argument('--grad-accumulation-steps', type=int, default=8,
                       help='Gradient accumulation steps (default: 8 for force stability)')
    parser.add_argument('--force-loss', type=str, default='mse', choices=['mse', 'huber'],
                       help='Force loss type: mse (default) or huber (outlier-robust)')
    args = parser.parse_args()
    print("starting high-performance DeepMD training with CUDA optimizations...")
    # Check CUDA info and enable optimizations
    check_cuda_info()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"Loading config from {config_path}")
    config = load_config(config_path)
    
    # Extract model config
    model_config = config.get('model', {})
    descriptor_config = model_config.get('descriptor', {})
    fitting_config = model_config.get('fitting_net', {})
    type_map = model_config.get('type_map', ['O', 'H'])
    
    training_config = config.get('training', {})
    training_data_config = training_config.get('training_data', {})
    # Force batch_size=1 to avoid shape ambiguity (use grad_accumulation for larger effective batch)
    batch_size = 1
    numb_steps = training_config.get('numb_steps', 100000)
    
    # Use the data directory from command line argument
    system_dirs = [args.data_dir]
    
    print(f"\n{'=' * 80}")
    print("TRAINING CONFIGURATION")
    print(f"{'=' * 80}")
    print(f"Training data systems: {system_dirs}")
    print(f"Batch size: {batch_size} (FORCED to 1 for shape safety)")
    print(f"Effective batch size: {batch_size * args.grad_accumulation_steps} (via gradient accumulation)")
    print(f"Number of workers: {args.num_workers}")
    print(f"Gradient accumulation steps: {args.grad_accumulation_steps}")
    print(f"Mixed precision: {args.mixed_precision}")
    print(f"Force loss type: {args.force_loss} {'(Huber: outlier-robust)' if args.force_loss == 'huber' else '(MSE)'}")
    print(f"Total training steps: {numb_steps}")
    print()
    
    # Create dataset and dataloader with full CUDA optimization
    dataset = DeepMDDataset(system_dirs, type_map=type_map)
    
    print(f"Loaded {len(dataset)} frames, {dataset[0][0].shape[0]} atoms per frame")
    print(f"Atom types distribution: {dataset[0][1].cpu().numpy()}")
    print(f"Training mode: Direct dataset access (no DataLoader)")
    print(f"Dataset will be accessed randomly via shuffled indices")
    print()
    
    # Create model
    print("Creating DeepMD model...")
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
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # Mixed precision training setup
    scaler = None
    if args.mixed_precision and torch.cuda.is_available():
        print("Using mixed precision training (AMP)")
        scaler = torch.cuda.amp.GradScaler()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create directories
    Path(args.checkpoint_dir).mkdir(exist_ok=True)
    Path(args.export_dir).mkdir(exist_ok=True)
    
    print(f"\n{'=' * 80}")
    print(f"Starting training for {numb_steps} steps (batch size: {batch_size})...")
    print(f"{'=' * 80}\n")
    
    step = 1
    update_step = 0  # Track actual optimizer.step() calls
    training_start_time = datetime.now()
    
    # EMA tracking for f_rmse (exponential moving average, alpha=0.01)
    f_rmse_ema = None
    ema_alpha = 0.01
    
    # Generate shuffled indices for random sampling
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    idx_position = 0
    
    try:
        # Gradient accumulation state
        accum_step = 0
        
        while step <= numb_steps:
            # Reshuffle when we've gone through all data
            if idx_position >= dataset_size:
                np.random.shuffle(indices)
                idx_position = 0
            
            # Get data directly from dataset (no batch dimension)
            data_idx = indices[idx_position]
            idx_position += 1
            
            data = dataset[data_idx]
            positions = data[0].to(device).requires_grad_(True)  # (natom, 3)
            atom_types = data[1].to(device)  # (natom,)
            box = data[2].to(device)  # (3, 3)
            target_energy = data[3].to(device)  # scalar
            target_forces = data[4].to(device)  # (natom, 3)
                
            # Shape assertions (training early-fail beats silent bugs)
            assert positions.dim() == 2 and positions.shape[1] == 3, \
                f"positions must be (N, 3), got {positions.shape}"
            assert target_forces.shape == positions.shape, \
                f"forces shape {target_forces.shape} != positions shape {positions.shape}"
            assert target_energy.dim() == 0, \
                f"target_energy must be scalar, got shape {target_energy.shape}"
                
            natoms = positions.shape[0]
                
            # Get learning rate and loss prefactors
            lr = get_learning_rate(step, config, numb_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            pref_e, pref_f = get_loss_prefactors(step, numb_steps, config)
                
            # Zero grad only at start of accumulation cycle
            if accum_step == 0:
                optimizer.zero_grad()
                
            # Forward pass
            if scaler is not None:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    pred_energy, atomic_energies, pred_forces = model.get_forces(
                        positions, atom_types, box
                    )
                    loss, e_loss, f_loss, e_rmse, f_rmse = compute_loss(
                        pred_energy, target_energy, pred_forces, target_forces,
                        pref_e, pref_f, natoms, force_loss_type=args.force_loss
                    )
                    
                # Scale loss by accumulation steps for correct gradient magnitude
                loss_scaled = loss / args.grad_accumulation_steps
                scaler.scale(loss_scaled).backward()
                    
                # Only step optimizer after accumulating gradients
                accum_step += 1
                if accum_step >= args.grad_accumulation_steps:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()  # Zero for next cycle
                    accum_step = 0
                    update_step += 1
            else:
                pred_energy, atomic_energies, pred_forces = model.get_forces(
                    positions, atom_types, box
                )
                loss, e_loss, f_loss, e_rmse, f_rmse = compute_loss(
                    pred_energy, target_energy, pred_forces, target_forces,
                    pref_e, pref_f, natoms, force_loss_type=args.force_loss
                )
                
                # Scale loss for gradient accumulation
                loss_scaled = loss / args.grad_accumulation_steps
                loss_scaled.backward()
                
                # Step optimizer after accumulation
                accum_step += 1
                if accum_step >= args.grad_accumulation_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()  # Zero for next cycle
                    accum_step = 0
                    update_step += 1
            
            # Update EMA for f_rmse
            f_rmse_val = f_rmse.item()
            if f_rmse_ema is None:
                f_rmse_ema = f_rmse_val
            else:
                f_rmse_ema = (1 - ema_alpha) * f_rmse_ema + ema_alpha * f_rmse_val
            
            # Logging with loss consistency check
            disp_freq = training_config.get('disp_freq', 100)
            if step % disp_freq == 0 or step<=5:
                # Loss consistency verification
                with torch.no_grad():
                    recon = pref_e * e_loss + pref_f * f_loss
                    diff = (loss - recon).abs().item()
                
                elapsed = (datetime.now() - training_start_time).total_seconds()
                speed = step / elapsed
                remaining_steps = numb_steps - step
                eta_seconds = remaining_steps / speed
                
                print(
                    f"Step {step:6d} (update={update_step:5d}) | lr={lr:.3e} | loss={loss.item():.6g} "
                    f"| e_rmse={e_rmse.item():.4f} f_rmse={f_rmse.item():.4f} f_ema={f_rmse_ema:.4f} "
                    f"| pref_e={pref_e:.4g} pref_f={pref_f:.4g} | diff={diff:.3e} "
                    f"| {speed:.2f} step/s | ETA: {eta_seconds/3600:.1f}h"
                )
                
                # Log to file (append mode)
                with open('cuda_training_opt.log', 'a') as f:
                    f.write(
                        f"Step {step:6d} | update={update_step:5d} | lr={lr:.3e} | loss={loss.item():.6g} | "
                        f"e_loss={e_loss.item():.6g} | f_loss={f_loss.item():.6g} | "
                        f"e_rmse={e_rmse.item():.4f} | f_rmse={f_rmse.item():.4f} | f_ema={f_rmse_ema:.4f} | "
                        f"pref_e={pref_e:.4g} pref_f={pref_f:.4g} | diff={diff:.3e}\n"
                    )
            
            # Checkpoint
            save_freq = training_config.get('save_freq', 10000)
            if step % save_freq == 0:
                checkpoint_path = Path(args.checkpoint_dir) / f'model_step{step}.pt'
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
            
            step += 1
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Final export
    print(f"\nTraining completed or interrupted at step {step}")
    print(f"Total time: {(datetime.now() - training_start_time).total_seconds() / 3600:.2f} hours")
    
    print("\nExporting final model...")
    export_path = Path(args.export_dir) / 'model_final.pth'
    torch.save(model.state_dict(), export_path)
    print(f"Exported model to {export_path}")
    
    print(f"\nAll checkpoints saved in: {args.checkpoint_dir}")
    print(f"Exported models in: {args.export_dir}")


if __name__ == '__main__':
    main()
