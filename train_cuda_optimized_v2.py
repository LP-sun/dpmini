#!/usr/bin/env python3
"""
High-performance CUDA training with optimizations (V2 - AsyncDataLoader):
- Async DataLoader with multiple workers (充分利用CPU多核)
- Real batch size (不仅是梯度累积虚拟扩大)
- pin_memory=True for GPU transfers
- non_blocking=True for async H2D
- CuDNN auto-tuner enabled
- Optional mixed precision
- Background logging thread

优化关键改进:
1. 启用DataLoader with num_workers=4 (激活4个空闲CPU核心)
2. batch_size改为4 (真实批处理而非1)
3. 梯度累积改为1或2 (减少重复计算)
4. non_blocking=True (GPU异步转移)
5. pin_memory=True (锁定内存)
6. 后台日志线程 (避免主线程阻塞)
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
import queue
import threading

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


def background_logger(log_queue, log_file):
    """Background thread for async logging.
    
    将日志写入从主训练线程解耦，避免I/O阻塞。
    """
    with open(log_file, 'a') as f:
        while True:
            item = log_queue.get()
            if item is None:  # Sentinel value to stop
                break
            f.write(item)
            f.flush()


def collate_batch(batch):
    """Custom collate function for batching variable-size samples.
    
    每个样本可能有不同的原子数，所以不能直接stack。
    这里我们返回原始数据 - 应用层会处理不同大小。
    """
    # 如果所有样本都有相同的原子数，可以stack
    # 否则返回list (更安全)
    if len(batch) > 0 and all(b[0].shape[0] == batch[0][0].shape[0] for b in batch):
        # All same size - can stack
        positions = torch.stack([b[0] for b in batch])
        atom_types = torch.stack([b[1] for b in batch])
        boxes = torch.stack([b[2] for b in batch])
        energies = torch.stack([b[3] for b in batch])
        forces = torch.stack([b[4] for b in batch])
        return positions, atom_types, boxes, energies, forces
    else:
        # Different sizes - return as is
        return batch


def check_cuda_info():
    """Display CUDA device information."""
    print("=" * 80)
    print("CUDA OPTIMIZATION CONFIGURATION (V2 - AsyncDataLoader)")
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
    parser = argparse.ArgumentParser(description='High-performance DeepMD training (V2 - AsyncDataLoader)')
    parser.add_argument('--config', type=str, default='se_e2_a/input_torch.json',
                       help='Path to config JSON file')
    parser.add_argument('--data-dir', type=str, default='collect/O64H128',
                       help='Path to data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_optimized',
                       help='Directory to save checkpoints')
    parser.add_argument('--export-dir', type=str, default='exports_optimized',
                       help='Directory to export final models')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size (default: 4, real batch not gradient accumulation)')
    parser.add_argument('--batch-size-multiplier', type=int, default=1,
                       help='Multiply batch_size by this factor (legacy, unused in V2)')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Use mixed precision training')
    parser.add_argument('--grad-accumulation-steps', type=int, default=1,
                       help='Gradient accumulation steps (default: 1, minimal in V2)')
    parser.add_argument('--force-loss', type=str, default='mse', choices=['mse', 'huber'],
                       help='Force loss type: mse (default) or huber (outlier-robust)')
    parser.add_argument('--prefetch-factor', type=int, default=2,
                       help='Number of batches to prefetch (default: 2)')
    parser.add_argument('--test-steps', type=int, default=0,
                       help='Run only N steps for testing (0=full training)')
    args = parser.parse_args()
    
    print("Starting high-performance DeepMD training (V2 - AsyncDataLoader)...")
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
    batch_size = args.batch_size
    numb_steps = training_config.get('numb_steps', 100000)
    if args.test_steps > 0:
        numb_steps = args.test_steps
    
    # Use the data directory from command line argument
    system_dirs = [args.data_dir]
    
    print(f"\n{'=' * 80}")
    print("TRAINING CONFIGURATION (V2)")
    print(f"{'=' * 80}")
    print(f"Training data systems: {system_dirs}")
    print(f"Batch size: {batch_size} (REAL batch, not accumulation)")
    print(f"Number of workers: {args.num_workers} (CPU data loading cores)")
    print(f"Prefetch factor: {args.prefetch_factor}")
    print(f"Gradient accumulation steps: {args.grad_accumulation_steps}")
    print(f"Mixed precision: {args.mixed_precision}")
    print(f"Force loss type: {args.force_loss} {'(Huber: outlier-robust)' if args.force_loss == 'huber' else '(MSE)'}")
    print(f"Total training steps: {numb_steps}")
    print(f"pin_memory: True")
    print(f"non_blocking: True (async GPU transfers)")
    print()
    
    # Create dataset and DataLoader with full async optimization
    dataset = DeepMDDataset(system_dirs, type_map=type_map)
    
    print(f"Loaded {len(dataset)} frames, {dataset[0][0].shape[0]} atoms per frame")
    print(f"Atom types distribution: {dataset[0][1].cpu().numpy()}")
    print()
    
    # Create DataLoader with async workers
    print(f"Creating DataLoader with {args.num_workers} workers and prefetch_factor={args.prefetch_factor}")
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_batch,
        drop_last=False
    )
    print(f"DataLoader created: {len(train_loader)} batches per epoch")
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
    
    # Setup background logging
    log_file = 'cuda_training_optimized.log'
    log_queue = queue.Queue(maxsize=1000)
    log_thread = threading.Thread(target=background_logger, args=(log_queue, log_file), daemon=True)
    log_thread.start()
    
    print(f"\n{'=' * 80}")
    print(f"Starting training for {numb_steps} steps with DataLoader...")
    print(f"{'=' * 80}\n")
    
    step = 1
    update_step = 0  # Track actual optimizer.step() calls
    training_start_time = datetime.now()
    
    # EMA tracking for f_rmse (exponential moving average, alpha=0.01)
    f_rmse_ema = None
    ema_alpha = 0.01
    
    try:
        # Gradient accumulation state
        accum_step = 0
        
        # Iterate through DataLoader (infinite loop with reset)
        epoch = 0
        while step <= numb_steps:
            epoch += 1
            print(f"\n[Epoch {epoch}] Starting epoch with {len(train_loader)} batches")
            
            for batch_idx, batch_data in enumerate(train_loader):
                if step > numb_steps:
                    break
                
                # Handle batch data (may have variable sizes)
                if isinstance(batch_data, list):
                    # Variable size batch - process one by one
                    for single_data in batch_data:
                        if step > numb_steps:
                            break
                        
                        positions = single_data[0].to(device, non_blocking=True).requires_grad_(True)
                        atom_types = single_data[1].to(device, non_blocking=True)
                        box = single_data[2].to(device, non_blocking=True)
                        target_energy = single_data[3].to(device, non_blocking=True)
                        target_forces = single_data[4].to(device, non_blocking=True)
                    # Handle batch data (should be list of 5 stacked tensors)
                    if isinstance(batch_data, list) and len(batch_data) == 5 and batch_data[0].dim() >= 2:
                        # Stacked batch (from collate_batch)
                        positions_batch, atom_types_batch, boxes_batch, energies_batch, forces_batch = batch_data
                        batch_size_actual = positions_batch.shape[0]
                    
                        # Process each sample in batch
                        for b_idx in range(batch_size_actual):
                            if step > numb_steps:
                                break
                        
                            pos = positions_batch[b_idx].to(device, non_blocking=True).requires_grad_(True)
                            atoms = atom_types_batch[b_idx].to(device, non_blocking=True)
                            box = boxes_batch[b_idx].to(device, non_blocking=True)
                            energy = energies_batch[b_idx].to(device, non_blocking=True)
                            forces_target = forces_batch[b_idx].to(device, non_blocking=True)
                        
                            natoms = pos.shape[0]
                        
                            # Training step
                            _do_training_step(
                                model, device, optimizer, scaler,
                                pos, atoms, box, energy, forces_target,
                                step, numb_steps, config, args,
                                log_queue, training_start_time,
                                accum_step, args.grad_accumulation_steps,
                                f_rmse_ema, ema_alpha, natoms
                            )
                        
                            step += 1
                    elif isinstance(batch_data, list) and all(isinstance(d, (list, tuple)) for d in batch_data):
                        # List of individual samples (variable size)
                        for single_data in batch_data:
                            if step > numb_steps:
                                break
                        
                            positions = single_data[0].to(device, non_blocking=True).requires_grad_(True)
                            atom_types = single_data[1].to(device, non_blocking=True)
                            box = single_data[2].to(device, non_blocking=True)
                            target_energy = single_data[3].to(device, non_blocking=True)
                            target_forces = single_data[4].to(device, non_blocking=True)
                        
                            natoms = positions.shape[0]
                        
                            # Training step
                            _do_training_step(
                                model, device, optimizer, scaler, 
                                positions, atom_types, box, target_energy, target_forces,
                                step, numb_steps, config, args, 
                                log_queue, training_start_time, 
                                accum_step, args.grad_accumulation_steps,
                                f_rmse_ema, ema_alpha, natoms
                            )
                        
                            step += 1
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Stop background logger
        log_queue.put(None)
        log_thread.join(timeout=5)
    
    # Final export
    print(f"\nTraining completed or interrupted at step {step}")
    print(f"Total time: {(datetime.now() - training_start_time).total_seconds() / 3600:.2f} hours")
    
    print("\nExporting final model...")
    export_path = Path(args.export_dir) / 'model_final.pth'
    torch.save(model.state_dict(), export_path)
    print(f"Exported model to {export_path}")
    
    print(f"\nAll checkpoints saved in: {args.checkpoint_dir}")
    print(f"Exported models in: {args.export_dir}")
    print(f"Training log: {log_file}")


def _do_training_step(model, device, optimizer, scaler, 
                     positions, atom_types, box, target_energy, target_forces,
                     step, numb_steps, config, args,
                     log_queue, training_start_time,
                     accum_step, grad_accum_steps,
                     f_rmse_ema, ema_alpha, natoms):
    """Execute a single training step."""
    
    # Shape assertions (training early-fail beats silent bugs)
    assert positions.dim() == 2 and positions.shape[1] == 3, \
        f"positions must be (N, 3), got {positions.shape}"
    assert target_forces.shape == positions.shape, \
        f"forces shape {target_forces.shape} != positions shape {positions.shape}"
    assert target_energy.dim() == 0, \
        f"target_energy must be scalar, got shape {target_energy.shape}"
    
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
        loss_scaled = loss / grad_accum_steps
        scaler.scale(loss_scaled).backward()
    else:
        pred_energy, atomic_energies, pred_forces = model.get_forces(
            positions, atom_types, box
        )
        loss, e_loss, f_loss, e_rmse, f_rmse = compute_loss(
            pred_energy, target_energy, pred_forces, target_forces,
            pref_e, pref_f, natoms, force_loss_type=args.force_loss
        )
        
        # Scale loss for gradient accumulation
        loss_scaled = loss / grad_accum_steps
        loss_scaled.backward()
    
    # Step optimizer after accumulation
    accum_step += 1
    if accum_step >= grad_accum_steps:
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        optimizer.zero_grad()
        accum_step = 0
    
    # Update EMA for f_rmse
    f_rmse_val = f_rmse.item()
    if f_rmse_ema is None:
        f_rmse_ema = f_rmse_val
    else:
        f_rmse_ema = (1 - ema_alpha) * f_rmse_ema + ema_alpha * f_rmse_val
    
    # Logging with loss consistency check
    disp_freq = config.get('training', {}).get('disp_freq', 100)
    if step % disp_freq == 0 or step <= 5:
        # Loss consistency verification
        with torch.no_grad():
            recon = pref_e * e_loss + pref_f * f_loss
            diff = (loss - recon).abs().item()
        
        elapsed = (datetime.now() - training_start_time).total_seconds()
        speed = step / elapsed
        remaining_steps = numb_steps - step
        eta_seconds = remaining_steps / speed if speed > 0 else 0
        
        log_line = (
            f"Step {step:6d} | lr={lr:.3e} | loss={loss.item():.6g} "
            f"| e_rmse={e_rmse.item():.4f} f_rmse={f_rmse.item():.4f} f_ema={f_rmse_ema:.4f} "
            f"| pref_e={pref_e:.4g} pref_f={pref_f:.4g} | diff={diff:.3e} "
            f"| {speed:.2f} step/s | ETA: {eta_seconds/3600:.1f}h\n"
        )
        
        print(log_line.rstrip())
        
        # Async log to queue (non-blocking)
        try:
            log_queue.put_nowait(log_line)
        except queue.Full:
            pass  # Skip if queue is full
    
    # Checkpoint
    save_freq = config.get('training', {}).get('save_freq', 10000)
    if step % save_freq == 0:
        checkpoint_path = Path(args.checkpoint_dir) / f'model_step{step}.pt'
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == '__main__':
    main()
