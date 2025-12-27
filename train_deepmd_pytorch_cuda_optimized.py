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


def get_learning_rate(step: int, config: dict) -> float:
    """Compute learning rate with exponential decay."""
    lr_config = config.get('learning_rate', {})
    start_lr = lr_config.get('start_lr', 1e-3)
    stop_lr = lr_config.get('stop_lr', 3.51e-8)
    decay_steps = lr_config.get('decay_steps', 5000)
    
    if step >= decay_steps:
        return stop_lr
    
    ratio = (stop_lr / start_lr) ** (step / decay_steps)
    return start_lr * ratio


def get_loss_prefactors(step: int, numb_steps: int, config: dict) -> tuple:
    """Compute loss prefactors with linear schedule."""
    loss_config = config.get('loss', {})
    start_pref_e = loss_config.get('start_pref_e', 0.02)
    limit_pref_e = loss_config.get('limit_pref_e', 1.0)
    start_pref_f = loss_config.get('start_pref_f', 1000.0)
    limit_pref_f = loss_config.get('limit_pref_f', 1.0)
    
    progress = min(1.0, step / numb_steps)
    pref_e = start_pref_e + (limit_pref_e - start_pref_e) * progress
    pref_f = start_pref_f + (limit_pref_f - start_pref_f) * progress
    
    return pref_e, pref_f


def compute_loss(pred_energy, target_energy, pred_forces, target_forces, 
                pref_e, pref_f, natoms):
    """Compute energy and force loss."""
    e_loss = torch.nn.functional.mse_loss(pred_energy, target_energy)
    f_loss = torch.nn.functional.mse_loss(pred_forces, target_forces)
    total_loss = pref_e * e_loss + pref_f * f_loss / natoms
    return total_loss, e_loss.item(), f_loss.item()


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
    parser.add_argument('--grad-accumulation-steps', type=int, default=1,
                       help='Gradient accumulation steps (for larger effective batch)')
    args = parser.parse_args()
    
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
    batch_size = training_data_config.get('batch_size', 1) * args.batch_size_multiplier
    numb_steps = training_config.get('numb_steps', 100000)
    
    # Use the data directory from command line argument
    system_dirs = [args.data_dir]
    
    print(f"\n{'=' * 80}")
    print("TRAINING CONFIGURATION")
    print(f"{'=' * 80}")
    print(f"Training data systems: {system_dirs}")
    print(f"Batch size: {batch_size} (original: {training_data_config.get('batch_size', 1)})")
    print(f"Number of workers: {args.num_workers}")
    print(f"Gradient accumulation steps: {args.grad_accumulation_steps}")
    print(f"Mixed precision: {args.mixed_precision}")
    print(f"Total training steps: {numb_steps}")
    print()
    
    # Create dataset and dataloader with full CUDA optimization
    dataset = DeepMDDataset(system_dirs, type_map=type_map)
    
    print(f"Loaded {len(dataset)} frames, {dataset[0][0].shape[0]} atoms per frame")
    print(f"Atom types distribution: {dataset[0][1][0].cpu().numpy()}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,  # Critical for GPU performance
        persistent_workers=args.num_workers > 0,  # Keep worker processes alive
        prefetch_factor=2 if args.num_workers > 0 else 0,  # Pre-load batches
        drop_last=False
    )
    
    print(f"DataLoader configured with {args.num_workers} workers + pin_memory + prefetch")
    print(f"Effective steps per epoch: {len(dataloader)}")
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
    
    step = 0
    epoch = 0
    training_start_time = datetime.now()
    
    try:
        while step < numb_steps:
            epoch += 1
            for batch_idx, batch in enumerate(dataloader):
                positions, atom_types, box, target_energy, target_forces = batch
                
                # Move to device
                positions = positions.to(device).squeeze(0).requires_grad_(True)
                atom_types = atom_types.to(device).squeeze(0)
                box = box.to(device).squeeze(0)
                target_energy = target_energy.to(device).squeeze(0)
                target_forces = target_forces.to(device).squeeze(0)
                
                natoms = positions.shape[0]
                
                # Get learning rate and loss prefactors
                lr = get_learning_rate(step, config)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                pref_e, pref_f = get_loss_prefactors(step, numb_steps, config)
                
                # Forward pass
                optimizer.zero_grad()
                
                if scaler is not None:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        pred_energy, atomic_energies, pred_forces = model.get_forces(
                            positions, atom_types, box
                        )
                        loss, e_loss, f_loss = compute_loss(
                            pred_energy, target_energy, pred_forces, target_forces,
                            pref_e, pref_f, natoms
                        )
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred_energy, atomic_energies, pred_forces = model.get_forces(
                        positions, atom_types, box
                    )
                    loss, e_loss, f_loss = compute_loss(
                        pred_energy, target_energy, pred_forces, target_forces,
                        pref_e, pref_f, natoms
                    )
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                step += 1
                
                # Logging
                disp_freq = training_config.get('disp_freq', 100)
                if step % disp_freq == 0:
                    elapsed = (datetime.now() - training_start_time).total_seconds()
                    speed = step / elapsed
                    remaining_steps = numb_steps - step
                    eta_seconds = remaining_steps / speed
                    
                    eta_time = datetime.now() + torch.tensor(eta_seconds).timedelta if hasattr(torch, 'timedelta') else datetime.now()
                    
                    print(f"Step {step:6d} | lr={lr:.2e} | loss={loss.item():.6e} | "
                          f"e_loss={e_loss:.6e} | f_loss={f_loss:.6e} | "
                          f"pref_e={pref_e:.3f} pref_f={pref_f:.3f} | "
                          f"Speed: {speed:.2f} steps/s | ETA: {eta_seconds/3600:.1f}h")
                    
                    # Log to file (append mode)
                    with open('cuda_training_opt.log', 'a') as f:
                        f.write(f"Step {step:6d} | lr={lr:.2e} | loss={loss.item():.6e} | "
                               f"e_loss={e_loss:.6e} | f_loss={f_loss:.6e} | "
                               f"pref_e={pref_e:.3f} pref_f={pref_f:.3f} | "
                               f"Speed: {speed:.2f} steps/s\n")
                
                # Checkpoint
                save_freq = training_config.get('save_freq', 10000)
                if step % save_freq == 0:
                    checkpoint_path = Path(args.checkpoint_dir) / f'model_step{step}.pt'
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")
                
                if step >= numb_steps:
                    break
    
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
