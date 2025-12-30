#!/usr/bin/env python3
"""Train a DeepMD-style model (se_e2_a) in PyTorch with CUDA acceleration.

Usage (CUDA版本):
    source activate cuda_env
    python3 train_cuda.py --config se_e2_a/input_torch.json --data-dir ../pj_cuda/collect/O64H128

Loads configuration from input_torch.json and trains on DeepMD format data.
Outputs checkpoints and exportable .pth models.
Uses CUDA GPU acceleration for faster training.
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
    """Compute learning rate with exponential decay.
    
    lr = start_lr * (stop_lr / start_lr) ^ (step / decay_steps)
    """
    lr_config = config.get('learning_rate', {})
    start_lr = lr_config.get('start_lr', 1e-3)
    stop_lr = lr_config.get('stop_lr', 3.51e-8)
    decay_steps = lr_config.get('decay_steps', 5000)
    
    if step >= decay_steps:
        return stop_lr
    
    ratio = (stop_lr / start_lr) ** (step / decay_steps)
    return start_lr * ratio


def get_loss_prefactors(step: int, numb_steps: int, config: dict) -> tuple:
    """Compute loss prefactors (pref_e, pref_f) with linear schedule.
    
    Linearly interpolate from start_pref to limit_pref over training.
    """
    loss_config = config.get('loss', {})
    start_pref_e = loss_config.get('start_pref_e', 0.02)
    limit_pref_e = loss_config.get('limit_pref_e', 1.0)
    start_pref_f = loss_config.get('start_pref_f', 1000.0)
    limit_pref_f = loss_config.get('limit_pref_f', 1.0)
    
    # Linear interpolation
    progress = min(1.0, step / numb_steps)
    pref_e = start_pref_e + (limit_pref_e - start_pref_e) * progress
    pref_f = start_pref_f + (limit_pref_f - start_pref_f) * progress
    
    return pref_e, pref_f


def compute_loss(pred_energy, target_energy, pred_forces, target_forces, 
                pref_e, pref_f, natoms):
    """Compute energy and force loss.
    
    Loss = pref_e * |E_pred - E_target|^2 + pref_f * |F_pred - F_target|^2 / natoms
    """
    e_loss = torch.nn.functional.mse_loss(pred_energy, target_energy)
    f_loss = torch.nn.functional.mse_loss(pred_forces, target_forces)
    
    # Normalize force loss by number of atoms
    total_loss = pref_e * e_loss + pref_f * f_loss / natoms
    
    return total_loss, e_loss.item(), f_loss.item()


def check_cuda_info():
    """Display CUDA device information."""
    print("=" * 80)
    print("CUDA Information")
    print("=" * 80)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"PyTorch Version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
            print(f"  Memory Reserved: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
            print(f"  Memory Total: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        
        # Enable cuDNN auto-tuner for better performance
        torch.backends.cudnn.benchmark = True
        print("\nCUDA optimizations enabled:")
        print("  - cuDNN auto-tuner: enabled")
        print("  - CUDA mixed precision: ready for use")
    else:
        print("WARNING: CUDA is not available. Training will run on CPU (slow)")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Train DeepMD-kit PyTorch model with CUDA acceleration')
    parser.add_argument('--config', type=str, default='se_e2_a/input_torch.json',
                       help='Path to config JSON file')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Override data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--export-dir', type=str, default='exports',
                       help='Directory to export final models')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers (increase for faster data loading)')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Use mixed precision training (fp16) for faster training and lower memory')
    parser.add_argument('--verbose', action='store_true',
                       help='Print additional debug information about data shapes, throughput, and GPU memory')
    args = parser.parse_args()
    
    # Check CUDA availability
    check_cuda_info()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"Loading config from {config_path}")
    config = load_config(config_path)
    
    # Extract config parameters
    model_config = config['model']
    training_config = config['training']
    
    type_map = model_config['type_map']
    descriptor_config = model_config['descriptor']
    fitting_config = model_config['fitting_net']
    
    # Training parameters
    numb_steps = training_config.get('numb_steps', 100000)
    batch_size = training_config['training_data'].get('batch_size', 1)
    save_freq = training_config.get('save_freq', 10000)
    disp_freq = training_config.get('disp_freq', 100)
    
    # Data directories
    if args.data_dir:
        system_dirs = [args.data_dir]
    else:
        system_dirs = training_config['training_data']['systems']
        # Convert relative paths to absolute (relative to config file)
        base_dir = config_path.parent
        system_dirs = [str(base_dir / Path(s)) for s in system_dirs]
    
    print(f"Loading training data from: {system_dirs}")
    
    # Create dataset and dataloader with CUDA optimizations
    dataset = DeepMDDataset(system_dirs, type_map=type_map)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=args.num_workers if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available(),  # Important for GPU training
        persistent_workers=True if args.num_workers > 0 else False
    )

    if args.verbose:
        try:
            vb_positions, vb_atom_types, vb_box, vb_target_energy, vb_target_forces = next(iter(dataloader))
            print("Verbose: first batch shapes → "
                  f"positions={tuple(vb_positions.shape)}, "
                  f"atom_types={tuple(vb_atom_types.shape)}, "
                  f"box={tuple(vb_box.shape)}, "
                  f"target_energy={tuple(vb_target_energy.shape)}, "
                  f"target_forces={tuple(vb_target_forces.shape)})")
        except Exception as e:
            print(f"Verbose: failed to inspect first batch: {e}")
    
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
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params} (trainable: {trainable_params})")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # Mixed precision training (optional)
    scaler = None
    if args.mixed_precision and torch.cuda.is_available():
        print("Using mixed precision training (torch.cuda.amp)")
        scaler = torch.cuda.amp.GradScaler()
    
    # Optimizer with CUDA-optimized settings
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    export_dir = Path(args.export_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    export_dir.mkdir(exist_ok=True)
    
    # Training loop
    print(f"Starting training for {numb_steps} steps...")
    print(f"Batch size: {batch_size}, Dataset size: {len(dataset)}")
    print(f"Number of workers: {args.num_workers if torch.cuda.is_available() else 0}")
    print()
    
    step = 0
    epoch = 0
    running_e_loss = 0.0
    running_f_loss = 0.0
    running_total_loss = 0.0
    training_start_time = datetime.now()
    last_disp_time = training_start_time
    
    try:
        while step < numb_steps:
            epoch += 1
            for batch_idx, batch in enumerate(dataloader):
                positions, atom_types, box, target_energy, target_forces = batch
                
                # Move to device
                positions = positions.to(device)
                atom_types = atom_types.to(device)
                box = box.to(device)
                target_energy = target_energy.to(device)
                target_forces = target_forces.to(device)
                
                # Handle both single sample and batch
                # If batch_size > 1, process each sample in the batch
                batch_size = positions.shape[0]
                
                for sample_idx in range(batch_size):
                    if batch_size > 1:
                        # Extract single sample from batch
                        pos = positions[sample_idx].requires_grad_(True)
                        atom_type = atom_types[sample_idx]
                        box_single = box[sample_idx]
                        target_e = target_energy[sample_idx].squeeze()  # Remove [1] dim
                        target_f = target_forces[sample_idx]
                    else:
                        # Single sample (batch_size == 1)
                        pos = positions.squeeze(0).requires_grad_(True)
                        atom_type = atom_types.squeeze(0)
                        box_single = box.squeeze(0)
                        target_e = target_energy.squeeze(0).squeeze()  # Remove both dims
                        target_f = target_forces.squeeze(0)
                    
                    natoms = pos.shape[0]
                    
                    # Get learning rate and loss prefactors
                    lr = get_learning_rate(step, config)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    
                    pref_e, pref_f = get_loss_prefactors(step, numb_steps, config)
                    
                    # Forward pass with optional mixed precision
                    optimizer.zero_grad()
                    
                    if scaler is not None:
                        # Mixed precision forward pass
                        with torch.cuda.amp.autocast():
                            pred_energy, atomic_energies, pred_forces = model.get_forces(
                                pos, atom_type, box_single
                            )
                            
                            # Compute loss
                            loss, e_loss, f_loss = compute_loss(
                                pred_energy, target_e, pred_forces, target_f,
                                pref_e, pref_f, natoms
                            )
                        
                        # Backward pass with scaling
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Standard forward/backward pass
                        pred_energy, atomic_energies, pred_forces = model.get_forces(
                            pos, atom_type, box_single
                        )
                        
                        # Compute loss
                        loss, e_loss, f_loss = compute_loss(
                            pred_energy, target_e, pred_forces, target_f,
                            pref_e, pref_f, natoms
                        )
                        
                        # Backward pass
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    
                    # Update step counter after processing each sample
                    step += 1
                    running_e_loss += e_loss
                    running_f_loss += f_loss
                    running_total_loss += loss.item()
                    
                    if step % disp_freq == 0:
                        avg_total = running_total_loss / disp_freq
                        avg_e = running_e_loss / disp_freq
                        avg_f = running_f_loss / disp_freq
                        
                        elapsed = (datetime.now() - training_start_time).total_seconds()
                        steps_per_sec = step / elapsed if elapsed > 0 else 0
                        eta_secs = (numb_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
                        eta_hours = eta_secs / 3600
                        interval_secs = (datetime.now() - last_disp_time).total_seconds()
                        interval_sps = disp_freq / interval_secs if interval_secs > 0 else steps_per_sec
                        mem_alloc = torch.cuda.memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0.0
                        mem_res = torch.cuda.memory_reserved(0) / (1024**2) if torch.cuda.is_available() else 0.0
                        try:
                            import subprocess
                            util_str = subprocess.check_output([
                                'nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'
                            ], timeout=1).decode().strip().splitlines()[0]
                            gpu_util = float(util_str)
                        except Exception:
                            gpu_util = -1.0

                        grad_norm = 0.0
                        for p in model.parameters():
                            if p.grad is not None:
                                try:
                                    grad_norm += p.grad.norm().item()
                                except Exception:
                                    pass

                        print(
                            f"Step {step:6d} | batch_size={batch_size:2d} | lr={lr:.2e} | loss={avg_total:.6e} | "
                            f"e_loss={avg_e:.6e} | f_loss={avg_f:.6e} | pref_e={pref_e:.3f} pref_f={pref_f:.3f} | "
                            f"ETA {eta_hours:.1f}h | speed {steps_per_sec:.2f} step/s (window {interval_sps:.2f}) | "
                            f"GPU mem {mem_alloc:.0f}/{mem_res:.0f} MiB | util {gpu_util:.0f}% | grad_norm {grad_norm:.2e}"
                        )
                        last_disp_time = datetime.now()
                        running_total_loss = 0.0
                        running_e_loss = 0.0
                        running_f_loss = 0.0
                    
                    # Save checkpoint
                    if step % save_freq == 0:
                        ckpt_path = checkpoint_dir / f"model_step{step}.pt"
                        torch.save({
                            'step': step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'config': config,
                            'device': str(device)
                        }, ckpt_path)
                        print(f"Saved checkpoint to {ckpt_path}")
                    
                    if step >= numb_steps:
                        break
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        # Save emergency checkpoint
        emergency_path = checkpoint_dir / f"model_emergency_step{step}.pt"
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'device': str(device)
        }, emergency_path)
        print(f"Emergency checkpoint saved to {emergency_path}")
        sys.exit(1)
    
    # Export final model
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    export_path = export_dir / f"model_cuda_{timestamp}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'type_map': type_map,
        'device': str(device)
    }, export_path)
    print(f"\nTraining complete! Exported model to {export_path}")
    
    # Print training summary
    total_time = (datetime.now() - training_start_time).total_seconds()
    print(f"\nTraining Summary:")
    print(f"  Total steps: {step}")
    print(f"  Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"  Average speed: {step / total_time:.2f} steps/second")
    print(f"  Device used: {device}")


if __name__ == '__main__':
    main()
