#!/usr/bin/env python3
"""Train a DeepMD-style model (se_e2_a) in PyTorch.

Usage:
    python3 train_cpu.py --config se_e2_a/input_torch.json

Loads configuration from input_torch.json and trains on DeepMD format data.
Outputs checkpoints and exportable .pth models.
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



def main():
    parser = argparse.ArgumentParser(description='Train DeepMD-kit PyTorch model')
    parser.add_argument('--config', type=str, default='se_e2_a/input_torch.json',
                       help='Path to config JSON file')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Override data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--export-dir', type=str, default='exports',
                       help='Directory to export final models')
    args = parser.parse_args()
    
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
    
    # Create dataset and dataloader
    dataset = DeepMDDataset(system_dirs, type_map=type_map)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=0, pin_memory=torch.cuda.is_available())
    
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    export_dir = Path(args.export_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    export_dir.mkdir(exist_ok=True)
    
    # Training loop
    print(f"Starting training for {numb_steps} steps...")
    print(f"Batch size: {batch_size}, Dataset size: {len(dataset)}")
    
    step = 0
    epoch = 0
    running_e_loss = 0.0
    running_f_loss = 0.0
    running_total_loss = 0.0
    
    while step < numb_steps:
        epoch += 1
        for batch_idx, batch in enumerate(dataloader):
            positions, atom_types, box, target_energy, target_forces = batch
            
            # Move to device. Keep the batch dimension so batch_size > 1 works.
            positions = positions.to(device).requires_grad_(True)
            atom_types = atom_types.to(device)
            box = box.to(device)
            target_energy = target_energy.to(device)
            target_forces = target_forces.to(device)
            
            natoms = positions.shape[-2]
            
            # Get learning rate and loss prefactors
            lr = get_learning_rate(step, config)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            pref_e, pref_f = get_loss_prefactors(step, numb_steps, config)
            
            # Forward pass
            optimizer.zero_grad()
            pred_energy, atomic_energies, pred_forces = model.get_forces(
                positions, atom_types, box
            )
            
            # Compute loss
            loss, e_loss, f_loss = compute_loss(
                pred_energy, target_energy, pred_forces, target_forces,
                pref_e, pref_f, natoms
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Logging
            running_total_loss += loss.item()
            running_e_loss += e_loss
            running_f_loss += f_loss
            
            if (step + 1) % disp_freq == 0:
                avg_total = running_total_loss / disp_freq
                avg_e = running_e_loss / disp_freq
                avg_f = running_f_loss / disp_freq
                print(f"Step {step+1:6d} | lr={lr:.2e} | loss={avg_total:.6e} | "
                      f"e_loss={avg_e:.6e} | f_loss={avg_f:.6e} | "
                      f"pref_e={pref_e:.3f} pref_f={pref_f:.3f}")
                running_total_loss = 0.0
                running_e_loss = 0.0
                running_f_loss = 0.0
            
            # Save checkpoint
            if (step + 1) % save_freq == 0:
                ckpt_path = checkpoint_dir / f"model_step{step+1}.pt"
                torch.save({
                    'step': step + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config
                }, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")
            
            step += 1
            if step >= numb_steps:
                break
    
    # Export final model
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    export_path = export_dir / f"model_{timestamp}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'type_map': type_map
    }, export_path)
    print(f"Training complete! Exported model to {export_path}")


if __name__ == '__main__':
    main()
