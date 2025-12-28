#!/usr/bin/env python3
"""
Test DeepMD model on collect/data0 dataset and compute error metrics
Similar to 'dp test' command
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from dpmini import DeepMDModel
from dpmini.data import DeepMDDataset
from torch.utils.data import DataLoader


def compute_metrics(pred, true):
    """Compute MAE and RMSE"""
    mae = np.mean(np.abs(pred - true))
    rmse = np.sqrt(np.mean((pred - true)**2))
    return mae, rmse


def main():
    parser = argparse.ArgumentParser(description='Test DeepMD model on dataset')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to saved .pth model checkpoint')
    parser.add_argument('--data-dir', type=str, default='collect/data0',
                       help='Path to test data system directory')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for testing')
    parser.add_argument('--num-frames', type=int, default=None,
                       help='Number of frames to test (default: all)')
    args = parser.parse_args()
    
    # Load model checkpoint
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract config and type_map
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        config = checkpoint['config']
        type_map = checkpoint.get('type_map', config['model']['type_map'])
        model_state = checkpoint['model_state_dict']
    else:
        # Try to load from config file
        print("Checkpoint doesn't contain config, loading from se_e2_a/input_torch.json")
        with open('se_e2_a/input_torch.json') as f:
            config = json.load(f)
        type_map = config['model']['type_map']
        model_state = checkpoint
    
    model_config = config['model']
    descriptor_config = model_config['descriptor']
    fitting_config = model_config['fitting_net']
    
    # Create model
    print(f"Creating model with type_map: {type_map}")
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
    
    # Load weights
    model.load_state_dict(model_state)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"\nLoading test data from {args.data_dir}")
    dataset = DeepMDDataset([args.data_dir], type_map=type_map)
    print(f"Dataset size: {len(dataset)} frames")
    
    if args.num_frames is not None:
        test_size = min(args.num_frames, len(dataset))
    else:
        test_size = len(dataset)
    
    # Test loop
    print(f"\nTesting on {test_size} frames...")
    
    pred_energies = []
    true_energies = []
    pred_forces = []
    true_forces = []
    
    for i in range(test_size):
        positions, atom_types, box, energy, forces = dataset[i]
        
        # Move to device
        positions = positions.to(device)
        atom_types = atom_types.to(device)
        box = box.to(device)
        
        # Need requires_grad for force calculation
        positions = positions.requires_grad_(True)
        
        # Forward pass
        total_energy, atomic_energies = model.forward(positions, atom_types, box)
        
        # Compute forces
        pred_force = -torch.autograd.grad(
            total_energy,
            positions,
            create_graph=False,
            retain_graph=False
        )[0]
            
            # Store predictions
            pred_energies.append(total_energy.item())
            true_energies.append(energy.item())
        pred_forces.append(pred_force.detach().cpu().numpy())
    print(f"  RMSE: {f_rmse:.6f} eV/Å")
    
    # Per-atom energy metrics
    natoms = len(dataset[0][1])  # Number of atoms
    e_mae_per_atom = e_mae / natoms
    e_rmse_per_atom = e_rmse / natoms
    print(f"\nEnergy per atom ({natoms} atoms):")
    print(f"  MAE:  {e_mae_per_atom:.6f} eV/atom")
    print(f"  RMSE: {e_rmse_per_atom:.6f} eV/atom")
    
    print("\n" + "="*60)
    print(f"Tested {test_size} frames from {args.data_dir}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
