#!/usr/bin/env python3
"""
Inference script: load trained DeepMD model and predict energy/forces.

Usage:
  python3 inference.py --model exports/model_YYYYMMDD-HHMMSS.pth --input collect/O64H128/set.000/coord.npy
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from dpmini import DeepMDModel
from dpmini.data import load_deepmd_system


def main():
    parser = argparse.ArgumentParser(description='DeepMD model inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to saved .pth model')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to coord.npy or system directory')
    parser.add_argument('--box', type=str, default=None,
                       help='Path to box.npy (optional, default cubic L=12.4447)')
    parser.add_argument('--types', type=str, default=None,
                       help='Path to type indices .npy (optional, default infer from natom)')
    args = parser.parse_args()
    
    # Load model checkpoint
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract config and type_map
    config = checkpoint.get('config')
    if config is None:
        raise ValueError("Checkpoint does not contain 'config' key")
    
    type_map = checkpoint.get('type_map')
    if type_map is None:
        type_map = config['model']['type_map']
    
    model_config = config['model']
    descriptor_config = model_config['descriptor']
    fitting_config = model_config['fitting_net']
    
    # Create model
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
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")
    print(f"Type map: {type_map}")
    
    # Load input data
    input_path = Path(args.input)
    
    if input_path.is_dir():
        # Load from system directory
        print(f"Loading data from system directory: {input_path}")
        data = load_deepmd_system(input_path)
        coords = data['coord']  # (nframe, natom, 3)
        boxes = data['box']     # (nframe, 3, 3)
        atom_types = None
    else:
        # Load from coord.npy
        coords = np.load(input_path).astype(np.float32)  # (nframe, natom*3)
        if coords.ndim == 1:
            coords = coords.reshape(-1, 3)
            natoms = coords.shape[0]
        else:
            nframe = coords.shape[0]
            natoms = coords.shape[1] // 3
            coords = coords.reshape(nframe, natoms, 3)
        
        # Load box if provided
        if args.box:
            boxes = np.load(args.box).astype(np.float32)  # (nframe, 9)
            boxes = boxes.reshape(nframe, 3, 3)
        else:
            # Default cubic box
            L = 12.4447
            if coords.ndim == 2:
                nframe = 1
                coords = coords[np.newaxis, :, :]
            else:
                nframe = coords.shape[0]
            boxes = np.tile(np.eye(3, dtype=np.float32) * L, (nframe, 1, 1))
        
        # Load atom types if provided
        if args.types:
            atom_types = np.load(args.types).astype(np.int64)
        else:
            atom_types = None
    
    # Infer atom types if not provided
    if atom_types is None:
        natoms = coords.shape[1]
        if natoms % 3 == 0:
            nO = natoms // 3
            nH = 2 * nO
            atom_types = np.array([0] * nO + [1] * nH, dtype=np.int64)
        else:
            atom_types = np.ones(natoms, dtype=np.int64)
    
    atom_types_tensor = torch.from_numpy(atom_types).long()
    
    print(f"Coordinates shape: {coords.shape}")
    print(f"Boxes shape: {boxes.shape}")
    print(f"Atom types: {atom_types}")
    
    # Run inference
    print("\nRunning inference...")
    nframes = coords.shape[0]
    
    with torch.no_grad():
        for frame_idx in range(min(nframes, 5)):  # Predict first 5 frames
            pos = torch.from_numpy(coords[frame_idx]).float().to(device)
            types = atom_types_tensor.to(device)
            box = torch.from_numpy(boxes[frame_idx]).float().to(device)
            
            energy, atomic_e, forces = model.get_forces(pos, types, box)
            
            print(f"\nFrame {frame_idx}:")
            print(f"  Total energy: {energy.item():.6f} eV")
            print(f"  Atomic energies (first 5): {atomic_e[:min(5, len(atomic_e))].detach().cpu().numpy()}")
            print(f"  Force magnitude (mean): {forces.norm(dim=1).mean().item():.6f} eV/Å")
            print(f"  Force magnitude (max): {forces.norm(dim=1).max().item():.6f} eV/Å")
    
    print("\nInference complete!")


if __name__ == '__main__':
    main()
