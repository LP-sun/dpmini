#!/usr/bin/env python3
"""
Simple Molecular Dynamics simulation using DeepMD PyTorch model.
Basic velocity Verlet integrator with energy conservation check.

Usage:
  python md_simulation.py --model exports_cuda_opt/model_cuda_20251228-121805.pth \
                          --system collect/data0 --steps 100 --dt 0.001
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from dpmini import DeepMDModel
from dpmini.data import load_deepmd_set


def load_initial_structure(system_dir, type_map=None):
    """Load initial atomic structure from dataset."""
    from dpmini.data import load_deepmd_set
    
    system_dir = Path(system_dir)
    set_dir = system_dir / 'set.000'
    data = load_deepmd_set(set_dir)
    
    # Use first frame as initial structure
    positions = data['coord'][0].copy()  # (natom, 3)
    box = data['box'][0].copy()  # (3, 3)
    natom = data['natom']
    
    # Infer atom types from data (default: O64H128 pattern)
    # For water: 64 O atoms, 128 H atoms
    if type_map is None:
        type_map = ['O', 'H']
    
    # Default: first natom//3 are O, rest are H
    nO = natom // 3
    nH = natom - nO
    atom_types = np.array([0] * nO + [1] * nH, dtype=np.int64)
    
    return positions, box, atom_types


def center_of_mass(positions, masses):
    """Compute center of mass."""
    return np.average(positions, axis=0, weights=masses)


def remove_translation(positions, velocities, masses):
    """Remove center of mass motion."""
    com_v = np.average(velocities, axis=0, weights=masses)
    velocities = velocities - com_v
    return positions, velocities


def remove_rotation(positions, velocities, masses):
    """Remove center of mass rotation (simple approximation)."""
    # Compute angular momentum about COM
    com = center_of_mass(positions, masses)
    r = positions - com
    
    # Angular velocity (approximation)
    L = np.sum(np.cross(r, velocities * masses[:, np.newaxis]), axis=0)
    I = np.sum(masses[:, np.newaxis] * (np.sum(r**2, axis=1, keepdims=True) - r**2))
    
    # This is a simplified version; full rotational removal is more complex
    return positions, velocities


def run_md(model, positions, box, atom_types, type_map, masses, 
           n_steps=100, dt=0.001, device='cuda', remove_translation_every=10,
           save_trajectory=False, output_prefix='md_trajectory'):
    """
    Run molecular dynamics simulation.
    
    Args:
        model: DeepMDModel instance
        positions: (natom, 3) initial positions
        box: (3, 3) simulation box
        atom_types: (natom,) atom type indices
        type_map: list of atom type names
        masses: (natom,) atomic masses
        n_steps: number of MD steps
        dt: time step in ps
        device: 'cuda' or 'cpu'
        remove_translation_every: remove COM motion every N steps
        save_trajectory: whether to save trajectory
        output_prefix: output file prefix
    
    Returns:
        trajectory: list of positions
        energies: list of total energies
        times: list of times
    """
    
    # Initialize velocities from Maxwell-Boltzmann distribution at T=300K
    k_B = 8.617333262e-5  # eV/K
    T = 300.0  # K
    sigma = np.sqrt(k_B * T / masses)
    velocities = np.random.normal(0, sigma[:, np.newaxis], positions.shape)
    
    # Remove COM motion
    positions, velocities = remove_translation(positions, velocities, masses)
    
    # Convert to torch tensors
    positions = torch.from_numpy(positions).float()
    atom_types = torch.from_numpy(atom_types).long()
    box = torch.from_numpy(box).float()
    velocities = torch.from_numpy(velocities).float()
    masses_torch = torch.from_numpy(masses).float()
    
    positions = positions.to(device)
    atom_types = atom_types.to(device)
    box = box.to(device)
    velocities = velocities.to(device)
    masses_torch = masses_torch.to(device)
    
    # Store trajectory
    trajectory = [positions.cpu().numpy().copy()]
    energies = []
    times = []
    
    print(f"\n{'Step':<8} {'Time (ps)':<12} {'KE (eV)':<12} {'PE (eV)':<12} {'Total (eV)':<12}")
    print("-" * 60)
    
    model.eval()
    # More frequent progress prints: every max(10, n_steps//50) steps, and always first few steps
    print_interval = max(10, n_steps // 50)
    
    # MD loop
    for step in range(n_steps):
        # Enable gradient computation for forces
        positions.requires_grad_(True)
        
        # Compute forces
        with torch.enable_grad():
            total_energy, atomic_energies = model.forward(positions, atom_types, box)
            forces = -torch.autograd.grad(
                total_energy,
                positions,
                create_graph=False,
                retain_graph=False
            )[0]
        
        # Velocity Verlet integration
        # v(t+dt/2) = v(t) + F(t)/(2m)*dt
        accelerations = forces / masses_torch[:, np.newaxis]
        velocities = velocities + 0.5 * accelerations * dt
        
        # r(t+dt) = r(t) + v(t+dt/2)*dt
        positions = positions + velocities * dt
        
        # Recompute forces at new position
        positions.requires_grad_(True)
        with torch.enable_grad():
            total_energy, atomic_energies = model.forward(positions, atom_types, box)
            forces = -torch.autograd.grad(
                total_energy,
                positions,
                create_graph=False,
                retain_graph=False
            )[0]
        
        # v(t+dt) = v(t+dt/2) + F(t+dt)/(2m)*dt
        accelerations = forces / masses_torch[:, np.newaxis]
        velocities = velocities + 0.5 * accelerations * dt
        
        # Periodic removal of COM motion
        if (step + 1) % remove_translation_every == 0:
            velocities_np = velocities.detach().cpu().numpy()
            positions_np = positions.detach().cpu().numpy()
            positions_np, velocities_np = remove_translation(
                positions_np, velocities_np, masses
            )
            positions = torch.from_numpy(positions_np).float().to(device)
            velocities = torch.from_numpy(velocities_np).float().to(device)
        
        # Compute energies
        with torch.no_grad():
            positions_no_grad = positions.detach()
            total_energy, _ = model.forward(positions_no_grad, atom_types, box)
        
        ke = 0.5 * (velocities**2 * masses_torch[:, np.newaxis]).sum().item()
        pe = total_energy.item()
        total_e = ke + pe
        
        # Store results
        trajectory.append(positions.detach().cpu().numpy().copy())
        energies.append((pe, ke, total_e))
        times.append((step + 1) * dt)
        
        # Print progress
        if step < 5 or (step + 1) % print_interval == 0 or step == 0:
            print(f"{step+1:<8} {(step+1)*dt:<12.4f} {ke:<12.4f} {pe:<12.4f} {total_e:<12.4f}")
    
    print("-" * 60)
    print(f"✓ MD simulation completed: {n_steps} steps, {n_steps*dt:.3f} ps")
    
    # Energy analysis
    energies = np.array(energies)
    if len(energies) > 1:
        print(f"\nEnergy Statistics:")
        print(f"  Potential Energy: {energies[:,0].mean():.4f} ± {energies[:,0].std():.4f} eV")
        print(f"  Kinetic Energy:   {energies[:,1].mean():.4f} ± {energies[:,1].std():.4f} eV")
        print(f"  Total Energy:     {energies[:,2].mean():.4f} ± {energies[:,2].std():.4f} eV")
        energy_drift = (energies[-1,2] - energies[0,2]) / energies[0,2]
        print(f"  Energy drift:     {energy_drift*100:.2f}%")
    
    return trajectory, energies, times


def main():
    parser = argparse.ArgumentParser(description='Run MD simulation with DeepMD model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--system', type=str, default='collect/data0',
                       help='Path to system directory')
    parser.add_argument('--steps', type=int, default=100,
                       help='Number of MD steps')
    parser.add_argument('--dt', type=float, default=0.001,
                       help='Time step in ps')
    parser.add_argument('--T', type=float, default=300.0,
                       help='Initial temperature in K')
    parser.add_argument('--output', type=str, default='md_trajectory',
                       help='Output file prefix')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu')
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        config = checkpoint['config']
        type_map = checkpoint.get('type_map', config['model']['type_map'])
        model_state = checkpoint['model_state_dict']
    else:
        with open('se_e2_a/input_torch.json') as f:
            config = json.load(f)
        type_map = config['model']['type_map']
        model_state = checkpoint
    
    model_config = config['model']
    descriptor_config = model_config['descriptor']
    fitting_config = model_config['fitting_net']
    
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
    
    model.load_state_dict(model_state)
    model.to(args.device)
    print(f"✓ Model loaded, using device: {args.device}")
    
    # Load initial structure
    print(f"\nLoading structure from {args.system}")
    positions, box, atom_types = load_initial_structure(args.system, type_map)
    print(f"  Atoms: {len(positions)}")
    print(f"  Atom types: {[type_map[t] for t in atom_types]}")
    
    # Set up atomic masses (for O=16, H=1)
    mass_map = {'O': 16.0, 'H': 1.0}
    masses = np.array([mass_map[type_map[t]] for t in atom_types], dtype=np.float32)
    
    # Run MD
    print(f"\nRunning MD simulation:")
    print(f"  Steps: {args.steps}")
    print(f"  Time step: {args.dt} ps")
    print(f"  Total time: {args.steps * args.dt:.3f} ps")
    
    trajectory, energies, times = run_md(
        model, positions, box, atom_types, type_map, masses,
        n_steps=args.steps,
        dt=args.dt,
        device=args.device,
        save_trajectory=True,
        output_prefix=args.output
    )
    
    # Save trajectory
    traj_file = f'{args.output}.npz'
    np.savez(traj_file,
             trajectory=np.array(trajectory),
             energies=energies,
             times=times,
             atom_types=atom_types,
             type_map=type_map)
    print(f"\n✓ Trajectory saved to: {traj_file}")
    print(f"  Frames: {len(trajectory)}")
    print(f"  Time range: 0 - {times[-1]:.4f} ps")


if __name__ == '__main__':
    main()
