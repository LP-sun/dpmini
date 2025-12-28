#!/usr/bin/env python3
"""
Analyze MD trajectory from NPZ file.
Extract energy, temperature, structure statistics, etc.
"""
import argparse
import numpy as np
import json
from pathlib import Path


def analyze_trajectory(trajectory_file):
    """Load and analyze MD trajectory."""
    print(f"Loading trajectory: {trajectory_file}")
    data = np.load(trajectory_file)
    
    traj = data['trajectory']        # (nframes, natoms, 3)
    energies = data['energies']      # (nframes-1, 3) - [PE, KE, E_total]
    times = data['times']            # (nframes-1,)
    atom_types = data['atom_types']  # (natoms,)
    type_map = data['type_map']      # atom type names
    
    nframes = len(traj)
    natoms = traj.shape[1]
    
    print(f"\n✓ Trajectory loaded:")
    print(f"  Frames: {nframes}")
    print(f"  Atoms: {natoms}")
    print(f"  Time range: 0 - {times[-1]:.4f} ps")
    print(f"  Atom types: {', '.join([f'{t}({(atom_types==i).sum()})' for i, t in enumerate(type_map)])}")
    
    # Energy analysis
    print(f"\n📊 Energy Analysis:")
    pe = energies[:, 0]
    ke = energies[:, 1]
    etot = energies[:, 2]
    
    print(f"\n  Potential Energy (eV):")
    print(f"    Mean:    {pe.mean():.6f}")
    print(f"    Std:     {pe.std():.6f}")
    print(f"    Min:     {pe.min():.6f}")
    print(f"    Max:     {pe.max():.6f}")
    
    print(f"\n  Kinetic Energy (eV):")
    print(f"    Mean:    {ke.mean():.6f}")
    print(f"    Std:     {ke.std():.6f}")
    print(f"    Min:     {ke.min():.6f}")
    print(f"    Max:     {ke.max():.6f}")
    
    print(f"\n  Total Energy (eV):")
    print(f"    Mean:    {etot.mean():.6f}")
    print(f"    Std:     {etot.std():.6f}")
    print(f"    Drift:   {(etot[-1] - etot[0]) / abs(etot[0]) * 100:.4f}%")
    
    # Temperature analysis (T = 2*KE / (3*N*k_B))
    k_B = 8.617333262e-5  # eV/K
    T = 2 * ke / (3 * natoms * k_B)
    
    print(f"\n  Temperature (K, from KE):")
    print(f"    Mean:    {T.mean():.1f}")
    print(f"    Std:     {T.std():.1f}")
    print(f"    Min:     {T.min():.1f}")
    print(f"    Max:     {T.max():.1f}")
    
    # Structure analysis
    print(f"\n🏗️  Structure Analysis:")
    
    # RMSD from first frame
    rmsd = np.zeros(nframes)
    ref = traj[0]
    for i in range(nframes):
        # Compute RMSD without alignment (simple version)
        rmsd[i] = np.sqrt(np.mean((traj[i] - ref)**2))
    
    print(f"\n  RMSD from initial structure (Å):")
    print(f"    Mean:    {rmsd[1:].mean():.4f}")
    print(f"    Max:     {rmsd.max():.4f}")
    
    # Atomic displacement
    disp = np.sqrt(np.sum((traj[-1] - traj[0])**2, axis=1))
    print(f"\n  Atomic displacements (Å):")
    print(f"    Mean:    {disp.mean():.4f}")
    print(f"    Min:     {disp.min():.4f}")
    print(f"    Max:     {disp.max():.4f}")
    
    # Gyration radius
    print(f"\n  Gyration radius (Å):")
    for frame_idx in [0, nframes//2, -1]:
        pos = traj[frame_idx]
        com = pos.mean(axis=0)
        r_gyr = np.sqrt(np.mean(np.sum((pos - com)**2, axis=1)))
        time_label = f"{times[frame_idx]:.4f} ps" if frame_idx > 0 else "0.0000 ps"
        print(f"    Frame {frame_idx} ({time_label}): {r_gyr:.4f}")
    
    # Statistical summary
    print(f"\n📈 Summary Statistics:")
    print(f"  System size:        {natoms} atoms")
    print(f"  Simulation time:    {times[-1]:.4f} ps")
    print(f"  Energy conserved:   {etot.std()/abs(etot.mean())*100:.4f}% (relative std)")
    print(f"  Temperature avg:    {T.mean():.1f} K")
    print(f"  Structure changed:  {rmsd[-1]:.4f} Å (RMSD)")
    
    return {
        'nframes': nframes,
        'natoms': natoms,
        'times': times,
        'energies': energies,
        'pe': pe,
        'ke': ke,
        'etot': etot,
        'T': T,
        'rmsd': rmsd,
        'displacements': disp,
        'trajectory': traj
    }


def save_summary(data, output_file='trajectory_summary.json'):
    """Save analysis summary as JSON."""
    summary = {
        'nframes': int(data['nframes']),
        'natoms': int(data['natoms']),
        'time_ps': float(data['times'][-1]),
        'energy': {
            'pe_mean': float(data['pe'].mean()),
            'pe_std': float(data['pe'].std()),
            'ke_mean': float(data['ke'].mean()),
            'ke_std': float(data['ke'].std()),
            'etot_drift_percent': float((data['etot'][-1] - data['etot'][0]) / abs(data['etot'][0]) * 100),
        },
        'temperature': {
            'mean': float(data['T'].mean()),
            'std': float(data['T'].std()),
            'min': float(data['T'].min()),
            'max': float(data['T'].max()),
        },
        'structure': {
            'rmsd_max': float(data['rmsd'].max()),
            'rmsd_mean': float(data['rmsd'][1:].mean()),
            'displacement_max': float(data['displacements'].max()),
            'displacement_mean': float(data['displacements'].mean()),
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze MD trajectory')
    parser.add_argument('trajectory', help='NPZ trajectory file')
    parser.add_argument('--save', type=str, default=None,
                       help='Save summary as JSON')
    args = parser.parse_args()
    
    if not Path(args.trajectory).exists():
        print(f"❌ File not found: {args.trajectory}")
        return
    
    data = analyze_trajectory(args.trajectory)
    
    if args.save:
        save_summary(data, args.save)


if __name__ == '__main__':
    main()
