#!/usr/bin/env python3
"""
Convert NPZ trajectory to XYZ format for visualization in VMD, Ovito, etc.
"""
import numpy as np
import argparse
from pathlib import Path


def write_xyz_frame(f, atom_types, coords, frame_idx=None, time=None, energy=None):
    """Write a single frame in XYZ format"""
    natoms = len(atom_types)
    
    # First line: number of atoms
    f.write(f"{natoms}\n")
    
    # Second line: comment (can include time, energy, etc.)
    comment_parts = []
    if frame_idx is not None:
        comment_parts.append(f"Frame {frame_idx}")
    if time is not None:
        time_val = float(np.asarray(time).flat[0])
        comment_parts.append(f"Time={time_val:.4f} ps")
    if energy is not None:
        # Energy might be array [KE, PE, Total] - use total or first value
        energy_arr = np.asarray(energy)
        if energy_arr.ndim == 0:
            energy_val = float(energy_arr)
        else:
            # Use last value if multiple (usually total energy)
            energy_val = float(energy_arr.flat[-1] if len(energy_arr.flat) > 0 else 0)
        comment_parts.append(f"E={energy_val:.6f} eV")
    
    comment = " ".join(comment_parts) if comment_parts else "MD trajectory"
    f.write(f"{comment}\n")
    
    # Atom lines: symbol x y z
    for atom_type, coord in zip(atom_types, coords):
        f.write(f"{atom_type:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")


def convert_npz_to_xyz(npz_file, xyz_file, skip=1, max_frames=None):
    """
    Convert NPZ trajectory to XYZ format
    
    Parameters:
    -----------
    npz_file : str
        Input NPZ trajectory file
    xyz_file : str
        Output XYZ file
    skip : int
        Use every N-th frame
    max_frames : int or None
        Maximum number of frames to write
    """
    print(f"Loading trajectory from: {npz_file}")
    data = np.load(npz_file)
    
    # Extract data
    trajectory = data['trajectory']
    atom_types = data.get('atom_types', None)
    type_map = data.get('type_map', None)
    times = data.get('times', None)
    energies = data.get('energies', None)
    
    nframes_total = len(trajectory)
    natoms = trajectory.shape[1]
    
    # Handle atom types
    if atom_types is not None and type_map is not None:
        # Convert integer types to symbols
        atom_symbols = [type_map[t] for t in atom_types]
    else:
        # Default to generic atoms
        atom_symbols = ['X'] * natoms
        print("Warning: Atom types not found, using 'X' for all atoms")
    
    # Apply skip
    frame_indices = list(range(0, nframes_total, skip))
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]
    
    nframes = len(frame_indices)
    
    print(f"\nTrajectory info:")
    print(f"  Total frames: {nframes_total}")
    print(f"  Output frames: {nframes} (every {skip})")
    print(f"  Atoms: {natoms}")
    print(f"  Atom types: {set(atom_symbols)}")
    
    # Write XYZ file
    print(f"\nWriting XYZ file: {xyz_file}")
    with open(xyz_file, 'w') as f:
        for i, frame_idx in enumerate(frame_indices):
            coords = trajectory[frame_idx]
            
            # Get time and energy if available (handle array size mismatch)
            time = times[min(frame_idx, len(times)-1)] if times is not None else None
            energy = energies[min(frame_idx, len(energies)-1)] if energies is not None else None
            
            write_xyz_frame(f, atom_symbols, coords, 
                          frame_idx=frame_idx, time=time, energy=energy)
            
            # Progress indicator
            if (i + 1) % 100 == 0 or (i + 1) == nframes:
                print(f"  Progress: {i+1}/{nframes} frames", end='\r')
    
    print(f"\n\n✓ XYZ file saved: {xyz_file}")
    
    # Calculate file size
    file_size = Path(xyz_file).stat().st_size
    if file_size < 1024**2:
        size_str = f"{file_size / 1024:.1f} KB"
    elif file_size < 1024**3:
        size_str = f"{file_size / 1024**2:.1f} MB"
    else:
        size_str = f"{file_size / 1024**3:.2f} GB"
    
    print(f"  File size: {size_str}")
    print(f"  Frames: {nframes}")
    print(f"  Atoms per frame: {natoms}")
    
    return xyz_file


def main():
    parser = argparse.ArgumentParser(
        description='Convert NPZ trajectory to XYZ format for VMD/Ovito',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert with default settings (all frames)
  python convert_trajectory_to_xyz.py trajectory.npz
  
  # Use every 10th frame to reduce file size
  python convert_trajectory_to_xyz.py trajectory.npz --skip 10
  
  # Limit to first 1000 frames
  python convert_trajectory_to_xyz.py trajectory.npz --max-frames 1000
  
  # Combine skip and max frames
  python convert_trajectory_to_xyz.py trajectory.npz --skip 50 --max-frames 500
  
  # Custom output name
  python convert_trajectory_to_xyz.py trajectory.npz --output my_traj.xyz
  
VMD Usage:
  vmd output.xyz
        """
    )
    
    parser.add_argument('trajectory', help='Input NPZ trajectory file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output XYZ file (default: <input>_trajectory.xyz)')
    parser.add_argument('--skip', type=int, default=1,
                       help='Use every N-th frame (default: 1, all frames)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to write')
    
    args = parser.parse_args()
    
    # Check input file
    if not Path(args.trajectory).exists():
        print(f"❌ Error: File not found: {args.trajectory}")
        return 1
    
    # Determine output filename
    if args.output is None:
        base = Path(args.trajectory).stem
        args.output = f"{base}_trajectory.xyz"
    
    # Convert
    try:
        convert_npz_to_xyz(args.trajectory, args.output, 
                          skip=args.skip, max_frames=args.max_frames)
        
        print("\n" + "="*60)
        print("To visualize in VMD:")
        print(f"  vmd {args.output}")
        print("\nTo visualize in Ovito:")
        print(f"  ovito {args.output}")
        print("="*60)
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
