#!/usr/bin/env python3
"""
Compute Radial Distribution Function (RDF) from MD trajectory.
Supports O-O, O-H, and H-H pair correlations.

Usage:
  python compute_rdf.py demo_md.npz --pairs O-O O-H H-H --rmax 8.0 --nbins 200
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def compute_rdf(positions, box, atom_types, type_map, pair, rmax=10.0, nbins=200):
    """
    Compute RDF for a specific atom pair type.
    
    Args:
        positions: (natom, 3) atomic positions
        box: (3, 3) or (3,) simulation box
        atom_types: (natom,) atom type indices
        type_map: list of atom type names
        pair: tuple like ('O', 'O') or ('O', 'H')
        rmax: maximum distance for RDF
        nbins: number of bins
        
    Returns:
        r: bin centers
        g_r: RDF values
    """
    type1, type2 = pair
    idx1 = type_map.index(type1)
    idx2 = type_map.index(type2)
    
    # Get atoms of each type
    atoms1 = np.where(atom_types == idx1)[0]
    atoms2 = np.where(atom_types == idx2)[0]
    
    # Box size (assume cubic or diagonal)
    if box.ndim == 2:
        box_lengths = np.diag(box)
    else:
        box_lengths = box
    
    # Bin setup
    dr = rmax / nbins
    bins = np.linspace(0, rmax, nbins + 1)
    r = (bins[:-1] + bins[1:]) / 2
    hist = np.zeros(nbins)
    
    # Compute pairwise distances with PBC
    for i in atoms1:
        for j in atoms2:
            if i == j and type1 == type2:
                continue  # Skip self-pairs
            
            # Distance with minimum image convention
            dr_vec = positions[j] - positions[i]
            
            # Apply PBC
            dr_vec = dr_vec - box_lengths * np.round(dr_vec / box_lengths)
            
            dist = np.linalg.norm(dr_vec)
            
            if dist < rmax:
                bin_idx = int(dist / dr)
                if bin_idx < nbins:
                    hist[bin_idx] += 1
    
    # Normalize by ideal gas
    natoms1 = len(atoms1)
    natoms2 = len(atoms2)
    
    # Number density
    volume = np.prod(box_lengths)
    if type1 == type2:
        # Same type: avoid double counting
        rho = (natoms2 - 1) / volume
        normalization = natoms1
    else:
        rho = natoms2 / volume
        normalization = natoms1
    
    # Shell volume
    shell_volume = 4 * np.pi * r**2 * dr
    
    # Normalize to g(r)
    g_r = hist / (normalization * rho * shell_volume)
    
    return r, g_r


def compute_rdf_trajectory(trajectory, box, atom_types, type_map, pairs, rmax=10.0, nbins=200):
    """
    Compute RDF averaged over trajectory.
    
    Args:
        trajectory: (nframes, natoms, 3)
        box: (3, 3) or (nframes, 3, 3)
        atom_types: (natoms,)
        type_map: list of atom type names
        pairs: list of tuples like [('O', 'O'), ('O', 'H')]
        rmax: maximum distance
        nbins: number of bins
        
    Returns:
        dict: {pair: (r, g_r)}
    """
    nframes = len(trajectory)
    results = {}
    
    for pair in pairs:
        print(f"  Computing {pair[0]}-{pair[1]} RDF...")
        
        # Accumulate RDF over all frames
        r = None
        g_r_sum = np.zeros(nbins)
        
        for frame_idx in range(nframes):
            positions = trajectory[frame_idx]
            if box.ndim == 3:
                box_frame = box[frame_idx]
            else:
                box_frame = box
            
            r, g_r = compute_rdf(positions, box_frame, atom_types, type_map, 
                                pair, rmax, nbins)
            g_r_sum += g_r
        
        # Average
        g_r_avg = g_r_sum / nframes
        results[pair] = (r, g_r_avg)
    
    return results


def plot_rdf(rdf_results, output_file='rdf_plot.png'):
    """Plot RDF for multiple pairs."""
    plt.figure(figsize=(10, 6))
    
    colors = {'O-O': 'red', 'O-H': 'blue', 'H-H': 'green'}
    
    for pair, (r, g_r) in rdf_results.items():
        pair_name = f"{pair[0]}-{pair[1]}"
        color = colors.get(pair_name, 'black')
        plt.plot(r, g_r, label=pair_name, color=color, linewidth=2)
    
    plt.xlabel('r (Å)', fontsize=12)
    plt.ylabel('g(r)', fontsize=12)
    plt.title('Radial Distribution Function', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, r[-1])
    plt.ylim(0, None)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"\n✓ RDF plot saved to: {output_file}")
    plt.close()


def save_rdf_data(rdf_results, output_file='rdf_data.txt'):
    """Save RDF data as text file."""
    with open(output_file, 'w') as f:
        # Header
        pairs = list(rdf_results.keys())
        f.write("# Radial Distribution Function\n")
        f.write(f"# Columns: r(Å) " + " ".join([f"g_{p[0]}{p[1]}(r)" for p in pairs]) + "\n")
        
        # Data
        r = rdf_results[pairs[0]][0]
        for i, r_val in enumerate(r):
            line = f"{r_val:.6f}"
            for pair in pairs:
                g_r = rdf_results[pair][1]
                line += f"  {g_r[i]:.6f}"
            f.write(line + "\n")
    
    print(f"✓ RDF data saved to: {output_file}")


def analyze_peaks(r, g_r, threshold=1.1):
    """Find and analyze peaks in RDF."""
    peaks = []
    for i in range(1, len(g_r) - 1):
        if g_r[i] > threshold and g_r[i] > g_r[i-1] and g_r[i] > g_r[i+1]:
            peaks.append((r[i], g_r[i]))
    return peaks


def main():
    parser = argparse.ArgumentParser(description='Compute RDF from MD trajectory')
    parser.add_argument('trajectory', help='NPZ trajectory file')
    parser.add_argument('--pairs', nargs='+', default=['O-O', 'O-H', 'H-H'],
                       help='Atom pairs to compute RDF (e.g., O-O O-H H-H)')
    parser.add_argument('--rmax', type=float, default=8.0,
                       help='Maximum distance for RDF (Å)')
    parser.add_argument('--nbins', type=int, default=200,
                       help='Number of bins')
    parser.add_argument('--output', type=str, default='rdf',
                       help='Output file prefix')
    parser.add_argument('--skip', type=int, default=1,
                       help='Use every N-th frame')
    args = parser.parse_args()
    
    # Load trajectory
    if not Path(args.trajectory).exists():
        print(f"❌ File not found: {args.trajectory}")
        return
    
    print(f"Loading trajectory: {args.trajectory}")
    data = np.load(args.trajectory)
    
    trajectory = data['trajectory']
    atom_types = data['atom_types']
    type_map = list(data['type_map'])
    
    # Infer box (assume cubic from first frame extent)
    if 'box' in data:
        box = data['box']
    else:
        # Estimate box from coordinates
        coords = trajectory[0]
        extent = coords.max(axis=0) - coords.min(axis=0)
        box = np.diag(extent * 1.5)  # Add some padding
    
    nframes_total = len(trajectory)
    trajectory = trajectory[::args.skip]
    nframes = len(trajectory)
    
    print(f"\n✓ Trajectory loaded:")
    print(f"  Total frames: {nframes_total}")
    print(f"  Using frames: {nframes} (every {args.skip})")
    print(f"  Atoms: {trajectory.shape[1]}")
    print(f"  Atom types: {', '.join([f'{t}({(atom_types==i).sum()})' for i, t in enumerate(type_map)])}")
    
    if box.ndim == 1:
        print(f"  Box: {box}")
    else:
        print(f"  Box (diagonal): {np.diag(box)}")
    
    # Parse pairs
    pairs = []
    for pair_str in args.pairs:
        parts = pair_str.split('-')
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))
        else:
            print(f"⚠️  Invalid pair format: {pair_str} (expected A-B)")
    
    if not pairs:
        print("❌ No valid pairs specified")
        return
    
    print(f"\n📊 Computing RDF:")
    print(f"  Pairs: {', '.join([f'{p[0]}-{p[1]}' for p in pairs])}")
    print(f"  r_max: {args.rmax} Å")
    print(f"  bins: {args.nbins}")
    
    # Compute RDF
    rdf_results = compute_rdf_trajectory(
        trajectory, box, atom_types, type_map, 
        pairs, rmax=args.rmax, nbins=args.nbins
    )
    
    # Analyze peaks
    print(f"\n🔍 Peak Analysis:")
    for pair, (r, g_r) in rdf_results.items():
        peaks = analyze_peaks(r, g_r, threshold=1.1)
        pair_name = f"{pair[0]}-{pair[1]}"
        print(f"\n  {pair_name}:")
        if peaks:
            for i, (r_peak, g_peak) in enumerate(peaks[:3], 1):
                print(f"    Peak {i}: r = {r_peak:.3f} Å, g(r) = {g_peak:.3f}")
        else:
            print(f"    No significant peaks found")
        
        # First minimum (coordination shell)
        first_min_idx = np.argmin(g_r[10:40]) + 10 if len(g_r) > 40 else 0
        if first_min_idx > 0:
            print(f"    First minimum: r = {r[first_min_idx]:.3f} Å")
    
    # Save results
    plot_file = f"{args.output}_plot.png"
    data_file = f"{args.output}_data.txt"
    
    plot_rdf(rdf_results, plot_file)
    save_rdf_data(rdf_results, data_file)
    
    print(f"\n✅ RDF analysis completed!")
    print(f"   Plot: {plot_file}")
    print(f"   Data: {data_file}")


if __name__ == '__main__':
    main()
