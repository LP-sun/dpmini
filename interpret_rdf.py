#!/usr/bin/env python3
"""
Interpret RDF results for water structures.
Compare with experimental/theoretical values.
"""
import numpy as np
import argparse
from pathlib import Path


def load_rdf_data(filename):
    """Load RDF data from text file."""
    data = np.loadtxt(filename)
    r = data[:, 0]
    
    # Assume columns: r, g_OO, g_OH, g_HH
    rdfs = {}
    if data.shape[1] >= 2:
        rdfs['O-O'] = data[:, 1]
    if data.shape[1] >= 3:
        rdfs['O-H'] = data[:, 2]
    if data.shape[1] >= 4:
        rdfs['H-H'] = data[:, 3]
    
    return r, rdfs


def find_peaks(r, g_r, threshold=1.1):
    """Find peaks in RDF."""
    peaks = []
    for i in range(1, len(g_r) - 1):
        if g_r[i] > threshold and g_r[i] > g_r[i-1] and g_r[i] > g_r[i+1]:
            peaks.append({'r': r[i], 'height': g_r[i], 'index': i})
    return peaks


def find_first_minimum(r, g_r, start_idx=10, end_idx=100):
    """Find first minimum (coordination shell boundary)."""
    if end_idx > len(g_r):
        end_idx = len(g_r)
    
    min_idx = np.argmin(g_r[start_idx:end_idx]) + start_idx
    return r[min_idx], g_r[min_idx]


def compute_coordination_number(r, g_r, r_cutoff):
    """
    Compute coordination number up to r_cutoff.
    N = 4π ρ ∫_0^r_cutoff r² g(r) dr
    
    Assuming number density ρ ≈ 0.033 atoms/Å³ for water
    """
    # Find index corresponding to r_cutoff
    idx_cutoff = np.argmin(np.abs(r - r_cutoff))
    
    # Numerical integration (trapezoidal rule)
    dr = r[1] - r[0]
    integrand = r[:idx_cutoff]**2 * g_r[:idx_cutoff]
    integral = np.trapezoid(integrand, dx=dr) if hasattr(np, 'trapezoid') else np.sum(integrand) * dr
    
    # Assume typical water density
    rho = 0.033  # atoms/Å³
    N = 4 * np.pi * rho * integral
    
    return N


def interpret_water_rdf(r, rdfs):
    """Interpret RDF for water system."""
    print("\n" + "="*70)
    print("  RDF Analysis for Water System")
    print("="*70)
    
    # Expected values for water
    expected = {
        'O-O': {
            'first_peak': 2.8,  # Å, nearest neighbor O-O distance
            'description': 'Nearest neighbor oxygen-oxygen distance'
        },
        'O-H': {
            'first_peak': 0.97,  # Å, O-H bond length
            'second_peak': 1.8,  # Å, hydrogen bonding distance
            'description': 'O-H covalent bond and hydrogen bond'
        },
        'H-H': {
            'first_peak': 1.55,  # Å, H-H distance in same molecule
            'description': 'Intra-molecular H-H distance'
        }
    }
    
    for pair_name, g_r in rdfs.items():
        print(f"\n📊 {pair_name} RDF Analysis:")
        print("-" * 70)
        
        # Find peaks
        peaks = find_peaks(r, g_r, threshold=1.1)
        
        if len(peaks) == 0:
            print("  ⚠️  No significant peaks found")
            continue
        
        # First peak
        first_peak = peaks[0]
        print(f"\n  First Peak:")
        print(f"    Position:  {first_peak['r']:.3f} Å")
        print(f"    Height:    g(r) = {first_peak['height']:.2f}")
        
        if pair_name in expected:
            exp_val = expected[pair_name].get('first_peak')
            if exp_val:
                diff = abs(first_peak['r'] - exp_val)
                print(f"    Expected:  ~{exp_val:.2f} Å")
                print(f"    Deviation: {diff:.3f} Å ({diff/exp_val*100:.1f}%)")
                
                if diff < 0.2:
                    print(f"    ✅ Good agreement!")
                elif diff < 0.5:
                    print(f"    ⚠️  Moderate deviation")
                else:
                    print(f"    ❌ Large deviation")
        
        # Second peak (if exists)
        if len(peaks) > 1:
            second_peak = peaks[1]
            print(f"\n  Second Peak:")
            print(f"    Position:  {second_peak['r']:.3f} Å")
            print(f"    Height:    g(r) = {second_peak['height']:.2f}")
            
            if pair_name in expected:
                exp_val = expected[pair_name].get('second_peak')
                if exp_val:
                    diff = abs(second_peak['r'] - exp_val)
                    print(f"    Expected:  ~{exp_val:.2f} Å")
                    print(f"    Deviation: {diff:.3f} Å")
        
        # First minimum (coordination shell)
        r_min, g_min = find_first_minimum(r, g_r, start_idx=20, end_idx=100)
        print(f"\n  First Minimum (coordination shell boundary):")
        print(f"    Position:  {r_min:.3f} Å")
        print(f"    Value:     g(r) = {g_min:.3f}")
        
        # Coordination number
        if pair_name == 'O-O':
            N_coord = compute_coordination_number(r, g_r, r_min)
            print(f"\n  Coordination Number (up to {r_min:.2f} Å):")
            print(f"    N = {N_coord:.1f}")
            print(f"    Expected: ~4-5 (tetrahedral water structure)")
            
            if 3.5 < N_coord < 5.5:
                print(f"    ✅ Reasonable coordination")
            else:
                print(f"    ⚠️  Unusual coordination number")
        
        # Physical interpretation
        if pair_name in expected:
            print(f"\n  💡 Physical Interpretation:")
            print(f"    {expected[pair_name]['description']}")
    
    print("\n" + "="*70)
    print("  Overall Assessment")
    print("="*70)
    
    # Check O-H bond
    if 'O-H' in rdfs:
        oh_peaks = find_peaks(r, rdfs['O-H'])
        if len(oh_peaks) > 0:
            oh_bond = oh_peaks[0]['r']
            if 0.9 < oh_bond < 1.1:
                print("  ✅ O-H covalent bond length: Normal")
            else:
                print("  ⚠️  O-H bond length unusual")
    
    # Check O-O distance
    if 'O-O' in rdfs:
        oo_peaks = find_peaks(r, rdfs['O-O'])
        if len(oo_peaks) > 0:
            oo_dist = oo_peaks[0]['r']
            if 2.6 < oo_dist < 3.0:
                print("  ✅ O-O neighbor distance: Normal water structure")
            else:
                print("  ⚠️  O-O distance suggests unusual structure")
    
    # Check H-H distance
    if 'H-H' in rdfs:
        hh_peaks = find_peaks(r, rdfs['H-H'])
        if len(hh_peaks) > 0:
            hh_dist = hh_peaks[0]['r']
            if 1.4 < hh_dist < 1.7:
                print("  ✅ H-H intra-molecular distance: Normal")
            else:
                print("  ⚠️  H-H distance unusual")
    
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Interpret RDF results')
    parser.add_argument('rdf_file', help='RDF data text file')
    args = parser.parse_args()
    
    if not Path(args.rdf_file).exists():
        print(f"❌ File not found: {args.rdf_file}")
        return
    
    print(f"Loading RDF data: {args.rdf_file}")
    r, rdfs = load_rdf_data(args.rdf_file)
    
    print(f"✓ Loaded {len(rdfs)} RDF curves")
    print(f"  Distance range: {r[0]:.2f} - {r[-1]:.2f} Å")
    print(f"  Number of bins: {len(r)}")
    
    interpret_water_rdf(r, rdfs)


if __name__ == '__main__':
    main()
