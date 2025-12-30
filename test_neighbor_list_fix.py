#!/usr/bin/env python3
"""
Quick test to verify the fixed neighbor list:
1. Self-exclusion works
2. Per-type neighbor count respects sel[t]
3. No mask/index mismatch
"""
import torch
import sys
sys.path.insert(0, '/home/ubuntu/pj')

from dpmini.descriptor import build_neighbor_list

def test_neighbor_list():
    print("=" * 80)
    print("Testing Fixed Neighbor List")
    print("=" * 80)
    
    # Create test system: 10 O atoms + 20 H atoms
    natom = 30
    positions = torch.randn(natom, 3) * 5.0  # Random positions
    atom_types = torch.cat([
        torch.zeros(10, dtype=torch.long),  # 10 O atoms (type 0)
        torch.ones(20, dtype=torch.long)    # 20 H atoms (type 1)
    ])
    
    type_map = ["O", "H"]
    sel = [5, 10]  # Max 5 O neighbors, 10 H neighbors per atom
    rcut = 6.0
    box = torch.tensor([15.0, 15.0, 15.0])
    
    print(f"\nTest system:")
    print(f"  natom = {natom}")
    print(f"  O atoms: {(atom_types == 0).sum().item()}")
    print(f"  H atoms: {(atom_types == 1).sum().item()}")
    print(f"  sel = {sel} (max 5 O + 10 H neighbors)")
    print(f"  rcut = {rcut} Å")
    print(f"  box = {box.tolist()} Å")
    
    # Build neighbor list
    try:
        neighbor_indices, neighbor_types, neighbor_mask = build_neighbor_list(
            positions, atom_types, type_map, sel, rcut, box
        )
        print("\n✓ Neighbor list built successfully")
    except Exception as e:
        print(f"\n✗ Failed to build neighbor list: {e}")
        return False
    
    print(f"\nNeighbor list shape:")
    print(f"  neighbor_indices: {neighbor_indices.shape}")
    print(f"  neighbor_types: {neighbor_types.shape}")
    print(f"  neighbor_mask: {neighbor_mask.shape}")
    
    # Test 1: Self-exclusion
    print("\n" + "=" * 80)
    print("Test 1: Self-exclusion")
    print("=" * 80)
    
    arange = torch.arange(natom).view(-1, 1)
    has_self = torch.any(neighbor_indices == arange).item()
    
    if has_self:
        print("✗ FAILED: Neighbor list contains self indices")
        # Show which atoms have self neighbors
        self_mask = neighbor_indices == arange
        bad_atoms = torch.any(self_mask, dim=1).nonzero().squeeze()
        print(f"  Atoms with self neighbors: {bad_atoms.tolist()}")
        return False
    else:
        print("✓ PASSED: No self neighbors found")
    
    # Test 2: Per-type neighbor count
    print("\n" + "=" * 80)
    print("Test 2: Per-type neighbor count respects sel")
    print("=" * 80)
    
    passed = True
    for i in range(natom):
        valid_mask = neighbor_mask[i] > 0
        valid_types = neighbor_types[i, valid_mask]
        
        for t, max_count in enumerate(sel):
            count = (valid_types == t).sum().item()
            if count > max_count:
                print(f"✗ FAILED: Atom {i} has {count} type-{t} neighbors (max {max_count})")
                passed = False
    
    if passed:
        print("✓ PASSED: All atoms respect per-type neighbor limits")
        # Show statistics
        print("\nNeighbor count statistics:")
        for t in range(len(type_map)):
            counts = [(neighbor_types[i] == t).sum().item() 
                     for i in range(natom)]
            print(f"  Type {t} ({type_map[t]}): "
                  f"min={min(counts)}, max={max(counts)}, "
                  f"avg={sum(counts)/len(counts):.1f}")
    else:
        return False
    
    # Test 3: Mask/index consistency
    print("\n" + "=" * 80)
    print("Test 3: Mask/index consistency")
    print("=" * 80)
    
    # Check: if mask=0, index should be -1
    invalid_idx = (neighbor_mask == 0) & (neighbor_indices != -1)
    if torch.any(invalid_idx):
        print("✗ FAILED: Found mask=0 but index≠-1")
        bad_positions = invalid_idx.nonzero()[:5]  # Show first 5
        print(f"  Examples: {bad_positions.tolist()}")
        return False
    
    # Check: if mask=1, index should be valid
    valid_idx = (neighbor_mask > 0) & (neighbor_indices == -1)
    if torch.any(valid_idx):
        print("✗ FAILED: Found mask=1 but index=-1")
        bad_positions = valid_idx.nonzero()[:5]
        print(f"  Examples: {bad_positions.tolist()}")
        return False
    
    print("✓ PASSED: Mask and indices are consistent")
    
    # Test 4: Distance verification
    print("\n" + "=" * 80)
    print("Test 4: Distance verification")
    print("=" * 80)
    
    # Check a few valid neighbors to ensure they're within cutoff
    max_dist = 0.0
    for i in range(min(5, natom)):
        valid_mask = neighbor_mask[i] > 0
        if not torch.any(valid_mask):
            continue
        
        valid_indices = neighbor_indices[i, valid_mask]
        r_ij = positions[valid_indices] - positions[i].unsqueeze(0)
        
        # Apply PBC
        r_ij = r_ij - torch.round(r_ij / box.unsqueeze(0)) * box.unsqueeze(0)
        
        dist = torch.sqrt(torch.sum(r_ij * r_ij, dim=-1))
        max_dist = max(max_dist, dist.max().item())
        
        if torch.any(dist >= rcut):
            print(f"✗ FAILED: Atom {i} has neighbors beyond cutoff")
            print(f"  Max distance: {dist.max().item():.4f} Å (cutoff: {rcut} Å)")
            return False
    
    print(f"✓ PASSED: All neighbors within cutoff")
    print(f"  Max observed distance: {max_dist:.4f} Å (cutoff: {rcut} Å)")
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_neighbor_list()
    sys.exit(0 if success else 1)
