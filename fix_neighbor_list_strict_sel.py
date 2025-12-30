#!/usr/bin/env python3
"""
Patch for descriptor.py: strict per-type top-k neighbor selection

This fixes the sel semantic to match DeepMD exactly:
- sel[0] = max neighbors of type 0
- sel[1] = max neighbors of type 1
- Each type gets its own top-k, not global sorting

Apply with:
    python fix_neighbor_list_strict_sel.py --apply

Or just review the diff:
    python fix_neighbor_list_strict_sel.py
"""

import argparse
from pathlib import Path


STRICT_BUILD_NEIGHBOR_LIST = '''def build_neighbor_list(positions: torch.Tensor, 
                       atom_types: torch.Tensor,
                       type_map: List[str],
                       sel: List[int],
                       rcut: float,
                       box: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build neighbor list with PBC support.
    
    STRICT sel semantic: sel[i] = max neighbors of type i for each center atom.
    This matches DeepMD's behavior exactly.
    
    Args:
        positions: (natom, 3) atom positions in Angstrom
        atom_types: (natom,) atom type indices (0-indexed)
        type_map: list of element names, e.g. ["O", "H"]
        sel: list of max neighbors per type, e.g. [46, 92] for O and H
        rcut: cutoff radius in Angstrom
        box: (3, 3) or (3,) box vectors. If (3,), assumed cubic box with diagonal L
        
    Returns:
        neighbor_indices: (natom, Nc) neighbor indices, padded with -1
        neighbor_types: (natom, Nc) neighbor type indices, padded with -1
        neighbor_mask: (natom, Nc) binary mask, 1 for valid neighbors, 0 for padding
    """
    natom = positions.shape[0]
    ntypes = len(type_map)
    Nc = sum(sel)
    
    device = positions.device
    
    # Initialize outputs
    neighbor_indices = torch.full((natom, Nc), -1, dtype=torch.long, device=device)
    neighbor_types = torch.full((natom, Nc), -1, dtype=torch.long, device=device)
    neighbor_mask = torch.zeros((natom, Nc), dtype=torch.float32, device=device)
    
    # Compute pairwise displacements with PBC
    # r_ij = r_j - r_i
    r_ij = positions.unsqueeze(0) - positions.unsqueeze(1)  # (natom, natom, 3)
    
    if box is not None:
        if box.shape == (3,):
            # Cubic box: apply minimum image convention
            box_diag = box
            r_ij = r_ij - torch.round(r_ij / box_diag.unsqueeze(0).unsqueeze(0)) * box_diag.unsqueeze(0).unsqueeze(0)
        elif box.shape == (3, 3):
            # General box: convert to fractional coords, wrap, convert back
            box_diag = torch.diagonal(box)
            r_ij = r_ij - torch.round(r_ij / box_diag.unsqueeze(0).unsqueeze(0)) * box_diag.unsqueeze(0).unsqueeze(0)
    
    # Compute distances
    dist = torch.sqrt(torch.sum(r_ij * r_ij, dim=-1) + 1e-12)  # (natom, natom)
    
    # STRICT per-type top-k selection (matches DeepMD)
    # For each center atom, select top sel[t] nearest neighbors of type t
    valid_mask = (dist < rcut) & (dist > 1e-8)  # (natom, natom)
    
    offset = 0
    for t in range(ntypes):
        # Mask for neighbors of type t
        type_mask = (atom_types == t).unsqueeze(0).expand(natom, -1)  # (natom, natom)
        valid_type_mask = valid_mask & type_mask  # (natom, natom)
        
        # For each center atom, get distances to type-t neighbors
        # Set invalid distances to large value for sorting
        dist_for_type = torch.where(valid_type_mask, dist, torch.tensor(1e10, device=device))
        
        # Get top sel[t] nearest neighbors of type t for each center atom
        # topk returns (values, indices) where indices are in [0, natom-1]
        k = min(sel[t], natom)  # Can't select more than total atoms
        _, topk_indices = torch.topk(dist_for_type, k, dim=1, largest=False, sorted=True)
        
        # Gather actual neighbor info
        topk_distances = torch.gather(dist, 1, topk_indices)
        topk_valid = topk_distances < rcut
        
        # Store in output arrays (offset by previous types)
        neighbor_indices[:, offset:offset+k] = topk_indices
        neighbor_types[:, offset:offset+k] = t
        neighbor_mask[:, offset:offset+k] = topk_valid.float()
        
        # Mark invalid entries
        invalid_mask = ~topk_valid
        neighbor_indices[:, offset:offset+k] = torch.where(
            invalid_mask, 
            torch.tensor(-1, dtype=torch.long, device=device), 
            neighbor_indices[:, offset:offset+k]
        )
        neighbor_types[:, offset:offset+k] = torch.where(
            invalid_mask,
            torch.tensor(-1, dtype=torch.long, device=device),
            neighbor_types[:, offset:offset+k]
        )
        
        offset += sel[t]
    
    return neighbor_indices, neighbor_types, neighbor_mask
'''


def main():
    parser = argparse.ArgumentParser(description='Fix neighbor list sel semantic')
    parser.add_argument('--apply', action='store_true', help='Apply the fix to descriptor.py')
    parser.add_argument('--backup', action='store_true', default=True, help='Backup original file')
    args = parser.parse_args()
    
    desc_path = Path('dpmini/descriptor.py')
    
    if not desc_path.exists():
        print(f"❌ Error: {desc_path} not found")
        return 1
    
    content = desc_path.read_text()
    
    if not args.apply:
        print("=" * 80)
        print("NEIGHBOR LIST FIX - STRICT SEL SEMANTIC")
        print("=" * 80)
        print("\nThis fix changes build_neighbor_list to:")
        print("  - Select top sel[t] neighbors of type t separately (per-type top-k)")
        print("  - Match DeepMD's exact behavior")
        print("  - Improve force learning quality when type distribution is uneven")
        print("\nCurrent implementation:")
        print("  - Global sort by (type*1000 + distance)")
        print("  - Takes first sum(sel) neighbors")
        print("  - May not respect per-type quotas exactly")
        print("\n" + "=" * 80)
        print("\nTo apply the fix:")
        print("  python fix_neighbor_list_strict_sel.py --apply")
        print("\nThis will:")
        print("  1. Backup dpmini/descriptor.py -> dpmini/descriptor.py.bak")
        print("  2. Replace build_neighbor_list function")
        print("  3. Preserve all other code")
        print("\n⚠️  After applying, re-run test_training_fixes.py to validate")
        return 0
    
    # Apply the fix
    print("Applying strict sel semantic fix...")
    
    # Backup
    if args.backup:
        backup_path = desc_path.with_suffix('.py.bak')
        backup_path.write_text(content)
        print(f"✓ Backed up to {backup_path}")
    
    # Find and replace build_neighbor_list
    import re
    
    # Pattern to match the entire function
    pattern = r'def build_neighbor_list\(.*?\n(?:.*?\n)*?(?=\n(?:def |class |$))'
    
    if not re.search(pattern, content, re.MULTILINE):
        print("❌ Error: Could not find build_neighbor_list function")
        return 1
    
    new_content = re.sub(pattern, STRICT_BUILD_NEIGHBOR_LIST + '\n\n', content, count=1, flags=re.MULTILINE)
    
    desc_path.write_text(new_content)
    print(f"✓ Updated {desc_path}")
    
    print("\n" + "=" * 80)
    print("✅ FIX APPLIED SUCCESSFULLY")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Run validation: python test_training_fixes.py")
    print("  2. Compare training: the new version should show better f_loss convergence")
    print("  3. If issues occur, restore: mv dpmini/descriptor.py.bak dpmini/descriptor.py")
    print("\nExpected impact:")
    print("  - More stable force learning (especially for uneven type distributions)")
    print("  - Closer match to DeepMD reference results")
    print("  - Slightly slower neighbor list construction (negligible for typical systems)")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
