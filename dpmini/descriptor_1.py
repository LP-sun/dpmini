"""
se_e2_a descriptor implementation for DeepMD-kit PyTorch.

Mathematical formulation:
- For each center atom i, construct neighbor list within cutoff r_c
- Environment matrix R^i: shape (Nc, 4), where Nc = sum(sel)
  R_j = [s(r), s(r)*x/r, s(r)*y/r, s(r)*z/r]
  s(r) is smooth cutoff function:
    - r < r_smth: s(r) = 1/r
    - r_smth <= r < r_c: smooth transition to 0
    - r >= r_c: s(r) = 0
- Embedding network G^i: (Nc, M), type_one_side=True means separate networks per neighbor type
- Descriptor matrix: D^i = (1/Nc) * (G^i)^T @ R^i @ (R^i)^T @ G^i_< 
  Shape: (M, M_<) where M_< = axis_neuron
  
Units: Angstrom (Å) for length
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


def build_neighbor_list(
    positions: torch.Tensor,
    atom_types: torch.Tensor,
    type_map: List[str],
    sel: List[int],
    rcut: float,
    box: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build neighbor list with strict self-exclusion and per-type selection.
    
    CRITICAL FIXES (2025-12-29 v2):
    - Use dist2.fill_diagonal_(inf) for guaranteed self-exclusion
    - Detach positions to avoid topk entering autograd graph
    - Per-type neighbor selection: for each atom, select sel[t] nearest neighbors of type t
    - Safety assertions to catch self-inclusion or mask/index mismatch
    
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
        neighbor_mask: (natom, Nc) float32 in {0,1}, 1 for valid neighbors
    """
    natom = positions.shape[0]
    device = positions.device
    ntypes = len(type_map)
    Nc = int(sum(sel))

    # Neighbor list does not need gradient: detach to avoid discrete ops in autograd
    pos = positions.detach()

    # r_ij = r_j - r_i
    r_ij = pos.unsqueeze(1) - pos.unsqueeze(0)  # (natom, natom, 3)

    # PBC: minimum image convention for diagonal box
    if box is not None:
        if box.shape == (3,):
            box_diag = box
        else:  # (3,3)
            box_diag = torch.diagonal(box)
        r_ij = r_ij - torch.round(r_ij / box_diag.view(1, 1, 3)) * box_diag.view(1, 1, 3)

    dist2 = (r_ij * r_ij).sum(dim=-1)  # (natom, natom)
    dist2.fill_diagonal_(float("inf"))  # CRITICAL:永远排除 self

    rcut2 = rcut * rcut

    idx_chunks = []
    type_chunks = []
    mask_chunks = []

    for t in range(ntypes):
        k = int(sel[t])
        # 选择"邻居原子 j 的类型为 t"
        type_mask = (atom_types == t).view(1, natom).expand(natom, natom)

        dist2_t = dist2.masked_fill(~type_mask, float("inf"))

        # 取最小的 k 个
        k_eff = min(k, natom)
        d2, idx = torch.topk(dist2_t, k=k_eff, dim=1, largest=False)

        valid = d2 < rcut2  # (natom, k_eff)

        # 不足 k 时 padding
        if k_eff < k:
            pad = k - k_eff
            idx = torch.cat([idx, idx.new_full((natom, pad), -1)], dim=1)
            valid = torch.cat([valid, valid.new_zeros((natom, pad), dtype=torch.bool)], dim=1)

        idx = torch.where(valid, idx, idx.new_full(idx.shape, -1))
        idx_chunks.append(idx)
        type_chunks.append(idx.new_full(idx.shape, t))
        mask_chunks.append(valid.float())

    neighbor_indices = torch.cat(idx_chunks, dim=1)  # (natom, Nc)
    neighbor_types = torch.cat(type_chunks, dim=1)   # (natom, Nc)
    neighbor_mask = torch.cat(mask_chunks, dim=1)    # (natom, Nc)

    # Safety assertions (optional but highly recommended during initial validation)
    arange = torch.arange(natom, device=device).view(-1, 1)
    if torch.any(neighbor_indices == arange):
        raise RuntimeError("Neighbor list contains self indices (self-exclusion failed).")
    if torch.any((neighbor_indices == -1) & (neighbor_mask > 0)):
        raise RuntimeError("Mask/index mismatch: idx=-1 but mask=1.")

    return neighbor_indices, neighbor_types, neighbor_mask


class EmbeddingNet(nn.Module):
    """Embedding network for se_e2_a descriptor.
    
    Maps radial information s(r) to embedding space.
    When type_one_side=True, uses separate networks per neighbor type.
    """
    
    def __init__(self, neuron: List[int], type_one_side: bool = True, ntypes: int = 2):
        super().__init__()
        self.neuron = neuron
        self.type_one_side = type_one_side
        self.ntypes = ntypes
        
        if type_one_side:
            # Separate network per type
            self.networks = nn.ModuleList([
                self._build_network(neuron) for _ in range(ntypes)
            ])
        else:
            # Single shared network
            self.network = self._build_network(neuron)
    
    def _build_network(self, neuron: List[int]) -> nn.Module:
        """Build MLP: input_dim=1 -> neuron[0] -> ... -> neuron[-1]"""
        layers = []
        in_dim = 1  # Input is s(r)
        for out_dim in neuron:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.Tanh())
            in_dim = out_dim
        return nn.Sequential(*layers)
    
    def forward(self, s_r: torch.Tensor, neighbor_types: torch.Tensor, 
                neighbor_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            s_r: (natom, Nc) smoothed radial values s(r)
            neighbor_types: (natom, Nc) neighbor type indices
            neighbor_mask: (natom, Nc) binary mask
            
        Returns:
            G: (natom, Nc, M) embedding features
        """
        natom, Nc = s_r.shape
        M = self.neuron[-1]
        
        G = torch.zeros(natom, Nc, M, device=s_r.device, dtype=s_r.dtype)
        
        if self.type_one_side:
            for t in range(self.ntypes):
                type_mask = (neighbor_types == t).float() * neighbor_mask
                if type_mask.sum() > 0:
                    # Get indices where this type is present
                    indices = torch.where(type_mask > 0)
                    s_input = s_r[indices].unsqueeze(-1)  # (n_selected, 1)
                    g_output = self.networks[t](s_input)  # (n_selected, M)
                    # Ensure dtype matches destination (autocast may produce half)
                    g_output = g_output.to(G.dtype)
                    G[indices] = g_output
        else:
            # Apply to all valid neighbors
            valid_indices = torch.where(neighbor_mask > 0)
            s_input = s_r[valid_indices].unsqueeze(-1)
            g_output = self.network(s_input)
            g_output = g_output.to(G.dtype)
            G[valid_indices] = g_output
        
        return G


class SEe2aDescriptor(nn.Module):
    """SE(e2_a) descriptor for DeepMD-kit.
    
    Two-body embedding (e2) with auto-correlation (a).
    """
    
    def __init__(self, 
                 rcut: float,
                 rcut_smth: float,
                 sel: List[int],
                 axis_neuron: int,
                 type_map: List[str] = ["O", "H"]):
        """
        Args:
            rcut: cutoff radius in Angstrom
            rcut_smth: smooth cutoff radius in Angstrom
            sel: max neighbors per type, e.g. [46, 92]
            axis_neuron: output dimension M_< for axis, e.g. 16
            type_map: list of element names, e.g. ["O", "H"]
        """
        super().__init__()
        self.rcut = rcut
        self.rcut_smth = rcut_smth
        self.sel = sel
        self.axis_neuron = axis_neuron
        self.type_map = type_map
        self.ntypes = len(type_map)
        self.Nc = sum(sel)
        
        
    def forward(self, 
                positions: torch.Tensor,
                atom_types: torch.Tensor,
                box: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute se_e2_a descriptor based on 1/rr, cos(theta), cos(phi), sin(phi).
        
        Args:
            positions: (natom, 3) atom positions in Angstrom
            atom_types: (natom,) atom type indices (0-indexed)
            box: (3, 3) or (3,) box vectors for PBC
            
        Returns:
            descriptor: (natom, M * axis_neuron) descriptor per atom
        """
        natom = positions.shape[0]
        device = positions.device
        
        # Build neighbor list
        neighbor_indices, neighbor_types, neighbor_mask = build_neighbor_list(
            positions, atom_types, self.type_map, self.sel, self.rcut, box
        )
        
        Nc = neighbor_indices.shape[1]
        neighbor_positions = torch.zeros(natom, Nc, 3, device=device, dtype=positions.dtype)
        for i in range(natom):
            valid_mask = neighbor_mask[i] > 0
            valid_indices = neighbor_indices[i, valid_mask]
            neighbor_positions[i, valid_mask] = positions[valid_indices]
            neighbor_positions[i, ~valid_mask.bool()] = positions[i]
        
        r_ij = neighbor_positions - positions.unsqueeze(1)  # (natom, Nc, 3)
        
        # Apply PBC if necessary
        if box is not None:
            if box.shape == (3,):
                box_diag = box
                r_ij = r_ij - torch.round(r_ij / box_diag.unsqueeze(0).unsqueeze(0)) * box_diag.unsqueeze(0).unsqueeze(0)
            elif box.shape == (3, 3):
                box_diag = torch.diagonal(box)
                r_ij = r_ij - torch.round(r_ij / box_diag.unsqueeze(0).unsqueeze(0)) * box_diag.unsqueeze(0).unsqueeze(0)
        
        # Compute distances and update with padding
        r = torch.sqrt(torch.sum(r_ij * r_ij, dim=-1) + 1e-12)  # (natom, Nc)
        r = r * neighbor_mask + (self.rcut + 1.0) * (1.0 - neighbor_mask)
        
        # Compute s(r) using smooth cutoff
        s_r = self.cutoff_fn(r)  # (natom, Nc)
        
        # Calculate the 1/rr, cos(theta), cos(phi), sin(phi) and apply rotation matrix
        descrpt_a = torch.zeros(natom, Nc, 4, device=device)
        for i in range(natom):
            for j in range(Nc):
                if neighbor_mask[i, j] > 0:
                    r_diff = r_ij[i, j]  # (3,)
                    
                    # Rotation matrix computation
                    # Select two reference neighbors (r1 and r2) for rotation
                    r1 = r_ij[i, 0]  # Use first neighbor for reference (can choose any)
                    r2 = r_ij[i, 1]  # Use second neighbor for reference
                    
                    # Compute unit vectors
                    xx = r1 / torch.norm(r1)  # r1 unit vector
                    yy = r2 - torch.dot(r2, xx) * xx  # Calculate yy perpendicular to r1
                    yy = yy / torch.norm(yy)  # yy unit vector
                    zz = torch.cross(xx, yy)  # zz is the cross product of xx and yy
                    
                    # Construct rotation matrix
                    rot_mat = torch.cat([xx, yy, zz], dim=0).view(3, 3)  # (3, 3)
                    r_diff_rot = torch.matmul(rot_mat, r_diff)  # Apply rotation
                    
                    # Calculate 1/rr for rotated r_diff
                    rr2_rot = torch.dot(r_diff_rot, r_diff_rot)  # r^2 after rotation
                    rr_rot = torch.sqrt(rr2_rot)  # r after rotation
                    
                    # Calculate cos(theta), cos(phi), sin(phi) from the rotated coordinates
                    cos_theta = r_diff_rot[2] / rr_rot
                    rxy_rot = torch.sqrt(r_diff_rot[0]**2 + r_diff_rot[1]**2)
                    cos_phi = r_diff_rot[0] / rxy_rot
                    sin_phi = r_diff_rot[1] / rxy_rot
                    
                    # Store the calculated values
                    descrpt_a[i, j, 0] = 1. / rr_rot  # 1/rr (rotated)
                    descrpt_a[i, j, 1] = cos_theta  # cos(theta) (rotated)
                    descrpt_a[i, j, 2] = cos_phi  # cos(phi) (rotated)
                    descrpt_a[i, j, 3] = sin_phi  # sin(phi) (rotated)

        # Flatten the descriptors (per atom)
        descrpt_flattened = descrpt_a.view(natom, -1)  # (natom, 4 * Nc)
        
        # Return the descriptor
        return descrpt_flattened

