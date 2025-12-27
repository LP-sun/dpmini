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


class SmoothCutoffFunction(nn.Module):
    """Smooth cutoff function for descriptor.
    
    s(r) = 1/r                            for r < r_smth
    s(r) = 1/r * u^3(-6u^2 + 15u - 10) + 1 for r_smth <= r < r_c  (smooth transition)
    s(r) = 0                              for r >= r_c
    
    where u = (r - r_smth)/(r_c - r_smth)
    """
    
    def __init__(self, rcut: float, rcut_smth: float):
        super().__init__()
        self.rcut = rcut
        self.rcut_smth = rcut_smth
        assert rcut > rcut_smth > 0, f"Must have 0 < rcut_smth < rcut"
        
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Compute s(r).
        
        Args:
            r: distances, shape (*, )
            
        Returns:
            s(r): smoothed 1/r values, shape (*, )
        """
        # Avoid division by zero
        r_safe = torch.clamp(r, min=1e-8)
        
        # Base value 1/r
        s = 1.0 / r_safe
        
        # Smooth region: r_smth <= r < r_c
        u = (r - self.rcut_smth) / (self.rcut - self.rcut_smth)
        u = torch.clamp(u, 0.0, 1.0)
        
        # Polynomial switching function: uu = u^3(10 - 15u + 6u^2)
        uu = u * u * u * (-6.0 * u * u + 15.0 * u - 10.0) + 1.0
        
        # Apply smoothing where r >= r_smth
        mask_smooth = (r >= self.rcut_smth).float()
        s = s * (1.0 - mask_smooth) + s * uu * mask_smooth
        
        # Zero out where r >= r_c
        mask_cutoff = (r < self.rcut).float()
        s = s * mask_cutoff
        
        return s


def build_neighbor_list(positions: torch.Tensor, 
                       atom_types: torch.Tensor,
                       type_map: List[str],
                       sel: List[int],
                       rcut: float,
                       box: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build neighbor list with PBC support.
    
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
            # This is simplified - for production use proper PBC with box matrix
            # Here we assume orthogonal box for simplicity
            box_diag = torch.diagonal(box)
            r_ij = r_ij - torch.round(r_ij / box_diag.unsqueeze(0).unsqueeze(0)) * box_diag.unsqueeze(0).unsqueeze(0)
    
    # Compute distances
    dist = torch.sqrt(torch.sum(r_ij * r_ij, dim=-1) + 1e-12)  # (natom, natom)
    
    # Build neighbor list for each atom (optimized vectorized version where possible)
    # Find all valid neighbors (within cutoff, not self)
    valid_mask = (dist < rcut) & (dist > 1e-8)  # (natom, natom)
    
    for i in range(natom):
        # Get valid neighbors for this atom
        valid_i = valid_mask[i]
        n_valid = valid_i.sum().item()
        
        if n_valid == 0:
            continue
            
        valid_indices = torch.where(valid_i)[0]
        valid_types = atom_types[valid_indices]
        valid_dists = dist[i, valid_indices]
        
        # Sort by type first, then by distance
        sort_key = valid_types.float() * 1000.0 + valid_dists
        sort_idx = torch.argsort(sort_key)
        valid_indices = valid_indices[sort_idx]
        valid_types = valid_types[sort_idx]
        
        # Fill neighbors respecting per-type limits (simplified)
        # For performance, we'll just take first Nc neighbors after sorting
        # This naturally respects type ordering
        n_to_take = min(len(valid_indices), Nc)
        neighbor_indices[i, :n_to_take] = valid_indices[:n_to_take]
        neighbor_types[i, :n_to_take] = valid_types[:n_to_take]
        neighbor_mask[i, :n_to_take] = 1.0
    
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
                 neuron: List[int],
                 axis_neuron: int,
                 type_one_side: bool = True,
                 type_map: List[str] = ["O", "H"]):
        """
        Args:
            rcut: cutoff radius in Angstrom
            rcut_smth: smooth cutoff radius in Angstrom
            sel: max neighbors per type, e.g. [46, 92]
            neuron: embedding network hidden dims, e.g. [25, 50, 100]
            axis_neuron: output dimension M_< for axis, e.g. 16
            type_one_side: use separate embedding nets per neighbor type
            type_map: list of element names, e.g. ["O", "H"]
        """
        super().__init__()
        self.rcut = rcut
        self.rcut_smth = rcut_smth
        self.sel = sel
        self.neuron = neuron
        self.axis_neuron = axis_neuron
        self.type_one_side = type_one_side
        self.type_map = type_map
        self.ntypes = len(type_map)
        self.Nc = sum(sel)
        
        # Embedding dimension M
        self.M = neuron[-1]
        
        # Smooth cutoff function
        self.cutoff_fn = SmoothCutoffFunction(rcut, rcut_smth)
        
        # Embedding network
        self.embedding_net = EmbeddingNet(neuron, type_one_side, self.ntypes)
        
        # Axis projection: from M to axis_neuron
        # This is G^i_< in the formula
        self.axis_projection = nn.Linear(self.M, axis_neuron, bias=False)
        self.axis_neuron = axis_neuron
        
    def forward(self, 
                positions: torch.Tensor,
                atom_types: torch.Tensor,
                box: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute se_e2_a descriptor.
        
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
        # neighbor_indices: (natom, Nc), values in [0, natom-1] or -1 for padding
        # neighbor_types: (natom, Nc), values in [0, ntypes-1] or -1 for padding
        # neighbor_mask: (natom, Nc), 1 for valid, 0 for padding
        
        # Compute displacements r_ij = r_j - r_i
        # Handle padding: set padded neighbor positions to center atom position (distance 0)
        neighbor_positions = torch.zeros(natom, self.Nc, 3, device=device, dtype=positions.dtype)
        for i in range(natom):
            valid_mask = neighbor_mask[i] > 0
            valid_indices = neighbor_indices[i, valid_mask]
            neighbor_positions[i, valid_mask] = positions[valid_indices]
            # Padding: set to self position
            neighbor_positions[i, ~valid_mask.bool()] = positions[i]
        
        r_ij = neighbor_positions - positions.unsqueeze(1)  # (natom, Nc, 3)
        
        # Apply PBC
        if box is not None:
            if box.shape == (3,):
                box_diag = box
                r_ij = r_ij - torch.round(r_ij / box_diag.unsqueeze(0).unsqueeze(0)) * box_diag.unsqueeze(0).unsqueeze(0)
            elif box.shape == (3, 3):
                box_diag = torch.diagonal(box)
                r_ij = r_ij - torch.round(r_ij / box_diag.unsqueeze(0).unsqueeze(0)) * box_diag.unsqueeze(0).unsqueeze(0)
        
        # Compute distances
        r = torch.sqrt(torch.sum(r_ij * r_ij, dim=-1) + 1e-12)  # (natom, Nc)
        
        # Compute s(r)
        s_r = self.cutoff_fn(r)  # (natom, Nc)
        
        # Build environment matrix R^i: (natom, Nc, 4)
        # R_j = [s(r), s(r)*x/r, s(r)*y/r, s(r)*z/r]
        r_safe = torch.clamp(r, min=1e-8)
        dir_vec = r_ij / r_safe.unsqueeze(-1)  # (natom, Nc, 3)
        R = torch.cat([
            s_r.unsqueeze(-1),  # (natom, Nc, 1)
            s_r.unsqueeze(-1) * dir_vec  # (natom, Nc, 3)
        ], dim=-1)  # (natom, Nc, 4)
        
        # Mask padding
        R = R * neighbor_mask.unsqueeze(-1)
        
        # Compute embedding G^i: (natom, Nc, M)
        G = self.embedding_net(s_r, neighbor_types, neighbor_mask)  # (natom, Nc, M)
        
        # Compute G^i_< via axis projection: (natom, Nc, axis_neuron)
        G_axis = self.axis_projection(G)  # (natom, Nc, axis_neuron)
        # se_e2_a: G_< 直接取前 M_< 列
        M_axis = min(self.axis_neuron, G.shape[-1])  # 防御性写法
        G_axis = G[..., :M_axis]                     # (B, Nc, M_<)
        # Compute descriptor matrix D^i = (1/Nc) * G^T @ R @ R^T @ G_<
        # Result shape: (natom, M, axis_neuron)
        
        # R @ R^T: (natom, Nc, 4) @ (natom, 4, Nc) -> need to compute per atom
        # Actually: R @ R^T gives (natom, Nc, Nc) which is large
        # More efficient: compute R^T @ G_< first
        # R: (natom, Nc, 4), G_<: (natom, Nc, axis_neuron)
        
        # Compute descriptor per atom
        descriptors = []
        for i in range(natom):
            R_i = R[i]  # (Nc, 4)
            G_i = G[i]  # (Nc, M)
            G_axis_i = G_axis[i]  # (Nc, axis_neuron)
            
            # D_i = (1/Nc) * G_i^T @ (R_i @ R_i^T) @ G_axis_i
            # = (1/Nc) * G_i^T @ R_i @ (R_i^T @ G_axis_i)
            RtG = torch.matmul(R_i.T, G_axis_i)  # (4, axis_neuron)
            RRtG = torch.matmul(R_i, RtG)  # (Nc, axis_neuron)
            D_i = torch.matmul(G_i.T, RRtG) / (self.Nc * self.Nc) # (M, axis_neuron)
            
            descriptors.append(D_i.flatten())  # (M * axis_neuron,)
        
        descriptor = torch.stack(descriptors, dim=0)  # (natom, M * axis_neuron)
        
        return descriptor
