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
        
        # Axis projection: learnable projection from M to axis_neuron
        # This creates G^i_< from G^i
        self.axis_projection = nn.Linear(self.M, axis_neuron, bias=False)
        # Initialize with small values to avoid numerical explosion
        nn.init.normal_(self.axis_projection.weight, mean=0.0, std=0.01)
        
        # Ensure axis_neuron <= M for numerical stability
        if axis_neuron > self.M:
            import warnings
            warnings.warn(f"axis_neuron ({axis_neuron}) > M ({self.M}), may cause issues")
        
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
        # Use actual neighbor list length returned by build_neighbor_list (may be < sum(sel) for small systems)
        Nc = neighbor_indices.shape[1]
        neighbor_positions = torch.zeros(natom, Nc, 3, device=device, dtype=positions.dtype)
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
        
        # CRITICAL FIX: Force padded r to be outside cutoff to avoid s(r)=1/r explosion
        # This prevents numerical instability from r≈0 for padding neighbors
        r = r * neighbor_mask + (self.rcut + 1.0) * (1.0 - neighbor_mask)
        
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
        
        # Compute G^i_< via learnable projection
        # Shape: (natom, Nc, M) -> (natom, Nc, axis_neuron)
        G_axis = self.axis_projection(G)  # (natom, Nc, axis_neuron)
        
        # Debug: check for numerical issues (can be removed after validation)
        if torch.isnan(G).any() or torch.isinf(G).any():
            raise ValueError(f"NaN/Inf detected in G embedding")
        if G.abs().max() > 1e3:
            # Embedding outputs are too large - may indicate training divergence
            import warnings
            warnings.warn(f"Large embedding values: max={G.abs().max().item():.3e}")
        
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
            # where Nc is the actual valid neighbor count for this atom
            n_valid = neighbor_mask[i].sum()
            if n_valid < 1:
                n_valid = 1.0
            
            RtG = torch.matmul(R_i.T, G_axis_i)  # (4, axis_neuron)
            RRtG = torch.matmul(R_i, RtG)  # (Nc, axis_neuron)
            # Correct DeepMD formula: D = (1/Nc) * G^T @ R @ R^T @ G_<
            # This means: D = G^T @ (R @ R^T @ G_<) / Nc
            # NOT dividing by Nc^2
            D_i = torch.matmul(G_i.T, RRtG) / n_valid  # (M, axis_neuron)
            
            descriptors.append(D_i.flatten())  # (M * axis_neuron,)
        
        descriptor = torch.stack(descriptors, dim=0)  # (natom, M * axis_neuron)
        
        # Numerical stability: clip extreme descriptor values to prevent gradient explosion
        # This is especially important during early training with random initialization
        descriptor = torch.clamp(descriptor, min=-1e4, max=1e4)
        
        return descriptor
