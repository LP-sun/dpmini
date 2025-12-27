"""
DeepMD model: descriptor + fitting network.

Outputs atomic energies and total energy.
Forces computed via autograd: F = -dE/dR
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from .descriptor import SEe2aDescriptor


class FittingNet(nn.Module):
    """Fitting network: maps descriptor to atomic energy.
    
    With resnet_dt=True, uses residual connections.
    """
    
    def __init__(self, 
                 input_dim: int,
                 neuron: List[int],
                 resnet_dt: bool = True):
        """
        Args:
            input_dim: descriptor dimension (M * axis_neuron)
            neuron: hidden layer sizes, e.g. [240, 240, 240]
            resnet_dt: use residual connections
        """
        super().__init__()
        self.input_dim = input_dim
        self.neuron = neuron
        self.resnet_dt = resnet_dt
        
        layers = []
        in_dim = input_dim
        
        for i, out_dim in enumerate(neuron):
            layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        
        # Final layer outputs 1 value (atomic energy)
        layers.append(nn.Linear(in_dim, 1))
        
        self.layers = nn.ModuleList(layers)
        
    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            descriptor: (natom, input_dim)
            
        Returns:
            atomic_energy: (natom,)
        """
        x = descriptor
        
        for i, layer in enumerate(self.layers[:-1]):
            x_new = torch.tanh(layer(x))
            
            # Residual connection (if dimensions match and resnet_dt=True)
            if self.resnet_dt and x.shape[-1] == x_new.shape[-1]:
                x = x + x_new
            else:
                x = x_new
        
        # Final layer (no activation)
        atomic_energy = self.layers[-1](x).squeeze(-1)  # (natom,)
        
        return atomic_energy


class DeepMDModel(nn.Module):
    """Complete DeepMD model: descriptor + fitting network."""
    
    def __init__(self,
                 type_map: List[str],
                 rcut: float,
                 rcut_smth: float,
                 sel: List[int],
                 descriptor_neuron: List[int],
                 axis_neuron: int,
                 fitting_neuron: List[int],
                 type_one_side: bool = True,
                 resnet_dt: bool = True):
        """
        Args:
            type_map: element names, e.g. ["O", "H"]
            rcut: cutoff radius in Angstrom
            rcut_smth: smooth cutoff radius
            sel: max neighbors per type, e.g. [46, 92]
            descriptor_neuron: embedding network dims, e.g. [25, 50, 100]
            axis_neuron: axis dimension M_<, e.g. 16
            fitting_neuron: fitting network dims, e.g. [240, 240, 240]
            type_one_side: separate embedding per neighbor type
            resnet_dt: use residual connections in fitting net
        """
        super().__init__()
        
        self.type_map = type_map
        self.ntypes = len(type_map)
        
        # Descriptor
        self.descriptor = SEe2aDescriptor(
            rcut=rcut,
            rcut_smth=rcut_smth,
            sel=sel,
            neuron=descriptor_neuron,
            axis_neuron=axis_neuron,
            type_one_side=type_one_side,
            type_map=type_map
        )
        
        # Descriptor dimension
        descriptor_dim = descriptor_neuron[-1] * axis_neuron
        
        # Fitting network
        self.fitting = FittingNet(
            input_dim=descriptor_dim,
            neuron=fitting_neuron,
            resnet_dt=resnet_dt
        )
        
    def forward(self, 
                positions: torch.Tensor,
                atom_types: torch.Tensor,
                box: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            positions: (natom, 3) positions in Angstrom, requires_grad=True for forces
            atom_types: (natom,) type indices
            box: (3,) or (3, 3) box for PBC
            
        Returns:
            total_energy: scalar
            atomic_energies: (natom,)
        """
        # Compute descriptor
        descriptor = self.descriptor(positions, atom_types, box)  # (natom, desc_dim)
        
        # Compute atomic energies
        atomic_energies = self.fitting(descriptor)  # (natom,)
        
        # Total energy
        total_energy = atomic_energies.sum()
        
        return total_energy, atomic_energies
    
    def get_forces(self,
                   positions: torch.Tensor,
                   atom_types: torch.Tensor,
                   box: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute energy and forces.
        
        Args:
            positions: (natom, 3) positions, will set requires_grad=True
            atom_types: (natom,) type indices
            box: (3,) or (3, 3) box
            
        Returns:
            total_energy: scalar
            atomic_energies: (natom,)
            forces: (natom, 3), F = -dE/dR
        """
        positions = positions.requires_grad_(True)
        total_energy, atomic_energies = self.forward(positions, atom_types, box)
        
        # Compute forces via autograd
        forces = -torch.autograd.grad(
            total_energy, 
            positions, 
            create_graph=True,
            retain_graph=True
        )[0]
        
        return total_energy, atomic_energies, forces
