#!/usr/bin/env python
import sys
sys.path.insert(0, '/home/ubuntu/pj')
import torch
from dpmini.model import DeepMDModel

print("Creating model...")
model = DeepMDModel(
    type_map=["O", "H"],
    rcut=6.0,
    rcut_smth=0.5,
    sel=[46, 92],
    descriptor_neuron=[25, 50, 100],
    axis_neuron=16,
    type_one_side=True,
    fitting_neuron=[240, 240, 240],
    resnet_dt=True
)

print("Creating test data...")
positions = torch.randn(192, 3).requires_grad_(True)
atom_types = torch.tensor([0]*64 + [1]*128, dtype=torch.long)
box = torch.tensor([15.0, 15.0, 15.0])

print("Forward pass...")
energy, atomic_energies, forces = model.get_forces(positions, atom_types, box)

print(f"Energy: {energy.item():.6f} eV")
print(f"Forces shape: {forces.shape}")
print("Success!")
