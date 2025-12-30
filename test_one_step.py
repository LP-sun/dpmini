#!/usr/bin/env python
"""Minimal training test - just one batch."""
import sys
sys.path.insert(0, '/home/ubuntu/pj')
import torch
from dpmini.model import DeepMDModel
from dpmini.data import load_deepmd_system

print("Loading data...")
data_dict = load_deepmd_system('collect/O64H128')
positions = torch.from_numpy(data_dict['coord'][0]).float().requires_grad_(True)
# Infer atom types from system name: O64H128 => 64 O atoms + 128 H atoms
atom_types = torch.tensor([0]*64 + [1]*128, dtype=torch.long)
box = torch.from_numpy(data_dict['box'][0]).float()
energy = torch.tensor([data_dict['energy'][0]], dtype=torch.float32)
forces = torch.from_numpy(data_dict['force'][0]).float()

print(f"Loaded: {positions.shape[0]} atoms")
print(f"Box: {box}")

print("\nCreating model...")
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

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nTraining one batch...")
optimizer.zero_grad()
pred_energy, atomic_energies, pred_forces = model.get_forces(positions, atom_types, box)

e_loss = ((pred_energy - energy)**2).item()
f_loss = torch.mean((pred_forces - forces)**2).item()
loss = 0.02 * e_loss + 1000 * f_loss

print(f"Energy: pred={pred_energy.item():.3f}, target={energy.item():.3f}")
print(f"Energy loss: {e_loss:.6e}")
print(f"Force loss: {f_loss:.6e}")
print(f"Total loss: {loss:.6e}")

loss_tensor = torch.tensor(loss, requires_grad=True)
loss_tensor.backward()
optimizer.step()

print("\n✅ One training step completed successfully!")
