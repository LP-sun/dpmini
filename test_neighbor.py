#!/usr/bin/env python
import sys
sys.path.insert(0, '/home/ubuntu/pj')
import torch
from dpmini.descriptor import build_neighbor_list

# Simple test
positions = torch.randn(10, 3)
atom_types = torch.tensor([0]*5 + [1]*5, dtype=torch.long)
box = torch.tensor([10.0, 10.0, 10.0])

print("Starting neighbor list test...")
indices, types, mask = build_neighbor_list(positions, atom_types, ["O", "H"], [5, 10], 6.0, box)
print("Success!")
print(f"Indices shape: {indices.shape}")
print(f"Types shape: {types.shape}")
print(f"Mask shape: {mask.shape}")
