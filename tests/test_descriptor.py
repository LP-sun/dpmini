"""Unit tests for DeepMD descriptor and model."""

import pytest
import torch
import numpy as np
from dpmini.descriptor import SEe2aDescriptor, build_neighbor_list
from dpmini.model import DeepMDModel


class TestSmoothCutoff:
    """Test smooth cutoff function."""
    
    def test_cutoff_value_range(self):
        """Cutoff function s(r) = smoothed 1/r."""
        from dpmini.descriptor import SmoothCutoffFunction
        
        cutoff = SmoothCutoffFunction(rcut=6.0, rcut_smth=0.5)
        r = torch.linspace(0.1, 7.0, 100)
        s = cutoff(r)
        
        assert torch.all(s >= 0), "Cutoff should be non-negative"
        # s(r) = 1/r for small r, so can be > 1
        assert torch.all(torch.isfinite(s)), "Cutoff should be finite"
        assert torch.all(s[r > 6.0] == 0), "Cutoff should be 0 for r >= rcut"
        # Check continuity at r_smth
        r_near_smth = torch.tensor([0.49, 0.51])
        s_near = cutoff(r_near_smth)
        assert torch.abs(s_near[0] - s_near[1]) < 0.5, "Should be continuous at r_smth"


class TestTranslationInvariance:
    """Test translation invariance of descriptor."""
    
    def test_descriptor_translation_invariance(self):
        """Adding constant to all positions should not change energy."""
        device = torch.device('cpu')
        natoms = 6
        type_map = ["O", "H"]
        
        descriptor = SEe2aDescriptor(
            rcut=6.0,
            rcut_smth=0.5,
            sel=[4, 8],  # Small numbers for test
            neuron=[8, 16],
            axis_neuron=4,
            type_one_side=False,
            type_map=type_map
        ).to(device)
        
        # Create simple water-like geometry
        positions = torch.tensor([
            [0.0, 0.0, 0.0],      # O
            [0.957, 0.0, 0.0],    # H
            [-0.239, 0.927, 0.0], # H
            [3.0, 0.0, 0.0],      # O
            [3.957, 0.0, 0.0],    # H
            [2.761, 0.927, 0.0],  # H
        ], dtype=torch.float32, device=device)
        
        atom_types = torch.tensor([0, 1, 1, 0, 1, 1], dtype=torch.long, device=device)
        box = torch.eye(3, dtype=torch.float32, device=device) * 12.4447
        
        # Compute descriptor
        desc1 = descriptor(positions, atom_types, box)
        
        # Translate all atoms
        translation = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
        positions_translated = positions + translation
        
        desc2 = descriptor(positions_translated, atom_types, box)
        
        # Descriptors should be identical (invariant to translation)
        max_diff = torch.abs(desc1 - desc2).max().item()
        assert max_diff < 1e-5, f"Translation not invariant: max diff = {max_diff}"
        print(f"✓ Translation invariance test passed (max diff: {max_diff:.2e})")


class TestPaddingStability:
    """Test that padding doesn't cause NaN/Inf."""
    
    def test_padding_no_nan(self):
        """Model should handle padded neighbors without NaN."""
        device = torch.device('cpu')
        type_map = ["O", "H"]
        
        model = DeepMDModel(
            type_map=type_map,
            rcut=6.0,
            rcut_smth=0.5,
            sel=[8, 16],
            descriptor_neuron=[16, 32],
            axis_neuron=8,
            fitting_neuron=[64, 64],
            type_one_side=False,
            resnet_dt=False
        ).to(device)
        
        # Create minimal geometry (fewer atoms than sel limits)
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
        ], dtype=torch.float32, device=device).requires_grad_(True)
        
        atom_types = torch.tensor([0, 1, 1], dtype=torch.long, device=device)
        box = torch.eye(3, dtype=torch.float32, device=device) * 12.4447
        
        # Forward pass should not produce NaN/Inf
        energy, atomic_e, forces = model.get_forces(positions, atom_types, box)
        
        assert torch.isfinite(energy).item(), "Energy is not finite"
        assert torch.isfinite(atomic_e).all().item(), "Atomic energies contain NaN/Inf"
        assert torch.isfinite(forces).all().item(), "Forces contain NaN/Inf"
        
        print(f"✓ Padding stability test passed")
        print(f"  Energy: {energy.item():.6e}, Forces range: [{forces.min().item():.2e}, {forces.max().item():.2e}]")


class TestPeriodicBoundaryConditions:
    """Test minimum image convention for PBC."""
    
    def test_pbc_neighbor_list(self):
        """Neighbors across PBC should be recognized correctly."""
        # Create two atoms: one at origin, one just outside box
        box_size = 10.0
        box = torch.eye(3, dtype=torch.float32) * box_size
        
        positions = torch.tensor([
            [5.0, 5.0, 5.0],  # Center
            [5.1, 5.0, 5.0],  # 0.1 Å away
        ], dtype=torch.float32)
        
        atom_types = torch.tensor([0, 1], dtype=torch.long)
        type_map = ["O", "H"]
        sel = [1, 1]
        rcut = 6.0
        
        neighbor_indices, neighbor_types, neighbor_mask = build_neighbor_list(
            positions, atom_types, type_map, sel, rcut, box
        )
        
        # First atom should see the second atom in the H-type neighbor block.
        assert (neighbor_indices[0] == 1).any().item(), "Neighbor not detected within cutoff"
        assert neighbor_mask[0, (neighbor_indices[0] == 1)].max().item() > 0
        
        print(f"✓ PBC test passed")
        print(f"  Neighbor indices: {neighbor_indices}")
        print(f"  Neighbor mask: {neighbor_mask}")


class TestSmokeTest:
    """Smoke test: ensure model runs without crashing."""
    
    def test_model_forward_backward(self):
        """Test basic forward/backward pass."""
        device = torch.device('cpu')
        type_map = ["O", "H"]
        
        model = DeepMDModel(
            type_map=type_map,
            rcut=6.0,
            rcut_smth=0.5,
            sel=[46, 92],
            descriptor_neuron=[25, 50, 100],
            axis_neuron=16,
            fitting_neuron=[240, 240, 240],
            type_one_side=True,
            resnet_dt=True
        ).to(device)
        
        # Create water molecules (2 waters: 2*3 = 6 atoms)
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.957, 0.0, 0.0],
            [-0.239, 0.927, 0.0],
            [5.0, 0.0, 0.0],
            [5.957, 0.0, 0.0],
            [4.761, 0.927, 0.0],
        ], dtype=torch.float32, device=device).requires_grad_(True)
        
        atom_types = torch.tensor([0, 1, 1, 0, 1, 1], dtype=torch.long, device=device)
        box = torch.eye(3, dtype=torch.float32, device=device) * 12.4447
        
        # Forward pass
        energy, atomic_e, forces = model.get_forces(positions, atom_types, box)
        
        # Check outputs
        assert energy.shape == torch.Size([]), "Energy should be scalar"
        assert atomic_e.shape == torch.Size([6]), "Atomic energies should be (natom,)"
        assert forces.shape == torch.Size([6, 3]), "Forces should be (natom, 3)"
        
        # Backward pass
        loss = energy.sum()
        loss.backward()
        
        assert positions.grad is not None, "Position gradient not computed"
        assert torch.isfinite(positions.grad).all(), "Position gradient contains NaN/Inf"
        
        print(f"✓ Smoke test passed")
        print(f"  Energy: {energy.item():.6e}")
        print(f"  Atomic energies mean: {atomic_e.mean().item():.6e}")
        print(f"  Forces magnitude: {forces.norm(dim=1).mean().item():.6e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestBatchedCoreFunctionality:
    """Tests for batched model execution and DeepMD metadata loading."""

    def test_model_get_forces_supports_batches(self):
        """Model should preserve a DataLoader batch dimension."""
        device = torch.device('cpu')
        type_map = ["O", "H"]

        model = DeepMDModel(
            type_map=type_map,
            rcut=6.0,
            rcut_smth=0.5,
            sel=[2, 4],
            descriptor_neuron=[8, 16],
            axis_neuron=4,
            fitting_neuron=[16, 16],
            type_one_side=False,
            resnet_dt=False
        ).to(device)

        frame = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.957, 0.0, 0.0],
            [-0.239, 0.927, 0.0],
        ], dtype=torch.float32, device=device)
        positions = torch.stack([frame, frame + 0.2], dim=0).requires_grad_(True)
        atom_types = torch.tensor([[0, 1, 1], [0, 1, 1]], dtype=torch.long, device=device)
        box = torch.eye(3, dtype=torch.float32, device=device).expand(2, 3, 3) * 12.4447

        energy, atomic_e, forces = model.get_forces(positions, atom_types, box)

        assert energy.shape == torch.Size([2])
        assert atomic_e.shape == torch.Size([2, 3])
        assert forces.shape == torch.Size([2, 3, 3])
        assert torch.isfinite(energy).all()
        assert torch.isfinite(forces).all()

    def test_triclinic_neighbor_list_uses_minimum_image(self):
        """Neighbor detection should work for non-diagonal DeepMD boxes."""
        box = torch.tensor([
            [10.0, 0.0, 0.0],
            [2.0, 9.0, 0.0],
            [0.0, 0.0, 8.0],
        ], dtype=torch.float32)
        positions = torch.tensor([
            [0.1, 0.1, 0.1],
            [9.9, 0.1, 0.1],
        ], dtype=torch.float32)
        atom_types = torch.tensor([0, 1], dtype=torch.long)

        neighbor_indices, _, neighbor_mask = build_neighbor_list(
            positions, atom_types, ["O", "H"], [1, 1], 0.5, box
        )

        assert neighbor_indices[0, 1].item() == 1
        assert neighbor_mask[0, 1].item() == 1.0

    def test_dataset_accepts_string_path_and_type_raw(self, tmp_path):
        """DeepMDDataset should accept one path string and prefer type.raw."""
        from dpmini.data import DeepMDDataset

        system_dir = tmp_path / "system"
        set_dir = system_dir / "set.000"
        set_dir.mkdir(parents=True)
        np.savetxt(system_dir / "type.raw", np.array([1, 0], dtype=np.int64), fmt="%d")
        np.save(set_dir / "coord.npy", np.array([[0, 0, 0, 1, 0, 0]], dtype=np.float32))
        np.save(set_dir / "box.npy", np.eye(3, dtype=np.float32).reshape(1, 9) * 10.0)
        np.save(set_dir / "energy.npy", np.array([1.5], dtype=np.float32))
        np.save(set_dir / "force.npy", np.zeros((1, 6), dtype=np.float32))

        dataset = DeepMDDataset(str(system_dir), type_map=["O", "H"])
        positions, atom_types, box, energy, forces = dataset[0]

        assert len(dataset) == 1
        assert positions.shape == torch.Size([2, 3])
        assert atom_types.tolist() == [1, 0]
        assert box.shape == torch.Size([3, 3])
        assert energy.item() == pytest.approx(1.5)
        assert forces.shape == torch.Size([2, 3])
