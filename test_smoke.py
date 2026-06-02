"""Smoke tests for the end-to-end DPMini training pipeline."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from dpmini import DeepMDModel, DeepMDDataset


def _write_tiny_deepmd_system(system_dir):
    set_dir = system_dir / "set.000"
    set_dir.mkdir(parents=True)

    coords = np.array([
        [0.0, 0.0, 0.0, 0.957, 0.0, 0.0, -0.239, 0.927, 0.0],
        [0.1, 0.0, 0.0, 1.057, 0.0, 0.0, -0.139, 0.927, 0.0],
    ], dtype=np.float32)
    boxes = np.tile((np.eye(3, dtype=np.float32) * 12.0).reshape(1, 9), (2, 1))
    energies = np.array([-1.0, -0.9], dtype=np.float32)
    forces = np.zeros((2, 9), dtype=np.float32)

    np.savetxt(system_dir / "type.raw", np.array([0, 1, 1], dtype=np.int64), fmt="%d")
    np.save(set_dir / "coord.npy", coords)
    np.save(set_dir / "box.npy", boxes)
    np.save(set_dir / "energy.npy", energies)
    np.save(set_dir / "force.npy", forces)


def _make_model():
    return DeepMDModel(
        type_map=["O", "H"],
        rcut=6.0,
        rcut_smth=0.5,
        sel=[2, 4],
        descriptor_neuron=[8, 16],
        axis_neuron=4,
        fitting_neuron=[16, 16],
        type_one_side=True,
        resnet_dt=False,
    )


def test_end_to_end_smoke_pipeline(tmp_path):
    """Load data, batch frames, compute forces, and run one optimizer step."""
    system_dir = tmp_path / "tiny"
    _write_tiny_deepmd_system(system_dir)

    dataset = DeepMDDataset(str(system_dir), type_map=["O", "H"])
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    positions, atom_types, box, target_energy, target_forces = next(iter(dataloader))

    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    positions = positions.requires_grad_(True)
    energy_pred, atomic_energy, forces_pred = model.get_forces(positions, atom_types, box)

    assert energy_pred.shape == torch.Size([2])
    assert atomic_energy.shape == torch.Size([2, 3])
    assert forces_pred.shape == torch.Size([2, 3, 3])
    assert torch.isfinite(energy_pred).all()
    assert torch.isfinite(forces_pred).all()

    loss_energy = torch.nn.functional.mse_loss(energy_pred, target_energy)
    loss_force = torch.nn.functional.mse_loss(forces_pred, target_forces)
    loss = loss_energy + 0.1 * loss_force

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
