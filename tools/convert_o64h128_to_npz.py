#!/usr/bin/env python3
"""
Convert collect/O64H128/set.000/*.npy into a single dataset .npz that `bycopilot.py` can load.
Produces `o64h128_dataset.npz` with keys: positions, energies, forces
positions shape: (M, N, 3)
energies shape: (M,)
forces shape: (M, N, 3)
"""
import numpy as np
import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "collect" / "O64H128" / "set.000"
OUT_PATH = Path(__file__).resolve().parents[1] / "o64h128_dataset.npz"

def main():
    print("Reading from:", DATA_DIR)
    coords_p = DATA_DIR / "coord.npy"
    box_p = DATA_DIR / "box.npy"
    energy_p = DATA_DIR / "energy.npy"
    force_p = DATA_DIR / "force.npy"

    if not coords_p.exists():
        raise FileNotFoundError(coords_p)

    coords = np.load(coords_p)
    print("coords shape:", coords.shape)

    # energies/forces may or may not exist; handle gracefully
    energies = None
    forces = None
    if energy_p.exists():
        energies = np.load(energy_p)
        print("energies shape:", energies.shape)
    else:
        print("Warning: energy.npy not found; creating zero energies")
        energies = np.zeros((coords.shape[0],), dtype=np.float32)

    if force_p.exists():
        forces = np.load(force_p)
        print("forces shape:", forces.shape)
    else:
        print("Warning: force.npy not found; creating zeros for forces")
        forces = np.zeros_like(coords)

    # ensure dtypes
    coords = coords.astype(np.float32)
    energies = energies.astype(np.float32)
    forces = forces.astype(np.float32)

    # pack to a single npz
    print(f"Saving to {OUT_PATH}")
    np.savez_compressed(OUT_PATH, positions=coords, energies=energies, forces=forces)
    print("Done.")

if __name__ == '__main__':
    main()
