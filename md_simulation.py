#!/usr/bin/env python3
"""
Simple Molecular Dynamics simulation using DeepMD PyTorch model.

Key fixes:
- Unit-consistent MD integration for common DeepMD convention:
    positions: Å
    energies:  eV
    forces:    eV/Å
    masses:    amu
    time:      ps
- Velocity Verlet (NVE).
- PBC wrap after each position update (general 3x3 box).
- --T (initial temperature) takes effect for velocity initialization.
- Avoid third model.forward per step (reuse energy from the 2nd force evaluation).
- Avoid autograd history accumulation across steps (detach each step).
- COM translation removal stays on GPU (no CPU round-trip).
- Output .npz filename includes timestamp; optional trajectory stride.

Usage:
  python md_simulation.py --model exports_final/model_cuda_xxx.pth \
                          --system collect/data0 --steps 10000 --dt 0.0005 --T 300 \
                          --traj-stride 10 --checkpoint-interval 5000
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dpmini import DeepMDModel
from dpmini.data import load_deepmd_set


# ----------------------------
# Unit conversions
# ----------------------------
# Common DeepMD conventions: E in eV, r in Å, so F in eV/Å.
EV_TO_J = 1.602176634e-19
AMU_TO_KG = 1.66053906660e-27
A_TO_M = 1e-10
PS_TO_S = 1e-12

# a(Å/ps^2) = F(eV/Å)/m(amu) * FORCE_TO_ACCEL
FORCE_TO_ACCEL = EV_TO_J * (PS_TO_S ** 2) / (AMU_TO_KG * (A_TO_M ** 2))  # ≈ 9648.5332157

# KE(eV) = 0.5 * sum( m(amu) * v(Å/ps)^2 ) * MV2_TO_EV
MV2_TO_EV = AMU_TO_KG * (A_TO_M / PS_TO_S) ** 2 / EV_TO_J  # ≈ 1.036426965e-4

# Boltzmann constant in eV/K
K_B_EV = 8.617333262e-5


def load_initial_structure(system_dir: str, type_map=None):
    """Load initial atomic structure from DeepMD dataset (set.000, first frame)."""
    system_dir = Path(system_dir)
    set_dir = system_dir / "set.000"
    data = load_deepmd_set(set_dir)

    # coords: (nframe, natom, 3) in Å
    positions = data["coord"][0].copy()
    natom = int(data.get("natom", positions.shape[0]))

    # box: (nframe, 3, 3) or (nframe, 9). If missing, fallback to project default.
    if "box" in data and data["box"] is not None:
        box0 = np.array(data["box"][0])
        if box0.size == 9:
            box = box0.reshape(3, 3)
        else:
            box = box0.reshape(3, 3)
    else:
        # Project convention default: cubic box with L = 12.4447 Å
        L = 12.4447
        box = np.eye(3, dtype=np.float32) * L

    # atom types: prefer dataset if present; else fallback to water O/H pattern
    atom_types = None
    if "type" in data and data["type"] is not None:
        t = np.array(data["type"])
        if t.ndim == 2:
            atom_types = t[0].copy()
        else:
            atom_types = t.copy()
        atom_types = atom_types.astype(np.int64)

    if atom_types is None:
        if type_map is None:
            type_map = ["O", "H"]
        # Default: first natom//3 are O, rest are H (water)
        nO = natom // 3
        atom_types = np.array([0] * nO + [1] * (natom - nO), dtype=np.int64)

    return positions.astype(np.float32), box.astype(np.float32), atom_types


def remove_translation(velocities: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Remove center-of-mass (COM) translational motion (velocities only)."""
    com_v = np.average(velocities, axis=0, weights=masses)
    return velocities - com_v


def wrap_pbc_torch(pos: torch.Tensor, box: torch.Tensor, inv_box: torch.Tensor) -> torch.Tensor:
    """Wrap positions into the primary cell for a general 3x3 box.

    Assumes the common convention:
        r = f @ box   (box rows are cell vectors)
        f = r @ inv(box)
    """
    frac = pos @ inv_box
    frac = frac - torch.floor(frac)  # [0, 1)
    return frac @ box


def run_md(
    model,
    positions: np.ndarray,
    box: np.ndarray,
    atom_types: np.ndarray,
    type_map,
    masses: np.ndarray,
    n_steps: int = 100,
    dt: float = 0.001,
    T: float = 300.0,
    device: str = "cuda",
    remove_translation_every: int = 10,
    save_trajectory: bool = True,
    traj_stride: int = 1,
    output_prefix: str = "md_trajectory",
    checkpoint_interval: int = 1000,
):
    """Run an NVE MD simulation with Velocity Verlet."""
    # Maxwell-Boltzmann velocity init in Å/ps:
    # 0.5*m*v^2*MV2_TO_EV = 0.5*k_B*T  =>  var(v) = k_B*T/(m*MV2_TO_EV)
    sigma = np.sqrt(K_B_EV * float(T) / (masses * MV2_TO_EV)).astype(np.float32)
    velocities = np.random.normal(0.0, sigma[:, None], positions.shape).astype(np.float32)
    velocities = remove_translation(velocities, masses)

    # To torch
    positions_t = torch.from_numpy(positions).to(device=device, dtype=torch.float32)
    velocities_t = torch.from_numpy(velocities).to(device=device, dtype=torch.float32)
    atom_types_t = torch.from_numpy(atom_types).to(device=device, dtype=torch.long)
    masses_t = torch.from_numpy(masses).to(device=device, dtype=torch.float32)
    box_t = torch.from_numpy(box).to(device=device, dtype=torch.float32)

    # Project convention: box constant across trajectory, so precompute inv(box) once.
    inv_box_t = torch.linalg.inv(box_t)

    trajectory = []
    energies = []
    times = []

    if save_trajectory:
        trajectory.append(positions_t.detach().cpu().numpy().copy())

    print(f"\n{'Step':<8} {'Time (ps)':<12} {'KE (eV)':<12} {'PE (eV)':<12} {'Total (eV)':<12}")
    print("-" * 60)

    model.eval()
    print_interval = max(10, n_steps // 50)

    for step in range(n_steps):
        # (1) Forces at r(t)
        positions_t = positions_t.detach().requires_grad_(True)
        velocities_t = velocities_t.detach()

        with torch.enable_grad():
            total_energy, _ = model.forward(positions_t, atom_types_t, box_t)
            forces = -torch.autograd.grad(
                total_energy,
                positions_t,
                create_graph=False,
                retain_graph=False,
            )[0]  # eV/Å

        # v(t+dt/2)
        accelerations = forces * FORCE_TO_ACCEL / masses_t[:, None]  # Å/ps^2
        velocities_t = velocities_t + 0.5 * accelerations * dt

        # r(t+dt) + PBC wrap
        positions_t = positions_t + velocities_t * dt
        positions_t = wrap_pbc_torch(positions_t, box_t, inv_box_t)

        # (2) Forces at r(t+dt)  (also provides PE at r(t+dt))
        positions_t = positions_t.detach().requires_grad_(True)
        with torch.enable_grad():
            total_energy, _ = model.forward(positions_t, atom_types_t, box_t)
            forces = -torch.autograd.grad(
                total_energy,
                positions_t,
                create_graph=False,
                retain_graph=False,
            )[0]

        # v(t+dt)
        accelerations = forces * FORCE_TO_ACCEL / masses_t[:, None]
        velocities_t = velocities_t + 0.5 * accelerations * dt

        # Remove COM translation on-GPU
        if remove_translation_every and remove_translation_every > 0 and (step + 1) % remove_translation_every == 0:
            com_v = (velocities_t * masses_t[:, None]).sum(dim=0) / masses_t.sum()
            velocities_t = velocities_t - com_v

        # Energies (reuse total_energy from 2nd forward)
        ke = 0.5 * (masses_t[:, None] * velocities_t**2).sum().item() * MV2_TO_EV
        pe = total_energy.detach().item()
        etot = ke + pe

        energies.append((pe, ke, etot))
        times.append((step + 1) * dt)

        # Store trajectory with stride
        if save_trajectory and ((step + 1) % max(1, int(traj_stride)) == 0):
            trajectory.append(positions_t.detach().cpu().numpy().copy())

        # Progress
        if step < 5 or (step + 1) % print_interval == 0:
            print(f"{step+1:<8} {(step+1)*dt:<12.4f} {ke:<12.4f} {pe:<12.4f} {etot:<12.4f}")

        # Checkpoint (lightweight: current state + energies/times so far)
        if checkpoint_interval and checkpoint_interval > 0 and (step + 1) % checkpoint_interval == 0:
            ckpt_file = f"{output_prefix}_checkpoint_{step+1}.npz"
            np.savez(
                ckpt_file,
                positions=positions_t.detach().cpu().numpy(),
                velocities=velocities_t.detach().cpu().numpy(),
                box=box,
                atom_types=atom_types,
                type_map=type_map,
                energies=np.array(energies, dtype=np.float64),
                times=np.array(times, dtype=np.float64),
            )
            print(f"\n✓ Checkpoint saved: {ckpt_file} ({step+1}/{n_steps} steps)\n")

    energies_arr = np.array(energies, dtype=np.float64)
    times_arr = np.array(times, dtype=np.float64)

    print("-" * 60)
    print(f"✓ MD simulation completed: {n_steps} steps, {n_steps*dt:.3f} ps")

    if len(energies_arr) > 1:
        print("\nEnergy Statistics:")
        print(f"  Potential Energy: {energies_arr[:,0].mean():.6f} ± {energies_arr[:,0].std():.6f} eV")
        print(f"  Kinetic Energy:   {energies_arr[:,1].mean():.6f} ± {energies_arr[:,1].std():.6f} eV")
        print(f"  Total Energy:     {energies_arr[:,2].mean():.6f} ± {energies_arr[:,2].std():.6f} eV")
        e0 = energies_arr[0, 2]
        if abs(e0) > 1e-12:
            drift = (energies_arr[-1, 2] - e0) / e0
            print(f"  Energy drift:     {drift*100:.3f}%")
        else:
            print("  Energy drift:     (skip; initial total energy ~ 0)")

    return trajectory, energies_arr, times_arr


def _safe_torch_load(path: str, map_location: str = "cpu"):
    """Prefer PyTorch 'weights_only' safe mode when available (suppresses security warning)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def main():
    parser = argparse.ArgumentParser(description="Run MD simulation with DeepMD model")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--system", type=str, default="collect/data0", help="Path to system directory")
    parser.add_argument("--steps", type=int, default=100, help="Number of MD steps")
    parser.add_argument("--dt", type=float, default=0.001, help="Time step in ps")
    parser.add_argument("--T", type=float, default=300.0, help="Initial temperature in K (velocity init)")
    parser.add_argument("--output", type=str, default="md_trajectory", help="Output file prefix")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--traj-stride", type=int, default=1, help="Store every Nth frame (memory control)")
    parser.add_argument("--remove-translation-every", type=int, default=10,
                        help="Remove COM translation every N steps (0 disables)")
    parser.add_argument("--checkpoint-interval", type=int, default=1000,
                        help="Save checkpoint every N steps (0 disables)")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_prefix = f"{args.output}_{timestamp}"

    # Load model
    print(f"Loading model from {args.model}")
    checkpoint = _safe_torch_load(args.model, map_location="cpu")

    if isinstance(checkpoint, dict) and "config" in checkpoint:
        config = checkpoint["config"]
        type_map = checkpoint.get("type_map", config["model"]["type_map"])
        model_state = checkpoint["model_state_dict"]
    else:
        with open("se_e2_a/input_torch.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        type_map = config["model"]["type_map"]
        model_state = checkpoint

    model_config = config["model"]
    descriptor_config = model_config["descriptor"]
    fitting_config = model_config["fitting_net"]

    model = DeepMDModel(
        type_map=type_map,
        rcut=descriptor_config["rcut"],
        rcut_smth=descriptor_config["rcut_smth"],
        sel=descriptor_config["sel"],
        descriptor_neuron=descriptor_config["neuron"],
        axis_neuron=descriptor_config["axis_neuron"],
        fitting_neuron=fitting_config["neuron"],
        type_one_side=descriptor_config.get("type_one_side", True),
        resnet_dt=fitting_config.get("resnet_dt", True),
    )
    model.load_state_dict(model_state)
    model.to(args.device)
    print(f"✓ Model loaded, using device: {args.device}")

    # Load initial structure
    print(f"\nLoading structure from {args.system}")
    positions, box, atom_types = load_initial_structure(args.system, type_map)
    print(f"  Atoms: {len(positions)}")
    print(f"  Atom types: {[type_map[t] for t in atom_types]}")
    print(f"  Box (Å):\n{box}")

    # Atomic masses (extend as needed)
    mass_map = {"O": 16.0, "H": 1.0, "Na": 22.98976928}
    masses = np.array([mass_map[type_map[t]] for t in atom_types], dtype=np.float32)

    # Run MD
    print("\nRunning MD simulation:")
    print(f"  Steps: {args.steps}")
    print(f"  Time step: {args.dt} ps")
    print(f"  Total time: {args.steps * args.dt:.3f} ps")
    print(f"  Initial temperature: {args.T} K")
    print(f"  Trajectory stride: {args.traj_stride}")

    trajectory, energies, times = run_md(
        model=model,
        positions=positions,
        box=box,
        atom_types=atom_types,
        type_map=type_map,
        masses=masses,
        n_steps=args.steps,
        dt=args.dt,
        T=args.T,
        device=args.device,
        remove_translation_every=args.remove_translation_every,
        save_trajectory=True,
        traj_stride=args.traj_stride,
        output_prefix=output_prefix,
        checkpoint_interval=args.checkpoint_interval,
    )

    # Save trajectory
    traj_file = f"{output_prefix}.npz"
    np.savez(
        traj_file,
        trajectory=np.array(trajectory, dtype=np.float32),
        energies=energies,
        times=times,
        atom_types=atom_types,
        type_map=type_map,
        box=box,
    )
    print(f"\n✓ Trajectory saved to: {traj_file}")
    print(f"  Frames: {len(trajectory)}")
    print(f"  Time range: 0 - {times[-1]:.4f} ps")


if __name__ == "__main__":
    main()
