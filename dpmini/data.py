"""
Data loading for DeepMD training.

Loads data from DeepMD format directories:
  system_dir/
    set.000/
      coord.npy   # (nframe, natom*3) flattened coordinates in Angstrom
      box.npy     # (nframe, 9) flattened box vectors
      energy.npy  # (nframe,) total energies in eV
      force.npy   # (nframe, natom*3) flattened forces in eV/Angstrom
      virial.npy  # (nframe, 9) optional

Units:
  - Length: Angstrom (Å)
  - Energy: eV
  - Force: eV/Å
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional, Tuple


def infer_natom_from_coord(coord_path: Path) -> int:
    """Infer number of atoms from coord.npy shape."""
    coord = np.load(coord_path)
    nframe = coord.shape[0]
    natom_times_3 = coord.shape[1]
    assert natom_times_3 % 3 == 0
    natom = natom_times_3 // 3
    return natom


def load_deepmd_set(set_dir: Path) -> dict:
    """Load a single DeepMD set.000 directory.
    
    Returns:
        dict with keys: coord, box, energy, force, natom, nframe
    """
    coord_path = set_dir / 'coord.npy'
    box_path = set_dir / 'box.npy'
    energy_path = set_dir / 'energy.npy'
    force_path = set_dir / 'force.npy'
    
    if not coord_path.exists():
        raise FileNotFoundError(f"coord.npy not found in {set_dir}")
    
    coord = np.load(coord_path).astype(np.float32)  # (nframe, natom*3)
    nframe = coord.shape[0]
    natom = coord.shape[1] // 3
    
    # Reshape coord to (nframe, natom, 3)
    coord = coord.reshape(nframe, natom, 3)
    
    # Load box
    if box_path.exists():
        box = np.load(box_path).astype(np.float32)  # (nframe, 9)
        # Reshape to (nframe, 3, 3)
        box = box.reshape(nframe, 3, 3)
    else:
        # Default cubic box L=12.4447 Angstrom
        L = 12.4447
        box = np.eye(3, dtype=np.float32) * L
        box = np.tile(box[np.newaxis, :, :], (nframe, 1, 1))
    
    # Load energy
    if energy_path.exists():
        energy = np.load(energy_path).astype(np.float32)  # (nframe,)
    else:
        energy = np.zeros(nframe, dtype=np.float32)
    
    # Load force
    if force_path.exists():
        force = np.load(force_path).astype(np.float32)  # (nframe, natom*3)
        force = force.reshape(nframe, natom, 3)
    else:
        force = np.zeros((nframe, natom, 3), dtype=np.float32)
    
    return {
        'coord': coord,
        'box': box,
        'energy': energy,
        'force': force,
        'natom': natom,
        'nframe': nframe
    }


def load_deepmd_system(system_dir) -> dict:
    """Load all sets from a DeepMD system directory.
    
    Args:
        system_dir: Path or string to system directory
    
    Returns:
        dict with concatenated data from all set.XXX subdirs
    """
    if isinstance(system_dir, str):
        system_dir = Path(system_dir)
    
    set_dirs = sorted([d for d in system_dir.iterdir() if d.is_dir() and d.name.startswith('set.')])
    
    if len(set_dirs) == 0:
        raise FileNotFoundError(f"No set.XXX directories found in {system_dir}")
    
    all_data = []
    for set_dir in set_dirs:
        data = load_deepmd_set(set_dir)
        all_data.append(data)
    
    # Concatenate
    natom = all_data[0]['natom']
    coord = np.concatenate([d['coord'] for d in all_data], axis=0)
    box = np.concatenate([d['box'] for d in all_data], axis=0)
    energy = np.concatenate([d['energy'] for d in all_data], axis=0)
    force = np.concatenate([d['force'] for d in all_data], axis=0)
    nframe = coord.shape[0]
    
    return {
        'coord': coord,
        'box': box,
        'energy': energy,
        'force': force,
        'natom': natom,
        'nframe': nframe
    }


class DeepMDDataset(Dataset):
    """PyTorch Dataset for DeepMD data.
    
    Loads data from one or more system directories.
    """
    
    def __init__(self, 
                 system_dirs: List[str],
                 type_map: List[str] = ["O", "H"],
                 atom_types: Optional[np.ndarray] = None):
        """
        Args:
            system_dirs: list of system directory paths
            type_map: element names, e.g. ["O", "H"]
            atom_types: (natom,) array of type indices. If None, infer from natom:
                        For water: O64H128 -> 64 O + 128 H
        """
        self.type_map = type_map
        self.ntypes = len(type_map)
        
        # Load all systems
        all_coords = []
        all_boxes = []
        all_energies = []
        all_forces = []
        natom = None
        
        for sys_dir in system_dirs:
            data = load_deepmd_system(Path(sys_dir))
            
            if natom is None:
                natom = data['natom']
            else:
                assert data['natom'] == natom, f"Inconsistent natom: {natom} vs {data['natom']}"
            
            all_coords.append(data['coord'])
            all_boxes.append(data['box'])
            all_energies.append(data['energy'])
            all_forces.append(data['force'])
        
        self.coords = np.concatenate(all_coords, axis=0)  # (nframe, natom, 3)
        self.boxes = np.concatenate(all_boxes, axis=0)    # (nframe, 3, 3)
        self.energies = np.concatenate(all_energies, axis=0)  # (nframe,)
        self.forces = np.concatenate(all_forces, axis=0)  # (nframe, natom, 3)
        
        self.nframe = self.coords.shape[0]
        self.natom = natom
        
        # Infer atom types if not provided
        if atom_types is None:
            # Assume water system: first 1/3 are O, rest are H
            # For O64H128: 64 O + 128 H = 192 atoms
            # For O128H256: 128 O + 256 H = 384 atoms
            # Pattern: natom = 3 * nO (nO oxygen, 2*nO hydrogen)
            if natom % 3 == 0:
                nO = natom // 3
                nH = 2 * nO
                atom_types = np.array([0] * nO + [1] * nH, dtype=np.int64)
            else:
                # Fallback: assume all H (type 1)
                print(f"Warning: natom={natom} not divisible by 3, assuming all H atoms")
                atom_types = np.ones(natom, dtype=np.int64)
        
        self.atom_types = torch.from_numpy(atom_types).long()
        
        print(f"Loaded {self.nframe} frames, {self.natom} atoms per frame")
        print(f"Atom types distribution: {np.bincount(atom_types)}")
    
    def __len__(self) -> int:
        return self.nframe
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single frame.
        
        Returns:
            positions: (natom, 3)
            atom_types: (natom,)
            box: (3, 3)
            energy: scalar
            forces: (natom, 3)
        """
        positions = torch.from_numpy(self.coords[idx]).float()
        box = torch.from_numpy(self.boxes[idx]).float()
        energy = torch.tensor(self.energies[idx], dtype=torch.float32)
        forces = torch.from_numpy(self.forces[idx]).float()
        
        return positions, self.atom_types, box, energy, forces
