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
from typing import List, Optional, Sequence, Tuple, Union


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


def load_atom_types(system_dir: Path, natom: int, type_map: Optional[List[str]] = None) -> Optional[np.ndarray]:
    """Load atom type indices from a DeepMD system directory when available.

    DeepMD stores per-atom type indices in ``type.raw`` at the system root. If
    the file is absent, callers can fall back to domain-specific inference.
    """
    type_path = system_dir / 'type.raw'
    if not type_path.exists():
        return None

    atom_types = np.loadtxt(type_path, dtype=np.int64).reshape(-1)
    if atom_types.shape[0] != natom:
        raise ValueError(
            f"type.raw in {system_dir} has {atom_types.shape[0]} entries, expected {natom}"
        )
    if np.any(atom_types < 0):
        raise ValueError(f"type.raw in {system_dir} contains negative type indices")
    if type_map is not None and np.any(atom_types >= len(type_map)):
        raise ValueError(
            f"type.raw in {system_dir} contains indices outside type_map of size {len(type_map)}"
        )
    return atom_types


def infer_water_atom_types(natom: int) -> np.ndarray:
    """Infer the common water ordering: all O atoms followed by all H atoms."""
    if natom % 3 == 0:
        nO = natom // 3
        nH = 2 * nO
        return np.array([0] * nO + [1] * nH, dtype=np.int64)

    print(f"Warning: natom={natom} not divisible by 3, assuming all H atoms")
    return np.ones(natom, dtype=np.int64)


def load_deepmd_system(system_dir, type_map: Optional[List[str]] = None) -> dict:
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
    
    atom_types = load_atom_types(system_dir, natom, type_map)

    return {
        'coord': coord,
        'box': box,
        'energy': energy,
        'force': force,
        'atom_types': atom_types,
        'natom': natom,
        'nframe': nframe
    }


class DeepMDDataset(Dataset):
    """PyTorch Dataset for DeepMD data.
    
    Loads data from one or more system directories.
    """
    
    def __init__(self, 
                 system_dirs: Union[str, Path, Sequence[Union[str, Path]]],
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

        if isinstance(system_dirs, (str, Path)):
            system_dirs = [system_dirs]
        else:
            system_dirs = list(system_dirs)

        if len(system_dirs) == 0:
            raise ValueError("system_dirs must contain at least one DeepMD system directory")
        
        # Load all systems
        all_coords = []
        all_boxes = []
        all_energies = []
        all_forces = []
        natom = None
        loaded_atom_types = None
        
        for sys_dir in system_dirs:
            data = load_deepmd_system(Path(sys_dir), type_map=type_map)
            
            if natom is None:
                natom = data['natom']
                loaded_atom_types = data.get('atom_types')
            else:
                if data['natom'] != natom:
                    raise ValueError(f"Inconsistent natom: {natom} vs {data['natom']}")
                current_types = data.get('atom_types')
                if loaded_atom_types is not None and current_types is not None and not np.array_equal(loaded_atom_types, current_types):
                    raise ValueError("All systems in one DeepMDDataset must use identical atom type ordering")
                if loaded_atom_types is None and current_types is not None:
                    loaded_atom_types = current_types
            
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
        
        # Prefer explicit caller input, then DeepMD type.raw, then legacy water inference.
        if atom_types is None:
            atom_types = loaded_atom_types if loaded_atom_types is not None else infer_water_atom_types(natom)
        else:
            atom_types = np.asarray(atom_types, dtype=np.int64).reshape(-1)

        if atom_types.shape[0] != natom:
            raise ValueError(f"atom_types has {atom_types.shape[0]} entries, expected {natom}")
        if np.any(atom_types < 0) or np.any(atom_types >= self.ntypes):
            raise ValueError(f"atom_types must be in [0, {self.ntypes}), got {atom_types}")
        
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
