#!/usr/bin/env python3
"""
Compute Radial Distribution Function (RDF) from MD trajectory (.npz).
Supports O-O, O-H, and H-H pair correlations.

Fixes compared to the original version:
- Proper MIC for general 3x3 cell (fractional coordinate wrapping) + correct volume = |det(box)|
- Remove stray leftover code that could crash / mislead
- First minimum is searched AFTER the first peak (not in a hard-coded tiny-r window)
- Output files include timestamp and are saved into pic/
- Optional sanity check: scan minimum pair distances to detect overlaps / broken frames
- Progress indicator via tqdm if available

Optional dependency:
  conda install -c conda-forge tqdm
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# PBC / geometry helpers
# -----------------------------
def _as_str_list(type_map_raw) -> List[str]:
    out = []
    for t in list(type_map_raw):
        if isinstance(t, (bytes, np.bytes_)):
            out.append(t.decode("utf-8"))
        else:
            out.append(str(t))
    return out


def box_volume(box: np.ndarray) -> float:
    """Return volume for box specified as (3,) lengths or (3,3) matrix."""
    box = np.asarray(box)
    if box.ndim == 1:
        return float(np.prod(box))
    return float(abs(np.linalg.det(box)))


def mic_displacements(dr: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Apply minimum image convention (MIC) to displacement vectors.

    dr: (..., 3) displacement vectors in Cartesian coordinates
    box:
      - (3,) orthorhombic lengths
      - (3,3) general cell matrix mapping fractional -> Cartesian: r = box @ s
    """
    box = np.asarray(box)
    dr = np.asarray(dr)

    if box.ndim == 1:
        # Orthorhombic
        L = box.reshape((1,) * (dr.ndim - 1) + (3,))
        return dr - L * np.round(dr / L)

    # General 3x3 cell
    inv_box = np.linalg.inv(box)
    # fractional: s = inv_box @ dr
    s = np.einsum("ij,...j->...i", inv_box, dr)
    s -= np.round(s)
    # back to cartesian: dr_mic = box @ s
    return np.einsum("ij,...j->...i", box, s)


def get_box_lengths_for_print(box: np.ndarray) -> np.ndarray:
    """For display only. If general cell, show diagonal as a quick reference."""
    box = np.asarray(box)
    if box.ndim == 1:
        return box
    return np.diag(box)


# -----------------------------
# RDF computation (vectorized)
# -----------------------------
@dataclass(frozen=True)
class RDFResult:
    r: np.ndarray
    g_r: np.ndarray
    hist: np.ndarray


def compute_rdf_frame(
    positions: np.ndarray,
    box: np.ndarray,
    atom_types: np.ndarray,
    type_map: List[str],
    pair: Tuple[str, str],
    rmax: float = 10.0,
    nbins: int = 200,
) -> RDFResult:
    """
    Compute RDF for one frame (vectorized), counting ordered pairs (i in A, j in B, j!=i for A=B).

    Returns:
      r (bin centers), g_r, hist (raw counts per bin).
    """
    positions = np.asarray(positions, dtype=float)
    atom_types = np.asarray(atom_types)

    type1, type2 = pair
    if type1 not in type_map or type2 not in type_map:
        raise ValueError(f"Unknown atom type in pair {pair}. Known types: {type_map}")

    idx1 = type_map.index(type1)
    idx2 = type_map.index(type2)

    atoms1 = np.where(atom_types == idx1)[0]
    atoms2 = np.where(atom_types == idx2)[0]

    if len(atoms1) == 0 or len(atoms2) == 0:
        # Return zeros gracefully
        bins = np.linspace(0.0, rmax, nbins + 1)
        r = 0.5 * (bins[:-1] + bins[1:])
        return RDFResult(r=r, g_r=np.zeros(nbins), hist=np.zeros(nbins))

    bins = np.linspace(0.0, rmax, nbins + 1)
    dr_bin = rmax / nbins
    r = 0.5 * (bins[:-1] + bins[1:])

    p1 = positions[atoms1]  # (n1,3)
    p2 = positions[atoms2]  # (n2,3)

    if type1 == type2:
        # Full (n,n,3) displacement matrix: p[j] - p[i]
        # This includes both directions; we then exclude diagonal (i==j)
        d = p1[None, :, :] - p1[:, None, :]  # (n,n,3) = p[j]-p[i] if transposed? here p1[None]-p1[:,None] gives p[j]-p[i]
        # Actually: d[i,j] = p1[0,j] - p1[i,0] => p[j]-p[i] (correct)
        d = mic_displacements(d, box)
        dist = np.linalg.norm(d, axis=-1)

        n = dist.shape[0]
        mask = ~np.eye(n, dtype=bool)  # exclude i==j
        dist_flat = dist[mask]
        dist_flat = dist_flat[dist_flat < rmax]
        hist, _ = np.histogram(dist_flat, bins=bins)

        n1 = len(atoms1)
        n2 = len(atoms2)
        vol = box_volume(box)
        rho = (n2 - 1) / vol if n2 > 1 else 0.0
        norm = n1
    else:
        d = p2[None, :, :] - p1[:, None, :]  # (n1,n2,3) with d[i,j]=p2[j]-p1[i]
        d = mic_displacements(d, box)
        dist = np.linalg.norm(d, axis=-1)

        dist_flat = dist.ravel()
        dist_flat = dist_flat[dist_flat < rmax]
        hist, _ = np.histogram(dist_flat, bins=bins)

        n1 = len(atoms1)
        n2 = len(atoms2)
        vol = box_volume(box)
        rho = n2 / vol
        norm = n1

    shell_volume = 4.0 * np.pi * (r**2) * dr_bin

    if rho <= 0.0:
        g_r = np.zeros_like(r)
    else:
        g_r = hist / (norm * rho * shell_volume)

    return RDFResult(r=r, g_r=g_r, hist=hist.astype(float))


def compute_rdf_trajectory(
    trajectory: np.ndarray,
    box: np.ndarray,
    atom_types: np.ndarray,
    type_map: List[str],
    pairs: List[Tuple[str, str]],
    rmax: float = 10.0,
    nbins: int = 200,
    use_tqdm: bool = True,
) -> Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]:
    """Compute RDF averaged over trajectory (average of per-frame g(r))."""
    nframes = trajectory.shape[0]
    results: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}

    # progress helper
    iterator = range(nframes)
    if use_tqdm:
        try:
            from tqdm import trange  # type: ignore

            iterator = trange(nframes)
        except Exception:
            iterator = range(nframes)

    for pair in pairs:
        print(f"  Computing {pair[0]}-{pair[1]} RDF...")
        g_sum = np.zeros(nbins, dtype=float)
        r_ref: Optional[np.ndarray] = None

        for fi in iterator:
            if not use_tqdm and fi % 100 == 0:
                print(f"    frame {fi}/{nframes}")

            pos = trajectory[fi]
            box_f = box[fi] if (np.asarray(box).ndim == 3 or np.asarray(box).ndim == 2 and np.asarray(box).shape[0] == nframes) else box

            res = compute_rdf_frame(
                positions=pos,
                box=box_f,
                atom_types=atom_types,
                type_map=type_map,
                pair=pair,
                rmax=rmax,
                nbins=nbins,
            )
            if r_ref is None:
                r_ref = res.r
            g_sum += res.g_r

        g_avg = g_sum / float(nframes)
        results[pair] = (r_ref if r_ref is not None else np.linspace(0, rmax, nbins), g_avg)

        # reset tqdm bar for next pair if it exists
        if use_tqdm:
            try:
                iterator = trange(nframes)  # type: ignore
            except Exception:
                iterator = range(nframes)

    return results


# -----------------------------
# Peak / minimum analysis
# -----------------------------
def find_local_peaks(r: np.ndarray, g: np.ndarray, threshold: float = 1.1) -> List[Tuple[float, float]]:
    peaks = []
    for i in range(1, len(g) - 1):
        if g[i] > threshold and g[i] > g[i - 1] and g[i] > g[i + 1]:
            peaks.append((float(r[i]), float(g[i])))
    return peaks


def first_peak_and_minimum(
    r: np.ndarray,
    g: np.ndarray,
    peak_threshold: float = 1.1,
    r_min_peak: float = 0.5,
    window_after_peak: float = 3.0,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Find the first peak above threshold (r >= r_min_peak), then the first minimum after that peak
    within a window of size window_after_peak.
    """
    idx_peak = None
    for k in range(1, len(g) - 1):
        if r[k] < r_min_peak:
            continue
        if g[k] > peak_threshold and g[k] > g[k - 1] and g[k] > g[k + 1]:
            idx_peak = k
            break

    if idx_peak is None:
        return None, None

    dr = r[1] - r[0]
    i0 = idx_peak + 1
    i1 = min(len(g), i0 + int(window_after_peak / dr))

    if i1 <= i0:
        return idx_peak, None

    idx_min = i0 + int(np.argmin(g[i0:i1]))
    return idx_peak, idx_min


# -----------------------------
# Plot / save
# -----------------------------
def plot_rdf(rdf_results: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]], output_file: Path) -> None:
    plt.figure(figsize=(10, 6))
    colors = {"O-O": "red", "O-H": "blue", "H-H": "green"}

    for pair, (r, g) in rdf_results.items():
        name = f"{pair[0]}-{pair[1]}"
        plt.plot(r, g, label=name, color=colors.get(name, "black"), linewidth=2)

    # Use mathtext for Angstrom to avoid font issues
    plt.xlabel(r"r ($\mathrm{\AA}$)", fontsize=12)
    plt.ylabel("g(r)", fontsize=12)
    plt.title("Radial Distribution Function", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, float(max(rdf_results[list(rdf_results.keys())[0]][0])))
    plt.ylim(0, None)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"\n✓ RDF plot saved to: {output_file}")


def save_rdf_data(rdf_results: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]], output_file: Path) -> None:
    pairs = list(rdf_results.keys())
    r = rdf_results[pairs[0]][0]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Radial Distribution Function\n")
        f.write("# Columns: r(A) " + " ".join([f"g_{p[0]}{p[1]}(r)" for p in pairs]) + "\n")
        for i, rv in enumerate(r):
            line = f"{rv:.6f}"
            for p in pairs:
                line += f"  {rdf_results[p][1][i]:.6f}"
            f.write(line + "\n")

    print(f"✓ RDF data saved to: {output_file}")


# -----------------------------
# Sanity check: minimum distances
# -----------------------------
def scan_min_distances(
    trajectory: np.ndarray,
    box: np.ndarray,
    atom_types: np.ndarray,
    type_map: List[str],
    pairs: List[Tuple[str, str]],
    use_tqdm: bool = True,
) -> None:
    """
    Scan per-frame minimum distances for requested pairs to detect overlaps / broken frames.
    """
    nframes = trajectory.shape[0]

    iterator = range(nframes)
    if use_tqdm:
        try:
            from tqdm import trange  # type: ignore

            iterator = trange(nframes, desc="sanity", leave=False)
        except Exception:
            iterator = range(nframes)

    for pair in pairs:
        type1, type2 = pair
        idx1 = type_map.index(type1)
        idx2 = type_map.index(type2)
        atoms1 = np.where(atom_types == idx1)[0]
        atoms2 = np.where(atom_types == idx2)[0]

        if len(atoms1) == 0 or len(atoms2) == 0:
            print(f"[sanity] {type1}-{type2}: skipped (no atoms).")
            continue

        mins = []
        for fi in iterator:
            pos = trajectory[fi]
            box_f = box[fi] if (np.asarray(box).ndim == 3 or np.asarray(box).ndim == 2 and np.asarray(box).shape[0] == nframes) else box

            p1 = pos[atoms1]
            p2 = pos[atoms2]

            if type1 == type2:
                d = p1[None, :, :] - p1[:, None, :]
                d = mic_displacements(d, box_f)
                dist = np.linalg.norm(d, axis=-1)
                n = dist.shape[0]
                mask = ~np.eye(n, dtype=bool)
                m = float(np.min(dist[mask]))
            else:
                d = p2[None, :, :] - p1[:, None, :]
                d = mic_displacements(d, box_f)
                dist = np.linalg.norm(d, axis=-1)
                m = float(np.min(dist))

            mins.append(m)

        mins = np.array(mins)
        print(f"[sanity] {type1}-{type2}: min={mins.min():.3f} A, p1%={np.percentile(mins,1):.3f} A, median={np.median(mins):.3f} A")


# -----------------------------
# CLI / main
# -----------------------------
def parse_pairs(pair_strs: List[str]) -> List[Tuple[str, str]]:
    pairs = []
    for s in pair_strs:
        parts = s.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid pair format: {s} (expected A-B)")
        pairs.append((parts[0], parts[1]))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute RDF from MD trajectory (.npz)")
    parser.add_argument("trajectory", help="NPZ trajectory file containing trajectory, atom_types, type_map, (optional) box")
    parser.add_argument("--pairs", nargs="+", default=["O-O", "O-H", "H-H"], help="Pairs like O-O O-H H-H")
    parser.add_argument("--rmax", type=float, default=8.0, help="Maximum distance for RDF (A)")
    parser.add_argument("--nbins", type=int, default=200, help="Number of bins")
    parser.add_argument("--output", type=str, default="rdf", help="Output file prefix")
    parser.add_argument("--skip", type=int, default=1, help="Use every N-th frame")
    parser.add_argument("--box", nargs=3, type=float, default=None, help="Override box lengths Lx Ly Lz (A) if npz has no box")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bar")
    parser.add_argument("--sanity-check", action="store_true", help="Scan minimum distances per frame for requested pairs")
    args = parser.parse_args()

    npz_path = Path(args.trajectory)
    if not npz_path.exists():
        raise FileNotFoundError(f"File not found: {npz_path}")

    print(f"Loading trajectory: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    trajectory = np.asarray(data["trajectory"], dtype=float)
    atom_types = np.asarray(data["atom_types"])
    type_map = _as_str_list(data["type_map"])

    nframes_total = trajectory.shape[0]
    trajectory = trajectory[:: args.skip]
    nframes = trajectory.shape[0]

    # box handling
    if "box" in data:
        box = np.asarray(data["box"])
    else:
        if args.box is not None:
            box = np.array(args.box, dtype=float)
            print("⚠️  NPZ has no box; using --box override.")
        else:
            # Project default (if not mentioned): 12.4447 A cubic box
            box = np.array([12.4447, 12.4447, 12.4447], dtype=float)
            print("⚠️  NPZ has no box and no --box provided; using default cubic box: 12.4447 A.")

    print("\n✓ Trajectory loaded:")
    print(f"  Total frames: {nframes_total}")
    print(f"  Using frames: {nframes} (every {args.skip})")
    print(f"  Atoms: {trajectory.shape[1]}")
    print("  Atom types: " + ", ".join([f"{t}({int((atom_types == i).sum())})" for i, t in enumerate(type_map)]))

    if np.asarray(box).ndim == 1:
        print(f"  Box lengths: {box}")
    elif np.asarray(box).ndim == 2:
        print(f"  Box (diag): {get_box_lengths_for_print(box)}")
        print(f"  Box volume: {box_volume(box):.6f} A^3")
    elif np.asarray(box).ndim == 3:
        print(f"  Box is time-dependent: shape={box.shape}")
        print(f"  Box[0] diag: {get_box_lengths_for_print(box[0])}")
        print(f"  Box[0] volume: {box_volume(box[0]):.6f} A^3")
    else:
        raise ValueError(f"Unsupported box array shape: {np.asarray(box).shape}")

    pairs = parse_pairs(args.pairs)

    print("\n📊 Computing RDF:")
    print(f"  Pairs: {', '.join([f'{p[0]}-{p[1]}' for p in pairs])}")
    print(f"  r_max: {args.rmax} A")
    print(f"  bins: {args.nbins}")

    use_tqdm = (not args.no_tqdm)

    if args.sanity_check:
        print("\n🧪 Sanity check: scanning minimum distances...")
        scan_min_distances(
            trajectory=trajectory,
            box=box,
            atom_types=atom_types,
            type_map=type_map,
            pairs=pairs,
            use_tqdm=use_tqdm,
        )

    rdf_results = compute_rdf_trajectory(
        trajectory=trajectory,
        box=box,
        atom_types=atom_types,
        type_map=type_map,
        pairs=pairs,
        rmax=args.rmax,
        nbins=args.nbins,
        use_tqdm=use_tqdm,
    )

    print("\n🔍 Peak Analysis (first 3 local peaks; first minimum searched after first peak):")
    for pair, (r, g) in rdf_results.items():
        name = f"{pair[0]}-{pair[1]}"
        print(f"\n  {name}:")
        peaks = find_local_peaks(r, g, threshold=1.1)
        if peaks:
            for k, (rp, gp) in enumerate(peaks[:3], 1):
                print(f"    Peak {k}: r = {rp:.3f} A, g(r) = {gp:.3f}")
        else:
            print("    No significant local peaks found (threshold=1.1).")

        idx_peak, idx_min = first_peak_and_minimum(r, g, peak_threshold=1.1, r_min_peak=0.5, window_after_peak=3.0)
        if idx_peak is not None:
            print(f"    First peak: r = {r[idx_peak]:.3f} A, g(r) = {g[idx_peak]:.3f}")
        if idx_min is not None:
            print(f"    First minimum (after peak): r = {r[idx_min]:.3f} A, g(r) = {g[idx_min]:.3f}")

    # timestamped outputs in pic/
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path("pic")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_file = out_dir / f"{args.output}_plot_{timestamp}.png"
    data_file = out_dir / f"{args.output}_data_{timestamp}.txt"

    plot_rdf(rdf_results, plot_file)
    save_rdf_data(rdf_results, data_file)

    print("\n✅ RDF analysis completed!")
    print(f"   Plot: {plot_file}")
    print(f"   Data: {data_file}")


if __name__ == "__main__":
    main()
