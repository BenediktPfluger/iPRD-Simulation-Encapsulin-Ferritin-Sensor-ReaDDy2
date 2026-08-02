"""
qtft.fibsem_export
==================
Extract final-frame **encapsulin** (Qt / QtC) positions from a ReaDDy trajectory and
write them in the same schema the FIB-SEM segmentation pipeline produces, so the
simulation can be pushed through the existing clustering / analysis notebooks.

Design notes
------------
- **Only Qt + QtC are exported.** Ferritin (Ft / FtC) is the sub-resolution linker and is
  invisible in FIB-SEM, so it is dropped from the output — but it *does* participate in the
  periodic-boundary unwrap, because it defines the Qt-Ft-Qt connectivity of a cluster.
- **Cluster membership is ground truth** from ReaDDy's topology graph (one topology = one
  cluster). No DBSCAN. Free (unbound) Qt become size-1 clusters, matching FIB-SEM singletons.
- **Positions are PBC-unwrapped per cluster** (minimum-image against the running centre of
  mass, mirroring ``qtft.analysis._unwrap_cluster_positions``), then globally shifted so the
  minimum corner sits at the origin (all coordinates >= 0).
- **Volumes are analytical** (4/3 pi r^3). Every row also carries ``radius_nm`` so a later
  notebook can compute an *exact* mass integral <M(R)> via ball-ball intersections, with no
  approximation and no voxelisation.
"""
from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ENCAPSULIN_TYPES = ("Qt", "QtC")
CLUSTERED_TYPE = "QtC"   # encapsulin bound in a cluster
FREE_TYPE = "Qt"         # free (unbound) encapsulin


def _unwrap(positions: np.ndarray, box_size, periodic: bool = True) -> np.ndarray:
    """Unwrap one cluster's positions across periodic boundaries.

    Mirrors ``qtft.analysis._unwrap_cluster_positions``: anchor on the first particle, then
    add particles one at a time, each shifted by the minimum-image convention relative to the
    running centre of mass. Returns positions made spatially contiguous (they may legitimately
    lie outside the primary box).
    """
    positions = np.asarray(positions, dtype=float)
    box = np.asarray(box_size, dtype=float)
    n = len(positions)
    if n == 0:
        return positions.copy()
    if not periodic:
        # Reflective walls: no periodic images, so a cluster is already contiguous.
        return positions.copy()
    unwrapped = np.zeros_like(positions)
    unwrapped[0] = positions[0]
    for i in range(1, n):
        com = unwrapped[:i].mean(axis=0)
        delta = positions[i] - com
        delta = delta - box * np.round(delta / box)   # minimum image
        unwrapped[i] = com + delta
    return unwrapped


def build_encapsulin_table(
    positions: np.ndarray,
    types: np.ndarray,
    topologies: List[np.ndarray],
    box_size,
    r_qt: float,
    voxel_nm: float = 4.0,
    periodic: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Build the FIB-SEM-schema encapsulin table from a single frame.

    Parameters
    ----------
    positions : (N, 3) particle positions (nm), ReaDDy (x, y, z) order.
    types : (N,) particle type-name strings ("Qt", "Ft", "QtC", "FtC").
    topologies : list of index arrays; each holds the particle indices of one ReaDDy
        topology (cluster). Indices refer into ``positions`` / ``types``.
    box_size : (3,) periodic box lengths (nm).
    r_qt : encapsulin radius (nm).
    voxel_nm : nominal voxel size for the ``*_vox`` columns (nm).

    Returns
    -------
    df : DataFrame with columns
        ``label, z_nm, y_nm, x_nm, z_vox, y_vox, x_vox, radius_nm, volume_nm3,
        cluster, is_clustered``.
    offset : (3,) the shift applied so the global minimum corner is the origin.
    """
    positions = np.asarray(positions, dtype=float)
    types = np.asarray(types)
    box = np.asarray(box_size, dtype=float)

    coords: List[np.ndarray] = []
    clusters: List[int] = []
    is_clustered: List[bool] = []
    next_cluster = 1
    assigned = np.zeros(len(positions), dtype=bool)

    # --- one cluster per topology ---
    # NOTE: engine.place_particles adds EVERY particle as its own single-particle topology,
    # so "free" here cannot mean "absent from the topology list" — free Qt sit in size-1
    # topologies. Boundness is therefore taken from the graph: a topology holding more than
    # one particle means its encapsulin is bonded to something (a Qt-Ft dimer counts), while
    # a size-1 topology is an unbound encapsulin -> a FIB-SEM singleton.
    for tp in topologies:
        tp = np.asarray(tp, dtype=int)
        if len(tp) == 0:
            continue
        assigned[tp] = True
        enc_mask = np.isin(types[tp], ENCAPSULIN_TYPES)   # Qt (free) or QtC (bound)
        if not enc_mask.any():
            continue                                       # ferritin-only topology: nothing to export
        uw = _unwrap(positions[tp], box, periodic=periodic)   # unwrap Qt + Ft together
        bound = len(tp) > 1
        for p in uw[enc_mask]:
            coords.append(p)
            clusters.append(next_cluster)
            is_clustered.append(bound)
        next_cluster += 1

    # --- defensive: encapsulins not covered by any topology (other placement paths) ---
    for i in np.flatnonzero(np.isin(types, ENCAPSULIN_TYPES) & ~assigned):
        coords.append(positions[i])
        clusters.append(next_cluster)
        is_clustered.append(False)
        next_cluster += 1

    if not coords:
        raise ValueError("No Qt/QtC encapsulins found in the frame.")

    xyz = np.asarray(coords, dtype=float)          # (M, 3), ReaDDy (x, y, z)
    offset = xyz.min(axis=0)
    xyz = xyz - offset                             # shift global min corner to origin (>= 0)

    vol = 4.0 / 3.0 * np.pi * float(r_qt) ** 3
    df = pd.DataFrame({
        "label": np.arange(1, len(xyz) + 1),
        "z_nm": xyz[:, 2], "y_nm": xyz[:, 1], "x_nm": xyz[:, 0],
        "z_vox": xyz[:, 2] / voxel_nm, "y_vox": xyz[:, 1] / voxel_nm, "x_vox": xyz[:, 0] / voxel_nm,
        "radius_nm": float(r_qt),
        "volume_nm3": vol,
        "cluster": np.asarray(clusters, dtype=int),
        "is_clustered": np.asarray(is_clustered, dtype=bool),
    })
    return df, offset


def load_final_frame(h5_file: str, config, verbose: bool = True):
    """Read the final frame of a ReaDDy trajectory.

    Returns ``(positions, types, topologies, box)`` where ``topologies`` is a list of
    particle-index arrays (ground-truth clusters). Handles the common case where the
    *particles observable* was not registered (``particles_observable_stride=None``) by
    streaming the trajectory and keeping only the last frame.
    """
    import readdy  # local import: only needed at runtime, keeps the module import light

    traj = readdy.Trajectory(h5_file)
    topo_times, topo_records = traj.read_observable_topologies()
    topo_times = np.asarray(topo_times)
    if len(topo_records) == 0:
        raise ValueError("Trajectory has no topology records.")

    type_id_to_name = {v: k for k, v in traj.particle_types.items()}

    positions = types = ids = None
    last_time = None
    have_real_time = False               # True only when the frame's step number is known
    try:
        ot, oty, oid, opos = traj.read_observable_particles()
        if len(ot) == 0:
            raise ValueError("empty particles observable")
        positions = np.asarray(opos[-1], dtype=float)
        ids = np.asarray(oid[-1])
        types = np.asarray([type_id_to_name.get(t, f"type_{t}") for t in oty[-1]])
        last_time = float(np.asarray(ot)[-1])
        have_real_time = True            # the observable carries genuine step numbers
        if verbose:
            logger.info("fibsem_export: using particles observable (final frame)")
    except (KeyError, ValueError, RuntimeError, OSError):
        # Fallback: stream frames and keep only the last one (memory-light).
        if verbose:
            logger.info("fibsem_export: particles observable absent; streaming to final frame")
        n_frames = 0
        # Extract each frame in-loop (overwriting) so the arrays hold the final frame
        # regardless of whether ReaDDy's read() iterator reuses its buffer.
        for frame in traj.read():
            positions = np.asarray([p.position for p in frame], dtype=float)
            types = np.asarray([p.type for p in frame])
            ids = np.asarray([p.id for p in frame])
            n_frames += 1
        if positions is None:
            raise ValueError("Trajectory has no frames.")
        if n_frames != len(topo_records):
            logger.warning(
                "fibsem_export: %d trajectory frames but %d topology records — the two "
                "observables were written at different strides; pairing the last of each.",
                n_frames, len(topo_records))

    # Pick the topology record that belongs to the extracted frame.
    #
    # trajectory.read() yields no step numbers, so the frame's step CANNOT be reconstructed
    # reliably: deriving it as (n_frames-1)*config.record_stride silently pairs final-frame
    # positions with clusters from a different time whenever the config's stride does not
    # match the file (a config from another run, an ensemble_config vs a replica, or a
    # stitched phased trajectory). Both series end at the end of the run, so pair the last
    # of each; only use nearest-in-time matching when a genuine step number is available.
    if len(topo_times) == 0:
        raise ValueError("Trajectory has no topology records.")
    if have_real_time:
        topo_idx = int(np.argmin(np.abs(topo_times - last_time)))
    else:
        topo_idx = len(topo_records) - 1
        last_time = float(topo_times[topo_idx])
    last_topos = topo_records[topo_idx]
    if verbose:
        logger.info("fibsem_export: final frame paired with topology record %d (step %g)",
                    topo_idx, last_time)

    # Map topology particle IDs -> array indices.
    id_to_idx = {int(pid): i for i, pid in enumerate(ids)}
    topologies: List[np.ndarray] = []
    for top in last_topos:
        idxs: List[int] = []
        for p in top.particles:
            if int(p) in id_to_idx:
                idxs.append(id_to_idx[int(p)])
            elif p < len(positions):
                idxs.append(int(p))
        topologies.append(np.asarray(idxs, dtype=int))

    box = np.asarray(config.box_size, dtype=float)
    return positions, types, topologies, box, last_time


def find_run_files(sim_dir: str, trajectory: str = None, config_json: str = None):
    """Resolve a run directory to ``(trajectory_path, config_path)``.

    The trajectory is resolved by :func:`qtft.analysis.resolve_trajectory`, the single place
    that knows this project's run layouts (plain run, ensemble replica, phased run, or a
    run whose phases were never combined). The config is found here: ``<param_string>_config.json``
    for a single run, or ``ensemble_config.json`` for a replica (also one level up).

    Explicit ``trajectory`` / ``config_json`` (absolute, or relative to ``sim_dir``) always win.
    """
    import glob
    from .analysis import resolve_trajectory

    traj = resolve_trajectory(sim_dir, explicit=trajectory)

    if config_json:
        cfg = config_json if os.path.isabs(config_json) else os.path.join(sim_dir, config_json)
    else:
        cands = sorted(glob.glob(os.path.join(sim_dir, "*_config.json")))
        cands += [os.path.join(sim_dir, "ensemble_config.json"),
                  os.path.join(os.path.dirname(sim_dir.rstrip("/")), "ensemble_config.json")]
        cfg = next((c for c in cands if os.path.isfile(c)), None)
        if cfg is None:
            raise FileNotFoundError(
                f"No config JSON found for {sim_dir!r} (looked for *_config.json and "
                f"ensemble_config.json here and one level up)")
    if not os.path.isfile(cfg):
        raise FileNotFoundError(f"Config not found: {cfg}")
    return traj, cfg


def export(
    trajectory_file: str,
    config,
    out_dir: str,
    voxel_nm: float = 4.0,
    file_tag: str = "_simulation",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Extract final-frame encapsulins and write the FIB-SEM-schema CSV + metadata JSON.

    Writes ``encapsulin_centroids{file_tag}.csv`` and
    ``structural_information_and_metadata{file_tag}.json`` into ``out_dir``.
    Returns ``(df, info)``.
    """
    os.makedirs(out_dir, exist_ok=True)
    positions, types, topologies, box, final_step = load_final_frame(
        trajectory_file, config, verbose=verbose)
    r_qt = float(config.qt.radius)
    df, offset = build_encapsulin_table(positions, types, topologies, box, r_qt,
                                       voxel_nm=voxel_nm,
                                       periodic=getattr(config, "is_periodic", True))

    csv_path = os.path.join(out_dir, f"encapsulin_centroids{file_tag}.csv")
    json_path = os.path.join(out_dir, f"structural_information_and_metadata{file_tag}.json")
    df.to_csv(csv_path, index=False)

    sizes = df["cluster"].value_counts()
    size_hist = {int(k): int(v) for k, v in sorted(sizes.value_counts().items())}
    info: Dict[str, Any] = {
        "source_trajectory": os.path.abspath(trajectory_file),
        "parameter_string": getattr(config, "output_file", None),
        "final_step": None if final_step is None else float(final_step),
        "final_time_us": (None if final_step is None
                          else float(final_step) * float(config.timestep) * 1e-3),
        "n_encapsulins": int(len(df)),
        "n_clustered": int(df["is_clustered"].sum()),
        "n_free": int((~df["is_clustered"]).sum()),
        "n_clusters": int(df["cluster"].nunique()),
        "n_multi_particle_clusters": int((sizes >= 2).sum()),
        "largest_cluster": int(sizes.max()),
        "cluster_size_distribution": size_hist,
        "qt_radius_nm": r_qt,
        "particle_volume_nm3": float(4.0 / 3.0 * np.pi * r_qt ** 3),
        "box_size_nm": [float(b) for b in box],
        "coordinate_offset_applied_nm": [float(o) for o in offset],
        "voxel_nm": float(voxel_nm),
        "note": (
            "Qt+QtC only (encapsulins); clusters are ReaDDy topologies (ground truth, no DBSCAN); "
            "free Qt are size-1 clusters; positions PBC-unwrapped per cluster then shifted so the "
            "min corner is the origin; volumes analytical (4/3 pi r^3); radius_nm is stored per row "
            "so an exact mass integral <M(R)> can be computed later without approximation."
        ),
    }
    # Full parameter set, if the config exposes a flat dict.
    try:
        info["config"] = config.to_flat_dict()
    except Exception:  # pragma: no cover - config API may differ
        pass

    with open(json_path, "w") as f:
        json.dump(info, f, indent=2, default=str)   # default=str: never crash on odd config values

    if verbose:
        logger.info("fibsem_export: wrote %s", csv_path)
        logger.info("fibsem_export: wrote %s", json_path)
    return df, info
