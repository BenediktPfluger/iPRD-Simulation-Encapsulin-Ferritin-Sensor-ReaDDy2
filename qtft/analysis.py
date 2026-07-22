"""
Qt-Ft Agglomeration Analysis Module

This module provides core analysis functions for ReaDDy2-based
Qt-Ft nanoparticle agglomeration simulations.

Contents:
    - Basic analysis functions (cluster statistics, bond counts, kinetics)
    - Advanced analysis functions (morphology, spatial distribution, contacts, composition)
    - Export utilities (XYZ conversion)
    - Ensemble data loading (JSON/NPZ files)

Related modules:
    - qtft.config / qtft.system / qtft.engine: configuration and simulation execution
    - qtft.plotting: all plotting and visualization (requires matplotlib)
    - qtft.ensemble: EnsembleSimulation class for multi-replica runs

This module has NO matplotlib dependency and can be used on headless servers.

Usage:
    import qtft
    import qtft.analysis as analysis

    # Analyze results
    cluster_stats = analysis.get_cluster_statistics(h5_file)
    bond_counts = analysis.get_bond_counts(h5_file)
    morphology = analysis.get_cluster_morphology(h5_file, config)
"""

from __future__ import annotations
import logging

import itertools
import json
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import readdy

# Import from simulation module
from .config import (
    SimulationConfig,
    ParticleConfig,
    TopologyConfig,
    LennardJonesConfig,
    NS_TO_US,
    _steps_to_us,
    format_duration,
)

# Try to import tqdm for progress bars, with fallback
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Create a dummy tqdm that just returns the iterator
    def tqdm(iterable, **kwargs):
        return iterable



# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
logger = logging.getLogger(__name__)


def get_cluster_statistics(
    h5_file: str,
    trajectory: Optional[readdy.Trajectory] = None,
) -> Dict[str, Any]:
    """
    Analyze cluster sizes and counts over time.
    
    Parameters
    ----------
    h5_file : str
        Path to trajectory HDF5 file
    trajectory : readdy.Trajectory, optional
        Pre-loaded trajectory object. If None, loads from h5_file.
    
    Returns
    -------
    dict with keys:
        times : ndarray
            Simulation step numbers (NOT time in ns). Multiply by timestep
            to get actual time in ns.
        n_clusters : ndarray
            Number of topologies at each time
        cluster_sizes : list of ndarray
            Size distribution at each time
        avg_sizes : ndarray
            Average cluster size over time
        max_sizes : ndarray
            Maximum cluster size over time
        trajectory : readdy.Trajectory
            The trajectory object (for reuse)
    """
    if trajectory is None:
        trajectory = readdy.Trajectory(h5_file)
    
    times, topology_records = trajectory.read_observable_topologies()
    
    n_clusters = []
    cluster_sizes = []
    
    for topologies in topology_records:
        n_clusters.append(len(topologies))
        sizes = [len(top.particles) for top in topologies]
        cluster_sizes.append(np.array(sizes) if sizes else np.array([]))
    
    times = np.array(times)
    n_clusters = np.array(n_clusters)
    avg_sizes = np.array([s.mean() if len(s) > 0 else 0 for s in cluster_sizes])
    max_sizes = np.array([s.max() if len(s) > 0 else 0 for s in cluster_sizes])
    
    return {
        "times": times,
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes,
        "avg_sizes": avg_sizes,
        "max_sizes": max_sizes,
        "trajectory": trajectory,
    }



def get_bond_counts(
    h5_file: str,
    trajectory: Optional[readdy.Trajectory] = None,
    verbose: bool = False,
    silent: bool = False,
) -> Dict[str, Any]:
    """
    Count bonds in topologies over time.
    
    Parameters
    ----------
    h5_file : str
        Path to trajectory HDF5 file
    trajectory : readdy.Trajectory, optional
        Pre-loaded trajectory object. If None, loads from h5_file.
    verbose : bool
        If True, print detailed breakdown of counting methods (default: False)
    silent : bool
        If True, suppress all output including the method summary (default: False)
    
    Returns
    -------
    dict with keys:
        times : ndarray
            Simulation step numbers (NOT time in ns). Multiply by timestep
            to get actual time in ns.
        n_bonds : ndarray
            Total bond count at each time
        bond_counting_method : str
            Description of the primary method used for counting
        trajectory : readdy.Trajectory
            The trajectory object (for reuse)
    """
    if trajectory is None:
        trajectory = readdy.Trajectory(h5_file)
    
    times, topology_records = trajectory.read_observable_topologies()
    
    n_bonds = []
    fallback_used_for_multiparticle = False
    
    # Track which methods are used
    method_counts = {"method1_edges": 0, "method2_graph": 0, "method3_fallback": 0}
    
    for topologies in topology_records:
        total = 0
        for top in topologies:
            # Try multiple ways to get edge count
            edge_count = None
            method_used = None
            
            # Method 1: Direct edges attribute
            if hasattr(top, 'edges') and top.edges is not None:
                try:
                    edge_count = len(top.edges)
                    method_used = "method1_edges"
                except (TypeError, AttributeError):
                    pass
            
            # Method 2: Graph edges (alternative ReaDDy API)
            if edge_count is None and hasattr(top, 'graph'):
                try:
                    edge_count = len(top.graph.edges)
                    method_used = "method2_graph"
                except (TypeError, AttributeError):
                    pass
            
            # Method 3: Fallback - only for single particles (0 bonds)
            if edge_count is None:
                n_particles = len(top.particles) if hasattr(top, 'particles') else 0
                if n_particles <= 1:
                    edge_count = 0
                    method_used = "method3_fallback"
                else:
                    # No explicit edge info: use n-1. When allow_loops is False (default),
                    # every reaction adds exactly one bond and never closes a ring, so clusters
                    # are acyclic trees and n_bonds == n_particles - 1 EXACTLY. If allow_loops
                    # is enabled, clusters may contain rings (edges >= n_particles) and this
                    # fallback would UNDERCOUNT — Methods 1/2 (len(edges)) remain exact and are
                    # used whenever edge data is readable (essentially always in practice).
                    edge_count = n_particles - 1
                    method_used = "method3_fallback"
                    fallback_used_for_multiparticle = True
            
            if method_used:
                method_counts[method_used] += 1
            
            total += edge_count
        n_bonds.append(total)
    
    # Determine primary method used
    total_counts = sum(method_counts.values())
    if total_counts > 0:
        if method_counts["method1_edges"] > 0 and method_counts["method1_edges"] >= method_counts["method2_graph"]:
            primary_method = "Method 1 (topology.edges) - exact count"
            method_type = "exact"
        elif method_counts["method2_graph"] > 0:
            primary_method = "Method 2 (topology.graph.edges) - exact count"
            method_type = "exact"
        else:
            # n-1 is exact for acyclic (allow_loops=False) tree clusters; with loops it
            # would undercount, but Methods 1/2 are preferred whenever edges are readable.
            primary_method = "Method 3 (n-1, exact for tree clusters / allow_loops=False)"
            method_type = "exact"
    else:
        primary_method = "No topologies found"
        method_type = "none"
    
    # Print method summary (unless silent)
    if not silent:
        logger.info(f"  Bond counting: {primary_method}")

    # Print detailed breakdown if verbose
    if verbose and not silent:
        logger.info(f"    Detailed breakdown:")
        logger.info(f"      Method 1 (top.edges):       {method_counts['method1_edges']} topologies")
        logger.info(f"      Method 2 (top.graph.edges): {method_counts['method2_graph']} topologies")
        logger.info(f"      Method 3 (n-1, tree-exact): {method_counts['method3_fallback']} topologies")

    return {
        "times": np.array(times),
        "n_bonds": np.array(n_bonds),
        "bond_counting_method": primary_method,
        "trajectory": trajectory,
    }


# =============================================================================
# ADVANCED CLUSTER ANALYSIS FUNCTIONS
# =============================================================================

def _extract_frame_data(
    h5_file: str,
    config: SimulationConfig,
    stride: int = 1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Extract detailed particle and topology data from trajectory.
    
    This is the foundation for structural analyses (morphology, spatial, contacts).
    Uses ReaDDy's API for robust data extraction.
    
    Parameters
    ----------
    h5_file : str
        Path to trajectory HDF5 file
    config : SimulationConfig
        Simulation configuration
    stride : int
        Analyze every Nth frame (default: 1 = all frames)
    verbose : bool
        Print progress messages (default: True)
    
    Returns
    -------
    dict with keys:
        times : ndarray - simulation step numbers (NOT ns; convert via _steps_to_us)
        n_frames : int - number of frames analyzed
        frame_indices : ndarray - original frame indices
        positions : list of ndarray - particle positions per frame (n_particles, 3)
        types : list of ndarray - particle type names per frame
        topology_ids : list of ndarray - topology index for each particle
        topology_particles : list of list of list - particle indices per topology per frame
        topology_edges : list of list of list - edge pairs per topology per frame
        box_size : tuple - simulation box dimensions
    """
    trajectory = readdy.Trajectory(h5_file)
    
    # Get topology data (for cluster membership and edges)
    topo_times_all, topology_records_all = trajectory.read_observable_topologies()
    topo_times_all = np.array(topo_times_all)
    
    # Get type ID to name mapping from ReaDDy
    # trajectory.particle_types returns {name: id}, we need {id: name}
    type_name_to_id = trajectory.particle_types
    type_id_to_name = {v: k for k, v in type_name_to_id.items()}
    
    # Try to use read_observable_particles() first (if particles observable was registered)
    # Fall back to trajectory.read() if not available
    use_particles_observable = False
    obs_times = obs_types = obs_ids = obs_positions = None
    try:
        # Check if particles observable exists by trying to read it
        obs_times, obs_types, obs_ids, obs_positions = trajectory.read_observable_particles()
        if len(obs_times) > 0:
            use_particles_observable = True
            if verbose:
                logger.info("    Using particles observable for data extraction")
    except (KeyError, ValueError, RuntimeError, OSError):
        # Raised when the particles observable was not registered for this trajectory.
        use_particles_observable = False
        if verbose:
            logger.info("    Using trajectory.read() for data extraction (particles observable not available)")
    
    positions_list = []
    types_list = []
    topology_ids_list = []
    topology_particles_list = []
    topology_edges_list = []
    extracted_times = []
    extracted_frame_indices = []
    
    if use_particles_observable:
        # Method 1: Use particles observable (faster, available if enabled)
        # IMPORTANT: particles observable may have different stride than topology observable
        # We need to use particles observable times as the basis and find matching topology records
        
        obs_times = np.array(obs_times)
        
        # Apply stride to particles observable frames
        obs_frame_indices = np.arange(0, len(obs_times), stride)
        
        frame_iter = tqdm(obs_frame_indices, desc="    Processing frames", 
                         disable=not TQDM_AVAILABLE or not verbose, unit="frame")
        
        for obs_idx in frame_iter:
            if obs_idx >= len(obs_times):
                break
            
            # Get the time for this particles observable frame
            frame_time = obs_times[obs_idx]
            
            # Get positions and types for this frame
            frame_positions = np.array(obs_positions[obs_idx])
            frame_type_ids = np.array(obs_types[obs_idx])
            frame_type_names = np.array([type_id_to_name.get(t, f"type_{t}") for t in frame_type_ids])
            frame_ids = np.array(obs_ids[obs_idx])
            
            positions_list.append(frame_positions)
            types_list.append(frame_type_names)
            extracted_times.append(frame_time)
            extracted_frame_indices.append(obs_idx)
            
            # Find the matching topology record by time
            # Topology observable may have finer resolution, find closest match
            topo_idx = np.argmin(np.abs(topo_times_all - frame_time))
            
            # Get topology info for this frame
            _extract_topology_info(
                topo_idx, frame_positions, frame_ids, topology_records_all,
                topology_ids_list, topology_particles_list, topology_edges_list
            )
    else:
        # Method 2: Use trajectory.read() with memory-efficient streaming
        # Instead of loading all frames into memory, we iterate and select
        if verbose:
            logger.info("    Reading trajectory frames...")
        
        # `stride` applies to trajectory frames. Each kept trajectory frame is matched to the
        # topology record at the SAME simulation step by comparing times, instead of assuming
        # trajectory-frame index == topology-record index. That assumption only holds when
        # record_stride == observable_stride; otherwise positions get paired with the wrong
        # topology record.
        record_stride = int(config.record_stride)
        current_frame = 0

        # Create iterator with progress bar
        traj_iter = trajectory.read()
        if TQDM_AVAILABLE and verbose:
            # We don't know total frames, but can estimate from topology records
            traj_iter = tqdm(traj_iter, desc="    Loading trajectory",
                            total=len(topo_times_all), unit="frame")

        for frame in traj_iter:
            if current_frame % stride == 0:
                # Extract positions, types, and IDs from frame
                frame_positions = []
                frame_type_names = []
                frame_ids = []

                for particle in frame:
                    frame_positions.append(particle.position)
                    frame_type_names.append(particle.type)
                    frame_ids.append(particle.id)

                frame_positions = np.array(frame_positions) if frame_positions else np.zeros((0, 3))
                frame_type_names = np.array(frame_type_names) if frame_type_names else np.array([])
                frame_ids = np.array(frame_ids) if frame_ids else np.array([])

                # Actual simulation step of this trajectory frame, and the topology record
                # recorded closest to that step.
                frame_step = current_frame * record_stride
                if len(topo_times_all) > 0:
                    topo_idx = int(np.argmin(np.abs(topo_times_all - frame_step)))
                else:
                    topo_idx = 0

                positions_list.append(frame_positions)
                types_list.append(frame_type_names)
                extracted_times.append(frame_step)
                extracted_frame_indices.append(current_frame)

                # Get topology info for this frame (time-matched topology record)
                _extract_topology_info(
                    topo_idx, frame_positions, frame_ids, topology_records_all,
                    topology_ids_list, topology_particles_list, topology_edges_list
                )

            current_frame += 1
    
    return {
        "times": np.array(extracted_times),
        "n_frames": len(positions_list),
        "frame_indices": np.array(extracted_frame_indices),
        "positions": positions_list,
        "types": types_list,
        "topology_ids": topology_ids_list,
        "topology_particles": topology_particles_list,
        "topology_edges": topology_edges_list,
        "box_size": config.box_size,
    }



def _extract_topology_info(
    frame_idx: int,
    frame_positions: np.ndarray,
    frame_ids: np.ndarray,
    topology_records_all: list,
    topology_ids_list: list,
    topology_particles_list: list,
    topology_edges_list: list,
) -> None:
    """
    Extract topology information for a single frame.
    
    Helper function to avoid code duplication between particles observable
    and trajectory.read() methods.
    
    Modifies the output lists in-place.
    """
    if frame_idx < len(topology_records_all):
        topologies = topology_records_all[frame_idx]
        
        # Build mapping from particle id to array index
        id_to_idx = {pid: idx for idx, pid in enumerate(frame_ids)}
        
        topo_ids = np.full(len(frame_positions), -1, dtype=int)
        topo_particles = []
        topo_edges = []
        
        for topo_idx, top in enumerate(topologies):
            # Get particle indices (convert from IDs if needed)
            particle_indices = []
            for p in top.particles:
                if p in id_to_idx:
                    particle_indices.append(id_to_idx[p])
                elif p < len(frame_positions):
                    particle_indices.append(p)
            
            topo_particles.append(particle_indices)
            
            for p_idx in particle_indices:
                if p_idx < len(topo_ids):
                    topo_ids[p_idx] = topo_idx
            
            # Get edges
            edges = []
            if hasattr(top, 'edges') and top.edges is not None:
                try:
                    edges = list(top.edges)
                except (TypeError, AttributeError):
                    pass
            topo_edges.append(edges)
        
        topology_ids_list.append(topo_ids)
        topology_particles_list.append(topo_particles)
        topology_edges_list.append(topo_edges)
    else:
        topology_ids_list.append(np.full(len(frame_positions), -1, dtype=int))
        topology_particles_list.append([])
        topology_edges_list.append([])



def _unwrap_cluster_positions(
    positions: np.ndarray,
    box_size: Tuple[float, float, float],
) -> np.ndarray:
    """
    Unwrap cluster positions to handle periodic boundary conditions.
    
    Uses the first particle as reference and unwraps others to be
    within half a box length of the growing cluster center.
    
    Parameters
    ----------
    positions : ndarray
        Particle positions (n_particles, 3)
    box_size : tuple
        Box dimensions (Lx, Ly, Lz)
    
    Returns
    -------
    ndarray
        Unwrapped positions
    """
    if len(positions) <= 1:
        return positions.copy()
    
    box = np.array(box_size)
    unwrapped = np.zeros_like(positions)
    unwrapped[0] = positions[0]
    
    # Iteratively add particles, unwrapping relative to current center
    for i in range(1, len(positions)):
        # Current center of mass of unwrapped particles
        com = unwrapped[:i].mean(axis=0)
        
        # Unwrap new particle relative to COM
        delta = positions[i] - com
        
        # Apply minimum image convention
        delta = delta - box * np.round(delta / box)
        
        unwrapped[i] = com + delta
    
    return unwrapped



def _calculate_radius_of_gyration(positions: np.ndarray) -> float:
    """
    Calculate radius of gyration for a set of positions.
    
    Rg = sqrt(1/N * sum(|r_i - r_com|^2))
    
    Parameters
    ----------
    positions : ndarray
        Particle positions (n_particles, 3), should be unwrapped
    
    Returns
    -------
    float
        Radius of gyration
    """
    if len(positions) < 2:
        return 0.0
    
    com = positions.mean(axis=0)
    squared_distances = np.sum((positions - com) ** 2, axis=1)
    rg = np.sqrt(np.mean(squared_distances))
    
    return rg



def _calculate_ideal_rg(n_particles: int, particle_radius: float) -> float:
    """
    Calculate ideal Rg for a compact spherical cluster.
    
    For a uniform sphere: Rg = sqrt(3/5) * R
    Estimate cluster radius from volume: R = (3N * v_particle / (4π))^(1/3)
    
    Parameters
    ----------
    n_particles : int
        Number of particles in cluster
    particle_radius : float
        Average particle radius
    
    Returns
    -------
    float
        Ideal Rg for compact arrangement
    """
    if n_particles < 2:
        return 0.0
    
    # Estimate cluster radius assuming close packing
    # Volume per particle ≈ (4/3)π r³, packing fraction ≈ 0.64
    v_particle = (4/3) * np.pi * particle_radius**3
    v_cluster = n_particles * v_particle / 0.64  # Account for packing
    r_cluster = (3 * v_cluster / (4 * np.pi)) ** (1/3)
    
    # Rg for uniform sphere
    rg_ideal = np.sqrt(3/5) * r_cluster
    
    return rg_ideal



def get_cluster_morphology(
    h5_file: str,
    config: SimulationConfig,
    stride: int = 1,
    min_cluster_size: int = 3,
    frame_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Calculate radius of gyration and compactness for clusters.
    
    Parameters
    ----------
    h5_file : str
        Path to trajectory HDF5 file
    config : SimulationConfig
        Simulation configuration
    stride : int
        Analyze every Nth frame
    min_cluster_size : int
        Minimum cluster size for Rg calculation (default: 3)
    frame_data : dict, optional
        Pre-extracted frame data from _extract_frame_data(). If provided,
        h5_file and stride are ignored. This allows sharing data between
        multiple analysis functions for better performance.
    
    Returns
    -------
    dict with keys:
        times : ndarray - simulation step numbers (NOT ns; convert via _steps_to_us)
        rg_per_cluster : list of list - Rg for each cluster at each frame
        size_per_cluster : list of list - size of each cluster at each frame
        mean_rg : ndarray - mean Rg per frame
        std_rg : ndarray - std of Rg per frame
        rg_normalized : list of list - Rg / Rg_ideal per cluster
        mean_rg_normalized : ndarray - mean normalized Rg per frame
    """
    if frame_data is None:
        logger.info("  Extracting frame data for morphology analysis...")
        frame_data = _extract_frame_data(h5_file, config, stride=stride)
    
    times = frame_data["times"]
    box_size = frame_data["box_size"]
    
    # Average particle radius for ideal Rg calculation
    avg_radius = (config.qt.radius + config.ft.radius) / 2
    
    rg_per_cluster = []
    size_per_cluster = []
    rg_normalized = []
    mean_rg = []
    std_rg = []
    mean_rg_normalized = []
    
    frame_iter = tqdm(range(frame_data["n_frames"]), desc="  Morphology", 
                     disable=not TQDM_AVAILABLE, unit="frame")
    for frame_idx in frame_iter:
        positions = frame_data["positions"][frame_idx]
        topo_particles = frame_data["topology_particles"][frame_idx]
        
        frame_rg = []
        frame_sizes = []
        frame_rg_norm = []
        
        for particle_indices in topo_particles:
            # Fix Issue #3: ensure particle_indices is array for proper indexing
            particle_indices = np.asarray(particle_indices)
            n_particles = len(particle_indices)
            
            if n_particles < min_cluster_size:
                continue
            
            # Get positions of particles in this cluster
            cluster_pos = positions[particle_indices]
            
            # Unwrap for PBC
            cluster_pos_unwrapped = _unwrap_cluster_positions(cluster_pos, box_size)
            
            # Calculate Rg
            rg = _calculate_radius_of_gyration(cluster_pos_unwrapped)
            rg_ideal = _calculate_ideal_rg(n_particles, avg_radius)
            
            frame_rg.append(rg)
            frame_sizes.append(n_particles)
            frame_rg_norm.append(rg / rg_ideal if rg_ideal > 0 else 1.0)
        
        rg_per_cluster.append(frame_rg)
        size_per_cluster.append(frame_sizes)
        rg_normalized.append(frame_rg_norm)
        
        if len(frame_rg) > 0:
            mean_rg.append(np.mean(frame_rg))
            std_rg.append(np.std(frame_rg))
            mean_rg_normalized.append(np.mean(frame_rg_norm))
        else:
            mean_rg.append(0.0)
            std_rg.append(0.0)
            mean_rg_normalized.append(0.0)
    
    return {
        "times": times,
        "rg_per_cluster": rg_per_cluster,
        "size_per_cluster": size_per_cluster,
        "mean_rg": np.array(mean_rg),
        "std_rg": np.array(std_rg),
        "rg_normalized": rg_normalized,
        "mean_rg_normalized": np.array(mean_rg_normalized),
        "min_cluster_size": min_cluster_size,
    }



def get_binding_kinetics(
    h5_file: str,
    config: SimulationConfig,
    trajectory: Optional[readdy.Trajectory] = None,
    smoothing_window: int = 10,
) -> Dict[str, Any]:
    """
    Analyze binding rates and reaction kinetics.
    
    Note: Unlike other structural analysis functions (morphology, spatial, contacts),
    this function always analyzes ALL frames because it uses pre-computed observables
    rather than extracting per-frame data. This makes it fast but means it doesn't
    support a stride parameter.
    
    Parameters
    ----------
    h5_file : str
        Path to trajectory HDF5 file
    config : SimulationConfig
        Simulation configuration
    trajectory : readdy.Trajectory, optional
        Pre-loaded trajectory
    smoothing_window : int
        Window size for smoothing bond rate calculation
    
    Returns
    -------
    dict with keys:
        times : ndarray - step numbers (NOT time in ns)
        n_bonds : ndarray - total bonds over time
        bond_rate : ndarray - d(bonds)/dt smoothed (bonds/step)
        free_qt : ndarray - number of free Qt
        free_ft : ndarray - number of free Ft
        clustered_qt : ndarray - number of QtC
        clustered_ft : ndarray - number of FtC
        fraction_bound_qt : ndarray - QtC / (Qt + QtC)
        fraction_bound_ft : ndarray - FtC / (Ft + FtC)
        half_time_qt : float or None - step number at 50% Qt bound
        half_time_ft : float or None - step number at 50% Ft bound
    """
    if trajectory is None:
        trajectory = readdy.Trajectory(h5_file)
    
    # Get particle counts
    times, counts = trajectory.read_observable_number_of_particles()
    times = np.array(times)
    counts = np.array(counts)
    
    # Get type indices (assuming order: Qt, Ft, QtC, FtC)
    # This matches how observables are registered
    free_qt = counts[:, 0]
    free_ft = counts[:, 1]
    clustered_qt = counts[:, 2]
    clustered_ft = counts[:, 3]
    
    # Get bond counts
    bond_data = get_bond_counts(h5_file, trajectory=trajectory)
    n_bonds = bond_data["n_bonds"]
    
    # Ensure same length (bonds might have different stride)
    if len(n_bonds) != len(times):
        # Interpolate bonds to match times
        bond_times = bond_data["times"]
        n_bonds = np.interp(times, bond_times, n_bonds)
    
    # Calculate fractions
    total_qt = free_qt + clustered_qt
    total_ft = free_ft + clustered_ft
    
    fraction_bound_qt = np.where(total_qt > 0, clustered_qt / total_qt, 0.0)
    fraction_bound_ft = np.where(total_ft > 0, clustered_ft / total_ft, 0.0)
    
    # Calculate bond rate (smoothed derivative)
    dt = np.diff(times)
    d_bonds = np.diff(n_bonds)
    
    # Avoid division by zero
    dt = np.where(dt > 0, dt, 1e-10)
    raw_rate = d_bonds / dt
    
    # Smooth the rate
    if smoothing_window > 1 and len(raw_rate) > smoothing_window:
        kernel = np.ones(smoothing_window) / smoothing_window
        bond_rate_smoothed = np.convolve(raw_rate, kernel, mode='same')
    else:
        bond_rate_smoothed = raw_rate
    
    # Pad to match original length
    bond_rate = np.zeros(len(times))
    bond_rate[:-1] = bond_rate_smoothed
    bond_rate[-1] = bond_rate_smoothed[-1] if len(bond_rate_smoothed) > 0 else 0
    
    # Find half-times
    def find_half_time(times, fraction):
        if fraction[-1] < 0.5:
            return None  # Never reached 50%
        # First step at/above 0.5. Use first-crossing rather than searchsorted, which
        # assumes a sorted array (fraction_bound is noisy and not strictly monotonic).
        crossings = np.flatnonzero(fraction >= 0.5)
        if len(crossings) == 0:
            return None
        idx = int(crossings[0])
        if idx == 0:
            return times[0]
        # Linear interpolation between the last sub-0.5 point and the first crossing
        f0, f1 = fraction[idx-1], fraction[idx]
        t0, t1 = times[idx-1], times[idx]
        if f1 == f0:
            return t0
        return t0 + (0.5 - f0) * (t1 - t0) / (f1 - f0)
    
    half_time_qt = find_half_time(times, fraction_bound_qt)
    half_time_ft = find_half_time(times, fraction_bound_ft)
    
    return {
        "times": times,
        "n_bonds": n_bonds,
        "bond_rate": bond_rate,
        "free_qt": free_qt,
        "free_ft": free_ft,
        "clustered_qt": clustered_qt,
        "clustered_ft": clustered_ft,
        "fraction_bound_qt": fraction_bound_qt,
        "fraction_bound_ft": fraction_bound_ft,
        "half_time_qt": half_time_qt,
        "half_time_ft": half_time_ft,
        "trajectory": trajectory,
    }



def load_phased_observables(
    config: SimulationConfig,
    phase_files: Optional[List[str]] = None,
    smoothing_window: int = 10,
) -> Dict[str, Any]:
    """Stitch per-phase trajectories onto one continuous time axis.

    A phased agglomeration<->deagglomeration run (see engine.run_phased) writes one
    trajectory per phase, each with step indices restarting at 0. This loads them in
    order, offsets each phase's steps by the cumulative steps of the prior phases
    (config.phase_step_offsets), converts to µs, and concatenates the bond, cluster, and
    binding-kinetics series so they can be plotted/analyzed as a single continuous cycle.

    This is matplotlib-free and is reused by both single-run plotting and the ensemble
    per-replica collection.

    Parameters
    ----------
    config : SimulationConfig
        Must have a non-empty ``phases`` list.
    phase_files : list of str, optional
        Per-phase trajectory paths in order. Defaults to ``config.phase_output_files``
        (pass explicitly for ensemble replicas whose files live under a replica dir).
    smoothing_window : int
        Forwarded to get_binding_kinetics.

    Returns
    -------
    dict with keys:
        time_us : ndarray              -- continuous µs axis for bonds & clusters
        n_bonds, n_clusters, avg_sizes, max_sizes : ndarray (aligned to time_us)
        kin_time_us : ndarray          -- continuous µs axis for binding kinetics
        fraction_bound_qt, fraction_bound_ft : ndarray (aligned to kin_time_us)
        phase_boundaries_us : list     -- µs positions of the internal phase switches
        phase_starts_us : list         -- µs start time of each phase
        phase_names : list             -- phase labels in order
        total_time_us : float
    """
    if not config.phases:
        raise ValueError("load_phased_observables requires config.phases to be set")

    files = phase_files if phase_files is not None else config.phase_output_files
    offsets = config.phase_step_offsets
    if len(files) != len(offsets):
        raise ValueError(
            f"Expected {len(offsets)} phase files, got {len(files)}"
        )
    ts = config.timestep

    bonds_t, bonds_v = [], []
    clus_t, clus_n, clus_avg, clus_max = [], [], [], []
    kin_t, kin_fbq, kin_fbf = [], [], []

    for f, off in zip(files, offsets):
        traj = readdy.Trajectory(f)
        bc = get_bond_counts(f, trajectory=traj, silent=True)
        cs = get_cluster_statistics(f, trajectory=traj)
        kin = get_binding_kinetics(f, config, trajectory=traj, smoothing_window=smoothing_window)

        bonds_t.append(np.asarray(bc["times"]) + off)
        bonds_v.append(np.asarray(bc["n_bonds"]))

        clus_t.append(np.asarray(cs["times"]) + off)
        clus_n.append(np.asarray(cs["n_clusters"]))
        clus_avg.append(np.asarray(cs["avg_sizes"]))
        clus_max.append(np.asarray(cs["max_sizes"]))

        kin_t.append(np.asarray(kin["times"]) + off)
        kin_fbq.append(np.asarray(kin["fraction_bound_qt"]))
        kin_fbf.append(np.asarray(kin["fraction_bound_ft"]))

    # Internal phase switches (start of each phase after the first) and per-phase starts.
    boundaries_us = list(_steps_to_us(np.asarray(offsets[1:]), ts)) if len(offsets) > 1 else []
    starts_us = list(_steps_to_us(np.asarray(offsets), ts))

    return {
        "time_us": _steps_to_us(np.concatenate(bonds_t), ts),
        "n_bonds": np.concatenate(bonds_v),
        "cluster_time_us": _steps_to_us(np.concatenate(clus_t), ts),
        "n_clusters": np.concatenate(clus_n),
        "avg_sizes": np.concatenate(clus_avg),
        "max_sizes": np.concatenate(clus_max),
        "kin_time_us": _steps_to_us(np.concatenate(kin_t), ts),
        "fraction_bound_qt": np.concatenate(kin_fbq),
        "fraction_bound_ft": np.concatenate(kin_fbf),
        "phase_boundaries_us": boundaries_us,
        "phase_starts_us": starts_us,
        "phase_names": [p.name for p in config.phases],
        "total_time_us": config.total_simulation_time_us,
    }


def get_spatial_distribution(
    h5_file: str,
    config: SimulationConfig,
    stride: int = 1,
    min_cluster_size: int = 2,
    frame_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Analyze spatial distribution of clusters.
    
    Parameters
    ----------
    h5_file : str
        Path to trajectory HDF5 file
    config : SimulationConfig
        Simulation configuration
    stride : int
        Analyze every Nth frame
    min_cluster_size : int
        Minimum cluster size to include (default: 2)
    frame_data : dict, optional
        Pre-extracted frame data from _extract_frame_data(). If provided,
        h5_file and stride are ignored. This allows sharing data between
        multiple analysis functions for better performance.
    
    Returns
    -------
    dict with keys:
        times : ndarray - simulation step numbers (NOT ns; convert via _steps_to_us)
        cluster_centers : list of ndarray - center of mass positions per frame
        cluster_sizes : list of ndarray - size of each cluster per frame
        nn_distances : list of ndarray - inter-cluster NN distance per cluster
        mean_nn_dist : ndarray - mean inter-cluster NN distance per frame
        std_nn_dist : ndarray - std of inter-cluster NN distance per frame
        mean_intra_nn_dist : ndarray - mean intra-cluster NN distance per frame
        std_intra_nn_dist : ndarray - std of intra-cluster NN distance per frame
        expected_nn_dist : ndarray - expected NN for random distribution per frame
        n_clusters : ndarray - number of clusters per frame
        box_size : tuple - simulation box dimensions
    """
    if frame_data is None:
        logger.info("  Extracting frame data for spatial analysis...")
        frame_data = _extract_frame_data(h5_file, config, stride=stride)
    
    times = frame_data["times"]
    box_size = frame_data["box_size"]
    box_volume = box_size[0] * box_size[1] * box_size[2]
    
    cluster_centers_list = []
    cluster_sizes_list = []
    nn_distances_list = []
    mean_nn_dist = []
    std_nn_dist = []
    mean_intra_nn_dist = []
    std_intra_nn_dist = []
    expected_nn_dist = []
    n_clusters_list = []
    
    frame_iter = tqdm(range(frame_data["n_frames"]), desc="  Spatial", 
                     disable=not TQDM_AVAILABLE, unit="frame")
    for frame_idx in frame_iter:
        positions = frame_data["positions"][frame_idx]
        topo_particles = frame_data["topology_particles"][frame_idx]
        
        # Calculate cluster centers and intra-cluster NN distances
        centers = []
        sizes = []
        intra_nn_per_cluster = []  # Mean intra-cluster NN distance for each cluster
        
        for particle_indices in topo_particles:
            # Fix Issue #3: ensure particle_indices is array for proper indexing
            particle_indices = np.asarray(particle_indices)
            n_particles = len(particle_indices)
            
            if n_particles < min_cluster_size:
                continue
            
            # Get positions and unwrap
            cluster_pos = positions[particle_indices]
            cluster_pos_unwrapped = _unwrap_cluster_positions(cluster_pos, box_size)
            
            # Center of mass
            com = cluster_pos_unwrapped.mean(axis=0)
            
            # Wrap COM back into box
            com = com - np.array(box_size) * np.floor(com / np.array(box_size) + 0.5)
            
            centers.append(com)
            sizes.append(n_particles)
            
            # Intra-cluster NN distance: for each particle in the cluster,
            # find the distance to its nearest neighbor within the same cluster
            if n_particles >= 3:
                # Compute pairwise distances within the cluster (unwrapped positions)
                diffs = cluster_pos_unwrapped[:, np.newaxis, :] - cluster_pos_unwrapped[np.newaxis, :, :]
                dists = np.linalg.norm(diffs, axis=2)
                # Set diagonal to inf so a particle doesn't match itself
                np.fill_diagonal(dists, np.inf)
                # Nearest neighbor distance for each particle
                particle_nn_dists = dists.min(axis=1)
                intra_nn_per_cluster.append(np.mean(particle_nn_dists))
        
        centers = np.array(centers) if len(centers) > 0 else np.zeros((0, 3))
        sizes = np.array(sizes) if len(sizes) > 0 else np.array([])
        
        cluster_centers_list.append(centers)
        cluster_sizes_list.append(sizes)
        n_clusters_list.append(len(centers))
        
        # Inter-cluster NN distances (between cluster centers of mass)
        nn_dists = []
        if len(centers) > 1:
            for i, c1 in enumerate(centers):
                min_dist = np.inf
                for j, c2 in enumerate(centers):
                    if i == j:
                        continue
                    # Minimum image distance
                    delta = c1 - c2
                    delta = delta - np.array(box_size) * np.round(delta / np.array(box_size))
                    dist = np.linalg.norm(delta)
                    min_dist = min(min_dist, dist)
                nn_dists.append(min_dist)
        
        nn_dists = np.array(nn_dists) if len(nn_dists) > 0 else np.array([])
        nn_distances_list.append(nn_dists)
        
        if len(nn_dists) > 0:
            mean_nn_dist.append(np.mean(nn_dists))
            std_nn_dist.append(np.std(nn_dists))
        else:
            mean_nn_dist.append(0.0)
            std_nn_dist.append(0.0)
        
        # Intra-cluster NN: average across all clusters in this frame
        if len(intra_nn_per_cluster) > 0:
            mean_intra_nn_dist.append(np.mean(intra_nn_per_cluster))
            std_intra_nn_dist.append(np.std(intra_nn_per_cluster))
        else:
            mean_intra_nn_dist.append(0.0)
            std_intra_nn_dist.append(0.0)
        
        # Expected NN distance for random distribution
        n_clusters = len(centers)
        if n_clusters > 1:
            # For N points in volume V: <d_NN> ≈ 0.554 * (V/N)^(1/3)
            expected = 0.554 * (box_volume / n_clusters) ** (1/3)
        else:
            expected = 0.0
        expected_nn_dist.append(expected)
    
    return {
        "times": times,
        "cluster_centers": cluster_centers_list,
        "cluster_sizes": cluster_sizes_list,
        "nn_distances": nn_distances_list,
        "mean_nn_dist": np.array(mean_nn_dist),
        "std_nn_dist": np.array(std_nn_dist),
        "mean_intra_nn_dist": np.array(mean_intra_nn_dist),
        "std_intra_nn_dist": np.array(std_intra_nn_dist),
        "expected_nn_dist": np.array(expected_nn_dist),
        "n_clusters": np.array(n_clusters_list),
        "box_size": box_size,
        "min_cluster_size": min_cluster_size,
    }



def get_contact_analysis(
    h5_file: str,
    config: SimulationConfig,
    stride: int = 1,
    frame_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Analyze bonding coordination within clusters.
    
    Parameters
    ----------
    h5_file : str
        Path to trajectory HDF5 file
    config : SimulationConfig
        Simulation configuration
    stride : int
        Analyze every Nth frame
    frame_data : dict, optional
        Pre-extracted frame data from _extract_frame_data(). If provided,
        h5_file and stride are ignored. This allows sharing data between
        multiple analysis functions for better performance.
    
    Returns
    -------
    dict with keys:
        times : ndarray - simulation step numbers (NOT ns; convert via _steps_to_us)
        mean_coord_qt : ndarray - mean coordination of QtC over time
        mean_coord_ft : ndarray - mean coordination of FtC over time
        std_coord_qt : ndarray - std of QtC coordination
        std_coord_ft : ndarray - std of FtC coordination
        max_coord_qt : ndarray - max QtC coordination per frame
        max_coord_ft : ndarray - max FtC coordination per frame
        coord_dist_qt : list of ndarray - coordination distribution per frame
        coord_dist_ft : list of ndarray - coordination distribution per frame
        bonds_per_cluster : list of ndarray - bonds in each cluster per frame
        sizes_per_cluster : list of ndarray - size of each cluster per frame
    """
    if frame_data is None:
        logger.info("  Extracting frame data for contact analysis...")
        frame_data = _extract_frame_data(h5_file, config, stride=stride)
    
    times = frame_data["times"]
    
    # Particle type names for identification
    qt_cluster_name = config.qt.cluster_name
    ft_cluster_name = config.ft.cluster_name
    
    mean_coord_qt = []
    mean_coord_ft = []
    std_coord_qt = []
    std_coord_ft = []
    max_coord_qt = []
    max_coord_ft = []
    coord_dist_qt = []
    coord_dist_ft = []
    bonds_per_cluster = []
    sizes_per_cluster = []
    
    frame_iter = tqdm(range(frame_data["n_frames"]), desc="  Contacts", 
                     disable=not TQDM_AVAILABLE, unit="frame")
    for frame_idx in frame_iter:
        types = frame_data["types"][frame_idx]
        topo_particles = frame_data["topology_particles"][frame_idx]
        topo_edges = frame_data["topology_edges"][frame_idx]
        
        # Build global coordination count per particle
        n_particles = len(types)
        coordination = np.zeros(n_particles, dtype=int)
        
        frame_bonds_per_cluster = []
        frame_sizes_per_cluster = []
        
        for topo_idx, (particle_indices, edges) in enumerate(zip(topo_particles, topo_edges)):
            # Count bonds in this topology
            n_bonds = len(edges)
            frame_bonds_per_cluster.append(n_bonds)
            frame_sizes_per_cluster.append(len(particle_indices))
            
            # Count coordination per particle
            # NOTE: edge indices are LOCAL to the topology (0, 1, 2, ...)
            # We must convert them to GLOBAL particle indices using particle_indices
            for edge in edges:
                if len(edge) >= 2:
                    local_p1, local_p2 = edge[0], edge[1]
                    # Convert local indices to global indices
                    if local_p1 < len(particle_indices) and local_p2 < len(particle_indices):
                        p1 = particle_indices[local_p1]
                        p2 = particle_indices[local_p2]
                        if p1 < n_particles:
                            coordination[p1] += 1
                        if p2 < n_particles:
                            coordination[p2] += 1
        
        bonds_per_cluster.append(np.array(frame_bonds_per_cluster))
        sizes_per_cluster.append(np.array(frame_sizes_per_cluster))
        
        # Separate by particle type
        qt_mask = types == qt_cluster_name
        ft_mask = types == ft_cluster_name
        
        coord_qt = coordination[qt_mask]
        coord_ft = coordination[ft_mask]
        
        coord_dist_qt.append(coord_qt)
        coord_dist_ft.append(coord_ft)
        
        # Statistics
        if len(coord_qt) > 0:
            mean_coord_qt.append(np.mean(coord_qt))
            std_coord_qt.append(np.std(coord_qt))
            max_coord_qt.append(np.max(coord_qt))
        else:
            mean_coord_qt.append(0.0)
            std_coord_qt.append(0.0)
            max_coord_qt.append(0)
        
        if len(coord_ft) > 0:
            mean_coord_ft.append(np.mean(coord_ft))
            std_coord_ft.append(np.std(coord_ft))
            max_coord_ft.append(np.max(coord_ft))
        else:
            mean_coord_ft.append(0.0)
            std_coord_ft.append(0.0)
            max_coord_ft.append(0)
    
    return {
        "times": times,
        "mean_coord_qt": np.array(mean_coord_qt),
        "mean_coord_ft": np.array(mean_coord_ft),
        "std_coord_qt": np.array(std_coord_qt),
        "std_coord_ft": np.array(std_coord_ft),
        "max_coord_qt": np.array(max_coord_qt),
        "max_coord_ft": np.array(max_coord_ft),
        "coord_dist_qt": coord_dist_qt,
        "coord_dist_ft": coord_dist_ft,
        "bonds_per_cluster": bonds_per_cluster,
        "sizes_per_cluster": sizes_per_cluster,
    }



def get_cluster_composition(
    h5_file: str,
    config: SimulationConfig,
    stride: int = 1,
    min_cluster_size: int = 2,
    frame_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Analyze the composition (QtC vs FtC) of each cluster.
    
    Parameters
    ----------
    h5_file : str
        Path to trajectory HDF5 file
    config : SimulationConfig
        Simulation configuration
    stride : int
        Analyze every Nth frame
    min_cluster_size : int
        Minimum cluster size to include (default: 2)
    frame_data : dict, optional
        Pre-extracted frame data from _extract_frame_data(). If provided,
        h5_file and stride are ignored. This allows sharing data between
        multiple analysis functions for better performance.
    
    Returns
    -------
    dict with keys:
        times : ndarray - simulation step numbers (NOT ns; convert via _steps_to_us)
        qt_per_cluster : list of list - QtC count per cluster per frame
        ft_per_cluster : list of list - FtC count per cluster per frame
        size_per_cluster : list of list - total size per cluster per frame
        qt_fraction_per_cluster : list of list - QtC/(QtC+FtC) per cluster per frame
        mean_qt_fraction : ndarray - mean Qt fraction across clusters per frame
        std_qt_fraction : ndarray - std of Qt fraction per frame
    """
    if frame_data is None:
        logger.info("  Extracting frame data for composition analysis...")
        frame_data = _extract_frame_data(h5_file, config, stride=stride)
    
    times = frame_data["times"]
    
    # Particle type names for identification
    qt_cluster_name = config.qt.cluster_name
    ft_cluster_name = config.ft.cluster_name
    
    qt_per_cluster = []
    ft_per_cluster = []
    size_per_cluster = []
    qt_fraction_per_cluster = []
    mean_qt_fraction = []
    std_qt_fraction = []
    
    frame_iter = tqdm(range(frame_data["n_frames"]), desc="  Composition", 
                     disable=not TQDM_AVAILABLE, unit="frame")
    for frame_idx in frame_iter:
        types = frame_data["types"][frame_idx]
        topo_particles = frame_data["topology_particles"][frame_idx]
        
        frame_qt = []
        frame_ft = []
        frame_sizes = []
        frame_fractions = []
        
        for particle_indices in topo_particles:
            # Fix Issue #3: ensure particle_indices is array for proper indexing
            particle_indices = np.asarray(particle_indices)
            n_particles = len(particle_indices)
            
            if n_particles < min_cluster_size:
                continue
            
            # Count QtC and FtC in this cluster
            cluster_types = types[particle_indices]
            n_qt = np.sum(cluster_types == qt_cluster_name)
            n_ft = np.sum(cluster_types == ft_cluster_name)
            
            frame_qt.append(n_qt)
            frame_ft.append(n_ft)
            frame_sizes.append(n_particles)
            
            # Qt fraction (handle edge case of empty cluster)
            total = n_qt + n_ft
            qt_frac = n_qt / total if total > 0 else 0.0
            frame_fractions.append(qt_frac)
        
        qt_per_cluster.append(frame_qt)
        ft_per_cluster.append(frame_ft)
        size_per_cluster.append(frame_sizes)
        qt_fraction_per_cluster.append(frame_fractions)
        
        if len(frame_fractions) > 0:
            mean_qt_fraction.append(np.mean(frame_fractions))
            std_qt_fraction.append(np.std(frame_fractions))
        else:
            mean_qt_fraction.append(0.0)
            std_qt_fraction.append(0.0)
    
    return {
        "times": times,
        "qt_per_cluster": qt_per_cluster,
        "ft_per_cluster": ft_per_cluster,
        "size_per_cluster": size_per_cluster,
        "qt_fraction_per_cluster": qt_fraction_per_cluster,
        "mean_qt_fraction": np.array(mean_qt_fraction),
        "std_qt_fraction": np.array(std_qt_fraction),
        "min_cluster_size": min_cluster_size,
    }



def compute_structural_analysis(
    h5_file: str,
    config: SimulationConfig,
    stride: int = 1,
    min_cluster_size_morphology: int = 3,
    min_cluster_size_spatial: int = 2,
) -> Dict[str, Any]:
    """
    Compute all structural cluster analysis data without plotting.
    
    This function performs the computationally expensive analysis and returns
    the data, which can then be plotted with plot_structural_analysis() or
    saved for later use.
    
    Parameters
    ----------
    h5_file : str
        Path to trajectory HDF5 file
    config : SimulationConfig
        Simulation configuration
    stride : int
        Analyze every Nth frame for detailed analyses (morphology, spatial, contacts).
        Note: Kinetics always uses all frames as it relies on pre-computed observables.
    min_cluster_size_morphology : int
        Minimum cluster size for Rg calculation
    min_cluster_size_spatial : int
        Minimum cluster size for spatial analysis
    
    Returns
    -------
    dict with keys:
        morphology : dict - Radius of gyration and compactness data
        kinetics : dict - Binding rates and particle counts
        spatial : dict - Nearest neighbor distances and cluster positions
        contacts : dict - Coordination numbers and bond counts per cluster
        config : SimulationConfig - The configuration used (for plotting)
    """
    logger.info("\n" + "=" * 60)
    logger.info("COMPUTING STRUCTURAL CLUSTER ANALYSIS")
    logger.info("=" * 60)
    
    # Extract frame data ONCE and share between analysis functions
    logger.info("\n[1/5] Extracting frame data...")
    frame_data = _extract_frame_data(h5_file, config, stride=stride)
    logger.info(f"       Extracted {frame_data['n_frames']} frames")
    
    # Run all analyses using shared frame_data
    logger.info("\n[2/5] Morphology analysis...")
    morphology = get_cluster_morphology(h5_file, config, stride=stride, 
                                         min_cluster_size=min_cluster_size_morphology,
                                         frame_data=frame_data)
    
    logger.info("\n[3/5] Binding kinetics analysis...")
    kinetics = get_binding_kinetics(h5_file, config)
    
    logger.info("\n[4/5] Spatial distribution analysis...")
    spatial = get_spatial_distribution(h5_file, config, stride=stride,
                                        min_cluster_size=min_cluster_size_spatial,
                                        frame_data=frame_data)
    
    logger.info("\n[5/5] Contact analysis...")
    contacts = get_contact_analysis(h5_file, config, stride=stride,
                                    frame_data=frame_data)
    
    logger.info("\n✓ Structural analysis complete")
    
    return {
        "morphology": morphology,
        "kinetics": kinetics,
        "spatial": spatial,
        "contacts": contacts,
        "config": config,
    }



def print_analysis_summary(h5_file: str, config: Optional[SimulationConfig] = None):
    """
    Print a summary of simulation results.
    
    Parameters
    ----------
    h5_file : str
        Path to trajectory HDF5 file
    config : SimulationConfig, optional
        Simulation configuration. Required for correct time display.
        If not provided, times will be shown as step numbers.
    """
    stats = get_cluster_statistics(h5_file)
    bonds = get_bond_counts(h5_file, trajectory=stats.get("trajectory"))
    
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS SUMMARY")
    print("=" * 60)
    
    # Convert times - need timestep from config for proper conversion
    if config is not None:
        times_us = _steps_to_us(stats["times"], config.timestep)
        time_unit = "µs"
    else:
        times_us = stats["times"]  # Just use step numbers
        time_unit = "steps"
    
    print(f"\nInitial state (t=0):")
    print(f"  Topologies: {stats['n_clusters'][0]}")
    print(f"  Average size: {stats['avg_sizes'][0]:.2f} particles")
    print(f"  Largest: {stats['max_sizes'][0]} particles")
    print(f"  Bonds: {bonds['n_bonds'][0]}")
    
    if config is not None:
        final_time_str = format_duration(times_us[-1] * 1e3)
    else:
        final_time_str = f"{times_us[-1]:.0f} {time_unit}"
    print(f"\nFinal state (t={final_time_str}):")
    print(f"  Topologies: {stats['n_clusters'][-1]}")
    print(f"  Average size: {stats['avg_sizes'][-1]:.2f} particles")
    print(f"  Largest: {stats['max_sizes'][-1]} particles")
    print(f"  Bonds: {bonds['n_bonds'][-1]}")
    
    final_sizes = stats["cluster_sizes"][-1]
    if len(final_sizes) > 0:
        print(f"\nFinal size distribution:")
        print(f"  Median: {np.median(final_sizes):.1f}")
        print(f"  Mean: {np.mean(final_sizes):.1f}")
        print(f"  Std: {np.std(final_sizes):.1f}")
        print(f"  Range: {np.min(final_sizes)} - {np.max(final_sizes)}")
        
        # Adaptive size categories based on total particles
        total = np.sum(final_sizes)
        categories = _get_size_categories(final_sizes, total, config)
        
        print(f"\nParticle distribution:")
        for name, mask in categories:
            n_particles = np.sum(final_sizes[mask])
            pct = 100 * n_particles / total if total > 0 else 0
            print(f"  {name}: {n_particles} particles ({pct:.1f}%)")
    
    print("=" * 60 + "\n")


def get_min_pair_distances(
    h5_file: str,
    config: SimulationConfig,
    frame: int = -1,
    trajectory: Optional[readdy.Trajectory] = None,
) -> Dict[str, Any]:
    """
    Minimum center-to-center distance per species pair in a single frame (default: last).

    A quick overlap diagnostic: for each of Qt-Qt, Qt-Ft, Ft-Ft it reports the closest
    center-to-center distance found (minimum-image convention for the periodic box) and the
    contact distance (sum of radii). ``min << contact`` means particles are interpenetrating;
    with the soft/weak potentials a small sub-contact overlap (~sqrt(2*kB*T/k)) is expected.
    Free and clustered particles are grouped together (Qt = {Qt, QtC}, Ft = {Ft, FtC}).

    Parameters
    ----------
    h5_file : str
        Path to trajectory HDF5 file.
    config : SimulationConfig
        Configuration (particle radii/names and box size).
    frame : int
        Frame index to analyze (default -1 = last recorded frame).
    trajectory : readdy.Trajectory, optional
        Unused placeholder for API symmetry with other analysis functions.

    Returns
    -------
    dict with keys:
        step : int               -- simulation step of the analyzed frame
        time_us : float          -- that step converted to microseconds
        pairs : dict             -- {"Qt-Qt"/"Qt-Ft"/"Ft-Ft": {"min","contact",
                                     "overlap_nm","overlap_frac","n_pairs"}}
    """
    data = _extract_frame_data(h5_file, config, verbose=False)
    if data["n_frames"] == 0:
        raise ValueError(f"No frames found in {h5_file}")
    pos = np.asarray(data["positions"][frame])
    types = list(data["types"][frame])
    box = np.asarray(config.box_size, dtype=float)

    qt_names = {config.qt.name, config.qt.cluster_name}
    ft_names = {config.ft.name, config.ft.cluster_name}
    qt_idx = [i for i, t in enumerate(types) if t in qt_names]
    ft_idx = [i for i, t in enumerate(types) if t in ft_names]

    def min_dist(ia, ib, same):
        pairs = itertools.combinations(ia, 2) if same else itertools.product(ia, ib)
        best = np.inf
        n = 0
        for i, j in pairs:
            if i == j:
                continue
            d = pos[i] - pos[j]
            d -= box * np.round(d / box)          # minimum-image
            r = float(np.linalg.norm(d))
            if r < best:
                best = r
            n += 1
        return (best if n else float("nan")), n

    rq, rf = config.qt.radius, config.ft.radius
    specs = [
        ("Qt-Qt", qt_idx, qt_idx, True, 2.0 * rq),
        ("Qt-Ft", qt_idx, ft_idx, False, rq + rf),
        ("Ft-Ft", ft_idx, ft_idx, True, 2.0 * rf),
    ]
    pairs = {}
    for label, ia, ib, same, contact in specs:
        mn, n = min_dist(ia, ib, same)
        overlap = max(contact - mn, 0.0) if mn == mn else float("nan")   # nan-safe
        pairs[label] = {
            "min": mn,
            "contact": contact,
            "overlap_nm": overlap,
            "overlap_frac": (overlap / contact) if (contact > 0 and overlap == overlap) else float("nan"),
            "n_pairs": n,
        }

    step = int(data["times"][frame]) if len(data["times"]) else 0
    return {"step": step, "time_us": float(_steps_to_us(step, config.timestep)), "pairs": pairs}


def print_min_distance_table(
    h5_file: str,
    config: SimulationConfig,
    frame: int = -1,
    trajectory: Optional[readdy.Trajectory] = None,
) -> Dict[str, Any]:
    """Print the minimum center-to-center distance / overlap table (see get_min_pair_distances)."""
    res = get_min_pair_distances(h5_file, config, frame=frame, trajectory=trajectory)
    print("=" * 60)
    print(f"MINIMUM CENTER-TO-CENTER DISTANCES (frame at t={format_duration(res['time_us'] * 1e3)})")
    print("=" * 60)
    print(f"  {'Pair':<7}{'min(nm)':>9}{'contact':>9}{'overlap (nm / %)':>20}")
    for label, p in res["pairs"].items():
        if p["min"] != p["min"]:   # nan -> too few particles
            print(f"  {label:<7}{'n/a':>9}{p['contact']:>9.1f}{'(too few particles)':>20}")
            continue
        ov = f"{p['overlap_nm']:.1f} ({100 * p['overlap_frac']:.0f}%)"
        print(f"  {label:<7}{p['min']:>9.1f}{p['contact']:>9.1f}{ov:>20}")
    print("  (min < contact => interpenetration; soft/weak potentials allow small sub-contact overlap)")
    print("=" * 60 + "\n")
    return res


def _get_size_category_boundaries(
    total_particles: int,
    config: Optional[SimulationConfig] = None,
) -> List[Tuple[str, int, Optional[int]]]:
    """
    Generate adaptive size category boundaries based on particle count.
    
    Returns structured boundary definitions that can be used both for
    masking arrays and for time-series plotting, without needing to
    reverse-engineer boundaries from label strings.
    
    Parameters
    ----------
    total_particles : int
        Total number of particles
    config : SimulationConfig, optional
        Configuration for getting particle counts (overrides total_particles)
    
    Returns
    -------
    list of (name, min_size, max_size) tuples
        max_size is None for the last (unbounded) category.
    """
    # Use config particle count if available, otherwise use provided total
    if config is not None:
        n_total = config.n_qt + config.n_ft
    else:
        n_total = total_particles
    
    # Define boundaries as percentages of total
    small_max = max(5, int(0.02 * n_total))      # 2% or at least 5
    medium_max = max(20, int(0.10 * n_total))    # 10% or at least 20
    large_max = max(50, int(0.25 * n_total))     # 25% or at least 50
    
    boundaries = [
        ("Monomers (1)", 1, 1),
        (f"Small (2-{small_max})", 2, small_max),
        (f"Medium ({small_max+1}-{medium_max})", small_max + 1, medium_max),
        (f"Large ({medium_max+1}-{large_max})", medium_max + 1, large_max),
        (f"Very large (>{large_max})", large_max + 1, None),
    ]
    
    return boundaries


def _apply_size_category(
    sizes: np.ndarray,
    min_size: int,
    max_size: Optional[int],
) -> np.ndarray:
    """
    Apply a size category boundary to an array of cluster sizes.
    
    Parameters
    ----------
    sizes : ndarray
        Array of cluster sizes
    min_size : int
        Minimum size (inclusive)
    max_size : int or None
        Maximum size (inclusive). None means unbounded (all sizes >= min_size).
    
    Returns
    -------
    ndarray (bool)
        Boolean mask for sizes matching this category
    """
    if max_size is None:
        return sizes >= min_size
    else:
        return (sizes >= min_size) & (sizes <= max_size)


def _get_size_categories(
    sizes: np.ndarray,
    total_particles: int,
    config: Optional[SimulationConfig] = None,
) -> List[Tuple[str, np.ndarray]]:
    """
    Generate adaptive size categories with boolean masks.
    
    Convenience wrapper around _get_size_category_boundaries() that
    applies the boundaries to a concrete array of sizes.
    
    Parameters
    ----------
    sizes : ndarray
        Array of cluster sizes
    total_particles : int
        Total number of particles
    config : SimulationConfig, optional
        Configuration for getting particle counts
    
    Returns
    -------
    list of (name, mask) tuples
    """
    boundaries = _get_size_category_boundaries(total_particles, config)
    categories = []
    for name, min_size, max_size in boundaries:
        mask = _apply_size_category(sizes, min_size, max_size)
        categories.append((name, mask))
    return categories


def _size_category_key(name: str) -> str:
    """Sanitize a size-category display name into an NPZ/stat-dict-safe key fragment.

    The producer (ensemble structural statistics) and the consumers (plotting) MUST use
    this so the ``size_frac_{key}_mean`` / ``_std`` keys always agree. For example
    ``"Very large (>50)" -> "Very_large_gt50"`` and ``"Large (21-50)" -> "Large_21_50"``.
    """
    return (name.replace(' ', '_').replace('(', '').replace(')', '')
            .replace('>', 'gt').replace('-', '_'))


def get_size_fractions(
    h5_file: str,
    config: Optional[SimulationConfig] = None,
    trajectory: Optional = None,
) -> Dict[str, Any]:
    """
    Compute fraction of particles in each size category over time.
    
    Uses adaptive size categories based on total particle count.
    This is the data behind the "Particles by Size Category" stacked area chart.
    
    Parameters
    ----------
    h5_file : str
        Path to trajectory HDF5 file
    config : SimulationConfig, optional
        Configuration for adaptive category boundaries and timestep
    trajectory : readdy.Trajectory, optional
        Pre-loaded trajectory object
    
    Returns
    -------
    dict with keys:
        times : ndarray - step numbers
        category_names : list of str - category display names
        category_fractions : dict of str -> ndarray - fraction time series per category
        boundaries : list of (name, min_size, max_size) tuples
    """
    stats = get_cluster_statistics(h5_file, trajectory=trajectory)
    cluster_sizes = stats["cluster_sizes"]
    
    # Get total particle count
    total_particles = (
        np.sum(cluster_sizes[0]) 
        if len(cluster_sizes) > 0 and len(cluster_sizes[0]) > 0 
        else 400
    )
    
    # Get structured boundaries
    boundaries = _get_size_category_boundaries(total_particles, config)
    
    # Compute fractions at each timestep
    category_fractions = {name: [] for name, _, _ in boundaries}
    
    for sizes in cluster_sizes:
        total = np.sum(sizes) if len(sizes) > 0 else 1
        for name, min_size, max_size in boundaries:
            if len(sizes) > 0:
                mask = _apply_size_category(sizes, min_size, max_size)
                frac = np.sum(sizes[mask]) / total if total > 0 else 0
            else:
                frac = 0
            category_fractions[name].append(frac)
    
    # Convert to arrays
    for name in category_fractions:
        category_fractions[name] = np.array(category_fractions[name])
    
    return {
        "times": stats["times"],
        "category_names": [name for name, _, _ in boundaries],
        "category_fractions": category_fractions,
        "boundaries": boundaries,
    }


# =============================================================================
# PHASED-TRAJECTORY COMBINING
# =============================================================================

def combine_phase_trajectories(
    phase_files: List[str],
    out_file: str,
    step_offsets: Optional[List[int]] = None,
    drop_duplicate_boundary: bool = True,
) -> str:
    """Stitch per-phase ReaDDy trajectories into one continuous trajectory file.

    A phased run (engine.run_phased) writes one ``trajectory.h5`` per phase, each with
    step numbers (``time``) restarting at 0. This concatenates them into a single
    ReaDDy-readable ``out_file`` spanning the whole cycle on a continuous step axis, so
    it can be opened with ``readdy.Trajectory``, re-analysed by the get_* functions, and
    exported to one ``.xyz`` (convert_h5_to_xyz). The per-phase files are left untouched.

    How it works (no ReaDDy write API exists, so this is direct HDF5 surgery):

    - ``readdy/config`` is copied verbatim from the first phase (particle/topology type
      metadata; per-phase reaction lists differ but are not needed to *read* frames).
    - For each phase i, every per-frame ``time`` is shifted by ``step_offsets[i]`` (the
      cumulative steps of prior phases). The first frame of each non-first phase is the
      checkpoint-restored copy of the previous phase's last frame, so it is dropped when
      ``drop_duplicate_boundary`` (default) to keep the axis strictly increasing.
    - ``readdy/trajectory`` (records/limits/time) and ``readdy/observables/topologies``
      (edges/particles/limits*/types/time) are concatenated by keeping each frame's data
      slices intact and rebuilding the per-frame ``limits`` as running offsets. Edge and
      particle indices are frame-local, so slice contents need no rewriting.
    - All other ``readdy/observables/<obs>`` with a ``time`` dataset are concatenated
      along the frame axis (per-frame datasets shifted/masked; non-per-frame datasets such
      as rdf ``bin_centers`` copied once from phase 0).
    - ``reaction_counts`` is skipped: its sub-structure differs between binding phases
      (spatialCounts) and breaking phases (structuralCounts). The per-phase files keep it.
      Any observable whose dtype h5py cannot write (notably the optional ``particles``
      observable, a vlen-of-array dataset) is also skipped with a message — it is
      redundant here since per-frame positions live in ``readdy/trajectory`` and analysis
      falls back to ``trajectory.read()``. The per-phase files retain these.

    Output datasets are written with gzip (built-in HDF5 filter) instead of ReaDDy's
    blosc, so the combined file is readable without the blosc plugin.

    Parameters
    ----------
    phase_files : list of str
        Per-phase trajectory.h5 paths, in order.
    out_file : str
        Path of the combined trajectory to write (overwritten if present).
    step_offsets : list of int, optional
        Cumulative step count before each phase. If None, derived by chaining each
        phase's own final ``readdy/trajectory/time`` value.
    drop_duplicate_boundary : bool
        Drop each non-first phase's first frame (the duplicated boundary state).

    Returns
    -------
    str
        ``out_file``.
    """
    import h5py  # readdy is imported at module top, which registers the blosc filter

    if not phase_files:
        raise ValueError("combine_phase_trajectories needs at least one phase file")
    for p in phase_files:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Phase trajectory not found: {p}")

    handles = [h5py.File(p, "r") for p in phase_files]
    try:
        # Resolve per-phase step offsets if not provided (chain each phase's final step).
        if step_offsets is None:
            step_offsets, cum = [], 0
            for h in handles:
                step_offsets.append(cum)
                cum += int(h["readdy/trajectory/time"][-1])
        offsets = [int(o) for o in step_offsets]

        out_dir = os.path.dirname(out_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        def _write(group, name, array, like=None):
            """Create a dataset (gzip for plain numeric arrays).

            Returns True on success, or False if h5py cannot write the value (e.g. a
            vlen-of-array dataset such as the `particles`/`forces` observables), after
            removing any partially-created dataset so the caller can cleanly skip it.
            """
            arr = np.asarray(array) if not isinstance(array, np.ndarray) else array
            dtype = like.dtype if like is not None else arr.dtype
            comp = {}
            if dtype != object and arr.dtype != object and arr.size:
                comp = {"compression": "gzip"}
            try:
                group.create_dataset(name, data=arr, dtype=dtype, **comp)
                return True
            except Exception:
                if name in group:
                    del group[name]
                return False

        def _write_required(group, name, array, like=None):
            if not _write(group, name, array, like=like):
                raise RuntimeError(
                    f"combine_phase_trajectories: cannot write required dataset "
                    f"{group.name}/{name}"
                )

        def _first_kept(time, offset, last_time):
            """Index of the first frame to keep for a phase (drop duplicated boundary)."""
            if last_time is None or not drop_duplicate_boundary or len(time) == 0:
                return 0
            return 1 if int(time[0]) + offset == last_time else 0

        with h5py.File(out_file, "w") as fout:
            readdy_out = fout.create_group("readdy")
            handles[0].copy("readdy/config", readdy_out)

            # ---- readdy/trajectory : records / limits / time ----
            traj_out = readdy_out.create_group("trajectory")
            rec_parts, lim_parts, time_parts = [], [], []
            running = 0
            last_time = None
            for h, off in zip(handles, offsets):
                g = h["readdy/trajectory"]
                time = g["time"][:]
                limits = g["limits"][:].astype(np.int64)
                first = _first_kept(time, off, last_time)
                base = int(limits[first, 0])
                rec_parts.append(g["records"][base:])
                lim_parts.append(limits[first:] - base + running)
                kt = time[first:].astype(np.uint64) + np.uint64(off)
                time_parts.append(kt)
                running += int(g["records"].shape[0]) - base
                if len(kt):
                    last_time = int(kt[-1])
            _write_required(traj_out, "records", np.concatenate(rec_parts), like=handles[0]["readdy/trajectory/records"])
            _write_required(traj_out, "limits", np.concatenate(lim_parts).astype(np.uint64))
            _write_required(traj_out, "time", np.concatenate(time_parts).astype(np.uint64))

            obs_out = readdy_out.create_group("observables")

            # ---- readdy/observables/topologies : edges/particles/limits*/types/time ----
            if "readdy/observables/topologies" in handles[0]:
                topo_out = obs_out.create_group("topologies")
                ed, pa, lE, lP, ty, tt = [], [], [], [], [], []
                runE = runP = 0
                last_time = None
                for h, off in zip(handles, offsets):
                    g = h["readdy/observables/topologies"]
                    time = g["time"][:]
                    limE = g["limitsEdges"][:].astype(np.int64)
                    limP = g["limitsParticles"][:].astype(np.int64)
                    first = _first_kept(time, off, last_time)
                    bE, bP = int(limE[first, 0]), int(limP[first, 0])
                    ed.append(g["edges"][bE:])
                    pa.append(g["particles"][bP:])
                    lE.append(limE[first:] - bE + runE)
                    lP.append(limP[first:] - bP + runP)
                    ty.append(g["types"][first:])
                    kt = time[first:].astype(np.uint64) + np.uint64(off)
                    tt.append(kt)
                    runE += int(g["edges"].shape[0]) - bE
                    runP += int(g["particles"].shape[0]) - bP
                    if len(kt):
                        last_time = int(kt[-1])
                _write_required(topo_out, "edges", np.concatenate(ed), like=handles[0]["readdy/observables/topologies/edges"])
                _write_required(topo_out, "particles", np.concatenate(pa), like=handles[0]["readdy/observables/topologies/particles"])
                _write_required(topo_out, "limitsEdges", np.concatenate(lE).astype(np.uint64))
                _write_required(topo_out, "limitsParticles", np.concatenate(lP).astype(np.uint64))
                _write_required(topo_out, "time", np.concatenate(tt).astype(np.uint64))
                _write_required(topo_out, "types", np.concatenate(ty), like=handles[0]["readdy/observables/topologies/types"])

            # ---- generic observables with a per-frame 'time' (skip reaction_counts) ----
            for name in handles[0]["readdy/observables"]:
                if name in ("topologies", "reaction_counts"):
                    continue
                src0 = handles[0][f"readdy/observables/{name}"]
                if not isinstance(src0, h5py.Group) or "time" not in src0:
                    continue
                g_out = obs_out.create_group(name)
                n_frames0 = src0["time"].shape[0]
                ok = True
                # per-dataset concatenation
                for dname, d0 in src0.items():
                    if not isinstance(d0, h5py.Dataset):
                        continue
                    if d0.shape and d0.shape[0] == n_frames0:
                        # per-frame dataset: concat across phases with boundary dedup
                        parts = []
                        last_time = None
                        for h, off in zip(handles, offsets):
                            g = h[f"readdy/observables/{name}"]
                            time = g["time"][:]
                            first = _first_kept(time, off, last_time)
                            if dname == "time":
                                kt = time[first:].astype(np.uint64) + np.uint64(off)
                                parts.append(kt)
                            else:
                                parts.append(g[dname][first:])
                                kt = time[first:].astype(np.uint64) + np.uint64(off)
                            if len(kt):
                                last_time = int(kt[-1])
                        ok = _write(g_out, dname, np.concatenate(parts), like=d0)
                    else:
                        # non-per-frame (e.g. rdf bin_centers): copy once from phase 0
                        ok = _write(g_out, dname, d0[:], like=d0)
                    if not ok:
                        break
                if not ok:
                    # An observable whose dtype h5py can't write (e.g. the vlen-of-array
                    # `particles` observable). It is redundant for the combined file —
                    # positions live in readdy/trajectory and analysis falls back to
                    # trajectory.read() — so drop the group and continue.
                    if name in obs_out:
                        del obs_out[name]
                    logger.info(f"  (combine: skipped observable '{name}' — dtype not writable by h5py)")

        return out_file
    finally:
        for h in handles:
            h.close()


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def convert_h5_to_xyz(
    h5_file: str,
    xyz_file: str,
    config: SimulationConfig,
    overwrite: bool = True,
) -> str:
    """
    Export trajectory to XYZ format for visualization (e.g., OVITO).
    
    Parameters
    ----------
    h5_file : str
        Input trajectory HDF5 file
    xyz_file : str
        Output XYZ file path
    config : SimulationConfig
        Configuration (for particle radii and names)
    overwrite : bool
        Overwrite existing file
    
    Returns
    -------
    str
        Path to created XYZ file
    
    Notes
    -----
    The output file is an Extended XYZ format compatible with OVITO:
    - Species names (Qt, Ft, QtC, FtC) instead of type_0, type_1, etc.
    - Radius column for each particle
    - Lattice/pbc/Origin information for periodic boxes
    - Particles at origin (0,0,0) are filtered out (these are "ghost" 
      particles for types not yet present in the simulation)
    """
    if not os.path.exists(h5_file):
        raise FileNotFoundError(f"Input file not found: {h5_file}")
    
    if os.path.exists(xyz_file):
        if overwrite:
            os.remove(xyz_file)
        else:
            raise FileExistsError(f"Output file exists: {xyz_file}")
    
    # Load trajectory
    traj = readdy.Trajectory(h5_file)
    
    # Define particle radii mapping
    radii = {
        config.qt.name: config.qt.radius,
        config.ft.name: config.ft.radius,
        config.qt.cluster_name: config.qt.radius,
        config.ft.cluster_name: config.ft.radius,
    }
    
    # First export using ReaDDy's built-in converter to a temporary file.
    # generate_tcl=False: we produce an OVITO-friendly XYZ below, not a VMD script, so the
    # otherwise-default "<temp>.tcl" companion file would just be leftover garbage.
    temp_readdy_file = xyz_file + ".readdy_tmp"
    traj.convert_to_xyz(temp_readdy_file, particle_radii=radii, generate_tcl=False)
    
    # Build extended XYZ header with box information
    Lx, Ly, Lz = config.box_size
    ox, oy, oz = -Lx / 2.0, -Ly / 2.0, -Lz / 2.0
    
    lattice_fragment = (
        f' Lattice="{Lx} 0 0 0 {Ly} 0 0 0 {Lz}"'
        f' pbc="T T T"'
        f' Origin="{ox} {oy} {oz}"'
    )
    
    # Tolerance for detecting particles at origin
    EPS = 1e-12
    
    # Type mapping: map whatever labels ReaDDy emits to (species_name, radius).
    # ReaDDy normally writes the real species names; as a fallback it may write
    # "type_<id>". Build the type_<id> -> name map from the trajectory's actual
    # {name: id} table rather than assuming the order species were added (the old
    # hard-coded type_0->Qt ... was silently wrong if _add_species was reordered).
    type_mapping = {
        config.qt.name: (config.qt.name, config.qt.radius),
        config.ft.name: (config.ft.name, config.ft.radius),
        config.qt.cluster_name: (config.qt.cluster_name, config.qt.radius),
        config.ft.cluster_name: (config.ft.cluster_name, config.ft.radius),
    }
    for name, type_id in traj.particle_types.items():
        if name in radii:
            type_mapping[f"type_{type_id}"] = (name, radii[name])
    
    # Process the ReaDDy file and create OVITO-friendly output
    with open(temp_readdy_file, "r", encoding="utf-8", errors="replace") as f_in, \
         open(xyz_file, "w", encoding="utf-8") as f_out:
        
        while True:
            # Read atom count line
            n_line = f_in.readline()
            if not n_line:
                break
            
            n_str = n_line.strip()
            if not n_str:
                continue
            
            try:
                n = int(n_str)
            except ValueError:
                warnings.warn(f"Skipping invalid atom count line: {n_str}")
                continue
            
            # Skip original comment line
            _ = f_in.readline()
            
            # Read and transform all particle lines for this frame
            transformed_particles = []
            
            for _ in range(n):
                line = f_in.readline()
                if not line:
                    break
                
                parts = line.strip().split()
                if len(parts) < 4:
                    warnings.warn(f"Skipping malformed line: {line.strip()}")
                    continue
                
                label = parts[0]
                x_str, y_str, z_str = parts[1], parts[2], parts[3]
                
                # Map type to species name and radius
                if label in type_mapping:
                    species, radius = type_mapping[label]
                else:
                    warnings.warn(f"Unknown particle type '{label}', using Qt as default")
                    species, radius = config.qt.name, config.qt.radius
                
                # Parse coordinates
                try:
                    x, y, z = float(x_str), float(y_str), float(z_str)
                except ValueError:
                    # Keep line if parse fails (safer than dropping data)
                    transformed_particles.append((species, x_str, y_str, z_str, radius))
                    continue
                
                # Filter out particles at exact origin (ghost particles)
                if abs(x) <= EPS and abs(y) <= EPS and abs(z) <= EPS:
                    continue
                
                transformed_particles.append((species, x_str, y_str, z_str, radius))
            
            # Write frame with adjusted particle count
            f_out.write(f"{len(transformed_particles)}\n")
            
            # Write extended XYZ header
            header = f"Properties=species:S:1:pos:R:3:radius:R:1{lattice_fragment}\n"
            f_out.write(header)
            
            # Write transformed particle lines
            for species, x_str, y_str, z_str, radius in transformed_particles:
                f_out.write(f"{species}\t{x_str}\t{y_str}\t{z_str}\t{radius}\n")
    
    # Clean up temporary files (the temp XYZ and, defensively, any ReaDDy VMD .tcl script).
    for tmp in (temp_readdy_file, temp_readdy_file + ".tcl"):
        try:
            os.remove(tmp)
        except OSError:
            pass
    
    logger.info(f"✓ Exported OVITO-friendly XYZ to {xyz_file}")
    
    return xyz_file


# =============================================================================
# ENSEMBLE DATA LOADING FUNCTIONS
# =============================================================================

def _load_ensemble_files(results_dir: str):
    """Read the three on-disk ensemble files (shared by both public loaders).

    Returns ``(stats, npz, config, meta)`` where ``stats`` is the statistics JSON with
    list values converted to ``np.ndarray``, ``npz`` is a plain dict of the structural
    arrays (``{}`` if absent), ``config`` is the config JSON (``{}`` if absent), and
    ``meta`` carries the paths / existence flags so callers keep their own error/print
    behaviour. Raises ``FileNotFoundError`` only if the statistics file is missing.
    """
    results_dir = results_dir.rstrip("/") + "/"
    stats_path = f"{results_dir}ensemble_statistics.json"
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Statistics file not found: {stats_path}")
    with open(stats_path, 'r') as f:
        raw = json.load(f)
    stats = {k: (np.array(v) if isinstance(v, list) else v) for k, v in raw.items()}

    npz_path = f"{results_dir}ensemble_structural.npz"
    npz = {}
    if os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=True) as data:
            npz = {k: data[k] for k in data.files}

    config_path = f"{results_dir}ensemble_config.json"
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)

    meta = {
        'stats_path': stats_path, 'npz_path': npz_path, 'config_path': config_path,
        'has_npz': os.path.exists(npz_path), 'has_config': os.path.exists(config_path),
    }
    return stats, npz, config, meta


def load_ensemble_data(results_dir: str) -> Tuple[Dict, Dict, Dict]:
    """
    Load ensemble analysis data from JSON and NPZ files.
    
    This function loads pre-computed ensemble results that were either:
    - Saved by EnsembleSimulation.save() locally
    - Generated by analyze_ensemble.py on a cluster
    
    Parameters
    ----------
    results_dir : str
        Path to ensemble results directory containing:
        - ensemble_statistics.json (basic statistics)
        - ensemble_structural.npz (structural analysis arrays)
        - ensemble_config.json (base simulation configuration)
    
    Returns
    -------
    stats : dict
        Basic statistics (times, mean/std for observables)
    structural : dict
        Advanced analysis data (morphology, spatial, contacts, composition)
    config : dict
        Base simulation configuration
    
    Example
    -------
    >>> stats, structural, config = load_ensemble_data("ensemble_results/")
    >>> plot_ensemble_observables(stats, structural, config)
    """
    stats, npz, config, meta = _load_ensemble_files(results_dir)
    logger.info(f"✓ Loaded statistics from {meta['stats_path']}")

    # All structural arrays (including per-replica *_all) live in `structural`.
    structural = npz
    if meta['has_npz']:
        logger.info(f"✓ Loaded structural data from {meta['npz_path']}")
    else:
        logger.info(f"  Note: No structural data file found at {meta['npz_path']}")

    if meta['has_config']:
        logger.info(f"✓ Loaded configuration from {meta['config_path']}")
    else:
        logger.info(f"  Note: No config file found at {meta['config_path']}")

    return stats, structural, config



def _fmt_mean_std(mean: float, std: float, decimals: int = 1) -> str:
    """Format a mean ± SD pair as a string, e.g. ``596.6 ± 1.3``."""
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def _final_state_rows(stats: Dict, config: Optional[Dict] = None,
                      structural: Optional[Dict] = None) -> List[Tuple[str, str, str]]:
    """
    Build the (Metric, Value, Unit) rows for a single ensemble's final state.

    Aggregation metrics come from the final value of the ensemble time-series arrays
    (``{key}_mean[-1]`` ± ``{key}_std[-1]``); kinetics come from the pre-computed
    ``stats['summary']`` dict; morphology (radius of gyration) and composition (Qt fraction)
    come from the final value of the ``structural`` time-series arrays. Missing metrics are
    silently skipped.

    Returns a list of ``(metric, value_string, unit)`` tuples (shared by the
    single-ensemble and comparison tables).
    """
    rows: List[Tuple[str, str, str]] = []

    def final_pair(key, source=stats):
        if source is None:
            return None
        mean = source.get(f"{key}_mean")
        std = source.get(f"{key}_std")
        if mean is None or std is None or len(mean) == 0:
            return None
        return float(np.asarray(mean)[-1]), float(np.asarray(std)[-1])

    # --- Aggregation / final state ---
    bonds = final_pair("bonds")
    if bonds:
        rows.append(("Number of bonds", _fmt_mean_std(*bonds, 1), "—"))

    clusters = final_pair("n_clusters")
    if clusters:
        rows.append(("Individual topologies", _fmt_mean_std(*clusters, 1), "—"))

    avg = final_pair("avg_cluster")
    if avg:
        rows.append(("Average cluster size", _fmt_mean_std(*avg, 1), "particles"))

    largest = final_pair("largest_cluster")
    if largest:
        rows.append(("Largest cluster size", _fmt_mean_std(*largest, 1), "particles"))
        total = None
        if config is not None:
            total = (config.get("n_qt") or 0) + (config.get("n_ft") or 0)
        if total:
            frac_mean = largest[0] / total * 100.0
            frac_std = largest[1] / total * 100.0
            rows.append(("Largest cluster fraction",
                         _fmt_mean_std(frac_mean, frac_std, 1), "% of particles"))

    fbound = final_pair("fraction_bound")
    if fbound:
        rows.append(("Fraction bound", _fmt_mean_std(*fbound, 3), "—"))

    # --- Kinetics (from the summary dict) ---
    m = stats.get("summary", {}) or {}
    if "half_time_mean" in m:
        rows.append(("Half-time t₅₀",
                     _fmt_mean_std(m["half_time_mean"] * NS_TO_US,
                                   m.get("half_time_std", 0.0) * NS_TO_US, 2), "µs"))

    # --- Morphology & composition (from the structural dict) ---
    rg = final_pair("mean_rg", structural)
    if rg:
        rows.append(("Radius of gyration", _fmt_mean_std(*rg, 1), "nm"))

    comp = final_pair("mean_composition", structural)
    if comp:
        rows.append(("Qt fraction (QtC/(QtC+FtC))", _fmt_mean_std(*comp, 3), "—"))

    return rows


def build_final_state_table(stats: Dict, config: Optional[Dict] = None,
                            structural: Optional[Dict] = None):
    """
    Build a final-state summary table for a single ensemble.

    Parameters
    ----------
    stats : dict
        Ensemble statistics (from JSON or EnsembleSimulation), including the per-replica
        ``{key}_mean``/``{key}_std`` time series and a ``summary`` sub-dict.
    config : dict, optional
        Configuration dictionary; used for the largest-cluster fraction
        (``n_qt`` + ``n_ft``).
    structural : dict, optional
        Structural statistics (from ``ensemble_structural.npz`` / ``load_ensemble_data``);
        supplies the radius-of-gyration and Qt-fraction composition rows. Omit to skip them.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``Metric`` with columns ``["Value", "Unit"]``, where ``Value`` is a
        formatted ``"mean ± SD"`` string. Rows: final-state aggregation metrics followed by
        kinetics, morphology and composition metrics.
    """
    import pandas as pd

    rows = _final_state_rows(stats, config, structural)
    df = pd.DataFrame(rows, columns=["Metric", "Value", "Unit"]).set_index("Metric")
    return df


def save_table_files(df, path_base: str, *, caption: Optional[str] = None,
                     label: Optional[str] = None):
    """
    Save a results table as ``{path_base}.csv`` and ``{path_base}.tex``.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to save (e.g. from :func:`build_final_state_table` or
        ``qtft.comparison.build_comparison_table``).
    path_base : str
        Output path without extension.
    caption, label : str, optional
        LaTeX caption/label passed to ``DataFrame.to_latex`` (for thesis tables).
    """
    csv_path = f"{path_base}.csv"
    tex_path = f"{path_base}.tex"

    df.to_csv(csv_path)
    logger.info(f"✓ Saved table to {csv_path}")

    # escape=True escapes LaTeX specials (notably % -> \%, which would otherwise comment out
    # the line) while leaving the unicode ±, µ and subscripts (t₅₀) untouched — those aren't in
    # pandas' escape set and render directly under a unicode-aware engine (XeLaTeX/LuaLaTeX or
    # utf8 inputenc).
    df.to_latex(tex_path, caption=caption, label=label, escape=True)
    logger.info(f"✓ Saved table to {tex_path}")


def print_ensemble_summary(stats: Dict, config: Optional[Dict] = None):
    """
    Print a summary of ensemble statistics.
    
    Parameters
    ----------
    stats : dict
        Statistics dictionary (from JSON or EnsembleSimulation)
    config : dict, optional
        Configuration dictionary for additional context
    """
    print("\n" + "=" * 60)
    print(f"ENSEMBLE SUMMARY (N={stats.get('n_replicas', '?')} replicas)")
    print("=" * 60)
    
    # Configuration info
    if config:
        n_qt = config.get('n_qt', '?')
        n_ft = config.get('n_ft', '?')
        print(f"\nSystem: {n_qt} Qt + {n_ft} Ft particles")
    
    # Summary metrics
    if 'summary' in stats:
        m = stats['summary']
        print(f"\nFinal state metrics:")
        if 'final_bonds_mean' in m:
            print(f"  Bonds: {m['final_bonds_mean']:.1f} ± {m.get('final_bonds_std', 0):.1f}")
        if 'final_largest_fraction_mean' in m:
            pct = m['final_largest_fraction_mean'] * 100
            pct_std = m.get('final_largest_fraction_std', 0) * 100
            print(f"  Largest cluster: {pct:.1f}% ± {pct_std:.1f}% of particles")
        if 'half_time_mean' in m:
            # half_time is stored in nanoseconds, convert to microseconds
            ht_us = m['half_time_mean'] * NS_TO_US
            ht_std_us = m.get('half_time_std', 0) * NS_TO_US
            print(f"  Half-time: {ht_us:.2f} ± {ht_std_us:.2f} µs")
    
    print("=" * 60 + "\n")
