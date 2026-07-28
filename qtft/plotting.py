"""
Qt-Ft Agglomeration Plotting Module

This module provides all plotting and visualization functions for ReaDDy2-based
Qt-Ft nanoparticle agglomeration simulations.

Contents:
    - Single-run observable plots (energy, bonds, pressure, particles, RDF)
    - Cluster analysis plots (size distributions, heatmaps)
    - Structural analysis plots (morphology, composition, spatial, contacts)
    - Ensemble plots (mean ± std bands, final distributions)

Related modules:
    - qtft.config / qtft.system / qtft.engine: configuration and simulation execution
    - qtft.analysis: core analysis functions (no matplotlib dependency)
    - qtft.ensemble: EnsembleSimulation class for multi-replica runs

Usage:
    import qtft
    import qtft.analysis as analysis
    import qtft.plotting as plotting

    # Single-run plots
    plotting.plot_observables(h5_file, config)
    plotting.plot_cluster_analysis(h5_file, config=config)

    # Ensemble plots (after loading data)
    stats, structural, config = analysis.load_ensemble_data(ensemble_dir)
    plotting.plot_ensemble_observables(stats, structural, config)
"""

from __future__ import annotations

import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import readdy
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# Import from simulation module
from .config import SimulationConfig, NS_TO_US, _steps_to_us, choose_time_unit


def _time_axis(times_us):
    """Scale a µs time array to an adaptive unit for plotting.

    Returns ``(scaled_array, axis_label)`` where the unit (µs/ms/s/…) is chosen from the
    array's maximum, so axis numbers stay readable. Panels built from the same trajectory
    share a unit automatically (same max => same choice).
    """
    arr = np.asarray(times_us, dtype=float)
    max_us = float(arr.max()) if arr.size else 0.0
    factor, unit = choose_time_unit(max_us)
    return arr * factor, f"Time ({unit})"

# Import analysis functions needed by plotting
from .analysis import (
    get_cluster_statistics,
    get_bond_counts,
    get_cluster_composition,
    get_size_fractions,
    compute_structural_analysis,
    load_phased_observables,
    _extract_frame_data,
    _get_size_categories,
    _get_size_category_boundaries,
    _apply_size_category,
    _size_category_key,
)


# =============================================================================
# CONSTANTS (Plotting)
# =============================================================================

# Plotting fontsize configuration (consistent across all plots)
FONTSIZE_TITLE = 14
FONTSIZE_LABEL = 12
FONTSIZE_LEGEND = 10
FONTSIZE_TICK = 10

# Species colours for the COORDINATION plots (Qt/QtC vs Ft/FtC). Scoped to those plots
# only — particle counts, composition and Rg keep their own blue/red scheme.
COORD_COLOR_QT = 'tab:green'
COORD_COLOR_FT = 'tab:red'


def _plot_coord_distribution(
    ax,
    coord_qt,
    coord_ft,
    *,
    weight: Optional[float] = None,
    ylabel: str = "Count",
    title: str = "Coordination Distribution (Final)",
):
    """Histogram of per-particle coordination number for QtC and FtC on one axes.

    Shared by the single-run structural figure and the ensemble panel. Both species use
    the same integer bin edges (centred on whole numbers) so their bars line up.

    Parameters
    ----------
    ax : matplotlib axes
    coord_qt, coord_ft : array-like
        Per-particle coordination numbers of the clustered species (QtC / FtC). Free
        particles are not included by ``analysis.get_contact_analysis``.
    weight : float, optional
        Per-sample weight. ``None`` (default) gives raw counts; pass ``1/n_replicas`` to
        turn counts pooled over replicas into a mean count per replica.
    ylabel, title : str
        Axis label and title (the ensemble panel relabels the y axis).
    """
    coord_qt = np.asarray(coord_qt)
    coord_ft = np.asarray(coord_ft)

    if len(coord_qt) > 0 or len(coord_ft) > 0:
        max_coord = max(
            coord_qt.max() if len(coord_qt) > 0 else 0,
            coord_ft.max() if len(coord_ft) > 0 else 0
        )
        bins = np.arange(-0.5, max_coord + 1.5, 1)

        for values, color, label in ((coord_qt, COORD_COLOR_QT, 'QtC'),
                                     (coord_ft, COORD_COLOR_FT, 'FtC')):
            if len(values) == 0:
                continue
            w = np.full(len(values), weight) if weight is not None else None
            ax.hist(values, bins=bins, alpha=0.6, color=color, weights=w,
                    label=f'{label} (mean={np.mean(values):.2f})', edgecolor='black')
        ax.legend(loc='best', fontsize=FONTSIZE_LEGEND)
    ax.set_xlabel("Coordination Number", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL)
    ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)


def plot_composition_analysis(
    composition: Dict[str, Any],
    config: SimulationConfig,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (15, 5),
) -> plt.Figure:
    """
    Plot pre-computed cluster composition data.
    
    Creates a 1x3 grid:
        [A: Composition histogram] [B: Mean composition over time] [C: Composition vs size]
    
    Parameters
    ----------
    composition : dict
        Pre-computed data from get_cluster_composition()
    config : SimulationConfig
        Simulation configuration (for timestep)
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    
    Returns
    -------
    fig : matplotlib.Figure
        The figure object
    """
    print("\nGenerating composition plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    ax_hist, ax_time, ax_scatter = axes
    
    times_us, time_label = _time_axis(_steps_to_us(composition["times"], config.timestep))

    # =========================================================================
    # Plot A: Composition histogram (final frame)
    # =========================================================================
    final_fractions = composition["qt_fraction_per_cluster"][-1] if composition["qt_fraction_per_cluster"] else []
    
    if len(final_fractions) > 0:
        ax_hist.hist(final_fractions, bins=20, range=(0, 1), color='purple', 
                    edgecolor='black', alpha=0.7)
        ax_hist.axvline(np.mean(final_fractions), color='red', linestyle='-', linewidth=2,
                       label=f'Mean={np.mean(final_fractions):.2f}')
        ax_hist.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Equal mix')
        ax_hist.legend(loc='best', fontsize=FONTSIZE_LEGEND)
    
    ax_hist.set_xlabel("Qt Fraction (QtC / total)", fontsize=FONTSIZE_LABEL)
    ax_hist.set_ylabel("Number of Clusters", fontsize=FONTSIZE_LABEL)
    ax_hist.set_title("Composition Distribution (Final)", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax_hist.set_xlim([0, 1])
    ax_hist.grid(True, alpha=0.3)
    
    # =========================================================================
    # Plot B: Mean composition over time with std
    # =========================================================================
    ax_time.plot(times_us, composition["mean_qt_fraction"], 'b-', linewidth=2, label='Mean Qt fraction')
    ax_time.fill_between(times_us,
                         composition["mean_qt_fraction"] - composition["std_qt_fraction"],
                         composition["mean_qt_fraction"] + composition["std_qt_fraction"],
                         alpha=0.3, color='blue', label='± 1 SD')
    ax_time.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Equal mix')
    
    ax_time.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax_time.set_ylabel("Qt Fraction", fontsize=FONTSIZE_LABEL)
    ax_time.set_title("Mean Cluster Composition Over Time", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax_time.set_ylim([0, 1])
    ax_time.legend(loc='best', fontsize=FONTSIZE_LEGEND)
    ax_time.grid(True, alpha=0.3)
    
    # =========================================================================
    # Plot C: Composition vs size scatter (colored by time)
    # =========================================================================
    all_fractions = []
    all_sizes = []
    all_times = []
    
    for i, (fracs, sizes) in enumerate(zip(composition["qt_fraction_per_cluster"],
                                            composition["size_per_cluster"])):
        for frac, size in zip(fracs, sizes):
            all_fractions.append(frac)
            all_sizes.append(size)
            all_times.append(times_us[i])
    
    if len(all_fractions) > 0:
        scatter = ax_scatter.scatter(all_sizes, all_fractions, c=all_times, cmap='viridis',
                                     alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax_scatter, label=time_label)
        ax_scatter.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    
    ax_scatter.set_xlabel("Cluster Size (particles)", fontsize=FONTSIZE_LABEL)
    ax_scatter.set_ylabel("Qt Fraction", fontsize=FONTSIZE_LABEL)
    ax_scatter.set_title("Composition vs Cluster Size", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax_scatter.set_ylim([0, 1])
    ax_scatter.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
        print(f"\n✓ Saved plot to {save_path}")
    
    plt.show()
    
    return fig



def plot_cluster_composition(
    h5_file: str,
    config: SimulationConfig,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (15, 5),
    stride: int = 1,
    min_cluster_size: int = 2,
    frame_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute and plot cluster composition analysis.
    
    This is a convenience wrapper that calls get_cluster_composition()
    followed by plot_composition_analysis(). For more control, use these
    functions separately.
    
    Creates a 1x3 grid:
        [A: Composition histogram] [B: Mean composition over time] [C: Composition vs size]
    
    Parameters
    ----------
    h5_file : str
        Path to trajectory HDF5 file
    config : SimulationConfig
        Simulation configuration
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    stride : int
        Analyze every Nth frame
    min_cluster_size : int
        Minimum cluster size to include
    frame_data : dict, optional
        Pre-extracted frame data from _extract_frame_data()
    
    Returns
    -------
    dict
        Composition analysis data
    """
    print("\n" + "=" * 60)
    print("CLUSTER COMPOSITION ANALYSIS")
    print("=" * 60)
    
    # Compute
    composition = get_cluster_composition(h5_file, config, stride=stride, 
                                          min_cluster_size=min_cluster_size,
                                          frame_data=frame_data)
    
    # Plot
    plot_composition_analysis(composition, config, save_path=save_path, figsize=figsize)
    
    print("\n" + "=" * 60)
    print("COMPOSITION ANALYSIS COMPLETE")
    print("=" * 60 + "\n")
    
    return composition



def plot_structural_analysis(
    data: Dict[str, Any],
    config: Optional[SimulationConfig] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (20, 20),
) -> plt.Figure:
    """
    Plot pre-computed structural cluster analysis data.
    
    Creates a 4x3 grid:
    
    Row 1 - Morphology:
        [Rg over time] [Rg vs size scatter] [Compactness distribution]
    
    Row 2 - Kinetics:
        [Bonds + rate] [Free particles] [Fraction bound]
    
    Row 3 - Spatial:
        [NN distance over time] [Cluster positions XY/XZ/YZ] [NN distribution]
    
    Row 4 - Contacts:
        [Coordination over time] [Coord distribution] [Bonds vs cluster size]
    
    Parameters
    ----------
    data : dict
        Pre-computed data from compute_structural_analysis()
    config : SimulationConfig, optional
        Simulation configuration. If not provided, uses config stored in data.
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    
    Returns
    -------
    fig : matplotlib.Figure
        The figure object
    """
    # Get config from data if not provided
    if config is None:
        config = data.get("config")
        if config is None:
            raise ValueError("config must be provided either as argument or in data dict")
    
    morphology = data["morphology"]
    kinetics = data["kinetics"]
    spatial = data["spatial"]
    contacts = data["contacts"]
    
    print("\nGenerating structural analysis plots...")
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.35)
    
    # One adaptive unit for every time axis / marker in this multi-panel figure.
    time_factor, time_unit = choose_time_unit(config.total_simulation_time_us)
    time_label = f"Time ({time_unit})"

    # =========================================================================
    # Row 1: Morphology
    # =========================================================================
    ax_rg_time = fig.add_subplot(gs[0, 0])
    ax_rg_size = fig.add_subplot(gs[0, 1])
    ax_rg_dist = fig.add_subplot(gs[0, 2])
    
    # Plot 1.1: Rg over time
    times_us = _steps_to_us(morphology["times"], config.timestep) * time_factor
    ax_rg_time.plot(times_us, morphology["mean_rg"], 'b-', linewidth=2, label='Mean Rg')
    ax_rg_time.fill_between(times_us, 
                            morphology["mean_rg"] - morphology["std_rg"],
                            morphology["mean_rg"] + morphology["std_rg"],
                            alpha=0.3, color='blue', label='± 1 SD')
    ax_rg_time.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax_rg_time.set_ylabel("Radius of Gyration (nm)", fontsize=FONTSIZE_LABEL)
    ax_rg_time.set_title("Cluster Rg Over Time", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax_rg_time.legend(loc='best', fontsize=FONTSIZE_LEGEND)
    ax_rg_time.grid(True, alpha=0.3)
    
    # Plot 1.2: Rg vs size (colored by time)
    all_rg = []
    all_sizes = []
    all_times = []
    for i, (rgs, sizes) in enumerate(zip(morphology["rg_per_cluster"], 
                                          morphology["size_per_cluster"])):
        for rg, size in zip(rgs, sizes):
            all_rg.append(rg)
            all_sizes.append(size)
            all_times.append(times_us[i])
    
    if len(all_rg) > 0:
        scatter = ax_rg_size.scatter(all_sizes, all_rg, c=all_times, cmap='viridis', 
                                      alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax_rg_size, label=time_label)
    ax_rg_size.set_xlabel("Cluster Size (particles)", fontsize=FONTSIZE_LABEL)
    ax_rg_size.set_ylabel("Radius of Gyration (nm)", fontsize=FONTSIZE_LABEL)
    ax_rg_size.set_title("Rg vs Cluster Size", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax_rg_size.grid(True, alpha=0.3)
    
    # Plot 1.3: Compactness distribution (final frame)
    final_rg_norm = morphology["rg_normalized"][-1] if morphology["rg_normalized"] else []
    if len(final_rg_norm) > 0:
        ax_rg_dist.hist(final_rg_norm, bins=20, color='green', edgecolor='black', alpha=0.7)
        ax_rg_dist.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Compact sphere')
        ax_rg_dist.axvline(np.mean(final_rg_norm), color='blue', linestyle='-', 
                          linewidth=2, label=f'Mean={np.mean(final_rg_norm):.2f}')
        ax_rg_dist.legend(loc='best', fontsize=FONTSIZE_LEGEND)
    ax_rg_dist.set_xlabel("Rg / Rg_ideal", fontsize=FONTSIZE_LABEL)
    ax_rg_dist.set_ylabel("Count", fontsize=FONTSIZE_LABEL)
    ax_rg_dist.set_title("Compactness Distribution (Final)", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax_rg_dist.grid(True, alpha=0.3)
    
    # =========================================================================
    # Row 2: Kinetics
    # =========================================================================
    ax_bonds = fig.add_subplot(gs[1, 0])
    ax_free = fig.add_subplot(gs[1, 1])
    ax_frac = fig.add_subplot(gs[1, 2])
    
    # Plot 2.1: Bonds over time with rate
    times_kin_us = _steps_to_us(kinetics["times"], config.timestep) * time_factor
    ax_bonds.plot(times_kin_us, kinetics["n_bonds"], 'b-', linewidth=2, label='Bonds')
    ax_bonds.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax_bonds.set_ylabel("Number of Bonds", color='blue', fontsize=FONTSIZE_LABEL)
    ax_bonds.tick_params(axis='y', labelcolor='blue')
    ax_bonds.grid(True, alpha=0.3)
    
    ax_rate = ax_bonds.twinx()
    ax_rate.plot(times_kin_us, kinetics["bond_rate"], 'r-', linewidth=1, alpha=0.7, label='Rate')
    ax_rate.set_ylabel("Bond Rate (bonds/ns)", color='red', fontsize=FONTSIZE_LABEL)
    ax_rate.tick_params(axis='y', labelcolor='red')
    ax_bonds.set_title("Bonds and Binding Rate", fontsize=FONTSIZE_TITLE, fontweight='bold')
    
    # Plot 2.2: Free particles
    ax_free.plot(times_kin_us, kinetics["free_qt"], 'b-', linewidth=2, label='Free Qt')
    ax_free.plot(times_kin_us, kinetics["free_ft"], 'r-', linewidth=2, label='Free Ft')
    ax_free.plot(times_kin_us, kinetics["clustered_qt"], 'b--', linewidth=1.5, label='QtC')
    ax_free.plot(times_kin_us, kinetics["clustered_ft"], 'r--', linewidth=1.5, label='FtC')
    ax_free.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax_free.set_ylabel("Particle Count", fontsize=FONTSIZE_LABEL)
    ax_free.set_title("Free vs Clustered Particles", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax_free.legend(loc='best', fontsize=FONTSIZE_LEGEND)
    ax_free.grid(True, alpha=0.3)
    
    # Plot 2.3: Fraction bound
    ax_frac.plot(times_kin_us, kinetics["fraction_bound_qt"], 'b-', linewidth=2, label='Qt')
    ax_frac.plot(times_kin_us, kinetics["fraction_bound_ft"], 'r-', linewidth=2, label='Ft')
    ax_frac.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Mark half-times
    if kinetics["half_time_qt"] is not None:
        ht_qt_us = _steps_to_us(np.array([kinetics["half_time_qt"]]), config.timestep)[0] * time_factor
        ax_frac.axvline(ht_qt_us, color='blue', linestyle=':', alpha=0.7)
        ax_frac.text(ht_qt_us, 0.52, f't½={ht_qt_us:.2f}', color='blue', fontsize=FONTSIZE_LEGEND)
    if kinetics["half_time_ft"] is not None:
        ht_ft_us = _steps_to_us(np.array([kinetics["half_time_ft"]]), config.timestep)[0] * time_factor
        ax_frac.axvline(ht_ft_us, color='red', linestyle=':', alpha=0.7)
        ax_frac.text(ht_ft_us, 0.48, f't½={ht_ft_us:.2f}', color='red', fontsize=FONTSIZE_LEGEND)
    
    ax_frac.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax_frac.set_ylabel("Fraction Bound", fontsize=FONTSIZE_LABEL)
    ax_frac.set_title("Binding Progress", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax_frac.legend(loc='best', fontsize=FONTSIZE_LEGEND)
    ax_frac.set_ylim([0, 1])
    ax_frac.grid(True, alpha=0.3)
    
    # =========================================================================
    # Row 3: Spatial Distribution
    # =========================================================================
    ax_nn_time = fig.add_subplot(gs[2, 0])
    ax_positions = fig.add_subplot(gs[2, 1])
    ax_nn_dist = fig.add_subplot(gs[2, 2])
    
    # Plot 3.1: NN distance over time
    times_sp_us = _steps_to_us(spatial["times"], config.timestep) * time_factor
    ax_nn_time.plot(times_sp_us, spatial["mean_nn_dist"], 'b-', linewidth=2, label='Observed')
    ax_nn_time.fill_between(times_sp_us,
                            spatial["mean_nn_dist"] - spatial["std_nn_dist"],
                            spatial["mean_nn_dist"] + spatial["std_nn_dist"],
                            alpha=0.3, color='blue')
    ax_nn_time.plot(times_sp_us, spatial["expected_nn_dist"], 'r--', linewidth=2, label='Random expected')
    ax_nn_time.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax_nn_time.set_ylabel("Inter-Cluster NN Dist (nm)", fontsize=FONTSIZE_LABEL)
    ax_nn_time.set_title("Nearest-Neighbor Distance", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax_nn_time.legend(loc='best', fontsize=FONTSIZE_LEGEND)
    ax_nn_time.grid(True, alpha=0.3)
    
    # Plot 3.2: Cluster positions (3 projections in one subplot)
    # Create sub-gridspec for 3 projections
    final_centers = spatial["cluster_centers"][-1] if len(spatial["cluster_centers"]) > 0 else np.zeros((0, 3))
    final_sizes = spatial["cluster_sizes"][-1] if len(spatial["cluster_sizes"]) > 0 else np.array([])
    box = spatial["box_size"]
    
    ax_positions.set_visible(False)  # Hide the main axis
    
    # Create 3 small subplots for projections
    gs_inner = gs[2, 1].subgridspec(1, 3, wspace=0.3)
    ax_xy = fig.add_subplot(gs_inner[0, 0])
    ax_xz = fig.add_subplot(gs_inner[0, 1])
    ax_yz = fig.add_subplot(gs_inner[0, 2])
    
    if len(final_centers) > 0:
        # Size scaling for scatter
        size_scale = 20 + 100 * (final_sizes - final_sizes.min()) / (final_sizes.max() - final_sizes.min() + 1)
        
        ax_xy.scatter(final_centers[:, 0], final_centers[:, 1], s=size_scale, alpha=0.6, c='blue')
        ax_xy.set_xlim([-box[0]/2, box[0]/2])
        ax_xy.set_ylim([-box[1]/2, box[1]/2])
        ax_xy.set_xlabel("X (nm)", fontsize=FONTSIZE_LEGEND)
        ax_xy.set_ylabel("Y (nm)", fontsize=FONTSIZE_LEGEND)
        ax_xy.set_title("XY", fontsize=FONTSIZE_LABEL)
        ax_xy.set_aspect('equal')
        ax_xy.tick_params(labelsize=8)
        
        ax_xz.scatter(final_centers[:, 0], final_centers[:, 2], s=size_scale, alpha=0.6, c='blue')
        ax_xz.set_xlim([-box[0]/2, box[0]/2])
        ax_xz.set_ylim([-box[2]/2, box[2]/2])
        ax_xz.set_xlabel("X (nm)", fontsize=FONTSIZE_LEGEND)
        ax_xz.set_ylabel("Z (nm)", fontsize=FONTSIZE_LEGEND)
        ax_xz.set_title("XZ", fontsize=FONTSIZE_LABEL)
        ax_xz.set_aspect('equal')
        ax_xz.tick_params(labelsize=8)
        
        ax_yz.scatter(final_centers[:, 1], final_centers[:, 2], s=size_scale, alpha=0.6, c='blue')
        ax_yz.set_xlim([-box[1]/2, box[1]/2])
        ax_yz.set_ylim([-box[2]/2, box[2]/2])
        ax_yz.set_xlabel("Y (nm)", fontsize=FONTSIZE_LEGEND)
        ax_yz.set_ylabel("Z (nm)", fontsize=FONTSIZE_LEGEND)
        ax_yz.set_title("YZ", fontsize=FONTSIZE_LABEL)
        ax_yz.set_aspect('equal')
        ax_yz.tick_params(labelsize=8)
    else:
        for ax in [ax_xy, ax_xz, ax_yz]:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes,
                   fontsize=FONTSIZE_LABEL, color='gray')
    
    # Add overall title for positions (y=0.395 positions it above row 3 subplots in 4-row layout)
    fig.text(0.5, 0.45, "Cluster Positions (Final)", ha='center', fontsize=FONTSIZE_TITLE, fontweight='bold')
    
    # Plot 3.3: NN distance distribution (final frame)
    final_nn = spatial["nn_distances"][-1] if len(spatial["nn_distances"]) > 0 else np.array([])
    if len(final_nn) > 0:
        ax_nn_dist.hist(final_nn, bins=20, color='purple', edgecolor='black', alpha=0.7, density=True)
        ax_nn_dist.axvline(np.mean(final_nn), color='blue', linestyle='-', linewidth=2, 
                          label=f'Mean={np.mean(final_nn):.1f}')
        if spatial["expected_nn_dist"][-1] > 0:
            ax_nn_dist.axvline(spatial["expected_nn_dist"][-1], color='red', linestyle='--', 
                              linewidth=2, label=f'Random={spatial["expected_nn_dist"][-1]:.1f}')
        ax_nn_dist.legend(loc='best', fontsize=FONTSIZE_LEGEND)
    ax_nn_dist.set_xlabel("Inter-Cluster NN Dist (nm)", fontsize=FONTSIZE_LABEL)
    ax_nn_dist.set_ylabel("Density", fontsize=FONTSIZE_LABEL)
    ax_nn_dist.set_title("Inter-Cluster NN Dist Distribution (Final)", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax_nn_dist.grid(True, alpha=0.3)
    
    # =========================================================================
    # Row 4: Contacts
    # =========================================================================
    ax_coord_time = fig.add_subplot(gs[3, 0])
    ax_coord_dist = fig.add_subplot(gs[3, 1])
    ax_bonds_size = fig.add_subplot(gs[3, 2])
    
    # Plot 4.1: Coordination over time
    times_ct_us = _steps_to_us(contacts["times"], config.timestep) * time_factor
    ax_coord_time.plot(times_ct_us, contacts["mean_coord_qt"], linestyle='-',
                       color=COORD_COLOR_QT, linewidth=2, label='QtC')
    ax_coord_time.fill_between(times_ct_us,
                               contacts["mean_coord_qt"] - contacts["std_coord_qt"],
                               contacts["mean_coord_qt"] + contacts["std_coord_qt"],
                               alpha=0.2, color=COORD_COLOR_QT)
    ax_coord_time.plot(times_ct_us, contacts["mean_coord_ft"], linestyle='-',
                       color=COORD_COLOR_FT, linewidth=2, label='FtC')
    ax_coord_time.fill_between(times_ct_us,
                               contacts["mean_coord_ft"] - contacts["std_coord_ft"],
                               contacts["mean_coord_ft"] + contacts["std_coord_ft"],
                               alpha=0.2, color=COORD_COLOR_FT)
    ax_coord_time.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax_coord_time.set_ylabel("Coordination Number", fontsize=FONTSIZE_LABEL)
    ax_coord_time.set_title("Mean Coordination Over Time", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax_coord_time.legend(loc='best', fontsize=FONTSIZE_LEGEND)
    ax_coord_time.grid(True, alpha=0.3)
    
    # Plot 4.2: Coordination distribution (final frame)
    final_coord_qt = contacts["coord_dist_qt"][-1] if contacts["coord_dist_qt"] else np.array([])
    final_coord_ft = contacts["coord_dist_ft"][-1] if contacts["coord_dist_ft"] else np.array([])
    _plot_coord_distribution(ax_coord_dist, final_coord_qt, final_coord_ft)
    
    # Plot 4.3: Bonds vs cluster size
    all_bonds = []
    all_cluster_sizes = []
    all_frame_times = []
    for i, (bonds, sizes) in enumerate(zip(contacts["bonds_per_cluster"], 
                                            contacts["sizes_per_cluster"])):
        for b, s in zip(bonds, sizes):
            if s >= 2:  # Only clusters with at least 2 particles
                all_bonds.append(b)
                all_cluster_sizes.append(s)
                all_frame_times.append(times_ct_us[i])
    
    if len(all_bonds) > 0:
        scatter = ax_bonds_size.scatter(all_cluster_sizes, all_bonds, c=all_frame_times, 
                                        cmap='viridis', alpha=0.5, s=20)
        plt.colorbar(scatter, ax=ax_bonds_size, label=time_label)

        # Add reference line (n-1 bonds for linear chain)
        x_ref = np.array([2, max(all_cluster_sizes)])
        ax_bonds_size.plot(x_ref, x_ref - 1, 'k--', linewidth=1, alpha=0.5, label='Linear (n-1)')
        ax_bonds_size.legend(loc='best', fontsize=FONTSIZE_LEGEND)
    ax_bonds_size.set_xlabel("Cluster Size (particles)", fontsize=FONTSIZE_LABEL)
    ax_bonds_size.set_ylabel("Number of Bonds", fontsize=FONTSIZE_LABEL)
    ax_bonds_size.set_title("Bonds vs Cluster Size", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax_bonds_size.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
        print(f"\n✓ Saved plot to {save_path}")
    
    plt.show()
    
    return fig



def plot_structural_cluster_analysis(
    h5_file: str,
    config: SimulationConfig,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (20, 20),
    stride: int = 1,
    min_cluster_size_morphology: int = 3,
    min_cluster_size_spatial: int = 2,
) -> Dict[str, Any]:
    """
    Compute and plot comprehensive structural cluster analysis.
    
    This is a convenience wrapper that calls compute_structural_analysis()
    followed by plot_structural_analysis(). For more control, use these
    functions separately.
    
    Creates a 4x3 grid:
    
    Row 1 - Morphology:
        [Rg over time] [Rg vs size scatter] [Compactness distribution]
    
    Row 2 - Kinetics:
        [Bonds + rate] [Free particles] [Fraction bound]
    
    Row 3 - Spatial:
        [NN distance over time] [Cluster positions XY/XZ/YZ] [NN distribution]
    
    Row 4 - Contacts:
        [Coordination over time] [Coord distribution] [Bonds vs cluster size]
    
    Parameters
    ----------
    h5_file : str
        Path to trajectory HDF5 file
    config : SimulationConfig
        Simulation configuration
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    stride : int
        Analyze every Nth frame for detailed analyses
    min_cluster_size_morphology : int
        Minimum cluster size for Rg calculation
    min_cluster_size_spatial : int
        Minimum cluster size for spatial analysis
    
    Returns
    -------
    dict
        All computed data from analyses (morphology, kinetics, spatial, contacts)
    """
    # Compute
    data = compute_structural_analysis(
        h5_file, config, stride=stride,
        min_cluster_size_morphology=min_cluster_size_morphology,
        min_cluster_size_spatial=min_cluster_size_spatial,
    )
    
    # Plot
    plot_structural_analysis(data, config, save_path=save_path, figsize=figsize)
    
    # Return data (without config to match original signature)
    return {
        "morphology": data["morphology"],
        "kinetics": data["kinetics"],
        "spatial": data["spatial"],
        "contacts": data["contacts"],
    }





# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_observables(
    h5_file: str,
    config: SimulationConfig,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (18, 15),
) -> Dict[str, Any]:
    """
    Create comprehensive observable plots.
    
    Creates a 3x3 grid with 7 plots:
    - Row 1: Particle counts, Energy, Pressure
    - Row 2: RDF, Cumulative reactions, Bonds
    - Row 3: Energy vs bonds (centered)
    
    Parameters
    ----------
    h5_file : str
        Path to trajectory HDF5 file
    config : SimulationConfig
        Simulation configuration (for labels)
    save_path : str, optional
        Path to save figure (SVG format)
    figsize : tuple
        Figure size
    
    Returns
    -------
    dict
        Raw data for further analysis
    """
    traj = readdy.Trajectory(h5_file)
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.45)
    
    # Row 1
    ax_np = fig.add_subplot(gs[0, 0])
    ax_energy = fig.add_subplot(gs[0, 1])
    ax_pressure = fig.add_subplot(gs[0, 2])
    
    # Row 2
    ax_rdf = fig.add_subplot(gs[1, 0])
    ax_cumulative = fig.add_subplot(gs[1, 1])
    ax_bonds = fig.add_subplot(gs[1, 2])
    
    # Row 3 (only one plot, centered)
    ax_energy_bonds = fig.add_subplot(gs[2, 1])
    
    time_label = "Time (µs)"
    types = [config.qt.name, config.ft.name, config.qt.cluster_name, config.ft.cluster_name]
    timestep = config.timestep
    
    # Collect data for return
    data = {}
    
    # Row 1
    # 1. Number of particles
    data["particles"] = _plot_particle_counts(ax_np, traj, types, time_label, timestep)
    
    # 2. Energy
    data["energy"] = _plot_energy(ax_energy, traj, time_label, timestep)
    
    # 3. Pressure
    _plot_pressure(ax_pressure, traj, time_label, timestep)
    
    # Row 2
    # 4. RDF
    _plot_rdf(ax_rdf, traj)
    
    # 5. Cumulative reactions
    _plot_cumulative_reactions(ax_cumulative, traj, time_label, timestep)
    
    # 6. Bonds
    data["bonds"] = _plot_bonds(ax_bonds, h5_file, time_label, timestep)
    
    # Row 3
    # 7. Energy vs bonds
    _plot_energy_vs_bonds(ax_energy_bonds, data.get("energy"), data.get("bonds"))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, format="svg", bbox_inches="tight", dpi=300)
        print(f"✓ Saved plot to {save_path}")
    
    plt.show()
    
    return data



def _plot_particle_counts(ax, traj, types, time_label, timestep: float) -> Optional[Dict]:
    """Plot particle counts over time."""
    try:
        result = traj.read_observable_number_of_particles()
        times, time_label = _time_axis(_steps_to_us(result[0], timestep))
        counts = np.array(result[1])
        
        for i, label in enumerate(types):
            ax.plot(times, counts[:, i], label=label, linewidth=1.5)
        
        ax.plot(times, counts.sum(axis=1), label="Total", color="black", linewidth=1.5)
        
        ax.set_title("Particle Counts", fontsize=FONTSIZE_TITLE, fontweight="bold")
        ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
        ax.set_ylabel("Count", fontsize=FONTSIZE_LABEL)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=FONTSIZE_LEGEND)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        
        return {"times": times, "counts": counts}
    except (KeyError, ValueError, IndexError) as e:
        print(f"  Warning: Could not plot particle counts: {e}")
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", 
               transform=ax.transAxes, fontsize=FONTSIZE_TITLE, color="gray")
        ax.axis("off")
        return None
    except Exception as e:
        print(f"  Warning: Unexpected error in particle counts plot: {e}")
        traceback.print_exc()
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", 
               transform=ax.transAxes, fontsize=FONTSIZE_TITLE, color="gray")
        ax.axis("off")
        return None



def _plot_energy(ax, traj, time_label, timestep: float) -> Optional[Dict]:
    """Plot energy over time."""
    try:
        times, energy = traj.read_observable_energy()
        times, time_label = _time_axis(_steps_to_us(times, timestep))
        energy = np.array(energy)
        
        ax.plot(times, energy, color="tab:red", linewidth=1.5)
        ax.set_title("Potential Energy", fontsize=FONTSIZE_TITLE, fontweight="bold")
        ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
        ax.set_ylabel("Energy (kJ/mol)", fontsize=FONTSIZE_LABEL)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        
        return {"times": times, "energy": energy, "time_label": time_label}
    except (KeyError, ValueError) as e:
        print(f"  Warning: Could not plot energy: {e}")
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", 
               transform=ax.transAxes, fontsize=FONTSIZE_TITLE, color="gray")
        ax.axis("off")
        return None
    except Exception as e:
        print(f"  Warning: Unexpected error in energy plot: {e}")
        traceback.print_exc()
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", 
               transform=ax.transAxes, fontsize=FONTSIZE_TITLE, color="gray")
        ax.axis("off")
        return None



def _plot_pressure(ax, traj, time_label, timestep: float):
    """Plot pressure over time."""
    try:
        times, pressure = traj.read_observable_pressure()
        times, time_label = _time_axis(_steps_to_us(times, timestep))
        
        ax.plot(times, pressure, color="tab:green", linewidth=1.5)
        ax.set_title("Pressure", fontsize=FONTSIZE_TITLE, fontweight="bold")
        ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
        ax.set_ylabel("Pressure (kJ/(mol·nm³))", fontsize=FONTSIZE_LABEL)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    except (KeyError, ValueError) as e:
        print(f"  Warning: Could not plot pressure: {e}")
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", 
               transform=ax.transAxes, fontsize=FONTSIZE_TITLE, color="gray")
        ax.axis("off")
    except Exception as e:
        print(f"  Warning: Unexpected error in pressure plot: {e}")
        traceback.print_exc()
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", 
               transform=ax.transAxes, fontsize=FONTSIZE_TITLE, color="gray")
        ax.axis("off")



def _plot_cumulative_reactions(ax, traj, time_label, timestep: float):
    """Plot cumulative reaction counts."""
    try:
        times, counts_dict = traj.read_observable_reaction_counts()
        times, time_label = _time_axis(_steps_to_us(times, timestep))
        
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        total_cumulative = np.zeros(len(times))
        color_idx = 0
        any_plotted = False
        
        def extract_series(obj, prefix=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_prefix = f"{prefix}/{k}" if prefix else k
                    yield from extract_series(v, new_prefix)
            else:
                yield prefix, np.asarray(obj).flatten()
        
        for name, series in extract_series(counts_dict):
            if len(series) == len(times) and series.sum() > 0:
                display_name = name.split("/")[-1]
                color = colors[color_idx % len(colors)]
                
                # Cumulative
                cumul = np.cumsum(series)
                total_cumulative += cumul
                ax.plot(times, cumul, label=display_name, linewidth=1.5, color=color)
                any_plotted = True
                
                color_idx += 1
        
        if any_plotted:
            ax.plot(times, total_cumulative, label="Total", linewidth=2, color="black")
        
        ax.set_title("Cumulative Reactions", fontsize=FONTSIZE_TITLE, fontweight="bold")
        ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
        ax.set_ylabel("Total reactions", fontsize=FONTSIZE_LABEL)
        ax.grid(True, alpha=0.3)
        if any_plotted:
            ax.legend(loc="upper left", fontsize=FONTSIZE_LEGEND)
            
    except (KeyError, ValueError) as e:
        print(f"  Warning: Could not plot reactions: {e}")
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", 
               transform=ax.transAxes, fontsize=FONTSIZE_TITLE, color="gray")
        ax.axis("off")
    except Exception as e:
        print(f"  Warning: Unexpected error in reactions plot: {e}")
        traceback.print_exc()
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", 
               transform=ax.transAxes, fontsize=FONTSIZE_TITLE, color="gray")
        ax.axis("off")



def _parse_rdf_result(result) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse RDF result from ReaDDy trajectory.
    
    ReaDDy's RDF output format can vary between versions.
    This function handles known formats.
    
    Parameters
    ----------
    result : tuple
        Result from traj.read_observable_rdf()
    
    Returns
    -------
    bins, values : tuple of ndarray
        Bin centers and RDF values (last frame if multiple)
    
    Raises
    ------
    ValueError
        If the RDF format is not recognized
    """
    if len(result) == 2:
        # Format: (bins, values)
        bins, values = np.array(result[0]), np.array(result[1])
    elif len(result) == 3:
        # Format: (times, bins, values)
        _, bins, values = result
        bins, values = np.array(bins), np.array(values)
    else:
        raise ValueError(
            f"Unexpected RDF format: got {len(result)} elements, expected 2 or 3. "
            "This may be due to a ReaDDy version change."
        )
    
    # Handle multi-frame RDF (use last frame)
    if values.ndim == 2:
        if values.shape[0] == len(bins):
            # Shape is (n_bins, n_frames)
            values = values[:, -1]
        else:
            # Shape is (n_frames, n_bins)
            values = values[-1, :]
    
    return bins, values



def _plot_rdf(ax, traj):
    """Plot radial distribution function."""
    try:
        result = traj.read_observable_rdf()
        bins, values = _parse_rdf_result(result)
        
        ax.plot(bins, values, color="tab:purple", linewidth=2)
        ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, label="Ideal gas")
        
        ax.set_title("Radial Distribution g(r)", fontsize=FONTSIZE_TITLE, fontweight="bold")
        ax.set_xlabel("Distance r (nm)", fontsize=FONTSIZE_LABEL)
        ax.set_ylabel("g(r)", fontsize=FONTSIZE_LABEL)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=FONTSIZE_LEGEND)
        
    except (KeyError, ValueError) as e:
        print(f"  Warning: Could not plot RDF: {e}")
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", 
               transform=ax.transAxes, fontsize=FONTSIZE_TITLE, color="gray")
        ax.axis("off")
    except Exception as e:
        print(f"  Warning: Unexpected error in RDF plot: {e}")
        traceback.print_exc()
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", 
               transform=ax.transAxes, fontsize=FONTSIZE_TITLE, color="gray")
        ax.axis("off")



def _plot_bonds(ax, h5_file, time_label, timestep: float, trajectory=None) -> Optional[Dict]:
    """Plot bond count over time."""
    try:
        data = get_bond_counts(h5_file, trajectory=trajectory)
        times, time_label = _time_axis(_steps_to_us(data["times"], timestep))
        bonds = data["n_bonds"]
        
        ax.plot(times, bonds, color="tab:orange", linewidth=2)
        ax.set_title("Number of Bonds", fontsize=FONTSIZE_TITLE, fontweight="bold")
        ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
        ax.set_ylabel("Bond count", fontsize=FONTSIZE_LABEL)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        
        ax.text(0.05, 0.95, f"Final: {bonds[-1]}", transform=ax.transAxes,
               fontsize=FONTSIZE_LEGEND, va="top",
               bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        
        return {"times": times, "bonds": bonds, "trajectory": data.get("trajectory"),
                "time_label": time_label}
    except (KeyError, ValueError, FileNotFoundError) as e:
        print(f"  Warning: Could not plot bonds: {e}")
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", 
               transform=ax.transAxes, fontsize=FONTSIZE_TITLE, color="gray")
        ax.axis("off")
        return None
    except Exception as e:
        print(f"  Warning: Unexpected error in bonds plot: {e}")
        traceback.print_exc()
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", 
               transform=ax.transAxes, fontsize=FONTSIZE_TITLE, color="gray")
        ax.axis("off")
        return None



def _plot_energy_vs_bonds(ax, energy_data, bond_data):
    """Scatter plot of energy vs bond count."""
    if energy_data is None or bond_data is None:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
               fontsize=FONTSIZE_TITLE, color="gray")
        ax.axis("off")
        return
    
    # Interpolate to match lengths
    energy = np.interp(bond_data["times"], energy_data["times"], energy_data["energy"])
    bonds = bond_data["bonds"]
    times = bond_data["times"]
    
    scatter = ax.scatter(bonds, energy, c=times, cmap="viridis", alpha=0.6, s=15)
    plt.colorbar(scatter, ax=ax, label=bond_data.get("time_label", "Time"))
    
    # Linear fit
    if len(bonds) > 10:
        coeffs = np.polyfit(bonds, energy, 1)
        x_fit = np.linspace(bonds.min(), bonds.max(), 100)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), "r--", linewidth=2, alpha=0.8,
               label=f"Slope: {coeffs[0]:.2f}")
        ax.legend(loc="best", fontsize=FONTSIZE_LEGEND)
    
    ax.set_title("Energy vs Bonds", fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax.set_xlabel("Number of bonds", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Energy (kJ/mol)", fontsize=FONTSIZE_LABEL)
    ax.grid(True, alpha=0.3)



def plot_cluster_analysis(
    h5_file: str,
    config: Optional[SimulationConfig] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (18, 10),
) -> Dict[str, Any]:
    """
    Create detailed cluster analysis plots.
    
    Creates a 2x3 grid with:
    - Number of topologies over time
    - Average cluster size over time
    - Final size distribution histogram
    - Size evolution heatmap
    - Largest cluster over time
    - Particle fraction by size category
    
    Parameters
    ----------
    h5_file : str
        Path to trajectory HDF5 file
    config : SimulationConfig, optional
        Simulation configuration. Required for correct time axis.
        If not provided, times shown as step numbers.
    save_path : str, optional
        Path to save figure (SVG format)
    figsize : tuple
        Figure size
    
    Returns
    -------
    dict
        Cluster statistics
    """
    stats = get_cluster_statistics(h5_file)
    
    # Convert times - need timestep from config for proper conversion
    if config is not None:
        times_us, time_label = _time_axis(_steps_to_us(stats["times"], config.timestep))
    else:
        times_us = stats["times"]  # Just use step numbers
        time_label = "Time (steps)"
    # Bare unit (e.g. "µs") for inline annotations, derived from the axis label.
    time_unit = time_label[time_label.find("(") + 1: time_label.find(")")] if "(" in time_label else ""

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    
    # 1. Number of topologies
    ax1.plot(times_us, stats["n_clusters"], linewidth=2, color="tab:blue")
    ax1.set_title("Number of Individual Topologies", fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax1.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax1.set_ylabel("Count", fontsize=FONTSIZE_LABEL)
    ax1.grid(True, alpha=0.3)
    _add_annotation(ax1, stats["n_clusters"], pos="top-right")
    
    # 2. Average size
    ax2.plot(times_us, stats["avg_sizes"], linewidth=2, color="tab:green")
    ax2.set_title("Average Cluster Size", fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax2.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax2.set_ylabel("Size (particles)", fontsize=FONTSIZE_LABEL)
    ax2.grid(True, alpha=0.3)
    _add_annotation(ax2, stats["avg_sizes"], fmt=".1f")
    
    # 3. Final histogram
    final_sizes = stats["cluster_sizes"][-1]
    if len(final_sizes) > 0:
        max_size = int(final_sizes.max())
        bins = np.arange(0.5, max_size + 1.5, 1)
        ax3.hist(final_sizes, bins=bins, color="tab:orange", edgecolor="black", alpha=0.7)
        ax3.axvline(np.median(final_sizes), color="red", linestyle="--",
                   label=f"Median={np.median(final_sizes):.1f}")
        ax3.axvline(np.mean(final_sizes), color="blue", linestyle="--",
                   label=f"Mean={np.mean(final_sizes):.1f}")
        ax3.legend(loc="upper right", fontsize=FONTSIZE_LEGEND)
    ax3.set_title(f"Size Distribution (t={times_us[-1]:.1f} {time_unit})", fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax3.set_xlabel("Cluster size", fontsize=FONTSIZE_LABEL)
    ax3.set_ylabel("Count", fontsize=FONTSIZE_LABEL)
    ax3.grid(True, alpha=0.3, axis="y")
    
    # 4. Size evolution heatmap
    _plot_size_heatmap(ax4, stats, times_us, time_label)
    
    # 5. Largest cluster
    ax5.plot(times_us, stats["max_sizes"], linewidth=2, color="tab:red")
    ax5.set_title("Largest Cluster", fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax5.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax5.set_ylabel("Size (particles)", fontsize=FONTSIZE_LABEL)
    ax5.grid(True, alpha=0.3)
    _add_annotation(ax5, stats["max_sizes"])
    
    # 6. Size category fractions (with adaptive categories)
    if config is not None:
        config_dict = config.to_dict()
        desc = _generate_ensemble_title(config_dict, n_replicas=None)
        size_title = f"Particles by Size Category — {desc}"
    else:
        size_title = "Particles by Size Category"
    _plot_size_fractions(ax6, stats, times_us, time_label, config=config, title=size_title)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, format="svg", bbox_inches="tight", dpi=300)
        print(f"✓ Saved plot to {save_path}")
    
    plt.show()
    
    return stats



def _add_annotation(ax, data, pos="top-left", fmt="d"):
    """Add initial/final annotation to axes."""
    if len(data) == 0:
        return
    
    if fmt == "d":
        text = f"Initial: {int(data[0])}\nFinal: {int(data[-1])}"
    else:
        text = f"Initial: {data[0]:{fmt}}\nFinal: {data[-1]:{fmt}}"
    
    ha = "right" if "right" in pos else "left"
    x = 0.95 if "right" in pos else 0.05
    
    ax.text(x, 0.95, text, transform=ax.transAxes, fontsize=FONTSIZE_LEGEND,
           va="top", ha=ha,
           bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"))



def _plot_size_heatmap(ax, stats, times_us, time_label):
    """Plot cluster size evolution as heatmap."""
    cluster_sizes = stats["cluster_sizes"]
    
    max_size = max(max(s) if len(s) > 0 else 0 for s in cluster_sizes)
    if max_size == 0:
        ax.text(0.5, 0.5, "No cluster data", ha="center", va="center",
               fontsize=FONTSIZE_TITLE, color="gray")
        ax.axis("off")
        return
    
    sizes = np.arange(1, int(max_size) + 1)
    matrix = np.zeros((len(sizes), len(cluster_sizes)))
    
    for t, frame_sizes in enumerate(cluster_sizes):
        for s in frame_sizes:
            if 1 <= s <= max_size:
                matrix[int(s) - 1, t] += 1
    
    row_mask = matrix.sum(axis=1) > 0
    matrix = matrix[row_mask]
    sizes = sizes[row_mask]
    
    if matrix.size == 0:
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center",
               fontsize=FONTSIZE_TITLE, color="gray")
        return
    
    vmax = matrix.max()
    vmin = max(1, matrix[matrix > 0].min()) if (matrix > 0).any() else 1
    
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", origin="lower",
                  extent=[times_us[0], times_us[-1], sizes[0] - 0.5, sizes[-1] + 0.5],
                  norm=LogNorm(vmin=vmin, vmax=vmax) if vmax / vmin > 10 else None)
    
    ax.set_title("Size Evolution", fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Cluster size", fontsize=FONTSIZE_LABEL)
    plt.colorbar(im, ax=ax, label="Count")



def _plot_size_fractions(ax, stats, times_us, time_label, config: Optional[SimulationConfig] = None,
                        title: Optional[str] = None):
    """Plot stacked area chart of particles by size category.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    stats : dict
        Cluster statistics (must contain 'cluster_sizes')
    times_us : ndarray
        Time array in microseconds
    time_label : str
        Label for time axis
    config : SimulationConfig, optional
        Configuration for adaptive category boundaries
    title : str, optional
        Custom plot title. If None, uses default.
    """
    cluster_sizes = stats["cluster_sizes"]
    
    # Get total particle count for adaptive categories
    total_particles = np.sum(cluster_sizes[0]) if len(cluster_sizes) > 0 and len(cluster_sizes[0]) > 0 else 400
    
    # Get structured category boundaries (no regex needed)
    boundaries = _get_size_category_boundaries(total_particles, config)
    
    # Compute fractions for each category at each time step
    fractions = {name: [] for name, _, _ in boundaries}
    
    for sizes in cluster_sizes:
        total = np.sum(sizes) if len(sizes) > 0 else 1
        for name, min_size, max_size in boundaries:
            if len(sizes) > 0:
                mask = _apply_size_category(sizes, min_size, max_size)
                frac = np.sum(sizes[mask]) / total if total > 0 else 0
            else:
                frac = 0
            fractions[name].append(frac)
    
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple"]
    
    ax.stackplot(times_us, *[fractions[n] for n, _, _ in boundaries],
                labels=[n for n, _, _ in boundaries], colors=colors[:len(boundaries)], alpha=0.8)
    
    plot_title = title if title is not None else "Particles by Size Category"
    ax.set_title(plot_title, fontsize=FONTSIZE_TITLE, fontweight="bold")
    ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Fraction", fontsize=FONTSIZE_LABEL)
    ax.set_ylim([0, 1])
    ax.legend(loc="upper right", fontsize=FONTSIZE_LEGEND)
    ax.grid(True, alpha=0.3, axis="y")


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================


def _ensemble_plot_with_band(
    ax, times, mean, std, color, n_replicas,
    all_data=None, show_individual=False, individual_alpha=0.3,
    label=None, band_label='± 1 SD'
) -> bool:
    """
    Helper function to plot mean with std band for ensemble data.

    Parameters such as ``label`` (mean-line legend label) and ``band_label`` allow
    overlaying multiple series in one axes without duplicate legend entries
    (pass ``band_label='_nolegend_'`` for the second series).

    Returns True if data was plotted, False otherwise.
    """
    if mean is None or len(mean) == 0:
        return False

    if n_replicas <= 0:
        n_replicas = 1  # Fallback to avoid confusing labels

    mean = np.asarray(mean)

    # Handle std being None (plot without error band)
    if std is None:
        std = np.zeros_like(mean)
    else:
        std = np.asarray(std)

    # A single run has no spread: drawing a zero-width "± 1 SD" band around every trace
    # (and labelling it "Mean (N=1)") reads as a bug, so plot the bare series instead.
    single_run = n_replicas == 1

    if label is not None:
        mean_label = label
    else:
        mean_label = 'Run' if single_run else f'Mean (N={n_replicas})'
    ax.plot(times, mean, color=color, linewidth=2, label=mean_label)
    if not single_run:
        ax.fill_between(times, mean - std, mean + std, color=color, alpha=0.3,
                        label=band_label)

    if show_individual and all_data is not None and not single_run:
        for data in all_data:
            ax.plot(times, data, color=color, alpha=individual_alpha, linewidth=0.5)
    
    return True



def _ensemble_show_no_data(ax):
    """Helper to show 'No data available' message on axes."""
    ax.text(0.5, 0.5, "No data available", ha='center', va='center',
           transform=ax.transAxes, fontsize=FONTSIZE_TITLE, color='gray')


def _ensemble_all_trace(stats: Dict, structural: Optional[Dict], key: str):
    """Return the per-replica traces array for ``key`` (``{key}_all``), or None.

    Checks ``structural`` first, then ``stats`` (per-replica arrays may live in either
    depending on the loader). Shared by the ensemble plots and the ensemble panel.
    """
    all_key = f'{key}_all'
    if structural is not None and all_key in structural:
        return structural[all_key]
    if all_key in stats:
        return stats[all_key]
    return None


def _ensemble_struct_ts(structural: Optional[Dict], timestep: float, time_key: str,
                        mean_key: str, std_key: str, all_key: Optional[str] = None):
    """Fetch a structural time series and convert its step axis to an adaptive time unit.

    Returns ``(times, mean, std, all_data, time_label)`` with any present arrays as ndarrays
    (times already scaled to the unit named in ``time_label``) and missing ones as None.
    Shared by the ensemble structural plot and the ensemble panel.
    """
    times = structural.get(time_key) if structural else None
    mean = structural.get(mean_key) if structural else None
    std = structural.get(std_key) if structural else None
    all_data = structural.get(all_key) if (structural and all_key) else None
    time_label = "Time (µs)"
    if times is not None:
        times, time_label = _time_axis(_steps_to_us(np.asarray(times), timestep))
    if mean is not None:
        mean = np.asarray(mean)
    if std is not None:
        std = np.asarray(std)
    return times, mean, std, all_data, time_label


def plot_ensemble_observables(
    stats: Dict,
    structural: Optional[Dict] = None,
    config: Optional[Dict] = None,
    show_individual: bool = False,
    individual_alpha: float = 0.3,
    figsize: Tuple[float, float] = (18, 18),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot 4×3 grid of ensemble observables with error bands.
    
    This is the main plotting function for ensemble results. It works with
    data from both local runs (via EnsembleSimulation) and cluster runs
    (via analyze_ensemble.py).
    
    Parameters
    ----------
    stats : dict
        Statistics dictionary containing times, mean/std arrays
    config : dict
        Configuration dictionary
    structural : dict, optional
        Structural data dictionary (for individual traces and distributions)
    show_individual : bool
        If True, show faint lines for each replica
    individual_alpha : float
        Transparency for individual traces
    figsize : tuple
        Figure size (default: 18x18 for 4x3 grid)
    save_path : str, optional
        If provided, save figure as SVG
    
    Returns
    -------
    fig : matplotlib.Figure
        The figure object
    
    Plot Layout (4×3)
    -----------------
    Row 1: [Energy] [Pressure] [Bonds]
    Row 2: [Particle Counts] [Number of Topologies] [Average Cluster Size]
    Row 3: [Largest Cluster] [Final Largest Distribution] [Final Fraction Distribution]
    Row 4: [Final Avg Cluster Distribution] [Fraction Bound] [Cumulative Reactions]
    """
    print("\nGenerating ensemble observable plots...")

    fig, axes = plt.subplots(4, 3, figsize=figsize)

    # Get timestep from config dict for proper time conversion
    config = config or {}
    timestep = config.get('timestep', 1e-4)  # Default if not found
    times_us, time_label = _time_axis(_steps_to_us(np.asarray(stats['times']), timestep))
    n_replicas = stats.get('n_replicas', 1)

    # ==========================================================================
    # Thermodynamics / bonds (grid cells assigned explicitly per block below)
    # ==========================================================================

    # Plot 1: Bonds
    ax = axes[0, 2]
    if 'bonds_mean' in stats:
        if _ensemble_plot_with_band(ax, times_us, stats['bonds_mean'], stats['bonds_std'],
                                    'tab:blue', n_replicas, _ensemble_all_trace(stats, structural, 'bonds'),
                                    show_individual, individual_alpha):
            ax.legend(loc='lower right', fontsize=FONTSIZE_LEGEND)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Number of Bonds", fontsize=FONTSIZE_LABEL)
    ax.set_title("Number of Bonds", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Energy
    ax = axes[0, 0]
    if 'energy_mean' in stats:
        if _ensemble_plot_with_band(ax, times_us, stats['energy_mean'], stats['energy_std'],
                                    'tab:red', n_replicas, _ensemble_all_trace(stats, structural, 'energy'),
                                    show_individual, individual_alpha):
            ax.legend(loc='upper right', fontsize=FONTSIZE_LEGEND)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Energy (kJ/mol)", fontsize=FONTSIZE_LABEL)
    ax.set_title("Potential Energy", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Pressure
    ax = axes[0, 1]
    if 'pressure_mean' in stats:
        if _ensemble_plot_with_band(ax, times_us, stats['pressure_mean'], stats['pressure_std'],
                                    'tab:green', n_replicas, _ensemble_all_trace(stats, structural, 'pressure'),
                                    show_individual, individual_alpha):
            ax.legend(loc='upper right', fontsize=FONTSIZE_LEGEND)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Pressure (kJ/mol/nm³)", fontsize=FONTSIZE_LABEL)
    ax.set_title("Pressure", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # ==========================================================================
    # Cluster-count / largest-cluster / fraction-bound kinetics
    # ==========================================================================

    # Plot 4: Cluster Count
    ax = axes[1, 1]
    if 'n_clusters_mean' in stats:
        if _ensemble_plot_with_band(ax, times_us, stats['n_clusters_mean'], stats['n_clusters_std'],
                                    'tab:purple', n_replicas, _ensemble_all_trace(stats, structural, 'n_clusters'),
                                    show_individual, individual_alpha):
            ax.legend(loc='upper right', fontsize=FONTSIZE_LEGEND)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Number of Individual Topologies", fontsize=FONTSIZE_LABEL)
    ax.set_title("Number of Individual Topologies", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Largest Cluster
    ax = axes[2, 0]
    if 'largest_cluster_mean' in stats:
        if _ensemble_plot_with_band(ax, times_us, stats['largest_cluster_mean'], stats['largest_cluster_std'],
                                    'tab:orange', n_replicas, _ensemble_all_trace(stats, structural, 'largest_cluster'),
                                    show_individual, individual_alpha):
            ax.legend(loc='lower right', fontsize=FONTSIZE_LEGEND)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Cluster Size (particles)", fontsize=FONTSIZE_LABEL)
    ax.set_title("Largest Cluster Size", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Fraction Bound
    ax = axes[3, 1]
    if 'fraction_bound_mean' in stats:
        if _ensemble_plot_with_band(ax, times_us, stats['fraction_bound_mean'], stats['fraction_bound_std'],
                                    'tab:cyan', n_replicas, _ensemble_all_trace(stats, structural, 'fraction_bound'),
                                    show_individual, individual_alpha):
            ax.legend(loc='lower right', fontsize=FONTSIZE_LEGEND)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Fraction Bound", fontsize=FONTSIZE_LABEL)
    ax.set_ylim([0, 1])
    ax.set_title("Fraction Bound", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # ==========================================================================
    # Particle counts / cumulative reactions / average cluster size
    # ==========================================================================

    # Plot 7: Particle Counts
    ax = axes[1, 0]
    particle_colors = {'qt': 'blue', 'ft': 'red', 'qtc': 'darkblue', 'ftc': 'darkred'}
    particle_labels = {'qt': 'Qt (free)', 'ft': 'Ft (free)', 'qtc': 'QtC', 'ftc': 'FtC'}
    has_particle_data = False
    for ptype in ['qt', 'ft', 'qtc', 'ftc']:
        key_mean = f'{ptype}_count_mean'
        key_std = f'{ptype}_count_std'
        if key_mean in stats:
            mean = np.asarray(stats[key_mean])
            std = np.asarray(stats[key_std])
            ax.plot(times_us, mean, color=particle_colors[ptype],
                   linewidth=2, label=particle_labels[ptype])
            ax.fill_between(times_us, mean - std, mean + std,
                           color=particle_colors[ptype], alpha=0.2)
            has_particle_data = True
    if has_particle_data:
        ax.legend(loc='upper right', fontsize=FONTSIZE_LEGEND)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Count", fontsize=FONTSIZE_LABEL)
    ax.set_title("Particle Counts", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 8: Cumulative Reactions
    ax = axes[3, 2]
    if 'cumulative_reactions_mean' in stats:
        if _ensemble_plot_with_band(ax, times_us, stats['cumulative_reactions_mean'], 
                                    stats['cumulative_reactions_std'],
                                    'tab:brown', n_replicas, _ensemble_all_trace(stats, structural, 'cumulative_reactions'),
                                    show_individual, individual_alpha):
            ax.legend(loc='lower right', fontsize=FONTSIZE_LEGEND)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Cumulative Reactions", fontsize=FONTSIZE_LABEL)
    ax.set_title("Cumulative Reactions", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 9: Average Cluster Size
    ax = axes[1, 2]
    if 'avg_cluster_mean' in stats:
        if _ensemble_plot_with_band(ax, times_us, stats['avg_cluster_mean'], stats['avg_cluster_std'],
                                    'tab:olive', n_replicas, _ensemble_all_trace(stats, structural, 'avg_cluster'),
                                    show_individual, individual_alpha):
            ax.legend(loc='lower right', fontsize=FONTSIZE_LEGEND)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Average Size (particles)", fontsize=FONTSIZE_LABEL)
    ax.set_title("Average Cluster Size", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # ==========================================================================
    # Final-frame distributions
    # ==========================================================================

    # Plot 10: Final largest cluster distribution
    ax = axes[2, 1]
    final_largest = None
    if structural is not None and 'final_largest_values' in structural:
        final_largest = structural['final_largest_values']
    elif 'largest_cluster_all' in stats:
        final_largest = np.asarray(stats['largest_cluster_all'])[:, -1]
    
    if final_largest is not None and len(final_largest) > 0:
        ax.hist(final_largest, bins=min(10, len(final_largest)), color='tab:orange',
               edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(final_largest), color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {np.mean(final_largest):.1f}')
        ax.legend(loc='upper right', fontsize=FONTSIZE_LEGEND)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel("Largest Cluster Size", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Count", fontsize=FONTSIZE_LABEL)
    ax.set_title("Final Largest Cluster Distribution", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 11: Final fraction bound distribution
    ax = axes[2, 2]
    final_fraction = None
    if structural is not None and 'final_fraction_bound_values' in structural:
        final_fraction = structural['final_fraction_bound_values']
    elif 'fraction_bound_all' in stats:
        final_fraction = np.asarray(stats['fraction_bound_all'])[:, -1]
    
    if final_fraction is not None and len(final_fraction) > 0:
        ax.hist(final_fraction, bins=min(10, len(final_fraction)), color='tab:cyan',
               edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(final_fraction), color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {np.mean(final_fraction):.3f}')
        ax.legend(loc='upper left', fontsize=FONTSIZE_LEGEND)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel("Fraction Bound", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Count", fontsize=FONTSIZE_LABEL)
    ax.set_xlim([0, 1])
    ax.set_title("Final Fraction Bound Distribution", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 12: Final average cluster size distribution
    ax = axes[3, 0]
    final_avg = None
    if structural is not None and 'final_avg_cluster_values' in structural:
        final_avg = structural['final_avg_cluster_values']
    elif 'avg_cluster_all' in stats:
        final_avg = np.asarray(stats['avg_cluster_all'])[:, -1]
    
    if final_avg is not None and len(final_avg) > 0:
        ax.hist(final_avg, bins=min(10, len(final_avg)), color='tab:olive',
               edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(final_avg), color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {np.mean(final_avg):.1f}')
        ax.legend(loc='upper right', fontsize=FONTSIZE_LEGEND)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel("Average Cluster Size", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Count", fontsize=FONTSIZE_LABEL)
    ax.set_title("Final Avg Cluster Distribution", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
        print(f"✓ Saved plot to {save_path}")
    
    plt.show()
    
    return fig



def plot_ensemble_structural(
    stats: Dict,
    structural: Dict,
    config: Dict,
    show_individual: bool = False,
    individual_alpha: float = 0.3,
    figsize: Tuple[float, float] = (18, 16),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot 3×3 grid of structural ensemble analysis with error bands.
    
    Requires pre-computed structural statistics (morphology, spatial, contacts,
    composition). For local runs, call EnsembleSimulation.compute_structural_statistics()
    first. For cluster runs, this data is generated by analyze_ensemble.py.
    
    Parameters
    ----------
    stats : dict
        Basic statistics dictionary
    structural : dict
        Structural data dictionary containing morphology, spatial, contacts, composition
    config : dict
        Configuration dictionary
    show_individual : bool
        If True, show faint lines for each replica
    individual_alpha : float
        Transparency for individual traces
    figsize : tuple
        Figure size (default: 18x16 for 3x3 grid)
    save_path : str, optional
        If provided, save figure as SVG
    
    Returns
    -------
    fig : matplotlib.Figure
        The figure object
    
    Plot Layout (3×3, last cell blank)
    ----------------------------------
    Row 1: [Mean Rg] [Coordination (Qt & Ft)] [NN Distance]
    Row 2: [Final Rg Distribution] [Final Coordination Distribution] [Composition Distribution]
    Row 3: [Mean Composition Over Time] [Composition vs Size] [blank]
    """
    print("\nGenerating structural ensemble plots...")
    
    if structural is None or len(structural) == 0:
        raise ValueError(
            "Structural data is required for this plot. "
            "For local runs: call ensemble.compute_structural_statistics() first. "
            "For cluster runs: ensure analyze_ensemble.py was run with structural analysis."
        )
    
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    n_replicas = stats.get('n_replicas', 1)
    time_label = "Time (µs)"
    
    # Get timestep from config dict for proper time conversion
    config = config or {}
    timestep = config.get('timestep', 1e-4)  # Default if not found

    # ==========================================================================
    # Row 1: Morphology and Coordination
    # ==========================================================================
    
    # Plot 1: Mean Rg
    ax = axes[0, 0]
    times, mean, std, all_data, time_label = _ensemble_struct_ts(structural, timestep, 
        'morphology_times', 'mean_rg_mean', 'mean_rg_std', 'mean_rg_all')
    if times is not None and mean is not None:
        if _ensemble_plot_with_band(ax, times, mean, std, 'tab:blue', n_replicas,
                                    all_data, show_individual, individual_alpha):
            ax.legend(loc='lower right', fontsize=FONTSIZE_LEGEND)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Mean Rg (nm)", fontsize=FONTSIZE_LABEL)
    ax.set_title("Mean Radius of Gyration", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Coordination Number (Qt & Ft fused into one axes)
    ax = axes[0, 1]
    t_qt, m_qt, s_qt, all_qt, time_label = _ensemble_struct_ts(structural, timestep, 
        'contacts_times', 'mean_coord_qt_mean', 'mean_coord_qt_std', 'mean_coord_qt_all')
    t_ft, m_ft, s_ft, all_ft, _ = _ensemble_struct_ts(structural, timestep, 
        'contacts_times', 'mean_coord_ft_mean', 'mean_coord_ft_std', 'mean_coord_ft_all')
    coord_plotted = False
    if t_qt is not None and m_qt is not None:
        coord_plotted |= _ensemble_plot_with_band(
            ax, t_qt, m_qt, s_qt, COORD_COLOR_QT, n_replicas,
            all_qt, show_individual, individual_alpha, label='Qt')
    if t_ft is not None and m_ft is not None:
        coord_plotted |= _ensemble_plot_with_band(
            ax, t_ft, m_ft, s_ft, COORD_COLOR_FT, n_replicas,
            all_ft, show_individual, individual_alpha, label='Ft')
    if coord_plotted:
        ax.legend(loc='lower right', fontsize=FONTSIZE_LEGEND)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Mean Coordination", fontsize=FONTSIZE_LABEL)
    ax.set_title("Coordination Number", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # ==========================================================================
    # Row 2: Spatial and Distributions
    # ==========================================================================

    # Plot 3: NN Distance
    ax = axes[0, 2]
    times, mean, std, all_data, time_label = _ensemble_struct_ts(structural, timestep, 
        'spatial_times', 'mean_nn_dist_mean', 'mean_nn_dist_std', 'mean_nn_dist_all')
    if times is not None and mean is not None:
        if _ensemble_plot_with_band(ax, times, mean, std, 'tab:purple', n_replicas,
                                    all_data, show_individual, individual_alpha):
            ax.legend(loc='upper right', fontsize=FONTSIZE_LEGEND)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Inter-Cluster NN Dist (nm)", fontsize=FONTSIZE_LABEL)
    ax.set_title("Inter-Cluster NN Distance", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Final Rg distribution
    ax = axes[1, 0]
    if 'final_rg_values' in structural:
        final_rg = structural['final_rg_values']
        if len(final_rg) > 0:
            ax.hist(final_rg, bins=min(10, len(final_rg)), color='tab:cyan',
                   edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(final_rg), color='red', linestyle='--',
                      linewidth=2, label=f'Mean: {np.mean(final_rg):.2f}')
            ax.legend(loc='upper right', fontsize=FONTSIZE_LEGEND)
        else:
            _ensemble_show_no_data(ax)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel("Mean Rg (nm)", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Count", fontsize=FONTSIZE_LABEL)
    ax.set_title("Final Mean Rg Distribution", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Final mean coordination, one value per replica (NOT the per-particle
    # coordination distribution — that is the ensemble panel's row-4 histogram).
    ax = axes[1, 1]
    has_coord_data = False
    if 'final_coord_qt_values' in structural:
        final_coord_qt = structural['final_coord_qt_values']
        if len(final_coord_qt) > 0:
            ax.hist(final_coord_qt, bins=min(10, len(final_coord_qt)), color=COORD_COLOR_QT,
                   edgecolor='black', alpha=0.5, label='Qt')
            has_coord_data = True
    if 'final_coord_ft_values' in structural:
        final_coord_ft = structural['final_coord_ft_values']
        if len(final_coord_ft) > 0:
            ax.hist(final_coord_ft, bins=min(10, len(final_coord_ft)), color=COORD_COLOR_FT,
                   edgecolor='black', alpha=0.5, label='Ft')
            has_coord_data = True
    if has_coord_data:
        ax.legend(loc='upper right', fontsize=FONTSIZE_LEGEND)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel("Mean Coordination", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Replicas", fontsize=FONTSIZE_LABEL)
    ax.set_title("Final Mean Coordination per Replica",
                 fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # ==========================================================================
    # Row 3: Composition Analysis
    # ==========================================================================

    # Plot 6: Composition distribution (final)
    ax = axes[1, 2]
    if 'final_composition_values' in structural:
        final_comp = structural['final_composition_values']
        if len(final_comp) > 0:
            ax.hist(final_comp, bins=20, range=(0, 1), color='purple',
                   edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(final_comp), color='red', linestyle='-', linewidth=2,
                      label=f'Mean={np.mean(final_comp):.2f}')
            ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Equal mix')
            ax.legend(loc='upper right', fontsize=FONTSIZE_LEGEND)
        else:
            _ensemble_show_no_data(ax)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel("Qt Fraction", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Count", fontsize=FONTSIZE_LABEL)
    ax.set_xlim([0, 1])
    ax.set_title("Composition Distribution (Final)", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 7: Mean composition over time
    ax = axes[2, 0]
    times, mean, std, all_data, time_label = _ensemble_struct_ts(structural, timestep, 
        'composition_times', 'mean_composition_mean', 'mean_composition_std', 'mean_composition_all')
    if times is not None and mean is not None:
        if _ensemble_plot_with_band(ax, times, mean, std, 'purple', n_replicas,
                                    all_data, show_individual, individual_alpha):
            ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Equal mix')
            ax.legend(loc='best', fontsize=FONTSIZE_LEGEND)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Mean Qt Fraction", fontsize=FONTSIZE_LABEL)
    ax.set_ylim([0, 1])
    ax.set_title("Mean Cluster Composition", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 8: Composition vs cluster size (scatter from final frame)
    ax = axes[2, 1]
    if 'composition_vs_size_fractions' in structural and 'composition_vs_size_sizes' in structural:
        fractions = structural['composition_vs_size_fractions']
        sizes = structural['composition_vs_size_sizes']
        if len(fractions) > 0 and len(sizes) > 0:
            ax.scatter(sizes, fractions, alpha=0.5, s=20, c='purple')
            ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
            # Add binned means (only if there's variation in sizes)
            if len(sizes) > 10 and np.max(sizes) > np.min(sizes):
                size_bins = np.linspace(np.min(sizes), np.max(sizes), 10)
                bin_means = []
                bin_centers = []
                for i in range(len(size_bins) - 1):
                    mask = (sizes >= size_bins[i]) & (sizes < size_bins[i+1])
                    if np.sum(mask) > 0:
                        bin_means.append(np.mean(fractions[mask]))
                        bin_centers.append((size_bins[i] + size_bins[i+1]) / 2)
                if bin_means:
                    ax.plot(bin_centers, bin_means, 'ro-', linewidth=2, markersize=6, label='Bin mean')
                    ax.legend(loc='best', fontsize=FONTSIZE_LEGEND)
        else:
            _ensemble_show_no_data(ax)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel("Cluster Size (particles)", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Qt Fraction", fontsize=FONTSIZE_LABEL)
    ax.set_ylim([0, 1])
    ax.set_title("Composition vs Cluster Size", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Coordination fused into one axes freed the last grid cell
    axes[2, 2].axis('off')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
        print(f"✓ Saved plot to {save_path}")

    plt.show()

    return fig



def _generate_ensemble_title(config: Dict, n_replicas: int = None) -> str:
    """
    Generate a descriptive title for ensemble/simulation plots.
    
    Format: "{n_qt}Qt + {n_ft}Ft ({potential_type}, N={n_replicas} replicas)"
    or for single simulations: "{n_qt}Qt + {n_ft}Ft ({potential_type})"
    
    Parameters
    ----------
    config : dict
        Configuration dictionary (from config.to_dict() or ensemble config JSON)
    n_replicas : int, optional
        Number of replicas. If None, omits replica count from title.
    
    Returns
    -------
    str
        Descriptive title string
    """
    n_qt = config.get('n_qt', '?')
    n_ft = config.get('n_ft', '?')
    
    # Get potential type: top-level (current) with fallback to lj (pre-move configs)
    lj = config.get('lj', {})
    potential = config.get('potential_type', lj.get('potential_type', '?'))
    
    if n_replicas is not None:
        return f"{n_qt}Qt + {n_ft}Ft ({potential}, N={n_replicas} replicas)"
    else:
        return f"{n_qt}Qt + {n_ft}Ft ({potential})"


def plot_ensemble_size_categories(
    stats: Dict,
    structural: Dict,
    config: Dict,
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot "Particles by Size Category" stacked area chart for a single ensemble.
    
    Shows the mean fraction of particles in each adaptive size category
    (monomers, small, medium, large, very large) over time, averaged
    across replicas. No std bands for clarity.
    
    Parameters
    ----------
    stats : dict
        Statistics dictionary (from JSON or EnsembleSimulation)
    structural : dict
        Structural data dictionary (from NPZ or EnsembleSimulation)
    config : dict
        Configuration dictionary
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib.Figure
        The figure object
    """
    # Check for size fraction data
    if 'size_fractions_times' not in structural:
        print("Warning: No size fraction data available in structural data.")
        print("  Re-run ensemble analysis to compute size fractions.")
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.text(0.5, 0.5, "Size fraction data not available\nRe-run ensemble analysis",
               ha='center', va='center', transform=ax.transAxes,
               fontsize=FONTSIZE_TITLE, color='gray')
        plt.show()
        return fig
    
    timestep = config.get('timestep', 1e-4)
    times_us, time_label = _time_axis(_steps_to_us(np.asarray(structural['size_fractions_times']), timestep))
    n_replicas = stats.get('n_replicas', '?')
    
    # Get category names
    if 'size_fractions_category_names' in structural:
        category_names = list(structural['size_fractions_category_names'])
    else:
        print("Warning: Category names not found in structural data.")
        return None
    
    # Collect mean fraction arrays in order
    mean_fractions = []
    for cat_name in category_names:
        safe_key = _size_category_key(cat_name)
        mean_key = f'size_frac_{safe_key}_mean'
        if mean_key in structural:
            mean_fractions.append(np.asarray(structural[mean_key]))
        else:
            # Fallback: zeros
            mean_fractions.append(np.zeros(len(times_us)))
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple"]
    
    ax.stackplot(times_us, *mean_fractions,
                labels=category_names, colors=colors[:len(category_names)], alpha=0.8)
    
    # Generate descriptive title
    desc = _generate_ensemble_title(config, n_replicas=n_replicas)
    ax.set_title(f"Particles by Size Category — {desc}",
                fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Fraction", fontsize=FONTSIZE_LABEL)
    ax.set_ylim([0, 1])
    ax.legend(loc="upper right", fontsize=FONTSIZE_LEGEND)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
        print(f"✓ Saved plot to {save_path}")
    
    plt.show()
    
    return fig


def plot_comparison_size_categories(
    comparison: dict,
    figsize: Tuple[float, float] = (10, 6),
    save_dir: Optional[str] = None,
) -> List[plt.Figure]:
    """
    Plot "Particles by Size Category" for each ensemble in a comparison.
    
    Generates one individual figure per ensemble, each showing the mean
    stacked area chart of particle fractions by adaptive size category.
    
    Parameters
    ----------
    comparison : dict
        Comparison data structure (from compare_ensembles or load_comparison_data)
    figsize : tuple
        Figure size for each individual plot
    save_dir : str, optional
        Directory to save figures. Files are named
        "size_categories_{label}.svg" for each ensemble.
    
    Returns
    -------
    list of matplotlib.Figure
        One figure per ensemble
    """
    figures = []
    labels = comparison['labels']
    
    for label in labels:
        ens = comparison['ensembles'][label]
        config = ens['config']
        structural = ens.get('structural', {})
        stats = ens.get('stats', {})
        n_replicas = ens.get('n_replicas', '?')
        
        # Check for size fraction data
        if 'size_fractions_times' not in structural:
            print(f"Warning: No size fraction data for '{label}', skipping.")
            continue
        
        timestep = ens.get('timestep', config.get('timestep', 1e-4))
        times_us, time_label = _time_axis(_steps_to_us(np.asarray(structural['size_fractions_times']), timestep))

        # Get category names
        if 'size_fractions_category_names' in structural:
            category_names = list(structural['size_fractions_category_names'])
        else:
            print(f"Warning: Category names not found for '{label}', skipping.")
            continue
        
        # Collect mean fraction arrays
        mean_fractions = []
        for cat_name in category_names:
            safe_key = _size_category_key(cat_name)
            mean_key = f'size_frac_{safe_key}_mean'
            if mean_key in structural:
                mean_fractions.append(np.asarray(structural[mean_key]))
            else:
                mean_fractions.append(np.zeros(len(times_us)))
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        colors = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple"]
        
        ax.stackplot(times_us, *mean_fractions,
                    labels=category_names, colors=colors[:len(category_names)], alpha=0.8)
        
        # Generate descriptive title
        desc = _generate_ensemble_title(config, n_replicas=n_replicas)
        ax.set_title(f"Particles by Size Category — {desc}",
                    fontsize=FONTSIZE_TITLE, fontweight='bold')
        ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
        ax.set_ylabel("Fraction", fontsize=FONTSIZE_LABEL)
        ax.set_ylim([0, 1])
        ax.legend(loc="upper right", fontsize=FONTSIZE_LEGEND)
        ax.grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            safe_label = label.replace(" ", "_").replace("/", "_")
            save_path = os.path.join(save_dir, f"size_categories_{safe_label}.svg")
            fig.savefig(save_path, format='svg', bbox_inches='tight', dpi=300)
            print(f"✓ Saved: {save_path}")
        
        plt.show()
        figures.append(fig)
    
    return figures


# =============================================================================
# ENSEMBLE COMPARISON PLOTS
# =============================================================================

COMPARISON_COLORS = plt.cm.tab10.colors


def _get_show_bands_default(n_ensembles: int, show_bands: bool = None) -> bool:
    """Determine whether to show error bands based on number of ensembles."""
    if show_bands is not None:
        return show_bands
    return n_ensembles <= 3


def _total_particles(ens: dict):
    """Total particle count N for one ensemble (constant), for ÷N normalization.

    Single source of truth for the normalization denominator (prefers the recorded
    ``stats['total_count_mean']``, falls back to ``config['n_qt']+['n_ft']``). Returns
    None if neither is available. Used by both the standalone comparison plots and the
    comparison panel so their normalized curves agree.
    """
    tc = ens['stats'].get('total_count_mean')
    if tc is not None and len(np.atleast_1d(tc)):
        n = float(np.asarray(tc).ravel()[0])
        if n > 0:
            return n
    cfg = ens.get('config', {}) or {}
    if cfg.get('n_qt') is not None and cfg.get('n_ft') is not None:
        n = float(cfg['n_qt']) + float(cfg['n_ft'])
        if n > 0:
            return n
    return None


def _comparison_timeseries(ax, comparison: dict, stat_key: str, ylabel: str, title: str, *,
                           show_bands: bool, divide_by_N: bool = False) -> bool:
    """Overlay a basic-stats time series (``ens['stats']``) per ensemble on ``ax``.

    Draws lines/bands + labels/title/grid; the caller places the legend (so standalone and
    panel can differ). When ``divide_by_N`` is True, mean/std are divided by ``_total_particles``.
    Returns True if any data was drawn.
    """
    labels = comparison['labels']
    # One time unit for the whole (multi-ensemble) axis, from the longest series.
    max_us = max(
        (float(np.asarray(comparison['ensembles'][l]['times_us']).max())
         for l in labels if len(comparison['ensembles'][l].get('times_us', []))),
        default=0.0,
    )
    time_factor, time_unit = choose_time_unit(max_us)
    has_data = False
    for i, label in enumerate(labels):
        ens = comparison['ensembles'][label]
        mean_key = f'{stat_key}_mean'
        std_key = f'{stat_key}_std'
        if mean_key not in ens['stats']:
            continue
        mean_vals = np.asarray(ens['stats'][mean_key], dtype=float)
        std_vals = np.asarray(ens['stats'].get(std_key, np.zeros_like(mean_vals)), dtype=float)
        if divide_by_N:
            N = _total_particles(ens)
            if not N:
                continue
            mean_vals = mean_vals / N
            std_vals = std_vals / N
        color = COMPARISON_COLORS[i % len(COMPARISON_COLORS)]
        t = np.asarray(ens['times_us']) * time_factor
        ax.plot(t, mean_vals, color=color, linewidth=2, label=label)
        if show_bands and len(std_vals) == len(mean_vals):
            ax.fill_between(t, mean_vals - std_vals, mean_vals + std_vals,
                            color=color, alpha=0.2)
        has_data = True
    if not has_data:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(f"Time ({time_unit})", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL)
    ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    return has_data


def _comparison_struct_ts(ax, comparison: dict, time_key: str, mean_key: str, std_key: str,
                          ylabel: str, title: str, *, show_bands: bool,
                          legend_loc: str = 'best') -> bool:
    """Overlay a structural time series (``ens['structural']``) per ensemble on ``ax``.

    Steps→µs via each ensemble's timestep; guards against mismatched array lengths. Places an
    inside legend at ``legend_loc``. Returns True if any data was drawn.
    """
    labels = comparison['labels']
    # One time unit for the whole (multi-ensemble) axis, from the longest series.
    max_us = 0.0
    for label in labels:
        s = comparison['ensembles'][label].get('structural', {})
        if time_key in s:
            arr = _steps_to_us(np.asarray(s[time_key]),
                               comparison['ensembles'][label].get('timestep', 0.001))
            if arr.size:
                max_us = max(max_us, float(arr.max()))
    time_factor, time_unit = choose_time_unit(max_us)
    has_data = False
    for i, label in enumerate(labels):
        ens = comparison['ensembles'][label]
        structural = ens.get('structural', {})
        if time_key not in structural or mean_key not in structural:
            continue
        times_us = _steps_to_us(np.asarray(structural[time_key]), ens.get('timestep', 0.001)) * time_factor
        mean_vals = np.asarray(structural[mean_key])
        std_vals = np.asarray(structural.get(std_key, np.zeros_like(mean_vals)))
        min_len = min(len(times_us), len(mean_vals))
        times_us, mean_vals, std_vals = times_us[:min_len], mean_vals[:min_len], std_vals[:min_len]
        color = COMPARISON_COLORS[i % len(COMPARISON_COLORS)]
        ax.plot(times_us, mean_vals, color=color, linewidth=2, label=label)
        if show_bands:
            ax.fill_between(times_us, mean_vals - std_vals, mean_vals + std_vals,
                            color=color, alpha=0.2)
        has_data = True
    if not has_data:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(f"Time ({time_unit})", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL)
    ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    if has_data:
        ax.legend(loc=legend_loc, fontsize=FONTSIZE_LEGEND)
    return has_data


def _comparison_coord_fused(ax, comparison: dict, *, show_bands: bool,
                            legend_loc: str = 'lower right') -> bool:
    """Overlay Qt (solid) and Ft (dashed) coordination per ensemble in one axes.

    Owns its legend (ensemble colours + a Qt/Ft linestyle key). Returns True if data drawn.
    """
    labels = comparison['labels']
    # One time unit for the whole (multi-ensemble) axis, from the longest series.
    max_us = 0.0
    for label in labels:
        s = comparison['ensembles'][label].get('structural', {})
        if 'contacts_times' in s:
            arr = _steps_to_us(np.asarray(s['contacts_times']),
                               comparison['ensembles'][label].get('timestep', 0.001))
            if arr.size:
                max_us = max(max_us, float(arr.max()))
    time_factor, time_unit = choose_time_unit(max_us)
    has_data = False
    for i, label in enumerate(labels):
        ens = comparison['ensembles'][label]
        structural = ens.get('structural', {})
        if 'contacts_times' not in structural:
            continue
        times_us = _steps_to_us(np.asarray(structural['contacts_times']), ens.get('timestep', 0.001)) * time_factor
        color = COMPARISON_COLORS[i % len(COMPARISON_COLORS)]
        for mean_key, std_key, ls in (
            ('mean_coord_qt_mean', 'mean_coord_qt_std', '-'),
            ('mean_coord_ft_mean', 'mean_coord_ft_std', '--'),
        ):
            if mean_key not in structural:
                continue
            mean_vals = np.asarray(structural[mean_key])
            std_vals = np.asarray(structural.get(std_key, np.zeros_like(mean_vals)))
            min_len = min(len(times_us), len(mean_vals))
            t, m, s = times_us[:min_len], mean_vals[:min_len], std_vals[:min_len]
            lbl = label if ls == '-' else '_nolegend_'
            ax.plot(t, m, color=color, linewidth=2, linestyle=ls, label=lbl)
            if show_bands:
                ax.fill_between(t, m - s, m + s, color=color, alpha=0.2)
            has_data = True
    if not has_data:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(f"Time ({time_unit})", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Mean Coordination", fontsize=FONTSIZE_LABEL)
    ax.set_title("Coordination Number", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONTSIZE_TICK)
    if has_data:
        ens_handles, ens_labels = ax.get_legend_handles_labels()
        style_handles = [
            Line2D([0], [0], color='black', linestyle='-', linewidth=2),
            Line2D([0], [0], color='black', linestyle='--', linewidth=2),
        ]
        ax.legend(ens_handles + style_handles, ens_labels + ['Qt', 'Ft'],
                  loc=legend_loc, fontsize=FONTSIZE_LEGEND)
    return has_data


def _plot_comparison_timeseries(
    comparison: dict,
    stat_key: str,
    ylabel: str,
    title: str,
    show_bands: bool = None,
    normalize_by_particles: bool = False,
    figsize: tuple = (10, 6),
    save_path: str = None,
) -> plt.Figure:
    """
    Generic time series comparison plot.
    
    Parameters
    ----------
    comparison : dict
        Comparison data structure
    stat_key : str
        Key for the statistic (e.g., 'bonds', 'n_clusters')
    ylabel : str
        Y-axis label
    title : str
        Plot title
    show_bands : bool, optional
        Show error bands. Default: True for ≤3 ensembles, False otherwise
    normalize_by_particles : bool
        Divide values by total particle count
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    show_bands = _get_show_bands_default(comparison['n_ensembles'], show_bands)
    has_data = _comparison_timeseries(ax, comparison, stat_key, ylabel, title,
                                      show_bands=show_bands, divide_by_N=normalize_by_particles)
    if has_data:
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=FONTSIZE_LEGEND)

    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_comparison_bonds(comparison: dict, show_bands: bool = None, 
                         normalize_by_particles: bool = False,
                         save_path: str = None) -> plt.Figure:
    """Plot bond count comparison across ensembles."""
    ylabel = "Bonds per Particle" if normalize_by_particles else "Number of Bonds"
    return _plot_comparison_timeseries(
        comparison, 'bonds', ylabel, "Bond Formation Comparison",
        show_bands=show_bands, normalize_by_particles=normalize_by_particles,
        save_path=save_path
    )


def plot_comparison_topologies(comparison: dict, show_bands: bool = None,
                               save_path: str = None) -> plt.Figure:
    """Plot topology count comparison across ensembles."""
    return _plot_comparison_timeseries(
        comparison, 'n_clusters', "Number of Individual Topologies", 
        "Individual Topologies Comparison",
        show_bands=show_bands, save_path=save_path
    )


def plot_comparison_largest_cluster(comparison: dict, show_bands: bool = None,
                                    normalize_by_particles: bool = False,
                                    save_path: str = None) -> plt.Figure:
    """Plot largest cluster size comparison across ensembles."""
    ylabel = "Fraction of Particles" if normalize_by_particles else "Largest Cluster Size (particles)"
    title = "Largest Cluster Fraction Comparison" if normalize_by_particles else "Largest Cluster Size Comparison"
    return _plot_comparison_timeseries(
        comparison, 'largest_cluster', ylabel, title,
        show_bands=show_bands, normalize_by_particles=normalize_by_particles,
        save_path=save_path
    )


def plot_comparison_avg_cluster(comparison: dict, show_bands: bool = None,
                                save_path: str = None) -> plt.Figure:
    """Plot average cluster size comparison across ensembles."""
    return _plot_comparison_timeseries(
        comparison, 'avg_cluster', "Average Cluster Size (particles)", 
        "Average Cluster Size Comparison",
        show_bands=show_bands, save_path=save_path
    )


def plot_comparison_energy(comparison: dict, show_bands: bool = None,
                          save_path: str = None) -> plt.Figure:
    """Plot energy comparison across ensembles."""
    return _plot_comparison_timeseries(
        comparison, 'energy', "Energy (kJ/mol)", "Energy Comparison",
        show_bands=show_bands, save_path=save_path
    )


def plot_comparison_pressure(comparison: dict, show_bands: bool = None,
                            save_path: str = None) -> plt.Figure:
    """Plot pressure comparison across ensembles."""
    return _plot_comparison_timeseries(
        comparison, 'pressure', "Pressure (kJ/(mol·nm³))", "Pressure Comparison",
        show_bands=show_bands, save_path=save_path
    )


def plot_comparison_fraction_bound(comparison: dict, show_bands: bool = None,
                                   save_path: str = None) -> plt.Figure:
    """Plot fraction bound comparison across ensembles (already normalized)."""
    return _plot_comparison_timeseries(
        comparison, 'fraction_bound', "Fraction Bound", "Fraction Bound Comparison",
        show_bands=show_bands, normalize_by_particles=False,
        save_path=save_path
    )


def plot_comparison_cumulative_reactions(comparison: dict, show_bands: bool = None,
                                         normalize_by_particles: bool = False,
                                         save_path: str = None) -> plt.Figure:
    """Plot cumulative reactions comparison across ensembles."""
    ylabel = "Reactions per Particle" if normalize_by_particles else "Cumulative Reactions"
    return _plot_comparison_timeseries(
        comparison, 'cumulative_reactions', ylabel, "Cumulative Reactions Comparison",
        show_bands=show_bands, normalize_by_particles=normalize_by_particles,
        save_path=save_path
    )


def _overlay_individual_points(ax, x_positions, all_replica_values, colors, width, rng_seed=42):
    """
    Overlay individual replica data points as jittered dots on bar charts.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    x_positions : ndarray
        Bar center x positions
    all_replica_values : list of array-like
        Per-ensemble list of individual replica final values
    colors : list
        Colors for each ensemble (darkened for dots)
    width : float
        Bar width (used to scale jitter)
    rng_seed : int
        Random seed for reproducible jitter
    """
    rng = np.random.default_rng(rng_seed)
    for i, replica_vals in enumerate(all_replica_values):
        if replica_vals is None or len(replica_vals) == 0:
            continue
        replica_vals = np.asarray(replica_vals)
        # Jitter x positions within ±30% of bar half-width
        jitter = rng.uniform(-0.3 * width / 2, 0.3 * width / 2, size=len(replica_vals))
        ax.scatter(
            x_positions[i] + jitter, replica_vals,
            color='black', alpha=0.6, s=20, zorder=5,
            edgecolors='white', linewidths=0.5,
        )


def plot_comparison_final_state(
    comparison: dict,
    show_individual_points: bool = False,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot bar charts comparing final state values across ensembles.
    
    Creates a 2x3 grid of bar charts for:
    - Final Bonds
    - Final Topology Count
    - Final Largest Cluster (fraction)
    - Final Fraction Bound
    - Final Average Cluster Size
    - Half-time (if available)
    
    Parameters
    ----------
    comparison : dict
        Comparison data structure
    show_individual_points : bool
        If True, overlay individual replica values as jittered dots on bars
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    matplotlib.Figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    labels = comparison['labels']
    x = np.arange(len(labels))
    width = 0.6
    
    def plot_bar(ax, values, errors, title, ylabel, is_available,
                 all_replica_values=None):
        """Helper to plot a single bar chart."""
        if not is_available:
            ax.text(0.5, 0.5, "Data not available", ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight='bold')
            return
        
        bar_color = 'steelblue'
        bars = ax.bar(x, values, width, yerr=errors, capsize=5, color=bar_color, 
                     edgecolor='black', linewidth=1)
        
        # Overlay individual replica points
        if show_individual_points and all_replica_values is not None:
            dot_colors = [bar_color] * len(labels)
            _overlay_individual_points(ax, x, all_replica_values, dot_colors, width)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=FONTSIZE_TICK)
        ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL)
        ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Extract final values for each metric
    metrics = [
        ('bonds', 'Final Bonds', 'Number of Bonds', False),
        ('n_clusters', 'Final Topology Count', 'Count', False),
        ('largest_cluster', 'Final Largest Cluster', 'Fraction of Particles', True),  # normalize
        ('fraction_bound', 'Final Fraction Bound', 'Fraction', False),  # already normalized
        ('avg_cluster', 'Final Avg Cluster Size', 'Particles', False),
    ]
    
    for idx, (key, title, ylabel, normalize) in enumerate(metrics):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        
        values = []
        errors = []
        all_replica_values = []
        available = True
        
        for label in labels:
            ens = comparison['ensembles'][label]
            mean_key = f'{key}_mean'
            std_key = f'{key}_std'
            all_key = f'{key}_all'
            
            if mean_key in ens['stats'] and len(ens['stats'][mean_key]) > 0:
                final_mean = ens['stats'][mean_key][-1]
                final_std = ens['stats'].get(std_key, np.zeros_like(ens['stats'][mean_key]))[-1]
                
                # Get per-replica final values
                replica_finals = None
                if all_key in ens['stats']:
                    all_data = np.asarray(ens['stats'][all_key])
                    if all_data.ndim == 2 and all_data.shape[1] > 0:
                        replica_finals = all_data[:, -1]
                
                if normalize:
                    n_particles = ens['config'].get('n_qt', 0) + ens['config'].get('n_ft', 0)
                    if n_particles > 0:
                        final_mean = final_mean / n_particles
                        final_std = final_std / n_particles
                        if replica_finals is not None:
                            replica_finals = replica_finals / n_particles
                
                values.append(final_mean)
                errors.append(final_std)
                all_replica_values.append(replica_finals)
            else:
                available = False
                break
        
        plot_bar(ax, values, errors, title, ylabel, available,
                 all_replica_values if show_individual_points else None)
    
    # Half-time plot (special handling for N/A values)
    ax = axes[1, 2]
    half_times = []
    half_time_errors = []
    half_time_available = []
    half_time_replica_values = []
    
    for label in labels:
        ens = comparison['ensembles'][label]
        summary = ens['stats'].get('summary', {})
        
        if 'half_time_mean' in summary and summary['half_time_mean'] is not None:
            # Convert to µs
            ht_mean = summary['half_time_mean'] / 1000
            ht_std = summary.get('half_time_std', 0) / 1000
            half_times.append(ht_mean)
            half_time_errors.append(ht_std)
            half_time_available.append(True)
            
            # Per-replica half-times (convert ns to µs)
            ht_all = summary.get('half_time_all', None)
            if ht_all is not None:
                half_time_replica_values.append(np.array(ht_all) / 1000)
            else:
                half_time_replica_values.append(None)
        else:
            half_times.append(0)
            half_time_errors.append(0)
            half_time_available.append(False)
            half_time_replica_values.append(None)
    
    if any(half_time_available):
        # Half-times are in µs; scale to one adaptive unit for the bar axis.
        ht_factor, ht_unit = choose_time_unit(max(half_times) if half_times else 0.0)
        half_times = [h * ht_factor for h in half_times]
        half_time_errors = [e * ht_factor for e in half_time_errors]
        half_time_replica_values = [
            (np.asarray(v) * ht_factor if v is not None else None)
            for v in half_time_replica_values
        ]
        colors = ['steelblue' if avail else 'lightgray'
                 for avail in half_time_available]
        bars = ax.bar(x, half_times, width, yerr=half_time_errors, capsize=5,
                     color=colors, edgecolor='black', linewidth=1)
        
        # Overlay individual replica points
        if show_individual_points:
            _overlay_individual_points(ax, x, half_time_replica_values, colors, width)
        
        # Add "N/A" text for unavailable half-times
        for i, (bar, avail) in enumerate(zip(bars, half_time_available)):
            if not avail:
                ax.text(bar.get_x() + bar.get_width()/2, 0.1, 'N/A', 
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=FONTSIZE_TICK)
        ax.set_ylabel(f"Time ({ht_unit})", fontsize=FONTSIZE_LABEL)
        ax.set_title("Half-time (50% bound)", fontsize=FONTSIZE_TITLE, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, "Half-time data not available", ha='center', va='center',
               transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title("Half-time (50% bound)", fontsize=FONTSIZE_TITLE, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_comparison_structural(
    comparison: dict,
    show_bands: bool = None,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot structural analysis comparison across ensembles.
    
    Creates a 2×3 grid of time-series comparing structural metrics:
    - Row 1: Coordination (Qt & Ft), Mean Cluster Composition, Inter-Cluster NN Distance
    - Row 2: Intra-Cluster NN Distance, Mean Rg, Normalized Rg (Compactness)
    
    Parameters
    ----------
    comparison : dict
        Comparison data structure
    show_bands : bool, optional
        Show error bands. Default: True for ≤3 ensembles
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    matplotlib.Figure
    """
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    show_bands = _get_show_bands_default(comparison['n_ensembles'], show_bands)

    # ======================================================================
    # Row 1: Coordination (Qt & Ft), Mean Composition, Inter-Cluster NN
    # ======================================================================
    _comparison_coord_fused(fig.add_subplot(gs[0, 0]), comparison,
                            show_bands=show_bands, legend_loc='lower right')

    _comparison_struct_ts(
        fig.add_subplot(gs[0, 1]), comparison,
        'composition_times', 'mean_composition_mean', 'mean_composition_std',
        'Mean Qt Fraction', 'Mean Cluster Composition',
        show_bands=show_bands, legend_loc='lower right')

    _comparison_struct_ts(
        fig.add_subplot(gs[0, 2]), comparison,
        'spatial_times', 'mean_nn_dist_mean', 'mean_nn_dist_std',
        'Inter-Cluster NN Dist (nm)', 'Inter-Cluster NN Distance',
        show_bands=show_bands)

    # ======================================================================
    # Row 2: Intra-Cluster NN, Mean Rg, Normalized Rg
    # ======================================================================
    _comparison_struct_ts(
        fig.add_subplot(gs[1, 0]), comparison,
        'spatial_times', 'mean_intra_nn_dist_mean', 'mean_intra_nn_dist_std',
        'Intra-Cluster NN Dist (nm)', 'Intra-Cluster NN Distance',
        show_bands=show_bands)

    _comparison_struct_ts(
        fig.add_subplot(gs[1, 1]), comparison,
        'morphology_times', 'mean_rg_mean', 'mean_rg_std',
        'Mean Rg (nm)', 'Mean Radius of Gyration',
        show_bands=show_bands, legend_loc='upper left')

    _comparison_struct_ts(
        fig.add_subplot(gs[1, 2]), comparison,
        'morphology_times', 'mean_rg_normalized_mean', 'mean_rg_normalized_std',
        'Rg / Rg_sphere', 'Normalized Rg (Compactness)',
        show_bands=show_bands, legend_loc='lower right')

    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_comparison_summary(
    comparison: dict,
    show_bands: bool = None,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot multi-panel comparison summary.
    
    Creates a 3×3 grid with:
    - Row 1: Energy, Pressure, Bonds (time-series)
    - Row 2: Individual Topologies, Avg Cluster Size, Largest Cluster (time-series)
    - Row 3: Free Particles, Complexed Particles, Fraction Bound (time-series)
    
    Parameters
    ----------
    comparison : dict
        Comparison data structure
    show_bands : bool, optional
        Show error bands. Default: True for ≤3 ensembles
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    matplotlib.Figure
    """
    fig = plt.figure(figsize=(18, 17))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    n_ensembles = comparison['n_ensembles']
    show_bands = _get_show_bands_default(n_ensembles, show_bands)
    labels = comparison['labels']
    
    def plot_timeseries(ax, stat_key, ylabel, title, legend_loc='best'):
        """Thin wrapper over the shared `_comparison_timeseries` helper (inside legend)."""
        if _comparison_timeseries(ax, comparison, stat_key, ylabel, title, show_bands=show_bands):
            ax.legend(loc=legend_loc, fontsize=FONTSIZE_LEGEND)

    def plot_particle_subset(ax, particle_types, type_labels, linestyles, title):
        """Plot a subset of particle counts with two-part legend below.
        
        Parameters
        ----------
        ax : matplotlib Axes
        particle_types : list of str
            Keys like ['qt', 'ft']
        type_labels : dict
            Mapping type key to display label
        linestyles : dict
            Mapping type key to linestyle
        title : str
        """
        has_data = False
        # One time unit for the whole (multi-ensemble) axis, from the longest series.
        max_us = max(
            (float(np.asarray(comparison['ensembles'][l]['times_us']).max())
             for l in labels if len(comparison['ensembles'][l].get('times_us', []))),
            default=0.0,
        )
        tf, tunit = choose_time_unit(max_us)
        for i, label in enumerate(labels):
            ens = comparison['ensembles'][label]
            times_us = np.asarray(ens['times_us']) * tf
            color = COMPARISON_COLORS[i % len(COMPARISON_COLORS)]
            
            for ptype in particle_types:
                mean_key = f'{ptype}_count_mean'
                if mean_key in ens['stats']:
                    mean_vals = ens['stats'][mean_key]
                    ax.plot(times_us, mean_vals, color=color,
                           linestyle=linestyles[ptype], linewidth=1.5,
                           label=f'{type_labels[ptype]} ({label})')
                    has_data = True
        
        if not has_data:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                   transform=ax.transAxes, fontsize=FONTSIZE_TITLE, color='gray')
        
        ax.set_xlabel(f"Time ({tunit})", fontsize=FONTSIZE_LABEL)
        ax.set_ylabel("Count", fontsize=FONTSIZE_LABEL)
        ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=FONTSIZE_TICK)
        
        if has_data:
            # Two-part legend: ensemble colors + line styles
            # Part 1: Ensemble colors (one solid line per ensemble)
            ensemble_handles = []
            for i, label in enumerate(labels):
                color = COMPARISON_COLORS[i % len(COMPARISON_COLORS)]
                ensemble_handles.append(
                    plt.Line2D([0], [0], color=color, linewidth=2, linestyle='-', label=label))
            
            # Part 2: Line style legend (black lines showing type distinction)
            style_handles = []
            for ptype in particle_types:
                style_handles.append(
                    plt.Line2D([0], [0], color='black', linewidth=1.5,
                              linestyle=linestyles[ptype], label=type_labels[ptype]))
            
            all_handles = ensemble_handles + style_handles
            ax.legend(handles=all_handles, loc='upper center',
                     bbox_to_anchor=(0.5, -0.18),
                     fontsize=max(FONTSIZE_LEGEND - 2, 7),
                     ncol=len(all_handles),
                     frameon=True, edgecolor='gray')
    
    # ======================================================================
    # Row 1: Energy, Pressure, Bonds
    # ======================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    plot_timeseries(ax1, 'energy', 'Energy (kJ/mol)', 'Potential Energy', legend_loc='lower left')
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_timeseries(ax2, 'pressure', 'Pressure (kJ/(mol·nm³))', 'Pressure', legend_loc='upper left')
    
    ax3 = fig.add_subplot(gs[0, 2])
    plot_timeseries(ax3, 'bonds', 'Number of Bonds', 'Bonds', legend_loc='lower right')
    
    # ======================================================================
    # Row 2: Topologies, Avg Cluster Size, Largest Cluster
    # ======================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    plot_timeseries(ax4, 'n_clusters', 'Count', 'Individual Topologies')
    
    ax5 = fig.add_subplot(gs[1, 1])
    plot_timeseries(ax5, 'avg_cluster', 'Particles', 'Average Cluster Size', legend_loc='upper left')
    
    ax6 = fig.add_subplot(gs[1, 2])
    plot_timeseries(ax6, 'largest_cluster', 'Particles', 'Largest Cluster', legend_loc='upper left')
    
    # ======================================================================
    # Row 3: Free Particles, Complexed Particles, Fraction Bound
    # ======================================================================
    ax7 = fig.add_subplot(gs[2, 0])
    plot_particle_subset(
        ax7,
        particle_types=['qt', 'ft'],
        type_labels={'qt': 'Qt', 'ft': 'Ft'},
        linestyles={'qt': '-', 'ft': '--'},
        title='Free Particles')
    
    ax8 = fig.add_subplot(gs[2, 1])
    plot_particle_subset(
        ax8,
        particle_types=['qtc', 'ftc'],
        type_labels={'qtc': 'QtC', 'ftc': 'FtC'},
        linestyles={'qtc': '-', 'ftc': '--'},
        title='Complexed Particles')
    
    ax9 = fig.add_subplot(gs[2, 2])
    plot_timeseries(ax9, 'fraction_bound', 'Fraction', 'Fraction Bound')

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.08)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"✓ Saved: {save_path}")

    return fig


def _mark_phase_boundaries(ax, boundaries_us, starts_us=None, names=None):
    """Draw vertical lines at phase switches and (optionally) label each phase span."""
    for b in boundaries_us:
        ax.axvline(b, color="0.4", linestyle="--", linewidth=1.0, zorder=1)
    if starts_us is not None and names is not None:
        ylim = ax.get_ylim()
        ytext = ylim[1] - 0.04 * (ylim[1] - ylim[0])
        ends = list(starts_us[1:]) + [ax.get_xlim()[1]]
        for name, s, e in zip(names, starts_us, ends):
            ax.text(0.5 * (s + e), ytext, name, ha="center", va="top",
                    fontsize=9, color="0.3",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.7))


def plot_phased_kinetics(
    config: "SimulationConfig",
    phase_files: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (11, 9),
    data: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
):
    """Plot a stitched agglomeration<->deagglomeration cycle on one continuous time axis.

    Loads the per-phase trajectories via analysis.load_phased_observables (unless
    pre-stitched ``data`` is supplied, e.g. ensemble means) and draws bonds, fraction
    bound, and average cluster size vs µs, with dashed lines and labels at the phase
    boundaries.

    Parameters
    ----------
    config : SimulationConfig
        Phased configuration (must have config.phases set).
    phase_files : list of str, optional
        Per-phase trajectory paths; defaults to config.phase_output_files.
    save_path : str, optional
        If given, save the figure here.
    figsize : tuple
        Figure size.
    data : dict, optional
        Pre-computed stitched series (same schema as load_phased_observables). When given,
        phase_files is ignored and the data is plotted directly (used for ensemble means).
    """
    if data is None:
        data = load_phased_observables(config, phase_files=phase_files)

    # One adaptive time unit for the whole figure (all axes + phase markers), from the
    # overall duration, so the axes and the phase-boundary lines stay aligned.
    _tarr = np.asarray(data["time_us"])
    time_factor, time_unit = choose_time_unit(float(_tarr.max()) if _tarr.size else 0.0)
    time_label = f"Time ({time_unit})"
    t_bonds = _tarr * time_factor
    t_kin = np.asarray(data["kin_time_us"]) * time_factor
    t_clus = np.asarray(data["cluster_time_us"]) * time_factor
    bnd = list(np.asarray(data["phase_boundaries_us"]) * time_factor) if data["phase_boundaries_us"] else []
    starts = list(np.asarray(data["phase_starts_us"]) * time_factor) if data["phase_starts_us"] else []
    names = data["phase_names"]

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # 1) Bonds
    ax = axes[0]
    ax.plot(t_bonds, data["n_bonds"], color="C0", lw=1.5)
    ax.set_ylabel("Number of bonds")
    # Default title describes a cycle; a non-phased run passes its own.
    ax.set_title(title if title is not None else "Agglomeration / deagglomeration cycle")
    ax.grid(True, alpha=0.3)
    _mark_phase_boundaries(ax, bnd, starts, names)

    # 2) Fraction bound
    ax = axes[1]
    ax.plot(t_kin, data["fraction_bound_qt"], color="C1", lw=1.5, label="Qt")
    ax.plot(t_kin, data["fraction_bound_ft"], color="C2", lw=1.5, label="Ft")
    ax.set_ylabel("Fraction bound")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="center left")
    ax.grid(True, alpha=0.3)
    _mark_phase_boundaries(ax, bnd)

    # 3) Average cluster size
    ax = axes[2]
    ax.plot(t_clus, data["avg_sizes"], color="C3", lw=1.5)
    ax.set_ylabel("Avg cluster size")
    ax.set_xlabel(time_label)
    ax.grid(True, alpha=0.3)
    _mark_phase_boundaries(ax, bnd)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"✓ Saved: {save_path}")

    return fig


# =============================================================================
# THESIS PANELS (composite multi-subplot figures; save paired SVG + PNG)
# =============================================================================

def plot_ensemble_panel(
    stats: Dict,
    structural: Dict,
    config: Dict,
    *,
    show_individual: bool = False,
    individual_alpha: float = 0.3,
    figsize: Tuple[float, float] = (18, 18),
    save_path_base: Optional[str] = None,
) -> plt.Figure:
    """Curated single-ensemble thesis panel (4x3 grid); optionally saves {base}.svg + .png.

    Layout:
        Row 1: Energy | Pressure | Number of Bonds
        Row 2: Particle Counts | Number of Individual Topologies | Average Cluster Size
        Row 3: Largest Cluster Size | Particles by Size Category | Mean Radius of Gyration
        Row 4: Mean Cluster Composition | Coordination Number | Coordination Distribution (Final)

    The final-row histogram needs the ``final_coord_dist_*`` keys in
    ``ensemble_structural.npz``; ensembles analysed before those were added render it as
    "No data" (re-run ``scripts/analyze_ensemble.py`` on the directory to populate them).
    """
    print("\nGenerating ensemble thesis panel...")

    fig, axes = plt.subplots(4, 3, figsize=figsize)

    config = config or {}
    timestep = config.get('timestep', 1e-4)
    times_us, time_label = _time_axis(_steps_to_us(np.asarray(stats['times']), timestep))
    n_replicas = stats.get('n_replicas', 1)

    def simple_band(ax, mean_key, std_key, color, title, ylabel, legend_loc='best'):
        if mean_key in stats:
            if _ensemble_plot_with_band(ax, times_us, stats[mean_key], stats[std_key],
                                        color, n_replicas,
                                        _ensemble_all_trace(stats, structural, mean_key[:-5]),
                                        show_individual, individual_alpha):
                ax.legend(loc=legend_loc, fontsize=FONTSIZE_LEGEND)
        else:
            _ensemble_show_no_data(ax)
        ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
        ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL)
        ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Row 1
    simple_band(axes[0, 0], 'energy_mean', 'energy_std', 'tab:red',
                "Potential Energy", "Energy (kJ/mol)", legend_loc='upper right')
    simple_band(axes[0, 1], 'pressure_mean', 'pressure_std', 'tab:green',
                "Pressure", "Pressure (kJ/mol/nm³)", legend_loc='upper right')
    simple_band(axes[0, 2], 'bonds_mean', 'bonds_std', 'tab:blue',
                "Number of Bonds", "Number of Bonds", legend_loc='lower right')

    # Row 2: particle counts (multi-line), topologies, avg cluster
    ax = axes[1, 0]
    particle_colors = {'qt': 'blue', 'ft': 'red', 'qtc': 'darkblue', 'ftc': 'darkred'}
    particle_labels = {'qt': 'Qt (free)', 'ft': 'Ft (free)', 'qtc': 'QtC', 'ftc': 'FtC'}
    has_particle_data = False
    for ptype in ['qt', 'ft', 'qtc', 'ftc']:
        key_mean = f'{ptype}_count_mean'
        key_std = f'{ptype}_count_std'
        if key_mean in stats:
            mean = np.asarray(stats[key_mean])
            std = np.asarray(stats[key_std])
            ax.plot(times_us, mean, color=particle_colors[ptype],
                    linewidth=2, label=particle_labels[ptype])
            ax.fill_between(times_us, mean - std, mean + std,
                            color=particle_colors[ptype], alpha=0.2)
            has_particle_data = True
    if has_particle_data:
        ax.legend(loc='upper right', fontsize=FONTSIZE_LEGEND)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Count", fontsize=FONTSIZE_LABEL)
    ax.set_title("Particle Counts", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)

    simple_band(axes[1, 1], 'n_clusters_mean', 'n_clusters_std', 'tab:purple',
                "Number of Individual Topologies", "Number of Individual Topologies",
                legend_loc='upper right')
    simple_band(axes[1, 2], 'avg_cluster_mean', 'avg_cluster_std', 'tab:olive',
                "Average Cluster Size", "Average Size (particles)", legend_loc='lower right')

    # Row 3: largest cluster, size categories, mean Rg
    simple_band(axes[2, 0], 'largest_cluster_mean', 'largest_cluster_std', 'tab:orange',
                "Largest Cluster Size", "Cluster Size (particles)", legend_loc='lower right')

    ax = axes[2, 1]
    if structural and 'size_fractions_times' in structural and \
            'size_fractions_category_names' in structural:
        sc_times, time_label = _time_axis(_steps_to_us(np.asarray(structural['size_fractions_times']), timestep))
        category_names = list(structural['size_fractions_category_names'])
        mean_fractions = []
        for cat_name in category_names:
            safe_key = _size_category_key(cat_name)
            mean_key = f'size_frac_{safe_key}_mean'
            if mean_key in structural:
                mean_fractions.append(np.asarray(structural[mean_key]))
            else:
                mean_fractions.append(np.zeros(len(sc_times)))
        colors = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple"]
        ax.stackplot(sc_times, *mean_fractions, labels=category_names,
                     colors=colors[:len(category_names)], alpha=0.8)
        ax.set_ylim([0, 1])
        ax.legend(loc='upper right', fontsize=FONTSIZE_LEGEND)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Fraction", fontsize=FONTSIZE_LABEL)
    ax.set_title("Particles by Size Category", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[2, 2]
    times, mean, std, all_data, time_label = _ensemble_struct_ts(
        structural, timestep, 'morphology_times', 'mean_rg_mean', 'mean_rg_std', 'mean_rg_all')
    if times is not None and mean is not None:
        if _ensemble_plot_with_band(ax, times, mean, std, 'tab:blue', n_replicas,
                                    all_data, show_individual, individual_alpha):
            ax.legend(loc='lower right', fontsize=FONTSIZE_LEGEND)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Mean Rg (nm)", fontsize=FONTSIZE_LABEL)
    ax.set_title("Mean Radius of Gyration", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Row 4: mean composition, coordination (Qt+Ft fused), final coordination distribution
    ax = axes[3, 0]
    times, mean, std, all_data, time_label = _ensemble_struct_ts(
        structural, timestep, 'composition_times', 'mean_composition_mean',
        'mean_composition_std', 'mean_composition_all')
    if times is not None and mean is not None:
        if _ensemble_plot_with_band(ax, times, mean, std, 'tab:blue', n_replicas,
                                    all_data, show_individual, individual_alpha):
            ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='_nolegend_')
            ax.legend(loc='best', fontsize=FONTSIZE_LEGEND)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Mean Qt Fraction", fontsize=FONTSIZE_LABEL)
    ax.set_ylim([0, 1])
    ax.set_title("Mean Cluster Composition", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax = axes[3, 1]
    t_qt, m_qt, s_qt, all_qt, time_label = _ensemble_struct_ts(
        structural, timestep, 'contacts_times', 'mean_coord_qt_mean', 'mean_coord_qt_std',
        'mean_coord_qt_all')
    t_ft, m_ft, s_ft, all_ft, _ = _ensemble_struct_ts(
        structural, timestep, 'contacts_times', 'mean_coord_ft_mean', 'mean_coord_ft_std',
        'mean_coord_ft_all')
    coord_plotted = False
    if t_qt is not None and m_qt is not None:
        coord_plotted |= _ensemble_plot_with_band(
            ax, t_qt, m_qt, s_qt, COORD_COLOR_QT, n_replicas,
            all_qt, show_individual, individual_alpha, label='Qt')
    if t_ft is not None and m_ft is not None:
        coord_plotted |= _ensemble_plot_with_band(
            ax, t_ft, m_ft, s_ft, COORD_COLOR_FT, n_replicas,
            all_ft, show_individual, individual_alpha, label='Ft')
    if coord_plotted:
        ax.legend(loc='lower right', fontsize=FONTSIZE_LEGEND)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(time_label, fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Mean Coordination", fontsize=FONTSIZE_LABEL)
    ax.set_title("Coordination Number", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Final-frame per-particle coordination, pooled over replicas and rescaled to a
    # per-replica count so the y axis is comparable with the single-run figure.
    ax = axes[3, 2]
    if structural is not None and 'final_coord_dist_qt' in structural:
        n_contrib = int(np.atleast_1d(
            structural.get('final_coord_dist_n_replicas', [1]))[0]) or 1
        _plot_coord_distribution(
            ax,
            structural.get('final_coord_dist_qt', []),
            structural.get('final_coord_dist_ft', []),
            weight=1.0 / n_contrib,
            ylabel="Mean count per replica",
        )
    else:
        # Ensembles analysed before final_coord_dist_* was added to ensemble_structural.npz.
        _ensemble_show_no_data(ax)
        ax.set_xlabel("Coordination Number", fontsize=FONTSIZE_LABEL)
        ax.set_ylabel("Mean count per replica", fontsize=FONTSIZE_LABEL)
        ax.set_title("Coordination Distribution (Final)",
                     fontsize=FONTSIZE_TITLE, fontweight='bold')

    plt.tight_layout()

    if save_path_base:
        for ext in ("svg", "png"):
            path = f"{save_path_base}.{ext}"
            fig.savefig(path, format=ext, bbox_inches='tight', dpi=300)
            print(f"✓ Saved panel to {path}")

    return fig


def plot_comparison_panel(
    comparison: dict,
    *,
    show_bands: Optional[bool] = None,
    figsize: Tuple[float, float] = (24, 17),
    save_path_base: Optional[str] = None,
) -> plt.Figure:
    """Cross-ensemble comparison thesis panel (3x4 grid); optionally saves {base}.svg + .png.

    Rows 1-2 overlay per-ensemble basic statistics (with two ÷N-normalized cluster-size panels);
    Row 3 overlays structural metrics (needs the live `compare_ensembles` structural data).
    """
    print("\nGenerating ensemble comparison thesis panel...")

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    show_bands = _get_show_bands_default(comparison['n_ensembles'], show_bands)

    def stat(ax, stat_key, ylabel, title, legend_loc='best', divide_by_N=False):
        if _comparison_timeseries(ax, comparison, stat_key, ylabel, title,
                                  show_bands=show_bands, divide_by_N=divide_by_N):
            ax.legend(loc=legend_loc, fontsize=FONTSIZE_LEGEND)

    # Row 1
    stat(fig.add_subplot(gs[0, 0]), 'energy', "Energy (kJ/mol)", "Potential Energy",
         legend_loc='lower left')
    stat(fig.add_subplot(gs[0, 1]), 'pressure', "Pressure (kJ/(mol·nm³))", "Pressure",
         legend_loc='upper left')
    stat(fig.add_subplot(gs[0, 2]), 'bonds', "Number of Bonds", "Number of Bonds",
         legend_loc='lower right')
    stat(fig.add_subplot(gs[0, 3]), 'n_clusters', "Number of Individual Topologies",
         "Number of Individual Topologies", legend_loc='upper right')

    # Row 2
    stat(fig.add_subplot(gs[1, 0]), 'avg_cluster', "Average Size (particles)",
         "Average Cluster Size", legend_loc='upper left')
    stat(fig.add_subplot(gs[1, 1]), 'avg_cluster', "Fraction of particles",
         "Average Cluster Size (normalized)", legend_loc='upper left', divide_by_N=True)
    stat(fig.add_subplot(gs[1, 2]), 'largest_cluster', "Cluster Size (particles)",
         "Largest Cluster Size", legend_loc='upper left')
    ax_largest_norm = fig.add_subplot(gs[1, 3])
    stat(ax_largest_norm, 'largest_cluster', "Fraction of particles",
         "Largest Cluster Size (normalized)", legend_loc='upper left', divide_by_N=True)
    ax_largest_norm.set_ylim([0, 1])

    # Row 3
    _comparison_struct_ts(fig.add_subplot(gs[2, 0]), comparison, 'morphology_times',
                          'mean_rg_mean', 'mean_rg_std', "Mean Rg (nm)",
                          "Mean Radius of Gyration", show_bands=show_bands, legend_loc='upper left')
    _comparison_struct_ts(fig.add_subplot(gs[2, 1]), comparison, 'morphology_times',
                          'mean_rg_normalized_mean', 'mean_rg_normalized_std',
                          r"Rg / Rg$_{\mathrm{ideal}}$", "Normalized Radius of Gyration",
                          show_bands=show_bands, legend_loc='lower right')
    _comparison_coord_fused(fig.add_subplot(gs[2, 2]), comparison,
                            show_bands=show_bands, legend_loc='lower right')
    ax_comp = fig.add_subplot(gs[2, 3])
    _comparison_struct_ts(ax_comp, comparison, 'composition_times', 'mean_composition_mean',
                          'mean_composition_std', "Mean Qt Fraction", "Mean Cluster Composition",
                          show_bands=show_bands, legend_loc='lower right')
    ax_comp.set_ylim([0, 1])

    if save_path_base:
        for ext in ("svg", "png"):
            path = f"{save_path_base}.{ext}"
            fig.savefig(path, format=ext, bbox_inches='tight', dpi=300)
            print(f"✓ Saved panel to {path}")

    return fig
