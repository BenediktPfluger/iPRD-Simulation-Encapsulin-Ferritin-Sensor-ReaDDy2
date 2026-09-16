"""
Qt-Ft Agglomeration Plotting Module

All matplotlib figures for the ReaDDy2 Qt-Ft agglomeration simulations. Every run type —
single run, ensemble, cross-ensemble comparison — produces the same three figures, built
from the same ``(stats, structural, config)`` triple:

    plot_metrics_panel         12-metric overview (single run or ensemble)
    plot_kinetics              bonds / fraction bound / avg cluster size (phase-aware)
    plot_large_cluster_count   number of clusters above a size threshold

``plot_comparison_panel`` is the cross-ensemble variant of the overview.

Related modules:
    - qtft.config / qtft.system / qtft.engine: configuration and simulation execution
    - qtft.analysis: core analysis functions (no matplotlib dependency)
    - qtft.ensemble: EnsembleSimulation class for multi-replica runs

Usage:
    import qtft.analysis as analysis
    import qtft.plotting as plotting

    # one triple per target; a single run uses build_single_run_plotting_data
    stats, structural, config = analysis.load_ensemble_data(ensemble_dir)
    plotting.plot_metrics_panel(stats, structural, config, save_path_base="Plots/panel")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Import from simulation module
from .config import SimulationConfig, _steps_to_us, choose_time_unit

# Import analysis functions needed by plotting
from .analysis import (
    load_phased_observables,
    _size_category_key,
)


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



# =============================================================================
# CONSTANTS (Plotting)
# =============================================================================

# Plotting fontsize configuration (consistent across all plots)
FONTSIZE_TITLE = 14
FONTSIZE_LABEL = 12
FONTSIZE_LEGEND = 10
FONTSIZE_TICK = 10

# Species colours used wherever Qt/Ft (or QtC/FtC) are drawn as separate series: the
# coordination plots and the fraction-bound panel of plot_kinetics. Deliberately
# NOT applied to particle counts, composition or Rg, which keep their own scheme.
SPECIES_COLOR_QT = 'tab:green'
SPECIES_COLOR_FT = 'tab:red'


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

        for values, color, label in ((coord_qt, SPECIES_COLOR_QT, 'QtC'),
                                     (coord_ft, SPECIES_COLOR_FT, 'FtC')):
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


# =============================================================================
# SHARED PLOTTING HELPERS
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


def plot_large_cluster_count(
    series: List[Dict[str, Any]],
    min_size: int,
    timestep: float,
    *,
    save_path: Optional[str] = None,
    save_path_base: Optional[str] = None,
    phase_boundaries_us: Optional[List[float]] = None,
    phase_starts_us: Optional[List[float]] = None,
    phase_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    show_individual: bool = False,
    individual_alpha: float = 0.3,
    figsize: Tuple[float, float] = (9, 5.5),
) -> plt.Figure:
    """Plot the number of clusters at or above a size threshold, over time.

    Works for every mode: one entry in ``series`` for a single run or one ensemble, several
    for a cross-ensemble comparison.

    Parameters
    ----------
    series : list of dict
        Each entry: ``label`` (str), ``times`` (step numbers), ``mean`` (counts), and
        optionally ``std``, ``all`` (per-replica traces) and ``n_replicas``. A single run
        passes ``n_replicas=1``, which suppresses the error band.
    min_size : int
        The threshold these counts were computed with (used for the title/label).
    timestep : float
        ns per step, for the step -> µs conversion.
    save_path / save_path_base : str, optional
        Save one file, or paired ``.svg`` + ``.png``.
    phase_boundaries_us, phase_starts_us, phase_names : optional
        Phase markers for an agglomeration<->deagglomeration run, as produced by
        ``analysis.load_phased_observables``.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # One adaptive time unit for the whole figure, shared with the phase markers.
    max_us = 0.0
    for s in series:
        t = _steps_to_us(np.asarray(s["times"], dtype=float), timestep)
        max_us = max(max_us, float(t.max()) if t.size else 0.0)
    time_factor, time_unit = choose_time_unit(max_us)

    plotted = False
    for i, s in enumerate(series):
        t_us = _steps_to_us(np.asarray(s["times"], dtype=float), timestep) * time_factor
        color = COMPARISON_COLORS[i % len(COMPARISON_COLORS)] if len(series) > 1 else 'tab:blue'
        plotted |= _ensemble_plot_with_band(
            ax, t_us, s["mean"], s.get("std"), color,
            s.get("n_replicas", 1),
            all_data=s.get("all"), show_individual=show_individual,
            individual_alpha=individual_alpha,
            label=s.get("label"),
            band_label='± 1 SD' if len(series) == 1 else '_nolegend_',
        )

    if not plotted:
        _ensemble_show_no_data(ax)
    else:
        ax.legend(loc='best', fontsize=FONTSIZE_LEGEND)

    if phase_boundaries_us:
        bnd = list(np.asarray(phase_boundaries_us) * time_factor)
        starts = list(np.asarray(phase_starts_us) * time_factor) if phase_starts_us else None
        _mark_phase_boundaries(ax, bnd, starts, phase_names)

    ax.set_xlabel(f"Time ({time_unit})", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Number of clusters", fontsize=FONTSIZE_LABEL)
    ax.set_title(title if title is not None else f"Clusters with size ≥ {min_size}",
                 fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✓ Saved plot to {save_path}")
    if save_path_base:
        for ext in ("svg", "png"):
            p = f"{save_path_base}.{ext}"
            fig.savefig(p, format=ext, bbox_inches='tight', dpi=300)
            print(f"✓ Saved plot to {p}")
    return fig


def plot_kinetics(
    series: List[Dict[str, Any]],
    *,
    save_path: Optional[str] = None,
    save_path_base: Optional[str] = None,
    figsize: Tuple[float, float] = (11, 9),
    title: Optional[str] = None,
):
    """Bonds / fraction bound / average cluster size on one continuous time axis.

    Takes the same ``series`` list shape as ``plot_large_cluster_count``, so one function
    serves every mode: one entry for a single run or one ensemble, several for a
    cross-ensemble comparison.

    Parameters
    ----------
    series : list of dict
        Each entry ``{"label": str, "data": dict}`` where ``data`` follows the
        ``analysis.load_phased_observables`` schema (also produced by
        ``analysis.build_kinetics_data_single`` and ``build_kinetics_data_ensemble``).
    save_path / save_path_base : str, optional
        Save one file, or paired ``.svg`` + ``.png``.
    title : str, optional
        Defaults to the cycle title when the first entry has phase boundaries, otherwise a
        plain kinetics title.

    Notes
    -----
    With one entry the three panels keep their dedicated colours (bonds blue, Qt green /
    Ft red, average cluster size orange). With several, each entry is coloured by ensemble
    and the two species are distinguished by linestyle (Qt solid, Ft dashed) — the same
    convention ``_comparison_coord_fused`` uses. Phase markers come from the first entry,
    whose boundaries are shared by construction.
    """
    if not series:
        raise ValueError("plot_kinetics requires at least one series entry")
    multi = len(series) > 1

    # One adaptive time unit for the whole figure (all axes + phase markers), taken from the
    # longest series so the axes and boundary lines stay aligned.
    max_us = max(float(np.asarray(s["data"]["time_us"]).max())
                 if len(s["data"]["time_us"]) else 0.0 for s in series)
    time_factor, time_unit = choose_time_unit(max_us)

    first = series[0]["data"]
    bnd = list(np.asarray(first["phase_boundaries_us"]) * time_factor) if first.get("phase_boundaries_us") else []
    starts = list(np.asarray(first["phase_starts_us"]) * time_factor) if first.get("phase_starts_us") else []
    names = first.get("phase_names")

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    for i, entry in enumerate(series):
        d = entry["data"]
        label = entry.get("label")
        colour = COMPARISON_COLORS[i % len(COMPARISON_COLORS)] if multi else None
        t_bonds = np.asarray(d["time_us"]) * time_factor
        t_kin = np.asarray(d["kin_time_us"]) * time_factor
        t_clus = np.asarray(d["cluster_time_us"]) * time_factor

        # 1) Bonds
        axes[0].plot(t_bonds, d["n_bonds"], lw=1.5,
                     color=colour if multi else "C0", label=label if multi else None)

        # 2) Fraction bound — per species when single, per ensemble (Qt solid / Ft dashed)
        #    when comparing, so N ensembles stay readable in one axes.
        if multi:
            axes[1].plot(t_kin, d["fraction_bound_qt"], color=colour, lw=1.5, ls="-",
                         label=label)
            axes[1].plot(t_kin, d["fraction_bound_ft"], color=colour, lw=1.5, ls="--",
                         label="_nolegend_")
        else:
            axes[1].plot(t_kin, d["fraction_bound_qt"], color=SPECIES_COLOR_QT, lw=1.5, label="Qt")
            axes[1].plot(t_kin, d["fraction_bound_ft"], color=SPECIES_COLOR_FT, lw=1.5, label="Ft")

        # 3) Average cluster size
        axes[2].plot(t_clus, d["avg_sizes"], lw=1.5,
                     color=colour if multi else "tab:orange", label=label if multi else None)

    axes[0].set_ylabel("Number of bonds")
    axes[0].set_title(title if title is not None
                      else ("Agglomeration / deagglomeration cycle" if bnd
                            else "Agglomeration kinetics"))
    axes[1].set_ylabel("Fraction bound")
    axes[1].set_ylim(-0.05, 1.05)
    axes[2].set_ylabel("Avg cluster size")
    axes[2].set_xlabel(f"Time ({time_unit})")

    for ax in axes:
        ax.grid(True, alpha=0.3)
    _mark_phase_boundaries(axes[0], bnd, starts, names)
    _mark_phase_boundaries(axes[1], bnd)
    _mark_phase_boundaries(axes[2], bnd)

    if multi:
        # ensemble colours, plus a solid/dashed key for the two species
        handles, labels = axes[1].get_legend_handles_labels()
        style = [Line2D([0], [0], color="black", ls="-", lw=1.5),
                 Line2D([0], [0], color="black", ls="--", lw=1.5)]
        axes[1].legend(handles + style, labels + ["Qt", "Ft"],
                       loc="center left", fontsize=FONTSIZE_LEGEND)
        axes[0].legend(loc="best", fontsize=FONTSIZE_LEGEND)
    else:
        axes[1].legend(loc="center left")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"✓ Saved plot to {save_path}")
    if save_path_base:
        for ext in ("svg", "png"):
            p = f"{save_path_base}.{ext}"
            fig.savefig(p, format=ext, bbox_inches="tight", dpi=300)
            print(f"✓ Saved plot to {p}")
    return fig


def plot_phased_kinetics(
    config: "SimulationConfig",
    phase_files: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (11, 9),
    data: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
):
    """Back-compat wrapper over :func:`plot_kinetics` for a single run.

    Loads the per-phase trajectories via ``analysis.load_phased_observables`` unless
    pre-stitched ``data`` is supplied.
    """
    if data is None:
        data = load_phased_observables(config, phase_files=phase_files)
    return plot_kinetics([{"label": None, "data": data}],
                         save_path=save_path, figsize=figsize, title=title)


# =============================================================================
# THESIS PANELS (composite multi-subplot figures; save paired SVG + PNG)
# =============================================================================

def plot_metrics_panel(
    stats: Dict,
    structural: Dict,
    config: Dict,
    *,
    show_individual: bool = False,
    individual_alpha: float = 0.3,
    figsize: Tuple[float, float] = (18, 18),
    save_path_base: Optional[str] = None,
) -> plt.Figure:
    """Curated 12-metric thesis panel (4x3 grid); optionally saves {base}.svg + .png.

    Takes the ``(stats, structural, config)`` triple, which a single run produces via
    ``analysis.build_single_run_plotting_data`` and an ensemble via
    ``EnsembleSimulation.to_plotting_format()`` — hence "metrics", not "ensemble".

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
            ax, t_qt, m_qt, s_qt, SPECIES_COLOR_QT, n_replicas,
            all_qt, show_individual, individual_alpha, label='Qt')
    if t_ft is not None and m_ft is not None:
        coord_plotted |= _ensemble_plot_with_band(
            ax, t_ft, m_ft, s_ft, SPECIES_COLOR_FT, n_replicas,
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


# Renamed in the P3 cleanup: a single run calls this too, so "ensemble" was misleading.
plot_ensemble_panel = plot_metrics_panel
