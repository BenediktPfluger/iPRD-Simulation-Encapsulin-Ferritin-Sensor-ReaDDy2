"""
Qt-Ft Agglomeration Plotting Module

All matplotlib figures for the ReaDDy2 Qt-Ft agglomeration simulations. Every run type —
single run, ensemble, cross-ensemble comparison — produces the same three figures, built
from the same ``(stats, structural, config)`` triple:

    plot_metrics_panel         12-metric overview (single run or ensemble)
    plot_kinetics              bonds / fraction bound / avg cluster size (phase-aware)
    plot_large_cluster_count   number of clusters above a size threshold

``plot_comparison_panel`` is the cross-ensemble variant of the overview.
``plot_overlap_timeseries`` plots particle-pair overlaps over time for one trajectory
(from ``analysis.get_overlap_timeseries``).

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

import functools
import glob
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullLocator

# Import from simulation module
from .config import SimulationConfig, _steps_to_us, choose_time_unit

# Import analysis functions needed by plotting
from .analysis import (
    load_phased_observables,
    _size_category_key,
)


# =============================================================================
# FIGURE STYLE (Nature)
# =============================================================================
# Arial 6/7 pt text, 0.5 pt axes, inward ticks on all four sides, 1 pt data lines; every
# figure is drawn at its final size (89 mm single column, 183 mm double) and saved as
# PDF + SVG + PNG with the text left editable. The style is applied per plot function
# (``_nature_style``), never globally, so other matplotlib figures are unaffected.

# Arial file names (regular, bold, italic, bold italic). Arial Narrow is skipped on purpose.
_ARIAL_FILES = ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf")


def _arial_search_dirs() -> List[str]:
    """Folders searched for the Arial files, in order of preference."""
    dirs = []
    if os.environ.get("QTFT_FONT_DIR"):
        dirs.append(os.environ["QTFT_FONT_DIR"])
    dirs += [
        "/mnt/c/Windows/Fonts",                       # WSL: the Windows font folder
        os.path.expanduser("~/.fonts"),
        os.path.expanduser("~/.local/share/fonts"),
    ]
    return dirs


def _register_arial() -> bool:
    """Make Arial available to matplotlib; return True if it is usable.

    matplotlib on Linux/WSL does not see the Windows fonts, so the four Arial files are
    registered from the first folder that has them. The font files are only read from
    the system — never copied into the repository (Arial is licensed by Microsoft).
    """
    if any(f.name == "Arial" for f in _fm.fontManager.ttflist):
        return True
    for folder in _arial_search_dirs():
        if not os.path.isdir(folder):
            continue
        # Match case-insensitively: Windows stores e.g. arial.ttf but sometimes ARIAL.TTF.
        by_name = {os.path.basename(p).lower(): p for p in glob.glob(os.path.join(folder, "*"))}
        found = [by_name[n] for n in _ARIAL_FILES if n in by_name]
        if not found:
            continue
        for path in found:
            try:
                _fm.fontManager.addfont(path)
            except Exception:  # unreadable file: try the remaining ones
                pass
        if any(f.name == "Arial" for f in _fm.fontManager.ttflist):
            return True
    return False


ARIAL_AVAILABLE = _register_arial()
if not ARIAL_AVAILABLE:
    warnings.warn(
        "Arial not found (searched $QTFT_FONT_DIR, /mnt/c/Windows/Fonts, ~/.fonts, "
        "~/.local/share/fonts); figures fall back to Liberation Sans / DejaVu Sans. "
        "Set QTFT_FONT_DIR to a folder containing arial.ttf to fix this.",
        stacklevel=2,
    )

NATURE_RC: Dict[str, Any] = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7,
    'figure.titlesize': 7, 'figure.labelsize': 7,
    'legend.fontsize': 7, 'legend.title_fontsize': 7,
    'xtick.labelsize': 6, 'ytick.labelsize': 6,
    'axes.linewidth': 0.5, 'lines.linewidth': 1.0, 'lines.markersize': 3,
    'lines.markeredgewidth': 0.3,                       # matplotlib's 1 pt outline looks too heavy
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'legend.frameon': False, 'legend.handlelength': 1.5, 'legend.markerscale': 1.5,
    'axes.grid': False,
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,   # editable text
    'savefig.dpi': 300,
    # bbox_inches='tight' pads the content by this much. Equal to constrained layout's own
    # margin (w_pad = h_pad = 3/72 in), so a figure drawn at 89 mm is also saved at 89 mm.
    'savefig.pad_inches': 3 / 72,
    'figure.dpi': 200,          # on-screen (Jupyter) size only; saved files are unaffected
}
if ARIAL_AVAILABLE:
    NATURE_RC.update({'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial',
                      'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
                      # unused slots, pointed at Arial so mathtext never looks for 'cursive'
                      'mathtext.sf': 'Arial', 'mathtext.cal': 'Arial:italic'})
else:
    NATURE_RC['mathtext.fontset'] = 'dejavusans'

# Exponents of 10^x tick labels at 5 pt (5/6 of the 6 pt tick label; matplotlib's 0.7 gives
# 4.2 pt). This is a private, process-wide matplotlib constant read at draw time, so it is
# set once here and guarded against matplotlib versions that drop it.
try:
    import matplotlib._mathtext as _mathtext
    _mathtext.SHRINK_FACTOR = 5 / 6
except Exception as exc:  # pragma: no cover - depends on matplotlib internals
    warnings.warn(f"10^x exponent size not changed: {exc}", stacklevel=2)

MM = 1 / 25.4               # inches per mm
WIDTH_1COL_MM = 89.0        # Nature single column
WIDTH_2COL_MM = 183.0       # Nature double column
MAX_HEIGHT_MM = 170.0


def figsize_mm(w: float = WIDTH_1COL_MM, h: float = 45.0) -> Tuple[float, float]:
    """Figure size in inches from millimetres (default: single-column time course)."""
    return (w * MM, h * MM)


def _draw_all(result) -> None:
    """Draw every figure in ``result`` (a Figure, or a dict/list of them).

    Tick artists are created lazily at draw time from the *current* rcParams. Drawing
    once inside the style context fixes the styled ticks in place, so a later draw
    outside it (e.g. Jupyter's inline display) shows the same figure that was saved.
    """
    if isinstance(result, plt.Figure):
        figs = [result]
    elif isinstance(result, dict):
        figs = [f for f in result.values() if isinstance(f, plt.Figure)]
    elif isinstance(result, (list, tuple)):
        figs = [f for f in result if isinstance(f, plt.Figure)]
    else:
        figs = []
    for fig in figs:
        fig.canvas.draw()


def _nature_style(func):
    """Run a public plot function inside the Nature rc context."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with plt.rc_context(NATURE_RC):
            result = func(*args, **kwargs)
            _draw_all(result)
        return result
    return wrapper


def _save_figure(fig: plt.Figure, path_stem: str,
                 formats: Tuple[str, ...] = ("pdf", "svg", "png")) -> List[str]:
    """Save ``fig`` as ``{path_stem}.pdf/.svg/.png`` (PNG at 300 dpi), tight bounding box.

    Creates the parent folder if needed and returns the written paths. Call it inside
    the style context so the editable-text settings (``svg.fonttype``/``pdf.fonttype``)
    apply.
    """
    parent = os.path.dirname(path_stem)
    if parent:
        os.makedirs(parent, exist_ok=True)
    written = []
    for fmt in formats:
        path = f"{path_stem}.{fmt}"
        fig.savefig(path, format=fmt, bbox_inches='tight',
                    dpi=300 if fmt == 'png' else None)
        written.append(path)
        print(f"✓ Saved plot to {path}")
    return written


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
    title: Optional[str] = None,
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
    ylabel : str
        Axis label (the ensemble panel relabels the y axis).
    title : str, optional
        Panel title; none by default.
    """
    coord_qt = np.asarray(coord_qt)
    coord_ft = np.asarray(coord_ft)

    if len(coord_qt) > 0 or len(coord_ft) > 0:
        max_coord = max(
            coord_qt.max() if len(coord_qt) > 0 else 0,
            coord_ft.max() if len(coord_ft) > 0 else 0
        )
        bins = np.arange(-0.5, max_coord + 1.5, 1)

        for values, color, label in ((coord_qt, SPECIES_COLOR_QT, 'Qt'),
                                     (coord_ft, SPECIES_COLOR_FT, 'Ft')):
            if len(values) == 0:
                continue
            w = np.full(len(values), weight) if weight is not None else None
            ax.hist(values, bins=bins, alpha=0.6, color=color, weights=w,
                    label=f'{label} (mean {np.mean(values):.2f})',
                    edgecolor='black', linewidth=0.3)
        ax.legend(loc='best')
    ax.set_xlabel("Coordination number")
    ax.set_ylabel(ylabel)
    # Coordination numbers are integers: whole-number ticks, no minor ticks in between.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_minor_locator(NullLocator())
    if title:
        ax.set_title(title)


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
    ax.plot(times, mean, color=color, label=mean_label)
    if not single_run:
        ax.fill_between(times, mean - std, mean + std, color=color, alpha=0.3,
                        linewidth=0, label=band_label)

    if show_individual and all_data is not None and not single_run:
        for data in all_data:
            ax.plot(times, data, color=color, alpha=individual_alpha, linewidth=0.4)
    
    return True



def _ensemble_show_no_data(ax):
    """Helper to show 'No data available' message on axes."""
    ax.text(0.5, 0.5, "No data available", ha='center', va='center',
           transform=ax.transAxes, color='gray')


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

    Draws lines/bands + labels/title; the caller places the legend (so standalone and
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


# Short forms used when a phase name does not fit its span on a narrow figure.
_PHASE_SHORT_NAMES = {"agglomerate": "agg.", "deagglomerate": "deagg."}


def _mark_phase_boundaries(ax, boundaries_us, starts_us=None, names=None):
    """Draw vertical lines at phase switches and (optionally) label each phase span.

    Labels sit in headroom added above the data (no background box, so they never hide
    data). A name wider than its span falls back to its short form ("agg."/"deagg.", or
    the first four letters), and is dropped if even that does not fit.
    """
    for b in boundaries_us:
        ax.axvline(b, color="0.4", linestyle="--", linewidth=0.5, zorder=1)
    if not starts_us or not names:          # None or empty: nothing to label
        return
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.14 * (hi - lo))           # headroom for one row of 6 pt labels
    ends = list(starts_us[1:]) + [ax.get_xlim()[1]]
    renderer = ax.figure.canvas.get_renderer()
    to_data = ax.transData.inverted()
    for name, s, e in zip(names, starts_us, ends):
        short = _PHASE_SHORT_NAMES.get(str(name).lower(), f"{str(name)[:4]}.")
        for candidate in (str(name), short):
            txt = ax.text(0.5 * (s + e), 0.98, candidate, ha="center", va="top",
                          fontsize=6, color="0.3", transform=ax.get_xaxis_transform())
            bb = txt.get_window_extent(renderer)
            (x0, _), (x1, _) = to_data.transform([[bb.x0, 0], [bb.x1, 0]])
            if x1 - x0 <= 0.95 * (e - s):
                break
            txt.remove()


@_nature_style
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
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Plot the number of agglomerates at or above a size threshold, over time.

    Works for every mode: one entry in ``series`` for a single run or one ensemble, several
    for a cross-ensemble comparison.

    Parameters
    ----------
    series : list of dict
        Each entry: ``label`` (str), ``times`` (step numbers), ``mean`` (counts), and
        optionally ``std``, ``all`` (per-replica traces) and ``n_replicas``. A single run
        passes ``n_replicas=1``, which suppresses the error band.
    min_size : int
        The threshold these counts were computed with (shown in the y-axis label).
    timestep : float
        ns per step, for the step -> µs conversion.
    save_path / save_path_base : str, optional
        Save one file, or ``{save_path_base}.pdf/.svg/.png``.
    phase_boundaries_us, phase_starts_us, phase_names : optional
        Phase markers for an agglomeration<->deagglomeration run, as produced by
        ``analysis.load_phased_observables``.
    title : str, optional
        Figure title; none by default (the information belongs in the figure legend).
    figsize : (float, float), optional
        Inches; defaults to 89 × 45 mm (Nature single column).
    """
    fig, ax = plt.subplots(figsize=figsize or figsize_mm(89, 45), layout='constrained')

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
        ax.legend(loc='best')

    if phase_boundaries_us:
        bnd = list(np.asarray(phase_boundaries_us) * time_factor)
        starts = list(np.asarray(phase_starts_us) * time_factor) if phase_starts_us else None
        _mark_phase_boundaries(ax, bnd, starts, phase_names)

    ax.set_xlabel(f"Time ({time_unit})")
    ax.set_ylabel(f"Agglomerates (size ≥ {min_size})")
    if title:
        ax.set_title(title)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✓ Saved plot to {save_path}")
    if save_path_base:
        _save_figure(fig, save_path_base)
    return fig


# Pair-family colours for the overlap plot: Qt-Qt / Ft-Ft reuse the species colours,
# the cross pair gets a third, distinct hue.
_OVERLAP_PAIR_COLORS = {"Qt-Qt": SPECIES_COLOR_QT, "Qt-Ft": 'tab:purple',
                        "Ft-Ft": SPECIES_COLOR_FT}

# metric key -> (y-axis label, scale factor applied to the stored values)
_OVERLAP_METRICS = {
    "n_overlapping": ("Overlapping pairs", 1.0),
    "frac_overlapping": ("Fraction of pairs overlapping", 1.0),
    "mean_overlap_all_frac": ("Mean interpenetration (% of contact)", 100.0),
    "max_overlap_frac": ("Deepest interpenetration (% of contact)", 100.0),
}


@_nature_style
def plot_overlap_timeseries(
    ts: Dict[str, Any],
    *,
    metric: str = "n_overlapping",
    log_y: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    save_path_base: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Plot particle-pair overlaps over time, one line per pair family.

    Parameters
    ----------
    ts : dict
        Result of ``analysis.get_overlap_timeseries`` (time axis already in µs).
    metric : str
        Which per-frame quantity to plot: ``"n_overlapping"`` (default — number of
        pairs with distance < r_i + r_j), ``"frac_overlapping"``,
        ``"mean_overlap_all_frac"`` or ``"max_overlap_frac"`` (the last two in % of
        contact).
    log_y : bool
        Log y-axis (default), since the families differ by orders of magnitude.
        Frames with zero overlap cannot be shown on a log axis and leave gaps; a family
        with no overlap at all is listed in the legend as "(none)". Falls back to a
        linear axis when nothing overlaps.
    title : str, optional
        Figure title; none by default, so the figure can carry its own caption.
    save_path / save_path_base : str, optional
        Save one file, or ``{save_path_base}.pdf/.svg/.png``.
    figsize : (float, float), optional
        Inches; defaults to 89 × 45 mm (Nature single column).
    """
    if metric not in _OVERLAP_METRICS:
        raise ValueError(f"metric must be one of {list(_OVERLAP_METRICS)}, got {metric!r}")
    ylabel, scale = _OVERLAP_METRICS[metric]

    fig, ax = plt.subplots(figsize=figsize or figsize_mm(89, 45), layout='constrained')
    t, time_label = _time_axis(ts["time_us"])

    any_positive = False
    for label, fam in ts["pairs"].items():
        y = np.asarray(fam[metric], dtype=float) * scale
        positive = np.isfinite(y) & (y > 0)
        color = _OVERLAP_PAIR_COLORS.get(label)
        if not positive.any():
            # Nothing to draw on either axis type beyond a flat zero; keep it in the legend.
            ax.plot([], [], color=color, label=f"{label} (none)")
            continue
        any_positive = True
        if log_y:
            y = np.where(positive, y, np.nan)
        ax.plot(t, y, marker='o', markersize=1.0, color=color, label=label)

    if log_y and any_positive:
        ax.set_yscale('log')
    ax.legend(loc='best')
    ax.set_xlabel(time_label)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"✓ Saved plot to {save_path}")
    if save_path_base:
        _save_figure(fig, save_path_base)
    return fig


# Kinetics panels: (name used for the separate subfigure file, y-axis label)
_KINETICS_PANELS = (
    ("bonds", "Number of bonds"),
    ("fraction_bound", "Fraction bound"),
    ("avg_agglomerate_size", "Average agglomerate\nsize (particles)"),
)


def _draw_kinetics_panel(ax, which: int, series: List[Dict[str, Any]], time_factor: float,
                         bnd: List[float], starts: List[float], names, *,
                         phase_labels: bool, legend: bool) -> None:
    """Draw kinetics panel ``which`` (0 bonds, 1 fraction bound, 2 average size) on ``ax``."""
    multi = len(series) > 1
    for i, entry in enumerate(series):
        d = entry["data"]
        label = entry.get("label")
        colour = COMPARISON_COLORS[i % len(COMPARISON_COLORS)] if multi else None
        if which == 0:
            ax.plot(np.asarray(d["time_us"]) * time_factor, d["n_bonds"],
                    color=colour if multi else "C0", label=label if multi else None)
        elif which == 1:
            # Per species when single, per ensemble (Qt solid / Ft dashed) when comparing,
            # so N ensembles stay readable in one axes.
            t_kin = np.asarray(d["kin_time_us"]) * time_factor
            if multi:
                ax.plot(t_kin, d["fraction_bound_qt"], color=colour, ls="-", label=label)
                ax.plot(t_kin, d["fraction_bound_ft"], color=colour, ls="--",
                        label="_nolegend_")
            else:
                ax.plot(t_kin, d["fraction_bound_qt"], color=SPECIES_COLOR_QT, label="Qt")
                ax.plot(t_kin, d["fraction_bound_ft"], color=SPECIES_COLOR_FT, label="Ft")
        else:
            ax.plot(np.asarray(d["cluster_time_us"]) * time_factor, d["avg_sizes"],
                    color=colour if multi else "tab:orange", label=label if multi else None)

    ax.set_ylabel(_KINETICS_PANELS[which][1])
    if which == 1:
        ax.set_ylim(-0.05, 1.05)
    _mark_phase_boundaries(ax, bnd, starts if phase_labels else None,
                           names if phase_labels else None)

    if not legend:
        return
    if which == 1 and multi:
        # ensemble colours, plus a solid/dashed key for the two species
        handles, labels = ax.get_legend_handles_labels()
        style = [Line2D([0], [0], color="black", ls="-"),
                 Line2D([0], [0], color="black", ls="--")]
        ax.legend(handles + style, labels + ["Qt", "Ft"], loc="best")
    elif which == 1 or multi:
        ax.legend(loc="best")


@_nature_style
def plot_kinetics(
    series: List[Dict[str, Any]],
    *,
    save_path: Optional[str] = None,
    save_path_base: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
):
    """Bonds / fraction bound / average agglomerate size on one continuous time axis.

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
        Save one file, or ``{save_path_base}.pdf/.svg/.png``. With ``save_path_base`` the
        three panels are additionally saved as separate, title-free 89 × 45 mm figures in
        ``{save_path_base}_subfigures/`` (``bonds``, ``fraction_bound``,
        ``avg_agglomerate_size``).
    figsize : (float, float), optional
        Inches; defaults to 89 × 95 mm (three stacked rows on one shared time axis).
    title : str, optional
        Figure title; none by default (the information belongs in the figure legend).

    Notes
    -----
    With one entry the three panels keep their dedicated colours (bonds blue, Qt green /
    Ft red, average agglomerate size orange). With several, each entry is coloured by
    ensemble and the two species are distinguished by linestyle (Qt solid, Ft dashed) —
    the same convention ``_comparison_coord_fused`` uses. Phase markers come from the
    first entry, whose boundaries are shared by construction.
    """
    if not series:
        raise ValueError("plot_kinetics requires at least one series entry")

    # One adaptive time unit for the whole figure (all axes + phase markers), taken from the
    # longest series so the axes and boundary lines stay aligned.
    max_us = max(float(np.asarray(s["data"]["time_us"]).max())
                 if len(s["data"]["time_us"]) else 0.0 for s in series)
    time_factor, time_unit = choose_time_unit(max_us)
    time_label = f"Time ({time_unit})"

    first = series[0]["data"]
    bnd = list(np.asarray(first["phase_boundaries_us"]) * time_factor) if first.get("phase_boundaries_us") else []
    starts = list(np.asarray(first["phase_starts_us"]) * time_factor) if first.get("phase_starts_us") else []
    names = first.get("phase_names")

    fig, axes = plt.subplots(3, 1, figsize=figsize or figsize_mm(89, 95), sharex=True,
                             layout='constrained')
    for which, ax in enumerate(axes):
        # Phase names only on the top row; the bonds legend (ensembles) only when comparing.
        _draw_kinetics_panel(ax, which, series, time_factor, bnd, starts, names,
                             phase_labels=(which == 0), legend=(which != 2))
    axes[2].set_xlabel(time_label)
    fig.align_ylabels(axes)
    if title:
        axes[0].set_title(title)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"✓ Saved plot to {save_path}")
    if save_path_base:
        _save_figure(fig, save_path_base)
        # The same three panels as separate single-column figures, without titles.
        sub_dir = f"{save_path_base}_subfigures"
        for which, (name, _) in enumerate(_KINETICS_PANELS):
            sub, ax = plt.subplots(figsize=figsize_mm(89, 45), layout='constrained')
            _draw_kinetics_panel(ax, which, series, time_factor, bnd, starts, names,
                                 phase_labels=True, legend=True)
            ax.set_xlabel(time_label)
            _save_figure(sub, os.path.join(sub_dir, name))
            plt.close(sub)
    return fig


def plot_phased_kinetics(
    config: "SimulationConfig",
    phase_files: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
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
# THESIS PANELS (composite multi-subplot figures; each panel also saved separately)
# =============================================================================
# One drawing function per metric, shared by the composite panel and the separate
# single-column figures, so the two can never drift apart. Each draws the data, the
# y-axis label and the x-axis label onto ``ax``; titles are added by the caller.

class _MetricCtx:
    """Inputs shared by every metric drawer of one ``(stats, structural, config)`` triple."""

    def __init__(self, stats, structural, config, show_individual, individual_alpha,
                 shared_legend=False):
        config = config or {}
        self.stats = stats
        self.structural = structural
        self.timestep = config.get('timestep', 1e-4)
        self.times_us, self.time_label = _time_axis(
            _steps_to_us(np.asarray(stats['times']), self.timestep))
        self.n_replicas = stats.get('n_replicas', 1)
        self.show_individual = show_individual
        self.individual_alpha = individual_alpha
        # In the composite panel the identical "Mean (N) / ± 1 SD" key of the single-series
        # panels is drawn once below the grid instead of in every panel.
        self.shared_legend = shared_legend


def _magnitude_scale(values) -> Tuple[float, str]:
    """``(factor, label prefix)`` for data far below 1, e.g. ``(1e5, '10⁻⁵ ')``.

    Folds the power of ten into the axis label ("Pressure (10⁻⁵ kJ/(mol·nm³))") instead
    of matplotlib's floating "1e-5" offset text above the axis.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    peak = float(np.max(np.abs(v))) if v.size else 0.0
    if peak == 0.0 or peak >= 1e-2:
        return 1.0, ""
    exp = int(np.floor(np.log10(peak)))
    return 10.0 ** (-exp), f"$10^{{{exp}}}$ "


def _with_unit_prefix(ylabel: str, prefix: str) -> str:
    """Insert ``prefix`` at the start of the parenthesised unit of ``ylabel``."""
    if not prefix:
        return ylabel
    if ylabel.endswith(")") and "(" in ylabel:
        head, unit = ylabel.split("(", 1)
        return f"{head}({prefix}{unit}"
    return f"{ylabel} ({prefix.strip()})"


def _draw_simple_band(ax, ctx: _MetricCtx, key: str, color: str, ylabel: str) -> None:
    """Mean ± SD of ``stats[key_mean]`` over time (the single-series metrics)."""
    mean_key, std_key = f'{key}_mean', f'{key}_std'
    if mean_key in ctx.stats:
        scale, prefix = _magnitude_scale(ctx.stats[mean_key])
        all_data = _ensemble_all_trace(ctx.stats, ctx.structural, key)
        if _ensemble_plot_with_band(ax, ctx.times_us,
                                    np.asarray(ctx.stats[mean_key]) * scale,
                                    np.asarray(ctx.stats[std_key]) * scale,
                                    color, ctx.n_replicas,
                                    None if all_data is None else np.asarray(all_data) * scale,
                                    ctx.show_individual, ctx.individual_alpha) \
                and not ctx.shared_legend:
            ax.legend(loc='best')
        ylabel = _with_unit_prefix(ylabel, prefix)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(ctx.time_label)
    ax.set_ylabel(ylabel)


def _draw_energy(ax, ctx):
    _draw_simple_band(ax, ctx, 'energy', 'tab:red', "Energy (kJ/mol)")


def _draw_pressure(ax, ctx):
    _draw_simple_band(ax, ctx, 'pressure', 'tab:green', "Pressure (kJ/(mol·nm³))")


def _draw_bonds(ax, ctx):
    _draw_simple_band(ax, ctx, 'bonds', 'tab:blue', "Number of bonds")


def _draw_n_topologies(ax, ctx):
    # A topology is either an agglomerate or a free particle (its own one-particle topology).
    _draw_simple_band(ax, ctx, 'n_clusters', 'tab:purple', "Agglomerates and\nfree particles")


def _draw_avg_size(ax, ctx):
    _draw_simple_band(ax, ctx, 'avg_cluster', 'tab:olive', "Average agglomerate\nsize (particles)")


def _draw_largest_size(ax, ctx):
    _draw_simple_band(ax, ctx, 'largest_cluster', 'tab:orange',
                      "Largest agglomerate\n(particles)")


def _draw_particle_counts(ax, ctx):
    colors = {'qt': 'blue', 'ft': 'red', 'qtc': 'darkblue', 'ftc': 'darkred'}
    labels = {'qt': 'Qt (free)', 'ft': 'Ft (free)',
              'qtc': 'Qt (agglomerated)', 'ftc': 'Ft (agglomerated)'}
    has_data = False
    for ptype in ('qt', 'ft', 'qtc', 'ftc'):
        key_mean, key_std = f'{ptype}_count_mean', f'{ptype}_count_std'
        if key_mean in ctx.stats:
            mean = np.asarray(ctx.stats[key_mean])
            std = np.asarray(ctx.stats[key_std])
            ax.plot(ctx.times_us, mean, color=colors[ptype], label=labels[ptype])
            ax.fill_between(ctx.times_us, mean - std, mean + std,
                            color=colors[ptype], alpha=0.2, linewidth=0)
            has_data = True
    if has_data:
        # The middle-right of the panel stays empty once the counts have settled; lifted a
        # little so the last entry clears the agglomerated-Qt plateau.
        ax.legend(loc='center right', bbox_to_anchor=(1.0, 0.58))
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(ctx.time_label)
    ax.set_ylabel("Particles")


def _draw_size_categories(ax, ctx):
    st = ctx.structural
    time_label = ctx.time_label
    if st and 'size_fractions_times' in st and 'size_fractions_category_names' in st:
        sc_times, time_label = _time_axis(
            _steps_to_us(np.asarray(st['size_fractions_times']), ctx.timestep))
        category_names = list(st['size_fractions_category_names'])
        fractions = []
        for cat_name in category_names:
            mean_key = f'size_frac_{_size_category_key(cat_name)}_mean'
            fractions.append(np.asarray(st[mean_key]) if mean_key in st
                             else np.zeros(len(sc_times)))
        colors = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple"]
        ax.stackplot(sc_times, *fractions, labels=category_names,
                     colors=colors[:len(category_names)], alpha=0.8, linewidth=0)
        ax.set_ylim([0, 1])
        # The stack fills the whole axes, so the legend always covers data: white background.
        # Entries listed top to bottom, in the order the bands are stacked.
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='upper right', frameon=True,
                  facecolor='white', edgecolor='none', framealpha=0.9)
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(time_label)
    ax.set_ylabel("Fraction of particles")


def _draw_struct_band(ax, ctx, time_key, base, color, ylabel, *, ylim=None, ref_line=None):
    """Mean ± SD of a structural time series (``{base}_mean/_std/_all``)."""
    times, mean, std, all_data, time_label = _ensemble_struct_ts(
        ctx.structural, ctx.timestep, time_key, f'{base}_mean', f'{base}_std', f'{base}_all')
    if times is not None and mean is not None:
        if _ensemble_plot_with_band(ax, times, mean, std, color, ctx.n_replicas,
                                    all_data, ctx.show_individual, ctx.individual_alpha):
            if ref_line is not None:
                ax.axhline(ref_line, color='gray', linestyle='--', linewidth=0.5,
                           alpha=0.7, label='_nolegend_')
            if not ctx.shared_legend:
                ax.legend(loc='best')
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(time_label)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)


def _draw_mean_rg(ax, ctx):
    _draw_struct_band(ax, ctx, 'morphology_times', 'mean_rg', 'tab:blue', "Mean Rg (nm)")


def _draw_composition(ax, ctx):
    _draw_struct_band(ax, ctx, 'composition_times', 'mean_composition', 'tab:blue',
                      "Mean Qt fraction", ylim=[0, 1], ref_line=0.5)


def _draw_coordination(ax, ctx):
    st = ctx.structural
    t_qt, m_qt, s_qt, all_qt, time_label = _ensemble_struct_ts(
        st, ctx.timestep, 'contacts_times', 'mean_coord_qt_mean', 'mean_coord_qt_std',
        'mean_coord_qt_all')
    t_ft, m_ft, s_ft, all_ft, _ = _ensemble_struct_ts(
        st, ctx.timestep, 'contacts_times', 'mean_coord_ft_mean', 'mean_coord_ft_std',
        'mean_coord_ft_all')
    plotted = False
    # Legend order Qt, Ft, ± 1 SD; in the panel the SD entry is in the shared legend.
    if t_qt is not None and m_qt is not None:
        plotted |= _ensemble_plot_with_band(
            ax, t_qt, m_qt, s_qt, SPECIES_COLOR_QT, ctx.n_replicas,
            all_qt, ctx.show_individual, ctx.individual_alpha, label='Qt',
            band_label='_nolegend_')
    if t_ft is not None and m_ft is not None:
        plotted |= _ensemble_plot_with_band(
            ax, t_ft, m_ft, s_ft, SPECIES_COLOR_FT, ctx.n_replicas,
            all_ft, ctx.show_individual, ctx.individual_alpha, label='Ft',
            band_label='_nolegend_')
    if plotted:
        handles, labels = ax.get_legend_handles_labels()
        if ctx.n_replicas > 1 and not ctx.shared_legend:
            # The band applies to both species, so its key entry is neutral grey.
            handles.append(plt.Rectangle((0, 0), 1, 1, color='0.2', alpha=0.3, linewidth=0))
            labels.append("± 1 SD")
        ax.legend(handles, labels, loc='best')
    else:
        _ensemble_show_no_data(ax)
    ax.set_xlabel(time_label)
    ax.set_ylabel("Mean coordination")


def _draw_coord_distribution(ax, ctx):
    # Final-frame per-particle coordination, pooled over replicas and rescaled to a
    # per-replica count so the y axis is comparable with the single-run figure.
    st = ctx.structural
    ylabel = "Mean count per replica"
    if st is not None and 'final_coord_dist_qt' in st:
        n_contrib = int(np.atleast_1d(st.get('final_coord_dist_n_replicas', [1]))[0]) or 1
        _plot_coord_distribution(ax, st.get('final_coord_dist_qt', []),
                                 st.get('final_coord_dist_ft', []),
                                 weight=1.0 / n_contrib, ylabel=ylabel)
    else:
        # Ensembles analysed before final_coord_dist_* was added to ensemble_structural.npz.
        _ensemble_show_no_data(ax)
        ax.set_xlabel("Coordination number")
        ax.set_ylabel(ylabel)


# (file name, panel title, drawer, separate-figure height in mm), in grid order (4 x 3).
_METRICS = (
    ("energy", "Potential energy", _draw_energy, 45),
    ("pressure", "Pressure", _draw_pressure, 45),
    ("bonds", "Number of bonds", _draw_bonds, 45),
    ("particle_counts", "Particle counts", _draw_particle_counts, 45),
    ("topologies", "Agglomerates and free particles", _draw_n_topologies, 45),
    ("avg_cluster_size", "Average agglomerate size", _draw_avg_size, 45),
    ("largest_cluster_size", "Largest agglomerate", _draw_largest_size, 45),
    ("cluster_size_distribution", "Particles by agglomerate size", _draw_size_categories, 45),
    ("mean_rg", "Mean radius of gyration", _draw_mean_rg, 45),
    ("mean_composition", "Mean agglomerate composition", _draw_composition, 45),
    ("coordination", "Coordination number", _draw_coordination, 45),
    ("coordination_distribution", "Final coordination distribution",
     _draw_coord_distribution, 55),
)
_METRICS_BY_NAME = {m[0]: m for m in _METRICS}


def _metric_figure(name: str, ctx: _MetricCtx, figsize=None) -> plt.Figure:
    """One metric as a separate, title-free single-column figure."""
    _, _, draw, height_mm = _METRICS_BY_NAME[name]
    fig, ax = plt.subplots(figsize=figsize or figsize_mm(89, height_mm), layout='constrained')
    draw(ax, ctx)
    return fig


@_nature_style
def plot_metrics_panel(
    stats: Dict,
    structural: Dict,
    config: Dict,
    *,
    show_individual: bool = False,
    individual_alpha: float = 0.3,
    figsize: Optional[Tuple[float, float]] = None,
    save_path_base: Optional[str] = None,
) -> plt.Figure:
    """Curated 12-metric thesis panel (4x3 grid, 183 mm wide).

    Takes the ``(stats, structural, config)`` triple, which a single run produces via
    ``analysis.build_single_run_plotting_data`` and an ensemble via
    ``EnsembleSimulation.to_plotting_format()`` — hence "metrics", not "ensemble".

    Layout:
        Row 1: Potential energy | Pressure | Number of bonds
        Row 2: Particle counts | Agglomerates and free particles | Average agglomerate size
        Row 3: Largest agglomerate | Particles by agglomerate size | Mean radius of gyration
        Row 4: Mean agglomerate composition | Coordination number | Final coordination distribution

    With ``save_path_base`` the panel is saved as ``{base}.pdf/.svg/.png`` and every one
    of the 12 panels additionally as a separate, title-free 89 mm figure in
    ``{base}_subfigures/{name}.pdf/.svg/.png`` (names as in ``_METRICS``).

    The final-row histogram needs the ``final_coord_dist_*`` keys in
    ``ensemble_structural.npz``; ensembles analysed before those were added render it as
    "No data" (re-run ``scripts/analyze_ensemble.py`` on the directory to populate them).
    """
    print("\nGenerating ensemble thesis panel...")

    ctx = _MetricCtx(stats, structural, config, show_individual, individual_alpha,
                     shared_legend=True)
    fig, axes = plt.subplots(4, 3, figsize=figsize or figsize_mm(WIDTH_2COL_MM, 165),
                             layout='constrained')
    for ax, (_, title, draw, _) in zip(axes.flat, _METRICS):
        draw(ax, ctx)
        ax.set_title(title)
    # Every panel above the bottom row shares the time axis: x label on the bottom row only.
    for ax in axes[:-1].flat:
        ax.set_xlabel("")

    # The single-series panels share one "Mean (N) / ± 1 SD" key, drawn once below the grid.
    if ctx.n_replicas > 1:
        handles = [Line2D([0], [0], color='0.2'),
                   plt.Rectangle((0, 0), 1, 1, color='0.2', alpha=0.3, linewidth=0)]
        fig.legend(handles, [f"Mean (N={ctx.n_replicas})", "± 1 SD"],
                   loc='outside lower center', ncol=2)

    if save_path_base:
        _save_figure(fig, save_path_base)
        sub_dir = f"{save_path_base}_subfigures"
        sub_ctx = _MetricCtx(stats, structural, config, show_individual, individual_alpha)
        for name, *_ in _METRICS:
            sub = _metric_figure(name, sub_ctx)
            _save_figure(sub, os.path.join(sub_dir, name))
            plt.close(sub)
    return fig


@_nature_style
def plot_metrics_separately(
    stats: Dict,
    structural: Optional[Dict],
    config: Dict,
    *,
    show_individual: bool = False,
    individual_alpha: float = 0.3,
    figsize: Optional[Tuple[float, float]] = None,
    save_dir: Optional[str] = None,
    suffix: str = "",
) -> Dict[str, plt.Figure]:
    """Four standalone, **title-free** 89 × 45 mm figures for use outside the panel.

    Same ``(stats, structural, config)`` triple and the same drawing code as
    :func:`plot_metrics_panel` (whose ``save_path_base`` exports all 12 metrics this way),
    minus the titles, so each figure can carry its own caption in a thesis or paper.

    Figures (returned as ``{name: Figure}``, saved as ``{save_dir}/{name}{suffix}`` in
    ``.pdf``, ``.svg`` and ``.png``):

    ========================== ==================================================
    ``particle_counts``        free and agglomerated Qt/Ft over time
    ``avg_cluster_size``       average agglomerate size over time
    ``largest_cluster_size``   largest agglomerate over time
    ``cluster_size_distribution``  stacked particle fractions per agglomerate size
    ========================== ==================================================

    A metric whose keys are absent renders the usual "No data" placeholder rather than
    raising, so a partially-analysed ensemble still produces the remaining figures.
    """
    ctx = _MetricCtx(stats, structural, config, show_individual, individual_alpha)
    figures: Dict[str, plt.Figure] = {}
    for name in ("particle_counts", "avg_cluster_size", "largest_cluster_size",
                 "cluster_size_distribution"):
        figures[name] = _metric_figure(name, ctx, figsize=figsize)
    if save_dir:
        for name, fig in figures.items():
            _save_figure(fig, os.path.join(save_dir, f"{name}{suffix}"))
    return figures


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
