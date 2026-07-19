"""qtft.comparison -- cross-ensemble comparison helpers (matplotlib-free).

Load multiple finished ensembles, diff their parameters, and build/save the
comparison data structure consumed by qtft.plotting.plot_comparison_* functions.
"""
import logging
import json
import os

import numpy as np

from .config import _steps_to_us, format_duration
from .analysis import _load_ensemble_files

logger = logging.getLogger(__name__)


# =============================================================================
# ENSEMBLE COMPARISON FUNCTIONS
# =============================================================================


def load_ensemble_for_comparison(ensemble_dir: str, label: str) -> dict:
    """
    Load a single ensemble's pre-computed statistics for comparison.
    
    Parameters
    ----------
    ensemble_dir : str
        Path to ensemble directory containing JSON/NPZ files
    label : str
        Label for this ensemble (used in plots/legends)
    
    Returns
    -------
    dict
        Ensemble data with config, statistics, structural data, and label
    
    Notes
    -----
    Per-replica time-series arrays (*_all keys like bonds_all, fraction_bound_all)
    are merged into stats for overlay of individual replica points on bar charts.
    Structural analysis data (morphology, spatial, contacts, composition) is
    loaded into the 'structural' dict.
    """
    stats, npz, config, meta = _load_ensemble_files(ensemble_dir)
    if not meta['has_config']:
        raise FileNotFoundError(f"Config file not found: {meta['config_path']}")

    structural_path = meta['npz_path'] if meta['has_npz'] else None

    # Per-replica time-series keys that go into stats (the rest of the NPZ is structural).
    time_series_all_keys = {
        'bonds_all', 'energy_all', 'pressure_all', 'n_clusters_all',
        'largest_cluster_all', 'fraction_bound_all', 'avg_cluster_all',
        'cumulative_reactions_all',
        'qt_count_all', 'ft_count_all', 'qtc_count_all', 'ftc_count_all',
        'total_count_all',
    }
    structural = {}
    for key, value in npz.items():
        if key.endswith('_all') and key in time_series_all_keys:
            stats[key] = value
        else:
            structural[key] = value

    # Get timestep for time conversion
    timestep = config.get('timestep', 0.001)

    # Convert times (step numbers) to microseconds
    if 'times' in stats:
        times_us = _steps_to_us(stats['times'], timestep)
    else:
        times_us = np.array([])

    return {
        'label': label,
        'dir': ensemble_dir,
        'config': config,
        'stats': stats,
        'structural': structural,
        'structural_path': structural_path,
        'times_us': times_us,
        'timestep': timestep,
        'n_replicas': stats.get('n_replicas', len(stats.get('available_replicas', []))),
    }


def compare_ensembles(ensembles_dict: dict) -> dict:
    """
    Build comparison data structure from multiple ensembles.
    
    Parameters
    ----------
    ensembles_dict : dict
        Dictionary mapping labels to ensemble directories
        Example: {"200 Ft": "ensemble_200ft/", "400 Ft": "ensemble_400ft/"}
    
    Returns
    -------
    dict
        Comparison data structure with all ensembles' data
    """
    logger.info("=" * 60)
    logger.info("LOADING ENSEMBLES FOR COMPARISON")
    logger.info("=" * 60)
    
    ensembles = {}
    
    for label, ensemble_dir in ensembles_dict.items():
        logger.info(f"\nLoading: {label}")
        logger.info(f"  Directory: {ensemble_dir}")
        try:
            data = load_ensemble_for_comparison(ensemble_dir, label)
            ensembles[label] = data
            struct_status = "with structural data" if data["structural"] else "basic stats only"
            logger.info(f"  ✓ Loaded {data['n_replicas']} replicas, {len(data['times_us'])} time points ({struct_status})")
        except Exception as e:
            logger.error(f"  ✗ Failed to load: {e}")
            continue
    
    if len(ensembles) < 2:
        raise ValueError(f"Need at least 2 ensembles for comparison, got {len(ensembles)}")
    
    # Detect parameter differences
    param_diffs = _detect_parameter_differences(ensembles)
    
    # Build comparison structure
    comparison = {
        'ensembles': ensembles,
        'labels': list(ensembles.keys()),
        'n_ensembles': len(ensembles),
        'parameter_differences': param_diffs,
        'metadata': {
            'created': str(np.datetime64('now')),
        }
    }
    
    return comparison


def _detect_parameter_differences(ensembles: dict) -> dict:
    """Detect which parameters differ between ensembles."""
    if len(ensembles) < 2:
        return {}
    
    # Parameters to check
    params_to_check = [
        ('n_qt', lambda c: c.get('n_qt')),
        ('n_ft', lambda c: c.get('n_ft')),
        ('qt_radius', lambda c: c.get('qt', {}).get('radius')),
        ('ft_radius', lambda c: c.get('ft', {}).get('radius')),
        ('qt_diffusion', lambda c: c.get('qt', {}).get('diffusion')),
        ('ft_diffusion', lambda c: c.get('ft', {}).get('diffusion')),
        ('binding_radius', lambda c: c.get('topology', {}).get('binding_radius')),
        ('kon', lambda c: c.get('topology', {}).get('kon')),
        ('k_bond', lambda c: c.get('topology', {}).get('k_bond')),
        ('epsilon_QtQt', lambda c: c.get('lj', {}).get('epsilon_QtQt')),
        ('epsilon_FtFt', lambda c: c.get('lj', {}).get('epsilon_FtFt')),
        ('epsilon_QtFt', lambda c: c.get('lj', {}).get('epsilon_QtFt')),
        ('epsilon_QtCFtC', lambda c: c.get('lj', {}).get('epsilon_QtCFtC')),
        ('potential_type', lambda c: c.get('lj', {}).get('potential_type')),
        ('box_size', lambda c: tuple(c.get('box_size', []))),
        ('temperature', lambda c: c.get('temperature')),
        ('timestep', lambda c: c.get('timestep')),
        ('n_steps', lambda c: c.get('n_steps')),
    ]
    
    diffs = {}
    labels = list(ensembles.keys())
    
    for param_name, getter in params_to_check:
        values = {}
        for label in labels:
            config = ensembles[label]['config']
            values[label] = getter(config)
        
        # Check if values differ
        unique_values = set(str(v) for v in values.values())
        if len(unique_values) > 1:
            diffs[param_name] = values
    
    return diffs


def print_parameter_differences(comparison: dict):
    """Print parameter differences between ensembles."""
    print("\n" + "=" * 60)
    print("PARAMETER DIFFERENCES")
    print("=" * 60)
    
    diffs = comparison.get('parameter_differences', {})
    
    if not diffs:
        print("No parameter differences detected (ensembles have identical configs)")
        return
    
    for param, values in diffs.items():
        print(f"\n{param}:")
        for label, value in values.items():
            print(f"  {label}: {value}")


def save_comparison_data(comparison: dict, output_dir: str):
    """
    Save comparison data to JSON and NPZ files.
    
    Parameters
    ----------
    comparison : dict
        Comparison data from compare_ensembles()
    output_dir : str
        Output directory
    """
    output_dir = output_dir.rstrip("/") + "/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare JSON-serializable data
    json_data = {
        'labels': comparison['labels'],
        'n_ensembles': comparison['n_ensembles'],
        'parameter_differences': comparison['parameter_differences'],
        'metadata': comparison['metadata'],
        'ensembles': {}
    }
    
    # Prepare NPZ data
    npz_data = {}
    
    for label, ens in comparison['ensembles'].items():
        # JSON: config and scalar values
        json_data['ensembles'][label] = {
            'config': ens['config'],
            'n_replicas': ens['n_replicas'],
            'timestep': ens['timestep'],
        }
        
        # Add summary metrics if available
        if 'summary' in ens['stats']:
            json_data['ensembles'][label]['summary'] = ens['stats']['summary']
        
        # NPZ: time series arrays
        safe_label = label.replace(" ", "_").replace("/", "_")
        npz_data[f'{safe_label}_times_us'] = ens['times_us']
        
        # Add all mean/std time series
        for key in ['bonds', 'energy', 'pressure', 'n_clusters', 'largest_cluster', 
                    'fraction_bound', 'avg_cluster', 'cumulative_reactions',
                    'qt_count', 'ft_count', 'qtc_count', 'ftc_count']:
            mean_key = f'{key}_mean'
            std_key = f'{key}_std'
            if mean_key in ens['stats']:
                npz_data[f'{safe_label}_{mean_key}'] = ens['stats'][mean_key]
            if std_key in ens['stats']:
                npz_data[f'{safe_label}_{std_key}'] = ens['stats'][std_key]
    
    # Save JSON
    json_path = f"{output_dir}comparison_data.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    logger.info(f"✓ Saved comparison metadata to {json_path}")
    
    # Save NPZ
    npz_path = f"{output_dir}comparison_timeseries.npz"
    np.savez_compressed(npz_path, **npz_data)
    logger.info(f"✓ Saved comparison time series to {npz_path}")


def load_comparison_data(output_dir: str) -> dict:
    """
    Load comparison data from JSON and NPZ files.
    
    Parameters
    ----------
    output_dir : str
        Directory containing comparison_data.json and comparison_timeseries.npz
    
    Returns
    -------
    dict
        Comparison data structure
    """
    output_dir = output_dir.rstrip("/") + "/"
    
    # Load JSON
    json_path = f"{output_dir}comparison_data.json"
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Load NPZ
    npz_path = f"{output_dir}comparison_timeseries.npz"
    npz_data = {}
    with np.load(npz_path, allow_pickle=True) as data:
        for key in data.files:
            npz_data[key] = data[key]
    
    # Reconstruct comparison structure
    comparison = {
        'labels': json_data['labels'],
        'n_ensembles': json_data['n_ensembles'],
        'parameter_differences': json_data['parameter_differences'],
        'metadata': json_data['metadata'],
        'ensembles': {}
    }
    
    for label in json_data['labels']:
        safe_label = label.replace(" ", "_").replace("/", "_")
        ens_json = json_data['ensembles'][label]
        
        # Reconstruct ensemble data
        ens = {
            'label': label,
            'config': ens_json['config'],
            'n_replicas': ens_json['n_replicas'],
            'timestep': ens_json['timestep'],
            'times_us': npz_data.get(f'{safe_label}_times_us', np.array([])),
            'stats': {},
        }
        
        # Add summary if available
        if 'summary' in ens_json:
            ens['stats']['summary'] = ens_json['summary']
        
        # Reconstruct stats from NPZ
        for key in ['bonds', 'energy', 'pressure', 'n_clusters', 'largest_cluster', 
                    'fraction_bound', 'avg_cluster', 'cumulative_reactions',
                    'qt_count', 'ft_count', 'qtc_count', 'ftc_count']:
            mean_key = f'{key}_mean'
            std_key = f'{key}_std'
            npz_mean_key = f'{safe_label}_{mean_key}'
            npz_std_key = f'{safe_label}_{std_key}'
            if npz_mean_key in npz_data:
                ens['stats'][mean_key] = npz_data[npz_mean_key]
            if npz_std_key in npz_data:
                ens['stats'][std_key] = npz_data[npz_std_key]
        
        comparison['ensembles'][label] = ens
    
    return comparison


def run_comparison(ensemble_specs: list, output_dir: str):
    """
    Run full ensemble comparison workflow (data only, no plotting).
    
    Loads ensembles, detects parameter differences, prints summaries,
    and saves comparison data to JSON/NPZ files for later plotting
    in notebooks.
    
    Parameters
    ----------
    ensemble_specs : list
        List of "Label=path" strings
    output_dir : str
        Output directory for comparison results
    
    Returns
    -------
    dict
        Comparison data structure
    """
    # Parse ensemble specifications
    ensembles_dict = {}
    for spec in ensemble_specs:
        if '=' not in spec:
            raise ValueError(f"Invalid ensemble spec '{spec}'. Use format: 'Label=path/to/ensemble/'")
        label, path = spec.split('=', 1)
        ensembles_dict[label] = path
    
    # Load and compare ensembles
    comparison = compare_ensembles(ensembles_dict)
    
    # Print parameter differences
    print_parameter_differences(comparison)
    
    # Print detailed summary for each ensemble
    logger.info("\n" + "=" * 60)
    logger.info("ENSEMBLE SUMMARIES")
    logger.info("=" * 60)
    
    for label in comparison['labels']:
        ens = comparison['ensembles'][label]
        logger.info(f"\n{label}:")
        logger.info(f"  Directory: {ens.get('dir', 'N/A')}")
        logger.info(f"  Replicas: {ens['n_replicas']}")
        logger.info(f"  Time points: {len(ens['times_us'])}")
        if len(ens['times_us']) > 0:
            logger.info(f"  Duration: {format_duration(ens['times_us'][-1] * 1e3)}")
        
        # Print key config parameters
        config = ens['config']
        logger.info(f"  Particles: {config.get('n_qt', '?')} Qt + {config.get('n_ft', '?')} Ft")
        
        # Print final values
        stats = ens['stats']
        if 'bonds_mean' in stats and len(stats['bonds_mean']) > 0:
            logger.info(f"  Final bonds: {stats['bonds_mean'][-1]:.1f} ± {stats.get('bonds_std', [0])[-1]:.1f}")
        if 'fraction_bound_mean' in stats and len(stats['fraction_bound_mean']) > 0:
            fb = stats['fraction_bound_mean'][-1]
            fb_std = stats.get('fraction_bound_std', [0])[-1]
            logger.info(f"  Final fraction bound: {fb*100:.1f}% ± {fb_std*100:.1f}%")
    
    # Save comparison data
    logger.info("\n" + "=" * 60)
    logger.info("SAVING COMPARISON DATA")
    logger.info("=" * 60)
    save_comparison_data(comparison, output_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nOutput files saved to: {output_dir}")
    logger.info("\nTo plot in notebook, use:")
    logger.info(f"  import qtft.comparison as ae")
    logger.info(f"  import qtft.plotting as plotting")
    logger.info(f"  comparison = ae.load_comparison_data('{output_dir}')")
    logger.info(f"  plotting.plot_comparison_summary(comparison)")

    return comparison


def build_comparison_table(comparison: dict):
    """
    Build a final-state comparison table across ensembles.

    Rows are metrics (final-state aggregation followed by kinetics, morphology and composition,
    with units appended to the metric name); columns are the ensemble labels. Each cell is a
    formatted ``"mean ± SD"`` string. Reuses the same per-ensemble row builder as the
    single-ensemble table (``qtft.analysis._final_state_rows``) so formatting and metric
    definitions stay identical.

    Parameters
    ----------
    comparison : dict
        Comparison data structure from :func:`compare_ensembles` (each ensemble must carry
        ``stats`` with the ``{key}_mean``/``{key}_std`` series and a ``summary`` sub-dict, plus
        an optional ``structural`` dict for the morphology/composition rows).

    Returns
    -------
    pandas.DataFrame
        Indexed by metric name (unit appended, e.g. ``"Half-time t₅₀ (µs)"``), one column per
        ensemble label.
    """
    import pandas as pd

    from .analysis import _final_state_rows

    def metric_name(metric: str, unit: str) -> str:
        return metric if unit in ("—", "", None) else f"{metric} ({unit})"

    columns = {}
    row_order = []
    seen = set()

    for label in comparison["labels"]:
        ens = comparison["ensembles"][label]
        rows = _final_state_rows(ens.get("stats", {}), ens.get("config"), ens.get("structural"))
        col = {}
        for metric, value, unit in rows:
            name = metric_name(metric, unit)
            col[name] = value
            if name not in seen:
                seen.add(name)
                row_order.append(name)
        columns[label] = col

    # Assemble into a DataFrame (rows in first-seen order, columns = ensemble labels)
    data = {label: [columns[label].get(name, "—") for name in row_order]
            for label in comparison["labels"]}
    df = pd.DataFrame(data, index=row_order)
    df.index.name = "Metric"
    return df


