#!/usr/bin/env python3
"""Ensemble analysis CLI.

    python scripts/analyze_ensemble.py --ensemble-dir DIR --stride 10 [--parallel --n-workers 8]
    python scripts/analyze_ensemble.py compare -e "Label1=path1/" -e "Label2=path2/" -o out/

Thin wrapper: the analysis pipeline lives in qtft.ensemble.EnsembleSimulation and
the comparison helpers in qtft.comparison.
"""
import argparse
import os
import sys

# Make the qtft package importable when run as `python scripts/analyze_ensemble.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtft import EnsembleSimulation, NS_TO_US
from qtft.comparison import run_comparison


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ensemble simulation results and save to JSON/NPZ"
    )
    parser.add_argument("--ensemble-dir", required=True,
                        help="Path to ensemble output directory")
    parser.add_argument("--stride", type=int, default=10,
                        help="Stride for structural analysis (default: 10)")
    parser.add_argument("--parallel", action="store_true",
                        help="Enable parallel processing for structural analysis")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Number of parallel workers (default: min(n_replicas, cpu_count))")
    args = parser.parse_args()

    ensemble_dir = args.ensemble_dir.rstrip("/") + "/"

    print("=" * 60)
    print("ENSEMBLE ANALYSIS")
    print("=" * 60)
    print(f"Ensemble directory: {ensemble_dir}")
    print(f"Structural analysis stride: {args.stride}")
    if args.parallel:
        print("Parallel processing: ENABLED")
    print()

    # Drive the shared EnsembleSimulation pipeline so this CLI and a local
    # run_local() produce byte-identical ensemble_statistics.json /
    # ensemble_structural.npz. save_for_plotting() is the single source of truth
    # for the on-disk schema (no duplicated aggregation/writer logic here).
    ensemble = EnsembleSimulation.load(ensemble_dir)
    ensemble.collect_results(require_all=False)
    ensemble.compute_statistics()
    ensemble.compute_summary_metrics()
    ensemble.compute_structural_statistics(
        stride=args.stride, parallel=args.parallel, n_workers=args.n_workers
    )
    ensemble.save_for_plotting()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Output files in {ensemble_dir}:")
    print("  ensemble_statistics.json")
    print("  ensemble_structural.npz")
    print("  ensemble_config.json")
    print("  ensemble_state.json")
    print("\nDownload these for local plotting with Plot_Ensemble_Results.ipynb")

    if ensemble.summary_metrics:
        m = ensemble.summary_metrics
        print("\n" + "-" * 60)
        print("SUMMARY METRICS")
        print("-" * 60)
        print(f"  Replicas: {m['n_replicas']}")
        if 'final_bonds_mean' in m:
            print(f"  Final bonds: {m['final_bonds_mean']:.1f} \u00b1 {m['final_bonds_std']:.1f}")
        if 'final_largest_fraction_mean' in m:
            print(f"  Final largest cluster: {m['final_largest_fraction_mean']*100:.1f}% "
                  f"\u00b1 {m['final_largest_fraction_std']*100:.1f}% of particles")
        if 'half_time_mean' in m:
            ht = m['half_time_mean'] * NS_TO_US
            ht_std = m['half_time_std'] * NS_TO_US
            print(f"  Half-time: {ht:.2f} \u00b1 {ht_std:.2f} \u00b5s")

def compare_main():
    """Entry point for compare subcommand."""
    parser = argparse.ArgumentParser(
        description="Compare multiple ensemble simulation results and save data for plotting"
    )
    parser.add_argument(
        "--ensemble", "-e",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Ensemble to include in comparison (format: 'Label=path/to/ensemble/'). "
             "Can be specified multiple times."
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for comparison results"
    )
    
    args = parser.parse_args()
    
    run_comparison(
        ensemble_specs=args.ensemble,
        output_dir=args.output,
    )



if __name__ == "__main__":
    # Check if 'compare' subcommand is used
    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        # Remove 'compare' from argv and run comparison
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        compare_main()
    else:
        main()
