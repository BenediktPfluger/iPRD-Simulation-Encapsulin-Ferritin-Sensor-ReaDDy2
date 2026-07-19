#!/usr/bin/env python
"""
Timestep & reachable-time calibration sweep for the "soft" potential mode.

The soft (harmonic-repulsion) potential removes the r^-12 Lennard-Jones wall that
forces the tiny production timestep, so a much larger dt can be numerically stable.
How large depends on both the potential softness AND how far particles diffuse per
step (sqrt(2*D*dt) must stay small relative to particle size / the binding radius).

This "measure-first" tool sweeps a grid of (timestep, diffusion) in soft mode, runs a
short simulation per cell, and reports, per cell:

  - stability      : did the run finish with all-finite positions, and do bonded
                     distances stay near the equilibrium bond length r0?
  - disp           : per-step displacement sqrt(2*D*dt) for Qt and Ft, and its ratio
                     to the particle radius and to the binding radius (the diffusion
                     criterion for reaction detection);
  - reaction        : per-step probabilities p = 1 - exp(-rate*dt) for kon/koff (they
                     saturate toward 1 at large dt -> reactions fire on first contact,
                     distorting kinetics) and the measured bound fraction reached. The
                     summary reports the largest "faithful" kon/koff at each dt
                     (-ln(1-p_target)/dt) and suggests a down-scaled rate when it saturates;
  - reachable time  : n_steps(budget) * dt in microseconds, i.e. how long a run of a
                     fixed STEP BUDGET would reach at this dt.

It then prints the largest stable dt per diffusion setting and the reachable time for
the chosen step budget. Nothing here changes production defaults; it only measures.

The sweep manages its own output directory (default: a temp dir, removed afterwards)
so the D-sweep does not collide under the auto-filename convention (which does not
encode diffusion).

Usage:
    python scripts/calibrate_timestep.py \
        --timesteps 0.05 0.5 2 5 \
        --diffusion-scales 1 0.1 0.01 \
        --qt-diffusion 0.5 --ft-diffusion 1.0 \
        --n-qt 60 --n-ft 60 --n-steps 4000 --box 300 \
        --step-budget 2000000 --output-csv calibration.csv

--timesteps are in PICOSECONDS. --diffusion-scales multiply the base Qt/Ft diffusion
(and cluster diffusion). A base SimulationConfig can be supplied with --config; any
CLI value given overrides it. potential_type is always forced to "soft".
"""

import argparse
import math
import os
import shutil
import sys
import tempfile

import numpy as np

# Make the qtft package importable when run as `python scripts/calibrate_timestep.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qtft
import qtft.analysis as analysis
from qtft.config import _steps_to_us


def _min_image(delta, box):
    """Minimum-image displacement under periodic boundaries."""
    box = np.asarray(box)
    return delta - box * np.round(delta / box)


def measure_bond_lengths(h5_file, config, stride=1):
    """Mean/std of bonded-pair distances (nm) across frames, via minimum image.

    Returns (mean, std, n_samples). n_samples == 0 when no bonds ever formed.
    """
    data = analysis._extract_frame_data(h5_file, config, stride=stride, verbose=False)
    box = config.box_size
    lengths = []
    for positions, topo_particles, topo_edges in zip(
        data["positions"], data["topology_particles"], data["topology_edges"]
    ):
        if positions is None or len(positions) == 0:
            continue
        for pidx, edges in zip(topo_particles, topo_edges):
            for edge in edges:
                a, b = int(edge[0]), int(edge[1])
                # edges use topology-local vertex indices -> map to frame position index
                if a < len(pidx) and b < len(pidx):
                    ia, ib = pidx[a], pidx[b]
                    if ia < len(positions) and ib < len(positions):
                        d = _min_image(positions[ia] - positions[ib], box)
                        lengths.append(float(np.linalg.norm(d)))
    if not lengths:
        return (float("nan"), float("nan"), 0)
    arr = np.asarray(lengths)
    return (float(arr.mean()), float(arr.std()), arr.size)


def positions_all_finite(h5_file, config, stride=1):
    """True if every particle position across the trajectory is finite."""
    data = analysis._extract_frame_data(h5_file, config, stride=stride, verbose=False)
    for positions in data["positions"]:
        if positions is None or len(positions) == 0:
            continue
        if not np.all(np.isfinite(np.asarray(positions))):
            return False
    return True


def evaluate_cell(base_config, dt_ns, d_scale, stage_dir, run_kwargs, p_target=0.1):
    """Run one (dt, diffusion-scale) cell and return a metrics dict."""
    cfg = qtft.SimulationConfig.from_dict(base_config.to_dict())
    cfg.lj.potential_type = "soft"
    cfg.timestep = dt_ns
    # Scale diffusion (free + cluster) by d_scale.
    for pc in (cfg.qt, cfg.ft):
        pc.diffusion = pc.diffusion * d_scale
        if pc.cluster_diffusion is not None:
            pc.cluster_diffusion = pc.cluster_diffusion * d_scale
    # Route output into the sweep's own staging dir (avoids filename collisions).
    tag = f"dt{dt_ns * 1000:.4g}ps_dscale{d_scale:g}"
    cfg.output_file = os.path.join(stage_dir, f"{tag}.h5")

    r0 = cfg.equilibrium_bond_length          # r_qt + r_ft
    kon = cfg.topology.kon
    koff = cfg.topology.koff
    binding_radius = cfg.topology.binding_radius

    # Diffusion criterion: per-step RMS displacement sqrt(2*D*dt) (D in nm^2/ns, dt in ns)
    disp_qt = math.sqrt(2.0 * cfg.qt.diffusion * dt_ns)
    disp_ft = math.sqrt(2.0 * cfg.ft.diffusion * dt_ns)
    disp_max = max(disp_qt, disp_ft)
    disp_over_bind = disp_max / binding_radius if binding_radius > 0 else float("inf")

    # Per-step reaction probabilities p = 1 - exp(-rate*dt). As dt grows, p -> 1 and the
    # reaction becomes contact-instantaneous, so a fast rate can no longer be resolved.
    p_bind = 1.0 - math.exp(-kon * dt_ns)
    p_break = 1.0 - math.exp(-koff * dt_ns) if koff > 0 else 0.0
    # Largest rate whose per-step probability stays <= p_target (the stochastic regime):
    # rate_max = -ln(1 - p_target) / dt. A faster intended rate needs a smaller dt.
    rate_cap = -math.log(1.0 - p_target) / dt_ns
    rate_stochastic = (p_bind <= p_target) and (p_break <= p_target)

    metrics = {
        "dt_ps": dt_ns * 1000.0,
        "d_scale": d_scale,
        "qt_D": cfg.qt.diffusion,
        "ft_D": cfg.ft.diffusion,
        "disp_qt": disp_qt,
        "disp_ft": disp_ft,
        "disp_over_r_ft": disp_ft / cfg.ft.radius if cfg.ft.radius > 0 else float("inf"),
        "disp_over_bind": disp_over_bind,
        "p_bind": p_bind,
        "p_break": p_break,
        "kon_max": rate_cap,
        "koff_max": rate_cap,
        "rate_stochastic": rate_stochastic,
        "completed": False,
        "finite": False,
        "bond_mean": float("nan"),
        "bond_std": float("nan"),
        "n_bonds_seen": 0,
        "frac_bound_qt": float("nan"),
        "frac_bound_ft": float("nan"),
        "stable": False,
        "note": "",
    }

    try:
        qtft.run_one(cfg, skip_equilibration=True, show_progress=False, **run_kwargs)
        metrics["completed"] = True
    except Exception as e:  # integrator blow-up typically surfaces as an exception
        metrics["note"] = f"run failed: {type(e).__name__}: {str(e)[:80]}"
        return metrics, cfg

    try:
        metrics["finite"] = positions_all_finite(cfg.output_file, cfg)
        bmean, bstd, bn = measure_bond_lengths(cfg.output_file, cfg)
        metrics["bond_mean"], metrics["bond_std"], metrics["n_bonds_seen"] = bmean, bstd, bn
        kin = analysis.get_binding_kinetics(cfg.output_file, cfg)
        metrics["frac_bound_qt"] = float(np.asarray(kin["fraction_bound_qt"])[-1])
        metrics["frac_bound_ft"] = float(np.asarray(kin["fraction_bound_ft"])[-1])
    except Exception as e:
        metrics["note"] = f"analysis failed: {type(e).__name__}: {str(e)[:80]}"
        return metrics, cfg

    # Stability verdict: finished, all-finite, and (if bonds formed) bonded distances
    # stay near r0 without excessive spread. No bonds formed -> can't confirm bond
    # stability, but a finite run is still "stable"; flagged in the note.
    stable = metrics["completed"] and metrics["finite"]
    if bn > 0:
        near_r0 = 0.5 * r0 <= bmean <= 2.0 * r0
        tight = bstd <= 0.5 * r0
        stable = stable and near_r0 and tight
        if not near_r0:
            metrics["note"] = f"bond mean {bmean:.1f} far from r0={r0:.1f}"
        elif not tight:
            metrics["note"] = f"bond std {bstd:.1f} large (r0={r0:.1f})"
    else:
        metrics["note"] = (metrics["note"] + "; " if metrics["note"] else "") + "no bonds formed"
    metrics["stable"] = bool(stable)
    return metrics, cfg


def print_table(rows, r0, step_budget):
    hdr = (
        f"{'dt[ps]':>8} {'Dscale':>7} {'Dqt':>7} {'Dft':>7} "
        f"{'disp_ft':>8} {'d/rbind':>8} {'p_bind':>8} {'kon_max':>9} {'rateOK':>6} "
        f"{'bond_mean':>9} {'bond_std':>8} {'fbQt':>5} {'fbFt':>5} "
        f"{'reach[us]':>10} {'stable':>7}  note"
    )
    print(hdr)
    print("-" * len(hdr))
    for m in rows:
        reach = _steps_to_us(step_budget, m["dt_ps"] / 1000.0)
        print(
            f"{m['dt_ps']:>8.4g} {m['d_scale']:>7g} {m['qt_D']:>7.3g} {m['ft_D']:>7.3g} "
            f"{m['disp_ft']:>8.2f} {m['disp_over_bind']:>8.3f} {m['p_bind']:>8.4f} "
            f"{m['kon_max']:>9.3g} {str(m['rate_stochastic']):>6} "
            f"{m['bond_mean']:>9.2f} {m['bond_std']:>8.2f} "
            f"{m['frac_bound_qt']:>5.2f} {m['frac_bound_ft']:>5.2f} "
            f"{reach:>10.3g} {str(m['stable']):>7}  {m['note']}"
        )


def summarize(rows, step_budget):
    print("\nLargest stable dt per diffusion scale (reachable time at "
          f"step budget = {step_budget:,}):")
    by_scale = {}
    for m in rows:
        if m["stable"]:
            cur = by_scale.get(m["d_scale"])
            if cur is None or m["dt_ps"] > cur["dt_ps"]:
                by_scale[m["d_scale"]] = m
    if not by_scale:
        print("  (no stable cell found — try softer k, lower diffusion, or smaller dt)")
        return
    for scale in sorted(by_scale, reverse=True):
        m = by_scale[scale]
        reach = _steps_to_us(step_budget, m["dt_ps"] / 1000.0)
        print(f"  Dscale={scale:g} (Dft={m['ft_D']:.3g} nm^2/ns): "
              f"max stable dt = {m['dt_ps']:.4g} ps -> reachable {reach:.3g} us "
              f"({reach / 1e6:.3g} s)")


def summarize_rates(rows, kon, koff, p_target):
    """Per-dt reaction-rate guidance: the largest rate still resolvable at each dt.

    kon_max/koff_max depend only on dt and p_target, so this collapses the grid to one
    line per timestep. When the configured rate saturates (p > p_target), it prints the
    down-scaled rate that would restore the stochastic regime. Down-scaling a rate does
    change the modeled kinetics, so this is a "you must choose" flag, not a free fix:
    keep the fast rate and shrink dt, or accept the slower rate at the large dt.
    """
    print(f"\nReaction-rate guidance (per-step probability target p_target={p_target}):")
    print("  'max faithful' rate = -ln(1 - p_target) / dt. A faster intended rate cannot")
    print("  be resolved at that dt (p saturates -> reactions fire on first contact).")
    seen = {}
    for m in rows:
        seen.setdefault(m["dt_ps"], m)
    for dt_ps in sorted(seen):
        m = seen[dt_ps]
        kon_verdict = ("OK" if m["p_bind"] <= p_target
                       else f"SATURATED -> down-scale kon to <= {m['kon_max']:.3g} (or reduce dt)")
        print(f"  dt={dt_ps:>10.4g} ps | max faithful kon={m['kon_max']:.3g} | "
              f"current kon={kon:g} -> p_bind={m['p_bind']:.3g}  [{kon_verdict}]")
        if koff > 0:
            koff_verdict = ("OK" if m["p_break"] <= p_target
                            else f"SATURATED -> down-scale koff to <= {m['koff_max']:.3g} (or reduce dt)")
            print(f"  {'':>13} | max faithful koff={m['koff_max']:.3g} | "
                  f"current koff={koff:g} -> p_break={m['p_break']:.3g}  [{koff_verdict}]")


def main():
    p = argparse.ArgumentParser(
        description="Soft-mode timestep & reachable-time calibration sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", help="Optional base config JSON; CLI values override it.")
    p.add_argument("--timesteps", type=float, nargs="+", default=[0.05, 0.5, 2.0, 5.0],
                   help="Timesteps to test, in PICOSECONDS (default: 0.05 0.5 2 5).")
    p.add_argument("--diffusion-scales", type=float, nargs="+", default=[1.0, 0.1, 0.01],
                   help="Multipliers on base Qt/Ft diffusion (default: 1 0.1 0.01).")
    p.add_argument("--qt-diffusion", type=float, help="Base Qt diffusion (nm^2/ns).")
    p.add_argument("--ft-diffusion", type=float, help="Base Ft diffusion (nm^2/ns).")
    p.add_argument("--repulsion-force-constant", type=float,
                   help="Soft repulsion force constant (kJ/mol/nm^2), applied uniformly to the "
                        "three free-free pairs (k_QtQt=k_FtFt=k_QtFt); cluster/mixed cascade. "
                        "The sweep varies uniform stiffness — set per-pair constants in a "
                        "--config JSON if you need them to differ.")
    p.add_argument("--k-bond", type=float,
                   help="Harmonic bond force constant (kJ/mol/nm^2). Primary timestep "
                        "lever: the bond-relaxation stability bound is dt < 2*kBT/(k_bond*D), "
                        "so a soft bond (paper uses ~0.01) permits a far larger dt.")
    p.add_argument("--n-qt", type=int, default=60)
    p.add_argument("--n-ft", type=int, default=60)
    p.add_argument("--n-steps", type=int, default=4000, help="Steps per calibration cell.")
    p.add_argument("--box", type=float, default=300.0, help="Cubic box side (nm).")
    p.add_argument("--kon", type=float, help="Binding rate kon (per unit time).")
    p.add_argument("--koff", type=float, help="Bond-breaking rate koff (per unit time); "
                                              "0 (default) means no breaking.")
    p.add_argument("--binding-radius", type=float, help="Spatial reaction radius (nm).")
    p.add_argument("--step-budget", type=int, default=2_000_000,
                   help="Step budget for reachable-time (default: 2,000,000 = production).")
    p.add_argument("--p-target", type=float, default=0.1,
                   help="Max per-step reaction probability considered 'stochastic' for the "
                        "rate-rescaling guidance (default: 0.1).")
    p.add_argument("--output-csv", help="Optional path to write the full table as CSV.")
    p.add_argument("--stage-dir", help="Where to write short runs (default: temp dir, removed).")
    p.add_argument("--keep", action="store_true", help="Keep the staging trajectories.")
    args = p.parse_args()

    # Build the base config: start from --config if given, else current standard values.
    if args.config:
        base = qtft.SimulationConfig.load_json(args.config)
    else:
        base = qtft.SimulationConfig(
            qt=qtft.ParticleConfig("Qt", radius=21.0, diffusion=0.5, cluster_diffusion=0.3),
            ft=qtft.ParticleConfig("Ft", radius=6.0, diffusion=1.0, cluster_diffusion=0.7),
            topology=qtft.TopologyConfig(binding_radius=27.25, kon=0.01, k_bond=10.0),
            lj=qtft.LennardJonesConfig(epsilon_QtQt=1.5, epsilon_FtFt=1.5, epsilon_QtFt=3.0,
                                       potential_type="soft"),
            box_size=(args.box, args.box, args.box),
        )
    # Apply CLI overrides.
    base.lj.potential_type = "soft"
    if args.qt_diffusion is not None:
        base.qt.diffusion = args.qt_diffusion
    if args.ft_diffusion is not None:
        base.ft.diffusion = args.ft_diffusion
    if args.repulsion_force_constant is not None:
        # Set the three free-free constants uniformly; cluster/mixed cascade from them.
        k = args.repulsion_force_constant
        base.soft = qtft.SoftPotentialConfig(k_QtQt=k, k_FtFt=k, k_QtFt=k)
    if args.k_bond is not None:
        base.topology.k_bond = args.k_bond
    if args.kon is not None:
        base.topology.kon = args.kon
    if args.koff is not None:
        base.topology.koff = args.koff
    if args.binding_radius is not None:
        base.topology.binding_radius = args.binding_radius
    base.box_size = (args.box, args.box, args.box)
    base.n_qt = args.n_qt
    base.n_ft = args.n_ft
    base.n_steps = args.n_steps
    # Record less often so short cells stay fast but still yield frames for analysis.
    base.record_stride = max(1, args.n_steps // 20)
    base.observable_stride = base.record_stride

    r0 = base.equilibrium_bond_length
    print("=" * 78)
    print("SOFT-MODE TIMESTEP CALIBRATION")
    print("=" * 78)
    print(f"soft k_rep (kJ/mol/nm^2): k_QQ={base.soft.k_QtQt} k_FF={base.soft.k_FtFt} "
          f"k_QF={base.soft.k_QtFt}")
    print(f"base Dqt={base.qt.diffusion} Dft={base.ft.diffusion} nm^2/ns | "
          f"r0={r0:.1f} nm | binding_radius={base.topology.binding_radius} nm | "
          f"kon={base.topology.kon} | k_bond={base.topology.k_bond}")
    print(f"cells: {len(args.timesteps)} dt x {len(args.diffusion_scales)} Dscale | "
          f"{base.n_qt}Qt+{base.n_ft}Ft, {base.n_steps} steps, box {args.box} nm")
    print("=" * 78)

    stage_dir = args.stage_dir or tempfile.mkdtemp(prefix="qtft_calib_")
    os.makedirs(stage_dir, exist_ok=True)
    run_kwargs = {"overwrite": True}

    rows = []
    try:
        for dt_ps in args.timesteps:
            for d_scale in args.diffusion_scales:
                dt_ns = dt_ps / 1000.0
                m, _ = evaluate_cell(base, dt_ns, d_scale, stage_dir, run_kwargs, args.p_target)
                rows.append(m)
                print(f"  ran dt={dt_ps:g}ps Dscale={d_scale:g} -> "
                      f"stable={m['stable']} {m['note']}")
    finally:
        if not args.keep and not args.stage_dir:
            shutil.rmtree(stage_dir, ignore_errors=True)

    print()
    print_table(rows, r0, args.step_budget)
    summarize(rows, args.step_budget)
    summarize_rates(rows, base.topology.kon, base.topology.koff, args.p_target)

    if args.output_csv:
        import csv
        fields = list(rows[0].keys()) + ["reachable_us"]
        with open(args.output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for m in rows:
                row = dict(m)
                row["reachable_us"] = _steps_to_us(args.step_budget, m["dt_ps"] / 1000.0)
                w.writerow(row)
        print(f"\n✓ Wrote calibration table to {args.output_csv}")


if __name__ == "__main__":
    main()
