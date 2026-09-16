#!/usr/bin/env python
"""
Soft-mode force-constant (``soft.k_*``) calibration sweep — overlap vs. stability.

In soft mode the only thing stopping particles interpenetrating is the harmonic
repulsion ``soft.k_QtQt / k_FtFt / k_QtFt``. Overlap falls monotonically with k
(the thermal overlap scale is ``delta ~ sqrt(2*kB*T/k)``), so there is no interior
optimum: the useful question is **the largest k that is still numerically stable at
the production timestep**. That ceiling comes from the per-step overshoot ratio

    alpha = k * D * dt / (kB*T)

— a particle pushed out of an overlap ``delta`` moves ``alpha * delta`` in one Euler
step, so ``alpha >= 1`` means it overshoots the overlap and the pair oscillates/blows
up. Note D is the *partner's* diffusion too, so a cross pair is governed by the
faster species (here Ft).

This "measure-first" tool sweeps k, runs a short simulation per cell, and reports:

  - alpha           : the analytic overshoot ratio per pair (see above);
  - overlap         : how widespread and how deep the interpenetration actually is,
                      from analysis.get_overlap_statistics (fraction of pairs
                      overlapping + mean/p95/max depth as a fraction of contact),
                      pooled over the trailing frames;
  - stability       : run completed, positions all finite, and bonded distances stay
                      near the equilibrium bond length r0 (reused from
                      calibrate_timestep.py);
  - aggregation     : bound fractions and mean/largest cluster size, so a k that
                      suppresses clustering (or one where overlap is geometrically
                      forced rather than k-limited) is visible.

Three sweep modes:

    # A. global scale on the config's k triple
    python scripts/calibrate_soft_k.py --config cfg.json --scales 0.5 1 2 3 4

    # B. one pair at a time, others held at the config value
    python scripts/calibrate_soft_k.py --config cfg.json --pair FtFt --values 1 2 3 5 8

    # C. explicit candidate triples (repeatable), e.g. to re-test winners with more seeds
    python scripts/calibrate_soft_k.py --config cfg.json --k 4,3,2.5 --k 6,3,2.5 --seeds 1 2 3

``--config`` is the single source of truth for the physics; this script only varies
``config.soft.k_*`` (plus ``--n-steps`` / ``--seeds`` for cost). potential_type is
forced to "soft". Cluster/mixed constants are left to the normal cascade.
"""

import argparse
import math
import os
import shutil
import sys
import tempfile

import numpy as np

# Make the qtft package importable when run as `python scripts/calibrate_soft_k.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qtft
import qtft.analysis as analysis

# Stability helpers are shared with the timestep sweep (scripts/ is sys.path[0] here).
from calibrate_timestep import measure_bond_lengths, positions_all_finite

KB_KJ_PER_MOL_K = 0.0083145            # kJ/(mol*K) — gas constant in ReaDDy's units
PAIR_KEYS = ("QtQt", "FtFt", "QtFt")


def kbt(config):
    """Thermal energy kB*T in kJ/mol."""
    return KB_KJ_PER_MOL_K * config.temperature


def alpha_ratios(config, k_triple):
    """Per-step overshoot ratio alpha = k*D*dt/(kB*T) for each free-free pair.

    A pair is governed by the *faster* of its two partners, so the cross pair uses
    max(D_Qt, D_Ft).
    """
    kt = kbt(config)
    dt = config.timestep
    d_qt, d_ft = config.qt.diffusion, config.ft.diffusion
    k_qq, k_ff, k_qf = k_triple
    return {
        "QtQt": k_qq * d_qt * dt / kt,
        "FtFt": k_ff * d_ft * dt / kt,
        "QtFt": k_qf * max(d_qt, d_ft) * dt / kt,
    }


def thermal_overlap(config, k_triple):
    """Expected thermal interpenetration sqrt(2*kB*T/k) as a fraction of contact."""
    kt = kbt(config)
    rq, rf = config.qt.radius, config.ft.radius
    contacts = {"QtQt": 2 * rq, "FtFt": 2 * rf, "QtFt": rq + rf}
    out = {}
    for key, k in zip(PAIR_KEYS, k_triple):
        out[key] = (math.sqrt(2.0 * kt / k) / contacts[key]) if k > 0 else float("inf")
    return out


def evaluate_cell(base_config, k_triple, seed, stage_dir, n_frames, equilibration_steps):
    """Run one (k triple, seed) cell and return a metrics dict."""
    cfg = qtft.SimulationConfig.from_dict(base_config.to_dict())
    cfg.potential_type = "soft"
    cfg.rng_seed = seed
    # Set only the three free-free constants; cluster/mixed pairs cascade from them.
    k_qq, k_ff, k_qf = k_triple
    cfg.soft = qtft.SoftPotentialConfig(k_QtQt=k_qq, k_FtFt=k_ff, k_QtFt=k_qf)

    tag = f"kQQ{k_qq:g}_kFF{k_ff:g}_kQF{k_qf:g}_seed{seed}"
    cfg.output_file = os.path.join(stage_dir, f"{tag}.h5")

    r0 = cfg.equilibrium_bond_length
    alphas = alpha_ratios(cfg, k_triple)
    therm = thermal_overlap(cfg, k_triple)

    m = {
        "k_QtQt": k_qq, "k_FtFt": k_ff, "k_QtFt": k_qf, "seed": seed,
        "alpha_QtQt": alphas["QtQt"], "alpha_FtFt": alphas["FtFt"],
        "alpha_QtFt": alphas["QtFt"], "alpha_max": max(alphas.values()),
        "therm_QtQt": therm["QtQt"], "therm_FtFt": therm["FtFt"], "therm_QtFt": therm["QtFt"],
        "completed": False, "finite": False,
        "bond_mean": float("nan"), "bond_std": float("nan"), "n_bonds_seen": 0,
        "frac_bound_qt": float("nan"), "frac_bound_ft": float("nan"),
        "avg_cluster": float("nan"), "max_cluster": float("nan"),
        "stable": False, "note": "",
    }
    for label in ("Qt-Qt", "Qt-Ft", "Ft-Ft"):
        key = label.replace("-", "")
        for stat in ("fracov", "meanov", "allov", "p95ov", "maxov"):
            m[f"{stat}_{key}"] = float("nan")
    m["worst_allov"] = float("nan")

    try:
        qtft.run_one(cfg, equilibration_steps=equilibration_steps,
                     skip_equilibration=(equilibration_steps <= 0),
                     overwrite=True, show_progress=False)
        m["completed"] = True
    except Exception as e:          # an integrator blow-up usually surfaces as an exception
        m["note"] = f"run failed: {type(e).__name__}: {str(e)[:80]}"
        return m

    try:
        m["finite"] = positions_all_finite(cfg.output_file, cfg)
        bmean, bstd, bn = measure_bond_lengths(cfg.output_file, cfg)
        m["bond_mean"], m["bond_std"], m["n_bonds_seen"] = bmean, bstd, bn

        kin = analysis.get_binding_kinetics(cfg.output_file, cfg)
        m["frac_bound_qt"] = float(np.asarray(kin["fraction_bound_qt"])[-1])
        m["frac_bound_ft"] = float(np.asarray(kin["fraction_bound_ft"])[-1])

        cl = analysis.get_cluster_statistics(cfg.output_file)
        m["avg_cluster"] = float(np.asarray(cl["avg_sizes"])[-1])
        m["max_cluster"] = float(np.asarray(cl["max_sizes"])[-1])

        ov = analysis.get_overlap_statistics(cfg.output_file, cfg, n_frames=n_frames)
        means = []
        for label, p in ov["pairs"].items():
            key = label.replace("-", "")
            m[f"fracov_{key}"] = p["frac_overlapping"]
            m[f"meanov_{key}"] = p["mean_overlap_frac"]
            m[f"allov_{key}"] = p["mean_overlap_all_frac"]
            m[f"p95ov_{key}"] = p["p95_overlap_frac"]
            m[f"maxov_{key}"] = p["max_overlap_frac"]
            if p["mean_overlap_all_frac"] == p["mean_overlap_all_frac"]:      # nan-safe
                means.append(p["mean_overlap_all_frac"])
        # Rank on the UNCONDITIONAL mean: the conditional one is selection-biased
        # (stiffening removes shallow overlaps first, flattening the conditional mean).
        m["worst_allov"] = max(means) if means else float("nan")
    except Exception as e:
        m["note"] = f"analysis failed: {type(e).__name__}: {str(e)[:80]}"
        return m

    # Stability verdict: same criteria as the timestep sweep — finished, all-finite,
    # and (if bonds formed) bonded distances stay near r0 without excessive spread.
    stable = m["completed"] and m["finite"]
    if bn > 0:
        near_r0 = 0.5 * r0 <= bmean <= 2.0 * r0
        tight = bstd <= 0.5 * r0
        stable = stable and near_r0 and tight
        if not near_r0:
            m["note"] = f"bond mean {bmean:.1f} far from r0={r0:.1f}"
        elif not tight:
            m["note"] = f"bond std {bstd:.1f} large (r0={r0:.1f})"
    else:
        m["note"] = (m["note"] + "; " if m["note"] else "") + "no bonds formed"
    m["stable"] = bool(stable)
    return m


def print_table(rows):
    hdr = (
        f"{'k_QQ':>6} {'k_FF':>6} {'k_QF':>6} {'seed':>5} {'a_max':>6} "
        f"{'all%QQ':>7} {'all%QF':>7} {'all%FF':>7} {'worst%':>7} "
        f"{'fr%QQ':>6} {'fr%QF':>6} {'fr%FF':>6} "
        f"{'bond':>6} {'bstd':>6} {'fbFt':>5} {'avgCl':>6} {'maxCl':>6} {'stable':>7}  note"
    )
    print(hdr)
    print("-" * len(hdr))
    for m in rows:
        print(
            f"{m['k_QtQt']:>6g} {m['k_FtFt']:>6g} {m['k_QtFt']:>6g} {m['seed']:>5} "
            f"{m['alpha_max']:>6.2f} "
            f"{100 * m['allov_QtQt']:>7.3f} {100 * m['allov_QtFt']:>7.3f} "
            f"{100 * m['allov_FtFt']:>7.3f} {100 * m['worst_allov']:>7.3f} "
            f"{100 * m['fracov_QtQt']:>6.2f} {100 * m['fracov_QtFt']:>6.2f} "
            f"{100 * m['fracov_FtFt']:>6.2f} "
            f"{m['bond_mean']:>6.1f} {m['bond_std']:>6.2f} {m['frac_bound_ft']:>5.2f} "
            f"{m['avg_cluster']:>6.1f} {m['max_cluster']:>6.0f} "
            f"{str(m['stable']):>7}  {m['note']}"
        )
    print("\n  all% = mean interpenetration over ALL pairs (zeros included), % of contact")
    print("         — the unbiased ranking metric; the conditional mean is selection-biased")
    print("  fr%  = fraction of all pairs of that family that overlap at all, %")
    print("  a_max= max over pairs of alpha = k*D*dt/(kB*T)  (>=1 => Euler overshoot)")
    print("  fbFt/avgCl: watch these — stiffening the REACTIVE Qt-Ft pair suppresses binding")


def summarize(rows, base_config):
    """Rank stable cells by worst-pair mean overlap and flag the alpha ceiling."""
    kt = kbt(base_config)
    dt = base_config.timestep
    print("\nStability ceilings at this dt and D (alpha = k*D*dt/(kB*T) = 1):")
    for key, d in (("QtQt", base_config.qt.diffusion),
                   ("FtFt", base_config.ft.diffusion),
                   ("QtFt", max(base_config.qt.diffusion, base_config.ft.diffusion))):
        k_max = kt / (d * dt) if d * dt > 0 else float("inf")
        print(f"  k_{key}: marginal k = {k_max:.3g}, alpha=0.5 safe k = {0.5 * k_max:.3g} "
              f"kJ/(mol*nm^2)   [D = {d:g} nm^2/ns]")

    stable = [m for m in rows if m["stable"] and m["worst_allov"] == m["worst_allov"]]
    if not stable:
        print("\n  (no stable cell — soften k, lower D, or reduce dt)")
        return
    print("\nStable cells ranked by worst-pair mean overlap over ALL pairs "
          "(lower = less interpenetration):")
    for m in sorted(stable, key=lambda r: r["worst_allov"]):
        print(f"  k=({m['k_QtQt']:g}, {m['k_FtFt']:g}, {m['k_QtFt']:g}) seed={m['seed']} -> "
              f"worst {100 * m['worst_allov']:.3f}% "
              f"(QQ {100 * m['allov_QtQt']:.3f} / QF {100 * m['allov_QtFt']:.3f} / "
              f"FF {100 * m['allov_FtFt']:.3f}), alpha_max={m['alpha_max']:.2f}, "
              f"boundFt={m['frac_bound_ft']:.2f}, avgCl={m['avg_cluster']:.1f}")
    unstable = [m for m in rows if not m["stable"]]
    if unstable:
        print("\nUnstable / rejected cells:")
        for m in unstable:
            print(f"  k=({m['k_QtQt']:g}, {m['k_FtFt']:g}, {m['k_QtFt']:g}) seed={m['seed']} "
                  f"alpha_max={m['alpha_max']:.2f} -> {m['note'] or 'failed'}")


def build_grid(args, base):
    """Resolve the CLI sweep mode into a list of (k_QtQt, k_FtFt, k_QtFt) triples."""
    base_k = (base.soft.k_QtQt, base.soft.k_FtFt, base.soft.k_QtFt)
    triples = []
    if args.k:
        for spec in args.k:
            parts = [float(x) for x in spec.split(",")]
            if len(parts) != 3:
                raise SystemExit(f"--k expects 'kQQ,kFF,kQF', got {spec!r}")
            triples.append(tuple(parts))
    elif args.pair:
        if args.pair not in PAIR_KEYS:
            raise SystemExit(f"--pair must be one of {PAIR_KEYS}")
        if not args.values:
            raise SystemExit("--pair requires --values")
        idx = PAIR_KEYS.index(args.pair)
        for v in args.values:
            t = list(base_k)
            t[idx] = float(v)
            triples.append(tuple(t))
    else:
        for s in args.scales:
            triples.append(tuple(s * k for k in base_k))
    # De-duplicate while preserving order (scales can collide, e.g. 1.0 with an explicit --k).
    seen, out = set(), []
    for t in triples:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def main():
    p = argparse.ArgumentParser(
        description="Soft-mode force-constant sweep: interpenetration vs. stability.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True,
                   help="Base SimulationConfig JSON (the physics under test). Only "
                        "soft.k_*, n_steps and rng_seed are overridden by this script.")
    p.add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0, 4.0],
                   help="Global multipliers on the config's k triple (default sweep mode).")
    p.add_argument("--pair", choices=PAIR_KEYS,
                   help="Sweep ONE pair (with --values), holding the others at the config value.")
    p.add_argument("--values", type=float, nargs="+", help="k values for --pair.")
    p.add_argument("--k", action="append",
                   help="Explicit triple 'kQQ,kFF,kQF'; repeatable. Overrides --scales/--pair.")
    p.add_argument("--seeds", type=int, nargs="+", default=[22],
                   help="RNG seeds per k triple (default: 22). More seeds = replicate check.")
    p.add_argument("--n-steps", type=int, default=25000,
                   help="Steps per cell (default: 25000). Must be long enough to form clusters.")
    p.add_argument("--record-stride", type=int,
                   help="Trajectory/observable stride (default: n_steps//50).")
    p.add_argument("--equilibration-steps", type=int, default=1000,
                   help="Reaction-free relaxation before production; <=0 skips it (default: 1000).")
    p.add_argument("--n-frames", type=int, default=5,
                   help="Trailing frames pooled for the overlap statistics (default: 5).")
    p.add_argument("--output-csv", help="Optional path to write the full table as CSV.")
    p.add_argument("--stage-dir", help="Where to write the short runs (default: temp dir, removed).")
    p.add_argument("--keep", action="store_true", help="Keep the staging trajectories.")
    args = p.parse_args()

    base = qtft.SimulationConfig.load_json(args.config)
    base.potential_type = "soft"
    base.n_steps = args.n_steps
    base.record_stride = args.record_stride or max(1, args.n_steps // 50)
    base.observable_stride = base.record_stride
    base.output_file = None            # re-derived per cell

    triples = build_grid(args, base)

    print("=" * 100)
    print("SOFT-MODE FORCE-CONSTANT (k) CALIBRATION — overlap vs. stability")
    print("=" * 100)
    print(f"base config : {args.config}")
    print(f"physics     : {base.n_qt}Qt(r={base.qt.radius}, D={base.qt.diffusion:g}) + "
          f"{base.n_ft}Ft(r={base.ft.radius}, D={base.ft.diffusion:g}), box {base.box_size[0]:g} nm, "
          f"T={base.temperature:g} K")
    print(f"              dt={base.timestep:g} ns, kon={base.topology.kon:g}, "
          f"k_bond={base.topology.k_bond:g}, r_bind={base.topology.binding_radius:g} nm, "
          f"allow_loops={base.topology.allow_loops}, ft_monovalent={base.topology.ft_monovalent}")
    print(f"base k      : ({base.soft.k_QtQt:g}, {base.soft.k_FtFt:g}, {base.soft.k_QtFt:g}) "
          f"kJ/(mol*nm^2)")
    print(f"cells       : {len(triples)} k triples x {len(args.seeds)} seeds, "
          f"{args.n_steps:,} steps each "
          f"({qtft.config.format_duration(args.n_steps * base.timestep)} per cell)")
    print("=" * 100)

    stage_dir = args.stage_dir or tempfile.mkdtemp(prefix="qtft_ksweep_")
    os.makedirs(stage_dir, exist_ok=True)

    rows = []
    try:
        for t in triples:
            for seed in args.seeds:
                m = evaluate_cell(base, t, seed, stage_dir, args.n_frames,
                                  args.equilibration_steps)
                rows.append(m)
                print(f"  ran k=({t[0]:g}, {t[1]:g}, {t[2]:g}) seed={seed} -> "
                      f"stable={m['stable']} worst_overlap="
                      f"{100 * m['worst_allov']:.3f}% boundFt={m['frac_bound_ft']:.2f} "
                      f"{m['note']}")
    finally:
        if not args.keep and not args.stage_dir:
            shutil.rmtree(stage_dir, ignore_errors=True)

    print()
    print_table(rows)
    summarize(rows, base)

    if args.output_csv:
        import csv
        with open(args.output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n✓ Wrote sweep table to {args.output_csv}")


if __name__ == "__main__":
    main()
