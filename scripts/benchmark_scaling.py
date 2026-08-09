#!/usr/bin/env python
"""
Performance diagnostics for the `qtft` simulations. Answers two questions:

  1. **How does cost scale with system size?** — whether a 2 µm box at the current
     concentration (38,400 particles) is affordable, and with how many threads.
  2. **Why does a run get slower as it proceeds?** — by isolating aggregation from I/O.

Nothing in `qtft/` is instrumented. Every measurement here comes from the public API
(`create_system` / `create_simulation` / `place_particles` / `run_simulation`) and from
existing config fields, so the numbers describe the code as it actually runs.

How the rate-vs-step curve is captured
--------------------------------------
`readdy.Simulation.run` does ``import tqdm`` and drives ``tqdm.tqdm`` from a progress
callback fired every ``progress_output_stride`` steps (readdy/api/simulation.py). Replacing
``tqdm.tqdm`` with a timestamping subclass — here, in this script only — turns that callback
into an exact wall-clock sample of the integration loop. No package code changes, and the
sampled loop is the real one, not a copy.

Each size/arm runs in its **own subprocess**: peak RSS is then a real per-run number rather
than a high-water mark carried over from the previous, and a size that exhausts memory does
not take the rest of the sweep with it.

Usage
-----
    python scripts/benchmark_scaling.py scaling  --out DIR [--sizes 500,707,1000]
    python scripts/benchmark_scaling.py threads  --out DIR [--box 2000]
    python scripts/benchmark_scaling.py slowdown --out DIR [--steps 150000]
    python scripts/benchmark_scaling.py report   --out DIR

Results accumulate as JSON under --out; `report` reads them back and prints the tables.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# Reference point: the production settings in Run_Simulation.ipynb.
BASE_BOX = 500.0
BASE_NQT = 400
BASE_NFT = 200

# Aggregation state at which the size sweep is measured. 5,000 steps is ~4 % of the final
# bond count (measured: 57 of 1350), so every size is timed in a comparably unaggregated
# state — otherwise the small, fast sizes would run far into aggregation within the same
# wall-clock budget and look artificially slow.
SWEEP_MAX_STEPS = 5000
SWEEP_MIN_STEPS = 200
SWEEP_TARGET_SECONDS = 90.0


# =============================================================================
# timing capture
# =============================================================================

_TICKS: list = []


def install_tick_timer(stride: int):
    """Make ``tqdm.tqdm`` timestamp every progress callback; returns the tick list.

    Must be called before ``simulation.run``. ``stride`` is only recorded for the caller's
    bookkeeping — the actual cadence is set via ``simulation.progress_output_stride``.
    """
    import tqdm as tqdm_mod

    ticks: list = []

    class TimingTqdm(tqdm_mod.tqdm):
        def update(self, n=1):
            ticks.append(time.perf_counter())
            return super().update(n)

    tqdm_mod.tqdm = TimingTqdm
    return ticks


def rates_from_ticks(ticks, stride, warmup=2):
    """Per-interval steps/s from progress timestamps, dropping ``warmup`` warm-up ticks.

    Returns ``(steps, rates)`` where ``steps[i]`` is the step at the END of the interval
    whose rate is ``rates[i]``. Tick *k* fires after ``(k+1)*stride`` steps, so the interval
    between ticks k and k+1 ends at ``(k+2)*stride`` — getting this off by one puts the
    periodic observable spike on the wrong step and hides what causes it.
    """
    t = np.asarray(ticks, dtype=float)
    if t.size < warmup + 2:
        return np.array([]), np.array([])
    dt = np.diff(t[warmup:])
    steps = (np.arange(warmup + 2, t.size + 1) * stride).astype(float)
    with np.errstate(divide="ignore"):
        rates = np.where(dt > 0, stride / dt, np.nan)
    return steps, rates


def sustained_rate(rates, stride):
    """Steps/s actually achieved over the measured window — the harmonic mean of the
    per-interval rates, i.e. total steps / total time.

    The *median* rate is the wrong headline number whenever an expensive event recurs on a
    stride of its own: observables firing every 100 steps are only 1 tick in 4 when the
    progress stride is 25, so the median reports the three cheap ticks and silently drops the
    one that dominates the wall clock. At 38,400 particles that is a 2x optimistic error.
    """
    r = np.asarray(rates, dtype=float)
    r = r[np.isfinite(r) & (r > 0)]
    if r.size == 0:
        return float("nan")
    return float(r.size / np.sum(1.0 / r))


def peak_rss_gb():
    """Peak resident set size of this process, in GB (Linux reports ru_maxrss in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


# =============================================================================
# configuration
# =============================================================================

def make_config(box, n_qt, n_ft, n_steps, out_file, *, n_threads=4, kon=1e-6,
                record_stride=100, observable_stride=100, seed=22, allow_loops=False,
                particles_observable_stride=None):
    """The notebook's production configuration, with the sweep knobs exposed."""
    import qtft as sim

    return sim.SimulationConfig(
        potential_type="soft",
        qt=sim.ParticleConfig(name="Qt", radius=25.0, diffusion=2e-4, cluster_diffusion=2e-4),
        ft=sim.ParticleConfig(name="Ft", radius=7.0, diffusion=5e-4, cluster_diffusion=5e-4),
        topology=sim.TopologyConfig(name="QtFt_Cluster", binding_radius=32.0, kon=kon,
                                    k_bond=1.0, ft_monovalent=False,
                                    allow_loops=bool(allow_loops)),
        soft=sim.SoftPotentialConfig(k_QtQt=4.0, k_FtFt=3.0, k_QtFt=1.5),
        equilibration_potential="soft",
        box_size=(box, box, box),
        boundary="periodic",
        temperature=300.0,
        timestep=1e3,
        n_steps=int(n_steps),
        record_stride=int(record_stride),
        observable_stride=int(observable_stride),
        particles_observable_stride=(None if particles_observable_stride is None
                                     else int(particles_observable_stride)),
        n_qt=int(n_qt),
        n_ft=int(n_ft),
        kernel="CPU",
        n_threads=int(n_threads),
        rng_seed=int(seed),
        output_file=out_file,
    )


def counts_for_box(box):
    """Particle counts that hold the concentration fixed as the box grows."""
    factor = (float(box) / BASE_BOX) ** 3
    return int(round(BASE_NQT * factor)), int(round(BASE_NFT * factor))


# =============================================================================
# worker — one measured run, in its own process
# =============================================================================

def timed_pipeline(config, equilibration_steps, progress_stride, positions=None):
    """Run one simulation, timing setup and the integration loop separately.

    Mirrors ``engine.run_one``'s sequence exactly (equilibrate -> system -> simulation ->
    place -> run) so the measured cost is the cost the notebook actually pays.
    """
    import qtft.engine as engine
    from qtft.system import create_system

    out = {}

    if positions is None and equilibration_steps > 0:
        t0 = time.perf_counter()
        positions = engine.equilibrate_system(config, n_steps=equilibration_steps)
        out["equilibrate_s"] = time.perf_counter() - t0
    pos_qt, pos_ft = positions if positions is not None else (None, None)

    t0 = time.perf_counter()
    system = create_system(config)
    simulation = engine.create_simulation(system, config, overwrite=True)
    simulation.progress_output_stride = int(progress_stride)
    engine.place_particles(simulation, config, positions_qt=pos_qt, positions_ft=pos_ft)
    out["setup_s"] = time.perf_counter() - t0

    ticks = install_tick_timer(progress_stride)
    t0 = time.perf_counter()
    engine.run_simulation(simulation, config, show_progress=False)
    out["run_s"] = time.perf_counter() - t0
    out["ticks"] = list(ticks)
    out["progress_stride"] = int(progress_stride)
    out["positions"] = (pos_qt, pos_ft)
    return out


def worker(spec):
    """Execute one spec dict and return the result dict (JSON-serializable)."""
    import logging
    logging.getLogger("qtft").setLevel(logging.ERROR)

    box = spec["box"]
    n_qt, n_ft = spec.get("n_qt"), spec.get("n_ft")
    if n_qt is None:
        n_qt, n_ft = counts_for_box(box)
    work = spec["work_dir"]
    os.makedirs(work, exist_ok=True)

    result = {k: spec[k] for k in ("mode", "label", "box", "n_threads")}
    result.update(n_qt=n_qt, n_ft=n_ft, n_particles=n_qt + n_ft)

    common = dict(box=box, n_qt=n_qt, n_ft=n_ft, n_threads=spec["n_threads"],
                  kon=spec.get("kon", 1e-6),
                  record_stride=spec.get("record_stride", 100),
                  observable_stride=spec.get("observable_stride", 100),
                  allow_loops=spec.get("allow_loops", False),
                  particles_observable_stride=spec.get("particles_observable_stride"))

    if spec["mode"] == "sweep":
        # Burst first: a cheap rate estimate that sets the length of the measured run, so
        # every size costs about the same wall-clock to measure.
        burst_steps = spec.get("burst_steps", 200)
        burst_stride = max(1, burst_steps // 8)
        cfg = make_config(n_steps=burst_steps, out_file=os.path.join(work, "burst.h5"), **common)
        burst = timed_pipeline(cfg, spec["equilibration_steps"], burst_stride)
        result["equilibrate_s"] = burst.get("equilibrate_s")
        result["setup_s"] = burst["setup_s"]
        burst_rate = burst_steps / burst["run_s"] if burst["run_s"] > 0 else float("inf")
        result["burst_rate"] = burst_rate

        steps = int(np.clip(SWEEP_TARGET_SECONDS * burst_rate, SWEEP_MIN_STEPS, SWEEP_MAX_STEPS))
        stride = max(1, steps // 30)
        steps = (steps // stride) * stride
        cfg = make_config(n_steps=steps, out_file=os.path.join(work, "main.h5"), **common)
        main = timed_pipeline(cfg, 0, stride, positions=burst["positions"])
        result["main_setup_s"] = main["setup_s"]
    else:
        steps = spec["steps"]
        stride = spec.get("progress_stride", max(1, steps // 150))
        cfg = make_config(n_steps=steps, out_file=os.path.join(work, "main.h5"), **common)
        main = timed_pipeline(cfg, spec["equilibration_steps"], stride)
        result["equilibrate_s"] = main.get("equilibrate_s")
        result["setup_s"] = main["setup_s"]

    n_ticks = len(main["ticks"])
    if n_ticks < 4:
        raise RuntimeError(
            f"progress callback produced only {n_ticks} ticks for {steps} steps "
            f"(stride {stride}) — the tqdm timing hook did not work; refusing to report a rate"
        )

    step_axis, rates = rates_from_ticks(main["ticks"], stride)
    fifth = max(1, len(rates) // 5)
    result.update(
        n_steps=steps, progress_stride=stride, run_s=main["run_s"],
        rate=sustained_rate(rates, stride),
        rate_median=float(np.nanmedian(rates)),
        rate_first=sustained_rate(rates[:fifth], stride),
        rate_last=sustained_rate(rates[-fifth:], stride),
        rate_steps=step_axis.tolist(), rates=rates.tolist(),
        peak_rss_gb=peak_rss_gb(),
        traj_path=cfg.output_file,
        traj_bytes=os.path.getsize(cfg.output_file) if os.path.exists(cfg.output_file) else 0,
        n_frames=int(steps // cfg.record_stride) + 1,
    )
    return result


# =============================================================================
# parent — dispatch specs to subprocesses
# =============================================================================

def run_spec(spec, out_dir, keep_traj=False):
    """Run one spec in a subprocess; returns the result dict (or an error dict)."""
    os.makedirs(out_dir, exist_ok=True)
    spec_path = os.path.join(out_dir, f"_spec_{spec['label']}.json")
    res_path = os.path.join(out_dir, f"result_{spec['label']}.json")
    with open(spec_path, "w") as fh:
        json.dump(spec, fh)

    env = dict(os.environ, PYTHONPATH=REPO)
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, os.path.abspath(__file__),
                           "--worker", spec_path, "--result", res_path],
                          env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0 or not os.path.exists(res_path):
        tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-6:])
        print(f"  ✗ {spec['label']}: failed after {elapsed:.0f} s\n{tail}")
        return {"label": spec["label"], "error": tail, "box": spec["box"]}

    with open(res_path) as fh:
        result = json.load(fh)
    if not keep_traj:
        shutil.rmtree(spec["work_dir"], ignore_errors=True)
    print(f"  ✓ {spec['label']:<22} N={result['n_particles']:>6}  "
          f"{rate_of(result):>8.1f} steps/s   peak RSS {result['peak_rss_gb']:.2f} GB   "
          f"({elapsed:.0f} s)")
    return result


def cmd_scaling(args):
    boxes = [float(b) for b in args.sizes.split(",")]
    print(f"Size sweep at fixed concentration (measured at <= {SWEEP_MAX_STEPS} steps, "
          f"i.e. ~4 % of final bonds):")
    for box in boxes:
        n_qt, n_ft = counts_for_box(box)
        spec = dict(mode="sweep", label=f"box{int(box)}", box=box, n_qt=n_qt, n_ft=n_ft,
                    n_threads=args.threads, equilibration_steps=args.equilibration,
                    work_dir=os.path.join(args.out, "work", f"box{int(box)}"))
        run_spec(spec, args.out, keep_traj=(box == boxes[-1] and args.keep))


def cmd_threads(args):
    box = args.box
    n_qt, n_ft = counts_for_box(box)
    print(f"Thread scaling at box {int(box)} nm (N={n_qt + n_ft}):")
    for nt in [int(t) for t in args.threads_list.split(",")]:
        spec = dict(mode="sweep", label=f"box{int(box)}_t{nt}", box=box, n_qt=n_qt, n_ft=n_ft,
                    n_threads=nt, equilibration_steps=args.equilibration,
                    work_dir=os.path.join(args.out, "work", f"box{int(box)}_t{nt}"))
        run_spec(spec, args.out)


def cmd_slowdown(args):
    """Three arms that separate aggregation from I/O, plus a size control."""
    steps = args.steps
    arms = [
        dict(label="A_baseline", box=500.0, kon=1e-6),
        # SimulationConfig rejects kon=0 ("Binding rate must be positive"), so this is the
        # smallest rate that is legal and still binds nothing: p = 1-exp(-kon*dt) ~ 1e-17 per
        # step, i.e. zero events in 1e5 steps.
        dict(label="B_no_binding", box=500.0, kon=1e-20),
        dict(label="C_no_io", box=500.0, kon=1e-6,
             record_stride=steps, observable_stride=steps),
    ]
    if args.control_box:
        arms.append(dict(label="D_control_box1000", box=args.control_box, kon=1e-6,
                         steps=args.control_steps or steps))
    print(f"Slowdown arms ({steps:,} steps each, allow_loops={args.loops}):")
    for arm in arms:
        arm["label"] += args.tag
        arm["allow_loops"] = args.loops
        label, box = arm.pop("label"), arm.pop("box")
        n_qt, n_ft = counts_for_box(box)
        spec = dict(mode="arm", label=label, box=box, n_qt=n_qt, n_ft=n_ft,
                    n_threads=args.threads, equilibration_steps=args.equilibration,
                    steps=arm.pop("steps", steps),
                    work_dir=os.path.join(args.out, "work", label), **arm)
        run_spec(spec, args.out, keep_traj=(label == "A_baseline"))


def rate_of(result):
    """Sustained steps/s, recomputed from the stored per-interval rates when a result
    predates the switch away from the median."""
    if "rate" in result:
        return result["rate"]
    return sustained_rate(result["rates"], result["progress_stride"])


def cmd_observables(args):
    """Split the per-step cost into integration, trajectory writing and observables.

    At 38,400 particles the observable evaluation is a 5.8x spike on the ticks where it
    fires; since it recurs every `observable_stride` steps it can dominate the wall clock
    while barely showing up in a median. Three runs, differing only in stride config, say how
    much of the cost each part owns.
    """
    box = args.box
    n_qt, n_ft = counts_for_box(box)
    # "Never fires again after step 0" = any stride past the end of the run. It must stay
    # small enough that effective_heavy_observable_stride (100x this) still fits the C++
    # int the forces/virial observables take — 1e9 overflows it.
    never = 10 * args.steps
    variants = [
        ("obs_on", dict(record_stride=100, observable_stride=100)),
        ("obs_off_traj_on", dict(record_stride=100, observable_stride=never)),
        ("obs_off_traj_off", dict(record_stride=never, observable_stride=never)),
    ]
    print(f"Observable/I-O cost split at box {int(box)} nm (N={n_qt + n_ft}):")
    for name, strides in variants:
        spec = dict(mode="arm", label=f"obs_{int(box)}_{name}", box=box, n_qt=n_qt, n_ft=n_ft,
                    n_threads=args.threads, equilibration_steps=args.equilibration,
                    steps=args.steps, progress_stride=args.steps // 30,
                    work_dir=os.path.join(args.out, "work", f"obs_{name}"), **strides)
        run_spec(spec, args.out)


def load_results(out_dir):
    out = []
    for name in sorted(os.listdir(out_dir)):
        if name.startswith("result_") and name.endswith(".json"):
            with open(os.path.join(out_dir, name)) as fh:
                out.append(json.load(fh))
    return [r for r in out if "error" not in r]


def cmd_report(args):
    results = load_results(args.out)
    if not results:
        print("No results found.")
        return

    sweep = sorted([r for r in results if r["label"].startswith("box") and "_t" not in r["label"]],
                   key=lambda r: r["n_particles"])
    if sweep:
        print("\nCost vs system size (unaggregated, t=0)")
        print(f"{'box (nm)':>9} {'N':>7} {'steps/s':>10} {'µs/step/particle':>18} "
              f"{'setup s':>9} {'peak RSS GB':>12} {'B/particle/frame':>17}")
        for r in sweep:
            per = 1e6 / rate_of(r) / r["n_particles"]
            bpf = r["traj_bytes"] / max(1, r["n_frames"]) / r["n_particles"]
            print(f"{r['box']:>9.0f} {r['n_particles']:>7} {rate_of(r):>10.1f} "
                  f"{per:>18.3f} {r.get('setup_s', 0):>9.1f} {r['peak_rss_gb']:>12.2f} "
                  f"{bpf:>17.1f}")

        n = np.array([r["n_particles"] for r in sweep], dtype=float)
        cost = np.array([1.0 / rate_of(r) for r in sweep])
        slope, intercept = np.polyfit(n, cost, 1)
        pred = slope * n + intercept
        print(f"\n  linear fit  cost/step = {slope:.3e} s x N + {intercept:.3e} s")
        print(f"  residuals: {', '.join(f'{100*(c-p)/c:+.1f}%' for c, p in zip(cost, pred))}")

    # match the thread-sweep label exactly: "_t<digits>" at the end. A substring test for
    # "_t" also catches "obs_..._traj_off" and silently mixes the two experiments.
    threads = sorted([r for r in results if re.search(r"_t\d+$", r["label"])],
                     key=lambda r: (r["n_particles"], r["n_threads"]))
    if threads:
        print("\nThread scaling")
        print(f"{'box':>6} {'N':>7} {'threads':>8} {'steps/s':>10} {'speedup vs 1':>13}")
        base = {}
        for r in threads:
            base.setdefault(r["n_particles"], rate_of(r) if r["n_threads"] == 1 else None)
        for r in threads:
            b = base.get(r["n_particles"])
            sp = f"{rate_of(r) / b:.2f}x" if b else "-"
            print(f"{r['box']:>6.0f} {r['n_particles']:>7} {r['n_threads']:>8} "
                  f"{rate_of(r):>10.1f} {sp:>13}")

    obs = sorted([r for r in results if r["label"].startswith("obs_")],
                 key=lambda r: -rate_of(r))
    if obs:
        print("\nWhere the per-step time goes")
        print(f"{'variant':>24} {'steps/s':>9} {'ms/step':>9} {'share of full cost':>20}")
        full = min(rate_of(r) for r in obs)
        for r in obs:
            ms = 1000.0 / rate_of(r)
            print(f"{r['label']:>24} {rate_of(r):>9.1f} {ms:>9.1f} "
                  f"{100 * (1000.0 / full - ms) / (1000.0 / full):>19.0f}% saved")

    arms = [r for r in results if r["label"][0] in "ABCD" and "_" in r["label"]]
    if arms:
        print("\nSlowdown arms (rate over the run)")
        print(f"{'arm':>20} {'N':>7} {'steps':>9} {'first 20%':>11} {'last 20%':>10} {'change':>9}")
        for r in sorted(arms, key=lambda r: r["label"]):
            change = 100.0 * (r["rate_last"] / r["rate_first"] - 1.0)
            print(f"{r['label']:>20} {r['n_particles']:>7} {r['n_steps']:>9,} "
                  f"{r['rate_first']:>11.1f} {r['rate_last']:>10.1f} {change:>8.1f}%")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worker", help=argparse.SUPPRESS)
    p.add_argument("--result", help=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="cmd")

    def common(sp):
        sp.add_argument("--out", required=True, help="directory for result JSON")
        sp.add_argument("--threads", type=int, default=4)
        sp.add_argument("--equilibration", type=int, default=1000)

    sp = sub.add_parser("scaling"); common(sp)
    sp.add_argument("--sizes", default="500,707,1000,1414,2000")
    sp.add_argument("--keep", action="store_true", help="keep the largest trajectory")
    sp.set_defaults(func=cmd_scaling)

    sp = sub.add_parser("threads"); common(sp)
    sp.add_argument("--box", type=float, default=2000.0)
    sp.add_argument("--threads-list", default="1,4,8,16")
    sp.set_defaults(func=cmd_threads)

    sp = sub.add_parser("slowdown"); common(sp)
    sp.add_argument("--steps", type=int, default=150000)
    sp.add_argument("--loops", action="store_true",
                    help="allow_loops=True (intra-cluster crosslinks keep adding bonds "
                         "long after the cluster count saturates)")
    sp.add_argument("--tag", default="", help="suffix so a second sweep does not overwrite")
    sp.add_argument("--control-box", type=float, default=1000.0)
    sp.add_argument("--control-steps", type=int, default=None)
    sp.set_defaults(func=cmd_slowdown)

    sp = sub.add_parser("observables"); common(sp)
    sp.add_argument("--box", type=float, default=2000.0)
    sp.add_argument("--steps", type=int, default=1500)
    sp.set_defaults(func=cmd_observables)

    sp = sub.add_parser("report")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_report)

    args = p.parse_args()

    if args.worker:
        with open(args.worker) as fh:
            spec = json.load(fh)
        result = worker(spec)
        with open(args.result, "w") as fh:
            json.dump(result, fh)
        return 0

    if not getattr(args, "func", None):
        p.print_help()
        return 1
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
