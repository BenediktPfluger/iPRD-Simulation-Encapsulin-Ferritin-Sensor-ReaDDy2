# Soft mode — reaching larger timesteps

How the soft harmonic-repulsion potential buys a µs timestep, how to calibrate it, and the
measured force constants the notebook preset ships. Background on the four potential modes is
in the README under [Potentials](../README.md#2-potentials).

---

Soft mode is what the notebook runs today; this page explains why. The earlier production
setup used stiff 12-6 Lennard-Jones, whose `r⁻¹²` wall turns any particle overlap into an
enormous force and so forced a very small timestep (50 ps) for EulerBD stability — capping a
run at ~100 µs of simulated time (the `Different_Particle_Ratios/` datasets).
**Soft mode** (`potential_type="soft"`) replaces that wall with **harmonic repulsion**
(ReaDDy's `add_harmonic_repulsion`): a bounded, linear force that vanishes at the contact
distance `r_i + r_j`. Overlaps then produce small finite forces instead of a blow-up, so a
much larger `dt` is numerically stable. This mirrors the approach of Arkfeld et al.,
*Whole-cell particle-based digital twin simulations from 4D lattice light-sheet microscopy
data* (2026; [schoeneberglab/readdy-cell](https://github.com/schoeneberglab/readdy-cell)),
which reaches minute-scale simulated time with millisecond timesteps.

There is **no attractive term** in soft mode — clustering comes purely from the topology
binding reactions + harmonic bonds (unchanged). Soft mode is **self-contained**: it reads only
`config.soft.*` and **ignores `lj.epsilon_*`** entirely (only `config.potential_type` selects
the mode). Each pair has its **own** force constant `soft.k_*` (kJ/(mol·nm²)), so you can stiffen
the small-particle pairs to stop them overlapping. The thermal overlap scale is
`δ ≈ √(2·kᵦT / k)`, so a value soft enough for a large `dt` lets small particles interpenetrate —
raise `k_FtFt` / `k_QtFt` to fix that. Setting any `k = 0` disables that pair. The constants
follow the **same free → cluster → mixed cascade** as the LJ epsilons (set the three free-free
values; cluster/mixed derive unless overridden).

```python
config = sim.SimulationConfig(
    qt=sim.ParticleConfig("Qt", radius=25.0, diffusion=2e-4, cluster_diffusion=2e-4),
    ft=sim.ParticleConfig("Ft", radius=7.0,  diffusion=5e-4, cluster_diffusion=5e-4),
    topology=sim.TopologyConfig(binding_radius=32.0, kon=1e-6, k_bond=1.0),
    potential_type="soft",   # <- top-level selector (epsilons/lj ignored in soft mode)
    soft=sim.SoftPotentialConfig(k_QtQt=4.0, k_FtFt=3.0, k_QtFt=1.5),  # calibrated, see below
    equilibration_potential="soft",
    box_size=(500.0, 500.0, 500.0), timestep=1e3, n_steps=750_000,
)
sim.run_one(config, skip_equilibration=True)   # soft repulsion tolerates initial overlaps
```

Soft mode round-trips through JSON, works for single runs, phases, and ensembles, and produces
a distinct `..._soft_kQQ…_kFF…_kQF…_…` filename (the three free-free constants; no `eQQ`, since
epsilon is unused). Existing WCA/LJ datasets and filenames are unchanged.

## Two things to keep in mind

1. **"Reachable time" is a statement about the model, not physical fidelity.** Reachable time
   `= n_steps × dt`. The paper reaches minutes because its particles are genuinely µm-scale and
   slow (`D ≈ 5×10⁻⁶ nm²/ns`). Qt/Ft are nanoscale and really diffuse fast; the per-step
   displacement `√(2·D·dt)` must stay small (≪ particle radius, and ≪ `binding_radius` for
   reaction detection), so a larger `dt` requires a **lower `D`**. `D` is a manual config input
   (no Stokes–Einstein helper) — choose it deliberately, and always report reachable time
   *together with the assumed `D`*.
2. **Reaction kinetics degrade at large `dt`.** Binding fires per step with
   `p = 1 − exp(−kon·dt)`; as `dt` grows, `p → 1` and every contact binds on the first step, so a
   fast rate can no longer be resolved. The calibration tool reports this and suggests the
   largest faithful `kon`/`koff` at each `dt`. (The internal retype rate already scales as
   `1/dt`, so it stays stable automatically.)

Two stability bounds govern the largest usable `dt`:

| Constraint | Bound | Lever |
|---|---|---|
| Diffusion / reaction detection | `√(2·D·dt) ≪ r_particle`, `binding_radius` | lower `D` |
| Harmonic-bond relaxation | `dt ≲ 2·kᵦT / (k_bond·D)` | softer `k_bond`, lower `D` |

The bond-relaxation bound is usually the binding one: at the stiff bond of the old LJ setup
(`k_bond=10`) bonds blow up long before diffusion does. This is why the current preset pairs a
**soft bond** (`k_bond=1.0`) with a much lower `D` (Qt 2e-4, Ft 5e-4 nm²/ns) — the µs timestep
is bought with genuine further coarse-graining, not for free.

## Calibrating (measure-first)

`scripts/calibrate_timestep.py` sweeps `(timestep × diffusion)` in soft mode with short runs and
reports, per cell: stability (finite positions + bond-length drift vs `r₀`), the diffusion
criterion `√(2·D·dt)`, per-step reaction saturation, the largest stable `dt`, and the reachable
time for a step budget. It manages its own output paths (so the `D`-sweep does not collide) and
can write the full table to CSV.

```bash
python scripts/calibrate_timestep.py \
    --timesteps 0.05 0.5 5 50 500 \      # PICOSECONDS
    --diffusion-scales 1 0.1 0.01 \       # multipliers on base Qt/Ft diffusion
    --qt-diffusion 0.5 --ft-diffusion 1.0 \
    --k-bond 0.5 --repulsion-force-constant 5.0 \
    --kon 0.01 --step-budget 2000000 --output-csv calibration.csv
```

Key flags: `--k-bond` and `--repulsion-force-constant` (the two softness levers; the latter sets
the three free-free `soft.k_*` uniformly for the sweep — use a `--config` JSON if you need them to
differ per pair), `--diffusion-scales`, `--p-target` (per-step probability treated as the
stochastic limit for the rate guidance), `--step-budget` (steps used for the reachable-time
column). Review the
largest-stable-`dt` / reachable-time table **before** committing to a production timescale, then
decide whether the physics-faithful `dt` is enough or further coarse-graining (softer bond, lower
`D`, rescaled `kon`/`koff`) is warranted.

## Choosing the force constants `soft.k_*` (least overlap)

Interpenetration falls monotonically with `k`, so there is no interior optimum — the
question is the largest `k` that is still stable at the production `dt`. The bound is the
per-step **overshoot ratio**

```
alpha = k · D · dt / (kB·T)          kB·T = 2.494 kJ/mol at 300 K
```

A particle pushed out of an overlap `δ` moves `alpha·δ` in one Euler step, so `alpha ≥ 1`
means it overshoots and the pair oscillates. A cross pair is governed by the *faster*
species. Sweep it with `scripts/calibrate_soft_k.py`, which reports `alpha` alongside the
measured overlap, stability, and — importantly — the bound fraction and cluster sizes.

Three things that are easy to get wrong:

- **Rank on the unconditional overlap** (`mean_overlap_all_frac`), not on the mean over
  overlapping pairs. Stiffening a pair removes the *shallow* overlaps first, so the
  conditional mean stays flat while total interpenetration falls several-fold.
- **The pairs are not equivalent — `Qt–Ft` is the reactive pair.** Since
  `binding_radius ≈ r_Qt + r_Ft`, the Qt–Ft repulsion acts over exactly the range where
  binding must happen, so stiffening `k_QtFt` shortens the contact residence time and
  suppresses aggregation. `k_QtQt` / `k_FtFt` are non-reactive and have no such cost.
- **`alpha < 1` was necessary but not the binding constraint** in the runs tested: no
  numerical blow-up appeared even at `alpha ≈ 1.6`, because deep overlaps are rare. What
  degraded first was the *physics* (aggregation), not the integrator.

Measured on the notebook's 200 Qt + 400 Ft soft preset (`dt = 1 µs`, `D_Qt = 2e-4`,
`D_Ft = 5e-4`, `k_bond = 1`, `allow_loops=True`), 100 000 steps = 100 ms, 3 seeds — mean
interpenetration over **all** pairs, as % of contact:

| `(k_QtQt, k_FtFt, k_QtFt)` | `alpha_max` | Qt–Qt | Qt–Ft | Ft–Ft | bound Ft | avg cluster |
|---|---|---|---|---|---|---|
| (0.5, 2, 1.5) — previous | 0.40 | 0.0096 ± 0.0009 | 0.0101 ± 0.0005 | 0.0011 ± 0.0002 | 0.934 ± 0.005 | 9.3 ± 1.0 |
| **(4, 3, 1.5)** — adopted | 0.60 | **0.0012 ± 0.0001** | 0.0094 ± 0.0001 | 0.0009 ± 0.0002 | 0.948 ± 0.008 | 8.2 ± 0.6 |
| (8, 4, 1.5) | 0.80 | 0.0006 ± 0.0001 | 0.0092 ± 0.0001 | 0.0005 ± 0.0002 | 0.944 ± 0.006 | 8.0 ± 0.7 |

`Run_Simulation.ipynb` now ships the adopted row, `soft.k_* = (4, 3, 1.5)`.

`k_QtQt` is the free win — it was ~10× below its ceiling, and raising it to 4 cuts Qt–Qt
interpenetration ~8× (fraction of Qt–Qt pairs overlapping: 0.26 % → 0.08 %) with no effect
on binding. `k_FtFt` matters little (Ft–Ft overlap is already rare). `k_QtFt` is the only
lever on the *dominant* Qt–Ft term, but raising it 1.5 → 6 drops bound Ft from 0.95 to 0.69
and mean cluster size from 10.5 to 2.3 — so leave it at 1.5. To reduce Qt–Ft
interpenetration without that cost, widen `binding_radius` beyond contact (giving a
reactive shell outside the repulsive core) or stiffen `k_bond`, rather than `k_QtFt`.

> Note: with `kernel="CPU"` and `n_threads > 1`, runs are **not** reproducible from
> `rng_seed` — repeating an identical config gives slightly different trajectories. Compare
> parameter sets across several seeds, not from single runs.

## Validating (calibrate-then-predict)

Following the paper's validation pattern: fix parameters in one condition, then run a *second*
condition (e.g. different particle counts or box size) **without retuning** and check that trends
hold. Use the existing ensemble machinery ([Running ensembles](../README.md#7-running-ensembles), `qtft/ensemble.py`)
for replicate statistics (SEM/SD over 3–4 replicates), exactly as for WCA/LJ runs.
