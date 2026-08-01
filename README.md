# iPRD Simulation of an Encapsulin–Ferritin Sensor (ReaDDy2)

Coarse-grained **interacting-particle reaction–dynamics (iPRD)** simulation of an
**encapsulin–ferritin sensor** — **Qt encapsulins (Qt)** and **ferritin (Ft)**
nanoparticles agglomerating in solution — built on [ReaDDy2](https://readdy.github.io/).
Two diffusing species bind into growing clusters ("topologies") through stochastic spatial
reactions; the code measures the resulting **agglomeration kinetics** and **cluster
morphology**, across single runs and multi-replica ensembles, locally or on a SLURM cluster.

The overall pipeline is:

```
configure ─▶ equilibrate (WCA, no reactions) ─▶ production (LJ + reactions)
          ─▶ analyze trajectory ─▶ plot ─▶ (ensemble averaging / cross-ensemble comparison)
```

<p align="center">
  <img src="docs/images/simulation_timeseries.png" alt="Qt–Ft agglomeration over time: initially dispersed Qt (green) and Ft (purple) particles bind into growing clusters (QtC blue, FtC red) by 100 µs" width="100%">
</p>

<p align="center"><em>A single 100 µs run in the earlier LJ setup (OVITO render): free Qt (green) and Ft (purple) start
dispersed and progressively bind into growing clusters of QtC (blue) and FtC (red). Bottom row is a
zoomed detail. The current preset (§5) uses soft mode to reach far longer simulated times.</em></p>

---

## 1. Physical model

**Species.** Two free particle types plus their auto-derived "clustered" counterparts, all
managed as ReaDDy *topology species* inside a single topology type `QtFt_Cluster`:

| Symbol | Meaning              | State            |
|--------|----------------------|------------------|
| `Qt`   | Qt encapsulin        | free / monomer   |
| `Ft`   | Ferritin             | free / monomer   |
| `QtC`  | Qt in a cluster      | bound            |
| `FtC`  | Ft in a cluster      | bound            |

**Reactions** (spatial topology reactions, Gillespie handler). All fire when two eligible
particles come within `binding_radius`, at rate `kon`:

| Name                       | Reaction                          | Role                       |
|----------------------------|-----------------------------------|----------------------------|
| `seed_QtFt_Cluster`        | `Qt + Ft → QtC–FtC`               | nucleate a new cluster     |
| `grow_QtC_Ft_QtFt_Cluster` | `QtC + Ft → QtC–FtC`              | cluster captures a free Ft |
| `grow_FtC_Qt_QtFt_Cluster` | `FtC + Qt → FtC–QtC`              | cluster captures a free Qt |
| `merge_QtC_FtC_QtFt_Cluster`| `QtC + FtC → QtC–FtC`            | two clusters merge         |

<p align="center">
  <img src="docs/images/reaction_types.png" alt="Schematic of the ReaDDy topology reactions: seed, grow, and merge, each firing at rate kon within the binding radius and reversible via koff" width="80%">
</p>

<p align="center"><em>The topology reactions: a <strong>seed</strong> nucleates a cluster from a free Qt + Ft, <strong>grow</strong>
reactions capture a free monomer onto an existing cluster, and <strong>merge</strong> joins two clusters — each
firing at rate k<sub>on</sub> within r<sub>bind</sub> and reversible at rate k<sub>off</sub> in deagglomeration phases.</em></p>

**Monovalent Ft (`topology.ft_monovalent`, default `False`).** ReaDDy has no built-in bond cap;
valence is governed purely by which particle types appear as reactants. In both `seed` and
`grow_QtC_Ft` the particle that becomes `FtC` is a *free* Ft gaining its first bond, whereas
`grow_FtC_Qt` and `merge_QtC_FtC` are the only reactions that give an already-bonded `FtC` a
*second* bond. Setting `ft_monovalent=True` skips those two reactions, so `FtC` is terminal and
every Ft forms **at most one bond**. Clusters then become **single-Qt stars** (one multivalent Qt + N monovalent Ft leaves): two clusters never merge, and a free Qt joins only by seeding with
a free Ft. Qt stays multivalent. Default `False` reproduces the original multivalent model.

**Loop-permitting binding (`topology.allow_loops`, default `False`).** All four reactions are
ReaDDy *fusions* between two **different** topologies, so a bond never forms between two particles
already in the same cluster — clusters are strictly **acyclic trees** (see §11). Only
`merge_QtC_FtC` ever has both partners already clustered (`seed`/`grow_*` always consume a *free*
monomer), so setting `allow_loops=True` registers just that one reaction with ReaDDy's `[self=true]`
flag, letting it fire **within** a cluster. Two already-clustered particles can then bond and close
a **ring**, turning clusters into crosslinked networks rather than trees. It has effect only when
`ft_monovalent=False` (a leaf Ft cannot form the second bond that closes a loop), and adds a
`_loops` filename tag. Default `False` reproduces the acyclic-tree model exactly.

**Potentials.**
- **Pairwise Lennard-Jones** for excluded volume, registered for all 10 type pairs.
  `potential_type="WCA"` → purely repulsive (cutoff `2^(1/6)·σ`); `"LJ"` → full attractive
  well (cutoff `2.5·σ`). σ is set so the LJ minimum / WCA exclusion fall at the contact
  distance: `σ = (r_i + r_j) / 2^(1/6) ≈ 0.8909·(r_i + r_j)`, which puts the well minimum at
  `r_i + r_j` = the harmonic bond length. ε is set per pair through a cascade of
  defaults.
- **Soft harmonic repulsion** (`potential_type="soft"`) is a third mode for reaching much
  larger timesteps — see **[§12 Soft mode & timestep](#12-soft-mode--reaching-larger-timesteps)**.
  Instead of the stiff `r⁻¹²` LJ wall, each pair gets ReaDDy's `add_harmonic_repulsion`
  (a bounded, linear force vanishing at contact `r_i + r_j`), with a **per-pair** force constant
  `soft.k_QtQt / k_FtFt / k_QtFt` (cluster/mixed pairs cascade, same as the LJ ε cascade;
  `k = 0` disables a pair). No attractive term. Soft mode is self-contained — `lj.epsilon_*` is
  **ignored** in soft mode. Because overlaps produce small finite forces (not an `r⁻¹²` blow-up),
  a far larger stable `dt` is possible.
- **Weak interaction, piecewise-harmonic** (`potential_type="weak"`) is a fourth mode: ReaDDy's
  `add_weak_interaction_piecewise_harmonic` — a *soft attractive* pair potential (harmonic
  repulsive branch + an attractive well of depth `depth` whose minimum sits at contact
  `r_i + r_j`, returning to 0 at `cutoff`). It gives LJ-like attraction **without** the `r⁻¹²`
  wall, so it runs at a much larger `dt` than 12-6 LJ. Per-pair force constants `weak.k_*` and
  well depths `weak.depth_*` (both cascade like the LJ ε; `k = 0` disables a pair);
  `cutoff = weak.cutoff_factor × contact`. Self-contained — `lj.epsilon_*` is ignored in weak
  mode.
- **Harmonic bonds** (`k_bond`) hold bonded particles inside a cluster at equilibrium length
  `r_Qt + r_Ft`.

**Equilibration vs production.** Equilibration runs with **reactions disabled** and a
purely-repulsive **WCA** potential (the `equilibration_potential` config field, default
`"WCA"`) to relax initial random positions without attraction; production then switches on
attractions (LJ via `config.potential_type`) and the binding reactions. This split is handled by
`equilibrate_system()` + `run_simulation()`. Set `equilibration_potential="LJ"` to equilibrate
under the full attractive potential instead. `equilibration_potential` (and each phase's
`potential_type`) also accepts `"soft"`; in soft mode equilibration is optional altogether
(harmonic repulsion tolerates initial overlaps), so a soft run can start straight from random
placement with `run_one(..., skip_equilibration=True)`.

**Deagglomeration & cycling (`config.phases`).** A run can be split into a sequence of *phases*
to model an **agglomeration ↔ deagglomeration cycle** (e.g. bind for 50 µs, then dissolve for
50 µs, repeatably). Each `PhaseConfig` specifies `n_steps` plus the physics for that phase:
`binding` (spatial binding reactions on/off), `breaking` (bond breaking on/off), and
`potential_type` (`"LJ"` attractive or `"WCA"` repulsive). Bond breaking uses **structural
topology dissociation**: a bonded cluster loses one uniformly-random bond at total rate
`n_edges × topology.koff`, splitting into sub-clusters; a freed monomer is automatically re-typed
back to its free species (`QtC→Qt`, `FtC→Ft`). Build it with `make_agg_deagg_phases(agg_steps, deagg_steps, n_cycles=...)`. Because
ReaDDy cannot change reactions mid-run, `run_phased()` runs each phase as a separate segment and
carries state (positions **and** bonds) across phases via ReaDDy checkpoints. Each phase writes its
own `phase_NNN/trajectory.h5`; analysis stitches them onto one continuous time axis
(`analysis.load_phased_observables`, `plotting.plot_phased_kinetics`). When `phases` is unset
(default), behavior and on-disk filenames are exactly as before. Works for single runs and
ensembles alike. Single dispatch point: `run_one()` calls `run_phased()` automatically
when `config.phases` is set.

**Integrator / environment.** EulerBD Brownian-dynamics integrator, Gillespie reactions,
cubic box, `T = 300 K`.

**Boundaries (`config.boundary`, default `"periodic"`).** `"periodic"` wraps the box.
`"reflective"` switches periodicity off and confines every species with a repulsive box
potential (ReaDDy's `add_box`, stiffness `config.wall_force_constant`, default 5.0) —
ReaDDy has no specular-reflection boundary, so a soft wall is the idiom. The wall spans the
**full** box, so particle *centres* are confined and the accessible volume, hence the
concentration, is identical to a periodic run; a particle's volume may protrude by up to its
radius, and thermal penetration of the wall itself is `≈ √(2·kᵦT/k)` ≈ 1 nm at the default
stiffness (measured max ≈ 2 nm, against a 25 nm Qt radius). Walls are registered inside
`create_system`, so they apply to equilibration, plain production **and every phase** of an
agglomeration↔deagglomeration cycle, and therefore to ensembles too. Reflective runs get a
`_reflective` filename tag (§10).

The analysis honours the setting: the minimum-image convention and the per-cluster
periodic unwrap are applied only when the box is periodic (`config.is_periodic`). This
matters — in a 500 nm box, a pair more than 250 nm apart would otherwise be wrapped and
reported far closer than it is. Two things remain periodic-flavoured and are mildly biased
near walls: ReaDDy's RDF observable, and the `expected_nn_dist` reference value in
`get_spatial_distribution`.

---

## 2. Repository layout

All code lives in the **`qtft`** package; `scripts/` holds thin CLI wrappers.

| Module | Purpose |
|--------|---------|
| `qtft.config` | Config dataclasses (`SimulationConfig` etc.), the `format_param_string` naming convention, and the `NS_TO_US`/`_steps_to_us` units helpers. Single source of truth. |
| `qtft.system` | ReaDDy system builders: `create_system`, species/potentials/topologies. |
| `qtft.engine` | Build + run: `create_simulation`, `place_particles`, `run_simulation`, `equilibrate_system`, and the one-shot `run_one`. |
| `qtft.ensemble` | `EnsembleSimulation` class — multi-replica orchestration, local/parallel runs, SLURM script generation, result collection, statistics, save/load. |
| `qtft.analysis` | Matplotlib-free trajectory analysis: cluster stats, bond counts, binding kinetics, morphology (Rg), spatial distribution, contacts, composition, size fractions. Also `convert_h5_to_xyz` (OVITO), `load_ensemble_data`, and numeric results tables (`build_final_state_table`, `save_table_files`). |
| `qtft.plotting` | All matplotlib plots: single-run, ensemble, and cross-ensemble comparison figures, including the composite "thesis" panels `plot_ensemble_panel` / `plot_comparison_panel` (each writes paired SVG + PNG via `save_path_base`). |
| `qtft.comparison` | Cross-ensemble comparison helpers (`compare_ensembles`, `save/load_comparison_data`, `build_comparison_table`, …). |
| `qtft.fibsem_export` | Export the **final frame** to the FIB-SEM segmentation schema for experiment comparison: encapsulin (Qt/QtC) centroids with ground-truth cluster IDs. Read-only — it consumes a finished trajectory and never runs a simulation. See **[§13](#13-fib-sem-comparison-export)**. |
| `scripts/analyze_ensemble.py` | CLI to (re)analyze an ensemble directory in parallel; `compare` subcommand. |
| `scripts/run_replica.py` | CLI to run **one** replica from a config JSON (used locally and by SLURM job arrays). |
| `scripts/calibrate_timestep.py` | CLI "measure-first" sweep over `(timestep, diffusion)` in **soft** mode: reports stability (finite + bond-length drift), the diffusion criterion, reaction-probability saturation, the largest stable `dt`, and reachable simulated time. See **[§12](#12-soft-mode--reaching-larger-timesteps)**. |
| `scripts/calibrate_soft_k.py` | CLI sweep over the **soft**-mode force constants `soft.k_*`: measures interpenetration (via `analysis.get_overlap_statistics`) against numerical stability, reported with the per-step overshoot ratio `alpha = k·D·dt/(kB·T)`. See **[§12](#12-soft-mode--reaching-larger-timesteps)**. |
| `Run_Simulation.ipynb` | Run-only notebook: one **Configuration** cell (all parameters) + one **Run** cell that dispatches on `RUN_MODE` (`single`/`ensemble`) and `ENABLE_DEAGG` (plain vs agglomeration↔deagglomeration cycling); optional SLURM cell. No plotting. |
| `Plot_Simulation_Results.ipynb` | Plotting/reporting notebook: one **Settings** cell + one **Run** cell selected by `MODE` (`single` trajectory / `ensemble` directory / `comparison` of several). Each mode auto-generates the plots **and** the text summary **and** the data/table exports (CSV/LaTeX) into a `Plots/` folder. |
| `Export_for_FIB-SEM_Comparison.ipynb` | Export notebook: points at a finished run, writes the encapsulin centroid CSV + metadata JSON in the FIB-SEM schema, and plots a cluster-coloured scatter as a periodic-unwrap sanity check. See **[§13](#13-fib-sem-comparison-export)**. |

---

## 3. Requirements

- Python 3.x with **ReaDDy2** (install via conda; the SLURM scripts assume a conda env named
  `readdy`):
  ```bash
  conda create -n readdy -c readdy -c conda-forge readdy
  conda activate readdy
  ```
- `numpy`, `matplotlib`, `pandas`, `h5py` (pulled in by ReaDDy / standard scientific stack).
- For visualization of `.xyz` exports: [OVITO](https://www.ovito.org/) (external, optional).

Progress messages are emitted through the `qtft` logger (streamed to stdout by default, so
notebook output is unchanged). Quiet or redirect it with
`qtft.set_log_level(logging.WARNING)`; the formatted `print_*` summary/report functions always
write to stdout.

---

## 4. Quick start — single run

The explicit building blocks are shown below. `Run_Simulation.ipynb` instead calls the one-shot
`sim.run_one(config)`, which wraps exactly these steps:

```python
import qtft as sim
import qtft.analysis as analysis
import qtft.plotting as plotting

# 1. Configure (the current notebook preset — soft mode, see §5 and §12)
config = sim.SimulationConfig(
    qt=sim.ParticleConfig("Qt", radius=25.0, diffusion=2e-4, cluster_diffusion=2e-4),
    ft=sim.ParticleConfig("Ft", radius=7.0, diffusion=5e-4, cluster_diffusion=5e-4),
    topology=sim.TopologyConfig(binding_radius=32.0, kon=1e-6, k_bond=1.0,
                                allow_loops=True),
    potential_type="soft",      # top-level selector: "WCA" | "LJ" | "soft" | "weak"
    soft=sim.SoftPotentialConfig(k_QtQt=4.0, k_FtFt=3.0, k_QtFt=1.5),
    equilibration_potential="soft",   # WCA/LJ would blow up at this dt
    box_size=(500.0, 500.0, 500.0),
    temperature=300.0,
    timestep=1e3,         # ns  (=1 µs)
    n_steps=750_000,      # → 750 ms total
    record_stride=100,
    observable_stride=100,
    particles_observable_stride=None,   # structural analysis reads positions from the trajectory
    n_qt=200,
    n_ft=400,
)

# 2. Run. Soft repulsion tolerates initial overlaps, so equilibration is optional:
trajectory = sim.run_one(config, skip_equilibration=True)

# ...or drive the same steps explicitly (equilibrate → build → place → run). With an
# attractive production potential ("LJ"), equilibrate first under the repulsive
# equilibration_potential to relax the random starting positions:
# pos_qt, pos_ft = sim.equilibrate_system(config, n_steps=10000)
# system     = sim.create_system(config)
# simulation = sim.create_simulation(system, config, overwrite=True)
# sim.place_particles(simulation, config, positions_qt=pos_qt, positions_ft=pos_ft)
# trajectory = sim.run_simulation(simulation, config)

# 6. Analyze + plot
analysis.print_analysis_summary(config.output_file, config)
plotting.plot_observables(config.output_file, config, save_path="plots_observables.svg")
plotting.plot_cluster_analysis(config.output_file, config=config, save_path="plots_clusters.svg")

# Optional: export for OVITO, and save the config
analysis.convert_h5_to_xyz(config.output_file, config.output_file.replace(".h5", ".xyz"), config, overwrite=True)
config.save_json("simulation_config.json")
```

`config.output_file` is auto-generated from the parameters if left `None`.

---

## 5. Configuration reference

`SimulationConfig` (in `qtft.config`) is the single source of truth and is
fully JSON-serializable (`config.save_json(...)` / `SimulationConfig.load_json(...)`,
`from_dict` / `to_dict` / `to_flat_dict`).

The table below leads with the **current values in `Run_Simulation.ipynb`** — the soft-mode,
750 ms preset. The code dataclass defaults are small smoke-test values — see the footnote.

> The ensembles in `Different_Particle_Ratios/` predate this preset: they were produced with
> the earlier **LJ** parameters (50 ps timestep, 100 µs, Qt r=21 / Ft r=6). Each dataset's
> exact parameters are recorded in its own `ensemble_config.json`, and are encoded in its
> directory name (§10).

| Parameter | Meaning | Units | Notebook value |
|-----------|---------|-------|----------------|
| `qt.radius`, `qt.diffusion` | Qt encapsulin size & diffusion | nm, nm²/ns | 25.0, 2e-4 |
| `qt.cluster_diffusion` | Qt diffusion once bound in a cluster | nm²/ns | 2e-4 (= monomer, see §11) |
| `ft.radius`, `ft.diffusion` | Ft ferritin size & diffusion | nm, nm²/ns | 7.0, 5e-4 |
| `ft.cluster_diffusion` | Ft diffusion once bound in a cluster | nm²/ns | 5e-4 (= monomer, see §11) |
| `n_qt`, `n_ft` | particle counts | – | 200 / 400 |
| `topology.binding_radius` | reaction capture distance | nm | 32.0 (= r_Qt+r_Ft, i.e. contact) |
| `topology.kon` | microscopic binding rate (per-pair; see §11) | 1/ns | 1e-6 |
| `topology.k_bond` | harmonic bond stiffness | kJ/(mol·nm²) | 1.0 (soft, for the large `dt`) |
| `topology.ft_monovalent` | cap Ft at one bond → single-Qt-star clusters | – | `False` |
| `topology.allow_loops` | let `merge_QtC_FtC` self-fuse → intra-cluster loops (crosslinked networks, not trees); needs `ft_monovalent=False`; adds `_loops` tag | – | `True` |
| `topology.koff` | bond-breaking rate per edge (deagglomeration phases only) | 1/ns | 0.0 |
| `phases` | optional list of `PhaseConfig` for agglomeration↔deagglomeration cycling; `None` = single run | – | `None` |
| `lj.epsilon_QtQt/FtFt/QtFt` | well depths for the three free pairs — **ignored in soft mode** | kJ/mol | 1.5 / 1.5 / 3.0 (unused) |
| `potential_type` | **top-level** production selector: `"WCA"` (repulsive), `"LJ"` (attractive), `"soft"` (harmonic repulsion, [§12](#12-soft-mode--reaching-larger-timesteps)), or `"weak"` (piecewise-harmonic weak interaction). Picks which block is registered (`lj.epsilon_*` / `soft.k_*` / `weak.k_*,depth_*`); the others are ignored | – | `soft` |
| `soft.k_QtQt/k_FtFt/k_QtFt` | per-pair harmonic-repulsion stiffness (free-free; cluster/mixed cascade), used only when `potential_type="soft"`; `k=0` disables a pair | kJ/(mol·nm²) | 4.0 / 3.0 / 1.5 (calibrated, [§12](#12-soft-mode--reaching-larger-timesteps)) |
| `weak.k_QtQt/…` / `weak.depth_QtQt/…` | per-pair force constant + well depth for `potential_type="weak"` (free-free; cluster/mixed cascade); `k=0` disables a pair | kJ/(mol·nm²), kJ/mol | 0.5 / 3.0 / 2.0 and 0.25 / 0.1 / 8.0 (unused in soft mode) |
| `weak.cutoff_factor` | weak-mode cutoff as a multiple of contact (`cutoff = factor × (r_i+r_j)`, must be > 1) | – | 1.1 |
| `box_size` | cubic box edge | nm | (500, 500, 500) |
| `boundary` | `"periodic"` (wrap) or `"reflective"` (periodicity off + repulsive box walls on all species; adds a `_reflective` filename tag). Read it via `config.is_periodic` | – | `periodic` |
| `wall_force_constant` | reflective-wall stiffness; ignored when periodic. Too soft leaks (k=1 → 10/600 out), 5.0 confines | kJ/(mol·nm²) | 5.0 |
| `temperature` | – | K | 300 |
| `equilibration_potential` | potential during equilibration (`"WCA"`, `"LJ"`, `"soft"`, or `"weak"`); reactions always off | – | `soft` (must be soft/weak at this `dt`) |
| `timestep` | integration step | ns | 1e3 (1 µs) |
| `n_steps` | total steps (→ 750 ms) | – | 750,000 |
| `record_stride`, `observable_stride` | save cadence | steps | 100 |
| `particles_observable_stride` | per-particle position cadence. **Optional/redundant** — `None` (default) is recommended: structural/morphology/overlap analyses read positions from the recorded trajectory (`record_stride`). Set an integer only to speed up per-frame structural analysis, at the cost of storing positions twice | steps | `None` |
| `heavy_observable_stride` | cadence for unread heavy observables (forces, virial); `None`=100×`observable_stride` | steps | optional |
| `kernel`, `n_threads` | `"CPU"`/`"SingleCPU"`, threads. Note: with `n_threads > 1` runs are **not** reproducible from the seed | – | CPU, 4 |
| `rng_seed` | RNG seed (per-replica in ensembles) | – | 22 |
| `output_file` | trajectory path (`None` = auto) | – | auto |

> **Code dataclass defaults** (smoke-test only, *not* the real runs): Qt r=1.0 D=5.0,
> Ft r=0.25 D=15.0, `binding_radius=1.5`, `kon=10.0`, `k_bond=20.0`, all ε=10.0,
> `potential_type="WCA"`, `box_size=(50,50,50)`, `timestep=1e-4`, `n_steps=200000`,
> `n_qt=n_ft=200`, `rng_seed=42`.

**Epsilon cascade.** Only the three free–free well depths are normally set; the seven
cluster/mixed pairs inherit them unless overridden:

```
epsilon_QtQt, epsilon_FtFt, epsilon_QtFt        (set these)
  └▶ epsilon_QtCQtC = epsilon_QtQt   (cluster pairs default to free–free)
  └▶ epsilon_FtCFtC = epsilon_FtFt
  └▶ epsilon_QtCFtC = epsilon_QtFt
       └▶ epsilon_QtQtC, epsilon_FtFtC, epsilon_QtCFt, epsilon_QtFtC  (mixed default to cluster)
```

Setting an ε to `0` disables that interaction entirely.

---

## 6. Running ensembles

`EnsembleSimulation` (in `qtft.ensemble`) replicates a base config with
independent RNG seeds, runs the replicas, and aggregates the results.

```python
from qtft import EnsembleSimulation

ensemble = EnsembleSimulation(
    base_config=config,
    n_replicas=10,
    base_dir="ensembles",   # output root
)

# Run all replicas locally (parallel), then auto-collect + compute statistics
ensemble.run_local(parallel=True, n_workers=10, overwrite=True, equilibration_steps=5000)

# Plot
stats, structural, cfg = ensemble.to_plotting_format()
plotting.plot_ensemble_observables(stats, structural, cfg, show_individual=True,
                                   save_path="ensemble_observables.svg")
plotting.plot_ensemble_structural(stats, structural, cfg, show_individual=True,
                                  save_path="ensemble_structural.svg")
```

`run_local` produces an output directory named from the parameter string containing:

```
<ensemble_dir>/
├── configs/config_000.json … config_009.json   # per-replica configs (differ only by seed)
├── replica_000/ … replica_009/                  # each has trajectory.h5 (+ optional trajectory.xyz)
├── logs/                                         # stdout/stderr (SLURM runs)
├── ensemble_config.json                          # base configuration
├── ensemble_statistics.json                      # time-series means ± std (+ per-replica traces)
├── ensemble_structural.npz                       # morphology / spatial / contacts / composition / size fractions
├── ensemble_state.json                           # full state for EnsembleSimulation.load()
└── submit_ensemble.slurm / submit_analysis.slurm # if SLURM scripts were generated
```

Reload a finished ensemble with `EnsembleSimulation.load("<ensemble_dir>")`.

**Phased (agglomeration↔deagglomeration) ensembles.** Give `base_config` a `phases` schedule
and run the ensemble exactly as above — replicas inherit cycling because they all run
through `run_one()`. Each replica then contains `replica_NNN/phase_000/trajectory.h5 …` instead of a
single `trajectory.h5`, and result collection automatically stitches the phases per replica onto one
continuous time axis before averaging, so `ensemble_statistics.json` / `ensemble_structural.npz` keep
the same format (the phase boundaries are identical across replicas).

---

## 7. Cluster (SLURM) execution

For HPC, generate job-array scripts instead of running locally:

```python
ensemble.generate_slurm_scripts(
    partition="cm4_tiny", cluster="cm4", time="08:00:00",
    cpus_per_task=12, memory="32G",
    conda_base="<CONDA_PATH>", conda_env="readdy",
)
ensemble.generate_analysis_slurm_script(
    partition="cm4_tiny", time="04:00:00", cpus_per_task=4, stride=10,
)
```

This writes `submit_ensemble.slurm` (a job array; each task runs one replica via
`scripts/run_replica.py --config configs/config_NNN.json`) and `submit_analysis.slurm` (runs
`scripts/analyze_ensemble.py` once all replicas finish). Submit them with `sbatch`. The SLURM
scripts ship the `qtft/` package and `scripts/` to the cluster (`scp -r qtft scripts ...`).

`scripts/run_replica.py` can also be invoked directly:

```bash
python scripts/run_replica.py --config configs/config_000.json
python scripts/run_replica.py --config configs/config_000.json --equilibration-steps 20000
python scripts/run_replica.py --config configs/config_000.json --skip-equilibration
```

---

## 8. Analysis & outputs

Re-analyze (or analyze for the first time) an ensemble directory in parallel:

```bash
python scripts/analyze_ensemble.py --ensemble-dir <ensemble_dir> --parallel --n-workers 4 --stride 10
```

This (re)writes `ensemble_statistics.json` and `ensemble_structural.npz`.

**Metrics computed** (`qtft.analysis`):
- **Kinetics:** bond counts over time, binding rate, free vs clustered Qt/Ft, fraction bound, half-times.
  `fraction_bound` is **particle-weighted** — `(QtC + FtC) / (all particles)`, each particle
  counted once (`analysis.weighted_fraction_bound`). Note this is not the mean of the two
  per-species fractions: those agree only when the species counts are equal, and diverge as the
  ratio becomes lopsided (measured final-value difference: 0.000 at 200:200, 0.046 at 600:200,
  0.298 at 600:50). The weighted form is used because the comparison tables put this number
  side by side across different Qt:Ft ratios. Per-species fractions remain available as
  `fraction_bound_qt` / `fraction_bound_ft` from `get_binding_kinetics`.
- **Cluster stats:** number of clusters, size distribution, average & largest cluster size, adaptive size-category fractions (monomers / small / medium / large / very large).
- **Morphology:** radius of gyration Rg per cluster and normalized compactness (Rg/Rg_ideal).
- **Spatial:** cluster centers (PBC-aware), inter- and intra-cluster nearest-neighbor distances.
- **Contacts:** coordination numbers per particle type, bonds per cluster.
- **Composition:** Qt-fraction per cluster and vs cluster size.
- **RDF:** Qt/QtC–Ft/FtC radial distribution (registered as a ReaDDy observable).
- **Overlap / interpenetration:** `get_overlap_statistics` pools every pair distance over
  the trailing frames and reports, per species pair, the closest approach, the fraction of
  pairs overlapping, and the mean/p95/max depth as a fraction of contact.
  `print_overlap_summary` prints the closest-approach, deepest-overlap and mean-overlap
  table (the notebooks call this after every run). When comparing parameter sets, rank on
  the mean over *all* pairs (`mean_overlap_all_frac`) — a minimum is an extreme-value
  statistic, and the mean over only the overlapping pairs is selection-biased.

**Output-file inventory:**

| File | Format | Contents |
|------|--------|----------|
| `replica_NNN/trajectory.h5` | HDF5 (ReaDDy) | frames + observables; read with `readdy.Trajectory(path)` |
| `replica_NNN/trajectory.xyz` | extended XYZ | OVITO-friendly export (large; optional) |
| `phase_NNN/trajectory.h5` (phased) | HDF5 (ReaDDy) | one per phase of a cycle (+ `phase_NNN/checkpoints/`) |
| `trajectory_combined.h5` (phased) | HDF5 (ReaDDy) | whole cycle stitched into one continuous trajectory (auto) |
| `ensemble_statistics.json` | JSON | time-series means/stds + per-replica traces + scalar `summary` |
| `ensemble_structural.npz` | NumPy npz | structural arrays (Rg, NN, coordination, composition, size fractions) |
| `ensemble_config.json` | JSON | base configuration |
| `ensemble_state.json` | JSON | full reconstruction state (can be ~130 MB) |

Load aggregated results for plotting with
`stats, structural, config = analysis.load_ensemble_data("<ensemble_dir>")` and summarize with
`analysis.print_ensemble_summary(stats, config)`. For a numeric final-state table (mean ± SD,
exportable to CSV/LaTeX) use `analysis.build_final_state_table(stats, config, structural)` for one
ensemble (pass `structural` to include the radius-of-gyration and Qt-fraction composition rows)
or `comparison.build_comparison_table(comparison)` across ensembles, then
`analysis.save_table_files(df, "<path_base>", caption=..., label=...)`.

---

## 9. Plotting

Driven from `Plot_Simulation_Results.ipynb` (one `MODE`-selected run cell); all plotting functions,
including the composite panels, live in `qtft.plotting`.

**Both modes report the same metrics**, so a figure can be picked for a talk regardless of how
the run was analysed. Some quantities therefore appear in more than one figure by design:

| `MODE` | figures written to `Plots/` |
|---|---|
| `single` (plain **and** phased) | `run_panel` (the 12-metric panel) + `kinetics` + the four detailed figures below |
| `ensemble`, plain | `ensemble_panel` |
| `ensemble`, phased | `ensemble_panel` + `ensemble_kinetics` (from replica means) |

A **single run** gets the ensemble panel via
`analysis.build_single_run_plotting_data(h5, config, stride)`, which returns the same
`(stats, structural, config)` triple as `load_ensemble_data` with `n_replicas=1` (every
`*_std` zero, so no error band is drawn). It composes the ordinary single-run analyses using
the ensemble's own key names, so a run plotted this way and the same run plotted as an
ensemble replica give identical numbers. For a phased run, pass `trajectory_combined.h5`.

The **kinetics** figure (bonds / fraction bound / average cluster size) is
`plot_phased_kinetics`, fed by `analysis.load_phased_observables` (phased single run),
`analysis.build_kinetics_data_single` (plain single run — no phase markers, pass your own
`title`), or `analysis.build_kinetics_data_ensemble` (replica means; per-species bound
fractions are derived from the stored particle counts, so no re-analysis is needed).

**Clusters above a size threshold** (its own cell + figure, `large_clusters_min{N}.svg/.png`,
produced for **all four modes**): `plot_large_cluster_count` draws the number of topologies
holding at least `MIN_CLUSTER_SIZE` particles over time — the quantity the other figures miss,
since `n_clusters` counts every free monomer as a topology and the size categories report
*particle fractions* rather than cluster counts. The curve is characteristically
non-monotonic (it rises as clusters nucleate, then falls as they coalesce into fewer, larger
ones). Data comes from `analysis.get_large_cluster_counts` /
`get_large_cluster_counts_ensemble`, which **re-read the replica trajectories** — the
aggregated `.npz` stores no per-frame size distribution, so this keeps the threshold freely
adjustable with no change to the saved format. Budget ≈ 2 s per replica (≈ 20 s for a
10-replica ensemble). Phased runs get the usual phase-boundary markers.

**Single run, detailed:** `plot_observables`, `plot_cluster_analysis`,
`plot_structural_cluster_analysis`, `plot_cluster_composition`. These now run for phased runs
too, reading the stitched `trajectory_combined.h5`; its `reaction_counts` observable is
deliberately dropped when the phases are combined, so the cumulative-reactions subplot shows
"No data available" there.

**Ensemble:** `plot_ensemble_observables`, `plot_ensemble_structural`,
`plot_ensemble_size_categories` (all support `show_individual=True` to overlay replica traces), or the
composite `plotting.plot_ensemble_panel(stats, structural, config, save_path_base=...)`. All ensemble
plotters share the argument order `(stats, structural, config)` — the same order returned by
`analysis.load_ensemble_data` and `EnsembleSimulation.to_plotting_format`.

> The panel's last-row **Coordination Distribution (Final)** histogram (per-particle QtC/FtC
> coordination, pooled over replicas and scaled to a mean count per replica) reads the
> `final_coord_dist_qt` / `final_coord_dist_ft` / `final_coord_dist_n_replicas` keys of
> `ensemble_structural.npz`. These were added later, so ensembles analysed before then render
> that cell as "No data" — re-run `scripts/analyze_ensemble.py --ensemble-dir <dir>` to
> populate them (no re-simulation needed). Not to be confused with
> `plot_ensemble_structural`'s *Final Mean Coordination per Replica*, which histograms one
> mean value per replica.

**Cross-ensemble comparison:** build a comparison with
`ae.compare_ensembles({label: dir, ...})` (from `qtft.comparison`, imported as `ae`), then:
`plot_comparison_summary`, `plot_comparison_final_state`, `plot_comparison_structural`,
`plot_comparison_size_categories`, or the composite
`plotting.plot_comparison_panel(comparison, save_path_base=...)`. Inspect differing
parameters with `ae.print_parameter_differences(comparison)` and persist with
`ae.save_comparison_data(...)`.

---

## 10. Output-file naming convention

Auto-generated trajectory / ensemble names encode the run parameters:

```
{n_qt}Qt_{n_ft}Ft_{POT}_eQQ{εQtQt}_eFF{εFtFt}_eQF{εQtFt}_kon{kon}_dt{timestep}ps_{total_time}us
```

Example — `600Qt_50Ft_LJ_eQQ1.5_eFF1.5_eQF3_kon0.001_dt50ps_100us`:
600 Qt + 50 Ft, full LJ potential, ε(QtQt)=1.5 / ε(FtFt)=1.5 / ε(QtFt)=3.0 kJ/mol,
binding rate kon=0.001, 50 ps timestep, 100 µs total.

In **soft** mode the epsilon block is replaced by the three free–free force constants (ε is
unused there), and the timestep/duration units adapt to the scale — the current notebook
preset produces `200Qt_400Ft_soft_kQQ4_kFF3_kQF1.5_kon1e-06_dt1us_750ms_loops`:
200 Qt + 400 Ft, soft mode with k(QtQt)=4 / k(FtFt)=3 / k(QtFt)=1.5 kJ/(mol·nm²),
kon=1e-6, 1 µs timestep, 750 ms total, with intra-cluster loops enabled.

When `boundary="reflective"`, a `_reflective` suffix is appended (before `_loops`/`_FtMono`)
so a walled run cannot overwrite a periodic dataset at otherwise identical parameters.

When `topology.ft_monovalent=True`, a `_FtMono` suffix is appended (e.g.
`…_dt50ps_100us_FtMono`) so monovalent and multivalent runs at otherwise-identical parameters
don't collide on disk. Likewise `topology.allow_loops=True` appends a `_loops` suffix (before
`_FtMono`). Both suffixes are absent by default, so existing names are unchanged.

When `config.phases` is set (agglomeration↔deagglomeration cycling), the tail after the shared
identity block is replaced by a phase-specific layout (the `kon…dt…` tail above is not used):

```
{n_qt}Qt_{n_ft}Ft_{POT}_eQQ{e}_eFF{e}_eQF{e}_phases{N}_kon{kon}_aggsteps{A}_koff{koff}_deaggsteps{D}_dt{timestep}ps_{total_time}us
```

Example — `200Qt_400Ft_WCA_eQQ1.5_eFF1.5_eQF3_phases2_kon0.001_aggsteps1000000_koff0.001_deaggsteps1000000_dt50ps_100us`:
the leading block is identical to a single run; then `phases{N}` is the number of phases
(`N = 2 × n_cycles` for `make_agg_deagg_phases`, so cycles = N/2); `kon` is paired with
`aggsteps{A}` (steps of the first agglomeration phase) and `koff` with `deaggsteps{D}` (steps of
the first deagglomeration phase); `{total_time}us` is the **sum** over all phases. A phased run's
outputs live under a directory derived from this name, with one `phase_NNN/trajectory.h5` per
phase (+ a `phase_NNN/checkpoints/` used to hand off state to the next phase). For ordinary single
runs (`phases=None`) the standard `kon…dt…` layout above is unchanged. `_loops` (when
`topology.allow_loops=True`) then `_FtMono` (when `topology.ft_monovalent=True`) are appended last.

After all phases, `run_phased` auto-writes a single **`trajectory_combined.h5`** in the run
directory: the per-phase trajectories stitched onto one continuous step axis (the duplicated
boundary frame is dropped), openable with `readdy.Trajectory`, re-analysable by the `get_*`
functions, and exportable to one `.xyz`. The per-phase files are **kept**. It omits the
`reaction_counts` observable (its schema differs between binding and breaking phases) and is written
with gzip rather than ReaDDy's blosc. Disable with `run_phased(..., combine=False)`; build one
manually with `analysis.combine_phase_trajectories(phase_files, out_file)`.

**Empty-directory cleanup.** The engine creates a run folder only when a simulation actually writes
to it (`create_simulation` makes the output dir), and after each run it removes empty leftover
directories under the output root via `qtft.cleanup_empty_run_dirs(root)` (only file-free trees are
removed; pass `cleanup_empty=False` to skip). You can also call that helper directly.

---

## 11. Limitations

These are deliberate simplifications / open questions in the current physical model, documented
here rather than silently fixed:

- **Cluster diffusion is a single fixed value, not size-dependent.** `ParticleConfig.cluster_diffusion`
  defaults to the monomer `diffusion`, and clusters do not slow down as `D ∝ 1/R`. The current
  notebook preset sets it **equal to the monomer value** (Qt 2e-4, Ft 5e-4 nm²/ns), so bound
  particles diffuse at the same rate as free ones. The knob to make bound particles slower exists
  — set `cluster_diffusion` below `diffusion` — but it is a single constant regardless of cluster
  size, so it cannot reproduce the size dependence.
- **`kon` is a microscopic rate.** It is passed straight to ReaDDy's spatial-reaction `rate`
  (a per-pair `1/time` rate), not the macroscopic `nm³/(ns·particle)` constant the older label
  implied. Treat the swept `kon` values as microscopic rates.
- **Diffusion ratio is not Stokes–Einstein consistent.** The Qt/Ft `D` values are a
  coarse-graining choice and do not follow `D ∝ 1/r` from the radii; this is intentional, noted
  here to avoid confusion.
- **Cluster bond graphs are spanning trees (when `allow_loops=False`, the default).** Every
  reaction is an inter-topology fusion that adds exactly one bond and never closes a ring, so
  clusters are acyclic (`n_bonds = n_particles − 1`); coordination numbers from the bond graph
  reflect that tree, not true spatial contact coordination. Setting `topology.allow_loops=True`
  lets `merge_QtC_FtC` self-fuse, so intra-cluster rings can form (`n_bonds ≥ n_particles`) and the
  clusters become crosslinked networks — bond counts stay exact (edge counts), but the tree
  identity no longer holds.
- **Bond breaking (`koff`) is a mean-field per-edge rate.** Each existing bond breaks at the
  same rate `koff` regardless of its location in the cluster (interior vs leaf) or local geometry;
  the broken edge is chosen uniformly at random, not by force or strain. It is a microscopic
  dissociation rate (1/time), the deagglomeration counterpart of the microscopic `kon`, not a
  macroscopic off-rate. A freed monomer is re-typed back to its free species essentially instantly
  (a fast internal cleanup reaction), so it is indistinguishable from an originally-free particle.
  Note: ReaDDy 2.0.13's built-in `add_topology_dissociation` is bypassed (it is broken in that
  build); `qtft` registers an equivalent custom structural reaction instead.

---

## 12. Soft mode — reaching larger timesteps

Soft mode is what the notebook runs today; this section explains why. The earlier production
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

### Two things to keep in mind

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

### Calibrating (measure-first)

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

### Choosing the force constants `soft.k_*` (least overlap)

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

### Validating (calibrate-then-predict)

Following the paper's validation pattern: fix parameters in one condition, then run a *second*
condition (e.g. different particle counts or box size) **without retuning** and check that trends
hold. Use the existing ensemble machinery (**[§6](#6-running-ensembles)**, `qtft/ensemble.py`)
for replicate statistics (SEM/SD over 3–4 replicates), exactly as for WCA/LJ runs.

---

## 13. FIB-SEM comparison export

To compare a simulated end state against **FIB-SEM** segmentation data, `qtft.fibsem_export`
writes the **final frame** in the same schema the segmentation pipeline produces, so a
simulation drops straight into the existing experimental analysis. It is **read-only**: it
consumes a finished trajectory and never builds or runs a simulation.

Driven from `Export_for_FIB-SEM_Comparison.ipynb` (a settings cell, a run cell, and a
cluster-coloured scatter as a visual unwrap check), or directly:

```python
from qtft.config import SimulationConfig
from qtft import fibsem_export

traj, cfg_path = fibsem_export.find_run_files("Simulation_Files_Single_Runs/<run_dir>")
config = SimulationConfig.load_json(cfg_path)
df, info = fibsem_export.export(traj, config, "FIBSEM_Comparison_Export", voxel_nm=4.0)
```

`find_run_files` resolves the run layouts used here: a plain run (`trajectory.h5`), a phased
agglomeration↔deagglomeration run (`trajectory_combined.h5`, else the last
`phase_NNN/trajectory.h5`), and the matching config (`<param_string>_config.json`, or
`ensemble_config.json` for an ensemble replica).

**What it extracts.**

- **Encapsulins only.** Qt (free) and QtC (bound) are exported; ferritin is sub-resolution in
  FIB-SEM and is dropped — but it still takes part in the periodic unwrap, since it defines
  the Qt–Ft–Qt connectivity of a cluster.
- **Clusters are ground truth**, taken from ReaDDy's topology graph — one topology = one
  cluster, so no DBSCAN is needed on the simulation side. Because every particle is placed as
  its own single-particle topology (`engine.place_particles`), boundness is read off the
  graph: a topology with more than one particle means its encapsulin is bonded (a Qt–Ft dimer
  counts), while a size-1 topology is an unbound encapsulin, i.e. a FIB-SEM singleton.
- **Positions are PBC-unwrapped per cluster**, then shifted so the minimum corner sits at the
  origin (all coordinates ≥ 0).
- **Volumes are analytical** (4/3·π·r³), and `radius_nm` is stored per row so a later notebook
  can compute an exact mass integral ⟨M(R)⟩ from ball–ball intersections, with no voxelisation.

**Outputs** (into the chosen `out_dir`, suffixed with `file_tag`, default `_simulation`):

| File | Format | Contents |
|------|--------|----------|
| `encapsulin_centroids_simulation.csv` | CSV | one row per encapsulin: `label, z/y/x_nm, z/y/x_vox, radius_nm, volume_nm3, cluster, is_clustered` |
| `structural_information_and_metadata_simulation.json` | JSON | source trajectory, final step / µs, encapsulin and cluster counts, cluster-size histogram, box, applied coordinate offset, and the full flattened config |

> **Caveat — periodicity is not preserved.** Each cluster is unwrapped independently and then
> everything is shifted by one global offset. Within a cluster the geometry is exact, but
> between clusters periodicity is gone: a cluster unwrapped past the boundary can end up
> spatially overlapping another, and the bounding volume exceeds the true box. That is
> harmless for per-cluster shape/size statistics, but it biases any metric that samples
> neighbourhoods *across* clusters (⟨M(R)⟩, RDF). Decide deliberately whether such a metric
> should be computed under PBC before the unwrap.
