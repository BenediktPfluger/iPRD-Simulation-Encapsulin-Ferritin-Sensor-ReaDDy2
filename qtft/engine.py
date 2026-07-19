"""qtft.engine -- simulation construction and execution (build, place, run, equilibrate)."""
from __future__ import annotations
import logging

import os
from typing import Optional, Tuple

import numpy as np
import readdy

from .config import SimulationConfig, format_duration
from .system import create_system
logger = logging.getLogger(__name__)


def _make_particle_rngs(seed: int) -> Tuple:
    """
    Create two statistically independent RNGs for Qt and Ft particle placement.
    
    Uses numpy's SeedSequence.spawn() to guarantee independent streams,
    rather than the fragile seed/seed+1 pattern.
    
    Parameters
    ----------
    seed : int
        Base random seed
    
    Returns
    -------
    rng_qt, rng_ft : tuple of numpy.random.Generator
        Independent RNGs for Qt and Ft particles
    """
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(2)
    rng_qt = np.random.default_rng(child_seeds[0])
    rng_ft = np.random.default_rng(child_seeds[1])
    return rng_qt, rng_ft


def cleanup_empty_run_dirs(root: str) -> list:
    """Remove empty directory trees under ``root`` (returns the removed paths).

    A directory is removed only if its **entire** subtree contains no files (i.e. it and
    all descendants are empty) — so real run output is never touched. This sweeps up
    leftover folders such as a run dir that was named/created but never written to
    (e.g. configuring a run then running a differently-named phased run instead).

    ``root`` itself is not removed. Missing ``root`` is a no-op.
    """
    removed = []
    if not root or not os.path.isdir(root):
        return removed
    # Walk bottom-up so empty leaves are removed before their (now-empty) parents.
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        if dirpath == root:
            continue
        if filenames:
            continue
        # Only remove if no remaining entries (sub-dirs were already removed if empty).
        try:
            if not os.listdir(dirpath):
                os.rmdir(dirpath)
                removed.append(dirpath)
        except OSError:
            pass
    if removed:
        logger.info(f"✓ Removed {len(removed)} empty leftover dir(s) under {root}")
    return removed


def _random_positions(config: "SimulationConfig") -> Tuple[np.ndarray, np.ndarray]:
    """Uniform random Qt and Ft positions inside the box.

    Uses independent per-species RNGs seeded from ``config.rng_seed`` (via
    ``_make_particle_rngs``), so equilibration placement and production placement
    are reproducible and consistent. Shared by ``place_particles`` and
    ``equilibrate_system`` to avoid duplicated placement logic.

    Returns
    -------
    pos_qt, pos_ft : tuple of ndarray
        Arrays of shape (n_qt, 3) and (n_ft, 3).
    """
    rng_qt, rng_ft = _make_particle_rngs(config.rng_seed)
    L = np.array(config.box_size, dtype=float)
    pos_qt = rng_qt.uniform(-L / 2.0, L / 2.0, size=(config.n_qt, 3))
    pos_ft = rng_ft.uniform(-L / 2.0, L / 2.0, size=(config.n_ft, 3))
    return pos_qt, pos_ft


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

def create_simulation(
    system: readdy.ReactionDiffusionSystem,
    config: SimulationConfig,
    overwrite: bool = True,
    output_file: Optional[str] = None,
) -> readdy.Simulation:
    """
    Create and configure a ReaDDy simulation.

    Parameters
    ----------
    system : readdy.ReactionDiffusionSystem
        System created by create_system()
    config : SimulationConfig
        Simulation configuration
    overwrite : bool
        If True, overwrite existing output file
    output_file : str, optional
        Trajectory output path. Defaults to config.output_file. Phased runs pass a
        per-phase path here so each phase writes its own trajectory.h5.

    Returns
    -------
    readdy.Simulation
        Configured simulation ready to run
    """
    out_file = output_file if output_file is not None else config.output_file

    # Ensure the output directory exists. The engine (which actually writes the file)
    # owns directory creation, so callers/notebooks don't need to pre-create folders —
    # a run folder is created only when a simulation really writes to it (avoids empty
    # leftover dirs from configuring-but-not-running).
    out_dir = os.path.dirname(out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Handle existing output file
    if os.path.exists(out_file):
        if overwrite:
            os.remove(out_file)
            logger.info(f"✓ Removed existing file: {out_file}")
        else:
            raise FileExistsError(
                f"Output file exists: {out_file}. "
                "Set overwrite=True or choose a different filename."
            )

    # Create simulation
    simulation = system.simulation(
        kernel=config.kernel,
        output_file=out_file,
        integrator="EulerBDIntegrator",
        reaction_handler="Gillespie",
        evaluate_topology_reactions=True,
        evaluate_forces=True,
        evaluate_observables=True,
        skin=0.0,
    )
    
    # Set temperature (timestep is passed to simulation.run() instead)
    simulation.temperature = float(config.temperature)
    
    # Configure threading
    if config.kernel == "CPU" and config.n_threads is not None:
        simulation.kernel_configuration.n_threads = int(config.n_threads)
    
    # Enable trajectory recording
    simulation.record_trajectory(stride=int(config.record_stride))
    
    # Register observables
    _register_observables(simulation, config)
    
    logger.info(f"✓ Simulation created: {config.kernel} kernel, {config.n_threads} threads")
    
    return simulation


def _register_observables(simulation: readdy.Simulation, config: SimulationConfig):
    """Register observables to record during simulation."""
    stride = config.observable_stride
    
    # Particle counts by type
    types = [config.qt.name, config.ft.name, config.qt.cluster_name, config.ft.cluster_name]
    simulation.observe.number_of_particles(stride=stride, types=types)
    
    # Particle positions and types (optional - for faster structural analysis)
    # When enabled, allows using read_observable_particles() instead of trajectory.read()
    # Trade-off: faster analysis but larger HDF5 files (position data stored twice)
    if config.particles_observable_stride is not None:
        simulation.observe.particles(stride=config.particles_observable_stride)
        particles_obs_str = f", particles observable stride={config.particles_observable_stride}"
    else:
        particles_obs_str = ""
    
    # Topologies
    simulation.observe.topologies(stride=stride)
    
    # Thermodynamic observables
    simulation.observe.energy(stride=stride)
    simulation.observe.pressure(stride=stride, physical_particles=None)

    # Heavy observables that are currently not read by any analysis (forces, virial).
    # Recorded on a much coarser stride to keep trajectory files small. Forces in
    # particular dominates trajectory.h5 size when recorded every step.
    heavy_stride = config.effective_heavy_observable_stride
    simulation.observe.forces(stride=heavy_stride, types=None)
    simulation.observe.virial(stride=heavy_stride)
    
    # Reaction counts
    simulation.observe.reaction_counts(stride=stride)
    
    # Radial distribution function
    # Configure RDF range based on particle sizes for better resolution
    box = config.box_size
    contact_distance = config.qt.radius + config.ft.radius
    rdf_max = min(3.0 * contact_distance, 0.5 * min(box))  # 3× contact or half box
    n_bins = 200  # Better resolution than default 100
    simulation.observe.rdf(
        stride=stride,
        bin_borders=(0.0, rdf_max, n_bins),
        types_count_from=[config.qt.name, config.qt.cluster_name],
        types_count_to=[config.ft.name, config.ft.cluster_name],
        particle_to_density=1.0 / (box[0] * box[1] * box[2])
    )
    
    logger.info(f"✓ Observables registered (stride={stride}, "
                f"forces/virial stride={heavy_stride}{particles_obs_str})")


def place_particles(
    simulation: readdy.Simulation,
    config: SimulationConfig,
    positions_qt: Optional[np.ndarray] = None,
    positions_ft: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Place particles in the simulation box.
    
    Parameters
    ----------
    simulation : readdy.Simulation
        Simulation to add particles to
    config : SimulationConfig
        Configuration with particle counts and box size
    positions_qt : ndarray, optional
        Pre-defined positions for Qt particles. If None, random positions are generated.
    positions_ft : ndarray, optional
        Pre-defined positions for Ft particles. If None, random positions are generated.
    
    Returns
    -------
    pos_qt, pos_ft : tuple of ndarray
        Arrays of positions used for Qt and Ft particles
    """
    # Random positions for any species not explicitly provided (deterministic from seed)
    rand_qt, rand_ft = _random_positions(config)

    # Generate or use provided positions for Qt
    if positions_qt is not None:
        pos_qt = np.asarray(positions_qt)
        if pos_qt.shape[0] != config.n_qt:
            raise ValueError(f"positions_qt has {pos_qt.shape[0]} particles, expected {config.n_qt}")
        placement_qt = "provided"
    else:
        pos_qt = rand_qt
        placement_qt = "random"

    # Generate or use provided positions for Ft
    if positions_ft is not None:
        pos_ft = np.asarray(positions_ft)
        if pos_ft.shape[0] != config.n_ft:
            raise ValueError(f"positions_ft has {pos_ft.shape[0]} particles, expected {config.n_ft}")
        placement_ft = "provided"
    else:
        pos_ft = rand_ft
        placement_ft = "random"
    
    # Add as single-particle topologies
    for p in pos_qt:
        simulation.add_topology(config.topology.name, [config.qt.name], p.reshape(1, 3))
    
    for p in pos_ft:
        simulation.add_topology(config.topology.name, [config.ft.name], p.reshape(1, 3))
    
    logger.info(f"✓ Placed {config.n_qt} Qt ({placement_qt}) + {config.n_ft} Ft ({placement_ft}) particles")
    
    return pos_qt, pos_ft


def run_simulation(
    simulation: readdy.Simulation,
    config: SimulationConfig,
    show_progress: bool = True,
) -> readdy.Trajectory:
    """
    Execute the simulation and return the trajectory.
    
    Parameters
    ----------
    simulation : readdy.Simulation
        Configured simulation with particles placed
    config : SimulationConfig
        Simulation configuration
    show_progress : bool
        Print progress messages
    
    Returns
    -------
    readdy.Trajectory
        Trajectory object for analysis
    """
    if show_progress:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"RUNNING SIMULATION")
        logger.info(f"  Particles: {config.n_qt} Qt + {config.n_ft} Ft")
        logger.info(f"  Duration: {format_duration(config.total_simulation_time_us * 1e3)} ({config.n_steps:,} steps)")
        logger.info(f"{'=' * 60}\n")
    
    simulation.run(int(config.n_steps), float(config.timestep))
    
    if show_progress:
        logger.info(f"\n{'=' * 60}")
        logger.info("SIMULATION COMPLETE")
        logger.info(f"{'=' * 60}\n")
    
    # Load trajectory
    if not os.path.exists(config.output_file):
        raise RuntimeError(f"Output file not created: {config.output_file}")
    
    return readdy.Trajectory(config.output_file)


def equilibrate_system(
    config: SimulationConfig,
    n_steps: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run an equilibration simulation to relax initial particle positions.
    
    This runs a short simulation WITHOUT binding reactions, allowing particles
    to move apart if they were initially overlapping. The final positions are
    returned and can be used as initial positions for the production run.
    
    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration
    n_steps : int
        Number of equilibration steps (default: 10000)
    
    Returns
    -------
    positions_qt, positions_ft : tuple of ndarray
        Equilibrated positions for Qt and Ft particles
    
    Example
    -------
    >>> # Equilibrate first
    >>> pos_qt, pos_ft = equilibrate_system(config, n_steps=10000)
    >>> # Then run production with equilibrated positions
    >>> system = create_system(config)
    >>> simulation = create_simulation(system, config)
    >>> place_particles(simulation, config, positions_qt=pos_qt, positions_ft=pos_ft)
    >>> trajectory = run_simulation(simulation, config)
    """
    logger.info(f"\n{'=' * 60}")
    logger.info("EQUILIBRATION")
    logger.info(f"  Running {n_steps:,} steps without reactions "
                f"({config.equilibration_potential} potential)")
    logger.info(f"{'=' * 60}\n")
    
    # Create system in equilibration mode (no reactions)
    system = create_system(config, equilibration_mode=True)
    
    # Create simulation without output file
    eq_simulation = system.simulation(
        kernel=config.kernel,
        output_file="",  # No output file
        integrator="EulerBDIntegrator",
        reaction_handler="Gillespie",
        evaluate_topology_reactions=True,
        evaluate_forces=True,
        evaluate_observables=False,  # No need to record observables
        skin=0.0,
    )
    
    # Set temperature
    eq_simulation.temperature = float(config.temperature)
    
    # Configure threading
    if config.kernel == "CPU" and config.n_threads is not None:
        eq_simulation.kernel_configuration.n_threads = int(config.n_threads)
    
    # Place particles randomly (same deterministic placement as place_particles)
    pos_qt, pos_ft = _random_positions(config)

    for p in pos_qt:
        eq_simulation.add_topology(config.topology.name, [config.qt.name], p.reshape(1, 3))
    
    for p in pos_ft:
        eq_simulation.add_topology(config.topology.name, [config.ft.name], p.reshape(1, 3))
    
    logger.info(f"✓ Placed {config.n_qt} Qt + {config.n_ft} Ft particles (random)")
    
    # Run equilibration
    logger.info(f"  Running equilibration...")
    eq_simulation.run(int(n_steps), float(config.timestep))
    
    # Extract final positions from the simulation context
    # We need to get current particle positions from the simulation
    context = eq_simulation.current_particles
    
    # Separate Qt and Ft particles by type
    qt_positions = []
    ft_positions = []
    
    for particle in context:
        particle_type = particle.type
        position = np.array(particle.pos)
        
        if particle_type == config.qt.name:
            qt_positions.append(position)
        elif particle_type == config.ft.name:
            ft_positions.append(position)
        # Ignore QtC and FtC (shouldn't exist in equilibration, but just in case)
    
    positions_qt = np.array(qt_positions)
    positions_ft = np.array(ft_positions)
    
    logger.info(f"\n{'=' * 60}")
    logger.info("EQUILIBRATION COMPLETE")
    logger.info(f"  Retrieved {len(positions_qt)} Qt + {len(positions_ft)} Ft positions")
    logger.info(f"{'=' * 60}\n")

    return positions_qt, positions_ft


def run_one(
    config: SimulationConfig,
    equilibration_steps: int = 10000,
    skip_equilibration: bool = False,
    overwrite: bool = True,
    show_progress: bool = True,
) -> readdy.Trajectory:
    """Run the full single-replica pipeline and return the trajectory.

    Pipeline: optional equilibration (WCA, no reactions) -> build system ->
    build simulation -> place particles -> production run. This is the single
    implementation shared by the ensemble runners and the run_replica.py CLI
    (previously duplicated in three places).

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration (its output_file determines where the trajectory goes).
    equilibration_steps : int
        Number of equilibration steps (ignored if skip_equilibration=True).
    skip_equilibration : bool
        If True, skip equilibration and start from random positions.
    overwrite : bool
        Overwrite an existing output file.
    show_progress : bool
        Print progress banners for the production run.

    Returns
    -------
    readdy.Trajectory
        Trajectory object for the completed production run.
    """
    # Ensure the output directory exists (create_simulation does not create it).
    output_dir = os.path.dirname(config.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Phased agglomeration<->deagglomeration cycle: delegate to run_phased so every
    # caller (single runs, ensemble workers, the run_replica.py CLI) gets cycling for
    # free, since they all funnel through run_one.
    if config.phases:
        return run_phased(
            config,
            equilibration_steps=equilibration_steps,
            skip_equilibration=skip_equilibration,
            overwrite=overwrite,
            show_progress=show_progress,
        )

    if skip_equilibration:
        pos_qt, pos_ft = None, None
    else:
        pos_qt, pos_ft = equilibrate_system(config, n_steps=equilibration_steps)

    system = create_system(config)
    simulation = create_simulation(system, config, overwrite=overwrite)
    place_particles(simulation, config, positions_qt=pos_qt, positions_ft=pos_ft)
    trajectory = run_simulation(simulation, config, show_progress=show_progress)

    # Sweep empty leftover dirs under the output root (sibling run folders that were
    # configured/named but never written to). Skip when there is no real parent.
    parent = os.path.dirname(output_dir)
    if parent:
        cleanup_empty_run_dirs(parent)

    return trajectory


def run_phased(
    config: SimulationConfig,
    equilibration_steps: int = 10000,
    skip_equilibration: bool = False,
    overwrite: bool = True,
    show_progress: bool = True,
    combine: bool = True,
    cleanup_empty: bool = True,
) -> list:
    """Run an agglomeration<->deagglomeration cycle defined by ``config.phases``.

    ReaDDy cannot change its reaction set mid-``run()``, so each phase is a separate
    simulation segment: the system is rebuilt with that phase's reactions and pair
    potential (via ``create_system(config, phase=...)``), and state is carried over
    between phases with ReaDDy checkpoints (which preserve particle positions *and*
    topology bonds). Phase i writes ``<phase_base_dir>/phase_{i:03d}/trajectory.h5``.

    Parameters
    ----------
    config : SimulationConfig
        Configuration with a non-empty ``phases`` list.
    equilibration_steps : int
        Equilibration steps before phase 0 (ignored if skip_equilibration=True).
    skip_equilibration : bool
        If True, start phase 0 from random positions.
    overwrite : bool
        Overwrite existing per-phase trajectory files.
    show_progress : bool
        Print per-phase progress banners.
    combine : bool
        After all phases, stitch the per-phase trajectories into a single continuous
        ReaDDy trajectory ``<base_dir>/trajectory_combined.h5`` (the per-phase files are
        kept). Only done when there are >= 2 phases.
    cleanup_empty : bool
        After the run, remove empty leftover directories under the output root (the
        parent of the run folder), e.g. a run dir that was configured but never written.

    Returns
    -------
    list of dict
        One entry per phase with keys: index, name, dir, trajectory, n_steps,
        step_offset (cumulative steps before this phase, for stitching a continuous
        time axis), binding, breaking, potential_type. The first entry also carries
        ``combined`` = path to trajectory_combined.h5 (or None) once combining is done.
    """
    if not config.phases:
        raise ValueError("run_phased requires config.phases to be a non-empty list")

    base_dir = config.phase_base_dir
    os.makedirs(base_dir, exist_ok=True)

    # Equilibrate once (WCA, no reactions) to relax initial overlaps before phase 0.
    if skip_equilibration:
        pos_qt, pos_ft = None, None
    else:
        pos_qt, pos_ft = equilibrate_system(config, n_steps=equilibration_steps)

    phase_dirs = config.phase_dirs
    phase_files = config.phase_output_files

    results = []
    prev_checkpoint_dir = None
    step_offset = 0

    for i, phase in enumerate(config.phases):
        phase_dir = phase_dirs[i]
        os.makedirs(phase_dir, exist_ok=True)
        checkpoint_dir = os.path.join(phase_dir, "checkpoints")
        out_file = phase_files[i]

        if show_progress:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"PHASE {i + 1}/{len(config.phases)}: {phase.name}")
            logger.info(f"  {phase.n_steps:,} steps ({format_duration(phase.n_steps * config.timestep)}), "
                        f"binding={phase.binding}, breaking={phase.breaking}, "
                        f"potential={phase.potential_type}")
            logger.info(f"{'=' * 60}\n")

        system = create_system(config, phase=phase)
        simulation = create_simulation(system, config, overwrite=overwrite, output_file=out_file)

        # Checkpoint at the end of the phase so the next phase can resume from it
        # (positions + topology bonds). Keep only the latest save.
        simulation.make_checkpoints(
            stride=int(phase.n_steps), output_directory=checkpoint_dir, max_n_saves=1
        )

        if i == 0:
            place_particles(simulation, config, positions_qt=pos_qt, positions_ft=pos_ft)
        else:
            simulation.load_particles_from_latest_checkpoint(prev_checkpoint_dir)
            logger.info(f"✓ Loaded state (positions + bonds) from {prev_checkpoint_dir}")

        simulation.run(int(phase.n_steps), float(config.timestep))

        if not os.path.exists(out_file):
            raise RuntimeError(f"Phase {i} output file not created: {out_file}")

        results.append({
            "index": i,
            "name": phase.name,
            "dir": phase_dir,
            "trajectory": out_file,
            "n_steps": int(phase.n_steps),
            "step_offset": step_offset,
            "binding": phase.binding,
            "breaking": phase.breaking,
            "potential_type": phase.potential_type,
        })
        step_offset += int(phase.n_steps)
        prev_checkpoint_dir = checkpoint_dir

    if show_progress:
        logger.info(f"\n{'=' * 60}")
        logger.info("PHASED RUN COMPLETE")
        logger.info(f"  {len(results)} phases, {step_offset:,} total steps -> "
                    f"{format_duration(step_offset * config.timestep)}")
        logger.info(f"{'=' * 60}\n")

    # Stitch the per-phase trajectories into one continuous ReaDDy trajectory (kept
    # alongside the per-phase files). Imported lazily to keep engine import light.
    combined_path = None
    if combine and len(results) >= 2:
        from .analysis import combine_phase_trajectories
        combined_path = os.path.join(base_dir, "trajectory_combined.h5")
        try:
            combine_phase_trajectories(
                phase_files, combined_path, step_offsets=config.phase_step_offsets
            )
            logger.info(f"✓ Combined trajectory written: {combined_path}")
        except Exception as e:
            logger.warning(f"Warning: could not write combined trajectory: {e}")
            combined_path = None
    if results:
        results[0]["combined"] = combined_path

    # Sweep empty leftover dirs under the output root (parent of this run's folder).
    # Skip when there is no real parent (avoid sweeping the current working directory).
    if cleanup_empty:
        parent = os.path.dirname(base_dir)
        if parent:
            cleanup_empty_run_dirs(parent)

    return results
