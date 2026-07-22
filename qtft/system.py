"""qtft.system -- ReaDDy ReactionDiffusionSystem builders (species, potentials, topologies)."""
from __future__ import annotations
import logging

from typing import Optional

import numpy as np
import readdy

from .config import SimulationConfig, PhaseConfig


# Multiplier (× 1/timestep) for the rate of the "re-type freed monomer" cleanup reaction.
# A freed monomer (an isolated 1-particle topology still typed QtC/FtC after a bond break)
# is re-typed back to Qt/Ft at rate _RETYPE_RATE_FACTOR / timestep, i.e. essentially within
# the first step, so it is indistinguishable from an originally-placed free particle.
_RETYPE_RATE_FACTOR = 10.0


# σ chosen so the 12-6 LJ minimum (and the WCA exclusion edge, both at 2^(1/6)·σ)
# sit at the contact distance r_i+r_j, matching the harmonic bond length r0
# (config.equilibrium_bond_length). Hence σ = (r_i+r_j) / 2^(1/6). This places the
# LJ attractive minimum and the WCA exclusion exactly at physical contact, so bonded
# pairs are not squeezed by a mismatched LJ minimum (resolves caveat P2).
_SIGMA_AT_CONTACT = 2.0 ** (-1.0 / 6.0)   # ≈ 0.8909
logger = logging.getLogger(__name__)


def create_system(
    config: SimulationConfig,
    equilibration_mode: bool = False,
    phase: Optional[PhaseConfig] = None,
) -> readdy.ReactionDiffusionSystem:
    """
    Create and configure a ReaDDy reaction-diffusion system.

    This sets up:
    - Particle species (free and clustered types)
    - Lennard-Jones potentials for excluded volume
    - Topology type with harmonic bonds
    - Spatial reactions for binding (unless equilibration_mode=True)
    - Bond-breaking (dissociation) reactions when a deagglomeration phase is given

    Parameters
    ----------
    config : SimulationConfig
        Complete simulation configuration
    equilibration_mode : bool
        If True, spatial reactions are NOT registered (no binding occurs) and the
        equilibration potential (config.equilibration_potential, WCA by default) is used.
        Use this for equilibration runs to relax initial particle positions.
    phase : PhaseConfig, optional
        When given (and not equilibration_mode), build the system for this phase of an
        agglomeration/deagglomeration cycle: the phase's pair potential
        (phase.potential_type), spatial binding reactions iff phase.binding, and bond
        breaking (dissociation + freed-monomer re-typing) iff phase.breaking. When None,
        the system is the ordinary single-run production system (binding on, no breaking,
        production potential config.potential_type).

    Returns
    -------
    readdy.ReactionDiffusionSystem
        Configured system ready for simulation
    """
    # Create system with correct box size from the start
    system = readdy.ReactionDiffusionSystem(box_size=list(config.box_size))

    if not config.periodic_boundary:
        system.periodic_boundary_conditions = [False, False, False]

    # Add species
    _add_species(system, config)

    # Resolve potential type and which reactions to register for this build.
    if equilibration_mode:
        # Equilibration: WCA (or config override), no reactions at all.
        potential_override = config.equilibration_potential
        register_binding = False
        register_breaking = False
    elif phase is not None:
        # One phase of a cycle: physics is fully specified by the phase.
        potential_override = phase.potential_type
        register_binding = phase.binding
        register_breaking = phase.breaking
    else:
        # Ordinary single production run (legacy behavior).
        potential_override = None
        register_binding = True
        register_breaking = False

    # Add potentials (production type unless overridden for equilibration / a phase).
    _add_potentials(system, config, potential_type=potential_override)

    # Add topology type, bonds, and the resolved set of reactions.
    _add_topologies(
        system, config,
        register_binding=register_binding,
        register_breaking=register_breaking,
    )

    if equilibration_mode:
        mode_str = " (equilibration mode - no reactions)"
    elif phase is not None:
        mode_str = (f" (phase '{phase.name}': binding={phase.binding}, "
                    f"breaking={phase.breaking}, {phase.potential_type})")
    else:
        mode_str = ""
    logger.info(f"✓ System created: {config.box_size[0]}×{config.box_size[1]}×{config.box_size[2]} nm box{mode_str}")

    return system


def _add_species(system: readdy.ReactionDiffusionSystem, config: SimulationConfig):
    """Add particle species to the system."""
    # Free particle types
    system.add_topology_species(config.qt.name, float(config.qt.diffusion))
    system.add_topology_species(config.ft.name, float(config.ft.diffusion))
    
    # Clustered particle types
    system.add_topology_species(config.qt.cluster_name, float(config.qt.cluster_diffusion))
    system.add_topology_species(config.ft.cluster_name, float(config.ft.cluster_diffusion))
    
    logger.info(f"✓ Species: {config.qt.name}, {config.ft.name}, "
                f"{config.qt.cluster_name}, {config.ft.cluster_name}")


def _add_potentials(
    system: readdy.ReactionDiffusionSystem,
    config: SimulationConfig,
    potential_type: Optional[str] = None,
):
    """Add the excluded-volume pair potentials to the system.

    Two families, selected by potential_type:
      - "WCA"/"LJ": 12-6 Lennard-Jones truncated at cf*sigma (WCA = purely repulsive,
        LJ = attractive well). sigma places the minimum/exclusion at contact r_i+r_j.
        Per-pair magnitude + on/off gate is config.lj.epsilon_* (epsilon == 0 skips a pair).
      - "soft": per-pair harmonic repulsion from config.soft.k_* (a bounded linear force
        that vanishes at the contact distance r_i+r_j; no attractive term, no cutoff;
        enables a much larger stable timestep). Self-contained: config.lj.epsilon_* is
        NOT read in soft mode; k == 0 skips a pair.

    Parameters
    ----------
    potential_type : str, optional
        Override for the potential type ("WCA", "LJ", "soft", or "weak"). If None, uses
        config.potential_type (production). When set (e.g. "WCA" for equilibration, or a
        phase's potential_type), that mode is registered instead.
    """
    lj = config.lj

    # Resolve the effective potential type. Production passes None -> use the top-level
    # config.potential_type selector; equilibration/phase pass an explicit override.
    if potential_type is None:
        potential_type = config.potential_type
    if potential_type not in ("WCA", "LJ", "soft", "weak"):
        raise ValueError(
            f"potential_type must be 'WCA', 'LJ', 'soft', or 'weak', got: {potential_type}")

    qt = config.qt.name
    ft = config.ft.name
    qtc = config.qt.cluster_name
    ftc = config.ft.cluster_name

    n_registered = 0
    n_skipped = 0

    # Build a per-pair registration helper `add_pair(t1, t2, suffix, geomkey)` where suffix
    # is the pair's field name (e.g. "QtQt") and geomkey is "qq" (Qt-family same), "ff"
    # (Ft-family same) or "qf" (cross). Each branch looks up its own per-pair coefficient
    # by suffix and uses it as both the magnitude and the on/off gate (0 => skip).
    if potential_type == "soft":
        # Harmonic repulsion: bounded linear force vanishing at the contact distance
        # r_i+r_j. No attractive term, no cutoff -> tolerant of overlaps, large-dt stable.
        # Self-contained: reads only config.soft (lj.epsilon is ignored in soft mode).
        soft = config.soft
        distances = {
            "qq": 2.0 * config.qt.radius,
            "ff": 2.0 * config.ft.radius,
            "qf": config.qt.radius + config.ft.radius,
        }

        def add_pair(t1, t2, suffix, geomkey):
            nonlocal n_registered, n_skipped
            k = float(getattr(soft, "k_" + suffix))
            if k == 0:
                n_skipped += 1
                return
            system.potentials.add_harmonic_repulsion(
                t1, t2,
                force_constant=k,
                interaction_distance=float(distances[geomkey]),
            )
            n_registered += 1
    elif potential_type == "weak":
        # Piecewise-harmonic weak interaction: soft attractive well of depth `depth` with
        # its minimum at the contact distance r_i+r_j, returning to 0 at cutoff. LJ-like
        # attraction without the r^-12 wall. Self-contained: reads only config.weak.
        weak = config.weak
        distances = {
            "qq": 2.0 * config.qt.radius,
            "ff": 2.0 * config.ft.radius,
            "qf": config.qt.radius + config.ft.radius,
        }

        def add_pair(t1, t2, suffix, geomkey):
            nonlocal n_registered, n_skipped
            k = float(getattr(weak, "k_" + suffix))
            if k == 0:
                n_skipped += 1
                return
            d = float(distances[geomkey])
            system.potentials.add_weak_interaction_piecewise_harmonic(
                t1, t2,
                force_constant=k,
                desired_distance=d,
                depth=float(getattr(weak, "depth_" + suffix)),
                cutoff=float(weak.cutoff_factor * d),
            )
            n_registered += 1
    else:
        # 12-6 Lennard-Jones truncated at cf*sigma. The cutoff factor is derived from the
        # mode (WCA -> 2^(1/6), LJ -> 2.5), unless the user set an explicit lj.cutoff_factor.
        if lj.cutoff_factor is not None:
            cf = lj.cutoff_factor
        elif potential_type == "WCA":
            cf = lj.WCA_CUTOFF_FACTOR
        else:  # "LJ"
            cf = lj.LJ_CUTOFF_FACTOR

        # σ = (r_i+r_j)/2^(1/6) puts the LJ minimum / WCA exclusion at the contact
        # distance r_i+r_j (= bond length); see _SIGMA_AT_CONTACT.
        sigma = {
            "qq": 2.0 * config.qt.radius * _SIGMA_AT_CONTACT,
            "ff": 2.0 * config.ft.radius * _SIGMA_AT_CONTACT,
            "qf": (config.qt.radius + config.ft.radius) * _SIGMA_AT_CONTACT,
        }

        def add_pair(t1, t2, suffix, geomkey):
            nonlocal n_registered, n_skipped
            epsilon = float(getattr(lj, "epsilon_" + suffix))
            if epsilon == 0:
                n_skipped += 1
                return
            s = sigma[geomkey]
            system.potentials.add_lennard_jones(
                t1, t2, m=12, n=6,
                epsilon=epsilon, sigma=float(s), cutoff=float(cf * s),
            )
            n_registered += 1

    # All 10 possible pairwise interactions: (type1, type2, field suffix, geom key).
    pair_specs = [
        # Free-free pairs
        (qt,  qt,  "QtQt",   "qq"),
        (ft,  ft,  "FtFt",   "ff"),
        (qt,  ft,  "QtFt",   "qf"),
        # Cluster-cluster pairs (same species) + cross-species
        (qtc, qtc, "QtCQtC", "qq"),
        (ftc, ftc, "FtCFtC", "ff"),
        (qtc, ftc, "QtCFtC", "qf"),
        # Mixed-state pairs (one free, one clustered)
        (qt,  qtc, "QtQtC",  "qq"),
        (ft,  ftc, "FtFtC",  "ff"),
        (qtc, ft,  "QtCFt",  "qf"),
        (qt,  ftc, "QtFtC",  "qf"),
    ]
    for t1, t2, suffix, geomkey in pair_specs:
        add_pair(t1, t2, suffix, geomkey)

    skip_str = f", {n_skipped} disabled" if n_skipped > 0 else ""
    if potential_type == "soft":
        s = config.soft
        logger.info(f"✓ soft (harmonic-repulsion) potentials ({n_registered} registered"
                    f"{skip_str}): k_QQ={s.k_QtQt}, k_FF={s.k_FtFt}, k_QF={s.k_QtFt}")
    elif potential_type == "weak":
        w = config.weak
        logger.info(f"✓ weak (piecewise-harmonic) potentials ({n_registered} registered"
                    f"{skip_str}): k_QQ={w.k_QtQt}, k_FF={w.k_FtFt}, k_QF={w.k_QtFt}; "
                    f"depth_QQ={w.depth_QtQt}, depth_FF={w.depth_FtFt}, depth_QF={w.depth_QtFt}")
    else:
        logger.info(f"✓ {potential_type} potentials ({n_registered} registered{skip_str}): "
                    f"ε_QQ={lj.epsilon_QtQt}, ε_FF={lj.epsilon_FtFt}, ε_QF={lj.epsilon_QtFt}")


def _add_topologies(
    system: readdy.ReactionDiffusionSystem,
    config: SimulationConfig,
    register_binding: bool = True,
    register_breaking: bool = False,
):
    """Add topology type, bonds, and the requested reactions.

    Harmonic bonds are always configured. Binding (spatial) and breaking (dissociation +
    freed-monomer re-typing) reactions are registered independently so a single function
    serves equilibration (neither), production (binding only) and deagglomeration phases
    (breaking only), per ``create_system``.

    Parameters
    ----------
    system : readdy.ReactionDiffusionSystem
        The system to configure
    config : SimulationConfig
        Simulation configuration
    register_binding : bool
        Register the spatial binding reactions (seed/grow/merge).
    register_breaking : bool
        Register bond breaking: the built-in topology dissociation (rate
        n_edges * topology.koff) plus a cleanup reaction that re-types freed monomers.
    """
    topo = config.topology
    qt = config.qt.name
    ft = config.ft.name
    qtc = config.qt.cluster_name
    ftc = config.ft.cluster_name
    r0 = config.equilibrium_bond_length

    # Register topology type
    system.topologies.add_type(topo.name)

    # Configure harmonic bonds for all possible connections
    bond_pairs = [(qt, ft), (qtc, ftc), (qtc, ft), (qt, ftc)]
    for t1, t2 in bond_pairs:
        system.topologies.configure_harmonic_bond(
            t1, t2, force_constant=float(topo.k_bond), length=float(r0)
        )

    n_binding = 0
    if register_binding:
        # Spatial reactions for binding. seed and grow_QtC_Ft always apply: in both, the
        # particle that ends up as FtC is a *free* Ft gaining its first bond, so Ft stays
        # monovalent. grow_FtC_Qt and merge_QtC_FtC are the only reactions where an
        # already-bonded FtC gains a second bond, so they are skipped when ft_monovalent.
        reactions = [
            # Seeding: Qt + Ft -> QtC--FtC
            (f"seed_{topo.name}",
             f"{topo.name}({qt}) + {topo.name}({ft}) -> {topo.name}({qtc}--{ftc})"),
            # Growth: QtC + Ft -> QtC--FtC
            (f"grow_QtC_Ft_{topo.name}",
             f"{topo.name}({qtc}) + {topo.name}({ft}) -> {topo.name}({qtc}--{ftc})"),
        ]
        if not topo.ft_monovalent:
            # allow_loops lets merge_QtC_FtC fire *within* one cluster (ReaDDy self-fusion),
            # so two already-clustered particles can bond and close a ring. Applied to the
            # merge reaction only: seed/grow always consume a free monomer (its own singleton
            # topology) and so are inherently inter-topology and can never close a loop.
            self_suffix = " [self=true]" if topo.allow_loops else ""
            reactions += [
                # Growth: FtC + Qt -> FtC--QtC
                (f"grow_FtC_Qt_{topo.name}",
                 f"{topo.name}({ftc}) + {topo.name}({qt}) -> {topo.name}({ftc}--{qtc})"),
                # Merging: QtC + FtC -> QtC--FtC  (self-fusion enabled when allow_loops)
                (f"merge_QtC_FtC_{topo.name}",
                 f"{topo.name}({qtc}) + {topo.name}({ftc}) -> {topo.name}({qtc}--{ftc}){self_suffix}"),
            ]

        for name, descriptor in reactions:
            system.topologies.add_spatial_reaction(
                f"{name}: {descriptor}",
                rate=float(topo.kon),
                radius=float(topo.binding_radius)
            )
        n_binding = len(reactions)

    if register_breaking:
        _add_breaking(system, config)

    bits = []
    if register_binding:
        bits.append(f"{n_binding} binding spatial reactions (kon={topo.kon}, "
                    f"binding_radius={topo.binding_radius} nm, ft_monovalent={topo.ft_monovalent}, "
                    f"allow_loops={topo.allow_loops})")
    if register_breaking:
        bits.append(f"dissociation (koff={topo.koff}) + freed-monomer re-typing")
    if not bits:
        bits.append("bonds only, no reactions")
    logger.info(f"✓ Topology '{topo.name}': k_bond={topo.k_bond}; " + "; ".join(bits))


def _add_breaking(system: readdy.ReactionDiffusionSystem, config: SimulationConfig):
    """Register bond-breaking for deagglomeration phases.

    Two structural reactions on the topology type:

    1. **Dissociation**: removes a uniformly-random edge at total rate
       `n_edges * topology.koff`, auto-splitting a cluster into its connected components
       (each becomes its own topology of the same type). This reimplements ReaDDy's
       built-in ``add_topology_dissociation`` because that helper is broken in this ReaDDy
       build (its reaction function returns the internal recipe, which lacks the ``_get``
       hook ``add_structural_reaction`` calls); here the reaction function returns the
       public ``StructuralReactionRecipe`` wrapper, which works.
    2. **Re-type cleanup**: dissociation does not change particle types, so a particle that
       ends up isolated stays typed QtC/FtC. This reaction fires (essentially immediately)
       on any 1-particle topology still typed QtC/FtC and changes it back to the free type
       (QtC->Qt, FtC->Ft). The result is identical to an originally-placed free particle
       (free particles are themselves single-particle topologies), so type-based counts in
       analysis (free vs clustered) and the pair potentials behave correctly.
    """
    topo = config.topology
    topo_name = topo.name
    koff = float(topo.koff)

    # (1) Per-edge dissociation: total rate n_edges * koff; remove one random edge.
    def dissociation_rate_function(topology):
        return koff * float(len(topology.get_graph().get_edges()))

    def dissociation_reaction_function(topology):
        recipe = readdy.StructuralReactionRecipe(topology)
        edges = topology.get_graph().get_edges()
        if edges:
            edge = edges[np.random.randint(0, len(edges))]
            recipe.remove_edge(edge[0], edge[1])
        return recipe

    system.topologies.add_structural_reaction(
        f"dissoc_{topo_name}", topo_name,
        dissociation_reaction_function, dissociation_rate_function,
    )

    # (2) Cleanup: re-type freed (isolated) cluster monomers back to their free species.
    free_of_cluster = {
        config.qt.cluster_name: config.qt.name,
        config.ft.cluster_name: config.ft.name,
    }
    cluster_types = set(free_of_cluster.keys())
    retype_rate = float(_RETYPE_RATE_FACTOR) / float(config.timestep)

    def retype_rate_function(topology):
        # Only isolated (single-particle) topologies whose particle is still a cluster type.
        if topology.n_particles != 1:
            return 0.0
        return retype_rate if topology.particles[0].type in cluster_types else 0.0

    def retype_reaction_function(topology):
        recipe = readdy.StructuralReactionRecipe(topology)
        free_name = free_of_cluster.get(topology.particles[0].type)
        if free_name is not None:
            recipe.change_particle_type(0, free_name)
        return recipe

    system.topologies.add_structural_reaction(
        f"retype_free_{topo_name}", topo_name,
        retype_reaction_function, retype_rate_function,
    )


