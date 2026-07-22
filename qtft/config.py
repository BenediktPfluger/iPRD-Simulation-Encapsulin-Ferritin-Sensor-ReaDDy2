"""qtft.config -- configuration dataclasses and the parameter-naming convention.

Single source of truth for all physical / simulation parameters (JSON-serialisable).
Free of matplotlib and ReaDDy so it can be imported anywhere.
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# Time unit conversion
# NOTE: ReaDDy observables return STEP NUMBERS, not time in ns!
# To get actual time: step_number * timestep (ns) * NS_TO_US (ns->µs)
NS_TO_US = 1e-3  # nanoseconds to microseconds


def _steps_to_us(steps: np.ndarray, timestep: float) -> np.ndarray:
    """
    Convert ReaDDy step numbers to microseconds.
    
    ReaDDy observables return step numbers, not time in ns.
    Actual time = step_number * timestep (ns) * NS_TO_US (ns->µs)
    
    Parameters
    ----------
    steps : array-like
        Step numbers from ReaDDy observables
    timestep : float
        Integration timestep in nanoseconds (must be > 0)
    
    Returns
    -------
    ndarray
        Time in microseconds
    
    Raises
    ------
    ValueError
        If timestep is not positive
    """
    if timestep <= 0:
        raise ValueError(f"timestep must be positive, got {timestep}")
    return np.asarray(steps) * timestep * NS_TO_US


# Adaptive time-unit ladder: (label, nanoseconds per unit). Used to render durations in
# whatever unit keeps the number in [1, 1000) (seconds uncapped), instead of always µs.
# Conversion of steps -> time is unchanged (_steps_to_us); this is display/naming only.
_TIME_UNITS = [("ps", 1e-3), ("ns", 1.0), ("µs", 1e3), ("ms", 1e6), ("s", 1e9)]


def _pick_time_unit(ns: float):
    """Return (label, ns_per_unit) for the largest unit whose ns_per_unit <= |ns|.

    Keeps the displayed number in [1, 1000) for ps..ms; seconds is the top unit and is
    left uncapped (e.g. 2000 s). Zero / sub-ps values fall back to picoseconds.
    """
    v = abs(float(ns))
    chosen = _TIME_UNITS[0]
    for label, nspu in _TIME_UNITS:
        if v >= nspu:
            chosen = (label, nspu)
    return chosen


def format_duration(ns: float, ascii: bool = False) -> str:
    """Format a duration given in nanoseconds with an appropriate SI unit.

    The unit is chosen from the magnitude (ps, ns, µs, ms, s) so the number stays in a
    readable range. The number is formatted compactly (no trailing zeros).

    Parameters
    ----------
    ns : float
        Duration in nanoseconds.
    ascii : bool
        If True, use "us" instead of "µs" and omit the space between number and unit —
        suitable for filenames (e.g. "50ps", "100us", "1ms", "300s"). If False, return a
        human-readable string with a space and the unicode µ (e.g. "100 µs", "5 ms").
    """
    label, nspu = _pick_time_unit(ns)
    value = float(ns) / nspu
    num = f"{value:g}"
    if ascii:
        label = "us" if label == "µs" else label
        return f"{num}{label}"
    return f"{num} {label}"


def choose_time_unit(max_us: float):
    """Pick a display unit for a time axis given in microseconds.

    Returns (factor, label): multiply a µs value/array by ``factor`` to convert it into
    the chosen unit, and use ``label`` (e.g. "µs", "ms", "s") for the axis. The unit is
    selected from ``max_us`` so the axis numbers stay readable.
    """
    label, nspu = _pick_time_unit(abs(float(max_us)) * 1e3)  # µs -> ns for the ladder
    factor = 1e3 / nspu  # µs -> chosen unit
    return factor, label


@dataclass
class ParticleConfig:
    """
    Configuration for a single particle species.
    
    Parameters
    ----------
    name : str
        Species name (e.g., "Qt", "Ft")
    radius : float
        Particle radius in nm
    diffusion : float
        Diffusion coefficient in nm²/ns
    cluster_diffusion : float, optional
        Diffusion coefficient when in a cluster. Defaults to same as diffusion.
    """
    name: str
    radius: float
    diffusion: float
    cluster_diffusion: Optional[float] = None
    
    def __post_init__(self):
        if self.cluster_diffusion is None:
            self.cluster_diffusion = self.diffusion
        self._validate()
    
    def _validate(self):
        if self.radius <= 0:
            raise ValueError(f"Radius must be positive: {self.radius}")
        if self.diffusion <= 0:
            raise ValueError(f"Diffusion must be positive: {self.diffusion}")
        if self.cluster_diffusion <= 0:
            raise ValueError(f"Cluster diffusion must be positive: {self.cluster_diffusion}")
    
    @property
    def cluster_name(self) -> str:
        """Name of the clustered particle type (e.g., 'Qt' -> 'QtC')."""
        return f"{self.name}C"


@dataclass
class TopologyConfig:
    """
    Configuration for topology-based binding.
    
    Parameters
    ----------
    name : str
        Topology type name
    binding_radius : float
        Distance within which binding can occur (nm)
    kon : float
        Binding rate, passed straight to ReaDDy's spatial-reaction ``rate``. It is a
        microscopic per-pair rate in 1/ns (timestep is in ns): any eligible pair within
        binding_radius reacts with probability p = 1 - exp(-kon * timestep) per step. It is
        NOT the macroscopic nm³/(ns·particle) mass-action constant (see README caveat P4).
    k_bond : float
        Harmonic bond force constant (kJ/(mol·nm²))
    ft_monovalent : bool
        If True, Ft can form at most one bond (monovalent leaf). The two reactions
        that would give an already-bonded Ft (FtC) a second bond — grow_FtC_Qt and
        merge_QtC_FtC — are not registered, so clusters become single-Qt stars
        (one Qt hub + N monovalent Ft leaves). Default False = fully multivalent.
    allow_loops : bool
        If True, allow the merge_QtC_FtC reaction to fire *within* a single cluster
        (ReaDDy self-fusion), so two already-clustered particles can bond and close a
        ring — clusters become crosslinked networks instead of acyclic trees. Has effect
        only when ft_monovalent is False (a leaf Ft cannot form the second bond that
        closes a loop). Default False reproduces the acyclic-tree model exactly. When
        enabled, bond-count Methods 1/2 (len(edges)) stay exact, but the n-1 fallback
        and the "tree" assumption no longer hold. Adds a `_loops` filename tag.
    koff : float
        Bond-breaking (dissociation) rate per edge, used only in deagglomeration
        phases (see SimulationConfig.phases). A topology with n_edges bonds breaks an
        edge at total rate n_edges * koff, possibly splitting into sub-clusters. 0
        (default) means no breaking, i.e. pure agglomeration as before.
    """
    name: str = "QtFt_Cluster"
    binding_radius: float = 1.5
    kon: float = 10.0
    k_bond: float = 20.0
    ft_monovalent: bool = False
    koff: float = 0.0
    allow_loops: bool = False

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if self.binding_radius <= 0:
            raise ValueError(f"Binding radius must be positive: {self.binding_radius}")
        if self.kon <= 0:
            raise ValueError(f"Binding rate must be positive: {self.kon}")
        if self.k_bond <= 0:
            raise ValueError(f"Bond constant must be positive: {self.k_bond}")
        if self.koff < 0:
            raise ValueError(f"Bond-breaking rate must be non-negative: {self.koff}")


@dataclass
class PhaseConfig:
    """One phase of an agglomeration/deagglomeration cycle.

    A SimulationConfig with a non-empty ``phases`` list runs each phase in order as a
    separate ReaDDy segment (state carried over via checkpoints), rebuilding the system
    with that phase's reactions and pair potential. The physics of each phase is fully
    explicit here so the config stays the single source of truth.

    Parameters
    ----------
    name : str
        Human label for the phase, e.g. "agglomerate" / "deagglomerate".
    n_steps : int
        Number of integration steps in this phase.
    binding : bool
        Register the spatial binding reactions (seed/grow/merge) for this phase.
    breaking : bool
        Register bond breaking for this phase: the built-in topology dissociation
        (rate n_edges * topology.koff) plus a cleanup reaction that re-types freed
        monomers (QtC->Qt, FtC->Ft). Requires topology.koff > 0 to have any effect.
    potential_type : str
        Pair potential during this phase: "LJ" (attractive) or "WCA" (purely
        repulsive). Deagglomeration typically uses "WCA" so freed particles disperse.
    """
    name: str
    n_steps: int
    binding: bool = True
    breaking: bool = False
    potential_type: str = "LJ"

    def __post_init__(self):
        if self.n_steps <= 0:
            raise ValueError(f"Phase '{self.name}' n_steps must be positive: {self.n_steps}")
        if self.potential_type not in ("WCA", "LJ", "soft", "weak"):
            raise ValueError(
                f"Phase '{self.name}' potential_type must be 'WCA', 'LJ', 'soft', or 'weak', "
                f"got: {self.potential_type}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_steps": self.n_steps,
            "binding": self.binding,
            "breaking": self.breaking,
            "potential_type": self.potential_type,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PhaseConfig":
        return cls(
            name=d.get("name", "phase"),
            n_steps=int(d["n_steps"]),
            binding=bool(d.get("binding", True)),
            breaking=bool(d.get("breaking", False)),
            potential_type=d.get("potential_type", "LJ"),
        )


def make_agg_deagg_phases(
    agg_steps: int,
    deagg_steps: int,
    n_cycles: int = 1,
    agg_potential: str = "LJ",
    deagg_potential: str = "WCA",
) -> List["PhaseConfig"]:
    """Build a standard agglomeration<->deagglomeration phase schedule.

    Each cycle is an agglomeration phase (binding on, breaking off, attractive LJ)
    followed by a deagglomeration phase (binding off, breaking on, repulsive WCA).
    Assign the result to ``SimulationConfig.phases`` and set ``topology.koff > 0``.

    Parameters
    ----------
    agg_steps, deagg_steps : int
        Steps per agglomeration / deagglomeration phase.
    n_cycles : int
        Number of agg->deagg cycles (default 1 => two phases).
    agg_potential, deagg_potential : str
        Pair potential for each phase type ("LJ" or "WCA").
    """
    if n_cycles < 1:
        raise ValueError(f"n_cycles must be >= 1, got {n_cycles}")
    phases: List[PhaseConfig] = []
    for _ in range(n_cycles):
        phases.append(PhaseConfig(
            name="agglomerate", n_steps=agg_steps,
            binding=True, breaking=False, potential_type=agg_potential,
        ))
        phases.append(PhaseConfig(
            name="deagglomerate", n_steps=deagg_steps,
            binding=False, breaking=True, potential_type=deagg_potential,
        ))
    return phases


@dataclass
class LennardJonesConfig:
    """
    Configuration for Lennard-Jones potentials with per-pair epsilon values.
    
    Epsilon values follow a cascade defaulting hierarchy:
    
    1. Free-free pairs (always explicit):
       - epsilon_QtQt, epsilon_FtFt, epsilon_QtFt
    
    2. Cluster pairs (default to their free-particle counterparts):
       - epsilon_QtCQtC  defaults to epsilon_QtQt
       - epsilon_FtCFtC  defaults to epsilon_FtFt
       - epsilon_QtCFtC  defaults to epsilon_QtFt
    
    3. Mixed-state pairs (default to cluster value for that species pair):
       - epsilon_QtQtC   defaults to epsilon_QtCQtC
       - epsilon_FtFtC   defaults to epsilon_FtCFtC
       - epsilon_QtCFt   defaults to epsilon_QtCFtC
       - epsilon_QtFtC   defaults to epsilon_QtCFtC
    
    Setting any epsilon to 0 disables that interaction entirely (the potential
    is not registered in ReaDDy).
    
    Parameters
    ----------
    epsilon_QtQt : float
        LJ well depth for Qt-Qt interaction (kJ/mol)
    epsilon_FtFt : float
        LJ well depth for Ft-Ft interaction (kJ/mol)
    epsilon_QtFt : float
        LJ well depth for Qt-Ft interaction (kJ/mol)
    epsilon_QtCQtC : float, optional
        LJ well depth for QtC-QtC. Defaults to epsilon_QtQt.
    epsilon_FtCFtC : float, optional
        LJ well depth for FtC-FtC. Defaults to epsilon_FtFt.
    epsilon_QtCFtC : float, optional
        LJ well depth for QtC-FtC. Defaults to epsilon_QtFt.
    epsilon_QtQtC : float, optional
        LJ well depth for Qt-QtC. Defaults to epsilon_QtCQtC.
    epsilon_FtFtC : float, optional
        LJ well depth for Ft-FtC. Defaults to epsilon_FtCFtC.
    epsilon_QtCFt : float, optional
        LJ well depth for QtC-Ft. Defaults to epsilon_QtCFtC.
    epsilon_QtFtC : float, optional
        LJ well depth for Qt-FtC. Defaults to epsilon_QtCFtC.
    cutoff_factor : float, optional
        Optional override for the cutoff as a multiple of sigma. If None (default), the
        cutoff is derived from the active mode in _add_potentials: WCA -> 2^(1/6)*sigma
        (purely repulsive), LJ -> 2.5*sigma (attractive well). The mode is selected by
        ``SimulationConfig.potential_type`` (not here); soft/weak ignore this field.
    """
    # Primary free-free epsilon values
    epsilon_QtQt: float = 10.0
    epsilon_FtFt: float = 10.0
    epsilon_QtFt: float = 10.0
    
    # Cluster epsilon values (cascade from free-free)
    epsilon_QtCQtC: Optional[float] = None
    epsilon_FtCFtC: Optional[float] = None
    epsilon_QtCFtC: Optional[float] = None
    
    # Mixed-state epsilon values (cascade from cluster)
    epsilon_QtQtC: Optional[float] = None
    epsilon_FtFtC: Optional[float] = None
    epsilon_QtCFt: Optional[float] = None
    epsilon_QtFtC: Optional[float] = None
    
    # Cutoff distance as a multiple of sigma. Optional OVERRIDE only: when None (default)
    # the WCA/LJ cutoff is chosen from SimulationConfig.potential_type in _add_potentials
    # (WCA -> 2^(1/6)*sigma, LJ -> 2.5*sigma). The mode selector lives on SimulationConfig,
    # not here (soft/weak ignore this).
    cutoff_factor: Optional[float] = None

    # WCA cutoff factor: 2^(1/6) ≈ 1.122462
    WCA_CUTOFF_FACTOR: float = field(default=1.122462, repr=False)
    LJ_CUTOFF_FACTOR: float = field(default=2.5, repr=False)
    
    def __post_init__(self):
        # Cascade defaults: cluster values from free-free values
        if self.epsilon_QtCQtC is None:
            self.epsilon_QtCQtC = self.epsilon_QtQt
        if self.epsilon_FtCFtC is None:
            self.epsilon_FtCFtC = self.epsilon_FtFt
        if self.epsilon_QtCFtC is None:
            self.epsilon_QtCFtC = self.epsilon_QtFt
        
        # Cascade defaults: mixed-state values from cluster values
        if self.epsilon_QtQtC is None:
            self.epsilon_QtQtC = self.epsilon_QtCQtC
        if self.epsilon_FtFtC is None:
            self.epsilon_FtFtC = self.epsilon_FtCFtC
        if self.epsilon_QtCFt is None:
            self.epsilon_QtCFt = self.epsilon_QtCFtC
        if self.epsilon_QtFtC is None:
            self.epsilon_QtFtC = self.epsilon_QtCFtC

        # cutoff_factor is left as-is: None means "derive from the mode in _add_potentials";
        # a number is an explicit override honored for WCA/LJ.
        self._validate()
    
    def _validate(self):
        # All epsilon values must be non-negative (0 = disabled)
        eps_pairs = {
            "epsilon_QtQt": self.epsilon_QtQt,
            "epsilon_FtFt": self.epsilon_FtFt,
            "epsilon_QtFt": self.epsilon_QtFt,
            "epsilon_QtCQtC": self.epsilon_QtCQtC,
            "epsilon_FtCFtC": self.epsilon_FtCFtC,
            "epsilon_QtCFtC": self.epsilon_QtCFtC,
            "epsilon_QtQtC": self.epsilon_QtQtC,
            "epsilon_FtFtC": self.epsilon_FtFtC,
            "epsilon_QtCFt": self.epsilon_QtCFt,
            "epsilon_QtFtC": self.epsilon_QtFtC,
        }
        for name, val in eps_pairs.items():
            if val < 0:
                raise ValueError(f"{name} must be non-negative, got {val}")

        if self.cutoff_factor is not None and self.cutoff_factor <= 0:
            raise ValueError(f"Cutoff factor must be positive: {self.cutoff_factor}")
    
    def get_epsilon(self, type1: str, type2: str) -> float:
        """
        Get the epsilon value for a specific pair of particle types.
        
        Parameters
        ----------
        type1, type2 : str
            Particle type names (e.g., "Qt", "Ft", "QtC", "FtC")
        
        Returns
        -------
        float
            Epsilon value for this pair
        """
        # Normalize pair order (alphabetical)
        pair = tuple(sorted([type1, type2]))
        
        pair_map = {
            ("Ft", "Ft"): self.epsilon_FtFt,
            ("Ft", "FtC"): self.epsilon_FtFtC,
            ("Ft", "Qt"): self.epsilon_QtFt,
            ("Ft", "QtC"): self.epsilon_QtCFt,
            ("FtC", "FtC"): self.epsilon_FtCFtC,
            ("FtC", "Qt"): self.epsilon_QtFtC,
            ("FtC", "QtC"): self.epsilon_QtCFtC,
            ("Qt", "Qt"): self.epsilon_QtQt,
            ("Qt", "QtC"): self.epsilon_QtQtC,
            ("QtC", "QtC"): self.epsilon_QtCQtC,
        }
        
        if pair not in pair_map:
            raise ValueError(f"Unknown particle pair: {type1}-{type2}")

        return pair_map[pair]


@dataclass
class SoftPotentialConfig:
    """
    Configuration for the "soft" excluded-volume mode (per-pair harmonic repulsion).

    Selected via ``SimulationConfig.potential_type == "soft"`` (or per-phase / equilibration).
    Instead of a stiff 12-6 Lennard-Jones wall, each enabled type pair gets a
    ReaDDy ``add_harmonic_repulsion`` potential: force is linear and bounded, and
    vanishes at the contact distance (r_i + r_j). Because overlaps produce small
    finite forces rather than an r^-12 blow-up, a much larger integration timestep
    is numerically stable. There is no attractive term — clustering comes purely
    from the topology binding reactions + harmonic bonds, which are unchanged.

    Soft mode is fully self-contained: when active, ``_add_potentials`` reads ONLY
    these force constants (``LennardJonesConfig.epsilon_*`` is ignored). Each pair
    has its own constant ``k_*`` (kJ/(mol*nm^2)); the thermal overlap scale is
    ``sqrt(2*kB*T / k)``, so stiffen the small-particle pairs (Ft-Ft, Qt-Ft) to
    stop them interpenetrating. Setting any ``k`` to 0 disables that pair entirely
    (the potential is not registered) — the soft-mode analog of ``epsilon == 0``.

    Force constants follow the same cascade hierarchy as ``LennardJonesConfig``:

    1. Free-free pairs (always explicit): k_QtQt, k_FtFt, k_QtFt
    2. Cluster pairs (default to their free-particle counterparts):
       k_QtCQtC<-k_QtQt, k_FtCFtC<-k_FtFt, k_QtCFtC<-k_QtFt
    3. Mixed-state pairs (default to the cluster value for that species pair):
       k_QtQtC<-k_QtCQtC, k_FtFtC<-k_FtCFtC, k_QtCFt<-k_QtCFtC, k_QtFtC<-k_QtCFtC

    Keep the constants soft enough that they do not reintroduce the stiffness that
    forces a tiny timestep; the stable upper bound is measured by
    scripts/calibrate_timestep.py.
    """
    # Primary free-free force constants (kJ/(mol*nm^2)); 0 disables that pair.
    k_QtQt: float = 10.0
    k_FtFt: float = 10.0
    k_QtFt: float = 10.0

    # Cluster force constants (cascade from free-free)
    k_QtCQtC: Optional[float] = None
    k_FtCFtC: Optional[float] = None
    k_QtCFtC: Optional[float] = None

    # Mixed-state force constants (cascade from cluster)
    k_QtQtC: Optional[float] = None
    k_FtFtC: Optional[float] = None
    k_QtCFt: Optional[float] = None
    k_QtFtC: Optional[float] = None

    def __post_init__(self):
        # Cascade defaults: cluster values from free-free values
        if self.k_QtCQtC is None:
            self.k_QtCQtC = self.k_QtQt
        if self.k_FtCFtC is None:
            self.k_FtCFtC = self.k_FtFt
        if self.k_QtCFtC is None:
            self.k_QtCFtC = self.k_QtFt

        # Cascade defaults: mixed-state values from cluster values
        if self.k_QtQtC is None:
            self.k_QtQtC = self.k_QtCQtC
        if self.k_FtFtC is None:
            self.k_FtFtC = self.k_FtCFtC
        if self.k_QtCFt is None:
            self.k_QtCFt = self.k_QtCFtC
        if self.k_QtFtC is None:
            self.k_QtFtC = self.k_QtCFtC

        self._validate()

    def _validate(self):
        # All force constants must be non-negative (0 = disabled)
        k_pairs = {
            "k_QtQt": self.k_QtQt,
            "k_FtFt": self.k_FtFt,
            "k_QtFt": self.k_QtFt,
            "k_QtCQtC": self.k_QtCQtC,
            "k_FtCFtC": self.k_FtCFtC,
            "k_QtCFtC": self.k_QtCFtC,
            "k_QtQtC": self.k_QtQtC,
            "k_FtFtC": self.k_FtFtC,
            "k_QtCFt": self.k_QtCFt,
            "k_QtFtC": self.k_QtFtC,
        }
        for name, val in k_pairs.items():
            if val < 0:
                raise ValueError(f"{name} must be non-negative, got {val}")

    def get_force_constant(self, type1: str, type2: str) -> float:
        """
        Get the harmonic-repulsion force constant for a pair of particle types.

        Parameters
        ----------
        type1, type2 : str
            Particle type names (e.g., "Qt", "Ft", "QtC", "FtC")

        Returns
        -------
        float
            Force constant (kJ/(mol*nm^2)) for this pair (0 = disabled).
        """
        # Normalize pair order (alphabetical)
        pair = tuple(sorted([type1, type2]))

        pair_map = {
            ("Ft", "Ft"): self.k_FtFt,
            ("Ft", "FtC"): self.k_FtFtC,
            ("Ft", "Qt"): self.k_QtFt,
            ("Ft", "QtC"): self.k_QtCFt,
            ("FtC", "FtC"): self.k_FtCFtC,
            ("FtC", "Qt"): self.k_QtFtC,
            ("FtC", "QtC"): self.k_QtCFtC,
            ("Qt", "Qt"): self.k_QtQt,
            ("Qt", "QtC"): self.k_QtQtC,
            ("QtC", "QtC"): self.k_QtCQtC,
        }

        if pair not in pair_map:
            raise ValueError(f"Unknown particle pair: {type1}-{type2}")

        return pair_map[pair]


@dataclass
class WeakInteractionConfig:
    """
    Configuration for the "weak" mode (per-pair piecewise-harmonic weak interaction).

    Selected via ``SimulationConfig.potential_type == "weak"`` (or per-phase / equilibration). Uses
    ReaDDy ``add_weak_interaction_piecewise_harmonic``: a *soft attractive* pair potential
    with a harmonic repulsive branch, a well of depth ``depth`` whose minimum sits at the
    contact distance (r_i + r_j), and a smooth return to zero at ``cutoff``. It provides
    LJ-like attraction WITHOUT the r^-12 wall, so it can run at a much larger timestep than
    12-6 Lennard-Jones.

    Weak mode is self-contained: when active, ``_add_potentials`` reads ONLY this config
    (``LennardJonesConfig.epsilon_*`` and ``SoftPotentialConfig`` are ignored). Each pair
    has its own force constant ``k_*`` (kJ/(mol*nm^2), the branch stiffness) and well depth
    ``depth_*`` (kJ/mol, the attraction strength — the analog of LJ epsilon). ReaDDy's
    ``desired_distance`` is derived from contact (r_i + r_j) and ``cutoff`` from a single
    global ``cutoff_factor`` (cutoff = cutoff_factor * contact). Setting a pair's ``k`` to 0
    disables that pair entirely (the potential is not registered).

    Both ``k_*`` and ``depth_*`` follow the same cascade hierarchy as ``LennardJonesConfig``:
    free-free explicit -> cluster (defaults to free-free) -> mixed (defaults to cluster).
    """
    # Primary free-free force constants (kJ/(mol*nm^2)); 0 disables that pair.
    k_QtQt: float = 10.0
    k_FtFt: float = 10.0
    k_QtFt: float = 10.0
    # Cluster force constants (cascade from free-free)
    k_QtCQtC: Optional[float] = None
    k_FtCFtC: Optional[float] = None
    k_QtCFtC: Optional[float] = None
    # Mixed-state force constants (cascade from cluster)
    k_QtQtC: Optional[float] = None
    k_FtFtC: Optional[float] = None
    k_QtCFt: Optional[float] = None
    k_QtFtC: Optional[float] = None

    # Primary free-free well depths (kJ/mol; attraction strength). 0 = no attraction.
    depth_QtQt: float = 1.0
    depth_FtFt: float = 1.0
    depth_QtFt: float = 1.0
    # Cluster depths (cascade from free-free)
    depth_QtCQtC: Optional[float] = None
    depth_FtCFtC: Optional[float] = None
    depth_QtCFtC: Optional[float] = None
    # Mixed-state depths (cascade from cluster)
    depth_QtQtC: Optional[float] = None
    depth_FtFtC: Optional[float] = None
    depth_QtCFt: Optional[float] = None
    depth_QtFtC: Optional[float] = None

    # Global cutoff as a multiple of the contact distance (must be > 1 for a well to exist).
    cutoff_factor: float = 2.0

    # (particle-type suffixes in cascade order, shared by both k_* and depth_*)
    _CLUSTER_FROM_FREE = (("QtCQtC", "QtQt"), ("FtCFtC", "FtFt"), ("QtCFtC", "QtFt"))
    _MIXED_FROM_CLUSTER = (("QtQtC", "QtCQtC"), ("FtFtC", "FtCFtC"),
                           ("QtCFt", "QtCFtC"), ("QtFtC", "QtCFtC"))

    def __post_init__(self):
        # Cascade both families (cluster<-free, then mixed<-cluster) exactly like LJ epsilon.
        for prefix in ("k_", "depth_"):
            for dst, src in self._CLUSTER_FROM_FREE:
                if getattr(self, prefix + dst) is None:
                    setattr(self, prefix + dst, getattr(self, prefix + src))
            for dst, src in self._MIXED_FROM_CLUSTER:
                if getattr(self, prefix + dst) is None:
                    setattr(self, prefix + dst, getattr(self, prefix + src))
        self._validate()

    def _validate(self):
        suffixes = ["QtQt", "FtFt", "QtFt", "QtCQtC", "FtCFtC", "QtCFtC",
                    "QtQtC", "FtFtC", "QtCFt", "QtFtC"]
        for s in suffixes:
            k = getattr(self, "k_" + s)
            d = getattr(self, "depth_" + s)
            if k < 0:
                raise ValueError(f"k_{s} must be non-negative, got {k}")
            if d < 0:
                raise ValueError(f"depth_{s} must be non-negative, got {d}")
        if self.cutoff_factor <= 0:
            raise ValueError(f"cutoff_factor must be positive, got {self.cutoff_factor}")

    @staticmethod
    def _pair_suffix(type1: str, type2: str) -> str:
        pair = tuple(sorted([type1, type2]))
        mapping = {
            ("Ft", "Ft"): "FtFt", ("Ft", "FtC"): "FtFtC", ("Ft", "Qt"): "QtFt",
            ("Ft", "QtC"): "QtCFt", ("FtC", "FtC"): "FtCFtC", ("FtC", "Qt"): "QtFtC",
            ("FtC", "QtC"): "QtCFtC", ("Qt", "Qt"): "QtQt", ("Qt", "QtC"): "QtQtC",
            ("QtC", "QtC"): "QtCQtC",
        }
        if pair not in mapping:
            raise ValueError(f"Unknown particle pair: {type1}-{type2}")
        return mapping[pair]

    def get_force_constant(self, type1: str, type2: str) -> float:
        """Force constant (kJ/(mol*nm^2)) for a pair (0 = disabled)."""
        return getattr(self, "k_" + self._pair_suffix(type1, type2))

    def get_depth(self, type1: str, type2: str) -> float:
        """Well depth (kJ/mol) for a pair (0 = no attraction)."""
        return getattr(self, "depth_" + self._pair_suffix(type1, type2))


@dataclass
class SimulationConfig:
    """
    Complete simulation configuration.
    
    This is the main configuration object that combines all settings.
    Can be created directly or via SimulationConfig.from_dict().
    
    Parameters
    ----------
    qt : ParticleConfig
        Qt particle configuration
    ft : ParticleConfig
        Ft particle configuration
    topology : TopologyConfig
        Topology/binding configuration
    lj : LennardJonesConfig
        Lennard-Jones potential configuration
    box_size : tuple of float
        Simulation box dimensions (Lx, Ly, Lz) in nm
    periodic_boundary : bool
        Use periodic boundary conditions
    temperature : float
        Temperature in Kelvin
    timestep : float
        Integration timestep in ns
    n_steps : int
        Total number of simulation steps
    record_stride : int
        Save trajectory every N steps
    observable_stride : int
        Record observables every N steps
    n_qt : int
        Number of Qt particles
    n_ft : int
        Number of Ft particles
    kernel : str
        ReaDDy kernel ("CPU" or "SingleCPU")
    n_threads : int
        Number of threads for CPU kernel
    rng_seed : int
        Random number generator seed
    output_file : str
        Output trajectory filename (.h5)
    """
    # Particle configurations
    qt: ParticleConfig = field(default_factory=lambda: ParticleConfig("Qt", 1.0, 5.0))
    ft: ParticleConfig = field(default_factory=lambda: ParticleConfig("Ft", 0.25, 15.0))
    
    # Topology configuration
    topology: TopologyConfig = field(default_factory=TopologyConfig)

    # Production potential selector (single source of truth). Picks which parameter block
    # below is registered: "WCA"/"LJ" -> lj.epsilon_*, "soft" -> soft.k_*,
    # "weak" -> weak.k_*/depth_*; the others are ignored.
    potential_type: str = "WCA"

    # Lennard-Jones configuration (epsilons used only when potential_type is "WCA"/"LJ")
    lj: LennardJonesConfig = field(default_factory=LennardJonesConfig)

    # Soft (harmonic-repulsion) configuration, used when potential_type == "soft".
    # Ignored by the WCA/LJ paths, so existing configs are unaffected.
    soft: SoftPotentialConfig = field(default_factory=SoftPotentialConfig)

    # Weak-interaction (piecewise-harmonic) configuration, used when potential_type == "weak".
    # Ignored by the WCA/LJ/soft paths, so existing configs are unaffected.
    weak: WeakInteractionConfig = field(default_factory=WeakInteractionConfig)

    # Simulation box
    box_size: Tuple[float, float, float] = (50.0, 50.0, 50.0)
    periodic_boundary: bool = True
    
    # Physical parameters
    temperature: float = 300.0

    # Potential used during equilibration (reactions are always off then). Defaults to
    # "WCA" (purely repulsive) so equilibration relaxes overlaps without attraction,
    # regardless of the production potential (config.potential_type). Set to "LJ"/"soft"/
    # "weak" to equilibrate under a different potential instead.
    equilibration_potential: str = "WCA"

    # Integration parameters
    timestep: float = 1e-4
    n_steps: int = 200000
    
    # Recording parameters
    record_stride: int = 10
    observable_stride: int = 10
    particles_observable_stride: Optional[int] = None  # None = disabled, saves disk space
    # Cadence for heavy, currently-unread observables (forces, virial). These are recorded
    # far less often than the rest to keep trajectory files small. None = 100 x observable_stride.
    heavy_observable_stride: Optional[int] = None
    
    # Particle counts
    n_qt: int = 200
    n_ft: int = 200
    
    # Kernel settings
    kernel: str = "CPU"
    n_threads: int = 4
    
    # Random seed
    rng_seed: int = 42
    
    # Output
    output_file: Optional[str] = None  # None = auto-generate from parameters

    # Optional agglomeration<->deagglomeration phase schedule. None/empty => a single
    # ordinary run using n_steps and lj.potential_type (unchanged legacy behavior).
    # When set, the run executes each phase in order (see engine.run_phased), and
    # n_steps is ignored in favor of the per-phase step counts.
    phases: Optional[List[PhaseConfig]] = None

    def __post_init__(self):
        # Auto-generate output filename if not specified
        if self.output_file is None:
            self.output_file = self._generate_output_filename()
        
        warnings_list = self._validate()
        if warnings_list:
            for w in warnings_list:
                warnings.warn(w)
    
    def _generate_output_filename(self) -> str:
        """
        Generate descriptive output filename from simulation parameters.

        Uses the shared `format_param_string()` convention (same string used for
        ensemble folder names) plus a `.h5` suffix:

            {n_qt}Qt_{n_ft}Ft_{potential_type}_eQQ{e}_eFF{e}_eQF{e}_kon{kon}_dt{dt}ps_{total_time}us.h5

        Examples:
            200Qt_200Ft_WCA_eQQ10_eFF10_eQF10_kon10_dt0.10ps_20us.h5
            200Qt_400Ft_LJ_eQQ10_eFF10_eQF5_kon5.5_dt10ps_30us.h5
        """
        return f"{format_param_string(self)}.h5"
    
    def _validate(self) -> List[str]:
        """Validate configuration and return list of warnings."""
        warnings_list = []
        
        # Check box dimensions
        if any(d <= 0 for d in self.box_size):
            raise ValueError(f"Box dimensions must be positive: {self.box_size}")
        
        # Check temperature
        if self.temperature <= 0:
            raise ValueError(f"Temperature must be positive: {self.temperature}")
        
        # Check timestep
        if self.timestep <= 0:
            raise ValueError(f"Timestep must be positive: {self.timestep}")

        # Check production potential selector
        if self.potential_type not in ("WCA", "LJ", "soft", "weak"):
            raise ValueError(
                f"potential_type must be 'WCA', 'LJ', 'soft', or 'weak', got: {self.potential_type}"
            )

        # Check equilibration potential
        if self.equilibration_potential not in ("WCA", "LJ", "soft", "weak"):
            raise ValueError(
                f"equilibration_potential must be 'WCA', 'LJ', 'soft', or 'weak', "
                f"got: {self.equilibration_potential}"
            )
        
        # Check counts
        if self.n_qt < 0 or self.n_ft < 0:
            raise ValueError(f"Particle counts must be non-negative: n_qt={self.n_qt}, n_ft={self.n_ft}")

        # Phase schedule checks
        if self.phases:
            any_breaking = any(p.breaking for p in self.phases)
            if any_breaking and self.topology.koff <= 0:
                warnings_list.append(
                    "Warning: a deagglomeration phase has breaking=True but topology.koff=0, "
                    "so no bonds will break. Set topology.koff > 0."
                )
            if not any(p.binding for p in self.phases):
                warnings_list.append(
                    "Warning: no phase has binding=True, so no clusters will ever form."
                )
        
        # Physics warnings
        r0_bond = self.qt.radius + self.ft.radius
        if self.topology.binding_radius < r0_bond:
            warnings_list.append(
                f"Warning: Binding radius ({self.topology.binding_radius} nm) < sum of radii "
                f"({r0_bond} nm). Particles may overlap before binding."
            )
        
        # Packing fraction warning
        box_volume = self.box_size[0] * self.box_size[1] * self.box_size[2]
        qt_volume = self.n_qt * (4/3) * np.pi * self.qt.radius**3
        ft_volume = self.n_ft * (4/3) * np.pi * self.ft.radius**3
        packing = (qt_volume + ft_volume) / box_volume
        
        if packing > 0.3:
            warnings_list.append(
                f"Warning: High packing fraction ({packing:.1%}). May cause initialization issues."
            )
        
        # Stride consistency
        if self.record_stride != self.observable_stride:
            warnings_list.append(
                f"Note: record_stride ({self.record_stride}) != observable_stride "
                f"({self.observable_stride}). This may complicate analysis."
            )
        
        return warnings_list
    
    @property
    def equilibrium_bond_length(self) -> float:
        """Equilibrium bond length for Qt-Ft bonds (nm)."""
        return self.qt.radius + self.ft.radius
    
    @property
    def effective_n_steps(self) -> int:
        """Total integration steps actually run.

        Sum of the phase step counts when a phase schedule is set, otherwise n_steps.
        """
        if self.phases:
            return int(sum(p.n_steps for p in self.phases))
        return int(self.n_steps)

    @property
    def total_simulation_time(self) -> float:
        """Total simulation time in ns (across all phases when a schedule is set)."""
        return self.effective_n_steps * self.timestep

    @property
    def phase_base_dir(self) -> Optional[str]:
        """Directory under which per-phase outputs live (None for non-phased runs).

        Derived from ``output_file`` so all consumers (engine, CLI, analysis, ensemble)
        agree without passing paths around:

        - ensemble replica ``.../replica_000/trajectory.h5`` -> ``.../replica_000``
          (phase dirs become siblings: ``.../replica_000/phase_000`` ...).
        - single run ``myrun.h5`` (no directory) -> ``myrun`` (a dedicated run folder).
        """
        if not self.phases:
            return None
        d = os.path.dirname(self.output_file)
        if d == "":
            return os.path.splitext(os.path.basename(self.output_file))[0]
        return d

    @property
    def phase_dirs(self) -> List[str]:
        """Ordered per-phase output directories (empty for non-phased runs)."""
        if not self.phases:
            return []
        base = self.phase_base_dir
        return [os.path.join(base, f"phase_{i:03d}") for i in range(len(self.phases))]

    @property
    def phase_output_files(self) -> List[str]:
        """Ordered per-phase trajectory.h5 paths (empty for non-phased runs)."""
        return [os.path.join(d, "trajectory.h5") for d in self.phase_dirs]

    @property
    def phase_step_offsets(self) -> List[int]:
        """Cumulative step count at the START of each phase (empty for non-phased runs).

        Used to stitch per-phase trajectories (whose step indices restart at 0) onto one
        continuous step/time axis: global_step = phase_local_step + phase_step_offsets[i].
        """
        if not self.phases:
            return []
        offsets, cum = [], 0
        for p in self.phases:
            offsets.append(cum)
            cum += int(p.n_steps)
        return offsets
    
    @property
    def total_simulation_time_us(self) -> float:
        """Total simulation time in µs."""
        return self.total_simulation_time / 1000.0

    @property
    def effective_heavy_observable_stride(self) -> int:
        """Resolved stride for heavy/unread observables (forces, virial).

        Defaults to 100 x observable_stride when heavy_observable_stride is None.
        """
        if self.heavy_observable_stride is not None:
            return int(self.heavy_observable_stride)
        return 100 * int(self.observable_stride)
    
    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "SimulationConfig":
        """
        Create SimulationConfig from a dictionary.
        
        Supports both nested format (from to_dict/save_json) and flat format
        (for backward compatibility and quick experimentation).
        
        Parameters
        ----------
        params : dict
            Dictionary with configuration parameters. Can be:
            
            Nested format (preferred, from to_dict()):
                {"qt": {"name": "Qt", "radius": 1.0, ...}, "ft": {...}, ...}
            
            Flat format (backward compatible):
                {"qt_name": "Qt", "qt_radius": 1.0, ...}
        
        Returns
        -------
        SimulationConfig
            Configured instance
        """
        # Nested format has 'qt' as a dict; flat format uses prefixed scalar keys.
        # One reconstruction path serves both: `g` reads a sub-config value from the
        # nested sub-dict when nested, else from the flat key.
        is_nested = isinstance(params.get("qt"), dict)
        qtp = params.get("qt", {})
        ftp = params.get("ft", {})
        topop = params.get("topology", {})
        ljp = params.get("lj", {})
        softp = params.get("soft", {})
        weakp = params.get("weak", {})

        def g(sub, key, default, flat_key):
            return sub.get(key, default) if is_nested else params.get(flat_key, default)

        qt = ParticleConfig(
            name=g(qtp, "name", "Qt", "qt_name"),
            radius=g(qtp, "radius", 1.0, "qt_radius"),
            diffusion=g(qtp, "diffusion", 5.0, "qt_diffusion"),
            cluster_diffusion=g(qtp, "cluster_diffusion", None, "qt_cluster_diffusion"),
        )
        ft = ParticleConfig(
            name=g(ftp, "name", "Ft", "ft_name"),
            radius=g(ftp, "radius", 0.25, "ft_radius"),
            diffusion=g(ftp, "diffusion", 15.0, "ft_diffusion"),
            cluster_diffusion=g(ftp, "cluster_diffusion", None, "ft_cluster_diffusion"),
        )
        topology = TopologyConfig(
            name=g(topop, "name", "QtFt_Cluster", "topology_name"),
            binding_radius=g(topop, "binding_radius", 1.5, "binding_radius"),
            kon=g(topop, "kon", 10.0, "kon"),
            k_bond=g(topop, "k_bond", 20.0, "k_bond"),
            ft_monovalent=g(topop, "ft_monovalent", False, "ft_monovalent"),
            koff=g(topop, "koff", 0.0, "koff"),
            allow_loops=g(topop, "allow_loops", False, "allow_loops"),
        )
        lj = LennardJonesConfig(
            epsilon_QtQt=g(ljp, "epsilon_QtQt", 10.0, "epsilon_QtQt"),
            epsilon_FtFt=g(ljp, "epsilon_FtFt", 10.0, "epsilon_FtFt"),
            epsilon_QtFt=g(ljp, "epsilon_QtFt", 10.0, "epsilon_QtFt"),
            epsilon_QtCQtC=g(ljp, "epsilon_QtCQtC", None, "epsilon_QtCQtC"),
            epsilon_FtCFtC=g(ljp, "epsilon_FtCFtC", None, "epsilon_FtCFtC"),
            epsilon_QtCFtC=g(ljp, "epsilon_QtCFtC", None, "epsilon_QtCFtC"),
            epsilon_QtQtC=g(ljp, "epsilon_QtQtC", None, "epsilon_QtQtC"),
            epsilon_FtFtC=g(ljp, "epsilon_FtFtC", None, "epsilon_FtFtC"),
            epsilon_QtCFt=g(ljp, "epsilon_QtCFt", None, "epsilon_QtCFt"),
            epsilon_QtFtC=g(ljp, "epsilon_QtFtC", None, "epsilon_QtFtC"),
            cutoff_factor=g(ljp, "cutoff_factor", None, "cutoff_factor"),
        )
        # Production potential selector: prefer the top-level key; migrate from the old
        # location (lj.potential_type) for configs written before it moved to SimulationConfig.
        potential_type = params.get("potential_type")
        if potential_type is None:
            potential_type = ljp.get("potential_type", "WCA")
        # Backward-compat: soft configs written before per-pair support stored a single
        # `repulsion_force_constant`. Map it onto the three free-free constants on load so
        # old soft config JSONs still round-trip faithfully (cluster/mixed then cascade).
        legacy_k = softp.get("repulsion_force_constant") if is_nested \
            else params.get("repulsion_force_constant")
        free_default = 10.0 if legacy_k is None else legacy_k
        soft = SoftPotentialConfig(
            k_QtQt=g(softp, "k_QtQt", free_default, "k_QtQt"),
            k_FtFt=g(softp, "k_FtFt", free_default, "k_FtFt"),
            k_QtFt=g(softp, "k_QtFt", free_default, "k_QtFt"),
            k_QtCQtC=g(softp, "k_QtCQtC", None, "k_QtCQtC"),
            k_FtCFtC=g(softp, "k_FtCFtC", None, "k_FtCFtC"),
            k_QtCFtC=g(softp, "k_QtCFtC", None, "k_QtCFtC"),
            k_QtQtC=g(softp, "k_QtQtC", None, "k_QtQtC"),
            k_FtFtC=g(softp, "k_FtFtC", None, "k_FtFtC"),
            k_QtCFt=g(softp, "k_QtCFt", None, "k_QtCFt"),
            k_QtFtC=g(softp, "k_QtFtC", None, "k_QtFtC"),
        )
        weak = WeakInteractionConfig(
            k_QtQt=g(weakp, "k_QtQt", 10.0, "weak_k_QtQt"),
            k_FtFt=g(weakp, "k_FtFt", 10.0, "weak_k_FtFt"),
            k_QtFt=g(weakp, "k_QtFt", 10.0, "weak_k_QtFt"),
            k_QtCQtC=g(weakp, "k_QtCQtC", None, "weak_k_QtCQtC"),
            k_FtCFtC=g(weakp, "k_FtCFtC", None, "weak_k_FtCFtC"),
            k_QtCFtC=g(weakp, "k_QtCFtC", None, "weak_k_QtCFtC"),
            k_QtQtC=g(weakp, "k_QtQtC", None, "weak_k_QtQtC"),
            k_FtFtC=g(weakp, "k_FtFtC", None, "weak_k_FtFtC"),
            k_QtCFt=g(weakp, "k_QtCFt", None, "weak_k_QtCFt"),
            k_QtFtC=g(weakp, "k_QtFtC", None, "weak_k_QtFtC"),
            depth_QtQt=g(weakp, "depth_QtQt", 1.0, "weak_depth_QtQt"),
            depth_FtFt=g(weakp, "depth_FtFt", 1.0, "weak_depth_FtFt"),
            depth_QtFt=g(weakp, "depth_QtFt", 1.0, "weak_depth_QtFt"),
            depth_QtCQtC=g(weakp, "depth_QtCQtC", None, "weak_depth_QtCQtC"),
            depth_FtCFtC=g(weakp, "depth_FtCFtC", None, "weak_depth_FtCFtC"),
            depth_QtCFtC=g(weakp, "depth_QtCFtC", None, "weak_depth_QtCFtC"),
            depth_QtQtC=g(weakp, "depth_QtQtC", None, "weak_depth_QtQtC"),
            depth_FtFtC=g(weakp, "depth_FtFtC", None, "weak_depth_FtFtC"),
            depth_QtCFt=g(weakp, "depth_QtCFt", None, "weak_depth_QtCFt"),
            depth_QtFtC=g(weakp, "depth_QtFtC", None, "weak_depth_QtFtC"),
            cutoff_factor=g(weakp, "cutoff_factor", 2.0, "weak_cutoff_factor"),
        )

        # Handle box_size - convert list to tuple if needed
        box_size = params.get("box_size", (50.0, 50.0, 50.0))
        if isinstance(box_size, list):
            box_size = tuple(box_size)

        # Reconstruct phase schedule if present (list of phase dicts, or None)
        phases_raw = params.get("phases", None)
        phases = [PhaseConfig.from_dict(p) for p in phases_raw] if phases_raw else None

        # Build SimulationConfig
        return cls(
            qt=qt,
            ft=ft,
            topology=topology,
            potential_type=potential_type,
            lj=lj,
            soft=soft,
            weak=weak,
            box_size=box_size,
            periodic_boundary=params.get("periodic_boundary", True),
            temperature=params.get("temperature", 300.0),
            equilibration_potential=params.get("equilibration_potential", "WCA"),
            timestep=params.get("timestep", 1e-4),
            n_steps=params.get("n_steps", 200000),
            record_stride=params.get("record_stride", 10),
            observable_stride=params.get("observable_stride", 10),
            particles_observable_stride=params.get("particles_observable_stride", None),
            heavy_observable_stride=params.get("heavy_observable_stride", None),
            n_qt=params.get("n_qt", 200),
            n_ft=params.get("n_ft", 200),
            kernel=params.get("kernel", "CPU"),
            n_threads=params.get("n_threads", 4),
            rng_seed=params.get("rng_seed", 42),
            output_file=params.get("output_file", None),
            phases=phases,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a nested dictionary suitable for JSON serialization.
        
        This format can be loaded back with from_dict() or load_json().
        """
        return {
            # Nested particle configs
            "qt": {
                "name": self.qt.name,
                "radius": self.qt.radius,
                "diffusion": self.qt.diffusion,
                "cluster_diffusion": self.qt.cluster_diffusion,
            },
            "ft": {
                "name": self.ft.name,
                "radius": self.ft.radius,
                "diffusion": self.ft.diffusion,
                "cluster_diffusion": self.ft.cluster_diffusion,
            },
            # Nested topology config
            "topology": {
                "name": self.topology.name,
                "binding_radius": self.topology.binding_radius,
                "kon": self.topology.kon,
                "k_bond": self.topology.k_bond,
                "ft_monovalent": self.topology.ft_monovalent,
                "koff": self.topology.koff,
                "allow_loops": self.topology.allow_loops,
            },
            # Production potential selector (single source of truth)
            "potential_type": self.potential_type,
            # Nested LJ config (resolved values, not None)
            "lj": {
                "epsilon_QtQt": self.lj.epsilon_QtQt,
                "epsilon_FtFt": self.lj.epsilon_FtFt,
                "epsilon_QtFt": self.lj.epsilon_QtFt,
                "epsilon_QtCQtC": self.lj.epsilon_QtCQtC,
                "epsilon_FtCFtC": self.lj.epsilon_FtCFtC,
                "epsilon_QtCFtC": self.lj.epsilon_QtCFtC,
                "epsilon_QtQtC": self.lj.epsilon_QtQtC,
                "epsilon_FtFtC": self.lj.epsilon_FtFtC,
                "epsilon_QtCFt": self.lj.epsilon_QtCFt,
                "epsilon_QtFtC": self.lj.epsilon_QtFtC,
                "cutoff_factor": self.lj.cutoff_factor,
            },
            # Nested soft (harmonic-repulsion) config (resolved per-pair values, not None)
            "soft": {
                "k_QtQt": self.soft.k_QtQt,
                "k_FtFt": self.soft.k_FtFt,
                "k_QtFt": self.soft.k_QtFt,
                "k_QtCQtC": self.soft.k_QtCQtC,
                "k_FtCFtC": self.soft.k_FtCFtC,
                "k_QtCFtC": self.soft.k_QtCFtC,
                "k_QtQtC": self.soft.k_QtQtC,
                "k_FtFtC": self.soft.k_FtFtC,
                "k_QtCFt": self.soft.k_QtCFt,
                "k_QtFtC": self.soft.k_QtFtC,
            },
            # Nested weak (piecewise-harmonic) config (resolved per-pair values, not None)
            "weak": {
                "k_QtQt": self.weak.k_QtQt, "k_FtFt": self.weak.k_FtFt, "k_QtFt": self.weak.k_QtFt,
                "k_QtCQtC": self.weak.k_QtCQtC, "k_FtCFtC": self.weak.k_FtCFtC,
                "k_QtCFtC": self.weak.k_QtCFtC, "k_QtQtC": self.weak.k_QtQtC,
                "k_FtFtC": self.weak.k_FtFtC, "k_QtCFt": self.weak.k_QtCFt,
                "k_QtFtC": self.weak.k_QtFtC,
                "depth_QtQt": self.weak.depth_QtQt, "depth_FtFt": self.weak.depth_FtFt,
                "depth_QtFt": self.weak.depth_QtFt, "depth_QtCQtC": self.weak.depth_QtCQtC,
                "depth_FtCFtC": self.weak.depth_FtCFtC, "depth_QtCFtC": self.weak.depth_QtCFtC,
                "depth_QtQtC": self.weak.depth_QtQtC, "depth_FtFtC": self.weak.depth_FtFtC,
                "depth_QtCFt": self.weak.depth_QtCFt, "depth_QtFtC": self.weak.depth_QtFtC,
                "cutoff_factor": self.weak.cutoff_factor,
            },
            # Simulation parameters
            "box_size": list(self.box_size),  # Convert tuple to list for JSON
            "periodic_boundary": self.periodic_boundary,
            "temperature": self.temperature,
            "equilibration_potential": self.equilibration_potential,
            "timestep": self.timestep,
            "n_steps": self.n_steps,
            "record_stride": self.record_stride,
            "observable_stride": self.observable_stride,
            "particles_observable_stride": self.particles_observable_stride,
            "heavy_observable_stride": self.heavy_observable_stride,
            "n_qt": self.n_qt,
            "n_ft": self.n_ft,
            "kernel": self.kernel,
            "n_threads": self.n_threads,
            "rng_seed": self.rng_seed,
            "output_file": self.output_file,
            # Phase schedule (None when this is a single ordinary run)
            "phases": [p.to_dict() for p in self.phases] if self.phases else None,
        }

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a flat dictionary.
        
        Useful for logging, display, or parameter sweeps.
        Note: Use to_dict() for serialization/deserialization.
        """
        return {
            # Qt parameters
            "qt_name": self.qt.name,
            "qt_radius": self.qt.radius,
            "qt_diffusion": self.qt.diffusion,
            "qt_cluster_diffusion": self.qt.cluster_diffusion,
            "qt_cluster_name": self.qt.cluster_name,
            # Ft parameters
            "ft_name": self.ft.name,
            "ft_radius": self.ft.radius,
            "ft_diffusion": self.ft.diffusion,
            "ft_cluster_diffusion": self.ft.cluster_diffusion,
            "ft_cluster_name": self.ft.cluster_name,
            # Topology parameters
            "topology_name": self.topology.name,
            "binding_radius": self.topology.binding_radius,
            "kon": self.topology.kon,
            "k_bond": self.topology.k_bond,
            "ft_monovalent": self.topology.ft_monovalent,
            "koff": self.topology.koff,
            "allow_loops": self.topology.allow_loops,
            "equilibrium_bond_length": self.equilibrium_bond_length,
            # LJ parameters
            "epsilon_QtQt": self.lj.epsilon_QtQt,
            "epsilon_FtFt": self.lj.epsilon_FtFt,
            "epsilon_QtFt": self.lj.epsilon_QtFt,
            "epsilon_QtCQtC": self.lj.epsilon_QtCQtC,
            "epsilon_FtCFtC": self.lj.epsilon_FtCFtC,
            "epsilon_QtCFtC": self.lj.epsilon_QtCFtC,
            "epsilon_QtQtC": self.lj.epsilon_QtQtC,
            "epsilon_FtFtC": self.lj.epsilon_FtFtC,
            "epsilon_QtCFt": self.lj.epsilon_QtCFt,
            "epsilon_QtFtC": self.lj.epsilon_QtFtC,
            "potential_type": self.potential_type,
            "cutoff_factor": self.lj.cutoff_factor,
            # Soft (harmonic-repulsion) per-pair force constants
            "k_QtQt": self.soft.k_QtQt,
            "k_FtFt": self.soft.k_FtFt,
            "k_QtFt": self.soft.k_QtFt,
            "k_QtCQtC": self.soft.k_QtCQtC,
            "k_FtCFtC": self.soft.k_FtCFtC,
            "k_QtCFtC": self.soft.k_QtCFtC,
            "k_QtQtC": self.soft.k_QtQtC,
            "k_FtFtC": self.soft.k_FtFtC,
            "k_QtCFt": self.soft.k_QtCFt,
            "k_QtFtC": self.soft.k_QtFtC,
            # Weak (piecewise-harmonic) per-pair force constants + depths (prefixed to avoid
            # colliding with the soft k_* keys above in this flat mapping)
            "weak_k_QtQt": self.weak.k_QtQt, "weak_k_FtFt": self.weak.k_FtFt,
            "weak_k_QtFt": self.weak.k_QtFt, "weak_k_QtCQtC": self.weak.k_QtCQtC,
            "weak_k_FtCFtC": self.weak.k_FtCFtC, "weak_k_QtCFtC": self.weak.k_QtCFtC,
            "weak_k_QtQtC": self.weak.k_QtQtC, "weak_k_FtFtC": self.weak.k_FtFtC,
            "weak_k_QtCFt": self.weak.k_QtCFt, "weak_k_QtFtC": self.weak.k_QtFtC,
            "weak_depth_QtQt": self.weak.depth_QtQt, "weak_depth_FtFt": self.weak.depth_FtFt,
            "weak_depth_QtFt": self.weak.depth_QtFt, "weak_depth_QtCQtC": self.weak.depth_QtCQtC,
            "weak_depth_FtCFtC": self.weak.depth_FtCFtC, "weak_depth_QtCFtC": self.weak.depth_QtCFtC,
            "weak_depth_QtQtC": self.weak.depth_QtQtC, "weak_depth_FtFtC": self.weak.depth_FtFtC,
            "weak_depth_QtCFt": self.weak.depth_QtCFt, "weak_depth_QtFtC": self.weak.depth_QtFtC,
            "weak_cutoff_factor": self.weak.cutoff_factor,
            # Simulation parameters
            "box_size": self.box_size,
            "periodic_boundary": self.periodic_boundary,
            "temperature": self.temperature,
            "equilibration_potential": self.equilibration_potential,
            "timestep": self.timestep,
            "n_steps": self.n_steps,
            "total_time_ns": self.total_simulation_time,
            "total_time_us": self.total_simulation_time_us,
            "record_stride": self.record_stride,
            "observable_stride": self.observable_stride,
            "particles_observable_stride": self.particles_observable_stride,
            "heavy_observable_stride": self.heavy_observable_stride,
            "effective_heavy_observable_stride": self.effective_heavy_observable_stride,
            "n_qt": self.n_qt,
            "n_ft": self.n_ft,
            "kernel": self.kernel,
            "n_threads": self.n_threads,
            "rng_seed": self.rng_seed,
            "output_file": self.output_file,
        }

    def print_summary(self):
        """Print a formatted summary of the configuration."""
        print("=" * 60)
        print("SIMULATION CONFIGURATION")
        print("=" * 60)
        print(f"\nParticles:")
        print(f"  Qt: r={self.qt.radius} nm, D={self.qt.diffusion} nm²/ns "
              f"(cluster: D={self.qt.cluster_diffusion})")
        print(f"  Ft: r={self.ft.radius} nm, D={self.ft.diffusion} nm²/ns "
              f"(cluster: D={self.ft.cluster_diffusion})")
        print(f"  Counts: {self.n_qt} Qt + {self.n_ft} Ft = {self.n_qt + self.n_ft} total")
        print(f"\nTopology:")
        print(f"  Binding radius: {self.topology.binding_radius} nm")
        print(f"  Binding rate (kon): {self.topology.kon} nm³/(ns·part)")
        print(f"  Bond-breaking rate (koff): {self.topology.koff} /(edge·ns)")
        print(f"  Bond stiffness: {self.topology.k_bond} kJ/(mol·nm²)")
        print(f"  Equilibrium bond length: {self.equilibrium_bond_length} nm")
        if self.potential_type == "soft":
            # Soft mode is self-contained: show the per-pair repulsion constants; the LJ
            # epsilon values are ignored entirely in this mode.
            s = self.soft
            print(f"\nSoft repulsion (harmonic):")
            print(f"  Potential type: soft   (lj.epsilon ignored in soft mode)")
            print(f"  k Qt-Qt: {s.k_QtQt} kJ/(mol·nm²)", end="")
            if s.k_QtCQtC == 0:
                print(f"  (QtC-QtC: disabled)", end="")
            elif s.k_QtCQtC != s.k_QtQt:
                print(f"  (QtC-QtC: {s.k_QtCQtC})", end="")
            print()
            print(f"  k Ft-Ft: {s.k_FtFt} kJ/(mol·nm²)", end="")
            if s.k_FtCFtC == 0:
                print(f"  (FtC-FtC: disabled)", end="")
            elif s.k_FtCFtC != s.k_FtFt:
                print(f"  (FtC-FtC: {s.k_FtCFtC})", end="")
            print()
            print(f"  k Qt-Ft: {s.k_QtFt} kJ/(mol·nm²)", end="")
            if s.k_QtCFtC == 0:
                print(f"  (QtC-FtC: disabled)", end="")
            elif s.k_QtCFtC != s.k_QtFt:
                print(f"  (QtC-FtC: {s.k_QtCFtC})", end="")
            print()
            all_cluster_default = (
                s.k_QtCQtC == s.k_QtQt and s.k_FtCFtC == s.k_FtFt and s.k_QtCFtC == s.k_QtFt
            )
            all_mixed_default = (
                s.k_QtQtC == s.k_QtCQtC and s.k_FtFtC == s.k_FtCFtC and
                s.k_QtCFt == s.k_QtCFtC and s.k_QtFtC == s.k_QtCFtC
            )
            if all_cluster_default and all_mixed_default:
                print(f"  Cluster/mixed k: same as free (default)")
            elif all_mixed_default:
                print(f"  Mixed-state k: same as cluster (default)")
            else:
                print(f"  Mixed-state k: Qt-QtC={s.k_QtQtC}, Ft-FtC={s.k_FtFtC}, "
                      f"QtC-Ft={s.k_QtCFt}, Qt-FtC={s.k_QtFtC}")
        elif self.potential_type == "weak":
            w = self.weak
            print(f"\nWeak interaction (piecewise-harmonic):")
            print(f"  Potential type: weak   (lj.epsilon ignored in weak mode)")
            print(f"  k     Qt-Qt={w.k_QtQt}, Ft-Ft={w.k_FtFt}, Qt-Ft={w.k_QtFt} kJ/(mol·nm²)")
            print(f"  depth Qt-Qt={w.depth_QtQt}, Ft-Ft={w.depth_FtFt}, Qt-Ft={w.depth_QtFt} kJ/mol")
            print(f"  cutoff: {w.cutoff_factor} × contact; well minimum at contact (r_i+r_j)")
            k_default = (
                w.k_QtCQtC == w.k_QtQt and w.k_FtCFtC == w.k_FtFt and w.k_QtCFtC == w.k_QtFt and
                w.k_QtQtC == w.k_QtCQtC and w.k_FtFtC == w.k_FtCFtC and
                w.k_QtCFt == w.k_QtCFtC and w.k_QtFtC == w.k_QtCFtC
            )
            d_default = (
                w.depth_QtCQtC == w.depth_QtQt and w.depth_FtCFtC == w.depth_FtFt and
                w.depth_QtCFtC == w.depth_QtFt and w.depth_QtQtC == w.depth_QtCQtC and
                w.depth_FtFtC == w.depth_FtCFtC and w.depth_QtCFt == w.depth_QtCFtC and
                w.depth_QtFtC == w.depth_QtCFtC
            )
            if k_default and d_default:
                print(f"  Cluster/mixed k, depth: same as free (default)")
        else:
            print(f"\nLennard-Jones:")
            print(f"  Potential type: {self.potential_type}")
            # cutoff_factor may be None (derive from the mode); show the effective value.
            _cf = self.lj.cutoff_factor
            if _cf is None:
                _cf = self.lj.LJ_CUTOFF_FACTOR if self.potential_type == "LJ" else self.lj.WCA_CUTOFF_FACTOR
            print(f"  Cutoff factor: {_cf:.3f}")
            lj = self.lj
            print(f"  ε Qt-Qt: {lj.epsilon_QtQt} kJ/mol", end="")
            if lj.epsilon_QtCQtC == 0:
                print(f"  (QtC-QtC: disabled)", end="")
            elif lj.epsilon_QtCQtC != lj.epsilon_QtQt:
                print(f"  (QtC-QtC: {lj.epsilon_QtCQtC})", end="")
            print()
            print(f"  ε Ft-Ft: {lj.epsilon_FtFt} kJ/mol", end="")
            if lj.epsilon_FtCFtC == 0:
                print(f"  (FtC-FtC: disabled)", end="")
            elif lj.epsilon_FtCFtC != lj.epsilon_FtFt:
                print(f"  (FtC-FtC: {lj.epsilon_FtCFtC})", end="")
            print()
            print(f"  ε Qt-Ft: {lj.epsilon_QtFt} kJ/mol", end="")
            if lj.epsilon_QtCFtC == 0:
                print(f"  (QtC-FtC: disabled)", end="")
            elif lj.epsilon_QtCFtC != lj.epsilon_QtFt:
                print(f"  (QtC-FtC: {lj.epsilon_QtCFtC})", end="")
            print()
            # Show cluster/mixed-state summary if all defaults
            all_cluster_default = (
                lj.epsilon_QtCQtC == lj.epsilon_QtQt and
                lj.epsilon_FtCFtC == lj.epsilon_FtFt and
                lj.epsilon_QtCFtC == lj.epsilon_QtFt
            )
            all_mixed_default = (
                lj.epsilon_QtQtC == lj.epsilon_QtCQtC and
                lj.epsilon_FtFtC == lj.epsilon_FtCFtC and
                lj.epsilon_QtCFt == lj.epsilon_QtCFtC and
                lj.epsilon_QtFtC == lj.epsilon_QtCFtC
            )
            if all_cluster_default and all_mixed_default:
                print(f"  Cluster/mixed ε: same as free (default)")
            elif all_mixed_default:
                print(f"  Mixed-state ε: same as cluster (default)")
            else:
                print(f"  Mixed-state ε: Qt-QtC={lj.epsilon_QtQtC}, Ft-FtC={lj.epsilon_FtFtC}, "
                      f"QtC-Ft={lj.epsilon_QtCFt}, Qt-FtC={lj.epsilon_QtFtC}")
        print(f"\nSimulation:")
        print(f"  Box: {self.box_size[0]} × {self.box_size[1]} × {self.box_size[2]} nm")
        print(f"  Temperature: {self.temperature} K")
        print(f"  Equilibration potential: {self.equilibration_potential}")
        print(f"  Timestep: {self.timestep} ns ({format_duration(self.timestep)})")
        if self.phases:
            print(f"  Phases: {len(self.phases)} "
                  f"({', '.join(p.name for p in self.phases)})")
            for p in self.phases:
                ph_ns = p.n_steps * self.timestep
                print(f"    - {p.name}: {p.n_steps:,} steps ({format_duration(ph_ns)}), "
                      f"binding={p.binding}, breaking={p.breaking}, pot={p.potential_type}")
            print(f"  Total steps: {self.effective_n_steps:,} "
                  f"({format_duration(self.total_simulation_time_us * 1e3)} total)")
        else:
            print(f"  Steps: {self.n_steps:,} "
                  f"({format_duration(self.total_simulation_time_us * 1e3)} total)")
        print(f"  Output: {self.output_file}")
        print("=" * 60)
    
    def save_json(self, filepath: str):
        """
        Save configuration to a JSON file.
        
        The saved configuration can be loaded later with load_json().
        
        Parameters
        ----------
        filepath : str
            Path to save the JSON file
        """
        config_dict = self.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✓ Configuration saved to {filepath}")
    
    @classmethod
    def load_json(cls, filepath: str) -> "SimulationConfig":
        """
        Load configuration from a JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to the JSON file (saved with save_json())
        
        Returns
        -------
        SimulationConfig
            Reconstructed configuration object
        """
        # Check file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                params = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in configuration file '{filepath}': {e.msg}",
                e.doc, e.pos
            )
        
        # Detect format and validate required fields
        is_nested = isinstance(params.get("qt"), dict)
        
        if is_nested:
            # Nested format - check for nested structure
            required_fields = ['qt', 'ft', 'topology', 'lj', 'box_size', 'n_steps']
            missing_fields = [f for f in required_fields if f not in params]
            if missing_fields:
                raise ValueError(
                    f"Configuration file '{filepath}' is missing required fields: {missing_fields}"
                )
        else:
            # Flat format - check for essential fields (more lenient)
            required_fields = ['box_size', 'n_steps']
            missing_fields = [f for f in required_fields if f not in params]
            if missing_fields:
                raise ValueError(
                    f"Configuration file '{filepath}' is missing required fields: {missing_fields}"
                )
        
        # Use from_dict to reconstruct (it handles both formats)
        try:
            config = cls.from_dict(params)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Error reconstructing configuration from '{filepath}': {e}"
            )
        
        print(f"✓ Configuration loaded from {filepath}")
        return config


# =============================================================================
# PARAMETER-STRING NAMING
# =============================================================================

def format_param_string(config: "SimulationConfig") -> str:
    """Canonical parameter string for auto filenames and ensemble folder names.

    Single source of truth for the naming convention so the single-run trajectory
    filename and the ensemble folder name can never drift apart.

    Format (ordinary single run)::

        {n_qt}Qt_{n_ft}Ft_{potential_type}_eQQ{e}_eFF{e}_eQF{e}_kon{kon}_dt{dt}ps_{total_time}us

    Format (phased agglomeration<->deagglomeration run, config.phases set)::

        {n_qt}Qt_{n_ft}Ft_{POT}_eQQ{e}_eFF{e}_eQF{e}_phases{N}_kon{kon}
            _aggsteps{A}_koff{koff}_deaggsteps{D}_dt{dt}ps_{total_time}us

    where N = number of phases (= 2*n_cycles for make_agg_deagg_phases; cycles = N/2),
    A = steps of the first agglomeration phase, D = steps of the first deagglomeration
    phase, and {total_time} is the sum over all phases. A `_FtMono` tag is appended last
    when ft_monovalent. Numbers are formatted without trailing zeros (e.g. kon0.001,
    eQQ2.5, dt20ps, 100us).
    """
    lj = config.lj

    def fmt_num(val):
        return f"{int(val)}" if val == int(val) else f"{val}"

    kon_str = f"kon{fmt_num(config.topology.kon)}"

    # Timestep and total time use an adaptive unit (ps/ns/µs/ms/s) chosen from magnitude.
    # For the ranges existing datasets live in (dt in ps, total in µs) this reproduces the
    # old "dt50ps"/"100us" strings exactly; only ms/s-scale runs get the new units.
    dt_str = f"dt{format_duration(config.timestep, ascii=True)}"
    time_str = format_duration(config.total_simulation_time_us * 1e3, ascii=True)

    # Coefficient block: soft/weak modes are self-contained (epsilon unused), so they encode
    # their own free-free parameters; WCA/LJ encode the free-free epsilons (eQQ/eFF/eQF)
    # exactly as before, so their filenames are byte-identical.
    if config.potential_type == "soft":
        s = config.soft
        coeff = f"kQQ{fmt_num(s.k_QtQt)}_kFF{fmt_num(s.k_FtFt)}_kQF{fmt_num(s.k_QtFt)}"
    elif config.potential_type == "weak":
        w = config.weak
        coeff = (f"kQQ{fmt_num(w.k_QtQt)}_kFF{fmt_num(w.k_FtFt)}_kQF{fmt_num(w.k_QtFt)}"
                 f"_dQQ{fmt_num(w.depth_QtQt)}_dFF{fmt_num(w.depth_FtFt)}_dQF{fmt_num(w.depth_QtFt)}")
    else:
        coeff = (f"eQQ{fmt_num(lj.epsilon_QtQt)}_eFF{fmt_num(lj.epsilon_FtFt)}"
                 f"_eQF{fmt_num(lj.epsilon_QtFt)}")

    # Leading identity block, shared by single and phased runs.
    prefix = f"{config.n_qt}Qt_{config.n_ft}Ft_{config.potential_type}_{coeff}"

    # Additive tags so monovalent-Ft / loop-permitting runs don't collide with the
    # default ones on disk. Both off by default => suffixes absent => existing
    # folder/file names are unchanged.
    mono_str = "_FtMono" if config.topology.ft_monovalent else ""
    loop_str = "_loops" if config.topology.allow_loops else ""

    if config.phases:
        # Phased layout: pair kon->agglomeration and koff->deagglomeration with their
        # per-phase step counts. Step counts come from the first phase of each kind, which
        # is identical across cycles for make_agg_deagg_phases.
        koff_str = f"koff{fmt_num(config.topology.koff)}"
        agg_steps = next(
            (p.n_steps for p in config.phases if p.binding and not p.breaking), None
        )
        deagg_steps = next((p.n_steps for p in config.phases if p.breaking), None)

        parts = [f"phases{len(config.phases)}", kon_str]
        if agg_steps is not None:
            parts.append(f"aggsteps{int(agg_steps)}")
        parts.append(koff_str)
        if deagg_steps is not None:
            parts.append(f"deaggsteps{int(deagg_steps)}")
        parts.extend([dt_str, time_str])
        return f"{prefix}_" + "_".join(parts) + loop_str + mono_str

    # Ordinary single run (unchanged when allow_loops/ft_monovalent are default).
    return f"{prefix}_{kon_str}_{dt_str}_{time_str}{loop_str}{mono_str}"


# =============================================================================
# SYSTEM SETUP FUNCTIONS
# =============================================================================

