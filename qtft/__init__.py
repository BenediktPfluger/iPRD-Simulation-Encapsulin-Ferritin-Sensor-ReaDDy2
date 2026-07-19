"""qtft -- Qt-Ft encapsulin / ferritin agglomeration simulation (ReaDDy2).

Public API (see README.md for the architecture):

    from qtft import SimulationConfig, run_one, EnsembleSimulation
    import qtft.analysis as analysis
    import qtft.plotting as plotting   # requires matplotlib

Layering: config (single source of truth) -> system / engine (build + run) ->
analysis (matplotlib-free metrics) -> ensemble (multi-replica) ; plotting and
comparison are separate. analysis and plotting are intentionally NOT imported
here so that headless/ReaDDy-only contexts stay light.

Progress messages go through the ``qtft`` logger (not bare ``print``); by default it
streams to stdout with plain message-only formatting so notebook output looks unchanged.
Quiet it with ``qtft.set_log_level(logging.WARNING)`` or reconfigure the logger yourself.
"""
import logging as _logging
import sys as _sys

# Package logger. Attach a plain stdout handler once so the library's progress output is
# visible by default (as it was with print), while staying fully controllable by the user.
logger = _logging.getLogger("qtft")
if not logger.handlers:
    _handler = _logging.StreamHandler(_sys.stdout)
    _handler.setFormatter(_logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(_logging.INFO)
    logger.propagate = False


def set_log_level(level):
    """Set the verbosity of qtft's log output (e.g. ``logging.WARNING`` to quiet progress)."""
    _logging.getLogger("qtft").setLevel(level)


from .config import (
    NS_TO_US,
    _steps_to_us,
    ParticleConfig,
    TopologyConfig,
    LennardJonesConfig,
    SoftPotentialConfig,
    WeakInteractionConfig,
    PhaseConfig,
    make_agg_deagg_phases,
    SimulationConfig,
    format_param_string,
)
from .system import create_system
from .engine import (
    create_simulation,
    place_particles,
    run_simulation,
    equilibrate_system,
    run_one,
    run_phased,
    cleanup_empty_run_dirs,
)
from .ensemble import EnsembleSimulation

__all__ = [
    "NS_TO_US",
    "_steps_to_us",
    "ParticleConfig",
    "TopologyConfig",
    "LennardJonesConfig",
    "SoftPotentialConfig",
    "WeakInteractionConfig",
    "PhaseConfig",
    "make_agg_deagg_phases",
    "SimulationConfig",
    "format_param_string",
    "create_system",
    "create_simulation",
    "place_particles",
    "run_simulation",
    "equilibrate_system",
    "run_one",
    "run_phased",
    "cleanup_empty_run_dirs",
    "EnsembleSimulation",
    "set_log_level",
]
