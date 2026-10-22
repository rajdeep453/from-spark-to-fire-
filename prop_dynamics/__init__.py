"""
prop_dynamics
"""

from .config import ExperimentConfig
from .topology import build_topology
from .payload import make_payload, is_infected
from .personas import get_persona, get_uniform_persona
from .fit import fit_from_log, grid_search_fit, FitResult

__all__ = [
    "ExperimentConfig",
    "build_topology",
    "make_payload",
    "is_infected",
    "get_persona",
    "get_uniform_persona",
    "fit_from_log",
    "grid_search_fit",
    "FitResult",
]
