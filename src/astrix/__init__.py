"""
AsTrIX — Aerospace Trajectory Imaging & Diagnostics toolbox.

This package provides tools for aerospace trajectory analysis and visualization.
"""

from . import primatives
from . import utils
from . import _backend_utils

if _backend_utils.HAS_JAX:
    _backend_utils.enforce_cpu_x64()

__all__ = ['primatives', 'utils']
