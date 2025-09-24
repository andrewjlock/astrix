"""
AsTrIX — Aerospace Trajectory Imaging & Diagnostics toolbox.

This package provides tools for aerospace trajectory analysis and visualization.
"""

from . import _backend_utils

if _backend_utils.HAS_JAX:
    _backend_utils.enforce_cpu_x64()

from . import primitives
from . import utils
from . import functs
from .primitives import Time, Point, Path, Frame, Ray, RotationSequence
from ._backend_utils import resolve_backend


__all__ = [
    "primitives",
    "utils",
    "functs",
    "Time",
    "Point",
    "Path",
    "resolve_backend",
    "Frame",
    "Ray",
    "RotationSequence",
]
