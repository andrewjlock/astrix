"""
AsTrIX — Aerospace Trajectory Imaging & Diagnostics toolbox.

This package provides tools for aerospace trajectory analysis and visualization.
"""

from . import _backend_utils

if _backend_utils.HAS_JAX:
    _backend_utils.enforce_cpu_x64()

from .time import Time, TIME_INVARIANT
from .spatial import Point, Path, Frame, Ray, RotationSequence
from .spatial import location, frame, rotation, ray
from .project import FixedZoomCamera, Pixel
from . import utils
from ._backend_utils import resolve_backend


__all__ = [
    "utils",
    "Time",
    "Point",
    "Path",
    "resolve_backend",
    "Frame",
    "Ray",
    "RotationSequence",
    "TIME_INVARIANT",
    "FixedZoomCamera",
    "Pixel",
    "location",
    "frame",
    "rotation",
    "ray",
]
