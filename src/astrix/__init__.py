"""
AsTrIX — Aerospace Trajectory Imaging & Diagnostics toolbox.

This package provides tools for aerospace trajectory analysis and visualization.
"""

from . import _backend_utils

_backend_utils.ensure_scipy_array_api_enabled()
if _backend_utils.HAS_JAX:
    _backend_utils.enforce_cpu_x64()

from .time import Time, TIME_INVARIANT, TimeGroup, time_linspace
from .spatial import Point, Path, Frame, Ray, RotationSequence, Velocity, Acceleration
from .spatial import location, frame, rotation, ray
from .project import FixedZoomCamera, Pixel
from . import utils
from ._backend_utils import resolve_backend
from . import functs


__all__ = [
    "utils",
    "Time",
    "TimeGroup",
    "time_linspace",
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
    "Velocity",
    "Acceleration",
    "functs",
]
