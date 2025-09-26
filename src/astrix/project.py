# pyright: reportExplicitAny=false, reportAny=false, reportImplicitOverride=false

from __future__ import annotations

from typing import Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ._backend_utils import (
    resolve_backend,
    Array,
    ArrayNS,
    BackendArg,
)

from astrix.functs import ensure_1d, ensure_2d
from astrix.time import Time, TimeLike, TIME_INVARIANT
from astrix.generic import AbstractValue

class CameraLike(ABC):
    """A marker interface for camera-like objects."""

    _res: tuple[int, int]
    _sensor_size: tuple[float, float]
    _rad_coef: ArrayNS | None
    _zoom: AbstractValue | None
    _xp: ArrayNS

    @property
    def res(self) -> tuple[int, int]:
        """Image resolution (width, height) in pixels."""
        return self._res

    @property
    def sensor_size(self):
        """Physical sensor size (width, height) in mm."""
        return self._sensor_size

    @property
    @abstractmethod
    def has_dist(self) -> bool:
        """Whether the camera has distortion coefficients."""
        pass

    @abstractmethod
    def mat(self, zoom: Array | None) -> Array:
        """Camera intrinsic matrix."""
        pass

    @abstractmethod
    def rad_coef(self, zoom: float | None) -> Array:
        """Radial distortion coefficients."""
        pass

    @abstractmethod
    def fov(self, zoom: float | None) -> tuple[float, float]:
        """Field of view in degrees (horizontal, vertical)."""
        pass

    @abstractmethod
    def convert_to(self, backend: BackendArg) -> CameraLike:
        """Convert the camera to a different backend.
        """
        pass


    def has_zoom(self) -> bool:
        """Whether the camera has a zoom level associated with it."""
        return self._zoom is not None

    def interp_zoom(self, time: TimeLike) -> Array:
        """Interpolate the zoom level at the given time.
        Returns None if the camera has no zoom level.
        """
        if self._zoom is not None:
            return self._zoom.interp(time)
        raise ValueError("Camera has no zoom level.")



@dataclass
class FixedZoomCamera:
    """A simple pinhole camera model with fixed zoom.

    Notes:
        The camera intrinsic matrix is given by:
            [ fx  0  cx ]
            [ 0  fy  cy ]
            [ 0   0   1 ]

        where:

            fx = focal_length * res[0] / sensor_size[0]
            fy = focal_length * res[1] / sensor_size[1]
            cx = res[0] / 2
            cy = res[1] / 2

        The field of view is given by:
            fov_x = 2 * arctan(sensor_size[0] / (2 * focal_length))
            fov_y = 2 * arctan(sensor_size[1] / (2 * focal_length))

    """

    _res: tuple[int, int]
    _sensor_size: tuple[float, float]
    _focal_length: float
    _mat: Array
    _rad_coef: Array | None
    _xp: ArrayNS

    def __init__(
        self,
        res: tuple[int, int],
        sensor_size: tuple[float, float],
        focal_length: float,
        rad_coef: Array | None = None,
        backend: BackendArg = None,
    ) -> None:
        """
        Create a FixedZoomCamera using focal length.

        Args:
            res: Image resolution (width, height) in pixels.
            sensor_size: Physical sensor size (width, height) in mm.
            focal_length: Focal length in mm.
            rad_coef (optional): Radial distortion coefficients, defaults to None
            backend (optional): Backend to use. Either "numpy" or "jax".

        """

        self._xp = resolve_backend(backend)
        self._res = res
        self._sensor_size = sensor_size
        self._focal_length = focal_length
        if rad_coef is not None:
            self._rad_coef = ensure_1d(rad_coef, self._xp)
        else:
            self._rad_coef = None

        fx = self._focal_length * self._res[0] / self._sensor_size[0]
        fy = self._focal_length * self._res[1] / self._sensor_size[1]
        cx = self._res[0] / 2
        cy = self._res[1] / 2
        self._mat = self._xp.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ]
        )

    # --- Constructors ---

    @classmethod
    def from_hoz_fov(
        cls,
        res: tuple[int, int],
        hoz_fov: float,
        sensor_size: tuple[float, float],
        rad_coef: Array | None = None,
        backend: BackendArg = None,
    ) -> FixedZoomCamera:
        """Create a FixedZoomCamera from horizontal field of view.

        Args:
            res: Image resolution (width, height) in pixels.
            hoz_fov: Horizontal field of view in degrees.
            sensor_size: Physical sensor size (width, height) in mm.
            rad_coef (optional): Radial distortion coefficients.
            backend (optional): Backend to use. Either "numpy" or "jax".
        """

        xp = resolve_backend(backend)
        focal_length = (sensor_size[0] / 2) / xp.tan(xp.deg2rad(hoz_fov) / 2)

        return cls(
            res=res,
            sensor_size=sensor_size,
            focal_length=float(focal_length),
            rad_coef=rad_coef,
            backend=backend,
        )

    # --- Dunder methods and properties ---

    def __repr__(self) -> str:
        return (
            f"FixedZoomCamera(res={self._res}, sensor_size={self._sensor_size}, "
            f"focal_length={self._focal_length}, backend='{self.backend}')"
        )

    def __str__(self) -> str:
        return f"FixedZoomCamera with resolution {self._res} and focal length {self._focal_length} mm"

    @property
    def backend(self) -> str:
        """Backend used by the camera."""
        return self._xp.__name__

    @property
    def has_dist(self):
        """Whether the camera has distortion coefficients."""
        return self._rad_coef is not None

    def fov(self, _):
        """Field of view in degrees (horizontal, vertical)."""

        return (
            self._xp.rad2deg(
                2 * self._xp.arctan(self._sensor_size[0] / (2 * self._focal_length))
            ),
            self._xp.rad2dev(
                2 * self._xp.arctan(self._sensor_size[1] / (2 * self._focal_length))
            ),
        )

    def mat(self, _: Any = None) -> Array:
        """Camera intrinsic matrix.
        No zoom parameter needed as this is a fixed zoom camera.
        """
        return self._mat

    def rad_coef(self, _):
        """Radial distortion coefficients."""
        if self._rad_coef is None:
            raise ValueError("Camera has no distortion coefficients.")
        return self._rad_coef

    def convert_to(self, backend: BackendArg) -> FixedZoomCamera:
        """Convert the camera to a different backend.
        """
        xp = resolve_backend(backend)
        if self._xp == xp:
            return self

        return FixedZoomCamera(
            res=self._res,
            sensor_size=self._sensor_size,
            focal_length=self._focal_length,
            rad_coef=self._rad_coef,
            backend=xp,
        )


@dataclass
class Pixel:
    """A pixel in an image.

    Notes:
        The pixel coordinates are given in the image coordinate system, where
        the origin is at the top-left corner of the image, and the u-axis
        points to the right and the v-axis points down.

    """

    _uv: Array
    _time: TimeLike
    _xp: ArrayNS

    def __init__(
        self,
        uv: Array,
        time: TimeLike = TIME_INVARIANT,
        backend: BackendArg = None,
    ) -> None:
        """Create a Pixel.

        Args:
            uv: Pixel coordinates (u, v) in pixels.
            time (optional): Time associated with the pixel.
            zoom (optional): Zoom level associated with the pixel, if any.
            backend (optional): Backend to use. Either "numpy" or "jax".
        """

        self._xp = resolve_backend(backend)
        self._uv = ensure_2d(uv, 2, self._xp)
        self._time = time

    # --- Constructors ---

    def _constructor(self, uv: Array, time: TimeLike, xp: ArrayNS) -> Pixel:
        """Initialise without data validation for speed"""
        obj = Pixel.__new__(Pixel)
        obj._uv = uv
        obj._time = time
        obj._xp = xp
        return obj

    # --- Dunder methods and properties ---

    @property
    def uv(self) -> Array:
        """Pixel coordinates (u, v) in pixels."""
        return self._uv

    @property
    def has_time(self) -> bool:
        """Whether the pixel has a time associated with it."""
        return isinstance(self._time, Time)

    @property
    def time(self) -> TimeLike:
        return self._time

    @property
    def backend(self) -> str:
        """Backend used by the pixel."""
        return self._xp.__name__


