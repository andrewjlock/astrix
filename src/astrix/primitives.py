# pyright: reportAny=false, reportImplicitOverride=false

from __future__ import annotations
import warnings

from dataclasses import dataclass
import datetime as dt
from scipy.spatial.transform import Rotation, Slerp

from ._backend_utils import (
    resolve_backend,
    Array,
    ArrayNS,
    BackendArg,
    backend_jit,
    _convert_rot_backend,
)
from .utils import ensure_1d, ensure_2d, ecef2geodet, geodet2ecef, is_increasing
from .functs import interp_nd, central_diff, interp_haversine


@dataclass
class Time:
    """One or more time instances.

    Represents time using seconds since Unix epoch (1970-01-01 00:00:00 UTC).
    Can handle single time instances or arrays of times with consistent
    backend support for JAX/NumPy compatibility.

    Parameters
    ----------
    secs : Array | list of float | float
        Time values in seconds since Unix epoch (1970-01-01 UTC)
    backend : BackendArg, optional
        Array backend to use (numpy, jax, etc.). Defaults to numpy.

    Examples
    --------
    Single time instance:

    >>> t = Time(1609459200.0)  # 2021-01-01 00:00:00 UTC

    Multiple times:

    >>> times = Time([1609459200.0, 1609545600.0])  # Jan 1-2, 2021

    From datetime:

    >>> from datetime import datetime, timezone
    >>> dt = datetime(2021, 1, 1, tzinfo=timezone.utc)
    >>> t = Time.from_datetime(dt)

    Notes
    -----
    All datetime objects must be timezone-aware to avoid ambiguity.
    """

    _secs: Array
    _min: float | Array
    _max: float | Array
    _xp: ArrayNS

    def __init__(
        self, secs: Array | list[float] | float, backend: BackendArg = None
    ) -> None:
        self._xp = resolve_backend(backend)
        self._secs = ensure_1d(secs, backend=self._xp)
        self._min = self._xp.min(self._secs)
        self._max = self._xp.max(self._secs)

    @backend_jit()
    def is_in_bounds(self, sec: Time) -> bool:
        """Check if the given time(s) are within the bounds of this Time object."""
        return bool(
            (self._xp.min(sec.secs) >= self._min)
            & (self._xp.max(sec.secs) <= self._max)
        )

    @classmethod
    def _constructor(cls, secs: Array, xp: ArrayNS) -> Time:
        """Internal constructor to create a Time object from seconds array
        Avoids type checking in __init__."""

        obj = cls.__new__(cls)
        obj._xp = xp
        obj._secs = secs
        obj._min = obj._xp.min(secs)
        obj._max = obj._xp.max(secs)
        return obj

    @classmethod
    def from_datetime(
        cls, time: dt.datetime | list[dt.datetime], backend: BackendArg = None
    ) -> Time:
        """Create a Time object from a list of datetime objects. \
        Will not accept timezone-unaware datetime obejects due to ambiguity.
        """
        if isinstance(time, dt.datetime):
            time = [time]

        if not all(
            t.tzinfo is not None and t.tzinfo.utcoffset(t) is not None for t in time
        ):
            raise ValueError("All datetime objects must be timezone-aware")
        xp = resolve_backend(backend)
        secs = xp.asarray([t.timestamp() for t in time])
        return cls(secs, backend=backend)

    @property
    def datetime(self) -> list[dt.datetime]:
        return [
            dt.datetime.fromtimestamp(float(s), tz=dt.timezone.utc) for s in self.secs
        ]

    def __getitem__(self, index: int) -> Time:
        return Time._constructor(
            self._xp.asarray(self.secs[index]).reshape(-1), xp=self._xp
        )

    def convert_to(self, backend: BackendArg) -> Time:
        """Convert the Time object to a different backend."""
        xp = resolve_backend(backend)
        if xp == self._xp:
            return self
        return Time._constructor(xp.asarray(self.secs), xp=xp)

    @property
    def is_increasing(self) -> bool:
        """Check if the time values are strictly increasing."""
        return is_increasing(self._secs, backend=self._xp)

    @property
    def secs(self) -> Array:
        """Get the time values in seconds since epoch."""
        # TODO: Decide whether to return a copy
        return self._secs

    @property
    def backend(self) -> str:
        """Get the name of the array backend in use (e.g., 'numpy', 'jax')."""
        return self._xp.__name__

    def __str__(self) -> str:
        return (
            f"Time array of length {self._secs.shape[0]} with {self._xp.__name__} backend. \n \
            Earliest time: {self.datetime[0]}, Latest time: {self.datetime[-1]}"
        )

    def __repr__(self) -> str:
        return f"Time, n={len(self)}, backend='{self._xp.__name__}')"

    def __len__(self) -> int:
        return self._secs.shape[0]

    def __add__(self, offset: float) -> Time:
        return Time(self.secs + offset, backend=self._xp)

    def __sub__(self, offset: float) -> Time:
        return Time(self.secs - offset, backend=self._xp)


@dataclass
class Point:
    _ecef: Array
    _xp: ArrayNS
    _time: Time | None = None

    def __init__(
        self, ecef: Array, time: Time | None = None, backend: BackendArg = None
    ) -> None:
        """Initialize a Point object with ECEF coordinates (x, y, z) in meters."""
        self._xp = resolve_backend(backend)
        self._ecef = ensure_2d(ecef, n=3, backend=self._xp)
        if time is not None:
            if ecef.shape[0] != time.secs.shape[0]:
                raise ValueError(
                    "Point and Time must be similar lengths if associated.\n"
                    + f"Found {ecef.shape[0]} points and {time.secs.shape[0]} times."
                )
            self._time = time

    @classmethod
    def _constructor(cls, ecef: Array, time: Time | None, xp: ArrayNS) -> Point:
        """Internal constructor to create a Point object from ECEF array
        Avoids type checking in __init__."""

        obj = cls.__new__(cls)
        obj._xp = xp
        obj._ecef = ecef
        obj._time = time
        return obj

    @classmethod
    def from_geodet(
        cls, geodet: Array, time: Time | None = None, backend: BackendArg = None
    ) -> Point:
        """Create a Point object from geodetic coordinates (lat, lon, alt).
        Lat and lon are in degrees, alt is in meters.
        """

        xp = resolve_backend(backend)
        geodet = ensure_2d(geodet, n=3, backend=xp)
        ecef = xp.asarray(geodet2ecef(geodet))
        return cls(ecef, time, backend=xp)

    @classmethod
    def from_list(cls, points: list[Point]) -> Point:
        """Create a Point object from a list of Point objects."""
        if not all(p.backend == points[0].backend for p in points):
            raise ValueError("All points must have the same backend.")
        if not all(p.has_time == points[0].has_time for p in points):
            raise ValueError("All points must either have time or not have time.")
        xp = resolve_backend(points[0].backend)

        ecef_joined = xp.concatenate([p.ecef for p in points]).reshape(-1, 3)
        if points[0].has_time:
            time_joined = Time(
                xp.concatenate([p.time.secs for p in points]),  # pyright: ignore[reportOptionalMemberAccess]
                backend=xp,
            )
        else:
            time_joined = None
        return cls(ecef_joined, time=time_joined, backend=xp)

    @property
    def ecef(self) -> Array:
        """Get the ECEF coordinates (x, y, z) in meters."""
        # TODO: Decide whether to return a copy
        return self._ecef.copy()

    @property
    def geodet(self) -> Array:
        """Convert to geodetic coordinates (lat [deg], lon [deg], alt [m])."""
        return self._xp.asarray(ecef2geodet(self.ecef))

    @property
    def has_time(self) -> bool:
        """Check if the Point has associated Time."""
        return isinstance(self._time, Time)

    @property
    def time(self) -> Time | None:
        """Get the associated Time object, if any."""
        return self._time

    @property
    def backend(self) -> str:
        """Get the name of the array backend in use (e.g., 'numpy', 'jax')."""
        return self._xp.__name__

    def convert_to(self, backend: BackendArg) -> Point:
        """Convert the Point object to a different backend."""
        xp = resolve_backend(backend)
        if xp == self._xp:
            return self
        if self.has_time:
            time_converted = self.time.convert_to(xp)  # pyright: ignore[reportOptionalMemberAccess]
        else:
            time_converted = None
        return Point._constructor(xp.asarray(self.ecef), time_converted, xp)

    def __getitem__(self, index: int) -> Point:
        return Point(self.ecef[index], backend=self._xp)

    def __str__(self) -> str:
        return f"""Point of length {self._ecef.shape[0]} 
            with {self._xp.__name__} backend. \n 
            First point (LLA): {self.geodet[0]}, 
            Last point (LLA): {self.geodet[-1]}"""

    def __repr__(self) -> str:
        return f"Point, n=<Select> {len(self)}, backend='{self._xp.__name__}'"

    def __len__(self) -> int:
        return self._ecef.shape[0]

    def __add__(self, added_point: Point) -> Point:
        if self.has_time != added_point.has_time:
            raise ValueError("Cannot combine points with and without time.")
        if self.backend != added_point.backend:
            raise ValueError("Cannot combine points with different backends.")

        ecef_joined = self._xp.concatenate([self.ecef, added_point.ecef])
        if self.has_time and added_point.has_time:
            time_joined = Time(
                self._xp.concatenate(
                    [
                        self.time.secs,  # pyright: ignore[reportOptionalMemberAccess]
                        added_point.time.secs,  # pyright: ignore[reportOptionalMemberAccess]
                    ]
                )
            )
        else:
            time_joined = None
        return Point(ecef_joined, time=time_joined, backend=self._xp)


@dataclass(frozen=True)
class Velocity:
    """
    Velocity vector(s) in ECEF coordinates (vx, vy, vz) in m/s.
    Associated with a Time object for the time instances of the velocities.
    Internal use only, typically created from Path objects.
    No data validation is performed.
    """

    vec: Array
    time: Time
    _xp: ArrayNS

    @property
    def magnitude(self) -> Array:
        """Get the velocity magnitude in m/s."""
        return self._xp.linalg.norm(self.vec, axis=1)

    @property
    def unit(self) -> Array:
        """Get the unit velocity vector."""
        mag = self.magnitude
        return self.vec / mag[:, self._xp.newaxis]

    def __str__(self) -> str:
        return f"Velocity array of length {self.vec.shape[0]} with {self._xp.__name__} backend."

    def __repr__(self) -> str:
        return f"Velocity, n={len(self)}, backend='{self._xp.__name__}')"

    def __len__(self) -> int:
        return self.vec.shape[0]

    @property
    def backend(self) -> str:
        return self._xp.__name__

    def convert_to(self, backend: BackendArg) -> Velocity:
        """Convert the Velocity object to a different backend."""
        xp = resolve_backend(backend)
        if xp == self._xp:
            return self
        return Velocity(xp.asarray(self.vec), self.time.convert_to(xp), xp)


class Path:
    _ecef: Array
    _vel: Array
    _time: Time
    _xp: ArrayNS

    def __init__(self, point: Point, backend: BackendArg = None) -> None:
        """Initialize a Path object from a Point object with associated Time."""
        if not point.has_time:
            raise ValueError("Point must have associated Time to create a Path.")
        self._xp = resolve_backend(backend)
        sort_indices = self._xp.argsort(point.time.secs)  # pyright: ignore[reportOptionalMemberAccess]
        self._time = Time(point.time.secs[sort_indices], backend=self._xp)  # pyright: ignore[reportOptionalMemberAccess]
        self._ecef = ensure_2d(point.ecef[sort_indices], n=3, backend=self._xp)
        self._vel = central_diff(self._time.secs, self._ecef, backend=self._xp)

    @property
    def time(self) -> Time:
        """Get the associated Time object."""
        return self._time

    @property
    def ecef(self) -> Array:
        """Get the ECEF coordinates (x, y, z) in meters."""
        return self._ecef

    @property
    def geodet(self) -> Array:
        """Convert to geodetic coordinates (lat [deg], lon [deg], alt [m])."""
        return self._xp.asarray(ecef2geodet(self.ecef))

    @property
    def backend(self) -> str:
        """Get the name of the array backend in use (e.g., 'numpy', 'jax')."""
        return self._xp.__name__

    @property
    def start_time(self) -> Time:
        """Get the start time of the Path."""
        return self.time[0]

    @property
    def end_time(self) -> Time:
        """Get the end time of the Path."""
        return self.time[-1]

    @property
    def vel(self) -> Velocity:
        """Get the Velocity object associated with the Path."""
        return Velocity(self._vel, self._time, self._xp)

    def __str__(self) -> str:
        return f"""Path of length {self._ecef.shape[0]} 
            with {self._xp.__name__} backend. \n 
            Start time: {self.time.datetime[0]}, 
            End time: {self.time.datetime[-1]} \n
            Start point (LLA): {self.geodet[0]},
            End point (LLA): {self.geodet[-1]}"""

    def __repr__(self) -> str:
        return f"Path, n={len(self)}, backend='{self._xp.__name__}')"

    def __len__(self) -> int:
        return self._ecef.shape[0]

    def convert_to(self, backend: BackendArg) -> Path:
        """Convert the Path object to a different backend."""
        xp = resolve_backend(backend)
        if xp == self._xp:
            return self
        return Path(
            Point(self.ecef, time=self.time.convert_to(xp), backend=xp), backend=xp
        )

    @backend_jit(["check_bounds"])
    def interp(
        self, time: Time, method: str = "linear", check_bounds: bool = True
    ) -> Point:
        """Interpolate the Path to the given times using the specified method.

        Args:
            time (Time): Times to interpolate to.
            method (str, optional): Interpolation method. Currently only 'linear' is supported. Defaults to 'linear'.
            check_bounds (bool, optional): Whether to check if the interpolation times are within the path time bounds. Defaults to True.

        Returns:
            Point: Interpolated Point object at the given times.
        """

        if check_bounds:
            if not self.time.is_in_bounds(time):
                warnings.warn(f"""Path interpolation times are out of bounds.
                    Path time range: {self.time.datetime[0]} to {self.time.datetime[-1]}
                    Interpolation time range: {time.datetime[0]} to {time.datetime[-1]}
                    Extrapolation is not supported and will raise an error.""")

        if method == "linear":
            interp_ecef = interp_nd(
                time.secs,
                self.time.secs,
                self.ecef,
                backend=self._xp,
            )
        elif method == "haversine":
            interp_ecef = interp_haversine(
                time.secs,
                self.time.secs,
                self.ecef,
                backend=self._xp,
            )
        else:
            raise ValueError("Currently only 'linear' interpolation is supported.")
        return Point._constructor(interp_ecef, time=time, xp=self._xp)

    @backend_jit(["method", "check_bounds"])
    def interp_vel(
        self, time: Time, method: str = "linear", check_bounds: bool = True
    ) -> Velocity:
        """Interpolate the Path velocity to the given times using the specified method.
        Currently only 'linear' interpolation is supported.
        """

        if method != "linear":
            raise ValueError("Currently only 'linear' interpolation is supported.")
        if check_bounds:
            if not self.time.is_in_bounds(time):
                warnings.warn(f"""Path interpolation times are out of bounds.
                    Path time range: {self.time.datetime[0]} to {self.time.datetime[-1]}
                    Interpolation time range: {time.datetime[0]} to {time.datetime[-1]}
                    Extrapolation is not supported and will raise an error.""")

        if method == "linear":
            interp_vel = interp_nd(
                time.secs,
                self.time.secs,
                self._vel,
                backend=self._xp,
            )
            return Velocity(interp_vel, time, self._xp)


class Frame:
    """ A static or moving reference frame defined by a rotation (and optionally a reference frame).

    Args:
        rot (scipy.spatial.transform.Rotation): Rotation object defining the frame orientation(s).
        ref_frame (Frame, optional): Reference frame that this frame is defined relative to. 
            If not provided, assumed to be ECEF frame. Defaults to None.
        time (Time, optional): Time object defining the time instances for the rotations. Must be provided
            if rot has multiple rotations, and be same length as rotations. Defaults to None.
        backend (BackendArg, optional): Array backend to use (numpy, jax, etc.). Defaults to numpy.

    Examples:
        Static frame with single rotation:
            >>> from scipy.spatial.transform import Rotation as R
            >>> from astrix import Frame
            >>> rot = R.from_euler('z', 45, degrees=True)
            >>> frame = Frame(rot)  # Static frame rotated 45 deg about Z from ECEF

        Static frame relative to another static frame:
            >>> rot1 = R.from_euler('z', 30, degrees=True)
            >>> rot2 = R.from_euler('z', 40, degrees=True)
            >>> frame1 = Frame(rot1)  # Reference frame
            >>> frame2 = Frame(rot2, ref_frame=frame1)  # 70 deg about Z from ECEF

        Dynamic frame with time-varying rotations:
            >>> from astrix import Time
            >>> times = Time([0, 10, 20])  # Times in seconds
            >>> rots = R.from_euler('z', [0, 90, 180], degrees=True)
            >>> frame = Frame(rots, time=times)  # Dynamic frame rotating 0-180 deg over 20s
        Interpolating a dynamic frame at specific times:
            >>> interp_times = Time([5, 15])  # Times to interpolate at
            >>> interp_rots = frame.interp(interp_times)  # Interpolated rotations

    Notes:
        - Frame interpolation is recursive through all reference frames.
        - If the frame is static (single rotation), time must be None.
        - If the frame is dynamic (multiple rotations), time must be provided and match the length
          of the rotations.
        - If a reference frame is provided, the absolute rotation is computed relative to it (intrinsic rotations).
        - Interpolation uses spherical linear interpolation (slerp) for smooth rotation transitions.
        - All inputs are converted to the specified backend when instantiated.

    """
    _rot: Rotation
    _xp: ArrayNS
    _ref_frame: Frame | None = None
    _time: Time | None = None
    _slerp: Slerp | None = None
    _static: bool = True

    def __init__(
        self,
        rot: Rotation,
        ref_frame: Frame | None = None,
        time: Time | None = None,
        backend: BackendArg = None,
    ) -> None:
        self._xp = resolve_backend(backend)
        self._rot = _convert_rot_backend(rot, self._xp)

        # Data validation
        if time is not None:
            if self._rot.single:
                raise ValueError("Time must be None for static Rotation frame")
            if len(time) != len(self._rot):
                raise ValueError("Time and rotations must have same length for Frame")
            if not time.is_increasing:
                raise ValueError(
                    "Time values must be strictly increasing to construct Frame"
                )
            time = time.convert_to(self._xp)
            self._time = time
            self._slerp = Slerp(time.secs, self._rot)
            self._static = False
        else:
            if not self._rot.single:
                raise ValueError("Time must be provided for dynamic Rotation frame")

        if ref_frame is not None:
            if ref_frame.backend != self.backend:
                ref_frame = ref_frame.convert_to(self.backend)
            self._ref_frame = ref_frame
            if not ref_frame.is_static:
                self._static = False

    @property
    def backend(self) -> str:
        """Get the name of the array backend in use (e.g., 'numpy', 'jax')."""
        return self._xp.__name__

    @property
    def time(self) -> Time | None:
        """Get the Time object associated with the frame, if any."""
        return self._time

    @property
    def rel_rot(self) -> Rotation:
        """Get the last rotation of the frame relative to the reference frame."""
        return self._rot

    @property
    def static_rot(self) -> Rotation:
        """Get the last absolute rotation of the frame (including reference frame)."""
        if self.is_static:
            # TODO: Check these are in the correct order
            if self._ref_frame is not None:
                return self._ref_frame.static_rot * self._rot
            else:
                return self._rot
        else:
            raise ValueError(
                "Cannot get absolute rotation of dynamic frame without time input"
            )

    @property
    def is_static(self) -> bool:
        """Check if the frame is static (single rotation) or dynamic (time-varying)."""
        return self._static

    @property
    def has_ref(self) -> bool:
        """Check if the frame has a reference frame."""
        return self._ref_frame is not None

    @backend_jit(["check_bounds"])
    def interp(self, time: Time, check_bounds: bool = True) -> Rotation:
        """Interpolate the frame at the given times to return the absolute rotation(s).

        Args:
            time (Time): Times to interpolate the frame at.
            check_bounds (bool, optional): Whether to check if the interpolation times are within
                the frame time bounds. Defaults to True.

        Returns:
            Rotation: Interpolated Rotation object at the given times.
        """

        if self.is_static:
            return Rotation._from_raw_quat(
                self._xp.repeat(self.static_rot.as_quat(), len(time), axis=0), xp=self._xp
            )

        if self.time is not None:
            if check_bounds:
                if not self.time.is_in_bounds(time):
                    warnings.warn(f"""Frame interpolation times are out of bounds.
                        Frame time range: {self.time.datetime[0]} to {self.time.datetime[-1]}
                        Interpolation time range: {time.datetime[0]} to {time.datetime[-1]}
                        Extrapolation is not supported and will raise an error.""")
            rel_rot = self._slerp(time.secs)  # pyright: ignore[reportOptionalCall]
        else:
            rel_rot = self._rot
        if self._ref_frame is None:
            return rel_rot
        return self._ref_frame.interp(time, check_bounds) * rel_rot

    def convert_to(self, backend: BackendArg) -> Frame:
        """Convert the Frame object (and all references) to a different backend."""
        xp = resolve_backend(backend)
        rot_converted = _convert_rot_backend(self._rot, xp)
        if self._ref_frame is not None:
            ref_converted = self._ref_frame.convert_to(backend)
        else:
            ref_converted = None
        if self._time is not None:
            time_converted = Time(self._time.secs, backend=xp)
        else:
            time_converted = None
        return Frame(
            rot_converted, ref_frame=ref_converted, time=time_converted, backend=xp
        )

# This needs more thought:
# - Should it support multiple rays?
# - Should it support length/non-length?
# - Should it support time/non-time?
# - Should it support interpolation itself (rather than separate function)?


class Ray:
    _origin: Array
    _unit: Array
    _xp: ArrayNS
    _time: Time | None = None

    def __init__(
        self,
        origin: Array,
        dir: Array,
        time: Time | None = None,
        backend: BackendArg = None,
    ) -> None:
        """Initialize a Ray object with origin, unit direction, optional length, and optional time.

        Args:
            origin (Array): Nx3 array of ray origin points in ECEF coordinates (meters).
            dir (Array): Nx3 array of unit direction vectors. Need not be normalised.
            time (Time, optional): Time object associated with the rays. 
                Must be same length as origin if provided. Defaults to None.
            backend (BackendArg, optional): Array backend to use (numpy, jax, etc.). Defaults to numpy.

        Raises:
            ValueError: If input arrays have incompatible shapes or if time length does not match origin length.
        """

        self._xp = resolve_backend(backend)
        self._origin = ensure_2d(origin, n=3, backend=self._xp)
        _dir = ensure_2d(dir, n=3, backend=self._xp)
        self._unit = _dir / self._xp.linalg.norm(_dir, axis=1)[:, self._xp.newaxis]
        if self._origin.shape[0] != self._unit.shape[0]:
            raise ValueError("Origin and unit direction arrays must have the same length.")
        if time is not None:
            if len(time) != self._origin.shape[0]:
                raise ValueError("Time length must match origin length if provided.")
            self._time = time.convert_to(self._xp)
        else:
            self._time = None

    @classmethod
    def _constructor(
        cls, origin: Array, unit: Array, time: Time | None, xp: ArrayNS
    ) -> Ray:
        """Internal constructor to create a Ray object from arrays
        Avoids type checking in __init__."""

        obj = cls.__new__(cls)
        obj._xp = xp
        obj._origin = origin
        obj._unit = unit
        obj._time = time
        return obj

    @classmethod
    def from_endpoint(
        cls,
        origin: Array,
        endpoint: Array,
        time: Time | None = None,
        backend: BackendArg = None,
    ) -> Ray:
        """Create a Ray object from origin and endpoint arrays.

        Args:
            origin (Array): Nx3 array of ray origin points in ECEF coordinates (meters).
            endpoint (Array): Nx3 array of ray endpoint points in ECEF coordinates (meters).
            time (Time, optional): Time object associated with the rays. 
                Must be same length as origin if provided. Defaults to None.
            backend (BackendArg, optional): Array backend to use (numpy, jax, etc.). Defaults to numpy.
        Returns:
            Ray: Ray object defined by the origin and direction from origin to endpoint.
        """

        xp = resolve_backend(backend)
        origin = ensure_2d(origin, n=3, backend=xp)
        endpoint = ensure_2d(endpoint, n=3, backend=xp)
        if origin.shape != endpoint.shape:
            raise ValueError("Origin and endpoint arrays must have the same shape.")
        dir = endpoint - origin
        return cls(origin, dir, time=time, backend=xp)

    @property
    def origin(self) -> Array:
        """Get the ray origin point(s)."""
        return self._origin

    @property
    def unit(self) -> Array:
        """Get the unit direction vector(s) of the ray."""
        return self._unit

    @property
    def has_time(self) -> bool:
        """Check if the Ray has associated Time."""
        return isinstance(self._time, Time)

    @property
    def time(self) -> Time:
        """Get the associated Time object, if any."""
        if self._time is None:
            raise ValueError("Ray does not have associated Time.")
        return self._time

    @property
    def backend(self) -> str:
        """Get the name of the array backend in use (e.g., 'numpy', 'jax')."""
        return self._xp.__name__

    def __len__(self) -> int:
        return self._origin.shape[0]

    def __str__(self) -> str:
        return f"""Ray of length {self._origin.shape[0]} 
            with {self._xp.__name__} backend. \n 
            First origin (LLA): {ecef2geodet(self._origin[0])}, 
            First direction (unit vector): {self._unit[0]}"""

    def __repr__(self) -> str:
        return f"Ray, n={len(self)}, backend='{self._xp.__name__}')"

    def convert_to(self, backend: BackendArg) -> Ray:
        """Convert the Ray object to a different backend."""
        xp = resolve_backend(backend)
        if xp == self._xp:
            return self
        if self.has_time:
            time_converted = self.time.convert_to(xp)  # pyright: ignore[reportOptionalMemberAccess]
        else:
            time_converted = None
        return Ray._constructor(
            xp.asarray(self.origin), xp.asarray(self.unit), time_converted, xp
        )

    @backend_jit(["check_bounds"])
    def interp(self, time: Time, check_bounds: bool = True) -> Ray:
        """Interpolate the Ray origin and direction to the given times.
            
        Args:
            time (Time): Times to interpolate to.
            check_bounds (bool, optional): Whether to check if the interpolation 
                times are within the ray time bounds. Defaults to True.
        Returns:
            Ray: Interpolated Ray object at the given times.
        """

        if not self.has_time:
            raise ValueError("Cannot interpolate Ray without associated Time.")
        if check_bounds:
            if not self.time.is_in_bounds(time):
                warnings.warn(f"""Ray interpolation times are out of bounds.
                    Ray time range: {self.time.datetime[0]} to {self.time.datetime[-1]}
                    Interpolation time range: {time.datetime[0]} to {time.datetime[-1]}
                    Extrapolation is not supported and will raise an error.""")

        interp_origin = interp_nd(
            time.secs,
            self.time.secs,  # pyright: ignore[reportOptionalMemberAccess]
            self.origin,
            backend=self._xp,
        )
        interp_unit = interp_nd(
            time.secs,
            self.time.secs,  # pyright: ignore[reportOptionalMemberAccess]
            self.unit,
            backend=self._xp,
        )
        interp_unit = interp_unit / self._xp.linalg.norm(interp_unit, axis=1)[:, self._xp.newaxis]
        return Ray._constructor(interp_origin, interp_unit, time=time.convert_to(self._xp), xp=self._xp)

