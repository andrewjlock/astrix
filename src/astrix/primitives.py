# pyright: reportAny=false

from __future__ import annotations
from dataclasses import dataclass
import datetime as dt
import warnings

from ._backend_utils import (
    resolve_backend,
    Array,
    ArrayNS,
    BackendArg,
    backend_jit

)
from .utils import ensure_1d, ensure_2d, ecef2geodet, geodet2ecef
from .functs import interp_nd


@dataclass
class Time:
    """One or more time instances.

    Represents time using seconds since Unix epoch (1970-01-01 00:00:00 UTC).
    Can handle single time instances or arrays of times with consistent
    backend support for JAX/NumPy compatibility.

    Parameters
    ----------
    secs : Array
        Time values in seconds since Unix epoch (1970-01-01 UTC)
    backend : BackendArg, optional
        Array backend to use (numpy, jax, etc.). Defaults to numpy.

    Attributes
    ----------
    secs : Array or list of floats
        Time values in seconds since epoch (Unix timestamp)
    datetime : datetime or list of datetime
        Python datetime objects (computed property)

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
    _xp: ArrayNS

    def __init__(
        self, secs: Array | list[float] | float, backend: BackendArg = None
    ) -> None:
        self._xp = resolve_backend(backend)
        self._secs = ensure_1d(secs, backend=self._xp)

    def is_in_bounds(self, sec: Time) -> bool:
        """Check if the given time(s) are within the bounds of this Time object."""
        return bool(
            (self._xp.min(sec.secs) >= self._xp.min(self._secs))
            & (self._xp.max(sec.secs) <= self._xp.max(self._secs))
        )

    @classmethod
    def _constructor(cls, secs: Array, backend: BackendArg = None) -> Time:
        """Internal constructor to create a Time object from seconds array
        Avoids type checking in __init__."""

        obj = cls.__new__(cls)
        obj._xp = resolve_backend(backend)
        obj._secs = ensure_1d(secs, backend=obj._xp)
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
    def datetime(self):
        """Convert to a list of datetime objects."""
        return [
            dt.datetime.fromtimestamp(float(s), tz=dt.timezone.utc) for s in self.secs
        ]

    def __getitem__(self, index: int) -> Time:
        return Time(self.secs[index], backend=self._xp)

    @property
    def secs(self) -> Array:
        """Get the time values in seconds since epoch."""
        #TODO: Decide whether to return a copy
        return self._secs

    @property
    def backend(self) -> str:
        """Get the name of the array backend in use (e.g., 'numpy', 'jax')."""
        return self._xp.__name__

    def __repr__(self) -> str:
        return (
            f"Time array of length {self._secs.shape[0]} with {self._xp.__name__} backend. \n \
            Earliest time: {self.datetime[0]}, Latest time: {self.datetime[-1]}"
        )

    def __len__(self) -> int:
        return self._secs.shape[0]

    def __add__(self, other: float) -> Time:
        return Time(self.secs + other, backend=self._xp)

    def __sub__(self, other: float) -> Time:
        return Time(self.secs - other, backend=self._xp)


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
    def _constructor(
        cls, ecef: Array, time: Time | None, backend: ArrayNS
    ) -> Point:
        """Internal constructor to create a Point object from ECEF array
        Avoids type checking in __init__."""

        obj = cls.__new__(cls)
        obj._xp = backend
        obj._ecef = ecef
        obj._time = time
        return obj

    @classmethod
    def from_geodet(cls, geodet: Array, time: Time | None = None, backend: BackendArg = None) -> Point:
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
                xp.concatenate([p.time.secs for p in points]), backend=xp # pyright: ignore[reportOptionalMemberAccess]
            )
        else:
            time_joined = None
        return cls(ecef_joined, time=time_joined, backend=xp)

    @property
    def ecef(self) -> Array:
        """Get the ECEF coordinates (x, y, z) in meters."""
        #TODO: Decide whether to return a copy
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

    def __getitem__(self, index: int) -> Point:
        return Point(self.ecef[index], backend=self._xp)

    def __repr__(self) -> str:
        return f"""Point of length {self._ecef.shape[0]} 
            with {self._xp.__name__} backend. \n 
            First point (LLA): {self.geodet[0]}, 
            Last point (LLA): {self.geodet[-1]}"""

    def __len__(self) -> int:
        return self._ecef.shape[0]


    def __add__(self, added_point: Point) -> Point:
        if self.has_time != added_point.has_time:
            raise ValueError("Cannot combine points with and without time.")
        if self.backend != added_point.backend:
            raise ValueError("Cannot combine points with different backends.")

        ecef_joined = self._xp.concatenate([self.ecef, added_point.ecef])
        if self.has_time and added_point.has_time:
            time_joined = Time(self._xp.concatenate([self.time.secs, # pyright: ignore[reportOptionalMemberAccess] 
                                                     added_point.time.secs, # pyright: ignore[reportOptionalMemberAccess]
                                                     ]))
        else:
            time_joined = None
        return Point(ecef_joined, time=time_joined, backend=self._xp)


class Path:
    _ecef: Array
    _time: Time
    _xp: ArrayNS

    def __init__(self, point: Point, backend: BackendArg = None) -> None:
        """Initialize a Path object from a Point object with associated Time."""
        if not point.has_time:
            raise ValueError("Point must have associated Time to create a Path.")
        self._xp = resolve_backend(backend)
        sort_indices = self._xp.argsort(point.time.secs) # pyright: ignore[reportOptionalMemberAccess]
        self._time = Time(point.time.secs[sort_indices], backend=self._xp) # pyright: ignore[reportOptionalMemberAccess]
        self._ecef = ensure_2d(point.ecef[sort_indices], n=3, backend=self._xp)

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

    def __repr__(self) -> str:
        return f"""Path of length {self._ecef.shape[0]} 
            with {self._xp.__name__} backend. \n 
            Start time: {self.time.datetime[0]}, 
            End time: {self.time.datetime[-1]} \n
            Start point (LLA): {self.geodet[0]},
            End point (LLA): {self.geodet[-1]}"""

    @backend_jit("method")
    def interpolate(self, time: Time, method: str = "linear") -> Point:
        """Interpolate the Path to the given times using the specified method.
        Currently only 'linear' interpolation is supported.
        """

        if method != "linear":
            raise ValueError("Currently only 'linear' interpolation is supported.")
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
            return Point._constructor(interp_ecef, time=time, backend=self._xp)


        # return Point(interp_ecef, time=time, backend=self._xp)




class Rotation:
    pass


class Frame:
    pass


@dataclass
class Pixels:
    pass
