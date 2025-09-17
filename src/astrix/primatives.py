# pyright: reportAny=false

from __future__ import annotations
from dataclasses import dataclass
import datetime as dt

from ._backend_utils import (
    resolve_backend,
    Array,
    ArrayNS,
    BackendArg,
)
from .utils import ensure_1d, ensure_2d, ecef2geodet, geodet2ecef


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
        self, secs: Array | list[float] | float = [], backend: BackendArg = None
    ) -> None:
        self._xp = resolve_backend(backend)
        self._secs = ensure_1d(secs)

    def is_in_bounds(self, sec: Time) -> bool:
        """Check if the given time(s) are within the bounds of this Time object."""
        return bool(
            (self._xp.min(sec.secs) >= self._xp.min(self._secs))
            & (self._xp.max(sec.secs) <= self._xp.max(self._secs))
        )

    @classmethod
    def from_datetime(
        cls, times: dt.datetime | list[dt.datetime], backend: BackendArg = None
    ) -> Time:
        """Create a Time object from a list of datetime objects. \
        Will not accept timezone-unaware datetime obejects due to ambiguity.
        """
        if isinstance(times, dt.datetime):
            times = [times]

        if not all(
            t.tzinfo is not None and t.tzinfo.utcoffset(t) is not None for t in times
        ):
            raise ValueError("All datetime objects must be timezone-aware")
        xp = resolve_backend(backend)
        secs = xp.asarray([t.timestamp() for t in times])
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
        return self._secs.copy()

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
    _times: Time | None = None

    def __init__(
        self, ecef: Array, time: Time | None = None, backend: BackendArg = None
    ) -> None:
        """Initialize a Point object with ECEF coordinates (x, y, z) in meters."""
        self._xp = resolve_backend(backend)
        self._ecef = ensure_2d(ecef, n=3)
        if time is not None:
            if ecef.shape[0] != time.secs.shape[0]:
                raise ValueError(
                    "Point and Time must be similar lengths if associated.\n"
                    + f"Found {ecef.shape[0]} points and {time.secs.shape[0]} times."
                )
            self._times = time

    @classmethod
    def from_geodet(cls, geodet: Array, time: Time | None = None, backend: BackendArg = None) -> Point:
        """Create a Point object from geodetic coordinates (lat, lon, alt).
        Lat and lon are in degrees, alt is in meters.
        """

        xp = resolve_backend(backend)
        geodet = ensure_2d(geodet, n=3)
        ecef = xp.asarray(geodet2ecef(geodet))
        return cls(ecef, time, backend=xp)

    @property
    def ecef(self) -> Array:
        """Get the ECEF coordinates (x, y, z) in meters."""
        return self._ecef.copy()

    @property
    def geodet(self) -> Array:
        """Convert to geodetic coordinates (lat [deg], lon [deg], alt [m])."""
        return self._xp.asarray(ecef2geodet(self.ecef))

    @property
    def backend(self) -> str:
        """Get the name of the array backend in use (e.g., 'numpy', 'jax')."""
        return self._xp.__name__

    def __getitem__(self, index: int) -> Point:
        return Point(self.ecef[index], backend=self._xp)

    def __repr__(self) -> str:
        return f"Point array of length {self._ecef.shape[0]} with {self._xp.__name__} backend. \n \
            First point (Geodet): {self.geodet[0]}, Last point (ECEF): {self.geodet[-1]}"

    def __len__(self) -> int:
        return self._ecef.shape[0]

    @property
    def has_time(self) -> bool:
        """Check if the Point has associated Time."""
        return self._times is not None

    @property
    def time(self) -> Time:
        """Get the associated Time object, if any."""
        if self._times is None:
            raise ValueError("This Point does not have associated Time.")
        return self._times


class Path:
    _secs: Array
    _ecef: Array
    _xp: ArrayNS

    def __init__(self, point: Point, backend: BackendArg) -> None:
        """Initialize a Path object from a Point object with associated Time."""
        if not point.has_time:
            raise ValueError("Point must have associated Time to create a Path.")
        self._xp = resolve_backend(backend)
        sort_indices = self._xp.argsort(point.time.secs)
        self._secs = ensure_1d(point.time.secs[sort_indices])
        self._ecef = ensure_2d(point.ecef[sort_indices])


class Rotation:
    pass


class Frame:
    pass


@dataclass
class Pixels:
    pass
