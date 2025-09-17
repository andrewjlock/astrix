# pyright: reportAny=false

from __future__ import annotations
from dataclasses import dataclass
import datetime as dt

from ._backend_utils import (
    resolve_backend,
    Array,
    ArrayNS,
    BackendArg,
    warn_if_not_numpy,
)
from .utils import ensure_1d, ensure_2d, ecef2geodet, geodet2ecef


@dataclass
class Time:
    """One or more time values. Stored as Unix timestamps (seconds since epoch),
    adjusted for leap seconds.
    """

    _secs: Array
    _xp: ArrayNS

    def __init__(self, secs: Array, backend: BackendArg = None) -> None:
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
        cls, times: list[dt.datetime], backend: BackendArg = None
    ) -> Time:
        """Create a Time object from a list of datetime objects. \
        Will not accept timezone-unaware datetime obejects due to ambiguity.
        """

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


@dataclass
class Point:
    _ecef: Array
    _xp: ArrayNS

    def __init__(self, ecef: Array, backend: BackendArg = None) -> None:
        self._xp = resolve_backend(backend)
        self._ecef = ensure_2d(ecef, n=3)

    @classmethod
    def from_geodet(cls, geodet: Array, backend: BackendArg = None) -> Point:
        """Create a Point object from geodetic coordinates (lat, lon, alt).
        Lat and lon are in degrees, alt is in meters.
        """

        xp = resolve_backend(backend)
        geodet = ensure_2d(geodet, n=3)
        ecef = xp.asarray(geodet2ecef(geodet))
        return cls(ecef, backend=xp)

    @property
    def ecef(self) -> Array:
        """Get the ECEF coordinates (x, y, z) in meters."""
        return self._ecef.copy()

    @property
    def geodet(self) -> Array:
        """Convert to geodetic coordinates (lat [deg], lon [deg], alt [m])."""
        return self._xp.asarray(ecef2geodet(self.ecef))

    def __getitem__(self, index: int) -> Point:
        return Point(self.ecef[index], backend=self._xp)


class Path:
    pass


class Rotation:
    pass


class Frame:
    pass


@dataclass
class Pixels:
    pass
