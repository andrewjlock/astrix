from __future__ import annotations
from dataclasses import dataclass
import datetime as dt

from ._backend_utils import resolve_backend, Array, ArrayNS, BackendArg
from .utils import ensure_1d, ensure_2d



@dataclass
class Time:
    """One or more time values. Stored as Unix timestamps (seconds since epoch),
    adjusted for leap seconds.
    """
    secs: Array
    xp: ArrayNS

    def __init__(self, secs: Array, backend: BackendArg = None):
        self.xp = resolve_backend(backend)
        self.secs = ensure_1d(secs, xp=self.xp)

    def is_in_bounds(self, sec: Time) -> bool:
        """Check if the given time(s) are within the bounds of this Time object."""
        return bool((self.xp.min(sec.secs) >= self.xp.min(self.secs)) & (
            self.xp.max(sec.secs) <= self.xp.max(self.secs)
        ))

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

    def to_datetime(self):
        """Convert to a list of datetime objects."""
        return [
            dt.datetime.fromtimestamp(float(s), tz=dt.timezone.utc) for s in self.secs
        ]

    def __getitem__(self, index: int) -> Time:
        return Time(self.secs[index], backend=self.xp)


@dataclass
class Point:
    def __init__(self, ecef: Array, backend: BackendArg = None):
        self.xp: ArrayNS = resolve_backend(backend)
        self.ecef: Array = ensure_2d(ecef, n=3, xp=self.xp)



class Path:
    pass


class Rotation:
    pass


class Frame:
    pass


@dataclass
class Pixels:
    pass
