# pyright: standard
# pyright: reportAny=false, reportImplicitOverride=false

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
from abc import ABC, abstractmethod
from typing import ClassVar

from ._backend_utils import (
    resolve_backend,
    Array,
    ArrayLike,
    ArrayNS,
    BackendArg,
)
from .functs import (
    ensure_1d,
    is_increasing,
)


class TimeLike(ABC):
    """Abstract base class for time-like objects (Time, TimeSequence).
    'in_bounds' function is required for integration with other modules.
    """

    @abstractmethod
    def __getitem__(self, index: int | slice) -> TimeLike:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    def is_singular(self) -> bool:
        """Check if the TimeLike object represents a single time instance.
        Override if not singular"""
        return len(self) == 1

    @abstractmethod
    def convert_to(self, backend: BackendArg) -> TimeLike:
        pass

    @abstractmethod
    def in_bounds(self, time: Time) -> bool:
        pass

    @abstractmethod
    def copy(self) -> TimeLike: ...


@dataclass(frozen=True)
class TimeInvariant(TimeLike):
    """Class for static time-like objects (static Time).
    'in_bounds' function is required for integration with other modules.
    """

    def __len__(self) -> int:
        return 1

    def __repr__(self) -> str:
        return "TimeInvariant object"

    def __getitem__(self, index: int | slice) -> TimeInvariant:
        return self

    def in_bounds(self, time: Time) -> bool:
        return True

    def convert_to(self, backend: BackendArg) -> TimeInvariant:
        return self

    def datetime(self) -> list[str]:
        return ["<Time Invariant Object>"]

    def copy(self) -> TimeInvariant:
        return self


TIME_INVARIANT = TimeInvariant()


@dataclass
class Time(TimeLike):
    """One or more time instances.

    Represents time using seconds since Unix epoch (1970-01-01 00:00:00 UTC).
    Can handle single time instances or arrays of times with consistent
    backend support for JAX/NumPy compatibility.

    Parameters
    ----------
    unix : Array | list of float | float
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
    >>> dt_list = [
    ...     datetime(2021, 1, 1, tzinfo=timezone.utc),
    ...     datetime(2021, 1, 2, tzinfo=timezone.utc),
    ... ]
    >>> times = Time.from_datetime(dt_list)

    Notes
    -----
    All datetime objects must be timezone-aware to avoid ambiguity.
    """

    _unix: Array
    _min: float | Array
    _max: float | Array
    _xp: ArrayNS
    _n: int
    _i: int = 0  # For compatibility with TimeSequence

    def __init__(
        self, unix: ArrayLike, backend: BackendArg = None
    ) -> None:
        self._xp = resolve_backend(backend)
        self._unix = ensure_1d(unix, backend=self._xp)
        self._min = self._xp.min(self._unix)
        self._max = self._xp.max(self._unix)
        self._n = self._unix.shape[0]

    # --- Constructors ---

    @classmethod
    def from_datetime(
        cls, time: dt.datetime | list[dt.datetime], backend: BackendArg = None
    ) -> Time:
        """Create a Time object from a single or list of datetime objects. \
        Will not accept timezone-unaware datetime obejects due to ambiguity.
        """

        if isinstance(time, dt.datetime):
            time = [time]

        if not all(
            t.tzinfo is not None and t.tzinfo.utcoffset(t) is not None for t in time
        ):
            raise ValueError("All datetime objects must be timezone-aware")
        xp = resolve_backend(backend)
        unix = xp.asarray([t.timestamp() for t in time])
        return cls(unix, backend=backend)

    @classmethod
    def _constructor(cls, unix: Array, xp: ArrayNS) -> Time:
        """Internal constructor to create a Time object from seconds array
        Avoids type checking in __init__."""

        obj = cls.__new__(cls)
        obj._xp = xp
        obj._unix = unix
        obj._min = obj._xp.min(unix)
        obj._max = obj._xp.max(unix)
        obj._n = unix.shape[0]
        return obj

    # --- Dunder methods and properties ---

    def __repr__(self) -> str:
        if len(self) == 1:
            return str(self.datetime[0])
        else:
            return f"Time array of length {len(self)} from {self.datetime[0]} to \
            {self.datetime[-1]} with {self._xp.__name__} backend."

    def __len__(self) -> int:
        return self._unix.shape[0]

    def __getitem__(self, index: int | slice) -> Time:
        return Time._constructor(
            self._xp.asarray(self.unix[index]).reshape(-1), xp=self._xp
        )

    def __iter__(self) -> Time:
        self._i = 0
        return self

    def __next__(self) -> Time:
        if self._i >= self._n:
            raise StopIteration
        value = self[self._i]
        self._i += 1
        return value

    def copy(self) -> Time:
        return Time._constructor(self.unix.copy(), xp=self._xp)

    @property
    def datetime(self) -> list[dt.datetime]:
        return [
            dt.datetime.fromtimestamp(float(s), tz=dt.timezone.utc) for s in self.unix
        ]

    @property
    def is_increasing(self) -> bool:
        """Check if the time values are strictly increasing."""
        return is_increasing(self._unix, backend=self._xp)

    @property
    def unix(self) -> Array:
        """Get the time values in seconds since epoch."""
        return self._unix

    @property
    def start_sec(self) -> float | Array:
        """Get the start time in seconds since epoch."""
        return self._min

    @property
    def end_sec(self) -> float | Array:
        """Get the end time in seconds since epoch."""
        return self._max

    @property
    def duration(self) -> float | Array:
        """Get the duration between the first and last time in seconds."""
        return self._max - self._min

    @property
    def backend(self) -> str:
        """Get the name of the array backend in use (e.g., 'numpy', 'jax.numpy')."""
        return self._xp.__name__

    # --- Methods ---

    def in_bounds(self, time: Time) -> bool:
        """Check if the given time(s) are within the bounds of this Time object."""
        return bool((time.start_sec >= self._min) & (time.end_sec <= self._max))

    def offset(self, offset: float) -> Time:
        """Return a new Time object with offset (seconds) added to all time values."""
        return Time(self.unix + offset, backend=self._xp)

    def _repeat_single(self, n: int) -> Time:
        """Private method to repeat a singular Time n times.
        Enables compatibility between static Times and dynamic TimeSequences in other modules."""

        if len(self) != 1:
            raise ValueError(
                "Attempting to repeat a non-singular Time object. \n"
                "This is not supported. Use TimeGroup objects for multiple times."
            )
        return Time._constructor(self._xp.repeat(self.unix, n), xp=self._xp)

    def return_in_bounds(self, time: Time) -> Time:
        """Return a new Time object containing only the times within the bounds of this Time object."""
        mask = (time.unix >= self._min) & (time.unix <= self._max)
        return Time._constructor(time.unix[mask], xp=time._xp)

    def nearest_idx(self, time: Time) -> Array:
        """Find the index of the nearest time in this Time object for each time in the input Time object.

        Parameters
        ----------
        time : Time
            Time object containing times to find nearest indices for.

        Returns
        -------
        Array
            Array of indices corresponding to the nearest times in this Time object.

        Examples
        --------

        >>> from astrix.primitives import Time
        >>> from datetime import datetime, timezone
        >>> t1 = Time.from_datetime([
        ...     datetime(2021, 1, 1, tzinfo=timezone.utc),
        ...     datetime(2021, 1, 2, tzinfo=timezone.utc),
        ...     datetime(2021, 1, 3, tzinfo=timezone.utc)
        ... ])
        >>> t2 = Time.from_datetime([
        ...     datetime(2021, 1, 1, 12, tzinfo=timezone.utc),
        ...     datetime(2021, 1, 2, 12, tzinfo=timezone.utc)
        ... ])
        >>> idx = t1.nearest_idx(t2)
        >>> idx.tolist()
        [0, 1]
        """

        idx = self._xp.abs(self.unix[:, None] - time.unix[None, :]).argmin(axis=0)
        return idx

    def convert_to(self, backend: BackendArg) -> Time:
        """Convert the Time object to a different backend."""
        xp = resolve_backend(backend)
        if xp == self._xp:
            return self
        return Time._constructor(xp.asarray(self.unix), xp=xp)


class TimeGroup:
    """A group of TimeLike objects (Time, TimeInvariant, TimeGroup).
    Used to manage multiple time instances and determine overlapping time bounds.

    Parameters
    ----------
    times : list of TimeLike
        List of TimeLike objects (Time, TimeInvariant, TimeGroup)
    backend : BackendArg, optional
        Array backend to use (numpy, jax, etc.). Defaults to numpy.

    Examples
    --------

    >>> t1 = Time.from_datetime(
    ...     [
    ...         datetime(2021, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    ...         datetime(2021, 1, 1, 13, 0, 0, tzinfo=timezone.utc),
    ...     ]
    ... )
    >>> t2 = Time.from_datetime(
    ...     [
    ...         datetime(2021, 1, 1, 12, 30, 0, tzinfo=timezone.utc),
    ...         datetime(2021, 1, 1, 14, 0, 0, tzinfo=timezone.utc),
    ...     ]
    ... )
    >>> tg = TimeGroup([t1, t2])
    >>> tg.duration  # Duration of overlap in seconds
    1800.0

    >>> overlap = tg.overlap_bounds  # Overlapping time range
    >>> assert overlap[0].datetime[0] == datetime(
    ...     2021, 1, 1, 12, 30, 0, tzinfo=timezone.utc
    ... )
    >>> assert overlap[1].datetime[0] == datetime(
    ...     2021, 1, 1, 13, 0, 0, tzinfo=timezone.utc
    ... )

    >>> tg.in_bounds(
    ...     Time.from_datetime(datetime(2021, 1, 1, 12, 45, 0, tzinfo=timezone.utc))
    ... )
    True
    """

    _times: list[TimeLike]
    _invariant: bool
    _xp: ArrayNS
    _overlap_bounds: tuple[float | Array, float | Array]
    _extreme_bounds: tuple[float | Array, float | Array]
    _duration: float | Array

    def __init__(
        self, times: list[TimeLike | TimeGroup], backend: BackendArg = None
    ) -> None:
        if not all(isinstance(t, TimeLike | TimeGroup) for t in times):
            raise ValueError("All elements of times must be TimeLike objects")
        if len(times) == 0:
            raise ValueError("times list cannot be empty")

        self._xp = resolve_backend(backend)
        self._times = []

        mins = []
        maxs = []
        for t in times:
            if isinstance(t, TimeInvariant):
                self._times.append(t)
            if isinstance(t, Time):
                mins.append(self._xp.min(t.unix))
                maxs.append(self._xp.max(t.unix))
                self._times.append(t.convert_to(self._xp))
            if isinstance(t, TimeGroup):
                if not t.is_invariant:
                    mins.append(t._overlap_bounds[0])
                    maxs.append(t._overlap_bounds[1])
                    mins.append(t._extreme_bounds[0])
                    maxs.append(t._extreme_bounds[1])
                self._times += [tt.convert_to(self._xp) for tt in t.times]

        if len(mins) == 0 or len(maxs) == 0:
            self._overlap_bounds = (-self._xp.inf, self._xp.inf)
            self._extreme_bounds = (-self._xp.inf, self._xp.inf)
            self._invariant = True
        else:
            self._invariant = False
            self._overlap_bounds = (
                self._xp.max(self._xp.array(mins)),
                self._xp.min(self._xp.array(maxs)),
            )
            self._extreme_bounds = (
                self._xp.min(self._xp.array(mins)),
                self._xp.max(self._xp.array(maxs)),
            )
        self._duration = self._overlap_bounds[1] - self._overlap_bounds[0]

    # --- Dunder methods and properties ---

    def __str__(self) -> str:
        if self.is_invariant:
            return "TimeGroup: Time invariant"
        else:
            return f"TimeGroup with {len(self)} TimeLike objects"

    def __repr__(self) -> str:
        if self.is_invariant:
            return "<Time Group: Time invariant>"
        else:
            return (
                f"TimeGroup with {len(self)} TimeLike objects \n"
                f"Overlap bounds: {self.overlap_bounds},  \n"
                f"Extreme bounds: {self.extreme_bounds},  \n"
                f"backend: {self._xp.__name__}"
            )

    def __getitem__(self, index: int) -> TimeLike:
        return self._times[index]

    def __len__(self) -> int:
        return len(self._times)


    @property
    def backend(self) -> str:
        return self._xp.__name__

    @property
    def is_invariant(self) -> bool:
        return self._invariant

    @property
    def times(self) -> tuple[TimeLike, ...]:
        return tuple(self._times)

    @property
    def duration(self) -> float | Array:
        """Get the duration of the overlap bounds in seconds."""
        return self._duration

    @property
    def overlap_bounds(self) -> tuple[TimeLike, TimeLike]:
        """Get the overlap bounds of the TimeGroup as Time objects."""
        if self.is_invariant:
            return (TIME_INVARIANT, TIME_INVARIANT)
        else:
            return (
                Time._constructor(
                    self._xp.asarray(self._overlap_bounds[0]).reshape(1), xp=self._xp
                ),
                Time._constructor(
                    self._xp.asarray(self._overlap_bounds[1]).reshape(1), xp=self._xp
                ),
            )

    @property
    def extreme_bounds(self) -> tuple[TimeLike, TimeLike]:
        """Get the extreme bounds of the TimeGroup as Time objects."""
        if self.is_invariant:
            return (TIME_INVARIANT, TIME_INVARIANT)
        else:
            return (
                Time._constructor(
                    self._xp.asarray(self._extreme_bounds[0]).reshape(-1), xp=self._xp
                ),
                Time._constructor(
                    self._xp.asarray(self._extreme_bounds[1]).reshape(-1), xp=self._xp
                ),
            )

    # --- Methods ---

    def in_bounds(self, time: Time) -> bool:
        """Check if the given time(s) are within the overlap bounds of this TimeGroup."""
        return bool(
            (time.start_sec >= self._overlap_bounds[0])
            & (time.end_sec <= self._overlap_bounds[1])
        )

    def convert_to(self, backend: BackendArg) -> TimeGroup:
        """Convert the TimeGroup object to a different backend."""
        xp = resolve_backend(backend)
        if xp == self._xp:
            return self
        times_converted: list[TimeLike | TimeGroup] = [
            t.convert_to(xp) for t in self._times
        ]
        return TimeGroup(times_converted, backend=xp)


def time_linspace(t1: Time, t2: Time, num: int) -> Time:
    """Create a Time object with linearly spaced times between two Time objects.
    If t1 and t2 are non-singular, uses t1[0] and t2[-1].

    Parameters
    ----------
    t1 : Time
        Start time.
    t2 : Time
        End time.
    num : int
        Number of time points to generate.

    Returns
    -------
    Time
        Time object with linearly spaced times.

    Examples
    --------

    >>> from astrix.primitives import Time
    >>> from datetime import datetime, timezone
    >>> t_start = Time.from_datetime(datetime(2021, 1, 1, tzinfo=timezone.utc))
    >>> t_end = Time.from_datetime(datetime(2021, 1, 2, tzinfo=timezone.utc))
    >>> t_lin = time_linspace(t_start, t_end, num=5)
    >>> t_lin.datetime
    [datetime.datetime(2021, 1, 1, 0, 0, tzinfo=datetime.timezone.utc),
     datetime.datetime(2021, 1, 1, 6, 0, tzinfo=datetime.timezone.utc),
     datetime.datetime(2021, 1, 1, 12, 0, tzinfo=datetime.timezone.utc),
     datetime.datetime(2021, 1, 1, 18, 0, tzinfo=datetime.timezone.utc),
     datetime.datetime(2021, 1, 2, 0, 0, tzinfo=datetime.timezone.utc)]
    """

    if not isinstance(t1, Time) or not isinstance(t2, Time):
        raise ValueError("t1 and t2 must be Time objects")
    if t1.backend != t2.backend:
        raise ValueError("t1 and t2 must have the same backend")
    if num < 2:
        raise ValueError("num must be at least 2")

    xp = resolve_backend(t1.backend)
    unix = xp.linspace(float(t1.start_sec), float(t2.end_sec), num=num)
    return Time._constructor(unix=unix, xp=xp)
