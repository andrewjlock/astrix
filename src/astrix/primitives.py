# pyright: standard
# pyright: reportAny=false, reportImplicitOverride=false

from __future__ import annotations
import warnings

from dataclasses import dataclass
import datetime as dt
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Callable
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
from .functs import (
    interp_nd,
    central_diff,
    finite_diff_2pt,
    interp_haversine,
    ned_rotation,
)


class TimeLike(ABC):
    """Abstract base class for time-like objects (Time, TimeSequence).
    'in_bounds' function is required for integration with other modules.
    """

    @abstractmethod
    def in_bounds(self, time: Time) -> bool:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> TimeLike:
        pass

    @abstractmethod
    def convert_to(self, backend: BackendArg) -> TimeLike:
        pass


@dataclass(frozen=True)
class TimeInvariant(TimeLike):
    """Class for static time-like objects (static Time).
    'in_bounds' function is required for integration with other modules.
    """

    def in_bounds(self, time: Time) -> bool:
        return True

    def __len__(self) -> int:
        return 0

    def __repr__(self) -> str:
        return "TimeInvariant object"

    def __getitem__(self, index: int) -> TimeInvariant:
        return self

    def convert_to(self, backend: BackendArg) -> TimeInvariant:
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
    >>> dt_list = [
    ...     datetime(2021, 1, 1, tzinfo=timezone.utc),
    ...     datetime(2021, 1, 2, tzinfo=timezone.utc),
    ... ]
    >>> times = Time.from_datetime(dt_list)

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

    # @backend_jit()
    def in_bounds(self, time: Time) -> bool:
        """Check if the given time(s) are within the bounds of this Time object."""
        return bool((time.start_sec >= self._min) & (time.end_sec <= self._max))

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
        return self._secs

    @property
    def backend(self) -> str:
        """Get the name of the array backend in use (e.g., 'numpy', 'jax.numpy')."""
        return self._xp.__name__

    def __repr__(self) -> str:
        if len(self) == 1:
            return str(self.datetime[0])
        else:
            return f"Time array of length {len(self)} from {self.datetime[0]} to \
            {self.datetime[-1]} with {self._xp.__name__} backend."

    def __len__(self) -> int:
        return self._secs.shape[0]

    def offset(self, offset: float) -> Time:
        return Time(self.secs + offset, backend=self._xp)

    @property
    def start_sec(self) -> float | Array:
        """Get the start time in seconds since epoch."""
        return self._min

    @property
    def end_sec(self) -> float | Array:
        """Get the end time in seconds since epoch."""
        return self._max


class TimeGroup(TimeLike):
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

    def __init__(self, times: list[TimeLike], backend: BackendArg = None) -> None:
        if not all(isinstance(t, TimeLike) for t in times):
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
                mins.append(self._xp.min(t.secs))
                maxs.append(self._xp.max(t.secs))
                self._times.append(t.convert_to(self._xp))
            if isinstance(t, TimeGroup):
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

    @property
    def backend(self) -> str:
        return self._xp.__name__

    @property
    def is_invariant(self) -> bool:
        return self._invariant

    @property
    def times(self) -> tuple[TimeLike, ...]:
        return tuple(self._times)

    def in_bounds(self, time: Time) -> bool:
        """Check if the given time(s) are within the overlap bounds of this TimeGroup."""
        return bool(
            (time.start_sec >= self._overlap_bounds[0])
            & (time.end_sec <= self._overlap_bounds[1])
        )

    @property
    def duration(self) -> float | Array:
        """Get the duration of the overlap bounds in seconds."""
        return self._duration

    def convert_to(self, backend: BackendArg) -> TimeGroup:
        """Convert the TimeGroup object to a different backend."""
        xp = resolve_backend(backend)
        if xp == self._xp:
            return self
        times_converted = [t.convert_to(xp) for t in self._times]
        return TimeGroup(times_converted, backend=xp)

    def __getitem__(self, index: int) -> TimeLike:
        return self._times[index]

    def __len__(self) -> int:
        return len(self._times)

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


T = TypeVar("T", bound=TimeLike, covariant=True)


class Location(Generic[T], ABC):
    """Abstract base class for location objects (Point, Path).
    'interp' function is required for integration with other modules.
    """

    _ecef: Array
    _time: T
    _xp: ArrayNS

    @property
    def ecef(self) -> Array:
        return self._ecef

    @property
    def geodet(self) -> Array:
        return self._xp.asarray(ecef2geodet(self._ecef))

    def __len__(self) -> int:
        return self._ecef.shape[0]

    @property
    def backend(self) -> str:
        return self._xp.__name__

    @property
    def time(self) -> T:
        return self._time

    @abstractmethod
    def _interp(self, time: Time) -> Point:
        pass

    @abstractmethod
    def convert_to(self, backend: BackendArg) -> Location:
        pass

    # @property
    # def is_singular(self) -> bool:
    #     """Check if the Location object represents a single point.
    #     Overridden if True in Point class."""
    #     return False


@dataclass
class Point(Location[TimeLike]):
    """
    Point(s) in ECEF coordinates, stored as (x, y, z) in metres.
    Can represent a single point or multiple points, and can be associated with
    a Time object for time instances of the points.

    Parameters
    ----------
    ecef : Array
        ECEF coordinates as (x, y, z) in metres. Shape (3,) or (1,3) for single points, (n, 3) for multiple points.
    time : TimeLike, optional
        Time object associated with the points. If provided, the length of time must match the number of points.
        Defaults to TIME_INVARIANT for static points.
    backend : BackendArg, optional
        Array backend to use (numpy, jax, etc.). Defaults to numpy.

    Examples
    --------

    Single static point:

    >>> from astrix.primitives import Point
    >>> p1 = Point(
    ...     [-5047162.4, 2568329.79, -2924521.17]
    ... )  # ECEF coordinates of Brisbane in metres
    >>> p.geodet  # Convert to geodetic coordinates (lat, lon, alt)
    array([[153.03, 27.47, 0.0]])
    >>> p2 = Point.from_geodet([27.47, 153.03, 0])  # lat, lon in degrees, alt in metres
    >>> p2.ecef  # Convert back to ECEF coordinates
    array([[-5047162.4, 2568329.79, -2924521.17]])

    Multiple static points:

    >>> pts = Point(
    ...     [
    ...         [-5047162.4, 2568329.79, -2924521.17],  # Brisbane
    ...         [-2694045.0, -4293642.0, 3857878.0],  # San Francisco
    ...         [3877000.0, 350000.0, 5027000.0],  # Somewhere else
    ...     ]
    ... )
    >>> pt_bris = pts[0]  # First point (Brisbane)
    >>> assert len(pts) == 3

    Dynamic point with time:

    >>> from datetime import datetime, timezone
    >>> from astrix.primitives import Time
    >>> times = Time.from_datetime(
    ...     [
    ...         datetime(2021, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    ...         datetime(2021, 1, 1, 13, 0, 0, tzinfo=timezone.utc),
    ...         datetime(2021, 1, 1, 14, 0, 0, tzinfo=timezone.utc),
    ...     ]
    ... )
    >>> pts_time = Point(
    ...     [
    ...         [-5047162.4, 2568329.79, -2924521.17],  # Brisbane
    ...         [-2694045.0, -4293642.0, 3857878.0],  # San Francisco
    ...         [3877000.0, 350000.0, 5027000.0],  # Somewhere else
    ...     ],
    ...     time=times,
    ... )
    >>> pts.has_time
    True
    >>> pts.is_singular
    False
    >>> pts_new = pts + Point(
    ...     [[-1000, -1000, -1000]],
    ...     time=Time.from_datetime(
    ...         datetime(2021, 1, 1, 15, 0, 0, tzinfo=timezone.utc)
    ...     ),
    ... )
    >>> assert len(pts_new) == 4

    Notes
    -----
    - When associating a Time object, the length of the Time must match the number of points.
    - Use Path objects for interpolating between multiple points over time.
    """

    def __init__(
        self, ecef: Array, time: TimeLike = TIME_INVARIANT, backend: BackendArg = None
    ) -> None:
        """Initialize a Point object with ECEF coordinates (x, y, z) in meters."""
        self._xp = resolve_backend(backend)
        self._ecef = ensure_2d(ecef, n=3, backend=self._xp)
        if isinstance(time, Time):
            time = time.convert_to(self._xp)
            if self._ecef.shape[0] != len(time):
                raise ValueError(
                    "Point and Time must be similar lengths if associated.\n"
                    + f"Found {self._ecef.shape[0]} points and {time.secs.shape[0]} times."
                )
        self._time = time

    @classmethod
    def _constructor(cls, ecef: Array, time: TimeLike, xp: ArrayNS) -> Point:
        """Internal constructor to create a Point object from ECEF array
        Avoids type checking in __init__."""

        obj = cls.__new__(cls)
        obj._xp = xp
        obj._ecef = ecef
        obj._time = time
        return obj

    @classmethod
    def from_geodet(
        cls, geodet: Array, time: TimeLike = TIME_INVARIANT, backend: BackendArg = None
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

        xp = resolve_backend(points[0].backend)
        if not all(p.backend == points[0].backend for p in points):
            raise ValueError("All points must have the same backend.")

        time_types = [type(p.time) for p in points]
        if not all(t == time_types[0] for t in time_types):
            raise ValueError("All points must either have time or not have time.")
        if time_types[0] is Time:
            time_joined = Time(
                xp.concatenate([p.time.secs for p in points]),  # pyright: ignore
                backend=xp,
            )
        else:
            time_joined = TIME_INVARIANT

        ecef_joined = xp.concatenate([p.ecef for p in points]).reshape(-1, 3)
        return cls(ecef_joined, time=time_joined, backend=xp)

    @property
    def is_singular(self) -> bool:
        """Check if the Point object represents a single point."""
        return self._ecef.shape[0] == 1

    @property
    def has_time(self) -> bool:
        """Check if the Point has associated Time."""
        return isinstance(self.time, Time)

    def convert_to(self, backend: BackendArg) -> Point:
        """Convert the Point object to a different backend."""
        xp = resolve_backend(backend)
        if xp == self._xp:
            return self
        if isinstance(self._time, Time):
            time_converted = self._time.convert_to(xp)
        else:
            time_converted = TIME_INVARIANT
        return Point._constructor(xp.asarray(self.ecef), time_converted, xp)

    def __getitem__(self, index: int) -> Point:
        return Point(self.ecef[index], backend=self._xp)

    def __repr__(self) -> str:
        return f"""Point of length {self._ecef.shape[0]} 
            with {self._xp.__name__} backend. \n 
            First point (LLA): {self.geodet[0]}, 
            Last point (LLA): {self.geodet[-1]}"""

    def __str__(self) -> str:
        return f"Point, n=<Select> {len(self)}, backend='{self._xp.__name__}'"

    def __add__(self, added_point: Point) -> Point:
        if self.has_time != added_point.has_time:
            raise ValueError("Cannot combine points with and without time.")
        if self.backend != added_point.backend:
            raise ValueError("Cannot combine points with different backends.")

        ecef_joined = self._xp.concatenate([self.ecef, added_point.ecef])
        if isinstance(self.time, Time) and isinstance(added_point.time, Time):
            time_joined = Time(
                self._xp.concatenate(
                    [
                        self.time.secs,
                        added_point.time.secs,
                    ]
                )
            )
        else:
            if any(isinstance(t, Time) for t in [self.time, added_point.time]):
                raise ValueError("Cannot combine points with and without time.")
            time_joined = TIME_INVARIANT
        return Point(ecef_joined, time=time_joined, backend=self._xp)

    def _interp(self, time: Time, check_bounds: bool = True) -> Point:
        """Private method to 'interpolate' (broadcast) a singular Point to multiple times.
        Enables compatibility between static Points and dynamic Paths in other modules."""

        if not self.is_singular:
            raise ValueError(
                "Attempting to 'interpolate' (broadcast) a non-singular Point object. \n"
                "This is not supported. Use Path objects for interpolation between multiple points."
            )
        return Point._constructor(
            self._xp.repeat(self.ecef, len(time), axis=0),
            Time._constructor(self._xp.asarray(time.secs).reshape(-1), xp=self._xp),
            self._xp,
        )


@dataclass(frozen=True)
class Velocity:
    """
    Velocity vector(s) in ECEF coordinates (vx, vy, vz) in m/s.
    Associated with a Time object for the time instances of the velocities.
    Internal use only, typically created from Path objects.
    No data validation is performed.

    Parameters
    ----------
    vec : Array
        Velocity vectors in ECEF coordinates (vx, vy, vz) in m/s. Shape (n, 3).
    time : Time
        Time object associated with the velocities. Length must match number of velocity vectors.
    backend : BackendArg, optional
        Array backend to use (numpy, jax, etc.). Defaults to numpy.

    Examples
    --------

    Velocity objects are typically created from Path objects.

    >>> from astrix.primitives import Point, Time, Path
    >>> from datetime import datetime, timezone
    >>> times = Time.from_datetime(
    ...     [
    ...         datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    ...         datetime(2025, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
    ...         datetime(2025, 1, 1, 12, 0, 2, tzinfo=timezone.utc),
    ...     ]
    ... )
    >>> path = Path(
    ...     [
    ...         Point([1, 2, 0], time=times[0]),
    ...         Point([2, 3.8, 0.4], time=times[1]),
    ...         Point([3, 6.0, 1], time=times[2]),
    ...     ]
    ... )  # Somewhere very hot in the middle of the Earth
    >>> vel = path.vel
    >>> vel.magnitude  # Velocity magnitudes in m/s
    array([1.91049732, 2.29128785, 2.6925824])
    >>> vel.unit  # Unit velocity vectors
    array([[0.52342392, 0.83747828, 0.15702718],
           [0.43643578, 0.87287156, 0.21821789],
           [0.37139068, 0.89133762, 0.25997347]])

    """

    vec: Array
    time: TimeLike
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


class Path(Location[Time]):
    """
    Path of multiple Point objects with associated Time.
    Enables interpolation between points over time and calculation of velocity.
    Must have at least 2 points with associated Time.

    Parameters
    ----------
    point : Point | list of Point
        Point object or list of Point objects with associated Time.
        If a list is provided, all Points must have the same backend and associated Time.
    backend : BackendArg, optional
        Array backend to use (numpy, jax, etc.). Defaults to numpy.

    Examples
    --------

    Instantiating a Path from a list of Points:

    >>> from astrix.primitives import Point, Time, Path
    >>> from datetime import datetime, timezone
    >>> times = Time.from_datetime(
    ...     [
    ...         datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    ...         datetime(2025, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
    ...         datetime(2025, 1, 1, 12, 0, 2, tzinfo=timezone.utc),
    ...     ]
    ... )
    >>> path = Path(
    ...     [
    ...         Point([1, 2, 0], time=times[0]),
    ...         Point([2, 3.8, 0.4], time=times[1]),
    ...         Point([3, 6.0, 1], time=times[2]),
    ...     ]
    ... )  # Somewhere very hot in the middle of the Earth

    Interpolate the Path to a new time and get velocity:

    >>> path.interp(
            Time.from_datetime(datetime(2025, 1, 1, 12, 0, 1, 500000, tzinfo=timezone.utc)),
            method="linear"
        ).ecef # Interpolate to halfway between second and third point, return ECEF array
    array([[2.5, 4.9, 0.7]])
    >>> vel = path.interp_vel(
            Time.from_datetime(datetime(2025, 1, 1, 12, 0, 1, 500000, tzinfo=timezone.utc)),
        )
    >>> vel.magnitude  # Interpolated velocity magnitude in m/s
    array([2.48997992])
    >>> vel.unit  # Interpolated unit velocity vector
    array([[0.40160966, 0.88354126, 0.2409658 ]])
    """

    _vel: Array

    def __init__(self, point: Point | list[Point], backend: BackendArg = None) -> None:
        """Initialize a Path object from a Point object with associated Time."""
        if isinstance(point, list):
            point = Point.from_list(point)
        if not isinstance(point.time, Time):
            raise ValueError("Point must have associated Time to create a Path.")
        self._xp = resolve_backend(backend)
        sort_indices = self._xp.argsort(point.time.secs)
        self._time = Time(point.time.secs[sort_indices], backend=self._xp)
        self._ecef = ensure_2d(point.ecef[sort_indices], n=3, backend=self._xp)

        if len(self.time) > 2:
            self._vel = central_diff(self._time.secs, self._ecef, backend=self._xp)
        elif len(self.time) == 2:
            self._vel = finite_diff_2pt(self._time.secs, self._ecef, backend=self._xp)
        else:
            raise ValueError("Path must have at least 2 points with associated Time.")

    @property
    def start_time(self) -> TimeLike:
        """Get the start time of the Path."""
        return self.time[0]

    @property
    def end_time(self) -> TimeLike:
        """Get the end time of the Path."""
        return self.time[-1]

    @property
    def vel(self) -> Velocity:
        """Get the Velocity object associated with the Path."""
        return Velocity(self._vel, self._time, self._xp)

    def __repr__(self) -> str:
        return f"""Path of length {self._ecef.shape[0]} 
            with {self._xp.__name__} backend. \n 
            Start time: {self.time[0]}, 
            End time: {self.time[1]} \n
            Start point (LLA): {self.geodet[0]},
            End point (LLA): {self.geodet[-1]}"""

    def __str__(self) -> str:
        return f"Path, n={len(self)}, backend='{self._xp.__name__}')"

    def convert_to(self, backend: BackendArg) -> Path:
        """Convert the Path object to a different backend."""
        xp = resolve_backend(backend)
        if xp == self._xp:
            return self
        return Path(
            Point(self.ecef, time=self.time.convert_to(xp), backend=xp), backend=xp
        )

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
            if not self.time.in_bounds(time):
                warnings.warn(f"""Path interpolation times are out of bounds.
                    Path time range: {self.time[0]} to {self.time[-1]}
                    Interpolation time range: {time[0]} to {time[-1]}
                    Extrapolation is not supported and will raise an error.""")
        time = time.convert_to(self._xp)
        return self._interp(time, method=method)

    # @backend_jit(["check_bounds"])
    def _interp(self, time: Time, method: str = "linear") -> Point:
        """Private method to interpolate the Path to the given times using the specified method.
        Avoids type and bounds checking in public interp()."""

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

    # @backend_jit(["method", "check_bounds"])
    def interp_vel(
        self, time: Time, method: str = "linear", check_bounds: bool = True
    ) -> Velocity:
        """Interpolate the Path velocity to the given times using the specified method.
        Currently only 'linear' interpolation is supported.
        """

        if method != "linear":
            raise ValueError("Currently only 'linear' interpolation is supported.")
        if check_bounds:
            if not self.time.in_bounds(time):
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


class RotationLike(ABC):
    """Abstract base class for rotation objects (RotationSingle, RotationSequence).
    'convert_to' function is required for integration with other modules.
    """

    _rot: Rotation
    _xp: ArrayNS

    @abstractmethod
    def convert_to(self, backend: BackendArg) -> RotationLike:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    @abstractmethod
    def time(self) -> TimeLike:
        pass

    @abstractmethod
    def interp(self, time: Time) -> Rotation:
        pass

    @abstractmethod
    def _interp_secs(self, secs: Array) -> Rotation:
        pass

    def backend(self) -> str:
        return self._xp.__name__

    def __str__(self) -> str:
        return f"{self.__class__.__name__} of length {len(self)} with {self._xp.__name__} backend."


class _RotationStatic(RotationLike):
    """A single, static rotation. Intended for internal use only."""

    _rot: Rotation
    _xp: ArrayNS

    def __init__(self, rot: Rotation, backend: BackendArg = None) -> None:
        self._xp = resolve_backend(backend)
        self._rot = _convert_rot_backend(rot, self._xp)
        if not self._rot.single:
            raise ValueError(
                "RotationSingle must be initialized with a single rotation"
            )

    def convert_to(self, backend: BackendArg) -> _RotationStatic:
        """Convert the RotationSingle object to a different backend."""
        xp = resolve_backend(backend)
        if xp == self._xp:
            return self
        return _RotationStatic(self._rot, xp)

    @property
    def time(self) -> TimeInvariant:
        """Get the Time object associated with the rotation (always static)."""
        return TIME_INVARIANT

    def __len__(self) -> int:
        return 1

    def interp(self, time: Time) -> Rotation:
        """Interpolate the rotation at the given times (always returns the same rotation)."""
        return Rotation._from_raw_quat(  # pyright: ignore
            self._xp.repeat(self._rot._quat, len(time), axis=0),  # pyright: ignore[reportAttributeAccessIssue]
            xp=self._xp,
        )

    def _interp_secs(self, secs: Array) -> Rotation:
        return Rotation._from_raw_quat(  # pyright: ignore
            self._xp.repeat(self._rot._quat, len(secs), axis=0),  # pyright: ignore[reportAttributeAccessIssue]
            xp=self._xp,
        )


class RotationSequence(RotationLike):
    """A sequence of time-tagged rotations, enabling interpolation between them.
    Uses scipy.spatial.transform.Slerp for interpolation.

    Parameters
    ----------
    rot : Rotation | list of Rotation
        A scipy Rotation object containing multiple rotations, or a list of such objects.
        If a list is provided, all elements must be scipy Rotation objects.
    time : Time
        A Time object with time instances corresponding to each rotation.
        Must be the same length as the number of rotations and strictly increasing.
    backend : BackendArg, optional
        Array backend to use (numpy, jax, etc.). Defaults to numpy.

    Examples
    --------

    >>> from astrix.primitives import Time, RotationSequence
    >>> from scipy.spatial.transform import Rotation
    >>> from datetime import datetime, timezone
    >>> times = Time.from_datetime(
    ...     [
    ...         datetime(2021, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    ...         datetime(2021, 1, 1, 13, 0, 0, tzinfo=timezone.utc),
    ...         datetime(2021, 1, 1, 14, 0, 0, tzinfo=timezone.utc),
    ...     ]
    ... )
    >>> rots = Rotation.from_euler(
    ...     "xyz",
    ...     [
    ...         [0, 0, 0],
    ...         [90, 0, 0],
    ...         [180, 0, 0],
    ...     ],
    ...     degrees=True,
    ... )
    >>> rot_seq = RotationSequence(rots, times)

    >>> interp_rot = rot_seq.interp(
    ...     Time.from_datetime(datetime(2021, 1, 1, 12, 30, 0, tzinfo=timezone.utc))
    ... )  # Interpolate to halfway between first and second rotation
    >>> interp_rot.as_euler(
    ...     "xyz", degrees=True
    ... )  # Get interpolated rotation as Euler angles
    array([[45.,  0.,  0.]])
    """

    _rot: Rotation
    _slerp: Slerp
    _time: Time
    _xp: ArrayNS

    def __init__(
        self,
        rot: Rotation | list[Rotation],
        time: Time,
        backend: BackendArg = None,
    ) -> None:
        self._xp = resolve_backend(backend)

        if isinstance(rot, list):
            if not all(isinstance(r, Rotation) for r in rot):
                raise ValueError(
                    "All elements of rot list must be scipy Rotation objects"
                )
            if len(rot) == 0:
                raise ValueError("rot list cannot be empty")
            rot = Rotation.concatenate(rot)
        self._rot = _convert_rot_backend(rot, self._xp)

        if len(time) != len(self._rot):
            raise ValueError(
                "Time and rotations must have same length for RotationSequence"
            )
        if not time.is_increasing:
            raise ValueError(
                "Time values must be strictly increasing to construct RotationSequence"
            )
        self._time = time.convert_to(self._xp)
        self._slerp = Slerp(self._time.secs, self._rot)

    def convert_to(self, backend: BackendArg) -> RotationSequence:
        """Convert the RotationSequence object to a different backend."""
        xp = resolve_backend(backend)
        if xp == self._xp:
            return self
        return RotationSequence(self._rot, self._time, xp)

    @property
    def time(self) -> Time:
        """Get the Time object associated with the rotation sequence."""
        return self._time

    def __len__(self) -> int:
        return len(self._rot)

    def interp(self, time: Time, check_bounds: bool = True) -> Rotation:
        """Interpolate the rotation sequence at the given times to return Rotation(s)."""
        time = time.convert_to(self._xp)
        if check_bounds:
            if not self._time.in_bounds(time):
                warnings.warn(f"""RotationSequence interpolation times are out of bounds.
                    RotationSequence time range: {self._time[0]} to {self._time[-1]}
                    Interpolation time range: {time[0]} to {time[-1]}
                    Extrapolation is not supported and will raise an error.""")
        return self._slerp(time.secs)

    def _interp_secs(self, secs: Array) -> Rotation:
        return self._slerp(secs)


class Frame:
    """ A reference frame defined by a rotation and location.
    Can be static or time-varying, and can have rotation defined relative to another Frame.
    Combines RotationLike and Location objects, and manages time associations.

    Parameters
    ----------
    rot : Rotation | RotationSequence
        A scipy Rotation object (single rotation) or RotationSequence (time-tagged rotations).
        If a single Rotation is provided, the frame rotation is static.
    loc : Location, optional
        A Location object (Point or Path) defining the frame origin in ECEF coordinates.
        If not provided, the frame origin is assumed to be at the origin of the reference frame.
        If loc is provided, it must be a singular Point (1x3) for static frames.
        Use Path objects for time-varying locations.
    ref_frame : Frame, optional
        A reference Frame object to define the rotation relative to.
        If not provided, the rotation is assumed to be absolute (e.g., from ECEF frame).
    backend : BackendArg, optional
        Array backend to use (numpy, jax, etc.). Defaults to numpy.

    Examples
    --------

    Static frame with static rotation and location:

    >>> from astrix.primitives import Frame, Point
    >>> from scipy.spatial.transform import Rotation
    >>>
    >>> rot = Rotation.from_euler('xyz', [90, 0, 0], degrees=True)  # 90 degree rotation about x-axis
    >>> loc = Point.from_geodet([27.47, 153.03, 0])  # Brisbane location
    >>> frame_static = Frame(rot, loc) # Frame with static rotation and location

    >>> frame_static.interp_rot().as_euler('xyz', degrees=True)  # Get absolute rotation
    array([[90.,  0.,  0.]])
    >>> frame_static.loc.geodet  # Get frame location in geodetic coordinates
    array([[153.03, 27.47, 0.0]])

    Time-varying frame with rotation sequence and static location:

    >>> from astrix.primitives import Time
    >>> from datetime import datetime, timezone
    >>>
    >>> times = Time.from_datetime(
    ...     [
    ...         datetime(2021, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    ...         datetime(2021, 1, 1, 13, 0, 0, tzinfo=timezone.utc),
    ...         datetime(2021, 1, 1, 14, 0, 0, tzinfo=timezone.utc),
    ...     ]
    ... )
    >>> rots = Rotation.from_euler(
    ...     "xyz",
    ...     [
    ...         [0, 0, 0],
    ...         [90, 0, 0],
    ...         [180, 0, 0],
    ...     ],
    ...     degrees=True,
    ... )
    >>> rot_seq = RotationSequence(rots, times)
    >>> loc = Point.from_geodet([27.47, 153.03, 0])  # Brisbane location
    >>> frame_dynamic_rot = Frame(rot_seq, loc) # Frame with time-varying rotation and static location

    >>> interp_rot = frame_dynamic_rot.interp_rot(
    ...     Time.from_datetime(datetime(2021, 1, 1, 12, 30, 0, tzinfo=timezone.utc))
    ... )  # Interpolate to halfway between first and second rotation
    >>> interp_rot.as_euler(
    ...     "xyz", degrees=True
    ... )  # Get interpolated absolute rotation as Euler angles
    array([[45.,  0.,  0.]])
    >>> frame_dynamic_rot.loc.geodet  # Get frame location in geodetic coordinates
    array([[153.03, 27.47, 0.0]])

    Frame defined relative to another frame:

    >>> rot_ref = Rotation.from_euler('xyz', [0, 30, 0], degrees=True)  # Reference frame
    >>> frame_ref = Frame(rot_ref, loc)  # Reference frame
    >>> rot_rel = Rotation.from_euler('xyz', [0, 40, 0], degrees=True)
    >>> frame = Frame(rot_rel, ref_frame=frame_ref) 
    >>>
    >>> frame.interp_rot().as_euler('xyz', degrees=True)  # Absolute rotation (rot_ref * rot_rel)
    array([[ 0., 70.,  0.]])
    >>> frame.loc.geodet  # (Same as reference frame)
    array([[153.03, 27.47, 0.0]])

    Notes
    -----
    - If both loc and ref_frame are provided, the new frame location is used and the reference frame location is disregarded.
    - A TimeGroup object is created internally to manage time associations between rotation, location, and reference frame. 
    - If the frame is static (single rotation and singular Point), the time properties return TIME_INVARIANT.
    - Use Path objects for time-varying locations.  
    """

    _rot: RotationLike
    _rot_chain: list[RotationLike]
    _interp_rot_fn: Callable[[Array], Rotation]
    _loc: Location
    _time_group: TimeGroup
    _xp: ArrayNS
    _has_ref: bool
    _static_rot: bool
    _static_loc: bool = False

    def __init__(
        self,
        rot: Rotation | RotationSequence,
        loc: Location | None = None,
        ref_frame: Frame | None = None,
        backend: BackendArg = None,
    ) -> None:
        self._xp = resolve_backend(backend)

        # Parse and validate rotation
        # Converts scipy Rotation to RotationLike if needed
        if isinstance(rot, Rotation):
            if rot._quat.shape[0] == 1:  # pyright: ignore[reportAttributeAccessIssue]
                self._rot = _RotationStatic(rot, backend=self._xp)
            else:
                raise ValueError(
                    "Rotation sequence must be wrapped in \
                RotationSequence to construct frame"
                )
        elif isinstance(rot, RotationSequence):
            self._rot = rot
            if rot.backend != self.backend:
                self._rot = rot.convert_to(self.backend)
        else:
            raise ValueError("Rotation must be a scipy Rotation or RotationLike object")

        # Parse location
        if loc is None:
            if ref_frame is None:
                raise ValueError(
                    "Either loc or ref_frame must be provided to construct Frame"
                )
            self._loc = ref_frame._loc.convert_to(self._xp)
        else:
            if isinstance(loc, Point):
                if not loc.is_singular:
                    raise ValueError(
                        "Location Point must be singular (1x3) to construct Frame \n \
                        Use Path objects for time-varying locations."
                    )
                if loc.has_time:
                    warnings.warn(
                        "Frame location Point has associated Time. \n"
                        "Disregarding this time and assuming time invariant location"
                    )
                    loc = Point(loc.ecef[0], backend=self._xp)
            self._loc = loc.convert_to(self._xp)
        self._static_loc = isinstance(self._loc, Point)

        # Parse time
        # Note that ref_frame location time is included, even if loc is provided
        # This is for potential future inclusion of relative location arguments
        _time_objs: list[TimeLike] = [self._rot.time]
        if ref_frame is not None:
            _time_objs.append(ref_frame.time_group)
        if loc is not None:
            _time_objs.append(loc.time)
        self._time_group = TimeGroup(_time_objs, backend=self._xp)
        if self._time_group.duration <= 0:
            raise ValueError(
                "Frame TimeGroup has non-positive duration. \n"
                "Check that all time objects have overlapping time ranges."
            )

        # Parse reference frame flag
        self._has_ref = ref_frame is not None

        # Parse reference frame and create rotation chain
        _rot_chain = []
        if ref_frame is not None:
            if ref_frame.backend != self.backend:
                ref_frame = ref_frame.convert_to(self.backend)
            _rot_chain += ref_frame._rot_chain
        _rot_chain.append(self._rot)  # New rotations applied to RHS

        if all(isinstance(r, _RotationStatic) for r in _rot_chain):
            self._static_rot = True
        else:
            self._static_rot = False

        self._rot_chain = _rot_chain

        # Create and store a flattened composite rotation interpolation function
        self._interp_rot_fn = self._create_interp_fn(self._rot_chain)

    def _create_interp_fn(
        self, rot_chain: list[RotationLike]
    ) -> Callable[[Array], Rotation]:
        """Create a function that computes the composite rotation at given times.
        Constructor function provided to allow backend conversion
        """

        @backend_jit()
        def _interp_rotation(secs: Array) -> Rotation:
            rots = [r._interp_secs(secs) for r in rot_chain]
            final_rot = rots[0]
            for r in rots[1:]:
                final_rot = final_rot * r
            return final_rot

        return _interp_rotation

    @property
    def backend(self) -> str:
        """Get the name of the array backend in use (e.g., 'numpy', 'jax')."""
        return self._xp.__name__

    @property
    def time_group(self) -> TimeGroup:
        """Get the Time object associated with the frame, if any."""
        return self._time_group

    @property
    def time_bounds(self) -> tuple[TimeLike, TimeLike]:
        """Get the time bounds of the frame as a tuple (start_time, end_time).
        If the frame is static, returns TIME_INVARIANT.
        """
        if self.time_group.is_invariant:
            return (TIME_INVARIANT, TIME_INVARIANT)
        return (
            self.time_group.overlap_bounds[0],
            self.time_group.overlap_bounds[1],
        )

    @property
    def rel_rot(self) -> RotationLike:
        """Get the last rotation of the frame relative to the reference frame."""
        return self._rot

    def interp_rot(
        self, time: Time | None = None, check_bounds: bool = True
    ) -> Rotation:
        """Get the interpolated absolute rotation of the frame at the given times.
        If all rotations are time invariant, time can be None.
        """
        if time is None:
            if not self._static_rot:
                raise ValueError(
                    "Time must be provided to interpolate time-varying frame rotation."
                )
            return self._interp_rot_fn(self._xp.array([0.0]))
        if check_bounds:
            if not self.time_group.in_bounds(time):
                warnings.warn(f"""Frame interpolation times are out of bounds.
                    Frame time range: {self.time_bounds[0]} to {self.time_bounds[1]}
                    Interpolation time range: {time.datetime[0]} to {time.datetime[-1]}
                    Extrapolation is not supported and will raise an error.""")
        return self._interp_rot_fn(time.secs)

    def interp_loc(self, time: Time | None = None, check_bounds: bool = True) -> Point:
        """Get the interpolated location of the frame at the given times.
        If the location is static, time can be None.
        """
        if time is None:
            if not self._static_loc:
                raise ValueError(
                    "Time must be provided to interpolate time-varying frame location."
                )
            return self._loc  # pyright: ignore[reportReturnType]
        if check_bounds:
            if not self.time_group.in_bounds(time):
                warnings.warn(f"""Frame interpolation times are out of bounds.
                    Frame time range: {self.time_bounds[0]} to {self.time_bounds[1]}
                    Interpolation time range: {time.datetime[0]} to {time.datetime[-1]}
                    Extrapolation is not supported and will raise an error.""")
        return self._loc._interp(time)

    @property
    def has_ref(self) -> bool:
        """Check if the frame has a reference frame."""
        return self._has_ref

    # @backend_jit(["check_bounds"])
    # def interp(self, time: Time, check_bounds: bool = True) -> Rotation:
    #     """Interpolate the frame at the given times to return the absolute rotation(s).
    #
    #     Args:
    #         time (Time): Times to interpolate the frame at.
    #         check_bounds (bool, optional): Whether to check if the interpolation times are within
    #             the frame time bounds. Defaults to True.
    #
    #     Returns:
    #         Rotation: Interpolated Rotation object at the given times.
    #     """
    #
    #     if self.is_static:
    #         return Rotation._from_raw_quat(  # pyright: ignore
    #             self._xp.repeat(self.static_rot.as_quat(), len(time), axis=0),
    #             xp=self._xp,
    #         )
    #
    #     if isinstance(self.time, Time):
    #         if check_bounds:
    #             if not self.time.in_bounds(time):
    #                 warnings.warn(f"""Frame interpolation times are out of bounds.
    #                     Frame time range: {self.time.datetime[0]} to {self.time.datetime[-1]}
    #                     Interpolation time range: {time.datetime[0]} to {time.datetime[-1]}
    #                     Extrapolation is not supported and will raise an error.""")
    #         rel_rot = self._slerp(time.secs)  # pyright: ignore[reportOptionalCall]
    #     else:
    #         rel_rot = self._rot
    #     if self._ref_frame is None:
    #         return rel_rot
    #     return self._ref_frame.interp(time, check_bounds) * rel_rot

    def convert_to(self, backend: BackendArg) -> Frame:
        """Convert the Frame object to a different backend."""
        xp = resolve_backend(backend)
        if xp == self._xp:
            return self
        _loc = self._loc.convert_to(xp)
        _rot = self._rot.convert_to(xp)
        _rot_chain = [r.convert_to(xp) for r in self._rot_chain]
        _time_group = self.time_group.convert_to(xp)
        _interp_rpt_fn = self._create_interp_fn(_rot_chain)

        obj = Frame.__new__(Frame)
        obj._xp = xp
        obj._loc = _loc
        obj._rot = _rot
        obj._rot_chain = _rot_chain
        obj._time_group = _time_group
        obj._interp_rot_fn = _interp_rpt_fn
        obj._has_ref = self._has_ref
        obj._static_rot = self._static_rot
        obj._static_loc = self._static_loc
        return obj


class Ray:
    """A ray defined by an origin point and a direction vector.
    Can represent multiple rays with associated times.

    Args:
        origin (Point): Point object with length N defining the ray origin(s) in ECEF coordinates (meters).
        dir (Array): Nx3 array of ray direction vectors in ECEF coordinates (meters).
            Direction vectors will be normalized to unit vectors.
        time (Time, optional): Time object associated with the rays.
            Must be same length as origin if provided. Defaults to None.
        backend (BackendArg, optional): Array backend to use (numpy, jax, etc.). Defaults to numpy.

    Notes:
        - Origin Point must be same length as dimension 0 of dir array.
        - Time cannot be provided by both origin Point and time argument.
            - If origin Point has associated Time, time argument must be None.
            - If origin Point does not have associated Time, time argument can be provided.
            - If neither origin Point nor time argument have Time, Ray will not have Time.
        - No check is made that times are monotonically increasing for interpolation. This is left to the user.

    Examples:
        TBC

    """

    # Should the origin input be en ECEF Array instead of Point?

    _origin: Point
    _unit: Array
    _xp: ArrayNS
    _time: TimeLike

    def __init__(
        self,
        origin: Point,
        dir: Array,
        time: TimeLike = TIME_INVARIANT,
        backend: BackendArg = None,
    ) -> None:
        self._xp = resolve_backend(backend)

        # Data validate direction
        _dir = ensure_2d(dir, n=3, backend=self._xp)
        dir_norm = self._xp.linalg.norm(_dir, axis=1)
        if self._xp.isclose(dir_norm, 0).any():
            raise ValueError("Direction vectors cannot be zero.")
        self._unit = _dir / dir_norm[:, self._xp.newaxis]

        # Data validate origin
        if not isinstance(origin, Point):
            raise ValueError("Origin must be a Point object.")
        if len(origin) != self._unit.shape[0]:
            raise ValueError(
                "Origin and unit direction arrays must have the same length."
            )
        self._origin = origin.convert_to(self._xp)

        # Data validate time
        if time is not None and self._origin.has_time:
            raise ValueError(
                "Ray instantiated Ray with time arg if origin already has time."
            )
        elif self._origin.has_time:
            self._time = origin.time
        elif isinstance(time, Time):
            if len(time) != len(origin):
                raise ValueError("Time length must match origin length if provided.")
            self._time = time
        else:
            self._time = TIME_INVARIANT

    @classmethod
    def _constructor(
        cls, origin: Point, unit: Array, time: TimeLike, xp: ArrayNS
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
        origin: Point,
        endpoint: Point,
        time: TimeLike = TIME_INVARIANT,
        backend: BackendArg = None,
    ) -> Ray:
        """Create a Ray object from origin and endpoint arrays.

        Args:
            origin (Point): Nx3 array of ray origin points in ECEF coordinates (meters).
            endpoint (Point): Nx3 array of ray endpoint points in ECEF coordinates (meters).
            time (Time, optional): Time object associated with the rays.
                Must be same length as origin if provided. Defaults to None.
            backend (BackendArg, optional): Array backend to use (numpy, jax, etc.). Defaults to numpy.
        Returns:
            Ray: Ray object defined by the origin and direction from origin to endpoint.
        """

        xp = resolve_backend(backend)
        if len(origin) != len(endpoint):
            raise ValueError("Origin and endpoint arrays must have the same shape.")
        dir = endpoint.ecef - origin.ecef
        return cls(origin, dir, time=time, backend=xp)

    @property
    def origin(self) -> Point:
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
    def time(self) -> TimeLike:
        """Get the associated Time object, if any."""
        return self._time

    @property
    def backend(self) -> str:
        """Get the name of the array backend in use (e.g., 'numpy', 'jax')."""
        return self._xp.__name__

    def __len__(self) -> int:
        return len(self._origin)

    def __repr__(self) -> str:
        return f"""Ray of length {len(self._origin)} 
            with {self._xp.__name__} backend. \n 
            First origin (LLA): {self._origin[0].geodet}, 
            First direction (unit vector): {self._unit[0].reshape(1, 3)[0]} \n"""

    def __str__(self) -> str:
        return f"Ray, n={len(self)}, backend='{self._xp.__name__}')"

    def __getitem__(self, index: int) -> Ray:
        return Ray._constructor(
            self.origin[index],
            self._xp.asarray(self.unit[index]).reshape(1, 3),
            self.time[index],
            self._xp,
        )

    def convert_to(self, backend: BackendArg) -> Ray:
        """Convert the Ray object to a different backend."""
        xp = resolve_backend(backend)
        if xp == self._xp:
            return self
        time_converted = self.time.convert_to(xp)  # pyright: ignore[reportOptionalMemberAccess]
        return Ray._constructor(
            self._origin.convert_to(xp), xp.asarray(self.unit), time_converted, xp
        )

    # @backend_jit(["check_bounds"])
    def interp(self, time: Time, check_bounds: bool = True) -> Ray:
        """Interpolate the Ray origin and direction to the given times.

        Args:
            time (Time): Times to interpolate to.
            check_bounds (bool, optional): Whether to check if the interpolation
                times are within the ray time bounds. Defaults to True.
        Returns:
            Ray: Interpolated Ray object at the given times.
        """

        if isinstance(self.time, Time):
            if check_bounds:
                if not self.time.in_bounds(time):
                    warnings.warn(f"""Ray interpolation times are out of bounds.
                        Ray time range: {self.time.datetime[0]} to {self.time.datetime[-1]}
                        Interpolation time range: {time.datetime[0]} to {time.datetime[-1]}
                        Extrapolation is not supported and will raise an error.""")

            interp_origin_ecef = interp_nd(
                time.secs,
                self.time.secs,  # pyright: ignore[reportOptionalMemberAccess]
                self.origin.ecef,
                backend=self._xp,
            )
            interp_origin = Point._constructor(
                interp_origin_ecef, time=time.convert_to(self._xp), xp=self._xp
            )

            interp_unit = interp_nd(
                time.secs,
                self.time.secs,  # pyright: ignore[reportOptionalMemberAccess]
                self.unit,
                backend=self._xp,
            )
            interp_unit = (
                interp_unit
                / self._xp.linalg.norm(interp_unit, axis=1)[:, self._xp.newaxis]
            )
            return Ray._constructor(
                interp_origin, interp_unit, time.convert_to(self._xp), xp=self._xp
            )
        else:
            raise ValueError("Cannot interpolate Ray without associated Time.")

    @property
    def head_el(self):
        """Return the heading (from north) and elevation (from horizontal) angles in degrees."""

        ned_rots = ned_rotation(self.origin.geodet).inv()
        ned_vecs = self._xp.einsum("ijk,ik->ij", ned_rots.as_matrix(), self.unit)
        head = self._xp.degrees(self._xp.arctan2(ned_vecs[:, 1], ned_vecs[:, 0])) % 360
        el = self._xp.degrees(
            self._xp.arctan2(
                -ned_vecs[:, 2], self._xp.linalg.norm(ned_vecs[:, 0:2], axis=1)
            )
        )
        return ensure_2d(self._xp.stack([head, el], axis=1), n=2, backend=self._xp)

    @classmethod
    def from_head_el(
        cls,
        origin: Array,
        head_el: Array,
        time: Time | None = None,
        backend: BackendArg = None,
    ) -> Ray:
        """Create a Ray object from origin points and heading/elevation angles.

        Args:
            origin (Array): Nx3 array of ray origin points in ECEF coordinates (meters).
            head_el (Array): Nx2 array of heading (from north) and elevation (from horizontal) angles in degrees.
            time (Time, optional): Time object associated with the rays.
                Must be same length as origin if provided. Defaults to None.
            backend (BackendArg, optional): Array backend to use (numpy, jax, etc.). Defaults to numpy.
        Returns:
            Ray: Ray object defined by the origin and direction from heading/elevation angles.
        """

        raise NotImplementedError("from_head_el not yet implemented")
