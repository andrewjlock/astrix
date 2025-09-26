# pyright: standard
# pyright: reportAny=false, reportImplicitOverride=false

from __future__ import annotations
import warnings

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from astrix._backend_utils import (
    resolve_backend,
    Array,
    ArrayNS,
    BackendArg,
)

from astrix.functs import (
    ensure_2d,
    ecef2geodet,
    geodet2ecef,
    interp_nd,
    central_diff,
    finite_diff_2pt,
    interp_haversine,
)

from astrix.time import Time, TimeInvariant, TIME_INVARIANT, TimeLike


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

    @property
    @abstractmethod
    def is_singular(self) -> bool:
        pass


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
        self,
        ecef: Array | list[float],
        time: TimeLike = TIME_INVARIANT,
        backend: BackendArg = None,
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

    # --- Constructors ---

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

    @classmethod
    def _constructor(cls, ecef: Array, time: TimeLike, xp: ArrayNS) -> Point:
        """Internal constructor to create a Point object from ECEF array
        Avoids type checking in __init__."""

        obj = cls.__new__(cls)
        obj._xp = xp
        obj._ecef = ecef
        obj._time = time
        return obj

    # --- Dunder methods and properties ---

    def __repr__(self) -> str:
        return f"""Point of length {self._ecef.shape[0]} 
            with {self._xp.__name__} backend. \n 
            First point (LLA): {self.geodet[0]}, 
            Last point (LLA): {self.geodet[-1]}"""

    def __str__(self) -> str:
        return f"Point, n=<Select> {len(self)}, backend='{self._xp.__name__}'"

    def __getitem__(self, index: int) -> Point:
        return Point(self.ecef[index], backend=self._xp)

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

    # --- Properties and methods ---

    @property
    def is_singular(self) -> bool:
        """Check if the Point object represents a single point."""
        return self._ecef.shape[0] == 1

    @property
    def has_time(self) -> bool:
        """Check if the Point has associated Time."""
        return isinstance(self.time, Time)

    def _interp(self, time: Time | TimeInvariant, check_bounds: bool = True) -> Point:
        """Private method to 'interpolate' (broadcast) a singular Point to multiple times.
        Enables compatibility between static Points and dynamic Paths in other modules."""

        if not self.is_singular:
            raise ValueError(
                "Attempting to 'interpolate' (broadcast) a non-singular Point object. \n"
                "This is not supported. Use Path objects for interpolation between multiple points."
            )
        return Point._constructor(
            self._xp.repeat(self.ecef, len(time), axis=0),
            time,
            self._xp,
        )

    def _repeat_single(self, n: int, time: TimeLike) -> Point:
        """Private method to repeat a singular Point n times.
        Enables compatibility between static Points and dynamic Paths in other modules."""

        if not self.is_singular:
            raise ValueError(
                "Attempting to repeat a non-singular Point object. \n"
                "This is not supported. Use Path objects for interpolation between multiple points."
            )
        return Point._constructor(
            self._xp.repeat(self.ecef, n, axis=0),
            time,
            self._xp,
        )

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

    # --- Dunder methods and properties ---

    def __str__(self) -> str:
        return f"Velocity array of length {self.vec.shape[0]} with {self._xp.__name__} backend."

    def __repr__(self) -> str:
        return f"Velocity, n={len(self)}, backend='{self._xp.__name__}')"

    def __len__(self) -> int:
        return self.vec.shape[0]

    @property
    def magnitude(self) -> Array:
        """Get the velocity magnitude in m/s."""
        return self._xp.linalg.norm(self.vec, axis=1)

    @property
    def unit(self) -> Array:
        """Get the unit velocity vector."""
        mag = self.magnitude
        return self.vec / mag[:, self._xp.newaxis]

    @property
    def backend(self) -> str:
        return self._xp.__name__

    # --- Methods ---

    def convert_to(self, backend: BackendArg) -> Velocity:
        """Convert the Velocity object to a different backend."""
        xp = resolve_backend(backend)
        if xp == self._xp:
            return self
        return Velocity(xp.asarray(self.vec), self.time.convert_to(xp), xp)


POINT_ORIGIN = Point([0., 0., 0.])


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

    # --- Constructors ---


    @classmethod
    def _constructor(cls, ecef: Array, time: Time, xp: ArrayNS) -> Path:
        """Internal constructor to create a Path object from ECEF array and Time object.
        Avoids type checking in __init__."""

        obj = cls.__new__(cls)
        obj._xp = xp
        obj._ecef = ecef
        obj._time = time
        if len(obj.time) > 2:
            obj._vel = central_diff(obj._time.secs, obj._ecef, backend=obj._xp)
        elif len(obj.time) == 2:
            obj._vel = finite_diff_2pt(obj._time.secs, obj._ecef, backend=obj._xp)
        return obj

    # --- Dunder methods and properties ---

    def __repr__(self) -> str:
        return f"""Path of length {self._ecef.shape[0]} 
            with {self._xp.__name__} backend. \n 
            Start time: {self.time[0]}, 
            End time: {self.time[1]} \n
            Start point (LLA): {self.geodet[0]},
            End point (LLA): {self.geodet[-1]}"""

    def __str__(self) -> str:
        return f"Path, n={len(self)}, backend='{self._xp.__name__}')"

    @property
    def start_time(self) -> TimeLike:
        """Get the start time of the Path."""
        return self.time[0]

    @property
    def end_time(self) -> TimeLike:
        """Get the end time of the Path."""
        return self.time[-1]

    @property
    def is_singular(self) -> bool:
        """Check if the Path object represents a single point.
        Always False for Path objects."""
        return False

    @property
    def vel(self) -> Velocity:
        """Get the Velocity object associated with the Path."""
        return Velocity(self._vel, self._time, self._xp)

    # --- Methods ---


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

    def convert_to(self, backend: BackendArg) -> Path:
        """Convert the Path object to a different backend."""
        xp = resolve_backend(backend)
        if xp == self._xp:
            return self
        return Path(
            Point(self.ecef, time=self.time.convert_to(xp), backend=xp), backend=xp
        )

