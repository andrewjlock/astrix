# pyright: standard
# pyright: reportAny=false, reportImplicitOverride=false

from __future__ import annotations
import warnings

from astrix._backend_utils import (
    resolve_backend,
    Array,
    ArrayNS,
    BackendArg,
    warn_if_not_numpy,
)
from astrix.utils import (
    ensure_2d,
    apply_rot,
)

from astrix.functs import (
    interp_nd,
    az_el_from_vec,
    vec_from_az_el,
)

from astrix.time import Time, TimeLike, TimeInvariant, TIME_INVARIANT
from astrix.spatial.location import Point, POINT_ORIGIN
from astrix.spatial.frame import Frame, FRAME_ECEF, ned_frame



class Ray:
    """A ray defined by an origin point, direction vector, reference frame, and optional time.
    The ray extends infinitely in one direction from the origin.
    The origin and direction are stored in local frame coordinates, where the default frame is ECEF (global).

    Args:
        origin (Point): Point object with length N defining the ray origin(s) in ECEF coordinates (meters).
        dir (Array): Nx3 array of ray direction vectors in ECEF coordinates (meters).
            Direction vectors will be normalized to unit vectors.
        frame (Frame, optional): Reference frame for the ray origin and direction.
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

    _unit_rel: Array
    _origin_rel: Array
    _frame: Frame
    _time: TimeLike
    _xp: ArrayNS

    def __init__(
        self,
        dir_rel: Array,
        origin_rel: Array = POINT_ORIGIN.ecef,
        frame: Frame = FRAME_ECEF,
        time: TimeLike = TIME_INVARIANT,
        backend: BackendArg = None,
    ) -> None:
        self._xp = resolve_backend(backend)

        # Data validate direction and origin arrays
        _dir_rel = ensure_2d(dir_rel, n=3, backend=self._xp)
        if self._xp.isclose(self._xp.linalg.norm(_dir_rel, axis=1), 0.).any():
            raise ValueError("Ray direction vectors cannot be zero.")
        self._unit_rel = _dir_rel / self._xp.linalg.norm(_dir_rel, axis=1, keepdims=True)
        self._origin_rel = ensure_2d(origin_rel, n=3, backend=self._xp)

        # Other data validation checks
        if not frame.is_static and isinstance(time, TimeInvariant):
            raise ValueError(
                "Time must be provided if frame is time-varying to create Ray."
            )
        if (self._unit_rel.shape[0] != self._origin_rel.shape[0]):
            if self._origin_rel.shape[0] == 1:
                self._origin_rel = self._xp.repeat(self._origin_rel, self._unit_rel.shape[0], axis=0)
            else:
                raise ValueError(
                    "Ray direction and origin arrays must have the same length or origin must be length 1."
                )
        if self._unit_rel.shape[0] != len(time) and not time.is_singular:
            raise ValueError(
                "Ray direction and time arrays must have the same length or time must be singular."
            )

        self._frame = frame.convert_to(self._xp)
        self._time = time.convert_to(self._xp)

    @classmethod
    def _constructor(
        cls, unit: Array, origin: Array, time: TimeLike, frame: Frame, xp: ArrayNS
    ) -> Ray:
        """Internal constructor to create a Ray object from arrays
        Avoids type checking in __init__."""

        obj = cls.__new__(cls)
        obj._xp = xp
        obj._unit_rel = unit
        obj._origin_rel = origin
        obj._frame = frame
        obj._time = time
        return obj

    @classmethod
    def from_points(
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
        if len(origin) != len(endpoint) or origin.is_singular:
            raise ValueError(
                "Origin and endpoint arrays must have the same shape or origin must be singular."
            )
        dir = endpoint.ecef - origin.ecef
        return cls(dir, origin.ecef, time=time, frame=FRAME_ECEF, backend=xp)

    def _to_frame(self, frame: Frame) -> Ray:
        """Internal method to convert ray origin and direction to a different frame."""
        if frame.backend != self.backend:
            frame = frame.convert_to(self.backend)
        rots_abs = self._frame.interp_rot(self.time, check_bounds=False)
        rots_new = frame.interp_rot(self.time, check_bounds=False)
        rots_rel = rots_new * rots_abs.inv()
        unit_new = apply_rot(rots_rel, self._unit_rel, xp=self._xp)
        origin_new = apply_rot(rots_new, 
            (apply_rot(rots_abs, self._origin_rel, inverse=True, xp=self._xp) + 
            self._frame.interp_loc(self.time, check_bounds=False).ecef -
            - frame.interp_loc(self.time, check_bounds=False).ecef), xp=self._xp
            # apply_rot(rots_new, self._origin_rel, inverse=True, xp=self._xp)
        )
        return Ray._constructor(
            unit_new,
            origin_new,
            self.time,
            frame,
            self._xp,
        )

    def to_ecef(self) -> Ray:
        """Convert the Ray object to ECEF coordinates."""
        return self._to_frame(FRAME_ECEF)

    def to_ned(self) -> Ray:
        warn_if_not_numpy(self._xp)
        frame_ned = ned_frame(self._frame._loc, name=self._frame._name + "-> NED")
        return self._to_frame(frame_ned)

    @property
    def origin_points(self) -> Point:
        """Get the ray origin point(s) as ECEF."""
        ray_ecef = self.to_ecef()
        return Point._constructor(ray_ecef.origin_rel, time=self.time, xp=self._xp)

    @property
    def origin_rel(self) -> Array:
        """Get the ray origin point(s) in the local frame coordinates."""
        return self._origin_rel

    @property
    def unit_ecef(self) -> Array:
        """Get the unit direction vector(s) of the ray."""
        ray_ecef = self.to_ecef()
        return ray_ecef._unit_rel

    @property
    def unit_rel(self) -> Array:
        """Get the unit direction vector(s) of the ray in the local frame coordinates."""
        return self._unit_rel

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
        return self._unit_rel.shape[0]

    def __repr__(self) -> str:
        return f"""Ray of length {len(self)} with
            origin in frame '{self._frame._name}' and
            {self._xp.__name__} backend. 
            """

    def __str__(self) -> str:
        return f"Ray, frame={self._frame._name}, n={len(self)})"

    def __getitem__(self, index: int) -> Ray:
        return Ray._constructor(
            self._unit_rel[index],
            self._origin_rel[index],
            self.time[index],
            self._frame,
            self._xp,
        )

    def convert_to(self, backend: BackendArg) -> Ray:
        """Convert the Ray object to a different backend."""
        xp = resolve_backend(backend)
        if xp == self._xp:
            return self
        unit_converted = xp.asarray(self._unit_rel)
        origin_converted = xp.asarray(self._origin_rel)
        time_converted = self.time.convert_to(xp)  # pyright: ignore[reportOptionalMemberAccess]
        frame_converted = self._frame.convert_to(xp)
        return Ray._constructor(
            unit_converted, origin_converted, time_converted, frame_converted, xp
        )

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

            interp_origin = interp_nd(
                time.secs,
                self.time.secs,  # pyright: ignore[reportOptionalMemberAccess]
                self._origin_rel,
                backend=self._xp,
            )

            interp_unit = interp_nd(
                time.secs,
                self.time.secs,  # pyright: ignore[reportOptionalMemberAccess]
                self._unit_rel,
                backend=self._xp,
            )
            interp_unit = (
                interp_unit
                / self._xp.linalg.norm(interp_unit, axis=1)[:, self._xp.newaxis]
            )
            return Ray._constructor(
                interp_unit, interp_origin, time, self._frame, self._xp
            )
        else:
            raise ValueError("Cannot interpolate Ray without associated Time.")

    @property
    def az_el(self):
        """Return the heading (from north) and elevation (from horizontal) angles in degrees."""
        return az_el_from_vec(self._unit_rel, backend=self._xp)

    @classmethod
    def from_az_el(
        cls,
        az_el: Array,
        frame: Frame = FRAME_ECEF,
        origin_rel: Array = POINT_ORIGIN.ecef,
        time: TimeLike = TIME_INVARIANT,
        backend: BackendArg = None,
    ) -> Ray:
        """Create a Ray object from origin points and heading/elevation angles.

        Args:
            az_el (Array): Nx2 array of azimuth and elevation angles in degrees, relative to the reference frame.
            frame (Frame, optional): Reference frame for the ray origin and direction.
                Defaults to ECEF frame.
            origin_rel (Array, optional): Nx3 array of ray origin points in local frame coordinates.
                Defaults to (0,0,0), which is the reference frame origin.
            time (Time, optional): Time object associated with the rays.
        """
        xp = resolve_backend(backend)
        az_el = ensure_2d(az_el, n=2, backend=xp)
        dir_rel = vec_from_az_el(az_el, backend=xp)
        return cls(dir_rel, origin_rel, frame, time, xp)
