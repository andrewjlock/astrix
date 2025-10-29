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

from astrix.functs import (
    ensure_2d,
    apply_rot,
    interp_nd,
    az_el_from_vec,
    vec_from_az_el,
    pixel_to_vec,
    vec_to_pixel,
    total_angle_from_vec,
    interp_unit_vec
)

from astrix.time import Time, TimeLike, TimeInvariant, TIME_INVARIANT
from astrix.spatial.location import Point, POINT_ORIGIN
from astrix.spatial.frame import Frame, FRAME_ECEF, ned_frame
from astrix.project import Pixel, CameraLike


class Ray:
    """A ray defined by an origin point, direction vector, reference frame, and optional time.

    Args:
        dir_rel (Array): Nx3 array of ray direction vectors in local frame.
            Need not be normalised. E.g., (1, 0, 0) is a ray pointing along axis 1 of reference frame.
        origin_rel (Array): 1x3 or Nx3 array defining the ray origin(s) in local frame (meters).
            Typically (0,0,0) for camera reference frames, or ECEF coordinates for ECEF frame rays.
        frame (Frame, optional): Reference frame for the ray origin and direction.
        time (Time, optional): Time object associated with the rays.
            Must be same length as origin if provided. Defaults to TIME_INVARIANT.
        backend (BackendArg, optional): Array backend to use (numpy, jax, etc.). Defaults to numpy.

    Notes:
        - For calculating metrics (e.g. az/el), the axis are assumed (1) forward, (2) right, (3) down (FRD frame).
        - Although stored in local coordiantes, rays are globally defined by their reference frame.
        - Monotonically increasing time is required for interpolation. But to prevent data-dependent control flow,
            this is not checked on initialization. Use Time.is_increasing to check if needed.

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
        check: bool = True,
        backend: BackendArg = None,
    ) -> None:
        self._xp = resolve_backend(backend)

        # Data validate direction and origin arrays
        _dir_rel = ensure_2d(dir_rel, n=3, backend=self._xp)
        if check:
            if self._xp.isclose(self._xp.linalg.norm(_dir_rel, axis=1), 0.0).any():
                raise ValueError("Ray direction vectors cannot be zero.")
        self._unit_rel = _dir_rel / self._xp.linalg.norm(
            _dir_rel, axis=1, keepdims=True
        )
        self._origin_rel = ensure_2d(origin_rel, n=3, backend=self._xp)

        # Other data validation checks
        if not frame.is_static and isinstance(time, TimeInvariant):
            raise ValueError(
                "Time must be provided if frame is time-varying to create Ray."
            )
        if self._unit_rel.shape[0] != self._origin_rel.shape[0]:
            if self._origin_rel.shape[0] == 1:
                self._origin_rel = self._xp.repeat(
                    self._origin_rel, self._unit_rel.shape[0], axis=0
                )
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

    # --- Constructors ---

    @classmethod
    def from_points(
        cls,
        endpoint: Point,
        origin: Point,
        time: TimeLike = TIME_INVARIANT,
        check: bool =  True,
        backend: BackendArg = None,
    ) -> Ray:
        """Create a Ray object from origin and endpoint arrays in ECEF frame.

        Args:
            origin (Point):  Origin points (ECEF coordinates). Must be length N or 1.
            endpoint (Point): End points (ECEF coordinates). Must be length N.
            time (Time, optional): Time object associated with the rays.
                Must be length N or 1. Defaults to TIME_INVARIANT (no time dependency).
            backend (BackendArg, optional): Array backend to use (numpy, jax, etc.). Defaults to numpy.
        Returns:
            Ray: Ray object defined by the origin and direction from origin to endpoint.

        Notes:
            - Origin and endpoint Point o
        """

        xp = resolve_backend(backend)
        if len(origin) != len(endpoint) and not origin.is_singular:
            raise ValueError(
                "Origin and endpoint arrays must have the same shape or origin must be singular."
            )
        dir = endpoint.ecef - origin.ecef
        return cls(dir, origin.ecef, time=time, frame=FRAME_ECEF, check=check, backend=xp)

    @classmethod
    def from_az_el(
        cls,
        az_el: Array,
        frame: Frame = FRAME_ECEF,
        time: TimeLike = TIME_INVARIANT,
        origin_rel: Array = POINT_ORIGIN.ecef,
        check: bool = True,
        backend: BackendArg = None,
    ) -> Ray:
        """Create a Ray object from origin points and heading/elevation angles.

        Args:
            az_el (Array): Nx2 array of azimuth and elevation angles in degrees, relative to the reference frame.
            frame (Frame, optional): Reference frame for the ray origin and direction.
                Defaults to ECEF frame.
            time (Time, optional): Time object associated with the rays.
            origin_rel (Array, optional): Nx3 array of ray origin points in local frame coordinates.
                Defaults to (0,0,0), which is the reference frame origin.
            check (bool, optional): Whether to check input arrays for validity (not JIT compatible).
                Defaults to True.
            backend (BackendArg, optional): Array backend to use (numpy, jax, etc.). Defaults to numpy.
        """
        xp = resolve_backend(backend)
        az_el = ensure_2d(az_el, n=2, backend=xp)
        dir_rel = vec_from_az_el(az_el, backend=xp)
        return cls(dir_rel, origin_rel, frame, time, check, xp)

    @classmethod
    def from_camera(
        cls,
        pixel: Pixel,
        camera: CameraLike,
        frame: Frame,
        backend: BackendArg = None,
    ) -> Ray:
        """Create a Ray object from pixel coordinates and a camera model.

        Args:
            pixel (Pixel): Pixel object defining the pixel coordinates and optional time.
            camera (CameraLike): Camera model defining the camera parameters and orientation.
            frame (Frame): Reference frame for the ray origin and direction.
            backend (BackendArg, optional): Array backend to use (numpy, jax, etc.). Defaults to numpy.

        Returns:
            Ray: Ray object defined by the pixel coordinates and camera model.
        """

        xp = resolve_backend(backend)
        mat = camera.mat(pixel.time)
        dir = pixel_to_vec(pixel.uv, mat, xp)
        return cls(
            dir,
            origin_rel=xp.zeros((1, 3)),
            frame=frame,
            time=pixel.time,
            backend=xp,
        )

    @classmethod
    def from_target_frame(
        cls,
        target: Point,
        frame: Frame,
        check_bounds: bool = True,
        backend: BackendArg = None,
    ) -> Ray:
        """Create a Ray object from a reference frame and target point(s).

        Args:
            target (Point): Target point(s) in ECEF coordinates. Must be length N or 1.
            frame (Frame): Reference frame for the ray origin and direction.
            backend (BackendArg, optional): Array backend to use (numpy, jax, etc.). Defaults to numpy.

        Returns:
            Ray: Ray object defined by the frame origin and direction to the target point(s).
        """
    
        if check_bounds:
            if not target.has_time and not frame.is_static:
                raise ValueError(
                    "Target Point must have associated Time if Frame is time-varying."
                )
            if isinstance(target.time, Time) and not frame.is_static:
                if not frame.time_group.in_bounds(target.time):
                    raise ValueError(
                        "Target Point time is out of bounds of Frame time range."
                    )

        xp = resolve_backend(backend)
        frame = frame.convert_to(xp)
        origin = frame.interp_loc(target.time, check_bounds=check_bounds)
        dir_ecef = target.ecef - origin.ecef
        dir_frame = apply_rot(
            frame.interp_rot(target.time, check_bounds=check_bounds).inv(),
            dir_ecef,
            xp=xp,
        )
        origin_frame = xp.zeros((1, 3))  # Ray origin at frame origin
        return cls(dir_frame, origin_frame, time=target.time, frame=frame, check=False, backend=xp)


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

    # --- Dune methods and properties ---

    def __len__(self) -> int:
        return self._unit_rel.shape[0]

    def __repr__(self) -> str:
        return f"""Ray of length {len(self)} with
            origin in frame '{self._frame.name}' and
            {self._xp.__name__} backend. 
            """

    def __str__(self) -> str:
        return f"Ray, frame={self._frame.name}, n={len(self)})"

    def __getitem__(self, index: int) -> Ray:
        return Ray._constructor(
            self._unit_rel[index],
            self._origin_rel[index],
            self.time[index],
            self._frame,
            self._xp,
        )

    @property
    def origin_points(self) -> Point:
        """Get the ray origin point(s) as ECEF.
        Note: this involves a frame transformation. For repeated access,
        recommend converting the Ray to ECEF frame first using to_ecef().
        """
        ray_ecef = self.to_ecef()
        return Point._constructor(ray_ecef.origin_rel, time=self.time, xp=self._xp)

    @property
    def origin_rel(self) -> Array:
        """Get the ray origin point(s) in the local frame coordinates.
        Typically zero for camera reference frames, or ECEF coordinates for ECEF frame rays.
        """
        return self._origin_rel

    @property
    def unit_ecef(self) -> Array:
        """Get the unit direction vector(s) of the ray in ECEF frame.
        Note: this involves a frame transformation. For repeated access,
        recommend converting the Ray to ECEF frame first using to_ecef().
        """
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
    def frame(self) -> Frame:
        """Get the reference Frame of the ray."""
        return self._frame

    @property
    def az_el(self):
        """Return the heading (from north) and elevation (from horizontal) angles in degrees."""
        return az_el_from_vec(self._unit_rel, backend=self._xp)

    @property
    def total_angle(self) -> Array:
        """Return the total angle from the forward axis in degrees."""
        return total_angle_from_vec(self._unit_rel, backend=self._xp)

    @property
    def backend(self) -> str:
        """Get the name of the array backend in use (e.g., 'numpy', 'jax')."""
        return self._xp.__name__

    # --- Methods ---

    def to_ecef(self) -> Ray:
        """Convert the Ray object to ECEF coordinates."""
        return self._to_frame(FRAME_ECEF)

    def to_ned(self) -> Ray:
        """Convert the Ray object to a local NED frame at the ray origin."""
        warn_if_not_numpy(self._xp)
        frame_ned = ned_frame(self._frame._loc, name=self._frame.name + "-> NED")
        return self._to_frame(frame_ned)

    def to_frame(self, frame: Frame) -> Ray:
        """Convert the Ray object to a different reference frame.

        Args:
            frame (Frame): Reference frame to convert the ray to.

        Returns:
            Ray: Ray object defined in the new reference frame.
        """
        return self._to_frame(frame)

    def _to_frame(self, frame: Frame) -> Ray:
        """Internal method to convert ray origin and direction to a different frame."""
        if frame.backend != self.backend:
            frame = frame.convert_to(self.backend)
        rots_abs = self._frame.interp_rot(self.time, check_bounds=False)
        rots_new = frame.interp_rot(self.time, check_bounds=False)
        rots_rel = rots_new.inv() * rots_abs
        unit_new = apply_rot(rots_rel, self._unit_rel, xp=self._xp)
        origin_new = apply_rot(
            rots_new,
            (
                apply_rot(rots_abs, self._origin_rel, inverse=True, xp=self._xp)
                + self._frame.interp_loc(self.time, check_bounds=False).ecef
                - -frame.interp_loc(self.time, check_bounds=False).ecef
            ),
            xp=self._xp,
            # apply_rot(rots_new, self._origin_rel, inverse=True, xp=self._xp)
        )
        return Ray._constructor(
            unit_new,
            origin_new,
            self.time,
            frame,
            self._xp,
        )

    def replace_frame(self, frame: Frame) -> Ray:
        """Replace the reference frame of the Ray without changing origin or direction.
        Not a transformation, but direct replacement. Use with caution.

        Args:
            frame (Frame): New reference frame for the ray.
        Returns:
            Ray: Ray object with the new reference frame.

        """

        if frame.backend != self.backend:
            frame = frame.convert_to(self.backend)
        return Ray._constructor(
            self._unit_rel,
            self._origin_rel,
            self.time,
            frame,
            self._xp,
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
                time.unix,
                self.time.unix,
                self._origin_rel,
                backend=self._xp,
            )

            # interp_unit = interp_nd(
            #     time.unix,
            #     self.time.unix,
            #     self._unit_rel,
            #     backend=self._xp,
            # )
            # interp_unit = (
            #     interp_unit
            #     / self._xp.linalg.norm(interp_unit, axis=1)[:, self._xp.newaxis]
            # )
            interp_unit = interp_unit_vec(
                time.unix,
                self.time.unix,
                self._unit_rel,
                backend=self._xp,
            )

            return Ray._constructor(
                interp_unit, interp_origin, time, self._frame, self._xp
            )
        else:
            raise ValueError("Cannot interpolate Ray without associated Time.")

    def project_to_cam(self, camera: CameraLike) -> Pixel:
        """Project the Ray object to pixel coordinates using a camera model.

        Args:
            camera (CameraLike): Camera model defining the camera parameters and orientation.

        Returns:
            Pixel: Pixel object defining the pixel coordinates and associated time.

        Notes:
            - The Ray must be defined in the same reference frame as the camera.
            - Rays that do not intersect the image plane will result in NaN pixel coordinates.
        """

        camera = camera.convert_to(self.backend)
        uv = vec_to_pixel(self._unit_rel, camera.mat(self.time), self._xp)
        return Pixel(uv, time=self.time, backend=self._xp)


    def convert_to(self, backend: BackendArg) -> Ray:
        """Convert the Ray object to a different backend."""
        xp = resolve_backend(backend)
        if xp == self._xp:
            return self
        unit_converted = xp.asarray(self._unit_rel)
        origin_converted = xp.asarray(self._origin_rel)
        time_converted = self.time.convert_to(xp)
        frame_converted = self._frame.convert_to(xp)
        return Ray._constructor(
            unit_converted, origin_converted, time_converted, frame_converted, xp
        )
