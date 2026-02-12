# pyright: reportAny=false, reportImplicitOverride=false

from __future__ import annotations
import warnings

from scipy.spatial.transform import Rotation

from astrix._backend_utils import (
    resolve_backend,
    Array,
    ArrayNS,
    BackendArg,
    backend_jit,
)

from astrix.functs import ned_rotation, apply_rot, az_el_from_vec


from astrix.time import Time, TIME_INVARIANT, TimeGroup, time_linspace
from astrix.spatial.location import Location, Point, POINT_ORIGIN, Path
from astrix.spatial.rotation import RotationLike, RotationSequence, _RotationStatic

ROT_IDENTITY = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])


class Frame:
    """A reference frame defined by a rotation and location.
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
    >>> rot = Rotation.from_euler(
    ...     "xyz", [90, 0, 0], degrees=True
    ... )  # 90 degree rotation about x-axis
    >>> loc = Point.from_geodet([27.47, 153.03, 0])  # Brisbane location
    >>> frame_static = Frame(rot, loc)  # Frame with static rotation and location

    >>> frame_static.interp_rot().as_euler("xyz", degrees=True)  # Get absolute rotation
    array([[90.,  0.,  0.]])
    >>> frame_static.loc.geodet  # Get frame location in geodetic coordinates
    array([[ 27.47, 153.03,   0.  ]])

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
    >>> frame_dynamic_rot = Frame(
    ...     rot_seq, loc
    ... )  # Frame with time-varying rotation and static location

    >>> interp_rot = frame_dynamic_rot.interp_rot(
    ...     Time.from_datetime(datetime(2021, 1, 1, 12, 30, 0, tzinfo=timezone.utc))
    ... )  # Interpolate to halfway between first and second rotation
    >>> interp_rot.as_euler(
    ...     "xyz", degrees=True
    ... )  # Get interpolated absolute rotation as Euler angles
    array([[45.,  0.,  0.]])
    >>> frame_dynamic_rot.loc.geodet  # Get frame location in geodetic coordinates
    array([[ 27.47, 153.03,   0.  ]])

    Frame defined relative to another frame:

    >>> rot_ref = Rotation.from_euler(
    ...     "xyz", [0, 30, 0], degrees=True
    ... )  # Reference frame
    >>> frame_ref = Frame(rot_ref, loc)  # Reference frame
    >>> rot_rel = Rotation.from_euler("xyz", [0, 40, 0], degrees=True)
    >>> frame = Frame(rot_rel, ref_frame=frame_ref)
    >>>
    >>> frame.interp_rot().as_euler(
    ...     "xyz", degrees=True
    ... )  # Absolute rotation (rot_ref * rot_rel)
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
    _rot_chain: dict[str, RotationLike]
    _loc: Location
    _time_group: TimeGroup
    _xp: ArrayNS
    _has_ref: bool
    _static_rot: bool
    _static_loc: bool
    _name: str

    def __init__(
        self,
        rot: Rotation | RotationSequence,
        loc: Location | None = None,
        ref_frame: Frame | None = None,
        backend: BackendArg = None,
        name: str = "unnamed-frame",
    ) -> None:
        self._xp = resolve_backend(backend)

        # Parse and validate rotation
        # Converts scipy Rotation to RotationLike if needed
        if isinstance(rot, Rotation):
            if rot.single or len(rot) == 1:
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
                        + "Disregarding this time and assuming time invariant location"
                    )
                    loc = Point(loc.ecef[0], backend=self._xp)
            self._loc = loc.convert_to(self._xp)
        self._static_loc = isinstance(self._loc, Point)

        # Parse time
        # Note that ref_frame location time is included, even if loc is provided
        # This is for potential future inclusion of relative location arguments
        _time_objs: list[Time | TimeGroup] = [self._rot.time]
        if ref_frame is not None:
            _time_objs.append(ref_frame.time_group)
        if loc is not None:
            _time_objs.append(loc.time)
        self._time_group = TimeGroup(_time_objs, backend=self._xp)
        if (not self._time_group.is_invariant) and self._time_group.duration <= 0:
            raise ValueError(
                "Frame TimeGroup has non-positive duration. \n"
                + "Check that all time objects have overlapping time ranges."
            )

        # Parse name
        if name == "unnamed-frame":
            if ref_frame is not None:
                n = len(ref_frame._rot_chain) + 1
                name = f"frame-{n}"
            else:
                name = "frame-1"
        elif ref_frame is not None and name in ref_frame._rot_chain:
            raise ValueError(
                f"Frame name '{name}' already exists in reference frame. \n"
                + "Choose a different name to avoid rotation chain conflicts."
            )
        self._name = name

        # Parse reference frame and create rotation chain
        self._has_ref = ref_frame is not None
        _rot_chain: dict[str, RotationLike] = {}
        if ref_frame is not None:
            if ref_frame.backend != self.backend:
                ref_frame = ref_frame.convert_to(self.backend)
            _rot_chain |= ref_frame._rot_chain
        _rot_chain |= {self._name: self._rot}  # New rotations applied to RHS

        if all(isinstance(r, _RotationStatic) for r in _rot_chain.values()):
            self._static_rot = True
        else:
            self._static_rot = False

        self._rot_chain = _rot_chain

    # --- Dunder methods and properties ---

    def __repr__(self) -> str:
        return f"Frame(name={self._name}, static_rot={self._static_rot}, \
                static_loc={self._static_loc}, has_ref={self._has_ref}, \
                time_bounds={self.time_bounds}, backend={self.backend})"

    def __str__(self) -> str:
        return f"Frame: {self._name}"

    @property
    def name(self) -> str:
        """Get the name of the frame."""
        return self._name

    @property
    def name_chain(self) -> list[str]:
        """Get the names of all frames in the rotation chain, from base to current."""
        return list(self._rot_chain.keys())

    @property
    def is_static(self) -> bool:
        """Check if the frame is static (single rotation and singular Point location)."""
        return self._static_rot and self._static_loc

    @property
    def has_ref(self) -> bool:
        """Check if the frame has a reference frame."""
        return self._has_ref

    @property
    def backend(self) -> str:
        """Get the name of the array backend in use (e.g., 'numpy', 'jax')."""
        return self._xp.__name__

    @property
    def time_group(self) -> TimeGroup:
        """Get the Time object associated with the frame, if any."""
        return self._time_group

    @property
    def time_bounds(self) -> tuple[Time, Time]:
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
    def path(self) -> Path:
        """Get the time-varying Path of the frame location, if applicable.
        If the frame location is static, raises AttributeError.
        """
        if isinstance(self._loc, Path):
            return self._loc
        raise AttributeError("Frame location is static; no Path available.")

    @property
    def point(self) -> Point:
        """Get the singular Point of the frame location, if applicable.
        If the frame location is time-varying, raises AttributeError.
        """
        if isinstance(self._loc, Point):
            return self._loc
        raise AttributeError(
            "Frame location is time-varying; no singular Point available."
        )

    @property
    def rel_rot(self) -> RotationLike:
        """Get the last rotation of the frame relative to the reference frame."""
        return self._rot

    @property
    def loc(self) -> Location:
        """Get the location of the frame in ECEF coordinates."""
        return self._loc

    # --- Methods ---

    def interp_rot(
        self, time: Time = TIME_INVARIANT, check_bounds: bool = True
    ) -> Rotation:
        """Get the interpolated absolute rotation of the frame at the given times.
        If all rotations are time invariant, time can be None.
        """
        if isinstance(time, Time):
            if check_bounds:
                if not self.time_group.in_bounds(time):
                    warnings.warn(f"""Frame interpolation times are out of bounds.
                        Frame time range: {self.time_bounds[0]} to {self.time_bounds[1]}
                        Interpolation time range: {time.datetime[0]} to {time.datetime[-1]}
                        Extrapolation is not supported and will raise an error.""")
            return self._interp_rotation_unix(time.unix)
        elif time.is_invariant:
            if not self._static_rot:
                raise ValueError(
                    "Time must be provided to interpolate time-varying frame rotation."
                )
            return self._interp_rotation_unix(self._xp.array([0.0]))
        else:
            raise ValueError("time must be a Time")

    def interp_loc(
        self, time: Time = TIME_INVARIANT, check_bounds: bool = True
    ) -> Point:
        """Get the interpolated location of the frame at the given times.
        If the location is static, time can be None.
        """
        if isinstance(time, Time):
            if check_bounds:
                if not self.time_group.in_bounds(time):
                    warnings.warn(f"""Frame interpolation times are out of bounds.
                        Frame time range: {self.time_bounds[0]} to {self.time_bounds[1]}
                        Interpolation time range: {time.datetime[0]} to {time.datetime[-1]}
                        Extrapolation is not supported and will raise an error.""")
            return self._loc._interp(time)
        elif time.is_invariant:
            if not self._static_loc:
                raise ValueError(
                    "Time must be provided to interpolate time-varying frame location."
                )
            return self._loc  # pyright: ignore[reportReturnType]
        else:
            raise ValueError("time must be a Time")

    def index_rot(self, index: int) -> Rotation:
        """Get the absolute rotation of the frame at the given index.

        Warning: This should only be used after downsampling so that location
        and rotation indeces align. Prefer interp_rot for general use.

        """
        return self._index_rotation(index)

    def index_loc(self, index: int) -> Point:
        """Get the location of the frame at the given index.

        Warning: This should only be used after downsampling so that rotation indices
        are time-aligned. Use with caution. Prefer interp_loc for general use.

        """
        return self._loc[index]

    @backend_jit(["self"])
    def _interp_rotation_unix(self, unix: Array) -> Rotation:
        rots = [r._interp_unix(unix) for r in self._rot_chain.values()]
        final_rot = rots[0]
        for r in rots[1:]:
            final_rot = final_rot * r
        return final_rot

    @backend_jit(["self"])
    def _index_rotation(self, index: int) -> Rotation:
        """Compute the composite rotation at a given index.
        """

        rots = [r[index] for r in self._rot_chain.values()]
        final_rot = rots[0]
        for r in rots[1:]:
            final_rot = final_rot * r
        return final_rot

    def replace_rot(self, frame_name: str, new_rot: Rotation) -> Frame:
        """Replace a rotation in the rotation chain with a new rotation.

        This is an advanced feature and currently only applicable for static rotations.
        Should primarily be used for optimisation purposes in autograd frameworks, such as 
        correcting misalignment.

        Args:
            frame_name (str): Name of the frame whose rotation is to be replaced.
            new_rot (Rotation): New scipy Rotation object to replace the existing rotation.
        """

        if frame_name not in self._rot_chain:
            raise ValueError(f"Frame name '{frame_name}' not found in rotation chain")
        if not isinstance(self._rot_chain[frame_name], _RotationStatic):
            raise ValueError(
                "Can only replace static rotations in composite frame for now"
            )
        _rot = _RotationStatic(new_rot, backend=self._xp)
        _rot_chain = self._rot_chain.copy()
        _rot_chain[frame_name] = _rot

        obj = Frame.__new__(Frame)
        obj._xp = self._xp
        obj._loc = self._loc
        obj._rot = _rot_chain[self._name]
        obj._rot_chain = _rot_chain
        obj._time_group = self._time_group
        obj._has_ref = self._has_ref
        obj._static_rot = self._static_rot
        obj._static_loc = self._static_loc
        obj._name = self._name

        return obj

    def sample_at_time(self, time: Time) -> Frame:
        """Sample the Frame object at specific times, returning a new Frame with 
        time-varying components sampled at those times. The new frame can then be 
        indexed at these times directly to avoid interpolation

        Args:
            time (Time): Time object specifying the times to sample the Frame at.

        Returns:
            Frame: New Frame object with components sampled at the specified times.
        """

        if not self.time_group.in_bounds(time):
            raise ValueError("Sampling times are out of bounds for Frame time range")

        if isinstance(self._loc, Path):
            _loc = Path(self._loc.interp(time))
        else:
            _loc = self._loc

        if isinstance(self._rot, RotationSequence):
            _rot = RotationSequence(self._rot.interp(time), time, backend=self._xp)
        else:
            _rot = self._rot

        _rot_chain = {}
        for k, r in self._rot_chain.items():
            if isinstance(r, RotationSequence):
                _rot_chain[k] = RotationSequence(r.interp(time), time, backend=self._xp)
            else:
                _rot_chain[k] = r

        _time_group = TimeGroup([time], backend=self._xp)

        obj = Frame.__new__(Frame)
        obj._xp = self._xp
        obj._loc = _loc
        obj._rot = _rot
        obj._rot_chain = _rot_chain
        obj._time_group = _time_group
        obj._has_ref = self._has_ref
        obj._static_rot = False
        obj._static_loc = False
        obj._name = self._name
        return obj

    def convert_to(self, backend: BackendArg) -> Frame:
        """Convert the Frame object to a different backend."""
        xp = resolve_backend(backend)
        if xp == self._xp:
            return self
        _loc = self._loc.convert_to(xp)
        _rot = self._rot.convert_to(xp)
        _rot_chain = {k: r.convert_to(xp) for k, r in self._rot_chain.items()}
        _time_group = self.time_group.convert_to(xp)

        obj = Frame.__new__(Frame)
        obj._xp = xp
        obj._loc = _loc
        obj._rot = _rot
        obj._rot_chain = _rot_chain
        obj._time_group = _time_group
        obj._has_ref = self._has_ref
        obj._static_rot = self._static_rot
        obj._static_loc = self._static_loc
        obj._name = self._name
        return obj


FRAME_ECEF = Frame(
    Rotation.from_quat([0.0, 0.0, 0.0, 1.0]), POINT_ORIGIN, name="ECEF"
)  # ECEF frame at origin


def ned_frame(
    loc: Location, downsample: float | None = 1.0, name: str = "NED frame"
) -> Frame:
    """Create a local NED (North-East-Down) frame at the given location(s).
    NED rotations are evaluated at all times in the Point/Path.

    Args:
        loc (Location): Location(s) to define the NED frame origin.
            Must be single point or time-varying Path.
        downsample (float, optional):  Downsample interval for Path objects in seconds.
            If None, no downsampling is performed. Defaults to 10s.
            If loc is a Path and time resolution is greater than downsample interval,
            the Path will be downsampled before creating the NED frame to reduce computational load.
            Note that NED frames likely vary slowly compared to other frames (gimbals, aircraft, etc.),
            so downsampling is recommended for high-resolution paths.
        name (str, optional): Name of the frame. Defaults to "NED frame".

    Returns:
        Frame: NED frame at the given location(s).

    Notes:
        - Adopts backend from Location object.
    """

    if isinstance(loc, Point) and not loc.is_singular:
        raise ValueError("Location Point must be singular (1x3) to create NED frame")

    if isinstance(loc, Path) and downsample is not None:
        if len(loc.time) > loc.time.duration / downsample:
            time_new = time_linspace(
                loc.time[0], loc.time[-1], int(loc.time.duration // downsample + 1)
            )
            loc = Path._constructor(
                loc.interp(time_new, check_bounds=False).ecef,
                time=time_new,
                xp=loc._xp,
            )

    rots = ned_rotation(loc.geodet, xp=loc._xp)
    if loc.time.is_invariant or len(rots) < 2:
        frame = Frame(rots[0], loc if isinstance(loc, Point) else loc[0], backend=loc.backend, name=name)
    else:
        rot_seq = RotationSequence(rots, loc.time, backend=loc.backend)
        frame = Frame(rot_seq, loc, backend=loc.backend, name=name)
    return frame


def velocity_frame(
    path: Path, downsample: float | None = 1.0, name: str = "velocity frame"
) -> Frame:
    """
    Create a velocity-aligned frame at each point along a Path.
    The frame is defined such that the x-axis aligns with the velocity vector,
    the z-axis aligns with the local vertical plane, and the y-axis completes the right-handed system.

    Args:
        path (Path): Path object defining the trajectory.
        downsample (float, optional): Downsample interval for Path objects in seconds.
            If None, no downsampling is performed. Defaults to 10s.
            If path time resolution is greater than downsample interval,
            the Path will be downsampled before creating the velocity frame to reduce computational load.
        name (str, optional): Name of the frame. Defaults to "velocity frame".

    Returns:
        Frame: Velocity-aligned frame along the Path.

    Notes:
        - Adopts backend from Path object.
    """

    if isinstance(path, Path) and downsample is not None:
        if len(path.time) > path.time.duration / downsample:
            time_new = time_linspace(
                path.time[0], path.time[-1], int(path.time.duration // downsample + 1)
            )
            path = Path._constructor(
                path.interp(time_new, check_bounds=False).ecef,
                time=time_new,
                xp=path._xp,
            )

    frame_ned_temp = ned_frame(path, downsample=None, name="temp-ned-frame")
    vel = path.vel
    vel_ned_vec = apply_rot(
        frame_ned_temp.interp_rot(vel.time), vel.unit, inverse=True, xp=path._xp
    )
    head_pitch = az_el_from_vec(vel_ned_vec, backend=path._xp)

    rots = Rotation.from_euler("ZY", head_pitch, degrees=True)
    rot_seq = RotationSequence(rots, path.time, backend=path.backend)
    frame = Frame(rot_seq, ref_frame=frame_ned_temp, backend=path.backend, name=name)
    return frame

def heading_frame(path: Path, downsample: float | None = 1.0, name: str = "heading frame") -> Frame:
    """
    Create a heading-aligned frame at each point along a Path.
    The frame is defined such that the first aligns with the heading direction (azimuth),
    the z-axis aligns with the local vertical plane, and the y-axis completes the right-handed system.

    Args:
        path (Path): Path object defining the trajectory.
        downsample (float, optional): Downsample interval for Path objects in seconds.
            If None, no downsampling is performed. Defaults to 10s.
            If path time resolution is greater than downsample interval,
            the Path will be downsampled before creating the heading frame to reduce computational load.
        name (str, optional): Name of the frame. Defaults to "heading frame".

    Returns:
        Frame: Heading-aligned frame along the Path.

    Notes:
        - Adopts backend from Path object.
    """

    if isinstance(path, Path) and downsample is not None:
        if len(path.time) > path.time.duration / downsample:
            time_new = time_linspace(
                path.time[0], path.time[-1], int(path.time.duration // downsample + 1)
            )
            path = Path._constructor(
                path.interp(time_new, check_bounds=False).ecef,
                time=time_new,
                xp=path._xp,
            )

    frame_ned_temp = ned_frame(path, downsample=None, name="temp-ned-frame")
    vel = path.vel
    vel_ned_vec = apply_rot(
        frame_ned_temp.interp_rot(vel.time), vel.unit, inverse=True, xp=path._xp
    )
    head_pitch = az_el_from_vec(vel_ned_vec, backend=path._xp)

    rots = Rotation.from_euler("Z", head_pitch[:, 0].reshape(-1, 1), degrees=True)
    rot_seq = RotationSequence(rots, path.time, backend=path.backend)
    frame = Frame(rot_seq, ref_frame=frame_ned_temp, backend=path.backend, name=name)
    return frame
