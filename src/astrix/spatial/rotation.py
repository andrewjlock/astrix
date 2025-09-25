# pyright: standard
# pyright: reportAny=false, reportImplicitOverride=false

from __future__ import annotations
import warnings

from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation, Slerp

from astrix._backend_utils import (
    resolve_backend,
    Array,
    ArrayNS,
    BackendArg,
    _convert_rot_backend,
)


from astrix.time import Time, TimeLike, TimeInvariant, TIME_INVARIANT

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
        if not (self._rot.single or len(self._rot) == 1):
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

    def interp(self, time: Time | TimeInvariant) -> Rotation:
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

