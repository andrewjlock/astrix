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
    warn_if_not_numpy,
)

from astrix.time import Time, TimeLike, TimeInvariant, TIME_INVARIANT, time_linspace


class RotationLike(ABC):
    """Abstract base class for rotation objects (RotationSingle, RotationSequence).
    'convert_to' function is required for integration with other modules.
    """

    _rot: Rotation
    _xp: ArrayNS

    def __str__(self) -> str:
        return f"{self.__class__.__name__} of length {len(self)} with {self._xp.__name__} backend."

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index: int | slice) -> Rotation:
        pass

    @property
    @abstractmethod
    def time(self) -> TimeLike:
        pass

    @property
    def backend(self) -> str:
        return self._xp.__name__

    @abstractmethod
    def interp(self, time: Time) -> Rotation:
        pass

    @abstractmethod
    def _interp_unix(self, unix: Array) -> Rotation:
        pass

    @abstractmethod
    def convert_to(self, backend: BackendArg) -> RotationLike:
        pass


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

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int | slice) -> Rotation:
        """Get the single rotation as a Rotation object."""
        return self._rot

    @property
    def time(self) -> TimeInvariant:
        """Get the Time object associated with the rotation (always static)."""
        return TIME_INVARIANT

    def convert_to(self, backend: BackendArg) -> _RotationStatic:
        """Convert the RotationSingle object to a different backend."""
        xp = resolve_backend(backend)
        if xp == self._xp:
            return self
        return _RotationStatic(self._rot, xp)

    def interp(self, time: Time | TimeInvariant) -> Rotation:
        """Interpolate the rotation at the given times (always returns the same rotation)."""
        return Rotation._from_raw_quat(  # pyright: ignore
            self._xp.repeat(self._rot._quat, len(time), axis=0),  # pyright: ignore[reportAttributeAccessIssue]
            xp=self._xp,
        )

    def _interp_unix(self, unix: Array) -> Rotation:
        return Rotation._from_raw_quat(  # pyright: ignore
            self._xp.repeat(self._rot._quat, len(unix), axis=0),  # pyright: ignore[reportAttributeAccessIssue]
            xp=self._xp,
        )

    def _replace_rot(self, rot: Rotation) -> None:
        if not (rot.single or len(rot) == 1):
            raise ValueError(
                "RotationSingle must be initialized with a single rotation"
            )
        self._rot = _convert_rot_backend(rot, self._xp)


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
        else:
            if rot.as_quat().ndim == 1:
                raise ValueError(
                    "RotationSequence requires multiple rotations; use RotationSingle for a single rotation"
                )
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
        self._slerp = Slerp(self._time.unix, self._rot)

    def __len__(self) -> int:
        return len(self._rot)

    def __getitem__(self, index: int | slice) -> Rotation:
        """Get a subset of the rotation sequence as a Rotation object."""
        return self._rot[index]

    @property
    def time(self) -> Time:
        """Get the Time object associated with the rotation sequence."""
        return self._time

    @property
    def rots(self) -> Rotation:
        """Get the underlying scipy Rotation object containing all rotations."""
        return self._rot

    def interp(self, time: Time, check_bounds: bool = True) -> Rotation:
        """Interpolate the rotation sequence at the given times to return Rotation(s)."""
        time = time.convert_to(self._xp)
        if check_bounds:
            if not self._time.in_bounds(time):
                warnings.warn(f"""RotationSequence interpolation times are out of bounds.
                    RotationSequence time range: {self._time[0]} to {self._time[-1]}
                    Interpolation time range: {time[0]} to {time[-1]}
                    Extrapolation is not supported and will raise an error.""")
        return self._slerp(time.unix)

    def _interp_unix(self, unix: Array) -> Rotation:
        return self._slerp(unix)

    def downsample(self, dt_max: float) -> RotationSequence:
        """Downsample the rotation sequence to a coarser time resolution.

        Parameters
        ----------
        dt_max : float
            Desired maximum time step in seconds for downsampling.

        Returns
        -------
        RotationSequence
            A new RotationSequence object with downsampled rotations.
        """

        warn_if_not_numpy(self._xp, "Rotation downsampling")

        new_times = time_linspace(
            self._time[0], self._time[-1], int(self._xp.ceil(self._time.duration / dt_max))
        )
        new_rots = self.interp(new_times)
        return RotationSequence(new_rots, new_times, backend=self._xp)

    def convert_to(self, backend: BackendArg) -> RotationSequence:
        """Convert the RotationSequence object to a different backend."""
        xp = resolve_backend(backend)
        if xp is self._xp:
            return self
        return RotationSequence(self._rot, self._time, xp)
