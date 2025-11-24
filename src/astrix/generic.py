from __future__ import annotations
from abc import ABC, abstractmethod
from typing import override

from ._backend_utils import Array, BackendArg, ArrayNS, get_backend
from .time import Time
from .functs import interp_nd, ensure_1d, ensure_2d

class AbstractValue(ABC):
    """A marker interface for variable objects."""

    _time: Time
    _data: Array
    _xp: ArrayNS

    @property
    def data(self) -> Array:
        return self._data

    @property
    def time(self) -> Time:
        return self._time

    @property
    def backend(self) -> str:
        return self._xp.__name__

    def __len__(self) -> int:
        return len(self.time)

    @property
    @abstractmethod
    def invariant(self) -> bool:
        ...

    @abstractmethod
    def interp(self, time: Time) -> Array:
        ...


class TimeSeriesValue(AbstractValue):

    _time: Time
    _data: Array
    _xp: ArrayNS

    def __init__(self,
    data: Array | list[float] | float,
    time: Time,
    backend: BackendArg = None) -> None:

        self._xp = get_backend(backend)
        self._data = ensure_2d(data, backend=self._xp)
        self._time = time

        if len(self._time) != self._data.shape[0]:
            raise ValueError("Length of time must match number of data rows. "+
                             "For time-invariant data, one row is expected.")
        if len(time) < 2:
            raise ValueError("For time-varying data, time must have at least two entries.")

    @property
    @override
    def invariant(self) -> bool:
        return False

    @override
    def interp(self, time: Time) -> Array:
        if getattr(time, "is_invariant", False):
            raise ValueError("Cannot interpolate time-varying data with invariant time.")
        elif isinstance(time, Time):
            return interp_nd(time.unix, self.time.unix, self.data, backend=self._xp)
        else:
            raise TypeError("time must be of type Time.")
