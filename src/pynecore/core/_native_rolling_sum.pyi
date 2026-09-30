"""Type stub of the compiled twin of :class:`pynecore.core.rolling_sum.SumMachine`."""
from typing import Any

__all__ = ['SumMachine']


class SumMachine:
    def step(self, source: Any, length: Any) -> float: ...

    def __pyne_snapshot__(self) -> Any: ...

    def __pyne_restore__(self, token: Any) -> None: ...
