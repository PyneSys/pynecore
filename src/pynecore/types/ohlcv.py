from typing import NamedTuple, Any
from .na import NA


class OHLCV(NamedTuple):
    timestamp: int  # Unix timestamp in seconds
    open: float | NA[float]
    high: float | NA[float]
    low: float | NA[float]
    close: float | NA[float]
    volume: float | NA[float]
    extra_fields: dict[str, Any] | None = None
