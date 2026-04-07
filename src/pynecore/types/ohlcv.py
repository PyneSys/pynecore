from typing import NamedTuple, Any


class OHLCV(NamedTuple):
    timestamp: int  # Unix timestamp in seconds
    open: float
    high: float
    low: float
    close: float
    volume: float
    extra_fields: dict[str, Any] | None = None
