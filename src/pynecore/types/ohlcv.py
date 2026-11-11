from typing import NamedTuple, Any


class AskOHLC(NamedTuple):
    """Ask-side prices of a bar whose main OHLC is the bid side."""
    open: float
    high: float
    low: float
    close: float


class OHLCV(NamedTuple):
    timestamp: int  # Unix timestamp in milliseconds
    open: float
    high: float
    low: float
    close: float
    volume: float
    extra_fields: dict[str, Any] | None = None
    is_closed: bool = True
    # Set when the feed quotes both sides: ``open``..``close`` are then the bid side
    ask: AskOHLC | None = None
