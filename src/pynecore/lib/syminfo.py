from typing import Literal
from ..types.session import Session
from ..types.na import NA
from .session import regular

from ..core.syminfo import SymInfoSession, SymInfoInterval

__all__ = [
    "prefix", "description", "ticker", "root", "tickerid", "currency", "basecurrency", "period", "type", "volumetype",
    "mintick", "pricescale", "minmove", "pointvalue", "timezone",
    "country", "session", "sector", "industry",
    "target_price_average", "target_price_high", "target_price_low", "target_price_date",
    "target_price_estimates"
]

_opening_hours: list[SymInfoInterval] = []
_session_starts: list[SymInfoSession] = []
_session_ends: list[SymInfoSession] = []

prefix: str = ""
description: str = ""
ticker: str = ""
root: str = ""
tickerid: str = ""
currency: str = ""
basecurrency: str = ""
period: str = ""
type: Literal['stock', 'future', 'option', 'forex', 'index', 'fund', 'bond', 'crypto'] | str = ""  # noqa
volumetype: Literal["base", "quote", "tick", "n/a"] | str = ""
mintick: float = 0.0
pricescale: int = 0
minmove: int = 1
pointvalue: float = 0.0
timezone: str = ""
country: str = ""
session: Session = regular
sector: str = ""
industry: str = ""

# Analyst price target information (na when no data available, like in TradingView)
target_price_average: float | NA = NA(float)
target_price_high: float | NA = NA(float)
target_price_low: float | NA = NA(float)
target_price_date: int | NA = NA(int)
target_price_estimates: int | NA = NA(int)

_size_round_factor: float
