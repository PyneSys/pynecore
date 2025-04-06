from typing import Literal
from ..types.na import NA
from ..types.session import Session
from .session import regular

from ..core.syminfo import SymInfoSession, SymInfoInterval

__all__ = [
    "prefix", "description", "ticker", "root", "tickerid", "currency", "basecurrency", "period", "type", "volumetype",
    "mintick", "pricescale", "minmove", "pointvalue", "timezone",
    "country", "session", "sector", "industry"
]

_opening_hours: list[SymInfoInterval] = []
_session_starts: list[SymInfoSession] = []
_session_ends: list[SymInfoSession] = []

prefix: str | NA[str] = NA(str)
description: str | NA[str] = NA(str)
ticker: str | NA[str] = NA(str)
root: str | NA[str] = NA(str)
tickerid: str | NA[str] = NA(str)
currency: str | NA[str] = NA(str)
basecurrency: str | NA[str] = NA(str)
period: str | NA[str] = NA(str)
type: Literal['stock', 'future', 'option', 'forex', 'index', 'fund', 'bond', 'crypto'] | NA[str] = NA(str)  # noqa
volumetype: Literal["base", "quote", "tick", "n/a"] | NA[str] = NA(str)
mintick: float | NA[float] = NA(float)
pricescale: int | NA[int] = NA(int)
minmove: int = 1
pointvalue: float | NA[float] = NA(float)
timezone: str | NA[str] = NA(str)
country: str | NA[str] = NA(str)
session: Session = regular
sector: str | NA[str] = NA(str)
industry: str | NA[str] = NA(str)
