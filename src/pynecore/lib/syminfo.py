from typing import Literal
from ..types.session import Session
from ..types.na import NA
from .session import regular

from ..core.syminfo import SymInfoSession, SymInfoInterval

__all__ = [
    "prefix", "description", "ticker", "root", "tickerid", "main_tickerid", "currency", "basecurrency",
    "period", "type", "volumetype",
    "mintick", "pricescale", "minmove", "pointvalue", "mincontract", "timezone",
    "country", "session", "sector", "industry", "isin",
    "expiration_date", "current_contract",
    "employees", "shareholders", "shares_outstanding_total", "shares_outstanding_float",
    "recommendations_buy", "recommendations_buy_strong", "recommendations_date", "recommendations_hold",
    "recommendations_sell", "recommendations_sell_strong", "recommendations_total",
    "target_price_average", "target_price_high", "target_price_low", "target_price_median",
    "target_price_date", "target_price_estimates"
]

_opening_hours: list[SymInfoInterval] = []
_session_starts: list[SymInfoSession] = []
_session_ends: list[SymInfoSession] = []

# Provider-supplied symbol details — annotation-only: the runner injects the values
# per run (the fee/spread fields only when the data provider supplies them)
opening_hours: list[SymInfoInterval]
session_starts: list[SymInfoSession]
session_ends: list[SymInfoSession]
avg_spread: float
taker_fee: float
maker_fee: float

prefix: str = ""
description: str = ""
ticker: str = ""
root: str = ""
tickerid: str = ""
main_tickerid: str = ""
currency: str = ""
basecurrency: str = ""
period: str = ""
type: Literal['stock', 'future', 'option', 'forex', 'index', 'fund', 'bond', 'crypto'] | str = ""  # noqa
volumetype: Literal["base", "quote", "tick", "n/a"] | str = ""
mintick: float = 0.0
pricescale: int = 0
minmove: int = 1
pointvalue: float = 0.0
mincontract: float = 1.0
timezone: str = ""
country: str = ""
session: Session = regular
sector: str = ""
industry: str = ""
isin: str = ""

# Futures contract information
expiration_date: int | NA = NA(int)
current_contract: str = ""

# Fundamentals (na when no data available, like in TradingView)
employees: int | NA = NA(int)
shareholders: int | NA = NA(int)
shares_outstanding_total: float | NA = NA(float)
shares_outstanding_float: float | NA = NA(float)

# Analyst recommendation counts (na when no data available, like in TradingView)
recommendations_buy: int | NA = NA(int)
recommendations_buy_strong: int | NA = NA(int)
recommendations_date: int | NA = NA(int)
recommendations_hold: int | NA = NA(int)
recommendations_sell: int | NA = NA(int)
recommendations_sell_strong: int | NA = NA(int)
recommendations_total: int | NA = NA(int)

# Analyst price target information (na when no data available, like in TradingView)
target_price_average: float | NA = NA(float)
target_price_high: float | NA = NA(float)
target_price_low: float | NA = NA(float)
target_price_median: float | NA = NA(float)
target_price_date: int | NA = NA(int)
target_price_estimates: int | NA = NA(int)

_size_round_factor: float
