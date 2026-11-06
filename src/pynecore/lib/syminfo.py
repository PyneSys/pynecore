from typing import Literal
from ..types.session import Session
from ..types.na import NA, na_float, na_int
from ..types.pine_types import PyneFloat, PyneInt
from .session import regular

from ..core.module_property import module_function_property
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
_extended_hours: list[SymInfoInterval] = []
# The regular hours and their dated exceptions; the same as the symbol's own session
# unless the bars follow the extended hours
_regular_hours: list[SymInfoInterval] = []
_regular_corrections: dict | None = None

# Provider-supplied symbol details — annotation-only: the runner injects the values
# per run (the fee/spread fields only when the data provider supplies them)
opening_hours: list[SymInfoInterval]
session_starts: list[SymInfoSession]
session_ends: list[SymInfoSession]
extended_hours: list[SymInfoInterval]
avg_spread: float
taker_fee: float
maker_fee: float

description: str = ""
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
expiration_date: int | NA = na_int
current_contract: str = ""

# Fundamentals (na when no data available, like in TradingView)
employees: int | NA = na_int
shareholders: int | NA = na_int
shares_outstanding_total: PyneInt = na_int
shares_outstanding_float: float | NA = na_float

# Analyst recommendation counts (na when no data available, like in TradingView)
recommendations_buy: int | NA = na_int
recommendations_buy_strong: int | NA = na_int
recommendations_date: int | NA = na_int
recommendations_hold: int | NA = na_int
recommendations_sell: int | NA = na_int
recommendations_sell_strong: int | NA = na_int
recommendations_total: int | NA = na_int

# Analyst price target information (na when no data available, like in TradingView)
target_price_average: float | NA = na_float
target_price_high: float | NA = na_float
target_price_low: float | NA = na_float
target_price_median: float | NA = na_float
target_price_date: int | NA = na_int
target_price_estimates: PyneFloat = na_float

_size_round_factor: float

# The chart symbol's exchange prefix and its name without that prefix, as the
# runner sets them. Pine reads ``syminfo.prefix`` / ``syminfo.ticker`` as these
# values and also calls them as functions of another symbol, so the two names
# are module properties over these cells.
_prefix: str = ""
_ticker: str = ""


def _symbol_parts(symbol: str) -> tuple[str, str]:
    """
    A ticker identifier's exchange prefix and the rest, split at its first colon.

    A chart-type marker (``ticker.heikinashi()``) belongs to neither part, and a
    symbol without a colon has neither: TradingView answers both with an empty
    string (``syminfo.ticker("ETHUSDT")`` is ``""``), and the rest of a spread
    keeps its later colons (``"BINANCE:ETHUSDT/BINANCE:BTCUSDT"`` ->
    ``"ETHUSDT/BINANCE:BTCUSDT"``).

    :param symbol: The ticker identifier
    :return: ``(prefix, ticker)``
    """
    from .ticker import _split_chart_type  # ``ticker`` imports this module
    base, _chart_type = _split_chart_type(symbol)
    prefix_part, colon, ticker_part = base.partition(':')
    return (prefix_part, ticker_part) if colon else ('', '')


@module_function_property
def prefix(symbol: str | None = None) -> str:
    """
    The exchange prefix of a symbol.

    :param symbol: A ticker identifier (``"BINANCE:BTCUSDT"``); None reads the
                   chart symbol's own prefix
    :return: The part before the first colon (``"BINANCE"``), or an empty
             string for a symbol without one
    """
    if symbol is None:
        return _prefix
    return _symbol_parts(str(symbol))[0]


@module_function_property
def ticker(symbol: str | None = None) -> str:
    """
    The name of a symbol without its exchange prefix.

    :param symbol: A ticker identifier (``"BINANCE:BTCUSDT"``); None reads the
                   chart symbol's own name
    :return: The part after the first colon (``"BTCUSDT"``), or an empty string
             for a symbol without one
    """
    if symbol is None:
        return _ticker
    return _symbol_parts(str(symbol))[1]
