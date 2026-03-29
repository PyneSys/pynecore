from __future__ import annotations

from math import nan
from typing import TYPE_CHECKING

from ..types.footprint import Footprint

if TYPE_CHECKING:
    from ..core.currency import CurrencyRateProvider

_currency_provider: CurrencyRateProvider | None = None


# noinspection PyUnusedLocal
def security(*args, **kwargs):
    """
    Request data from another symbol/timeframe.

    This function exists for IDE support only. In compiled scripts, the
    SecurityTransformer rewrites all calls into the signal/write/read protocol
    at AST level — this function is never called at runtime.

    Only ``barmerge.lookahead_off`` is supported (deliberate safety decision).
    """
    raise RuntimeError(
        "request.security() should not be called directly. "
        "It is rewritten by SecurityTransformer during compilation."
    )


# noinspection PyUnusedLocal
def security_lower_tf(
    symbol, timeframe, expression,
    ignore_invalid_symbol=False, currency=None,
    ignore_invalid_timeframe=False, calc_bars_count=None,
):
    """
    Request intrabar data from a lower timeframe.

    Returns an array of values, one per intrabar within each chart bar.
    This function exists for IDE support only. In compiled scripts, the
    SecurityTransformer rewrites all calls into the LTF signal/write/read
    protocol at AST level — this function is never called at runtime.

    :param symbol: Symbol to request data from
    :param timeframe: Lower timeframe string (must be <= chart timeframe)
    :param expression: Expression to evaluate in the lower timeframe context
    :param ignore_invalid_symbol: If True, return empty array for invalid symbols
    :param currency: Currency for conversion (not yet supported)
    :param ignore_invalid_timeframe: If True, ignore invalid timeframe
    :param calc_bars_count: Number of bars to calculate (not yet supported)
    :return: array of expression values per intrabar
    """
    raise RuntimeError(
        "request.security_lower_tf() should not be called directly. "
        "It is rewritten by SecurityTransformer during compilation."
    )


def currency_rate(from_currency: str, to_currency: str) -> float:
    """
    Get the currency conversion rate between two currencies.

    Returns the exchange rate to convert from ``from_currency`` to ``to_currency``
    at the current bar's timestamp. The rate is looked up from OHLCV data files
    whose TOML metadata matches the requested currency pair.

    :param from_currency: Source currency code (e.g. ``"EUR"``, ``currency.EUR``)
    :param to_currency: Target currency code (e.g. ``"USD"``, ``currency.USD``)
    :return: Exchange rate as float, or ``na`` if no data is available
    """
    if _currency_provider is None:
        return nan
    from .. import lib
    timestamp = int(lib._datetime.timestamp())
    return _currency_provider.get_rate(str(from_currency), str(to_currency), timestamp)


# noinspection PyUnusedLocal
def dividends(
    ticker=None, field=None, gaps=None, lookahead=None,
    ignore_invalid_symbol=False,
) -> float:
    """
    Request dividend data for a symbol.

    :param ticker: Symbol ticker
    :param field: Dividend field (dividends.gross, dividends.net)
    :param gaps: Gap handling mode (barmerge.gaps_on/off)
    :param lookahead: Lookahead mode (barmerge.lookahead_on/off)
    :param ignore_invalid_symbol: If True, return na instead of raising
    :return: Dividend value or na
    :raises NotImplementedError: When ignore_invalid_symbol is False
    """
    if ignore_invalid_symbol:
        return nan
    raise NotImplementedError("request.dividends() is not yet implemented in PyneCore")


# noinspection PyUnusedLocal
def splits(
    ticker=None, field=None, gaps=None, lookahead=None,
    ignore_invalid_symbol=False,
) -> float:
    """
    Request stock split data for a symbol.

    :param ticker: Symbol ticker
    :param field: Split field (splits.numerator, splits.denominator)
    :param gaps: Gap handling mode (barmerge.gaps_on/off)
    :param lookahead: Lookahead mode (barmerge.lookahead_on/off)
    :param ignore_invalid_symbol: If True, return na instead of raising
    :return: Split value or na
    :raises NotImplementedError: When ignore_invalid_symbol is False
    """
    if ignore_invalid_symbol:
        return nan
    raise NotImplementedError("request.splits() is not yet implemented in PyneCore")


# noinspection PyUnusedLocal
def earnings(
    ticker=None, field=None, gaps=None, lookahead=None,
    ignore_invalid_symbol=False,
) -> float:
    """
    Request earnings data for a symbol.

    :param ticker: Symbol ticker
    :param field: Earnings field (earnings.actual, earnings.estimate, earnings.standardized)
    :param gaps: Gap handling mode (barmerge.gaps_on/off)
    :param lookahead: Lookahead mode (barmerge.lookahead_on/off)
    :param ignore_invalid_symbol: If True, return na instead of raising
    :return: Earnings value or na
    :raises NotImplementedError: When ignore_invalid_symbol is False
    """
    if ignore_invalid_symbol:
        return nan
    raise NotImplementedError("request.earnings() is not yet implemented in PyneCore")


# noinspection PyUnusedLocal
def financial(*args, **kwargs) -> float:
    """
    Request financial data from FactSet.

    :raises NotImplementedError: Not yet implemented in PyneCore
    """
    raise NotImplementedError("request.financial() is not yet implemented in PyneCore")


# noinspection PyUnusedLocal
def economic(*args, **kwargs) -> float:
    """
    Request economic data.

    :raises NotImplementedError: Not yet implemented in PyneCore
    """
    raise NotImplementedError("request.economic() is not yet implemented in PyneCore")


# noinspection PyUnusedLocal
def quandl(*args, **kwargs) -> float:
    """
    Request data from Quandl/Nasdaq.

    :raises NotImplementedError: Not yet implemented in PyneCore
    """
    raise NotImplementedError("request.quandl() is not yet implemented in PyneCore")


# noinspection PyUnusedLocal
def seed(*args, **kwargs):
    """
    Request data from user-maintained GitHub repositories.

    :raises NotImplementedError: Not yet implemented in PyneCore
    """
    raise NotImplementedError("request.seed() is not yet implemented in PyneCore")


# noinspection PyUnusedLocal
def footprint(ticks_per_row: int, va_percent: int) -> Footprint:
    """
    Request volume footprint data for the current bar.

    :param ticks_per_row: Number of ticks per footprint row
    :param va_percent: Value Area percentage
    :return: Footprint object with volume data
    :raises NotImplementedError: Not yet implemented in PyneCore
    """
    raise NotImplementedError("request.footprint() is not yet implemented in PyneCore")


def _reset_request_state() -> None:
    """Reset request module state between script runs."""
    global _currency_provider
    _currency_provider = None
