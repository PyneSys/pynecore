"""
Pine Script ``ticker.*`` namespace - ticker identifier construction.
"""
from . import syminfo

__all__ = [
    'heikinashi',
    'inherit',
    'kagi',
    'linebreak',
    'modify',
    'new',
    'pointfigure',
    'renko',
    'standard',
]

# Internal separator embedded in a ticker identifier by the chart-type helpers
# (currently only ``heikinashi``). ``\x1f`` (ASCII Unit Separator) can never
# appear in a real ticker identifier, so a marked ticker never collides with a
# genuine symbol and never compares equal to the chart symbol — which keeps a
# Heikin Ashi request out of the same-context inline fast path and routes it to
# a subprocess that applies the chart-type transform per bar.
# ``_split_chart_type`` strips the marker back to the base symbol wherever the
# ticker is resolved to data.
_CHART_TYPE_SEP = '\x1f'


def _split_chart_type(tickerid: str) -> 'tuple[str, str | None]':
    """
    Split a chart-type-marked ticker identifier into its base symbol and type.

    Internal helper (not a Pine builtin): the security machinery calls it to
    strip the marker added by :func:`heikinashi` before resolving data.

    :param tickerid: A ticker identifier, optionally carrying a chart-type marker
                     produced by :func:`heikinashi`
    :return: ``(base_symbol, chart_type)``; ``chart_type`` is ``None`` for a plain
             ticker identifier (no marker)
    """
    s = str(tickerid)
    idx = s.find(_CHART_TYPE_SEP)
    if idx < 0:
        return s, None
    return s[:idx], s[idx + len(_CHART_TYPE_SEP):]


# noinspection PyUnusedLocal
def new(prefix: str, ticker: str, session: str | None = None,
        adjustment: str | None = None, backadjustment: str | None = None,
        settlement_as_close: str | None = None) -> str:
    """
    Create a ticker identifier from an exchange prefix and a ticker name.

    The ``session``, ``adjustment``, ``backadjustment`` and ``settlement_as_close``
    parameters are accepted for Pine compatibility but do not alter the identifier:
    PyneCore's data layer does not provide per-request session or adjustment variants
    of a feed.

    :param prefix: Exchange prefix (e.g. ``"BINANCE"``)
    :param ticker: Ticker name (e.g. ``"BTCUSDT"``); a full identifier
                   (``"BINANCE:BTCUSDT"``) is used as-is
    :param session: Session type, ignored (see above)
    :param adjustment: Price adjustment type, ignored (see above)
    :param backadjustment: Continuous contract back-adjustment, ignored (see above)
    :param settlement_as_close: Settlement-as-close setting, ignored (see above)
    :return: The ticker identifier (``"PREFIX:TICKER"``)
    """
    ticker = str(ticker)
    if ':' in ticker:
        return ticker
    return f"{prefix}:{ticker}"


# noinspection PyUnusedLocal
def modify(tickerid: str, session: str | None = None, adjustment: str | None = None,
           backadjustment: str | None = None, settlement_as_close: str | None = None) -> str:
    """
    Create a copy of a ticker identifier with modified session/adjustment settings.

    The settings are accepted for Pine compatibility but do not alter the identifier:
    PyneCore's data layer does not provide per-request session or adjustment variants
    of a feed.

    :param tickerid: The ticker identifier to modify
    :param session: Session type, ignored (see above)
    :param adjustment: Price adjustment type, ignored (see above)
    :param backadjustment: Continuous contract back-adjustment, ignored (see above)
    :param settlement_as_close: Settlement-as-close setting, ignored (see above)
    :return: The ticker identifier
    """
    return str(tickerid)


def standard(symbol: str | None = None) -> str:
    """
    Return the ticker identifier of the standard chart type for a symbol.

    PyneCore feeds carry no synthetic chart-type modifiers, so the standard form is
    the identifier itself.

    :param symbol: The ticker identifier; if None, the chart's symbol is used
    :return: The standard-chart ticker identifier
    """
    if symbol is None:
        return str(syminfo.tickerid)
    return str(symbol)


def inherit(from_tickerid: str, symbol: str) -> str:
    """
    Create a ticker identifier for ``symbol`` inheriting the parameters of
    ``from_tickerid``.

    PyneCore feeds carry no chart-type or adjustment modifiers to inherit, so only the
    exchange prefix is taken over when ``symbol`` has none.

    :param from_tickerid: The ticker identifier to inherit parameters from
    :param symbol: The symbol to build the new identifier for
    :return: The ticker identifier
    """
    symbol = str(symbol)
    if ':' in symbol:
        return symbol
    from_tickerid = str(from_tickerid)
    if ':' in from_tickerid:
        return f"{from_tickerid.split(':', 1)[0]}:{symbol}"
    return symbol


def heikinashi(symbol: str) -> str:
    """
    Create a ticker identifier for requesting Heikin Ashi bar values.

    The returned identifier carries an internal chart-type marker (see
    :data:`_CHART_TYPE_SEP`). When passed to ``request.security()``, the security
    child transforms each bar to Heikin Ashi before the script reads it; every
    other consumer strips the marker back to the base symbol via
    :func:`_split_chart_type`.

    Works in both backtest and live mode. ``request.security_lower_tf()`` with a
    chart type is not supported (it would need per-intrabar transformation).

    :param symbol: The ticker identifier
    :return: The base ticker identifier with the Heikin Ashi chart-type marker
    """
    return f"{symbol}{_CHART_TYPE_SEP}heikinashi"


# noinspection PyUnusedLocal
def renko(symbol: str, style: str | None = None, param: float | None = None,
          request_wicks: bool = False, source: str | None = None) -> str:
    """
    Create a ticker identifier for requesting Renko bar values.

    :param symbol: The ticker identifier
    :param style: Renko box size style
    :param param: Value of the selected style
    :param request_wicks: Whether wick values are reflected
    :param source: The price source
    :raises NotImplementedError: PyneCore has no synthetic chart-type data feeds
    """
    raise NotImplementedError("ticker.renko() is not supported: "
                              "PyneCore has no synthetic chart-type data feeds")


# noinspection PyUnusedLocal
def pointfigure(symbol: str, source: str | None = None, style: str | None = None,
                param: float | None = None, reversal: int | None = None) -> str:
    """
    Create a ticker identifier for requesting Point & Figure values.

    :param symbol: The ticker identifier
    :param source: The price source
    :param style: Point & Figure box size style
    :param param: Value of the selected style
    :param reversal: The reversal amount
    :raises NotImplementedError: PyneCore has no synthetic chart-type data feeds
    """
    raise NotImplementedError("ticker.pointfigure() is not supported: "
                              "PyneCore has no synthetic chart-type data feeds")


# noinspection PyUnusedLocal
def kagi(symbol: str, reversal: float) -> str:
    """
    Create a ticker identifier for requesting Kagi values.

    :param symbol: The ticker identifier
    :param reversal: The reversal amount
    :raises NotImplementedError: PyneCore has no synthetic chart-type data feeds
    """
    raise NotImplementedError("ticker.kagi() is not supported: "
                              "PyneCore has no synthetic chart-type data feeds")


# noinspection PyUnusedLocal
def linebreak(symbol: str, number_of_lines: int) -> str:
    """
    Create a ticker identifier for requesting Line Break values.

    :param symbol: The ticker identifier
    :param number_of_lines: The number of lines
    :raises NotImplementedError: PyneCore has no synthetic chart-type data feeds
    """
    raise NotImplementedError("ticker.linebreak() is not supported: "
                              "PyneCore has no synthetic chart-type data feeds")
