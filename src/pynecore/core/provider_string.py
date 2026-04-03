"""
Parse provider strings used in ``pyne run`` and ``request.security()``.

Format::

    <provider>:<symbol>@<timeframe>

Examples::

    ccxt:BYBIT:BTC/USDT:USDT@1D       # CCXT, Bybit futures, daily
    ccxt:BINANCE:ETH/USDT@4H          # CCXT, Binance spot, 4-hour
    capitalcom:EURUSD@1H              # Capital.com, EURUSD, 1-hour

The ``@timeframe`` suffix is only required in ``pyne run`` (the main data
source).  In ``request.security()`` the timeframe is a separate parameter,
so the provider string contains only ``<provider>:<symbol>``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderString:
    """Parsed components of a provider string."""

    provider: str
    """Plugin entry point name (e.g. ``"ccxt"``, ``"capitalcom"``)."""

    symbol: str
    """Symbol in provider-specific format (e.g. ``"BYBIT:BTC/USDT:USDT"``)."""

    timeframe: str | None = None
    """Timeframe in TradingView format (e.g. ``"1D"``), or None if not specified."""


def is_provider_string(value: str) -> bool:
    """
    Check whether a string looks like a provider string rather than a file path.

    A provider string contains ``:`` and its first segment (before the first ``:``)
    does not look like a drive letter or path component.

    :param value: The string to check.
    :return: True if it looks like a provider string.
    """
    if ':' not in value:
        return False
    first_segment = value.split(':', 1)[0]
    if len(first_segment) == 1 and first_segment.isalpha():
        return False
    return True


def parse_provider_string(value: str, *, require_timeframe: bool = False) -> ProviderString:
    """
    Parse a provider string into its components.

    :param value: The provider string (e.g. ``"ccxt:BYBIT:BTC/USDT:USDT@1D"``).
    :param require_timeframe: If True, raise ValueError when ``@timeframe`` is missing.
    :return: A :class:`ProviderString` with the parsed components.
    :raises ValueError: If the string is malformed or timeframe is required but missing.
    """
    if ':' not in value:
        raise ValueError(
            f"Invalid provider string: '{value}'. "
            f"Expected format: provider:symbol[@timeframe]"
        )

    provider, rest = value.split(':', 1)

    if not provider:
        raise ValueError(
            f"Invalid provider string: '{value}'. "
            f"Provider name is empty."
        )

    if not rest:
        raise ValueError(
            f"Invalid provider string: '{value}'. "
            f"Symbol is missing after '{provider}:'."
        )

    timeframe = None
    if '@' in rest:
        symbol_part, timeframe = rest.rsplit('@', 1)
        if not timeframe:
            raise ValueError(
                f"Invalid provider string: '{value}'. "
                f"Timeframe is empty after '@'."
            )
        if not symbol_part:
            raise ValueError(
                f"Invalid provider string: '{value}'. "
                f"Symbol is missing before '@'."
            )
        rest = symbol_part

    if require_timeframe and timeframe is None:
        raise ValueError(
            f"Timeframe is required. Use '@' to specify it: "
            f"'{value}@<timeframe>' (e.g. '{value}@1D')"
        )

    return ProviderString(provider=provider, symbol=rest, timeframe=timeframe)
