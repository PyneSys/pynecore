"""
Parse provider strings used in ``pyne run`` and ``request.security()``.

Format::

    <provider>:<symbol>@<timeframe>

Examples::

    ccxt:BYBIT:BTC/USDT:USDT@1D       # CCXT, Bybit futures, daily
    ccxt:BINANCE:ETH/USDT@240         # CCXT, Binance spot, 4-hour
    capitalcom:EURUSD@60              # Capital.com, EURUSD, 1-hour

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
    """Symbol in provider-specific format (e.g. ``"BTC/USDT:USDT"``).

    For multi-broker providers the broker prefix is split off into
    :attr:`broker`, so this holds only the instrument; for single-broker
    providers the whole symbol is kept verbatim. May be an empty string
    for a broker-only string such as ``"ccxt:BYBIT"``.
    """

    timeframe: str | None = None
    """Timeframe in TradingView format (e.g. ``"1D"``), or None if not specified."""

    broker: str | None = None
    """Broker / exchange selector for multi-broker providers (e.g. ``"BYBIT"``),
    or ``None`` for single-broker providers."""

    @property
    def provider_symbol(self) -> str:
        """Symbol in the form the provider plugin expects.

        Multi-broker providers receive the broker folded back into the
        symbol (``"BYBIT:BTC/USDT:USDT"``) and split it off internally; a
        broker-only string yields just the broker so the provider can list
        that broker's symbols. Single-broker providers get :attr:`symbol`
        unchanged.
        """
        if self.broker:
            return f"{self.broker}:{self.symbol}" if self.symbol else self.broker
        return self.symbol


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


def parse_provider_string(value: str, *, require_timeframe: bool = False,
                          multi_broker: bool = False) -> ProviderString:
    """
    Parse a provider string into its components.

    :param value: The provider string (e.g. ``"ccxt:BYBIT:BTC/USDT:USDT@1D"``).
    :param require_timeframe: If True, raise ValueError when ``@timeframe`` is missing.
    :param multi_broker: If True, treat the first segment after the provider name
        as the broker/exchange selector (e.g. ``"BYBIT"``) and split it into
        :attr:`ProviderString.broker`; the remainder is the symbol (which may be
        empty for a broker-only string such as ``"ccxt:BYBIT"``). Callers learn
        whether a provider is multi-broker from its ``multi_broker`` class
        attribute.
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

    broker: str | None = None
    symbol = rest
    if multi_broker:
        broker, _, symbol = rest.partition(':')
        if not broker:
            raise ValueError(
                f"Invalid provider string: '{value}'. "
                f"Broker name is empty."
            )

    if require_timeframe and timeframe is None:
        raise ValueError(
            f"Timeframe is required. Use '@' to specify it: "
            f"'{value}@<timeframe>' (e.g. '{value}@1D')"
        )

    return ProviderString(provider=provider, symbol=symbol,
                          timeframe=timeframe, broker=broker)
