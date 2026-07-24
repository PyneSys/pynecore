"""
Global workdir-level symbol map for ``request.security()`` resolution.

Scripts use TradingView-canonical symbols (``NASDAQ:AAPL``); the user's own
data files carry provider-native symbols (``capitalcom:AAPL``, or a
multi-broker ``ccxt:BYBIT:BTC/USDT:USDT``). The global map
(``<workdir>/config/symbol_map.toml``) translates one into the other so both
backtest (file resolution) and live (``PluginSymbol`` construction) can find
the right data without an explicit ``--security`` mapping.

File schema::

    [symbol_map]
    "BINANCE:BTCUSDT" = "ccxt:BYBIT:BTC/USDT:USDT"
    "NASDAQ:AAPL" = "capitalcom:AAPL"
    "NASDAQ:AAPL:60" = "capitalcom:AAPL"   # optional per-timeframe override

The KEY is the TradingView-style ``PREFIX:TICKER`` written in the script, with
an optional ``:TF`` suffix for a per-timeframe override. The VALUE is a
provider-qualified NATIVE symbol string in the same format as the ``[download]``
provider string (``provider:rest``, where ``rest`` may itself contain colons).
It is NOT a filename — the backtest ``.ohlcv`` path is derived deterministically
via :meth:`ProviderPlugin.get_ohlcv_path`.

A missing or malformed file yields an empty map (never crashes a run); parse
errors are logged as warnings.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ['MappedSymbol', 'SymbolMap', 'SYMBOL_MAP_FILENAME']

logger = logging.getLogger(__name__)

#: Name of the global symbol map file inside ``<workdir>/config``.
SYMBOL_MAP_FILENAME = 'symbol_map.toml'


@dataclass(frozen=True)
class MappedSymbol:
    """A resolved global-map entry: a provider-qualified native symbol.

    :ivar provider: Plugin entry-point name (e.g. ``"ccxt"``, ``"capitalcom"``).
    :ivar native_symbol: The provider-native symbol (the ``rest`` of the
        ``provider:rest`` value; may itself contain further colons, e.g.
        ``"BYBIT:BTC/USDT:USDT"`` for a multi-broker provider).
    """
    provider: str
    native_symbol: str

    @classmethod
    def parse(cls, value: str) -> 'MappedSymbol | None':
        """Parse a ``"provider:rest"`` value into a :class:`MappedSymbol`.

        :param value: The provider-qualified native symbol string.
        :return: The parsed :class:`MappedSymbol`, or ``None`` when the value
            is malformed (no colon, empty provider, or empty native symbol).
        """
        provider, sep, rest = value.partition(':')
        if not sep or not provider or not rest:
            return None
        return cls(provider=provider, native_symbol=rest)


@dataclass(frozen=True)
class SymbolMap:
    """The parsed ``[symbol_map]`` table of ``config/symbol_map.toml``."""

    entries: dict[str, MappedSymbol] = field(default_factory=dict)

    @classmethod
    def load(cls, config_dir: 'Path | str | None') -> 'SymbolMap':
        """Load the global symbol map from ``config_dir/symbol_map.toml``.

        A missing file, an unreadable file, or a malformed ``[symbol_map]``
        table degrades to an empty map (a run is never crashed by the map).
        Parse errors and individual malformed entries are logged as warnings.

        :param config_dir: The workdir ``config`` directory, or ``None``.
        :return: A :class:`SymbolMap` (empty when nothing could be loaded).
        """
        if config_dir is None:
            return cls()
        path = Path(config_dir) / SYMBOL_MAP_FILENAME
        if not path.is_file():
            return cls()
        try:
            with open(path, 'rb') as f:
                data = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError) as e:
            logger.warning("Could not read symbol map %s: %s", path, e)
            return cls()
        table = data.get('symbol_map')
        if not isinstance(table, dict):
            if table is not None:
                logger.warning("Ignoring [symbol_map] in %s: expected a table", path)
            return cls()
        entries: dict[str, MappedSymbol] = {}
        for key, value in table.items():
            if not isinstance(value, str):
                logger.warning("Ignoring non-string symbol_map entry %r in %s", key, path)
                continue
            mapped = MappedSymbol.parse(value)
            if mapped is None:
                logger.warning(
                    "Ignoring malformed symbol_map entry %r = %r in %s "
                    "(expected 'provider:native_symbol')", key, value, path)
                continue
            entries[key] = mapped
        return cls(entries=entries)

    def resolve(self, symbol: str, timeframe: 'str | None' = None) -> 'MappedSymbol | None':
        """Resolve a TradingView-style symbol to a provider-native mapping.

        Key precedence mirrors ``_resolve_security_data``: a ``"SYMBOL:TF"``
        per-timeframe override wins over a bare ``"SYMBOL"`` entry.

        :param symbol: The TradingView-style ``PREFIX:TICKER`` from the script.
        :param timeframe: The security timeframe (used for the ``:TF`` override).
        :return: The :class:`MappedSymbol`, or ``None`` when unmapped.
        """
        if timeframe:
            tf_hit = self.entries.get(f"{symbol}:{timeframe}")
            if tf_hit is not None:
                return tf_hit
        return self.entries.get(symbol)

    def __bool__(self) -> bool:
        return bool(self.entries)
