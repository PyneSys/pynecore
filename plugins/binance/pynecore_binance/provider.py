"""Binance data provider — a :class:`CCXTProvider` pinned to the ``binance``
exchange, with its own credential source.

Why not plain ``ccxt:BINANCE:…``? Two reasons:

* The broker (``binance_broker``) must build on a provider whose config carries
  broker fields (sandbox, slippage, mainnet guard) in ONE file
  (``workdir/config/plugins/binance_broker.toml``) instead of splitting
  credentials into ``ccxt.toml``.
* :class:`CCXTProvider` *replaces* its exchange config wholesale from
  ``ccxt.toml [binance]`` when that section exists. A testnet broker run
  picking up MAINNET keys from ``ccxt.toml`` would be catastrophic, so this
  subclass rebuilds the client strictly from its OWN config and never reads
  ``ccxt.toml``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pynecore.core.plugin import override
from pynecore.providers.ccxt import CCXTProvider, CCXTConfig, _PYNECORE_ONLY_CONFIG_KEYS

__all__ = ['BinanceProvider', 'BinanceConfig']

#: Config fields that are pynecore/plugin concerns and must NEVER be spread
#: into the ccxt client constructor.
_NON_CCXT_CONFIG_KEYS: frozenset[str] = _PYNECORE_ONLY_CONFIG_KEYS | frozenset({
    'allow_mainnet', 'stop_slippage_ticks', 'poll_interval',
})


@dataclass
class BinanceConfig(CCXTConfig):
    """Binance plugin configuration (``workdir/config/plugins/binance.toml``).

    Inherits ``apiKey`` / ``secret`` / ``sandbox`` from :class:`CCXTConfig`.
    ``sandbox = true`` targets the spot testnet (testnet.binance.vision) —
    testnet keys are separate from mainnet keys.
    """


class BinanceProvider(CCXTProvider):
    """Binance spot market data (history + ccxt.pro live streaming)."""

    plugin_name = "Binance"
    Config = BinanceConfig

    #: ccxt exchange id this provider is pinned to.
    exchange_id = 'binance'

    @override
    def __init__(self, *, symbol: str | None = None, timeframe: str | None = None,
                 ohlcv_dir: Path | None = None, config=None):
        import ccxt

        # The framework passes the bare symbol ("BTC/USDT"); CCXTProvider
        # expects "exchange:symbol", so pin the exchange prefix here.
        if symbol and not symbol.lower().startswith(f"{self.exchange_id}:"):
            symbol = f"{self.exchange_id}:{symbol}"

        super().__init__(symbol=symbol, timeframe=timeframe,
                         ohlcv_dir=ohlcv_dir, config=config)

        # Rebuild the client strictly from OUR config — CCXTProvider.__init__
        # may have replaced the exchange config from ccxt.toml [binance],
        # which would leak mainnet credentials into a sandbox run.
        exchange_config = {}
        if self.config:
            exchange_config = {
                k: v for k, v in vars(self.config).items()
                if v and k not in _NON_CCXT_CONFIG_KEYS
            }
        self._exchange_config = dict(exchange_config)
        self._client = ccxt.binance({
            'enableRateLimit': True,
            'adjustForTimeDifference': True,
            **exchange_config,
        })
        if self.config and getattr(self.config, 'sandbox', False):
            self._client.set_sandbox_mode(True)
