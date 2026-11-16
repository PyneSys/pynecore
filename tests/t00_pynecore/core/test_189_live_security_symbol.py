"""
How a live run resolves the data source of a ``request.security`` context: the
timeframe spelling, the chart's own symbol, the ccxt market lookup of a
TradingView symbol and the coded error of a symbol the provider cannot serve.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from pynecore.core.plugin.live_provider import LiveProviderPlugin, PluginSymbol
from pynecore.core.plugin.provider import canonical_timeframe
from pynecore.core.script_runner import ScriptRunner
from pynecore.core.symbol_map import SymbolMap
from pynecore.errorcodes import LIVE_SECURITY_SYMBOL, LiveSecuritySymbolError
from pynecore.providers.ccxt import CCXTProvider



def _live_runner(tmp_path: Path, provider: object) -> ScriptRunner:
    """A bare live ScriptRunner on a chart provider, without importing a script."""
    runner = object.__new__(ScriptRunner)
    runner._security_data = {}
    runner._symbol_map = SymbolMap.load(tmp_path / "config")
    runner._chart_provider_instance = provider
    runner._chart_provider_name = "ccxt"
    runner._chart_data_path = None
    runner._time_from = None
    runner._sec_syminfos = {}
    runner.syminfo = SimpleNamespace(prefix="BINANCE", ticker="BTCUSDT", period="60")
    return runner


def _chart_provider(**overrides: object) -> SimpleNamespace:
    """The chart provider attributes the live resolution reads."""
    values: dict[str, object] = dict(
        provider_symbol="BINANCE:BTC/USDT:USDT", config=None, ohlcv_path=None,
        resolve_symbol=lambda key: key)
    values.update(overrides)
    return SimpleNamespace(**values)


def __test_canonical_timeframe__():
    """A bare unit gets its multiplier, a multiplied timeframe stays as is"""
    assert canonical_timeframe("D") == "1D"
    assert canonical_timeframe("W") == "1W"
    assert canonical_timeframe("M") == "1M"
    assert canonical_timeframe("1D") == "1D"
    assert canonical_timeframe("3M") == "3M"
    assert canonical_timeframe("60") == "60"
    assert canonical_timeframe("30S") == "30S"


def __test_plugin_symbol_canonical_timeframe__():
    """A live security source is created with the multiplied timeframe"""
    assert PluginSymbol(provider_name="ccxt", symbol="x", timeframe="D").timeframe == "1D"
    assert PluginSymbol(provider_name="ccxt", symbol="x", timeframe="240").timeframe == "240"


def __test_live_own_symbol_uses_chart_provider_symbol__(tmp_path: Path):
    """The chart's own symbol resolves to the chart's provider symbol, never to PREFIX:TICKER"""
    asked: list[str] = []

    def resolve(key: str) -> str:
        asked.append(key)
        return key

    runner = _live_runner(tmp_path, _chart_provider(resolve_symbol=resolve))
    result = runner._resolve_security_data({
        "own": {"symbol": "BINANCE:BTCUSDT", "timeframe": "D"},
        "bare": {"symbol": "BTCUSDT", "timeframe": "240"},
        "other": {"symbol": "BINANCE:ETHUSDT", "timeframe": "60"},
    })
    own, bare, other = result["own"], result["bare"], result["other"]
    assert isinstance(own, PluginSymbol) and isinstance(bare, PluginSymbol)
    assert isinstance(other, PluginSymbol)
    assert (own.symbol, own.timeframe) == ("BINANCE:BTC/USDT:USDT", "1D")
    assert (bare.symbol, bare.timeframe) == ("BINANCE:BTC/USDT:USDT", "240")
    assert other.symbol == "BINANCE:ETHUSDT"
    assert asked == ["BINANCE:ETHUSDT"]


def __test_ccxt_resolve_symbol_market_id__():
    """A TradingView symbol of the chart's exchange resolves to the spot or the .P perpetual market"""
    spot = {"symbol": "ETH/USDT", "type": "spot"}
    swap = {"symbol": "ETH/USDT:USDT", "type": "swap"}
    client = SimpleNamespace(id="binance", load_markets=lambda: None,
                             markets_by_id={"ETHUSDT": [spot, swap], "SOLUSDT": [swap]})
    provider = object.__new__(CCXTProvider)
    provider._client = client
    provider.config = None
    provider.global_symbol_map = None
    provider.provider_name = "ccxt"

    assert provider.resolve_symbol("BINANCE:ETHUSDT") == "binance:ETH/USDT"
    assert provider.resolve_symbol("BINANCE:ETHUSDT.P") == "binance:ETH/USDT:USDT"
    # No spot market for the id: the key is no symbol of this exchange
    assert provider.resolve_symbol("BINANCE:SOLUSDT") == "BINANCE:SOLUSDT"
    # Another exchange, a unified symbol and an unknown id stay as written
    assert provider.resolve_symbol("BYBIT:ETHUSDT") == "BYBIT:ETHUSDT"
    assert provider.resolve_symbol("BINANCE:ETH/USDT") == "BINANCE:ETH/USDT"
    assert provider.resolve_symbol("BINANCE:XYZ") == "BINANCE:XYZ"


def __test_ccxt_resolve_symbol_map_first__():
    """The plugin's symbol_map wins over the market lookup"""
    client = SimpleNamespace(id="binance", load_markets=lambda: None,
                             markets_by_id={"ETHUSDT": [{"symbol": "ETH/USDT", "type": "spot"}]})
    provider = object.__new__(CCXTProvider)
    provider._client = client
    provider.config = SimpleNamespace(symbol_map={"BINANCE:ETHUSDT": "binance:ETH/BTC"})
    provider.global_symbol_map = None
    provider.provider_name = "ccxt"
    assert provider.resolve_symbol("BINANCE:ETHUSDT") == "binance:ETH/BTC"


class _UnknownSymbolProvider(LiveProviderPlugin):
    """A live provider whose symbol lookup always fails."""

    def __init__(self, **_kwargs: object) -> None:
        pass

    def update_symbol_info(self):
        raise RuntimeError("Unknown symbol 'XYZ' on binance.")


_UnknownSymbolProvider.__abstractmethods__ = frozenset()


def __test_live_unknown_symbol_is_coded__(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A symbol the provider cannot serve stops the run with R0055, naming the Pine symbol"""
    monkeypatch.setattr("pynecore.core.plugin.load_plugin",
                        lambda _name: _UnknownSymbolProvider)
    runner = _live_runner(tmp_path, _chart_provider())
    source = PluginSymbol(provider_name="ccxt", symbol="BINANCE:XYZ", timeframe="60")
    contexts = {"sid": {"symbol": "BINANCE:XYZ", "timeframe": "60"}}
    with pytest.raises(LiveSecuritySymbolError) as info:
        runner._prefetch_sec_syminfos({"sid": source}, contexts)
    assert info.value.code is LIVE_SECURITY_SYMBOL
    message = str(info.value)
    assert "BINANCE:XYZ" in message and "`ccxt`" in message
    assert "Unknown symbol 'XYZ' on binance." in message

    ignored = {"sid": {**contexts["sid"], "ignore_invalid_symbol": True}}
    assert runner._prefetch_sec_syminfos({"sid": source}, ignored) == {"sid": None}
