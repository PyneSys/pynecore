"""
Tests for the global workdir symbol map (``config/symbol_map.toml``).

Covers :class:`SymbolMap` loading/parsing precedence and an integration test
of ``ScriptRunner._resolve_security_data`` against a temp workdir: a global-map
hit resolving to a synthetic ``.ohlcv`` file, the missing-file error message,
and an explicit ``security_data`` mapping overriding the global map.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from pynecore.core.symbol_map import MappedSymbol, SymbolMap, SYMBOL_MAP_FILENAME
from pynecore.core.script_runner import ScriptRunner


# --- MappedSymbol.parse ---

def __test_mapped_symbol_parse_simple__():
    """A single-colon value splits into provider + native symbol"""
    m = MappedSymbol.parse("capitalcom:AAPL")
    assert m == MappedSymbol(provider="capitalcom", native_symbol="AAPL")


def __test_mapped_symbol_parse_multi_colon__():
    """Extra colons stay in the native symbol (multi-broker providers)"""
    m = MappedSymbol.parse("ccxt:BYBIT:BTC/USDT:USDT")
    assert m == MappedSymbol(provider="ccxt", native_symbol="BYBIT:BTC/USDT:USDT")


def __test_mapped_symbol_parse_malformed__():
    """No colon / empty parts return None"""
    assert MappedSymbol.parse("AAPL") is None
    assert MappedSymbol.parse(":AAPL") is None
    assert MappedSymbol.parse("capitalcom:") is None


# --- SymbolMap.load ---

def _write_map(config_dir: Path, body: str) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / SYMBOL_MAP_FILENAME).write_text(body, encoding="utf-8")


def __test_load_missing_file_is_empty__(tmp_path: Path):
    """A missing config dir / file yields an empty map, never crashes"""
    assert not SymbolMap.load(None)
    assert not SymbolMap.load(tmp_path / "config")


def __test_load_basic_entries__(tmp_path: Path):
    """A well-formed [symbol_map] table parses into MappedSymbol values"""
    _write_map(tmp_path, """
[symbol_map]
"NASDAQ:AAPL" = "capitalcom:AAPL"
"BINANCE:BTCUSDT" = "ccxt:BYBIT:BTC/USDT:USDT"
""")
    sm = SymbolMap.load(tmp_path)
    assert sm.resolve("NASDAQ:AAPL") == MappedSymbol("capitalcom", "AAPL")
    assert sm.resolve("BINANCE:BTCUSDT") == MappedSymbol("ccxt", "BYBIT:BTC/USDT:USDT")
    assert sm.resolve("UNKNOWN:X") is None


def __test_resolve_timeframe_precedence__(tmp_path: Path):
    """A "SYMBOL:TF" override wins over the bare "SYMBOL" entry"""
    _write_map(tmp_path, """
[symbol_map]
"NASDAQ:AAPL" = "capitalcom:AAPL"
"NASDAQ:AAPL:60" = "ccxt:BINANCE:AAPL"
""")
    sm = SymbolMap.load(tmp_path)
    # TF match wins
    assert sm.resolve("NASDAQ:AAPL", "60") == MappedSymbol("ccxt", "BINANCE:AAPL")
    # Non-matching TF falls back to the bare symbol entry
    assert sm.resolve("NASDAQ:AAPL", "1D") == MappedSymbol("capitalcom", "AAPL")
    # No TF given -> bare symbol entry
    assert sm.resolve("NASDAQ:AAPL") == MappedSymbol("capitalcom", "AAPL")


def __test_load_malformed_entries_skipped__(tmp_path: Path):
    """Malformed / non-string entries are skipped, valid ones kept"""
    _write_map(tmp_path, """
[symbol_map]
"GOOD:ONE" = "capitalcom:AAPL"
"BAD:NOCOLON" = "AAPL"
"BAD:NUMBER" = 42
""")
    sm = SymbolMap.load(tmp_path)
    assert sm.resolve("GOOD:ONE") == MappedSymbol("capitalcom", "AAPL")
    assert sm.resolve("BAD:NOCOLON") is None
    assert sm.resolve("BAD:NUMBER") is None


def __test_load_broken_toml_is_empty__(tmp_path: Path):
    """A syntactically broken toml degrades to an empty map (no crash)"""
    _write_map(tmp_path, "[symbol_map]\n\"X:Y\" = \n")
    assert not SymbolMap.load(tmp_path)


def __test_load_no_symbol_map_table__(tmp_path: Path):
    """A file without a [symbol_map] table yields an empty map"""
    _write_map(tmp_path, "[other]\nfoo = \"bar\"\n")
    assert not SymbolMap.load(tmp_path)


# --- ScriptRunner._resolve_security_data integration ---

def _make_runner(tmp_path: Path, *, security_data=None) -> ScriptRunner:
    """Build a bare ScriptRunner with only the attributes
    ``_resolve_security_data`` needs, without importing a script.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    runner = object.__new__(ScriptRunner)
    runner._security_data = security_data or {}
    runner._symbol_map = SymbolMap.load(tmp_path / "config")
    runner._chart_provider_instance = None
    runner._chart_provider_name = None
    runner._chart_data_path = data_dir / "chart.ohlcv"
    runner._time_from = None
    runner.syminfo = SimpleNamespace(prefix="NASDAQ", ticker="AAPL", period="1D")
    return runner


def __test_resolve_security_data_map_hit__(tmp_path: Path):
    """A global-map hit resolves to the derived .ohlcv file when it exists"""
    _write_map(tmp_path / "config", """
[symbol_map]
"BINANCE:BTCUSDT" = "ccxt:BYBIT:BTC/USDT:USDT"
""")
    # Create the synthetic .ohlcv/.toml pair the map derives.
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    expected = data_dir / "ccxt_BYBIT_BTC_USDT_USDT_1D.ohlcv"
    expected.touch()
    expected.with_suffix(".toml").touch()

    runner = _make_runner(tmp_path)
    result = runner._resolve_security_data(
        {"sid1": {"symbol": "BINANCE:BTCUSDT", "timeframe": "1D"}})
    assert result["sid1"] == str(expected)


def __test_resolve_security_data_map_missing_file__(tmp_path: Path):
    """A mapped-but-missing file raises an actionable ValueError"""
    _write_map(tmp_path / "config", """
[symbol_map]
"BINANCE:BTCUSDT" = "ccxt:BYBIT:BTC/USDT:USDT"
""")
    runner = _make_runner(tmp_path)
    with pytest.raises(ValueError) as ei:
        runner._resolve_security_data(
            {"sid1": {"symbol": "BINANCE:BTCUSDT", "timeframe": "1D"}})
    msg = str(ei.value)
    assert "ccxt_BYBIT_BTC_USDT_USDT_1D.ohlcv" in msg
    assert "pyne data download" in msg
    assert "ccxt:BYBIT:BTC/USDT:USDT@1D" in msg


def __test_resolve_security_data_explicit_overrides_map__(tmp_path: Path):
    """An explicit security_data mapping wins over the global map"""
    _write_map(tmp_path / "config", """
[symbol_map]
"BINANCE:BTCUSDT" = "ccxt:BYBIT:BTC/USDT:USDT"
""")
    explicit = tmp_path / "data" / "my_explicit"
    runner = _make_runner(
        tmp_path, security_data={"BINANCE:BTCUSDT": str(explicit)})
    result = runner._resolve_security_data(
        {"sid1": {"symbol": "BINANCE:BTCUSDT", "timeframe": "1D"}})
    # Explicit path returned even though the mapped file does not exist.
    assert result["sid1"] == str(explicit)


def __test_resolve_security_data_ignore_invalid__(tmp_path: Path):
    """ignore_invalid_symbol swallows a mapped-but-missing file (returns None)"""
    _write_map(tmp_path / "config", """
[symbol_map]
"BINANCE:BTCUSDT" = "ccxt:BYBIT:BTC/USDT:USDT"
""")
    runner = _make_runner(tmp_path)
    result = runner._resolve_security_data(
        {"sid1": {"symbol": "BINANCE:BTCUSDT", "timeframe": "1D",
                  "ignore_invalid_symbol": True}})
    assert result["sid1"] is None
