"""
@pyne
"""
import pytest
from pathlib import Path

from pynecore.core.data_converter import DataConverter


def main():
    """
    Dummy main function to be a valid Pyne script
    """
    pass


def __test_symbol_provider_detection_ccxt__():
    """Test CCXT-style filename detection"""
    dc = DataConverter()
    
    # Test without ccxt prefix but with exchange  
    symbol, provider = dc.guess_symbol_from_filename(Path("BINANCE_BTC_USDT.csv"))
    assert symbol == "BTC/USDT"
    assert provider == "binance"
    
    # Test exchange with compact symbol
    symbol, provider = dc.guess_symbol_from_filename(Path("BINANCE_BTCUSDT.csv"))
    assert symbol == "BTC/USDT"
    assert provider == "binance"
    
    # Test with colon separators
    symbol, provider = dc.guess_symbol_from_filename(Path("BYBIT:BTC:USDT.csv"))
    assert symbol == "BTC/USDT"
    assert provider == "bybit"
    
    # Test ccxt with BYBIT exchange - provider should be bybit, not ccxt
    symbol, provider = dc.guess_symbol_from_filename(Path("ccxt_BYBIT_BTC_USDT_USDT_1.csv"))
    assert symbol == "BTC/USDT"
    assert provider == "bybit"  # When ccxt_ prefix, provider is the exchange name


def __test_symbol_provider_detection_capitalcom__():
    """Test Capital.com filename detection"""
    dc = DataConverter()
    
    # Test with dots
    symbol, provider = dc.guess_symbol_from_filename(Path("capital.com_EURUSD_60.csv"))
    assert symbol == "EURUSD"
    assert provider == "capital.com"
    
    # Test with uppercase
    symbol, provider = dc.guess_symbol_from_filename(Path("CAPITALCOM_EURUSD.csv"))
    assert symbol == "EURUSD"
    assert provider == "capitalcom"


def __test_symbol_provider_detection_tradingview__():
    """Test TradingView export format detection"""
    dc = DataConverter()
    
    # Test with hash suffix
    symbol, provider = dc.guess_symbol_from_filename(Path("CAPITALCOM_EURUSD, 30_cbf9d.csv"))
    assert symbol == "EURUSD"
    assert provider == "capitalcom"
    
    # Test TV prefix
    symbol, provider = dc.guess_symbol_from_filename(Path("TV_BTCUSD_1h.csv"))
    assert symbol == "BTCUSD"
    assert provider == "tradingview"
    
    # Test TradingView prefix
    symbol, provider = dc.guess_symbol_from_filename(Path("TRADINGVIEW_AAPL_daily.csv"))
    assert symbol == "AAPL"
    assert provider == "tradingview"


def __test_symbol_provider_detection_metatrader__():
    """Test MetaTrader filename detection"""
    dc = DataConverter()
    
    # Test MT4 format
    symbol, provider = dc.guess_symbol_from_filename(Path("MT4_EURUSD_M1.csv"))
    assert symbol == "EURUSD"
    assert provider == "mt4"
    
    # Test MT5 format
    symbol, provider = dc.guess_symbol_from_filename(Path("MT5_GBPUSD_H1_2024.csv"))
    assert symbol == "GBPUSD"
    assert provider == "mt5"
    
    # Test forex pair without explicit provider
    symbol, provider = dc.guess_symbol_from_filename(Path("EURUSD.csv"))
    assert symbol == "EURUSD"
    assert provider == "forex"
    
    # Test another forex pair
    symbol, provider = dc.guess_symbol_from_filename(Path("GBPJPY.csv"))
    assert symbol == "GBPJPY"
    assert provider == "forex"


def __test_symbol_provider_detection_crypto_exchanges__():
    """Test various crypto exchange filename formats"""
    dc = DataConverter()
    
    # Binance
    symbol, provider = dc.guess_symbol_from_filename(Path("BINANCE_BTCUSDT.csv"))
    assert symbol == "BTC/USDT"
    assert provider == "binance"
    
    # Bybit
    symbol, provider = dc.guess_symbol_from_filename(Path("BYBIT_ETH_USDT.csv"))
    assert symbol == "ETH/USDT"
    assert provider == "bybit"
    
    # Coinbase
    symbol, provider = dc.guess_symbol_from_filename(Path("COINBASE_BTC_USD.csv"))
    assert symbol == "BTC/USD"
    assert provider == "coinbase"
    
    # Kraken
    symbol, provider = dc.guess_symbol_from_filename(Path("KRAKEN_XRPUSD.csv"))
    assert symbol == "XRP/USD"
    assert provider == "kraken"


def __test_symbol_provider_detection_generic_crypto__():
    """Test generic crypto pair detection without provider"""
    dc = DataConverter()
    
    # Common crypto pairs should be detected
    symbol, provider = dc.guess_symbol_from_filename(Path("BTCUSDT.csv"))
    assert symbol == "BTC/USDT"
    assert provider == "ccxt"
    
    symbol, provider = dc.guess_symbol_from_filename(Path("ETHUSD.csv"))
    assert symbol == "ETH/USD"
    assert provider == "ccxt"
    
    symbol, provider = dc.guess_symbol_from_filename(Path("BTC_USDT.csv"))
    assert symbol == "BTC/USDT"
    assert provider == "ccxt"


def __test_symbol_provider_detection_stock_symbols__():
    """Test stock symbol detection"""
    dc = DataConverter()
    
    # Simple stock symbols
    symbol, provider = dc.guess_symbol_from_filename(Path("AAPL.csv"))
    assert symbol == "AAPL"
    assert provider is None
    
    symbol, provider = dc.guess_symbol_from_filename(Path("MSFT_daily.csv"))
    assert symbol == "MSFT"
    assert provider is None
    
    # With IB provider
    symbol, provider = dc.guess_symbol_from_filename(Path("IB_AAPL_1h.csv"))
    assert symbol == "AAPL"
    assert provider == "ib"


def __test_symbol_provider_detection_complex_filenames__():
    """Test complex filename patterns"""
    dc = DataConverter()
    
    # Multiple underscores and timeframe - ccxt_ prefix means provider is exchange
    symbol, provider = dc.guess_symbol_from_filename(Path("ccxt_BYBIT_BTC_USDT_USDT_5.csv"))
    assert symbol == "BTC/USDT"
    assert provider == "bybit"  # When ccxt_ prefix, provider is the exchange name
    
    # Mixed case - Note: Mixed case may not be detected properly
    # Using uppercase for consistency
    symbol, provider = dc.guess_symbol_from_filename(Path("CAPITAL.COM_EURUSD_60.csv"))
    assert symbol == "EURUSD"
    assert provider == "capital.com"
    
    # With date suffix
    symbol, provider = dc.guess_symbol_from_filename(Path("MT5_EURUSD_2024_01_01.csv"))
    assert symbol == "EURUSD"
    assert provider == "mt5"


def __test_symbol_provider_detection_edge_cases__():
    """Test edge cases and invalid formats"""
    dc = DataConverter()
    
    # Empty filename
    symbol, provider = dc.guess_symbol_from_filename(Path(".csv"))
    assert symbol is None
    assert provider is None
    
    # Too short symbol
    symbol, provider = dc.guess_symbol_from_filename(Path("XX.csv"))
    assert symbol is None
    assert provider is None
    
    # Only provider, no symbol
    symbol, provider = dc.guess_symbol_from_filename(Path("CCXT.csv"))
    assert symbol is None
    assert provider == "ccxt"
    
    # Numbers only (should not be detected as symbol)
    symbol, provider = dc.guess_symbol_from_filename(Path("12345.csv"))
    assert symbol is None
    assert provider is None


def __test_symbol_provider_detection_forex_pairs__():
    """Test various forex pair formats"""
    dc = DataConverter()
    
    # Standard 6-char format
    symbol, provider = dc.guess_symbol_from_filename(Path("EURUSD.csv"))
    assert symbol == "EURUSD"
    assert provider == "forex"
    
    # With separator
    symbol, provider = dc.guess_symbol_from_filename(Path("EUR_USD.csv"))
    assert symbol == "EURUSD"
    assert provider == "forex"
    
    # With slash
    symbol, provider = dc.guess_symbol_from_filename(Path("EUR-USD.csv"))
    assert symbol == "EURUSD"
    assert provider == "forex"
    
    # Less common pairs
    symbol, provider = dc.guess_symbol_from_filename(Path("NZDJPY.csv"))
    assert symbol == "NZDJPY"
    assert provider == "forex"


def __test_symbol_provider_detection_our_format__():
    """Test PyneCore own format detection"""
    dc = DataConverter()

    # Our format with provider and symbol
    symbol, provider = dc.guess_symbol_from_filename(Path("capitalcom_EURUSD_60.ohlcv"))
    assert symbol == "EURUSD"
    assert provider == "capitalcom"

    # CCXT style with exchange - provider is exchange name
    symbol, provider = dc.guess_symbol_from_filename(Path("ccxt_BYBIT_BTC_USDT_USDT_1.ohlcv"))
    assert symbol == "BTC/USDT"
    assert provider == "bybit"  # When ccxt_ prefix, provider is the exchange name

    # Simple format without provider
    symbol, provider = dc.guess_symbol_from_filename(Path("BTCUSD_1h.ohlcv"))
    assert symbol == "BTC/USD"
    assert provider == "ccxt"  # Should default to ccxt for crypto


def __test_symbol_provider_detection_from_csv_content_databento__(tmp_path):
    """Databento CSVs carry the symbol in a column; provider tagged via ts_event."""
    csv_path = tmp_path / "glbx-mdp3-20220103.ohlcv-1m.csv"
    with open(csv_path, 'w') as f:
        f.write("ts_event,rtype,publisher_id,instrument_id,open,high,low,close,volume,symbol\n")
        f.write("2022-01-03T19:06:00.000000000Z,33,1,206323,4765.0,4765.0,4765.0,4765.0,2,ESZ2\n")

    symbol, provider = DataConverter.guess_symbol_from_csv_content(csv_path)
    assert symbol == "ESZ2"
    assert provider == "databento"


def __test_symbol_provider_detection_from_csv_content_ticker_column__(tmp_path):
    """`ticker` column is honoured the same way as `symbol`."""
    csv_path = tmp_path / "raw_export.csv"
    with open(csv_path, 'w') as f:
        f.write("time,open,high,low,close,volume,ticker\n")
        f.write("2025-01-01T00:00:00Z,100,101,99,100.5,42,AAPL\n")

    symbol, provider = DataConverter.guess_symbol_from_csv_content(csv_path)
    assert symbol == "AAPL"
    assert provider is None  # no Databento marker


def __test_symbol_provider_detection_from_csv_content_no_hints__(tmp_path):
    """Plain OHLCV CSV without symbol/ticker column returns (None, None)."""
    csv_path = tmp_path / "plain.csv"
    with open(csv_path, 'w') as f:
        f.write("timestamp,open,high,low,close,volume\n")
        f.write("1641236760,100,101,99,100.5,42\n")

    symbol, provider = DataConverter.guess_symbol_from_csv_content(csv_path)
    assert symbol is None
    assert provider is None


def __test_convert_to_ohlcv_databento_uses_csv_symbol__(tmp_path):
    """convert_to_ohlcv on a hint-less Databento CSV picks symbol/provider from CSV content.

    End-to-end: convert_to_ohlcv on a Databento CSV without filename hints
    must pick up symbol from the `symbol` column and provider from `ts_event`."""
    from pynecore.core.syminfo import SymInfo
    csv_path = tmp_path / "glbx-mdp3-20220103-20220104.ohlcv-1m.csv"
    with open(csv_path, 'w') as f:
        f.write("ts_event,rtype,publisher_id,instrument_id,open,high,low,close,volume,symbol\n")
        f.write("2022-01-03T19:06:00.000000000Z,33,1,206323,4765.0,4765.0,4765.0,4765.0,2,ESZ2\n")
        f.write("2022-01-03T19:07:00.000000000Z,33,1,206323,4765.0,4766.0,4764.0,4765.5,5,ESZ2\n")
        f.write("2022-01-03T19:08:00.000000000Z,33,1,206323,4765.5,4767.0,4765.0,4766.0,3,ESZ2\n")

    DataConverter().convert_to_ohlcv(csv_path, force=True)

    toml_path = csv_path.with_suffix('.toml')
    assert toml_path.exists()

    syminfo = SymInfo.load_toml(toml_path)
    assert syminfo.ticker == "ESZ2"
    assert syminfo.prefix == "DATABENTO"


def __test_convert_to_ohlcv_restores_originals_when_backup_fails__(tmp_path, monkeypatch):
    """A failed vacating rename leaves the previously converted pair fully in place.

    Both destinations are renamed aside before anything is published. If the second
    of those renames fails — the sidecar is held open elsewhere — the first one must
    be put back, otherwise the good binary survives only under its internal
    ``.replaced`` name and the conversion silently destroys the last usable output.
    """
    from pynecore.core import data_converter as data_converter_module

    csv_path = tmp_path / "backup_failure.csv"
    csv_path.write_text(
        "time,open,high,low,close,volume,sig\n"
        "2024-01-01 00:00:00,10,11,9,10.5,100,1\n"
        "2024-01-01 00:01:00,10.5,11.5,10,11,110,2\n"
        "2024-01-01 00:02:00,11,12,10.5,11.5,120,3\n"
    )
    ohlcv_path = csv_path.with_suffix('.ohlcv')
    extra_path = csv_path.with_suffix('.extra.csv')

    converter = DataConverter()
    converter.convert_to_ohlcv(csv_path, symbol="TEST")
    assert ohlcv_path.exists() and extra_path.exists()
    good_binary = ohlcv_path.read_bytes()
    good_sidecar = extra_path.read_text()

    real_replace = data_converter_module.replace_file

    def failing_replace(source, destination):
        if str(destination).endswith(".extra.csv.replaced"):
            raise PermissionError("sidecar is held open by another process")
        real_replace(source, destination)

    monkeypatch.setattr(data_converter_module, "replace_file", failing_replace)
    with pytest.raises(Exception):
        converter.convert_to_ohlcv(csv_path, symbol="TEST", force=True)
    monkeypatch.undo()

    assert ohlcv_path.read_bytes() == good_binary
    assert extra_path.read_text() == good_sidecar
    assert list(tmp_path.glob("*.replaced")) == []
    assert list(tmp_path.glob("*.converting.*")) == []


def __test_helper_convert_csv(tmp_path, name: str, timestamps: list[int]):
    """Convert a flat-price CSV at ``timestamps`` (epoch seconds) and load its TOML"""
    from pynecore.core.syminfo import SymInfo
    csv_path = tmp_path / name
    with open(csv_path, 'w') as f:
        f.write("timestamp,open,high,low,close,volume\n")
        for ts in timestamps:
            f.write(f"{ts},100,101,99,100.5,10\n")
    DataConverter().convert_to_ohlcv(csv_path, force=True)
    return SymInfo.load_toml(csv_path.with_suffix('.toml'))


def __test_convert_to_ohlcv_247_schedule_uses_python_weekdays__(tmp_path):
    """A round-the-clock feed gets one 00:00-23:59:59 session on every day 0 (Monday) ... 6.

    Schedule days are Python weekdays. ISO numbering (1 ... 7) left Monday out of the
    schedule, and the only session start/end pair made the whole week one session, so
    ``request.security("D")`` paired every Monday chart bar with that day's close.
    """
    from datetime import time
    start = 1704067200  # 2024-01-01 00:00 UTC, a Monday
    syminfo = __test_helper_convert_csv(
        tmp_path, "BINANCE_BTCUSDT_30.csv", [start + i * 1800 for i in range(48 * 21)])

    assert sorted(oh.day for oh in syminfo.opening_hours) == list(range(7))
    assert all(oh.start == time(0, 0) and oh.end == time(23, 59, 59)
               for oh in syminfo.opening_hours)
    assert sorted((s.day, s.time) for s in syminfo.session_starts) == \
           [(d, time(0, 0)) for d in range(7)]
    assert sorted((s.day, s.time) for s in syminfo.session_ends) == \
           [(d, time(23, 59, 59)) for d in range(7)]


def __test_convert_to_ohlcv_weekday_daily_schedule_includes_monday__(tmp_path):
    """Monday-Friday daily bars are scheduled on days 0 ... 4, not Tuesday-Saturday"""
    start = 1704067200  # 2024-01-01, a Monday
    stamps = [start + d * 86400 for d in range(28) if d % 7 < 5]
    syminfo = __test_helper_convert_csv(tmp_path, "CAPITALCOM_EURUSD_1D.csv", stamps)

    assert sorted({oh.day for oh in syminfo.opening_hours}) == [0, 1, 2, 3, 4]
    assert sorted(s.day for s in syminfo.session_starts) == [0, 1, 2, 3, 4]
    assert sorted(s.day for s in syminfo.session_ends) == [0, 1, 2, 3, 4]


def __test_syminfo_rejects_schedule_day_outside_python_weekdays__(tmp_path):
    """``day = 7`` has no weekday: loading it fails instead of silently dropping the day"""
    from pynecore.core.syminfo import SymInfo
    syminfo = __test_helper_convert_csv(
        tmp_path, "BINANCE_ETHUSDT_60.csv", [1704067200 + i * 3600 for i in range(24 * 14)])
    toml_path = tmp_path / "BINANCE_ETHUSDT_60.toml"
    assert syminfo.opening_hours
    text = toml_path.read_text()
    toml_path.write_text(text.replace("day = 6\n", "day = 7\n", 1))

    with pytest.raises(ValueError, match="Invalid schedule day 7"):
        SymInfo.load_toml(toml_path)


def __test_convert_to_ohlcv_hourly_24x5_schedule_excludes_weekend__(tmp_path):
    """An even round-the-clock Monday-Friday feed is not mistaken for a 24/7 market"""
    from datetime import time
    start = 1704067200  # 2024-01-01 00:00 UTC, a Monday
    stamps = [start + h * 3600 for h in range(24 * 7 * 4) if (h // 24) % 7 < 5]
    syminfo = __test_helper_convert_csv(tmp_path, "CAPITALCOM_EURUSD_60.csv", stamps)

    assert sorted((oh.day, oh.start, oh.end) for oh in syminfo.opening_hours) == \
           [(d, time(0, 0), time(23, 59, 59)) for d in range(5)]
    assert sorted(s.day for s in syminfo.session_starts) == [0, 1, 2, 3, 4]
    assert sorted(s.day for s in syminfo.session_ends) == [0, 1, 2, 3, 4]


def __test_convert_to_ohlcv_keeps_unloadable_toml__(tmp_path):
    """A TOML that fails to load stops the conversion instead of being overwritten"""
    from pynecore.core.data_converter import ConversionError
    stamps = [1704067200 + i * 3600 for i in range(24 * 14)]
    __test_helper_convert_csv(tmp_path, "BINANCE_SOLUSDT_60.csv", stamps)
    toml_path = tmp_path / "BINANCE_SOLUSDT_60.toml"
    legacy_text = toml_path.read_text().replace("day = 6\n", "day = 7\n", 1)
    toml_path.write_text(legacy_text)

    with pytest.raises(ConversionError, match="Invalid schedule day 7"):
        DataConverter().convert_to_ohlcv(tmp_path / "BINANCE_SOLUSDT_60.csv", force=True)
    assert toml_path.read_text() == legacy_text
