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
    """End-to-end: convert_to_ohlcv on a Databento CSV without filename hints
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


# Test runner functions that pytest will find
def test_symbol_provider_detection_ccxt():
    __test_symbol_provider_detection_ccxt__()


def test_symbol_provider_detection_capitalcom():
    __test_symbol_provider_detection_capitalcom__()


def test_symbol_provider_detection_tradingview():
    __test_symbol_provider_detection_tradingview__()


def test_symbol_provider_detection_metatrader():
    __test_symbol_provider_detection_metatrader__()


def test_symbol_provider_detection_crypto_exchanges():
    __test_symbol_provider_detection_crypto_exchanges__()


def test_symbol_provider_detection_generic_crypto():
    __test_symbol_provider_detection_generic_crypto__()


def test_symbol_provider_detection_stock_symbols():
    __test_symbol_provider_detection_stock_symbols__()


def test_symbol_provider_detection_complex_filenames():
    __test_symbol_provider_detection_complex_filenames__()


def test_symbol_provider_detection_edge_cases():
    __test_symbol_provider_detection_edge_cases__()


def test_symbol_provider_detection_forex_pairs():
    __test_symbol_provider_detection_forex_pairs__()


def test_symbol_provider_detection_our_format():
    __test_symbol_provider_detection_our_format__()


def test_symbol_provider_detection_from_csv_content_databento(tmp_path):
    __test_symbol_provider_detection_from_csv_content_databento__(tmp_path)


def test_symbol_provider_detection_from_csv_content_ticker_column(tmp_path):
    __test_symbol_provider_detection_from_csv_content_ticker_column__(tmp_path)


def test_symbol_provider_detection_from_csv_content_no_hints(tmp_path):
    __test_symbol_provider_detection_from_csv_content_no_hints__(tmp_path)


def test_convert_to_ohlcv_databento_uses_csv_symbol(tmp_path):
    __test_convert_to_ohlcv_databento_uses_csv_symbol__(tmp_path)