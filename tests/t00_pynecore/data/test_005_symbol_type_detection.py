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


def __test_detect_symbol_type_forex__():
    """Test forex pair detection"""
    dc = DataConverter()
    
    # Standard forex pairs
    symbol_type, currency, base = dc.guess_symbol_type("EURUSD")
    assert symbol_type == "forex"
    assert currency == "USD"
    assert base == "EUR"
    
    symbol_type, currency, base = dc.guess_symbol_type("EUR/USD")
    assert symbol_type == "forex"
    assert currency == "USD"
    assert base == "EUR"
    
    symbol_type, currency, base = dc.guess_symbol_type("GBPUSD")
    assert symbol_type == "forex"
    assert currency == "USD"
    assert base == "GBP"
    
    symbol_type, currency, base = dc.guess_symbol_type("USDJPY")
    assert symbol_type == "forex"
    assert currency == "JPY"
    assert base == "USD"
    
    # Less common forex pairs
    symbol_type, currency, base = dc.guess_symbol_type("NZDJPY")
    assert symbol_type == "forex"
    assert currency == "JPY"
    assert base == "NZD"
    
    symbol_type, currency, base = dc.guess_symbol_type("EURCHF")
    assert symbol_type == "forex"
    assert currency == "CHF"
    assert base == "EUR"


def __test_detect_symbol_type_crypto__():
    """Test crypto pair detection"""
    dc = DataConverter()
    
    # Common crypto pairs
    symbol_type, currency, base = dc.guess_symbol_type("BTC/USDT")
    assert symbol_type == "crypto"
    assert currency == "USDT"
    assert base == "BTC"
    
    symbol_type, currency, base = dc.guess_symbol_type("BTCUSDT")
    assert symbol_type == "crypto"
    assert currency == "USDT"
    assert base == "BTC"
    
    symbol_type, currency, base = dc.guess_symbol_type("ETH/USD")
    assert symbol_type == "crypto"
    assert currency == "USD"
    assert base == "ETH"
    
    symbol_type, currency, base = dc.guess_symbol_type("ETHUSDT")
    assert symbol_type == "crypto"
    assert currency == "USDT"
    assert base == "ETH"
    
    # Other crypto coins
    symbol_type, currency, base = dc.guess_symbol_type("ADAUSDT")
    assert symbol_type == "crypto"
    assert currency == "USDT"
    assert base == "ADA"
    
    symbol_type, currency, base = dc.guess_symbol_type("DOGEUSDT")
    assert symbol_type == "crypto"
    assert currency == "USDT"
    assert base == "DOGE"
    
    symbol_type, currency, base = dc.guess_symbol_type("SOL/USDC")
    assert symbol_type == "crypto"
    assert currency == "USDC"
    assert base == "SOL"


def __test_detect_symbol_type_other__():
    """Test that unknown symbols default to 'other' type"""
    dc = DataConverter()
    
    # Stock-like symbols should be 'other' now
    symbol_type, currency, base = dc.guess_symbol_type("AAPL")
    assert symbol_type == "other"
    assert currency == "USD"
    assert base is None
    
    symbol_type, currency, base = dc.guess_symbol_type("MSFT")
    assert symbol_type == "other"
    assert currency == "USD"
    assert base is None
    
    # Unknown patterns
    symbol_type, currency, base = dc.guess_symbol_type("UNKNOWN")
    assert symbol_type == "other"
    assert currency == "USD"
    assert base is None
    
    symbol_type, currency, base = dc.guess_symbol_type("ABC123")
    assert symbol_type == "other"
    assert currency == "USD"
    assert base is None
    
    # Too short
    symbol_type, currency, base = dc.guess_symbol_type("XY")
    assert symbol_type == "other"
    assert currency == "USD"
    assert base is None


def __test_detect_symbol_type_edge_cases__():
    """Test edge cases"""
    dc = DataConverter()
    
    # Ambiguous 6-letter that could be forex or other
    symbol_type, currency, base = dc.guess_symbol_type("ABCDEF")
    assert symbol_type == "other"  # Not recognized forex pair
    assert currency == "USD"
    assert base is None
    
    # Crypto with USD (not USDT)
    symbol_type, currency, base = dc.guess_symbol_type("BTCUSD")
    assert symbol_type == "crypto"
    assert currency == "USD"
    assert base == "BTC"
    
    # Test with uppercase (method expects uppercase input)
    symbol_type, currency, base = dc.guess_symbol_type("BTCUSDT")
    assert symbol_type == "crypto"
    assert currency == "USDT"
    assert base == "BTC"
    
    # With underscores (cleaned internally)
    symbol_type, currency, base = dc.guess_symbol_type("EUR_USD")
    assert symbol_type == "forex"
    assert currency == "USD"
    assert base == "EUR"


def __test_detect_symbol_type_forex_with_slash__():
    """Test forex pairs with explicit slash notation"""
    dc = DataConverter()
    
    # These should be detected as forex because both parts are 3-letter currency codes
    symbol_type, currency, base = dc.guess_symbol_type("EUR/USD")
    assert symbol_type == "forex"
    assert currency == "USD"
    assert base == "EUR"
    
    symbol_type, currency, base = dc.guess_symbol_type("GBP/JPY")
    assert symbol_type == "forex"
    assert currency == "JPY"
    assert base == "GBP"
    
    symbol_type, currency, base = dc.guess_symbol_type("AUD/CAD")
    assert symbol_type == "forex"
    assert currency == "CAD"
    assert base == "AUD"


def __test_detect_symbol_type_special_cases__():
    """Test special handling cases"""
    dc = DataConverter()
    
    # BTCUSD vs BTCUSDT - both should be crypto
    symbol_type, currency, base = dc.guess_symbol_type("BTCUSD")
    assert symbol_type == "crypto"
    assert currency == "USD"
    assert base == "BTC"
    
    symbol_type, currency, base = dc.guess_symbol_type("BTCUSDT")
    assert symbol_type == "crypto"
    assert currency == "USDT"
    assert base == "BTC"
    
    # 6-letter code that happens to have a crypto symbol in it
    # but isn't a standard pair
    symbol_type, currency, base = dc.guess_symbol_type("BTCXYZ")
    assert symbol_type == "crypto"  # BTC is detected
    # Currency extraction might vary
    
    # Forex currencies in non-standard order (still detected as forex)
    symbol_type, currency, base = dc.guess_symbol_type("JPYEUR")
    assert symbol_type == "forex"  # Both JPY and EUR are forex currencies
    assert currency == "EUR"
    assert base == "JPY"


# Test runner functions that pytest will find
def test_detect_symbol_type_forex():
    __test_detect_symbol_type_forex__()


def test_detect_symbol_type_crypto():
    __test_detect_symbol_type_crypto__()


def test_detect_symbol_type_other():
    __test_detect_symbol_type_other__()


def test_detect_symbol_type_edge_cases():
    __test_detect_symbol_type_edge_cases__()


def test_detect_symbol_type_forex_with_slash():
    __test_detect_symbol_type_forex_with_slash__()


def test_detect_symbol_type_special_cases():
    __test_detect_symbol_type_special_cases__()