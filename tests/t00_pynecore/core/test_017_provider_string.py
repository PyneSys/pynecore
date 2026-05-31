"""
Tests for provider string parsing.

Provider string format: <provider>:<symbol>@<timeframe>
Examples:
    ccxt:BYBIT:BTC/USDT:USDT@1D
    ccxt:BINANCE:ETH/USDT@4H
    capitalcom:EURUSD@1H
"""

import pytest

from pynecore.core.provider_string import (
    ProviderString,
    is_provider_string,
    parse_provider_string,
)


# --- is_provider_string ---

def __test_is_provider_string_with_ccxt__():
    """CCXT provider string is recognized"""
    assert is_provider_string("ccxt:BYBIT:BTC/USDT:USDT@1D") is True


def __test_is_provider_string_with_capitalcom__():
    """Capital.com provider string is recognized"""
    assert is_provider_string("capitalcom:EURUSD@1H") is True


def __test_is_provider_string_file_path__():
    """Regular file paths are not provider strings"""
    assert is_provider_string("data.csv") is False
    assert is_provider_string("path/to/data.ohlcv") is False
    assert is_provider_string("my_data") is False


def __test_is_provider_string_windows_drive__():
    """Windows drive letters (single char before colon) are not provider strings"""
    assert is_provider_string("C:data.csv") is False
    assert is_provider_string("D:path") is False


def __test_is_provider_string_without_colon__():
    """Strings without colon are not provider strings"""
    assert is_provider_string("ccxt") is False


# --- parse_provider_string ---

def __test_parse_ccxt_futures_with_timeframe__():
    """Parse CCXT futures provider string with timeframe"""
    result = parse_provider_string("ccxt:BYBIT:BTC/USDT:USDT@1D")
    assert result == ProviderString(provider="ccxt", symbol="BYBIT:BTC/USDT:USDT", timeframe="1D")


def __test_parse_ccxt_spot_with_timeframe__():
    """Parse CCXT spot provider string with timeframe"""
    result = parse_provider_string("ccxt:BINANCE:ETH/USDT@4H")
    assert result == ProviderString(provider="ccxt", symbol="BINANCE:ETH/USDT", timeframe="4H")


def __test_parse_capitalcom_with_timeframe__():
    """Parse Capital.com provider string with timeframe"""
    result = parse_provider_string("capitalcom:EURUSD@1H")
    assert result == ProviderString(provider="capitalcom", symbol="EURUSD", timeframe="1H")


def __test_parse_without_timeframe__():
    """Parse provider string without timeframe (for request.security)"""
    result = parse_provider_string("ccxt:BYBIT:BTC/USDT:USDT")
    assert result == ProviderString(provider="ccxt", symbol="BYBIT:BTC/USDT:USDT", timeframe=None)


def __test_parse_minute_timeframe__():
    """Parse provider string with minute timeframe"""
    result = parse_provider_string("ccxt:BINANCE:BTC/USDT@15")
    assert result == ProviderString(provider="ccxt", symbol="BINANCE:BTC/USDT", timeframe="15")


def __test_parse_second_timeframe__():
    """Parse provider string with second timeframe"""
    result = parse_provider_string("ccxt:BINANCE:BTC/USDT@5S")
    assert result == ProviderString(provider="ccxt", symbol="BINANCE:BTC/USDT", timeframe="5S")


def __test_parse_require_timeframe_present__():
    """require_timeframe passes when timeframe is present"""
    result = parse_provider_string("ccxt:BYBIT:BTC/USDT:USDT@1D", require_timeframe=True)
    assert result.timeframe == "1D"


def __test_parse_require_timeframe_missing__():
    """require_timeframe raises ValueError when timeframe is missing"""
    with pytest.raises(ValueError, match="Timeframe is required"):
        parse_provider_string("ccxt:BYBIT:BTC/USDT:USDT", require_timeframe=True)


def __test_parse_no_colon__():
    """String without colon raises ValueError"""
    with pytest.raises(ValueError, match="Invalid provider string"):
        parse_provider_string("ccxt")


def __test_parse_empty_provider__():
    """Empty provider name raises ValueError"""
    with pytest.raises(ValueError, match="Provider name is empty"):
        parse_provider_string(":BYBIT:BTC/USDT@1D")


def __test_parse_empty_symbol__():
    """Empty symbol raises ValueError"""
    with pytest.raises(ValueError, match="Symbol is missing"):
        parse_provider_string("ccxt:")


def __test_parse_empty_timeframe__():
    """Empty timeframe after @ raises ValueError"""
    with pytest.raises(ValueError, match="Timeframe is empty"):
        parse_provider_string("ccxt:BYBIT:BTC/USDT@")


def __test_parse_empty_symbol_with_timeframe__():
    """Empty symbol before @ raises ValueError"""
    with pytest.raises(ValueError, match="Symbol is missing"):
        parse_provider_string("ccxt:@1D")


def __test_provider_string_frozen__():
    """ProviderString is immutable (frozen dataclass)"""
    result = parse_provider_string("ccxt:BINANCE:BTC/USDT@1D")
    with pytest.raises(AttributeError):
        result.provider = "other"


# --- multi_broker parsing ---

def __test_parse_multi_broker_futures__():
    """multi_broker splits the broker off a futures symbol"""
    result = parse_provider_string("ccxt:BYBIT:BTC/USDT:USDT@1D", multi_broker=True)
    assert result == ProviderString(
        provider="ccxt", symbol="BTC/USDT:USDT", timeframe="1D", broker="BYBIT")


def __test_parse_multi_broker_spot__():
    """multi_broker splits the broker off a spot symbol"""
    result = parse_provider_string("ccxt:BINANCE:ETH/USDT@4H", multi_broker=True)
    assert result == ProviderString(
        provider="ccxt", symbol="ETH/USDT", timeframe="4H", broker="BINANCE")


def __test_parse_multi_broker_broker_only__():
    """Broker-only string yields an empty symbol (for listing that broker)"""
    result = parse_provider_string("ccxt:BYBIT", multi_broker=True)
    assert result == ProviderString(
        provider="ccxt", symbol="", timeframe=None, broker="BYBIT")


def __test_parse_multi_broker_broker_only_with_timeframe__():
    """Broker-only string keeps the timeframe and leaves the symbol empty"""
    result = parse_provider_string("ccxt:BYBIT@1D", multi_broker=True)
    assert result == ProviderString(
        provider="ccxt", symbol="", timeframe="1D", broker="BYBIT")


def __test_parse_multi_broker_empty_broker__():
    """Empty broker segment raises ValueError"""
    with pytest.raises(ValueError, match="Broker name is empty"):
        parse_provider_string("ccxt::BTC/USDT@1D", multi_broker=True)


def __test_parse_default_keeps_broker_in_symbol__():
    """Without multi_broker the broker stays folded into the symbol, broker is None"""
    result = parse_provider_string("ccxt:BYBIT:BTC/USDT:USDT@1D")
    assert result.symbol == "BYBIT:BTC/USDT:USDT"
    assert result.broker is None


# --- provider_symbol property ---

def __test_provider_symbol_multi_broker__():
    """provider_symbol re-folds the broker into the symbol"""
    result = parse_provider_string("ccxt:BYBIT:BTC/USDT:USDT@1D", multi_broker=True)
    assert result.provider_symbol == "BYBIT:BTC/USDT:USDT"


def __test_provider_symbol_broker_only__():
    """provider_symbol of a broker-only string is just the broker"""
    result = parse_provider_string("ccxt:BYBIT", multi_broker=True)
    assert result.provider_symbol == "BYBIT"


def __test_provider_symbol_single_broker__():
    """provider_symbol of a single-broker provider is the symbol unchanged"""
    result = parse_provider_string("capitalcom:EURUSD@1H")
    assert result.provider_symbol == "EURUSD"
