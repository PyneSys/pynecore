"""
Tests for LiveProviderPlugin and OHLCV is_closed field.
"""

from abc import ABCMeta

from pynecore.core.plugin import Plugin, ProviderPlugin, LiveProviderPlugin
from pynecore.types.ohlcv import OHLCV


def __test_live_provider_inherits_from_provider__():
    """LiveProviderPlugin is a subclass of ProviderPlugin"""
    assert issubclass(LiveProviderPlugin, ProviderPlugin)
    assert issubclass(LiveProviderPlugin, Plugin)


def __test_live_provider_is_abstract__():
    """LiveProviderPlugin cannot be instantiated directly"""
    assert isinstance(LiveProviderPlugin, ABCMeta)


def __test_live_provider_has_abstract_methods__():
    """LiveProviderPlugin requires connect, disconnect, is_connected, watch_ohlcv"""
    abstract_methods = LiveProviderPlugin.__abstractmethods__
    assert 'connect' in abstract_methods
    assert 'disconnect' in abstract_methods
    assert 'is_connected' in abstract_methods
    assert 'watch_ohlcv' in abstract_methods


def __test_live_provider_inherits_provider_abstract_methods__():
    """LiveProviderPlugin also requires ProviderPlugin abstract methods"""
    abstract_methods = LiveProviderPlugin.__abstractmethods__
    assert 'download_ohlcv' in abstract_methods
    assert 'to_tradingview_timeframe' in abstract_methods
    assert 'to_exchange_timeframe' in abstract_methods
    assert 'get_list_of_symbols' in abstract_methods
    assert 'update_symbol_info' in abstract_methods


def __test_live_provider_default_reconnect_values__():
    """LiveProviderPlugin has default reconnect configuration"""
    assert LiveProviderPlugin.reconnect_delay == 1.0
    assert LiveProviderPlugin.max_reconnect_attempts == 10


def __test_ohlcv_is_closed_field__():
    """OHLCV has is_closed field with default True"""
    ohlcv = OHLCV(timestamp=1000, open=100.0, high=105.0, low=95.0, close=102.0, volume=1000.0)
    assert ohlcv.is_closed is True

    closed = ohlcv._replace(is_closed=True)
    assert closed.is_closed is True

    update = ohlcv._replace(is_closed=False)
    assert update.is_closed is False
