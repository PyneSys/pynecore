"""
Tests for LiveProviderPlugin and BarUpdate.
"""

from abc import ABCMeta
from dataclasses import fields

from pynecore.core.plugin import Plugin, ProviderPlugin, LiveProviderPlugin
from pynecore.core.plugin.live_provider import BarUpdate
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


def __test_bar_update_fields__():
    """BarUpdate has ohlcv and is_closed fields"""
    field_names = {f.name for f in fields(BarUpdate)}
    assert field_names == {'ohlcv', 'is_closed'}


def __test_bar_update_creation__():
    """BarUpdate can be created with OHLCV and is_closed flag"""
    ohlcv = OHLCV(timestamp=1000, open=100.0, high=105.0, low=95.0, close=102.0, volume=1000.0)

    closed = BarUpdate(ohlcv=ohlcv, is_closed=True)
    assert closed.ohlcv is ohlcv
    assert closed.is_closed is True

    update = BarUpdate(ohlcv=ohlcv, is_closed=False)
    assert update.is_closed is False
