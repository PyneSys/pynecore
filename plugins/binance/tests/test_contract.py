"""Startup contract + safety guards."""
import pytest

from pynecore.core.broker.exceptions import ExchangeCapabilityError
from pynecore.core.broker.validation import validate_plugin_contract

from pynecore_binance.broker import BinanceBroker, BinanceBrokerConfig


def __test_contract_clean__(make_broker):
    broker, _ = make_broker()
    errors, warnings = validate_plugin_contract(broker)
    assert errors == []
    assert warnings == []


def __test_mainnet_refused_without_optin__():
    with pytest.raises(ExchangeCapabilityError, match="MAINNET"):
        BinanceBroker(symbol='BTC/USDT', timeframe='60',
                      config=BinanceBrokerConfig(sandbox=False))


def __test_mainnet_allowed_with_explicit_optin__():
    broker = BinanceBroker(symbol='BTC/USDT', timeframe='60',
                           config=BinanceBrokerConfig(sandbox=False,
                                                      allow_mainnet=True))
    assert broker is not None


def __test_entry_points_discoverable__():
    from pynecore.core.plugin import discover_plugins
    names = discover_plugins()
    assert 'binance' in names
    assert 'binance_broker' in names
