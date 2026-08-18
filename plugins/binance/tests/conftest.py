"""Shared fixtures for Binance plugin unit tests — no live network.

Every venue call funnels through the ccxt client, so a single fake injected as
``broker._client`` intercepts the whole REST surface. Canned replies are set
per method; every call is recorded in ``.calls``. Test functions use the repo
convention ``__test_*__`` (see ``pytest.ini``).
"""
import asyncio
from decimal import Decimal

import pytest

import pynecore.lib as lib
from pynecore.core.broker.models import DispatchEnvelope, EntryIntent, ExitIntent, OrderType

from pynecore_binance.broker import BinanceBroker, BinanceBrokerConfig
from pynecore_binance.inventory import BinanceSpotPort

lib.bar_index = 0  # let the [BROKER] log formatter render

MARKET = {
    'id': 'BTCUSDT', 'base': 'BTC', 'quote': 'USDT', 'symbol': 'BTC/USDT',
    'limits': {'amount': {'min': 1e-5}, 'cost': {'min': 5.0}},
    'precision': {'price': 0.01, 'amount': 1e-5},
}

RUN_TAG = 'ab12'
BAR_TS_MS = 1_755_000_000_000


class FakeCCXT:
    """Configurable stand-in for the ccxt binance client.

    ``FakeCCXT(create_order={'id': '1', ...})`` — a response may be a canned
    value or a callable receiving the call's args. Unset methods return ``{}``
    (or ``[]`` for list-returning reads). Calls are recorded in ``.calls``.
    """

    _LIST_METHODS = {'fetch_open_orders', 'fetch_my_trades'}

    def __init__(self, **responses):
        self._responses = responses
        self.calls = []
        self.markets = {'BTC/USDT': MARKET}

    def market(self, symbol):
        return self.markets[symbol]

    def price_to_precision(self, symbol, price):
        return f"{float(price):.2f}"

    def amount_to_precision(self, symbol, amount):
        return f"{float(amount):.5f}"

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)

        def _call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            resp = self._responses.get(name)
            if callable(resp):
                return resp(*args, **kwargs)
            if resp is not None:
                return resp
            return [] if name in self._LIST_METHODS else {}

        return _call

    def count(self, method):
        return sum(1 for call in self.calls if call[0] == method)

    def last(self, method):
        for name, args, kwargs in reversed(self.calls):
            if name == method:
                return args, kwargs
        raise AssertionError(f"no call to {method}")


@pytest.fixture
def make_broker():
    """Factory: a broker with the fake client injected and startup pre-seeded
    (no network; no store — the in-memory inventory fallback is active)."""
    def _make(**responses):
        broker = BinanceBroker(symbol='BTC/USDT', timeframe='60',
                               config=BinanceBrokerConfig(sandbox=True))
        fake = FakeCCXT(**responses)
        broker._client = fake
        broker._market = MARKET
        broker._spot_port = BinanceSpotPort(broker, MARKET)
        broker.spot_inventory_port = broker._spot_port
        broker._trade_cursor = 0
        broker._broker_started = True
        broker._last_price = 60_000.0
        return broker, fake
    return _make


def envelope_for(intent) -> DispatchEnvelope:
    return DispatchEnvelope(intent=intent, run_tag=RUN_TAG,
                            bar_ts_ms=BAR_TS_MS, coid_max_len=36)


@pytest.fixture
def entry_envelope():
    def _make(side='buy', qty=0.001, limit=None, stop=None):
        order_type = (OrderType.STOP if stop is not None
                      else OrderType.LIMIT if limit is not None
                      else OrderType.MARKET)
        return envelope_for(EntryIntent(
            pine_id='Long', symbol='BTC/USDT', side=side, qty=qty,
            order_type=order_type, limit=limit, stop=stop))
    return _make


@pytest.fixture
def exit_envelope():
    def _make(qty=0.001, tp_price=None, sl_price=None):
        return envelope_for(ExitIntent(
            pine_id='Exit', from_entry='Long', symbol='BTC/USDT', side='sell',
            qty=qty, tp_price=tp_price, sl_price=sl_price))
    return _make


@pytest.fixture
def run():
    """Run a coroutine to completion."""
    return lambda coro: asyncio.run(coro)


def unified_order(order_id='101', coid='', status='open', side='buy',
                  amount=0.001, filled=0.0, price=None, stop_price=None,
                  order_type='limit'):
    return {
        'id': order_id, 'clientOrderId': coid, 'status': status, 'side': side,
        'amount': amount, 'filled': filled, 'price': price,
        'stopPrice': stop_price, 'type': order_type,
        'timestamp': BAR_TS_MS, 'info': {},
    }


def trade_row(trade_id, order_id, side='buy', amount=0.001, price=60_000.0,
              fee_cost=0.0, fee_currency='USDT'):
    return {
        'id': str(trade_id), 'order': str(order_id), 'side': side,
        'amount': amount, 'price': price, 'cost': amount * price,
        'timestamp': BAR_TS_MS,
        'fee': {'cost': fee_cost, 'currency': fee_currency},
    }
