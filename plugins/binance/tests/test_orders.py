"""Intent -> venue order mapping, preflight and cancel semantics."""
import pytest

import ccxt

from pynecore.core.broker.exceptions import OrderSkippedByPlugin
from pynecore.core.broker.models import CancelDispositionOutcome, LegType

from .conftest import unified_order


def __test_entry_market_carries_client_id__(make_broker, entry_envelope, run):
    broker, fake = make_broker(
        create_order=lambda sym, typ, side, amt, price, params:
            unified_order(coid=params['newClientOrderId'], order_type=typ))
    orders = run(broker.execute_entry(entry_envelope()))
    args, _ = fake.last('create_order')
    assert args[1] == 'market' and args[2] == 'buy'
    assert args[5]['newClientOrderId']            # engine-minted id present
    assert orders[0].client_order_id == args[5]['newClientOrderId']


def __test_entry_limit_uses_gtc_limit__(make_broker, entry_envelope, run):
    broker, fake = make_broker(create_order=unified_order(price=59_000.0))
    run(broker.execute_entry(entry_envelope(limit=59_000.0)))
    args, _ = fake.last('create_order')
    assert args[1] == 'limit'
    assert args[4] == 59_000.0
    assert args[5]['timeInForce'] == 'GTC'
    assert 'stopPrice' not in args[5]


def __test_entry_stop_prices_through_trigger__(make_broker, entry_envelope, run):
    broker, fake = make_broker(create_order=unified_order(stop_price=61_000.0))
    run(broker.execute_entry(entry_envelope(stop=61_000.0)))
    args, _ = fake.last('create_order')
    assert args[1] == 'limit'
    assert args[5]['stopPrice'] == 61_000.0
    # buy stop: the limit sits ABOVE the trigger by stop_slippage_ticks.
    assert args[4] > 61_000.0


def __test_exit_oco_places_native_list__(make_broker, exit_envelope, run):
    def oco(payload):
        return {'orderListId': 777, 'orderReports': [
            unified_order(order_id='201', coid=payload['aboveClientOrderId'],
                          side='sell', price=float(payload['abovePrice'])),
            unified_order(order_id='202', coid=payload['belowClientOrderId'],
                          side='sell', price=float(payload['belowPrice']),
                          stop_price=float(payload['belowStopPrice'])),
        ]}
    broker, fake = make_broker(private_post_orderlist_oco=oco)
    orders = run(broker.execute_exit(exit_envelope(tp_price=65_000.0,
                                                   sl_price=58_000.0)))
    (payload,), _ = fake.last('private_post_orderlist_oco')
    assert payload['symbol'] == 'BTCUSDT' and payload['side'] == 'SELL'
    assert payload['aboveType'] == 'LIMIT_MAKER'
    assert payload['belowType'] == 'STOP_LOSS_LIMIT'
    assert float(payload['belowPrice']) < float(payload['belowStopPrice'])
    legs = {broker._identity[order.id][2] for order in orders}
    assert legs == {LegType.TAKE_PROFIT, LegType.STOP_LOSS}
    assert all(broker._oco_list_id[order.id] == '777' for order in orders)


def __test_preflight_skips_below_lot_size__(make_broker, entry_envelope, run):
    broker, fake = make_broker()
    with pytest.raises(OrderSkippedByPlugin, match="LOT_SIZE"):
        run(broker.execute_entry(entry_envelope(qty=1e-7)))
    assert fake.count('create_order') == 0


def __test_preflight_skips_below_min_notional__(make_broker, entry_envelope, run):
    broker, fake = make_broker()
    # 0.00004 BTC @ 60k = 2.4 USDT < 5 USDT MIN_NOTIONAL
    with pytest.raises(OrderSkippedByPlugin, match="MIN_NOTIONAL"):
        run(broker.execute_entry(entry_envelope(limit=60_000.0, qty=0.00004)))
    assert fake.count('create_order') == 0


def __test_cancel_unknown_order_verifies_disposition__(
        make_broker, entry_envelope, run):
    """-2011 "Unknown order" counts as gone ONLY once fetch_order agrees."""
    def raise_gone(*args, **kwargs):
        raise ccxt.OrderNotFound('binance -2011 Unknown order sent.')
    broker, fake = make_broker(
        create_order=unified_order(order_id='301'),
        cancel_order=raise_gone,
        fetch_order=unified_order(order_id='301', status='canceled'))
    envelope = entry_envelope(limit=59_000.0)
    run(broker.execute_entry(envelope))
    from pynecore.core.broker.models import CancelIntent
    from .conftest import envelope_for
    cancel = envelope_for(CancelIntent(pine_id='Long', symbol='BTC/USDT'))
    assert run(broker.execute_cancel(cancel)) is True
    assert fake.count('fetch_order') == 1


def __test_cancel_outcome_already_filled__(make_broker, entry_envelope, run):
    def raise_gone(*args, **kwargs):
        raise ccxt.OrderNotFound('binance -2011 Unknown order sent.')
    broker, fake = make_broker(
        create_order=unified_order(order_id='302'),
        cancel_order=raise_gone,
        fetch_order=unified_order(order_id='302', status='closed',
                                  filled=0.001))
    run(broker.execute_entry(entry_envelope(limit=59_000.0)))
    from pynecore.core.broker.models import CancelIntent
    from .conftest import envelope_for
    cancel = envelope_for(CancelIntent(pine_id='Long', symbol='BTC/USDT'))
    assert (run(broker.execute_cancel_with_outcome(cancel))
            is CancelDispositionOutcome.ALREADY_FILLED)


def __test_oco_cancel_targets_the_list_once__(make_broker, exit_envelope, run):
    def oco(payload):
        return {'orderListId': 888, 'orderReports': [
            unified_order(order_id='401', coid=payload['aboveClientOrderId'],
                          side='sell'),
            unified_order(order_id='402', coid=payload['belowClientOrderId'],
                          side='sell', stop_price=58_000.0),
        ]}
    broker, fake = make_broker(private_post_orderlist_oco=oco)
    run(broker.execute_exit(exit_envelope(tp_price=65_000.0, sl_price=58_000.0)))
    from pynecore.core.broker.models import CancelIntent
    from .conftest import envelope_for
    cancel = envelope_for(CancelIntent(pine_id='Exit', symbol='BTC/USDT',
                                       from_entry='Long'))
    assert run(broker.execute_cancel(cancel)) is True
    assert fake.count('private_delete_orderlist') == 1     # once, not per leg
    assert fake.count('cancel_order') == 0


def __test_crossed_stop_falls_back_to_market__(make_broker, entry_envelope, run):
    """Venue -2010 "would trigger immediately" -> MARKET with the same coid."""
    def create(sym, typ, side, amt, price, params):
        if typ == 'limit' and 'stopPrice' in params:
            raise ccxt.OrderImmediatelyFillable(
                'binance Stop price would trigger immediately.')
        return unified_order(order_id='801', coid=params['newClientOrderId'],
                             order_type=typ)
    broker, fake = make_broker(create_order=create)
    orders = run(broker.execute_entry(entry_envelope(stop=60_100.0)))
    args, _ = fake.last('create_order')
    assert args[1] == 'market'
    assert fake.count('create_order') == 2
    stop_args = fake.calls[[c[0] for c in fake.calls].index('create_order')][1]
    assert stop_args[5]['newClientOrderId'] == args[5]['newClientOrderId']
    assert orders[0].client_order_id == args[5]['newClientOrderId']


def __test_crossed_stop_limit_falls_back_to_limit__(make_broker, entry_envelope, run):
    """A crossed stop-LIMIT keeps its price cap: fallback is LIMIT, not market."""
    def create(sym, typ, side, amt, price, params):
        if 'stopPrice' in params:
            raise ccxt.OrderImmediatelyFillable(
                'binance Stop price would trigger immediately.')
        return unified_order(order_id='802', coid=params['newClientOrderId'],
                             order_type=typ, price=price)
    broker, fake = make_broker(create_order=create)
    run(broker.execute_entry(entry_envelope(stop=60_100.0, limit=60_150.0)))
    args, _ = fake.last('create_order')
    assert args[1] == 'limit'
    assert args[4] == 60_150.0
    assert 'stopPrice' not in args[5]
