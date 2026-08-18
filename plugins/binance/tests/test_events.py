"""Fill / cancel event synthesis from the REST poll."""
from .conftest import trade_row, unified_order


def __test_partial_then_filled_events__(make_broker, entry_envelope, run):
    broker, fake = make_broker(
        create_order=lambda sym, typ, side, amt, price, params:
            unified_order(order_id='501', coid=params['newClientOrderId'],
                          amount=0.002, price=59_000.0))
    run(broker.execute_entry(entry_envelope(limit=59_000.0, qty=0.002)))
    coid = broker._coid_by_order_id['501']

    fake._responses['fetch_my_trades'] = [
        trade_row(9001, '501', amount=0.001, price=59_000.0),
        trade_row(9002, '501', amount=0.001, price=59_000.0),
    ]
    events = run(broker._poll_once())
    assert [event.event_type for event in events] == ['partial', 'filled']
    assert events[0].fill_id == '9001' and events[1].fill_id == '9002'
    assert events[0].fill_qty == 0.001                     # incremental slice
    assert events[1].order.filled_qty == 0.002             # cumulative
    assert all(event.pine_id == 'Long' for event in events)
    assert events[0].order.client_order_id == coid
    assert '501' not in broker._live_ids                   # terminal, unpolled


def __test_foreign_fills_are_ignored__(make_broker, run):
    broker, fake = make_broker(
        fetch_my_trades=[trade_row(9100, '999')],
        fetch_order=unified_order(order_id='999', coid='someone-elses-order'))
    events = run(broker._poll_once())
    assert events == []
    assert broker._trade_cursor == 9100                    # cursor still advances


def __test_cancel_transition_emits_cancelled__(make_broker, entry_envelope, run):
    broker, fake = make_broker(
        create_order=unified_order(order_id='601', amount=0.001),
        fetch_order=unified_order(order_id='601', status='canceled'))
    run(broker.execute_entry(entry_envelope(limit=59_000.0)))
    events = run(broker._poll_once())
    assert [event.event_type for event in events] == ['cancelled']
    assert events[0].pine_id == 'Long'
    assert '601' not in broker._live_ids


def __test_in_memory_position_tracks_fills__(make_broker, entry_envelope, run):
    """No store (tests): fills still synthesize a position for the engine."""
    broker, fake = make_broker(
        create_order=unified_order(order_id='701', amount=0.002))
    run(broker.execute_entry(entry_envelope(qty=0.002)))
    fake._responses['fetch_my_trades'] = [
        trade_row(9200, '701', amount=0.002, price=60_000.0)]
    run(broker._poll_once())
    position = run(broker.get_position('BTC/USDT'))
    assert position is not None
    assert position.side == 'long'
    assert abs(position.size - 0.002) < 1e-12
    assert abs(position.entry_price - 60_000.0) < 1e-6
