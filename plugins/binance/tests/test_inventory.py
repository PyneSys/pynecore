"""Spot inventory port: SpotExecution conversion rules and cursor behavior."""
from decimal import Decimal

from pynecore_binance.inventory import execution_from_trade

from .conftest import trade_row, unified_order


def _convert(trade):
    return execution_from_trade(trade, base_asset='BTC', quote_asset='USDT',
                                client_order_id='ab12-x-y-e0')


def __test_buy_signs_and_quote_fee__():
    execution = _convert(trade_row(1, '10', side='buy', amount=0.001,
                                   price=60_000.0, fee_cost=0.06,
                                   fee_currency='USDT'))
    assert execution.base_delta == Decimal('0.001')
    assert execution.quote_delta == Decimal('-60.06')      # cost + quote fee
    assert execution.venue_seq == 1


def __test_buy_base_fee_reduces_base_delta__():
    execution = _convert(trade_row(2, '10', side='buy', amount=0.001,
                                   price=60_000.0, fee_cost=0.000001,
                                   fee_currency='BTC'))
    assert execution.base_delta == Decimal('0.000999')
    assert execution.quote_delta == Decimal('-60')


def __test_sell_signs__():
    execution = _convert(trade_row(3, '11', side='sell', amount=0.001,
                                   price=60_000.0, fee_cost=0.06,
                                   fee_currency='USDT'))
    assert execution.base_delta == Decimal('-0.001')
    assert execution.quote_delta == Decimal('59.94')


def __test_bnb_fee_touches_neither_delta__():
    execution = _convert(trade_row(4, '12', side='buy', amount=0.001,
                                   price=60_000.0, fee_cost=0.0001,
                                   fee_currency='BNB'))
    assert execution.base_delta == Decimal('0.001')
    assert execution.quote_delta == Decimal('-60')
    assert execution.fee_currency == 'BNB'


def __test_first_startup_anchors_watermark_empty__(make_broker, run):
    broker, fake = make_broker(fetch_my_trades=[trade_row(4242, '99')])
    batch = run(broker._spot_port.fetch_executions(None))
    assert batch.executions == ()
    assert batch.next_cursor == '4242'                     # anchored, not replayed


def __test_cursor_advances_past_foreign_fills__(make_broker, run):
    broker, fake = make_broker(
        fetch_my_trades=lambda symbol, limit=None, params=None:
            [trade_row(100, '55')] if params else [],
        fetch_order=unified_order(order_id='55', coid='not-ours'))
    batch = run(broker._spot_port.fetch_executions('99'))
    assert batch.executions == ()                          # foreign excluded
    assert batch.next_cursor == '100'                      # but cursor advances


def __test_unparsable_cursor_fails_closed__(make_broker, run):
    broker, _ = make_broker()
    batch = run(broker._spot_port.fetch_executions('not-a-number'))
    assert batch.conclusive is False
