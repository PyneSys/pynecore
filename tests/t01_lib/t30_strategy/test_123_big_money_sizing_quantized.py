"""
Regression test: default sizing quantizes its money budget at every magnitude.

TradingView rounds the money side of default sizing to 10 significant decimal
digits before dividing it by the unit cost. That was measured below 1e7 first
(BINANCE:SHIBUSDT 1h flat-cycle sweeps) and used to be applied only there; the
wild corpus script `Fractal Breakout Strategy [KL]` (BINANCE:BTCUSDT 30m, 5% of
a 1e9 account, so a ~5e7 budget) shows the same grid above 1e7. Four of its
sizing events disagreed with the raw quotient, and TradingView's own
full-precision ``strategy.equity`` plot supplies the budgets used here.

Quantizing is not free to overshoot: `Hybrid: RSI + Breakout + Dashboard`
(100% percent_of_equity) buys a position the account cannot margin once the
budget rounds up, and TradingView refuses to open it.
"""
import math

# noinspection PyProtectedMember
from pynecore.lib.strategy import _gate_entry_lots, _money_ticks, _sig10_money

MINTICK = 0.01
RFACTOR = 100000.0

#: Fractal Breakout entries whose contract count the raw quotient floors one lot
#: short: (bar, budget, unit cost, contracts TradingView sized in lots).
SNAP_CASES = (
    ('2025-03-11 05:30', 50339185.945264585, 80261.78, 62718751),
    ('2026-01-31 08:00', 49966033.61529095, 83097.99, 60129052),
    ('2026-03-29 10:00', 49869517.34895752, 66468.21, 75027622),
)


def __test_quantized_budget_lifts_the_lot_count__():
    """The raw quotient floors one lot short of what TradingView sized."""
    for label, money, unit_cost, lots in SNAP_CASES:
        assert math.floor(money / unit_cost * RFACTOR) == lots - 1, label
        assert math.floor(_sig10_money(money) / unit_cost * RFACTOR) == lots, label


def __test_gate_grants_on_the_quantized_budget__():
    """Fractal Breakout 2026-08-25 18:30.

    The floored size is the same either way; the raw budget lands 0.05 tick
    under the cost's grid ceiling on an EVEN multiple, which the gate rejects,
    while the quantized budget lands exactly on it and TradingView fills the
    full 632.93356 contracts.
    """
    money, unit_cost, lots = 50082766.739484586, 79128.0, 63293356
    assert math.floor(_sig10_money(money) / unit_cost * RFACTOR) == lots
    assert _gate_entry_lots(money / MINTICK, lots, RFACTOR, unit_cost,
                            MINTICK, unit_cost) is None
    assert _gate_entry_lots(_money_ticks(money, MINTICK), lots, RFACTOR, unit_cost,
                            MINTICK, unit_cost) == lots


def __test_bumped_gate_trims_back_to_the_floor__():
    """Hybrid 2026-01-29 15:30.

    The budget reaches the next lot's grid threshold, so the gate is judged one
    lot larger; its own cost then exceeds the budget and the odd-multiple parity
    branch hands back the floored size — 1212.66322 contracts, as TradingView
    sized it.
    """
    money, unit_cost = 102996363.0495567, 84934.02
    lots = math.floor(_sig10_money(money) / unit_cost * RFACTOR)
    assert lots == 121266322
    assert _gate_entry_lots(_money_ticks(money, MINTICK), lots + 1, RFACTOR,
                            unit_cost, MINTICK, unit_cost) == lots


def __test_creation_time_gate_rejects_a_quantized_oversize__():
    """Hybrid 2026-05-14 14:30.

    At 100% percent_of_equity the rounded-up budget buys a position costing more
    than the account holds. The sizing gate, reading the quantized budget, is
    happy with it; the creation-time margin check reads the RAW equity and
    refuses to open — TradingView closes the reversed short at the next open and
    leaves the long unopened.
    """
    equity, check_price = 468246869.3555109, 80964.85
    lots = math.floor(_sig10_money(equity) / check_price * RFACTOR)
    assert lots == 578333523
    assert _gate_entry_lots(_money_ticks(equity, MINTICK), lots, RFACTOR,
                            check_price, MINTICK, check_price) == lots
    assert _gate_entry_lots(equity / MINTICK, lots, RFACTOR, check_price,
                            MINTICK, check_price) is None
