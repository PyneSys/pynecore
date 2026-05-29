"""
Tests for :func:`pynecore.core.broker.intent_builder.build_intents`.

The builder is a pure translation of the Pine ``Order`` objects that
``strategy.entry/exit/order/close/close_all`` create into the broker
:class:`EntryIntent` / :class:`ExitIntent` / :class:`CloseIntent` dataclasses.
Tests construct ``Order`` instances directly — no ScriptRunner, no Pine
function call — and assert the resulting intent fields.
"""
from __future__ import annotations

from pynecore.core.broker.intent_builder import build_intents
from pynecore.core.broker.models import (
    EntryIntent,
    ExitIntent,
    CloseIntent,
    OrderType,
)
from pynecore.lib.strategy import (
    Order,
    Trade,
    _order_type_normal,
    _order_type_entry,
    _order_type_close,
)
from pynecore.lib.strategy import oca as _oca

SYMBOL = "BTCUSDT"


def _entry(order_id, size, **kw) -> Order:
    return Order(order_id, size, order_type=_order_type_entry, **kw)


def _normal(order_id, size, **kw) -> Order:
    return Order(order_id, size, order_type=_order_type_normal, **kw)


def _exit(from_entry, size, exit_id, **kw) -> Order:
    return Order(from_entry, size, order_type=_order_type_close,
                 exit_id=exit_id, **kw)


def _close(id_, size) -> Order:
    return Order(id_, size, order_type=_order_type_close,
                 exit_id=f"Close entry(s) order {id_}")


def _close_all(size) -> Order:
    return Order(None, size, order_type=_order_type_close,
                 exit_id="Close position order")


def _trade(entry_id, size) -> Trade:
    return Trade(size=size, entry_id=entry_id, entry_bar_index=0,
                 entry_time=0, entry_price=100.0, commission=0.0)


def _bracket_exit(from_entry, size) -> Order:
    return _exit(from_entry, size, "TP", limit=60_000.0, stop=45_000.0)


def _only_exit(intents) -> ExitIntent:
    return next(i for i in intents if isinstance(i, ExitIntent))


# === Entry / Order ===

def __test_market_entry_produces_market_intent__():
    intents = build_intents({"L": _entry("L", 1.0)}, {}, SYMBOL)
    assert intents == [EntryIntent(
        pine_id="L", symbol=SYMBOL, side="buy", qty=1.0,
        order_type=OrderType.MARKET,
    )]


def __test_limit_entry__():
    intents = build_intents({"L": _entry("L", 1.0, limit=50_000.0)}, {}, SYMBOL)
    i = intents[0]
    assert isinstance(i, EntryIntent)
    assert i.order_type is OrderType.LIMIT
    assert i.limit == 50_000.0 and i.stop is None


def __test_stop_entry__():
    intents = build_intents({"L": _entry("L", 1.0, stop=49_000.0)}, {}, SYMBOL)
    i = intents[0]
    assert i.order_type is OrderType.STOP
    assert i.stop == 49_000.0 and i.limit is None


def __test_stop_limit_entry__():
    intents = build_intents(
        {"L": _entry("L", 1.0, limit=50_000.0, stop=49_500.0)}, {}, SYMBOL,
    )
    i = intents[0]
    assert i.order_type is OrderType.STOP_LIMIT
    assert i.limit == 50_000.0 and i.stop == 49_500.0


def __test_short_entry_side_is_sell__():
    intents = build_intents({"S": _entry("S", -1.0)}, {}, SYMBOL)
    assert intents[0].side == "sell"


def __test_qty_is_absolute__():
    intents = build_intents({"S": _entry("S", -2.5)}, {}, SYMBOL)
    assert intents[0].qty == 2.5


def __test_strategy_order_sets_is_strategy_order__():
    intents = build_intents({"X": _normal("X", 1.0)}, {}, SYMBOL)
    assert intents[0].is_strategy_order is True


def __test_strategy_entry_is_not_strategy_order__():
    intents = build_intents({"L": _entry("L", 1.0)}, {}, SYMBOL)
    assert intents[0].is_strategy_order is False


# === OCA propagation ===

def __test_oca_group_is_propagated__():
    e = _entry("L", 1.0, limit=50_000.0, oca_name="grp", oca_type=_oca.cancel)
    i = build_intents({"L": e}, {}, SYMBOL)[0]
    assert i.oca_name == "grp"
    assert i.oca_type == "cancel"


def __test_no_oca_means_both_none__():
    i = build_intents({"L": _entry("L", 1.0, limit=50_000.0)}, {}, SYMBOL)[0]
    assert i.oca_name is None and i.oca_type is None


# === Exit ===

def __test_exit_with_prices_maps_tp_sl__():
    e = _exit("L", -1.0, "TP", limit=60_000.0, stop=45_000.0)
    i = build_intents({}, {"L": e}, SYMBOL)[0]
    assert isinstance(i, ExitIntent)
    assert i.pine_id == "TP" and i.from_entry == "L"
    assert i.tp_price == 60_000.0 and i.sl_price == 45_000.0
    assert i.profit_ticks is None and i.loss_ticks is None
    assert i.has_unresolved_ticks is False
    assert i.intent_key == "TP\0L"


def __test_exit_with_ticks_defers_resolution__():
    e = _exit("L", -1.0, "TP", profit_ticks=100.0, loss_ticks=50.0)
    i = build_intents({}, {"L": e}, SYMBOL)[0]
    assert isinstance(i, ExitIntent)
    assert i.tp_price is None and i.sl_price is None
    assert i.profit_ticks == 100.0 and i.loss_ticks == 50.0
    assert i.has_unresolved_ticks is True


def __test_exit_ticks_override_explicit_prices__():
    # If both are syntactically present, Pine uses ticks at fill time.
    e = _exit("L", -1.0, "TP",
              limit=60_000.0, stop=45_000.0,
              profit_ticks=100.0, loss_ticks=50.0)
    i = build_intents({}, {"L": e}, SYMBOL)[0]
    assert i.tp_price is None and i.sl_price is None
    assert i.profit_ticks == 100.0 and i.loss_ticks == 50.0


def __test_exit_with_trailing__():
    e = _exit("L", -1.0, "TR", trail_price=55_000.0, trail_offset=50)
    i = build_intents({}, {"L": e}, SYMBOL)[0]
    assert i.trail_price == 55_000.0
    assert i.trail_offset == 50


def __test_exit_without_trailing_has_null_trail_offset__():
    # Order.__init__ defaults trail_offset to 0; the intent should expose
    # None when no trailing context exists, so a plugin doesn't mistake
    # a plain TP/SL for a zero-offset trailing stop.
    e = _exit("L", -1.0, "TP", limit=60_000.0)
    i = build_intents({}, {"L": e}, SYMBOL)[0]
    assert i.trail_offset is None and i.trail_price is None


def __test_exit_tick_trailing__():
    e = _exit("L", -1.0, "TR", trail_points_ticks=100.0, trail_offset=25)
    i = build_intents({}, {"L": e}, SYMBOL)[0]
    assert i.trail_price is None
    assert i.trail_points_ticks == 100.0
    assert i.trail_offset == 25
    assert i.has_unresolved_ticks is True


# === Close / close_all ===

def __test_close_produces_close_intent__():
    i = build_intents({}, {"L": _close("L", -1.0)}, SYMBOL)[0]
    assert isinstance(i, CloseIntent)
    assert i.pine_id == "L" and i.side == "sell"


def __test_close_all_has_empty_pine_id__():
    i = build_intents({}, {None: _close_all(-1.0)}, SYMBOL)[0]
    assert isinstance(i, CloseIntent)
    assert i.pine_id == ""


# === Cancellation / filtering ===

def __test_cancelled_orders_are_skipped__():
    o = _entry("L", 1.0)
    o.cancelled = True
    assert build_intents({"L": o}, {}, SYMBOL) == []


def __test_mixed_entry_and_exit_produce_both_intents__():
    e = _entry("L", 1.0, limit=50_000.0)
    x = _exit("L", -1.0, "TP", limit=60_000.0, stop=45_000.0)
    intents = build_intents({"L": e}, {"L": x}, SYMBOL)
    kinds = [type(i).__name__ for i in intents]
    assert kinds == ["EntryIntent", "ExitIntent"]


# === reduce_only invariant (WS2) ===

def __test_exit_intent_defaults_to_reduce_only_true__():
    x = _exit("L", -1.0, "TP", limit=60_000.0, stop=45_000.0)
    i = build_intents({}, {"L": x}, SYMBOL)[0]
    assert isinstance(i, ExitIntent)
    assert i.reduce_only is True


def __test_close_intent_defaults_to_reduce_only_true__():
    i = build_intents({}, {"L": _close("L", -1.0)}, SYMBOL)[0]
    assert isinstance(i, CloseIntent)
    assert i.reduce_only is True


def __test_close_all_intent_defaults_to_reduce_only_true__():
    i = build_intents({}, {None: _close_all(-1.0)}, SYMBOL)[0]
    assert isinstance(i, CloseIntent)
    assert i.reduce_only is True


def __test_exit_intent_rejects_reduce_only_false__():
    import pytest
    with pytest.raises(ValueError, match="reduce_only must be True"):
        ExitIntent(
            pine_id="TP", from_entry="L", symbol=SYMBOL,
            side="sell", qty=1.0, reduce_only=False,
        )


def __test_close_intent_rejects_reduce_only_false__():
    import pytest
    with pytest.raises(ValueError, match="reduce_only must be True"):
        CloseIntent(
            pine_id="L", symbol=SYMBOL, side="sell", qty=1.0,
            reduce_only=False,
        )


# === Partial-qty bracket: parent-total sourcing (pyramiding vs single-row) ===
#
# ``build_intents`` reads the simulator ``SimPosition.open_trades`` — one
# ``Trade`` row per Pine entry event (no split fills). The open row count under
# a ``from_entry`` id therefore distinguishes pyramiding/strategy.order adds
# (>1 row → summed parent qty) from a single entry (==1 row → script-declared
# qty, immune to broker fill noise).

def __test_pyramiding_partial_bracket_uses_summed_parent_qty__():
    # Two same-id open_trades (pyramided to 2.0); the latest declared entry row
    # is only 1.0. A bracket exit of qty 1.0 is partial against the AGGREGATE,
    # not the declared-latest — proving the multi-row branch wins.
    entry = {"L": _entry("L", 1.0)}
    ex = {"L": _bracket_exit("L", -1.0)}
    open_trades = [_trade("L", 1.0), _trade("L", 1.0)]
    x = _only_exit(build_intents(entry, ex, SYMBOL, open_trades))
    assert x.is_partial_qty_bracket is True


def __test_pyramiding_whole_row_exit_is_not_partial__():
    # Exit qty equals the summed parent (2.0) → whole-row, not partial.
    entry = {"L": _entry("L", 1.0)}
    ex = {"L": _bracket_exit("L", -2.0)}
    open_trades = [_trade("L", 1.0), _trade("L", 1.0)]
    x = _only_exit(build_intents(entry, ex, SYMBOL, open_trades))
    assert x.is_partial_qty_bracket is False


def __test_single_row_overfill_stays_whole_row__():
    # One open_trade overfilled to 1.2 vs declared 1.0. A whole-row exit of the
    # declared 1.0 must stay whole-row (declared wins on single-row), so broker
    # fill noise cannot flip the flag and churn the native<->partial bracket.
    entry = {"L": _entry("L", 1.0)}
    ex = {"L": _bracket_exit("L", -1.0)}
    open_trades = [_trade("L", 1.2)]
    x = _only_exit(build_intents(entry, ex, SYMBOL, open_trades))
    assert x.is_partial_qty_bracket is False


def __test_single_row_partial_bracket_uses_declared_qty__():
    # Single entry of 2.0, bracket exit of 1.0 → partial against declared 2.0.
    entry = {"L": _entry("L", 2.0)}
    ex = {"L": _bracket_exit("L", -1.0)}
    open_trades = [_trade("L", 2.0)]
    x = _only_exit(build_intents(entry, ex, SYMBOL, open_trades))
    assert x.is_partial_qty_bracket is True


def __test_partial_flag_false_without_bracket_leg__():
    # A bare reduce (no TP/SL/trail) under pyramiding is never a partial-qty
    # *bracket* — the flag gates bracket dispatch, not plain partial closes.
    entry = {"L": _entry("L", 1.0)}
    ex = {"L": _exit("L", -1.0, "X")}
    open_trades = [_trade("L", 1.0), _trade("L", 1.0)]
    x = _only_exit(build_intents(entry, ex, SYMBOL, open_trades))
    assert x.is_partial_qty_bracket is False
