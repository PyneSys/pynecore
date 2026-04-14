"""
Unit tests for the broker plugin data models.

Focus on the stable diff keys (:attr:`intent_key`) used by the Order Sync
Engine and the tick-unresolved flag on :class:`ExitIntent`.
"""
from pynecore.core.broker.models import (
    OrderType,
    EntryIntent,
    ExitIntent,
    CloseIntent,
    CancelIntent,
    ScriptRequirements,
)


def __test_entry_intent_key_is_pine_id__():
    intent = EntryIntent(
        pine_id="Long",
        symbol="BTCUSDT",
        side="buy",
        qty=1.0,
        order_type=OrderType.MARKET,
    )
    assert intent.intent_key == "Long"


def __test_exit_intent_key_combines_pine_id_and_from_entry__():
    """
    Two strategy.exit(id="TP") calls with different from_entry values must
    produce different intent_keys — otherwise the sync engine would treat
    them as the same intent and lose one.
    """
    a = ExitIntent(pine_id="TP", from_entry="Long", symbol="BTCUSDT",
                   side="sell", qty=1.0)
    b = ExitIntent(pine_id="TP", from_entry="Short", symbol="BTCUSDT",
                   side="buy", qty=1.0)
    assert a.intent_key != b.intent_key
    assert a.intent_key == "TP\0Long"
    assert b.intent_key == "TP\0Short"


def __test_exit_intent_has_unresolved_ticks__():
    """
    Tick-based exit against an unfilled entry cannot compute absolute
    prices yet; the sync engine must defer until the entry fills.
    """
    resolved = ExitIntent(pine_id="TP", from_entry="L", symbol="S",
                          side="sell", qty=1.0, tp_price=50000.0)
    unresolved = ExitIntent(pine_id="TP", from_entry="L", symbol="S",
                            side="sell", qty=1.0, profit_ticks=100.0)
    assert resolved.has_unresolved_ticks is False
    assert unresolved.has_unresolved_ticks is True


def __test_close_intent_key_is_pine_id__():
    intent = CloseIntent(pine_id="Long", symbol="BTCUSDT",
                         side="sell", qty=1.0)
    assert intent.intent_key == "Long"


def __test_cancel_intent_key_with_and_without_from_entry__():
    bare = CancelIntent(pine_id="TP", symbol="BTCUSDT")
    scoped = CancelIntent(pine_id="TP", symbol="BTCUSDT", from_entry="Long")
    assert bare.intent_key == "TP"
    assert scoped.intent_key == "TP\0Long"


def __test_script_requirements_defaults_all_false__():
    """
    The AST detector should only set flags for features the script uses —
    the default must start with everything off.
    """
    reqs = ScriptRequirements()
    assert not any([
        reqs.market_orders, reqs.limit_orders, reqs.stop_orders,
        reqs.stop_limit_orders, reqs.tp_sl_bracket,
        reqs.trailing_stop, reqs.strategy_order,
    ])
