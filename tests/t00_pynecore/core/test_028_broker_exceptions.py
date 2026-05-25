"""
Tests for the WS3 broker-exception taxonomy additions.

Covers:
- ``AuthenticationError`` construction and reason echo.
- ``InsufficientMarginError`` as a typed ``ExchangeOrderRejectedError``.
- ``AuthenticationFailedEvent`` dataclass.
- ``BrokerPlugin._map_exception`` default stdlib mapping.
"""
from __future__ import annotations

from pynecore.core.broker.exceptions import (
    AuthenticationError,
    BracketAttachAfterFillRejectedError,
    BrokerError,
    ExchangeConnectionError,
    ExchangeOrderRejectedError,
    InsufficientMarginError,
)
from pynecore.core.broker.models import (
    AuthenticationFailedEvent,
    BrokerEvent,
    ExchangeOrder,
    OrderStatus,
    OrderType,
)


# === AuthenticationError ===

def __test_authentication_error_is_broker_error__():
    exc = AuthenticationError("Invalid API key")
    assert isinstance(exc, BrokerError)


def __test_authentication_error_reason_defaults_to_message__():
    exc = AuthenticationError("API key revoked")
    assert exc.reason == "API key revoked"


def __test_authentication_error_reason_can_be_distinct__():
    exc = AuthenticationError(
        "Broker authentication failed at startup — cannot begin trading: bad key",
        reason="bad key",
    )
    assert exc.reason == "bad key"
    assert "bad key" in str(exc)


# === InsufficientMarginError ===

def __test_insufficient_margin_is_rejected_error__():
    order = ExchangeOrder(
        id="1", symbol="BTCUSDT", side="buy", order_type=OrderType.MARKET,
        qty=10.0, filled_qty=0.0, remaining_qty=10.0,
        price=None, stop_price=None, average_fill_price=None,
        status=OrderStatus.REJECTED, timestamp=0.0, fee=0.0, fee_currency="",
    )
    exc = InsufficientMarginError("Not enough margin", order=order)
    assert isinstance(exc, ExchangeOrderRejectedError)
    assert isinstance(exc, BrokerError)
    assert exc.order is order


def __test_insufficient_margin_typed_match__():
    """Risk engine pattern-match on type instead of string."""
    try:
        raise InsufficientMarginError("Balance too low")
    except ExchangeOrderRejectedError as exc:
        assert isinstance(exc, InsufficientMarginError)


# === BracketAttachAfterFillRejectedError ===

def __test_bracket_attach_after_fill_rejected_is_typed_subclass__():
    exc = BracketAttachAfterFillRejectedError(
        "rejected",
        position_deal_id='deal-1', position_coid='coid-1',
        symbol='EURUSD', position_side='buy', qty=1.0,
    )
    assert isinstance(exc, ExchangeOrderRejectedError)
    assert isinstance(exc, BrokerError)


def __test_bracket_attach_after_fill_rejected_carries_position_context__():
    """The sync engine reads these fields to build the defensive
    CloseIntent — they must survive construction unchanged."""
    exc = BracketAttachAfterFillRejectedError(
        "rejected",
        position_deal_id='deal-99', position_coid='coid-99',
        symbol='BTCUSDT', position_side='sell', qty=2.5,
        from_entry='Short',
    )
    assert exc.position_deal_id == 'deal-99'
    assert exc.position_coid == 'coid-99'
    assert exc.symbol == 'BTCUSDT'
    assert exc.position_side == 'sell'
    assert exc.qty == 2.5
    assert exc.from_entry == 'Short'


def __test_bracket_attach_after_fill_rejected_from_entry_optional__():
    """``from_entry`` is informational — recovery dispatches do not
    require it (CloseIntent has no from_entry field)."""
    exc = BracketAttachAfterFillRejectedError(
        "rejected",
        position_deal_id='deal-1', position_coid='coid-1',
        symbol='EURUSD', position_side='buy', qty=1.0,
    )
    assert exc.from_entry is None


def __test_bracket_attach_after_fill_rejected_position_deal_id_optional__():
    """Cross-broker recovery key is ``position_coid``; ``position_deal_id``
    is plugin-specific (Capital.com deal id, IB permId, Bybit orderId) and
    may not exist for every plugin — must be omittable."""
    exc = BracketAttachAfterFillRejectedError(
        "rejected",
        position_coid='coid-1',
        symbol='EURUSD', position_side='buy', qty=1.0,
    )
    assert exc.position_deal_id is None


def __test_bracket_attach_after_fill_rejected_optional_diagnostic_fields__():
    """Plugins may surface filled qty + error code/message + exit id for
    operator diagnostics. All optional, default to None."""
    exc = BracketAttachAfterFillRejectedError(
        "rejected",
        position_coid='coid-1', symbol='EURUSD',
        position_side='buy', qty=2.0,
        exit_id='TP', filled_qty=1.5,
        error_code='E_LIMIT', error_message='price too far',
    )
    assert exc.exit_id == 'TP'
    assert exc.filled_qty == 1.5
    assert exc.error_code == 'E_LIMIT'
    assert exc.error_message == 'price too far'

    bare = BracketAttachAfterFillRejectedError(
        "rejected",
        position_coid='coid-1', symbol='EURUSD',
        position_side='buy', qty=1.0,
    )
    assert bare.exit_id is None
    assert bare.filled_qty is None
    assert bare.error_code is None
    assert bare.error_message is None


# === BracketAttachRejectContext ===

def __test_bracket_attach_reject_context_from_exit_intent__():
    """For an ExitIntent the entry id falls back to ``intent.from_entry``
    when the exception omits it."""
    from pynecore.core.broker.models import BracketAttachRejectContext, ExitIntent
    err = BracketAttachAfterFillRejectedError(
        "rejected",
        position_coid='coid-99', symbol='BTCUSDT',
        position_side='sell', qty=2.5,
    )
    intent = ExitIntent(
        pine_id='TP', from_entry='Short',
        symbol='BTCUSDT', side='buy', qty=2.5, tp_price=99.0,
    )
    ctx = BracketAttachRejectContext.from_exception(err, intent)
    assert ctx.intent_key == intent.intent_key
    assert ctx.from_entry == 'Short'
    assert ctx.position_coid == 'coid-99'
    assert ctx.position_deal_id is None
    assert ctx.exit_id is None


def __test_bracket_attach_reject_context_from_entry_intent__():
    """For an EntryIntent the entry id falls back to ``intent.pine_id``
    when the exception omits it."""
    from pynecore.core.broker.models import (
        BracketAttachRejectContext, EntryIntent, OrderType,
    )
    err = BracketAttachAfterFillRejectedError(
        "rejected",
        position_coid='coid-1', symbol='EURUSD',
        position_side='buy', qty=1.0,
    )
    intent = EntryIntent(
        pine_id='Long', symbol='EURUSD', side='buy', qty=1.0,
        order_type=OrderType.MARKET,
    )
    ctx = BracketAttachRejectContext.from_exception(err, intent)
    assert ctx.from_entry == 'Long'


def __test_bracket_attach_reject_context_exception_from_entry_wins__():
    """When the exception supplies ``from_entry`` it overrides the
    intent fallback (plugin knows best, e.g. partial-fill remainder)."""
    from pynecore.core.broker.models import BracketAttachRejectContext, ExitIntent
    err = BracketAttachAfterFillRejectedError(
        "rejected",
        position_coid='coid-1', symbol='EURUSD',
        position_side='buy', qty=1.0,
        from_entry='ExplicitParent',
    )
    intent = ExitIntent(
        pine_id='TP', from_entry='IntentFallback',
        symbol='EURUSD', side='sell', qty=1.0, tp_price=1.1,
    )
    ctx = BracketAttachRejectContext.from_exception(err, intent)
    assert ctx.from_entry == 'ExplicitParent'


def __test_bracket_attach_reject_context_carries_optional_diagnostics__():
    """Optional fields pass through ``getattr(..., None)`` — when set on
    the exception they show up on the context."""
    from pynecore.core.broker.models import BracketAttachRejectContext, ExitIntent
    err = BracketAttachAfterFillRejectedError(
        "rejected",
        position_coid='coid-1', symbol='EURUSD',
        position_side='buy', qty=2.0,
        exit_id='TP', filled_qty=1.5,
        error_code='E_LIMIT', error_message='price too far',
    )
    intent = ExitIntent(
        pine_id='TP', from_entry='Long',
        symbol='EURUSD', side='sell', qty=2.0, tp_price=1.1,
    )
    ctx = BracketAttachRejectContext.from_exception(err, intent)
    assert ctx.exit_id == 'TP'
    assert ctx.filled_qty == 1.5
    assert ctx.error_code == 'E_LIMIT'
    assert ctx.error_message == 'price too far'


# === BracketAttachRejectContext (de)serialize ===

def __test_bracket_attach_reject_context_roundtrip__():
    """Full field set round-trips losslessly through to_dict/from_dict —
    the path used by BrokerStore extras serialization."""
    from pynecore.core.broker.models import BracketAttachRejectContext
    import json
    ctx = BracketAttachRejectContext(
        intent_key='TP\x00Long',
        position_coid='coid-1',
        position_side='buy',
        qty=1.0,
        symbol='EURUSD',
        position_deal_id='deal-1',
        from_entry='Long',
        exit_id='TP',
        filled_qty=0.5,
        error_code='E1',
        error_message='msg',
    )
    payload = ctx.to_dict()
    # Must be JSON-serializable.
    rehydrated = BracketAttachRejectContext.from_dict(json.loads(json.dumps(payload)))
    assert rehydrated == ctx


def __test_bracket_attach_reject_context_from_dict_rejects_missing_required__():
    """Required-field absence raises ValueError so the engine can
    log+skip instead of crashing."""
    from pynecore.core.broker.models import BracketAttachRejectContext
    import pytest as _pytest
    with _pytest.raises(ValueError):
        BracketAttachRejectContext.from_dict({
            'intent_key': 'x', 'position_coid': 'y',
            # missing position_side, qty, symbol
        })


def __test_bracket_attach_reject_context_from_dict_rejects_wrong_type__():
    """Type mismatch on a required field raises ValueError."""
    from pynecore.core.broker.models import BracketAttachRejectContext
    import pytest as _pytest
    with _pytest.raises(ValueError):
        BracketAttachRejectContext.from_dict({
            'intent_key': 'x', 'position_coid': 'y',
            'position_side': 123,  # wrong type
            'qty': 1.0, 'symbol': 'EURUSD',
        })


def __test_bracket_attach_reject_context_from_dict_rejects_non_dict__():
    from pynecore.core.broker.models import BracketAttachRejectContext
    import pytest as _pytest
    with _pytest.raises(ValueError):
        BracketAttachRejectContext.from_dict('not a dict')  # type: ignore[arg-type]


# === PendingDefensiveClose ===

def __test_pending_defensive_close_roundtrip__():
    """Full marker round-trips through to_extras_dict / from_extras_dict
    including the nested context."""
    from pynecore.core.broker.models import (
        BracketAttachRejectContext, PendingDefensiveClose,
    )
    import json
    marker = PendingDefensiveClose(
        entry_id='Long',
        close_intent_key='__pyne_defensive_close__coid-1',
        close_order_ref='broker-ref-42',
        pending_since=1700000000.123,
        reject_context=BracketAttachRejectContext(
            intent_key='TP\x00Long', position_coid='coid-1',
            position_side='buy', qty=1.0, symbol='EURUSD',
        ),
    )
    payload = marker.to_extras_dict()
    rehydrated = PendingDefensiveClose.from_extras_dict(
        json.loads(json.dumps(payload))
    )
    assert rehydrated == marker


def __test_pending_defensive_close_close_order_ref_may_be_none__():
    """close_order_ref is allowed to be None (position-attached close
    plugins never surface a ref)."""
    from pynecore.core.broker.models import (
        BracketAttachRejectContext, PendingDefensiveClose,
    )
    marker = PendingDefensiveClose(
        entry_id='Long',
        close_intent_key='ck',
        close_order_ref=None,
        pending_since=1.0,
        reject_context=BracketAttachRejectContext(
            intent_key='k', position_coid='c',
            position_side='buy', qty=1.0, symbol='S',
        ),
    )
    rehydrated = PendingDefensiveClose.from_extras_dict(marker.to_extras_dict())
    assert rehydrated.close_order_ref is None


def __test_pending_defensive_close_from_extras_dict_rejects_malformed__():
    """Missing/incorrect fields raise ValueError so the engine's startup
    replay can log+skip instead of crashing."""
    from pynecore.core.broker.models import PendingDefensiveClose
    import pytest as _pytest
    with _pytest.raises(ValueError):
        PendingDefensiveClose.from_extras_dict('not a dict')  # type: ignore[arg-type]
    with _pytest.raises(ValueError):
        PendingDefensiveClose.from_extras_dict({
            # missing entry_id
            'close_intent_key': 'k',
            'pending_since': 1.0,
            'reject_context': {},
        })
    with _pytest.raises(ValueError):
        PendingDefensiveClose.from_extras_dict({
            'entry_id': 'e', 'close_intent_key': 'k',
            'pending_since': 'not-numeric',
            'reject_context': {
                'intent_key': 'i', 'position_coid': 'c',
                'position_side': 'buy', 'qty': 1.0, 'symbol': 'S',
            },
        })


# === AuthenticationFailedEvent ===

def __test_authentication_failed_event_is_broker_event__():
    evt = AuthenticationFailedEvent(reason="Invalid API key")
    assert isinstance(evt, BrokerEvent)
    assert evt.reason == "Invalid API key"


# === BrokerPlugin._map_exception default behaviour ===
#
# The base implementation does not touch ``self``, so the tests exercise it
# as an unbound method — avoiding the full abstract-method + LiveProvider
# implementation surface a real subclass would require.

def _map(raw: Exception):
    from pynecore.core.plugin.broker import BrokerPlugin
    return BrokerPlugin._map_exception(None, raw)  # type: ignore[arg-type]


def __test_map_exception_maps_connection_error__():
    mapped = _map(ConnectionError("peer closed"))
    assert isinstance(mapped, ExchangeConnectionError)
    assert "peer closed" in str(mapped)


def __test_map_exception_returns_none_for_unknown__():
    assert _map(ValueError("no idea")) is None


def __test_map_exception_connection_error_without_message__():
    mapped = _map(ConnectionError())
    assert isinstance(mapped, ExchangeConnectionError)
    assert str(mapped) == "Connection lost"


# === Defensive-close plugin contract ===

def __test_broker_plugin_default_residuals_is_empty_list__():
    """Plugins that never raise BracketAttachAfterFillRejectedError
    (or do but have no residuals to cancel — Capital.com's
    full-row position-attached bracket) inherit a no-op default."""
    from pynecore.core.plugin.broker import BrokerPlugin
    from pynecore.core.broker.models import BracketAttachRejectContext
    ctx = BracketAttachRejectContext(
        intent_key='k', position_coid='c',
        position_side='buy', qty=1.0, symbol='S',
    )
    result = BrokerPlugin.get_residual_orders_after_bracket_attach_reject(
        None, ctx,  # type: ignore[arg-type]
    )
    assert result == []


def __test_broker_plugin_default_cancel_broker_order_ref_raises__():
    """The default ``cancel_broker_order_ref`` raises NotImplementedError
    — plugins that return non-empty residual lists MUST override."""
    import asyncio
    from pynecore.core.plugin.broker import BrokerPlugin
    try:
        asyncio.run(BrokerPlugin.cancel_broker_order_ref(  # type: ignore[arg-type]
            None, 'broker-ref-1',
        ))
    except NotImplementedError as exc:
        assert "cancel_broker_order_ref" in str(exc)
    else:
        raise AssertionError("expected NotImplementedError")


def __test_broker_plugin_idempotency_contract_documented__():
    """The plugin contract for ``cancel_broker_order_ref`` is part of
    the docstring — engine recovery flows rely on plugins normalising
    not-found / already-cancelled / already-filled exchange responses
    to a successful no-op."""
    from pynecore.core.plugin.broker import BrokerPlugin
    doc = BrokerPlugin.cancel_broker_order_ref.__doc__ or ""
    assert "Idempotency contract" in doc
    assert "not found" in doc
    assert "already cancelled" in doc
    assert "already filled" in doc
