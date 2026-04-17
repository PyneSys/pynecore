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
