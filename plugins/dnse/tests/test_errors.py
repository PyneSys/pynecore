"""Tests for the DNSE error-code classifier and the broker's reactions to it.

Table-driven against canned ``(status, body)`` replies — the same fake-client seam
that proved the cancel fix — covering every action class in
``docs/plan/dnse-error-handling.md`` plus the codes observed live.
"""
import pytest

import pynecore.lib as lib
lib.bar_index = 0  # let the [BROKER] log formatter render during _emit()

from pynecore_dnse import errors, broker
from pynecore_dnse.errors import Disposition as D
from pynecore.core.broker.models import LegType
from pynecore.core.broker.exceptions import (
    AuthenticationError, ExchangeOrderRejectedError, ExchangeRateLimitError,
    InsufficientMarginError, OrderDispositionUnknownError,
)


# --- classify --------------------------------------------------------------

@pytest.mark.parametrize("status, body, is_write, want", [
    (201, {}, True, None),
    (204, {}, False, None),
    (0, {}, True, D.DISPOSITION_UNKNOWN),    # no response, write -> park+verify
    (0, {}, False, D.CONNECTION),            # no response, read  -> reconnect
    (400, {"code": "INVALID_TRADING_TOKEN"}, True, D.AUTH_TOKEN),
    (403, {"code": "FORBIDDEN"}, True, D.AUTH),
    (401, {}, True, D.AUTH),                 # HTTP-status fallback
    (429, {"code": "OA-429"}, True, D.RATE_LIMIT),
    (400, {"code": "PURCHASING_POWER_NOT_ENOUGH"}, True, D.MARGIN),
    (400, {"code": "OUT_OF_MARGIN_BASKET"}, True, D.MARGIN),
    (400, {"code": "CAN_NOT_PLACE_ORDER_ON_THIS_SESSION"}, True, D.SESSION_REJECT),
    (400, {"code": "CANNOT_PLACE_ORDER_IN_THE_CLOSED_SESSION"}, True, D.SESSION_REJECT),   # observed
    (400, {"code": "CANNOT_CANCEL_THE_ORDER_IN_THE_ATO_SESSION"}, True, D.CONNECTION),     # observed
    (404, {"code": "RESOURCE_NOT_FOUND"}, True, D.NOT_FOUND),
    (400, {"code": "INVALID_ORDER_ID"}, True, D.NOT_FOUND),
    (400, {"code": "ORDER_IS_DONE"}, True, D.TERMINAL),
    (400, {"code": "CO-ORD-013"}, True, D.TERMINAL),                                       # observed
    (500, {"code": "REMOTE_SERVER_ERROR"}, True, D.DISPOSITION_UNKNOWN),
    (503, {"code": "REMOTE_SERVER_ERROR"}, False, D.CONNECTION),
    (400, {"code": "TIMEOUT"}, False, D.CONNECTION),
    (400, {"code": "INVALID_PRICE"}, True, D.REJECT),
    (400, {"code": "PRICE_MUST_LESS_THAN_OR_EQUAL_TO_CEILING_PRICE"}, True, D.REJECT),
    (404, {"code": "OA-404"}, True, D.REJECT),
    (400, {"code": "TOTALLY_UNKNOWN_CODE"}, True, D.REJECT),   # unknown -> safe default
])
def __test_classify__(status, body, is_write, want):
    c = errors.classify(status, body, is_write=is_write)
    assert (c.disposition if c else None) is want


def __test_rate_limit_carries_retry_after__():
    c = errors.classify(429, {"code": "OA-429", "X-RateLimit-Reset": "12"}, is_write=True)
    assert c.disposition is D.RATE_LIMIT and c.retry_after == 12.0


def __test_log_message_is_structured__():
    c = errors.classify(400, {"code": "INVALID_PRICE", "message": "bad"}, is_write=True)
    line = c.log_message("place", "L/entry intent=k")
    assert "code=INVALID_PRICE" in line and "http=400" in line and "-> rejected" in line


# --- broker reactions (real methods, fake client) --------------------------

def _stub(**attrs):
    """A minimal object carrying the real DNSEBroker error/cancel methods."""
    class S:
        pass
    for name in ("_emit", "_raise_write_error", "_cancel_one",
                 "_order_category_for", "_identity_for", "_write"):
        setattr(S, name, getattr(broker.DNSEBroker, name))
    s = S()
    s.account_id = "A"
    s.market_type = "DERIVATIVE"
    s._token = lambda: "t"
    s._order_category = {}
    s._identity = {}
    for key, val in attrs.items():
        setattr(s, key, val)
    return s


@pytest.mark.parametrize("status, body, exc", [
    (201, {"id": "x"}, None),
    (400, {"code": "INVALID_PRICE"}, ExchangeOrderRejectedError),
    (400, {"code": "PURCHASING_POWER_NOT_ENOUGH"}, InsufficientMarginError),
    (429, {"code": "OA-429"}, ExchangeRateLimitError),
    (0, {}, OrderDispositionUnknownError),
    (403, {"code": "FORBIDDEN"}, AuthenticationError),
    (400, {"code": "INVALID_TRADING_TOKEN"}, AuthenticationError),  # persists post-reread
    (400, {"code": "CANNOT_PLACE_ORDER_IN_THE_CLOSED_SESSION"}, ExchangeOrderRejectedError),
])
def __test_raise_write_error__(status, body, exc):
    stub = _stub()
    if exc is None:
        stub._raise_write_error(status, body, action="place", ident="L/entry", coid="c")
    else:
        with pytest.raises(exc):
            stub._raise_write_error(status, body, action="place", ident="L/entry", coid="c")


class _FakeClient:
    def __init__(self, behavior):
        self.behavior = behavior
        self.calls = []

    def cancel_order(self, account, order_id, market, token, order_category=None):
        self.calls.append(order_category)
        return self.behavior(order_category)


def _book(category):
    """STOP book cancels; NORMAL book 404s (the STOP id isn't there)."""
    return (200, {"orderStatus": "Canceled"}) if category == "STOP" \
        else (404, {"code": "RESOURCE_NOT_FOUND"})


@pytest.mark.parametrize("name, behavior, attrs, want", [
    ("recorded-STOP", _book, {"_order_category": {"X": "STOP"}}, (True, ["STOP"])),
    ("fallback-ENTRY", _book, {"_identity": {"X": (None, None, LegType.ENTRY)}},
     (True, ["NORMAL", "STOP"])),
    ("terminal", lambda c: (400, {"code": "CO-ORD-013"}),
     {"_order_category": {"X": "STOP"}}, (True, ["STOP"])),
    ("session-refused", lambda c: (400, {"code": "CANNOT_CANCEL_THE_ORDER_IN_THE_ATO_SESSION"}),
     {"_order_category": {"X": "STOP"}}, (False, ["STOP"])),
    ("gone-everywhere", lambda c: (404, {"code": "RESOURCE_NOT_FOUND"}),
     {"_identity": {"X": (None, None, LegType.ENTRY)}}, (True, ["NORMAL", "STOP"])),
])
def __test_cancel_one__(name, behavior, attrs, want):
    fake = _FakeClient(behavior)
    stub = _stub(client=fake, **attrs)
    # These cases pin WHICH BOOK is probed and how each error code is classified. The
    # separate question — whether a 2xx cancel actually took effect at the venue — is
    # covered by __test_cancel_2xx_is_not_trusted_until_the_venue_agrees__ and friends
    # in test_broker_lifecycle.py, so here the venue simply agrees.
    stub._cancel_took_effect = lambda *_args: True
    assert (stub._cancel_one("X"), fake.calls) == want
