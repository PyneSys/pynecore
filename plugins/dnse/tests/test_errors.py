"""Tests for the DNSE error-code classifier and the broker's reactions to it.

Table-driven against canned ``(status, body)`` replies — the same fake-client seam
that proved the cancel fix — covering every action class in
``docs/plan/dnse-error-handling.md`` plus the codes observed live.
"""
import asyncio

import pytest

import pynecore.lib as lib
lib.bar_index = 0  # let the [BROKER] log formatter render during _emit()

from pynecore_dnse import errors, broker
from pynecore_dnse.errors import Disposition as D
from pynecore.core.broker.models import CancelDispositionOutcome, LegType
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
    for name in ("_emit", "_raise_write_error", "_cancel_one_disposition",
                 "_order_category_for", "_identity_for", "_write"):
        setattr(S, name, getattr(broker.DNSEBroker, name))
    s = S()
    s.account_id = "A"
    s.market_type = "DERIVATIVE"
    s._token = lambda: "t"
    s._order_category = {}
    s._identity = {}
    s._pending_oco = set()
    s.store_ctx = None               # #36: cancel core journals when a store exists
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
    (400, {"code": "INVALID_TRADING_TOKEN"}, AuthenticationError),  # surfaced on FIRST refusal (#58: no retry)
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


_CONFIRMED = CancelDispositionOutcome.CANCEL_CONFIRMED
_UNKNOWN = CancelDispositionOutcome.UNKNOWN


@pytest.mark.parametrize("name, behavior, attrs, want", [
    ("recorded-STOP", _book, {"_order_category": {"X": "STOP"}}, (_CONFIRMED, ["STOP"])),
    ("fallback-ENTRY", _book, {"_identity": {"X": (None, None, LegType.ENTRY)}},
     (_CONFIRMED, ["NORMAL", "STOP"])),   # STOP cancels -> loop stops before OCO
    ("terminal", lambda c: (400, {"code": "CO-ORD-013"}),
     {"_order_category": {"X": "STOP"}}, (_CONFIRMED, ["STOP"])),
    ("session-refused", lambda c: (400, {"code": "CANNOT_CANCEL_THE_ORDER_IN_THE_ATO_SESSION"}),
     {"_order_category": {"X": "STOP"}}, (_UNKNOWN, ["STOP"])),
    ("gone-everywhere", lambda c: (404, {"code": "RESOURCE_NOT_FOUND"}),
     {"_identity": {"X": (None, None, LegType.ENTRY)}},
     (_UNKNOWN, ["NORMAL", "STOP", "OCO"])),   # #45: unknown ids probe the OCO book too
])
def __test_cancel_one__(name, behavior, attrs, want):
    fake = _FakeClient(behavior)
    stub = _stub(client=fake, **attrs)
    # These cases pin WHICH BOOK is probed and how each error code is classified.
    # The separate question — what the venue read-back actually says — is pinned
    # in test_broker_lifecycle.py and test_cancel_disposition.py, so the
    # observation stages simply report success (or, for history, no row) here.
    # #55 declared change: "session-refused" is a FAILED WRITE (G3) and
    # "gone-everywhere" is absence — both now UNKNOWN (the engine retries),
    # never a confirmed cancel.
    async def _confirmed(*_args):
        return _CONFIRMED

    async def _no_history_row(*_args):
        return _UNKNOWN

    stub._readback_disposition = _confirmed
    stub._terminal_reject_disposition = _confirmed
    stub._history_disposition = _no_history_row
    assert (asyncio.run(stub._cancel_one_disposition("X")), fake.calls) == want


def __test_cancel_one_unknown_id_probes_every_book__():
    """#45 (RED until fixed): an id with NO category record and NO identity —
    every venue.py cancel of a conditional, and the engine after a restart
    before re-hydration — must probe EVERY book, as ``_cancel_one``'s own
    docstring promises. The ``_order_category_for`` catch-all instead answered
    "NORMAL", so the probe loop saw only the NORMAL book's 404 and returned
    True ("gone from every book") while the conditional kept working — measured
    live 2026-08-24 on the T04 orphan: three cancel_one=True in a row, order
    still on the STOP book."""
    fake = _FakeClient(
        lambda category: (404, {"code": "RESOURCE_NOT_FOUND"})
        if category == "NORMAL"
        else (400, {"code": "INVALID_TRADING_TOKEN", "message": "Invalid trading token"}))
    stub = _stub(client=fake)

    async def _no_history_row(*_args):
        return CancelDispositionOutcome.UNKNOWN

    stub._history_disposition = _no_history_row
    ok = asyncio.run(stub._cancel_one_disposition("da5unknownconditional"))

    assert "STOP" in fake.calls, \
        "the STOP book was never probed — the #45 false-True path"
    assert ok is CancelDispositionOutcome.UNKNOWN, \
        "a rejected STOP cancel must NOT read as cancelled (G3: a failed write is not a disposition)"


# === session-phase cancel codes ===============================================

@pytest.mark.parametrize("code", [
    "CANNOT_CANCEL_THE_ORDER_IN_THE_ATO_SESSION",
    "CANNOT_CANCEL_THE_ORDER_IN_THE_ATC_SESSION",
])
def __test_auction_cancel_refusals_are_transient_not_terminal__(code):
    """Both auction twins must be retryable: the order STILL RESTS after the refusal.

    Measured live 2026-08-13 — DNSE refused 18 cancels across the 15-minute ATC window
    and both orders then filled in the auction. The ATO code was already classified
    transient while its ATC twin fell through to REJECT, so the engine would have given
    up on an order that was merely un-cancellable *for now*.
    """
    classified = errors.classify(400, {"code": code, "message": "x"}, is_write=True)
    assert classified is not None
    assert classified.disposition is errors.Disposition.CONNECTION, \
        f"{code} must be transient (retry after the phase flips), not a terminal reject"


def __test_cancel_of_already_canceled_normal_order_is_terminal__():
    """Observed live 2026-08-18 (Live-L1-T17): re-cancel of a Canceled NORMAL order
    -> 400 ORDER_CANCEL_STATUS_REJECTED. Must classify TERMINAL (treated-gone) like
    its conditional-book twin CO-ORD-013 — not a REJECT-with-ERROR."""
    c = errors.classify(400, {"code": "ORDER_CANCEL_STATUS_REJECTED",
                              "message": "Order status is not valid to cancel"}, is_write=True)
    assert c.disposition is D.TERMINAL
    assert c.level == "info"


# === #68: OA-400 (bad HMAC secret) is an AUTH failure, not an order reject ===
# MEASURED live 2026-08-31 (read-only battery, real venue): a wrong API secret
# returns http=400 code=OA-400 msg="Authorization field missing, malformed or
# invalid". Today classify maps it to REJECT — so a rotated secret looks like
# order rejections on writes and can never reach #54's all-books-AUTH halt.

def __test_oa400_with_authorization_message_is_auth_class__():
    """RED (#68): the measured bad-secret shape must classify AUTH-class."""
    classified = errors.classify(
        400, {"code": "OA-400",
              "message": "Authorization field missing, malformed or invalid"},
        is_write=False)
    assert classified is not None
    assert classified.disposition is errors.Disposition.AUTH, (
        f"the measured bad-secret refusal classified {classified.disposition.name} "
        f"— it must be exactly AUTH: AUTH_TOKEN would route #58's do-not-re-mint "
        f"guidance, actively wrong advice for a dead secret (panel P3)")
    lowered = classified.message.lower()
    assert "secret" in lowered and "clock" in lowered, (
        "the run-ending message must name BOTH documented causes (secret, clock) "
        "— clock skew produces the identical venue reply (panel P1)")


def __test_oa400_write_side_raises_authentication_error__():
    """RED (#68): on the write path a dead secret must surface as a credential
    problem, never a terminal order reject the operator chases per-order."""
    stub = _stub()
    with pytest.raises(AuthenticationError):
        stub._raise_write_error(
            400, {"code": "OA-400",
                  "message": "Authorization field missing, malformed or invalid"},
            action="place", ident="L/entry", coid="c")


def __test_oa400_without_authorization_message_stays_reject__():
    """GREEN control (#68): OA-400 may be the venue's GENERIC bad-request code —
    without the Authorization message it must keep today's REJECT classification
    (mis-promoting a malformed order payload to AUTH would mask real bugs)."""
    classified = errors.classify(
        400, {"code": "OA-400", "message": "field 'price' is malformed"},
        is_write=True)
    assert classified is not None
    assert classified.disposition is errors.Disposition.REJECT
