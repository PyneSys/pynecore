"""Tests for :class:`DNSEBroker`'s STATE-READ and FILL-DETECTION methods.

Covers ``get_position`` (the weighted-avg / net-to-zero money-path method),
``get_open_orders`` (union-of-books + the "both books fail must raise, never
return []" regression), ``watch_orders`` (async-generator fill detection:
delta/dedup/unknown-order/status-map/transient-survival/clamp), ``_iter_orders``,
and ``get_balance``. Same fake-client seam as ``test_errors.py``: a real
``DNSEBroker`` instance is built with a tiny in-memory config and
``broker._client`` is swapped for a canned :class:`_FakeClient`. ``asyncio.sleep``
is monkeypatched (autouse) so the ``watch_orders`` poll loop never really waits.
Test functions use the repo convention ``__test_*__`` (see ``pytest.ini``).
"""
import asyncio

import pytest
import pynecore.lib as lib
lib.bar_index = 0  # let the [BROKER] log formatter render during broker._emit()

from pynecore_dnse import broker
from pynecore.core.broker.models import LegType, OrderStatus
from pynecore.core.broker.exceptions import ExchangeConnectionError


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """The ``watch_orders`` poll loop calls ``asyncio.sleep`` every cycle —
    make it instant so tests are fast and deterministic."""
    async def _fast_sleep(_delay, result=None):
        return result
    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)


def _broker(fake_client, *, symbol="VN30F1M", account_no="ACC1", **responses):
    """A real :class:`DNSEBroker` wired to a fake client — the seam every
    state-read/fill-detection method funnels through. ``account_no`` is set
    explicitly so the ``account_id`` property never needs a network call."""
    cfg = broker.DNSEBrokerConfig(api_key="k", api_secret="s", account_no=account_no)
    instance = broker.DNSEBroker(symbol=symbol, timeframe="5", config=cfg)
    instance._client = fake_client(**responses)
    return instance


def _order_row(order_id, status, *, symbol="C1", side="NB", qty=10.0, fill=0.0,
               avg_price=None):
    row = {"id": order_id, "symbol": symbol, "side": side, "quantity": qty,
           "fillQuantity": fill, "orderStatus": status}
    if avg_price is not None:
        row["averagePrice"] = avg_price
    return row


def _books(normal=None, stop=None):
    """A ``get_orders`` callable that answers per ``order_category``; an unset
    book defaults to an empty (but successful) order list."""
    def get_orders(account, market_type, order_category=None, page_index=0,
                   page_size=100):
        body = normal if order_category == "NORMAL" else stop
        return body if body is not None else (200, {"orders": []})
    return get_orders


# --- get_position ------------------------------------------------------

def __test_get_position_single_row_long__(fake_client):
    b = _broker(fake_client, get_positions=(200, {"positions": [
        {"symbol": "VN30F1M", "side": "NB", "openQuantity": 5, "costPrice": 100.0},
    ]}))
    position = asyncio.run(b.get_position("VN30F1M"))
    assert position is not None, "a single NB row must produce a position, not None"
    assert position.side == "buy" and position.size == 5.0
    assert position.entry_price == 100.0, "entry_price must equal costPrice for one row"


@pytest.mark.parametrize("rows, want_size, want_entry", [
    ([{"symbol": "VN30F1M", "side": "NB", "openQuantity": 3, "costPrice": 90.0},
      {"symbol": "VN30F1M", "side": "NB", "openQuantity": 2, "costPrice": 120.0}],
     5.0, 102.0),  # (3*90 + 2*120) / 5 = 102
    ([{"symbol": "VN30F1M", "side": "NB", "openQuantity": 1, "costPrice": 80.0},
      {"symbol": "VN30F1M", "side": "NB", "openQuantity": 1, "costPrice": 100.0},
      {"symbol": "VN30F1M", "side": "NB", "openQuantity": 1, "costPrice": 120.0}],
     3.0, 100.0),
])
def __test_get_position_weighted_average_across_multiple_rows__(
        fake_client, rows, want_size, want_entry):
    b = _broker(fake_client, get_positions=(200, {"positions": rows}))
    position = asyncio.run(b.get_position("VN30F1M"))
    assert position.size == want_size
    assert position.entry_price == pytest.approx(want_entry), \
        "entry_price must be the cost-weighted average (cost/volume), not a plain mean"


def __test_get_position_ns_nets_negative__(fake_client):
    b = _broker(fake_client, get_positions=(200, {"positions": [
        {"symbol": "VN30F1M", "side": "NS", "openQuantity": 4, "costPrice": 50.0},
    ]}))
    position = asyncio.run(b.get_position("VN30F1M"))
    assert position.side == "sell", "NS must net negative -> side sell"
    assert position.size == 4.0 and position.entry_price == 50.0


def __test_get_position_long_alias_treated_as_positive__(fake_client):
    b = _broker(fake_client, get_positions=(200, {"positions": [
        {"symbol": "VN30F1M", "side": "LONG", "openQuantity": 2, "averagePrice": 10.0},
    ]}))
    position = asyncio.run(b.get_position("VN30F1M"))
    assert position.side == "buy", "the LONG alias must be treated the same as NB"
    assert position.entry_price == 10.0, "averagePrice must be used when costPrice is absent"


def __test_get_position_net_to_zero_returns_none__(fake_client):
    b = _broker(fake_client, get_positions=(200, {"positions": [
        {"symbol": "VN30F1M", "side": "NB", "openQuantity": 3, "costPrice": 100.0},
        {"symbol": "VN30F1M", "side": "NS", "openQuantity": 3, "costPrice": 100.0},
    ]}))
    position = asyncio.run(b.get_position("VN30F1M"))
    assert position is None, \
        "a fully netted-to-zero position must be None, NOT a zero-size ExchangePosition"


@pytest.mark.parametrize("key", ["positions", "data"])
def __test_get_position_positions_vs_data_key_fallback__(fake_client, key):
    b = _broker(fake_client, get_positions=(200, {key: [
        {"symbol": "VN30F1M", "side": "NB", "openQuantity": 1, "costPrice": 77.0},
    ]}))
    position = asyncio.run(b.get_position("VN30F1M"))
    assert position is not None, f"rows under body[{key!r}] must be read"
    assert position.size == 1.0 and position.entry_price == 77.0


@pytest.mark.parametrize("extra_fields, want_entry", [
    ({"costPrice": 10.0, "averagePrice": 20.0, "price": 30.0}, 10.0),  # costPrice wins
    ({"averagePrice": 20.0, "price": 30.0}, 20.0),                     # averagePrice next
    ({"price": 30.0}, 30.0),                                          # price last resort
    ({}, 0.0),                                                        # none -> 0 cost
])
def __test_get_position_price_fallback_chain__(fake_client, extra_fields, want_entry):
    row = {"symbol": "VN30F1M", "side": "NB", "openQuantity": 1, **extra_fields}
    b = _broker(fake_client, get_positions=(200, {"positions": [row]}))
    position = asyncio.run(b.get_position("VN30F1M"))
    assert position is not None
    assert position.entry_price == want_entry, \
        "fallback order must be costPrice > averagePrice > price > 0"


def __test_get_position_filters_wrong_symbol_via_resolve_contract__(fake_client):
    b = _broker(
        fake_client,
        get_instruments=(200, {"data": [{"symbolType": "VN30F1M", "symbol": "41I1G8000"}]}),
        get_positions=(200, {"positions": [
            {"symbol": "41I1G8000", "side": "NB", "openQuantity": 2, "costPrice": 50.0},
            # the raw alias itself must NOT match once it resolves to a dated contract
            {"symbol": "VN30F1M", "side": "NB", "openQuantity": 99, "costPrice": 1.0},
            {"symbol": "OTHER", "side": "NB", "openQuantity": 50, "costPrice": 1.0},
        ]}))
    position = asyncio.run(b.get_position("VN30F1M"))
    assert position is not None
    assert position.size == 2.0, "only the row matching the RESOLVED contract must count"
    assert position.entry_price == 50.0


def __test_get_position_non200_classifies_emits_then_raises__(fake_client):
    b = _broker(fake_client, get_positions=(500, {"code": "REMOTE_SERVER_ERROR"}))
    emitted = []
    b._emit = lambda classified, *, action, ident: emitted.append(
        (classified.disposition, action, ident))
    with pytest.raises(ExchangeConnectionError):
        asyncio.run(b.get_position("VN30F1M"))
    assert emitted, "a classifiable failure must be routed through classify()+_emit() before raising"
    assert emitted[0][1] == "read:positions" and emitted[0][2] == "VN30F1M"


def __test_get_position_200_with_non_dict_body_raises_without_emit__(fake_client):
    """A 200 status is a success HTTP code, so errors.classify() returns None
    and _emit is skipped for a malformed (non-dict) 200 body — but get_position
    must still raise rather than silently treating it as flat/no-position."""
    b = _broker(fake_client, get_positions=(200, None))
    emitted = []
    b._emit = lambda *a, **k: emitted.append((a, k))
    with pytest.raises(ExchangeConnectionError):
        asyncio.run(b.get_position("VN30F1M"))
    assert emitted == [], "classify() returns None for a 2xx status, so _emit must not fire here"


# --- get_open_orders -----------------------------------------------------

def __test_get_open_orders_unions_normal_and_stop_books__(fake_client):
    normal = (200, {"orders": [_order_row("N1", "New")]})
    stop = (200, {"orders": [_order_row("S1", "New")]})
    b = _broker(fake_client, get_orders=_books(normal, stop))
    orders = asyncio.run(b.get_open_orders())
    ids = {order.id for order in orders}
    assert ids == {"N1", "S1"}, "must be the UNION of the NORMAL and STOP books"
    assert len(orders) == 2


def __test_get_open_orders_excludes_terminal_statuses__(fake_client):
    normal = (200, {"orders": [
        _order_row("N1", "New"),
        _order_row("N2", "Filled", fill=10.0, qty=10.0),
        _order_row("N3", "Cancelled"),
        _order_row("N4", "Rejected"),
        _order_row("N5", "Expired"),
    ]})
    b = _broker(fake_client, get_orders=_books(normal, (200, {"orders": []})))
    orders = asyncio.run(b.get_open_orders())
    ids = {order.id for order in orders}
    assert ids == {"N1"}, "FILLED/CANCELLED/REJECTED/EXPIRED rows must be excluded"
    assert len(orders) == 1


def __test_get_open_orders_one_book_fails_still_returns_good_book__(fake_client):
    normal = (200, {"orders": [_order_row("N1", "New")]})
    stop = (500, {"code": "REMOTE_SERVER_ERROR"})
    b = _broker(fake_client, get_orders=_books(normal, stop))
    orders = asyncio.run(b.get_open_orders())  # must NOT raise
    assert [order.id for order in orders] == ["N1"], \
        "a failed STOP book must not blank out a healthy NORMAL book (any_ok)"
    assert len(orders) == 1


def __test_get_open_orders_both_books_fail_raises_never_returns_empty__(fake_client):
    """Regression: a silently-empty [] when BOTH books fail would look
    identical to 'flat, no open orders' and could let a live risk position
    go unmanaged. This must raise instead."""
    b = _broker(fake_client, get_orders=_books((500, {"code": "REMOTE_SERVER_ERROR"}),
                                               (503, {"code": "SERVICE_UNAVAILABLE"})))
    with pytest.raises(ExchangeConnectionError) as exc_info:
        asyncio.run(b.get_open_orders())
    assert "unavailable" in str(exc_info.value).lower(), \
        "both order books failing must raise ExchangeConnectionError, never []"


def __test_get_open_orders_symbol_filter_via_resolve_contract__(fake_client):
    normal = (200, {"orders": [
        _order_row("N1", "New", symbol="VN30F1M"),
        _order_row("N2", "New", symbol="OTHER"),
    ]})
    b = _broker(fake_client, get_orders=_books(normal, (200, {"orders": []})))
    orders = asyncio.run(b.get_open_orders(symbol="VN30F1M"))
    assert [order.id for order in orders] == ["N1"], \
        "only rows matching the resolved contract must be returned"
    assert len(orders) == 1


# --- _iter_orders ----------------------------------------------------------

def __test_iter_orders_yields_rows_from_both_books__(fake_client):
    normal = (200, {"orders": [_order_row("N1", "New")]})
    stop = (200, {"orders": [_order_row("S1", "New")]})
    b = _broker(fake_client, get_orders=_books(normal, stop))
    rows = list(b._iter_orders())
    ids = {row["id"] for row in rows}
    assert ids == {"N1", "S1"}
    assert len(rows) == 2


def __test_iter_orders_skips_failed_book_without_raising__(fake_client):
    normal = (200, {"orders": [_order_row("N1", "New")]})
    stop = (500, {"code": "REMOTE_SERVER_ERROR"})
    b = _broker(fake_client, get_orders=_books(normal, stop))
    rows = list(b._iter_orders())  # must not raise; the bad book is skipped
    assert [row["id"] for row in rows] == ["N1"]
    assert len(rows) == 1


# --- watch_orders ----------------------------------------------------------

def __test_watch_orders_first_sighting_yields_fill_event__(fake_client, collect):
    row = _order_row("O1", "PartiallyFilled", fill=4.0, qty=10.0, avg_price=101.5)
    b = _broker(fake_client, get_orders=_books((200, {"orders": [row]}), None))
    b._identity["O1"] = ("pineA", None, LegType.ENTRY)
    events = collect(b.watch_orders(), 1)
    assert len(events) == 1, "the first sighting of a fill must yield exactly one event"
    event = events[0]
    assert event.event_type == "partial"
    assert event.fill_qty == 4.0, "fill_qty must be the DELTA (cumulative - previous), here 4-0"
    assert event.fill_price == 101.5
    assert event.pine_id == "pineA"


def __test_watch_orders_dedup_no_reyield_on_identical_poll__(fake_client, collect):
    seen_row = _order_row("O1", "PartiallyFilled", fill=4.0, qty=10.0)
    fresh_row = _order_row("O2", "Filled", fill=6.0, qty=6.0)
    b = _broker(fake_client, get_orders=_books(
        (200, {"orders": [seen_row, fresh_row]}), None))
    b._identity["O1"] = ("pineA", None, LegType.ENTRY)
    b._identity["O2"] = ("pineB", None, LegType.ENTRY)
    b._last_seen["O1"] = (4.0, OrderStatus.PARTIALLY_FILLED)  # already reported last poll
    events = collect(b.watch_orders(), 1)
    assert len(events) == 1, "only the un-seen order should yield; O1 must be deduped"
    assert events[0].order.id == "O2", "dedup must be per-id, not a global freeze"


def __test_watch_orders_unknown_order_skipped__(fake_client, collect):
    unknown_row = _order_row("O1", "Filled", fill=5.0, qty=5.0)
    known_row = _order_row("O2", "Filled", fill=3.0, qty=3.0)
    b = _broker(fake_client, get_orders=_books(
        (200, {"orders": [unknown_row, known_row]}), None))
    b._identity["O2"] = ("pineB", None, LegType.ENTRY)  # O1 deliberately unregistered
    events = collect(b.watch_orders(), 1)
    assert len(events) == 1, "an order with no recorded identity (pine_id None) must be skipped"
    assert events[0].order.id == "O2" and events[0].pine_id == "pineB"


@pytest.mark.parametrize("raw_status, fill, expected_event, expected_status", [
    ("Filled", 10.0, "filled", OrderStatus.FILLED),
    ("PartiallyFilled", 4.0, "partial", OrderStatus.PARTIALLY_FILLED),
    ("Cancelled", 0.0, "cancelled", OrderStatus.CANCELLED),
    ("Rejected", 0.0, "rejected", OrderStatus.REJECTED),
    ("New", 0.0, "created", OrderStatus.OPEN),
    ("Activated", 0.0, "created", OrderStatus.OPEN),
])
def __test_watch_orders_status_to_event_type_map__(
        fake_client, collect, raw_status, fill, expected_event, expected_status):
    row = _order_row("O1", raw_status, fill=fill, qty=10.0)
    b = _broker(fake_client, get_orders=_books((200, {"orders": [row]}), None))
    b._identity["O1"] = ("pineA", None, LegType.ENTRY)
    events = collect(b.watch_orders(), 1)
    assert len(events) == 1, f"status {raw_status!r} must yield exactly one event"
    assert events[0].event_type == expected_event
    assert events[0].order.status is expected_status


def __test_watch_orders_survives_transient_iter_orders_exception__(fake_client, collect):
    calls = {"n": 0}
    good_row = _order_row("O1", "Filled", fill=7.0, qty=7.0)

    def flaky_get_orders(account, market_type, order_category=None, page_index=0,
                         page_size=100):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("transient wire fault")
        if order_category == "NORMAL":
            return (200, {"orders": [good_row]})
        return (200, {"orders": []})

    b = _broker(fake_client, get_orders=flaky_get_orders)
    b._identity["O1"] = ("pineA", None, LegType.ENTRY)
    events = collect(b.watch_orders(), 1)
    assert len(events) == 1, "a transient _iter_orders blip on one poll must not crash the generator"
    assert events[0].order.id == "O1"
    assert calls["n"] > 1, "the flaky get_orders must actually have been called more than once"


def __test_watch_orders_cumulative_decrease_clamps_no_negative_fill__(fake_client, collect):
    calls = {"n": 0}

    def get_orders(account, market_type, order_category=None, page_index=0,
                   page_size=100):
        if order_category != "NORMAL":
            return (200, {"orders": []})
        calls["n"] += 1
        fill = 10.0 if calls["n"] == 1 else 4.0  # cumulative goes DOWN on poll 2+
        return (200, {"orders": [_order_row("O1", "PartiallyFilled", fill=fill, qty=10.0)]})

    b = _broker(fake_client, get_orders=get_orders)
    b._identity["O1"] = ("pineA", None, LegType.ENTRY)
    events = collect(b.watch_orders(), 2)
    assert len(events) == 2, "both the up-poll and the down-poll must each report a change"
    assert events[0].fill_qty == 10.0
    assert events[1].fill_qty is None, \
        "a decreasing cumulative must clamp delta to 0 (never a negative fill_qty)"
    assert events[1].order.filled_qty == 4.0, \
        "the order's own cumulative filled_qty still reflects the raw drop"


# --- get_balance -------------------------------------------------------

def __test_get_balance_derivative_branch__(fake_client):
    b = _broker(fake_client, get_balances=(200, {"derivative": {"remainSecure": 123456.0}}))
    balance = asyncio.run(b.get_balance())
    assert balance == {"VND": 123456.0}
    assert set(balance) == {"VND"}, "only the VND key is reported"


def __test_get_balance_stock_fallback_branch__(fake_client):
    b = _broker(fake_client, get_balances=(200, {"stock": {"availableCash": 5000.0}}))
    balance = asyncio.run(b.get_balance())
    assert balance == {"VND": 5000.0}
    assert balance["VND"] > 0


def __test_get_balance_derivative_wins_when_both_present__(fake_client):
    b = _broker(fake_client, get_balances=(200, {
        "derivative": {"remainSecure": 111.0}, "stock": {"availableCash": 999.0}}))
    balance = asyncio.run(b.get_balance())
    assert balance == {"VND": 111.0}, "derivative.remainSecure must win over stock.availableCash"
    assert balance["VND"] != 999.0


def __test_get_balance_zero_remain_secure_falls_back_to_stock__(fake_client):
    """remainSecure == 0 is falsy, so ``or`` must fall through to the stock figure."""
    b = _broker(fake_client, get_balances=(200, {
        "derivative": {"remainSecure": 0}, "stock": {"availableCash": 250.0}}))
    balance = asyncio.run(b.get_balance())
    assert balance == {"VND": 250.0}
    assert balance["VND"] != 0.0


@pytest.mark.parametrize("status, body", [
    (500, {"code": "REMOTE_SERVER_ERROR"}),
    (200, None),
    (200, ["not", "a", "dict"]),
])
def __test_get_balance_non200_or_non_dict_returns_empty_dict__(fake_client, status, body):
    """Pinned contract: get_balance silently returns {} on any non-200/non-dict
    reply — it never raises. This is the documented, deliberate behavior."""
    b = _broker(fake_client, get_balances=(status, body))
    balance = asyncio.run(b.get_balance())
    assert balance == {}, "non-200/non-dict must yield the silent-{} contract"
    assert isinstance(balance, dict)
