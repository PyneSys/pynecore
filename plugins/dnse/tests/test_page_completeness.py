"""#62/#57/#61 repro — a truncated page must NEVER read as flat/complete.

Positions surface (#57): the docs envelope carries ``total``/``pageSize``
(sample default 20, fields "chỉ dành cho phái sinh") but ``get_position``
ignores them — ``{"positions": [], "total": 3}`` parses as flat (None), and
None triggers the engine's external-flatten wipe → Pine re-enters → double
exposure. A ``status: CLOSED`` row with residual ``openQuantity`` is adopted
as live. Orders surface (#61): the docs envelope carries ``totalPages`` but
``get_open_orders``/``_iter_orders`` read only ``page_index=0`` — a book past
one page silently truncates under recovery, disappearance checks and cancel
dispatch. Gate (#62): the L0 gate (``l0_order_semantics.py``) and ``venue.py
flat`` read through these exact parsers, so a parser that raises on PROVEN
truncation flips the gate from false-green to exit-2 structurally.

Whether the live venue actually pages >20 positions is UNMEASURED (#57 states
it honestly) — these tests pin the PARSER's blindness, which is provable
offline. R1/R2/R3/R4 are RED on the unmodified tree; R5 is the green control
(a complete envelope parses exactly as today).
"""
import asyncio

import pytest
import pynecore.lib as lib

lib.bar_index = 0  # let the [BROKER] log formatter render during broker._emit()

from pynecore_dnse import broker
from pynecore.core.broker.exceptions import ExchangeConnectionError

_SECDEF_ROW = [{"ceilingPrice": "1550", "floorPrice": "1450", "securityGroupId": "FU"}]
_LOAN_OK = (200, {"loanPackages": [{"id": 42}]})


def _broker(fake_client, tmp_path, **client_responses):
    responses = {"get_security_definition": (200, _SECDEF_ROW), "get_loan_packages": _LOAN_OK}
    responses.update(client_responses)
    config = broker.DNSEBrokerConfig(
        api_key="k", api_secret="s", account_no="ACC001", trading_token="tok-A",
        token_file=str(tmp_path / "missing_token.json"))
    instance = broker.DNSEBroker(symbol="VN30F1M", timeframe="15", config=config)
    instance._client = fake_client(**responses)
    return instance


def _pos_row(symbol="VN30F1M", qty=2, side="NB", status="OPEN", cost=1800.0):
    return {"symbol": symbol, "openQuantity": qty, "side": side,
            "status": status, "costPrice": cost}


# --- R1 (RED): empty page with total>0 must not read as FLAT -----------------

def __test_empty_page_with_nonzero_total_never_reads_flat__(fake_client, tmp_path):
    """``{"positions": [], "total": 3}`` — the venue says 3 positions exist
    but delivered none of them. Today this returns None (flat), and None is
    what arms the engine's external-flatten wipe. A successful-but-incomplete
    read must raise, never conclude."""
    b = _broker(fake_client, tmp_path,
                get_positions=(200, {"positions": [], "total": 3}))

    with pytest.raises(ExchangeConnectionError):
        asyncio.run(b.get_position("VN30F1M"))


# --- R2 (RED): rows < total (our row beyond the page) must not read as flat --

def __test_truncated_page_with_rows_below_total_raises__(fake_client, tmp_path):
    """20 foreign rows delivered, ``total: 21`` — our VN30F1M row is in the
    undelivered remainder. Today: None (flat) → wipe. The parser must refuse
    the unprovable page set."""
    rows = [_pos_row(symbol=f"FOREIGN{i}") for i in range(20)]
    b = _broker(fake_client, tmp_path,
                get_positions=(200, {"positions": rows, "total": 21}))

    with pytest.raises(ExchangeConnectionError):
        asyncio.run(b.get_position("VN30F1M"))


# --- R3 (RED): a CLOSED row must not be adopted as a live position -----------

def __test_closed_row_with_residual_open_quantity_is_not_a_position__(fake_client, tmp_path):
    """The docs enumerate status OPEN / PENDING_CLOSE / CLOSED / ODD_LOT.
    A CLOSED row that still carries ``openQuantity`` is history, not
    exposure — adopting it makes the engine defend a position that does not
    exist."""
    b = _broker(fake_client, tmp_path,
                get_positions=(200, {"positions": [_pos_row(status="CLOSED")],
                                     "total": 1}))

    position = asyncio.run(b.get_position("VN30F1M"))

    assert position is None, (
        f"a CLOSED row must be filtered, not adopted; got {position!r}")


# --- R4 (RED): orders book past page 0 must be drained -----------------------

def __test_open_orders_drains_every_page__(fake_client, tmp_path):
    """The orders envelope carries ``totalPages``; a book past one page must
    be drained — our working order on page 1 is invisible to a page-0-only
    read (recovery, disappearance checks and cancel dispatch all reason over
    that partial book)."""
    def _orders(account, market, order_category=None, page_index=0, page_size=100):
        if order_category != "NORMAL":
            return (200, {"orders": [], "totalPages": 1, "pageIndex": page_index})
        if page_index == 0:
            rows = [{"id": 1000 + i, "symbol": "VN30F1M", "side": "NB", "quantity": 1,
                     "orderStatus": "New"} for i in range(100)]
            return (200, {"orders": rows, "totalPages": 2, "pageIndex": 0})
        return (200, {"orders": [{"id": 9999, "symbol": "VN30F1M", "side": "NB",
                                  "quantity": 1, "orderStatus": "New"}],
                      "totalPages": 2, "pageIndex": 1})

    b = _broker(fake_client, tmp_path, get_orders=_orders)

    orders = asyncio.run(b.get_open_orders("VN30F1M"))

    ids = {order.id for order in orders}
    assert "9999" in ids or 9999 in ids, (
        f"the page-1 order is invisible — got {len(orders)} orders, all from "
        f"page 0; a >100-row book silently truncates (#61)")


# --- R5 (GREEN control): a complete envelope parses exactly as today ---------

def __test_complete_envelope_parses_as_before__(fake_client, tmp_path):
    """Control: rows == total, OPEN status — the net-position arithmetic must
    be untouched by the completeness guard."""
    b = _broker(fake_client, tmp_path,
                get_positions=(200, {"positions": [_pos_row(qty=3)], "total": 1}))

    position = asyncio.run(b.get_position("VN30F1M"))

    assert position is not None
    assert position.side == "long" and position.size == 3.0
    assert position.entry_price == 1800.0


# --- G2b: completeness on RAW rows, BEFORE the CLOSED filter -----------------

def __test_closed_rows_count_toward_total_no_false_raise__(fake_client, tmp_path):
    """``total`` counts CLOSED rows too — filtering before the completeness
    check would false-raise this healthy read (panel P1, G2b)."""
    rows = [_pos_row(status="CLOSED", qty=5), _pos_row(status="OPEN", qty=2)]
    b = _broker(fake_client, tmp_path,
                get_positions=(200, {"positions": rows, "total": 2}))

    position = asyncio.run(b.get_position("VN30F1M"))

    assert position is not None and position.size == 2.0, (
        "the OPEN row nets, the CLOSED row is filtered, and the RAW count "
        "satisfies total — no raise")


# --- ODD_LOT is real exposure ------------------------------------------------

def __test_odd_lot_row_nets_as_exposure__(fake_client, tmp_path):
    """ODD_LOT is a real (stock) holding — filtering it would re-import
    wrong-flat for STOCK accounts (panel P1)."""
    b = _broker(fake_client, tmp_path,
                get_positions=(200, {"positions": [_pos_row(status="ODD_LOT", qty=1)],
                                     "total": 1}))

    position = asyncio.run(b.get_position("VN30F1M"))

    assert position is not None and position.size == 1.0


# --- G3': ONE unreadable book poisons the whole answer -----------------------

def __test_one_unreadable_book_raises_never_partial_union__(fake_client, tmp_path):
    """The old any_ok union returned the readable book's rows as a complete
    answer while the other book 500'd — a resting STOP invisible to
    ``venue.py flat`` (exit 0) and the restart scan (COID collision). ANY
    unreadable book must raise (panel P1+P3, found independently)."""
    def _orders(account, market, order_category=None, page_index=0, page_size=100):
        if order_category == "NORMAL":
            return (200, {"orders": [], "totalPages": 1})
        return (500, {"code": "INTERNAL"})

    b = _broker(fake_client, tmp_path, get_orders=_orders)

    with pytest.raises(ExchangeConnectionError):
        asyncio.run(b.get_open_orders("VN30F1M"))


# --- G5': over-cap totalPages = unprovable, never a partial drain ------------

def __test_over_cap_total_pages_is_unreadable_not_partial__(fake_client, tmp_path):
    """A lying ``totalPages`` must not authorize a request storm (cap 10) —
    and hitting the cap renders the BOOK unreadable, never a partial return
    that looks complete (panel P1+P2+P3 convergent)."""
    def _orders(account, market, order_category=None, page_index=0, page_size=100):
        return (200, {"orders": [{"id": 1, "symbol": "VN30F1M", "side": "NB",
                                  "quantity": 1, "orderStatus": "New"}],
                      "totalPages": 99})

    b = _broker(fake_client, tmp_path, get_orders=_orders)

    with pytest.raises(ExchangeConnectionError):
        asyncio.run(b.get_open_orders("VN30F1M"))
    pages_fetched = b._client.count("get_orders")
    assert pages_fetched <= 2, (
        f"an over-cap totalPages must be refused up front, not drained — "
        f"{pages_fetched} pages were fetched")


# --- deadline: a hung read raises instead of blocking the loop ---------------

def __test_positions_read_deadline_raises_fail_closed__(fake_client, tmp_path):
    """The client socket can block 60 s; the read runs off-loop with a
    deadline under the engine's execute budget — timeout raises (retry, the
    safe direction), never a silent long block (panel P2)."""
    import time as _time

    def _slow(*a, **k):
        _time.sleep(0.5)
        return (200, {"positions": [], "total": 0})

    b = _broker(fake_client, tmp_path, get_positions=_slow)
    b._book_read_deadline_s = 0.05

    with pytest.raises(ExchangeConnectionError):
        asyncio.run(b.get_position("VN30F1M"))


# --- the client wrapper threads pageSize to the venue ------------------------

def __test_client_wrapper_sends_page_size__():
    """The fake-client seam bypasses client.py entirely (panel P3), so the
    explicit ``get_positions`` wrapper gets its own anchor: pageSize reaches
    the SDK query, path and marketType intact, vendored SDK untouched."""
    from pynecore_dnse.client import DNSEClient

    client = DNSEClient(api_key="k", api_secret="s")
    seen = {}

    class _StubSdk:
        def _request(self, method, path, query=None, **kwargs):
            seen.update(method=method, path=path, query=query)
            return (200, '{"positions": [], "total": 0}')

    client._sdk = _StubSdk()
    status, body = client.get_positions("ACC001", "DERIVATIVE", 500)

    assert status == 200 and body == {"positions": [], "total": 0}
    assert seen["method"] == "GET"
    assert seen["path"] == "/accounts/ACC001/positions"
    assert seen["query"] == {"marketType": "DERIVATIVE", "pageSize": 500}
