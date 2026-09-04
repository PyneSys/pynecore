"""#56 + item-5 repro — fill slices must book at their OWN price, once.

Two defects, one data source:

- **#56 (today, no restart needed):** `_scan_row` books every fill event at
  `averagePrice` — the venue's CUMULATIVE VWAP — so any multi-slice fill gets
  a wrong cost basis on every slice after the first. The venue has the truth:
  `GET /accounts/{account}/executions/{orderId}` returns per-execution
  `lastQuantity`/`lastPrice` (+ `metadata` — a JSON STRING whose `eventNo`
  arrives as float OR int), own 10k/h bucket, in the vendored SDK, unused.
- **item 5 (restart):** `_last_seen` is in-memory, so a relaunch re-emits a
  live PARTIAL fill wholesale (completed fills stopped re-emitting when #36
  began closing their rows — the partial window is what remains).

R1/R2/R3 are RED on the unmodified tree; the control pins that a clean
single-slice full fill still emits exactly one event.
"""
import asyncio
import json

import pynecore.lib as lib

lib.bar_index = 0  # let the [BROKER] log formatter render during broker._emit()

from pynecore_dnse import broker
from pynecore.core.broker.models import LegType
from pynecore.core.broker.storage import BrokerStore
from pynecore.core.broker.run_identity import RunIdentity

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


def _row(order_id="437346", status="PartiallyFilled", cumulative=1.0, avg=1500.0, qty=3):
    return {"id": order_id, "symbol": "VN30F1M", "side": "NB", "quantity": qty,
            "orderStatus": status, "fillQuantity": cumulative, "averagePrice": avg}


def _report(event_no, qty, price, *, cum, status="PartiallyFilled"):
    """The DOCUMENTED report shape: per-report cumulative fillQuantity, and
    metadata as a JSON string whose eventNo may arrive float or int."""
    return {"orderStatus": status, "fillQuantity": cum,
            "lastQuantity": qty, "lastPrice": price,
            "metadata": json.dumps({"eventNo": event_no})}


def _lifecycle(event_no, status):
    """A qty-0 lifecycle row (PendingNew/New) — present in the doc sample."""
    return {"orderStatus": status, "fillQuantity": 0, "lastQuantity": 0,
            "lastPrice": 0, "metadata": json.dumps({"eventNo": float(event_no)})}


def _executions(*reports):
    return (200, {"reports": list(reports)})


def _own(b, order_id="437346"):
    b._identity[order_id] = ("L", None, LegType.ENTRY)
    b._order_category[order_id] = "NORMAL"


def _scan(b, raw):
    result = asyncio.run(b._scan_row(raw))
    if result is None:
        return []
    return result if isinstance(result, list) else [result]


# --- R1 (RED): two slices across two polls book TWO slice prices, not VWAP ---

def __test_two_polls_book_slice_prices_not_vwap__(fake_client, tmp_path):
    """Slice 1: 1 @ 1500. Slice 2: 1 @ 1510 (VWAP now 1505). Today BOTH
    events carry averagePrice — the second slice books 1505 instead of 1510:
    wrong cost basis, live money, no restart involved."""
    b = _broker(fake_client, tmp_path,
                get_execution_detail=_executions(_report(3, 1, 1500.0, cum=1.0)))
    _own(b)

    first = _scan(b, _row(cumulative=1.0, avg=1500.0))
    b._client._responses["get_execution_detail"] = _executions(
        _report(4, 1, 1510.0, cum=2.0), _report(3, 1, 1500.0, cum=1.0),
        _lifecycle(1, "PendingNew"))
    second = _scan(b, _row(cumulative=2.0, avg=1505.0))

    assert len(first) == 1 and first[0].fill_price == 1500.0
    assert len(second) == 1, f"expected the ONE new slice, got {len(second)} events"
    assert second[0].fill_price == 1510.0, (
        f"slice 2 executed at 1510 but booked {second[0].fill_price} — that is "
        f"the cumulative VWAP, not the slice price (#56)")
    assert second[0].fill_qty == 1.0


# --- R2 (RED): one poll, two slices -> two events with distinct prices -------

def __test_single_poll_multi_slice_emits_per_slice_events__(fake_client, tmp_path):
    """The poll can jump 0 -> 2 with two executions behind it. One VWAP event
    loses both real prices; the executions feed has each slice."""
    # documented-hostile payload: UNORDERED, a literal DUPLICATE eventNo-3
    # row, and qty-0 lifecycle rows interleaved (dnse-get-executions.md).
    b = _broker(fake_client, tmp_path,
                get_execution_detail=_executions(
                    _report(3, 1, 1500.0, cum=1.0),
                    _report(3, 1, 1500.0, cum=1.0),
                    _lifecycle(1, "PendingNew"), _lifecycle(2, "New"),
                    _report(4, 1, 1510.0, cum=2.0)))
    _own(b)

    events = _scan(b, _row(cumulative=2.0, avg=1505.0))

    prices = sorted(e.fill_price for e in events)
    assert prices == [1500.0, 1510.0], (
        f"two executions at 1500/1510 must book as two slice events; got "
        f"{[(e.fill_qty, e.fill_price) for e in events]} (#56)")


# --- R3 (RED): a restart must NOT re-emit an already-booked partial ----------

def __test_restart_does_not_reemit_partial_fill__(fake_client, tmp_path):
    """Instance 1 books the partial slice; instance 2 (same store, #36
    restore) sees the SAME venue cumulative — nothing new happened, nothing
    may be emitted. Today `_last_seen` dies with the process and the whole
    partial re-emits (double-counted fill in the engine)."""
    b1 = _broker(fake_client, tmp_path,
                 post_order=(201, {"id": "437346", "symbol": "VN30F1M",
                                   "side": "NB", "quantity": 3,
                                   "orderStatus": "New"}),
                 get_execution_detail=_executions(_report(3, 1, 1500.0, cum=1.0)))
    store = BrokerStore(tmp_path / "broker.sqlite", plugin_name=b1.plugin_name)
    identity = RunIdentity(strategy_id="t", symbol="VN30F1M", timeframe="15",
                           account_id="ACC001")
    ctx = store.open_run(identity, script_source="// t")
    b1.store_ctx = ctx
    from pynecore.core.broker.models import DispatchEnvelope, EntryIntent, OrderType
    envelope = DispatchEnvelope(
        intent=EntryIntent(pine_id="L", symbol="VN30F1M", side="buy", qty=3,
                           order_type=OrderType.LIMIT, limit=1500.0),
        run_tag="abcd", bar_ts_ms=1_700_000_000_000, retry_seq=0, coid_max_len=30)
    asyncio.run(b1.execute_entry(envelope))
    booked = _scan(b1, _row(cumulative=1.0, avg=1500.0))
    assert len(booked) == 1, "instance 1 books the partial slice"
    ctx.close(); store.close()

    b2 = _broker(fake_client, tmp_path,
                 get_execution_detail=_executions(_report(3, 1, 1500.0, cum=1.0)))
    store2 = BrokerStore(tmp_path / "broker.sqlite", plugin_name=b2.plugin_name)
    ctx2 = store2.open_run(identity, script_source="// t")
    b2.store_ctx = ctx2
    try:
        asyncio.run(b2.connect())          # #36 restore + item-5 cursor seed

        events = _scan(b2, _row(cumulative=1.0, avg=1500.0))

        assert events == [], (
            f"the restarted instance re-emitted {len(events)} event(s) for a "
            f"fill already booked before the restart — double-counted fill "
            f"(item 5: the cursor must survive the process)")
    finally:
        store2.close()


# --- GREEN control: a clean single-slice full fill emits exactly once --------

def __test_single_slice_full_fill_emits_once__(fake_client, tmp_path):
    b = _broker(fake_client, tmp_path,
                get_execution_detail=_executions(_report(3, 3, 1500.0, cum=3.0)))
    _own(b)

    events = _scan(b, _row(status="Filled", cumulative=3.0, avg=1500.0))

    assert len(events) == 1
    assert events[0].event_type == "filled"
    assert events[0].fill_qty == 3.0
    assert events[0].fill_price == 1500.0


# --- panel anchors: the hard dedup/conservation cases ------------------------

def __test_late_row_after_remainder_does_not_double_count__(fake_client, tmp_path):
    """P1's hard case: poll 1's executions LAG (no rows) -> the delta books
    as a VWAP remainder, ADVANCING the watermark. The late row arriving on
    poll 2 sits at-or-below the watermark and must be discarded — only the
    genuinely new slice books."""
    b = _broker(fake_client, tmp_path, get_execution_detail=_executions())
    _own(b)

    first = _scan(b, _row(cumulative=1.0, avg=1500.0))
    assert [(e.fill_qty, e.fill_price) for e in first] == [(1.0, 1500.0)], \
        "lagging executions must book the delta at VWAP (conserved, degraded)"

    b._client._responses["get_execution_detail"] = _executions(
        _report(3, 1, 1495.0, cum=1.0),      # the LATE row for the booked qty
        _report(4, 1, 1510.0, cum=2.0))
    second = _scan(b, _row(cumulative=2.0, avg=1505.0))

    assert [(e.fill_qty, e.fill_price) for e in second] == [(1.0, 1510.0)], (
        f"the late row re-emitted: {[(e.fill_qty, e.fill_price) for e in second]} "
        f"— quantity above the watermark only (budget clamp)")


def __test_overcovering_slices_are_clamped_to_the_budget__(fake_client, tmp_path):
    """Slices claiming MORE than the observed delta must clamp — the venue
    cumulative is authoritative for quantity."""
    b = _broker(fake_client, tmp_path,
                get_execution_detail=_executions(
                    _report(3, 2, 1500.0, cum=2.0),   # claims 2
                    _report(4, 2, 1510.0, cum=4.0)))  # claims 2 more
    _own(b)

    events = _scan(b, _row(cumulative=3.0, avg=1505.0, qty=5))

    assert sum(e.fill_qty for e in events) == 3.0, (
        f"emitted {sum(e.fill_qty for e in events)} for a delta of 3 — the "
        f"budget clamp must conserve quantity")
    assert [e.fill_qty for e in events] == [2.0, 1.0], "last slice clamps"


def __test_executions_read_failure_conserves_the_delta__(fake_client, tmp_path):
    """G2: a transport failure on the slice read falls back to ONE VWAP-delta
    event — a fill is NEVER lost to the slice feed."""
    import urllib3

    def _boom(*_a, **_k):
        raise urllib3.exceptions.MaxRetryError(
            None, "https://x",
            reason=urllib3.exceptions.NewConnectionError(None, "refused"))

    b = _broker(fake_client, tmp_path, get_execution_detail=_boom)
    _own(b)

    events = _scan(b, _row(cumulative=2.0, avg=1505.0))

    assert [(e.fill_qty, e.fill_price) for e in events] == [(2.0, 1505.0)]


def __test_status_only_transition_reads_no_executions__(fake_client, tmp_path):
    """P2's gate: a status change with NO fill delta must not spend the
    executions bucket."""
    b = _broker(fake_client, tmp_path,
                get_execution_detail=_executions(_report(3, 1, 1500.0, cum=1.0)))
    _own(b)
    b._last_seen["437346"] = (0.0, "PendingNew")

    events = _scan(b, _row(status="New", cumulative=0.0, avg=0.0))

    assert b._client.count("get_execution_detail") == 0, \
        "status-only transitions must not read executions (delta > 0 gate)"
    assert len(events) == 1 and events[0].event_type == "created"


def __test_parse_reports_survives_hostile_metadata__():
    """Pure-module robustness: missing metadata, malformed JSON, float
    eventNo, bare-object body — none may raise or produce phantom slices."""
    from pynecore_dnse.fill_slices import parse_reports

    slices = parse_reports({"reports": [
        {"orderStatus": "PartiallyFilled", "fillQuantity": 1,
         "lastQuantity": 1, "lastPrice": 1500.0},                # no metadata
        {"orderStatus": "PartiallyFilled", "fillQuantity": 2,
         "lastQuantity": 1, "lastPrice": 1510.0, "metadata": "{not json"},
        {"orderStatus": "PartiallyFilled", "fillQuantity": 3,
         "lastQuantity": 1, "lastPrice": 1520.0,
         "metadata": json.dumps({"eventNo": 5.0})},              # float eventNo
        "not-a-dict",
    ]})
    assert [s.cumulative for s in slices] == [1.0, 2.0, 3.0]
    assert slices[2].event_no == 5
    assert parse_reports("garbage") == []
    assert parse_reports({"lastQuantity": 0}) == []
