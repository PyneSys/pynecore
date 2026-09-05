"""Phase B1 repro — the recovery verdict ladder over #36's journal rows.

What #36 landed: rows + refs + identity restore. What it deliberately did NOT
land (its DECISION comment): recovery POLICY. The gaps, red-proven here:

- a ``disposition_unknown`` row (the #67 lost-reply park, persisted by #36) is
  restored blind — nothing RESOLVES it against the venue at startup, and an
  unresolvable one produces NO loud signal (the order may exist at the venue
  with real exposure);
- the repo's own documented crash-relaunch recipe (``--run-label <x>``,
  live_test/README.md) CHANGES run_id, so the new run adopts nothing and the
  old run's live rows become invisible strands — #60, review-verified
  (#59 item 2: "the exact scenario Item 2 exists for is defeated by the
  repo's own documented relaunch recipe"). Core provides the purpose-built
  read: ``foreign_live_exchange_order_ids(symbol=)``.

The controls pin the ladder's load-bearing rule (still-unknown TOUCHES
NOTHING — on a SOFTWARE-idempotency venue, doubt + redispatch is a double
order) and the operator-FOREIGN distinction (a book row journalled by NOBODY
is the operator's; a row journalled by a SIBLING RUN is a strand — different
verdicts, both never blindly cancelled).
"""
import asyncio
import logging

import pytest
import urllib3
import pynecore.lib as lib

lib.bar_index = 0  # let the [BROKER] log formatter render during broker._emit()

from pynecore_dnse import broker
from pynecore.core.broker.storage import BrokerStore
from pynecore.core.broker.run_identity import RunIdentity
from pynecore.core.broker.models import DispatchEnvelope, EntryIntent, OrderType

_SECDEF_ROW = [{"ceilingPrice": "1550", "floorPrice": "1450", "securityGroupId": "FU"}]
_LOAN_OK = (200, {"loanPackages": [{"id": 42}]})
_PLACED = (201, {"id": "437346", "symbol": "VN30F1M", "side": "NB",
                 "quantity": 1, "orderStatus": "New"})


def _broker(fake_client, tmp_path, **client_responses):
    responses = {"get_security_definition": (200, _SECDEF_ROW), "get_loan_packages": _LOAN_OK}
    responses.update(client_responses)
    config = broker.DNSEBrokerConfig(
        api_key="k", api_secret="s", account_no="ACC001", trading_token="tok-A",
        token_file=str(tmp_path / "missing_token.json"))
    instance = broker.DNSEBroker(symbol="VN30F1M", timeframe="15", config=config)
    instance._client = fake_client(**responses)
    instance._cancel_verify_attempts = 1
    instance._cancel_verify_delay = 0.0
    return instance


def _entry_envelope(pine_id="L"):
    return DispatchEnvelope(
        intent=EntryIntent(pine_id=pine_id, symbol="VN30F1M", side="buy", qty=1,
                           order_type=OrderType.LIMIT, limit=1500.0),
        run_tag="abcd", bar_ts_ms=1_700_000_000_000, retry_seq=0, coid_max_len=30)


def _identity(label="t"):
    return RunIdentity(strategy_id=label, symbol="VN30F1M", timeframe="15",
                       account_id="ACC001")


def _crash_a_lost_reply(fake_client, tmp_path):
    """Instance 1: POST times out (#67 park) -> #36 persists the
    disposition_unknown row -> 'crash' (run ends, row stays live)."""
    from pynecore.core.broker.exceptions import OrderDispositionUnknownError

    def _post_timeout(*_a, **_k):
        raise urllib3.exceptions.ReadTimeoutError(None, "https://x", "t")

    b1 = _broker(fake_client, tmp_path, post_order=_post_timeout)
    store = BrokerStore(tmp_path / "broker.sqlite", plugin_name=b1.plugin_name)
    ctx = store.open_run(_identity(), script_source="// b1")
    b1.store_ctx = ctx
    with pytest.raises(OrderDispositionUnknownError):
        asyncio.run(b1.execute_entry(_entry_envelope()))
    ctx.close()
    store.close()


def _loud_records(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


# --- R1 (RED): a resolvable disposition_unknown row is RESOLVED at startup ---

def __test_du_row_resolved_by_venue_read_at_startup__(fake_client, tmp_path, caplog):
    """The lost-reply order turns out to EXIST at the venue (resting New on
    the book). Startup recovery must resolve the row by reading the venue and
    re-own the order — today the DU row is restored blind (no id known, no
    read, no resolution) and the resting order is invisible exposure."""
    _crash_a_lost_reply(fake_client, tmp_path)

    b2 = _broker(fake_client, tmp_path,
                 get_orders=(200, {"orders": [_PLACED[1]], "totalPages": 1}),
                 get_order_detail=(200, _PLACED[1]))
    store2 = BrokerStore(tmp_path / "broker.sqlite", plugin_name=b2.plugin_name)
    ctx2 = store2.open_run(_identity(), script_source="// b1")
    b2.store_ctx = ctx2
    caplog.clear()                     # drop the crash-phase park+verify WARNING
    try:
        with caplog.at_level(logging.DEBUG):
            asyncio.run(b2.connect())

        resolved_or_loud = (
            b2._identity_for("437346")[0] is not None or _loud_records(caplog))
        assert resolved_or_loud, (
            "a disposition_unknown row was restored BLIND: the venue holds a "
            "resting order from the lost-reply POST and startup neither "
            "resolved it nor said a word (Phase B1 R1)")
    finally:
        store2.close()


# --- R2 (RED): an UNRESOLVABLE disposition_unknown row gets LOUD -------------

def __test_unresolvable_du_row_is_loud_at_startup__(fake_client, tmp_path, caplog):
    """No book row, no history row — the disposition stays unknown. The
    operator MUST hear it (WARNING+ naming the pine id, or the designed
    halt): an order may exist at the venue. Today: total silence."""
    _crash_a_lost_reply(fake_client, tmp_path)

    b2 = _broker(fake_client, tmp_path,
                 get_orders=(200, {"orders": [], "totalPages": 1}),
                 get_order_detail=(400, {"code": "RESOURCE_NOT_FOUND"}),
                 get_order_history=(200, {"data": [], "total": 0}))
    store2 = BrokerStore(tmp_path / "broker.sqlite", plugin_name=b2.plugin_name)
    ctx2 = store2.open_run(_identity(), script_source="// b1")
    b2.store_ctx = ctx2
    caplog.clear()                     # drop the crash-phase park+verify WARNING
    try:
        raised = None
        with caplog.at_level(logging.DEBUG):
            try:
                asyncio.run(b2.connect())
            except Exception as exc:                                # noqa: BLE001
                raised = exc

        assert raised is not None or _loud_records(caplog), (
            "an UNRESOLVABLE lost-reply row produced no WARNING+ and no "
            "raise at startup — the order may exist at the venue and nobody "
            "was told (Phase B1 R2; still-unknown touches nothing, but it "
            "must never be SILENT)")
    finally:
        store2.close()


# --- R3 (RED, #60): a different-run-label relaunch reports the strands -------

def __test_foreign_label_relaunch_reports_stranded_rows__(fake_client, tmp_path, caplog):
    """The documented crash-relaunch recipe changes --run-label, so the new
    run_id adopts nothing — the old run's LIVE rows become invisible strands
    (#60, review-verified). Core provides
    ``foreign_live_exchange_order_ids``: the new run must REPORT the strands
    loudly (never adopt them, never cancel them). Today: silence."""
    b1 = _broker(fake_client, tmp_path, post_order=_PLACED)
    store = BrokerStore(tmp_path / "broker.sqlite", plugin_name=b1.plugin_name)
    ctx = store.open_run(_identity("label-A"), script_source="// a")
    b1.store_ctx = ctx
    asyncio.run(b1.execute_entry(_entry_envelope()))
    ctx.close()
    store.close()                      # label-A's row stays LIVE (order rests)

    b2 = _broker(fake_client, tmp_path,
                 get_orders=(200, {"orders": [_PLACED[1]], "totalPages": 1}))
    store2 = BrokerStore(tmp_path / "broker.sqlite", plugin_name=b2.plugin_name)
    ctx2 = store2.open_run(_identity("label-B"), script_source="// b")
    b2.store_ctx = ctx2
    try:
        with caplog.at_level(logging.DEBUG):
            asyncio.run(b2.connect())

        strand_mentions = [r for r in _loud_records(caplog)
                           if "437346" in r.getMessage()]
        assert strand_mentions, (
            "label-A's live journalled order is stranded and label-B said "
            "NOTHING — the documented relaunch recipe silently orphans the "
            "previous run's orders (#60)")
        pine_id, _f, _l = b2._identity_for("437346")
        assert pine_id is None, "a sibling run's strand must be REPORTED, never adopted"
    finally:
        store2.close()


# --- controls: the rules the ladder must NEVER break -------------------------

def __test_clean_store_startup_is_quiet__(fake_client, tmp_path, caplog):
    b = _broker(fake_client, tmp_path,
                get_orders=(200, {"orders": [], "totalPages": 1}))
    store = BrokerStore(tmp_path / "broker.sqlite", plugin_name=b.plugin_name)
    ctx = store.open_run(_identity(), script_source="// c")
    b.store_ctx = ctx
    try:
        with caplog.at_level(logging.DEBUG):
            asyncio.run(b.connect())
        assert not _loud_records(caplog), "a clean store must start silently"
    finally:
        store.close()


def __test_still_unknown_touches_nothing_no_venue_writes__(fake_client, tmp_path):
    """The single most load-bearing rule (#59 item 2, SC2): recovery in doubt
    performs NO venue WRITE — on a SOFTWARE-idempotency venue, doubt plus a
    redispatch/cancel is how a double order is born. Pinned against the
    future implementation."""
    _crash_a_lost_reply(fake_client, tmp_path)

    b2 = _broker(fake_client, tmp_path,
                 get_orders=(200, {"orders": [], "totalPages": 1}),
                 get_order_detail=(400, {"code": "RESOURCE_NOT_FOUND"}),
                 get_order_history=(200, {"data": [], "total": 0}))
    store2 = BrokerStore(tmp_path / "broker.sqlite", plugin_name=b2.plugin_name)
    ctx2 = store2.open_run(_identity(), script_source="// b1")
    b2.store_ctx = ctx2
    try:
        try:
            asyncio.run(b2.connect())
        except Exception:                                           # noqa: BLE001
            pass                       # a designed halt is allowed; writes are not
        assert b2._client.count("cancel_order") == 0
        assert b2._client.count("post_order") == 0
    finally:
        store2.close()


def __test_operator_foreign_row_still_untouched_with_strands_present__(fake_client, tmp_path):
    """The operator-FOREIGN distinction survives the ladder: a book row
    journalled by NOBODY stays untouched even while sibling strands exist."""
    b1 = _broker(fake_client, tmp_path, post_order=_PLACED)
    store = BrokerStore(tmp_path / "broker.sqlite", plugin_name=b1.plugin_name)
    ctx = store.open_run(_identity("label-A"), script_source="// a")
    b1.store_ctx = ctx
    asyncio.run(b1.execute_entry(_entry_envelope()))
    ctx.close()
    store.close()

    operator_row = {"id": "999111", "symbol": "VN30F1M", "side": "NB",
                    "quantity": 5, "orderStatus": "New"}
    b2 = _broker(fake_client, tmp_path,
                 get_orders=(200, {"orders": [_PLACED[1], operator_row],
                                   "totalPages": 1}))
    store2 = BrokerStore(tmp_path / "broker.sqlite", plugin_name=b2.plugin_name)
    ctx2 = store2.open_run(_identity("label-B"), script_source="// b")
    b2.store_ctx = ctx2
    try:
        try:
            asyncio.run(b2.connect())
        except Exception:                                           # noqa: BLE001
            pass
        assert b2._identity_for("999111")[0] is None, \
            "the operator's order must never be adopted"
        assert b2._client.count("cancel_order") == 0, \
            "the operator's order must never be cancelled"
    finally:
        store2.close()
