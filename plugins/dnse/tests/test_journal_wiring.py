"""#36 repro (Phase A2 core) — placed orders must be JOURNALED, and a restart
must re-own them.

Measured live (Live-L1-T16, 2026-08-18): TERM leaves orders resting at the
venue; a same-label relaunch adopts nothing and ``cancel_all`` reaches
nothing. The store seam is fully core-provided (`BrokerPlugin.store_ctx` is
injected by the runner, script_runner.py:856; `store_helpers` has the whole
vocabulary) — the DNSE plugin simply never calls any of it: `_place` journals
nothing, fills/cancels transition nothing, startup adopts nothing.

The #67 join: a post-write transport loss (sentinel phase='sent') currently
parks the dispatch ENGINE-side only for the envelope's lifetime — no durable
row exists, so a crash during the park loses the order entirely.

B1/B2/B3 are RED on the unmodified tree; the control pins that a run WITHOUT
a store (store_ctx=None — backtests, unit tests) behaves exactly as today.
"""
import asyncio

import pytest
import urllib3
import pynecore.lib as lib

lib.bar_index = 0  # let the [BROKER] log formatter render during broker._emit()

from pynecore_dnse import broker
from pynecore.core.broker.storage import BrokerStore
from pynecore.core.broker.run_identity import RunIdentity
from pynecore.core.broker.models import (
    DispatchEnvelope, EntryIntent, OrderType,
)

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
    return instance


def _open_store_ctx(tmp_path, instance):
    """The reference idiom (capitalcom's journal-contract tests)."""
    store = BrokerStore(tmp_path / "broker.sqlite",
                        plugin_name=instance.plugin_name)
    identity = RunIdentity(strategy_id="t16", symbol="VN30F1M", timeframe="15",
                           account_id="ACC001")
    ctx = store.open_run(identity, script_source="// t16")
    instance.store_ctx = ctx
    return store, ctx


def _entry_envelope(pine_id="L"):
    return DispatchEnvelope(
        intent=EntryIntent(pine_id=pine_id, symbol="VN30F1M", side="buy", qty=1,
                           order_type=OrderType.LIMIT, limit=1500.0),
        run_tag="abcd", bar_ts_ms=1_700_000_000_000, retry_seq=0, coid_max_len=30)


# --- B1 (RED): a placed order must leave a journal row with its venue ref ----

def __test_place_journals_a_row_with_the_exchange_ref__(fake_client, tmp_path):
    """With a real store attached, `_place` must journal the order and record
    the venue id as a ref — the row IS the restart bridge (DNSE sends no coid
    to the venue, idempotency=SOFTWARE, so the journal is the ONLY link).
    Today: zero rows ever written (T16's measured mechanism)."""
    b = _broker(fake_client, tmp_path, post_order=_PLACED)
    store, ctx = _open_store_ctx(tmp_path, b)
    try:
        asyncio.run(b.execute_entry(_entry_envelope()))

        rows = list(ctx.iter_live_orders(symbol="VN30F1M"))
        assert rows, ("the placed order left NO journal row — a crash/TERM "
                      "strands it at the venue with no restart bridge (#36)")
        refs = [v for _t, v in ctx.iter_refs_for_coid(rows[0].client_order_id)]
        assert "437346" in refs, (
            f"the venue id must be recorded as a ref on the row; refs={refs}")
    finally:
        store.close()


# --- B2 (RED): a restarted instance must re-own the resting order ------------

def __test_restart_reowns_the_resting_order__(fake_client, tmp_path):
    """Instance 1 places; instance 2 (same store, fresh process) must adopt at
    startup: identity restored, the id cancellable. Today `_identity`/
    `_order_ids` are in-memory only — the relaunch adopts nothing (T16)."""
    b1 = _broker(fake_client, tmp_path, post_order=_PLACED)
    store, ctx = _open_store_ctx(tmp_path, b1)
    try:
        asyncio.run(b1.execute_entry(_entry_envelope()))
    finally:
        ctx.close()      # controlled shutdown ends the run (a crash would
        store.close()    # instead age out via the heartbeat-stale cleanup)

    b2 = _broker(fake_client, tmp_path,
                 get_orders=(200, {"orders": [_PLACED[1]], "totalPages": 1}))
    store2 = BrokerStore(tmp_path / "broker.sqlite", plugin_name=b2.plugin_name)
    identity = RunIdentity(strategy_id="t16", symbol="VN30F1M", timeframe="15",
                           account_id="ACC001")
    ctx2 = store2.open_run(identity, script_source="// t16")
    b2.store_ctx = ctx2
    try:
        asyncio.run(b2.connect())

        pine_id, _from_entry, _leg = b2._identity_for("437346")
        assert pine_id is not None, (
            "after restart the resting venue order has NO identity — "
            "cancel_all cannot reach it (Live-L1-T16, measured)")
    finally:
        store2.close()


# --- B3 (RED): a lost POST reply must leave a DURABLE disposition row --------

def __test_lost_reply_leaves_a_durable_row__(fake_client, tmp_path):
    """#67 parks the dispatch engine-side (OrderDispositionUnknownError), but
    with no journal row a crash DURING the park loses the order entirely.
    Persist-first: the row must exist (submitted/disposition_unknown) even
    though the reply never arrived."""
    from pynecore.core.broker.exceptions import OrderDispositionUnknownError

    def _post_timeout(*_a, **_k):
        raise urllib3.exceptions.ReadTimeoutError(None, "https://x", "read timed out")

    b = _broker(fake_client, tmp_path, post_order=_post_timeout)
    store, ctx = _open_store_ctx(tmp_path, b)
    try:
        with pytest.raises(OrderDispositionUnknownError):
            asyncio.run(b.execute_entry(_entry_envelope()))

        rows = list(ctx.iter_live_orders(symbol="VN30F1M"))
        assert rows, ("a lost-reply POST left NO durable row — the order may "
                      "exist at the venue and a crash during the engine park "
                      "loses it entirely (persist-first, capitalcom 6-point)")
    finally:
        store.close()


# --- GREEN control: no store, no behavior change -----------------------------

def __test_without_store_ctx_place_works_as_today__(fake_client, tmp_path):
    """store_ctx=None (backtests, unit tests, degraded runs) must keep
    today's behavior exactly — the journal is additive, never required."""
    b = _broker(fake_client, tmp_path, post_order=_PLACED)
    assert b.store_ctx is None

    orders = asyncio.run(b.execute_entry(_entry_envelope()))

    assert orders[0].id == "437346"


# --- G1: persist-FIRST — the row exists BEFORE the POST leaves ---------------

def __test_row_exists_before_the_post__(fake_client, tmp_path):
    """The six-point discipline's first window: a crash inside the POST call
    itself must already have the `submitted` row on disk."""
    b = _broker(fake_client, tmp_path)
    store, ctx = _open_store_ctx(tmp_path, b)
    seen_at_post = {}

    def _post(*_a, **_k):
        seen_at_post["rows"] = len(list(ctx.iter_live_orders(symbol="VN30F1M")))
        return _PLACED

    b._client._responses["post_order"] = _post
    try:
        asyncio.run(b.execute_entry(_entry_envelope()))
        assert seen_at_post["rows"] == 1, (
            "the journal row must be written BEFORE the POST (persist-first); "
            f"rows visible at POST time: {seen_at_post['rows']}")
    finally:
        store.close()


# --- G3: FOREIGN book rows are never adopted ---------------------------------

def __test_foreign_book_row_is_never_adopted__(fake_client, tmp_path):
    """The operator shares the account: a venue order with NO journal root
    must stay foreign — adoption is journal-ROOTED, never book-sourced."""
    b = _broker(fake_client, tmp_path,
                get_orders=(200, {"orders": [{"id": "999111", "symbol": "VN30F1M",
                                              "side": "NB", "quantity": 5,
                                              "orderStatus": "New"}],
                                  "totalPages": 1}))
    store, ctx = _open_store_ctx(tmp_path, b)   # journal is EMPTY
    try:
        asyncio.run(b.connect())
        pine_id, _f, _l = b._identity_for("999111")
        assert pine_id is None, (
            "a book row with no journal root was adopted — that is the "
            "operator's order (netting venue, shared account)")
    finally:
        store.close()


# --- crash-window state: a rejected place is journalled REJECTED, not adopted

def __test_rejected_place_is_journalled_and_skipped_by_restore__(fake_client, tmp_path):
    from pynecore.core.broker.exceptions import ExchangeOrderRejectedError
    from pynecore_dnse.journal_wiring import iter_journal_identities

    b = _broker(fake_client, tmp_path,
                post_order=(400, {"code": "INVALID_PRICE", "message": "bad"}))
    store, ctx = _open_store_ctx(tmp_path, b)
    try:
        with pytest.raises(ExchangeOrderRejectedError):
            asyncio.run(b.execute_entry(_entry_envelope()))
        assert list(iter_journal_identities(ctx)) == [], (
            "a REJECTED row has no venue order — restore must skip it")
    finally:
        store.close()


# --- terminal observations: fills keep the ledger, cancels close it ----------

def __test_fill_observation_keeps_the_exposure_row_live__(fake_client, tmp_path):
    """#73 (supersedes the original close-on-fill anchor): a FILLED terminal
    keeps the row LIVE with the watermark fields written — the rows are the
    run's exposure ledger (core's ``_durable_owned_signed_size`` sums
    ``filled_qty`` over LIVE rows only; closing the filled entry row made
    the startup ownership clamp refuse the run's own position). A
    zero-exposure terminal (cancel/reject) still closes the row."""
    from pynecore_dnse.journal_wiring import journal_terminal

    b = _broker(fake_client, tmp_path, post_order=_PLACED)
    store, ctx = _open_store_ctx(tmp_path, b)
    try:
        asyncio.run(b.execute_entry(_entry_envelope()))
        assert len(list(ctx.iter_live_orders(symbol="VN30F1M"))) == 1

        journal_terminal(ctx, venue_id="437346", terminal_status="Filled",
                         filled_qty=1.0)

        rows = list(ctx.iter_live_orders(symbol="VN30F1M"))
        assert len(rows) == 1, (
            "a filled row is the run's exposure ledger — it must stay LIVE")
        assert float(rows[0].filled_qty) == 1.0
        extras = rows[0].extras or {}
        assert extras.get("last_raw_status") == "Filled", (
            "#56 watermark: without last_raw_status the restart seed gate "
            "fails and the first poll re-emits the full fill")
        assert extras.get("last_fill_venue_id") == "437346"
        journal_terminal(ctx, venue_id="437346", terminal_status="Filled")  # idempotent
        assert len(list(ctx.iter_live_orders(symbol="VN30F1M"))) == 1
    finally:
        store.close()


def __test_zero_exposure_terminal_still_closes_the_row__(fake_client, tmp_path):
    from pynecore_dnse.journal_wiring import journal_terminal

    b = _broker(fake_client, tmp_path, post_order=_PLACED)
    store, ctx = _open_store_ctx(tmp_path, b)
    try:
        asyncio.run(b.execute_entry(_entry_envelope()))
        journal_terminal(ctx, venue_id="437346", terminal_status="Canceled")
        assert list(ctx.iter_live_orders(symbol="VN30F1M")) == [], (
            "a cancelled-unfilled order carries no exposure — its row closes")
    finally:
        store.close()


# --- widened None-store control: the cancel path -----------------------------

def __test_without_store_ctx_cancel_works_as_today__(fake_client, tmp_path):
    from pynecore.core.broker.models import CancelIntent, CancelDispositionOutcome

    b = _broker(fake_client, tmp_path,
                cancel_order=(200, {"orderStatus": "New"}),
                get_order_detail=(200, {"id": "ID1", "symbol": "VN30F1M",
                                        "side": "NB", "quantity": 1,
                                        "orderStatus": "Canceled",
                                        "fillQuantity": 0.0}))
    b._cancel_verify_attempts, b._cancel_verify_delay = 1, 0.0
    b._order_ids["K"] = ["ID1"]
    b._order_category["ID1"] = "NORMAL"
    assert b.store_ctx is None

    envelope = DispatchEnvelope(
        intent=CancelIntent(pine_id="K", symbol="VN30F1M"),
        run_tag="abcd", bar_ts_ms=1_700_000_000_000, retry_seq=0, coid_max_len=30)
    outcome = asyncio.run(b.execute_cancel_with_outcome(envelope))

    assert outcome is CancelDispositionOutcome.CANCEL_CONFIRMED
