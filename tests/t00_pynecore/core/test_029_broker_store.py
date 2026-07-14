"""
Standalone unit tests for the unified SQLite-based ``BrokerStore``.

These tests assume nothing about ``OrderSyncEngine`` or plugin
integration — only the storage layer in isolation. End-to-end
(sync_engine + storage) scenarios live under
``test_030_broker_store_sync_engine.py``.
"""
import json
import sqlite3
from pathlib import Path

import pytest

from pynecore.core.broker.run_identity import RunIdentity
from pynecore.core.broker.storage import (
    BrokerStore,
    EnvelopeRecord,
    HEARTBEAT_INTERVAL_MS,
    OrderRow,
    PendingRecord,
    PURGE_INTERVAL_MS,
    RETENTION_DAYS,
    RunContext,
    STALE_THRESHOLD_MS,
)


PLUGIN = "Capital.com"
SCRIPT_SOURCE = "// stub source for run_tag derivation\n"
BAR_TS = 1_700_000_000_000


def _make_identity(
        *, strategy_id: str = "ema_cross",
        symbol: str = "EURUSD",
        timeframe: str = "60",
        account_id: str = "capitalcom-demo-1234567",
        label: str | None = None,
) -> RunIdentity:
    return RunIdentity(
        strategy_id=strategy_id, symbol=symbol, timeframe=timeframe,
        account_id=account_id, label=label,
    )


def _open_run(store: BrokerStore, **overrides) -> RunContext:
    return store.open_run(
        _make_identity(**overrides),
        script_source=SCRIPT_SOURCE,
        script_path="strategies/ema_cross.py",
    )


# === Migration + PRAGMA ===================================================


def __test_migrations_applied_on_fresh_db__(tmp_path: Path) -> None:
    """First open creates the schema and adds a ``_migrations`` row."""
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN):
        pass

    # Inspect with raw sqlite3 — independent of BrokerStore.
    conn = sqlite3.connect(str(path))
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        assert version >= 1

        rows = conn.execute(
            "SELECT version, description FROM _migrations ORDER BY version"
        ).fetchall()
        assert rows, "the migrations table must hold at least one row"
        assert rows[0][0] == 1

        # Schema check: every main table exists.
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        expected = {
            '_migrations', 'runs', 'envelopes', 'pending_verifications',
            'orders', 'order_refs', 'events',
        }
        assert expected.issubset(tables)

        # The ``live_runs`` VIEW is also created.
        views = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='view'"
            )
        }
        assert 'live_runs' in views
    finally:
        conn.close()


def __test_live_runs_view_heals_on_threshold_drift__(tmp_path: Path) -> None:
    """A ``live_runs`` VIEW carrying a stale threshold is recreated on open.

    The migration chain is append-only history, so a change to
    ``STALE_THRESHOLD_MS`` cannot reach existing DBs through it — the
    self-heal on open is what keeps the stored VIEW aligned with the
    running code.
    """
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN):
        pass

    def _view_sql() -> str:
        conn = sqlite3.connect(str(path))
        try:
            return conn.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type='view' AND name='live_runs'"
            ).fetchone()[0]
        finally:
            conn.close()

    from pynecore.core.broker.storage import STALE_THRESHOLD_MS

    # Fresh DB: the VIEW already carries the current constant.
    assert f"- {STALE_THRESHOLD_MS}" in _view_sql()

    # Simulate a DB migrated under an older constant.
    conn = sqlite3.connect(str(path))
    try:
        with conn:
            conn.execute("DROP VIEW live_runs")
            conn.execute(
                "CREATE VIEW live_runs AS SELECT * FROM runs "
                "WHERE ended_ts_ms IS NULL "
                "AND last_heartbeat_ts_ms > ("
                "CAST(strftime('%s', 'now') AS INTEGER) * 1000 - 999)"
            )
    finally:
        conn.close()

    with BrokerStore(path, plugin_name=PLUGIN):
        pass
    healed = _view_sql()
    assert "- 999" not in healed
    assert f"- {STALE_THRESHOLD_MS}" in healed


def __test_migrations_idempotent_on_reopen__(tmp_path: Path) -> None:
    """The second open does not drop or duplicate the schema."""
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN):
        pass
    with BrokerStore(path, plugin_name=PLUGIN):
        pass

    conn = sqlite3.connect(str(path))
    try:
        rows = conn.execute("SELECT version FROM _migrations").fetchall()
        versions = sorted(r[0] for r in rows)
        # Each migration runs exactly once — no duplicates, and the
        # versions are strictly increasing.
        assert versions == sorted(set(versions))
        assert versions[0] == 1
        # Current schema cursor: the length of the _MIGRATIONS list.
        from pynecore.core.broker.storage import _MIGRATIONS
        assert versions == [v for v, _, _ in _MIGRATIONS]
    finally:
        conn.close()


def __test_wal_mode_enabled__(tmp_path: Path) -> None:
    """At start-up the DB must be in WAL mode — crash-safety + concurrent read."""
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"


# === open_run lifecycle ===================================================


def __test_open_run_inserts_runs_row_and_returns_context__(tmp_path: Path) -> None:
    """``open_run`` inserts an active ``runs`` row and returns its ``RunContext``."""
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        ctx = _open_run(store)

        assert ctx.run_id == "ema_cross@capitalcom-demo-1234567:EURUSD:60"
        assert len(ctx.run_tag) == 4
        assert ctx.run_instance_id >= 1

        row = store._conn.execute(
            "SELECT * FROM runs WHERE run_instance_id = ?",
            (ctx.run_instance_id,),
        ).fetchone()
        assert row['run_id'] == ctx.run_id
        assert row['run_tag'] == ctx.run_tag
        assert row['ended_ts_ms'] is None
        assert row['plugin_name'] == PLUGIN


def __test_open_run_with_label_includes_suffix__(tmp_path: Path) -> None:
    """A run ``label`` appends a ``#label`` suffix to ``run_id`` and changes the ``run_tag``."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx_a = _open_run(store, label="a")
        assert ctx_a.run_id.endswith("#a")
        ctx_a.close()

        ctx_b = _open_run(store, label="b")
        assert ctx_b.run_id.endswith("#b")
        # Different label → different run_tag too (input set widened).
        assert ctx_a.run_tag != ctx_b.run_tag


def __test_open_run_collision_raises__(tmp_path: Path) -> None:
    """Same run_id active twice → RuntimeError on the second open_run."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        _open_run(store)
        with pytest.raises(RuntimeError, match="Active run_id already exists"):
            _open_run(store)


def __test_close_allows_reopen_with_same_run_id__(tmp_path: Path) -> None:
    """After a run is closed, the same run_id can be reopened (with a new run_instance_id)."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx1 = _open_run(store)
        ctx1.close()

        ctx2 = _open_run(store)
        assert ctx2.run_id == ctx1.run_id
        assert ctx2.run_instance_id != ctx1.run_instance_id

        # Both rows are kept historically — "show every run of this bot"
        # remains queryable.
        run_rows = store._conn.execute(
            "SELECT run_instance_id, ended_ts_ms FROM runs WHERE run_id = ? "
            "ORDER BY run_instance_id",
            (ctx1.run_id,),
        ).fetchall()
        assert len(run_rows) == 2
        assert run_rows[0]['ended_ts_ms'] is not None  # first one is closed
        assert run_rows[1]['ended_ts_ms'] is None      # second one is active


# === Stale-run cleanup ====================================================


def __test_stale_run_auto_cleaned_on_open_run__(tmp_path: Path) -> None:
    """SIGKILL simulation: last_heartbeat_ts_ms expires → open_run closes it."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx_crashed = _open_run(store)
        # Simulate the missing heartbeat by faking the value directly in the DB.
        fake_stale_hb = 1_000_000  # far in the past
        store._conn.execute(
            "UPDATE runs SET last_heartbeat_ts_ms = ? WHERE run_instance_id = ?",
            (fake_stale_hb, ctx_crashed.run_instance_id),
        )
        store._conn.commit()

        # Restart simulation: new open_run on the same run_id.
        ctx_new = _open_run(store)
        assert ctx_new.run_instance_id != ctx_crashed.run_instance_id

        # The old row is closed, the new one is alive.
        old = store._conn.execute(
            "SELECT ended_ts_ms FROM runs WHERE run_instance_id = ?",
            (ctx_crashed.run_instance_id,),
        ).fetchone()
        assert old['ended_ts_ms'] == fake_stale_hb

        # Cleanup event logged.
        ev = store._conn.execute(
            "SELECT kind, payload FROM events "
            "WHERE run_instance_id = ? AND kind = 'stale_run_cleaned'",
            (ctx_crashed.run_instance_id,),
        ).fetchone()
        assert ev is not None
        payload = json.loads(ev['payload'])
        assert payload['run_id'] == ctx_crashed.run_id


def __test_cleanup_stale_runs_public_api__(tmp_path: Path) -> None:
    """The public ``cleanup_stale_runs`` also works (debug-CLI scenario)."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        store._conn.execute(
            "UPDATE runs SET last_heartbeat_ts_ms = 1 WHERE run_instance_id = ?",
            (ctx.run_instance_id,),
        )
        store._conn.commit()

        cleaned = store.cleanup_stale_runs()
        assert cleaned == 1


def __test_live_runs_view_excludes_stale__(tmp_path: Path) -> None:
    """The ``live_runs`` VIEW filters zombies even when no physical cleanup ran."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx_fresh = _open_run(store)
        ctx_stale = _open_run(store, label="stale")

        # The stale run's heartbeat is too old — the VIEW must skip it.
        store._conn.execute(
            "UPDATE runs SET last_heartbeat_ts_ms = 1 WHERE run_instance_id = ?",
            (ctx_stale.run_instance_id,),
        )
        store._conn.commit()

        rows = store._conn.execute(
            "SELECT run_instance_id FROM live_runs"
        ).fetchall()
        ids = {r['run_instance_id'] for r in rows}
        assert ctx_fresh.run_instance_id in ids
        assert ctx_stale.run_instance_id not in ids, (
            "live_runs VIEW must skip the stale row even when the runs "
            "table still has ``ended_ts_ms IS NULL`` for it"
        )


# === Envelope / pending replay ============================================


def __test_replay_round_trip_envelope_and_pending__(tmp_path: Path) -> None:
    """``replay`` returns recorded envelopes and parked pending records keyed by intent/coid."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        ctx.record_envelope(key="Long", bar_ts_ms=BAR_TS, retry_seq=0)
        ctx.record_envelope(key="TP\0Long", bar_ts_ms=BAR_TS, retry_seq=1)
        ctx.record_park(coid="coid-1", key="Long")

        envelopes, pending = ctx.replay()

    assert envelopes == {
        "Long": EnvelopeRecord(key="Long", bar_ts_ms=BAR_TS, retry_seq=0),
        "TP\0Long": EnvelopeRecord(key="TP\0Long", bar_ts_ms=BAR_TS, retry_seq=1),
    }
    park_row = pending.pop("coid-1")
    assert pending == {}
    assert park_row.parked_ts_ms > 0  # stamped at park time, replayed verbatim
    assert park_row == PendingRecord(
        key="Long", coid="coid-1", parked_ts_ms=park_row.parked_ts_ms,
    )


def __test_record_complete_drops_envelope_and_pending__(tmp_path: Path) -> None:
    """``record_complete`` clears the intent's envelope and all of its pending records."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        ctx.record_envelope(key="Long", bar_ts_ms=BAR_TS, retry_seq=0)
        ctx.record_park(coid="coid-1", key="Long")
        ctx.record_complete(key="Long")

        envelopes, pending = ctx.replay()
        assert envelopes == {}
        assert pending == {}


def __test_record_unpark_drops_only_that_coid__(tmp_path: Path) -> None:
    """``record_unpark`` removes only the named coid, leaving the envelope and other pendings."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        ctx.record_envelope(key="Long", bar_ts_ms=BAR_TS, retry_seq=0)
        ctx.record_park(coid="coid-1", key="Long")
        ctx.record_park(coid="coid-2", key="Long")
        ctx.record_unpark(coid="coid-1")

        envelopes, pending = ctx.replay()
        assert "Long" in envelopes
        assert list(pending) == ["coid-2"]


def __test_record_envelope_upsert_on_retry_seq_bump__(tmp_path: Path) -> None:
    """A repeated ``record_envelope`` for the same ``intent_key`` overwrites the row."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        ctx.record_envelope(key="Long", bar_ts_ms=BAR_TS, retry_seq=0)
        ctx.record_envelope(key="Long", bar_ts_ms=BAR_TS, retry_seq=1)

        envelopes, _ = ctx.replay()
        assert envelopes["Long"].retry_seq == 1


def __test_record_resolution_rejected_is_sticky_against_attached__(
        tmp_path: Path) -> None:
    """A later ``'attached'`` ``record_resolution`` must not overwrite a sticky ``'rejected'``.

    ``'rejected'`` is sticky: a later ``'attached'`` write must not
    overwrite it. Bracket scenario: the TP and SL legs each call
    ``record_resolution`` for the same parent COID. If the TP is
    ``'rejected'`` and the SL is ``'attached'``, an order-dependent
    naive UPDATE could end up with ``'attached'`` and the engine would
    keep the ExitIntent despite the missing TP. ``'rejected'`` always
    wins; re-dispatch is idempotent (a missing leg goes out again with
    the same parameters, an already-attached leg is a broker-side
    no-op).
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        ctx.record_envelope(key="Long", bar_ts_ms=BAR_TS, retry_seq=0)
        ctx.record_park(coid="coid-entry", key="Long")
        # TP missing: rejected.
        ctx.record_resolution("coid-entry", "rejected")
        # SL present: attached. Must NOT overwrite.
        ctx.record_resolution("coid-entry", "attached")

        resolutions = ctx.iter_pending_resolutions()
        assert len(resolutions) == 1
        assert resolutions[0].coid == "coid-entry"
        assert resolutions[0].resolution == "rejected", (
            f"'rejected' must remain sticky against later 'attached'; "
            f"got {resolutions[0].resolution!r}"
        )


def __test_record_resolution_attached_then_rejected_flips_to_rejected__(
        tmp_path: Path) -> None:
    """A later ``'rejected'`` ``record_resolution`` overwrites an earlier ``'attached'``.

    Reverse order: SL ``'attached'`` → TP ``'rejected'``. Here
    ``'attached'`` is written first and the later ``'rejected'``
    overwrites it. Guarantees that whatever order leg resolutions
    arrive in, the final state is ``'rejected'`` whenever any leg is
    missing.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        ctx.record_envelope(key="Long", bar_ts_ms=BAR_TS, retry_seq=0)
        ctx.record_park(coid="coid-entry", key="Long")
        ctx.record_resolution("coid-entry", "attached")
        ctx.record_resolution("coid-entry", "rejected")

        resolutions = ctx.iter_pending_resolutions()
        assert len(resolutions) == 1
        assert resolutions[0].resolution == "rejected"


def __test_record_resolution_attached_then_attached_stays_attached__(
        tmp_path: Path) -> None:
    """Two consecutive ``'attached'`` ``record_resolution`` writes are idempotent.

    Two consecutive ``'attached'`` writes are idempotent — the final
    state stays ``'attached'``. (TP attached + SL attached = bracket
    fully attached.)"""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        ctx.record_envelope(key="Long", bar_ts_ms=BAR_TS, retry_seq=0)
        ctx.record_park(coid="coid-entry", key="Long")
        ctx.record_resolution("coid-entry", "attached")
        ctx.record_resolution("coid-entry", "attached")

        resolutions = ctx.iter_pending_resolutions()
        assert len(resolutions) == 1
        assert resolutions[0].resolution == "attached"


def __test_record_park_resets_stale_resolution_on_repark__(
        tmp_path: Path) -> None:
    """Re-parking the same ``(run_id, client_order_id)`` clears the previous ``resolution`` to NULL.

    When the same ``(run_id, client_order_id)`` is re-parked, the
    previous ``'attached'`` (or any) ``resolution`` is *cleared* (reset
    to NULL). Otherwise a modify/retry timeout that re-parks the same
    coid would keep the old ``'attached'`` decision and the next
    restart's
    :meth:`OrderSyncEngine._consume_plugin_resolutions` would
    immediately adopt the freshly parked dispatch — yet the whole point
    of the new park is that the exchange-side state is *unknown*.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        ctx.record_envelope(key="Long", bar_ts_ms=BAR_TS, retry_seq=0)
        ctx.record_park(coid="coid-entry", key="Long")
        ctx.record_resolution("coid-entry", "attached")

        # Sanity: the 'attached' resolution is present.
        before = ctx.iter_pending_resolutions()
        assert len(before) == 1 and before[0].resolution == "attached"

        # Re-park onto the same coid (modify/retry timeout simulation).
        ctx.record_park(coid="coid-entry", key="Long")

        # ``resolution`` is now NULL → ``iter_pending_resolutions``
        # (which only requests resolved rows) returns an empty list.
        after = ctx.iter_pending_resolutions()
        assert after == [], (
            f"re-park must clear stale resolution; got {after!r}"
        )

        # Belt-and-braces: ``replay()`` also sees the NULL-reset row, so
        # after a restart the engine will not produce a stale adopt.
        _, pending = ctx.replay()
        assert "coid-entry" in pending
        assert pending["coid-entry"].resolution is None, (
            f"re-parked row's persisted resolution must be NULL, "
            f"got {pending['coid-entry'].resolution!r}"
        )


# === Multi-run isolation ==================================================


def __test_replay_isolated_per_run_instance__(tmp_path: Path) -> None:
    """Two parallel runs do not see each other's envelopes."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx_a = _open_run(store, timeframe="15")
        ctx_b = _open_run(store, timeframe="60")

        ctx_a.record_envelope(key="Long", bar_ts_ms=BAR_TS, retry_seq=0)
        ctx_b.record_envelope(key="Short", bar_ts_ms=BAR_TS, retry_seq=0)

        env_a, _ = ctx_a.replay()
        env_b, _ = ctx_b.replay()

        assert set(env_a) == {"Long"}
        assert set(env_b) == {"Short"}


# === Orders + order_refs ==================================================


def __test_upsert_order_new_then_update__(tmp_path: Path) -> None:
    """``upsert_order`` inserts a new order, then a state update leaves other fields intact."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)

        # New row
        ctx.upsert_order(
            "coid-1", symbol="EURUSD", side="buy", qty=1.0,
            state="submitted", from_entry="L", pine_entry_id="L",
            sl_level=1.0, tp_level=1.2,
            extras={"deal_reference": "o_abc"},
        )

        row = ctx.get_order("coid-1")
        assert row is not None
        assert row.symbol == "EURUSD"
        assert row.qty == 1.0
        assert row.state == "submitted"
        assert row.sl_level == 1.0
        assert row.extras == {"deal_reference": "o_abc"}

        # State update
        ctx.set_order_state("coid-1", "confirmed")
        row = ctx.get_order("coid-1")
        assert row is not None
        assert row.state == "confirmed"
        # Fields not passed must remain untouched.
        assert row.sl_level == 1.0
        assert row.extras == {"deal_reference": "o_abc"}


def __test_upsert_order_requires_core_fields_for_new_row__(tmp_path: Path) -> None:
    """``upsert_order`` raises ``ValueError`` when a new row omits required core fields."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        with pytest.raises(ValueError, match="missing required fields"):
            ctx.upsert_order("coid-missing", state="submitted")


def __test_add_ref_and_find_by_ref_round_trip__(tmp_path: Path) -> None:
    """Generic alias lookup: deal_reference → OrderRow via one indexed SELECT."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        ctx.upsert_order(
            "coid-1", symbol="EURUSD", side="buy", qty=1.0, state="submitted",
        )
        ctx.add_ref("coid-1", "deal_reference", "o_abc")
        ctx.add_ref("coid-1", "deal_id", "p_xyz")

        found_by_ref = ctx.find_by_ref("deal_reference", "o_abc")
        found_by_id = ctx.find_by_ref("deal_id", "p_xyz")
        assert found_by_ref is not None and found_by_ref.client_order_id == "coid-1"
        assert found_by_id is not None and found_by_id.client_order_id == "coid-1"

        # Unknown key → None, no raise.
        assert ctx.find_by_ref("deal_id", "ghost") is None


def __test_close_order_cascades_delete_refs__(tmp_path: Path) -> None:
    """``close_order`` marks the order closed and cascade-deletes its ``order_refs`` aliases."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        ctx.upsert_order(
            "coid-1", symbol="EURUSD", side="buy", qty=1.0, state="submitted",
        )
        ctx.add_ref("coid-1", "deal_reference", "o_abc")
        ctx.add_ref("coid-1", "deal_id", "p_xyz")

        ctx.close_order("coid-1")

        # The order still exists but is closed.
        row = ctx.get_order("coid-1")
        assert row is not None
        assert row.closed_ts_ms is not None
        # The refs were cascade-deleted.
        assert ctx.find_by_ref("deal_reference", "o_abc") is None
        assert ctx.find_by_ref("deal_id", "p_xyz") is None


def __test_reopen_order_clears_closed_ts_ms_and_logs_event__(tmp_path: Path) -> None:
    """``reopen_order`` nulls ``closed_ts_ms`` and writes an ``order_reopened`` event.

    ``reopen_order`` nulls ``closed_ts_ms`` and writes an
    ``order_reopened`` event. A previously closed row can come back to
    the live range (e.g. a bracket leg COID that ``close_order`` had
    closed after an earlier REJECTED attach, and now a fresh protective
    leg has been attached at the broker under the same COID).
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        ctx.upsert_order(
            "coid-1", symbol="EURUSD", side="buy", qty=1.0, state="rejected",
        )
        ctx.close_order("coid-1")
        pre = ctx.get_order("coid-1")
        assert pre is not None and pre.closed_ts_ms is not None

        ctx.reopen_order("coid-1")
        post = ctx.get_order("coid-1")
        assert post is not None
        assert post.closed_ts_ms is None, (
            f"reopen_order must null closed_ts_ms; got {post.closed_ts_ms!r}"
        )
        # The row is now visible to iter_live_orders again.
        live_ids = {r.client_order_id for r in ctx.iter_live_orders()}
        assert "coid-1" in live_ids, (
            "reopened order must appear in iter_live_orders"
        )
        # Audit-event written.
        kinds = [p for p in ctx.iter_events_by_kind_since(
            'order_reopened', 0,
        )]
        # No payload on the reopen event — just verify presence by
        # walking the events table directly via the existing helper:
        # the iterator yields parsed payloads only for non-empty
        # payloads, so we confirm via SQL that the audit row exists.
        cur = store._conn.execute(
            "SELECT COUNT(*) AS n FROM events "
            "WHERE kind = 'order_reopened' AND client_order_id = ?",
            ("coid-1",),
        )
        assert cur.fetchone()['n'] == 1, (
            f"order_reopened event must be written; "
            f"iter_events sample={kinds!r}"
        )


def __test_reopen_order_no_op_on_unknown_or_already_open_row__(tmp_path: Path) -> None:
    """``reopen_order`` is a silent no-op on an unknown or already-open ``client_order_id``.

    ``reopen_order`` does not raise on an unknown client_order_id or
    on an already-open row: it returns silently and writes no audit
    event.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        # Unknown coid — must not raise.
        ctx.reopen_order("nonexistent-coid")
        # Already-open row — must not raise, must not write event.
        ctx.upsert_order(
            "coid-open", symbol="EURUSD", side="buy", qty=1.0, state="submitted",
        )
        ctx.reopen_order("coid-open")
        cur = store._conn.execute(
            "SELECT COUNT(*) AS n FROM events WHERE kind = 'order_reopened'",
        )
        assert cur.fetchone()['n'] == 0, (
            "reopen_order on a still-open row must not write an audit event"
        )


def __test_iter_live_orders_filters_closed__(tmp_path: Path) -> None:
    """``iter_live_orders`` excludes orders that have been closed."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        for coid in ("a", "b", "c"):
            ctx.upsert_order(
                coid, symbol="EURUSD", side="buy", qty=1.0, state="submitted",
            )
        ctx.close_order("b")

        live_ids = {r.client_order_id for r in ctx.iter_live_orders()}
        assert live_ids == {"a", "c"}


def __test_iter_live_orders_symbol_filter__(tmp_path: Path) -> None:
    """``iter_live_orders(symbol=...)`` returns only orders for the requested symbol."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        ctx.upsert_order(
            "a", symbol="EURUSD", side="buy", qty=1.0, state="submitted",
        )
        ctx.upsert_order(
            "b", symbol="GBPUSD", side="buy", qty=1.0, state="submitted",
        )

        eur = {r.client_order_id for r in ctx.iter_live_orders(symbol="EURUSD")}
        assert eur == {"a"}


def __test_set_risk_preserves_unset_fields__(tmp_path: Path) -> None:
    """``set_risk`` ``None`` parameters do not erase existing values."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        ctx.upsert_order(
            "c", symbol="EURUSD", side="buy", qty=1.0, state="submitted",
            sl_level=1.0, tp_level=1.2,
        )
        ctx.set_risk("c", sl=1.1)  # only set SL

        row = ctx.get_order("c")
        assert row is not None
        assert row.sl_level == 1.1
        assert row.tp_level == 1.2, "set_risk=None must not clear the TP"


# === Extras JSON ==========================================================


def __test_extras_round_trip_preserves_dict__(tmp_path: Path) -> None:
    """An order's ``extras`` dict round-trips through JSON, preserving nested and unicode values."""
    payload = {
        "deal_reference": "o_abc",
        "working_order": True,
        "nested": {"x": 1, "y": [1, 2, 3]},
        "unicode": "Falsches Üben von Xylophonmusik quält größere Zwerge",
    }
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        ctx.upsert_order(
            "coid-1", symbol="EURUSD", side="buy", qty=1.0, state="submitted",
            extras=payload,
        )
        row = ctx.get_order("coid-1")
    assert row is not None
    assert row.extras == payload


# === Events ===============================================================


def __test_log_event_written_with_payload__(tmp_path: Path) -> None:
    """``log_event`` writes an events row with its kind, coid, intent key and JSON payload."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        ctx.log_event(
            "dispatch", client_order_id="coid-1", intent_key="Long",
            payload={"status": "submitted", "details": {"foo": 1}},
        )

        rows = store._conn.execute(
            "SELECT kind, client_order_id, intent_key, payload FROM events "
            "WHERE run_instance_id = ?",
            (ctx.run_instance_id,),
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]['kind'] == "dispatch"
    assert rows[0]['client_order_id'] == "coid-1"
    assert json.loads(rows[0]['payload']) == {
        "status": "submitted", "details": {"foo": 1}
    }


# === Heartbeat ============================================================


def __test_heartbeat_rate_limited__(tmp_path: Path) -> None:
    """Two heartbeats in quick succession → only the first writes to the DB."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        # open_run already wrote a last_heartbeat_ts_ms; capture it.
        hb_after_open = store._conn.execute(
            "SELECT last_heartbeat_ts_ms FROM runs WHERE run_instance_id = ?",
            (ctx.run_instance_id,),
        ).fetchone()[0]

        # Pretend we wrote recently (now ms). This activates the gate.
        from pynecore.core.broker import storage as _storage
        ctx._last_heartbeat_write_ms = _storage._now_ms()
        ctx.heartbeat()  # no-op — too soon after the last write

        hb_after_nothing = store._conn.execute(
            "SELECT last_heartbeat_ts_ms FROM runs WHERE run_instance_id = ?",
            (ctx.run_instance_id,),
        ).fetchone()[0]
        assert hb_after_nothing == hb_after_open, "rate-limit gate must skip the UPDATE"

        # Now pretend the previous write is older than the interval.
        ctx._last_heartbeat_write_ms = (
            _storage._now_ms() - HEARTBEAT_INTERVAL_MS - 1
        )
        ctx.heartbeat()

        hb_after_write = store._conn.execute(
            "SELECT last_heartbeat_ts_ms FROM runs WHERE run_instance_id = ?",
            (ctx.run_instance_id,),
        ).fetchone()[0]
        assert hb_after_write >= hb_after_open, "the allowed call must write to the DB"


# === Retention purge =======================================================


def _backdated_ts() -> int:
    """A timestamp safely past the default retention window."""
    from pynecore.core.broker import storage as _storage
    return _storage._now_ms() - (RETENTION_DAYS + 10) * 86_400_000


def __test_retention_purge_removes_expired_rows__(tmp_path: Path) -> None:
    """Old closed orders + their events are purged; live orders (and
    their events) and recent events survive the same cutoff."""
    old_ts = _backdated_ts()
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        ctx.upsert_order(
            "coid-old", symbol="EURUSD", side="buy", qty=1.0, state="filled",
        )
        ctx.close_order("coid-old")
        ctx.upsert_order(
            "coid-live", symbol="EURUSD", side="buy", qty=1.0, state="filled",
        )
        ctx.log_event("dispatch", client_order_id="coid-old")
        ctx.log_event("dispatch", client_order_id="coid-live")
        ctx.log_event("misc")

        # Backdate everything written so far past the retention window.
        store._conn.execute("UPDATE events SET ts_ms = ?", (old_ts,))
        store._conn.execute(
            "UPDATE orders SET closed_ts_ms = ? WHERE closed_ts_ms IS NOT NULL",
            (old_ts,),
        )
        # Written after the backdating — recent, must survive.
        ctx.log_event("fresh")

        deleted = store.cleanup_old_data()
        assert deleted > 0

        coids = {
            r[0] for r in store._conn.execute(
                "SELECT client_order_id FROM orders"
            )
        }
        assert coids == {"coid-live"}, "only the live order row survives"

        events = {
            (r[0], r[1]) for r in store._conn.execute(
                "SELECT kind, client_order_id FROM events"
            )
        }
        assert events == {
            ("dispatch", "coid-live"),  # live-order guard
            ("fresh", None),            # inside the retention window
        }


def __test_retention_purge_keeps_events_of_live_intents__(tmp_path: Path) -> None:
    """An old event whose ``intent_key`` still has a live envelope is
    protected; completing the intent makes it purgeable."""
    old_ts = _backdated_ts()
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        ctx.record_envelope("Long", BAR_TS, 0)
        ctx.log_event("defensive_close_filled", intent_key="Long")
        store._conn.execute("UPDATE events SET ts_ms = ?", (old_ts,))

        store.cleanup_old_data()
        remaining = store._conn.execute(
            "SELECT COUNT(*) FROM events WHERE intent_key = 'Long'"
        ).fetchone()[0]
        assert remaining == 1, "live envelope must protect the event"

        ctx.record_complete("Long")
        store.cleanup_old_data()
        remaining = store._conn.execute(
            "SELECT COUNT(*) FROM events WHERE intent_key = 'Long'"
        ).fetchone()[0]
        assert remaining == 0, "completed intent's old event is purgeable"


def __test_retention_purge_removes_childless_ended_runs__(tmp_path: Path) -> None:
    """An old ended run with no child rows is deleted; the live run stays."""
    old_ts = _backdated_ts()
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx_old = _open_run(store, label="old")
        ctx_old.close()
        ctx = _open_run(store)

        store._conn.execute(
            "UPDATE runs SET ended_ts_ms = ? WHERE ended_ts_ms IS NOT NULL",
            (old_ts,),
        )
        store.cleanup_old_data()

        instance_ids = {
            r[0] for r in store._conn.execute(
                "SELECT run_instance_id FROM runs"
            )
        }
        assert instance_ids == {ctx.run_instance_id}


def __test_heartbeat_daily_purge_gate__(tmp_path: Path) -> None:
    """Heartbeat triggers the retention purge at most once per
    ``PURGE_INTERVAL_MS``."""
    from pynecore.core.broker import storage as _storage
    old_ts = _backdated_ts()
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        ctx.log_event("misc")
        store._conn.execute("UPDATE events SET ts_ms = ?", (old_ts,))

        # Gate is closed: open_run just purged.
        ctx._last_heartbeat_write_ms = 0
        ctx.heartbeat()
        count = store._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == 1, "purge must not run again within the interval"

        # Open the gate and heartbeat again.
        store._last_purge_ms = _storage._now_ms() - PURGE_INTERVAL_MS - 1
        ctx._last_heartbeat_write_ms = 0
        ctx.heartbeat()
        count = store._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == 0, "heartbeat past the gate must purge the old event"


# === Close (happy path) ====================================================


def __test_close_run_sets_ended_ts_ms__(tmp_path: Path) -> None:
    """Closing a ``RunContext`` stamps the run's ``ended_ts_ms`` (was NULL while active)."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        assert store._conn.execute(
            "SELECT ended_ts_ms FROM runs WHERE run_instance_id = ?",
            (ctx.run_instance_id,),
        ).fetchone()[0] is None

        ctx.close()
        ended = store._conn.execute(
            "SELECT ended_ts_ms FROM runs WHERE run_instance_id = ?",
            (ctx.run_instance_id,),
        ).fetchone()[0]
    assert ended is not None


def __test_close_idempotent__(tmp_path: Path) -> None:
    """Calling ``RunContext.close`` a second time is a no-op and does not raise."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        ctx.close()
        # A second close must not raise.
        ctx.close()


# === Torn write / recovery ================================================


def __test_persistence_survives_reopen__(tmp_path: Path) -> None:
    """Writes from the first open are readable in the second open."""
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        ctx.record_envelope(key="Long", bar_ts_ms=BAR_TS, retry_seq=0)
        ctx.upsert_order(
            "coid-1", symbol="EURUSD", side="buy", qty=1.0, state="submitted",
        )
        saved_instance_id = ctx.run_instance_id

    # Second open with a separate BrokerStore — must still be queryable.
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        row = store._conn.execute(
            "SELECT run_id FROM runs WHERE run_instance_id = ?",
            (saved_instance_id,),
        ).fetchone()
        assert row is not None
        assert row['run_id'] == "ema_cross@capitalcom-demo-1234567:EURUSD:60"

        # Reopening the same run would now collide (the previous run is
        # still alive, heartbeat is fresh). That is the correct
        # behaviour — the ``close`` and stale-cleanup scenarios cover
        # the rest separately.


# === OrderRow field coverage ==============================================


def __test_order_row_fields_are_typed__(tmp_path: Path) -> None:
    """The dataclass returns fields with their proper types (not raw sqlite.Row)."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        ctx.upsert_order(
            "coid-1", symbol="EURUSD", side="buy", qty=1.0, state="submitted",
            trailing_stop=True, trailing_distance=5.0,
        )
        row = ctx.get_order("coid-1")
    assert isinstance(row, OrderRow)
    assert row is not None
    assert row.trailing_stop is True
    assert row.trailing_distance == 5.0
    assert row.closed_ts_ms is None
