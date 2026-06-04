"""
Unified SQLite-backed broker storage.

This module replaces the previously split persistence:

- the core ``state_store.py`` (append-only JSONL, envelope + parked verifications),
- plugin-level ledgers (Capital.com ``DealLedger`` etc.).

:class:`BrokerStore` writes every broker-relevant piece of state into a
single SQLite file: the sync-engine envelope identity, parked
dispatches, the live view of orders, the generic alias table (for
broker-specific lookup keys) and the structured audit log.

Main abstractions of the module:

- :class:`RunIdentity` — the run's human-readable logical key
  (``{strategy_id}@{account}:{symbol}:{timeframe}[#label]``).
  Constructed before storage is opened.
- :class:`RunContext` — context object for a concrete invocation. The
  ``run_instance_id`` (physical autoincrement FK) lives here; the
  caller (sync engine, plugin) never sees it.
- :class:`BrokerStore` — the lifecycle: ``open_run()`` returns a
  ``RunContext``, handles stale-run cleanup and schema migration.

Two crash-recovery mechanisms run side by side:

1. **Passive** — the ``live_runs`` VIEW automatically excludes rows
   whose ``last_heartbeat_ts_ms`` is past the threshold. The dashboard
   always sees the right answer even if physical cleanup has not run
   yet.
2. **Active** — every ``open_run()`` closes expired rows by setting
   ``ended_ts_ms`` and writing a ``stale_run_cleaned`` event.

Transactionality: every compound operation (e.g. ``close_order`` =
update orders + delete order_refs + insert events) runs in a single
``BEGIN IMMEDIATE ... COMMIT`` block. No half-written state, no
half-lost log.
"""
import json
import logging
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from pynecore.core.broker.run_identity import RunIdentity

__all__ = [
    'BrokerStore',
    'RunContext',
    'RunIdentity',
    'EnvelopeRecord',
    'PendingRecord',
    'OrderRow',
    'HEARTBEAT_INTERVAL_MS',
    'STALE_THRESHOLD_MS',
]

_log = logging.getLogger(__name__)

# Heartbeat cadence: a denser ``RunContext.heartbeat()`` call is a no-op.
HEARTBEAT_INTERVAL_MS: Final[int] = 60_000  # 1 minute
# A run is considered stale after this much heartbeat silence. The same
# value is HARD-CODED inside the ``live_runs`` VIEW's SQL — if it
# changes here, the VIEW must be replaced via a new migration
# (see ``_MIGRATIONS``).
STALE_THRESHOLD_MS: Final[int] = 5 * HEARTBEAT_INTERVAL_MS  # 5 minutes


def _now_ms() -> int:
    """Current time in ms, UTC. Centralised so the time import lives in one place."""
    return int(time.time() * 1000)


# === Replay-result dataclasses =============================================

@dataclass(frozen=True)
class EnvelopeRecord:
    """Replay output for a live envelope.

    Same shape as the former ``state_store.EnvelopeRecord`` — the sync
    engine consumes it unchanged.
    """
    key: str
    bar_ts_ms: int
    retry_seq: int


@dataclass(frozen=True)
class PendingRecord:
    """Replay output for a parked dispatch.

    ``resolution`` is ``None`` while the dispatch is parked; it flips to
    ``'attached'`` or ``'rejected'`` once the plugin decides the outcome
    via a snapshot-recovery path (see
    :meth:`RunContext.record_resolution`). The engine consumes and
    deletes the row on the next
    :meth:`OrderSyncEngine._verify_pending_dispatches` cycle.

    ``dispatch_kind`` distinguishes a parked new dispatch
    (``'new'``: ``execute_entry`` / ``execute_exit`` / ``execute_close``)
    from a parked amend (``'modify'``: ``modify_entry`` /
    ``modify_exit``). On a ``'rejected'`` resolution the engine uses
    this to decide whether to clear the ``_active_intents`` /
    ``_order_mapping`` slot tied to a now-live exchange order (yes for
    new dispatches — no broker-side order ever materialised) or to
    keep the original mapping and only drop the parked envelope
    (modify case — the original order is still live and the next
    :meth:`OrderSyncEngine._diff_and_dispatch` re-emits a modify rather
    than a fresh order). Defaults to ``'new'`` for pre-v3 rows and any
    code path that did not specify the kind explicitly.
    """
    key: str
    coid: str
    resolution: str | None = None
    dispatch_kind: str = 'new'
    order_ids: list[str] = field(default_factory=list)


@dataclass
class OrderRow:
    """One row of the ``orders`` table, exposed to the caller.

    ``extras`` is already a parsed dict (decoded from JSON), not the raw
    string — so plugins can access broker-specific fields natively.
    """
    client_order_id: str
    plugin_name: str
    intent_key: str | None
    exchange_order_id: str | None
    symbol: str
    side: str
    qty: float
    filled_qty: float
    state: str
    from_entry: str | None
    pine_entry_id: str | None
    sl_level: float | None
    tp_level: float | None
    trailing_stop: bool
    trailing_distance: float | None
    created_ts_ms: int
    updated_ts_ms: int
    closed_ts_ms: int | None
    extras: dict = field(default_factory=dict)


# === Schema migrations =====================================================

# This list is **append-only** — old tuples are never modified because
# they already represent state applied in production DBs. A new column
# or table arrives as a new tuple.
_MIGRATIONS: list[tuple[int, str, str]] = [
    (1, "initial schema", """
        CREATE TABLE _migrations (
            version       INTEGER PRIMARY KEY,
            applied_ts_ms INTEGER NOT NULL,
            description   TEXT NOT NULL
        );

        CREATE TABLE runs (
            run_instance_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id               TEXT NOT NULL,
            run_tag              TEXT NOT NULL,
            strategy_id          TEXT NOT NULL,
            script_path          TEXT NOT NULL,
            symbol               TEXT NOT NULL,
            timeframe            TEXT NOT NULL,
            account_id           TEXT NOT NULL,
            run_label            TEXT,
            plugin_name          TEXT NOT NULL,
            started_ts_ms        INTEGER NOT NULL,
            last_heartbeat_ts_ms INTEGER NOT NULL,
            ended_ts_ms          INTEGER
        );
        CREATE INDEX idx_runs_run_id    ON runs(run_id);
        CREATE INDEX idx_runs_active    ON runs(run_id)
            WHERE ended_ts_ms IS NULL;
        CREATE INDEX idx_runs_heartbeat ON runs(last_heartbeat_ts_ms)
            WHERE ended_ts_ms IS NULL;

        -- The 300000 ms (5 minute) threshold is HARD-CODED because SQLite
        -- does not support parameterised VIEWs. If STALE_THRESHOLD_MS
        -- changes, a new migration must DROP VIEW + CREATE VIEW.
        CREATE VIEW live_runs AS
            SELECT *
            FROM runs
            WHERE ended_ts_ms IS NULL
              AND last_heartbeat_ts_ms > (
                  CAST(strftime('%s', 'now') AS INTEGER) * 1000 - 300000
              );

        -- Envelopes and pending_verifications are scoped to the LOGICAL
        -- run_id, not run_instance_id — they are the broker-side
        -- idempotency anchors that every restart (new instance)
        -- inherits, because the same bot starts again with the same
        -- intents. Orders/order_refs/events, by contrast, are
        -- instance-scoped because they belong to the historical runs.
        CREATE TABLE envelopes (
            run_id           TEXT NOT NULL,
            intent_key       TEXT NOT NULL,
            bar_ts_ms        INTEGER NOT NULL,
            retry_seq        INTEGER NOT NULL,
            updated_ts_ms    INTEGER NOT NULL,
            PRIMARY KEY (run_id, intent_key)
        );

        CREATE TABLE pending_verifications (
            run_id           TEXT NOT NULL,
            client_order_id  TEXT NOT NULL,
            intent_key       TEXT NOT NULL,
            parked_ts_ms     INTEGER NOT NULL,
            PRIMARY KEY (run_id, client_order_id)
        );

        CREATE TABLE orders (
            run_instance_id    INTEGER NOT NULL,
            client_order_id    TEXT NOT NULL,
            plugin_name        TEXT NOT NULL,
            intent_key         TEXT,
            exchange_order_id  TEXT,
            symbol             TEXT NOT NULL,
            side               TEXT NOT NULL,
            qty                REAL NOT NULL,
            filled_qty         REAL DEFAULT 0.0,
            state              TEXT NOT NULL,
            from_entry         TEXT,
            pine_entry_id      TEXT,
            sl_level           REAL,
            tp_level           REAL,
            trailing_stop      INTEGER DEFAULT 0,
            trailing_distance  REAL,
            created_ts_ms      INTEGER NOT NULL,
            updated_ts_ms      INTEGER NOT NULL,
            closed_ts_ms       INTEGER,
            extras             TEXT,
            PRIMARY KEY (run_instance_id, client_order_id),
            FOREIGN KEY (run_instance_id) REFERENCES runs(run_instance_id)
        );
        CREATE INDEX idx_orders_live  ON orders(run_instance_id, symbol)
            WHERE closed_ts_ms IS NULL;
        CREATE INDEX idx_orders_entry ON orders(run_instance_id, from_entry)
            WHERE closed_ts_ms IS NULL;

        CREATE TABLE order_refs (
            run_instance_id  INTEGER NOT NULL,
            ref_type         TEXT NOT NULL,
            ref_value        TEXT NOT NULL,
            client_order_id  TEXT NOT NULL,
            created_ts_ms    INTEGER NOT NULL,
            PRIMARY KEY (run_instance_id, ref_type, ref_value),
            FOREIGN KEY (run_instance_id) REFERENCES runs(run_instance_id)
        );
        CREATE INDEX idx_order_refs_coid ON order_refs(run_instance_id, client_order_id);

        CREATE TABLE events (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            run_instance_id    INTEGER NOT NULL,
            ts_ms              INTEGER NOT NULL,
            plugin_name        TEXT NOT NULL,
            kind               TEXT NOT NULL,
            client_order_id    TEXT,
            exchange_order_id  TEXT,
            intent_key         TEXT,
            payload            TEXT,
            FOREIGN KEY (run_instance_id) REFERENCES runs(run_instance_id)
        );
        CREATE INDEX idx_events_run_ts  ON events(run_instance_id, ts_ms);
        CREATE INDEX idx_events_coid    ON events(run_instance_id, client_order_id);
        CREATE INDEX idx_events_kind_ts ON events(run_instance_id, kind, ts_ms);
    """),
    (2, "pending_verifications resolution column", """
        -- Plugin-driven resolution channel for parked dispatches whose
        -- exchange-side disposition the engine cannot observe via
        -- get_open_orders (e.g. position-attached brackets on Capital.com).
        -- The plugin's snapshot recovery writes 'attached' or 'rejected'
        -- here; OrderSyncEngine._verify_pending_dispatches consumes it on
        -- the next sync. NULL = still parked, default behaviour.
        ALTER TABLE pending_verifications
            ADD COLUMN resolution TEXT;
    """),
    (3, "pending_verifications dispatch_kind column", """
        -- Distinguishes a parked 'new' dispatch (execute_entry/exit/close)
        -- from a parked 'modify' dispatch (modify_entry/modify_exit). The
        -- 'rejected' path in OrderSyncEngine._consume_plugin_resolutions
        -- must NOT clear _active_intents/_order_mapping when the original
        -- exchange order is still live and only the amend failed —
        -- otherwise the next _diff_and_dispatch treats the Pine intent as
        -- brand new and re-dispatches via execute_*, creating a duplicate
        -- order alongside the still-live original. Default 'new' covers
        -- pre-migration rows and the unspecified-kind path.
        ALTER TABLE pending_verifications
            ADD COLUMN dispatch_kind TEXT NOT NULL DEFAULT 'new';
    """),
    (4, "pending_verifications order_ids column", """
        -- Stores the ``_order_mapping[key]`` snapshot at park time as a
        -- JSON array so a post-restart modify-rejected resolution can
        -- recover the original exchange order IDs and prevent a duplicate
        -- ``execute_*`` dispatch.
        ALTER TABLE pending_verifications
            ADD COLUMN order_ids TEXT NOT NULL DEFAULT '[]';
    """),
]


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Migrate the schema from ``PRAGMA user_version`` up to the latest.

    :param conn: An open ``sqlite3.Connection``. The function opens
        transaction blocks on the connection; the caller must be in a
        transaction-free state (``conn.isolation_level`` at its default
        ``""``).
    """
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    for version, description, sql in _MIGRATIONS:
        if version <= current:
            continue
        # ``with conn`` opens an implicit BEGIN DEFERRED and commits on
        # exit. One version = one transaction; a half-migrated schema
        # must not remain.
        with conn:
            conn.executescript(sql)
            # The ``_migrations`` table exists only after the first
            # migration runs — on the very first invocation the
            # CREATE TABLE is part of the script, so the INSERT below
            # is already safe.
            conn.execute(
                "INSERT INTO _migrations (version, applied_ts_ms, description) "
                "VALUES (?, ?, ?)",
                (version, _now_ms(), description),
            )
            conn.execute(f"PRAGMA user_version = {version}")
        _log.info("broker storage migrated to version %d (%s)", version, description)


# === BrokerStore ===========================================================

class BrokerStore:
    """Unified SQLite broker-state store for one workdir.

    Construction opens the DB, applies migrations and sets up the
    WAL + crash-safe PRAGMAs. Every ``open_run()`` returns a fresh
    :class:`RunContext` — every actual data movement goes through it.

    :param path: Absolute path of the SQLite file. The parent directory
        is created automatically.
    :param plugin_name: The BrokerPlugin's ``plugin_name`` attribute
        (e.g. ``"Capital.com"``). Every ``events`` / ``orders`` row
        carries this value so a multi-plugin workdir can be filtered.
    """

    def __init__(self, path: Path | str, *, plugin_name: str) -> None:
        self._path = Path(path)
        self._plugin_name = plugin_name
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # ``isolation_level=""`` = default; we open explicit transactions
        # via ``with conn:`` blocks. The sqlite3 module's autocommit
        # mode is not what it looks like at first glance — the default
        # behaviour is to start an implicit BEGIN before DML and close
        # it on the next commit. That fits our needs.
        # noinspection PyTypeChecker
        self._conn: sqlite3.Connection = sqlite3.connect(
            str(self._path),
            isolation_level="",
            check_same_thread=False,
            timeout=5.0,
        )
        # ``sqlite3.Row`` returns columns accessible by name — the
        # query helpers stay position-insensitive, so adding a column
        # later is easy.
        self._conn.row_factory = sqlite3.Row
        self._configure_pragmas()
        _apply_migrations(self._conn)

    def _configure_pragmas(self) -> None:
        """Configure WAL + crash-safety + FK + busy-timeout."""
        # WAL: concurrent read + single writer; crash-safe under power loss.
        self._conn.execute("PRAGMA journal_mode=WAL")
        # NORMAL: crash-safe with WAL, faster than FULL.
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        # 5 s wait on lock collision — rare with a single writer, but
        # parallel debug-CLI invocations can trigger it.
        self._conn.execute("PRAGMA busy_timeout=5000")

    # --- Lifecycle ---------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def plugin_name(self) -> str:
        return self._plugin_name

    def close(self) -> None:
        """Close the connection. Repeated calls are no-ops."""
        if self._conn is None:
            return
        try:
            self._conn.close()
        finally:
            self._conn = None  # type: ignore[assignment]

    def __enter__(self) -> 'BrokerStore':
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # --- Run lifecycle -----------------------------------------------------

    def open_run(
            self,
            identity: RunIdentity,
            *,
            script_source: str,
            script_path: str | Path = "",
    ) -> 'RunContext':
        """Open a new run instance.

        Four steps in a single transaction:

        1. Stale-run cleanup: every row marked alive whose
           ``last_heartbeat_ts_ms`` has expired is closed by setting
           ``ended_ts_ms`` and gets a ``stale_run_cleaned`` event.
        2. Live-collision check: if a live row with the same ``run_id``
           still exists after cleanup, raise ``RuntimeError``.
        3. INSERT a new ``runs`` row.
        4. Order adoption: every live order
           (``closed_ts_ms IS NULL``) and its ``order_refs`` rows
           that still belong to a previous (now ended) instance of the
           same logical ``run_id`` get re-pointed to the fresh
           ``run_instance_id`` and audited via an ``order_adopted``
           event. Without this step, ``iter_live_orders`` /
           :func:`~pynecore.core.broker.store_helpers.find_pending_dispatch`
           would not see pending dispatches left behind by a crashed
           instance, and :meth:`DispatchJournal.recover_pending` would
           return an empty result after a real restart.

        :param identity: Logical identity of the run
            (strategy, symbol, ...).
        :param script_source: Full source of the Pine script — fed into
            ``run_tag`` generation.
        :param script_path: Path to the script file (audit metadata
            only; empty string is allowed).
        :raises RuntimeError: If a live run already exists with the
            same ``run_id``.
        :return: A freshly opened :class:`RunContext`.
        """
        run_id = identity.run_id
        run_tag = identity.make_run_tag(script_source)
        now = _now_ms()

        with self._conn:
            # (1) Stale cleanup — close every expired live row
            self._cleanup_stale_runs_inside_tx(now=now)

            # (2) Collision check AFTER cleanup
            row = self._conn.execute(
                "SELECT run_instance_id, last_heartbeat_ts_ms FROM runs "
                "WHERE run_id = ? AND ended_ts_ms IS NULL",
                (run_id,),
            ).fetchone()
            if row is not None:
                raise RuntimeError(
                    f"Active run_id already exists: {run_id!r} "
                    f"(run_instance_id={row['run_instance_id']}, "
                    f"last_heartbeat={row['last_heartbeat_ts_ms']}). "
                    f"Pass `--run-label`, or stop the previous instance."
                )

            # (3) INSERT a new instance
            cur = self._conn.execute(
                "INSERT INTO runs ("
                "  run_id, run_tag, strategy_id, script_path, symbol, timeframe,"
                "  account_id, run_label, plugin_name,"
                "  started_ts_ms, last_heartbeat_ts_ms"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id, run_tag,
                    identity.strategy_id, str(script_path),
                    identity.symbol, identity.timeframe,
                    identity.account_id, identity.label,
                    self._plugin_name,
                    now, now,
                ),
            )
            run_instance_id = cur.lastrowid
            if run_instance_id is None:
                # Theoretical case: AUTOINCREMENT always returns a
                # lastrowid; handled only to keep the static analyzer
                # happy.
                raise RuntimeError("sqlite3 lastrowid is None after INSERT")

            # (4) Adopt orphan live orders + refs left behind by previous
            #     instances of the same run_id (crash recovery).
            self._adopt_orphan_rows_inside_tx(
                now=now,
                new_run_instance_id=run_instance_id,
                run_id=run_id,
            )

        return RunContext(
            run_id=run_id,
            run_instance_id=run_instance_id,
            run_tag=run_tag,
            _store=self,
        )

    def _adopt_orphan_rows_inside_tx(
            self, *, now: int, new_run_instance_id: int, run_id: str,
    ) -> int:
        """Re-point live orders + refs from previous instances onto the new one.

        Caller owns the surrounding ``with self._conn`` block. Idempotent
        for an empty input (no orphan rows → no-op). Every adopted COID
        gets a per-row ``order_adopted`` audit event tied to the new
        ``run_instance_id`` (the new owner), with a payload carrying the
        ``prior_run_instance_id`` and ``prior_state`` for forensics.

        Adoption only touches rows whose ``closed_ts_ms IS NULL`` — a
        properly finalised order stays linked to the instance that
        finalised it. ``order_refs`` rows for adopted COIDs follow the
        same migration; refs for already-closed orders are left alone
        (they will be cleaned up by the standard
        :meth:`RunContext.close_order` path).

        :return: Count of adopted COIDs (useful for tests / diagnostics).
        """
        all_orphan_rows = self._conn.execute(
            "SELECT o.run_instance_id AS prior_run_instance_id, "
            "       o.client_order_id, o.exchange_order_id, "
            "       o.intent_key, o.state "
            "FROM orders o "
            "JOIN runs r ON o.run_instance_id = r.run_instance_id "
            "WHERE r.run_id = ? "
            "  AND o.run_instance_id != ? "
            "  AND o.closed_ts_ms IS NULL "
            "ORDER BY o.run_instance_id DESC",
            (run_id, new_run_instance_id),
        ).fetchall()
        if not all_orphan_rows:
            return 0

        # Deduplicate by COID. Repeated crash/restart cycles before
        # adoption existed could leave the same live ``client_order_id``
        # under multiple ended ``run_instance_id``s of this ``run_id``.
        # The PRIMARY KEY ``(run_instance_id, client_order_id)`` forbids
        # collapsing them onto ``new_run_instance_id`` in a single UPDATE,
        # so adopt only the most recent prior instance's row (the highest
        # ``run_instance_id``) and terminalize the older duplicates with
        # ``closed_ts_ms`` so they vanish from recovery's view.
        adopted_rows: list = []
        superseded_rows: list = []
        seen_coids: set[str] = set()
        for row in all_orphan_rows:
            coid = row['client_order_id']
            if coid in seen_coids:
                superseded_rows.append(row)
            else:
                seen_coids.add(coid)
                adopted_rows.append(row)

        adopted_priors = sorted({row['prior_run_instance_id'] for row in adopted_rows})
        adopted_coids = sorted({row['client_order_id'] for row in adopted_rows})
        adopted_prior_placeholders = ','.join('?' * len(adopted_priors))
        coid_placeholders = ','.join('?' * len(adopted_coids))

        # Close superseded duplicates BEFORE migrating the canonical
        # rows, so the UPDATE below cannot accidentally pick them up via
        # the ``run_instance_id IN (...)`` predicate. Also drop their
        # ``order_refs`` rows — leaving them attached to the closed prior
        # instance would still collide with the canonical refs once they
        # land on ``new_run_instance_id`` (PK includes ``ref_value``).
        for row in superseded_rows:
            self._conn.execute(
                "UPDATE orders SET closed_ts_ms = ?, updated_ts_ms = ? "
                "WHERE run_instance_id = ? AND client_order_id = ? "
                "  AND closed_ts_ms IS NULL",
                (now, now,
                 row['prior_run_instance_id'], row['client_order_id']),
            )
            self._conn.execute(
                "DELETE FROM order_refs "
                "WHERE run_instance_id = ? AND client_order_id = ?",
                (row['prior_run_instance_id'], row['client_order_id']),
            )
            self._conn.execute(
                "INSERT INTO events ("
                "  run_instance_id, ts_ms, plugin_name, kind,"
                "  client_order_id, exchange_order_id, intent_key, payload"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    new_run_instance_id, now, self._plugin_name,
                    'order_adopt_superseded',
                    row['client_order_id'],
                    row['exchange_order_id'],
                    row['intent_key'],
                    json.dumps({
                        'prior_run_instance_id': row['prior_run_instance_id'],
                        'prior_state': row['state'],
                    }),
                ),
            )
            # Forensic only — the operator has nothing to do with this.
            # Full audit row lands in the ``events`` table as
            # ``order_adopt_superseded``.
            _log.debug(
                "broker storage: superseded orphan order coid=%r from "
                "run_instance_id=%d (state=%r) — newer prior instance "
                "exists for the same run_id; closed to resolve ambiguity",
                row['client_order_id'], row['prior_run_instance_id'],
                row['state'],
            )

        # Migrate the canonical orders rows.
        self._conn.execute(
            f"UPDATE orders SET run_instance_id = ?, updated_ts_ms = ? "
            f"WHERE run_instance_id IN ({adopted_prior_placeholders}) "
            f"  AND client_order_id IN ({coid_placeholders}) "
            f"  AND closed_ts_ms IS NULL",
            (new_run_instance_id, now, *adopted_priors, *adopted_coids),
        )

        # Migrate the order_refs rows for the same COIDs. order_refs has
        # PRIMARY KEY (run_instance_id, ref_type, ref_value) — colliding
        # refs from superseded duplicates would also fail the UPDATE, so
        # restrict the migration to refs belonging to the canonical prior
        # instances only.
        self._conn.execute(
            f"UPDATE order_refs SET run_instance_id = ? "
            f"WHERE run_instance_id IN ({adopted_prior_placeholders}) "
            f"  AND client_order_id IN ({coid_placeholders})",
            (new_run_instance_id, *adopted_priors, *adopted_coids),
        )

        # Per-COID audit event under the NEW run_instance_id.
        for row in adopted_rows:
            self._conn.execute(
                "INSERT INTO events ("
                "  run_instance_id, ts_ms, plugin_name, kind,"
                "  client_order_id, exchange_order_id, intent_key, payload"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    new_run_instance_id, now, self._plugin_name,
                    'order_adopted',
                    row['client_order_id'],
                    row['exchange_order_id'],
                    row['intent_key'],
                    json.dumps({
                        'prior_run_instance_id': row['prior_run_instance_id'],
                        'prior_state': row['state'],
                    }),
                ),
            )
            _log.debug(
                "broker storage: adopted order coid=%r from "
                "run_instance_id=%d to %d (state=%r)",
                row['client_order_id'], row['prior_run_instance_id'],
                new_run_instance_id, row['state'],
            )

        # Summary INFO so the operator sees a single, actionable line on a
        # crash-recovery restart, while the per-row noise stays at DEBUG.
        if adopted_coids:
            _log.info(
                "broker storage: adopted %d order(s) from %d prior "
                "instance(s) of run_id; per-row details at DEBUG",
                len(adopted_coids), len(adopted_priors),
            )

        return len(adopted_coids)

    def cleanup_stale_runs(
            self, *, stale_threshold_ms: int = STALE_THRESHOLD_MS,
    ) -> int:
        """Public stale-cleanup, callable manually (e.g. from debug CLI).

        :param stale_threshold_ms: A heartbeat older than this counts as
            stale.
        :return: Number of rows closed.
        """
        now = _now_ms()
        with self._conn:
            return self._cleanup_stale_runs_inside_tx(
                now=now, stale_threshold_ms=stale_threshold_ms,
            )

    def _cleanup_stale_runs_inside_tx(
            self, *, now: int, stale_threshold_ms: int = STALE_THRESHOLD_MS,
    ) -> int:
        """Stale-cleanup inside a transaction. Caller owns the ``with self._conn``.

        Split from the public ``cleanup_stale_runs`` because
        ``open_run`` calls this inside an already-open transaction — a
        nested ``with self._conn`` would open a block savepoint, adding
        complexity for no benefit here.
        """
        threshold = now - stale_threshold_ms
        rows = self._conn.execute(
            "SELECT run_instance_id, last_heartbeat_ts_ms, run_id "
            "FROM runs WHERE ended_ts_ms IS NULL AND last_heartbeat_ts_ms < ?",
            (threshold,),
        ).fetchall()
        for row in rows:
            rid = row['run_instance_id']
            last_hb = row['last_heartbeat_ts_ms']
            self._conn.execute(
                "UPDATE runs SET ended_ts_ms = ? WHERE run_instance_id = ?",
                (last_hb, rid),
            )
            self._conn.execute(
                "INSERT INTO events ("
                "  run_instance_id, ts_ms, plugin_name, kind, payload"
                ") VALUES (?, ?, ?, ?, ?)",
                (
                    rid, now, self._plugin_name, 'stale_run_cleaned',
                    json.dumps({
                        'run_id': row['run_id'],
                        'last_heartbeat_ts_ms': last_hb,
                        'cleaned_at_ts_ms': now,
                    }),
                ),
            )
            _log.warning(
                "broker storage: stale run cleaned run_instance_id=%d run_id=%r "
                "last_heartbeat=%d", rid, row['run_id'], last_hb,
            )
        return len(rows)

    def cleanup_old_events(self, retention_days: int = 180) -> int:
        """Purge stale event rows.

        Not implemented yet — relevant once the DB grows past its size
        threshold (~1 GB). The stub reserves the API slot so caller
        code can already reference it from a ``not-implemented`` branch.

        :raises NotImplementedError: Always; implementation lands in v2.
        """
        raise NotImplementedError(
            "event retention cleanup — scheduled for v2"
        )


# === RunContext ============================================================

# noinspection PyProtectedMember
@dataclass
class RunContext:
    """Context object for one concrete running run.

    Every actual data movement goes through it. The ``run_instance_id``
    (physical FK) is stored here but not exposed on the caller surface —
    every method already filters on this run.

    ``close()`` is the happy-path teardown (``SIGINT`` / ``SIGTERM`` /
    context manager). The crash path is handled by the stale-cleanup
    that runs at the start of ``BrokerStore.open_run``.
    """
    run_id: str
    run_instance_id: int
    run_tag: str
    _store: BrokerStore
    _last_heartbeat_write_ms: int = 0

    # --- Core sync engine: envelope-identity ------------------------------

    def record_envelope(
            self, key: str, bar_ts_ms: int, retry_seq: int,
    ) -> None:
        """Persist the first envelope for an ``intent_key``.

        UPSERT on the ``(run_id, key)`` pair: ``run_id`` is the logical
        key, so every new instance inherits the previous envelopes of
        the same bot. Because of the sync engine's pinning semantics, a
        conflict only occurs on a retry_seq bump.
        """
        now = _now_ms()
        with self._store._conn:
            self._store._conn.execute(
                "INSERT INTO envelopes ("
                "  run_id, intent_key, bar_ts_ms, retry_seq, updated_ts_ms"
                ") VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(run_id, intent_key) DO UPDATE SET "
                "  bar_ts_ms = excluded.bar_ts_ms, "
                "  retry_seq = excluded.retry_seq, "
                "  updated_ts_ms = excluded.updated_ts_ms",
                (self.run_id, key, bar_ts_ms, retry_seq, now),
            )

    # noinspection SqlResolve
    def record_park(
            self, coid: str, key: str, *, kind: str = 'new',
            order_ids: list[str] | None = None,
    ) -> None:
        """Persist a parked dispatch (unknown-disposition response).

        On a re-park for the same ``(run_id, client_order_id)`` the
        ``resolution`` column is also reset to ``NULL``: the row becomes
        parked again after a modify/retry timeout, so the *previous*
        attach/reject decision is now stale. Leaving an old
        ``'attached'`` value in place would make the next restart's
        :meth:`OrderSyncEngine._consume_plugin_resolutions` immediately
        adopt the freshly parked dispatch (skipping the broker call) —
        exactly the wrong outcome, since the new park exists precisely
        because the exchange-side state is unknown.

        :param coid: The dispatch's ``client_order_id`` — the broker-side
            idempotency key the park row is anchored on.
        :param key: The ``intent_key`` this parked dispatch belongs to.
        :param kind: ``'new'`` (default) when the parked dispatch was an
            ``execute_*`` call (new order), ``'modify'`` when it was a
            ``modify_entry`` / ``modify_exit``. The value is overwritten
            on re-park — a modify-park can be replaced by a later
            new-park and vice versa. The engine uses this when
            processing a ``'rejected'`` resolution to decide whether to
            clear the ``_active_intents`` / ``_order_mapping`` slot
            (kind='new') or to keep the original mapping and only drop
            the envelope (kind='modify' — the original exchange order is
            still live).
        :param order_ids: The ``_order_mapping[key]`` snapshot captured at
            park time (the exchange order IDs), persisted as a JSON array
            so a post-restart modify-rejected resolution can recover them
            and avoid a duplicate ``execute_*`` dispatch. Defaults to an
            empty list.
        """
        if kind not in ('new', 'modify'):
            raise ValueError(
                f"record_park: unknown kind {kind!r}; "
                f"expected 'new' or 'modify'"
            )
        now = _now_ms()
        ids_json = json.dumps(order_ids) if order_ids else '[]'
        with self._store._conn:
            self._store._conn.execute(
                "INSERT INTO pending_verifications ("
                "  run_id, client_order_id, intent_key, parked_ts_ms, "
                "  dispatch_kind, order_ids"
                ") VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(run_id, client_order_id) DO UPDATE SET "
                "  intent_key = excluded.intent_key, "
                "  parked_ts_ms = excluded.parked_ts_ms, "
                "  resolution = NULL, "
                "  dispatch_kind = excluded.dispatch_kind, "
                "  order_ids = excluded.order_ids",
                (self.run_id, coid, key, now, kind, ids_json),
            )

    def record_unpark(self, coid: str) -> None:
        """Remove a parked dispatch (it has shown up at the broker)."""
        with self._store._conn:
            self._store._conn.execute(
                "DELETE FROM pending_verifications "
                "WHERE run_id = ? AND client_order_id = ?",
                (self.run_id, coid),
            )

    # noinspection SqlResolve
    def record_resolution(self, coid: str, resolution: str) -> None:
        """Record a plugin-resolved disposition for a parked COID.

        Used by plugins that can determine the parked dispatch's outcome
        through a path other than ``get_open_orders`` (e.g. a position
        snapshot). The engine consumes and deletes the row on the next
        sync.

        ``'rejected'`` is *sticky*: once a row is ``'rejected'`` a later
        ``'attached'`` write does not overwrite it. The motivation is
        the Capital.com bracket scenario: TP and SL legs each call
        ``record_resolution`` for the same parent COID (the bracket has
        a single park entry). If the TP is missing (``'rejected'``) and
        the SL is attached (``'attached'``), an order-dependent naive
        UPDATE could store ``'attached'`` last, the engine would keep
        the ExitIntent and never re-dispatch the TP, leaving protection
        permanently incomplete. ``'rejected'`` means "at least one leg
        is definitely missing" and always wins, because re-dispatch is
        idempotent (the already-attached leg is re-emitted with the
        same parameters and is a no-op at the broker).

        :param coid: The parent ``client_order_id`` whose parked dispatch
            disposition is being recorded.
        :param resolution: ``'attached'`` if the dispatch landed at the
            broker (engine keeps the ``_active_intents`` entry),
            ``'rejected'`` if it definitely did not (engine drops the
            intent so the next sync re-dispatches). Any other value
            raises ``ValueError``.
        """
        if resolution not in ('attached', 'rejected'):
            raise ValueError(
                f"record_resolution: unknown resolution {resolution!r}; "
                f"expected 'attached' or 'rejected'"
            )
        with self._store._conn:
            if resolution == 'attached':
                self._store._conn.execute(
                    "UPDATE pending_verifications "
                    "SET resolution = ? "
                    "WHERE run_id = ? AND client_order_id = ? "
                    "  AND (resolution IS NULL OR resolution != 'rejected')",
                    (resolution, self.run_id, coid),
                )
            else:
                self._store._conn.execute(
                    "UPDATE pending_verifications "
                    "SET resolution = ? "
                    "WHERE run_id = ? AND client_order_id = ?",
                    (resolution, self.run_id, coid),
                )

    def record_complete(self, key: str) -> None:
        """Fully close an ``intent_key`` (cancel / close / rejected).

        Atomically deletes the envelope and every parked dispatch
        attached to it within the ``run_id`` logical scope.
        """
        with self._store._conn:
            self._store._conn.execute(
                "DELETE FROM envelopes "
                "WHERE run_id = ? AND intent_key = ?",
                (self.run_id, key),
            )
            self._store._conn.execute(
                "DELETE FROM pending_verifications "
                "WHERE run_id = ? AND intent_key = ?",
                (self.run_id, key),
            )

    # noinspection SqlResolve
    def replay(
            self,
    ) -> tuple[dict[str, EnvelopeRecord], dict[str, PendingRecord]]:
        """Reconstruct the in-memory state after a restart.

        Replays on the ``run_id`` logical key — a new instance inherits
        the previous envelopes and parked dispatches of the same logical
        bot.

        :return: ``(envelopes_by_key, pending_by_coid)`` — same shape as
            the former ``state_store.replay`` returned.
        """
        envelopes: dict[str, EnvelopeRecord] = {}
        pending: dict[str, PendingRecord] = {}

        for row in self._store._conn.execute(
                "SELECT intent_key, bar_ts_ms, retry_seq FROM envelopes "
                "WHERE run_id = ?",
                (self.run_id,),
        ):
            envelopes[row['intent_key']] = EnvelopeRecord(
                key=row['intent_key'],
                bar_ts_ms=int(row['bar_ts_ms']),
                retry_seq=int(row['retry_seq']),
            )

        for row in self._store._conn.execute(
                "SELECT client_order_id, intent_key, resolution, "
                "       dispatch_kind, order_ids "
                "FROM pending_verifications "
                "WHERE run_id = ?",
                (self.run_id,),
        ):
            raw_ids = row['order_ids'] or '[]'
            pending[row['client_order_id']] = PendingRecord(
                key=row['intent_key'],
                coid=row['client_order_id'],
                resolution=row['resolution'],
                dispatch_kind=row['dispatch_kind'] or 'new',
                order_ids=json.loads(raw_ids),
            )

        return envelopes, pending

    # noinspection SqlResolve
    def iter_pending_resolutions(self) -> list[PendingRecord]:
        """Fetch parked rows that the plugin has already resolved.

        Called by the engine's ``_verify_pending_dispatches`` at the
        start of every sync to learn which COIDs the plugin wrote a
        ``record_resolution`` entry for. Returned records always have a
        non-``None`` ``resolution`` — still-parked (unresolved) rows are
        skipped.
        """
        rows = self._store._conn.execute(
            "SELECT client_order_id, intent_key, resolution, "
            "       dispatch_kind, order_ids "
            "FROM pending_verifications "
            "WHERE run_id = ? AND resolution IS NOT NULL",
            (self.run_id,),
        ).fetchall()
        return [
            PendingRecord(
                key=row['intent_key'],
                coid=row['client_order_id'],
                resolution=row['resolution'],
                dispatch_kind=row['dispatch_kind'] or 'new',
                order_ids=json.loads(row['order_ids'] or '[]'),
            )
            for row in rows
        ]

    # --- Orders ------------------------------------------------------------

    def upsert_order(
            self, client_order_id: str, **fields: Any,
    ) -> None:
        """UPSERT an order row — insert a new one or update an existing one.

        Accepted fields: ``symbol``, ``side``, ``qty``, ``state``,
        ``intent_key``, ``exchange_order_id``, ``from_entry``,
        ``pine_entry_id``, ``sl_level``, ``tp_level``, ``trailing_stop``,
        ``trailing_distance``, ``filled_qty``, ``extras``. Missing
        fields are filled with the DB defaults on insert; on update
        only the explicitly passed fields are written.

        ``extras`` is supplied as a dict and serialised to a JSON string.

        :raises ValueError: When inserting a new row with required
            fields missing (``symbol``, ``side``, ``qty``, ``state``).
        """
        now = _now_ms()
        extras = fields.pop('extras', None)
        extras_json = json.dumps(extras) if extras is not None else None

        with self._store._conn:
            existing = self._store._conn.execute(
                "SELECT 1 FROM orders "
                "WHERE run_instance_id = ? AND client_order_id = ?",
                (self.run_instance_id, client_order_id),
            ).fetchone()

            if existing is None:
                required = ('symbol', 'side', 'qty', 'state')
                missing = [r for r in required if r not in fields]
                if missing:
                    raise ValueError(
                        f"upsert_order({client_order_id!r}) new row, "
                        f"missing required fields: {missing}"
                    )
                self._store._conn.execute(
                    "INSERT INTO orders ("
                    "  run_instance_id, client_order_id, plugin_name,"
                    "  intent_key, exchange_order_id, symbol, side, qty,"
                    "  filled_qty, state, from_entry, pine_entry_id,"
                    "  sl_level, tp_level, trailing_stop, trailing_distance,"
                    "  created_ts_ms, updated_ts_ms, extras"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        self.run_instance_id, client_order_id, self._store._plugin_name,
                        fields.get('intent_key'), fields.get('exchange_order_id'),
                        fields['symbol'], fields['side'], fields['qty'],
                        fields.get('filled_qty', 0.0), fields['state'],
                        fields.get('from_entry'), fields.get('pine_entry_id'),
                        fields.get('sl_level'), fields.get('tp_level'),
                        int(bool(fields.get('trailing_stop', False))),
                        fields.get('trailing_distance'),
                        now, now, extras_json,
                    ),
                )
                return

            # Update path: only the explicitly passed fields are written.
            sets: list[str] = []
            params: list[Any] = []
            for col in (
                    'intent_key', 'exchange_order_id', 'symbol', 'side', 'qty',
                    'filled_qty', 'state', 'from_entry', 'pine_entry_id',
                    'sl_level', 'tp_level', 'trailing_distance',
            ):
                if col in fields:
                    sets.append(f"{col} = ?")
                    params.append(fields[col])
            if 'trailing_stop' in fields:
                sets.append("trailing_stop = ?")
                params.append(int(bool(fields['trailing_stop'])))
            if extras_json is not None:
                sets.append("extras = ?")
                params.append(extras_json)
            sets.append("updated_ts_ms = ?")
            params.append(now)
            params.extend([self.run_instance_id, client_order_id])

            self._store._conn.execute(
                f"UPDATE orders SET {', '.join(sets)} "
                f"WHERE run_instance_id = ? AND client_order_id = ?",
                params,
            )

    def set_order_state(self, client_order_id: str, state: str) -> None:
        """Single-field update: ``orders.state``."""
        self.upsert_order(client_order_id, state=state)

    def set_exchange_id(
            self, client_order_id: str, exchange_order_id: str,
    ) -> None:
        """Populate ``orders.exchange_order_id`` (Capital.com confirm, IB orderId, ...)."""
        self.upsert_order(client_order_id, exchange_order_id=exchange_order_id)

    def set_risk(
            self, client_order_id: str, *,
            sl: float | None = None,
            tp: float | None = None,
            trailing_stop: bool | None = None,
            trailing_distance: float | None = None,
    ) -> None:
        """Update SL/TP/trailing attributes in one go.

        A ``None`` parameter *does not* erase the existing value — it
        just indicates that the caller is not setting it now. To clear
        a value, pass an explicit ``sl=0.0`` or use a dedicated UPDATE
        (no delete method exists yet — added if a real need arises).
        """
        fields: dict[str, Any] = {}
        if sl is not None:
            fields['sl_level'] = sl
        if tp is not None:
            fields['tp_level'] = tp
        if trailing_stop is not None:
            fields['trailing_stop'] = trailing_stop
        if trailing_distance is not None:
            fields['trailing_distance'] = trailing_distance
        if fields:
            self.upsert_order(client_order_id, **fields)

    def set_filled(self, client_order_id: str, filled_qty: float) -> None:
        """Update ``orders.filled_qty`` (non-incremental — caller passes the full amount)."""
        self.upsert_order(client_order_id, filled_qty=filled_qty)

    def reopen_order(self, client_order_id: str) -> None:
        """Re-activate a previously closed order: ``closed_ts_ms = NULL``.

        Typical use: a bracket leg row was closed by :meth:`close_order`
        on an earlier REJECTED attach (``state='rejected'``,
        ``closed_ts_ms`` set), then a later ``modify_exit`` /
        ``execute_exit`` re-attached a fresh protective leg at the
        broker under the same ``client_order_id`` — the row has to
        return to the live range (``iter_live_orders``, recovery,
        fill-fallback) or the post-persistence logic will not find it.

        Reopen only nulls ``closed_ts_ms``; the caller is responsible
        for harmonising ``state`` and other fields via
        :meth:`upsert_order`. For audit purposes the reopen itself
        writes an event (``order_reopened``), so the full lifecycle
        remains traceable from the ``events`` table.

        :raises: no specific error signalling. If the row does not
            exist or is no longer closed the SQL UPDATE affects zero
            rows and the method returns silently.
        """
        now = _now_ms()
        with self._store._conn:
            existing = self._store._conn.execute(
                "SELECT closed_ts_ms FROM orders "
                "WHERE run_instance_id = ? AND client_order_id = ?",
                (self.run_instance_id, client_order_id),
            ).fetchone()
            if existing is None or existing['closed_ts_ms'] is None:
                return
            self._store._conn.execute(
                "UPDATE orders SET closed_ts_ms = NULL, updated_ts_ms = ? "
                "WHERE run_instance_id = ? AND client_order_id = ?",
                (now, self.run_instance_id, client_order_id),
            )
            self._store._conn.execute(
                "INSERT INTO events ("
                "  run_instance_id, ts_ms, plugin_name, kind,"
                "  client_order_id, exchange_order_id, intent_key, payload"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.run_instance_id, now, self._store._plugin_name,
                    'order_reopened', client_order_id, None, None, None,
                ),
            )

    def close_order(self, client_order_id: str) -> None:
        """Close an order: set ``closed_ts_ms``, delete the related
        ``order_refs`` rows and write an ``order_closed`` audit event in
        a single transaction.

        Eagerly trimming ``order_refs`` keeps the table's size in line
        with the number of live orders. Historical dealReference /
        dealId lookups are served by the ``events`` table — that is why
        we also write an event here, carrying the ``exchange_order_id``
        valid at close time.
        """
        now = _now_ms()
        with self._store._conn:
            exchange_order_id: str | None = None
            row = self._store._conn.execute(
                "SELECT exchange_order_id FROM orders "
                "WHERE run_instance_id = ? AND client_order_id = ?",
                (self.run_instance_id, client_order_id),
            ).fetchone()
            if row is not None:
                exchange_order_id = row['exchange_order_id']
            self._store._conn.execute(
                "UPDATE orders SET closed_ts_ms = ?, updated_ts_ms = ? "
                "WHERE run_instance_id = ? AND client_order_id = ?",
                (now, now, self.run_instance_id, client_order_id),
            )
            self._store._conn.execute(
                "DELETE FROM order_refs "
                "WHERE run_instance_id = ? AND client_order_id = ?",
                (self.run_instance_id, client_order_id),
            )
            self._store._conn.execute(
                "INSERT INTO events ("
                "  run_instance_id, ts_ms, plugin_name, kind,"
                "  client_order_id, exchange_order_id, intent_key, payload"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.run_instance_id, now, self._store._plugin_name,
                    'order_closed', client_order_id, exchange_order_id,
                    None, None,
                ),
            )

    # --- Order refs (generic alias lookup) --------------------------------

    def add_ref(
            self, client_order_id: str, ref_type: str, ref_value: str,
    ) -> None:
        """Record a broker-specific alias key.

        E.g. Capital.com's ``deal_reference`` from the POST response,
        followed by ``deal_id`` from the confirm. For IB:
        ``perm_id`` / ``order_id``. The
        ``(run_instance_id, ref_type, ref_value)`` triplet is unique.
        """
        now = _now_ms()
        with self._store._conn:
            self._store._conn.execute(
                "INSERT INTO order_refs ("
                "  run_instance_id, ref_type, ref_value, client_order_id, created_ts_ms"
                ") VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(run_instance_id, ref_type, ref_value) DO UPDATE SET "
                "  client_order_id = excluded.client_order_id, "
                "  created_ts_ms = excluded.created_ts_ms",
                (self.run_instance_id, ref_type, ref_value, client_order_id, now),
            )

    def iter_refs_for_coid(
            self, client_order_id: str,
    ) -> Iterator[tuple[str, str]]:
        """Yield ``(ref_type, ref_value)`` pairs for one COID.

        Used by recovery to materialise all alias keys that were
        durably recorded before a crash. The narrow but real crash
        window is between ``add_ref(deal_reference, ...)`` (commits
        the alias) and the subsequent ``upsert_order(extras={...})``
        that mirrors it into ``orders.extras``: in that gap the
        ``deal_reference`` is only present in ``order_refs``, and the
        resume hook needs it to confirm the already-submitted order
        against the exchange.

        Filtered by the current ``run_instance_id`` — adoption (see
        :meth:`BrokerStore.open_run`) already migrates orphan refs
        into the live instance, so this matches the row's owner.
        """
        rows = self._store._conn.execute(
            "SELECT ref_type, ref_value FROM order_refs "
            "WHERE run_instance_id = ? AND client_order_id = ?",
            (self.run_instance_id, client_order_id),
        )
        for row in rows:
            yield row['ref_type'], row['ref_value']

    def find_by_ref(
            self, ref_type: str, ref_value: str,
    ) -> OrderRow | None:
        """Alias-based order lookup in O(log n).

        Joins ``order_refs`` × ``orders`` on the PK. A single indexed
        SELECT that reduces this use case to one DB call.
        """
        row = self._store._conn.execute(
            "SELECT o.* FROM orders o "
            "JOIN order_refs r ON "
            "  r.run_instance_id = o.run_instance_id "
            "  AND r.client_order_id = o.client_order_id "
            "WHERE r.run_instance_id = ? AND r.ref_type = ? AND r.ref_value = ?",
            (self.run_instance_id, ref_type, ref_value),
        ).fetchone()
        return _row_to_order(row) if row is not None else None

    # --- Queries ----------------------------------------------------------

    def get_order(self, client_order_id: str) -> OrderRow | None:
        """Direct lookup by CO-ID."""
        row = self._store._conn.execute(
            "SELECT * FROM orders "
            "WHERE run_instance_id = ? AND client_order_id = ?",
            (self.run_instance_id, client_order_id),
        ).fetchone()
        return _row_to_order(row) if row is not None else None

    def iter_live_orders(
            self, *,
            symbol: str | None = None,
            from_entry: str | None = None,
    ) -> Iterator[OrderRow]:
        """Iterator over live (not yet closed) orders.

        The partial index (``idx_orders_live``) serves the filter; a
        realistic one-way Pine strategy has fewer than 50 live rows at
        any time, so the query cost is negligible.
        """
        sql = (
            "SELECT * FROM orders "
            "WHERE run_instance_id = ? AND closed_ts_ms IS NULL"
        )
        params: list[Any] = [self.run_instance_id]
        if symbol is not None:
            sql += " AND symbol = ?"
            params.append(symbol)
        if from_entry is not None:
            sql += " AND from_entry = ?"
            params.append(from_entry)
        for row in self._store._conn.execute(sql, params):
            yield _row_to_order(row)

    # --- Events -----------------------------------------------------------

    def log_event(
            self, kind: str, *,
            client_order_id: str | None = None,
            exchange_order_id: str | None = None,
            intent_key: str | None = None,
            payload: dict | None = None,
    ) -> None:
        """Write an audit event.

        ``payload`` is serialised to JSON; plugin-specific fields can
        be added freely.
        """
        now = _now_ms()
        payload_json = json.dumps(payload) if payload is not None else None
        with self._store._conn:
            self._store._conn.execute(
                "INSERT INTO events ("
                "  run_instance_id, ts_ms, plugin_name, kind,"
                "  client_order_id, exchange_order_id, intent_key, payload"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.run_instance_id, now, self._store._plugin_name, kind,
                    client_order_id, exchange_order_id, intent_key, payload_json,
                ),
            )

    def find_event_by_intent_key(
            self, intent_key: str, kind: str,
    ) -> bool:
        """Return ``True`` iff at least one event with the given
        ``intent_key`` and ``kind`` exists for this *logical* run.

        Scoped to ``run_id``, not ``run_instance_id`` — the engine's
        startup replay uses this to detect whether a defensive-close
        FILL event was recorded in a *previous* process instance (whose
        adopted orders carry over into the current run). A query
        scoped to the current ``run_instance_id`` would always miss
        cross-restart settlements and re-arm markers that are already
        done.

        The JOIN cost is amortised over the rare set of startup-replay
        invocations; the ``runs.run_id`` lookup uses the existing
        ``idx_runs_run_id`` index.
        """
        row = self._store._conn.execute(
            "SELECT 1 FROM events AS e "
            "JOIN runs AS r ON e.run_instance_id = r.run_instance_id "
            "WHERE r.run_id = ? AND e.intent_key = ? AND e.kind = ? "
            "LIMIT 1",
            (self.run_id, intent_key, kind),
        ).fetchone()
        return row is not None

    def iter_events_by_kind_since(
            self, kind: str, since_ts_ms: int,
    ) -> Iterator[dict]:
        """Iterate event payloads of a given ``kind`` since ``since_ts_ms``.

        ASC by ``ts_ms``; only payloads that JSON-deserialise
        successfully are yielded (empty or malformed payloads are
        skipped). Used by plugin-side cross-restart recovery (e.g.
        activity-cursor rebuild) so the persisted audit-event tail can
        be read without dropping down to raw SQL.

        :param kind: Filter on ``events.kind``.
        :param since_ts_ms: Lower bound on ``ts_ms`` (inclusive).
        :return: Iterator of payload dicts in insertion order.
        """
        rows = self._store._conn.execute(
            "SELECT payload FROM events "
            "WHERE run_instance_id = ? AND kind = ? AND ts_ms >= ? "
            "ORDER BY ts_ms",
            (self.run_instance_id, kind, since_ts_ms),
        )
        for row in rows:
            raw = row['payload']
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except ValueError:
                continue

    def iter_events_by_kind_for_run_id(
            self, kind: str,
    ) -> Iterator[tuple[str | None, str | None, str | None, dict]]:
        """Iterate events of a given ``kind`` across every run instance
        sharing this logical ``run_id``.

        Scoped to ``runs.run_id`` (not ``run_instance_id``) — the engine's
        startup replay uses this to recover dedup state from
        ``defensive_close_filled`` audit events written by prior process
        instances. The ``find_event_by_intent_key`` helper only answers
        "does any matching event exist?"; this iterator returns the
        identifying columns + payload so callers can reseed in-memory
        caches.

        Yields ``(intent_key, client_order_id, exchange_order_id,
        payload_dict)`` tuples in insertion order. Rows with malformed
        payloads yield an empty dict (the column data is still
        useful).
        """
        rows = self._store._conn.execute(
            "SELECT e.intent_key, e.client_order_id, e.exchange_order_id, "
            "       e.payload "
            "FROM events AS e "
            "JOIN runs AS r ON e.run_instance_id = r.run_instance_id "
            "WHERE r.run_id = ? AND e.kind = ? "
            "ORDER BY e.ts_ms",
            (self.run_id, kind),
        )
        for row in rows:
            raw = row['payload']
            payload: dict = {}
            if raw:
                try:
                    parsed = json.loads(raw)
                except ValueError:
                    parsed = None
                if isinstance(parsed, dict):
                    payload = parsed
            yield (
                row['intent_key'],
                row['client_order_id'],
                row['exchange_order_id'],
                payload,
            )

    # --- Lifecycle --------------------------------------------------------

    def heartbeat(self) -> None:
        """Heartbeat for the current run. Rate-limited to ``HEARTBEAT_INTERVAL_MS``.

        The caller can call this every sync cycle — the internal gate
        ensures at most one UPDATE per minute. Does NOT run cross-run
        cleanup (that is exclusively ``open_run()``'s responsibility —
        clear separation of concerns).
        """
        now = _now_ms()
        if now - self._last_heartbeat_write_ms < HEARTBEAT_INTERVAL_MS:
            return
        with self._store._conn:
            self._store._conn.execute(
                "UPDATE runs SET last_heartbeat_ts_ms = ? "
                "WHERE run_instance_id = ?",
                (now, self.run_instance_id),
            )
        self._last_heartbeat_write_ms = now

    def close(self) -> None:
        """Happy-path run teardown: populate ``ended_ts_ms``.

        Repeated calls are no-ops (after the first UPDATE
        ``ended_ts_ms`` is non-NULL and the WHERE clause excludes the
        row). SIGKILL is handled by stale-cleanup; this method is only
        the controlled-shutdown path.
        """
        now = _now_ms()
        with self._store._conn:
            self._store._conn.execute(
                "UPDATE runs SET ended_ts_ms = ?, last_heartbeat_ts_ms = ? "
                "WHERE run_instance_id = ? AND ended_ts_ms IS NULL",
                (now, now, self.run_instance_id),
            )

    def __enter__(self) -> 'RunContext':
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


# === Private helpers =======================================================

def _row_to_order(row: sqlite3.Row) -> OrderRow:
    """Convert ``sqlite3.Row`` → :class:`OrderRow`, parsing extras JSON."""
    extras_raw = row['extras']
    extras: dict = json.loads(extras_raw) if extras_raw else {}
    return OrderRow(
        client_order_id=row['client_order_id'],
        plugin_name=row['plugin_name'],
        intent_key=row['intent_key'],
        exchange_order_id=row['exchange_order_id'],
        symbol=row['symbol'],
        side=row['side'],
        qty=float(row['qty']),
        filled_qty=float(row['filled_qty'] or 0.0),
        state=row['state'],
        from_entry=row['from_entry'],
        pine_entry_id=row['pine_entry_id'],
        sl_level=None if row['sl_level'] is None else float(row['sl_level']),
        tp_level=None if row['tp_level'] is None else float(row['tp_level']),
        trailing_stop=bool(row['trailing_stop']),
        trailing_distance=(
            None if row['trailing_distance'] is None
            else float(row['trailing_distance'])
        ),
        created_ts_ms=int(row['created_ts_ms']),
        updated_ts_ms=int(row['updated_ts_ms']),
        closed_ts_ms=(
            None if row['closed_ts_ms'] is None else int(row['closed_ts_ms'])
        ),
        extras=extras,
    )
