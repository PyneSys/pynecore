"""
Egyesített SQLite-alapú broker storage.

Ez a modul lép a korábbi kétfelé szórt perzisztálás helyére:

- a core ``state_store.py`` (append-only JSONL, envelope + parked verifications),
- plugin-szintű ledger-ek (Capital.com ``DealLedger`` stb.).

A :class:`BrokerStore` egyetlen SQLite fájlba ír minden broker-relevans
állapotot: a sync-engine envelope-identitását, a parked dispatch-eket, az
order-ok élő nézetét, a generikus alias-táblát (broker-specifikus lookup
kulcsokhoz) és a strukturált audit-log-ot.

A modul fő absztrakciói:

- :class:`RunIdentity` — a run humán-olvasható logikai kulcsa
  (``{strategy_id}@{account}:{symbol}:{timeframe}[#label]``). A storage
  előtt képződik.
- :class:`RunContext` — egy konkrét futtatás context-objektuma. A
  ``run_instance_id`` (fizikai autoincrement FK) itt rejtőzik, a caller
  (sync engine, plugin) soha nem látja.
- :class:`BrokerStore` — a lifecycle: ``open_run()`` ad vissza ``RunContext``-et,
  kezeli a stale-run cleanupot és a séma-migrációt.

Két crash-recovery mechanizmus fut egymás mellett:

1. **Passzív** — a ``live_runs`` VIEW automatikusan kizárja azokat a sorokat,
   ahol a ``last_heartbeat_ts_ms`` túllépte a küszöböt. Dashboard mindig
   helyeset kérdez akkor is, ha fizikai takarítás még nem futott.
2. **Aktív** — minden ``open_run()`` elején a lejárt sorokat ``ended_ts_ms``-szel
   zárjuk és egy ``stale_run_cleaned`` event-et írunk.

Tranzakcionalitás: minden compound művelet (pl. ``close_order`` = update orders
+ delete order_refs + insert events) egyetlen ``BEGIN IMMEDIATE ... COMMIT``
blokkban fut. Se félig írott állapot, se félig elvesztett log.
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

# Heartbeat gyakoriság: ennél sűrűbb ``RunContext.heartbeat()`` hívás no-op.
HEARTBEAT_INTERVAL_MS: Final[int] = 60_000  # 1 perc
# Ennyi heartbeat-hiány után egy run stale-nek számít. A ``live_runs`` VIEW
# SQL-jében HARDKÓDOLVA is szerepel ugyanez az érték — ha itt változik, a
# VIEW-t migrációval cserélni kell (lásd ``_MIGRATIONS``).
STALE_THRESHOLD_MS: Final[int] = 5 * HEARTBEAT_INTERVAL_MS  # 5 perc


def _now_ms() -> int:
    """Aktuális idő ms-ben, UTC. Egy helyen kibontva a time-import-ot."""
    return int(time.time() * 1000)


# === Replay-eredmény dataclass-ek ==========================================

@dataclass(frozen=True)
class EnvelopeRecord:
    """Replay-output egy élő envelope-ra.

    Ugyanolyan shape, mint a korábbi ``state_store.EnvelopeRecord`` — a
    sync engine változatlanul használja.
    """
    key: str
    bar_ts_ms: int
    retry_seq: int


@dataclass(frozen=True)
class PendingRecord:
    """Replay-output egy parked dispatch-re."""
    key: str
    coid: str


@dataclass
class OrderRow:
    """Egy sor az ``orders`` táblából, caller számára kinyerve.

    Az ``extras`` már parse-olt dict (JSON-ból visszaolvasva), nem a nyers
    string — így a plugin-ok broker-specifikus mezőket natívan elérnek.
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


# === Séma-migrációk ========================================================

# A lista **append-only** — régi tuple-öket soha nem módosítunk, mert az már
# éles DB-kben alkalmazott állapotot reprezentál. Új oszlop vagy tábla egy
# új tuple-ként jön.
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

        -- A 300000 ms (5 perc) küszöb HARD-CODED, mert SQLite nem támogat
        -- paraméterezhető VIEW-t. Ha STALE_THRESHOLD_MS változik, új
        -- migrációban DROP VIEW + CREATE VIEW kell.
        CREATE VIEW live_runs AS
            SELECT *
            FROM runs
            WHERE ended_ts_ms IS NULL
              AND last_heartbeat_ts_ms > (
                  CAST(strftime('%s', 'now') AS INTEGER) * 1000 - 300000
              );

        -- Envelopes és pending_verifications a LOGIKAI run_id-re kötöttek,
        -- nem a run_instance_id-re — ezek a broker-oldali idempotency
        -- anchor-jai, amelyeket minden restart (új instance) örököl,
        -- mivel ugyanaz a bot ugyanazokkal az intentkkel indul újra.
        -- Orders/order_refs/events ezzel szemben instance-szintűek, mert
        -- a historikus run-okhoz tartoznak.
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
]


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Migrate the schema from ``PRAGMA user_version`` up to the latest.

    :param conn: A nyitott ``sqlite3.Connection``. A függvény tranzakció-
        blokkokat nyit a connection-ön; a caller legyen tranzakciótól-mentes
        állapotban (``conn.isolation_level`` a default ``""``).
    """
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    for version, description, sql in _MIGRATIONS:
        if version <= current:
            continue
        # A ``with conn`` implicit BEGIN DEFERRED-et nyit, COMMIT-ra zár.
        # Egy verzió = egy tranzakció; félig migrált séma nem maradhat.
        with conn:
            conn.executescript(sql)
            # A ``_migrations`` tábla csak az első migráció futása után
            # létezik — az első alkalommal a CREATE TABLE benne van a
            # script-ben, úgyhogy az INSERT már biztonságos.
            conn.execute(
                "INSERT INTO _migrations (version, applied_ts_ms, description) "
                "VALUES (?, ?, ?)",
                (version, _now_ms(), description),
            )
            conn.execute(f"PRAGMA user_version = {version}")
        _log.info("broker storage migrated to version %d (%s)", version, description)


# === BrokerStore ===========================================================

class BrokerStore:
    """Egyesített SQLite broker-állapot tároló egy workdir-re.

    A konstrukció megnyitja a DB-t, alkalmazza a migrációkat, és beállítja
    a WAL + crash-safe PRAGMA-kat. Az ``open_run()`` minden hívásra egy új
    :class:`RunContext`-et ad vissza — azon keresztül megy minden tényleges
    adatmozgás.

    :param path: A SQLite fájl abszolút útja. A szülő könyvtár automatikusan
        létrehozódik.
    :param plugin_name: A BrokerPlugin ``plugin_name`` attribútuma
        (pl. ``"Capital.com"``). Minden ``events`` / ``orders`` sor ezt
        kapja meg, hogy több-plugin-os workdir-ben filterezhető legyen.
    """

    def __init__(self, path: Path | str, *, plugin_name: str) -> None:
        self._path = Path(path)
        self._plugin_name = plugin_name
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # ``isolation_level=""`` = default; explicit tranzakciókat mi nyitunk
        # ``with conn:`` blokkokkal. A sqlite3-modul autocommit módja nem az,
        # aminek első olvasásra tűnik — a default úgy viselkedik, hogy
        # DML előtt implicit BEGIN van, és a next commit zárja. Ez nekünk
        # megfelel.
        self._conn = sqlite3.connect(
            str(self._path),
            isolation_level="",
            check_same_thread=False,
            timeout=5.0,
        )
        # ``sqlite3.Row`` névvel elérhető oszlopokat ad — a query-helperek
        # ettől nem pozíció-érzékenyek, könnyebb utólag oszlopot hozzáadni.
        self._conn.row_factory = sqlite3.Row
        self._configure_pragmas()
        _apply_migrations(self._conn)

    def _configure_pragmas(self) -> None:
        """WAL + crash-safety + FK + busy-timeout beállítása."""
        # WAL: concurrent read + single writer; crash-safe power loss esetén.
        self._conn.execute("PRAGMA journal_mode=WAL")
        # NORMAL: WAL mellett crash-safe, gyorsabb mint FULL.
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        # 5 s blokkolás lock-collision-kor — egyetlen writer esetén ritka,
        # de debug-CLI párhuzamos futása ezt felélesztheti.
        self._conn.execute("PRAGMA busy_timeout=5000")

    # --- Lifecycle ---------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def plugin_name(self) -> str:
        return self._plugin_name

    def close(self) -> None:
        """Zárja a connection-t. Ismétlődő hívásra no-op."""
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
        """Nyit egy új run-instance-t.

        Három lépés egyetlen tranzakcióban:

        1. Stale-run cleanup: minden élőként jelölt, de ``last_heartbeat_ts_ms``
           túllépett sor ``ended_ts_ms``-szel záródik, és kap egy
           ``stale_run_cleaned`` event-et.
        2. Élő collision check: ha ugyanazon ``run_id``-vel még van élő sor
           a cleanup után, ``RuntimeError``.
        3. INSERT új ``runs`` sor.

        :param identity: A run logikai identitása (strategy, symbol, ...).
        :param script_source: A Pine-script teljes forráskódja — a
            ``run_tag`` képzésébe megy bemenetként.
        :param script_path: A script fájl elérési útja (csak metaadat az
            audit-hoz; üres string megengedett).
        :raises RuntimeError: Ha azonos ``run_id``-val létezik élő run.
        :return: Egy frissen nyitott :class:`RunContext`.
        """
        run_id = identity.run_id
        run_tag = identity.make_run_tag(script_source)
        now = _now_ms()

        with self._conn:
            # (1) stale cleanup — minden lejárt élő sort zárunk
            self._cleanup_stale_runs_inside_tx(now=now)

            # (2) collision check a cleanup UTÁN
            row = self._conn.execute(
                "SELECT run_instance_id, last_heartbeat_ts_ms FROM runs "
                "WHERE run_id = ? AND ended_ts_ms IS NULL",
                (run_id,),
            ).fetchone()
            if row is not None:
                raise RuntimeError(
                    f"Aktív run_id már létezik: {run_id!r} "
                    f"(run_instance_id={row['run_instance_id']}, "
                    f"last_heartbeat={row['last_heartbeat_ts_ms']}). "
                    f"Adj meg `--run-label`-t, vagy állítsd le az előző instance-t."
                )

            # (3) INSERT új instance
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
                # Elvi eset: AUTOINCREMENT mindig ad lastrowid-ot; csak a
                # static analyzer békéje miatt kezeljük.
                raise RuntimeError("sqlite3 lastrowid None az INSERT után")

        return RunContext(
            run_id=run_id,
            run_instance_id=run_instance_id,
            run_tag=run_tag,
            _store=self,
        )

    def cleanup_stale_runs(
            self, *, stale_threshold_ms: int = STALE_THRESHOLD_MS,
    ) -> int:
        """Publikus stale-cleanup, manuálisan is hívható (pl. debug-CLI).

        :param stale_threshold_ms: Ennél régebbi heartbeat = stale.
        :return: A lezárt sorok száma.
        """
        now = _now_ms()
        with self._conn:
            return self._cleanup_stale_runs_inside_tx(
                now=now, stale_threshold_ms=stale_threshold_ms,
            )

    def _cleanup_stale_runs_inside_tx(
            self, *, now: int, stale_threshold_ms: int = STALE_THRESHOLD_MS,
    ) -> int:
        """Tranzakción belüli stale-cleanup. Hívó felel a ``with self._conn``-ért.

        Azért szétvágva a publikus ``cleanup_stale_runs``-től, mert az
        ``open_run`` már nyitott tranzakción belül hívja — egy beágyazott
        ``with self._conn`` block-savepoint-ot nyitna, ami itt felesleges
        összetettséget ad.
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
        """Elavult event-sorok takarítása.

        Még nincs implementálva — a DB méretkritikus küszöbe (~1 GB) felett
        lesz aktuális. A stub megőrzi a helyet az API-ban, hogy a hívó
        kódban már most lehessen rá hivatkozni ``not-implemented`` branch-en.

        :raises NotImplementedError: Mindig; implementáció a v2-ben.
        """
        raise NotImplementedError(
            "event retention cleanup — scheduled for v2"
        )


# === RunContext ============================================================

@dataclass
class RunContext:
    """Egy konkrét futó run context-objektuma.

    Az összes tényleges adatmozgás ezen keresztül megy. A
    ``run_instance_id`` (fizikai FK) itt tároljuk, de a caller-felületen
    nem jelenik meg — minden metódus eleve erre a run-ra szűr.

    A ``close()`` happy-path-lezárás (``SIGINT`` / ``SIGTERM`` /
    context-manager). Crash-path-ot a storage ``open_run`` elején futó
    stale-cleanup kezeli.
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
        """Az első envelope perzisztálása egy ``intent_key``-re.

        UPSERT a ``(run_id, key)`` páron: a ``run_id`` logikai kulcs, így
        minden új instance örökli ugyanezen bot korábbi envelope-jait.
        A sync engine pinning szemantikája miatt a konfliktus csak
        retry_seq-bump esetén fordul elő.
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

    def record_park(self, coid: str, key: str) -> None:
        """Parked dispatch perzisztálása (unknown-disposition válaszra)."""
        now = _now_ms()
        with self._store._conn:
            self._store._conn.execute(
                "INSERT INTO pending_verifications ("
                "  run_id, client_order_id, intent_key, parked_ts_ms"
                ") VALUES (?, ?, ?, ?) "
                "ON CONFLICT(run_id, client_order_id) DO UPDATE SET "
                "  intent_key = excluded.intent_key, "
                "  parked_ts_ms = excluded.parked_ts_ms",
                (self.run_id, coid, key, now),
            )

    def record_unpark(self, coid: str) -> None:
        """Parked dispatch eltávolítása (bróker-oldalon megjelent)."""
        with self._store._conn:
            self._store._conn.execute(
                "DELETE FROM pending_verifications "
                "WHERE run_id = ? AND client_order_id = ?",
                (self.run_id, coid),
            )

    def record_complete(self, key: str) -> None:
        """Egy ``intent_key`` teljes lezárása (cancel / close / rejected).

        Atomikusan törli az envelope-ot és minden hozzá tartozó parked
        dispatch-et a ``run_id`` logikai scope-ján belül.
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

    def replay(
            self,
    ) -> tuple[dict[str, EnvelopeRecord], dict[str, PendingRecord]]:
        """In-memory állapot rekonstrukció restart után.

        A ``run_id`` logikai kulcson replay-el — egy új instance örökli
        ugyanezen logikai bot korábbi envelope-jait és parked dispatch-jeit.

        :return: ``(envelopes_by_key, pending_by_coid)`` — ugyanolyan shape,
            mint amit a régi ``state_store.replay`` adott.
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
                "SELECT client_order_id, intent_key FROM pending_verifications "
                "WHERE run_id = ?",
                (self.run_id,),
        ):
            pending[row['client_order_id']] = PendingRecord(
                key=row['intent_key'],
                coid=row['client_order_id'],
            )

        return envelopes, pending

    # --- Orders ------------------------------------------------------------

    def upsert_order(
            self, client_order_id: str, **fields: Any,
    ) -> None:
        """Order sor UPSERT — új sor vagy meglévő frissítés.

        Elfogadott mezők: ``symbol``, ``side``, ``qty``, ``state``,
        ``intent_key``, ``exchange_order_id``, ``from_entry``,
        ``pine_entry_id``, ``sl_level``, ``tp_level``, ``trailing_stop``,
        ``trailing_distance``, ``filled_qty``, ``extras``. A hiányzó
        mezőket a DB default-ja szolgálja új sor esetén; meglévő sor
        frissítésekor a nem megadott mezőket nem változtatja.

        ``extras`` paraméter dict-ként érkezik és JSON-stringbe szerializálódik.

        :raises ValueError: Új sor beszúrásakor hiányzó kötelező mező
            (``symbol``, ``side``, ``qty``, ``state``).
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
                        f"upsert_order({client_order_id!r}) új sor, hiányzó "
                        f"kötelező mezők: {missing}"
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

            # Update path: csak a kifejezetten megadott mezőket írja át.
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
        """Egyetlen mező-frissítés: ``orders.state``."""
        self.upsert_order(client_order_id, state=state)

    def set_exchange_id(
            self, client_order_id: str, exchange_order_id: str,
    ) -> None:
        """``orders.exchange_order_id`` kitöltése (Capital.com confirm, IB orderId, ...)."""
        self.upsert_order(client_order_id, exchange_order_id=exchange_order_id)

    def set_risk(
            self, client_order_id: str, *,
            sl: float | None = None,
            tp: float | None = None,
            trailing_stop: bool | None = None,
            trailing_distance: float | None = None,
    ) -> None:
        """SL/TP/trailing attribútumok egyben-frissítése.

        A ``None`` paraméter *nem* törli a meglévő értéket — csak azt
        jelzi, hogy a caller most nem állítja be. Ha törölni kell, írj
        explicit ``sl=0.0``-t vagy használj külön UPDATE-et (jelenleg
        nincs külön delete-metódus — későbbi igény esetén bekerül).
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
        """``orders.filled_qty`` frissítése (delta-nem-kumulatív, a caller átadja a teljes mennyiséget)."""
        self.upsert_order(client_order_id, filled_qty=filled_qty)

    def close_order(self, client_order_id: str) -> None:
        """Order lezárása: ``closed_ts_ms`` + kapcsolódó ``order_refs`` törlés
        + ``order_closed`` audit-event egy tranzakcióban.

        Az ``order_refs`` azonnali takarítása tartja a tábla méretét a
        live-ordereknél nagyságrendileg. Historikus dealReference / dealId
        visszakereshetőségre az ``events`` tábla szolgál — ezért írunk be
        ide is egy eseményt a lezáráskor érvényes `exchange_order_id`-vel.
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
        """Broker-specifikus alias kulcs rögzítése.

        Pl. Capital.com ``deal_reference`` a POST-válaszból, később
        ``deal_id`` a confirm-ből. IB esetén ``perm_id`` / ``order_id``.
        A ``(run_instance_id, ref_type, ref_value)` triplet egyedi.
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

    def find_by_ref(
            self, ref_type: str, ref_value: str,
    ) -> OrderRow | None:
        """Alias-alapú order-lookup O(log n)-ben.

        Join ``order_refs`` × ``orders`` a PK-n. Egyetlen indexelt SELECT,
        ami ezt a use-case-t egyszeri DB-hívássá redukálja.
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
        """Direkt lookup a CO-ID-re."""
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
        """Élő (nem lezárt) orderek iterátora.

        Partial index (``idx_orders_live``) szolgálja ki a filtert; egy
        realisztikus one-way Pine stratégiának <50 élő sora van bármikor,
        a query-költség elhanyagolható.
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
        """Egy audit-event írása.

        A ``payload`` JSON-ba szerializálódik; plugin-specifikus mezők
        szabadon bekerülhetnek.
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

    # --- Lifecycle --------------------------------------------------------

    def heartbeat(self) -> None:
        """Aktuális run heartbeat-je. Rate-limited ``HEARTBEAT_INTERVAL_MS``-re.

        A caller nyugodtan hívhatja minden sync-ciklusban — a belső
        gate biztosítja, hogy ne legyen egynél több UPDATE / perc.
        Cross-run cleanupot NEM futtat (az kizárólag ``open_run()``
        felelőssége — felelősségi határok).
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
        """Happy-path run-lezárás: ``ended_ts_ms`` kitöltése.

        Ismétlődő hívásra no-op (az első UPDATE után az ended_ts_ms már
        nem-NULL, a WHERE kidobja). SIGKILL-t a stale-cleanup kezel,
        ez csak a kontrollált leállás útvonala.
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
    """``sqlite3.Row`` → :class:`OrderRow` konverzió, extras JSON-parse."""
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
