"""
Standalone unit tests for the unified SQLite-based ``BrokerStore``.

Ezek a tesztek semmit nem feltételeznek az ``OrderSyncEngine``-ről vagy
plugin integrációról — csak a storage réteg önmagában. Az end-to-end
(sync_engine + storage) forgatókönyvek a ``test_030_broker_store_sync_engine.py``
alá kerülnek.
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
    """Első megnyitáskor a séma létrejön, a ``_migrations`` tábla bejegyzést kap."""
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN):
        pass

    # Közvetlenül sqlite3-mal nézzük — függetlenül a BrokerStore-tól.
    conn = sqlite3.connect(str(path))
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        assert version >= 1

        rows = conn.execute(
            "SELECT version, description FROM _migrations ORDER BY version"
        ).fetchall()
        assert rows, "migration táblának kell legalább egy sor"
        assert rows[0][0] == 1

        # Séma-ellenőrzés: minden fő tábla létezik.
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

        # A ``live_runs`` VIEW is létrejött.
        views = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='view'"
            )
        }
        assert 'live_runs' in views
    finally:
        conn.close()


def __test_migrations_idempotent_on_reopen__(tmp_path: Path) -> None:
    """Második megnyitás nem dobja el / nem duplikálja a sémát."""
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN):
        pass
    with BrokerStore(path, plugin_name=PLUGIN):
        pass

    conn = sqlite3.connect(str(path))
    try:
        rows = conn.execute("SELECT version FROM _migrations").fetchall()
        # Pontosan egy darab — nem futott újra a v1 migration.
        assert [r[0] for r in rows] == [1]
    finally:
        conn.close()


def __test_wal_mode_enabled__(tmp_path: Path) -> None:
    """Induláskor WAL mode-ban kell lennie — crash-safety + concurrent read."""
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"


# === open_run lifecycle ===================================================


def __test_open_run_inserts_runs_row_and_returns_context__(tmp_path: Path) -> None:
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
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx_a = _open_run(store, label="a")
        assert ctx_a.run_id.endswith("#a")
        ctx_a.close()

        ctx_b = _open_run(store, label="b")
        assert ctx_b.run_id.endswith("#b")
        # Eltérő label → eltérő run_tag is (az input bővült).
        assert ctx_a.run_tag != ctx_b.run_tag


def __test_open_run_collision_raises__(tmp_path: Path) -> None:
    """Ugyanaz a run_id kétszer aktív → RuntimeError a második open_run-nál."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        _open_run(store)
        with pytest.raises(RuntimeError, match="Aktív run_id már létezik"):
            _open_run(store)


def __test_close_allows_reopen_with_same_run_id__(tmp_path: Path) -> None:
    """Lezárt run után ugyanaz a run_id újra nyitható (új run_instance_id)."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx1 = _open_run(store)
        ctx1.close()

        ctx2 = _open_run(store)
        assert ctx2.run_id == ctx1.run_id
        assert ctx2.run_instance_id != ctx1.run_instance_id

        # Történelmileg mindkét sor megvan — "mutasd a bot összes futását"
        # lekérdezhető.
        run_rows = store._conn.execute(
            "SELECT run_instance_id, ended_ts_ms FROM runs WHERE run_id = ? "
            "ORDER BY run_instance_id",
            (ctx1.run_id,),
        ).fetchall()
        assert len(run_rows) == 2
        assert run_rows[0]['ended_ts_ms'] is not None  # első lezárva
        assert run_rows[1]['ended_ts_ms'] is None      # második aktív


# === Stale-run cleanup ====================================================


def __test_stale_run_auto_cleaned_on_open_run__(tmp_path: Path) -> None:
    """SIGKILL szimuláció: last_heartbeat_ts_ms elavul → open_run lezárja."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx_crashed = _open_run(store)
        # Szimuláljuk a heartbeat-hiányt: közvetlenül DB-be hamisítjuk.
        fake_stale_hb = 1_000_000  # messze a múltban
        store._conn.execute(
            "UPDATE runs SET last_heartbeat_ts_ms = ? WHERE run_instance_id = ?",
            (fake_stale_hb, ctx_crashed.run_instance_id),
        )
        store._conn.commit()

        # Újraindulás szimuláció: új open_run ugyanarra a run_id-ra.
        ctx_new = _open_run(store)
        assert ctx_new.run_instance_id != ctx_crashed.run_instance_id

        # A régi sor lezárt, az új él.
        old = store._conn.execute(
            "SELECT ended_ts_ms FROM runs WHERE run_instance_id = ?",
            (ctx_crashed.run_instance_id,),
        ).fetchone()
        assert old['ended_ts_ms'] == fake_stale_hb

        # Esemény logja a cleanup-nak.
        ev = store._conn.execute(
            "SELECT kind, payload FROM events "
            "WHERE run_instance_id = ? AND kind = 'stale_run_cleaned'",
            (ctx_crashed.run_instance_id,),
        ).fetchone()
        assert ev is not None
        payload = json.loads(ev['payload'])
        assert payload['run_id'] == ctx_crashed.run_id


def __test_cleanup_stale_runs_public_api__(tmp_path: Path) -> None:
    """Publikus ``cleanup_stale_runs`` is működik (debug CLI forgatókönyv)."""
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
    """A ``live_runs`` VIEW akkor is szűri a zombikat, ha fizikai cleanup nem futott."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx_fresh = _open_run(store)
        ctx_stale = _open_run(store, label="stale")

        # A stale run sorának heartbeat-je túl régi — a VIEW-nek ki kell hagynia.
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
            "live_runs VIEW-nek ki kell hagynia a stale sort akkor is, "
            "ha a runs táblában még ``ended_ts_ms IS NULL`` állapotú"
        )


# === Envelope / pending replay ============================================


def __test_replay_round_trip_envelope_and_pending__(tmp_path: Path) -> None:
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
    assert pending == {
        "coid-1": PendingRecord(key="Long", coid="coid-1"),
    }


def __test_record_complete_drops_envelope_and_pending__(tmp_path: Path) -> None:
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        ctx.record_envelope(key="Long", bar_ts_ms=BAR_TS, retry_seq=0)
        ctx.record_park(coid="coid-1", key="Long")
        ctx.record_complete(key="Long")

        envelopes, pending = ctx.replay()
        assert envelopes == {}
        assert pending == {}


def __test_record_unpark_drops_only_that_coid__(tmp_path: Path) -> None:
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
    """Azonos ``intent_key``-re ismételt ``record_envelope`` felülírja a sort."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        ctx.record_envelope(key="Long", bar_ts_ms=BAR_TS, retry_seq=0)
        ctx.record_envelope(key="Long", bar_ts_ms=BAR_TS, retry_seq=1)

        envelopes, _ = ctx.replay()
        assert envelopes["Long"].retry_seq == 1


# === Multi-run izoláció ===================================================


def __test_replay_isolated_per_run_instance__(tmp_path: Path) -> None:
    """Két párhuzamos run nem látja egymás envelope-jait."""
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
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)

        # Új sor
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

        # Állapot-frissítés
        ctx.set_order_state("coid-1", "confirmed")
        row = ctx.get_order("coid-1")
        assert row is not None
        assert row.state == "confirmed"
        # A meg nem adott mezőket nem változtatta.
        assert row.sl_level == 1.0
        assert row.extras == {"deal_reference": "o_abc"}


def __test_upsert_order_requires_core_fields_for_new_row__(tmp_path: Path) -> None:
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        with pytest.raises(ValueError, match="hiányzó kötelező mezők"):
            ctx.upsert_order("coid-missing", state="submitted")


def __test_add_ref_and_find_by_ref_round_trip__(tmp_path: Path) -> None:
    """Generikus alias lookup: deal_reference → OrderRow egy indexelt SELECT-tel."""
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

        # Nem létező kulcs → None, nem raise.
        assert ctx.find_by_ref("deal_id", "ghost") is None


def __test_close_order_cascades_delete_refs__(tmp_path: Path) -> None:
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        ctx.upsert_order(
            "coid-1", symbol="EURUSD", side="buy", qty=1.0, state="submitted",
        )
        ctx.add_ref("coid-1", "deal_reference", "o_abc")
        ctx.add_ref("coid-1", "deal_id", "p_xyz")

        ctx.close_order("coid-1")

        # Az order megvan, de closed.
        row = ctx.get_order("coid-1")
        assert row is not None
        assert row.closed_ts_ms is not None
        # A refs cascade-törlődtek.
        assert ctx.find_by_ref("deal_reference", "o_abc") is None
        assert ctx.find_by_ref("deal_id", "p_xyz") is None


def __test_iter_live_orders_filters_closed__(tmp_path: Path) -> None:
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
    """``set_risk`` ``None`` paraméterei nem törlik a meglévő értékeket."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        ctx.upsert_order(
            "c", symbol="EURUSD", side="buy", qty=1.0, state="submitted",
            sl_level=1.0, tp_level=1.2,
        )
        ctx.set_risk("c", sl=1.1)  # csak SL-t állítunk

        row = ctx.get_order("c")
        assert row is not None
        assert row.sl_level == 1.1
        assert row.tp_level == 1.2, "a set_risk=None nem törölheti a TP-t"


# === Extras JSON ==========================================================


def __test_extras_round_trip_preserves_dict__(tmp_path: Path) -> None:
    payload = {
        "deal_reference": "o_abc",
        "working_order": True,
        "nested": {"x": 1, "y": [1, 2, 3]},
        "unicode": "árvíztűrő tükörfúrógép",
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
    """Két gyors egymás után hívott heartbeat → csak az első ír DB-be."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        # Az open_run már beírt egy last_heartbeat_ts_ms-t; rögzítsük.
        hb_after_open = store._conn.execute(
            "SELECT last_heartbeat_ts_ms FROM runs WHERE run_instance_id = ?",
            (ctx.run_instance_id,),
        ).fetchone()[0]

        # Szimuláljuk, hogy már írtunk nemrég (most ms). Ez aktiválja a gate-et.
        from pynecore.core.broker import storage as _storage
        ctx._last_heartbeat_write_ms = _storage._now_ms()
        ctx.heartbeat()  # no-op — túl gyorsan a legutóbbi írás óta

        hb_after_nothing = store._conn.execute(
            "SELECT last_heartbeat_ts_ms FROM runs WHERE run_instance_id = ?",
            (ctx.run_instance_id,),
        ).fetchone()[0]
        assert hb_after_nothing == hb_after_open, "rate-limit gate kihagyta az UPDATE-et"

        # Most szimuláljuk, hogy az előző írás régebbi mint az interval.
        ctx._last_heartbeat_write_ms = (
            _storage._now_ms() - HEARTBEAT_INTERVAL_MS - 1
        )
        ctx.heartbeat()

        hb_after_write = store._conn.execute(
            "SELECT last_heartbeat_ts_ms FROM runs WHERE run_instance_id = ?",
            (ctx.run_instance_id,),
        ).fetchone()[0]
        assert hb_after_write >= hb_after_open, "az engedélyezett hívás írt DB-be"


# === Close (happy path) ====================================================


def __test_close_run_sets_ended_ts_ms__(tmp_path: Path) -> None:
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
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        ctx.close()
        # Kétszeri close nem dobhat.
        ctx.close()


# === Torn write / recovery ================================================


def __test_persistence_survives_reopen__(tmp_path: Path) -> None:
    """Első megnyitás írása a második megnyitásban visszaolvasható."""
    path = tmp_path / "broker.sqlite"
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        ctx = _open_run(store)
        ctx.record_envelope(key="Long", bar_ts_ms=BAR_TS, retry_seq=0)
        ctx.upsert_order(
            "coid-1", symbol="EURUSD", side="buy", qty=1.0, state="submitted",
        )
        saved_instance_id = ctx.run_instance_id

    # Második nyitás, különálló BrokerStore — megint le kell kérdezhetőnek lennie.
    with BrokerStore(path, plugin_name=PLUGIN) as store:
        row = store._conn.execute(
            "SELECT run_id FROM runs WHERE run_instance_id = ?",
            (saved_instance_id,),
        ).fetchone()
        assert row is not None
        assert row['run_id'] == "ema_cross@capitalcom-demo-1234567:EURUSD:60"

        # Ha újra megnyitjuk ugyanazt a run-t: collision (az előző még él,
        # heartbeat friss). Ez a helyes viselkedés — a test a ``close`` és
        # stale-cleanup scenariókban külön-külön kipróbálja.


# === OrderRow field coverage ==============================================


def __test_order_row_fields_are_typed__(tmp_path: Path) -> None:
    """A dataclass visszaadja a mezőket a megfelelő típusokkal (nem raw sqlite.Row)."""
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
