"""#48 baseline — the MEASURED divergence-detection matrix (engine + DNSE).

Each probe drives the REAL OrderSyncEngine over the REAL DNSEBroker (fake
client seam) and pins what the stack does TODAY when a strategy's believed
state diverges from the venue book. These are measurement probes, not wishes:
they assert the CURRENT behaviour — holes included — so the matrix is
executable and any silent change to it fails loudly. Each hole names the card
that owns its fix; flip the corresponding assertion when that card lands
(same characterization discipline the #47 panel documented: a pinned
behaviour must carry its WHY).

Matrix (recon 2026-08-25, full citations on card #48):

  (a) startup qty mismatch  -> engine WOULD adopt, but the ownership clamp
      reads the orders journal, which DNSE never writes (#36): adopts NOTHING.
  (a) periodic qty mismatch -> DETECTED-WARN-ONLY since #48's detector:
      signed-pair persistence streak (>=2 reconciles), deferred while fills
      are recent or cancel dispositions pending; one warning per delta value.
      (Was: silent forever — flipped 2026-08-25.)
  (b) tracked id vanishes   -> no detection anywhere: engine delegates to the
      plugin (by design), plugin has no store/DisappearanceTracker (#36).
  (c) two engines, one netted account -> nothing prevents it; run_id keys on
      strategy_id so both open_run calls succeed.
"""
import asyncio
from types import SimpleNamespace

import pytest

import pynecore.lib as lib
lib.bar_index = 0

from pynecore.core.broker.position import BrokerPosition
from pynecore.core.broker.run_identity import RunIdentity
from pynecore.core.broker.storage import BrokerStore, RunContext
from pynecore.core.broker.sync_engine import OrderSyncEngine

from pynecore_dnse import broker as dnse_broker


def _broker(fake_client, **responses):
    """Real DNSEBroker wired to the canned fake client (same seam as
    test_broker_state's helper — duplicated because pytest does not import
    sibling test modules)."""
    cfg = dnse_broker.DNSEBrokerConfig(api_key="k", api_secret="s", account_no="ACC1")
    instance = dnse_broker.DNSEBroker(symbol="VN30F1M", timeframe="5", config=cfg)
    instance._client = fake_client(**responses)
    return instance

SYMBOL = "VN30F1M"


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """watch_orders sleeps its poll interval each cycle — make it instant."""
    async def _fast_sleep(_delay, result=None):
        return result
    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)


@pytest.fixture(autouse=True)
def _stub_script():
    prev = getattr(lib, "_script", None)
    lib._script = SimpleNamespace(initial_capital=1_000_000.0)
    try:
        yield
    finally:
        lib._script = prev


def _open_ctx(store: BrokerStore, strategy_id: str = "divergence") -> RunContext:
    identity = RunIdentity(
        strategy_id=strategy_id, symbol=SYMBOL, timeframe="60",
        account_id="shared-netted-acct", label=None,
    )
    return store.open_run(identity, script_source="// divergence matrix")


def _mk_engine(broker, ctx) -> tuple[OrderSyncEngine, BrokerPosition]:
    pos = BrokerPosition()
    engine = OrderSyncEngine(
        broker=broker,  # type: ignore[arg-type]
        position=pos,
        symbol=SYMBOL,
        run_tag=ctx.run_tag if ctx is not None else "dvrg",
        mintick=0.1,
        store_ctx=ctx,
    )
    return engine, pos


def _venue_long(qty: float):
    return (200, {"positions": [{
        "symbol": SYMBOL, "side": "NB", "openQuantity": qty, "costPrice": 1930.0}]})


# --- (a) startup: venue holds 3, engine believes 0 --------------------------

def __test_matrix_a_startup_adoption_is_clamped_to_nothing__(
        fake_client, tmp_path, caplog):
    """CHARACTERIZATION (annotation corrected at #73 — "#36 owns the fix"
    was stale): over an EMPTY journal, clamped startup adoption over a live
    3-lot venue position adopts NOTHING. Post-#36+#73 this is the CORRECT
    verdict — a run with no journalled fills owns nothing on the shared
    netting account (that net is the operator's); a genuine restart over
    the run's OWN fill is covered by
    ``__test_restart_over_own_filled_position_adopts_it__`` below."""
    store = BrokerStore(tmp_path / "broker.sqlite", plugin_name="dnse")
    ctx = _open_ctx(store)
    b = _broker(fake_client, get_positions=_venue_long(3),
                get_orders=(200, {"orders": []}))
    engine, pos = _mk_engine(b, ctx)

    engine.reconcile()

    assert pos.size == 0.0, (
        "measured hole moved: startup adoption no longer clamps to zero — "
        "if #36 wired the journal, flip this to assert ADOPTION (size 3)")
    assert any("run-owned exposure" in r.message for r in caplog.records), \
        "the clamp is at least LOUD today — losing the warning would be a regression"
    # ("unrecognised side" arm removed: dead since #49 fixed the vocabulary)


# --- (a) periodic: engine believes 4, venue says 3 --------------------------

def __test_matrix_a_periodic_partial_drift_is_detected_warn_only__(
        fake_client, tmp_path, caplog):
    """FLIPPED 2026-08-25 (#48 detector landed; was ...drift_is_silent):
    a persistent partial external close (4 -> 3) must produce EXACTLY ONE
    'position drift' warning after two clean reconciles — and never adopt.
    Deferral guards (recent fill / pending cancel disposition) are covered by
    the dedicated debounce tests below."""
    store = BrokerStore(tmp_path / "broker2.sqlite", plugin_name="dnse")
    ctx = _open_ctx(store, "periodic")
    b = _broker(fake_client, get_positions=_venue_long(3),
                get_orders=(200, {"orders": []}))
    engine, pos = _mk_engine(b, ctx)
    engine._sync_count = 1              # past startup
    pos.size = 4.0                      # the strategy's belief

    caplog.clear()
    engine.reconcile()                  # streak 1 — no warning yet (debounce)
    drift = [r for r in caplog.records if "position drift" in r.message]
    assert not drift, "one observation must not warn — a stale replica would false-alarm"

    engine.reconcile()                  # streak 2 — warn once
    engine.reconcile()                  # streak 3 — same delta, no re-warn
    assert pos.size == 4.0, "engine adopted mid-run?! the documented design refuses that"
    drift = [r for r in caplog.records if "position drift" in r.message]
    assert len(drift) == 1, "exactly one warning per persisted delta value"
    assert "venue 3.0 vs internal 4.0" in drift[0].message


def __test_matrix_a_drift_warning_defers_while_fills_are_recent__(
        fake_client, tmp_path, caplog, monkeypatch):
    """#48 panel guard (R1): a pipelined fill lag can hold a constant delta —
    the detector must stay quiet while ``_last_position_fill_monotonic`` is
    inside the grace window, then warn once the book has been still."""
    import time as _time
    store = BrokerStore(tmp_path / "broker4.sqlite", plugin_name="dnse")
    ctx = _open_ctx(store, "defer")
    b = _broker(fake_client, get_positions=_venue_long(3),
                get_orders=(200, {"orders": []}))
    engine, pos = _mk_engine(b, ctx)
    engine._sync_count = 1
    pos.size = 4.0
    engine._last_position_fill_monotonic = _time.monotonic()  # a fill JUST landed

    caplog.clear()
    engine.reconcile(); engine.reconcile(); engine.reconcile()
    assert not [r for r in caplog.records if "position drift" in r.message], \
        "recent fills must defer the drift warning (fill-race guard)"

    engine._last_position_fill_monotonic = _time.monotonic() - 999.0  # book still
    engine.reconcile(); engine.reconcile()
    assert len([r for r in caplog.records if "position drift" in r.message]) == 1


def __test_matrix_a_drift_rewarns_on_progressive_unwind__(
        fake_client, tmp_path, caplog):
    """#48 panel guard (R3): a progressive external unwind (4->3->2) changes
    the delta each time — the streak must NOT reset (any non-zero delta
    counts), and each new delta value earns exactly one fresh warning."""
    sizes = iter([3, 3, 2, 2, 2])
    def positions(*_a, **_k):
        return _venue_long(next(sizes))
    store = BrokerStore(tmp_path / "broker5.sqlite", plugin_name="dnse")
    ctx = _open_ctx(store, "unwind")
    b = _broker(fake_client, get_positions=positions,
                get_orders=(200, {"orders": []}))
    engine, pos = _mk_engine(b, ctx)
    engine._sync_count = 1
    pos.size = 4.0

    caplog.clear()
    for _ in range(5):
        engine.reconcile()
    drift = [r.message for r in caplog.records if "position drift" in r.message]
    assert len(drift) == 2, "one warning for delta -1, one fresh warning for delta -2"
    assert "delta -1" in drift[0] and "delta -2" in drift[1]


# --- (b) tracked order id vanishes from every book --------------------------

def __test_matrix_b_vanished_order_is_silent_forever__(fake_client, collect):
    """MEASURED HOLE (#36 owns the seam; engine delegates by design): a row
    the venue stops returning is simply never visited again — watch_orders
    dedups rows that ARE present and has no absence path. Zero events, no
    warning, the identity maps still believe the order exists."""
    row = {"id": "777001", "symbol": SYMBOL, "side": "NB", "quantity": 1,
           "fillQuantity": 0, "orderStatus": "New", "price": 1900.0}
    books = {"n": 0}

    def orders(*_a, order_category=None, **_k):
        books["n"] += 1
        if books["n"] <= 2:             # first cycle: NORMAL + STOP books
            return (200, {"orders": [row]} if order_category == "NORMAL"
                    else {"orders": []})
        return (200, {"orders": []})    # then the id vanishes everywhere

    from pynecore.core.broker.models import LegType
    b = _broker(fake_client, get_orders=orders)
    b._identity["777001"] = ("pineV", None, LegType.ENTRY)
    b._order_category["777001"] = "NORMAL"

    events = collect(b.watch_orders(), 2, timeout=0.3)

    assert [e.event_type for e in events] == ["created"], \
        "only the presence cycle emits; absence emits nothing"
    assert "777001" in b._identity, (
        "measured hole moved: the vanished id was retired — if disappearance "
        "detection landed (#36 + tracker wiring), flip this to assert the "
        "cancelled/lost event instead")


# --- (c) two strategies, one netted account ---------------------------------

def __test_matrix_c_nothing_prevents_two_engines_on_one_account__(tmp_path):
    """MEASURED HOLE (#48 scenario-c design doc): run_id keys on strategy_id,
    so a long strategy and a short hedge strategy open the SAME netted
    account with no error, no warning, no lease. The venue nets per symbol;
    neither engine will ever see the other's fills as anything but
    unexplained position drift (see probe a-periodic: which is silent)."""
    store = BrokerStore(tmp_path / "broker3.sqlite", plugin_name="dnse")
    ctx_long = _open_ctx(store, "long-strategy")
    ctx_short = _open_ctx(store, "short-hedge")

    assert ctx_long.run_tag != ctx_short.run_tag
    assert ctx_long is not None and ctx_short is not None, (
        "measured hole moved: open_run now refuses a second engine on the "
        "same account/symbol — flip this when a netting lease exists")


# --- B2 (#59 item 4): the UNCLAMPED sibling startup branch ------------------

def __test_replayed_close_startup_adoption_respects_the_clamp__(
        fake_client, tmp_path, caplog):
    """RED (Phase B2 owns the fix): matrix (a) proves the clamped startup
    branch adopts NOTHING over a no-journal store — but its SIBLING branch
    (`_adopt_size_with_replayed_close`, routed when a replayed defensive-
    close marker exists at startup) adopts the RAW venue snapshot with the
    ownership clamp never applied (#59 item-4, the first of the four
    uncovered raw-net consumers). Same fixture, same empty journal: the
    sibling must reach the same owned figure (nothing) — today it adopts
    the operator's 3 lots as bot exposure in exactly the crash-mid-close
    scenario Phase A exists for."""
    import time as _time
    from pynecore.core.broker.models import (
        BracketAttachRejectContext, PendingDefensiveClose,
    )

    store = BrokerStore(tmp_path / "broker3.sqlite", plugin_name="dnse")
    ctx = _open_ctx(store, "replayed")
    b = _broker(fake_client, get_positions=_venue_long(3),
                get_orders=(200, {"orders": []}))
    engine, pos = _mk_engine(b, ctx)
    engine._pending_defensive_close["Long"] = PendingDefensiveClose(
        entry_id="Long",
        close_intent_key="__pyne_defensive_close__coid-1",
        close_order_ref=None,
        pending_since=_time.time(),
        reject_context=BracketAttachRejectContext(
            intent_key="Bracket\0Long", position_coid="coid-1",
            position_side="buy", qty=3.0, symbol=SYMBOL,
        ),
        close_client_order_id="CLOSE-COID-1",
        pre_close_position_size=3.0,
    )
    engine._replayed_defensive_close_entry_ids.add("Long")

    engine.reconcile()

    assert pos.size == 0.0, (
        f"the replayed-close sibling branch adopted {pos.size} — the RAW "
        f"venue snapshot, clamp never applied, while the clamped branch on "
        f"the identical empty journal adopts nothing (matrix a). On the "
        f"shared netting account that is the operator's exposure adopted "
        f"as the bot's (#59 item-4 consumer 1; Phase B2)")


def __test_restart_over_own_filled_position_adopts_it__(
        fake_client, tmp_path, caplog):
    """RED (Phase B2 owns the fix — a #36/#56 interaction defect found at
    B2's Step 0): core's owned sum reads `filled_qty` on LIVE rows and its
    docstring assumes 'a genuine restart over its own open position finds
    its entry row's cursor' — but `journal_terminal` CLOSES the row on
    FILLED, so the run's own filled 3-lot position contributes ZERO and the
    clamp adopts nothing. The startup-clamp-adopts-zero hole Phase A was
    meant to fix is re-created by our own terminal-close policy."""
    from pynecore.core.broker.models import DispatchEnvelope, EntryIntent, OrderType
    from pynecore_dnse.journal_wiring import journal_terminal

    store = BrokerStore(tmp_path / "broker4.sqlite", plugin_name="dnse")
    ctx = _open_ctx(store, "own-fill")
    b1 = _broker(fake_client,
                 get_security_definition=(200, [{"ceilingPrice": "1550",
                                                 "floorPrice": "1450",
                                                 "securityGroupId": "FU"}]),
                 get_loan_packages=(200, {"loanPackages": [{"id": 42}]}),
                 post_order=(201, {"id": "437346", "symbol": SYMBOL, "side": "NB",
                                   "quantity": 3, "orderStatus": "New"}))
    b1.store_ctx = ctx
    envelope = DispatchEnvelope(
        intent=EntryIntent(pine_id="Long", symbol=SYMBOL, side="buy", qty=3,
                           order_type=OrderType.LIMIT, limit=1500.0),
        run_tag="abcd", bar_ts_ms=1_700_000_000_000, retry_seq=0, coid_max_len=30)
    asyncio.run(b1.execute_entry(envelope))
    journal_terminal(ctx, venue_id="437346", terminal_status="Filled",
                     filled_qty=3.0)      # the watch scan's fill observation
    ctx.close()

    ctx2 = _open_ctx(store, "own-fill")
    b2 = _broker(fake_client, get_positions=_venue_long(3),
                 get_orders=(200, {"orders": []}))
    b2.store_ctx = ctx2
    engine, pos = _mk_engine(b2, ctx2)

    engine.reconcile()

    assert pos.size == 3.0, (
        f"restart over the run's OWN filled position adopted {pos.size} — "
        f"the filled entry's row was CLOSED by journal_terminal, so the "
        f"owned sum sees zero and the clamp refuses our own exposure "
        f"(#36/#56 interaction; Phase B2)")
