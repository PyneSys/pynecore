"""
Unit tests for the cancel-tentative state machine.

Covers the two-level (leg-state + envelope/intent shadow map) design
that protects partial bracket pending legs against ambiguous broker
cancel dispositions (``OrderDispositionUnknownError`` swallowed by
:meth:`OrderSyncEngine._dispatch_cancel`). See the cancel-tentative
state design dossier for the full specification.

Engine-isolated tests (1-7) exercise the
:class:`SoftwarePartialBracketEngine` API directly without a sync
engine or broker. Sync-engine integration tests (8-12) wire a minimal
:class:`_MockBroker` that controls the cancel-pathway return values
deterministically.
"""
import contextlib
import dataclasses
import heapq
import itertools
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from pynecore import lib
from pynecore.core.broker.exceptions import (
    OrderDispositionUnknownError,
)
from pynecore.core.broker.models import (
    CANCEL_REASON_VENUE_POSITION_CLEARED,
    BrokerEvent,
    CancelDispositionOutcome,
    CancelIntent,
    CapabilityLevel,
    CloseIntent,
    DispatchEnvelope,
    EntryDeferredCancelDispositionPendingEvent,
    EntryIntent,
    ExchangeCapabilities,
    ExchangeOrder,
    ExchangePosition,
    ExitIntent,
    LegType,
    OrderExecution,
    OrderStatus,
    OrderType,
    PartialBracketCancelTentativeDegradedEvent,
    PartialBracketCancelTentativeResolvedEvent,
    PartialBracketCancelTentativeStartedEvent,
    QuarantineEndedEvent,
    QuarantineEnteredEvent,
    QuarantineKind,
)
from pynecore.core.broker.intent_builder import build_intents
from pynecore.core.broker.native_failsafe_manager import FailsafeHealth
from pynecore.core.broker import software_partial_bracket_engine as spbe_mod
from pynecore.core.broker.position import BrokerPosition
from pynecore.core.broker.software_partial_bracket_engine import (
    PartialBracketLeg,
    SoftwarePartialBracketEngine,
    _leg_from_row,
)
from pynecore.core.broker.run_identity import RunIdentity
from pynecore.core.broker.storage import (
    ENTRY_FILL_EVIDENCE_ARRIVAL_EVENT_KIND,
    ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND,
    BrokerStore,
    EnvelopeRecord,
    OrderRow,
)
from pynecore.core.broker.store_helpers import (
    EXTRAS_KEY_CANCEL_TENTATIVE_SINCE_TS_MS,
    EXTRAS_KEY_INTENT_PARTIAL_QTY,
    EXTRAS_KEY_LEG_KIND,
    EXTRAS_KEY_LEG_STATE,
    EXTRAS_KEY_PARENT_ENTRY_DISPATCH_REF,
    EXTRAS_KEY_PARENT_PINE_ENTRY_ID,
    EXTRAS_KEY_TRIGGER_LEVEL,
    EXTRAS_KEY_TRIGGER_OFFSET,
    LEG_KIND_SL_PARTIAL,
    LEG_KIND_TP_PARTIAL,
    LEG_KIND_TRAIL_PARTIAL,
    LEG_STATE_ABORTED_PARENT_GONE,
    LEG_STATE_ABORTED_PARENT_NEVER_ARRIVED,
    LEG_STATE_ARMED,
    LEG_STATE_CANCEL_TENTATIVE,
    LEG_STATE_PENDING_ENTRY,
    STATE_PARTIAL_BRACKET_LEG,
)
from pynecore.core.broker.sync_engine import (
    _VENUE_RECOVERY_CAP_S,
    _VENUE_RECOVERY_FIRST_S,
    OrderSyncEngine,
    _CancelTentativeMeta,
)
from pynecore.lib.strategy import Trade
from types import SimpleNamespace


SYMBOL = "BTCUSDT"
# 4 alphanumeric chars — the COID builder validates the tag shape once a
# store-backed test journals a cancel COID (idempotency.build_client_order_id).
RUN_TAG = "ct27"
BAR_TS = 1_700_000_000_000


@pytest.fixture(autouse=True)
def _stub_script():
    prev = lib._script
    lib._script = SimpleNamespace(initial_capital=1_000_000.0)
    try:
        yield
    finally:
        lib._script = prev


# === Engine-isolated leg fixtures =====================================


def _mk_leg(
        *,
        coid: str = 'leg-1',
        pine_id: str = 'exit_tp',
        from_entry: str = 'parent',
        leg_kind: str = LEG_KIND_TP_PARTIAL,
        leg_state: str = LEG_STATE_PENDING_ENTRY,
        side: str = 'sell',
        trigger_level: float | None = 110.0,
) -> PartialBracketLeg:
    """Build a minimal :class:`PartialBracketLeg` for engine tests."""
    intent_key = f"{pine_id}\x00{from_entry}"
    return PartialBracketLeg(
        coid=coid,
        symbol=SYMBOL,
        pine_id=pine_id,
        from_entry=from_entry,
        leg_kind=leg_kind,
        leg_state=leg_state,
        side=side,
        qty=1.0,
        intent_key=intent_key,
        parent_pine_entry_id=from_entry,
        parent_entry_dispatch_ref='parent-coid',
        intent_partial_qty=1.0,
        trigger_level=trigger_level,
    )


def _mk_engine_only() -> SoftwarePartialBracketEngine:
    """Engine instance without storage — extras-only persistence."""
    return SoftwarePartialBracketEngine(store_ctx=None)


# === Tests 1-7: engine-isolated ========================================


def __test_cancel_tentative_marks_pending_entry_legs__():
    """``mark_legs_cancel_tentative`` flips PENDING_ENTRY → CANCEL_TENTATIVE."""
    engine = _mk_engine_only()
    leg = _mk_leg()
    engine.register_leg(leg)
    flipped = engine.mark_legs_cancel_tentative(
        'parent', reason='broker_timeout', now_ms=BAR_TS,
    )
    assert len(flipped) == 1
    assert leg.leg_state == LEG_STATE_CANCEL_TENTATIVE
    assert leg.extras[EXTRAS_KEY_CANCEL_TENTATIVE_SINCE_TS_MS] == BAR_TS
    assert leg.extras['cancel_tentative_reason'] == 'broker_timeout'
    assert engine.has_cancel_tentative_legs('parent') is True


def __test_cancel_tentative_does_not_affect_armed_legs__():
    """Armed legs MUST NOT be flipped by ``mark_legs_cancel_tentative``."""
    engine = _mk_engine_only()
    leg = _mk_leg(leg_state=LEG_STATE_ARMED)
    engine.register_leg(leg)
    flipped = engine.mark_legs_cancel_tentative(
        'parent', reason='broker_timeout', now_ms=BAR_TS,
    )
    assert flipped == []
    assert leg.leg_state == LEG_STATE_ARMED
    assert EXTRAS_KEY_CANCEL_TENTATIVE_SINCE_TS_MS not in leg.extras


def __test_iter_cancel_tentative_parents_dedups_trio__():
    """A TP+SL+TRAIL trio under one parent yields ONE parent key, not three."""
    engine = _mk_engine_only()
    for kind in (LEG_KIND_TP_PARTIAL, LEG_KIND_SL_PARTIAL, LEG_KIND_TRAIL_PARTIAL):
        engine.register_leg(_mk_leg(
            coid=f'leg-{kind}', pine_id=f'exit_{kind}', leg_kind=kind,
        ))
    engine.mark_legs_cancel_tentative(
        'parent', reason='broker_timeout', now_ms=BAR_TS,
    )
    parents = engine.iter_cancel_tentative_parents()
    assert parents == {'parent'}


def __test_restore_cancel_tentative_parent_filled_returns_armed__():
    """``restore(parent_filled=True)`` → leg armed."""
    engine = _mk_engine_only()
    leg = _mk_leg()
    engine.register_leg(leg)
    engine.mark_legs_cancel_tentative(
        'parent', reason='r', now_ms=BAR_TS,
    )
    restored = engine.restore_legs_from_cancel_tentative(
        'parent', parent_filled=True, reason='already_filled',
    )
    assert len(restored) == 1
    assert leg.leg_state == LEG_STATE_ARMED
    assert leg.extras[EXTRAS_KEY_CANCEL_TENTATIVE_SINCE_TS_MS] is None
    assert engine.has_cancel_tentative_legs('parent') is False


def __test_restore_cancel_tentative_parent_pending_returns_pending_entry__():
    """``restore(parent_filled=False)`` → leg pending_entry."""
    engine = _mk_engine_only()
    leg = _mk_leg()
    engine.register_leg(leg)
    engine.mark_legs_cancel_tentative(
        'parent', reason='r', now_ms=BAR_TS,
    )
    engine.restore_legs_from_cancel_tentative(
        'parent', parent_filled=False, reason='still_pending',
    )
    assert leg.leg_state == LEG_STATE_PENDING_ENTRY


def __test_confirm_cancel_tentative_marks_aborted_and_closes__():
    """``confirm_cancel_tentative`` → ABORTED_PARENT_NEVER_ARRIVED + evicted."""
    engine = _mk_engine_only()
    leg = _mk_leg()
    engine.register_leg(leg)
    engine.mark_legs_cancel_tentative(
        'parent', reason='r', now_ms=BAR_TS,
    )
    confirmed = engine.confirm_cancel_tentative('parent', reason='confirmed')
    assert len(confirmed) == 1
    assert leg.leg_state == LEG_STATE_ABORTED_PARENT_NEVER_ARRIVED
    # Evicted from in-memory ledger (close_row=True).
    assert engine.get_leg(leg.key) is None
    assert engine.has_cancel_tentative_legs('parent') is False


def __test_idempotent_restore_after_confirm_is_noop__():
    """Restore after confirm: idempotent no-op (no exception, empty list)."""
    engine = _mk_engine_only()
    leg = _mk_leg()
    engine.register_leg(leg)
    engine.mark_legs_cancel_tentative(
        'parent', reason='r', now_ms=BAR_TS,
    )
    engine.confirm_cancel_tentative('parent', reason='confirmed')
    # A late event-driven restore arrives after the reconcile-driven confirm.
    result = engine.restore_legs_from_cancel_tentative(
        'parent', parent_filled=True, reason='late_event',
    )
    assert result == []


# === Sync-engine integration: mock broker ==============================


@dataclass
class _MockBroker:
    """Minimal stand-in for :class:`BrokerPlugin` exercising only the
    cancel pathway and the partial-qty-bracket SOFTWARE capability.

    Distinct knobs control the original ``execute_cancel`` (which can
    raise :class:`OrderDispositionUnknownError`) and the new
    ``execute_cancel_with_outcome`` (which returns a normalized
    :class:`CancelDispositionOutcome`). Tests configure each independently
    so the cancel-tentative entry path and the reconcile resolution path
    can be exercised in isolation.
    """
    client_order_id_max_len = 30  # BrokerPlugin contract attribute
    raise_on_cancel: list[Exception] = field(default_factory=list)
    cancel_returns: list[bool] = field(default_factory=list)
    cancel_outcomes: list[CancelDispositionOutcome] = field(default_factory=list)
    raise_on_cancel_with_outcome: list[Exception] = field(default_factory=list)
    cancel_with_outcome_calls: list[DispatchEnvelope] = field(default_factory=list)
    cancel_calls: list[DispatchEnvelope] = field(default_factory=list)
    capabilities: ExchangeCapabilities = field(default_factory=lambda: ExchangeCapabilities(
        partial_qty_bracket_exit=CapabilityLevel.SOFTWARE,
    ))
    on_unexpected_cancel: str = "stop"

    def get_capabilities(self) -> ExchangeCapabilities:
        return self.capabilities

    async def execute_cancel(self, envelope: DispatchEnvelope) -> bool:
        self.cancel_calls.append(envelope)
        if self.raise_on_cancel:
            raise self.raise_on_cancel.pop(0)
        if self.cancel_returns:
            return self.cancel_returns.pop(0)
        return True

    async def execute_cancel_with_outcome(
            self, envelope: DispatchEnvelope,
    ) -> CancelDispositionOutcome:
        self.cancel_with_outcome_calls.append(envelope)
        if self.raise_on_cancel_with_outcome:
            raise self.raise_on_cancel_with_outcome.pop(0)
        if self.cancel_outcomes:
            return self.cancel_outcomes.pop(0)
        return CancelDispositionOutcome.UNKNOWN

    async def get_position(self, symbol: str) -> ExchangePosition | None:
        return None

    # The cancel-tentative tests never call entry/exit/close — provide
    # only the surface the reconcile path touches.
    async def get_open_orders(self, symbol: str | None = None):
        return []

    def watch_orders(self):
        async def _gen():
            # The dead yield only makes _gen an async generator
            # noinspection PyUnreachableCode
            if False:
                yield  # pragma: no cover — empty stream
        return _gen()


def _mk_engine(
        broker: _MockBroker,
        events: list[BrokerEvent] | None = None,
        cancel_tentative_stale_grace_s: float = 10.0,
        store_ctx=None,
) -> OrderSyncEngine:
    pos = BrokerPosition()
    return OrderSyncEngine(
        broker=broker,  # type: ignore[arg-type]
        position=pos,
        symbol=SYMBOL,
        run_tag=RUN_TAG,
        mintick=1.0,
        broker_event_sink=(events.append if events is not None else None),
        cancel_tentative_stale_grace_s=cancel_tentative_stale_grace_s,
        store_ctx=store_ctx,
    )


def _run_identity() -> RunIdentity:
    return RunIdentity(
        strategy_id="t027", symbol=SYMBOL, timeframe="60",
        account_id="testbroker-demo", label=None,
    )


def _seed_parent_intent_and_leg(
        engine: OrderSyncEngine,
        *,
        intent_key: str = 'parent',
        with_leg: bool = True,
) -> EntryIntent:
    """Populate the engine state as if a parent EntryIntent had been
    dispatched and its partial brackets were registered. Returns the
    seeded EntryIntent so tests can pass it back into the engine
    internals. ``with_leg=False`` seeds the structural corner the
    parent-level journal persistence exists for: an entry with NO
    partial-bracket legs, where the shadow map alone guards the key."""
    intent = EntryIntent(
        pine_id=intent_key,
        symbol=SYMBOL,
        side='buy',
        qty=1.0,
        order_type=OrderType.MARKET,
    )
    engine._active_intents[intent.intent_key] = intent
    engine._order_mapping[intent.intent_key] = ['xchg-parent']
    envelope = DispatchEnvelope(
        intent=intent,
        run_tag=RUN_TAG,
        bar_ts_ms=BAR_TS,
        retry_seq=0,
    )
    engine._envelopes[intent.intent_key] = envelope
    if with_leg:
        leg = _mk_leg(from_entry=intent.intent_key)
        engine._partial_bracket_engine.register_leg(leg)
    engine._current_bar_ts_ms = BAR_TS
    return intent


# === Tests 8-12: sync-engine integration ===============================


def __test_dispatch_cancel_unknown_disposition_enters_cancel_tentative__():
    """``_dispatch_cancel`` swallows the unknown disposition → shadow-map populated + leg flipped.

    Mapping and envelope are RETAINED (the reconcile retry needs them);
    the diff-loop's adoption check sees the tentative flag and defers.
    """
    events: list[BrokerEvent] = []
    broker = _MockBroker(
        raise_on_cancel=[OrderDispositionUnknownError(
            "cancel timed out", client_order_id='xchg-parent',
        )],
    )
    engine = _mk_engine(broker, events=events)
    intent = _seed_parent_intent_and_leg(engine)
    wall_before_ms = int(time.time() * 1000)
    engine._dispatch_cancel(intent)
    wall_after_ms = int(time.time() * 1000)
    assert intent.intent_key in engine._cancel_disposition_pending
    meta = engine._cancel_disposition_pending[intent.intent_key]
    # The stale-grace deadline is anchored on wall-clock time, NOT
    # the bar timestamp — a seconds-based grace must age in real time
    # so it does not depend on the chart timeframe. See
    # :meth:`OrderSyncEngine._cancel_tentative_now_ms`.
    assert wall_before_ms <= meta.since_ts_ms <= wall_after_ms
    assert meta.since_ts_ms != BAR_TS
    assert intent.intent_key in engine._order_mapping  # retained
    assert intent.intent_key in engine._envelopes  # retained
    # Started-event emitted exactly once.
    started = [e for e in events
               if isinstance(e, PartialBracketCancelTentativeStartedEvent)]
    assert len(started) == 1
    assert started[0].intent_key == intent.intent_key
    # Leg in CANCEL_TENTATIVE.
    legs = list(engine._partial_bracket_engine.iter_legs())
    assert len(legs) == 1
    assert legs[0].leg_state == LEG_STATE_CANCEL_TENTATIVE


def __test_dispatch_cancel_strict_remains_unaffected__():
    """``_dispatch_cancel_strict`` still raises and sets no cancel-tentative state.

    ``_dispatch_cancel_strict`` continues to raise
    ``OrderDispositionUnknownError`` — no cancel-tentative state set."""
    broker = _MockBroker(
        raise_on_cancel=[OrderDispositionUnknownError(
            "strict cancel timed out", client_order_id='xchg-parent',
        )],
    )
    engine = _mk_engine(broker)
    intent = _seed_parent_intent_and_leg(engine)
    with pytest.raises(OrderDispositionUnknownError):
        engine._dispatch_cancel_strict(intent)
    # Strict path did NOT mark cancel-tentative.
    assert engine._cancel_disposition_pending == {}


def __test_diff_loop_defers_reissue_while_cancel_tentative__():
    """A fresh EntryIntent on a tentative parent defers with an audit event and no broker dispatch.

    A fresh EntryIntent on a tentative parent ⇒ defer + audit event,
    no broker dispatch."""
    events: list[BrokerEvent] = []
    broker = _MockBroker()
    engine = _mk_engine(broker, events=events)
    # Seed shadow map directly so the test doesn't depend on
    # _dispatch_cancel's plumbing.
    from pynecore.core.broker.sync_engine import _CancelTentativeMeta
    engine._cancel_disposition_pending['parent'] = _CancelTentativeMeta(
        since_ts_ms=BAR_TS, reason='broker_timeout',
    )
    engine._current_bar_ts_ms = BAR_TS
    new_intent = EntryIntent(
        pine_id='parent', symbol=SYMBOL, side='buy', qty=1.0,
        order_type=OrderType.MARKET,
    )
    engine._diff_and_dispatch([new_intent])
    # Refuse-and-defer: not adopted, not dispatched.
    assert 'parent' not in engine._active_intents
    # Audit event emitted.
    deferred = [e for e in events
                if isinstance(e, EntryDeferredCancelDispositionPendingEvent)]
    assert len(deferred) == 1
    assert deferred[0].intent_key == 'parent'


def __test_reconcile_cancel_confirmed_resolves_and_clears__():
    """``reconcile()`` retry → CANCEL_CONFIRMED → leg aborted, state cleared."""
    events: list[BrokerEvent] = []
    broker = _MockBroker(
        cancel_outcomes=[CancelDispositionOutcome.CANCEL_CONFIRMED],
    )
    engine = _mk_engine(broker, events=events)
    intent = _seed_parent_intent_and_leg(engine)
    from pynecore.core.broker.sync_engine import _CancelTentativeMeta
    engine._cancel_disposition_pending[intent.intent_key] = _CancelTentativeMeta(
        since_ts_ms=BAR_TS, reason='broker_timeout',
    )
    # Drive the cancel-retry-loop directly (skip the full reconcile
    # pre-amble that would need a stocked broker).
    engine._drive_cancel_tentative(now_ms=BAR_TS + 1_000)
    assert engine._cancel_disposition_pending == {}
    assert intent.intent_key not in engine._order_mapping
    assert intent.intent_key not in engine._envelopes
    assert intent.intent_key not in engine._active_intents
    # Resolved-event emitted with the correct outcome.
    resolved = [e for e in events
                if isinstance(e, PartialBracketCancelTentativeResolvedEvent)]
    assert len(resolved) == 1
    assert resolved[0].outcome is CancelDispositionOutcome.CANCEL_CONFIRMED
    assert resolved[0].via == 'reconcile_retry'


def _degraded(engine: OrderSyncEngine, key: str) -> bool:
    """The parent is degraded: an entry block holds under its cause, and the
    entry stays tentative so the probe goes on."""
    return (engine.quarantined
            and engine.quarantine_kind is QuarantineKind.ENTRY_BLOCK
            and engine._entry_block_causes.get(f"cancel_disposition:{key}")
            == 'partial_bracket_cancel_disposition_unresolved'
            and key in engine._cancel_disposition_pending)


def __test_reconcile_stale_grace_promotes_to_a_degraded_entry_block__():
    """Stale-grace expiry after an in-session retry ⇒ degraded: new entries
    are blocked, the process stays alive and the entry stays tentative.

    ``retry_count=1`` records that one ``execute_cancel_with_outcome``
    attempt already ran in this session — the precondition the
    degradation requires (a zero-retry expired entry gets probed first,
    see ``__test_stale_anchor_restart_probes_before_degrading__``).
    """
    events: list[BrokerEvent] = []
    broker = _MockBroker()
    engine = _mk_engine(broker, events=events,
                       cancel_tentative_stale_grace_s=1.0)
    intent = _seed_parent_intent_and_leg(engine)
    from pynecore.core.broker.sync_engine import _CancelTentativeMeta
    # Anchor 5s in the past — well beyond the 1s stale grace.
    engine._cancel_disposition_pending[intent.intent_key] = _CancelTentativeMeta(
        since_ts_ms=BAR_TS - 5_000, reason='broker_timeout', retry_count=1,
    )
    engine._drive_cancel_tentative(now_ms=BAR_TS)
    # Degraded: entry block latched under its own cause.
    assert _degraded(engine, intent.intent_key)
    # Degraded-event emitted.
    degraded = [e for e in events
                if isinstance(e, PartialBracketCancelTentativeDegradedEvent)]
    assert len(degraded) == 1
    assert degraded[0].intent_key == intent.intent_key
    # The quarantine is entered once, with the stable reason.
    entered = [e for e in events if isinstance(e, QuarantineEnteredEvent)]
    assert len(entered) == 1
    assert entered[0].reason == 'partial_bracket_cancel_disposition_unresolved'
    assert entered[0].kind is QuarantineKind.ENTRY_BLOCK


def __test_degraded_cancel_disposition_keeps_probing_and_ends_its_entry_block__():
    """A degraded parent is probed on with the venue re-read backoff, never
    abandoned: the venue's definite answer resolves it, and the next
    reconcile ends the entry block by itself."""
    events: list[BrokerEvent] = []
    broker = _MockBroker(cancel_outcomes=[
        CancelDispositionOutcome.UNKNOWN, CancelDispositionOutcome.CANCEL_CONFIRMED,
    ])
    engine = _mk_engine(broker, events=events, cancel_tentative_stale_grace_s=1.0)
    intent = _seed_parent_intent_and_leg(engine)
    engine._cancel_disposition_pending[intent.intent_key] = _CancelTentativeMeta(
        since_ts_ms=BAR_TS - 5_000, reason='broker_timeout', retry_count=1,
    )
    engine._mark_reads_confirmed()
    engine._drive_cancel_tentative(now_ms=BAR_TS)
    assert _degraded(engine, intent.intent_key)
    assert broker.cancel_with_outcome_calls == []
    # Within the first backoff delay nothing is asked.
    engine._drive_cancel_tentative(now_ms=BAR_TS + 1_000)
    assert broker.cancel_with_outcome_calls == []
    # Past it the venue is asked again; an UNKNOWN keeps the entry degraded.
    engine._drive_cancel_tentative(now_ms=BAR_TS + 6_000)
    assert len(broker.cancel_with_outcome_calls) == 1
    assert _degraded(engine, intent.intent_key)
    # The delay doubled: the next probe waits 10 s, then the venue answers.
    engine._drive_cancel_tentative(now_ms=BAR_TS + 12_000)
    assert len(broker.cancel_with_outcome_calls) == 1
    engine._drive_cancel_tentative(now_ms=BAR_TS + 17_000)
    assert len(broker.cancel_with_outcome_calls) == 2
    assert intent.intent_key not in engine._cancel_disposition_pending
    assert len([e for e in events
                if isinstance(e, PartialBracketCancelTentativeDegradedEvent)]) == 1
    engine._resolve_entry_block_causes(now_ms=BAR_TS + 18_000)
    assert not engine.quarantined
    assert not engine._stale_cancel_dispositions


def __test_stale_grace_waits_while_broker_reads_unconfirmed__():
    """A REST outage must not turn cancel-tentative into a halt.

    With the broker view unconfirmed every probe answers ``UNKNOWN``
    because the venue is unreachable, not because it is ambiguous: the
    loop keeps probing, never promotes, and drops the retry count so the
    halt needs one fresh in-session ``UNKNOWN`` once reads recover.
    """
    events: list[BrokerEvent] = []
    broker = _MockBroker()
    engine = _mk_engine(broker, events=events,
                        cancel_tentative_stale_grace_s=1.0)
    intent = _seed_parent_intent_and_leg(engine)
    engine._cancel_disposition_pending[intent.intent_key] = _CancelTentativeMeta(
        since_ts_ms=BAR_TS - 5_000, reason='broker_timeout', retry_count=1,
    )
    engine._mark_reads_unconfirmed()
    engine._drive_cancel_tentative(now_ms=BAR_TS)
    engine._drive_cancel_tentative(now_ms=BAR_TS + 1_000)
    # Grace expired with an in-session retry, yet no halt: the venue is down.
    assert engine.quarantined is False
    assert len(broker.cancel_with_outcome_calls) == 2
    meta = engine._cancel_disposition_pending[intent.intent_key]
    assert meta.retry_count == 0
    assert not [e for e in events
                if isinstance(e, PartialBracketCancelTentativeDegradedEvent)]
    # Reads recover: the first pass probes again (fresh evidence), the
    # second pass halts on the persistent UNKNOWN.
    engine._mark_reads_confirmed()
    engine._drive_cancel_tentative(now_ms=BAR_TS + 2_000)
    assert engine.quarantined is False
    assert len(broker.cancel_with_outcome_calls) == 3
    assert meta.retry_count == 1
    engine._drive_cancel_tentative(now_ms=BAR_TS + 3_000)
    assert _degraded(engine, intent.intent_key)
    assert len(broker.cancel_with_outcome_calls) == 3


def __test_transport_failure_on_cancel_probe_is_not_venue_evidence__():
    """A cancel probe that never reached the venue must not feed the halt.

    A flaky link can refuse individual requests while the periodic reads
    still succeed, so the broker view stays confirmed. Each refused probe
    raises ``OrderDispositionUnknownError`` from the plugin; the loop keeps
    the entry armed, resets the retry count (no answer is no evidence), and
    never promotes on those alone. Only a probe the venue answered with
    ``UNKNOWN`` counts, and the pass after it halts.
    """
    events: list[BrokerEvent] = []
    broker = _MockBroker(raise_on_cancel_with_outcome=[
        OrderDispositionUnknownError("refused", client_order_id='xchg-parent'),
        OrderDispositionUnknownError("refused", client_order_id='xchg-parent'),
        OrderDispositionUnknownError("refused", client_order_id='xchg-parent'),
    ])
    engine = _mk_engine(broker, events=events,
                        cancel_tentative_stale_grace_s=1.0)
    intent = _seed_parent_intent_and_leg(engine)
    engine._cancel_disposition_pending[intent.intent_key] = _CancelTentativeMeta(
        since_ts_ms=BAR_TS - 5_000, reason='broker_timeout',
    )
    engine._mark_reads_confirmed()
    for step in range(1, 4):
        engine._drive_cancel_tentative(now_ms=BAR_TS + step * 1_000)
    meta = engine._cancel_disposition_pending[intent.intent_key]
    assert engine.quarantined is False
    assert len(broker.cancel_with_outcome_calls) == 3
    assert meta.retry_count == 0
    assert meta.last_retry_outcome is CancelDispositionOutcome.UNKNOWN
    assert not [e for e in events
                if isinstance(e, PartialBracketCancelTentativeDegradedEvent)]
    # The venue answers UNKNOWN once the link is back: that probe is the
    # first real evidence, and the next pass halts on it.
    engine._drive_cancel_tentative(now_ms=BAR_TS + 4_000)
    assert engine.quarantined is False
    assert len(broker.cancel_with_outcome_calls) == 4
    assert meta.retry_count == 1
    engine._drive_cancel_tentative(now_ms=BAR_TS + 5_000)
    assert _degraded(engine, intent.intent_key)
    assert len(broker.cancel_with_outcome_calls) == 4


def __test_dispatch_cancel_unknown_preserves_entry_envelope__():
    """Cancel-tentative entry keeps the original ``EntryIntent`` envelope in ``_envelopes``.

    Cancel-tentative entry must KEEP the original ``EntryIntent`` envelope
    in ``_envelopes``. A later ALREADY_FILLED resolution recovers the
    parent and downstream dispatchers (e.g.
    ``_dispatch_engine_trigger_partial_bracket``) compute the deterministic
    ``KIND_ENTRY`` ``client_order_id`` from this envelope; replacing it
    with a ``CancelIntent`` envelope at the cancel bar would yield a wrong
    COID. The reconcile retry loop builds its own ``CancelIntent``
    envelope on the fly for each retry."""
    broker = _MockBroker(
        raise_on_cancel=[OrderDispositionUnknownError(
            "cancel timed out", client_order_id='xchg-parent',
        )],
    )
    engine = _mk_engine(broker)
    intent = _seed_parent_intent_and_leg(engine)
    engine._dispatch_cancel(intent)
    # Envelope under the parent's intent_key is STILL the EntryIntent.
    stored = engine._envelopes[intent.intent_key]
    assert isinstance(stored.intent, EntryIntent)
    assert stored.intent.intent_key == intent.intent_key


def __test_reconcile_rebuilds_cancel_envelope_when_missing__():
    """Cancel-retry rebuilds a ``CancelIntent`` envelope when only the original intent was retained.

    If the retained envelope is the original (non-cancel) intent — the
    case after restart-replay rehydrate, which repopulates the shadow map
    from leg extras but never the ``_envelopes`` map — the cancel-retry
    loop must rebuild a ``CancelIntent`` envelope on the fly rather than
    forwarding the wrong intent type to the plugin."""
    broker = _MockBroker(
        cancel_outcomes=[CancelDispositionOutcome.CANCEL_CONFIRMED],
    )
    engine = _mk_engine(broker)
    intent = _seed_parent_intent_and_leg(engine)
    # Simulate post-rehydrate state: shadow map populated, but the
    # envelope slot still holds the original EntryIntent (or any non-
    # CancelIntent envelope).
    from pynecore.core.broker.sync_engine import _CancelTentativeMeta
    engine._cancel_disposition_pending[intent.intent_key] = _CancelTentativeMeta(
        since_ts_ms=BAR_TS, reason='rehydrated_from_replay',
    )
    engine._drive_cancel_tentative(now_ms=BAR_TS + 1_000)
    # The plugin saw a CancelIntent, not the EntryIntent we seeded.
    assert len(broker.cancel_with_outcome_calls) == 1
    seen = broker.cancel_with_outcome_calls[0]
    assert isinstance(seen.intent, CancelIntent)
    assert seen.intent.pine_id == intent.pine_id


def __test_reconcile_resolved_drops_child_partial_exit_slots__():
    """CANCEL_CONFIRMED retires every child partial-exit ``ExitIntent`` slot tied to the parent.

    When a parent cancel-tentative resolves as CANCEL_CONFIRMED, every
    child partial-exit ``ExitIntent`` slot tied to that parent's
    ``from_entry`` must be retired alongside the parent. Otherwise a
    later same-id entry sees the stale exit as already active and skips
    the partial-bracket dispatch, landing the position without TP/SL
    protection."""
    broker = _MockBroker(
        cancel_outcomes=[CancelDispositionOutcome.CANCEL_CONFIRMED],
    )
    engine = _mk_engine(broker)
    intent = _seed_parent_intent_and_leg(engine)
    # Register a partial-exit ExitIntent with its own intent_key that
    # carries the parent's intent_key as its ``from_entry``.
    child = ExitIntent(
        pine_id='exit_tp', from_entry=intent.intent_key, symbol=SYMBOL,
        side='sell', qty=0.5, tp_price=110.0,
        is_partial_qty_bracket=True,
    )
    engine._active_intents[child.intent_key] = child
    engine._order_mapping[child.intent_key] = ['xchg-child']
    engine._envelopes[child.intent_key] = DispatchEnvelope(
        intent=child, run_tag=RUN_TAG, bar_ts_ms=BAR_TS, retry_seq=0,
    )
    from pynecore.core.broker.sync_engine import _CancelTentativeMeta
    engine._cancel_disposition_pending[intent.intent_key] = _CancelTentativeMeta(
        since_ts_ms=BAR_TS, reason='broker_timeout',
    )
    engine._drive_cancel_tentative(now_ms=BAR_TS + 1_000)
    # Parent slot retired.
    assert intent.intent_key not in engine._active_intents
    # Child partial-exit slot also retired across all three maps.
    assert child.intent_key not in engine._active_intents
    assert child.intent_key not in engine._order_mapping
    assert child.intent_key not in engine._envelopes


def __test_leg_from_row_accepts_cancel_tentative_state__():
    """``_leg_from_row`` loads ``cancel_tentative`` rows so restart-replay rebuilds them.

    ``_leg_from_row`` MUST load ``cancel_tentative`` rows so that
    :meth:`SoftwarePartialBracketEngine.restart_replay` rebuilds them
    in the in-memory ledger after a process restart.

    Without this, the post-restart engine would observe the persisted
    tentative rows from
    :func:`iter_active_engine_trigger_partial_legs` but ``_leg_from_row``
    would return ``None`` (the older ``LEG_STATE_ACTIVE``-only gate
    excluded ``cancel_tentative``), so the rehydrate guard
    (``_rehydrate_cancel_tentative_from_replayed_legs``) and the
    ledger-based ``iter_cancel_tentative_parents`` would never see them
    — the ambiguous parent cancel could be reissued or adopted
    incorrectly on the next bar.
    """
    row = OrderRow(
        client_order_id='leg-coid-1',
        plugin_name='test',
        intent_key='exit_tp\x00parent',
        exchange_order_id=None,
        symbol=SYMBOL,
        side='sell',
        qty=1.0,
        filled_qty=0.0,
        state=STATE_PARTIAL_BRACKET_LEG,
        from_entry='parent',
        pine_entry_id='exit_tp',
        sl_level=None,
        tp_level=None,
        trailing_stop=False,
        trailing_distance=None,
        created_ts_ms=BAR_TS,
        updated_ts_ms=BAR_TS,
        closed_ts_ms=None,
        extras={
            EXTRAS_KEY_LEG_KIND: LEG_KIND_TP_PARTIAL,
            EXTRAS_KEY_LEG_STATE: LEG_STATE_CANCEL_TENTATIVE,
            EXTRAS_KEY_PARENT_PINE_ENTRY_ID: 'parent',
            EXTRAS_KEY_PARENT_ENTRY_DISPATCH_REF: 'parent-coid',
            EXTRAS_KEY_INTENT_PARTIAL_QTY: 1.0,
            EXTRAS_KEY_TRIGGER_OFFSET: 5.0,
            EXTRAS_KEY_CANCEL_TENTATIVE_SINCE_TS_MS: BAR_TS,
            'cancel_tentative_reason': 'broker_timeout',
        },
    )
    leg = _leg_from_row(row)
    assert leg is not None
    assert leg.leg_state == LEG_STATE_CANCEL_TENTATIVE
    assert leg.from_entry == 'parent'
    assert leg.trigger_offset == 5.0
    assert leg.extras[EXTRAS_KEY_CANCEL_TENTATIVE_SINCE_TS_MS] == BAR_TS


def __test_clear_already_filled_resolves_trigger_level_via_open_trades__():
    """ALREADY_FILLED promotes legs via ``pending_entry`` → ``armed`` so ``trigger_level`` resolves.

    ALREADY_FILLED resolution from the reconcile-retry path must
    promote tentative legs through the normal ``pending_entry`` → ``armed``
    promotion so ``trigger_level`` is resolved from ``trigger_offset``
    against the parent's fill price.

    The bug fixed here: previously
    :meth:`OrderSyncEngine._clear_intent_cancel_disposition_pending`
    restored the legs directly to ``armed`` via
    ``restore_legs_from_cancel_tentative(parent_filled=True)``. For
    pending-parent tentative legs whose trigger came from
    ``profit_ticks`` / ``loss_ticks`` / ``trail_points_ticks``,
    ``trigger_level`` was ``None`` at mark time and was never resolved
    — the leg ended up armed but inert (the WATCH tick skips
    ``trigger_level=None`` legs).
    """
    events: list[BrokerEvent] = []
    broker = _MockBroker(
        cancel_outcomes=[CancelDispositionOutcome.ALREADY_FILLED],
    )
    engine = _mk_engine(broker, events=events)
    intent = _seed_parent_intent_and_leg(engine)
    # Override the seeded leg to carry trigger_offset only — mirrors a
    # pending-parent partial bracket where the absolute level is resolved
    # from ticks at parent-fill time.
    seeded_leg = next(iter(engine._partial_bracket_engine.iter_legs()))
    seeded_leg.trigger_level = None
    seeded_leg.trigger_offset = 5.0
    # Simulate the parent fill having been recorded by a prior event
    # (the reconcile-retry path runs after the broker confirms the
    # cancel as ALREADY_FILLED — by that point the parent fill must
    # have landed for the broker to report this outcome).
    parent_trade = Trade(
        size=1.0,  # long, +1.0 contracts
        entry_id=intent.intent_key,
        entry_bar_index=0,
        entry_time=BAR_TS,
        entry_price=100.0,  # parent fill price
        commission=0.0,
    )
    engine._position.open_trades.append(parent_trade)
    # Enter cancel-tentative.
    from pynecore.core.broker.sync_engine import _CancelTentativeMeta
    engine._cancel_disposition_pending[intent.intent_key] = _CancelTentativeMeta(
        since_ts_ms=BAR_TS, reason='broker_timeout',
    )
    engine._partial_bracket_engine.mark_legs_cancel_tentative(
        intent.intent_key, reason='broker_timeout', now_ms=BAR_TS,
    )
    assert seeded_leg.leg_state == LEG_STATE_CANCEL_TENTATIVE
    # Drive the cancel-retry loop — the mock broker returns ALREADY_FILLED.
    engine._drive_cancel_tentative(now_ms=BAR_TS + 1_000)
    # Leg is now armed AND its trigger_level was resolved from
    # trigger_offset against the parent's entry_price.
    # TP leg, long parent (side='sell'): trigger_level = fill + offset = 105.0.
    assert seeded_leg.leg_state == LEG_STATE_ARMED
    assert seeded_leg.trigger_level == 105.0
    # Shadow map cleared, resolved event emitted.
    assert engine._cancel_disposition_pending == {}
    resolved = [e for e in events
                if isinstance(e, PartialBracketCancelTentativeResolvedEvent)]
    assert len(resolved) == 1
    assert resolved[0].outcome is CancelDispositionOutcome.ALREADY_FILLED


@pytest.mark.parametrize(
    ("with_actuator", "leg_kind", "trigger_level"),
    [
        (False, LEG_KIND_TP_PARTIAL, 110.0),
        (False, LEG_KIND_SL_PARTIAL, 90.0),
        (True, LEG_KIND_TP_PARTIAL, 110.0),
        (True, LEG_KIND_SL_PARTIAL, 90.0),
    ],
    ids=["state-only-tp", "state-only-sl", "actuator-tp", "actuator-sl"],
)
def __test_clear_already_filled_awaits_a_confirmation_only_when_one_can_come__(
        with_actuator: bool, leg_kind: str, trigger_level: float,
):
    """The restored parent waits for a broker confirmation only when one can come.

    A confirmation needs a native fail-safe actuator AND an SL / trail leg
    seeding a worst-SL: without either no PUT is ever sent and no
    observation confirms anything, so a ``DEGRADING`` start would escalate
    to ``DEGRADED`` and block every entry on the symbol until the position
    is gone (a Bybit lane after a rotation, 2026-10-18).
    """
    broker = _MockBroker(cancel_outcomes=[CancelDispositionOutcome.ALREADY_FILLED])
    engine = _mk_engine(broker)
    if with_actuator:
        engine.set_native_bracket_dispatcher(lambda _snapshot: None)
    intent = _seed_parent_intent_and_leg(engine, with_leg=False)
    leg = _mk_leg(from_entry=intent.intent_key, leg_kind=leg_kind,
                  trigger_level=trigger_level)
    engine._partial_bracket_engine.register_leg(leg)
    engine._position.open_trades.append(Trade(
        size=1.0, entry_id=intent.intent_key, entry_bar_index=0, entry_time=BAR_TS,
        entry_price=100.0, commission=0.0,
    ))
    engine._cancel_disposition_pending[intent.intent_key] = _CancelTentativeMeta(
        since_ts_ms=BAR_TS, reason='broker_timeout',
    )
    engine._partial_bracket_engine.mark_legs_cancel_tentative(
        intent.intent_key, reason='broker_timeout', now_ms=BAR_TS,
    )
    engine._drive_cancel_tentative(now_ms=BAR_TS + 1_000)
    manager = engine._native_failsafe_manager
    state = manager.get_state(leg.parent_entry_dispatch_ref)
    assert state is not None
    expects_confirmation = with_actuator and leg_kind == LEG_KIND_SL_PARTIAL
    expected = FailsafeHealth.DEGRADING if expects_confirmation else FailsafeHealth.HEALTHY
    assert state.health is expected
    if not expects_confirmation:
        engine.drive_native_failsafe(now_ms=BAR_TS + 3_600_000)
        assert state.health is FailsafeHealth.HEALTHY
        assert not manager.block_new_entry(symbol=SYMBOL, pine_id='next', bar_ts_ms=BAR_TS)


def __test_clear_already_filled_leaves_pending_when_parent_not_in_open_trades__():
    """ALREADY_FILLED keeps legs in ``pending_entry`` while the parent fill is not in open_trades.

    Reconcile-retry ALREADY_FILLED resolution must NOT leave the legs
    in armed-without-trigger_level when the parent fill event has not
    yet been routed. Instead the legs stay in ``pending_entry`` so the
    subsequent FILL event's natural
    :meth:`OrderSyncEngine._promote_pending_partial_bracket_legs` call
    can complete the promotion with the event's ``fill_price``.
    """
    broker = _MockBroker(
        cancel_outcomes=[CancelDispositionOutcome.ALREADY_FILLED],
    )
    engine = _mk_engine(broker)
    intent = _seed_parent_intent_and_leg(engine)
    seeded_leg = next(iter(engine._partial_bracket_engine.iter_legs()))
    seeded_leg.trigger_level = None
    seeded_leg.trigger_offset = 5.0
    # No parent trade in open_trades — simulates the race where the
    # broker reports ALREADY_FILLED but the FILL event has not yet been
    # processed by ``_route_event``.
    assert engine._position.open_trades == []
    from pynecore.core.broker.sync_engine import _CancelTentativeMeta
    engine._cancel_disposition_pending[intent.intent_key] = _CancelTentativeMeta(
        since_ts_ms=BAR_TS, reason='broker_timeout',
    )
    engine._partial_bracket_engine.mark_legs_cancel_tentative(
        intent.intent_key, reason='broker_timeout', now_ms=BAR_TS,
    )
    engine._drive_cancel_tentative(now_ms=BAR_TS + 1_000)
    # Leg restored to pending_entry (NOT armed-with-None-level): the
    # next FILL event will promote it properly.
    assert seeded_leg.leg_state == LEG_STATE_PENDING_ENTRY
    assert seeded_leg.trigger_level is None
    # Shadow map cleared regardless.
    assert engine._cancel_disposition_pending == {}


# === Parent-level restart durability (journal rows) =====================


def __test_cancel_tentative_legless_survives_restart_from_journal__(tmp_path):
    """A leg-less parent's unresolved cancel disposition survives a restart.

    An EntryIntent WITHOUT partial-bracket legs enters cancel-tentative
    on an ambiguous cancel: the leg flip inside the mark is a no-op, so
    the durable ``dispatch_kind='cancel_tentative'`` journal row is the
    only restart-persistent trace. The row must be the atomically
    CONVERTED forced-cancel pre-park (one row, kind flipped — no
    delete+insert window, no forced-cancel residue), and a restart must
    re-arm the shadow map from it with the ORIGINAL stale-grace anchor
    so the retry loop resolves the disposition instead of the
    obligation silently vanishing.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(
            _run_identity(), script_source="src", script_path="t027.py",
        )
        broker = _MockBroker(
            raise_on_cancel=[OrderDispositionUnknownError(
                "cancel timed out", client_order_id='xchg-parent',
            )],
        )
        engine = _mk_engine(broker, store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        engine._dispatch_cancel(intent)
        meta = engine._cancel_disposition_pending[intent.intent_key]
        assert meta.park_coid is not None
        # Atomic ownership transfer: exactly ONE journal row for the key,
        # already 'cancel_tentative' — the speculative forced-cancel
        # pre-park was converted in place, not deleted and re-inserted.
        _, pending = ctx.replay()
        rows = [r for r in pending.values() if r.key == intent.intent_key]
        assert len(rows) == 1
        assert rows[0].dispatch_kind == 'cancel_tentative'
        assert rows[0].coid == meta.park_coid
        assert rows[0].parked_ts_ms == meta.since_ts_ms
        assert rows[0].order_ids == ['xchg-parent']
        # Cancel-tentative owns the ambiguity — no forced-cancel double-drive.
        assert engine._forced_cancel_pending == {}
        ctx.close()

        # --- Restart: no legs exist, the journal row alone re-arms. ---
        ctx2 = store.open_run(
            _run_identity(), script_source="src", script_path="t027.py",
        )
        broker2 = _MockBroker(
            cancel_outcomes=[CancelDispositionOutcome.CANCEL_CONFIRMED],
        )
        engine2 = _mk_engine(broker2, store_ctx=ctx2)
        assert intent.intent_key in engine2._cancel_disposition_pending
        meta2 = engine2._cancel_disposition_pending[intent.intent_key]
        assert meta2.since_ts_ms == meta.since_ts_ms  # original anchor, no slip
        assert meta2.park_coid == meta.park_coid
        # Event routing re-seeded from the row's order_ids snapshot.
        assert engine2._order_mapping.get(intent.intent_key) == ['xchg-parent']
        assert engine2._forced_cancel_pending == {}  # no phantom forced cancel
        # The retry loop resolves the disposition; the confirm-cancel
        # teardown purges the journal row (record_complete by key).
        engine2._drive_cancel_tentative(now_ms=meta.since_ts_ms + 1_000)
        assert engine2._cancel_disposition_pending == {}
        assert engine2.quarantined is False
        assert len(broker2.cancel_with_outcome_calls) == 1
        _, pending2 = ctx2.replay()
        assert pending2 == {}


def _seed_parent_sell_intent(
        engine: OrderSyncEngine, *, intent_key: str = 'parent', qty: float = 5.0,
) -> EntryIntent:
    """Seed a SELL parent EntryIntent as if it had been dispatched (active slot,
    mapping, envelope). Mirrors :func:`_seed_parent_intent_and_leg` but on the
    sell side and without registering a leg — the short-gate reservation only
    depends on the parent entry."""
    intent = EntryIntent(
        pine_id=intent_key, symbol=SYMBOL, side='sell', qty=qty,
        order_type=OrderType.MARKET,
    )
    engine._active_intents[intent.intent_key] = intent
    engine._order_mapping[intent.intent_key] = ['xchg-parent']
    engine._envelopes[intent.intent_key] = DispatchEnvelope(
        intent=intent, run_tag=RUN_TAG, bar_ts_ms=BAR_TS,
    )
    engine._current_bar_ts_ms = BAR_TS
    return intent


def __test_short_gate_counts_in_session_cancel_tentative_sell__():
    """F3 regression (in-session): a SOFTWARE partial-bracket parent SELL whose
    cancel did not land is moved into ``_cancel_disposition_pending`` — out of
    ``_active_intents`` — while the cancel branch deliberately KEEPS its
    envelope and the order rests live at the broker.
    ``_parked_working_sell_qty`` must fold the preserved-envelope working
    residual into the short-gate reservation; pre-fix the aggregation ignored
    ``_cancel_disposition_pending`` entirely, so a different sell was admitted
    against the same inventory and both could oversell."""
    broker = _MockBroker()  # spot default: short_selling UNSUPPORTED
    engine = _mk_engine(broker)
    intent = EntryIntent(
        pine_id='P', symbol=SYMBOL, side='sell', qty=5.0,
        order_type=OrderType.MARKET,
    )
    # The diff loop popped 'P' from _active_intents; the cancel branch marked it
    # cancel-tentative and preserved the envelope (never added to active here).
    engine._envelopes['P'] = DispatchEnvelope(
        intent=intent, run_tag=RUN_TAG, bar_ts_ms=BAR_TS,
    )
    engine._cancel_disposition_pending['P'] = _CancelTentativeMeta(
        since_ts_ms=BAR_TS, reason='parent_cancel_did_not_land',
    )
    assert 'P' not in engine._active_intents
    # Pre-fix this returned 0.0 (only _forced_cancel_pending was scanned).
    assert engine._parked_working_sell_qty() == 5.0


def __test_short_gate_recovers_restart_stranded_cancel_tentative_sell__(tmp_path):
    """F3 regression (restart-stranded): a cancel-tentative parent SELL re-armed
    from the journal has NO in-memory envelope, but its resting order is still
    live at the broker. The short-gate reservation recovers the working residual
    from the authoritative ``orders`` table (side + qty + filled_qty) so a
    different sell cannot be admitted against the same inventory."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(
            _run_identity(), script_source="src", script_path="t027.py",
        )
        # A real plugin persisted the parent SELL entry order row; seed it so the
        # recovery has an authoritative residual to read after the restart.
        ctx.upsert_order(
            'xchg-parent', symbol=SYMBOL, side='sell', qty=5.0,
            state='confirmed', intent_key='parent', filled_qty=0.0,
        )
        broker = _MockBroker(
            raise_on_cancel=[OrderDispositionUnknownError(
                "cancel timed out", client_order_id='xchg-parent',
            )],
        )
        engine = _mk_engine(broker, store_ctx=ctx)
        intent = _seed_parent_sell_intent(engine)
        engine._dispatch_cancel(intent)  # -> cancel-tentative + journal row
        assert intent.intent_key in engine._cancel_disposition_pending
        ctx.close()

        # Restart: the journal re-arms _cancel_disposition_pending, but the
        # in-memory envelope did not survive.
        ctx2 = store.open_run(
            _run_identity(), script_source="src", script_path="t027.py",
        )
        broker2 = _MockBroker()
        engine2 = _mk_engine(broker2, store_ctx=ctx2)
        assert 'parent' in engine2._cancel_disposition_pending
        assert 'parent' not in engine2._envelopes
        # Pre-fix: no scan of _cancel_disposition_pending and no envelope to read
        # -> 0.0. The orders-table recovery now reads the live resting sell.
        assert engine2._parked_working_sell_qty() == 5.0


def __test_cancel_tentative_already_filled_purges_journal_row__(tmp_path):
    """ALREADY_FILLED deletes the journal row without touching the envelope.

    The ALREADY_FILLED resolution RETAINS the envelope (the parent is
    live and filled), so the wholesale ``record_complete`` purge is not
    available — the targeted ``record_unpark`` via ``meta.park_coid``
    must delete the ``cancel_tentative`` row, and a subsequent restart
    must NOT phantom-re-arm the already-resolved disposition.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(
            _run_identity(), script_source="src", script_path="t027.py",
        )
        broker = _MockBroker(
            raise_on_cancel=[OrderDispositionUnknownError(
                "cancel timed out", client_order_id='xchg-parent',
            )],
            cancel_outcomes=[CancelDispositionOutcome.ALREADY_FILLED],
        )
        engine = _mk_engine(broker, store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        engine._dispatch_cancel(intent)
        meta = engine._cancel_disposition_pending[intent.intent_key]
        engine._drive_cancel_tentative(now_ms=meta.since_ts_ms + 1_000)
        assert engine._cancel_disposition_pending == {}
        # Envelope retained (parent filled) — only the row was deleted.
        assert intent.intent_key in engine._envelopes
        _, pending = ctx.replay()
        assert pending == {}
        ctx.close()

        ctx2 = store.open_run(
            _run_identity(), script_source="src", script_path="t027.py",
        )
        engine2 = _mk_engine(_MockBroker(), store_ctx=ctx2)
        assert engine2._cancel_disposition_pending == {}
        assert engine2._forced_cancel_pending == {}


def __test_filled_and_closed_resolves_without_restore_booking_or_halt__():
    """FILLED_AND_CLOSED resolves the tentative like a confirmed cancel.

    The order filled before the cancel and its position has closed since,
    both booked already: the resolution restores nothing (no pending or
    armed leg, no fail-safe registration), books no fill, fires no OCA
    cascade, retires the parent's and its partial exits' state, and the
    expired stale grace never halts the bot.
    """
    events: list[BrokerEvent] = []
    broker = _MockBroker(
        raise_on_cancel=[OrderDispositionUnknownError(
            "cancel timed out", client_order_id='xchg-parent',
        )],
        cancel_outcomes=[CancelDispositionOutcome.FILLED_AND_CLOSED],
    )
    engine = _mk_engine(broker, events=events, cancel_tentative_stale_grace_s=1.0)
    intent = _seed_parent_intent_and_leg(engine)
    child = ExitIntent(
        pine_id='exit_tp', from_entry=intent.intent_key, symbol=SYMBOL,
        side='sell', qty=0.5, tp_price=110.0, is_partial_qty_bracket=True,
    )
    engine._active_intents[child.intent_key] = child
    engine._order_mapping[child.intent_key] = ['xchg-child']
    cascades: list[str] = []
    engine._cascade_oca_cancel_for_key = cascades.append  # type: ignore[method-assign]
    closed_before = len(engine._position.closed_trades)
    engine._dispatch_cancel(intent)
    meta = engine._cancel_disposition_pending[intent.intent_key]
    legs = list(engine._partial_bracket_engine.iter_legs())
    assert [leg.leg_state for leg in legs] == [LEG_STATE_CANCEL_TENTATIVE]

    engine._drive_cancel_tentative(now_ms=meta.since_ts_ms + 5_000)
    engine._drive_cancel_tentative(now_ms=meta.since_ts_ms + 6_000)

    assert engine.quarantined is False
    assert len(broker.cancel_with_outcome_calls) == 1
    assert engine._cancel_disposition_pending == {}
    assert [leg.leg_state for leg in legs] == [LEG_STATE_ABORTED_PARENT_GONE]
    assert list(engine._partial_bracket_engine.iter_legs()) == []
    assert engine._native_failsafe_manager.get_state(legs[0].parent_entry_dispatch_ref) is None
    for key in (intent.intent_key, child.intent_key):
        assert key not in engine._order_mapping
        assert key not in engine._envelopes
        assert key not in engine._active_intents
    assert engine._position.open_trades == []
    assert len(engine._position.closed_trades) == closed_before
    assert cascades == []
    resolved = [e for e in events
                if isinstance(e, PartialBracketCancelTentativeResolvedEvent)]
    assert [(e.outcome, e.via) for e in resolved] == [
        (CancelDispositionOutcome.FILLED_AND_CLOSED, 'reconcile_retry'),
    ]
    assert not [e for e in events
                if isinstance(e, PartialBracketCancelTentativeDegradedEvent)]


@pytest.mark.parametrize('outcome,leg_state', [
    (CancelDispositionOutcome.CANCEL_CONFIRMED, LEG_STATE_ABORTED_PARENT_NEVER_ARRIVED),
    (CancelDispositionOutcome.FILLED_AND_CLOSED, LEG_STATE_ABORTED_PARENT_GONE),
    (CancelDispositionOutcome.ALREADY_FILLED, LEG_STATE_PENDING_ENTRY),
])
def __test_a_resolution_a_leg_write_failure_ended_is_finished_with_its_verdict__(
        monkeypatch, outcome, leg_state):
    """A storage failure while a resolution writes its legs never strands the parent.

    The parent stays pending with the verdict it got, its leg as the store
    holds it, and the next cancel-retry pass finishes the same resolution
    without probing the venue again.
    """
    events: list[BrokerEvent] = []
    broker = _MockBroker(
        raise_on_cancel=[OrderDispositionUnknownError(
            "cancel timed out", client_order_id='xchg-parent',
        )],
        cancel_outcomes=[outcome],
    )
    engine = _mk_engine(broker, events=events)
    intent = _seed_parent_intent_and_leg(engine)
    engine._dispatch_cancel(intent)
    meta = engine._cancel_disposition_pending[intent.intent_key]
    (leg,) = list(engine._partial_bracket_engine.iter_legs())
    written: list[str] = []
    failures = [sqlite3.OperationalError("disk I/O error")]

    def _write(_store_ctx, *, new_leg_state, **_kwargs):
        if failures:
            raise failures.pop()
        written.append(new_leg_state)

    engine._partial_bracket_engine._store_ctx = object()  # type: ignore[assignment]
    monkeypatch.setattr(spbe_mod, 'update_engine_trigger_partial_leg_state', _write)

    with pytest.raises(sqlite3.OperationalError):
        engine._drive_cancel_tentative(now_ms=meta.since_ts_ms + 1_000)
    assert engine._cancel_disposition_pending[intent.intent_key] is meta
    assert leg.leg_state == LEG_STATE_CANCEL_TENTATIVE
    assert written == []

    engine._drive_cancel_tentative(now_ms=meta.since_ts_ms + 2_000)
    assert len(broker.cancel_with_outcome_calls) == 1
    assert engine._cancel_disposition_pending == {}
    assert written == [leg_state]
    assert leg.leg_state == leg_state
    assert (intent.intent_key in engine._envelopes) is (
        outcome is CancelDispositionOutcome.ALREADY_FILLED)
    resolved = [e for e in events
                if isinstance(e, PartialBracketCancelTentativeResolvedEvent)]
    assert [e.outcome for e in resolved] == [outcome]


def __test_filled_and_closed_after_a_restart_is_not_re_armed__(tmp_path):
    """A restart resolving FILLED_AND_CLOSED retires the durable row for good.

    The cancel was left unsettled by the earlier process, which also booked
    the order's fill and its close. The restarted run probes once with the
    stale grace long expired, resolves without a halt and drops the
    ``cancel_tentative`` row, so a further restart re-arms nothing and
    probes nothing.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(
            _run_identity(), script_source="src", script_path="t027.py",
        )
        broker = _MockBroker(
            raise_on_cancel=[OrderDispositionUnknownError(
                "cancel timed out", client_order_id='xchg-parent',
            )],
        )
        engine = _mk_engine(broker, store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        engine._dispatch_cancel(intent)
        meta = engine._cancel_disposition_pending[intent.intent_key]
        ctx.close()

        ctx2 = store.open_run(
            _run_identity(), script_source="src", script_path="t027.py",
        )
        broker2 = _MockBroker(
            cancel_outcomes=[CancelDispositionOutcome.FILLED_AND_CLOSED],
        )
        engine2 = _mk_engine(broker2, store_ctx=ctx2, cancel_tentative_stale_grace_s=1.0)
        assert intent.intent_key in engine2._cancel_disposition_pending
        engine2._drive_cancel_tentative(now_ms=meta.since_ts_ms + 1_200_000)
        engine2._drive_cancel_tentative(now_ms=meta.since_ts_ms + 1_201_000)
        assert engine2.quarantined is False
        assert engine2._cancel_disposition_pending == {}
        assert intent.intent_key not in engine2._order_mapping
        assert intent.intent_key not in engine2._envelopes
        assert len(broker2.cancel_with_outcome_calls) == 1
        _, pending2 = ctx2.replay()
        assert pending2 == {}
        ctx2.close()

        ctx3 = store.open_run(
            _run_identity(), script_source="src", script_path="t027.py",
        )
        broker3 = _MockBroker()
        engine3 = _mk_engine(broker3, store_ctx=ctx3, cancel_tentative_stale_grace_s=1.0)
        assert engine3._cancel_disposition_pending == {}
        assert engine3._forced_cancel_pending == {}
        assert intent.intent_key not in engine3._order_mapping
        engine3._drive_cancel_tentative(now_ms=meta.since_ts_ms + 1_300_000)
        assert broker3.cancel_with_outcome_calls == []
        assert engine3.quarantined is False


def __test_cancel_tentative_stale_grace_degrades_again_after_restart__(tmp_path):
    """Stale-grace expiry keeps the journal row: a restart probes, then degrades again.

    Expiry degrades the entry (new entries blocked) and deliberately KEEPS
    the ``cancel_tentative`` row. A restart re-arms the entry with the
    original (already expired) anchor; the retry loop must then probe
    the venue at least once IN THE NEW SESSION before it may degrade again —
    a definite outcome resolves the entry, only a persistent ``UNKNOWN``
    degrades it again. Deleting the row would let a leg-less parent's
    unresolved disposition silently vanish across the restart instead.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(
            _run_identity(), script_source="src", script_path="t027.py",
        )
        broker = _MockBroker(
            raise_on_cancel=[OrderDispositionUnknownError(
                "cancel timed out", client_order_id='xchg-parent',
            )],
        )
        engine = _mk_engine(
            broker, store_ctx=ctx, cancel_tentative_stale_grace_s=1.0,
        )
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        engine._dispatch_cancel(intent)
        meta = engine._cancel_disposition_pending[intent.intent_key]
        # First drive past the grace: probes once (UNKNOWN), not degraded yet.
        engine._drive_cancel_tentative(now_ms=meta.since_ts_ms + 5_000)
        assert engine.quarantined is False
        assert len(broker.cancel_with_outcome_calls) == 1
        # Second drive: retry already ran, grace still expired — degraded.
        engine._drive_cancel_tentative(now_ms=meta.since_ts_ms + 6_000)
        assert _degraded(engine, intent.intent_key)
        # The obligation must not vanish across a restart: row kept.
        _, pending = ctx.replay()
        kinds = [r.dispatch_kind for r in pending.values()]
        assert kinds == ['cancel_tentative']
        ctx.close()

        ctx2 = store.open_run(
            _run_identity(), script_source="src", script_path="t027.py",
        )
        broker2 = _MockBroker()
        engine2 = _mk_engine(
            broker2, store_ctx=ctx2, cancel_tentative_stale_grace_s=1.0,
        )
        assert intent.intent_key in engine2._cancel_disposition_pending
        # Fresh session: the expired anchor alone must NOT degrade — the
        # venue is probed once first (persistent UNKNOWN here).
        engine2._drive_cancel_tentative(now_ms=meta.since_ts_ms + 10_000)
        assert engine2.quarantined is False
        assert len(broker2.cancel_with_outcome_calls) == 1
        engine2._drive_cancel_tentative(now_ms=meta.since_ts_ms + 11_000)
        assert _degraded(engine2, intent.intent_key)


def __test_stale_anchor_restart_probes_before_degrading__(tmp_path):
    """Restart with a long-expired anchor resolves via the venue, no entry block.

    Regression for the bybit-spot cycle 18-66 restart churn (measured
    2026-08-22): a rotation shutdown stranded a cancel-tentative parent,
    and every restarted run re-armed the entry with the original anchor,
    saw the 10s grace long expired with ``retry_count == 0`` and promoted
    straight to the degraded state — no venue probe ever ran, so the loop
    could never terminate even though ``execute_cancel_with_outcome``
    resolves the disposition authoritatively on the first attempt.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(
            _run_identity(), script_source="src", script_path="t027.py",
        )
        broker = _MockBroker(
            raise_on_cancel=[OrderDispositionUnknownError(
                "cancel timed out", client_order_id='xchg-parent',
            )],
        )
        engine = _mk_engine(broker, store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        engine._dispatch_cancel(intent)
        meta = engine._cancel_disposition_pending[intent.intent_key]
        ctx.close()

        # Restart 20 minutes later — far beyond the 10s stale grace.
        ctx2 = store.open_run(
            _run_identity(), script_source="src", script_path="t027.py",
        )
        broker2 = _MockBroker(
            cancel_outcomes=[CancelDispositionOutcome.CANCEL_CONFIRMED],
        )
        engine2 = _mk_engine(broker2, store_ctx=ctx2)
        assert intent.intent_key in engine2._cancel_disposition_pending
        engine2._drive_cancel_tentative(now_ms=meta.since_ts_ms + 1_200_000)
        assert engine2.quarantined is False
        assert engine2._cancel_disposition_pending == {}
        assert len(broker2.cancel_with_outcome_calls) == 1
        # The resolved obligation purged its journal row.
        _, pending2 = ctx2.replay()
        assert pending2 == {}


# === Pre-park crash-window durability (cancel_probe rows) ================


def __test_cancel_probe_crash_window_rehydrates_via_outcome_not_bool_retry__(
        tmp_path,
):
    """A crash before the disposition lands survives as an OUTCOME-driven
    rehydrate, not a bool-only forced-cancel retry.

    The pre-park of a SOFTWARE partial-bracket parent-entry cancel writes a
    ``dispatch_kind='cancel_probe'`` row BEFORE the broker round-trip. If the
    process dies before the disposition lands (here: the cancel times out,
    then the mark that would convert the row crashes), the ``cancel_probe``
    row is the only durable trace. A restart MUST re-arm it through the
    cancel-tentative machine (``execute_cancel_with_outcome``, which tells an
    ALREADY_FILLED parent from a confirmed cancel), NOT the bool-only
    ``_retry_forced_cancels`` — the latter reads the benign no-op ``True`` of
    a parent that filled during the window as a confirmed cancel and retires
    the live position's partial protection. Driving ALREADY_FILLED after the
    restart must RESTORE the protection (envelope kept), not retire it.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(
            _run_identity(), script_source="src", script_path="t027.py",
        )
        broker = _MockBroker(
            raise_on_cancel=[OrderDispositionUnknownError(
                "cancel timed out", client_order_id='xchg-parent',
            )],
        )
        engine = _mk_engine(broker, store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)

        # Crash the mark that would convert the pre-park to cancel_tentative,
        # simulating a process death inside the disposition-unknown window.
        def _crash_mark(*_args, **_kwargs):
            raise RuntimeError("simulated crash before cancel-tentative mark")
        engine._mark_intent_cancel_disposition_pending = (  # type: ignore[method-assign]
            _crash_mark
        )
        with pytest.raises(RuntimeError, match="simulated crash"):
            engine._dispatch_cancel(intent)

        # The stranded durable trace is a single ``cancel_probe`` row (the
        # speculative pre-park) — NOT a resolved cancel_tentative, NOT a
        # forced_cancel.
        _, pending = ctx.replay()
        rows = [r for r in pending.values() if r.key == intent.intent_key]
        assert len(rows) == 1
        assert rows[0].dispatch_kind == 'cancel_probe'
        assert rows[0].parked_ts_ms > 0
        assert rows[0].order_ids == ['xchg-parent']
        ctx.close()

        # --- Restart: the probe row re-arms the OUTCOME-based machine. ---
        ctx2 = store.open_run(
            _run_identity(), script_source="src", script_path="t027.py",
        )
        broker2 = _MockBroker(
            cancel_outcomes=[CancelDispositionOutcome.ALREADY_FILLED],
        )
        engine2 = _mk_engine(broker2, store_ctx=ctx2)
        # Rehydrated into cancel-tentative, NOT the bool-only forced-cancel
        # retry that cannot tell ALREADY_FILLED from a confirmed cancel.
        assert intent.intent_key in engine2._cancel_disposition_pending
        assert engine2._forced_cancel_pending == {}
        meta2 = engine2._cancel_disposition_pending[intent.intent_key]
        assert meta2.since_ts_ms == rows[0].parked_ts_ms  # original probe anchor
        assert meta2.park_coid == rows[0].coid
        # Event routing re-seeded from the row's order_ids snapshot.
        assert engine2._order_mapping.get(intent.intent_key) == ['xchg-parent']

        # Drive the retry: the broker reports the parent filled during the
        # window. Resolution went through the OUTCOME path
        # (``execute_cancel_with_outcome``), NOT the bool-only strict retry
        # (``execute_cancel``) that would have retired the protection on a
        # benign no-op ``True``.
        engine2._drive_cancel_tentative(now_ms=meta2.since_ts_ms + 1_000)
        assert len(broker2.cancel_with_outcome_calls) == 1
        assert broker2.cancel_calls == []  # never took the bool-only retry
        assert engine2._cancel_disposition_pending == {}
        assert engine2.quarantined is False
        # Row purged by the ALREADY_FILLED targeted record_unpark.
        _, pending2 = ctx2.replay()
        assert pending2 == {}


def __test_cancel_probe_rehydrate_flips_pending_entry_legs__():
    """The leg-side rehydrate flips a re-armed parent's still-pending legs.

    A ``cancel_probe`` row re-arms the shadow map from the journal in
    ``__init__`` — before ``restart_replay`` loads the leg ledger — and its
    legs were never flipped to ``cancel_tentative`` (the mark never ran). The
    leg-side rehydrate must flip any ``pending_entry`` leg under a re-armed
    parent so a later CANCEL_CONFIRMED resolution
    (:meth:`SoftwarePartialBracketEngine.confirm_cancel_tentative`, which acts
    only on cancel_tentative legs) does not orphan it. The anchor comes from
    the shadow map, keeping the persisted stale-grace deadline in one epoch.
    """
    broker = _MockBroker()
    engine = _mk_engine(broker)
    intent = _seed_parent_intent_and_leg(engine, with_leg=True)
    # Emulate the __init__ absorption of a cancel_probe row: the shadow map
    # is re-armed, but the leg is still pending_entry (the mark never flipped
    # it, and this parent was NOT re-armed from a cancel_tentative leg row).
    engine._cancel_disposition_pending[intent.intent_key] = _CancelTentativeMeta(
        since_ts_ms=BAR_TS,
        reason='rehydrated_from_probe',
    )
    legs = list(engine._partial_bracket_engine.iter_legs())
    assert len(legs) == 1
    assert legs[0].leg_state == LEG_STATE_PENDING_ENTRY  # precondition

    engine._rehydrate_cancel_tentative_from_replayed_legs()

    legs = list(engine._partial_bracket_engine.iter_legs())
    assert legs[0].leg_state == LEG_STATE_CANCEL_TENTATIVE
    assert legs[0].extras.get(EXTRAS_KEY_CANCEL_TENTATIVE_SINCE_TS_MS) == BAR_TS


def __test_dispatch_cancel_false_on_software_parent_enters_cancel_tentative__():
    """A ``False`` cancel on a SOFTWARE parent resolves via the OUTCOME machine.

    ``execute_cancel`` returning ``False`` (its "cancel did not land, still
    pending" signal) for a SOFTWARE partial-bracket parent entry must route the
    key into the OUTCOME-based cancel-tentative machine — NOT the bool-only
    forced-cancel retry. A ``False`` proves the parent live only at THIS
    instant; a resting parent can fill before a later retry, whose bool-only
    ``execute_cancel`` cannot distinguish a confirmed cancel from the benign
    no-op ``True`` a filled parent returns. Routing to cancel-tentative lets a
    subsequent ``execute_cancel_with_outcome`` see ALREADY_FILLED and RESTORE
    the live position's partial protection instead of retiring it.
    """
    broker = _MockBroker(cancel_returns=[False])
    engine = _mk_engine(broker)
    intent = _seed_parent_intent_and_leg(engine, with_leg=True)

    engine._dispatch_cancel(intent)

    # Cancel-tentative owns the ambiguity; the bool-only forced retry does NOT.
    assert intent.intent_key in engine._cancel_disposition_pending
    assert intent.intent_key not in engine._forced_cancel_pending
    # The PENDING_ENTRY leg was flipped to CANCEL_TENTATIVE and the parent's
    # mapping / envelope are retained for the outcome-based retry loop.
    legs = list(engine._partial_bracket_engine.iter_legs())
    assert len(legs) == 1
    assert legs[0].leg_state == LEG_STATE_CANCEL_TENTATIVE
    assert engine._order_mapping.get(intent.intent_key) == ['xchg-parent']
    assert engine._envelopes.get(intent.intent_key) is not None

    # The parent then fills: the OUTCOME-based retry sees ALREADY_FILLED and
    # RESTORES the partial protection (leg preserved as PENDING_ENTRY, re-armed
    # with the fill price by ``on_parent_entry_filled``). The pre-fix bool-only
    # path would instead have ABORTED the leg on a benign no-op ``True``,
    # stripping the now-live position's protection.
    broker.cancel_outcomes = [CancelDispositionOutcome.ALREADY_FILLED]
    engine._drive_cancel_tentative(now_ms=BAR_TS + 1_000)

    assert len(broker.cancel_with_outcome_calls) == 1  # outcome-driven, not bool
    assert intent.intent_key not in engine._cancel_disposition_pending
    assert engine.quarantined is False
    legs = list(engine._partial_bracket_engine.iter_legs())
    assert len(legs) == 1
    # Preserved, NOT ``aborted_parent_never_arrived`` — the protection survives.
    assert legs[0].leg_state == LEG_STATE_PENDING_ENTRY


def _ct_order_event(
        event_type: str, *, filled_qty: float, fill_qty: float | None,
        fill_id: str | None = None,
):
    """An event of the tentative parent ``xchg-parent`` (a 1-unit buy)."""
    from pynecore.core.broker.models import (
        ExchangeOrder, LegType, OrderEvent, OrderStatus,
    )
    status = {
        'cancelled': OrderStatus.CANCELLED, 'partial': OrderStatus.PARTIALLY_FILLED,
    }[event_type]
    order = ExchangeOrder(
        id='xchg-parent', symbol=SYMBOL, side='buy', order_type=OrderType.MARKET,
        qty=1.0, filled_qty=filled_qty, remaining_qty=1.0 - filled_qty,
        price=None, stop_price=None,
        average_fill_price=100.0 if filled_qty > 0.0 else None,
        status=status, timestamp=BAR_TS / 1000.0, fee=0.0, fee_currency='USDT',
    )
    return OrderEvent(
        order=order, event_type=event_type,
        fill_price=100.0 if fill_qty else None, fill_qty=fill_qty,
        timestamp=BAR_TS / 1000.0, pine_id='parent', leg_type=LegType.ENTRY,
        fill_id=fill_id, journal_fill_qty=fill_qty,
    )


@pytest.mark.parametrize('order', ['cancel_first', 'fill_first'])
def __test_a_cancelled_partly_executed_tentative_parent_keeps_its_legs__(order):
    """The cancel of a partly executed tentative parent keeps the legs for what filled.

    The venue cancels the unfilled remainder of an order that executed 0.4:
    whichever of the cancellation and the execution it delivers first, the
    partial-bracket leg ends armed against the booked 0.4, never as
    ``parent_never_arrived``.
    """
    broker = _MockBroker()
    engine = _mk_engine(broker)
    intent = _seed_parent_intent_and_leg(engine)
    leg = next(iter(engine._partial_bracket_engine.iter_legs()))
    # The leg was dispatched against the live parent envelope.
    leg.parent_entry_dispatch_ref = sorted(engine._parent_dispatch_refs('parent'))[0]
    engine._cancel_disposition_pending[intent.intent_key] = _CancelTentativeMeta(
        since_ts_ms=BAR_TS, reason='broker_timeout',
    )
    engine._partial_bracket_engine.mark_legs_cancel_tentative(
        intent.intent_key, reason='broker_timeout', now_ms=BAR_TS,
    )
    cancelled = _ct_order_event('cancelled', filled_qty=0.4, fill_qty=None)
    fill = _ct_order_event('partial', filled_qty=0.4, fill_qty=0.4, fill_id='x-1')
    sequence = [cancelled, fill] if order == 'cancel_first' else [fill, cancelled]

    for event in sequence:
        engine._route_event(event)

    assert engine._cancel_disposition_pending == {}
    assert engine._position.size == pytest.approx(0.4)
    assert leg.leg_state == LEG_STATE_ARMED
    assert engine._partial_bracket_engine.get_leg(leg.key) is leg
    assert intent.intent_key not in engine._order_mapping
    assert intent.intent_key in engine._envelopes
    assert engine.quarantined is False


def __test_a_retried_partly_executed_cancel_resolution_drops_the_dead_mapping__():
    """A storage failure midway through the partial-cancel resolution keeps its whole verdict.

    The cancellation of a partly executed tentative parent resolves as filled
    and drops the dead remainder's mapping. A leg write failing midway leaves
    the verdict for the cancel-retry pass, which finishes it: the mapping is
    gone (the cancelled order is never adopted as a working one), the
    envelope stays, and the execution arriving afterwards arms the leg.
    """
    import sqlite3
    broker = _MockBroker()
    engine = _mk_engine(broker)
    intent = _seed_parent_intent_and_leg(engine)
    leg = next(iter(engine._partial_bracket_engine.iter_legs()))
    leg.parent_entry_dispatch_ref = sorted(engine._parent_dispatch_refs('parent'))[0]
    engine._cancel_disposition_pending[intent.intent_key] = _CancelTentativeMeta(
        since_ts_ms=BAR_TS, reason='broker_timeout',
    )
    engine._partial_bracket_engine.mark_legs_cancel_tentative(
        intent.intent_key, reason='broker_timeout', now_ms=BAR_TS,
    )
    partial_engine = engine._partial_bracket_engine
    real_restore = partial_engine.restore_legs_from_cancel_tentative
    failed: list[str] = []

    def restore_once(*args, **kwargs):
        if not failed:
            failed.append(args[0])
            raise sqlite3.OperationalError("disk I/O error")
        return real_restore(*args, **kwargs)

    partial_engine.restore_legs_from_cancel_tentative = restore_once  # type: ignore[method-assign]
    with pytest.raises(sqlite3.OperationalError):
        engine._route_event(_ct_order_event('cancelled', filled_qty=0.4, fill_qty=None))
    assert failed == [intent.intent_key]
    assert intent.intent_key in engine._cancel_disposition_pending

    engine._drive_cancel_tentative(now_ms=BAR_TS + 1_000)

    assert engine._cancel_disposition_pending == {}
    assert intent.intent_key not in engine._order_mapping
    assert intent.intent_key in engine._envelopes
    assert broker.cancel_with_outcome_calls == []
    engine._route_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.4, fill_id='x-1'))
    assert engine._position.size == pytest.approx(0.4)
    assert leg.leg_state == LEG_STATE_ARMED
    assert engine.quarantined is False


def __test_a_retried_partial_fill_resolution_still_cancels_the_remainder__():
    """The residual cancel of a partial-fill resolution survives a storage failure midway.

    A ``partial`` event of a tentative parent resolves it as filled and ends
    with the one-shot cancel of the unfilled remainder still working at the
    venue. When a leg write fails midway, the cancel-retry pass finishes the
    resolution from the saved verdict and sends that cancel, once.
    """
    import sqlite3
    broker = _MockBroker()
    engine = _mk_engine(broker)
    intent = _seed_parent_intent_and_leg(engine)
    engine._cancel_disposition_pending[intent.intent_key] = _CancelTentativeMeta(
        since_ts_ms=BAR_TS, reason='broker_timeout',
    )
    engine._partial_bracket_engine.mark_legs_cancel_tentative(
        intent.intent_key, reason='broker_timeout', now_ms=BAR_TS,
    )
    partial_engine = engine._partial_bracket_engine
    real_restore = partial_engine.restore_legs_from_cancel_tentative
    failed: list[str] = []

    def restore_once(*args, **kwargs):
        if not failed:
            failed.append(args[0])
            raise sqlite3.OperationalError("disk I/O error")
        return real_restore(*args, **kwargs)

    partial_engine.restore_legs_from_cancel_tentative = restore_once  # type: ignore[method-assign]
    with pytest.raises(sqlite3.OperationalError):
        engine._route_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.4,
                                            fill_id='x-1'))
    assert broker.cancel_calls == []

    engine._drive_cancel_tentative(now_ms=BAR_TS + 1_000)

    assert engine._cancel_disposition_pending == {}
    assert len(broker.cancel_calls) == 1
    assert broker.cancel_with_outcome_calls == []
    engine._drive_cancel_tentative(now_ms=BAR_TS + 2_000)
    assert len(broker.cancel_calls) == 1


def _ct_unknown_execution(event):
    """``event`` with an executed quantity the venue did not report."""
    import dataclasses
    return dataclasses.replace(
        event, order=dataclasses.replace(event.order, filled_qty_known=False))


def __test_a_tentative_cancel_without_execution_evidence_stays_tentative__():
    """A cancellation whose executed quantity is unknown confirms nothing.

    The venue's cancel push of the tentative parent omits the executed
    quantity. The parent stays tentative with its legs and mapping; the
    execution reported afterwards resolves it as filled and arms the leg. The
    same push with a reported zero confirms the cancel
    (:func:`__test_an_unexecuted_tentative_parent_cancel_still_aborts_its_legs__`).
    """
    broker = _MockBroker()
    engine = _mk_engine(broker)
    intent = _seed_parent_intent_and_leg(engine)
    leg = next(iter(engine._partial_bracket_engine.iter_legs()))
    leg.parent_entry_dispatch_ref = sorted(engine._parent_dispatch_refs('parent'))[0]
    engine._cancel_disposition_pending[intent.intent_key] = _CancelTentativeMeta(
        since_ts_ms=BAR_TS, reason='broker_timeout',
    )
    engine._partial_bracket_engine.mark_legs_cancel_tentative(
        intent.intent_key, reason='broker_timeout', now_ms=BAR_TS,
    )

    engine._route_event(_ct_unknown_execution(
        _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))

    assert intent.intent_key in engine._cancel_disposition_pending
    assert leg.leg_state == LEG_STATE_CANCEL_TENTATIVE
    assert engine._order_mapping.get(intent.intent_key) == ['xchg-parent']
    assert intent.intent_key in engine._envelopes

    engine._route_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.4, fill_id='x-1'))

    assert engine._cancel_disposition_pending == {}
    assert engine._position.size == pytest.approx(0.4)
    assert leg.leg_state == LEG_STATE_ARMED
    assert engine.quarantined is False


@pytest.mark.parametrize('terminal', ['cancelled', 'venue_cancelled', 'rejected'])
def __test_an_entry_terminal_without_execution_evidence_keeps_the_entry__(terminal):
    """A terminal event of an entry with an unknown executed quantity retires nothing.

    Neither the legs nor the mapping or the entry intent are dropped; the
    terminal event delivered again with the venue's zero then ends the entry
    and aborts its legs as before.
    """
    import dataclasses
    broker = _MockBroker()
    engine = _mk_engine(broker)
    intent = _seed_parent_intent_and_leg(engine)
    leg = next(iter(engine._partial_bracket_engine.iter_legs()))
    event = _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)
    if terminal == 'venue_cancelled':
        event = dataclasses.replace(event, cancel_reason='venue_rejected')
    elif terminal == 'rejected':
        event = dataclasses.replace(
            event, event_type='rejected',
            order=dataclasses.replace(event.order, status=OrderStatus.REJECTED))

    engine._route_event(_ct_unknown_execution(event))

    assert leg.leg_state == LEG_STATE_PENDING_ENTRY
    assert engine._order_mapping.get(intent.intent_key) == ['xchg-parent']
    assert engine._active_intents.get(intent.intent_key) is not None
    assert intent.intent_key in engine._envelopes
    assert engine.quarantined is False and not engine.quarantined

    engine._route_event(event)

    assert leg.leg_state == LEG_STATE_ABORTED_PARENT_NEVER_ARRIVED
    assert intent.intent_key not in engine._order_mapping


def __test_an_entry_cancel_without_execution_evidence_lets_a_late_fill_arm_the_legs__():
    """The execution reported after an evidence-less cancel still arms the entry's legs."""
    broker = _MockBroker()
    engine = _mk_engine(broker)
    _seed_parent_intent_and_leg(engine)
    leg = next(iter(engine._partial_bracket_engine.iter_legs()))
    leg.parent_entry_dispatch_ref = sorted(engine._parent_dispatch_refs('parent'))[0]

    engine._route_event(_ct_unknown_execution(
        _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))
    engine._route_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.4, fill_id='x-1'))

    assert engine._position.size == pytest.approx(0.4)
    assert leg.leg_state == LEG_STATE_ARMED


@pytest.mark.parametrize('diff', ['removed', 'replaced'])
@pytest.mark.parametrize('evidence', ['fill', 'zero'])
def __test_a_strategy_diff_keeps_an_entry_awaiting_execution_evidence__(diff, evidence):
    """A strategy cancel or amend between an evidence-less terminal event and its evidence.

    The entry's order is terminal at the venue with an unknown executed
    quantity. The next diff drops the entry (or re-places it at another
    quantity): nothing is sent to the venue and the mapping, envelope, intent
    and pending leg stay. A later execution then arms the leg; the venue's
    reported zero instead ends the entry and aborts the leg.
    """
    import dataclasses
    broker = _MockBroker()
    engine = _mk_engine(broker)
    intent = _seed_parent_intent_and_leg(engine)
    leg = next(iter(engine._partial_bracket_engine.iter_legs()))
    leg.parent_entry_dispatch_ref = sorted(engine._parent_dispatch_refs('parent'))[0]
    cancelled = _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)
    engine._route_event(_ct_unknown_execution(cancelled))

    engine._diff_and_dispatch(
        [] if diff == 'removed' else [dataclasses.replace(intent, qty=2.0)])

    assert broker.cancel_calls == [] and broker.cancel_with_outcome_calls == []
    assert engine._active_intents.get(intent.intent_key) == intent
    assert engine._order_mapping.get(intent.intent_key) == ['xchg-parent']
    assert intent.intent_key in engine._envelopes
    assert leg.leg_state == LEG_STATE_PENDING_ENTRY
    assert engine._forced_cancel_pending == {} and engine._cancel_disposition_pending == {}

    if evidence == 'fill':
        engine._route_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.4,
                                            fill_id='x-1'))
        assert engine._position.size == pytest.approx(0.4)
        assert leg.leg_state == LEG_STATE_ARMED
    else:
        engine._route_event(cancelled)
        assert leg.leg_state == LEG_STATE_ABORTED_PARENT_NEVER_ARRIVED
        assert intent.intent_key not in engine._order_mapping
    assert engine._entries_awaiting_fill_evidence == {}


@pytest.mark.parametrize('evidence', ['fill', 'zero'])
def __test_an_oca_cascade_keeps_an_entry_awaiting_execution_evidence__(evidence):
    """An OCA sibling's fill between an evidence-less terminal event and its evidence.

    The entry's order is terminal at the venue with an unknown executed
    quantity when a sibling of its ``cancel`` OCA group fills. The cascade
    sends nothing and keeps the entry's intent, mapping, envelope and pending
    leg, so the evidence that follows is applied to the entry as without the
    cascade: a later execution arms the leg, the venue's reported zero ends
    the entry and aborts the leg.
    """
    import dataclasses
    broker = _MockBroker()
    engine = _mk_engine(broker)
    seeded = _seed_parent_intent_and_leg(engine)
    intent = dataclasses.replace(seeded, oca_name='grp', oca_type='cancel')
    engine._active_intents[intent.intent_key] = intent
    leg = next(iter(engine._partial_bracket_engine.iter_legs()))
    leg.parent_entry_dispatch_ref = sorted(engine._parent_dispatch_refs('parent'))[0]
    cancelled = _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)
    engine._route_event(_ct_unknown_execution(cancelled))
    sibling = EntryIntent(pine_id='sibling', symbol=SYMBOL, side='sell', qty=1.0,
                          order_type=OrderType.MARKET, oca_name='grp', oca_type='cancel')
    engine._active_intents[sibling.intent_key] = sibling

    engine._cascade_oca_cancel_for_key(sibling.intent_key)

    assert broker.cancel_calls == [] and broker.cancel_with_outcome_calls == []
    assert engine._active_intents.get(intent.intent_key) == intent
    assert engine._order_mapping.get(intent.intent_key) == ['xchg-parent']
    assert intent.intent_key in engine._envelopes
    assert leg.leg_state == LEG_STATE_PENDING_ENTRY
    assert engine._forced_cancel_pending == {} and engine._cancel_disposition_pending == {}

    if evidence == 'fill':
        engine._route_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.4,
                                            fill_id='x-1'))
        assert engine._position.size == pytest.approx(0.4)
        assert leg.leg_state == LEG_STATE_ARMED
    else:
        engine._route_event(cancelled)
        assert leg.leg_state == LEG_STATE_ABORTED_PARENT_NEVER_ARRIVED
        assert intent.intent_key not in engine._order_mapping
    assert engine._entries_awaiting_fill_evidence == {}


def __test_a_cancel_of_an_entry_awaiting_execution_evidence_does_not_land__():
    """Both cancel helpers answer ``False`` for an entry held for its execution evidence.

    A caller acting on a landed cancel would retire the entry's state or
    send its replacement; ``False`` keeps both, with no venue round-trip and
    nothing parked.
    """
    broker = _MockBroker()
    engine = _mk_engine(broker)
    intent = _seed_parent_intent_and_leg(engine)
    engine._route_event(_ct_unknown_execution(
        _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))

    assert engine._dispatch_cancel(intent) is False
    assert engine._dispatch_cancel_strict(intent) is False
    assert broker.cancel_calls == [] and broker.cancel_with_outcome_calls == []
    assert engine._forced_cancel_pending == {}
    assert engine._order_mapping.get(intent.intent_key) == ['xchg-parent']
    assert intent.intent_key in engine._envelopes


def __test_a_residual_cancel_that_did_not_land_keeps_the_remainder_mapped__():
    """A ``False`` residual cancel after a partial-fill resolution keeps the remainder tracked.

    ``execute_cancel`` returning ``False`` says the cancel did not land: the
    unfilled remainder still rests at the venue, so its mapping stays and its
    later execution still routes to the entry whose legs protect the position.
    """
    broker = _MockBroker(cancel_returns=[False])
    engine = _mk_engine(broker)
    intent = _seed_parent_intent_and_leg(engine)
    leg = next(iter(engine._partial_bracket_engine.iter_legs()))
    leg.parent_entry_dispatch_ref = sorted(engine._parent_dispatch_refs('parent'))[0]
    engine._cancel_disposition_pending[intent.intent_key] = _CancelTentativeMeta(
        since_ts_ms=BAR_TS, reason='broker_timeout',
    )
    engine._partial_bracket_engine.mark_legs_cancel_tentative(
        intent.intent_key, reason='broker_timeout', now_ms=BAR_TS,
    )

    engine._route_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.4, fill_id='x-1'))

    assert engine._cancel_disposition_pending == {}
    assert len(broker.cancel_calls) == 1
    assert engine._order_mapping.get(intent.intent_key) == ['xchg-parent']
    assert engine._position.size == pytest.approx(0.4)
    assert leg.leg_state == LEG_STATE_ARMED
    assert engine.quarantined is False


def __test_an_unexecuted_tentative_parent_cancel_still_aborts_its_legs__():
    """A cancellation with nothing executed stays a confirmed cancel."""
    broker = _MockBroker()
    engine = _mk_engine(broker)
    intent = _seed_parent_intent_and_leg(engine)
    leg = next(iter(engine._partial_bracket_engine.iter_legs()))
    engine._cancel_disposition_pending[intent.intent_key] = _CancelTentativeMeta(
        since_ts_ms=BAR_TS, reason='broker_timeout',
    )
    engine._partial_bracket_engine.mark_legs_cancel_tentative(
        intent.intent_key, reason='broker_timeout', now_ms=BAR_TS,
    )

    engine._route_event(_ct_order_event('cancelled', filled_qty=0.0, fill_qty=None))

    assert engine._cancel_disposition_pending == {}
    assert leg.leg_state == LEG_STATE_ABORTED_PARENT_NEVER_ARRIVED
    assert intent.intent_key not in engine._envelopes


@pytest.mark.parametrize('cancel_reason', [None, 'venue_rejected'])
def __test_a_cancel_delivered_before_its_execution_keeps_the_entry_legs__(cancel_reason):
    """A cancelled entry that executed 0.4 keeps its legs, the execution arms them.

    No cancel is in flight: the remainder was cancelled by someone else or by
    the venue itself, and the venue pushed the cancellation ahead of the
    execution it reports in ``filled_qty``.
    """
    import dataclasses
    broker = _MockBroker()
    engine = _mk_engine(broker)
    _seed_parent_intent_and_leg(engine)
    leg = next(iter(engine._partial_bracket_engine.iter_legs()))
    leg.parent_entry_dispatch_ref = sorted(engine._parent_dispatch_refs('parent'))[0]
    cancelled = dataclasses.replace(
        _ct_order_event('cancelled', filled_qty=0.4, fill_qty=None),
        cancel_reason=cancel_reason,
    )

    engine._route_event(cancelled)
    assert leg.leg_state == LEG_STATE_PENDING_ENTRY
    engine._route_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.4, fill_id='x-1'))

    assert engine._position.size == pytest.approx(0.4)
    assert leg.leg_state == LEG_STATE_ARMED


@pytest.mark.parametrize('oca_type', ['cancel', 'reduce'])
def __test_an_oca_cascade_over_a_held_entry_is_not_undone_by_its_evidence__(oca_type):
    """The OCA verdict on a held entry survives the venue's later zero.

    A sibling of the held entry's OCA group fills. The cascade sends nothing
    for the held entry, but its Pine order follows the group: a ``cancel``
    group drops it, a ``reduce`` group shrinks it by the sibling's fill. When
    the venue then reports that the held order executed nothing, the entry is
    retired and its Pine order stays as the group left it, so the next sync
    neither sends the cancelled entry again nor its full original size.
    """
    import dataclasses
    broker = _MockBroker()
    engine = _mk_engine(broker)
    seeded = _seed_parent_intent_and_leg(engine)
    intent = dataclasses.replace(seeded, oca_name='grp', oca_type=oca_type)
    engine._active_intents[intent.intent_key] = intent
    pine_order = SimpleNamespace(size=1.0, sign=1.0)
    engine._position.entry_orders['parent'] = pine_order  # type: ignore[assignment]
    cancelled = _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)
    engine._route_event(_ct_unknown_execution(cancelled))
    sibling = EntryIntent(pine_id='sibling', symbol=SYMBOL, side='sell', qty=1.0,
                          order_type=OrderType.MARKET, oca_name='grp', oca_type=oca_type)
    engine._active_intents[sibling.intent_key] = sibling

    if oca_type == 'cancel':
        engine._cascade_oca_cancel_for_key(sibling.intent_key)
        assert 'parent' not in engine._position.entry_orders
    else:
        sibling_fill = dataclasses.replace(
            _ct_order_event('partial', filled_qty=0.4, fill_qty=0.4, fill_id='s-1'),
            pine_id='sibling')
        engine._cascade_oca_reduce(sibling_fill)
        assert engine._position.entry_orders['parent'] is pine_order
        assert pine_order.size == pytest.approx(0.6)

    assert broker.cancel_calls == [] and broker.cancel_with_outcome_calls == []
    assert engine._active_intents.get(intent.intent_key) == intent
    assert engine._order_mapping.get(intent.intent_key) == ['xchg-parent']

    engine._route_event(cancelled)

    assert intent.intent_key not in engine._order_mapping
    assert intent.intent_key not in engine._active_intents
    assert engine._entries_awaiting_fill_evidence == {}
    if oca_type == 'cancel':
        assert 'parent' not in engine._position.entry_orders
    else:
        assert pine_order.size == pytest.approx(0.6)


def __test_an_entry_held_for_its_evidence_stays_held_across_a_restart__(tmp_path):
    """A restart re-arms the hold instead of sending the entry again.

    The entry's order is terminal at the venue with an unknown executed
    quantity when the process stops. The restarted process re-arms the hold
    and the order mapping from the store: the script's re-issued entry adopts
    the mapping (no dispatch), and the execution reported afterwards routes to
    the entry and ends the hold, in memory and in the store.
    """
    from pynecore.core.broker.storage import ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine = _mk_engine(_MockBroker(), store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        engine._route_event(_ct_unknown_execution(
            _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))
        stored = ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND)
        assert stored[intent.intent_key]['order_ids'] == ['xchg-parent']
        ctx.close()

        ctx2 = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        broker2 = _MockBroker()
        engine2 = _mk_engine(broker2, store_ctx=ctx2)
        engine2._current_bar_ts_ms = BAR_TS
        assert intent.intent_key in engine2._entries_awaiting_fill_evidence
        assert engine2._order_mapping.get(intent.intent_key) == ['xchg-parent']

        # ``_MockBroker`` has no ``execute_entry``: a fresh dispatch would raise.
        engine2._diff_and_dispatch([intent])
        assert engine2._active_intents.get(intent.intent_key) == intent
        assert engine2._order_mapping.get(intent.intent_key) == ['xchg-parent']
        assert broker2.cancel_calls == []

        # A re-delivered event without the quantity stays deferred.
        engine2._route_event(_ct_unknown_execution(
            _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))
        assert engine2._order_mapping.get(intent.intent_key) == ['xchg-parent']

        engine2._route_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.4,
                                             fill_id='x-1'))
        assert engine2._position.size == pytest.approx(0.4)
        assert engine2._entries_awaiting_fill_evidence == {}
        assert ctx2.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND) == {}
        assert engine2.quarantined is False and not engine2.quarantined


def __test_a_hold_without_evidence_quarantines_once_and_the_venue_releases_it__(
        tmp_path):
    """A hold the stream never resolves ends in one quarantine, which the bot lifts itself.

    Past the fill-report grace without evidence the bot enters an
    ``ENTRY_BLOCK`` quarantine, once; the hold stays, so evidence arriving
    later still applies. A restart re-arms the expired hold and quarantines
    again. A resume of that quarantine keeps the hold: the venue may show
    what the order executed while the journal owns none of it. The venue's
    order history showing nothing executed releases it, the quarantine ends,
    and the restart after it neither holds the entry nor maps its order.
    """
    from pynecore.core.broker.models import QuarantineKind
    from pynecore.core.broker.storage import ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine = _mk_engine(_MockBroker(), store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        engine._route_event(_ct_unknown_execution(
            _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))
        hold = engine._entries_awaiting_fill_evidence[intent.intent_key]
        grace_ms = int(engine._fill_report_grace_s() * 1000)

        engine._expire_fill_evidence_holds(now_ms=hold.since_ms + grace_ms - 1)
        assert not engine.quarantined

        engine._expire_fill_evidence_holds(now_ms=hold.since_ms + grace_ms)
        assert engine.quarantined
        assert engine.quarantine_kind is QuarantineKind.ENTRY_BLOCK
        assert engine._quarantine_reason == 'entry_execution_evidence_missing'
        quarantine_id = engine.quarantine_id
        engine._expire_fill_evidence_holds(now_ms=hold.since_ms + 2 * grace_ms)
        assert engine.quarantine_id == quarantine_id
        assert intent.intent_key in engine._entries_awaiting_fill_evidence
        assert ctx.keyed_events_for_run_id(
            ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND)[intent.intent_key]['expired'] is True
        ctx.close()

        # A plain restart re-arms the expired hold and quarantines again.
        ctx2 = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine2 = _mk_engine(_MockBroker(), store_ctx=ctx2)
        assert not engine2.quarantined
        assert engine2._order_mapping.get(intent.intent_key) == ['xchg-parent']
        engine2._expire_fill_evidence_holds(now_ms=int(time.time() * 1000))
        assert engine2.quarantined
        assert engine2._quarantine_reason == 'entry_execution_evidence_missing'
        resumed_id = engine2.quarantine_id
        assert resumed_id is not None
        store.request_quarantine_resume(ctx2.run_id, resumed_id)
        ctx2.close()

        # A plain resume keeps the hold: the restart quarantines again.
        ctx3 = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        broker3 = _HeldCloseBroker()
        broker3.executions = [OrderExecution(order=_venue_parent(OrderStatus.CANCELLED, 0.0))]
        engine3 = _mk_engine(broker3, store_ctx=ctx3)
        assert intent.intent_key in engine3._entries_awaiting_fill_evidence
        assert engine3._order_mapping.get(intent.intent_key) == ['xchg-parent']
        engine3._expire_fill_evidence_holds(now_ms=int(time.time() * 1000))
        assert engine3.quarantined
        engine3._recover_held_executions(now_ms=int(time.time() * 1000))
        engine3._drain_events()
        assert engine3._entries_awaiting_fill_evidence == {}
        assert intent.intent_key not in engine3._order_mapping
        engine3._resolve_entry_block_causes(now_ms=int(time.time() * 1000))
        assert not engine3.quarantined
        engine3._flush_owed_store_writes()
        ctx3.close()

        ctx4 = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine4 = _mk_engine(_HeldCloseBroker(), store_ctx=ctx4)
        assert ctx4.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND) == {}
        assert intent.intent_key not in engine4._order_mapping
        engine4._expire_fill_evidence_holds(now_ms=int(time.time() * 1000))
        assert not engine4.quarantined


def __test_repeated_oca_reductions_of_a_held_entry_accumulate__():
    """Each OCA reduction of a held entry shrinks what the group already left.

    Two sibling fills (0.4, then 0.3) reduce a held one-unit entry. The held
    entry's venue intent keeps its sent quantity for the evidence, while its
    Pine order ends at 0.3, the size the next diff sends once the venue's zero
    retired the old order.
    """
    import dataclasses
    engine = _mk_engine(_MockBroker())
    seeded = _seed_parent_intent_and_leg(engine)
    intent = dataclasses.replace(seeded, oca_name='grp', oca_type='reduce')
    engine._active_intents[intent.intent_key] = intent
    pine_order = SimpleNamespace(size=1.0, sign=1.0)
    engine._position.entry_orders['parent'] = pine_order  # type: ignore[assignment]
    cancelled = _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)
    engine._route_event(_ct_unknown_execution(cancelled))
    sibling = EntryIntent(pine_id='sibling', symbol=SYMBOL, side='sell', qty=1.0,
                          order_type=OrderType.MARKET, oca_name='grp', oca_type='reduce')
    engine._active_intents[sibling.intent_key] = sibling

    for qty, fill_id in ((0.4, 's-1'), (0.3, 's-2')):
        engine._cascade_oca_reduce(dataclasses.replace(
            _ct_order_event('partial', filled_qty=qty, fill_qty=qty, fill_id=fill_id),
            pine_id='sibling'))

    assert pine_order.size == pytest.approx(0.3)
    assert engine._active_intents.get(intent.intent_key) == intent

    engine._cascade_oca_reduce(dataclasses.replace(
        _ct_order_event('partial', filled_qty=0.3, fill_qty=0.3, fill_id='s-3'),
        pine_id='sibling'))
    assert 'parent' not in engine._position.entry_orders


def __test_evidence_settling_a_restart_held_entry_retires_its_pending_protection__():
    """The venue's zero for a hold re-armed by a restart aborts the entry's pending legs.

    After a restart the hold and the order mapping are back, but the entry has
    no active intent until the script re-issues it. The proven zero that
    arrives first must still end the entry's pending partial-bracket leg
    (``parent_never_arrived``), so a later order under the same Pine id cannot
    arm it.
    """
    engine = _mk_engine(_MockBroker())
    intent = _seed_parent_intent_and_leg(engine)
    leg = next(iter(engine._partial_bracket_engine.iter_legs()))
    engine._route_event(_ct_unknown_execution(
        _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))
    # The restart state: the hold and the mapping, no active intent.
    engine._active_intents.pop(intent.intent_key)
    assert intent.intent_key in engine._entries_awaiting_fill_evidence
    assert leg.leg_state == LEG_STATE_PENDING_ENTRY

    engine._route_event(_ct_order_event('cancelled', filled_qty=0.0, fill_qty=None))

    assert leg.leg_state == LEG_STATE_ABORTED_PARENT_NEVER_ARRIVED
    assert intent.intent_key not in engine._order_mapping
    assert engine._entries_awaiting_fill_evidence == {}
    assert intent.intent_key not in engine._fill_evidence_entry_keys


def __test_evidence_received_before_a_restart_settles_the_hold_after_it__(tmp_path):
    """The venue's quantity received but not applied before the process stopped still settles the hold.

    The plugin delivers the terminal event with the venue's zero; the engine
    stores it on receipt and queues it, and the process stops before the
    drain. The restarted engine re-arms the hold and queues the stored
    evidence again: its first drain retires the entry without the plugin
    delivering anything, and the store keeps neither the hold nor the
    evidence. The plugin delivering the same event once more afterwards
    changes nothing.
    """
    import dataclasses
    from pynecore.core.broker.models import CANCEL_REASON_VENUE_POSITION_CLEARED
    from pynecore.core.broker.storage import (
        ENTRY_FILL_EVIDENCE_ARRIVAL_EVENT_KIND, ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND,
    )
    zero = dataclasses.replace(
        _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None),
        cancel_reason=CANCEL_REASON_VENUE_POSITION_CLEARED)
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine = _mk_engine(_MockBroker(), store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        engine._route_event(_ct_unknown_execution(zero))
        engine.on_order_event(zero)
        assert set(ctx.keyed_events_for_run_id(
            ENTRY_FILL_EVIDENCE_ARRIVAL_EVENT_KIND)) == {'xchg-parent'}
        ctx.close()

        ctx2 = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine2 = _mk_engine(_MockBroker(), store_ctx=ctx2)
        engine2._current_bar_ts_ms = BAR_TS
        assert intent.intent_key in engine2._entries_awaiting_fill_evidence

        engine2._drain_events()

        assert engine2._entries_awaiting_fill_evidence == {}
        assert intent.intent_key not in engine2._order_mapping
        assert ctx2.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND) == {}
        assert ctx2.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_ARRIVAL_EVENT_KIND) == {}

        engine2.on_order_event(zero)
        engine2._drain_events()
        assert ctx2.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_ARRIVAL_EVENT_KIND) == {}
        assert intent.intent_key not in engine2._order_mapping
        assert engine2.quarantined is False and not engine2.quarantined


class _ProcessDeath(BaseException):
    """Stands for the process dying at the point a test patches it in."""


def __test_a_process_death_after_the_settling_evidence_replays_it_at_the_restart__(tmp_path):
    """The hold and its evidence stay stored until the terminal event's teardown ran.

    The venue's zero for a held entry is received (stored, queued) and the
    process dies while its teardown runs, right after the point where the
    hold used to be released. The store still holds the hold and the
    evidence, and the restarted engine's first drain completes the teardown
    and only then releases both.
    """
    zero = dataclasses.replace(
        _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None),
        cancel_reason=CANCEL_REASON_VENUE_POSITION_CLEARED)
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine = _mk_engine(_MockBroker(), store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        engine._route_event(_ct_unknown_execution(zero))
        engine.on_order_event(zero)

        def _die(*_args, **_kwargs) -> None:
            raise _ProcessDeath()

        engine._abort_pending_partial_legs_for_dead_entry = _die  # type: ignore[method-assign]
        with pytest.raises(_ProcessDeath):
            engine._drain_events()
        assert set(ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND)) == {
            intent.intent_key}
        assert set(ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_ARRIVAL_EVENT_KIND)) == {
            'xchg-parent'}
        ctx.close()

        ctx2 = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine2 = _mk_engine(_MockBroker(), store_ctx=ctx2)
        engine2._current_bar_ts_ms = BAR_TS
        assert engine2._order_mapping.get(intent.intent_key) == ['xchg-parent']

        engine2._drain_events()

        assert intent.intent_key not in engine2._order_mapping
        assert engine2._entries_awaiting_fill_evidence == {}
        assert ctx2.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND) == {}
        assert ctx2.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_ARRIVAL_EVENT_KIND) == {}
        assert ctx2.fill_evidence_holds() == {}
        assert engine2.quarantined is False and not engine2.quarantined


def __test_a_plugin_resend_next_to_the_replayed_evidence_is_harmless__(tmp_path):
    """A plugin re-sending the owed terminal event after a restart changes nothing.

    The restarted engine queues the evidence it stored before the process
    stopped, and the plugin delivers the same event again before the first
    drain. The hold is read through the documented interface, the first copy
    settles it, the second finds no mapping and is only logged, and nothing
    is quarantined or halted.
    """
    zero = dataclasses.replace(
        _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None),
        cancel_reason=CANCEL_REASON_VENUE_POSITION_CLEARED)
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine = _mk_engine(_MockBroker(), store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        engine._route_event(_ct_unknown_execution(zero))
        assert ctx.fill_evidence_holds() == {intent.intent_key: ('xchg-parent',)}
        engine.on_order_event(zero)
        ctx.close()

        ctx2 = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine2 = _mk_engine(_MockBroker(), store_ctx=ctx2)
        engine2._current_bar_ts_ms = BAR_TS
        assert ctx2.fill_evidence_holds() == {intent.intent_key: ('xchg-parent',)}
        engine2.on_order_event(zero)

        engine2._drain_events()

        assert intent.intent_key not in engine2._order_mapping
        assert intent.intent_key not in engine2._envelopes
        assert ctx2.fill_evidence_holds() == {}
        assert ctx2.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_ARRIVAL_EVENT_KIND) == {}
        assert engine2.quarantined is False and not engine2.quarantined


def __test_positive_evidence_for_a_restart_held_entry_waits_for_its_intent__():
    """The venue's partial execution of a restart-held entry is applied through its intent.

    After a restart the hold and the mapping are back but the script has not
    re-issued the entry. A venue cancel reporting 0.4 of 1.0 executed must not
    take the generic teardown, which would drop the mapping and let the script
    send the whole entry again: the event waits with the hold. Once the entry
    is active again, the next drain applies it, and the entry is consumed at
    the executed quantity.
    """
    engine = _mk_engine(_MockBroker())
    intent = _seed_parent_intent_and_leg(engine)
    engine._route_event(_ct_unknown_execution(
        _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))
    engine._active_intents.pop(intent.intent_key)
    partial = dataclasses.replace(
        _ct_order_event('cancelled', filled_qty=0.4, fill_qty=None),
        cancel_reason=CANCEL_REASON_VENUE_POSITION_CLEARED)

    engine.on_order_event(partial)
    engine._drain_events()

    assert engine._order_mapping.get(intent.intent_key) == ['xchg-parent']
    assert intent.intent_key in engine._entries_awaiting_fill_evidence
    assert intent.intent_key in engine._envelopes

    engine._active_intents[intent.intent_key] = intent
    engine._drain_events()

    assert engine._entries_awaiting_fill_evidence == {}
    assert intent.intent_key not in engine._order_mapping
    consumed = engine._active_intents[intent.intent_key]
    assert isinstance(consumed, EntryIntent)
    assert consumed.qty == pytest.approx(0.4)
    assert engine.quarantined is False and not engine.quarantined


def __test_a_venue_proven_unexecuted_entry_retires_its_pending_protection__(tmp_path):
    """The venue's record of nothing executed tears the entry down like a proven zero.

    The restarted bot reads the held order from the venue, which shows it
    unexecuted: on the next drain its pending partial-bracket leg is aborted
    (``parent_never_arrived``), so a later order under the same Pine id
    cannot arm it, and the mapping, the envelope and the stored hold are gone.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine = _mk_engine(_MockBroker(), store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        engine._route_event(_ct_unknown_execution(
            _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))
        hold = engine._entries_awaiting_fill_evidence[intent.intent_key]
        engine._expire_fill_evidence_holds(
            now_ms=hold.since_ms + int(engine._fill_report_grace_s() * 1000))
        assert engine.quarantine_id is not None
        # The entry's pending partial-bracket leg, as the journal persisted it.
        ctx.upsert_order(
            'leg-1', symbol=SYMBOL, side='sell', qty=1.0, filled_qty=0.0,
            state=STATE_PARTIAL_BRACKET_LEG, intent_key='exit_tp\x00parent',
            from_entry=intent.intent_key, pine_entry_id='exit_tp',
            extras={
                EXTRAS_KEY_LEG_KIND: LEG_KIND_TP_PARTIAL,
                EXTRAS_KEY_LEG_STATE: LEG_STATE_PENDING_ENTRY,
                EXTRAS_KEY_PARENT_PINE_ENTRY_ID: intent.intent_key,
                EXTRAS_KEY_PARENT_ENTRY_DISPATCH_REF: 'parent-coid',
                EXTRAS_KEY_INTENT_PARTIAL_QTY: 1.0,
                EXTRAS_KEY_TRIGGER_OFFSET: 5.0,
            },
        )
        ctx.close()

        ctx2 = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        broker2 = _HeldCloseBroker()
        broker2.executions = [OrderExecution(order=_venue_parent(OrderStatus.CANCELLED, 0.0))]
        engine2 = _mk_engine(broker2, store_ctx=ctx2)
        engine2._current_bar_ts_ms = BAR_TS
        (leg,) = list(engine2._partial_bracket_engine.iter_legs())
        assert leg.leg_state == LEG_STATE_PENDING_ENTRY
        assert engine2._order_mapping.get(intent.intent_key) == ['xchg-parent']

        engine2._expire_fill_evidence_holds(now_ms=int(time.time() * 1000))
        engine2._recover_held_executions(now_ms=int(time.time() * 1000))
        engine2._drain_events()
        engine2._flush_owed_store_writes()

        assert leg.leg_state == LEG_STATE_ABORTED_PARENT_NEVER_ARRIVED
        assert intent.intent_key not in engine2._order_mapping
        assert engine2._entries_awaiting_fill_evidence == {}
        assert ctx2.fill_evidence_holds() == {}
        engine2._resolve_entry_block_causes(now_ms=int(time.time() * 1000))
        assert not engine2.quarantined


def __test_an_arrival_without_its_hold_is_dropped_at_the_restart__(tmp_path):
    """Stored evidence of a hold that already ended is not replayed."""
    from pynecore.core.broker.storage import ENTRY_FILL_EVIDENCE_ARRIVAL_EVENT_KIND
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        ctx.replace_keyed_events(ENTRY_FILL_EVIDENCE_ARRIVAL_EVENT_KIND,
                                 {'xchg-gone': {'event_type': 'cancelled'}})
        ctx.close()

        ctx2 = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine2 = _mk_engine(_MockBroker(), store_ctx=ctx2)
        assert engine2._event_queue.empty()
        assert ctx2.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_ARRIVAL_EVENT_KIND) == {}


@dataclass
class _QueueingCancelBroker(_MockBroker):
    """A plugin that journals an execution and delivers its event during the cancel.

    The first ``execute_cancel`` hands ``queued`` to the engine's event queue
    and then answers ``True`` (the venue order is gone, its executions are in
    the plugin's journal), so the cancel lands before the engine routed the
    execution.
    """
    engine: OrderSyncEngine | None = None
    queued: list = field(default_factory=list)

    async def execute_cancel(self, envelope: DispatchEnvelope) -> bool:
        assert self.engine is not None
        while self.queued:
            self.engine.on_order_event(self.queued.pop(0))
        return await super().execute_cancel(envelope)


def _queueing_engine(queued, *, capability=CapabilityLevel.SOFTWARE):
    broker = _QueueingCancelBroker(queued=list(queued))
    broker.capabilities = ExchangeCapabilities(partial_qty_bracket_exit=capability)
    engine = _mk_engine(broker)
    broker.engine = engine
    intent = _seed_parent_intent_and_leg(engine)
    leg = next(iter(engine._partial_bracket_engine.iter_legs()))
    leg.parent_entry_dispatch_ref = sorted(engine._parent_dispatch_refs('parent'))[0]
    return broker, engine, intent, leg


def __test_a_landed_cancel_with_a_queued_fill_keeps_the_filled_part_protected__():
    """A cancel landing behind a queued fill tears nothing down before the fill is booked.

    The plugin books 0.4 of the entry and queues its event, then answers the
    strategy's cancel ``True``. The entry keeps its mapping, envelope and leg
    until the drain routes the fill: the fill is booked once and arms the leg,
    never ``parent_never_arrived``.
    """
    fill = _ct_order_event('partial', filled_qty=0.4, fill_qty=0.4, fill_id='x-1')
    broker, engine, intent, leg = _queueing_engine([fill])

    assert engine._dispatch_cancel(intent) is True

    assert engine._order_mapping.get(intent.intent_key) == ['xchg-parent']
    assert intent.intent_key in engine._envelopes
    assert leg.leg_state == LEG_STATE_CANCEL_TENTATIVE

    engine._drain_events()

    assert engine._position.size == pytest.approx(0.4)
    assert leg.leg_state == LEG_STATE_ARMED
    assert engine._cancel_disposition_pending == {}
    # The unfilled remainder's cancel lands with nothing queued: its mapping goes.
    assert intent.intent_key not in engine._order_mapping
    engine._route_event(fill)
    engine._drain_events()
    assert engine._position.size == pytest.approx(0.4)
    assert leg.leg_state == LEG_STATE_ARMED
    assert engine.quarantined is False


def __test_a_landed_cancel_with_a_queued_partly_executed_terminal_keeps_the_legs__():
    """The same for a queued terminal event that reports a positive executed quantity."""
    cancelled = _ct_order_event('cancelled', filled_qty=0.4, fill_qty=None)
    broker, engine, intent, leg = _queueing_engine([cancelled])

    engine._dispatch_cancel(intent)

    assert engine._order_mapping.get(intent.intent_key) == ['xchg-parent']
    assert leg.leg_state == LEG_STATE_CANCEL_TENTATIVE

    engine._drain_events()

    assert leg.leg_state != LEG_STATE_ABORTED_PARENT_NEVER_ARRIVED
    assert engine._cancel_disposition_pending == {}
    engine._route_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.4, fill_id='x-1'))
    assert engine._position.size == pytest.approx(0.4)
    assert leg.leg_state == LEG_STATE_ARMED
    assert engine.quarantined is False


def __test_a_landed_cancel_with_nothing_queued_still_tears_the_entry_down__():
    """A cancel with no execution of the entry queued retires it at once, as before."""
    other = dataclasses.replace(
        _ct_order_event('partial', filled_qty=0.4, fill_qty=0.4, fill_id='x-9'),
        order=dataclasses.replace(
            _ct_order_event('partial', filled_qty=0.4, fill_qty=0.4).order, id='xchg-other'),
    )
    broker, engine, intent, leg = _queueing_engine([other])

    assert engine._dispatch_cancel(intent) is True

    assert intent.intent_key not in engine._order_mapping
    assert intent.intent_key not in engine._envelopes
    assert leg.leg_state == LEG_STATE_ABORTED_PARENT_NEVER_ARRIVED
    assert engine._forced_cancel_pending == {} and engine._cancel_disposition_pending == {}


def __test_a_landed_non_software_cancel_with_a_queued_fill_waits_for_the_fill__():
    """Outside the SOFTWARE bracket mode the cancel is parked until the fill is routed.

    The strict helper reports the cancel as not landed while the fill waits,
    the forced-cancel retry after the drain lands it once the fill is booked.
    """
    fill = _ct_order_event('partial', filled_qty=0.4, fill_qty=0.4, fill_id='x-1')
    broker, engine, intent, _leg = _queueing_engine(
        [fill], capability=CapabilityLevel.NATIVE)

    assert engine._dispatch_cancel(intent) is False

    assert intent.intent_key in engine._forced_cancel_pending
    assert engine._order_mapping.get(intent.intent_key) == ['xchg-parent']

    engine._drain_events()

    assert engine._position.size == pytest.approx(0.4)
    assert engine._forced_cancel_pending == {}
    assert intent.intent_key not in engine._order_mapping
    assert len(broker.cancel_calls) == 2
    assert engine.quarantined is False


def __test_a_confirmed_tentative_cancel_with_a_queued_fill_resolves_as_filled__():
    """A cancel-tentative retry confirming the cancel waits for a queued execution.

    The retry's ``CANCEL_CONFIRMED`` would abort the legs; the queued fill
    instead resolves the entry as filled when the drain routes it.
    """
    broker, engine, intent, leg = _queueing_engine([])
    engine._cancel_disposition_pending[intent.intent_key] = _CancelTentativeMeta(
        since_ts_ms=BAR_TS, reason='broker_timeout',
    )
    engine._partial_bracket_engine.mark_legs_cancel_tentative(
        intent.intent_key, reason='broker_timeout', now_ms=BAR_TS,
    )
    engine.on_order_event(
        _ct_order_event('partial', filled_qty=0.4, fill_qty=0.4, fill_id='x-1'))
    broker.cancel_outcomes = [CancelDispositionOutcome.CANCEL_CONFIRMED]

    engine._drive_cancel_tentative(now_ms=BAR_TS + 1_000)

    assert intent.intent_key in engine._cancel_disposition_pending
    assert engine._cancel_disposition_pending[intent.intent_key].retry_count == 0
    assert leg.leg_state == LEG_STATE_CANCEL_TENTATIVE

    engine._drain_events()

    assert engine._cancel_disposition_pending == {}
    assert engine._position.size == pytest.approx(0.4)
    assert leg.leg_state == LEG_STATE_ARMED


def __test_a_landed_residual_cancel_with_a_queued_fill_routes_that_fill__():
    """A queued execution of a remainder whose residual cancel landed is still booked.

    A ``partial`` event resolves the tentative parent as filled and cancels
    the unfilled remainder; the plugin queues a further 0.2 execution of it
    and answers ``True``. That execution is booked to the entry and the dead
    remainder's mapping goes.
    """
    more = dataclasses.replace(
        _ct_order_event('partial', filled_qty=0.6, fill_qty=0.2, fill_id='x-2'),
    )
    broker, engine, intent, leg = _queueing_engine([more])
    engine._cancel_disposition_pending[intent.intent_key] = _CancelTentativeMeta(
        since_ts_ms=BAR_TS, reason='broker_timeout',
    )
    engine._partial_bracket_engine.mark_legs_cancel_tentative(
        intent.intent_key, reason='broker_timeout', now_ms=BAR_TS,
    )
    engine.on_order_event(
        _ct_order_event('partial', filled_qty=0.4, fill_qty=0.4, fill_id='x-1'))

    engine._drain_events()

    assert len(broker.cancel_calls) == 1
    assert engine._position.size == pytest.approx(0.6)
    assert intent.intent_key not in engine._order_mapping
    assert engine.quarantined is False


def __test_a_storage_failure_while_evidence_settles_a_hold_retries_it_in_process__(tmp_path):
    """The terminal event that settles a hold is routed again after a storage failure.

    The venue's zero for a held entry is delivered once. Its teardown fails
    once with a storage error: the hold stands again and the event stays for
    the next drain, which retires the entry and releases the stored hold and
    evidence without a restart and without the plugin delivering it again.
    """
    zero = dataclasses.replace(
        _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None),
        cancel_reason=CANCEL_REASON_VENUE_POSITION_CLEARED)
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine = _mk_engine(_MockBroker(), store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        engine._route_event(_ct_unknown_execution(zero))
        engine.on_order_event(zero)
        real_abort = engine._abort_pending_partial_legs_for_dead_entry
        failed: list[str] = []

        def abort_once(*args, **kwargs):
            if not failed:
                failed.append(args[0])
                raise sqlite3.OperationalError("disk I/O error")
            return real_abort(*args, **kwargs)

        engine._abort_pending_partial_legs_for_dead_entry = abort_once  # type: ignore[method-assign]
        with pytest.raises(sqlite3.OperationalError):
            engine._drain_events()
        assert failed == [intent.intent_key]
        assert intent.intent_key in engine._entries_awaiting_fill_evidence

        engine._drain_events()

        assert engine._entries_awaiting_fill_evidence == {}
        assert intent.intent_key not in engine._order_mapping
        assert intent.intent_key not in engine._envelopes
        engine._flush_owed_store_writes()
        assert ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND) == {}
        assert ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_ARRIVAL_EVENT_KIND) == {}
        assert engine.quarantined is False and not engine.quarantined


def __test_a_second_restart_before_the_execution_is_booked_keeps_the_hold__(tmp_path):
    """Applied positive evidence keeps the stored hold until its execution is booked.

    A restart-held entry gets the venue's cancel reporting 0.4 of 1.0
    executed; the re-issued entry is consumed at 0.4, but the 0.4 fill has not
    arrived yet. The store keeps the hold and the evidence, so a second
    restart in that window re-arms the hold and maps the order again instead
    of letting the script send a fresh entry. The fill arriving afterwards is
    booked once and only then releases both.
    """
    partial = dataclasses.replace(
        _ct_order_event('cancelled', filled_qty=0.4, fill_qty=None),
        cancel_reason=CANCEL_REASON_VENUE_POSITION_CLEARED)
    fill = _ct_order_event('partial', filled_qty=0.4, fill_qty=0.4, fill_id='x-1')
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine = _mk_engine(_MockBroker(), store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        engine._route_event(_ct_unknown_execution(
            _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))
        ctx.close()

        ctx2 = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine2 = _mk_engine(_MockBroker(), store_ctx=ctx2)
        engine2._current_bar_ts_ms = BAR_TS
        engine2.on_order_event(partial)
        engine2._drain_events()
        engine2._active_intents[intent.intent_key] = intent
        engine2._drain_events()
        engine2._flush_owed_store_writes()
        assert engine2._entries_awaiting_fill_evidence == {}
        assert engine2._active_intents[intent.intent_key].qty == pytest.approx(0.4)
        assert set(ctx2.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND)) == {
            intent.intent_key}
        assert set(ctx2.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_ARRIVAL_EVENT_KIND)) == {
            'xchg-parent'}
        ctx2.close()

        ctx3 = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine3 = _mk_engine(_MockBroker(), store_ctx=ctx3)
        engine3._current_bar_ts_ms = BAR_TS
        assert intent.intent_key in engine3._entries_awaiting_fill_evidence
        assert engine3._order_mapping.get(intent.intent_key) == ['xchg-parent']
        engine3._active_intents[intent.intent_key] = intent
        engine3._drain_events()
        assert engine3._active_intents[intent.intent_key].qty == pytest.approx(0.4)
        engine3._flush_owed_store_writes()
        assert set(ctx3.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND)) == {
            intent.intent_key}

        engine3.on_order_event(fill)
        engine3._drain_events()
        engine3._flush_owed_store_writes()

        assert engine3._position.size == pytest.approx(0.4)
        assert ctx3.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND) == {}
        assert ctx3.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_ARRIVAL_EVENT_KIND) == {}
        assert engine3._fill_evidence_awaiting_execution == {}
        assert engine3.quarantined is False and not engine3.quarantined


def _hold_fsm_evidence(evidence: str) -> list:
    """The events that bring the held entry's execution evidence, in delivery order."""
    fill = _ct_order_event('partial', filled_qty=0.4, fill_qty=0.4, fill_id='x-1')
    if evidence == 'zero':
        return [dataclasses.replace(
            _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None),
            cancel_reason=CANCEL_REASON_VENUE_POSITION_CLEARED)]
    if evidence == 'fill':
        return [fill]
    if evidence == 'fill_then_positive':
        # A first 0.2 slice queued ahead of the cancel reporting 0.4 executed,
        # then the second slice.
        return [_ct_order_event('partial', filled_qty=0.2, fill_qty=0.2, fill_id='x-1'),
                _hold_positive_terminal(),
                _ct_order_event('partial', filled_qty=0.4, fill_qty=0.2, fill_id='x-2')]
    return [dataclasses.replace(
        _ct_order_event('cancelled', filled_qty=0.4, fill_qty=None),
        cancel_reason=CANCEL_REASON_VENUE_POSITION_CLEARED), fill]


def _hold_fsm_restart(store, intent):
    """A restarted engine of the run whose script issued ``intent`` again."""
    ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
    engine = _mk_engine(_MockBroker(), store_ctx=ctx)
    engine._current_bar_ts_ms = BAR_TS
    assert intent.intent_key in engine._entries_awaiting_fill_evidence
    assert engine._order_mapping.get(intent.intent_key) == ['xchg-parent']
    engine._active_intents[intent.intent_key] = intent
    return engine, ctx


@pytest.mark.parametrize('crash', ['none', 'error_before', 'error_after', 'death_before_release'])
@pytest.mark.parametrize('evidence', ['zero', 'fill', 'positive', 'fill_then_positive'])
@pytest.mark.parametrize('state', ['held', 'restarted'])
def __test_the_fill_evidence_hold_reaches_its_outcome_from_every_state_and_crash_point__(
        tmp_path, state, evidence, crash):
    """The fill-evidence hold state machine, table-driven (state x evidence x crash point).

    An entry is held (in this process, or re-armed by a restart with the entry
    issued again). Its evidence arrives: the venue's zero, a 0.4 fill, or a
    cancel reporting 0.4 executed followed by the 0.4 fill, or a first 0.2
    slice queued ahead of that cancel and the second slice after it. The routing of the
    first evidence event raises a storage error once before or after its
    work, or the process dies after the last one before the hold's release
    lands. Whatever the crash, the hold reaches its outcome without a
    quarantine or halt: no hold in memory or store, the 0.4 booked once, the
    unexecuted entry unmapped, and a plugin re-sending the evidence afterwards
    changes nothing.
    """
    events = _hold_fsm_evidence(evidence)
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine = _mk_engine(_MockBroker(), store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        engine._route_event(_ct_unknown_execution(
            _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))
        assert intent.intent_key in engine._entries_awaiting_fill_evidence
        if state == 'restarted':
            ctx.close()
            engine, ctx = _hold_fsm_restart(store, intent)

        if crash in ('error_before', 'error_after'):
            real_route = engine._route_order_event
            raised: list[bool] = []

            def route_once(event):
                if not raised and crash == 'error_before':
                    raised.append(True)
                    raise sqlite3.OperationalError("disk I/O error")
                real_route(event)
                if not raised:
                    raised.append(True)
                    raise sqlite3.OperationalError("disk I/O error")

            engine._route_order_event = route_once  # type: ignore[method-assign]
            engine.on_order_event(events[0])
            with pytest.raises(sqlite3.OperationalError):
                engine._drain_events()
            for event in events[1:]:
                engine.on_order_event(event)
            engine._drain_events()
        elif crash == 'death_before_release':
            def no_release(_key, _hold) -> None:
                return None

            for event in events[:-1]:
                engine.on_order_event(event)
            engine._drain_events()
            engine._owe_fill_evidence_release = no_release  # type: ignore[method-assign]
            engine.on_order_event(events[-1])
            engine._drain_events()
            engine._flush_owed_store_writes()
            assert set(ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND)) == {
                intent.intent_key}
            ctx.close()
            engine, ctx = _hold_fsm_restart(store, intent)
            if evidence != 'zero':
                # The startup takes in the fills the journal shows.
                engine._position.seed_fill_cursors({'xchg-parent': 0.4})
            engine._drain_events()
        else:
            for event in events:
                engine.on_order_event(event)
            engine._drain_events()

        engine._expire_fill_evidence_holds(now_ms=int(time.time() * 1000))
        engine._flush_owed_store_writes()

        assert engine._entries_awaiting_fill_evidence == {}
        assert engine._settled_fill_evidence == {}
        assert engine._fill_evidence_awaiting_execution == {}
        assert ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND) == {}
        assert ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_ARRIVAL_EVENT_KIND) == {}
        size = engine._position.size
        if evidence == 'zero':
            assert intent.intent_key not in engine._order_mapping
            assert size == 0.0
        elif crash != 'death_before_release':
            assert size == pytest.approx(0.4)
        assert engine.quarantined is False and not engine.quarantined

        for event in events:
            engine.on_order_event(event)
        engine._drain_events()
        engine._expire_fill_evidence_holds(now_ms=int(time.time() * 1000))
        engine._flush_owed_store_writes()

        assert engine._entries_awaiting_fill_evidence == {}
        assert ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND) == {}
        if evidence == 'zero' or crash != 'death_before_release':
            # A restarted engine owns a pre-restart fill through the startup
            # adoption, which this test does not run.
            assert engine._position.size == pytest.approx(size)
        assert engine.quarantined is False and not engine.quarantined


def _hold_positive_terminal(event_type: str = 'cancelled'):
    """The venue's terminal event of ``xchg-parent`` reporting 0.4 of 1.0 executed."""
    event = dataclasses.replace(
        _ct_order_event('cancelled', filled_qty=0.4, fill_qty=None),
        cancel_reason=CANCEL_REASON_VENUE_POSITION_CLEARED)
    if event_type == 'rejected':
        event = dataclasses.replace(
            event, event_type='rejected', cancel_reason=None,
            order=dataclasses.replace(event.order, status=OrderStatus.REJECTED))
    return event


def __test_an_out_of_order_slice_does_not_release_the_awaited_execution__(tmp_path):
    """The reported execution is released only once the book took all of it in.

    A held entry's venue cancel reports 0.4 executed. Its second 0.2 slice
    arrives first, carrying the cumulative 0.4: the book holds 0.2, so the
    stored hold and evidence stay. The first slice then completes the 0.4 and
    releases both.
    """
    second = _ct_order_event('partial', filled_qty=0.4, fill_qty=0.2, fill_id='x-2')
    first = _ct_order_event('partial', filled_qty=0.2, fill_qty=0.2, fill_id='x-1')
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine = _mk_engine(_MockBroker(), store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        engine._route_event(_ct_unknown_execution(
            _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))
        engine.on_order_event(_hold_positive_terminal())
        engine.on_order_event(second)
        engine._drain_events()
        engine._flush_owed_store_writes()

        assert engine._position.size == pytest.approx(0.2)
        assert set(ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND)) == {
            intent.intent_key}
        assert 'xchg-parent' in engine._fill_evidence_awaiting_execution

        engine.on_order_event(first)
        engine._drain_events()
        engine._flush_owed_store_writes()

        assert engine._position.size == pytest.approx(0.4)
        assert ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND) == {}
        assert ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_ARRIVAL_EVENT_KIND) == {}
        assert engine._fill_evidence_awaiting_execution == {}


@pytest.mark.parametrize('when', ['after_terminal', 'row_not_rewritten', 'parked'])
def __test_a_restart_between_slices_keeps_the_hold_until_the_reported_execution_is_booked__(
        tmp_path, when):
    """A restart after the first slice of a reported execution keeps the hold.

    The held entry's venue cancel reports 0.4 executed and a first 0.2 slice is
    booked, either after the cancel was applied through the entry
    (``after_terminal``; ``row_not_rewritten``: the process died before the
    hold row took the 0.4, which the stored evidence still names) or while the
    cancel waits for the entry after an earlier restart (``parked``). The
    restarted engine's hold knows the 0.4:
    neither the 0.2 its startup takes in nor its reconcile releases it, only
    the second slice does.
    """
    first = _ct_order_event('partial', filled_qty=0.2, fill_qty=0.2, fill_id='x-1')
    second = _ct_order_event('partial', filled_qty=0.4, fill_qty=0.2, fill_id='x-2')
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine = _mk_engine(_MockBroker(), store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        engine._route_event(_ct_unknown_execution(
            _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))
        if when == 'parked':
            ctx.close()
            ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
            engine = _mk_engine(_MockBroker(), store_ctx=ctx)
            engine._current_bar_ts_ms = BAR_TS
        elif when == 'row_not_rewritten':
            def keep_row(_key, _hold) -> None:
                return None

            engine._persist_fill_evidence_hold = keep_row  # type: ignore[method-assign]
        engine.on_order_event(_hold_positive_terminal())
        engine.on_order_event(first)
        engine._drain_events()
        engine._flush_owed_store_writes()
        assert engine._position.size == pytest.approx(0.2)
        assert set(ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND)) == {
            intent.intent_key}
        ctx.close()

        ctx2 = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine2 = _mk_engine(_MockBroker(), store_ctx=ctx2)
        engine2._current_bar_ts_ms = BAR_TS
        # The startup takes in the fills the journal shows.
        engine2._position.seed_fill_cursors({'xchg-parent': 0.2})
        # A reconcile before and after the first drain.
        engine2._expire_fill_evidence_holds(now_ms=int(time.time() * 1000))
        engine2._drain_events()
        engine2._expire_fill_evidence_holds(now_ms=int(time.time() * 1000))
        engine2._flush_owed_store_writes()

        assert set(ctx2.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND)) == {
            intent.intent_key}
        assert engine2._order_mapping.get(intent.intent_key) == ['xchg-parent']

        engine2._active_intents[intent.intent_key] = intent
        engine2._drain_events()
        engine2.on_order_event(second)
        engine2._drain_events()
        engine2._expire_fill_evidence_holds(now_ms=int(time.time() * 1000))
        engine2._flush_owed_store_writes()

        assert ctx2.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND) == {}
        assert ctx2.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_ARRIVAL_EVENT_KIND) == {}
        assert engine2._entries_awaiting_fill_evidence == {}
        assert engine2._fill_evidence_awaiting_execution == {}
        assert engine2.quarantined is False and not engine2.quarantined


def __test_a_landed_cancel_with_a_queued_unknown_execution_terminal_keeps_the_legs__():
    """A cancel landing behind a queued terminal event without the venue's quantity.

    The plugin queues the entry's cancellation that does not say how much
    executed, then answers the cancel ``True``. Nothing is torn down before the
    drain routes that event, which holds the entry; the 0.4 fill reported
    afterwards arms the leg.
    """
    unknown = _ct_unknown_execution(_ct_order_event('cancelled', filled_qty=0.0, fill_qty=None))
    broker, engine, intent, leg = _queueing_engine([unknown])

    engine._dispatch_cancel(intent)

    assert engine._order_mapping.get(intent.intent_key) == ['xchg-parent']
    assert leg.leg_state == LEG_STATE_CANCEL_TENTATIVE

    engine._drain_events()

    assert intent.intent_key in engine._entries_awaiting_fill_evidence
    assert leg.leg_state != LEG_STATE_ABORTED_PARENT_NEVER_ARRIVED
    engine._route_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.4, fill_id='x-1'))
    assert engine._position.size == pytest.approx(0.4)
    assert leg.leg_state == LEG_STATE_ARMED
    assert engine.quarantined is False


@pytest.mark.parametrize('terminal', ['cancelled', 'rejected'])
def __test_a_reported_execution_not_yet_booked_blocks_a_new_cycle_of_the_entry__(terminal):
    """An entry whose terminal event reported 0.4 executed is not sent again before the 0.4 is booked.

    The held entry's venue cancel (or reject) reports 0.4 executed: the entry
    is consumed at 0.4, not retired, so the diff does not re-place it, and
    while that fill is not booked the entry counts as awaiting it (the gate
    of a fresh same-parameter cycle). Once the fill is booked it no longer
    does.
    """
    engine = _mk_engine(_MockBroker())
    intent = _seed_parent_intent_and_leg(engine, with_leg=False)
    engine._route_event(_ct_unknown_execution(
        _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))
    engine.on_order_event(_hold_positive_terminal(terminal))
    engine._drain_events()

    consumed = engine._active_intents.get(intent.intent_key)
    assert isinstance(consumed, EntryIntent)
    assert consumed.qty == pytest.approx(0.4)
    assert intent.intent_key in engine._truncated_entries
    assert engine._entry_execution_awaiting_booking(intent.intent_key)

    engine._route_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.4, fill_id='x-1'))

    assert engine._position.size == pytest.approx(0.4)
    assert not engine._entry_execution_awaiting_booking(intent.intent_key)
    assert engine.quarantined is False


def __test_a_fill_queued_ahead_of_the_reported_execution_keeps_the_restart_barrier__(tmp_path):
    """A first slice settling the hold ahead of the terminal event that reports 0.4 executed.

    Both events are queued before the drain: the 0.2 slice settles the hold
    first, the cancel reporting 0.4 comes after it. The hold waits for the 0.4
    (``HELD`` + received evidence -> ``SETTLING`` -> ``AWAITING_EXECUTION``), so
    a restart before the second slice re-arms it with its mapping; only the
    second slice releases it.
    """
    first = _ct_order_event('partial', filled_qty=0.2, fill_qty=0.2, fill_id='x-1')
    second = _ct_order_event('partial', filled_qty=0.4, fill_qty=0.2, fill_id='x-2')
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine = _mk_engine(_MockBroker(), store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        engine._route_event(_ct_unknown_execution(
            _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))
        engine.on_order_event(first)
        engine.on_order_event(_hold_positive_terminal())
        engine._drain_events()
        engine._flush_owed_store_writes()

        assert engine._position.size == pytest.approx(0.2)
        assert 'xchg-parent' in engine._fill_evidence_awaiting_execution
        assert set(ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND)) == {
            intent.intent_key}
        ctx.close()

        ctx2 = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine2 = _mk_engine(_MockBroker(), store_ctx=ctx2)
        engine2._current_bar_ts_ms = BAR_TS
        engine2._position.seed_fill_cursors({'xchg-parent': 0.2})
        engine2._expire_fill_evidence_holds(now_ms=int(time.time() * 1000))
        assert intent.intent_key in engine2._entries_awaiting_fill_evidence
        assert engine2._order_mapping.get(intent.intent_key) == ['xchg-parent']

        engine2._active_intents[intent.intent_key] = intent
        engine2._drain_events()
        engine2.on_order_event(second)
        engine2._drain_events()
        engine2._expire_fill_evidence_holds(now_ms=int(time.time() * 1000))
        engine2._flush_owed_store_writes()

        assert ctx2.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND) == {}
        assert ctx2.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_ARRIVAL_EVENT_KIND) == {}
        assert engine2._fill_evidence_awaiting_execution == {}
        assert engine2.quarantined is False and not engine2.quarantined


@pytest.mark.parametrize('evidence', ['zero', 'fill'])
def __test_a_native_strategy_cancel_with_a_queued_unknown_execution_holds_the_entry__(evidence):
    """The strategy drops an entry; its cancel lands behind a cancellation of unknown execution.

    NATIVE mode parks the landed cancel as a forced cancel (an execution may
    still be queued). The queued cancellation that does not say how much
    executed holds the entry (``HELD``, entered through the forced-cancel
    park), its mapping stays and the bot is not quarantined for its own
    cancel. The venue's zero then retires the entry through the forced-cancel
    retry; a 0.4 fill instead is booked first.
    """
    unknown = _ct_unknown_execution(_ct_order_event('cancelled', filled_qty=0.0, fill_qty=None))
    broker, engine, intent, _leg = _queueing_engine(
        [unknown], capability=CapabilityLevel.NATIVE)

    engine._diff_and_dispatch([])

    assert intent.intent_key in engine._forced_cancel_pending
    engine._drain_events()

    assert intent.intent_key in engine._entries_awaiting_fill_evidence
    assert engine._order_mapping.get(intent.intent_key) == ['xchg-parent']
    assert not engine.quarantined and engine.quarantined is False

    if evidence == 'zero':
        engine.on_order_event(_ct_order_event('cancelled', filled_qty=0.0, fill_qty=None))
    else:
        engine.on_order_event(
            _ct_order_event('partial', filled_qty=0.4, fill_qty=0.4, fill_id='x-1'))
    engine._drain_events()

    assert engine._entries_awaiting_fill_evidence == {}
    assert engine._forced_cancel_pending == {}
    assert intent.intent_key not in engine._order_mapping
    assert engine._position.size == pytest.approx(0.0 if evidence == 'zero' else 0.4)
    assert not engine.quarantined and engine.quarantined is False


@pytest.mark.parametrize('diff', ['changed', 'removed_then_readded'])
def __test_an_entry_awaiting_its_reported_execution_sends_nothing_on_its_key__(diff):
    """No diff path sends an entry while a terminal event's reported execution is unbooked.

    The held entry's venue reject reports 0.4 of 1.0 executed and the entry is
    consumed at 0.4. Before the 0.4 fill is booked, re-issuing it at another
    quantity, or dropping it and issuing it again, sends nothing and keeps
    the consumed entry; once the fill is booked the changed entry goes out.
    """
    engine = _mk_engine(_MockBroker())
    intent = _seed_parent_intent_and_leg(engine, with_leg=False)
    sent: list = []

    def record_new(new_intent) -> None:
        sent.append(('new', new_intent))

    def record_modify(*args, **kwargs) -> None:
        sent.append(('modify', args))

    engine._dispatch_new = record_new  # type: ignore[method-assign]
    engine._dispatch_modify = record_modify  # type: ignore[method-assign]
    engine._route_event(_ct_unknown_execution(
        _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))
    engine.on_order_event(_hold_positive_terminal('rejected'))
    engine._drain_events()
    assert engine._entry_execution_awaiting_booking(intent.intent_key)

    changed = dataclasses.replace(intent, qty=1.1)
    if diff == 'changed':
        engine._diff_and_dispatch([changed])
    else:
        engine._diff_and_dispatch([])
        engine._diff_and_dispatch([intent])

    assert sent == []
    consumed = engine._active_intents.get(intent.intent_key)
    assert isinstance(consumed, EntryIntent) and consumed.qty == pytest.approx(0.4)

    engine._route_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.4, fill_id='x-1'))
    assert not engine._entry_execution_awaiting_booking(intent.intent_key)
    engine._diff_and_dispatch([changed])

    assert sent and sent[0][0] == 'new'
    assert engine.quarantined is False


# === The fill-evidence hold matrix =====================================
#
# Every state of the hold (:class:`~pynecore.core.broker.sync_engine._FillEvidenceHold`)
# crossed with what arrives next, what the strategy does on the next sync,
# the partial-bracket mode, whether the arrival is still queued when the diff
# runs, and a crash (a storage error in the routing, one restart, two).

_MATRIX_EXECUTED = 0.4
_MATRIX_STATES = ('held', 'tentative', 'restarted', 'received', 'parked', 'awaiting',
                  'awaiting_sibling', 'reported_sibling', 'rising_sibling', 'terminal',
                  'terminal_sibling', 'tentative_cancelled', 'tentative_rejected')
#: A sibling venue order of the entry, mapped next to the held one, whose
#: fills the book took in up to more than the held order reports executed.
_MATRIX_SIBLING_ORDER = 'xchg-sib'
_MATRIX_SIBLING_BOOKED = 0.5
#: What the sibling's own terminal event reports executed in the
#: ``reported_sibling`` state, none of it booked yet: its fill arrives last.
_MATRIX_SIBLING_REPORTED = 0.2
#: In the ``rising_sibling`` state the sibling's report is a lower bound: its
#: first 0.2 slice shows this much executed, and a 0.1 slice follows a sync later.
_MATRIX_SIBLING_RISEN = 0.3
_MATRIX_AWAITING_STATES = ('received', 'parked', 'awaiting', 'awaiting_sibling',
                           'reported_sibling', 'rising_sibling')
#: States entered by a terminal event reporting the execution ahead of its
#: fills, with no hold before it: of the active entry, or of a cancel-tentative
#: parent (its venue ``cancelled`` / ``rejected``).
_MATRIX_UNHELD_STATES = ('terminal', 'terminal_sibling', 'tentative_cancelled',
                         'tentative_rejected')
_MATRIX_ARRIVALS = ('slice', 'cum_slice', 'cum_rise', 'terminal_rise', 'queued_rise',
                    'queued_cum', 'queued_cum_bound', 'stale_restart', 'fill', 'zero', 'positive', 'unknown', 'duplicate',
                    'nothing')
#: A restart of the bot inside an arrival's sequence, before what was received
#: is drained.
_MATRIX_RESTART = 'restart'
#: ``reverse``: a market stop-and-reverse under another id (a 1.0 sell) next to
#: the kept entry; ``close_all``: a ``strategy.close_all`` sized from the book.
#: Both act on the venue (:class:`_MatrixVenueBroker`).
_MATRIX_ACTIONS = ('keep', 'remove', 'qty', 'price', 'readd', 'close', 'oca_cancel',
                   'oca_reduce', 'close_sweep', 'reverse', 'close_all')
#: The actions that write to the matrix venue, and what each sibling state's
#: sibling order executed there (the fills its arrivals deliver).
_MATRIX_VENUE_ACTIONS = ('reverse', 'close_all')
_MATRIX_SIBLING_VENUE = {'reported_sibling': _MATRIX_SIBLING_REPORTED,
                         'rising_sibling': _MATRIX_SIBLING_RISEN}
_MATRIX_MODES = ('software', 'native')
_MATRIX_TIMINGS = ('drained', 'queued')
#: ``torn``: the process dies right after the first store write of the
#: routing that enters the state (an unheld state) or of the arrival's sync
#: (a held one) landed, and the bot restarts.
_MATRIX_CRASHES = ('none', 'storage_error', 'restart', 'restart2', 'resume', 'torn')


class _ProcessDeath(BaseException):
    """The process died: nothing after it runs or reaches the store."""


def _matrix_truths(state: str, arrival: str) -> tuple[float, ...]:
    """The executed quantities of the held order the case is consistent with."""
    if state in _MATRIX_AWAITING_STATES + _MATRIX_UNHELD_STATES:
        return () if arrival == 'zero' else (_MATRIX_EXECUTED,)
    if arrival == 'zero':
        return (0.0,)
    if arrival in ('slice', 'cum_slice', 'cum_rise', 'terminal_rise', 'queued_rise',
                   'queued_cum', 'queued_cum_bound', 'stale_restart', 'fill', 'positive'):
        return (_MATRIX_EXECUTED,)
    return (0.0, _MATRIX_EXECUTED)


def _matrix_combinations() -> list[tuple]:
    """Every consistent ``(state, arrival, action, mode, timing, crash, truth)`` case."""
    combinations = []
    for state, arrival, action, mode, timing, crash in itertools.product(
            _MATRIX_STATES, _MATRIX_ARRIVALS, _MATRIX_ACTIONS, _MATRIX_MODES,
            _MATRIX_TIMINGS, _MATRIX_CRASHES):
        if state.startswith('tentative') and mode != 'software':
            # Only a SOFTWARE entry's cancel becomes a cancel-tentative probe.
            continue
        for truth in _matrix_truths(state, arrival):
            combinations.append((state, arrival, action, mode, timing, crash, truth))
    return combinations


def _matrix_pairwise_slice(combinations: list[tuple]) -> set[tuple]:
    """A deterministic subset in which every value pair of any two dimensions occurs.

    Greedy set cover: the next case taken is the one covering the most value
    pairs no case taken before covers, ties broken by generation order. The
    gains only shrink, so a heap of stale gains re-scores just the popped
    case (lazy greedy). The slice exercises every state against every
    arrival, action, mode, timing, crash and truth at a small fraction of
    the full product.
    """
    case_pairs = [frozenset((i, case[i], j, case[j])
                            for i, j in itertools.combinations(range(len(case)), 2))
                  for case in combinations]
    uncovered = set().union(*case_pairs)
    heap = [(-len(pairs), index) for index, pairs in enumerate(case_pairs)]
    heapq.heapify(heap)
    taken: set[tuple] = set()
    while uncovered:
        _, index = heapq.heappop(heap)
        gain = len(case_pairs[index] & uncovered)
        if gain == 0:
            continue
        if heap and (-gain, index) > heap[0]:
            heapq.heappush(heap, (-gain, index))
            continue
        uncovered -= case_pairs[index]
        taken.add(combinations[index])
    return taken


def _matrix_cases() -> list:
    """The matrix cases; all but the pairwise slice carry the ``exhaustive`` marker.

    The default run (``-m "not live and not exhaustive"``) takes the slice
    only. ``-m exhaustive`` runs the rest; see "Exhaustive tests" in
    ``docs/development/testing-system.md`` for when the full product is due.
    """
    combinations = _matrix_combinations()
    default_slice = _matrix_pairwise_slice(combinations)
    cases = []
    for case in combinations:
        state, arrival, action, mode, timing, crash, truth = case
        cases.append(pytest.param(
            *case,
            id=f"{state}-{arrival}-{action}-{mode}-{timing}-{crash}-x{truth}",
            marks=() if case in default_slice else pytest.mark.exhaustive))
    return cases


@dataclass
class _MatrixVenueBroker(_MockBroker):
    """The matrix venue of the ``reverse`` and ``close_all`` actions.

    It holds the position the venue executed (:attr:`_MatrixRun.venue`, which
    survives the bot's restarts), closes reduce-only, never past flat, and
    reports each close and entry it executes as a fill on the engine's event
    queue. It closes a ``held_execution_close`` by its venue quantity.
    """
    run: '_MatrixRun | None' = None
    supports_held_execution_close: bool = True

    def _write(self, kind: str, envelope: DispatchEnvelope, executed: float):
        from pynecore.core.broker.models import ExchangeOrder, LegType, OrderEvent
        run = self.run
        assert run is not None and run.engine is not None
        intent = envelope.intent
        run.venue_writes += 1
        order_id = f"xchg-v{run.venue_writes}"
        order = ExchangeOrder(
            id=order_id, symbol=SYMBOL, side=intent.side, order_type=OrderType.MARKET,
            qty=intent.qty, filled_qty=0.0, remaining_qty=intent.qty, price=None,
            stop_price=None, average_fill_price=None, status=OrderStatus.OPEN,
            timestamp=BAR_TS / 1000.0, fee=0.0, fee_currency='USDT')
        if executed > 0.0:
            filled = dataclasses.replace(
                order, filled_qty=executed, remaining_qty=0.0, average_fill_price=100.0,
                status=OrderStatus.FILLED)
            run.engine.on_order_event(OrderEvent(
                order=filled, event_type='filled', fill_price=100.0, fill_qty=executed,
                timestamp=BAR_TS / 1000.0, pine_id=intent.pine_id,
                from_entry=intent.pine_id if kind == 'close' else None,
                leg_type=LegType.CLOSE if kind == 'close' else LegType.ENTRY,
                fill_id=f"v-{run.venue_writes}", journal_fill_qty=executed))
        return order

    async def get_position(self, symbol: str) -> ExchangePosition | None:
        run = self.run
        assert run is not None
        if abs(run.venue) <= 1e-12:
            return None
        return ExchangePosition(
            symbol=SYMBOL, side='long' if run.venue > 0.0 else 'short', size=abs(run.venue),
            entry_price=100.0, unrealized_pnl=0.0, liquidation_price=None, leverage=1.0,
            margin_mode='cross')

    async def execute_entry(self, envelope: DispatchEnvelope):
        run = self.run
        assert run is not None
        run.on_venue_write('entry', envelope.intent)
        intent = envelope.intent
        run.venue = round(run.venue + (intent.qty if intent.side == 'buy' else -intent.qty), 12)
        return [self._write('entry', envelope, intent.qty)]

    async def execute_close(self, envelope: DispatchEnvelope):
        run = self.run
        assert run is not None
        run.on_venue_write('close', envelope.intent)
        intent = envelope.intent
        closes_long = intent.side == 'sell'
        held = run.venue if closes_long else -run.venue
        executed = round(min(intent.qty, max(held, 0.0)), 12)
        run.venue = round(run.venue - executed if closes_long else run.venue + executed, 12)
        return self._write('close', envelope, executed)


class _MatrixRun:
    """One generated case: the engines it runs and what they did on the entry's key."""

    def __init__(self, store, mode: str, action: str):
        self.store = store
        self.mode = mode
        self.action = action
        self.ctx = None
        self.engine: OrderSyncEngine | None = None
        self.broker: _MockBroker | None = None
        self.history: list = []
        self.fills: dict[str, float] = {}
        self.violations: list[str] = []
        self.writes: list[tuple[str, object]] = []
        self.leg: PartialBracketLeg | None = None
        self.intent: EntryIntent | None = None
        self.consumed_here = False
        self.restarts = 0
        self.seeded = 0.0
        self.swept_resolved = False
        #: The most executed the venue reported of the order on any event.
        self.reported = 0.0
        #: Whether the entry maps a sibling order.
        self.sibling = False
        #: What the book held of the sibling when it was mapped.
        self.sibling_seeded = 0.0
        #: The sibling's fills delivered, by fill id.
        self.sibling_fills: dict[str, float] = {}
        #: The most executed the venue reported of the sibling on any event.
        self.sibling_reported = 0.0
        #: Whether the sibling's first slice shows more executed than its
        #: terminal event reported (:data:`_MATRIX_SIBLING_RISEN`).
        self.sibling_rises = False
        #: The venue position of the venue actions (:class:`_MatrixVenueBroker`),
        #: and how many writes it took.
        self.venue = 0.0
        self.venue_writes = 0
        #: Whether a venue action went out while the book held less of the held
        #: order than it executed with no hold standing: a slice the venue had
        #: not announced (the evidence settles the hold on what it shows).
        self.sized_short = False

    # --- engines

    def _broker(self) -> _MockBroker:
        broker = (_MatrixVenueBroker(run=self) if self.action in _MATRIX_VENUE_ACTIONS
                  else _MockBroker())
        broker.capabilities = ExchangeCapabilities(
            partial_qty_bracket_exit=(CapabilityLevel.SOFTWARE if self.mode == 'software'
                                      else CapabilityLevel.NATIVE),
            short_selling=CapabilityLevel.NATIVE)
        return broker

    def on_venue_write(self, kind: str, intent) -> None:
        """Judge a write of a venue action against the hold, when it reaches the venue."""
        engine = self.engine
        assert engine is not None
        holds = engine._unresolved_execution_holds()
        expired = any(hold.expired for hold in holds.values())
        held_close = getattr(intent, 'synthetic_kind', None) == 'held_execution_close'
        if kind == 'entry' and intent.pine_id == 'short' and engine.quarantined:
            self.violations.append(f"the reversal's new side opened under the quarantine: {intent}")
        if not holds and engine._position.fill_cursor(['xchg-parent']) < self.truth - 1e-9:
            self.sized_short = True
        if held_close and not expired:
            self.violations.append(f"a venue close over a hold that did not expire: {intent}")
        elif holds and not held_close and not expired:
            self.violations.append(
                f"{kind} sized from the book while the execution is unresolved: {intent}")

    def start(self) -> None:
        self.ctx = self.store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        self.broker = self._broker()
        self.engine = _mk_engine(self.broker, store_ctx=self.ctx)
        oca = None
        if self.action == 'oca_cancel':
            oca = 'cancel'
        elif self.action == 'oca_reduce':
            oca = 'reduce'
        intent = _seed_parent_intent_and_leg(self.engine, with_leg=self.mode == 'software')
        if oca is not None:
            intent = dataclasses.replace(intent, oca_name='grp', oca_type=oca)
            self.engine._active_intents[intent.intent_key] = intent
            self.engine._envelopes[intent.intent_key] = dataclasses.replace(
                self.engine._envelopes[intent.intent_key], intent=intent)
        self.intent = intent
        if self.mode == 'software':
            self.leg = next(iter(self.engine._partial_bracket_engine.iter_legs()))
            self.leg.parent_entry_dispatch_ref = self.engine._resolve_parent_opening_ref('parent')
            # The entry's pending partial-bracket leg, as the journal persisted it.
            self.ctx.upsert_order(
                'leg-1', symbol=SYMBOL, side='sell', qty=1.0, filled_qty=0.0,
                state=STATE_PARTIAL_BRACKET_LEG, intent_key='exit_tp\x00parent',
                from_entry='parent', pine_entry_id='exit_tp',
                extras={
                    EXTRAS_KEY_LEG_KIND: LEG_KIND_TP_PARTIAL,
                    EXTRAS_KEY_LEG_STATE: LEG_STATE_PENDING_ENTRY,
                    EXTRAS_KEY_PARENT_PINE_ENTRY_ID: 'parent',
                    EXTRAS_KEY_PARENT_ENTRY_DISPATCH_REF: self.leg.parent_entry_dispatch_ref,
                    EXTRAS_KEY_INTENT_PARTIAL_QTY: 1.0,
                    EXTRAS_KEY_TRIGGER_LEVEL: self.leg.trigger_level,
                },
            )
            # The script's strategy.exit bracket of the entry.
            self.engine._seed_partial_bracket_exit_from_legs('exit_tp\x00parent', [self.leg])
        self._probe()

    def restart(self) -> None:
        assert self.ctx is not None and self.engine is not None
        # The journal's envelope anchor of the entry (its dispatch row).
        envelope = self.engine._envelopes.get('parent')
        anchor = (EnvelopeRecord(key='parent', bar_ts_ms=envelope.bar_ts_ms,
                                 retry_seq=envelope.retry_seq, run_tag=RUN_TAG)
                  if envelope is not None
                  else self.engine._persisted_envelope_anchors.get('parent'))
        self.ctx.close()
        self.restarts += 1
        self.leg = None
        self.consumed_here = False
        self.ctx = self.store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        self.broker = self._broker()
        self.engine = _mk_engine(self.broker, store_ctx=self.ctx)
        self.engine._current_bar_ts_ms = BAR_TS
        if anchor is not None:
            self.engine._persisted_envelope_anchors.setdefault('parent', anchor)
        if self.engine._pending_cancel_tentative_rehydrate:
            # The first sync re-arms the cancel-tentative state of replayed legs.
            self.engine._pending_cancel_tentative_rehydrate = False
            self.engine._rehydrate_cancel_tentative_from_replayed_legs()
        legs = list(self.engine._partial_bracket_engine.iter_legs())
        self.leg = legs[0] if legs else None
        if legs and not self.engine._position.exit_orders:
            # The script's strategy.exit bracket, issued again after the restart.
            self.engine._seed_partial_bracket_exit_from_legs('exit_tp\x00parent', legs)
        journaled = sum(self.fills.values())
        self.seeded = journaled
        if journaled > 0.0:
            # The startup takes in the fills the plugin journaled.
            self.engine._position.seed_fill_cursors({'xchg-parent': journaled})
        sibling_journaled = self.sibling_seeded + sum(self.sibling_fills.values())
        if self.sibling and sibling_journaled > 0.0:
            self.engine._position.seed_fill_cursors({_MATRIX_SIBLING_ORDER: sibling_journaled})
        self._probe()

    def add_sibling(self, booked: float = _MATRIX_SIBLING_BOOKED) -> None:
        """Map the sibling order to the entry, ``booked`` of it taken in by the book."""
        engine = self.engine
        assert engine is not None
        self.sibling = True
        self.sibling_seeded = booked
        engine._order_mapping['parent'].append(_MATRIX_SIBLING_ORDER)
        if booked > 0.0:
            engine._position.seed_fill_cursors({_MATRIX_SIBLING_ORDER: booked})

    def resume(self) -> None:
        """The hold expires, the operator resumes the quarantine, the bot restarts.

        A resume states nothing about the entry: the restarted bot keeps the
        hold and recovers the execution from the venue itself.
        """
        engine, ctx = self.engine, self.ctx
        assert engine is not None and ctx is not None
        engine._expire_fill_evidence_holds(now_ms=int(time.time() * 1000) + 10 ** 9)
        engine._flush_owed_store_writes()
        quarantine_id = engine.quarantine_id
        if quarantine_id is not None:
            self.store.request_quarantine_resume(ctx.run_id, quarantine_id)
        self.restart()

    def arm_death(self) -> None:
        """Kill the process right after its next store write landed.

        From the death on no store span opens, so nothing the dying process
        runs afterwards lands.
        """
        store = self.store
        dead: list[bool] = []
        depth: list[int] = [0]

        @contextlib.contextmanager
        def transaction():
            if dead:
                raise _ProcessDeath
            depth[0] += 1
            try:
                with BrokerStore.transaction(store) as conn:
                    yield conn
            finally:
                depth[0] -= 1
            if depth[0] == 0:
                dead.append(True)

        store.transaction = transaction  # type: ignore[method-assign]

    def revive(self) -> None:
        """The dead process's bot starts again on the store it left."""
        vars(self.store).pop('transaction', None)
        self.restart()

    def unresolved(self) -> bool:
        """Whether the entry's execution is unknown, or reported and not booked."""
        engine = self.engine
        assert engine is not None
        return ('parent' in engine._entries_awaiting_fill_evidence
                or 'parent' in engine._settled_fill_evidence
                or 'parent' in engine._positive_evidence_awaiting_intent
                or engine._entry_execution_awaiting_booking('parent'))

    def _probe(self) -> None:
        engine = self.engine
        broker = self.broker
        assert engine is not None and broker is not None

        def write(kind: str, intent) -> None:
            if intent.intent_key != 'parent':
                return
            self.writes.append((kind, intent))
            if self.unresolved():
                self.violations.append(f"{kind} of the held entry: {intent}")
            elif (kind in ('new', 'modify') and engine._position.fill_cursor(['xchg-parent'])
                    < self.reported - 1e-9):
                self.violations.append(
                    f"{kind} while {self.reported} reported executed is not all booked: {intent}")
            elif (kind in ('new', 'modify')
                    and engine._position.fill_cursor([_MATRIX_SIBLING_ORDER])
                    < self.sibling_reported - 1e-9):
                self.violations.append(
                    f"{kind} while the sibling's reported {self.sibling_reported} is not all "
                    f"booked: {intent}")
            elif (kind == 'new' and self.consumed_here and isinstance(intent, EntryIntent)
                    and intent == self.intent and self.action == 'keep'):
                self.violations.append(f"resend of the executed entry: {intent}")

        real_new = engine._dispatch_new
        real_modify = engine._dispatch_modify

        def reaches_venue(intent) -> bool:
            # The venue actions' own intents and the closes the engine makes
            # for them; the entry and its bracket stay recorded only.
            return self.action in _MATRIX_VENUE_ACTIONS and (
                intent.intent_key in ('short', '')
                or getattr(intent, 'synthetic_kind', None) is not None)

        def venue_new(intent, **kwargs) -> None:
            write('new', intent)
            if reaches_venue(intent):
                real_new(intent, **kwargs)

        def venue_modify(old, new) -> None:
            write('modify', new)
            if reaches_venue(new):
                real_modify(old, new)

        engine._dispatch_new = venue_new  # type: ignore[method-assign]
        engine._dispatch_modify = venue_modify  # type: ignore[method-assign]
        real_cancel = broker.execute_cancel
        real_cancel_outcome = broker.execute_cancel_with_outcome

        async def execute_cancel(envelope):
            write('cancel', envelope.intent)
            return await real_cancel(envelope)

        async def execute_cancel_with_outcome(envelope):
            # The cancel-tentative probe is the venue query that resolves a
            # tentative parent; it is not a write that changes the order.
            self.writes.append(('cancel_probe', envelope.intent))
            return await real_cancel_outcome(envelope)

        broker.execute_cancel = execute_cancel  # type: ignore[method-assign]
        broker.execute_cancel_with_outcome = execute_cancel_with_outcome  # type: ignore[method-assign]
        real_retire = engine._retire_native_failsafe_for_entry

        def retire(key, *args, **kwargs):
            if key == 'parent' and self.truth > 0.0 and self.unresolved():
                self.violations.append("native fail-safe retired for an executed entry")
            return real_retire(key, *args, **kwargs)

        engine._retire_native_failsafe_for_entry = retire  # type: ignore[method-assign]

    truth: float = 0.0

    # --- venue and strategy

    def deliver(self, events: list) -> None:
        """Hand ``events`` to the engine.

        A ``None`` drains what was handed so far, :data:`_MATRIX_RESTART`
        restarts the bot with what it received still undrained.
        """
        assert self.engine is not None
        for event in events:
            if event is None:
                try:
                    self.engine._drain_events()
                except sqlite3.Error:
                    # The event stays queued for the next sync's drain.
                    pass
                continue
            if event == _MATRIX_RESTART:
                self.engine._flush_owed_store_writes()
                self.restart()
                continue
            self.history.append(event)
            sibling = event.order.id == _MATRIX_SIBLING_ORDER
            if sibling:
                self.sibling_reported = max(self.sibling_reported, event.order.filled_qty)
            else:
                self.reported = max(self.reported, event.order.filled_qty)
            if (event.fill_qty or 0.0) > 0.0 and event.fill_id is not None:
                (self.sibling_fills if sibling else self.fills)[event.fill_id] = event.fill_qty
            self.engine.on_order_event(event)

    def intents(self, step: int) -> list:
        """The script's intents on sync ``step``: the action's entry, and its bracket."""
        engine = self.engine
        assert engine is not None
        exits = [exit_intent for exit_intent in build_intents(
            {}, engine._position.exit_orders, SYMBOL, engine._position.bindings)
            if isinstance(exit_intent, ExitIntent)]
        return self._entry_intents(step) + exits

    def _entry_intents(self, step: int) -> list:
        intent = self.intent
        assert intent is not None
        action = self.action
        if action == 'keep':
            return [intent]
        if action in ('remove', 'oca_cancel', 'oca_reduce', 'close_sweep'):
            return [intent] if step == 0 else []
        if action == 'qty':
            return [dataclasses.replace(intent, qty=1.1)]
        if action == 'price':
            return [dataclasses.replace(intent, order_type=OrderType.LIMIT, limit=99.0)]
        if action == 'readd':
            return [] if step == 0 else [intent]
        if action == 'reverse':
            return [intent, EntryIntent(pine_id='short', symbol=SYMBOL, side='sell', qty=1.0,
                                        order_type=OrderType.MARKET)]
        if action == 'close_all':
            assert self.engine is not None
            booked = self.engine._position.size
            if booked <= 0.0:
                # Pine closes nothing of a flat book.
                return [intent]
            return [intent, CloseIntent(pine_id='', symbol=SYMBOL, side='sell', qty=booked)]
        assert action == 'close'
        return [CloseIntent(pine_id='parent', symbol=SYMBOL, side='sell', qty=1.0)]

    def strategy_side_effect(self) -> None:
        """The OCA sibling's fill and the close sweep of the action's first sync."""
        engine = self.engine
        assert engine is not None
        if self.action in ('oca_cancel', 'oca_reduce'):
            sibling = EntryIntent(pine_id='sib', symbol=SYMBOL, side='buy', qty=1.0,
                                  order_type=OrderType.LIMIT, limit=95.0, oca_name='grp',
                                  oca_type='cancel' if self.action == 'oca_cancel' else 'reduce')
            engine._active_intents['sib'] = sibling
            if self.action == 'oca_cancel':
                engine._cascade_oca_cancel_for_key('sib')
            else:
                event = dataclasses.replace(
                    _ct_order_event('partial', filled_qty=0.3, fill_qty=0.3, fill_id='s-1'),
                    pine_id='sib')
                engine._cascade_oca_reduce(event)
            engine._active_intents.pop('sib', None)
        elif self.action == 'close_sweep':
            # Once the execution is booked the sweep may retire the protection
            # (this test sweeps an open position).
            self.swept_resolved = not self.unresolved()
            engine._cleanup_position_tracking('parent')

    def sync(self, step: int, *, queued: bool = False, side_effect: bool = False) -> None:
        engine = self.engine
        assert engine is not None
        for _attempt in range(3):
            try:
                if not queued:
                    engine._drain_events()
                engine._retry_forced_cancels()
                # The cancel-retry probe, past its stale grace.
                engine._drive_cancel_tentative(now_ms=int(time.time() * 1000) + 120_000)
                engine._diff_and_dispatch(self.intents(step))
                if side_effect:
                    self.strategy_side_effect()
                    side_effect = False
                engine._drain_events()
                # The engine's reconcile: the expiry, the bot's own recovery
                # and the end of the causes resolved.
                now_ms = int(time.time() * 1000)
                engine._expire_fill_evidence_holds(now_ms=now_ms)
                engine._recover_held_executions(now_ms=now_ms)
                engine._resolve_entry_block_causes(now_ms=now_ms)
                engine._flush_owed_store_writes()
                break
            except sqlite3.Error:
                queued = False
        self.check_protection()
        if (self.truth > 0.0 and isinstance(engine._active_intents.get('parent'), EntryIntent)
                and 'parent' in engine._truncated_entries):
            self.consumed_here = True

    def check_protection(self) -> None:
        engine = self.engine
        assert engine is not None
        booked = engine._position.fill_cursor(['xchg-parent'])
        if booked > self.truth + 1e-9:
            self.violations.append(f"booked {booked} of an execution of {self.truth}")
        if self.leg is not None and self.truth > 0.0:
            if self.leg.leg_state == LEG_STATE_ABORTED_PARENT_NEVER_ARRIVED:
                self.violations.append("the executed entry's leg aborted as never arrived")

    def finish_evidence(self) -> list:
        """The venue's remaining evidence of what the order executed."""
        if self.truth <= 0.0:
            return [dataclasses.replace(
                _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None),
                cancel_reason=CANCEL_REASON_VENUE_POSITION_CLEARED)]
        remaining = round(self.truth - sum(self.fills.values()), 12)
        if remaining <= 0.0:
            return []
        return [_ct_order_event('partial', filled_qty=self.truth, fill_qty=remaining,
                                fill_id='x-rest')]

    def finish_sibling_evidence(self) -> list:
        """The sibling's remaining fill, delivered a sync after the held order's.

        A rising sibling first delivers a 0.2 slice showing
        :data:`_MATRIX_SIBLING_RISEN` executed, then the rest a sync later.
        """
        if self.sibling_rises and not self.sibling_fills:
            return [_matrix_sibling_event('partial', filled_qty=_MATRIX_SIBLING_RISEN,
                                          fill_qty=_MATRIX_SIBLING_REPORTED,
                                          fill_id='sib-1')]
        remaining = round(self.sibling_reported - sum(self.sibling_fills.values()), 12)
        if remaining <= 0.0:
            return []
        return [_matrix_sibling_event('partial', filled_qty=self.sibling_reported,
                                      fill_qty=remaining, fill_id='sib-rest')]


def _matrix_sibling_event(event_type: str, *, filled_qty: float, fill_qty: float | None,
                          fill_id: str | None = None):
    """An event of the sibling order :data:`_MATRIX_SIBLING_ORDER` of the entry."""
    event = _ct_order_event(event_type, filled_qty=filled_qty, fill_qty=fill_qty,
                            fill_id=fill_id)
    if event_type == 'cancelled':
        event = dataclasses.replace(event, cancel_reason=CANCEL_REASON_VENUE_POSITION_CLEARED)
    return dataclasses.replace(event, order=dataclasses.replace(event.order,
                                                                id=_MATRIX_SIBLING_ORDER))


def _matrix_arrival(run: _MatrixRun, arrival: str) -> list:
    if arrival == 'slice':
        return [_ct_order_event('partial', filled_qty=0.2, fill_qty=0.2, fill_id='x-1')]
    if arrival == 'cum_slice':
        # A 0.2 slice whose order already shows 0.4 executed: the other slice
        # is still to come.
        return [_ct_order_event('partial', filled_qty=0.4, fill_qty=0.2, fill_id='x-1')]
    if arrival == 'cum_rise':
        # Two 0.1 slices whose order shows more executed each time (0.2, then
        # 0.4): the second raises what the book must hold.
        return [_ct_order_event('partial', filled_qty=0.2, fill_qty=0.1, fill_id='x-1'),
                _ct_order_event('partial', filled_qty=0.4, fill_qty=0.1, fill_id='x-2')]
    if arrival == 'terminal_rise':
        # A terminal event reporting 0.2 as a lower bound of the executed
        # quantity, routed before the venue's next terminal event reports 0.4,
        # then a 0.2 slice: the second report raises what the book must hold
        # after the first unmapped the order.
        lower = dataclasses.replace(
            _hold_positive_terminal('rejected'),
            order=dataclasses.replace(_hold_positive_terminal('rejected').order,
                                      filled_qty=0.2, filled_qty_known=False))
        return [lower, None, _hold_positive_terminal('rejected'),
                _ct_order_event('partial', filled_qty=0.2, fill_qty=0.2, fill_id='x-1')]
    if arrival == 'queued_rise':
        # The same lower bound of 0.2 routed, then a 0.2 slice and the venue's
        # next terminal event reporting 0.4 received together, in that order:
        # the slice reaches what the hold awaits before the higher report is
        # routed.
        lower = dataclasses.replace(
            _hold_positive_terminal('rejected'),
            order=dataclasses.replace(_hold_positive_terminal('rejected').order,
                                      filled_qty=0.2, filled_qty_known=False))
        return [lower, None,
                _ct_order_event('partial', filled_qty=0.2, fill_qty=0.2, fill_id='x-1'),
                _hold_positive_terminal('rejected')]
    if arrival in ('queued_cum', 'queued_cum_bound'):
        # A 0.2 slice showing 0.2 executed and a 0.1 slice showing 0.4
        # executed received together, behind the lower bound of 0.2 routed
        # first (``queued_cum_bound``) or with nothing reported: the first
        # slice reaches everything routed so far, the second one shows more
        # executed than the two slices book (the rest comes a sync later).
        slices = [_ct_order_event('partial', filled_qty=0.2, fill_qty=0.2, fill_id='x-1'),
                  _ct_order_event('partial', filled_qty=0.4, fill_qty=0.1, fill_id='x-2')]
        if arrival == 'queued_cum':
            return slices
        lower = dataclasses.replace(
            _hold_positive_terminal('rejected'),
            order=dataclasses.replace(_hold_positive_terminal('rejected').order,
                                      filled_qty=0.2, filled_qty_known=False))
        return [lower, None, *slices]
    if arrival == 'stale_restart':
        # The same lower bound of 0.2 routed, then the venue's terminal event
        # reporting 0.4 and a stale repeat of the 0.2 lower bound received
        # together; the bot restarts before draining them, then a 0.2 slice
        # arrives: the restart must still await the 0.4.
        lower = dataclasses.replace(
            _hold_positive_terminal('rejected'),
            order=dataclasses.replace(_hold_positive_terminal('rejected').order,
                                      filled_qty=0.2, filled_qty_known=False))
        return [lower, None, _hold_positive_terminal('rejected'), lower, _MATRIX_RESTART,
                _ct_order_event('partial', filled_qty=0.2, fill_qty=0.2, fill_id='x-1')]
    if arrival == 'fill':
        return [_ct_order_event('partial', filled_qty=0.4, fill_qty=0.4, fill_id='x-full')]
    if arrival == 'zero':
        return [dataclasses.replace(
            _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None),
            cancel_reason=CANCEL_REASON_VENUE_POSITION_CLEARED)]
    if arrival == 'positive':
        return [_hold_positive_terminal()]
    if arrival == 'unknown':
        return [_ct_unknown_execution(_ct_order_event('cancelled', filled_qty=0.0, fill_qty=None))]
    if arrival == 'duplicate':
        return list(run.history)
    return []


def _matrix_enter_state(run: _MatrixRun, state: str, crash: str) -> None:
    engine = run.engine
    assert engine is not None
    if state in _MATRIX_UNHELD_STATES:
        if state == 'terminal_sibling':
            run.add_sibling()
        elif state != 'terminal':
            # The strategy's cancel of the entry timed out, its grace spent.
            engine._cancel_disposition_pending['parent'] = _CancelTentativeMeta(
                since_ts_ms=int(time.time() * 1000) - 3_600_000, reason='broker_timeout',
                retry_count=1)
            engine._partial_bracket_engine.mark_legs_cancel_tentative(
                'parent', reason='broker_timeout', now_ms=BAR_TS)
        run.deliver([_hold_positive_terminal(
            'rejected' if state == 'tentative_rejected' else 'cancelled')])
        if state in ('terminal', 'terminal_sibling'):
            run.consumed_here = True
        if crash == 'torn':
            run.arm_death()
            try:
                engine._drain_events()
                engine._flush_owed_store_writes()
            except _ProcessDeath:
                pass
            run.revive()
            return
        engine._drain_events()
        engine._flush_owed_store_writes()
        return
    if state == 'awaiting_sibling':
        run.add_sibling()
    elif state in ('reported_sibling', 'rising_sibling'):
        run.add_sibling(booked=0.0)
        run.sibling_rises = state == 'rising_sibling'
    unknown = _ct_unknown_execution(_ct_order_event('cancelled', filled_qty=0.0, fill_qty=None))
    run.history.append(unknown)
    engine._route_event(unknown)
    assert 'parent' in engine._entries_awaiting_fill_evidence
    if state == 'restarted':
        run.restart()
    elif state == 'tentative':
        # The strategy's cancel of the entry timed out before the venue's
        # cancellation arrived: HELD and cancel-tentative, its grace spent.
        engine._cancel_disposition_pending['parent'] = _CancelTentativeMeta(
            since_ts_ms=int(time.time() * 1000) - 3_600_000, reason='broker_timeout',
            retry_count=1)
        engine._partial_bracket_engine.mark_legs_cancel_tentative(
            'parent', reason='broker_timeout', now_ms=BAR_TS)
    elif state == 'received':
        run.deliver([_hold_positive_terminal()])
    elif state == 'parked':
        run.restart()
        run.deliver([_hold_positive_terminal()])
        run.engine._drain_events()
        assert 'parent' in run.engine._positive_evidence_awaiting_intent
    elif state in ('reported_sibling', 'rising_sibling'):
        # Both orders of the entry end at the venue reporting executions the
        # book holds none of: 0.2 of the sibling, 0.4 of the held order.
        run.deliver([_matrix_sibling_event('cancelled', filled_qty=_MATRIX_SIBLING_REPORTED,
                                           fill_qty=None),
                     _hold_positive_terminal()])
        engine._drain_events()
        engine._flush_owed_store_writes()
        assert engine._entry_execution_awaiting_booking('parent')
        run.consumed_here = True
    elif state in ('awaiting', 'awaiting_sibling'):
        run.deliver([_hold_positive_terminal()])
        engine._drain_events()
        engine._flush_owed_store_writes()
        assert engine._entry_execution_awaiting_booking('parent')
        run.consumed_here = True


def _matrix_self_recovery(run: '_MatrixRun', truth: float) -> None:
    """The bot gets out of an expired hold by itself: it reads the held order from the venue.

    The venue's order history shows what the order executed (``truth``). The
    hold ends with no human: the execution is booked exactly once (or the
    entry is released as unexecuted), the book never goes short, and the
    hold's quarantine cause ends exactly when the venue holds what the book
    holds.
    """
    engine = run.engine
    assert engine is not None
    order = ExchangeOrder(
        id='xchg-parent', symbol=SYMBOL, side='buy', order_type=OrderType.MARKET, qty=1.0,
        filled_qty=truth, remaining_qty=round(1.0 - truth, 12), price=None, stop_price=None,
        average_fill_price=100.0 if truth > 0.0 else None, status=OrderStatus.CANCELLED,
        timestamp=BAR_TS / 1000.0, fee=0.0, fee_currency='USDT')

    async def fetch_order_execution(order_id, client_order_id):
        return OrderExecution(order=order)

    run.broker.fetch_order_execution = fetch_order_execution  # type: ignore[attr-defined]
    now = int(time.time() * 1000)
    for step in range(3):
        _recover(engine, now + step * int(_VENUE_RECOVERY_CAP_S * 1000))
    assert not engine._entry_execution_unresolved('parent'), "the bot did not end its own hold"
    assert engine._position.fill_cursor(['xchg-parent']) == pytest.approx(truth)
    assert engine._position.size >= -1e-9
    venue = run.venue if isinstance(run.broker, _MatrixVenueBroker) else 0.0
    matches = abs(venue - engine._position.size) <= 1e-9
    if engine.quarantine_kind is QuarantineKind.ENTRY_BLOCK or not engine.quarantined:
        assert ('fill_evidence:parent' in engine._entry_block_causes) is not matches, (
            f"venue {venue}, book {engine._position.size}")


@pytest.mark.parametrize('state,arrival,action,mode,timing,crash,truth', _matrix_cases())
def __test_the_fill_evidence_hold_matrix__(
        tmp_path, state, arrival, action, mode, timing, crash, truth):
    """The fill-evidence hold over state x arrival x strategy x mode x timing x crash.

    Invariants, whatever the combination: no order is sent, amended or
    cancelled on the entry's key while its execution is unknown or reported
    and not booked; the executed quantity is booked exactly once; the
    pending protection of an executed entry is never aborted as never
    arrived and its native fail-safe never retired; an executed entry the
    strategy keeps is never sent again; the hold ends with the evidence
    applied (no hold, settlement, parked event or awaited execution in
    memory, no hold or arrival row in the store) or in the expiry
    quarantine, never in a halt.

    The venue actions (a market reversal under another id, a close_all)
    reach the venue only against the true book: nothing is closed or opened
    while the execution is unresolved and the hold stands; once the evidence
    settles the reversal ends 1.0 short (TradingView) and the close_all flat.
    Over an expired hold a close the strategy requested is never left open by
    the quarantine (the venue is flat after the next sync), the reversal's
    new side never opens under it, and the book never goes short. The bot
    then gets out of the expired hold by itself (:func:`_matrix_self_recovery`).
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        run = _MatrixRun(store, mode, action)
        run.truth = truth
        # What the venue executed: the held order's truth and the sibling's fills.
        run.venue = round(truth + _MATRIX_SIBLING_VENUE.get(state, 0.0), 12)
        run.start()
        _matrix_enter_state(run, state, crash)
        engine = run.engine
        assert engine is not None

        if crash == 'storage_error':
            real_route = engine._route_order_event
            raised: list[bool] = []

            def route_once(event):
                if not raised:
                    raised.append(True)
                    raise sqlite3.OperationalError("disk I/O error")
                real_route(event)

            engine._route_order_event = route_once  # type: ignore[method-assign]

        run.deliver(_matrix_arrival(run, arrival))
        if crash == 'torn' and state not in _MATRIX_UNHELD_STATES:
            run.arm_death()
            try:
                run.sync(0, queued=timing == 'queued', side_effect=True)
            except _ProcessDeath:
                pass
            run.revive()
        else:
            run.sync(0, queued=timing == 'queued', side_effect=True)
        if crash in ('restart', 'restart2'):
            run.restart()
        elif crash == 'resume':
            run.resume()
        run.sync(1)

        finish = run.finish_evidence()
        run.deliver(finish)
        if crash == 'restart2':
            run.restart()
        for step in range(2, 5):
            run.sync(step)
            if step in (2, 3):
                run.deliver(run.finish_sibling_evidence())

        engine = run.engine
        assert engine is not None
        ctx = run.ctx
        assert not run.violations, run.violations
        booked = engine._position.fill_cursor(['xchg-parent'])
        assert booked == pytest.approx(truth), f"booked {booked} of {truth}"
        clean = (engine._entries_awaiting_fill_evidence == {}
                 and engine._settled_fill_evidence == {}
                 and engine._fill_evidence_awaiting_execution == {}
                 and engine._positive_evidence_awaiting_intent == {}
                 and ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND) == {}
                 and ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_ARRIVAL_EVENT_KIND) == {})
        if not clean:
            assert 'parent' in engine._entries_awaiting_fill_evidence, (
                "the hold neither ended nor stands for its expiry",
                engine._settled_fill_evidence, engine._fill_evidence_awaiting_execution,
                list(engine._positive_evidence_awaiting_intent),
                ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND),
                list(ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_ARRIVAL_EVENT_KIND)))
            engine._expire_fill_evidence_holds(now_ms=int(time.time() * 1000) + 10 ** 9)
            assert engine.quarantined, "a standing hold did not quarantine at its expiry"
            if action in _MATRIX_VENUE_ACTIONS:
                # The strategy keeps asking to close or reverse over the expired
                # hold: the quarantine must not leave that position open.
                asked = action == 'reverse' or engine._position.size > 0.0
                run.sync(5)
                assert not run.violations, run.violations
                if asked and not engine.venue_writes_blocked:
                    assert abs(run.venue) <= 1e-9, (
                        f"a close the strategy requested is left open by the quarantine: "
                        f"the venue holds {run.venue}")
                assert engine._position.size >= -1e-9, "the book went short over the hold"
            _matrix_self_recovery(run, truth)
        elif crash != 'resume' or truth == 0.0:
            # An executed entry's expired hold re-quarantines after the resume
            # and its cause ends once the venue matches the book (the matrix
            # venue of the non-venue actions holds nothing).
            assert not engine.quarantined
        # A new cycle of the entry the strategy changed after the evidence was
        # applied re-binds the bracket to it.
        new_cycle = (any(kind == 'new' for kind, _intent in run.writes)
                     or run.swept_resolved
                     # A venue action that closed the position retires its protection.
                     or (action in _MATRIX_VENUE_ACTIONS and engine._position.size <= 1e-9))
        if (run.leg is not None and truth > 0.0 and booked > run.seeded + 1e-9
                and not new_cycle):
            # The engine that routed the execution armed its protection.
            assert run.leg.leg_state == LEG_STATE_ARMED
        if run.leg is not None and truth > 0.0 and not new_cycle:
            assert run.leg.leg_state in (LEG_STATE_ARMED, LEG_STATE_PENDING_ENTRY)
        if (clean and action in _MATRIX_VENUE_ACTIONS and state not in _MATRIX_SIBLING_VENUE
                and crash in ('none', 'storage_error')):
            # (The harness's restart rebuilds no book from the venue, so the
            # venue is compared only within one process.) The book tracks the
            # venue; once the evidence settled, the request went out against
            # the true book: a reversal ends 1.0 short (TradingView), only
            # closed under a quarantine that blocks its new side; a close_all
            # ends flat. A slice the venue never announced can only be booked
            # when it arrives, after the request went out.
            assert engine._position.size == pytest.approx(run.venue), (
                f"book {engine._position.size}, venue {run.venue}")
            if not run.sized_short:
                expected = -1.0 if action == 'reverse' and not engine.quarantined else 0.0
                assert engine._position.size == pytest.approx(expected), (
                    f"book {engine._position.size}, venue {run.venue}")


def __test_a_held_tentative_entry_waits_for_its_hold_expiry_instead_of_halting__():
    """A SOFTWARE entry both held and cancel-tentative never halts on inconclusive probes.

    The strategy's cancel timed out and the venue's cancellation came without
    the executed quantity. Every cancel-retry probe past the stale grace
    answers ``UNKNOWN``: the bot is not halted, the hold stands, and its own
    expiry quarantines it (exits keep working) instead.
    """
    engine = _mk_engine(_MockBroker())
    _seed_parent_intent_and_leg(engine)
    engine._route_event(_ct_unknown_execution(
        _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))
    engine._cancel_disposition_pending['parent'] = _CancelTentativeMeta(
        since_ts_ms=int(time.time() * 1000) - 3_600_000, reason='broker_timeout',
        retry_count=1)
    engine._partial_bracket_engine.mark_legs_cancel_tentative(
        'parent', reason='broker_timeout', now_ms=BAR_TS)

    for step in range(3):
        engine._drive_cancel_tentative(now_ms=int(time.time() * 1000) + 120_000 + step)

    assert engine.quarantined is False
    assert 'parent' in engine._entries_awaiting_fill_evidence
    engine._expire_fill_evidence_holds(now_ms=int(time.time() * 1000) + 10 ** 9)
    assert engine.quarantined


def __test_a_fill_slice_showing_more_executed_keeps_the_dispatch_barrier__():
    """A 0.2 slice of a held order that shows 0.4 executed is a lower bound, not the total.

    Nothing reported the venue's executed quantity for the held entry. The
    first slice books 0.2 while the order shows 0.4 executed: the entry waits
    in ``AWAITING_EXECUTION``, so a changed entry sends nothing until the
    second slice is booked.
    """
    engine = _mk_engine(_MockBroker())
    intent = _seed_parent_intent_and_leg(engine, with_leg=False)
    sent: list = []

    def record_new(new_intent) -> None:
        sent.append(new_intent)

    engine._dispatch_new = record_new  # type: ignore[method-assign]
    engine._route_event(_ct_unknown_execution(
        _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))
    engine.on_order_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.2, fill_id='x-1'))
    engine._drain_events()

    assert engine._entry_execution_awaiting_booking(intent.intent_key)
    changed = dataclasses.replace(intent, qty=1.1)
    engine._diff_and_dispatch([changed])
    assert sent == []

    engine.on_order_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.2, fill_id='x-2'))
    engine._drain_events()
    assert not engine._entry_execution_unresolved(intent.intent_key)
    assert engine._position.size == pytest.approx(0.4)
    engine._diff_and_dispatch([changed])
    assert sent and sent[0] == changed


def __test_an_awaited_execution_that_never_arrives_expires_into_the_quarantine__():
    """``AWAITING_EXECUTION`` past the fill-report grace quarantines the bot.

    The held entry's venue cancel reports 0.4 executed and its fill never
    arrives: the expiry quarantines the bot (it is not left blocked silently),
    and the hold stays until the fill is booked.
    """
    engine = _mk_engine(_MockBroker())
    intent = _seed_parent_intent_and_leg(engine, with_leg=False)
    engine._route_event(_ct_unknown_execution(
        _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))
    engine.on_order_event(_hold_positive_terminal())
    engine._drain_events()
    assert engine._entry_execution_awaiting_booking(intent.intent_key)

    engine._expire_fill_evidence_holds(now_ms=int(time.time() * 1000) + 10 ** 9)

    assert engine.quarantined
    assert engine._entry_execution_awaiting_booking(intent.intent_key)
    engine._route_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.4, fill_id='x-1'))
    assert not engine._entry_execution_unresolved(intent.intent_key)


def __test_booked_parked_evidence_releases_the_hold_without_a_reissue__(tmp_path):
    """A parked positive event whose execution the book holds ends the hold.

    After a restart the held entry's venue cancel reporting 0.4 executed waits
    for the script's re-issue, which never comes; the 0.4 fill is booked. The
    hold ends with no quarantine, no parked event and no mapping of the ended
    order left, so a later re-issue cannot adopt it as a working order.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine = _mk_engine(_MockBroker(), store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        engine._route_event(_ct_unknown_execution(
            _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))
        ctx.close()

        ctx2 = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine2 = _mk_engine(_MockBroker(), store_ctx=ctx2)
        engine2._current_bar_ts_ms = BAR_TS
        engine2.on_order_event(_hold_positive_terminal())
        engine2._drain_events()
        assert intent.intent_key in engine2._positive_evidence_awaiting_intent

        engine2.on_order_event(
            _ct_order_event('partial', filled_qty=0.4, fill_qty=0.4, fill_id='x-1'))
        engine2._drain_events()
        engine2._expire_fill_evidence_holds(now_ms=int(time.time() * 1000) + 10 ** 9)
        engine2._flush_owed_store_writes()

        assert engine2._position.size == pytest.approx(0.4)
        assert not engine2._entry_execution_unresolved(intent.intent_key)
        assert intent.intent_key not in engine2._order_mapping
        assert ctx2.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND) == {}
        assert ctx2.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_ARRIVAL_EVENT_KIND) == {}
        assert not engine2.quarantined and engine2.quarantined is False


def __test_a_later_fill_showing_more_executed_raises_the_awaited_execution__(tmp_path):
    """Each fill's cumulative executed quantity raises what the hold waits for.

    The held entry's first 0.2 slice shows 0.4 executed, its second 0.2
    slice shows 0.6: the book holds 0.4, so the hold still waits (in memory
    and in the stored hold) and a changed entry sends nothing until the
    third slice is booked.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine = _mk_engine(_MockBroker(), store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        sent: list = []

        def record_new(new_intent) -> None:
            sent.append(new_intent)

        engine._dispatch_new = record_new  # type: ignore[method-assign]
        engine._route_event(_ct_unknown_execution(
            _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))
        engine.on_order_event(
            _ct_order_event('partial', filled_qty=0.4, fill_qty=0.2, fill_id='x-1'))
        engine.on_order_event(
            _ct_order_event('partial', filled_qty=0.6, fill_qty=0.2, fill_id='x-2'))
        engine._drain_events()
        engine._flush_owed_store_writes()

        assert engine._entry_execution_awaiting_booking(intent.intent_key)
        stored = ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND)
        assert stored[intent.intent_key]['awaiting_execution'] == ['xchg-parent', 0.6]
        changed = dataclasses.replace(intent, qty=1.1)
        engine._diff_and_dispatch([changed])
        assert sent == []

        engine.on_order_event(
            _ct_order_event('partial', filled_qty=0.6, fill_qty=0.2, fill_id='x-3'))
        engine._drain_events()
        engine._flush_owed_store_writes()
        assert not engine._entry_execution_unresolved(intent.intent_key)
        assert engine._position.size == pytest.approx(0.6)
        assert ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND) == {}
        engine._diff_and_dispatch([changed])
        assert sent and sent[0] == changed


def __test_a_venue_record_contradicting_a_reported_execution_releases_nothing__(tmp_path):
    """A record of no such order cannot undo an execution the venue already reported.

    A 0.2 slice of the held entry shows 0.4 executed and the rest never
    arrives: the hold expires into the quarantine. After a resume and a
    restart the venue answers that it holds no such order: nothing is
    released (no synthetic zero is queued, the mapping stays), the hold
    re-quarantines and ends once the missing 0.2 is booked.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine = _mk_engine(_MockBroker(), store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        key = intent.intent_key
        engine._route_event(_ct_unknown_execution(
            _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))
        engine.on_order_event(
            _ct_order_event('partial', filled_qty=0.4, fill_qty=0.2, fill_id='x-1'))
        engine._drain_events()
        engine._expire_fill_evidence_holds(now_ms=int(time.time() * 1000) + 10 ** 9)
        engine._flush_owed_store_writes()
        quarantine_id = engine.quarantine_id
        assert quarantine_id is not None
        store.request_quarantine_resume(ctx.run_id, quarantine_id)
        ctx.close()

        ctx2 = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        broker2 = _HeldCloseBroker()
        broker2.executions = [OrderExecution(order=None)]
        engine2 = _mk_engine(broker2, store_ctx=ctx2)
        engine2._current_bar_ts_ms = BAR_TS
        engine2._position.seed_fill_cursors({'xchg-parent': 0.2})
        engine2._expire_fill_evidence_holds(now_ms=int(time.time() * 1000))
        engine2._recover_held_executions(now_ms=int(time.time() * 1000))
        assert broker2.execution_reads == [('xchg-parent', None)]
        assert engine2._event_queue.empty()
        engine2._drain_events()
        assert engine2._order_mapping.get(key) == ['xchg-parent']
        assert engine2._entry_execution_unresolved(key)
        engine2._expire_fill_evidence_holds(now_ms=int(time.time() * 1000))
        assert engine2.quarantined

        engine2.on_order_event(
            _ct_order_event('partial', filled_qty=0.4, fill_qty=0.2, fill_id='x-2'))
        engine2._drain_events()
        engine2._flush_owed_store_writes()
        assert engine2._position.fill_cursor(['xchg-parent']) == pytest.approx(0.4)
        assert not engine2._entry_execution_unresolved(key)
        assert ctx2.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND) == {}


@pytest.mark.parametrize('executed', [0.0, 0.4])
def __test_a_rejected_terminal_resolves_a_cancel_tentative_parent__(executed):
    """A venue ``rejected`` ends a tentative parent's order just as ``cancelled`` does.

    The strategy's cancel of the entry timed out (cancel-tentative). The
    venue then reports the order rejected with its executed quantity
    (nothing, or 0.4): the tentative cancel is resolved (confirmed, or as
    filled with the legs kept for the execution), so the cancel-retry probe
    past the stale grace no longer halts the bot over an order that ended.
    It is the terminal the venue's own record of a held order proves.
    """
    engine = _mk_engine(_MockBroker())
    intent = _seed_parent_intent_and_leg(engine)
    (leg,) = list(engine._partial_bracket_engine.iter_legs())
    engine._cancel_disposition_pending[intent.intent_key] = _CancelTentativeMeta(
        since_ts_ms=int(time.time() * 1000) - 3_600_000, reason='broker_timeout',
        retry_count=1)
    engine._partial_bracket_engine.mark_legs_cancel_tentative(
        intent.intent_key, reason='broker_timeout', now_ms=BAR_TS)
    event = _ct_order_event('cancelled', filled_qty=executed, fill_qty=None)
    event = dataclasses.replace(
        event, event_type='rejected',
        order=dataclasses.replace(event.order, status=OrderStatus.REJECTED))

    engine._route_event(event)
    engine._drive_cancel_tentative(now_ms=int(time.time() * 1000) + 120_000)

    assert intent.intent_key not in engine._cancel_disposition_pending
    assert engine.quarantined is False
    if executed > 0.0:
        assert leg.leg_state == LEG_STATE_PENDING_ENTRY
    else:
        assert leg.leg_state == LEG_STATE_ABORTED_PARENT_NEVER_ARRIVED


class _ReversalBroker(_MockBroker):
    """A mock broker that also places entries and closes, recording each placement."""

    def __init__(self) -> None:
        super().__init__(capabilities=ExchangeCapabilities(
            partial_qty_bracket_exit=CapabilityLevel.SOFTWARE,
            short_selling=CapabilityLevel.NATIVE))
        self.placed: list[tuple[str, Any]] = []

    @staticmethod
    def _order(order_id: str, intent) -> object:
        from pynecore.core.broker.models import ExchangeOrder
        return ExchangeOrder(
            id=order_id, symbol=SYMBOL, side=intent.side, order_type=OrderType.MARKET,
            qty=intent.qty, filled_qty=0.0, remaining_qty=intent.qty, price=None,
            stop_price=None, average_fill_price=None, status=OrderStatus.OPEN,
            timestamp=BAR_TS / 1000.0, fee=0.0, fee_currency='USDT')

    async def execute_entry(self, envelope: DispatchEnvelope):
        self.placed.append(('entry', envelope.intent))
        return [self._order(f"xchg-{envelope.intent.pine_id}", envelope.intent)]

    async def execute_close(self, envelope: DispatchEnvelope):
        self.placed.append(('close', envelope.intent))
        return self._order('xchg-close', envelope.intent)


def _reversal_fill(order_id: str, pine_id: str, side: str, qty: float, fill_id: str,
                   leg_type=None):
    from pynecore.core.broker.models import ExchangeOrder, LegType, OrderEvent
    order = ExchangeOrder(
        id=order_id, symbol=SYMBOL, side=side, order_type=OrderType.MARKET, qty=qty,
        filled_qty=qty, remaining_qty=0.0, price=None, stop_price=None,
        average_fill_price=100.0, status=OrderStatus.FILLED, timestamp=BAR_TS / 1000.0,
        fee=0.0, fee_currency='USDT')
    return OrderEvent(
        order=order, event_type='filled', fill_price=100.0, fill_qty=qty,
        timestamp=BAR_TS / 1000.0, pine_id=pine_id, leg_type=leg_type or LegType.ENTRY,
        fill_id=fill_id, journal_fill_qty=qty)


@pytest.mark.parametrize('reverse_id', ['parent', 'short'])
def __test_a_reversal_over_an_entry_awaiting_its_execution__(reverse_id):
    """A stop-and-reverse while the reversed entry's execution is still being booked.

    The held entry's 0.2 slice shows 0.4 executed (``AWAITING_EXECUTION``,
    the book long 0.2). The strategy reverses with a MARKET sell entry: under
    the same id it is a change of the held key and nothing is sent on that
    key; under another id the reversal waits (nothing is closed, cancelled or
    sent while the book holds only part of the execution). Once the missing
    0.2 is booked the order's 0.4 is booked exactly once, the reversal closes
    the whole 0.4 and opens the 1.0 short: the TradingView result.
    """
    from pynecore.core.broker.models import LegType
    broker = _ReversalBroker()
    engine = _mk_engine(broker)
    intent = _seed_parent_intent_and_leg(engine, with_leg=False)
    key = intent.intent_key
    cancels: list = []
    real_cancel = broker.execute_cancel

    async def execute_cancel(envelope):
        cancels.append(envelope.intent)
        return await real_cancel(envelope)

    broker.execute_cancel = execute_cancel  # type: ignore[method-assign]
    engine._route_event(_ct_unknown_execution(
        _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))
    engine._route_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.2, fill_id='x-1'))
    assert engine._entry_execution_awaiting_booking(key)
    assert engine._position.size == pytest.approx(0.2)

    reverse = EntryIntent(pine_id=reverse_id, symbol=SYMBOL, side='sell', qty=1.0,
                          order_type=OrderType.MARKET)
    intents = [reverse] if reverse_id == key else [intent, reverse]
    engine._diff_and_dispatch(intents)
    engine._diff_and_dispatch(intents)
    assert broker.placed == []
    assert cancels == []

    engine._route_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.2, fill_id='x-2'))
    engine._route_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.2, fill_id='x-2'))
    assert not engine._entry_execution_unresolved(key)
    assert engine._position.fill_cursor(['xchg-parent']) == pytest.approx(0.4)
    assert engine._position.size == pytest.approx(0.4)

    def on_held_key(placed) -> bool:
        return getattr(placed, 'intent_key', None) == key

    if reverse_id == key:
        assert broker.placed == []
    else:
        engine._diff_and_dispatch(intents)
        (kind, close), = broker.placed
        assert kind == 'close' and close.synthetic_kind == 'reversal_close'
        assert close.qty == pytest.approx(0.4)
        engine._route_event(_reversal_fill('xchg-close', close.pine_id, 'sell', 0.4, 'c-1',
                                           leg_type=LegType.CLOSE))
        assert engine._position.size == pytest.approx(0.0)
        engine._diff_and_dispatch(intents)
        assert [kind for kind, _placed in broker.placed] == ['close', 'entry']
        assert broker.placed[1][1].pine_id == reverse_id
        assert broker.placed[1][1].qty == pytest.approx(1.0)
        engine._route_event(_reversal_fill('xchg-short', reverse_id, 'sell', 1.0, 's-1'))
    assert not any(on_held_key(placed) for _kind, placed in broker.placed)
    assert not any(on_held_key(cancelled) for cancelled in cancels)
    # The venue: 0.4 long, the reduce-only close of 0.4, the 1.0 short.
    expected = 0.4 if reverse_id == key else -1.0
    assert engine._position.size == pytest.approx(expected)
    assert engine.quarantined is False


class _HeldCloseBroker(_ReversalBroker):
    """A venue that keeps its own position: 0.4 long of the held order, closes reduce-only.

    Netting by default (``get_position``, ``execute_close`` reduces the venue
    position by the close's quantity, never past flat); ``hedging`` exposes
    the venue as one leg per opened order through a position port instead.
    """

    def __init__(self, *, hedging: bool = False, supports: bool = True) -> None:
        super().__init__()
        self.supports_held_execution_close = supports
        self.position_port = self if hedging else None
        self.read_error: Exception | None = None
        self.close_error: Exception | None = None
        #: The venue's legs: leg id -> (side, qty, entry id).
        self.legs: dict[str, tuple[str, float, str | None]] = {}
        self.closed_legs: list[tuple[str, float]] = []
        self.placed_legs: list[float] = []
        self.position_reads = 0
        #: The venue's answers to ``fetch_order_execution``, in order (the last
        #: one repeats); empty: the plugin cannot read an order's execution.
        self.executions: list[Any] = []
        self.execution_reads: list[tuple[str | None, str | None]] = []

    async def fetch_order_execution(self, order_id, client_order_id):
        self.execution_reads.append((order_id, client_order_id))
        if not self.executions:
            raise NotImplementedError
        answer = self.executions[0] if len(self.executions) == 1 else self.executions.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    @property
    def venue(self) -> float:
        return round(sum(qty if side == 'buy' else -qty
                         for side, qty, _entry in self.legs.values()), 12)

    def hold(self, qty: float) -> None:
        """The venue holds ``qty`` long of the held order."""
        self.legs = {'L1': ('buy', qty, 'parent')}

    def _reduce(self, side: str, qty: float) -> float:
        """Reduce the venue position by ``qty`` closing ``side``; what executed."""
        remaining = qty
        for leg_id, (leg_side, leg_qty, entry) in list(self.legs.items()):
            if leg_side == side or remaining <= 1e-12:
                continue
            part = min(leg_qty, remaining)
            remaining = round(remaining - part, 12)
            if leg_qty - part <= 1e-12:
                del self.legs[leg_id]
            else:
                self.legs[leg_id] = (leg_side, round(leg_qty - part, 12), entry)
        return round(qty - remaining, 12)

    async def get_position(self, symbol: str) -> ExchangePosition | None:
        self.position_reads += 1
        if self.read_error is not None:
            raise self.read_error
        venue = self.venue
        if abs(venue) <= 1e-12:
            return None
        return ExchangePosition(
            symbol=SYMBOL, side='long' if venue > 0.0 else 'short', size=abs(venue),
            entry_price=100.0, unrealized_pnl=0.0, liquidation_price=None, leverage=1.0,
            margin_mode='cross')

    async def execute_entry(self, envelope: DispatchEnvelope):
        intent = envelope.intent
        self.legs[f"E{len(self.legs) + 1}"] = (intent.side, intent.qty, intent.pine_id)
        return await super().execute_entry(envelope)

    async def execute_close(self, envelope: DispatchEnvelope):
        self.placed.append(('close', envelope.intent))
        if self.close_error is not None:
            raise self.close_error
        self._reduce(envelope.intent.side, envelope.intent.qty)
        return self._order('xchg-close', envelope.intent)

    # --- PositionPort surface (active with ``hedging=True``)

    async def fetch_raw_positions(self, symbol: str):
        from pynecore.core.broker.models import PositionLeg
        self.position_reads += 1
        if self.read_error is not None:
            raise self.read_error
        return [PositionLeg(leg_id=leg_id, symbol=SYMBOL, side=side, qty=qty,
                            entry_price=100.0, open_time=float(index), unrealized_pnl=0.0,
                            entry_id=entry)
                for index, (leg_id, (side, qty, entry)) in enumerate(self.legs.items())]

    async def get_volume_quantizer(self, symbol: str):
        return lambda units: int(round(units * 100))

    async def close_leg(self, symbol: str, leg_id: str, volume: int, coid: str) -> None:
        qty = volume / 100
        self.closed_legs.append((leg_id, qty))
        side, leg_qty, entry = self.legs[leg_id]
        if leg_qty - qty <= 1e-12:
            del self.legs[leg_id]
        else:
            self.legs[leg_id] = (side, round(leg_qty - qty, 12), entry)

    async def close_leg_executed(self, symbol, leg_id, coids, since_ms):
        return 0

    async def reject_out_of_range(self, envelope, qty):
        return None

    async def place_leg(self, envelope, qty):
        self.placed_legs.append(qty)
        intent = envelope.intent
        self.legs[f"P{len(self.placed_legs)}"] = (intent.side, qty, intent.pine_id)
        return [self._order(f"xchg-{intent.pine_id}", intent)]


def _held_close_setup(state: str = 'awaiting', **broker_kw):
    """A held long entry whose order executed 0.4 at the venue.

    ``awaiting``: the venue's 0.2 slice showed 0.4 executed, the book holds 0.2
    (``AWAITING_EXECUTION``). ``held``: the order ended without the venue's
    executed quantity and no fill reached the book (``HELD``).
    """
    broker = _HeldCloseBroker(**broker_kw)
    broker.hold(0.4)
    engine = _mk_engine(broker)
    intent = _seed_parent_intent_and_leg(engine, with_leg=False)
    engine._route_event(_ct_unknown_execution(
        _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))
    if state == 'awaiting':
        engine._route_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.2,
                                            fill_id='x-1'))
        assert engine._entry_execution_awaiting_booking('parent')
    else:
        assert 'parent' in engine._entries_awaiting_fill_evidence
    return broker, engine, intent


def _expire(engine: OrderSyncEngine) -> None:
    engine._expire_fill_evidence_holds(now_ms=int(time.time() * 1000) + 10 ** 9)
    assert engine.quarantined


def _late_fill(qty: float, fill_id: str = 'x-2'):
    """The held order's missing fill: the order shows 0.4 executed."""
    return _ct_order_event('partial', filled_qty=0.4, fill_qty=qty, fill_id=fill_id)


def _held_close_requests(request: str, intent: EntryIntent, booked: float) -> list:
    """The strategy's intents for ``request`` over the held long entry.

    A close is emitted while the book holds a position, as Pine sizes it.
    """
    if request == 'reverse':
        return [intent, EntryIntent(pine_id='short', symbol=SYMBOL, side='sell', qty=1.0,
                                    order_type=OrderType.MARKET)]
    if request == 'limit_reverse':
        return [intent, EntryIntent(pine_id='short', symbol=SYMBOL, side='sell',
                                    qty=round(1.0 + booked, 12), order_type=OrderType.LIMIT,
                                    limit=101.0, flip_qty=booked)]
    if request == 'close_all':
        if booked <= 0.0:
            return [intent]
        return [intent, CloseIntent(pine_id='', symbol=SYMBOL, side='sell', qty=booked)]
    assert request == 'close_held'
    if booked <= 0.0:
        return []
    return [CloseIntent(pine_id='parent', symbol=SYMBOL, side='sell', qty=booked)]


def __test_a_waiting_reversal_never_tears_down_the_held_entrys_protection__():
    """The reversal waits before its pre-clear sweep: the held entry keeps its exit.

    The sweep of a stop-and-reverse cancels the closing legs of every
    consumed entry. While the held entry's execution is not all booked, the
    reversal does not start, so the exit that protects the executed entry is
    neither cancelled nor retired.
    """
    broker, engine, intent = _held_close_setup()
    cancels: list = []
    real_cancel = broker.execute_cancel

    async def execute_cancel(envelope):
        cancels.append(envelope.intent)
        return await real_cancel(envelope)

    broker.execute_cancel = execute_cancel  # type: ignore[method-assign]
    exit_intent = ExitIntent(pine_id='tp', from_entry='parent', symbol=SYMBOL, side='sell',
                             qty=1.0, tp_price=120.0)
    engine._active_intents[exit_intent.intent_key] = exit_intent
    engine._order_mapping[exit_intent.intent_key] = ['xchg-tp']

    engine._diff_and_dispatch([*_held_close_requests('reverse', intent, 0.2), exit_intent])

    assert broker.placed == []
    assert cancels == []
    assert engine._active_intents[exit_intent.intent_key] == exit_intent
    assert engine._order_mapping[exit_intent.intent_key] == ['xchg-tp']
    assert engine._pending_reversal_opens == {}


@pytest.mark.parametrize('request_kind', ['reverse', 'limit_reverse', 'close_all'])
def __test_a_position_sized_dispatch_waits_for_the_held_execution__(request_kind):
    """A reversal or close_all over a held entry waits, then goes out against the true book.

    Nothing is sent while the book holds 0.2 of the 0.4 the order executed.
    The sync that books the missing 0.2 sends it sized from the whole 0.4: the
    market reversal's close takes 0.4, a close_all goes out on the same bar
    (no decline gate delays it), a price-based reversal is placed as the
    strategy sized it then.
    """
    from pynecore.core.broker.models import LegType
    broker, engine, intent = _held_close_setup()
    for _sync in range(2):
        engine._diff_and_dispatch(_held_close_requests(request_kind, intent, 0.2))
    assert broker.placed == []
    assert not engine.quarantined

    engine._route_event(_late_fill(0.2))
    assert not engine._entry_execution_unresolved('parent')
    assert engine._position.size == pytest.approx(0.4)

    engine._diff_and_dispatch(_held_close_requests(request_kind, intent, 0.4))
    (kind, placed), = broker.placed
    if request_kind == 'reverse':
        assert kind == 'close' and placed.synthetic_kind == 'reversal_close'
        assert placed.qty == pytest.approx(0.4)
        engine._route_event(_reversal_fill('xchg-close', placed.pine_id, 'sell', 0.4, 'c-1',
                                           leg_type=LegType.CLOSE))
        engine._diff_and_dispatch(_held_close_requests(request_kind, intent, 0.0))
        assert [kind for kind, _placed in broker.placed] == ['close', 'entry']
        engine._route_event(_reversal_fill('xchg-short', 'short', 'sell', 1.0, 's-1'))
        assert engine._position.size == pytest.approx(-1.0)
    elif request_kind == 'close_all':
        assert kind == 'close' and placed.synthetic_kind is None and placed.pine_id == ''
        assert placed.qty == pytest.approx(0.4)
    else:
        assert kind == 'entry' and placed.pine_id == 'short'
        assert placed.qty == pytest.approx(1.4)
    assert engine.quarantined is False


@pytest.mark.parametrize('fill_order', ['close_first', 'late_fill_first', 'never'])
@pytest.mark.parametrize('request_kind,state', [
    ('reverse', 'awaiting'), ('reverse', 'held'), ('close_all', 'awaiting'),
    ('close_held', 'awaiting'),
])
def __test_an_expired_hold_closes_the_venue_position_then_stays_quarantined__(
        request_kind, state, fill_order):
    """Over an expired hold the strategy's close or reversal closes the venue position.

    The evidence never came within the grace: the bot is quarantined. The
    strategy asks to close (or reverse) the long, so the engine reads the
    venue (0.4 long, the book holds 0.2 or nothing) and closes all of it
    with one ``held_execution_close``; the reversal's new side never opens.
    The close's fill is booked up to the book and its excess waits for the
    held order's late fill, which it closes: the book never goes short, the
    late fill is booked exactly once, and the book ends flat with the venue.
    Without the late fill the book stays flat and the excess waits. The
    quarantine stands throughout.
    """
    from pynecore.core.broker.models import LegType
    broker, engine, intent = _held_close_setup(state)
    booked = engine._position.size
    _expire(engine)
    requests = _held_close_requests(request_kind, intent, booked)

    engine._diff_and_dispatch(requests)
    (kind, close), = broker.placed
    assert kind == 'close' and close.synthetic_kind == 'held_execution_close'
    assert close.side == 'sell' and close.qty == pytest.approx(0.4)
    assert broker.venue == pytest.approx(0.0)
    # Settling: no venue read, no second close.
    reads = broker.position_reads
    engine._diff_and_dispatch(requests)
    assert len(broker.placed) == 1 and broker.position_reads == reads

    close_fill = _reversal_fill('xchg-close', close.pine_id, 'sell', 0.4, 'c-1',
                                leg_type=LegType.CLOSE)
    sizes: list[float] = []
    if fill_order == 'late_fill_first':
        engine._route_event(_late_fill(0.4 - booked))
        sizes.append(engine._position.size)
        engine._route_event(close_fill)
    else:
        engine._route_event(close_fill)
        sizes.append(engine._position.size)
        if fill_order == 'close_first':
            engine._route_event(_late_fill(0.4 - booked))
            engine._route_event(_late_fill(0.4 - booked))
    sizes.append(engine._position.size)
    assert all(size >= -1e-12 for size in sizes)
    assert engine._position.size == pytest.approx(0.0)
    engine._diff_and_dispatch(_held_close_requests(request_kind, intent, 0.0))
    assert [kind for kind, _placed in broker.placed] == ['close']
    assert engine.quarantined
    if fill_order == 'never':
        assert engine._entry_execution_unresolved('parent')
        assert engine._position.close_excess_qty(['xchg-parent']) == pytest.approx(0.4 - booked)
        assert engine._held_execution_close is not None
    else:
        assert not engine._entry_execution_unresolved('parent')
        assert engine._position.fill_cursor(['xchg-parent']) == pytest.approx(0.4)
        assert engine._position.close_excess_qty(['xchg-parent']) == 0.0
        assert engine._held_execution_close is None
        # The round trip of the execution is booked with both real prices.
        assert sum(abs(trade.size) for trade in engine._position.closed_trades) == (
            pytest.approx(0.4))


def __test_an_expired_hold_with_an_unreadable_venue_position_closes_nothing__():
    """A venue position the engine cannot read is never closed blindly.

    The strategy asks to reverse over the expired hold, but ``get_position``
    fails: nothing is sent, the bot stays quarantined, and the close goes out
    once the venue can be read again.
    """
    from pynecore.core.broker.exceptions import ExchangeConnectionError
    broker, engine, intent = _held_close_setup()
    _expire(engine)
    broker.read_error = ExchangeConnectionError("positions endpoint down")
    engine._diff_and_dispatch(_held_close_requests('reverse', intent, 0.2))
    assert broker.placed == []
    assert engine.quarantined

    broker.read_error = None
    engine._held_close_read_ms = 0
    engine._diff_and_dispatch(_held_close_requests('reverse', intent, 0.2))
    (kind, close), = broker.placed
    assert close.synthetic_kind == 'held_execution_close' and close.qty == pytest.approx(0.4)


def __test_an_expired_hold_under_an_external_intervention_writes_nothing__():
    """An external-intervention latch forbids the held-execution close too.

    Someone else touched the account: the bot writes nothing to the venue,
    whatever the strategy asks.
    """
    from pynecore.core.broker.models import QuarantineKind
    broker, engine, intent = _held_close_setup()
    _expire(engine)
    engine.record_quarantine('order_moved', {'client_order_id': 'x'},
                             kind=QuarantineKind.EXTERNAL_INTERVENTION)
    for request_kind in ('reverse', 'close_all', 'close_held'):
        engine._diff_and_dispatch(_held_close_requests(request_kind, intent, 0.2))
    assert broker.placed == []
    assert broker.position_reads == 0
    assert broker.venue == pytest.approx(0.4)


def __test_an_expired_hold_on_a_plugin_without_venue_close_closes_the_booked_part__():
    """A netting plugin that cannot close by venue quantity closes what the book holds.

    Without :attr:`BrokerPlugin.supports_held_execution_close` the engine
    sends the book's own symbol-wide close for the booked 0.2 and leaves the
    rest to the operator; once the book is flat it sends nothing for the
    unbooked part the plugin cannot reach.
    """
    from pynecore.core.broker.models import LegType
    broker, engine, intent = _held_close_setup(supports=False)
    _expire(engine)
    engine._diff_and_dispatch(_held_close_requests('reverse', intent, 0.2))
    (kind, close), = broker.placed
    assert close.synthetic_kind == 'reversal_close' and close.qty == pytest.approx(0.2)
    assert broker.venue == pytest.approx(0.2)
    engine._route_event(_reversal_fill('xchg-close', close.pine_id, 'sell', 0.2, 'c-1',
                                       leg_type=LegType.CLOSE))
    assert engine._position.size == pytest.approx(0.0)
    marker = engine._held_execution_close
    assert marker is not None
    marker.sent_ms -= 10 ** 6
    engine._held_close_read_ms = 0
    engine._diff_and_dispatch(_held_close_requests('reverse', intent, 0.2))
    assert len(broker.placed) == 1
    assert broker.position_reads == 2
    assert engine.quarantined


@pytest.mark.parametrize('outcome', ['landed', 'lost'])
def __test_an_unknown_outcome_held_close_is_never_sent_again_blindly__(outcome):
    """A held-execution close of unknown outcome waits, then the venue decides.

    The close's dispatch timed out. Inside the settle window nothing is read
    or sent. Past it the venue is read: a close that landed left the venue
    flat and nothing more goes out; one that never landed is sent again
    under a fresh id, sized from the fresh read.
    """
    broker, engine, intent = _held_close_setup()
    _expire(engine)
    broker.close_error = OrderDispositionUnknownError("close timed out",
                                                      client_order_id='held-close')
    if outcome == 'landed':
        real_close = broker.execute_close

        async def landing_close(envelope):
            broker._reduce(envelope.intent.side, envelope.intent.qty)
            return await real_close(envelope)

        broker.execute_close = landing_close  # type: ignore[method-assign]
    engine._diff_and_dispatch(_held_close_requests('reverse', intent, 0.2))
    assert len(broker.placed) == 1
    first = broker.placed[0][1]
    broker.close_error = None
    engine._diff_and_dispatch(_held_close_requests('reverse', intent, 0.2))
    assert len(broker.placed) == 1

    marker = engine._held_execution_close
    assert marker is not None
    marker.sent_ms -= 10 ** 6
    engine._diff_and_dispatch(_held_close_requests('reverse', intent, 0.2))
    if outcome == 'landed':
        assert len(broker.placed) == 1
    else:
        (_kind, _first), (_kind2, second) = broker.placed
        assert second.pine_id != first.pine_id
        assert second.synthetic_kind == 'held_execution_close'
        assert second.qty == pytest.approx(0.4)


def __test_a_failed_held_close_write_sends_nothing_and_the_next_sync_sends_it__(tmp_path):
    """The held-execution close is stored before it is sent (persist-first).

    A storage error on that write ends the sync with nothing sent; the next
    sync sends the close. The stored record is what a restart re-arms.
    """
    from pynecore.core.broker.storage import HELD_EXECUTION_CLOSE_EVENT_KIND
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        broker = _HeldCloseBroker()
        broker.hold(0.4)
        engine = _mk_engine(broker, store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        engine._route_event(_ct_unknown_execution(
            _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))
        engine._route_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.2,
                                            fill_id='x-1'))
        engine._flush_owed_store_writes()
        _expire(engine)
        real_replace = ctx.replace_keyed_events
        failed: list[bool] = []

        def replace_keyed_events(event_kind, *args, **kwargs):
            if event_kind == HELD_EXECUTION_CLOSE_EVENT_KIND and not failed:
                failed.append(True)
                raise sqlite3.OperationalError("disk I/O error")
            return real_replace(event_kind, *args, **kwargs)

        ctx.replace_keyed_events = replace_keyed_events  # type: ignore[method-assign]
        with pytest.raises(sqlite3.OperationalError):
            engine._diff_and_dispatch(_held_close_requests('close_all', intent, 0.2))
        assert broker.placed == []
        assert engine._held_execution_close is None

        engine._diff_and_dispatch(_held_close_requests('close_all', intent, 0.2))
        (kind, close), = broker.placed
        assert close.synthetic_kind == 'held_execution_close'
        stored = ctx.keyed_events_for_run_id(HELD_EXECUTION_CLOSE_EVENT_KIND)['close']
        assert stored['pine_ids'] == [close.pine_id]
        assert stored['order_ids'] == ['xchg-parent']


def __test_a_restart_after_the_held_close_was_sent_reads_the_venue_before_another__(tmp_path):
    """A restart re-arms the sent held-execution close from the store.

    The restarted bot does not send another close while the first one
    settles, caps the first close's late fill at the book (the excess waits
    for the held order's fill), and once the venue shows the position closed
    sends nothing more. The held order's fill then closes against the excess
    and the record is retired.
    """
    from pynecore.core.broker.models import LegType
    from pynecore.core.broker.storage import HELD_EXECUTION_CLOSE_EVENT_KIND
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        broker = _HeldCloseBroker()
        broker.hold(0.4)
        engine = _mk_engine(broker, store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        engine._route_event(_ct_unknown_execution(
            _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))
        _expire(engine)
        engine._diff_and_dispatch(_held_close_requests('reverse', intent, 0.0))
        (_kind, close), = broker.placed
        engine._flush_owed_store_writes()
        ctx.close()

        ctx2 = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        broker2 = _HeldCloseBroker()
        engine2 = _mk_engine(broker2, store_ctx=ctx2)
        engine2._current_bar_ts_ms = BAR_TS
        marker = engine2._held_execution_close
        assert marker is not None and marker.pine_ids == [close.pine_id]
        _expire(engine2)
        engine2._diff_and_dispatch(_held_close_requests('reverse', intent, 0.0))
        assert broker2.placed == [] and broker2.position_reads == 0

        engine2._route_event(_reversal_fill('xchg-close', close.pine_id, 'sell', 0.4, 'c-1',
                                            leg_type=LegType.CLOSE))
        assert engine2._position.size == pytest.approx(0.0)
        assert engine2._position.close_excess_qty(['xchg-parent']) == pytest.approx(0.4)

        marker.sent_ms -= 10 ** 6
        engine2._diff_and_dispatch(_held_close_requests('reverse', intent, 0.0))
        assert broker2.placed == [] and broker2.position_reads == 1

        engine2._route_event(_late_fill(0.4))
        assert engine2._position.size == pytest.approx(0.0)
        assert not engine2._entry_execution_unresolved('parent')
        assert engine2._held_execution_close is None
        engine2._flush_owed_store_writes()
        assert ctx2.keyed_events_for_run_id(HELD_EXECUTION_CLOSE_EVENT_KIND) == {}


def __test_a_close_of_a_held_entry_parked_for_its_cancel_closes_from_the_venue__():
    """A close of an expired held entry is not refused for good by its parked cancel.

    The strategy dropped the entry and its cancel is parked for retry, which
    the hold keeps from landing; the strategy then closes the entry. Over the
    expired hold the close goes out as the venue close, not deferred behind
    the cancel that never retries.
    """
    broker, engine, intent = _held_close_setup()
    engine._forced_cancel_pending['parent'] = intent
    _expire(engine)
    engine._diff_and_dispatch(_held_close_requests('close_held', intent, 0.2))
    (kind, close), = broker.placed
    assert close.synthetic_kind == 'held_execution_close' and close.qty == pytest.approx(0.4)


@pytest.mark.parametrize('latch', ['storage_failure', 'close_rejected', 'plugin_cause'])
def __test_a_market_reversal_under_a_quarantine_closes_the_position_only__(latch):
    """An ENTRY_BLOCK quarantine runs a reversal's close leg and blocks its new side.

    No entry is held: the book is the venue's. The quarantine (any cause, the
    engine's own or a plugin's) blocks new exposure, but
    the strategy asked to close the long: the close leg goes out, and once
    the book is flat the short is not opened.
    """
    from pynecore.core.broker.models import LegType
    broker = _HeldCloseBroker()
    broker.hold(0.5)
    engine = _mk_engine(broker)
    intent = _seed_parent_intent_and_leg(engine, with_leg=False)
    engine._route_event(_ct_order_event('partial', filled_qty=0.5, fill_qty=0.5,
                                        fill_id='x-1'))
    assert engine._position.size == pytest.approx(0.5)
    engine.record_quarantine(latch, cause={'storage_failure': 'storage_failure',
                                           'close_rejected': 'close_rejected:x',
                                           'plugin_cause': 'vanished_order:7'}[latch])
    requests = _held_close_requests('reverse', intent, 0.5)

    engine._diff_and_dispatch(requests)
    (kind, close), = broker.placed
    assert close.synthetic_kind == 'reversal_close' and close.qty == pytest.approx(0.5)
    engine._route_event(_reversal_fill('xchg-close', close.pine_id, 'sell', 0.5, 'c-1',
                                       leg_type=LegType.CLOSE))
    assert engine._position.size == pytest.approx(0.0)
    for _sync in range(2):
        engine._diff_and_dispatch(requests)
    assert [kind for kind, _placed in broker.placed] == ['close']
    assert engine._position.size == pytest.approx(0.0)


@pytest.mark.parametrize('evidence', ['settled', 'expired'])
def __test_a_hedging_reversal_over_a_held_entry_closes_the_whole_venue_leg__(evidence):
    """On a hedging account the reversal's per-leg close takes the held execution too.

    The held order's leg holds 0.4 at the venue, the book 0.2. A per-leg
    close of the booked 0.2 would leave 0.2 of that leg as a separate venue
    long next to the new short. The reversal waits instead; once the evidence
    settles, its close takes the whole 0.4 leg and the short opens (-1.0).
    Over an expired hold the engine closes the venue legs it reads (the whole
    0.4) and the short never opens.
    """
    from pynecore.core.broker.models import LegType
    broker, engine, intent = _held_close_setup(hedging=True)
    requests = _held_close_requests('reverse', intent, 0.2)
    engine._diff_and_dispatch(requests)
    assert broker.closed_legs == [] and broker.placed_legs == []

    if evidence == 'settled':
        engine._route_event(_late_fill(0.2))
        engine._diff_and_dispatch(requests)
        assert broker.closed_legs == [('L1', pytest.approx(0.4))]
        close_key = next(key for key in engine._envelopes
                         if key.startswith('__pyne_reversal_close__'))
        engine._route_event(dataclasses.replace(
            _reversal_fill('xchg-close-L1', close_key, 'sell', 0.4, 'c-1',
                           leg_type=LegType.CLOSE), from_entry=close_key))
        assert engine._position.size == pytest.approx(0.0)
        engine._diff_and_dispatch(requests)
        assert broker.placed_legs == [pytest.approx(1.0)]
        engine._route_event(_reversal_fill('xchg-short', 'short', 'sell', 1.0, 's-1'))
        assert engine._position.size == pytest.approx(-1.0)
        assert broker.venue == pytest.approx(-1.0)
        return

    _expire(engine)
    engine._diff_and_dispatch(requests)
    assert broker.closed_legs == [('L1', pytest.approx(0.4))]
    assert broker.venue == pytest.approx(0.0)
    marker = engine._held_execution_close
    assert marker is not None
    engine._route_event(dataclasses.replace(
        _reversal_fill('xchg-close-L1', marker.pine_ids[0], 'sell', 0.4, 'c-1',
                       leg_type=LegType.CLOSE), from_entry=marker.pine_ids[0]))
    engine._route_event(_late_fill(0.2))
    assert engine._position.size == pytest.approx(0.0)
    assert not engine._entry_execution_unresolved('parent')
    engine._diff_and_dispatch(requests)
    assert broker.placed_legs == []
    assert engine.quarantined


@pytest.mark.parametrize('terminal', ['cancelled', 'rejected'])
def __test_a_tentative_parent_terminal_ahead_of_its_fill_keeps_the_dispatch_barrier__(
        tmp_path, terminal):
    """A tentative parent's terminal event reports an execution its fill has not delivered yet.

    The strategy's cancel of the 1.0 entry timed out (cancel-tentative, no
    hold). The venue then ends the order reporting 0.4 executed, ahead of the
    fill. The entry waits in ``AWAITING_EXECUTION`` with the reported
    quantity in its stored hold, so neither the kept nor a changed entry is
    sent before that fill is booked; once it is, the hold ends.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine = _mk_engine(_MockBroker(), store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        sent: list = []

        def record_new(new_intent) -> None:
            sent.append(new_intent)

        engine._dispatch_new = record_new  # type: ignore[method-assign]
        engine._cancel_disposition_pending[intent.intent_key] = _CancelTentativeMeta(
            since_ts_ms=BAR_TS, reason='broker_timeout')
        engine._partial_bracket_engine.mark_legs_cancel_tentative(
            intent.intent_key, reason='broker_timeout', now_ms=BAR_TS)

        engine._route_event(_hold_positive_terminal(terminal))
        engine._flush_owed_store_writes()

        assert intent.intent_key not in engine._cancel_disposition_pending
        assert engine._entry_execution_awaiting_booking(intent.intent_key)
        stored = ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND)
        assert stored[intent.intent_key]['awaiting_execution'] == ['xchg-parent', 0.4]
        engine._diff_and_dispatch([intent])
        engine._diff_and_dispatch([dataclasses.replace(intent, qty=1.1)])
        assert sent == []

        engine._route_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.4,
                                            fill_id='x-1'))
        engine._flush_owed_store_writes()

        assert engine._position.size == pytest.approx(0.4)
        assert not engine._entry_execution_unresolved(intent.intent_key)
        assert ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND) == {}
        assert engine.quarantined is False and not engine.quarantined
        ctx.close()


@pytest.mark.parametrize('tentative', [True, False])
@pytest.mark.parametrize('terminal', ['cancelled', 'rejected'])
def __test_the_reported_execution_hold_is_stored_before_the_terminal_handling__(
        tmp_path, terminal, tentative):
    """The hold of an execution reported ahead of its fill is stored before the order retires.

    The venue ends the entry's order (a cancel-tentative parent's, or an
    active entry's) reporting 0.4 executed, ahead of the fill. When the
    terminal handling retires the tentative cancel or consumes the entry,
    the hold naming 0.4 is already in the store: a process dying at that
    point restarts with the barrier, never with neither.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine = _mk_engine(_MockBroker(), store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        key = intent.intent_key
        if tentative:
            engine._cancel_disposition_pending[key] = _CancelTentativeMeta(
                since_ts_ms=BAR_TS, reason='broker_timeout')
            engine._partial_bracket_engine.mark_legs_cancel_tentative(
                key, reason='broker_timeout', now_ms=BAR_TS)
        stored_at_retirement: list = []

        def spy(real):
            def wrapper(*args, **kwargs):
                stored_at_retirement.append(
                    ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND).get(key))
                return real(*args, **kwargs)
            return wrapper

        engine._clear_intent_cancel_disposition_pending = spy(  # type: ignore[method-assign]
            engine._clear_intent_cancel_disposition_pending)
        engine._retire_venue_cancelled_entry = spy(  # type: ignore[method-assign]
            engine._retire_venue_cancelled_entry)

        engine._route_event(_hold_positive_terminal(terminal))

        assert stored_at_retirement
        assert all(row is not None and row['awaiting_execution'] == ['xchg-parent', 0.4]
                   for row in stored_at_retirement), stored_at_retirement
        assert engine._entry_execution_awaiting_booking(key)
        ctx.close()


def __test_a_later_terminal_report_raises_the_awaited_execution_after_unmapping__(tmp_path):
    """A second terminal event of the ended order raises what the book must hold.

    The venue rejects the entry reporting 0.4 executed as a lower bound (the
    executed quantity not confirmed), so the entry is consumed, its order
    unmapped and ``AWAITING_EXECUTION`` waits for 0.4. A second rejection
    then confirms 0.6. The wait (and its stored row) takes 0.6: the 0.4
    fill does not end it, a changed entry sends nothing until the last 0.2
    is booked too.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine = _mk_engine(_MockBroker(), store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        key = intent.intent_key
        sent: list = []

        def record_new(new_intent) -> None:
            sent.append(new_intent)

        engine._dispatch_new = record_new  # type: ignore[method-assign]
        rejected = _hold_positive_terminal('rejected')
        engine._route_event(dataclasses.replace(
            rejected, order=dataclasses.replace(rejected.order, filled_qty_known=False)))
        assert engine._find_key_for_order_id('xchg-parent') is None
        engine._route_event(dataclasses.replace(
            rejected, order=dataclasses.replace(rejected.order, filled_qty=0.6)))
        engine._flush_owed_store_writes()

        stored = ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND)
        assert stored[key]['awaiting_execution'] == ['xchg-parent', 0.6]
        engine._route_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.4,
                                            fill_id='x-1'))
        assert engine._entry_execution_awaiting_booking(key)
        changed = dataclasses.replace(intent, qty=1.1)
        engine._diff_and_dispatch([changed])
        assert sent == []

        engine._route_event(_ct_order_event('partial', filled_qty=0.6, fill_qty=0.2,
                                            fill_id='x-2'))
        engine._flush_owed_store_writes()
        assert not engine._entry_execution_unresolved(key)
        assert engine._position.size == pytest.approx(0.6)
        assert ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND) == {}
        assert engine.quarantined is False and not engine.quarantined
        ctx.close()


def __test_a_higher_terminal_report_queued_behind_a_fill_keeps_the_awaited_execution__(tmp_path):
    """A terminal event received behind the fill that reaches the awaited quantity still counts.

    ``AWAITING_EXECUTION`` waits for 0.4 (a lower bound). A 0.4 fill and a
    rejection confirming 0.6 are received together, in that order, and
    drained at once. The rejection is recorded on receipt, so the fill does
    not release the wait: a changed entry sends nothing until the last 0.2 is
    booked, and the stored wait names 0.6.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine = _mk_engine(_MockBroker(), store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        key = intent.intent_key
        sent: list = []

        def record_new(new_intent) -> None:
            sent.append(new_intent)

        engine._dispatch_new = record_new  # type: ignore[method-assign]
        rejected = _hold_positive_terminal('rejected')
        engine._route_event(dataclasses.replace(
            rejected, order=dataclasses.replace(rejected.order, filled_qty_known=False)))
        assert engine._entry_execution_awaiting_booking(key)

        engine.on_order_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.4,
                                              fill_id='x-1'))
        engine.on_order_event(dataclasses.replace(
            rejected, order=dataclasses.replace(rejected.order, filled_qty=0.6)))
        engine._drain_events()
        engine._flush_owed_store_writes()

        assert engine._entry_execution_awaiting_booking(key)
        stored = ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND)
        assert stored[key]['awaiting_execution'] == ['xchg-parent', 0.6]
        engine._diff_and_dispatch([dataclasses.replace(intent, qty=1.1)])
        assert sent == []

        engine._route_event(_ct_order_event('partial', filled_qty=0.6, fill_qty=0.2,
                                            fill_id='x-2'))
        engine._flush_owed_store_writes()
        assert not engine._entry_execution_unresolved(key)
        assert engine._position.size == pytest.approx(0.6)
        assert ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND) == {}
        assert ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_ARRIVAL_EVENT_KIND) == {}
        assert engine.quarantined is False and not engine.quarantined
        ctx.close()


def __test_a_stale_lower_terminal_repeat_keeps_the_stored_execution_across_a_restart__(tmp_path):
    """A stale repeat of a lower report never replaces the stored higher one.

    ``AWAITING_EXECUTION`` waits for 0.4 (a lower bound). A rejection
    confirming 0.6 and a stale repeat of the 0.4 report are received, and
    the bot restarts before draining either: the stored arrival still names
    0.6, so the restarted bot's 0.4 fill does not release the hold, and a
    changed entry sends nothing until the last 0.2 is booked.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine = _mk_engine(_MockBroker(), store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        key = intent.intent_key
        rejected = _hold_positive_terminal('rejected')
        lower = dataclasses.replace(
            rejected, order=dataclasses.replace(rejected.order, filled_qty_known=False))
        engine._route_event(lower)
        engine._flush_owed_store_writes()
        assert engine._entry_execution_awaiting_booking(key)

        engine.on_order_event(dataclasses.replace(
            rejected, order=dataclasses.replace(rejected.order, filled_qty=0.6)))
        engine.on_order_event(lower)
        arrivals = ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_ARRIVAL_EVENT_KIND)
        assert arrivals['xchg-parent']['order']['filled_qty'] == pytest.approx(0.6)
        ctx.close()

        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine = _mk_engine(_MockBroker(), store_ctx=ctx)
        engine._current_bar_ts_ms = BAR_TS
        _seed_parent_intent_and_leg(engine, with_leg=False)
        sent: list = []

        def record_new(new_intent) -> None:
            sent.append(new_intent)

        engine._dispatch_new = record_new  # type: ignore[method-assign]
        engine.on_order_event(lower)
        engine._drain_events()
        engine._route_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.4,
                                            fill_id='x-1'))
        engine._flush_owed_store_writes()
        assert engine._entry_execution_unresolved(key)
        engine._diff_and_dispatch([dataclasses.replace(intent, qty=1.1)])
        assert sent == []

        engine._route_event(_ct_order_event('partial', filled_qty=0.6, fill_qty=0.2,
                                            fill_id='x-2'))
        engine._flush_owed_store_writes()
        assert not engine._entry_execution_unresolved(key)
        assert ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND) == {}
        assert ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_ARRIVAL_EVENT_KIND) == {}
        assert engine.quarantined is False and not engine.quarantined
        ctx.close()


@pytest.mark.parametrize('restart', [False, True])
def __test_a_hold_awaits_every_reporting_order_and_settles_on_the_last_fill__(tmp_path, restart):
    """Both orders of an entry report executions; the hold ends with the last one booked.

    The entry maps two venue orders, the book holds none of either. The venue
    ends the sibling reporting 0.2 and the held order reporting 0.4, with an
    optional restart before either is drained. Booking the held order's 0.4
    does not release the hold while the sibling's 0.2 is unbooked; the
    sibling's fill, booked last, does: nothing stays held and nothing is
    stored.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine = _mk_engine(_MockBroker(), store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        key = intent.intent_key
        engine._order_mapping[key].append(_MATRIX_SIBLING_ORDER)
        engine._route_event(_ct_unknown_execution(
            _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))
        engine._flush_owed_store_writes()
        assert key in engine._entries_awaiting_fill_evidence
        engine.on_order_event(_matrix_sibling_event('cancelled', filled_qty=0.2, fill_qty=None))
        engine.on_order_event(_hold_positive_terminal())
        if restart:
            ctx.close()
            ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
            engine = _mk_engine(_MockBroker(), store_ctx=ctx)
            engine._current_bar_ts_ms = BAR_TS
            _seed_parent_intent_and_leg(engine, with_leg=False)
            engine._order_mapping[key].append(_MATRIX_SIBLING_ORDER)
        sent: list = []

        def record_new(new_intent) -> None:
            sent.append(new_intent)

        engine._dispatch_new = record_new  # type: ignore[method-assign]
        engine._drain_events()
        engine._drain_events()
        engine._flush_owed_store_writes()
        assert engine._entry_execution_unresolved(key)

        engine._route_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.4,
                                            fill_id='x-1'))
        engine._flush_owed_store_writes()
        assert engine._entry_execution_unresolved(key)
        engine._diff_and_dispatch([dataclasses.replace(intent, qty=1.1)])
        assert sent == []

        engine._route_event(_matrix_sibling_event('partial', filled_qty=0.2, fill_qty=0.2,
                                                  fill_id='sib-1'))
        engine._flush_owed_store_writes()
        assert not engine._entry_execution_unresolved(key)
        assert engine._fill_evidence_awaiting_execution == {}
        assert ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND) == {}
        assert ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_ARRIVAL_EVENT_KIND) == {}
        assert engine.quarantined is False and not engine.quarantined
        ctx.close()


@pytest.mark.parametrize('restart', [False, True])
def __test_a_sibling_slice_showing_more_executed_raises_its_booking_target__(tmp_path, restart):
    """A sibling's slice showing more than its terminal reported keeps the hold for the rest.

    Both orders of the entry end at the venue: the sibling reporting 0.2 as a
    lower bound, the held order reporting 0.4. The held order's 0.4 is
    booked, then a 0.2 slice of the sibling whose order shows 0.3 executed:
    the hold stands for the sibling's 0.3, also across a restart (the startup
    takes in the journaled 0.4 and 0.2, and the reconcile's booked-hold check
    runs), and a changed entry sends nothing. The sibling's 0.1 slice ends it.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine = _mk_engine(_MockBroker(), store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        key = intent.intent_key
        engine._order_mapping[key].append(_MATRIX_SIBLING_ORDER)
        engine._route_event(_ct_unknown_execution(
            _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))
        engine.on_order_event(_matrix_sibling_event('cancelled', filled_qty=0.2, fill_qty=None))
        engine.on_order_event(_hold_positive_terminal())
        engine._drain_events()
        engine._flush_owed_store_writes()
        engine._route_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.4,
                                            fill_id='x-1'))
        engine._route_event(_matrix_sibling_event('partial', filled_qty=0.3, fill_qty=0.2,
                                                  fill_id='sib-1'))
        engine._flush_owed_store_writes()
        assert engine._entry_execution_unresolved(key)
        if restart:
            ctx.close()
            ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
            engine = _mk_engine(_MockBroker(), store_ctx=ctx)
            engine._current_bar_ts_ms = BAR_TS
            engine._position.seed_fill_cursors({'xchg-parent': 0.4, _MATRIX_SIBLING_ORDER: 0.2})
            _seed_parent_intent_and_leg(engine, with_leg=False)
            engine._order_mapping[key].append(_MATRIX_SIBLING_ORDER)
            engine._drain_events()
            engine._flush_owed_store_writes()
            engine._release_booked_fill_evidence_holds()
            engine._flush_owed_store_writes()
        sent: list = []

        def record_new(new_intent) -> None:
            sent.append(new_intent)

        engine._dispatch_new = record_new  # type: ignore[method-assign]
        assert engine._entry_execution_unresolved(key)
        engine._diff_and_dispatch([dataclasses.replace(intent, qty=1.1)])
        assert sent == []

        engine._route_event(_matrix_sibling_event('partial', filled_qty=0.3, fill_qty=0.1,
                                                  fill_id='sib-2'))
        engine._release_booked_fill_evidence_holds()
        engine._flush_owed_store_writes()
        assert not engine._entry_execution_unresolved(key)
        assert engine._fill_evidence_floors == {}
        assert ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND) == {}
        assert ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_ARRIVAL_EVENT_KIND) == {}
        assert engine.quarantined is False and not engine.quarantined
        ctx.close()


@pytest.mark.parametrize('restart', [False, True])
def __test_a_queued_fill_showing_more_executed_keeps_the_hold_after_an_earlier_fill__(
        tmp_path, restart):
    """A queued slice showing more executed than an earlier queued fill books keeps the hold.

    The held entry's venue cancel reports 0.4 executed. Its 0.4 fill and a 0.1
    slice whose order shows 0.6 executed are received together: the 0.4 fill
    reaches what the cancel reported, but the hold stands for the 0.6, also
    across a restart (the startup takes in the journaled 0.5 and the
    reconcile's booked-hold check runs), and a changed entry sends nothing.
    The last 0.1 slice ends it.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine = _mk_engine(_MockBroker(), store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        key = intent.intent_key
        engine._route_event(_ct_unknown_execution(
            _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))
        engine.on_order_event(_hold_positive_terminal())
        engine._drain_events()
        engine._flush_owed_store_writes()
        assert engine._entry_execution_awaiting_booking(key)
        engine.on_order_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.4,
                                              fill_id='x-1'))
        engine.on_order_event(_ct_order_event('partial', filled_qty=0.6, fill_qty=0.1,
                                              fill_id='x-2'))
        engine._drain_events()
        engine._flush_owed_store_writes()
        assert engine._position.fill_cursor(['xchg-parent']) == pytest.approx(0.5)
        assert engine._entry_execution_unresolved(key)
        if restart:
            ctx.close()
            ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
            engine = _mk_engine(_MockBroker(), store_ctx=ctx)
            engine._current_bar_ts_ms = BAR_TS
            engine._position.seed_fill_cursors({'xchg-parent': 0.5})
            _seed_parent_intent_and_leg(engine, with_leg=False)
            engine._drain_events()
            engine._flush_owed_store_writes()
            engine._release_booked_fill_evidence_holds()
            engine._flush_owed_store_writes()
        sent: list = []

        def record_new(new_intent) -> None:
            sent.append(new_intent)

        engine._dispatch_new = record_new  # type: ignore[method-assign]
        assert engine._entry_execution_unresolved(key)
        engine._diff_and_dispatch([dataclasses.replace(intent, qty=1.1)])
        assert sent == []

        engine.on_order_event(_ct_order_event('partial', filled_qty=0.6, fill_qty=0.1,
                                              fill_id='x-3'))
        engine._drain_events()
        engine._release_booked_fill_evidence_holds()
        engine._flush_owed_store_writes()
        assert not engine._entry_execution_unresolved(key)
        assert engine._fill_evidence_queued == {}
        assert ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_HOLD_EVENT_KIND) == {}
        assert ctx.keyed_events_for_run_id(ENTRY_FILL_EVIDENCE_ARRIVAL_EVENT_KIND) == {}
        assert engine.quarantined is False and not engine.quarantined
        ctx.close()


def __test_a_received_sibling_slice_stores_its_booking_target_for_a_restart__(tmp_path):
    """A sibling slice received through the queue writes its raised target into the hold row.

    As in the routed variant, the sibling reports 0.2 as a lower bound and
    the held order 0.4; the 0.4 is booked. The sibling's 0.2 slice showing 0.3
    executed is received and drained: the hold row holds the 0.3, so after a
    restart (the startup takes in the journaled 0.4 and 0.2) the hold stands
    and a changed entry sends nothing.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine = _mk_engine(_MockBroker(), store_ctx=ctx)
        intent = _seed_parent_intent_and_leg(engine, with_leg=False)
        key = intent.intent_key
        engine._order_mapping[key].append(_MATRIX_SIBLING_ORDER)
        engine._route_event(_ct_unknown_execution(
            _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))
        engine.on_order_event(_matrix_sibling_event('cancelled', filled_qty=0.2, fill_qty=None))
        engine.on_order_event(_hold_positive_terminal())
        engine._drain_events()
        engine._flush_owed_store_writes()
        engine._route_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.4,
                                            fill_id='x-1'))
        engine._flush_owed_store_writes()
        engine.on_order_event(_matrix_sibling_event('partial', filled_qty=0.3, fill_qty=0.2,
                                                    fill_id='sib-1'))
        engine._drain_events()
        engine._flush_owed_store_writes()
        ctx.close()
        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        engine = _mk_engine(_MockBroker(), store_ctx=ctx)
        engine._current_bar_ts_ms = BAR_TS
        engine._position.seed_fill_cursors({'xchg-parent': 0.4, _MATRIX_SIBLING_ORDER: 0.2})
        _seed_parent_intent_and_leg(engine, with_leg=False)
        engine._order_mapping[key].append(_MATRIX_SIBLING_ORDER)
        engine._drain_events()
        engine._flush_owed_store_writes()
        engine._release_booked_fill_evidence_holds()
        engine._flush_owed_store_writes()
        sent: list = []

        def record_new(new_intent) -> None:
            sent.append(new_intent)

        engine._dispatch_new = record_new  # type: ignore[method-assign]
        assert engine._entry_execution_unresolved(key)
        engine._diff_and_dispatch([dataclasses.replace(intent, qty=1.1)])
        assert sent == []
        ctx.close()


def __test_a_sibling_orders_booked_fills_do_not_release_the_awaited_execution__():
    """Only the reporting order's own booked fills end ``AWAITING_EXECUTION``.

    The entry maps two venue orders; the book took in 0.5 of the sibling. The
    venue cancels the other reporting 0.4 executed. Its 0.2 slice does not end
    the wait though the sibling's cursor is above 0.4: a changed entry sends
    nothing until the reporting order's 0.4 is booked.
    """
    engine = _mk_engine(_MockBroker())
    intent = _seed_parent_intent_and_leg(engine, with_leg=False)
    key = intent.intent_key
    engine._order_mapping[key].append('xchg-sib')
    engine._position.seed_fill_cursors({'xchg-sib': 0.5})
    sent: list = []

    def record_new(new_intent) -> None:
        sent.append(new_intent)

    engine._dispatch_new = record_new  # type: ignore[method-assign]
    engine._route_event(_hold_positive_terminal())
    assert engine._entry_execution_awaiting_booking(key)

    engine._route_event(_ct_order_event('partial', filled_qty=0.2, fill_qty=0.2, fill_id='x-1'))
    assert engine._entry_execution_awaiting_booking(key)
    engine._diff_and_dispatch([dataclasses.replace(intent, qty=1.1)])
    assert sent == []

    engine._route_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.2, fill_id='x-2'))
    assert not engine._entry_execution_unresolved(key)
    assert engine.quarantined is False


@pytest.mark.parametrize('terminal', ['cancelled', 'rejected'])
def __test_a_tentative_parent_proven_unexecuted_retires_its_native_failsafe__(terminal):
    """A tentative parent's terminal event with a reported zero retires its native fail-safe.

    The parent's ``NativeStopState`` is DEGRADING and blocks new entries on
    the symbol. The venue ends the order reporting nothing executed: the
    tentative cancel is confirmed and the state is retired, as for a parent
    that was not tentative, so another entry on the flat symbol is no longer
    blocked.
    """
    engine = _mk_engine(_MockBroker())
    intent = _seed_parent_intent_and_leg(engine)
    (leg,) = list(engine._partial_bracket_engine.iter_legs())
    ref = engine._resolve_parent_opening_ref(intent.intent_key)
    assert ref is not None
    leg.parent_entry_dispatch_ref = ref
    manager = engine._native_failsafe_manager
    manager.register_parent(
        parent_entry_dispatch_ref=ref, symbol=SYMBOL, parent_side='long',
        mintick=1.0, pending_confirmation=True, now_ms=float(BAR_TS))
    assert manager.get_state(ref).health is FailsafeHealth.DEGRADING
    assert manager.block_new_entry(symbol=SYMBOL, pine_id='other', bar_ts_ms=BAR_TS)
    engine._cancel_disposition_pending[intent.intent_key] = _CancelTentativeMeta(
        since_ts_ms=BAR_TS, reason='broker_timeout')
    engine._partial_bracket_engine.mark_legs_cancel_tentative(
        intent.intent_key, reason='broker_timeout', now_ms=BAR_TS)
    event = _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)
    if terminal == 'rejected':
        event = dataclasses.replace(
            event, event_type='rejected',
            order=dataclasses.replace(event.order, status=OrderStatus.REJECTED))

    engine._route_event(event)

    assert intent.intent_key not in engine._cancel_disposition_pending
    assert leg.leg_state == LEG_STATE_ABORTED_PARENT_NEVER_ARRIVED
    assert manager.get_state(ref).health is FailsafeHealth.RETIRED
    assert not manager.block_new_entry(symbol=SYMBOL, pine_id='other', bar_ts_ms=BAR_TS)


# === The bot's own recovery of a held entry's execution =======================


def _venue_parent(status: OrderStatus, filled: float, *, avg: float | None = 100.5,
                  fee: float = 0.04):
    """The venue's record of the held order ``xchg-parent`` (a 1-unit buy), read again."""
    return ExchangeOrder(
        id='xchg-parent', symbol=SYMBOL, side='buy', order_type=OrderType.MARKET,
        qty=1.0, filled_qty=filled, remaining_qty=round(1.0 - filled, 12), price=None,
        stop_price=None, average_fill_price=avg if filled > 0.0 else None, status=status,
        timestamp=BAR_TS / 1000.0, fee=fee, fee_currency='USDT')


def _recover(engine: OrderSyncEngine, now_ms: int) -> None:
    """One reconcile pass of the recovery: read, drain what it queued, resolve the causes."""
    engine._recover_held_executions(now_ms=now_ms)
    engine.apply_async_events()
    engine._resolve_entry_block_causes(now_ms=now_ms)


def __test_an_expired_awaiting_hold_is_booked_from_the_venues_executions__():
    """The fill never came: the bot reads the order's executions and books the missing one.

    The book holds 0.2 of the 0.4 the venue reported. The venue's executions
    are the stream's own fills: the booked one is taken once, the missing
    one is booked at its own price, the hold ends, and the quarantine ends by
    itself once the venue holds what the book holds. Nothing waits for a human.
    """
    broker, engine, _intent = _held_close_setup()
    _expire(engine)
    now = int(time.time() * 1000)
    fills = (_ct_order_event('partial', filled_qty=0.2, fill_qty=0.2, fill_id='x-1'),
             dataclasses.replace(_late_fill(0.2), fill_price=100.9))
    broker.executions = [OrderExecution(order=_venue_parent(OrderStatus.CANCELLED, 0.4),
                                        fills=fills)]

    _recover(engine, now)

    assert broker.execution_reads == [('xchg-parent', None)]
    assert engine._position.size == pytest.approx(0.4)
    assert sorted(t.entry_price for t in engine._position.open_trades) == [100.0, 100.9]
    assert not engine._entry_execution_unresolved('parent')
    assert not engine.quarantined
    assert engine.quarantine_kind is None


@pytest.mark.parametrize('fee_reported', [True, False])
def __test_an_expired_unknown_hold_is_booked_from_the_venues_totals__(fee_reported, caplog):
    """The venue never said what the order executed: its totals are booked, exactly once.

    Nothing of the order is booked, so the venue's average is the price of
    the whole execution and its cumulative fee its commission; a fee the
    venue does not report is booked as none and said so. A later stream fill
    of the same execution, under its own fill id, is taken without booking
    it again.
    """
    broker, engine, _intent = _held_close_setup('held')
    _expire(engine)
    broker.executions = [OrderExecution(order=_venue_parent(OrderStatus.CANCELLED, 0.4),
                                        fee_reported=fee_reported)]

    _recover(engine, int(time.time() * 1000))

    assert engine._position.size == pytest.approx(0.4)
    (trade,) = engine._position.open_trades
    assert trade.entry_price == pytest.approx(100.5)
    assert trade.commission == pytest.approx(0.04 if fee_reported else 0.0)
    assert ('unreported' in caplog.text) is (not fee_reported)
    assert not engine._entry_execution_unresolved('parent')
    assert not engine.quarantined

    engine._route_event(_ct_order_event('partial', filled_qty=0.4, fill_qty=0.4,
                                        fill_id='stream-1'))
    assert engine._position.size == pytest.approx(0.4)


def __test_a_partly_booked_order_is_never_booked_from_its_totals__():
    """The average alone does not give the price of the unbooked rest: nothing is invented.

    The book holds 0.2 of the order, the venue keeps only its totals. Nothing
    is booked, the quarantine stands and the bot asks again after its backoff
    (the plugin owes the executions); once the venue returns them the gap is
    booked and the quarantine ends.
    """
    broker, engine, _intent = _held_close_setup()
    _expire(engine)
    now = int(time.time() * 1000)
    broker.executions = [OrderExecution(order=_venue_parent(OrderStatus.CANCELLED, 0.4))]

    _recover(engine, now)
    assert engine._position.size == pytest.approx(0.2)
    assert engine.quarantined

    _recover(engine, now + 1000)
    assert len(broker.execution_reads) == 1, "the backoff holds the next read"

    broker.executions = [OrderExecution(
        order=_venue_parent(OrderStatus.CANCELLED, 0.4),
        fills=(_ct_order_event('partial', filled_qty=0.2, fill_qty=0.2, fill_id='x-1'),
               _late_fill(0.2)))]
    _recover(engine, now + 6000)
    assert len(broker.execution_reads) == 2
    assert engine._position.size == pytest.approx(0.4)
    assert not engine.quarantined


@pytest.mark.parametrize('answer', ['zero_executed', 'no_such_order'])
def __test_an_order_the_venue_shows_unexecuted_releases_the_held_entry__(answer):
    """The venue proves nothing executed: the entry is released by the bot itself.

    The venue's order history shows zero executed (or no such order at
    all), so the hold ends as unexecuted, and with the venue flat the
    quarantine ends.
    """
    broker, engine, _intent = _held_close_setup('held')
    broker.legs = {}
    _expire(engine)
    order = _venue_parent(OrderStatus.CANCELLED, 0.0) if answer == 'zero_executed' else None
    broker.executions = [OrderExecution(order=order)]

    _recover(engine, int(time.time() * 1000))

    assert engine._position.size == pytest.approx(0.0)
    assert not engine._entry_execution_unresolved('parent')
    assert not engine.quarantined


def __test_the_recovery_reads_the_venue_with_a_capped_backoff_and_no_deadline__(caplog):
    """A venue that cannot answer is asked again, never in a tight loop and never given up.

    A failing read is logged once per situation; a failing read and a
    ``None`` answer are asked again after a doubling delay capped at a few
    minutes, also hours later.
    """
    broker, engine, _intent = _held_close_setup('held')
    _expire(engine)
    now = int(time.time() * 1000)
    broker.executions = [ConnectionError('venue down'), None, ConnectionError('venue down')]
    for step in range(3):
        engine._recover_held_executions(now_ms=now + step)
    assert len(broker.execution_reads) == 1
    assert caplog.text.count("reading its execution from the venue failed") == 1

    delays = []
    at = now
    for _read in range(12):
        state = engine._venue_recoveries['order:xchg-parent']
        delays.append((state.next_ms - at) / 1000.0)
        at = state.next_ms
        engine._recover_held_executions(now_ms=at)
    assert delays[0] == pytest.approx(_VENUE_RECOVERY_FIRST_S)
    assert all(later >= earlier for earlier, later in zip(delays, delays[1:]))
    assert max(delays) == pytest.approx(_VENUE_RECOVERY_CAP_S)
    assert len(broker.execution_reads) == 13
    assert engine.quarantined and engine._entry_execution_unresolved('parent')


def __test_a_restart_during_the_recovery_reads_the_venue_again_and_books_once__(tmp_path):
    """The process stops while the venue cannot answer; the next one recovers by itself.

    The expired hold survives the restart, the restarted bot reads the
    venue at once, books the totals and ends its quarantine. A stream fill
    of the same execution delivered after a second restart is taken once:
    what the totals booked is kept with the book.
    """
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        broker = _HeldCloseBroker()
        broker.hold(0.4)
        broker.executions = [ConnectionError('venue down')]
        engine = _mk_engine(broker, store_ctx=ctx)
        _seed_parent_intent_and_leg(engine, with_leg=False)
        engine._route_event(_ct_unknown_execution(
            _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))
        _expire(engine)
        _recover(engine, int(time.time() * 1000))
        assert engine._position.size == pytest.approx(0.0) and engine.quarantined
        engine._flush_owed_store_writes()
        ctx.close()

        ctx2 = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        broker2 = _HeldCloseBroker()
        broker2.hold(0.4)
        broker2.executions = [OrderExecution(order=_venue_parent(OrderStatus.CANCELLED, 0.4))]
        engine2 = _mk_engine(broker2, store_ctx=ctx2)
        engine2._current_bar_ts_ms = BAR_TS
        assert engine2._entry_execution_unresolved('parent')
        _expire(engine2)
        _recover(engine2, int(time.time() * 1000))
        assert broker2.execution_reads == [('xchg-parent', None)]
        assert engine2._position.size == pytest.approx(0.4)
        assert not engine2._entry_execution_unresolved('parent')
        assert not engine2.quarantined
        engine2._persist_entry_book()
        engine2._flush_owed_store_writes()
        snapshot = engine2._position.entry_book_snapshot()
        assert [record['qty'] for record in snapshot['recovered_executions']] == [0.4]
        ctx2.close()


def __test_a_failed_audit_write_never_stops_the_recovery__(tmp_path):
    """The recovery's audit row is not part of the booking: a store that refuses it is logged."""
    with BrokerStore(tmp_path / "broker.sqlite", plugin_name="testbroker") as store:
        ctx = store.open_run(_run_identity(), script_source="src", script_path="t027.py")
        broker = _HeldCloseBroker()
        broker.hold(0.4)
        broker.executions = [OrderExecution(order=_venue_parent(OrderStatus.CANCELLED, 0.4))]
        engine = _mk_engine(broker, store_ctx=ctx)
        _seed_parent_intent_and_leg(engine, with_leg=False)
        engine._route_event(_ct_unknown_execution(
            _ct_order_event('cancelled', filled_qty=0.0, fill_qty=None)))
        _expire(engine)

        def refuse(*_args, **_kwargs):
            raise sqlite3.OperationalError("disk I/O error")

        ctx.log_event = refuse  # type: ignore[method-assign]
        _recover(engine, int(time.time() * 1000))
        assert engine._position.size == pytest.approx(0.4)
        assert not engine._entry_execution_unresolved('parent')
        ctx.close()


def __test_a_venue_close_over_an_expired_hold_then_the_recovery_ends_the_quarantine__():
    """The strategy reversed over the expired hold: the venue close, then the bot's own recovery.

    The venue close flattened the venue and its excess waits for the held
    order's execution; the recovery books that execution, which the excess
    closes at the venue close's price. The book never went short, the
    reversal's new side did not open while the cause stood, and the
    quarantine ends by itself with the book and the venue flat.
    """
    broker, engine, intent = _held_close_setup('held')
    _expire(engine)
    engine._diff_and_dispatch(_held_close_requests('reverse', intent, 0.0))
    (_kind, close), = broker.placed
    engine._route_event(_reversal_fill('xchg-close', close.pine_id, 'sell', 0.4, 'c-1',
                                       leg_type=LegType.CLOSE))
    assert engine._position.size == pytest.approx(0.0)
    broker.executions = [OrderExecution(order=_venue_parent(OrderStatus.CANCELLED, 0.4))]

    _recover(engine, int(time.time() * 1000))

    assert engine._position.size == pytest.approx(0.0)
    assert not engine._entry_execution_unresolved('parent')
    assert engine._held_execution_close is None
    assert not engine.quarantined
    assert [kind for kind, _placed in broker.placed] == ['close']


def __test_a_waited_price_reversal_is_sized_from_the_true_book_without_a_reissue__():
    """A limit reversal created over the booked 0.2 goes out closing the whole 0.4.

    The strategy sized it 1.2 (flip 0.2) and does not issue it again. Once
    the held execution is booked, the order goes out as 1.4 (flip 0.4), and an
    amend of its level keeps that flip.
    """
    broker, engine, intent = _held_close_setup()
    stale = _held_close_requests('limit_reverse', intent, 0.2)
    engine._diff_and_dispatch(stale)
    assert broker.placed == []
    engine._route_event(_late_fill(0.2))
    assert engine._position.size == pytest.approx(0.4)

    engine._diff_and_dispatch(stale)
    (kind, placed), = broker.placed
    assert kind == 'entry' and placed.qty == pytest.approx(1.4)
    assert engine._active_intents['short'].qty == pytest.approx(1.2)

    moved = [stale[0], dataclasses.replace(stale[1], limit=102.0)]
    resized = engine._resize_waited_dispatch(moved[1])
    assert resized.qty == pytest.approx(1.4) and resized.flip_qty == pytest.approx(0.4)


def __test_a_waited_close_all_closes_the_true_position__():
    """A close_all the strategy sized from the booked 0.2 closes the whole 0.4 once it goes out."""
    broker, engine, intent = _held_close_setup()
    stale = _held_close_requests('close_all', intent, 0.2)
    engine._diff_and_dispatch(stale)
    assert broker.placed == []
    engine._route_event(_late_fill(0.2))

    engine._diff_and_dispatch(stale)
    (kind, placed), = broker.placed
    assert kind == 'close' and placed.pine_id == '' and placed.qty == pytest.approx(0.4)


@pytest.mark.parametrize('held_side', ['buy', 'sell'])
def __test_an_unbooked_held_execution_never_masks_an_external_reduction__(held_side):
    """The venue holds the reported unbooked execution too: a real shrink is still seen.

    The book holds 0.2 of a long entry the venue reported 0.4 executed of. A
    venue of 0.3 is an external reduction of 0.1, not "more than the book".
    A held entry on the other side could explain a lower venue by itself:
    nothing is judged while it stands.
    """
    broker, engine, _intent = _held_close_setup()
    (_key, hold), = engine._fill_evidence_awaiting_execution.values()
    hold.side = held_side
    venue = ExchangePosition(symbol=SYMBOL, side='long', size=0.3, entry_price=100.0,
                             unrealized_pnl=0.0, liquidation_price=None, leverage=1.0,
                             margin_mode='cross')
    expected = 0.1 if held_side == 'buy' else 0.0
    assert engine._external_shrink_delta(venue) == pytest.approx(expected)


# === ENTRY_BLOCK causes end by themselves ========================================


def __test_an_entry_block_ends_with_its_last_cause_and_never_on_its_own_otherwise__():
    """Each cause ends by its producer; the latch ends with the last one, in this process.

    A cause-less request is refused and latches nothing (nothing could ever
    end it), and an external-intervention latch never ends by the bot,
    whatever its entry-block causes do.
    """
    events: list[BrokerEvent] = []
    engine = _mk_engine(_MockBroker(), events)
    engine.record_quarantine('first', cause='a')
    engine.record_quarantine('second', cause='b')
    assert engine.end_quarantine_cause('a', 'gone') is False
    assert engine.quarantined
    assert engine.end_quarantine_cause('missing', 'gone') is False
    assert engine.end_quarantine_cause('b', 'gone') is True
    assert not engine.quarantined and engine.quarantine_id is None
    assert [type(event) for event in events if isinstance(event, QuarantineEndedEvent)] == [
        QuarantineEndedEvent]

    with pytest.raises(ValueError, match='needs a cause'):
        engine.record_quarantine('plugin latch without a cause')
    assert not engine.quarantined and engine._entry_block_causes == {}
    engine.record_quarantine('third', cause='c')
    assert engine.end_quarantine_cause('c', 'gone') is True
    assert not engine.quarantined

    engine2 = _mk_engine(_MockBroker())
    engine2.record_quarantine('own trouble', cause='d')
    engine2.record_quarantine('someone moved the stop', kind=QuarantineKind.EXTERNAL_INTERVENTION)
    assert engine2.end_quarantine_cause('d', 'gone') is False
    assert engine2.quarantine_kind is QuarantineKind.EXTERNAL_INTERVENTION


def __test_an_ended_entry_block_lets_the_next_entry_out__():
    """Once the bot ended its quarantine the strategy's next entry is dispatched."""
    broker = _HeldCloseBroker()
    engine = _mk_engine(broker)
    engine.record_quarantine('close_rejected_repeatedly', cause='close_rejected:x')
    entry = EntryIntent(pine_id='long', symbol=SYMBOL, side='buy', qty=1.0,
                        order_type=OrderType.MARKET)
    engine._diff_and_dispatch([entry])
    assert broker.placed == []
    engine._resolve_entry_block_causes(now_ms=int(time.time() * 1000))
    assert not engine.quarantined
    engine._diff_and_dispatch([dataclasses.replace(entry, pine_id='long2')])
    assert [kind for kind, _placed in broker.placed] == ['entry']


def __test_a_protective_exit_cause_ends_once_the_unprotected_position_is_closed__():
    """The venue refused to protect entry L; the cause ends once nothing of L is open."""
    broker = _HeldCloseBroker()
    engine = _mk_engine(broker)
    _seed_parent_intent_and_leg(engine, with_leg=False)
    engine._route_event(_ct_order_event('partial', filled_qty=0.5, fill_qty=0.5,
                                        fill_id='x-1'))
    engine.record_quarantine('protective_exit_rejected', cause='protective_exit_rejected:parent')
    engine._resolve_entry_block_causes(now_ms=int(time.time() * 1000))
    assert engine.quarantined
    engine._position.record_fill(_reversal_fill('xchg-c', 'c', 'sell', 0.5, 'c-1',
                                                leg_type=LegType.CLOSE))
    assert not any(b.entry_id == 'parent' for b in engine._position.bindings)
    engine._resolve_entry_block_causes(now_ms=int(time.time() * 1000))
    assert not engine.quarantined


@pytest.mark.parametrize(('answer', 'expected', 'fraction'), [
    ('no_such_order', 'refused', 0.0),
    ('zero', 'refused', 0.0),
    ('full', 'sent', 0.0),
    ('part', 'partial', 0.25),
    ('working', 'working', 0.0),
    ('cannot_tell', None, 0.0),
])
def __test_a_close_no_record_settles_is_settled_from_the_venues_order_history__(
        answer, expected, fraction):
    """The bot reads a held partial-bracket close by its client order id instead of asking a human."""
    broker = _HeldCloseBroker()
    engine = _mk_engine(broker)

    def order(status, filled):
        return ExchangeOrder(
            id='xchg-c', symbol=SYMBOL, side='sell', order_type=OrderType.MARKET, qty=0.4,
            filled_qty=filled, remaining_qty=round(0.4 - filled, 12), price=None,
            stop_price=None, average_fill_price=100.0 if filled else None, status=status,
            timestamp=BAR_TS / 1000.0, fee=0.0, fee_currency='USDT')

    broker.executions = [{
        'no_such_order': OrderExecution(order=None),
        'zero': OrderExecution(order=order(OrderStatus.CANCELLED, 0.0)),
        'full': OrderExecution(order=order(OrderStatus.FILLED, 0.4)),
        'part': OrderExecution(order=order(OrderStatus.CANCELLED, 0.1)),
        'working': OrderExecution(order=order(OrderStatus.OPEN, 0.0)),
        'cannot_tell': None,
    }[answer]]
    evidence = engine._venue_close_outcome('close-coid')
    assert broker.execution_reads == [(None, 'close-coid')]
    if expected is None:
        assert evidence is None
        return
    assert evidence is not None and evidence.outcome == expected
    assert evidence.executed_fraction == pytest.approx(fraction)
    assert engine._venue_close_outcome('close-coid') is None, "the backoff holds the next read"


# === The fan's unsettled leg send, settled in this process ======================


def __test_a_held_fan_close_runs_the_leg_settlement_in_process_after_its_window__():
    """No restart and no operator: the bot settles the fan's unsettled leg send itself.

    The emulator's leg settlement runs first after the settle window (a send
    still in flight lands first), then with the backoff; a settlement that
    could not complete is tried again; nothing runs while the venue writes are
    blocked.
    """
    broker = _HeldCloseBroker(hedging=True)
    engine = _mk_engine(broker)
    calls: list[int] = []
    answers = [False, True]

    async def settle(port):
        calls.append(1)
        return answers.pop(0)

    engine._one_way_emulator.settle_pending_close_legs = settle  # type: ignore[method-assign]
    now = int(time.time() * 1000)
    engine._replay_held_fan_close('close-1', broker)
    assert calls == []
    state = engine._venue_recoveries['fanreplay:close-1']
    assert state.next_ms >= now + 60_000

    state.next_ms = 0
    engine._replay_held_fan_close('close-1', broker)
    assert calls == [1]
    engine._replay_held_fan_close('close-1', broker)
    assert calls == [1], "the backoff holds the next settlement"

    engine.record_quarantine('someone moved the stop', kind=QuarantineKind.EXTERNAL_INTERVENTION)
    state.next_ms = 0
    engine._replay_held_fan_close('close-1', broker)
    assert calls == [1], "no settlement sends while the venue writes are blocked"


@pytest.mark.parametrize('answer', [
    'fills_above_filled', 'fills_below_filled', 'other_order', 'filled_above_qty',
    'fill_without_id', 'executions_of_no_order',
])
def __test_an_execution_record_that_breaks_the_contract_books_nothing__(answer, caplog):
    """A record whose figures do not add up is refused: nothing booked, the quarantine stands.

    Every figure is in the journal unit. The bot asks again after its
    backoff; once the plugin answers consistently the gap is booked.
    """
    broker, engine, _intent = _held_close_setup()
    _expire(engine)
    stream_fill = _ct_order_event('partial', filled_qty=0.2, fill_qty=0.2, fill_id='x-1')
    late = _late_fill(0.2)
    order = _venue_parent(OrderStatus.CANCELLED, 0.4)
    bad = {
        'fills_above_filled': OrderExecution(order=order, fills=(stream_fill, late, _late_fill(
            0.2, fill_id='x-3'))),
        'fills_below_filled': OrderExecution(order=order, fills=(late,)),
        'other_order': OrderExecution(order=dataclasses.replace(order, id='xchg-other'),
                                      fills=(stream_fill, late)),
        'filled_above_qty': OrderExecution(order=dataclasses.replace(order, filled_qty=1.5)),
        'fill_without_id': OrderExecution(order=order, fills=(
            stream_fill, dataclasses.replace(late, fill_id=None))),
        'executions_of_no_order': OrderExecution(order=None, fills=(late,)),
    }[answer]
    broker.executions = [bad, OrderExecution(order=order, fills=(stream_fill, late))]
    now = int(time.time() * 1000)

    _recover(engine, now)
    assert engine._position.size == pytest.approx(0.2)
    assert engine.quarantined
    assert 'breaks the BrokerPlugin.fetch_order_execution contract' in caplog.text

    _recover(engine, now + int(_VENUE_RECOVERY_FIRST_S * 1000))
    assert engine._position.size == pytest.approx(0.4)
    assert not engine.quarantined
