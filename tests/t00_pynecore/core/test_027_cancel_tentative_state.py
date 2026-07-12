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
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from pynecore import lib
from pynecore.core.broker.exceptions import (
    OrderDispositionUnknownError,
)
from pynecore.core.broker.models import (
    BrokerEvent,
    CancelDispositionOutcome,
    CancelIntent,
    CapabilityLevel,
    DispatchEnvelope,
    EntryDeferredCancelDispositionPendingEvent,
    EntryIntent,
    ExchangeCapabilities,
    ExchangePosition,
    ExitIntent,
    ManualInterventionRequiredEvent,
    OcaPartialFillPolicy,
    OrderType,
    PartialBracketCancelTentativeDegradedEvent,
    PartialBracketCancelTentativeResolvedEvent,
    PartialBracketCancelTentativeStartedEvent,
)
from pynecore.core.broker.position import BrokerPosition
from pynecore.core.broker.software_partial_bracket_engine import (
    PartialBracketLeg,
    SoftwarePartialBracketEngine,
    _leg_from_row,
)
from pynecore.core.broker.storage import OrderRow
from pynecore.core.broker.store_helpers import (
    EXTRAS_KEY_CANCEL_TENTATIVE_SINCE_TS_MS,
    EXTRAS_KEY_INTENT_PARTIAL_QTY,
    EXTRAS_KEY_LEG_KIND,
    EXTRAS_KEY_LEG_STATE,
    EXTRAS_KEY_PARENT_ENTRY_DISPATCH_REF,
    EXTRAS_KEY_PARENT_PINE_ENTRY_ID,
    EXTRAS_KEY_TRIGGER_OFFSET,
    LEG_KIND_SL_PARTIAL,
    LEG_KIND_TP_PARTIAL,
    LEG_KIND_TRAIL_PARTIAL,
    LEG_STATE_ABORTED_PARENT_NEVER_ARRIVED,
    LEG_STATE_ARMED,
    LEG_STATE_CANCEL_TENTATIVE,
    LEG_STATE_PENDING_ENTRY,
    STATE_PARTIAL_BRACKET_LEG,
)
from pynecore.core.broker.sync_engine import OrderSyncEngine
from pynecore.lib.strategy import Trade
from pynecore.lib.strategy import (
    Order,
    _order_type_entry,
)
from types import SimpleNamespace


SYMBOL = "BTCUSDT"
RUN_TAG = "ct-test"
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
    cancel_outcomes: list[CancelDispositionOutcome] = field(default_factory=list)
    cancel_with_outcome_calls: list[DispatchEnvelope] = field(default_factory=list)
    cancel_calls: list[DispatchEnvelope] = field(default_factory=list)
    capabilities: ExchangeCapabilities = field(default_factory=lambda: ExchangeCapabilities(
        partial_qty_bracket_exit=CapabilityLevel.SOFTWARE,
    ))

    def get_capabilities(self) -> ExchangeCapabilities:
        return self.capabilities

    async def execute_cancel(self, envelope: DispatchEnvelope) -> bool:
        self.cancel_calls.append(envelope)
        if self.raise_on_cancel:
            raise self.raise_on_cancel.pop(0)
        return True

    async def execute_cancel_with_outcome(
            self, envelope: DispatchEnvelope,
    ) -> CancelDispositionOutcome:
        self.cancel_with_outcome_calls.append(envelope)
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
            if False:
                yield  # pragma: no cover — empty stream
        return _gen()


def _mk_engine(
        broker: _MockBroker,
        events: list[BrokerEvent] | None = None,
        cancel_tentative_stale_grace_s: float = 10.0,
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
    )


def _seed_parent_intent_and_leg(
        engine: OrderSyncEngine,
        *,
        intent_key: str = 'parent',
) -> EntryIntent:
    """Populate the engine state as if a parent EntryIntent had been
    dispatched and its partial brackets were registered. Returns the
    seeded EntryIntent so tests can pass it back into the engine
    internals."""
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


def __test_reconcile_stale_grace_promotes_to_degraded_halt__():
    """Stale-grace expiry ⇒ ManualInterventionRequiredEvent + halt latched."""
    events: list[BrokerEvent] = []
    broker = _MockBroker()
    engine = _mk_engine(broker, events=events,
                       cancel_tentative_stale_grace_s=1.0)
    intent = _seed_parent_intent_and_leg(engine)
    from pynecore.core.broker.sync_engine import _CancelTentativeMeta
    # Anchor 5s in the past — well beyond the 1s stale grace.
    engine._cancel_disposition_pending[intent.intent_key] = _CancelTentativeMeta(
        since_ts_ms=BAR_TS - 5_000, reason='broker_timeout',
    )
    engine._drive_cancel_tentative(now_ms=BAR_TS)
    # Halt latched.
    assert engine.halted is True
    assert engine._halted_reason == 'partial_bracket_cancel_disposition_unresolved'
    # Degraded-event emitted.
    degraded = [e for e in events
                if isinstance(e, PartialBracketCancelTentativeDegradedEvent)]
    assert len(degraded) == 1
    assert degraded[0].intent_key == intent.intent_key
    # ManualInterventionRequiredEvent emitted.
    mire = [e for e in events
            if isinstance(e, ManualInterventionRequiredEvent)]
    assert len(mire) == 1
    assert mire[0].reason == 'partial_bracket_cancel_disposition_unresolved'


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
