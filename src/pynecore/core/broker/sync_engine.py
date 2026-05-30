"""
:class:`OrderSyncEngine` — the bridge between the Pine Script order book and
a :class:`~pynecore.core.plugin.broker.BrokerPlugin`.

On each bar the engine:

1. Drains any :class:`OrderEvent` objects that the broker posted
   asynchronously (via :meth:`on_order_event`), routing fills to the
   :class:`~pynecore.core.broker.position.BrokerPosition` and unfreezing
   tick-based exits once their entry fill price is known.
2. Builds intents from the position's pending order dicts.
3. Runs the interceptor chain to let extensions reject or amend intents.
4. Diffs the resulting intent set against the previously-active one and
   dispatches the **new**, **modified** and **removed** intents to the
   plugin — tick-deferred exits are held back until the referenced entry
   has filled.
5. Every ``reconcile_every_n_syncs`` calls (optional) performs a read-side
   state reconciliation with the exchange.

The engine is synchronous; the broker plugin is async. :meth:`_run_async`
bridges the two, using ``run_coroutine_threadsafe`` on a background event
loop in live mode and ``asyncio.run`` for single-shot unit tests.
"""
from __future__ import annotations

import asyncio
import dataclasses
import itertools
import logging
import queue
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pynecore.core.broker.exceptions import (
    BracketAttachAfterFillRejectedError,
    BrokerManualInterventionError,
    ExchangeConnectionError,
    ExchangeOrderRejectedError,
    OrderDispositionUnknownError,
    OrderSkippedByPlugin,
)
from pynecore.core.broker.idempotency import (
    KIND_CLOSE,
    KIND_ENTRY,
    KIND_ENTRY_STOP,
    KIND_ENTRY_STOP_WATCH,
    KIND_EXIT_SL_PARTIAL,
    KIND_EXIT_TP_PARTIAL,
    KIND_EXIT_TRAIL_PARTIAL,
    build_client_order_id,
)
from pynecore.core.broker.intent_builder import build_intents
from pynecore.lib.log import (
    broker_info as _blog_info,
    broker_warning as _blog_warning,
    broker_error as _blog_error,
)
from pynecore.core.broker.models import (
    BracketAttachRejectContext,
    BrokerEvent,
    CancelDispositionOutcome,
    CancelIntent,
    CapabilityLevel,
    CloseIntent,
    DispatchEnvelope,
    EntryDeferredCancelDispositionPendingEvent,
    EntryIntent,
    ExchangePosition,
    ExitIntent,
    InterceptorResult,
    LegPartialRepairedEvent,
    LegRepairFailedEvent,
    LegType,
    ManualInterventionRequiredEvent,
    OcaPartialFillPolicy,
    OcaType,
    OrderEvent,
    OrderType,
    PartialBracketCancelTentativeDegradedEvent,
    PartialBracketCancelTentativeResolvedEvent,
    PartialBracketCancelTentativeStartedEvent,
    PendingDefensiveClose,
)
from pynecore.core.broker.native_failsafe_manager import (
    FailsafeHealth,
    FailsafeOwner,
    NativeBracketSnapshot,
    NativeFailsafeManager,
)
from pynecore.core.broker.software_partial_bracket_engine import (
    PartialBracketLeg,
    SoftwarePartialBracketEngine,
    TickOffsetResolver,
)
from pynecore.core.broker.software_entry_stop_engine import (
    EntryStopWatch,
    SoftwareEntryStopEngine,
)
from pynecore.core.broker.storage import (
    EnvelopeRecord,
    PendingRecord,
    RunContext,
)
from pynecore.core.broker.store_helpers import (
    EXTRAS_KEY_CANCEL_TENTATIVE_SINCE_TS_MS,
    EXTRAS_KEY_INTENT_PARTIAL_QTY,
    EXTRAS_KEY_LEG_KIND,
    EXTRAS_KEY_LEG_STATE,
    EXTRAS_KEY_OCA_GROUP,
    EXTRAS_KEY_OCA_TYPE,
    EXTRAS_KEY_PARENT_ENTRY_DISPATCH_REF,
    EXTRAS_KEY_PARENT_PINE_ENTRY_ID,
    EXTRAS_KEY_TRIGGER_LEVEL,
    EXTRAS_KEY_TRIGGER_OFFSET,
    LEG_KIND_SL_PARTIAL,
    LEG_KIND_TP_PARTIAL,
    LEG_KIND_TRAIL_PARTIAL,
    LEG_STATE_ACTIVE,
    LEG_STATE_ARMED,
    LEG_STATE_CANCEL_TENTATIVE,
    LEG_STATE_PENDING_ENTRY,
    ENTRY_STOP_STATE_ARMED,
    ENTRY_STOP_STATE_CANCEL_PENDING,
    ENTRY_STOP_STATE_MARKET_PENDING,
    create_engine_trigger_partial_leg_row,
    create_entry_stop_watch_row,
)
from pynecore.types.na import na_float

if TYPE_CHECKING:
    from pynecore.core.broker.position import BrokerPosition
    from pynecore.core.plugin.broker import BrokerPlugin

__all__ = ['OrderSyncEngine']

_log = logging.getLogger(__name__)

Intent = EntryIntent | ExitIntent | CloseIntent

CANCEL_TENTATIVE_STALE_GRACE_S = 10.0
"""Default stale-grace window (seconds) for cancel-tentative resolution.

A pending-entry partial bracket leg may enter ``cancel_tentative`` when
the broker returns :class:`OrderDispositionUnknownError` from
``execute_cancel`` (network timeout / ambiguous response). The
:meth:`OrderSyncEngine.reconcile` cancel-retry-loop drives the leg to
resolution within this window via
:meth:`BrokerPlugin.execute_cancel_with_outcome`. If the disposition
remains :attr:`CancelDispositionOutcome.UNKNOWN` past the deadline,
the engine promotes the parent to ``DEGRADED_HALT`` (the partial
brackets cannot be safely re-armed against a parent whose live/dead
status the engine cannot determine).

The default of 10s is more aggressive than
:data:`DEFENSIVE_CLOSE_RESOLUTION_GRACE_S` because the consequence is
more direct (unprotected position vs. waiting for an in-flight close
to settle). Config-tunable up to 30s for slow brokers via the
constructor.
"""


def _coid_is_close(coid: str) -> bool:
    """Return ``True`` when ``coid`` was minted for a :data:`KIND_CLOSE`
    dispatch.

    The canonical client-order-id format is
    ``{run_tag}-{pid_hash}-{bar}-{kind}{retry}`` (see
    :func:`pynecore.core.broker.idempotency.build_client_order_id`). The
    last dash-separated segment starts with the kind character — checking
    that single position is sufficient to distinguish a parked
    ``execute_close`` from a parked ``execute_entry`` / ``execute_exit``
    on the cross-restart pending-anchor recovery path.
    """
    tail = coid.rsplit('-', 1)
    if len(tail) != 2 or not tail[1]:
        return False
    return tail[1][0] == KIND_CLOSE


@dataclasses.dataclass
class _CancelTentativeMeta:
    """Per-``intent_key`` shadow-map entry tracking a cancel-tentative
    disposition that has not yet been resolved.

    ``since_ts_ms`` anchors the stale-grace deadline; ``retry_count``
    and ``last_retry_ts_ms`` give the cancel-retry-loop the audit data
    it needs to back off if the broker is consistently returning
    :attr:`CancelDispositionOutcome.UNKNOWN`. Mirrored on the
    persisted leg row's
    :data:`EXTRAS_KEY_CANCEL_TENTATIVE_SINCE_TS_MS` so a restart can
    rehydrate the deadline.
    """
    since_ts_ms: int
    reason: str
    retry_count: int = 0
    last_retry_ts_ms: int | None = None
    last_retry_outcome: CancelDispositionOutcome | None = None


DEFENSIVE_CLOSE_RESOLUTION_GRACE_S = 30.0
"""Default grace window (seconds) for a defensive close FILL to settle.

Used by :meth:`OrderSyncEngine.reconcile` when a plugin does not
override the value via
``BrokerPlugin.defensive_close_resolution_grace_s``. A pending marker
older than the grace window means the FILL we are waiting on has not
arrived in time and the engine halts so an operator can investigate.
"""


class _PartialBracketModifyDeferred(Exception):
    """Internal control-flow signal raised by :meth:`OrderSyncEngine._dispatch_modify`
    when an engine-trigger partial-bracket modify cannot be dispatched right now
    (e.g. the §2.6.7 broker-native fail-safe gate blocks the replacement).

    Caught by :meth:`OrderSyncEngine._diff_and_dispatch` to skip the
    ``_active_intents[key] = intent`` promotion: the old legs are still armed
    and the next sync must re-run the diff against the previous intent. Without
    this signal the diff would treat Pine == active and never retry once the
    fail-safe recovers, leaving the engine watching stale TP/SL/qty levels.

    Module-private — never propagates outside :mod:`sync_engine`.
    """


@dataclasses.dataclass(frozen=True)
class _EngineTriggerLegSpec:
    """Per-leg spec produced by :meth:`OrderSyncEngine._enumerate_engine_trigger_legs`.

    Bundles the four leg attributes the engine-trigger partial bracket
    dispatch needs to write a leg row and seed the in-memory
    :class:`~pynecore.core.broker.software_partial_bracket_engine.PartialBracketLeg`.
    All distance fields are normalised to price units (raw tick inputs
    are multiplied by ``mintick`` before reaching here).
    """
    leg_kind: str
    kind_code: str
    trigger_level: float | None
    trigger_offset: float | None
    trail_activation_level: float | None = None
    trail_activation_offset: float | None = None


class OrderSyncEngine:
    """Translate Pine orders to broker calls and route fills back.

    :param broker: The concrete :class:`BrokerPlugin` instance to drive.
    :param position: The live :class:`BrokerPosition` this engine updates.
    :param symbol: The trading symbol (as the plugin expects it).
    :param run_tag: 4-char base36 session tag (see
        :meth:`~pynecore.core.broker.run_identity.RunIdentity.make_run_tag`) — seeds
        every :class:`DispatchEnvelope` this engine builds, so restarting the
        same script under the same config regenerates the same
        ``client_order_id`` values and the exchange dedups duplicates.
    :param event_loop: A running ``asyncio`` loop on which to execute the
        broker's coroutines. Pass ``None`` for unit tests — each broker call
        will then spin up a transient loop via ``asyncio.run``.
    :param execute_timeout: Seconds to wait for any single ``execute_*``
        coroutine when bridging from a background loop.
    :param reconcile_every_n_syncs: If non-zero, perform a read-side
        reconciliation every N :meth:`sync` calls.
    :param mintick: Symbol minimum tick — used to resolve tick-based exits
        (``profit=`` / ``loss=`` / ``trail_points=``) into absolute prices.
    :param oca_partial_fill_policy: How OCA-cancel groups react to partial
        fills (see :class:`OcaPartialFillPolicy`). Defaults to
        :data:`OcaPartialFillPolicy.FILL_CANCELS` — matches the Pine
        backtester, which treats the first touch as the winning leg.
    :param broker_event_sink: Optional callable invoked for structured
        broker-side :class:`BrokerEvent` objects (bracket repairs, overfill
        guards, ...). ``None`` disables emission — useful in tests and
        single-shot backtests; production wires the runner's observability
        bus here.
    :param store_ctx: Optional :class:`RunContext` from the unified
        :class:`BrokerStore`. When provided the engine persists envelope
        identity and parked-verification entries through it; on construction
        the context is replayed (``SELECT``-ed from SQLite) so a restarted
        process re-uses the same ``client_order_id`` for every live intent
        and matches up parked dispatches against ``get_open_orders`` on the
        next sync. Pass ``None`` for unit tests and single-shot backtests
        where restart safety is not required. The context also carries the
        ``run_tag`` the engine needs — but the engine still takes ``run_tag``
        explicitly so test paths that do not use a storage context can run.
    """

    def __init__(
            self,
            broker: 'BrokerPlugin',
            position: 'BrokerPosition',
            symbol: str,
            *,
            run_tag: str,
            event_loop: asyncio.AbstractEventLoop | None = None,
            execute_timeout: float = 30.0,
            reconcile_every_n_syncs: int = 0,
            mintick: float = 0.01,
            minmove: float = 0.0,
            pricescale: int = 0,
            oca_partial_fill_policy: OcaPartialFillPolicy = OcaPartialFillPolicy.FILL_CANCELS,
            broker_event_sink: Callable[[BrokerEvent], None] | None = None,
            cancel_tentative_stale_grace_s: float = CANCEL_TENTATIVE_STALE_GRACE_S,
            store_ctx: RunContext | None = None,
    ) -> None:
        self._broker = broker
        self._position = position
        self._symbol = symbol
        self._run_tag = run_tag
        self._loop = event_loop
        self._timeout = execute_timeout
        self._reconcile_every = reconcile_every_n_syncs
        self._mintick = mintick
        # Tick-grid factors for the native fail-safe rounding (mintick ==
        # minmove / pricescale). Sentinel ``0`` means "grid unknown" — the
        # manager then leaves levels unrounded. Forwarded to
        # ``register_parent`` so observed broker stops snap to the same grid
        # the engine's worst-SL was computed on.
        self._minmove = minmove
        self._pricescale = pricescale
        self._oca_partial_policy = oca_partial_fill_policy
        self._broker_event_sink = broker_event_sink
        self._cancel_tentative_stale_grace_s = cancel_tentative_stale_grace_s
        self._store_ctx = store_ctx
        # Capabilities are declared once at plugin startup — cache the lookup
        # so the cascade-cancel fast path does not pay a method call per event.
        # Only the NATIVE level suppresses the engine fallback path:
        # PARTIAL_NATIVE keeps the engine running because the exchange only
        # owns part of the semantics, and we cannot safely guess which part.
        # Over-cancel / over-amend is idempotent; under-cancel leaves an open
        # exposure on the book.
        caps = broker.get_capabilities()
        self._oca_cancel_native = caps.oca_cancel is CapabilityLevel.NATIVE
        self._tp_sl_bracket_native = caps.tp_sl_bracket is CapabilityLevel.NATIVE
        # Capability cache for the partial-qty bracket dispatch switch.
        # The companion ``partial_qty_bracket_exit_pyramiding`` level
        # is enforced once at startup by ``validate_at_startup``;
        # the engine itself only needs the mode value at dispatch time.
        self._partial_qty_bracket_exit_mode = caps.partial_qty_bracket_exit
        # §2.6 broker-native fail-safe worst-SL manager (Slice A.7).
        # Like the partial-bracket state machine it is always
        # constructed — the dispatch path is the one that decides
        # whether any parent gets registered for fail-safe tracking.
        # The :attr:`_native_bracket_dispatcher` (set by the runner /
        # plugin opt-in) is the bridge to the actual ``PUT /positions``
        # call; without it, the manager still owns the state machine
        # and emits events but no broker traffic is generated. Slice B
        # wires the dispatcher on Capital.com.
        self._native_failsafe_manager = NativeFailsafeManager(
            event_sink=self._emit_broker_event,
        )
        self._native_bracket_dispatcher: (
            Callable[[NativeBracketSnapshot], None] | None
        ) = None
        # Engine-trigger partial bracket state machine. The state
        # machine itself is cheap to keep around (empty ledger when the
        # mode is UNSUPPORTED), so it is constructed unconditionally —
        # the dispatch switch decides whether it ever sees a leg.
        self._partial_bracket_engine = SoftwarePartialBracketEngine(
            store_ctx=store_ctx,
            state_change_listener=self._on_partial_bracket_leg_state_change,
        )
        # Software state machine for the STOP leg of a both-set Pine entry
        # (``strategy.entry(limit=, stop=)``). The LIMIT leg rests natively;
        # this engine watches the price and, on a stop cross, cancels the
        # native LIMIT and fires a MARKET order — so only one OCO leg is ever
        # native and a double-fill race is impossible. Constructed
        # unconditionally; armed only when a both-set entry is dispatched.
        self._entry_stop_engine = SoftwareEntryStopEngine(store_ctx=store_ctx)

        self._active_intents: dict[str, Intent] = {}
        self._order_mapping: dict[str, list[str]] = {}
        self._envelopes: dict[str, DispatchEnvelope] = {}
        self._pending_verification: dict[str, DispatchEnvelope] = {}
        # Keyed by ``ExitIntent.intent_key`` (``f"{pine_id}\x00{from_entry}"``)
        # so pyramiding entries with multiple tick-deferred exits per
        # ``from_entry`` each get their own slot.
        self._deferred_exits: dict[str, ExitIntent] = {}
        self._event_queue: queue.Queue[OrderEvent] = queue.Queue()
        # §2.6.7 broker-native fail-safe recovery feed. The plugin's reconcile
        # pass runs on the broker event-loop thread and enqueues observed
        # bracket triples here; they are applied to the manager on the MAIN
        # thread in :meth:`drive_native_failsafe` (mirroring how ``_event_queue``
        # marshals async fills). A plain ``queue.Queue`` is the thread-safe
        # hand-off; direct cross-thread ``on_native_bracket_observed`` calls
        # would race the main-thread ``recompute_worst_sl`` / ``tick_stale_window``.
        # Tuple shape: ``(ref, stop_level, profit_level, trailing_stop)``.
        self._native_bracket_observed_queue: queue.Queue[
            tuple[str, float | None, float | None, float | None]
        ] = queue.Queue()
        self._exchange_position: ExchangePosition | None = None
        self._interceptors: list[Callable[[Intent], InterceptorResult]] = []
        self._sync_count = 0
        self._current_bar_ts_ms: int = 0
        # Set during ``__init__`` when the persisted partial-bracket leg
        # ledger contains active rows: defers
        # :meth:`_rehydrate_native_failsafe_from_replayed_legs` to the
        # first :meth:`sync` call. ``_current_bar_ts_ms`` is still ``0``
        # at construction time, so anchoring ``register_parent``'s
        # ``degrading_since_ts_ms`` / ``recompute_worst_sl``'s
        # ``last_desired_change_ts_ms`` here would let the first real
        # :meth:`drive_native_failsafe` tick the stale window from epoch
        # zero and immediately promote the freshly rehydrated state to
        # ``DEGRADED`` — dropping the queued reattach PUT. Running the
        # rehydrate after :meth:`sync` assigns a real ``bar_ts_ms``
        # gives the stale-window timer a sane anchor.
        self._pending_native_failsafe_rehydrate: bool = False
        # Cancel-tentative state machine shadow map. Keyed by
        # ``intent_key``; each entry tracks one envelope whose broker
        # cancel disposition is unresolved
        # (:class:`OrderDispositionUnknownError` swallowed by
        # :meth:`_dispatch_cancel`). The ``reconcile()`` cancel-retry-
        # loop iterates this map every pass, re-invokes
        # :meth:`BrokerPlugin.execute_cancel_with_outcome`, and resolves
        # the entry to either a confirmed cancel (legs flip to
        # ``aborted_parent_never_arrived``, mapping/envelope dropped)
        # or a late-fill restore (legs return to ``pending_entry`` or
        # ``armed``). The diff-loop's adoption branch refuses to adopt
        # / dispatch any new intent whose key sits in this map — the
        # prior dispatch's broker state is still ambiguous, and a
        # parallel new dispatch would create a double-life. Stale-
        # grace expiry promotes the parent to ``DEGRADED_HALT``.
        self._cancel_disposition_pending: dict[str, _CancelTentativeMeta] = {}
        # Deferred rehydrate flag — analogous to
        # ``_pending_native_failsafe_rehydrate``. Set in ``__init__``
        # when the persisted leg ledger contains any
        # :data:`LEG_STATE_CANCEL_TENTATIVE` rows; cleared on the first
        # :meth:`sync` after the shadow map is rebuilt from the leg
        # extras (since_ts_ms / reason). Rebuilding at construction
        # time is not viable because ``_current_bar_ts_ms`` is still
        # ``0`` and the cancel-retry-loop's stale-grace timer would
        # immediately expire against epoch zero.
        self._pending_cancel_tentative_rehydrate: bool = False
        # OCA groups already processed inside the current :meth:`sync` pass.
        # Cleared at the start of every sync so a fresh bar re-enables cascade,
        # but kept stable within the pass so two fills in the same group do
        # not emit duplicate CancelIntents.
        self._cancelled_oca_groups_this_sync: set[str] = set()

        # Entry ids defensively closed during the current bar cycle via
        # :meth:`_handle_bracket_attach_after_fill_reject`. Populated after
        # :meth:`_cleanup_position_tracking` removes the entry/bracket
        # state; consulted by :meth:`_diff_and_dispatch` so that intents in
        # the precomputed ``new_map`` referencing the same ``from_entry``
        # (sibling brackets, alternate ``exit_id`` for the same parent,
        # OR an exit the user script re-emitted in the SAME bar after
        # observing the parent fill via :meth:`apply_async_events`) are
        # NOT re-dispatched as fresh orders into the just-flattened
        # position. Cleared at the END of :meth:`sync` (after
        # ``_diff_and_dispatch``) so a marker set by an ``apply_async_events``
        # drain that ran BEFORE the script on this bar survives until the
        # diff pass consumes it.
        self._defensively_closed_entries_this_sync: set[str] = set()

        # Defensive-close pending markers keyed by ``entry_id``.
        # Populated by :meth:`_handle_bracket_attach_after_fill_reject`
        # *before* the synthetic CloseIntent dispatches; cleared by
        # :meth:`_route_event` when the close FILL lands (or by startup
        # replay when the audit log shows the FILL already settled
        # pre-restart). While a marker lives, the parent entry intent
        # stays in ``_active_intents`` so :meth:`reconcile` cannot
        # mistakenly treat the soon-to-be-flat broker snapshot as an
        # external flatten and trip the duplicate-entry race the
        # design dossier calls out. Mirrored to BrokerStore extras
        # (``defensive_close_pending``) so a restart between dispatch
        # and FILL re-arms the same window via
        # :meth:`_replay_pending_defensive_closes`.
        self._pending_defensive_close: dict[str, PendingDefensiveClose] = {}

        # Entry ids whose :class:`PendingDefensiveClose` marker was
        # re-armed by :meth:`_replay_pending_defensive_closes` after a
        # process restart (rather than created in this process via
        # :meth:`_handle_bracket_attach_after_fill_reject`).
        #
        # The distinction matters for
        # :meth:`_broker_matches_post_close_expectation`: that helper's
        # subtraction logic assumes ``_position.size`` is the pre-close
        # aggregate, which is true intra-process but breaks across a
        # restart where startup adoption pulled the broker's already
        # reduced size into ``_position``. For replayed markers the
        # helper accepts the engine's current view directly as the
        # post-close state when it matches the broker, instead of
        # subtracting the qty a second time.
        #
        # Entries are removed when the marker is dropped (settlement,
        # halt, or explicit clear).
        self._replayed_defensive_close_entry_ids: set[str] = set()

        # Settled defensive-close identity cache used by :meth:`_route_event`
        # to deduplicate replayed FILL events before they reach
        # :meth:`BrokerPosition.record_fill`. A single defensive close can
        # be delivered to ``run_event_stream`` twice when the live WS
        # replays after a reconnect or the polled-orders fallback races the
        # stream — the second delivery would otherwise be applied to a flat
        # position by ``record_fill`` (opening a phantom opposite trade)
        # before :meth:`_route_defensive_close_fill` notices the marker is
        # already gone. Populated *after* the marker matches the first FILL,
        # consulted *before* ``record_fill`` runs on subsequent events whose
        # pine_id matches ``__pyne_defensive_close__`` or whose order id
        # matches a known settled close ref. In-memory only — a restart
        # before the second delivery falls back to the broker store's
        # ``defensive_close_filled`` audit event written by
        # :meth:`_route_defensive_close_fill`.
        self._settled_defensive_close_pine_ids: set[str] = set()
        self._settled_defensive_close_order_refs: set[str] = set()
        # ``client_order_id`` cache for parked defensive closes whose FILL
        # arrives with ``pine_id=None`` and a broker-allocated ``order.id``
        # that ``_settled_defensive_close_order_refs`` never observed
        # (the marker was created with ``close_order_ref=None`` because the
        # close parked via :class:`OrderDispositionUnknownError` and never
        # appeared in ``get_open_orders``). The exchange echoes the
        # dispatch-time ``client_order_id`` on every redelivery of the same
        # order, so this set is the only stable duplicate-detector for the
        # polled-orders identity path used by
        # :meth:`_route_defensive_close_fill`.
        self._settled_defensive_close_client_order_ids: set[str] = set()

        # Intent keys whose ``_order_mapping`` entry was seeded from a
        # recovered close-park anchor (see :meth:`_verify_pending_dispatches`).
        # The diff-loop adoption guard at :meth:`_diff_and_dispatch`
        # otherwise excludes every :class:`CloseIntent` from adoption to
        # avoid colliding with a stale parent-entry mapping left after an
        # ``ALREADY_FILLED`` cancel-tentative resolution. That blanket
        # exclusion is unsafe when the mapping itself originated from a
        # parked ``execute_close``: the script's next ``strategy.close``
        # would otherwise fall through to ``_dispatch_new`` and emit a
        # second market close against the just-recovered broker close
        # order. Membership unblocks adoption for the matching key
        # exactly once; consumed by the adoption branch and cleared on
        # any disposition that drops the close mapping.
        self._recovered_close_anchor_keys: set[str] = set()

        # Neutralised parent-ENTRY identity cache. When the no-FIFO settle
        # branch in :meth:`_route_event` settles a defensive close while
        # ``not _position.open_trades`` (re-armed marker, empty FIFO),
        # the parent ENTRY ``filled`` event may still be queued behind the
        # close FILL in the WS / poll stream — its arrival order is not
        # guaranteed because the bracket-attach reject is raised inside the
        # entry dispatch call, BEFORE the broker has emitted the entry
        # ``filled`` event (or the engine has drained it). When the close
        # FILL races ahead and lands first, the close handler settles
        # cleanly, but the delayed ENTRY ``filled`` event then walks an
        # empty FIFO and ``record_fill`` opens a fresh trade against a
        # broker that is already flat. Capture the parent's three identity
        # keys (pine_id, position_coid, intent_key) at no-FIFO settle time
        # so :meth:`_route_event` can drop the late ENTRY fill before it
        # corrupts position state. In-memory only — a restart between the
        # settle and the late ENTRY delivery falls back to startup
        # :meth:`reconcile`, which adopts the (now-flat) broker snapshot
        # and clears the parent intent without touching ``open_trades``.
        self._neutralised_parent_entry_pine_ids: set[str] = set()
        self._neutralised_parent_entry_coids: set[str] = set()

        # Entry ids whose FIFO ``Trade`` rows were just closed by the most
        # recent :meth:`BrokerPosition.record_fill` call. Captured in
        # :meth:`_route_event` from the ``new_closed_trades`` slice added
        # by the FILL and consumed by :meth:`_route_defensive_close_fill`
        # so its cleanup targets the actually-reduced entries rather than
        # blindly ``marker.entry_id``. In pyramiding (LongA + LongB, with
        # a bracket-attach reject on the newer LongB), FIFO closes LongA
        # but ``marker.entry_id`` is LongB — without this list the
        # cleanup would clear LongB's intents while LongA's intents stay
        # live against a Trade that no longer exists, corrupting later
        # exits and P&L. List (not set) to preserve walk order for
        # callers that care; entries are not deduplicated here because
        # ``_cleanup_position_tracking`` is idempotent.
        self._last_fifo_closed_entry_ids: list[str] = []

        # Keys that received an empty :attr:`_order_mapping` adoption marker
        # from :meth:`_consume_plugin_resolutions` during the current sync.
        # The marker tells :meth:`_diff_and_dispatch` "this intent is already
        # live, just adopt it" — but if no Pine intent shows up under the same
        # key in the same sync, the marker becomes a stale trap that would
        # silently absorb a *future* same-key dispatch into the gone-position's
        # slot. End-of-sync cleanup drops markers that no intent claimed.
        self._attached_adoption_keys: set[str] = set()

        # Per-run dedup for ``'attached'`` resolutions. The persisted row
        # is intentionally NOT deleted on the first attached consume —
        # otherwise a late ``'rejected'`` write (per-leg bracket
        # resolvers, cross-poll re-evaluation) would update zero rows
        # and the engine would never learn the leg is missing. Keeping
        # the row alive lets the sticky-rejected SQL flip it; this set
        # prevents re-processing the same attached row on every sync
        # while it lives. Cleared on ``'rejected'`` flip-back so a
        # future re-park (different coid) is processed correctly.
        self._consumed_attached_coids: set[str] = set()

        # Pre-modify intent rollback table — keyed by ``intent_key``, set
        # only when ``_dispatch_modify`` parks a timed-out amend. The
        # current ``_diff_and_dispatch`` flow promotes ``_active_intents[key]``
        # to the NEW intent immediately after ``_dispatch_modify`` returns
        # (line 1328), so by the time a parked modify resolves as
        # ``'rejected'`` the engine no longer remembers the pre-modify
        # state — leaving ``_active_intents[key]`` set to the NEW intent
        # would make the next diff observe Pine == active and never retry
        # the amend, leaving the original exchange order indefinitely
        # stale. Restoring from this dict on the rejected path forces
        # the next diff to see a Pine-vs-active delta again and re-emit
        # ``modify_*``.
        #
        # Limitation: in-memory only — a restart between park and
        # resolution loses the snapshot, so the post-restart rejected
        # path falls back to the previous Round 19 behaviour (preserve
        # promoted active, accept the slim chance of a stale amend).
        # Persisting the snapshot on the ``pending_verifications`` row
        # would close that gap but requires an :class:`Intent`
        # serialisation contract; not worth the surface area until a
        # real restart-during-park incident motivates it.
        self._modify_old_intents: dict[str, Intent] = {}

        # Manual-intervention halt flag. Once set (via :meth:`_record_halt`),
        # every subsequent :meth:`sync` returns early without dispatching or
        # draining events — the strategy must be restarted after the operator
        # resolves the broker-side ambiguity. Plugins signal the halt by
        # raising :class:`BrokerManualInterventionError` from any ``execute_*``
        # or ``watch_orders``; the engine catches once, emits a
        # :class:`ManualInterventionRequiredEvent`, and re-raises so the
        # runner performs a graceful stop.
        self._halted: bool = False
        self._halted_reason: str | None = None
        self._halted_intent_key: str | None = None
        self._halted_context: dict = {}

        # Cross-restart recovery anchors. The state store persists envelope
        # identity and parked-verification entries; replay rebuilds these
        # *anchor* dicts (intent objects are not persisted — they are rebuilt
        # from the Pine order book on the first post-restart sync). The first
        # _build_envelope / _verify_pending_dispatches call for an anchored key
        # promotes the anchor into the live in-memory state and clears it.
        # ``sync()`` re-replays both caches at the start of every cycle
        # so a slow plugin ``connect()`` whose ``_retire_startup_orphans``
        # finishes between syncs cannot leave a stale anchor that
        # ``_build_envelope`` would pop into a recycled ``client_order_id``.
        self._persisted_envelope_anchors: dict[str, EnvelopeRecord] = {}
        self._persisted_pending_anchors: dict[str, PendingRecord] = {}
        if store_ctx is not None:
            envelopes, pending = store_ctx.replay()
            self._persisted_envelope_anchors = dict(envelopes)
            self._persisted_pending_anchors = self._unresolved_pending(pending)
            if envelopes or pending:
                _log.info(
                    "broker state replay: %d envelope(s), %d pending verification(s)",
                    len(envelopes), len(pending),
                )
            # Rebuild the engine-trigger partial bracket ledger from
            # persisted leg rows. Without this call a process restart
            # with an active partial bracket leaves the in-memory
            # ledger empty while the rows live on in SQLite — the
            # WATCH phase would never re-arm them, and the dispatch
            # guard would happily accept a duplicate bracket on the
            # same parent. See the partial-qty bracket exit design
            # dossier §3.5.
            self._partial_bracket_engine.restart_replay()
            # Rebuild the both-set entry-stop watch ledger from persisted
            # watch rows. Latched intermediate states (cancel_pending /
            # stop_market_pending) are reloaded as-is; the first
            # :meth:`_drive_entry_stop_triggers` re-drives them
            # deterministically (idempotent cancel / idempotent market POST),
            # so no demotion or price re-evaluation is needed.
            self._entry_stop_engine.restart_replay()
            # Rehydrate the broker-native fail-safe manager from the
            # replayed legs. ``NativeFailsafeManager`` was just
            # constructed empty, so without this call there is no
            # ``NativeStopState`` for any parent that owned partial
            # legs across the restart — the §2.6.7 worst-SL recompute
            # path returns early, no broker-native stop snapshot is
            # queued, and the degraded-state gates do not fire until a
            # new leg transition happens. Walk the replayed legs and
            # register each unique parent + trigger an immediate
            # worst-SL recompute so the safety state machine reflects
            # the persisted partial bracket.
            #
            # Defer the work to the first :meth:`sync`: at construction
            # time ``_current_bar_ts_ms`` is still ``0``, and running
            # rehydrate with that anchor would let the first real
            # :meth:`drive_native_failsafe` tick the stale window from
            # epoch zero and immediately demote the freshly rehydrated
            # state to ``DEGRADED``. The deferred flag is only set when
            # there is something to rehydrate (active legs present); a
            # restart without partial brackets is a no-op.
            self._pending_native_failsafe_rehydrate = any(
                True for _ in self._partial_bracket_engine.iter_legs()
            )
            # Cancel-tentative shadow-map rehydrate: defer to the first
            # :meth:`sync` so ``_current_bar_ts_ms`` is non-zero before
            # the stale-grace timer runs. The flag is only set when the
            # restart-replay surfaced at least one cancel-tentative leg
            # (the engine's ``iter_cancel_tentative_intent_keys`` peek
            # against the now-rehydrated ledger).
            self._pending_cancel_tentative_rehydrate = bool(
                self._partial_bracket_engine.iter_cancel_tentative_parents()
            )

    # === Public API ===

    def refresh_anchors_from_store(self) -> None:
        """Re-read persisted envelope and pending anchors from the store.

        The engine loads ``_persisted_envelope_anchors`` /
        ``_persisted_pending_anchors`` in ``__init__`` and re-replays them
        at the start of every :meth:`sync` cycle. This method exists for
        callers that need an explicit mid-bar refresh (currently the
        runner's ``start_broker`` pre-sync handshake, kept as a
        belt-and-braces fence against pre-first-sync code that touches
        ``_persisted_envelope_anchors`` directly).

        Safe to call repeatedly. No-op when ``store_ctx`` is unset
        (tests, single-shot backtests).
        """
        if self._store_ctx is None:
            return
        envelopes, pending = self._store_ctx.replay()
        self._persisted_envelope_anchors = dict(envelopes)
        self._persisted_pending_anchors = self._unresolved_pending(pending)

    @staticmethod
    def _unresolved_pending(
            pending: dict[str, PendingRecord],
    ) -> dict[str, PendingRecord]:
        """Strip plugin-resolved rows from a replayed pending-anchor map.

        ``RunContext.replay`` returns every ``pending_verifications`` row
        for the run, including those a plugin has already flipped to
        ``'attached'`` or ``'rejected'`` via ``record_resolution``. Those
        rows are intentionally kept in SQLite so a late per-leg
        ``'rejected'`` flip can still be observed against an earlier
        ``'attached'`` write (see :meth:`_consume_plugin_resolutions`'s
        docstring), but they have no business in
        ``_persisted_pending_anchors``: that map drives
        :meth:`_verify_pending_dispatches`'s ``get_open_orders`` matching
        path, which is only meaningful for *unresolved* parked dispatches.

        Leaving resolved rows in the map would force every sync after the
        first attached consume to call ``get_open_orders`` for COIDs the
        plugin has already accounted for. The in-memory
        ``_consumed_attached_coids`` dedup short-circuits the resolution
        bookkeeping but does **not** re-pop the anchor (the row's pop
        already happened on the first consume), so after an in-process
        replay the anchor is back. A transient connection error on that
        spurious call would skip the entire sync via the
        ``ExchangeConnectionError`` guard in :meth:`sync` — a correctness
        regression for plugins whose protective brackets resolve via
        ``record_resolution`` rather than ``get_open_orders`` (e.g.
        Capital.com's position-attached TP/SL).

        Resolved rows are still picked up directly from SQLite by
        :meth:`RunContext.iter_pending_resolutions` in
        :meth:`_consume_plugin_resolutions`; filtering them here loses no
        information.
        """
        return {
            coid: record
            for coid, record in pending.items()
            if record.resolution is None
        }

    @property
    def active_intents(self) -> dict[str, Intent]:
        return self._active_intents

    @property
    def deferred_exits(self) -> dict[str, ExitIntent]:
        return self._deferred_exits

    @property
    def pending_defensive_close(self) -> dict[str, PendingDefensiveClose]:
        """In-memory map of in-flight defensive-close markers, keyed by entry id."""
        return self._pending_defensive_close

    @property
    def order_mapping(self) -> dict[str, list[str]]:
        return self._order_mapping

    @property
    def pending_verification(self) -> dict[str, DispatchEnvelope]:
        """Envelopes whose exchange-side disposition is still unknown."""
        return self._pending_verification

    @property
    def exchange_position(self) -> ExchangePosition | None:
        """Latest position snapshot returned by the broker, if any."""
        return self._exchange_position

    def register_interceptor(
            self, fn: Callable[[Intent], InterceptorResult],
    ) -> None:
        """Add an interceptor that may reject or amend intents before dispatch."""
        self._interceptors.append(fn)

    def on_order_event(self, event: OrderEvent) -> None:
        """Queue a broker :class:`OrderEvent` for processing on the next sync.

        Called from the :meth:`run_event_stream` background task or by
        tests injecting synthetic events.
        """
        self._event_queue.put(event)

    async def run_event_stream(self) -> None:
        """Drain :meth:`BrokerPlugin.watch_orders` into the event queue.

        Meant to run as a long-lived task on the shared live-provider event
        loop. If the plugin does not implement WebSocket streaming, the
        method logs and returns — the engine then relies on
        :meth:`reconcile` for fill detection.

        Each incoming event is logged immediately on arrival — the actual
        :meth:`_route_event` processing is deferred to the next
        :meth:`_drain_events` call (i.e. the next bar's :meth:`sync`), so
        a log emitted only at drain time would falsely tag the fill with
        the *next* bar's ``bar_index``.  Logging here, on the broker event
        loop, captures the moment the broker actually observed the
        transition.
        """
        try:
            stream = self._broker.watch_orders()
        except NotImplementedError:
            _log.info(
                "broker does not implement watch_orders; "
                "reconcile() will poll for fills instead",
            )
            return
        try:
            async for event in stream:
                _blog_info("event %s", event)
                self._event_queue.put(event)
        except NotImplementedError:
            _log.info(
                "broker does not implement watch_orders; "
                "reconcile() will poll for fills instead",
            )
            return
        except asyncio.CancelledError:
            raise
        except BrokerManualInterventionError as e:
            self._record_halt(e)
            raise
        except Exception:  # pragma: no cover — defensive
            _log.exception("watch_orders stream terminated with an error")
            raise

    @property
    def halted(self) -> bool:
        """``True`` once :meth:`_record_halt` has latched a manual-intervention halt."""
        return self._halted

    def raise_if_halted(self) -> None:
        """Re-raise the latched halt as :class:`BrokerManualInterventionError`.

        Cheap, drain-free check meant for the script runner's tick loop:
        an async halt set on the broker event-loop thread (e.g. from
        :meth:`run_event_stream` reacting to an
        :class:`UnexpectedCancelError`) should surface in the runner thread
        on the very next tick — *not* one full bar later when
        :meth:`apply_async_events` runs at bar close.
        """
        if self._halted:
            raise BrokerManualInterventionError(
                self._halted_reason or "manual intervention required",
                intent_key=self._halted_intent_key,
                context=dict(self._halted_context),
            )

    def apply_async_events(self) -> None:
        """Drain any async-arrived broker events into the position state.

        Call this from the script runner BEFORE running the user script on
        each bar.  Without it, fills observed asynchronously between bars
        only become visible to ``position.size`` when the next bar's
        :meth:`sync` runs (i.e. AFTER that bar's script has executed),
        leaving the script's view of the position one bar stale.

        Also propagates an async-recorded halt (e.g. an
        :class:`UnexpectedCancelError` from ``run_event_stream``) so the
        bar loop exits via its ``finally`` block instead of running the
        script with stale state.
        """
        self.raise_if_halted()
        self._drain_events()

    def sync(self, bar_ts_ms: int, *, last_price: float | None = None) -> None:
        """Run one diff/dispatch cycle.

        Reads the Pine order book from ``position.entry_orders`` and
        ``position.exit_orders``, resolves tick-deferred exits where the
        referenced entry price is now known, and dispatches whatever
        changed to the broker plugin.

        :param bar_ts_ms: Current bar open timestamp in milliseconds — seeds
            every :class:`DispatchEnvelope` built in this cycle. The caller
            (typically the script runner) sources this from ``lib._time``.
        :param last_price: Last observed price for the script's symbol —
            sourced from ``lib.close`` at the call site. Drives the
            engine-trigger partial-bracket WATCH phase
            (:meth:`SoftwarePartialBracketEngine.on_price_tick`). ``None``
            (the default) suppresses the WATCH phase this cycle, which is
            the right behaviour for callers that have no price context yet
            (e.g. startup before the first bar has been ingested).
        """
        # Surface a latched halt before any state mutation so an async halt
        # triggered from ``run_event_stream`` (e.g. an
        # :class:`UnexpectedCancelError` observed by the polling plugin)
        # exits the bar loop via its ``finally`` block instead of letting
        # the engine silently keep iterating.
        self.raise_if_halted()
        # Re-replay persisted anchors at the start of every sync to keep
        # the in-memory caches aligned with the journal. The runner's
        # pre-sync refresh races with the broker plugin's ``connect()``
        # orphan-retire pass when ``live_ohlcv_generator`` times out and
        # proceeds before connect completes; ``_retire_startup_orphans``
        # finishing between any two syncs would otherwise leave a stale
        # anchor that the next ``_build_envelope`` pops into a recycled
        # ``client_order_id`` (the journal's ``upsert_order`` then
        # UPDATEs a closed row instead of inserting a fresh one — the
        # row stays invisible to ``iter_live_orders`` and the next
        # ``execute_exit`` raises ``no confirmed entry row``).
        #
        # Safe to re-populate wholesale: ``_build_envelope`` checks the
        # live in-memory ``_envelopes`` map FIRST, so an anchor we
        # already promoted and consumed cannot be resurrected — its DB
        # row is still present (the envelope is live, not yet completed)
        # and its in-memory entry takes precedence. An anchor whose
        # ``record_complete`` already DELETEd the row simply does not
        # come back. For ``_persisted_pending_anchors`` we additionally
        # strip plugin-resolved rows via :meth:`_unresolved_pending`:
        # those rows are intentionally kept in SQLite so a late
        # ``'rejected'`` flip can still be observed, but they have no
        # business in the map that drives ``get_open_orders`` matching
        # (see the helper's docstring for the full rationale).
        if self._store_ctx is not None:
            envelopes, pending = self._store_ctx.replay()
            self._persisted_envelope_anchors = dict(envelopes)
            self._persisted_pending_anchors = self._unresolved_pending(pending)
        self._current_bar_ts_ms = bar_ts_ms
        # Restart-replay native fail-safe rehydrate: deferred from
        # ``__init__`` so ``_current_bar_ts_ms`` carries a real epoch-ms
        # anchor before any :meth:`register_parent` /
        # :meth:`recompute_worst_sl` call. Running it here — after the
        # assignment above and before any state mutation — guarantees
        # that the stale-window timer in :meth:`drive_native_failsafe`
        # (called at the end of this ``sync``) measures from the bar
        # timestamp, not from epoch zero, so the freshly rehydrated
        # ``DEGRADING`` state stays in ``DEGRADING`` long enough for the
        # queued reattach PUT to dispatch.
        if self._pending_native_failsafe_rehydrate:
            self._pending_native_failsafe_rehydrate = False
            self._rehydrate_native_failsafe_from_replayed_legs()
        if self._pending_cancel_tentative_rehydrate:
            self._pending_cancel_tentative_rehydrate = False
            self._rehydrate_cancel_tentative_from_replayed_legs()
        self._cancelled_oca_groups_this_sync.clear()
        # ``_defensively_closed_entries_this_sync`` is intentionally NOT
        # cleared here: :meth:`_handle_bracket_attach_after_fill_reject`
        # can populate it from :meth:`apply_async_events` (which runs
        # BEFORE the script on every bar) when a queued entry fill
        # triggers :meth:`_resolve_deferred_for_entry` → ``_dispatch_new``
        # → bracket-attach reject. Clearing at the top of ``sync`` would
        # erase that marker before ``_diff_and_dispatch`` consumes it,
        # letting a sibling/recreated bracket re-dispatch into the just
        # defensively-closed position. The clear is deferred to the END
        # of ``sync`` instead, after ``_diff_and_dispatch``.
        # Drain again here in case events arrived between
        # ``apply_async_events`` (start of this bar) and now.  ``sync`` is
        # also called from contexts that don't pre-drain (e.g. tests, the
        # backtest path with broker mode), so this remains the safety net.
        self._drain_events()
        try:
            self._verify_pending_dispatches()
        except ExchangeConnectionError as e:
            _blog_warning(
                "sync skipped after pending dispatch verification connection error: %s",
                e,
            )
            return

        raw = build_intents(
            self._position.entry_orders,
            self._position.exit_orders,
            self._symbol,
            self._position.open_trades,
        )
        resolved = [self._resolve_ticks(i) for i in raw]
        final = self._apply_interceptors(resolved)

        dispatchable: list[Intent] = []
        new_deferred: dict[str, ExitIntent] = {}
        for i in final:
            if isinstance(i, ExitIntent) and i.has_unresolved_ticks:
                new_deferred[i.intent_key] = i
            else:
                dispatchable.append(i)
        self._deferred_exits = new_deferred

        self._diff_and_dispatch(dispatchable)
        self._cleanup_unused_adoption_markers()
        # Engine-trigger partial bracket WATCH phase. Fires before
        # ``drive_native_failsafe`` so a leg that triggers in this
        # tick can flip to ``triggered`` (and the OCA cascade can run)
        # before the failsafe pass observes the new leg-state set —
        # otherwise the worst-SL recompute would still see the now-fired
        # leg as armed and emit a stale broker-native level for one bar.
        self._drive_partial_bracket_triggers(last_price=last_price)
        # Both-set entry STOP leg WATCH phase: a stop cross cancels the
        # native LIMIT and (only once that cancel is confirmed) fires a
        # MARKET order. Latched cancel_pending / stop_market_pending watches
        # are re-driven here too (restart / retry), independent of price.
        self._drive_entry_stop_triggers(last_price=last_price)
        # Drain native failsafe dispatches into the plugin dispatcher
        # (§2.6 worst-SL drive loop). The manager accumulates snapshots
        # via the leg state-change listener / ``recompute_worst_sl``;
        # without this drain the broker-native stop would never get
        # placed or refreshed in live runs. Idempotent — a no-op when
        # no plugin dispatcher is installed or the manager has nothing
        # queued.
        self.drive_native_failsafe(now_ms=float(self._current_bar_ts_ms))
        # End-of-sync clear: the defensive-close markers populated during
        # this sync (whether by an ``apply_async_events`` drain on this
        # bar or by the in-line ``_drain_events`` above) have now been
        # consumed by ``_diff_and_dispatch``. Reset so the next bar
        # starts fresh. If the sync above returned early (e.g. connection
        # error in ``_verify_pending_dispatches``) we deliberately leave
        # the set intact for the next sync attempt to consume.
        self._defensively_closed_entries_this_sync.clear()

        self._sync_count += 1
        if self._reconcile_every and self._sync_count % self._reconcile_every == 0:
            try:
                self.reconcile()
            except ExchangeConnectionError as e:
                _blog_warning(
                    "periodic reconcile skipped after connection error: %s",
                    e,
                )

    def _verify_pending_dispatches(self) -> None:
        """Match parked timeouts against the exchange's open-orders view.

        When a plugin raises :class:`OrderDispositionUnknownError` the sync
        engine cannot tell whether the order landed on the exchange; it parks
        the envelope here. Every subsequent :meth:`sync` calls this method
        first: it queries ``get_open_orders`` and, for each pending
        ``client_order_id`` that now appears on the exchange, promotes the
        envelope back into ``_order_mapping`` without re-dispatching.

        After a restart the persisted parked entries are also matched here —
        the in-memory envelope is gone, but the persisted ``key`` is enough to
        attach the recovered exchange order to the right ``_order_mapping``
        slot.

        For dispatches whose disposition the broker cannot expose through
        ``get_open_orders`` (e.g. position-attached brackets on Capital.com,
        which never show up there once attached), the plugin writes a
        resolution into the persisted park row via
        :meth:`~pynecore.core.broker.storage.RunContext.record_resolution`.
        This method consumes those resolutions first: ``'attached'`` clears
        the park (the dispatch is live, leave the active intent alone),
        ``'rejected'`` clears the park *and* drops the active intent so the
        next sync re-dispatches the original Pine intent.

        A pending entry that does *not* show up stays parked — the engine
        deliberately does not re-dispatch because the original may still land
        (slow network round-trip). The user can inspect
        :attr:`pending_verification` to surface stuck entries.
        """
        if self._store_ctx is not None:
            self._consume_plugin_resolutions()
        if not self._pending_verification and not self._persisted_pending_anchors:
            return
        orders = self._run_async(self._broker.get_open_orders(self._symbol))
        by_coid = {o.client_order_id: o for o in orders if o.client_order_id}
        for coid in list(self._pending_verification):
            order = by_coid.get(coid)
            if order is None:
                continue
            envelope = self._pending_verification.pop(coid)
            key = envelope.intent.intent_key
            current = self._order_mapping.setdefault(key, [])
            if order.id not in current:
                current.append(order.id)
            if self._store_ctx is not None:
                self._store_ctx.record_unpark(coid)
            self._maybe_attach_defensive_close_ref(key, order.id)
            # Tag close-park recoveries so the diff-loop adoption guard
            # can adopt a re-emitted ``CloseIntent`` instead of falling
            # through to ``_dispatch_new`` and emitting a duplicate
            # market close. See the ``_persisted_pending_anchors`` loop
            # below for the cross-restart counterpart.
            if isinstance(envelope.intent, CloseIntent):
                self._recovered_close_anchor_keys.add(key)
            _log.info(
                "recovered pending dispatch %s -> exchange order %s "
                "for intent %s", coid, order.id, key,
            )
        for coid in list(self._persisted_pending_anchors):
            order = by_coid.get(coid)
            if order is None:
                continue
            anchor = self._persisted_pending_anchors.pop(coid)
            current = self._order_mapping.setdefault(anchor.key, [])
            if order.id not in current:
                current.append(order.id)
            if self._store_ctx is not None:
                self._store_ctx.record_unpark(coid)
            self._maybe_attach_defensive_close_ref(anchor.key, order.id)
            # A close-park's COID encodes :data:`KIND_CLOSE` in its
            # trailing kind+retry segment. Tag the seeded key so the
            # diff-loop adoption guard can recognise the mapping as a
            # genuine close anchor rather than a stale parent-entry
            # leftover, allowing the script's re-emitted
            # :class:`CloseIntent` to adopt instead of double-dispatching.
            if _coid_is_close(coid):
                self._recovered_close_anchor_keys.add(anchor.key)
            _log.info(
                "recovered persisted pending dispatch %s -> exchange order %s "
                "for intent %s", coid, order.id, anchor.key,
            )

    def _maybe_attach_defensive_close_ref(
            self, intent_key: str, order_id: str,
    ) -> None:
        """Backfill ``close_order_ref`` when a parked defensive close lands.

        :meth:`_handle_bracket_attach_after_fill_reject` only stamps the
        broker-side close ref on the synchronous success path — when the
        dispatch parks via :class:`OrderDispositionUnknownError`, the marker
        is armed but ``close_order_ref`` stays ``None``. Once
        :meth:`_verify_pending_dispatches` (or its persisted-anchor sibling)
        recovers the parked order from ``get_open_orders``, the matching
        marker must learn the recovered ``order.id`` — otherwise the
        eventual FILL routed via the polled-orders path (where
        ``OrderEvent.pine_id`` is unreliable) cannot be matched in
        :meth:`_route_defensive_close_fill`, leaving the marker armed
        until the stale-pending halt fires even though the close filled.

        Recovering the parked close is also the first moment the engine
        can safely cancel the bracket residuals the parked path skipped:
        :meth:`_handle_bracket_attach_after_fill_reject` runs the
        dispatch-time residual cancel only on the synchronous success
        branch (``dispatch_succeeded == True``). When the dispatch parks,
        residuals (partial entry remainder, separate TP/SL orders, ...)
        stay live exchange-side for the entire dispatch → FILL window —
        a window that can stretch arbitrarily long depending on broker
        latency. Cancelling them here once we *know* the close landed
        narrows that exposure to roughly the open-orders-poll interval
        instead of waiting for the FILL itself. The plugin's
        :meth:`~pynecore.core.plugin.broker.BrokerPlugin.cancel_broker_order_ref`
        idempotency contract makes the eventual retry inside
        :meth:`_route_defensive_close_fill` a safe no-op when this
        attempt already cleared everything. On transient failure we
        stamp ``residual_cleanup_pending=True`` so
        :meth:`_retry_residual_cleanup_after_transient_fill` replays the
        cancel on the next reconcile rather than waiting for the close
        FILL to land.
        """
        if not self._pending_defensive_close:
            return
        for entry_id, marker in self._pending_defensive_close.items():
            if (marker.close_intent_key == intent_key
                    and marker.close_order_ref is None):
                updated_marker = dataclasses.replace(
                    marker, close_order_ref=order_id,
                )
                self._set_pending_defensive_close(entry_id, updated_marker)
                # Mirror the synchronous-success ``natural_close_at`` stamp
                # now that the parked close is confirmed live on the
                # exchange. Without it, brokers whose position disappears
                # before the close activity is ingested would let the
                # plugin-side missing-position reconciler see the
                # unstamped parent row as an unexpected manual close and
                # halt the run.
                self._stamp_natural_close_at(
                    updated_marker.reject_context.position_coid,
                )
                try:
                    self._cancel_bracket_reject_residuals(
                        updated_marker.reject_context,
                        raise_on_transient=True,
                    )
                except (ExchangeConnectionError,
                        OrderDispositionUnknownError) as exc:
                    _blog_warning(
                        "residual cancel after parked defensive-close "
                        "recovery transiently failed for entry %s (%s: %s) "
                        "— marker flagged for retry via reconcile",
                        updated_marker.entry_id,
                        type(exc).__name__, exc,
                    )
                    if not updated_marker.residual_cleanup_pending:
                        self._set_pending_defensive_close(
                            entry_id,
                            dataclasses.replace(
                                updated_marker,
                                residual_cleanup_pending=True,
                            ),
                        )
                break

    def _maybe_run_attached_defensive_close_cleanup(
            self, intent_key: str,
    ) -> None:
        """Run residual cleanup when a parked defensive close resolves attached.

        Counterpart of :meth:`_maybe_attach_defensive_close_ref` for the
        ``record_resolution(..., 'attached')`` path: plugins whose
        position-attached closes never surface through ``get_open_orders``
        signal a successful land via the resolution row instead. The
        close is confirmed live on the exchange, so the parked-path
        residual cleanup (partial entry remainder, separate TP/SL
        orders, ...) that :meth:`_handle_bracket_attach_after_fill_reject`
        deferred can run now — without waiting for the FILL to route
        through :meth:`_route_defensive_close_fill`.

        ``close_order_ref`` is intentionally left ``None`` on this path:
        the attached plugin contract does not expose a real exchange
        order id, and the eventual FILL routes via
        :meth:`_match_pending_defensive_close`'s ``client_order_id``
        fallback. Transient cleanup failures stamp
        ``residual_cleanup_pending=True`` so
        :meth:`_retry_residual_cleanup_after_transient_fill` replays the
        cancel on the next reconcile.
        """
        if not self._pending_defensive_close:
            return
        for entry_id, marker in self._pending_defensive_close.items():
            if marker.close_intent_key != intent_key:
                continue
            # Mirror the synchronous-success ``natural_close_at`` stamp:
            # the attached-resolution branch likewise proves the close
            # landed on the exchange, and brokers whose position
            # disappears before the close activity is ingested would
            # otherwise let the plugin-side missing-position reconciler
            # halt the run on an unstamped parent row.
            self._stamp_natural_close_at(
                marker.reject_context.position_coid,
            )
            try:
                self._cancel_bracket_reject_residuals(
                    marker.reject_context,
                    raise_on_transient=True,
                )
            except (ExchangeConnectionError,
                    OrderDispositionUnknownError) as exc:
                _blog_warning(
                    "residual cancel after attached defensive-close "
                    "resolution transiently failed for entry %s (%s: %s) "
                    "— marker flagged for retry via reconcile",
                    marker.entry_id,
                    type(exc).__name__, exc,
                )
                if not marker.residual_cleanup_pending:
                    self._set_pending_defensive_close(
                        entry_id,
                        dataclasses.replace(
                            marker,
                            residual_cleanup_pending=True,
                        ),
                    )
            break

    def _escalate_rejected_defensive_close_resolution(
            self, intent_key: str,
    ) -> None:
        """Halt the run when the plugin rejects a parked defensive close.

        Mirrors the ``OrderSkippedByPlugin`` branch of
        :meth:`_handle_bracket_attach_after_fill_reject`: if the broker
        rejected the synthetic close, the parent position is open and
        unprotected with no order in flight, so the only safe outcome is
        manual intervention. Leaving the marker armed would suppress
        retries until the stale-pending grace window fires — by then the
        unprotected position has been exposed for the full timeout.

        The marker is dropped (automated recovery has failed) and the
        halt is recorded via :meth:`_record_halt` so the engine surfaces
        a :class:`ManualInterventionRequiredEvent` and refuses to act on
        further syncs.
        """
        if not self._pending_defensive_close:
            return
        for entry_id, marker in list(self._pending_defensive_close.items()):
            if marker.close_intent_key != intent_key:
                continue
            context = marker.reject_context
            self._clear_pending_defensive_close(entry_id)
            halt = BrokerManualInterventionError(
                f"Defensive close after bracket attach reject was "
                f"rejected by the broker for {intent_key}: position is "
                f"open and unprotected, no close order is live — manual "
                f"intervention required",
                intent_key=intent_key,
                context={
                    'entry_id': entry_id,
                    'position_deal_id': context.position_deal_id,
                    'position_coid': context.position_coid,
                    'symbol': context.symbol,
                    'position_side': context.position_side,
                    'qty': context.qty,
                },
            )
            self._record_halt(halt)
            raise halt

    def _halt_if_defensive_close_terminal(
            self, event: OrderEvent, *, terminal: str,
    ) -> bool:
        """Halt the run when ``watch_orders`` reports a terminal failure
        for an in-flight synthetic defensive close.

        ``terminal`` is the lifecycle event tag (``'cancelled'`` or
        ``'rejected'``) — used purely in the halt message and audit log
        for operator clarity.

        Routes :meth:`_match_pending_defensive_close` (pine_id → order
        ref → client_order_id) so the WS, polled-orders, and
        parked-and-filled-fast paths all surface the same halt. When
        the event does not belong to a pending defensive-close marker
        (the common case), returns ``False`` so the caller's generic
        cancel / reject handling continues. When matched, drops the
        marker, records the halt via :meth:`_record_halt`, and raises
        :class:`BrokerManualInterventionError` — the method never
        returns ``True`` (the raise propagates), the return type is
        kept for readability at the call sites.

        Mirrors :meth:`_escalate_rejected_defensive_close_resolution`,
        which handles the plugin-resolution path (persisted
        ``rejected`` row). The parent position is open and unprotected
        with no close order live, so manual intervention is the only
        safe outcome.
        """
        matched_entry_id = self._match_pending_defensive_close(event)
        if matched_entry_id is None:
            return False
        marker = self._pending_defensive_close[matched_entry_id]
        context = marker.reject_context
        order = event.order
        order_id = order.id if order is not None else None
        # Cleanup ordering: drop the synthetic close intent's in-memory
        # mapping + envelope state (and any parked-dispatch verification
        # row) BEFORE clearing the marker — and BEFORE raising the halt.
        # Mirrors the FILL-settlement path
        # (see :meth:`_route_defensive_close_fill`); the marker itself
        # is dropped last and its extras cleanup runs inside
        # :meth:`_clear_pending_defensive_close`. Without this cleanup,
        # a terminal cancel/reject would leave the close ``intent_key``
        # in ``_order_mapping`` (next ``_find_key_for_order_id`` for any
        # late status update on that order would misroute), the parked
        # ``client_order_id`` in :attr:`_persisted_pending_anchors` (so
        # :meth:`_verify_pending_dispatches` would keep polling
        # ``get_open_orders`` for a close the operator already
        # acknowledged as terminal), and the persisted envelope row
        # uncompleted — :meth:`_replay_pending_defensive_closes` would
        # have no marker to reattach the orphaned envelope to and the
        # JSONL would never self-compact for this key.
        close_coid = marker.close_client_order_id
        if close_coid is not None:
            self._pending_verification.pop(close_coid, None)
            self._persisted_pending_anchors.pop(close_coid, None)
            if self._store_ctx is not None:
                self._store_ctx.record_unpark(close_coid)
        close_intent_key = marker.close_intent_key
        if close_intent_key:
            self._order_mapping.pop(close_intent_key, None)
            self._drop_envelope(close_intent_key)
        self._clear_pending_defensive_close(matched_entry_id)
        halt = BrokerManualInterventionError(
            f"Defensive close after bracket attach reject was "
            f"{terminal} by the broker for "
            f"{marker.close_intent_key} (order_id={order_id}): "
            f"position is open and unprotected, no close order is "
            f"live — manual intervention required",
            intent_key=marker.close_intent_key,
            context={
                'entry_id': matched_entry_id,
                'terminal_event': terminal,
                'order_id': order_id,
                'position_deal_id': context.position_deal_id,
                'position_coid': context.position_coid,
                'symbol': context.symbol,
                'position_side': context.position_side,
                'qty': context.qty,
            },
        )
        self._record_halt(halt)
        raise halt

    def _consume_plugin_resolutions(self) -> None:
        """Apply plugin-driven resolutions written via ``record_resolution``.

        Brokers whose protective brackets are position-attributes (e.g.
        Capital.com native TP/SL) cannot expose them through
        ``get_open_orders``; the plugin's snapshot recovery determines the
        outcome and records it on the persisted park row. This method
        consumes those rows on every sync — see
        :meth:`_verify_pending_dispatches` for the contract.

        ``attached``: the dispatch landed; clear the in-memory park and
        leave ``_active_intents`` alone so the engine keeps treating the
        intent as live. The persisted ``pending_verifications`` row is
        intentionally NOT deleted here: a per-leg resolver (or a slow
        plugin re-evaluating the bracket on a later poll) may still
        flip the row to ``'rejected'`` via the sticky-rejected SQL in
        :meth:`RunContext.record_resolution`. Deleting eagerly would
        force that late UPDATE to find zero rows and the engine would
        never learn the leg is missing. The row is reaped instead by
        :meth:`_drop_envelope` (cancel / fill cleanup / rejected flip)
        or by :meth:`_cleanup_unused_adoption_markers` when no Pine
        intent claimed the marker.

        ``rejected``: the dispatch did not land; clear the park *and*
        drop the matching ``_active_intents`` / ``_order_mapping`` /
        envelope entries (in-memory + persisted) so
        :meth:`_diff_and_dispatch` re-dispatches the same Pine intent on
        the next sync with a fresh envelope (the bracket goes out again,
        restoring protection).
        """
        assert self._store_ctx is not None
        # Group resolutions by intent_key so a single key resolves
        # deterministically even when the snapshot contains multiple
        # rows under it. Multiple rows can occur when a bracket
        # resolver writes per-leg (Round 14 plugin-side aggregation
        # is the typical path, but other plugins or edge replay
        # sequences may still produce per-leg writes), or when a
        # restart finds older 'attached' rows alongside a newer
        # 'rejected' row that arrived during the same retry storm.
        # Within a single key, **'rejected' dominates**: any rejected
        # write means at least one expected exchange-side artefact is
        # confirmed missing, so the intent must be re-dispatched —
        # processing a stale 'attached' row last would otherwise
        # restore the adoption marker the rejected branch had cleared
        # and the engine would silently adopt an unverified dispatch
        # on the next ``_diff_and_dispatch``. The same precedence
        # rule already lives at the storage layer
        # (:meth:`RunContext.record_resolution`'s sticky-rejected SQL);
        # this loop mirrors it for the snapshot we just fetched.
        records_by_key: dict[str, list] = {}
        for record in self._store_ctx.iter_pending_resolutions():
            if record.resolution not in ('attached', 'rejected'):
                _log.error(
                    "ignoring unknown plugin resolution %r for coid=%s "
                    "intent=%s",
                    record.resolution, record.coid, record.key,
                )
                continue
            records_by_key.setdefault(record.key, []).append(record)

        for key, records in records_by_key.items():
            has_rejected = any(r.resolution == 'rejected' for r in records)
            if has_rejected:
                # ``dispatch_kind == 'modify'`` on ANY participating row
                # is enough to flag the whole group as a modify-rejected
                # event: only one parked record per (run_id, coid) ever
                # exists, and re-park overwrites the kind, so an
                # ``'attached'`` survivor row in the same group either
                # came from a prior new-dispatch attached consume that
                # was later flipped to rejected (still 'new') or from a
                # genuine modify amend bookkeeping. The conservative
                # treatment for the mixed case is the modify path:
                # preserving an already-live original order can never
                # produce a duplicate, while clearing it on a real
                # modify-rejected definitely can.
                kind_is_modify = any(
                    r.dispatch_kind == 'modify' for r in records
                )
                for record in records:
                    self._pending_verification.pop(record.coid, None)
                    self._persisted_pending_anchors.pop(record.coid, None)
                    # Drop any prior attached-consume dedup so a future
                    # re-park (different coid, same key) is processed
                    # correctly. Belt-and-braces — the new park gets a
                    # fresh coid which would not collide regardless.
                    self._consumed_attached_coids.discard(record.coid)
                    _log.warning(
                        "plugin-resolved pending dispatch %s as %s "
                        "for intent %s (kind=%s); rejected wins for key "
                        "— %s",
                        record.coid, record.resolution, key,
                        record.dispatch_kind,
                        ("scheduling re-dispatch"
                         if not kind_is_modify
                         else "preserving original order, "
                              "scheduling modify retry"),
                    )
                if not kind_is_modify:
                    # New-dispatch rejected: no original order exists
                    # on the exchange, so clear everything and let the
                    # next ``_diff_and_dispatch`` re-issue the Pine
                    # intent via ``execute_*``.
                    self._active_intents.pop(key, None)
                    self._order_mapping.pop(key, None)
                    # Drop the recovered-close-anchor tag in lockstep
                    # with the mapping: a parked close that the
                    # broker then rejected is terminally dead, so the
                    # adoption exception must NOT survive into a
                    # future EntryIntent reusing the same ``pine_id``
                    # (see :meth:`_drop_envelope` for the same
                    # invariant on the regular cleanup path; this
                    # rejected-new-dispatch branch deliberately does
                    # not call :meth:`_drop_envelope`).
                    self._recovered_close_anchor_keys.discard(key)
                    # Defensive: a 'new' rejected record should never
                    # have left a modify rollback snapshot on the same
                    # key (snapshots are only stashed for kind='modify'
                    # parks), but drop any stale entry to avoid
                    # resurrecting it on a later modify rollback.
                    self._modify_old_intents.pop(key, None)
                else:
                    # Modify-rejected: the ORIGINAL exchange order is
                    # still live (the parked dispatch was an amend that
                    # the broker did NOT apply). Clearing
                    # ``_active_intents`` / ``_order_mapping`` here
                    # would make ``_diff_and_dispatch`` treat the Pine
                    # intent as brand new and call ``execute_*``,
                    # placing a SECOND order alongside the still-live
                    # original. Restore ``_active_intents[key]`` from
                    # the pre-modify snapshot captured at park time
                    # (:meth:`_park_pending`); without this restoration
                    # the slot still holds the NEW intent that
                    # :meth:`_diff_and_dispatch` promoted right after
                    # the parked ``_dispatch_modify`` returned, and the
                    # next diff observes Pine == active so the amend
                    # is silently dropped — leaving the original
                    # exchange order indefinitely on the OLD parameters.
                    # ``_order_mapping[key]`` is intentionally kept (it
                    # still references the live original order id).
                    restored = self._modify_old_intents.pop(key, None)
                    if restored is not None:
                        self._active_intents[key] = restored
                    else:
                        # Post-restart: the in-memory snapshot did not
                        # survive the process bounce. Without recovery,
                        # both ``_active_intents`` and ``_order_mapping``
                        # stay empty for this key, and
                        # ``_diff_and_dispatch`` issues a fresh
                        # ``execute_*`` alongside the still-live original
                        # — duplicating the order. Recover the exchange
                        # order IDs from the persisted park row (v4
                        # ``order_ids`` column) and seed
                        # ``_order_mapping`` so the cross-restart
                        # adoption path picks the intent up without
                        # re-dispatching. The adoption sets
                        # ``_active_intents[key]`` to the current Pine
                        # intent, which is the NEW (unapplied) intent —
                        # the exchange keeps the OLD parameters. A
                        # subsequent Pine parameter change triggers a
                        # normal modify retry; if Pine stays unchanged
                        # the desync persists as a documented limitation.
                        #
                        # The ``modify`` row is the authoritative source —
                        # ``_park_pending`` snapshots the live
                        # ``_order_mapping[key]`` only on the modify park.
                        # An older ``new``/``attached`` sibling row from
                        # the initial dispatch carries ``order_ids=[]``
                        # (parked before the mapping existed); SQL
                        # returns the group unordered, so picking
                        # ``records[0]`` could land on that empty
                        # snapshot and skip the recovery, duplicating
                        # the still-live original on the next dispatch.
                        recovered_ids: list[str] = []
                        for rec in records:
                            if (rec.dispatch_kind == 'modify'
                                    and rec.order_ids):
                                recovered_ids = list(rec.order_ids)
                                break
                        if not recovered_ids:
                            for rec in records:
                                if rec.order_ids:
                                    recovered_ids = list(rec.order_ids)
                                    break
                        if recovered_ids:
                            self._order_mapping[key] = recovered_ids
                            # Track for end-of-sync cleanup. Without this,
                            # if the first post-restart Pine pass no longer
                            # contains this intent (strategy cancelled it,
                            # position closed while the bot was down),
                            # :meth:`_diff_and_dispatch` has no
                            # ``_active_intents`` entry to remove and the
                            # recovered mapping silently adopts the next
                            # same-key dispatch — skipping the broker call
                            # entirely. Mirror the attached-path behaviour:
                            # the cleanup pass drops markers that no Pine
                            # intent claimed.
                            self._attached_adoption_keys.add(key)
                # ``_drop_envelope`` clears both ``_envelopes`` and
                # ``_persisted_envelope_anchors`` (the in-memory replayed
                # anchor) and calls ``record_complete(key)`` which DELETEs
                # every pending_verifications row sharing this intent_key
                # — no separate ``record_unpark`` needed. The explicit pop
                # ordering matters historically (the anchor used to be a
                # silent stale trap); the pop here is now idempotent
                # belt-and-suspenders.
                self._persisted_envelope_anchors.pop(key, None)
                self._drop_envelope(key)
                # Drop any in-flight attached-adoption marker placed by
                # an earlier sync (or this loop iteration before the
                # grouping rewrite). Without this the next call to
                # :meth:`_cleanup_unused_adoption_markers` would not
                # touch it (the `_order_mapping` entry has just been
                # popped above for kind='new', or is still set for
                # kind='modify'), but a future same-sync attached
                # consume would re-add it from the now-already-cleaned
                # state. Belt-and-braces consistency.
                self._attached_adoption_keys.discard(key)
                # Defensive-close synthetic dispatch rejected by plugin
                # resolution: the close order is confirmed missing from
                # the exchange, so the parent position is open and
                # unprotected with no order in flight. Leaving the
                # marker armed would only suppress retries until the
                # stale-pending grace window halts the run — escalate
                # immediately to manual intervention, mirroring the
                # ``OrderSkippedByPlugin`` branch of
                # :meth:`_handle_bracket_attach_after_fill_reject`.
                self._escalate_rejected_defensive_close_resolution(key)
                continue

            # All resolutions for this key are 'attached' — install
            # the adoption marker once (vs. once per coid) and process
            # each row's per-coid bookkeeping.
            #
            # If any record under this key was a parked modify, drop the
            # rollback snapshot stashed by :meth:`_park_pending`. The
            # amend actually went through (despite the timeout) — the
            # promoted-new ``_active_intents[key]`` is correct, and
            # leaving the snapshot in place would make a *future*
            # genuine modify rollback to the wrong (now-superseded)
            # state if its first attempt also times out and is then
            # rejected.
            if any(r.dispatch_kind == 'modify' for r in records):
                self._modify_old_intents.pop(key, None)
            newly_consumed_any = False
            for record in records:
                coid = record.coid
                if coid in self._consumed_attached_coids:
                    # Already processed this attached resolution earlier
                    # in the same run. The row stays alive (so a late
                    # ``'rejected'`` flip can still be observed); the
                    # in-memory state is unchanged on subsequent syncs
                    # until either cleanup, retire, or a flip arrives.
                    continue
                self._consumed_attached_coids.add(coid)
                newly_consumed_any = True
                self._pending_verification.pop(coid, None)
                self._persisted_pending_anchors.pop(coid, None)
                _log.info(
                    "plugin-resolved pending dispatch %s as attached "
                    "for intent %s; keeping active intent", coid, key,
                )
            # After a restart ``_active_intents`` is empty (intents are
            # rebuilt from the Pine order book on the first post-restart
            # sync), so the upcoming :meth:`_diff_and_dispatch` would
            # dispatch this key again unless it sees an existing
            # :attr:`_order_mapping` slot. Mirror the same adoption
            # signal :meth:`_verify_pending_dispatches` uses for
            # recovered ``get_open_orders`` matches: a present (possibly
            # empty) mapping is the "already live, just adopt it" marker.
            # Capital.com's bracket legs do not surface real exchange
            # order ids on this path (their ``id`` is synthesised from
            # the parent ``dealId`` and is never returned by
            # ``get_open_orders``), so the empty-list shape matches how
            # those plugins already populate the slot.
            self._order_mapping.setdefault(key, [])
            # Synthetic defensive-close keys (``__pyne_defensive_close__<coid>``)
            # are engine-generated and never appear in Pine's order book, so
            # ``_active_intents[key]`` is intentionally absent. Adding such a
            # key to ``_attached_adoption_keys`` would make
            # :meth:`_cleanup_unused_adoption_markers` unconditionally drop it
            # at the end of the same sync, and the ``_drop_envelope`` call
            # there DELETEs the ``pending_verifications`` row. A subsequent
            # ``record_resolution(coid, 'rejected')`` (e.g. a per-leg
            # resolver flipping to rejected on a later poll) would then UPDATE
            # zero rows, the engine would never observe the rejection, and
            # :meth:`_escalate_rejected_defensive_close_resolution` would
            # never fire — the unprotected position would wait for the
            # stale-pending timeout instead of halting immediately. Skip the
            # marker for these synthetic keys: the cleanup paths that retire
            # a defensive close (:meth:`_route_defensive_close_fill` on FILL,
            # the rejected branch on a later flip, or the grace-window halt)
            # already call ``_drop_envelope`` at the correct settlement time,
            # and the synthetic ``<coid>`` is unique per position so no
            # future Pine intent can collide with the empty ``_order_mapping``
            # slot.
            if not key.startswith("__pyne_defensive_close__"):
                # Track the marker so end-of-sync cleanup can drop it if no
                # Pine intent claimed it (e.g. the position has since closed
                # and the strategy moved on). Without this cleanup the empty
                # list would silently adopt a *future* same-key dispatch
                # and skip the broker call entirely.
                self._attached_adoption_keys.add(key)
            # If this attached resolution belongs to a parked defensive
            # close, the close did land at the broker but the standard
            # ``get_open_orders`` path inside
            # :meth:`_verify_pending_dispatches` never observed it (the
            # plugin uses the documented ``record_resolution(..., 'attached')``
            # contract because position-attached closes never appear in
            # ``get_open_orders``). Trigger the same residual cleanup
            # :meth:`_maybe_attach_defensive_close_ref` performs on the
            # ``get_open_orders`` recovery branch so partial-fill
            # remainders / separate TP/SL residuals do not linger until
            # the FILL routes (the FILL-time residual pass only fires
            # once :meth:`_route_defensive_close_fill` runs).
            #
            # Only run cleanup when at least one record was NEWLY consumed
            # in this sync — the persisted ``'attached'`` row is kept alive
            # (see ``_consumed_attached_coids`` rationale above) so a later
            # ``'rejected'`` flip can still be observed, which means this
            # block would otherwise run on EVERY sync. Re-issuing the
            # residual cancel calls each sync after the first successful
            # cleanup wastes broker API calls and risks rate-limit hits
            # even though the plugin's ``cancel_broker_order_ref``
            # idempotency contract keeps the operation semantically safe.
            # Transient cleanup failures stamp ``residual_cleanup_pending``
            # and are retried by
            # :meth:`_retry_residual_cleanup_after_transient_fill`, not by
            # this consume path.
            if newly_consumed_any:
                self._maybe_run_attached_defensive_close_cleanup(key)

    def _cleanup_unused_adoption_markers(self) -> None:
        """Drop adoption markers that no Pine intent claimed.

        :meth:`_consume_plugin_resolutions` seeds a slot in
        :attr:`_order_mapping` for two distinct cases:

        - ``'attached'`` resolution: an empty list, signalling "this
          intent is already live, just adopt it on the next diff".
        - Modify-rejected post-restart recovery: the original exchange
          order ids carried over from the persisted park row, so the
          next diff adopts the live original instead of dispatching a
          duplicate ``execute_*`` alongside it.

        Both shapes assume Pine still wants this intent. If the upcoming
        :meth:`_diff_and_dispatch` does not register an
        ``_active_intents[key]`` entry (strategy cancelled the intent,
        position closed while the bot was down), the slot becomes a
        stale trap: a *future* sync producing a fresh intent at the same
        key would hit the adoption branch and skip the broker dispatch
        entirely — a silent loss of order. The same logic applies to the
        envelope anchor: keeping it would pin the new intent to the old
        ``client_order_id``, which the broker may dedupe against the
        previously-attached order. The persisted ``envelopes`` and
        ``pending_verifications`` rows are deleted alongside the
        in-memory state; otherwise a *future restart* would replay the
        anchor and the same staleness would resurface.

        Cleanup runs once per sync, after :meth:`_diff_and_dispatch`. Only
        markers placed *this* sync are tracked (the set is cleared here),
        so legitimate adopters from earlier syncs are not affected.
        """
        for key in list(self._attached_adoption_keys):
            # Set membership already guarantees this key was seeded by
            # :meth:`_consume_plugin_resolutions` (either the attached
            # adoption path with an empty mapping, or the modify-rejected
            # post-restart recovery path with non-empty recovered ids).
            # Both shapes are stale traps if no Pine intent claimed them
            # this sync — drop them uniformly without inspecting the
            # mapping value.
            if (key not in self._active_intents
                    and key in self._order_mapping):
                self._order_mapping.pop(key, None)
                # ``_drop_envelope`` removes the in-memory ``_envelopes``
                # entry (defensive — adoption never populated it for
                # this key, but cancel/retire paths use the same call),
                # also clears the replayed ``_persisted_envelope_anchors``
                # slot, and persists the cleanup via
                # :meth:`RunContext.record_complete`, which DELETEs the
                # ``envelopes`` row AND every ``pending_verifications``
                # row sharing this intent_key. Without that DELETE a
                # future restart would replay the stale anchor through
                # :meth:`_build_envelope` and reuse the old
                # ``client_order_id`` for a genuinely fresh order.
                self._drop_envelope(key)
                _log.info(
                    "cleared stale attached-adoption marker for intent %s "
                    "(no Pine intent claimed it this sync)", key,
                )
        self._attached_adoption_keys.clear()

    def reconcile(self) -> None:
        """Read-side position reconciliation with the exchange.

        The exchange is authoritative for position state. ``get_position`` is
        compared against ``self._position``: at startup the engine adopts the
        exchange size unconditionally; on periodic passes it only acts on
        shrink-to-zero (external flatten). No orders are ever **sent** from a
        reconciliation pass — that would risk duplicate entries.

        **What this method does NOT do:** it does not diff
        ``_order_mapping`` against ``get_open_orders``. Detecting bot-owned
        orders that disappear from the exchange (manual close from the broker
        UI, broker-side liquidation, exchange-side cancel) is a
        :class:`~pynecore.core.plugin.broker.BrokerPlugin` responsibility,
        because the relevant resource namespace is broker-specific (working
        orders vs open positions vs position-attached brackets vs child
        orders), and ``get_open_orders`` only sees one of those namespaces on
        most brokers. Plugins detect disappearance via their own internal
        snapshot loop and signal the engine through ``watch_orders`` —
        either by emitting a synthesised ``cancelled`` :class:`OrderEvent`
        (which the engine's ``_route_event`` cleans out of
        ``_order_mapping``) or by raising
        :class:`~pynecore.core.broker.exceptions.UnexpectedCancelError` for a
        graceful halt. See the Capital.com plugin's
        ``_reconcile_snapshot`` + ``_emit_unexpected_cancellations`` for the
        reference implementation.
        """
        # Drain queued broker events before any pending-close bookkeeping.
        # Startup ``reconcile`` (called from ``start_broker``) runs after
        # ``run_event_stream`` has been scheduled — by the time we get here
        # the WS task may have already pushed replayed FILLs into
        # ``_event_queue``. A re-armed defensive-close marker carries the
        # original ``pending_since`` from the prior process, so when the
        # restart gap exceeds the grace window the marker is stale on its
        # face even though its settling FILL is sitting in the queue.
        # Routing those events first lets :meth:`_route_defensive_close_fill`
        # clear the matching marker so the stale-grace halt does not fire
        # on a pending close that is, in fact, already done.
        #
        # Capture ``_position.size`` BEFORE the drain so the startup
        # adoption block below can detect when the queued events mutated
        # the in-memory size. If ``watch_orders`` delivered a FILL ahead
        # of the ``/positions`` snapshot catching up, ``record_fill``
        # bumps ``_position.size`` here but ``exch_pos`` below may still
        # report the pre-fill size (often zero). Unconditional startup
        # adoption would then clobber the just-applied fill and let the
        # strategy think it is flat — re-entering on the next bar.
        # Rehydrate the cancel-tentative shadow map BEFORE the event drain
        # so that ``_route_event`` can resolve replayed FILL / cancelled
        # events against tentative parents seeded from the prior process.
        # Without this, the startup ``reconcile`` would drain those events
        # against an empty ``_cancel_disposition_pending`` (the rehydrate
        # otherwise only runs at the top of the first :meth:`sync`),
        # missing the event-driven resolution and leaving the legs stuck
        # until a later retry might resolve them incorrectly. Idempotent:
        # the first-sync rehydrate gate clears the flag here so it does
        # not run a second time.
        if self._pending_cancel_tentative_rehydrate:
            self._pending_cancel_tentative_rehydrate = False
            self._rehydrate_cancel_tentative_from_replayed_legs()
        pre_drain_size = self._position.size
        self._drain_events()
        drained_mutated_position = self._position.size != pre_drain_size
        # Cancel-tentative cancel-retry-loop: drives unresolved cancel
        # dispositions to a terminal outcome (or to ``DEGRADED_HALT`` on
        # stale-grace expiry). The drain above already gave every
        # event-driven resolution path a chance to run; this pass is the
        # idempotent fallback for parents whose broker never pushed an
        # explicit cancelled / filled event after the original
        # ``execute_cancel`` timed out. Runs even when the shadow map
        # is empty (cheap no-op) so future-restart rehydrate paths flow
        # through the same code as the steady-state case.
        #
        # Skip the retry on the startup ``reconcile`` (called from
        # ``start_broker`` BEFORE the first :meth:`sync` assigns a
        # real bar timestamp). At that point ``_current_bar_ts_ms`` is
        # still ``0``, and the retry path's
        # :meth:`_build_cancel_envelope` fallback (used when no
        # ``CancelIntent`` envelope is retained — the typical
        # restart-replay case) would mint a ``client_order_id`` from
        # ``bar_ts_ms=0``. That COID differs from every later retry
        # (which uses the real bar timestamp), breaking the broker's
        # idempotency dedup and producing a bogus first attempt. The
        # event-driven drain above still resolves cancel-tentative
        # entries that have a queued FILL / ``cancelled``; truly
        # silent parents wait for the next ``sync``-triggered reconcile
        # (periodic) or a steady-state retry, both with a real
        # ``bar_ts_ms`` anchor.
        if self._current_bar_ts_ms > 0:
            self._drive_cancel_tentative(now_ms=self._cancel_tentative_now_ms())
        # Retry residual cleanup for markers whose FILL already routed
        # but the bracket-reject residual cancel failed transiently. The
        # duplicate-fill cache is already seeded (so no second FILL can
        # ever reach :meth:`_route_defensive_close_fill` again to retry
        # from there), the marker is still armed (so settlement audit +
        # marker drop did NOT happen), and the post-restart replay path
        # is unreachable without an actual restart — without this
        # in-process retry the transient blip would force the bot to
        # wait out the stale-pending grace window and halt even though
        # the broker just needed one more attempt.
        self._retry_residual_cleanup_after_transient_fill()
        # Stale pending defensive close: if a close FILL has been
        # missing past the configured grace window, the deferred
        # cleanup invariant is broken — the parent entry intent is
        # still in ``_active_intents`` (so the engine cannot make
        # forward progress on this position) but the broker has gone
        # silent on the FILL we asked for. Halt so an operator can
        # investigate before further reconcile passes act on stale
        # state.
        #
        # The helper returns ``True`` when it deferred final settlement
        # this pass (broker snapshot already reflects the close but the
        # residual cancel still failed transiently — markers stay armed).
        # In that state the broker view is reduced by qty that the
        # marker still represents for the next grace probe; if we then
        # ran the startup adoption block below it would copy that
        # reduced size into ``_position.size`` and the next
        # ``_broker_matches_post_close_expectation`` would subtract the
        # marker qty a second time and false-halt. Short-circuit the
        # rest of reconcile so the in-memory position stays at the
        # pre-close aggregate and the next pass can retry cleanly.
        deferred = self._raise_if_stale_pending_defensive_close()
        if deferred:
            return

        exch_pos = self._run_async(self._broker.get_position(self._symbol))
        self._exchange_position = exch_pos
        if exch_pos is not None:
            self._position.openprofit = float(exch_pos.unrealized_pnl)
        elif self._position.size == 0.0:
            self._position.openprofit = 0.0
        # The exchange is the single source of truth for position state.
        # ``get_position`` returns ``None`` when no row exists for the symbol,
        # which is functionally a flat position — fold both branches into one
        # ``new_size`` comparison.
        new_size = exch_pos.size if exch_pos is not None else 0.0

        # Periodic reconcile (``sync_count > 0``) only acts on **shrink-to-zero**
        # transitions: the exchange went flat while we still think we hold a
        # position (manual web-UI close, broker liquidation, …). Any other
        # mismatch — including ``new_size > internal`` — could be a fill that
        # raced /positions ahead of the activity stream; adopting it would
        # double-count the moment the matching ``record_fill`` finally drains.
        # The startup call (``sync_count == 0``) still adopts unconditionally
        # so a fresh process restart over an existing exchange position does
        # not double-enter on the first bar.
        is_startup = self._sync_count == 0

        # Skip startup adoption when queued fills mutated the position
        # during the pre-drain above OR when the post-restart replay
        # pre-arming has detected an ambiguous broker snapshot that the
        # marker's persisted ``pre_close_position_size`` cannot
        # disambiguate.
        # In the first case the in-memory size is fresher than the
        # ``/positions`` snapshot the broker just returned (the FILL beat
        # the REST refresh); adopting ``new_size`` would clobber the fill.
        # For the replayed defensive close case we adopt the broker view
        # below — see :meth:`_adopt_size_with_replayed_close` — and only
        # skip when the snapshot cannot be matched against either the
        # marker's stored pre-close size or its post-close expectation
        # (which would risk a double-apply through the no-FIFO settle
        # branch).
        replayed_markers_to_consider = [
            m for m in self._pending_defensive_close.values()
            if m.entry_id in self._replayed_defensive_close_entry_ids
        ]
        if is_startup and drained_mutated_position:
            _blog_info(
                "skipping startup size adoption (exchange=%s, internal=%s) "
                "— queued fills mutated position during pre-drain",
                new_size, self._position.size,
            )
        elif (is_startup and replayed_markers_to_consider
              and new_size != 0.0):
            self._adopt_size_with_replayed_close(
                new_size, exch_pos, replayed_markers_to_consider,
            )
        elif is_startup and new_size != self._position.size:
            _blog_warning(
                "position size mismatch (exchange=%s, internal=%s) — "
                "adopting exchange",
                new_size, self._position.size,
            )
            self._position.size = new_size
            self._position.sign = (
                1.0 if new_size > 0.0
                else (-1.0 if new_size < 0.0 else 0.0)
            )
            if new_size == 0.0:
                self._position.avg_price = na_float
                self._position.open_trades.clear()
                self._position.openprofit = 0.0
                self._position.open_commission = 0.0
            else:
                self._position.avg_price = (
                    exch_pos.entry_price if exch_pos is not None else na_float
                )
        elif not is_startup and new_size == 0.0 and self._position.size != 0.0:
            # Skip while bot-initiated work is in flight — a close we
            # dispatched ourselves will flatten /positions seconds before
            # the matching ``OrderEvent`` reaches the queue. Pre-empting the
            # event here would zero the position; the closing fill (which
            # arrives with a non-zero ``signed_delta``) would then enter
            # ``record_fill``'s ``Opening`` branch and be miscounted as a
            # fresh entry in the opposite direction.
            if self._active_intents:
                return
            # External flatten detected — wipe ALL trade state so a re-entry
            # on the next bar starts from a clean slate. Leaving stale
            # ``open_trades`` would corrupt P&L bookkeeping the moment the
            # next ``record_fill`` runs (FIFO close against trades that no
            # longer exist on the broker).
            _blog_warning(
                "exchange shows flat, internal=%s — external close detected, "
                "clearing position state",
                self._position.size,
            )
            self._position.size = 0.0
            self._position.sign = 0.0
            self._position.avg_price = na_float
            self._position.open_trades.clear()
            self._position.openprofit = 0.0
            self._position.open_commission = 0.0

    def _adopt_size_with_replayed_close(
            self,
            new_size: float,
            exch_pos: ExchangePosition | None,
            replayed: list[PendingDefensiveClose],
    ) -> None:
        """Adopt the broker snapshot at startup with replayed close marker(s).

        Skipping adoption entirely (the previous behaviour) leaves
        ``_position.size`` at its fresh-init default of zero while the
        broker reports a reduced-but-not-flat size. The no-FIFO settle
        branch in :meth:`_route_event` would then see ``size == 0.0``
        when the delayed FILL routes and finalise the marker without
        restoring the residual broker position; with no FILL,
        :meth:`_broker_matches_post_close_expectation` would derive its
        post-close expectation from ``engine_signed == 0`` and
        false-halt the run.

        We disambiguate via the marker's persisted
        :attr:`PendingDefensiveClose.pre_close_position_size` — the
        engine view captured at marker creation, i.e. the pre-close
        aggregate from the prior process:

        - **Broker matches pre-close.** The close has not landed at the
          broker yet (or has not flushed to ``/positions``). Adopt the
          broker view as-is — it is the pre-close aggregate and the
          FILL, when it routes, will reduce ``_position.size`` through
          the no-FIFO apply branch.
        - **Broker matches post-close.** The close landed at the broker
          but the prior process crashed before
          :meth:`_route_defensive_close_fill` flipped
          :attr:`PendingDefensiveClose.fill_observed`. Adopt the broker
          view AND seed the duplicate-fill caches so any redelivered
          FILL is dropped before reaching the no-FIFO apply branch
          (which would otherwise double-subtract the close qty from an
          already-reduced engine view).
        - **Ambiguous.** Marker carries no pre-close snapshot (legacy
          persisted payload), multiple sibling replayed markers whose
          combined deltas do not align with the broker snapshot, or the
          arithmetic does not match either expectation within
          tolerance. Keep the conservative skip behaviour and let the
          stale-grace timer surface a manual-intervention halt rather
          than guess.
        """
        # Marker(s) without a persisted pre-close size cannot
        # disambiguate the snapshot — fall back to the conservative
        # skip behaviour. This also covers payloads written by an
        # earlier schema where the field did not exist yet.
        if any(m.pre_close_position_size is None for m in replayed):
            _blog_info(
                "skipping startup size adoption (exchange=%s, internal=%s) "
                "— replayed defensive-close marker(s) carry no pre-close "
                "snapshot; broker view remains ambiguous until FILL routes",
                new_size, self._position.size,
            )
            return
        # Decode the broker snapshot to a signed scalar — same convention
        # as :meth:`_sync_position_size_to_broker` and
        # :meth:`_broker_matches_post_close_expectation`.
        # ``ExchangePosition.size`` is the unsigned magnitude with the
        # direction carried by ``side``; using ``new_size`` raw would
        # compare ``+abs`` against ``pre_close_position_size`` captured
        # from the signed :attr:`BrokerPosition.size`, false-skipping
        # adoption on every short replay.
        if exch_pos is None:
            broker_signed = 0.0
        else:
            broker_abs = float(exch_pos.size)
            broker_side = (exch_pos.side or "").lower()
            if broker_abs == 0.0 or broker_side == "flat":
                broker_signed = 0.0
            elif broker_side == "long":
                broker_signed = broker_abs
            elif broker_side == "short":
                broker_signed = -broker_abs
            else:
                # Unknown side label with non-zero size — refuse to
                # guess; fall through to the stale-grace handler.
                _blog_info(
                    "skipping startup size adoption (exchange=%s, "
                    "internal=%s) — broker snapshot carries an "
                    "unrecognised side label %r, cannot disambiguate "
                    "replayed defensive-close marker(s)",
                    new_size, self._position.size, exch_pos.side,
                )
                return

        def _signed_close_delta(
                marker: PendingDefensiveClose, magnitude: float,
        ) -> float | None:
            """Return the signed delta a close of ``magnitude`` lots
            applies to :attr:`_position.size` for ``marker``."""
            side = marker.reject_context.position_side
            if magnitude < 0.0:
                magnitude = 0.0
            if side == "buy":
                return -magnitude
            if side == "sell":
                return magnitude
            return None

        # ``pre_close`` is the engine's view BEFORE any of these
        # markers' closes filled. With a single marker the value is
        # taken directly; with multiple markers we use the smallest
        # ``pre_close_position_size`` because each marker captured the
        # state at its own creation time and the EARLIEST capture is
        # the pre-close aggregate before any sibling close shrank it.
        # That value is by construction the broker-side aggregate at
        # restart minus the cumulative close deltas — assuming all
        # siblings' closes landed at the broker.
        # ``min`` by absolute value picks the pre-close anchor: for a
        # long position the largest positive value, for a short the
        # most-negative value, both of which precede any shrinking
        # close. ``key=lambda...`` is None-safe because we filtered
        # ``None`` out above.
        candidates = [
            float(m.pre_close_position_size)  # type: ignore[arg-type]
            for m in replayed
        ]
        # All replayed markers share a sign for the position-side
        # (you cannot have a long and a short marker on the same
        # symbol at the same time — the broker netting forbids it).
        # Pick the candidate with the largest absolute value as the
        # pre-close anchor.
        pre_close = max(candidates, key=abs)
        # Two signed-delta sums on the marker batch:
        # - ``full_deltas`` uses the marker's full ``reject_context.qty``
        #   for each close. ``pre_close + sum(full_deltas)`` is the
        #   broker state once every replayed close has fully landed.
        # - ``partial_deltas`` uses only the slice the prior process
        #   already booked via partial fills
        #   (``marker.partial_filled_qty``). ``pre_close +
        #   sum(partial_deltas)`` is the broker state when the partials
        #   landed at the broker but the terminal fills are still
        #   outstanding — the "pre-close at restart" anchor.
        # When every marker carries ``partial_filled_qty == 0`` the
        # latter sum is zero and the partial-restart anchor collapses
        # to ``pre_close``, matching the legacy single-shot flow.
        full_deltas: list[float] = []
        partial_deltas: list[float] = []
        for marker in replayed:
            qty = float(marker.reject_context.qty)
            partial = float(marker.partial_filled_qty)
            full_delta = _signed_close_delta(marker, qty)
            partial_delta = _signed_close_delta(marker, partial)
            if full_delta is None or partial_delta is None:
                _blog_info(
                    "skipping startup size adoption (exchange=%s, "
                    "internal=%s) — replayed marker carries an "
                    "unrecognised side label, cannot disambiguate "
                    "broker snapshot",
                    new_size, self._position.size,
                )
                return
            full_deltas.append(full_delta)
            partial_deltas.append(partial_delta)
        post_close = pre_close + sum(full_deltas)
        # Anchor for "partials landed at broker, terminal fills still
        # outstanding". Equals ``pre_close`` when no partials were
        # booked in the prior process (the legacy pre-close anchor).
        pre_close_at_restart = pre_close + sum(partial_deltas)

        def _matches(expected: float) -> bool:
            scale = max(abs(expected), abs(broker_signed), 1.0)
            return abs(broker_signed - expected) <= scale * 1e-6 + 1e-9

        if _matches(pre_close_at_restart):
            _blog_warning(
                "startup adopting broker pre-close size (exchange=%s, "
                "internal=%s) — replayed defensive-close marker(s) "
                "expect FILL to reduce the in-memory aggregate",
                new_size, self._position.size,
            )
            self._position.size = broker_signed
            self._position.sign = (
                1.0 if broker_signed > 0.0
                else (-1.0 if broker_signed < 0.0 else 0.0)
            )
            if broker_signed == 0.0:
                self._position.avg_price = na_float
                self._position.open_trades.clear()
                self._position.openprofit = 0.0
                self._position.open_commission = 0.0
            else:
                self._position.avg_price = (
                    exch_pos.entry_price if exch_pos is not None else na_float
                )
            return
        if _matches(post_close):
            _blog_warning(
                "startup adopting broker post-close size (exchange=%s, "
                "internal=%s) — replayed defensive-close marker(s) "
                "appear settled at the broker but FILL was not observed; "
                "seeding duplicate-fill caches",
                new_size, self._position.size,
            )
            self._position.size = broker_signed
            self._position.sign = (
                1.0 if broker_signed > 0.0
                else (-1.0 if broker_signed < 0.0 else 0.0)
            )
            if broker_signed == 0.0:
                self._position.avg_price = na_float
                self._position.open_trades.clear()
                self._position.openprofit = 0.0
                self._position.open_commission = 0.0
            else:
                self._position.avg_price = (
                    exch_pos.entry_price if exch_pos is not None else na_float
                )
            # Seed the dedup caches so any redelivered FILL is dropped
            # by :meth:`_is_duplicate_defensive_close_fill` BEFORE the
            # no-FIFO apply branch could over-reduce the just-adopted
            # post-close size. Also flip ``fill_observed`` on each
            # marker (in-memory) so the restart-safe fallback in
            # :meth:`_broker_matches_post_close_expectation` recognises
            # the snapshot as already post-close should the FILL never
            # arrive and the stale-grace timer fire.
            for marker in replayed:
                if marker.close_intent_key:
                    self._settled_defensive_close_pine_ids.add(
                        marker.close_intent_key,
                    )
                if marker.close_order_ref is not None:
                    self._settled_defensive_close_order_refs.add(
                        marker.close_order_ref,
                    )
                if marker.close_client_order_id is not None:
                    self._settled_defensive_close_client_order_ids.add(
                        marker.close_client_order_id,
                    )
                if marker.fill_exchange_order_id is not None:
                    self._settled_defensive_close_order_refs.add(
                        marker.fill_exchange_order_id,
                    )
                # Mirror the synchronous-success / parked-recovery /
                # attached-resolution ``natural_close_at`` stamp: the
                # broker snapshot we just adopted proves this replayed
                # defensive close landed on the exchange. Without the
                # stamp, the eventual retry cleanup in
                # :meth:`_retry_residual_cleanup_after_transient_fill`
                # writes the audit + clears ``defensive_close_pending``
                # WITHOUT setting ``natural_close_at``, so plugins that
                # rely on that breadcrumb classify the disappeared
                # parent row as an unexpected manual close and halt the
                # run on the next missing-position reconcile.
                self._stamp_natural_close_at(
                    marker.reject_context.position_coid,
                )
                if not marker.fill_observed:
                    self._set_pending_defensive_close(
                        marker.entry_id,
                        dataclasses.replace(marker, fill_observed=True),
                    )
            return
        _blog_info(
            "skipping startup size adoption (exchange=%s, internal=%s) "
            "— broker snapshot matches neither replayed pre-close=%s "
            "nor post-close=%s expectations, leaving in-memory view to "
            "the stale-grace handler",
            new_size, self._position.size,
            pre_close_at_restart, post_close,
        )

    # === Event routing ===

    def _drain_events(self) -> None:
        while True:
            try:
                event = self._event_queue.get_nowait()
            except queue.Empty:
                return
            self._route_event(event)

    def _route_event(self, event: OrderEvent) -> None:
        # Generic ``event %s`` arrival logging happens in
        # :meth:`run_event_stream` so the timestamp + ``bar_index`` reflect
        # when the broker observed the transition, not when this drain
        # pulled it off the queue.  Only intent-key-specific lines (which
        # need engine state) are emitted here.
        t = event.event_type
        if t in ('filled', 'partial'):
            # Event-driven cancel-tentative restore: if a FILL (terminal
            # or partial) belongs to a parent whose cancel disposition is
            # currently unresolved, treat it as a broker-pushed
            # :attr:`CancelDispositionOutcome.ALREADY_FILLED` resolution
            # (the cancel attempt that timed out never actually landed).
            # ``_clear_intent_cancel_disposition_pending`` flips the
            # tentative legs back to ``pending_entry`` (then to ``armed``
            # via the standard ``_promote_pending_partial_bracket_legs``
            # below for ``ENTRY`` events), re-registers the parent with
            # the §2.6 manager (``pending_confirmation=True``), and clears
            # the shadow-map entry — without dropping ``_order_mapping``
            # / envelope (the parent is in fact alive). The normal
            # fill-routing path (record_fill, defensive-close inspection,
            # etc.) runs immediately afterwards on the same event.
            # Idempotent against a later ``reconcile()`` retry that also
            # observes ``ALREADY_FILLED``.
            #
            # ``partial`` events MUST resolve cancel-tentative too. A
            # partial fill leaves residual qty live on the broker, but
            # the just-filled qty is now an open position that needs
            # bracket protection (TP/SL) immediately — leaving the legs
            # in ``LEG_STATE_CANCEL_TENTATIVE`` would let
            # ``_promote_pending_partial_bracket_legs`` (called below for
            # ``ENTRY`` events) skip them, and a subsequent broker
            # ``cancelled`` event for the residual would then abort the
            # legs as ``parent_never_arrived`` — leaving the live
            # partial position unprotected. Resolving as
            # ``ALREADY_FILLED`` here arms the legs against the partial
            # fill price; the residual order remains in
            # ``_order_mapping``.
            #
            # For ``partial`` events specifically, the script's original
            # cancel intent is NOT obsolete: the broker still holds the
            # unfilled residual qty live and could fill more, growing
            # the position past what the just-armed legs cover. The
            # script will not reissue a CancelIntent (from its view the
            # cancel already went out), so we honor the script's intent
            # by attempting a one-shot best-effort cancel of the
            # residual here. On success, ``_order_mapping`` /
            # ``_envelopes`` are cleaned by
            # :meth:`_dispatch_cancel_strict`. On timeout
            # (``OrderDispositionUnknownError``), we deliberately do NOT
            # re-enter cancel-tentative — re-marking would flip the
            # just-armed legs back to ``LEG_STATE_CANCEL_TENTATIVE``,
            # defeating the protection we just established. Log a
            # warning so the operator can intervene if the broker
            # ultimately fills the residual against an unprotected
            # position.
            if event.order is not None:
                _ct_key = self._find_key_for_order_id(event.order.id)
                if (_ct_key is not None
                        and _ct_key in self._cancel_disposition_pending):
                    _blog_info(
                        "cancel-tentative %r resolved by broker %s event",
                        _ct_key, t,
                    )
                    self._clear_intent_cancel_disposition_pending(
                        _ct_key,
                        outcome=CancelDispositionOutcome.ALREADY_FILLED,
                        via='order_event',
                        now_ms=self._cancel_tentative_now_ms(),
                        is_partial_fill=(t == 'partial'),
                    )
                    if t == 'partial':
                        self._cancel_residual_after_partial_resolution(
                            _ct_key,
                        )
            # Reset the per-event FIFO closure cache. Populated below only
            # when :meth:`BrokerPosition.record_fill` actually walks FIFO
            # for this event; consumed by
            # :meth:`_route_defensive_close_fill` so its cleanup targets
            # the entries the FIFO walk reduced, not blindly
            # ``marker.entry_id``. Routing branches that bypass
            # ``record_fill`` (no-FIFO settle, duplicate-drop) must leave
            # this list empty so the cleanup falls back to
            # ``marker.entry_id`` rather than reusing the previous event's
            # ids.
            self._last_fifo_closed_entry_ids = []
            # Defensive-close fill deduplication. The marker-based
            # :meth:`_route_defensive_close_fill` only catches the case
            # where the second delivery still sees a live marker — once
            # the first terminal FILL has cleared it, a duplicate WS
            # replay / polled-orders race would otherwise reach
            # ``record_fill`` and be applied to a flat (or pyramided)
            # ``BrokerPosition`` as a brand-new opposite-direction
            # trade. The same hazard exists for a replayed ``partial``
            # event that arrives AFTER the terminal ``filled`` already
            # settled the marker: the no-FIFO ``partial`` branch below
            # only triggers when the marker is still live, so a stale
            # partial would otherwise fall through to ``record_fill``
            # and corrupt position state the same way. Reject duplicate
            # fill-like events (both ``filled`` and ``partial``) before
            # any state mutation by matching against the settled-close
            # identity cache populated by
            # :meth:`_route_defensive_close_fill`.
            if self._is_duplicate_defensive_close_fill(event):
                _blog_warning(
                    "duplicate defensive-close %s ignored (pine=%r, "
                    "order_id=%s) — already settled this session",
                    t,
                    event.pine_id,
                    event.order.id if event.order is not None else None,
                )
                return
            # Late parent-ENTRY fill after no-FIFO defensive-close settle.
            # The bracket-attach reject is raised synchronously inside the
            # entry dispatch call, BEFORE the broker has emitted the parent
            # ``filled`` event (or the engine has drained it). If the
            # synthetic defensive close fills first and the no-FIFO settle
            # branch below already flattened the in-memory position, a
            # delayed parent ENTRY ``filled`` / ``partial`` event would
            # otherwise fall through to :meth:`record_fill` and open a
            # fresh trade against a broker that is already flat (or
            # corrupt a pyramided remainder). Match the event identity
            # against the cache seeded at no-FIFO settle time
            # (pine_id / position_coid / client_order_id) and drop.
            if self._is_neutralised_parent_entry_fill(event):
                _blog_warning(
                    "late parent-ENTRY %s ignored (pine=%r, order_id=%s) "
                    "— defensive close already settled the position via "
                    "the no-FIFO settle branch",
                    t,
                    event.pine_id,
                    event.order.id if event.order is not None else None,
                )
                return
            # Post-restart settling-close guard. When a defensive close
            # FILL arrives matching a re-armed marker AND
            # :attr:`_position` has no FIFO state to walk, the FILL is
            # the closing leg for a position the engine cannot meaningfully
            # update via :meth:`record_fill`. Two reachable substates:
            #
            # - ``size == 0.0`` — broker snapshot was already flat at
            #   reconcile time (no audit event written before the prior
            #   crash, so dedup caches above are empty), OR the parent
            #   fill was applied via external-flatten cleanup
            #   (:meth:`reconcile` cleared the position) before the close
            #   FILL landed.
            # - ``size != 0.0`` and ``not open_trades`` — startup
            #   :meth:`reconcile` adopted the exchange position size but
            #   did NOT reconstruct ``open_trades`` (the trade rows live
            #   in the persisted ``BrokerStore`` and are not replayed
            #   here). Running :meth:`record_fill` would walk an empty
            #   FIFO, reduce ``size`` toward zero, then take the
            #   ``remaining > 0`` flip branch and create a phantom
            #   opposite-direction trade.
            #
            # In either substate, route directly to the defensive-close
            # handler: it seeds the dedup cache, runs the final residual
            # cancel, writes the audit event, and clears the marker —
            # exactly what the FILL needs without touching
            # ``record_fill``. When the adopted position is still open,
            # flatten the in-memory ``_position`` here so the close FILL
            # leaves the engine consistent with the broker snapshot the
            # close just produced.
            no_fifo_matched_id = (
                self._match_pending_defensive_close(event)
                if not self._position.open_trades else None
            )
            if no_fifo_matched_id is not None:
                # Seed the late-parent-ENTRY-fill cache so a delayed parent
                # ``filled`` / ``partial`` arriving after this settle is
                # dropped by :meth:`_is_neutralised_parent_entry_fill`
                # instead of opening a phantom trade on a flat broker.
                # Restrict the cache seed to the terminal ``'filled'``
                # branch: on ``'partial'`` the defensive close has not
                # settled yet and the parent's full close qty has not been
                # applied — neutralising here would drop a legitimate
                # later parent ENTRY ``filled`` event, leaving the engine
                # flat (or under-sized) while the broker still holds the
                # un-closed remainder until the terminal close fill
                # arrives. The terminal ``'filled'`` branch below seeds
                # the cache before ``_route_defensive_close_fill`` drops
                # the marker.
                if t == 'filled':
                    self._mark_parent_entry_neutralised(
                        self._pending_defensive_close[no_fifo_matched_id],
                    )
                    _blog_warning(
                        "defensive-close FILL routed without FIFO trade state "
                        "(pine=%r, order_id=%s, size=%s) — settling re-armed "
                        "marker without applying to BrokerPosition",
                        event.pine_id,
                        event.order.id if event.order is not None else None,
                        self._position.size,
                    )
                    # Adopted-position state: the defensive close reduces the
                    # broker-side aggregate by exactly its own qty. With
                    # pyramiding (Capital.com aggregates multiple entries
                    # into one netted row), the broker can still be open
                    # AFTER the close fills — flattening the in-memory
                    # position here would over-reduce by the remaining
                    # entries' qty and corrupt subsequent order decisions.
                    # Apply the signed FILL qty (clamped to zero on tiny
                    # residuals) so the engine view tracks the broker.
                    # ``fill_qty`` on the terminal 'filled' event is the
                    # last segment when the close completed across multiple
                    # partials (each prior 'partial' already applied its
                    # slice via the branch below); on a single-shot fill it
                    # equals the full close qty. Falling back to the
                    # marker's recorded qty when ``fill_qty`` is missing
                    # keeps a single-shot fill correct on plugins that
                    # populate only the final aggregate.
                    if self._position.size != 0.0:
                        if event.order is None:
                            order_side = None
                        else:
                            order_side = event.order.side
                        fill_qty = event.fill_qty or 0.0
                        # ``True`` when this terminal 'filled' was
                        # preceded by one or more ``partial`` events in
                        # this lifecycle — i.e. the in-memory position
                        # has already been reduced by those partials and
                        # the only quantity left to apply is the
                        # remainder, NOT the full marker qty.
                        partials_already_booked = False
                        if fill_qty <= 0.0:
                            # Plugin reported terminal 'filled' without
                            # per-segment qty — fall back to the marker's
                            # recorded close qty MINUS whatever was
                            # already applied via prior ``partial`` events
                            # in this lifecycle. Subtracting the full
                            # close qty here when partials have already
                            # been booked would double-reduce the
                            # in-memory position and corrupt subsequent
                            # order decisions (flipping side or under-
                            # shooting zero). On single-shot fills
                            # ``partial_filled_qty`` is 0.0 and the
                            # subtraction is a no-op.
                            matched_id = (
                                self._match_pending_defensive_close(event)
                            )
                            if matched_id is not None:
                                marker = self._pending_defensive_close[
                                    matched_id
                                ]
                                partials_already_booked = (
                                    marker.partial_filled_qty > 0.0
                                )
                                remaining = (
                                    float(marker.reject_context.qty)
                                    - float(marker.partial_filled_qty)
                                )
                                if remaining < 0.0:
                                    remaining = 0.0
                                fill_qty = remaining
                                if order_side is None:
                                    # Buy-close shrinks long; sell-close
                                    # shrinks short — map from position
                                    # side to the order side that closes
                                    # it.
                                    pos_side = (
                                        marker.reject_context.position_side
                                    )
                                    if pos_side == "buy":
                                        order_side = "sell"
                                    elif pos_side == "sell":
                                        order_side = "buy"
                        if fill_qty > 0.0 and order_side in ("buy", "sell"):
                            signed_delta = (
                                fill_qty if order_side == "buy"
                                else -fill_qty
                            )
                            self._position.size += signed_delta
                            if abs(self._position.size) < 1e-12:
                                self._position.size = 0.0
                                self._position.sign = 0.0
                                self._position.avg_price = na_float
                                self._position.openprofit = 0.0
                                self._position.open_commission = 0.0
                            else:
                                self._position.sign = (
                                    1.0 if self._position.size > 0.0 else -1.0
                                )
                        elif partials_already_booked:
                            # All slices of the no-FIFO close were
                            # already applied via prior ``partial``
                            # events (remaining qty rounds to zero).
                            # The terminal 'filled' carries no further
                            # quantity to apply — leave ``_position``
                            # as-is. Flattening here would discard the
                            # legitimate residual position of other
                            # (pyramided / netted) entries the close
                            # was NOT meant to touch.
                            pass
                        else:
                            # Truly unknown size/side — fall back to the
                            # historic flatten behaviour rather than leave
                            # a partially-applied state.
                            self._position.size = 0.0
                            self._position.sign = 0.0
                            self._position.avg_price = na_float
                            self._position.openprofit = 0.0
                            self._position.open_commission = 0.0
                    self._route_defensive_close_fill(event)
                    return
                # ``t == 'partial'`` for the same no-FIFO defensive close.
                # Running :meth:`BrokerPosition.record_fill` here would
                # walk the empty FIFO, take the leftover-qty flip branch
                # and create a phantom opposite-direction trade — e.g. a
                # long-1.0 adopted position receiving a sell-0.4 partial
                # fill ends up sized 0.4 long with a phantom trade
                # instead of 0.6 long. Reduce the in-memory size by the
                # signed partial qty (clamped to zero on tiny residuals)
                # WITHOUT touching ``open_trades`` and WITHOUT
                # short-circuiting to :meth:`_route_defensive_close_fill`
                # (the close has not settled yet — the final ``'filled'``
                # event will run the marker-drop / audit / residual-cancel
                # finalisation above).
                fill_qty = event.fill_qty or 0.0
                if fill_qty <= 0.0:
                    _blog_warning(
                        "defensive-close partial fill with zero qty ignored "
                        "(pine=%r, order_id=%s) — position size NOT updated",
                        event.pine_id,
                        event.order.id if event.order is not None else None,
                    )
                    return
                # Only apply the signed close delta when the in-memory
                # position still carries a non-zero size from the prior
                # adoption. When ``_position.size == 0.0`` here, the
                # parent entry fill has NOT yet been reflected in
                # :attr:`_position` (e.g. the entry filled inside dispatch,
                # bracket attach rejected, but the entry FILL event has
                # not routed to :meth:`record_fill` yet, or the broker
                # snapshot was already flat at reconcile time). Adding
                # the close-side delta from flat would flip the engine
                # to a phantom opposite-direction position — a sell
                # partial against a long defensive close would leave
                # :attr:`_position` short, and the eventual terminal
                # 'filled' event would compound from that phantom state.
                # The skipped slice is accumulated on the marker's
                # :attr:`unapplied_partial_qty` so the engine can drain
                # it once the parent ENTRY fill finally arrives (see
                # :meth:`_drain_unapplied_defensive_close_partials`);
                # without that drain the terminal close FILL would
                # subtract only the remaining slice and leave
                # :attr:`_position` carrying the un-applied partial qty
                # while the broker is flat.
                matched_id = self._match_pending_defensive_close(event)
                if self._position.size != 0.0:
                    signed_delta = (fill_qty if event.order.side == "buy"
                                    else -fill_qty)
                    self._position.size += signed_delta
                    if abs(self._position.size) < 1e-12:
                        self._position.size = 0.0
                        self._position.sign = 0.0
                        self._position.avg_price = na_float
                        self._position.openprofit = 0.0
                        self._position.open_commission = 0.0
                    else:
                        self._position.sign = (
                            1.0 if self._position.size > 0.0 else -1.0
                        )
                    # The slice was applied — count it on
                    # ``partial_filled_qty`` so the terminal close FILL's
                    # missing/zero ``fill_qty`` fallback subtracts only
                    # the remainder. This matches the historical
                    # contract: ``partial_filled_qty`` reflects slices
                    # already applied to :attr:`_position`.
                    if matched_id is not None:
                        existing = self._pending_defensive_close[matched_id]
                        self._set_pending_defensive_close(
                            matched_id,
                            dataclasses.replace(
                                existing,
                                partial_filled_qty=(
                                    existing.partial_filled_qty + fill_qty
                                ),
                            ),
                        )
                else:
                    # Defer the slice: ``_position.size == 0.0`` means the
                    # parent ENTRY fill has not yet routed. Track on
                    # :attr:`unapplied_partial_qty` so the parent-fill
                    # drain (above the ``record_fill`` call below) can
                    # apply the signed delta atomically once the parent
                    # arrives. Critically we do NOT bump
                    # ``partial_filled_qty`` here — the terminal close
                    # FILL's fallback derives ``remaining =
                    # reject_context.qty - partial_filled_qty``; counting
                    # an un-applied slice there would let the terminal
                    # subtract only the remainder, leaving
                    # :attr:`_position` long/short by the deferred slice
                    # while the broker is flat.
                    if matched_id is not None:
                        existing = self._pending_defensive_close[matched_id]
                        self._set_pending_defensive_close(
                            matched_id,
                            dataclasses.replace(
                                existing,
                                unapplied_partial_qty=(
                                    existing.unapplied_partial_qty + fill_qty
                                ),
                            ),
                        )
                _blog_warning(
                    "defensive-close partial fill applied without FIFO trade "
                    "state (pine=%r, order_id=%s, fill_qty=%s, size=%s) — "
                    "marker stays armed until final 'filled' event",
                    event.pine_id,
                    event.order.id if event.order is not None else None,
                    fill_qty,
                    self._position.size,
                )
                return
            # FIFO-path defensive-close ``partial``: accumulate the slice on
            # the marker BEFORE :meth:`record_fill` reduces ``_position.size``
            # via the FIFO walk. The no-FIFO partial branch above already
            # updates :attr:`partial_filled_qty`; without the symmetric
            # update here the marker stays at zero while FIFO partials shrink
            # ``_position.size`` in place. If the terminal ``filled`` event is
            # then delayed / missing / zero-qty, the stale-grace check in
            # :meth:`_broker_matches_post_close_expectation` derives the
            # expected post-close size from ``reject_context.qty -
            # partial_filled_qty`` (i.e. the full close qty) and subtracts
            # that from the already-reduced engine view, false-halting in
            # pyramided runs where the broker snapshot actually matches the
            # remaining netted position.
            if t == 'partial':
                matched_id = self._match_pending_defensive_close(event)
                if matched_id is not None:
                    fill_qty_value = event.fill_qty or 0.0
                    fill_price_value = event.fill_price or 0.0
                    # Mirror :meth:`BrokerPosition.record_fill`'s gate
                    # (``fill_qty > 0 AND fill_price > 0``). A defensive-close
                    # ``partial`` arriving with positive ``fill_qty`` but
                    # missing/zero ``fill_price`` is silently dropped by
                    # :meth:`record_fill` — without the same gate here the
                    # marker's :attr:`partial_filled_qty` would still
                    # accumulate the slice. The terminal ``filled``
                    # math (and the stale-grace check in
                    # :meth:`_broker_matches_post_close_expectation`)
                    # would then treat that quantity as already applied
                    # and only subtract the remainder, leaving
                    # ``_position`` / ``open_trades`` too large or
                    # falsely declaring the close settled.
                    if fill_qty_value > 0.0 and fill_price_value > 0.0:
                        existing = self._pending_defensive_close[matched_id]
                        self._set_pending_defensive_close(
                            matched_id,
                            dataclasses.replace(
                                existing,
                                partial_filled_qty=(
                                    existing.partial_filled_qty + fill_qty_value
                                ),
                            ),
                        )
            # Snapshot the FIFO frontier BEFORE :meth:`record_fill` walks
            # ``open_trades``. ``record_fill`` closes the OLDEST trades
            # first (Pine FIFO semantics), which in pyramiding can differ
            # from ``marker.entry_id``: e.g. with LongA (older) and LongB
            # (newer, bracket-attach reject → defensive close for
            # LongB.qty), FIFO reduces LongA. The downstream cleanup in
            # :meth:`_route_defensive_close_fill` would otherwise clear
            # tracking for ``marker.entry_id`` (LongB) while LongA's
            # Trade was the one actually closed — corrupting later exits
            # and P&L. We capture the pre-FILL ``new_closed_trades``
            # length here so the post-FILL slice gives us the exact set
            # of entry ids the FIFO walk consumed.
            closed_count_before = len(self._position.new_closed_trades)
            self._position.record_fill(event)
            self._last_fifo_closed_entry_ids = [
                trade.entry_id
                for trade in self._position.new_closed_trades[closed_count_before:]
                if trade.entry_id is not None
            ]
            # Persist defensive-close ``partial`` FIFO closures onto the
            # marker. The transient ``_last_fifo_closed_entry_ids`` list is
            # reset at the top of every :meth:`_route_event`, so a
            # subsequent event (the terminal ``filled``, an unrelated fill,
            # or no event at all when the terminal arrives with zero qty
            # and stale-grace settles the marker) would otherwise discard
            # the entries this partial closed. In pyramiding (LongA older,
            # LongB newer with bracket-attach reject), a defensive-close
            # partial for LongB.qty can fully consume LongA via FIFO; if
            # the terminal ``filled`` then closes LongB, the snapshot
            # captured in :meth:`_route_defensive_close_fill` carries only
            # LongB and the cleanup in
            # :meth:`_retry_residual_cleanup_after_transient_fill` /
            # :meth:`_replay_pending_defensive_closes` falls back to
            # ``marker.entry_id`` for LongA — leaving LongA's intents
            # orphaned. Merge here so the marker accumulates every entry
            # FIFO touched across the partials → terminal sequence.
            if t == 'partial' and self._last_fifo_closed_entry_ids:
                matched_id = self._match_pending_defensive_close(event)
                if matched_id is not None:
                    existing = self._pending_defensive_close[matched_id]
                    merged = list(existing.fifo_closed_entry_ids)
                    seen = set(merged)
                    for entry_id in self._last_fifo_closed_entry_ids:
                        if entry_id not in seen:
                            merged.append(entry_id)
                            seen.add(entry_id)
                    if len(merged) != len(existing.fifo_closed_entry_ids):
                        self._set_pending_defensive_close(
                            matched_id,
                            dataclasses.replace(
                                existing,
                                fifo_closed_entry_ids=tuple(merged),
                            ),
                        )
            _blog_info(
                "position size=%s sign=%s avg=%s (after %s pine=%r)",
                self._position.size, self._position.sign,
                self._position.avg_price, t, event.pine_id,
            )
            if event.leg_type == LegType.ENTRY and event.pine_id:
                # Drain any defensive-close ``partial`` slices the no-FIFO
                # branch deferred while :attr:`_position.size` was still 0.
                # The deferred slices were stashed on
                # :attr:`PendingDefensiveClose.unapplied_partial_qty`; now
                # that :meth:`record_fill` has populated ``_position.size``
                # / ``open_trades`` from the parent ENTRY fill, apply the
                # accumulated close-side delta so the terminal close FILL
                # sees a consistent state. The engine assumes per-segment
                # plugin reporting (see the comment block in the no-FIFO
                # terminal branch above): without this drain, a per-segment
                # terminal that subtracts only the remaining close slice
                # would leave :attr:`_position` long/short by the deferred
                # qty while the broker is flat.
                self._drain_unapplied_defensive_close_partials(event.pine_id)
                self._resolve_deferred_for_entry(event.pine_id)
                self._amend_bracket_qty_for_entry_fill(event)
                self._promote_pending_partial_bracket_legs(event)
                self._retire_entry_stop_watch_on_fill(event)
            self._cascade_oca_cancel(event)
            # When a closing leg fully closes the position, drop the
            # entry + matching exit intents and clear Pine's order
            # dicts. Pine's ``strategy.exit`` is unconditional in most
            # scripts; the simulator gates it via ``open_trades`` —
            # broker mode needs equivalent gating, otherwise the next
            # bar's ``sync()`` rebuilds the same exit and dispatches a
            # pointless ``modify_exit`` against a position that no
            # longer exists on the broker (which on Capital.com fails
            # because the bracket-attached entry row is naturally
            # closed).
            if (event.leg_type in (LegType.TAKE_PROFIT, LegType.STOP_LOSS,
                                    LegType.TRAILING_STOP, LegType.CLOSE)
                    and self._position.size == 0):
                self._cleanup_closed_position(event)
            # Defensive-close FILL: when a defensive-close marker is in
            # flight, the FILL event arrives carrying the synthetic
            # CloseIntent's pine_id rather than the parent entry id, so
            # the natural :meth:`_cleanup_closed_position` path above
            # cannot reach the parent state. Match the event against
            # the pending markers (by close_intent_key OR by the
            # captured close_order_ref) and run the deferred cleanup
            # against the *parent* entry id.
            #
            # Guard against the broker reporting the terminal FILL with
            # missing/zero ``fill_qty`` or ``fill_price`` while FIFO
            # ``open_trades`` are still live: :meth:`record_fill` skips
            # that event (it returns ``False`` without mutating FIFO
            # state), so settling the marker here would drop tracking
            # for a position the engine never actually reduced — leaving
            # ``_position.size`` and ``open_trades`` looking as if the
            # trade is still open while the broker close has settled.
            # Leave the marker armed so the stale-grace timer
            # (:meth:`_raise_if_stale_pending_defensive_close`) or the
            # next reconcile can resolve it once a fully-formed FILL or
            # broker snapshot arrives.
            if t == 'filled':
                fill_qty_value = event.fill_qty or 0.0
                fill_price_value = event.fill_price or 0.0
                if (fill_qty_value <= 0.0 or fill_price_value <= 0.0) \
                        and self._match_pending_defensive_close(event) is not None:
                    _blog_warning(
                        "defensive-close FILL with zero qty/price ignored "
                        "(pine=%r, order_id=%s, qty=%s, price=%s) — marker "
                        "stays armed for stale-grace resolution",
                        event.pine_id,
                        event.order.id if event.order is not None else None,
                        fill_qty_value, fill_price_value,
                    )
                else:
                    self._route_defensive_close_fill(event)
        elif t == 'cancelled':
            # Terminal failure for an in-flight defensive close: the
            # synthetic flatten was cancelled by the broker (or an
            # operator) without ever filling. The parent position is
            # open AND unprotected with no close order live — leaving
            # the marker armed would defer detection until the
            # stale-grace window
            # (:meth:`_raise_if_stale_pending_defensive_close`) fires.
            # Halt immediately, mirroring
            # :meth:`_escalate_rejected_defensive_close_resolution`.
            if self._halt_if_defensive_close_terminal(
                    event, terminal='cancelled',
            ):
                return
            key = self._find_key_for_order_id(event.order.id)
            if (key is not None
                    and key in self._cancel_disposition_pending):
                # Event-driven cancel-tentative confirm: the broker pushed
                # a CANCELLED for a parent currently in cancel-tentative.
                # Resolve forward (legs → aborted_parent_never_arrived,
                # mapping / envelope / active_intents dropped) without
                # running the eager-retire path below — the confirm helper
                # owns the full cleanup. Idempotent against a follow-up
                # ``reconcile()`` retry that may also see the outcome.
                _blog_info(
                    "cancel-tentative %r resolved by broker cancelled event",
                    key,
                )
                self._clear_intent_cancel_disposition_pending(
                    key,
                    outcome=CancelDispositionOutcome.CANCEL_CONFIRMED,
                    via='order_event',
                    now_ms=self._cancel_tentative_now_ms(),
                )
                return
            if key is not None:
                _blog_error(
                    "unexpected cancel for intent %s (%s)",
                    key, event,
                )
                # §2.6.7: retire the parent's native fail-safe state BEFORE the
                # teardown below evicts the legs / envelope used to resolve its
                # COID — but ONLY when the cancelled order IS the parent entry
                # AND that entry never filled. A cancelled child exit / close
                # carries ``from_entry`` naming a still-open parent; retiring
                # there would RETIRE a live parent's state and silently disable
                # its broker-native fail-safe (``recompute_worst_sl`` no-ops on
                # RETIRED). Equally, a CANCELLED for the unfilled residual of a
                # PARTIALLY filled entry leaves ``_active_intents[key]`` an
                # ``EntryIntent`` while the partial position is live and its
                # legs are armed — retiring there strands that position without
                # the broker-native SL. Gate on the entry intent having zero
                # filled qty (``event.order.filled_qty <= 0``), mirroring
                # ``_abort_pending_partial_legs_for_dead_entry`` (also a no-op
                # for already-armed partial-fill legs). ``key`` is the matched
                # entry intent's Pine id (``intent_key`` == ``pine_id`` for an
                # ``EntryIntent``), which is exactly what
                # ``_retire_native_failsafe_for_entry`` resolves on — use it
                # directly. A broker-synthesized status event may carry only the
                # exchange order id (``pine_id`` / ``from_entry`` both ``None``),
                # so deriving the id from the event would pass ``''`` and leave a
                # DEGRADING / DEGRADED state parked under the COID. Idempotent —
                # no-op when no state is registered (the never-filled cancel case).
                if (isinstance(self._active_intents.get(key), EntryIntent)
                        and event.order.filled_qty <= 0.0):
                    self._retire_native_failsafe_for_entry(key)
                self._abort_pending_partial_legs_for_dead_entry(
                    key, reason='parent_cancelled',
                )
                self._entry_stop_engine.mark_aborted(
                    key, reason='parent_cancelled',
                )
                self._order_mapping.pop(key, None)
                self._active_intents.pop(key, None)
                self._drop_envelope(key)
            else:
                _blog_info(
                    "external cancel observed (%s)", event,
                )
        elif t == 'rejected':
            # Mirror of the 'cancelled' branch above: a watch_orders
            # ``rejected`` for the synthetic defensive close means the
            # close never landed. Halt before the stale-grace window
            # so the operator learns about the unprotected position
            # immediately.
            if self._halt_if_defensive_close_terminal(
                    event, terminal='rejected',
            ):
                return
            key = self._find_key_for_order_id(event.order.id)
            if key is not None:
                _blog_warning(
                    "order rejected for intent %s (%s)",
                    key, event,
                )
                # §2.6.7: mirror the 'cancelled' branch — retire the parent's
                # native fail-safe state before the teardown evicts its COID
                # sources, but ONLY when the rejected order IS the parent entry
                # AND that entry never filled. A rejected child exit / close
                # names a still-open parent via ``from_entry``; retiring there
                # would disable a live parent's fail-safe. A REJECTED for the
                # unfilled residual of a partially filled entry likewise leaves
                # the partial position live with armed legs — retiring there
                # strands it without the broker-native SL. Gate on the entry
                # intent having zero filled qty (``event.order.filled_qty <=
                # 0``). ``key`` is the matched entry intent's Pine id — the
                # exact id ``_retire_native_failsafe_for_entry`` resolves on —
                # and is used directly so a broker-synthesized status event
                # lacking ``pine_id`` / ``from_entry`` does not pass ``''`` and
                # strand a parked state under the COID. Idempotent.
                if (isinstance(self._active_intents.get(key), EntryIntent)
                        and event.order.filled_qty <= 0.0):
                    self._retire_native_failsafe_for_entry(key)
                self._abort_pending_partial_legs_for_dead_entry(
                    key, reason='parent_rejected',
                )
                self._entry_stop_engine.mark_aborted(
                    key, reason='parent_rejected',
                )
                self._order_mapping.pop(key, None)
                self._active_intents.pop(key, None)
                self._drop_envelope(key)
            else:
                _blog_warning(
                    "order rejected (%s)", event,
                )

    def _abort_pending_partial_legs_for_dead_entry(
            self, intent_key: str, *, reason: str,
    ) -> None:
        """Cancel ``pending_entry`` partial-bracket legs when the parent
        entry never fills (cancel/reject terminal event).

        An absolute-price ``strategy.exit`` issued before the parent fill
        persists durable ``LEG_STATE_PENDING_ENTRY`` rows
        (see :meth:`_dispatch_engine_trigger_partial_bracket`). The cancel
        / reject branches that drop the entry intent + mapping must also
        retire those legs, otherwise a later same-``from_entry`` reuse can
        promote them against the wrong parent (different
        ``parent_entry_dispatch_ref``) and dispatch closes / fail-safe PUTs
        on the new position.

        ``intent_key`` is whatever :meth:`_find_key_for_order_id` returned;
        for entry intents that is the parent's ``pine_id`` (== the leg's
        ``from_entry``). Exit / close keys carry a NUL separator and do
        not match any leg's ``from_entry``, so this method becomes a
        no-op on the exit / close branches.

        The aborted legs leave behind their parent :class:`ExitIntent`
        slot in :attr:`_active_intents` / :attr:`_order_mapping`. Without
        retiring those slots here, a later sync that re-issues the same
        entry under the same ``pine_id`` together with the same partial
        ``strategy.exit`` would diff the still-active ExitIntent against
        the freshly emitted one, see them as equal (dataclass equality on
        identical params) and skip
        :meth:`_dispatch_engine_trigger_partial_bracket` — the new
        position would land without partial TP / SL protection. Drop
        every ExitIntent whose ``from_entry`` matches the dying parent
        so the next sync re-dispatches its legs.
        """
        intent = self._active_intents.get(intent_key)
        if not isinstance(intent, EntryIntent):
            return
        self._partial_bracket_engine.abort_pending_legs_for_parent_never_arrived(
            symbol=self._symbol,
            from_entry=intent_key,
            reason=reason,
        )
        # Drop any ExitIntent slots whose ``from_entry`` matches the
        # dying parent. Collect first to avoid mutating ``_active_intents``
        # during iteration.
        exit_keys_to_drop = [
            key for key, active in self._active_intents.items()
            if isinstance(active, ExitIntent)
            and active.from_entry == intent_key
        ]
        for exit_key in exit_keys_to_drop:
            self._active_intents.pop(exit_key, None)
            self._order_mapping.pop(exit_key, None)
            self._drop_envelope(exit_key)

    def _drop_envelope(self, key: str) -> None:
        """Remove envelope state for ``key`` and persist a ``complete`` marker.

        Called from every site that retires an intent (cancel dispatch,
        unexpected cancel event, reject event). The persisted marker lets the
        replay path skip the envelope and any still-pending verifications
        attached to the same ``key`` — keeping the JSONL self-compacting.

        Also clears the in-memory ``_persisted_envelope_anchors`` entry. The
        anchor map is repopulated wholesale at the top of every :meth:`sync`
        from :meth:`storage.RunContext.replay`, so an event-driven drop
        within the same sync (e.g. a queued ``cancelled``/``rejected``
        drained from :meth:`_drain_events`) would otherwise leave a stale
        anchor that :meth:`_build_envelope` then pops into a recycled
        ``client_order_id`` for the retry — brokers with idempotency caches
        dedupe/reject the replacement.

        Mirrors the same wholesale cleanup for ``_persisted_pending_anchors``:
        :meth:`storage.RunContext.record_complete` DELETEs every
        ``pending_verifications`` row tied to this ``key`` from SQLite, but
        :meth:`sync` only repopulates the in-memory map at the START of the
        cycle. An event-driven drop after that replay (cancelled/rejected
        drained from :meth:`_drain_events`) would otherwise leave stale
        same-key entries that :meth:`_verify_pending_dispatches` — invoked
        immediately afterward — could re-adopt via ``get_open_orders``,
        resurrecting an intent that was just cancelled/rejected/closed.
        """
        self._envelopes.pop(key, None)
        self._persisted_envelope_anchors.pop(key, None)
        for coid in [
            coid for coid, record in self._persisted_pending_anchors.items()
            if record.key == key
        ]:
            self._persisted_pending_anchors.pop(coid, None)
        # Drop the recovered-close-anchor tag in lockstep with the
        # envelope. The tag was seeded by
        # :meth:`_verify_pending_dispatches` (or its persisted-anchor
        # sibling) so the diff-loop adoption guard would let a
        # re-emitted :class:`CloseIntent` adopt the parked broker
        # order. Once the envelope/mapping for ``key`` is retired
        # (fill, cancel, reject, or any other terminal cleanup), the
        # tag must NOT outlive it — a future EntryIntent sharing the
        # same ``pine_id`` could repopulate ``_order_mapping[key]``,
        # and a subsequent CloseIntent for that ``pine_id`` would then
        # silently adopt the entry's mapping under the close-anchor
        # exception instead of dispatching a fresh market close.
        # ``set.discard`` is O(1) and idempotent — safe to call for
        # every retired key.
        self._recovered_close_anchor_keys.discard(key)
        if self._store_ctx is not None:
            self._store_ctx.record_complete(key)

    def _find_key_for_order_id(self, order_id: str) -> str | None:
        for key, ids in self._order_mapping.items():
            if order_id in ids:
                return key
        return None

    def _resolve_deferred_for_entry(self, entry_id: str) -> None:
        """An entry fill unblocks every exit that references it via ticks.

        Pyramiding can attach multiple tick-deferred exits (different
        ``pine_id``) to the same ``from_entry``; all of them must be
        resolved on the entry's fill, not just one. Iterating with a
        snapshot of the keys lets the loop mutate ``_deferred_exits``
        safely.
        """
        matches = [
            (key, intent)
            for key, intent in self._deferred_exits.items()
            if intent.from_entry == entry_id
        ]
        for key, deferred in matches:
            # Defensive-close-arming short-circuit. ``_dispatch_new`` on the
            # FIRST resolved sibling can hit a bracket-attach-after-fill
            # reject, run :meth:`_handle_bracket_attach_after_fill_reject`
            # (which arms the defensive close and adds ``entry_id`` to
            # :attr:`_defensively_closed_entries_this_sync`) and re-raise
            # :class:`OrderSkippedByPlugin`. Without this guard, the next
            # snapshot sibling — also keyed on the same ``from_entry`` —
            # would call ``_dispatch_new`` and emit a duplicate market
            # close against a parent we just flattened defensively
            # (or, when the marker is still pending settlement, race the
            # in-flight defensive close). The same guard pattern lives in
            # :meth:`_diff_and_dispatch` for the in-loop new-intents pass.
            if (entry_id in self._defensively_closed_entries_this_sync
                    or entry_id in self._pending_defensive_close):
                # Drop the deferred entry — its parent is no longer eligible
                # for a fresh exit attach. The next bar's diff pass will
                # re-evaluate the strategy state from scratch.
                del self._deferred_exits[key]
                continue
            del self._deferred_exits[key]
            resolved = self._resolve_ticks(deferred)
            if resolved.has_unresolved_ticks:
                self._deferred_exits[key] = deferred
                continue
            try:
                self._dispatch_new(resolved)
            except OrderSkippedByPlugin as e:
                _blog_warning("%s", e)
                continue
            self._active_intents[resolved.intent_key] = resolved

    # === OCA cascade cancel ===

    def _cascade_oca_cancel(self, event: OrderEvent) -> None:
        """Cancel OCA-cancel siblings of a freshly filled intent.

        Pine semantics: an ``oca_type='cancel'`` group keeps exactly one live
        leg at a time. The Pine backtester enforces this at fill time; this
        method is the live-trading equivalent — without it, a fill on leg A
        leaves leg B open until the next bar's diff pass, and a same-bar
        reversal may fill B too.

        The cascade is **suppressed** when:

        - The plugin declared ``oca_cancel = CapabilityLevel.NATIVE`` — the
          exchange registers and cancels the group natively.
        - The filled intent has no OCA group, or its type is not ``cancel``.
          (``reduce`` groups amend quantities on fill; that belongs to the
          partial-fill qty-amend workstream, not here.)
        - The partial-fill policy is :data:`OcaPartialFillPolicy.FULL_FILL_ONLY`
          and the event is ``partial``.
        - The group was already processed in this sync — prevents a
          double-fill (e.g. TP and entry both filling on the same bar) from
          emitting duplicate cancels.
        """
        if event.event_type == 'partial' and (
                self._oca_partial_policy is OcaPartialFillPolicy.FULL_FILL_ONLY
        ):
            return
        filled_key = self._filled_intent_key(event)
        if filled_key is None:
            return
        self._cascade_oca_cancel_for_key(filled_key)

    def _cascade_oca_cancel_for_key(self, filled_key: str) -> None:
        """Run the OCA-cancel cascade for ``filled_key``.

        Event-agnostic core extracted from :meth:`_cascade_oca_cancel` so
        proactive paths (notably
        :meth:`_handle_bracket_attach_after_fill_reject`) can fire the
        cascade BEFORE dropping the parent intent — otherwise the later
        in-band cascade call in :meth:`_route_event` finds
        ``_active_intents`` empty for ``filled_key`` and the OCA-cancel
        siblings stay live exchange-side.

        The cascade is **suppressed** when the plugin declared
        ``oca_cancel = CapabilityLevel.NATIVE`` (exchange handles the
        group) or the filled intent has no ``cancel`` OCA group.
        Idempotent within a sync via
        :attr:`_cancelled_oca_groups_this_sync`, so a follow-up
        in-band call on the same fill is a no-op.
        """
        if self._oca_cancel_native:
            return
        filled_intent = self._active_intents.get(filled_key)
        if filled_intent is None:
            return
        oca_name = getattr(filled_intent, 'oca_name', None)
        oca_type = getattr(filled_intent, 'oca_type', None)
        if not oca_name or oca_type != OcaType.CANCEL.value:
            return
        if oca_name in self._cancelled_oca_groups_this_sync:
            return
        self._cancelled_oca_groups_this_sync.add(oca_name)

        siblings = [
            (key, intent)
            for key, intent in list(self._active_intents.items())
            if key != filled_key
            and getattr(intent, 'oca_name', None) == oca_name
            and getattr(intent, 'oca_type', None) == OcaType.CANCEL.value
        ]
        for key, intent in siblings:
            _log.info(
                "OCA cascade cancel: fill on %s cancels sibling %s in group %r",
                filled_key, key, oca_name,
            )
            self._active_intents.pop(key, None)
            self._remove_pine_order_for_intent(intent)
            self._dispatch_cancel(intent)

    def _remove_pine_order_for_intent(self, intent: Intent) -> None:
        """Delete the Pine-side :class:`Order` backing ``intent``.

        Mirrors :meth:`SimPosition._cancel_oca_group` for the live path: once
        an OCA-cancel sibling is cancelled exchange-side, the Pine-level order
        book must drop it too — otherwise the next :meth:`sync` rebuilds an
        intent from the stale entry and re-dispatches onto the now-cancelled
        exchange state.
        """
        entry_orders = getattr(self._position, 'entry_orders', None)
        exit_orders = getattr(self._position, 'exit_orders', None)
        if isinstance(intent, EntryIntent) and entry_orders is not None:
            entry_orders.pop(intent.pine_id, None)
        elif isinstance(intent, ExitIntent) and exit_orders is not None:
            exit_orders.pop((intent.pine_id, intent.from_entry), None)

    def _drain_unapplied_defensive_close_partials(
            self, entry_pine_id: str,
    ) -> None:
        """Apply any deferred no-FIFO defensive-close partial slices.

        Invoked from :meth:`_route_event` right after the parent ENTRY
        ``record_fill`` populates :attr:`_position.size` / ``open_trades``.

        The no-FIFO ``partial`` branch in :meth:`_route_event` stashes
        slices it could not apply (because ``_position.size == 0.0`` —
        the parent entry fill had not routed yet) on
        :attr:`PendingDefensiveClose.unapplied_partial_qty`. Now that
        the parent fill has landed, we apply the accumulated signed
        delta against :attr:`_position.size` AND the freshly-opened
        FIFO trade so the terminal defensive-close ``filled`` event
        (which goes through the FIFO branch with a per-segment
        ``fill_qty == close_qty - applied_partial_qty``) sees a
        consistent pre-state. Without this drain, ``record_fill`` on
        the terminal subtracts only the remaining slice and leaves
        :attr:`_position` long/short by the previously-skipped partial
        qty while the broker is flat.

        ``partial_filled_qty`` is bumped by the same amount so the
        no-FIFO ``fill_qty <= 0`` fallback (used when the plugin omits
        per-segment qty on the terminal) computes ``remaining ==
        reject_context.qty - partial_filled_qty`` correctly.

        No-op when the engine carries no marker for ``entry_pine_id``
        or when ``unapplied_partial_qty == 0``.
        """
        marker: PendingDefensiveClose | None = (
            self._pending_defensive_close.get(entry_pine_id)
        )
        if marker is None:
            return
        deferred = float(marker.unapplied_partial_qty)
        if deferred <= 0.0:
            return
        # The close side is opposite the parent side: a long parent's
        # defensive close is a sell; ``unapplied_partial_qty`` always
        # represents close-side qty. Map the parent's recorded
        # ``position_side`` (the side that OPENS the parent) to the
        # signed delta needed to REDUCE that position.
        parent_side = marker.reject_context.position_side
        if parent_side == "buy":
            signed_delta = -deferred  # close = sell → shrink long
        elif parent_side == "sell":
            signed_delta = deferred  # close = buy → shrink short
        else:
            _blog_warning(
                "defensive-close partial drain: unknown parent side %r "
                "for entry %r — leaving deferred qty=%s un-applied",
                parent_side, entry_pine_id, deferred,
            )
            return
        # Mirror :meth:`BrokerPosition.record_fill`'s arithmetic on
        # :attr:`_position.size`. We update ``open_trades`` separately
        # below because the parent ENTRY fill has just opened a fresh
        # FIFO trade for this entry and FIFO walks of the terminal
        # close FILL must see that trade already reduced by the
        # deferred slice.
        self._position.size += signed_delta
        if abs(self._position.size) < 1e-12:
            self._position.size = 0.0
            self._position.sign = 0.0
            self._position.avg_price = na_float
            self._position.openprofit = 0.0
            self._position.open_commission = 0.0
        else:
            self._position.sign = (
                1.0 if self._position.size > 0.0 else -1.0
            )
        # Reduce the freshly-opened parent trade by the deferred qty.
        # FIFO walks the OLDEST trade first; the parent fill just
        # appended its trade at the end. With pyramiding (older same-id
        # trades), the FIFO close that would normally absorb the
        # deferred partial would consume the OLDEST trade first — to
        # mirror that, walk ``open_trades`` from the front.
        remaining = deferred
        survivors: list = []
        for trade in self._position.open_trades:
            if remaining <= 0.0:
                survivors.append(trade)
                continue
            trade_abs = abs(trade.size)
            if trade_abs <= remaining + 1e-12:
                # Fully consume this trade.
                remaining -= trade_abs
                continue
            # Partial close — keep the trade with reduced size.
            # :class:`pynecore.lib.strategy.Trade` is a slotted class, not
            # a dataclass, so ``dataclasses.replace`` would raise
            # ``TypeError`` here. Mutate ``size`` in-place — the same
            # pattern :meth:`BrokerPosition.record_fill_close` and the
            # flat-broker FIFO walk below already use.
            new_size = (
                trade.size - remaining
                if trade.size > 0.0
                else trade.size + remaining
            )
            trade.size = new_size
            survivors.append(trade)
            remaining = 0.0
        self._position.open_trades[:] = survivors
        if remaining > 1e-12:
            _blog_warning(
                "defensive-close partial drain for entry %r: deferred "
                "qty=%s exceeded ``open_trades`` total by %s — leftover "
                "discarded (broker likely netted multiple entries)",
                entry_pine_id, deferred, remaining,
            )
        # Move the now-applied qty from ``unapplied_partial_qty`` to
        # ``partial_filled_qty`` so the terminal close FILL's
        # ``fill_qty <= 0`` fallback derives the correct ``remaining``.
        self._set_pending_defensive_close(
            entry_pine_id,
            dataclasses.replace(
                marker,
                partial_filled_qty=(
                    marker.partial_filled_qty + (deferred - remaining)
                ),
                unapplied_partial_qty=0.0,
            ),
        )

    def _match_pending_defensive_close(
            self, event: OrderEvent,
    ) -> str | None:
        """Resolve ``event`` to a pending defensive-close marker entry_id.

        Centralises the three-step identity match used by both the
        in-flight settlement handler :meth:`_route_defensive_close_fill`
        and the post-restart settling-close guard inside
        :meth:`_route_event`. The match priority is:

        - ``event.pine_id`` against ``marker.close_intent_key`` — the WS
          path, where ``OrderEvent.pine_id`` is reliable.
        - ``event.order.id`` against ``marker.close_order_ref`` — the
          polled-orders path once
          :meth:`_maybe_attach_defensive_close_ref` has backfilled the
          ref from ``get_open_orders``.
        - ``event.order.client_order_id`` against
          ``marker.close_client_order_id`` — the parked-and-filled-fast
          fallback (see the matching comment in
          :meth:`_route_defensive_close_fill`).

        Returns ``None`` when no marker matches, including the trivial
        ``_pending_defensive_close`` empty case.
        """
        if not self._pending_defensive_close:
            return None
        pine_id = event.pine_id
        if pine_id is not None:
            for entry_id, marker in self._pending_defensive_close.items():
                if marker.close_intent_key == pine_id:
                    return entry_id
        order = event.order
        if order is not None:
            order_id = order.id
            for entry_id, marker in self._pending_defensive_close.items():
                if (marker.close_order_ref is not None
                        and marker.close_order_ref == order_id):
                    return entry_id
            event_coid = order.client_order_id
            if event_coid is not None:
                for entry_id, marker in self._pending_defensive_close.items():
                    if (marker.close_client_order_id is not None
                            and marker.close_client_order_id == event_coid):
                        return entry_id
        return None

    def _route_defensive_close_fill(self, event: OrderEvent) -> None:
        """Run deferred-cleanup for a defensive-close FILL, if any.

        The synthetic close dispatched by
        :meth:`_handle_bracket_attach_after_fill_reject` carries an
        engine-generated ``pine_id`` (``__pyne_defensive_close__<coid>``)
        that the standard :meth:`_cleanup_closed_position` cannot map to
        the parent entry. This handler matches the FILL event against
        the live :attr:`_pending_defensive_close` markers via
        :meth:`_match_pending_defensive_close` (pine_id → order ref →
        client_order_id fallback).

        On match:

        - The parent entry's tracking + Pine order-book entries are
          cleaned via :meth:`_cleanup_position_tracking`.
        - The marker is dropped from memory and from the parent entry
          row's ``extras`` (see :meth:`_clear_pending_defensive_close`).
        - An audit event ``'defensive_close_filled'`` is written so
          startup replay can detect post-restart that the FILL already
          settled.

        Idempotent: a second invocation finds no marker and returns.
        """
        matched_entry_id = self._match_pending_defensive_close(event)
        if matched_entry_id is None:
            return
        marker = self._pending_defensive_close[matched_entry_id]
        # Seed the duplicate-fill cache BEFORE attempting residual
        # cleanup. ``record_fill`` has already applied this FILL to
        # ``BrokerPosition`` (we run from :meth:`_route_event` after
        # the in-band ``record_fill`` call), so a WS replay /
        # polled-orders re-delivery of the same FILL would otherwise
        # be applied a second time and open a phantom opposite-side
        # trade from flat. The settled-identity cache is the only
        # guard against that, so it MUST be populated regardless of
        # whether the residual-cancel step below succeeds. The audit
        # log + marker drop further down still depend on residual
        # cancel success.
        self._mark_defensive_close_settled(marker, event)
        # Persist ``fill_observed=True`` on the marker BEFORE attempting
        # the final residual cancel. The in-memory duplicate-fill caches
        # seeded above do NOT survive a restart — without a durable
        # signal that the FILL already landed, a crash between here and
        # the audit-log write below would leave
        # :meth:`_replay_pending_defensive_closes` treating the marker
        # as pre-FILL on the next startup. The replay would re-cancel
        # the (now empty) residual set as a no-op, then wait forever
        # for a FILL that already happened in the prior instance —
        # eventually halting via
        # :meth:`_raise_if_stale_pending_defensive_close`. Persisting
        # the flag lets startup replay re-seed the duplicate-fill
        # caches and route the marker through the FILL-side branch of
        # :meth:`_retry_residual_cleanup_after_transient_fill` so the
        # documented post-FILL lifecycle finishes after restart.
        if not marker.fill_observed:
            # Mirror the synchronous-success / parked-recovery /
            # attached-resolution / flat-broker-snapshot
            # ``natural_close_at`` stamp. When ``execute_close`` parked and
            # the FILL arrives BEFORE :meth:`_maybe_attach_defensive_close_ref`
            # or :meth:`_maybe_run_attached_defensive_close_cleanup` runs,
            # this is the first proof the close landed on the exchange —
            # without the stamp, brokers whose position disappears before
            # the close activity is ingested would let the plugin-side
            # missing-position reconciler see the unstamped parent row as
            # an unexpected manual close and halt the run. Stamping BEFORE
            # the ``fill_observed`` flip means a crash between here and the
            # flag persist is recovered cleanly: the FILL replays through
            # this branch on restart and re-stamps idempotently. A crash
            # AFTER the flag persist routes the marker through
            # :meth:`_retry_residual_cleanup_after_transient_fill`'s
            # ``cache_seeded`` branch, where the stamp has already been
            # written here so the lifecycle finish is safe.
            self._stamp_natural_close_at(
                marker.reject_context.position_coid,
            )
            # Capture the actual FILL ``event.order.id`` alongside the
            # ``fill_observed`` flag. The transient residual-cancel
            # failure branch below returns BEFORE writing the audit
            # row, and on the next restart
            # :meth:`_replay_pending_defensive_closes` re-seeds
            # :attr:`_settled_defensive_close_order_refs` from
            # ``marker.close_order_ref`` /
            # ``marker.close_client_order_id`` only. When the FILL
            # arrived with an ``order.id`` different from
            # ``close_order_ref`` (polled-orders fallback or broker
            # rekey) AND the duplicate replay does not echo
            # ``client_order_id``, the dedup cache would miss it and
            # a stale replay of the same FILL could reach
            # :meth:`record_fill`. Persisting the observed fill id
            # closes that hole — startup replay seeds the cache from
            # this field so the duplicate is filtered out before any
            # state mutation.
            observed_fill_order_id = (
                event.order.id if event.order is not None else None
            )
            # Snapshot the FIFO-closed entry ids captured in
            # :meth:`_route_event` before ``record_fill`` walked
            # ``open_trades``. Persisting them alongside ``fill_observed``
            # lets the post-FILL retry branch in
            # :meth:`_retry_residual_cleanup_after_transient_fill` run the
            # same FIFO-aware cleanup as the in-flight path even after
            # intervening :meth:`record_fill` calls have overwritten the
            # live ``_last_fifo_closed_entry_ids`` cache. Without this,
            # the retry path would always clean ``marker.entry_id``,
            # which in pyramiding can target a different entry than the
            # one FIFO actually closed.
            #
            # Merge with any closures already accumulated on the marker
            # by prior defensive-close ``partial`` events (see the
            # ``t == 'partial'`` branch in :meth:`_route_event`): the
            # live cache only holds THIS terminal fill's slice, so
            # overwriting would drop entries an earlier partial FIFO
            # walk consumed (e.g. LongA via partial, LongB via terminal).
            merged_fifo_closed = list(marker.fifo_closed_entry_ids)
            seen_fifo_closed = set(merged_fifo_closed)
            for entry_id in self._last_fifo_closed_entry_ids:
                if entry_id not in seen_fifo_closed:
                    merged_fifo_closed.append(entry_id)
                    seen_fifo_closed.add(entry_id)
            fifo_closed_snapshot = tuple(merged_fifo_closed)
            marker = dataclasses.replace(
                marker,
                fill_observed=True,
                fill_exchange_order_id=observed_fill_order_id,
                fifo_closed_entry_ids=fifo_closed_snapshot,
            )
            self._set_pending_defensive_close(marker.entry_id, marker)
        # Final residual-cancel retry before the marker is dropped. The
        # dispatch-time loop in :meth:`_cancel_bracket_reject_residuals`
        # swallows transient
        # :class:`ExchangeConnectionError`/:class:`OrderDispositionUnknownError`
        # failures on the assumption that the marker (still armed) will
        # cause a later reconcile/restart to retry. Once we clear the
        # marker below, that retry pathway is gone — so a transient
        # failure here must keep the marker armed: we run the cancel
        # with ``raise_on_transient=True``, and on transient failure we
        # return WITHOUT writing the audit / clearing the marker so the
        # stale-grace timer (:meth:`_raise_if_stale_pending_defensive_close`)
        # or the next reconcile can retry. The plugin's
        # ``cancel_broker_order_ref`` idempotency contract guarantees the
        # replay is safe (already-cancelled / already-filled refs become
        # no-ops). Any explicit broker reject still halts via the
        # ``BrokerError`` branch inside the helper.
        try:
            self._cancel_bracket_reject_residuals(
                marker.reject_context, raise_on_transient=True,
            )
        except (ExchangeConnectionError, OrderDispositionUnknownError) as exc:
            _blog_warning(
                "defensive-close FILL routed but residual cancel transiently "
                "failed for entry %s (%s: %s) — marker stays armed for retry "
                "via reconcile / stale-grace halt",
                marker.entry_id, type(exc).__name__, exc,
            )
            return
        # Residual cancel succeeded — only now is it safe to declare the
        # defensive close settled (audit log + marker drop). Writing the
        # audit before the cancel would let
        # :meth:`_replay_pending_defensive_closes` decide the close is
        # done on the next restart even though the residual was never
        # cancelled.
        if self._store_ctx is not None:
            # Persist BOTH the dispatch-time close ref AND the actual
            # filled order id. The polled-orders fallback can deliver a
            # FILL whose ``event.order.id`` differs from
            # ``marker.close_order_ref`` (e.g. a parked close whose
            # marker carried ``close_order_ref=None`` settles via
            # ``client_order_id`` echo and the broker assigned a fresh
            # id at fill time, or the plugin re-keyed the order during
            # reconcile). After a restart,
            # :meth:`_replay_pending_defensive_closes` seeds
            # :attr:`_settled_defensive_close_order_refs` from the
            # audit row's ``exchange_order_id`` column AND from the
            # ``fill_exchange_order_id`` payload key written here, so a
            # duplicate FILL replayed with either id is rejected by
            # :meth:`_is_duplicate_defensive_close_fill` before it
            # reaches :meth:`record_fill`.
            fill_order_id = (
                event.order.id if event.order is not None else None
            )
            audit_exchange_order_id = (
                marker.close_order_ref
                if marker.close_order_ref is not None
                else fill_order_id
            )
            payload: dict[str, object] = {
                'entry_id': marker.entry_id,
                'symbol': marker.reject_context.symbol,
                'position_side': marker.reject_context.position_side,
                'qty': marker.reject_context.qty,
                'close_client_order_id': marker.close_client_order_id,
            }
            if (fill_order_id is not None
                    and fill_order_id != marker.close_order_ref):
                payload['fill_exchange_order_id'] = fill_order_id
            self._store_ctx.log_event(
                kind='defensive_close_filled',
                intent_key=marker.close_intent_key,
                client_order_id=marker.reject_context.position_coid,
                exchange_order_id=audit_exchange_order_id,
                payload=payload,
            )
        # Align cleanup with the entry/entries whose FIFO ``Trade`` was
        # actually closed by :meth:`record_fill`. In pyramiding scenarios
        # (LongA older, LongB newer with bracket-attach reject), Pine FIFO
        # semantics close the OLDEST trade first — so ``record_fill``
        # reduces LongA while ``marker.entry_id`` is LongB. Cleaning only
        # ``marker.entry_id`` would clear LongB's intents but leave
        # LongA's intents pointing at a Trade that no longer exists,
        # corrupting later exits and P&L. ``_last_fifo_closed_entry_ids``
        # was captured in :meth:`_route_event` from the
        # ``new_closed_trades`` slice the FIFO walk just appended.
        #
        # When ``marker.entry_id`` is NOT among the FIFO-closed entries
        # (i.e. the protected entry's Trade survived because FIFO took
        # an older sibling instead), its intents must stay live so the
        # remaining Trade row keeps a path to its exits — otherwise the
        # next ``sync()`` rebuilds nothing and the Trade goes orphaned.
        # Fall back to ``marker.entry_id`` only when the FIFO walk
        # recorded no closure at all (degenerate / no-FIFO path that
        # bypassed this method), so the marker's own pine-id is still
        # cleared rather than left dangling.
        #
        # Partial-close survivor guard: ``record_fill``'s partial-close
        # branch splits an oversized trade and keeps the residual in
        # ``open_trades`` while appending the closed slice to
        # ``new_closed_trades`` — both pieces share the same
        # ``entry_id``. Cleaning that id here would wipe intents for an
        # entry whose Trade is still partially open. Exclude entry ids
        # that still appear in ``open_trades`` (the cleanup helper is
        # idempotent against ids it cannot find, so the exclusion only
        # narrows the cleanup, never widens it).
        #
        # Empty-cleanup distinction: when the FIFO walk recorded NO
        # closure at all (the marker's snapshot is empty — the
        # degenerate / no-FIFO path that bypassed ``record_fill``'s
        # FIFO branch), the marker's own pine-id is still the right
        # cleanup anchor. But when the FIFO walk DID record closures
        # and every one of them was filtered out by the survivor guard
        # (e.g. the close partially consumed a single tranche so the
        # residual is still open with the same ``entry_id``), the
        # entries are by construction still partially open — falling
        # back to ``marker.entry_id`` would clear intents for a trade
        # that still has live exposure. Distinguish the two with the
        # raw FIFO list, not the post-filter cleanup_targets length.
        #
        # Source the FIFO closures from ``marker.fifo_closed_entry_ids``
        # (just merged above with ``_last_fifo_closed_entry_ids``):
        # this includes every entry FIFO touched across the partials →
        # terminal sequence, not only the terminal slice. Reading from
        # the live cache would lose closures from earlier partial events
        # whose ``_last_fifo_closed_entry_ids`` was reset at the top of
        # each subsequent :meth:`_route_event`.
        surviving_entry_ids = {
            trade.entry_id
            for trade in self._position.open_trades
            if trade.entry_id is not None
        }
        cleanup_targets: list[str] = [
            entry_id
            for entry_id in marker.fifo_closed_entry_ids
            if entry_id not in surviving_entry_ids
        ]
        if not cleanup_targets and not marker.fifo_closed_entry_ids:
            cleanup_targets.append(marker.entry_id)
        for cleanup_id in cleanup_targets:
            self._cleanup_position_tracking(cleanup_id)
        # Drop any parked-dispatch envelope for the synthetic close.
        # When ``execute_close`` parked with
        # :class:`OrderDispositionUnknownError`, ``_park_pending`` left
        # the close ``client_order_id`` in ``_pending_verification`` (and
        # a persisted ``pending_verifications`` row) keyed on the close
        # COID. ``_verify_pending_dispatches`` normally clears that row
        # via ``record_unpark`` when the order shows up in
        # ``get_open_orders`` — but the FILL we just routed proves the
        # close already settled, and for plugins where the parked close
        # never appears in ``get_open_orders`` (e.g. position-attached
        # market closes that move straight from queued → filled) the
        # cleanup pathway would otherwise never fire and the engine
        # would keep polling ``get_open_orders`` forever for a COID it
        # cannot find.
        close_coid = marker.close_client_order_id
        if close_coid is not None:
            self._pending_verification.pop(close_coid, None)
            self._persisted_pending_anchors.pop(close_coid, None)
            if self._store_ctx is not None:
                self._store_ctx.record_unpark(close_coid)
        # Drop the synthetic close intent's ``_order_mapping`` slot and
        # envelope state. ``_cleanup_position_tracking`` above only
        # cleans entries keyed by ``marker.entry_id`` (the parent
        # position's ``pine_id``) and exit intents whose ``from_entry``
        # matches — the synthetic ``CloseIntent`` carries its own
        # ``pine_id`` (``__pyne_defensive_close__<coid>``) and was
        # never wired as an ``ExitIntent`` with a ``from_entry``, so
        # neither pass touches it. Without this cleanup, the close
        # broker order ref(s) stay in ``_order_mapping`` after
        # settlement and any later broker event for that order id
        # (e.g. a status update arriving on the polled-orders fallback
        # after the FILL was already routed) would route via
        # :meth:`_find_key_for_order_id` as if a live intent still
        # exists. The persisted envelope row would likewise survive
        # restart and replay a stale close anchor. ``_drop_envelope``
        # also persists ``record_complete`` so the JSONL self-compacts.
        close_intent_key = marker.close_intent_key
        if close_intent_key:
            self._order_mapping.pop(close_intent_key, None)
            self._drop_envelope(close_intent_key)
        self._clear_pending_defensive_close(matched_entry_id)

    def _is_duplicate_defensive_close_fill(self, event: OrderEvent) -> bool:
        """``True`` if ``event`` is a replay of an already-settled defensive close.

        The check runs in :meth:`_route_event` before
        :meth:`BrokerPosition.record_fill` so a duplicate delivery cannot
        corrupt position state. Matches in priority order:

        - ``event.pine_id`` against the cache of synthetic
          ``__pyne_defensive_close__<coid>`` ids;
        - ``event.order.id`` against the cache of broker order refs
          captured for settled closes.
        """
        pine_id = event.pine_id
        if (pine_id is not None
                and pine_id in self._settled_defensive_close_pine_ids):
            return True
        order = event.order
        if order is not None:
            if order.id in self._settled_defensive_close_order_refs:
                return True
            # Polled-orders identity path: a parked defensive close whose
            # marker carried ``close_order_ref=None`` settles by matching
            # ``event.order.client_order_id`` (see
            # :meth:`_route_defensive_close_fill`). A WS / poll replay of
            # that same FILL after restart arrives with ``pine_id=None``
            # and a ``order.id`` the engine never saw at dispatch time,
            # so neither set above can dedupe it — the
            # ``client_order_id`` echo is the only stable identity.
            coid = order.client_order_id
            if (coid is not None
                    and coid in self._settled_defensive_close_client_order_ids):
                return True
        return False

    def _is_neutralised_parent_entry_fill(self, event: OrderEvent) -> bool:
        """``True`` if ``event`` is a late parent ENTRY fill the engine must drop.

        Consulted by :meth:`_route_event` BEFORE
        :meth:`BrokerPosition.record_fill`. Match order mirrors
        :meth:`_match_pending_defensive_close`:

        - ``event.pine_id`` against the cache of parent entry pine ids;
        - ``event.order.id`` (broker-side identity) against the parent
          ``position_coid`` cache — plugins that emit the COID as the
          order id (Capital.com, Bybit) match here;
        - ``event.order.client_order_id`` echo against the same coid set
          for plugins that mint a fresh broker order id on the fill but
          echo the original COID via ``client_order_id``.

        Triggered exclusively from the no-FIFO defensive-close settle
        branch — that branch only matches a re-armed marker against an
        empty ``open_trades``, so a true positive here is always a parent
        fill the engine already flattened in memory.
        """
        pine_id = event.pine_id
        if (pine_id is not None
                and pine_id in self._neutralised_parent_entry_pine_ids):
            return True
        order = event.order
        if order is not None:
            if order.id in self._neutralised_parent_entry_coids:
                return True
            coid = order.client_order_id
            if (coid is not None
                    and coid in self._neutralised_parent_entry_coids):
                return True
        return False

    def _mark_parent_entry_neutralised(
            self, marker: PendingDefensiveClose,
    ) -> None:
        """Cache the parent ENTRY identifiers carried by ``marker``.

        Called from the no-FIFO defensive-close settle branch in
        :meth:`_route_event` so a late parent ENTRY ``filled`` /
        ``partial`` event arriving after the close has settled is
        recognised and discarded before it reaches
        :meth:`BrokerPosition.record_fill` (where it would otherwise open
        a phantom fresh trade against an already-flat broker).
        """
        ctx = marker.reject_context
        if ctx.from_entry:
            self._neutralised_parent_entry_pine_ids.add(ctx.from_entry)
        if ctx.position_coid:
            self._neutralised_parent_entry_coids.add(ctx.position_coid)

    def _mark_defensive_close_settled(
            self, marker: PendingDefensiveClose, event: OrderEvent,
    ) -> None:
        """Record a settled defensive close in the duplicate-fill cache.

        Stores both the synthetic ``close_intent_key`` and any broker
        order ref(s) known for the close: ``marker.close_order_ref`` (set
        when dispatch succeeded synchronously) and ``event.order.id`` (the
        ref that actually filled — usually identical, but the polled-
        orders fallback can deliver a FILL whose ``order.id`` differs
        from the dispatch-time ref when the broker assigned a fresh id).
        """
        if marker.close_intent_key:
            self._settled_defensive_close_pine_ids.add(marker.close_intent_key)
        if marker.close_order_ref is not None:
            self._settled_defensive_close_order_refs.add(marker.close_order_ref)
        if marker.close_client_order_id is not None:
            self._settled_defensive_close_client_order_ids.add(
                marker.close_client_order_id,
            )
        if event.order is not None:
            self._settled_defensive_close_order_refs.add(event.order.id)
            event_coid = event.order.client_order_id
            if event_coid is not None:
                self._settled_defensive_close_client_order_ids.add(event_coid)

    def _cleanup_closed_position(self, event: OrderEvent) -> None:
        """Drop tracking for an entry fully closed by a TP/SL/CLOSE fill.

        Identifies the closed entry's ``pine_id`` from the event:
        ``event.from_entry`` is set on bracket-leg fills emitted by
        plugins that own a separate exit order (Bybit, Binance USDM);
        ``event.pine_id`` carries it on plugins where the closing
        activity references the entry's own exchange id (Capital.com's
        position-attached bracket). Falling back across both fields
        keeps the cleanup correct on every plugin.

        Delegates to :meth:`_cleanup_position_tracking` for the actual
        intent / Pine-order-book teardown so the bracket-reject
        defensive-close path can reuse the same logic without
        synthesising a fake event.
        """
        closed_entry_id = event.from_entry or event.pine_id
        if not closed_entry_id:
            return
        self._cleanup_position_tracking(closed_entry_id)

    def _resolve_parent_opening_ref(self, from_entry: str) -> str | None:
        """Resolve the dispatch ref the parent position actually OPENED under.

        A normal / limit-won entry opens under its native :data:`KIND_ENTRY`
        client-order-id. A both-set entry whose STOP leg won the OCO opens via
        the stop-fired MARKET under the deterministic :data:`KIND_ENTRY_STOP`
        id — the SAME pinned envelope anchor as :data:`KIND_ENTRY`, only the
        kind char differs. The broker reconcile feed reports the live position
        under *that* id (the opening row's ``client_order_id``), so every
        native fail-safe parent-ref resolver must key on it: a state registered
        under :data:`KIND_ENTRY` would never see its confirming snapshot,
        escalate to ``DEGRADED``, and silently block new entries / brackets on
        the symbol; the close-time retire would likewise miss it.

        Returns the :data:`KIND_ENTRY_STOP` ref iff a position row was persisted
        under it — the durable, restart-surviving signal that the stop fired
        (the entry-stop watch row is terminal-filtered once the stop wins and
        cannot be relied on after restart). Otherwise the :data:`KIND_ENTRY`
        ref. ``None`` when neither a live envelope nor a persisted anchor is
        tracked for ``from_entry`` (mirrors the callers' existing ``None``
        guards).
        """
        parent_envelope = self._envelopes.get(from_entry)
        if parent_envelope is not None:
            kind_entry_ref = parent_envelope.client_order_id(KIND_ENTRY)
            stop_ref = parent_envelope.client_order_id(KIND_ENTRY_STOP)
        else:
            parent_anchor = self._persisted_envelope_anchors.get(from_entry)
            if parent_anchor is None:
                return None
            kind_entry_ref = build_client_order_id(
                run_tag=self._run_tag,
                pine_id=from_entry,
                bar_ts_ms=parent_anchor.bar_ts_ms,
                kind=KIND_ENTRY,
                retry_seq=parent_anchor.retry_seq,
            )
            stop_ref = build_client_order_id(
                run_tag=self._run_tag,
                pine_id=from_entry,
                bar_ts_ms=parent_anchor.bar_ts_ms,
                kind=KIND_ENTRY_STOP,
                retry_seq=parent_anchor.retry_seq,
            )
        if (self._store_ctx is not None
                and self._store_ctx.get_order(stop_ref) is not None):
            return stop_ref
        return kind_entry_ref

    def _retire_native_failsafe_for_entry(self, closed_entry_id: str) -> None:
        """Retire any §2.6.7 ``NativeStopState`` parked under ``closed_entry_id``.

        The manager keys state on the parent entry COID, not the Pine entry id,
        so resolve the COID two ways (union, idempotent):

        - walk the leg ledger for legs still carrying
          ``parent_entry_dispatch_ref`` under this entry (must run before the
          partial-bracket cascade evicts them);
        - :meth:`_resolve_parent_opening_ref` for the COID the position opened
          under (``KIND_ENTRY`` normally, ``KIND_ENTRY_STOP`` when the both-set
          STOP leg won the OCO), live envelope or restart-anchor rebuild.

        The latter covers the case where every partial leg already terminated
        (no leg to walk) but the state is still parked — without it a
        ``DEGRADING`` / ``DEGRADED`` state strands and
        :meth:`NativeFailsafeManager.block_new_entry` blocks new entries on an
        already-flat symbol. ``on_deal_id_disappeared`` is a no-op for an
        unknown / already-retired ref, so this is safe to call from the
        never-filled cancel / reject paths too.
        """
        if not closed_entry_id:
            return
        retired_refs: set[str] = set()
        for leg in self._partial_bracket_engine.iter_legs():
            if leg.from_entry == closed_entry_id and leg.parent_entry_dispatch_ref:
                retired_refs.add(leg.parent_entry_dispatch_ref)
        opening_ref = self._resolve_parent_opening_ref(closed_entry_id)
        if opening_ref is not None:
            retired_refs.add(opening_ref)
        for ref in retired_refs:
            self._native_failsafe_manager.on_deal_id_disappeared(
                ref, now_ms=float(self._current_bar_ts_ms),
            )

    def _cleanup_position_tracking(
            self,
            closed_entry_id: str,
            *,
            cascade_reason: str = 'parent_closed',
    ) -> None:
        """Drop active intent + Pine order-book entries for ``closed_entry_id``.

        Cleans:

        - ``_active_intents`` — entry intent keyed by ``pine_id``, every
          exit intent whose ``from_entry`` matches the closed entry.
        - ``_order_mapping`` and envelope state for the dropped keys.
        - ``position.entry_orders`` (single-key by ``pine_id``) and
          ``position.exit_orders`` (composite ``(exit_id, from_entry)``;
          every key whose ``from_entry`` matches is dropped).

        Idempotent: callable from both the natural close-fill handler
        (:meth:`_cleanup_closed_position`) and proactive paths such as
        :meth:`_handle_bracket_attach_after_fill_reject` — a follow-up
        invocation on the same id finds nothing to do.
        """
        # §2.6.7 retire: drop any NativeStopState parked under this entry id
        # BEFORE the partial-bracket cascade evicts the legs (the leg-walk
        # branch needs them populated). Idempotent / no-op when nothing is
        # registered, so the never-filled cancel path can call it safely.
        self._retire_native_failsafe_for_entry(closed_entry_id)
        # Cascade-cancel any active engine-trigger partial legs under
        # this entry — the parent just flattened, so every leg's row
        # must be marked ``cascaded_cancel_by_parent_close`` and the
        # in-memory ledger purged. Without this, a proactive cleanup
        # path (e.g. bracket-attach-after-fill defensive close) leaves
        # the leg rows live until a later sync; they could be replayed
        # after restart or matched against a new position reusing the
        # same ``from_entry``.
        self._partial_bracket_engine.cascade_cancel_by_parent_close(
            symbol=self._symbol,
            from_entry=closed_entry_id,
            reason=cascade_reason,
        )
        # Entry intent + its mapping/envelope.
        self._active_intents.pop(closed_entry_id, None)
        self._order_mapping.pop(closed_entry_id, None)
        self._drop_envelope(closed_entry_id)
        # Every exit intent that points at this entry.
        for key in list(self._active_intents.keys()):
            intent = self._active_intents[key]
            if (isinstance(intent, ExitIntent)
                    and intent.from_entry == closed_entry_id):
                self._active_intents.pop(key, None)
                self._order_mapping.pop(key, None)
                self._drop_envelope(key)
        # Pine-side order dicts.
        entry_orders = getattr(self._position, 'entry_orders', None)
        exit_orders = getattr(self._position, 'exit_orders', None)
        if entry_orders is not None:
            entry_orders.pop(closed_entry_id, None)
        if exit_orders is not None:
            for ex_key in list(exit_orders.keys()):
                if isinstance(ex_key, tuple) and ex_key[1] == closed_entry_id:
                    exit_orders.pop(ex_key, None)

    def _filled_intent_key(self, event: OrderEvent) -> str | None:
        """Resolve a fill event to the ``intent_key`` of the owning intent.

        Exits track identity as ``(pine_id, from_entry)``; entries / closes
        as just ``pine_id``. An event coming from a plugin that did not tag
        the Pine identity cannot be routed and the method returns ``None``.
        """
        if event.pine_id is None:
            return None
        if event.leg_type in (LegType.TAKE_PROFIT, LegType.STOP_LOSS):
            if event.from_entry is None:
                return None
            return f"{event.pine_id}\0{event.from_entry}"
        return event.pine_id

    # === Partial entry fill → bracket qty amend (WS5, Option A) ===

    def _amend_bracket_qty_for_entry_fill(self, event: OrderEvent) -> None:
        """Track partial entry fills with an incremental bracket qty amend.

        Canonical semantics (Option A): the bracket's qty follows the entry's
        cumulative ``filled_qty`` — every partial fill dispatches a
        :meth:`BrokerPlugin.modify_exit` with ``new_qty = filled_qty``. This
        mirrors the Pine backtester, where exits exist against the actually
        filled entry portion; it also guarantees that if the entry ends with
        unfilled remainder (cancel/expire), the bracket is not over-sized.

        Suppressed when:

        - The plugin declared ``tp_sl_bracket = CapabilityLevel.NATIVE`` —
          the exchange tracks partial entry fills natively (Bybit V5
          attached TP/SL, Capital.com position-attribute bracket).
        - No bracket is active for ``event.pine_id`` (plain entry, no exit).
        - The current ExitIntent already matches the target qty — avoids
          redundant dispatch churn.

        Over-fill guard: if ``event.order.filled_qty`` exceeds the entry
        intent's intended qty (exchange rounding or adversarial event), the
        amend is capped at the intended qty and a
        :class:`LegRepairFailedEvent` is emitted so the runner can surface
        the anomaly.
        """
        if self._tp_sl_bracket_native:
            return
        pine_id = event.pine_id
        if pine_id is None:
            return
        if pine_id in self._pending_defensive_close:
            # A sibling bracket attach reject for this same parent fill
            # just armed a defensive close (see
            # :meth:`_handle_bracket_attach_after_fill_reject`) and
            # ``_route_event`` resumed from the exception into this amend
            # path. The same-entry ``ExitIntent`` guard in
            # :meth:`_diff_and_dispatch` deliberately keeps already-active
            # brackets in ``_active_intents`` until the close FILL
            # settles, and ``_cancel_bracket_reject_residuals`` may have
            # just cancelled the broker-side bracket order this method
            # would now amend. Dispatching ``modify_exit`` here would
            # therefore either retarget a cancelled bracket or
            # recreate protection against a position the engine is
            # actively flattening — both fight the marker. Skip; the
            # close FILL settlement path cleans up the bracket slot.
            return

        filled_qty = event.order.filled_qty
        if filled_qty <= 0.0:
            return

        bracket_key: str | None = None
        bracket_intent: ExitIntent | None = None
        for key, intent in self._active_intents.items():
            if not isinstance(intent, ExitIntent):
                continue
            if intent.from_entry != pine_id:
                continue
            # Partial-qty (scale-out) brackets carry a script-declared
            # qty that is intentionally smaller than the parent fill —
            # e.g. ``strategy.exit(qty=2)`` against a 10-unit entry. The
            # cumulative-fill amend path is designed for whole-row
            # brackets whose qty should track the parent fill; applying
            # it to a partial bracket would silently grow the scale-out
            # to the parent fill size and close more than the strategy
            # requested. The engine-trigger / reduce-only working order
            # paths own the per-leg qty bookkeeping for partial brackets.
            if intent.is_partial_qty_bracket:
                continue
            bracket_key = key
            bracket_intent = intent
            break
        if bracket_key is None or bracket_intent is None:
            return

        entry_intent = self._active_intents.get(pine_id)
        target_qty = filled_qty
        overfill = False
        if isinstance(entry_intent, EntryIntent) and filled_qty > entry_intent.qty:
            target_qty = entry_intent.qty
            overfill = True

        if target_qty == bracket_intent.qty:
            if overfill:
                self._emit_overfill_event(
                    bracket_intent, entry_intent, filled_qty,
                )
            return

        old_qty = bracket_intent.qty
        new_intent = dataclasses.replace(bracket_intent, qty=target_qty)
        self._dispatch_modify(bracket_intent, new_intent)
        self._active_intents[bracket_key] = new_intent
        self._sync_pine_exit_qty(new_intent, target_qty)

        self._emit_broker_event(LegPartialRepairedEvent(
            pine_id=new_intent.pine_id,
            from_entry=new_intent.from_entry,
            leg='bracket',
            generation=0,
            old_qty=old_qty,
            new_qty=target_qty,
        ))
        if overfill:
            self._emit_overfill_event(new_intent, entry_intent, filled_qty)

    def _promote_pending_partial_bracket_legs(self, event: OrderEvent) -> None:
        """Promote ``pending_entry`` partial-bracket legs on a parent fill.

        Engine-trigger partial brackets dispatched against an unfilled
        limit/stop parent are persisted as :data:`LEG_STATE_PENDING_ENTRY`
        with a deliberate gate in :meth:`_dispatch_engine_trigger_partial_bracket`:
        the leg cannot be ``armed`` until the parent is live in
        ``open_trades``, otherwise :meth:`SoftwarePartialBracketEngine.on_price_tick`
        would refuse the trigger via the ``parent_flat`` safety check.
        That fill arrives here; without this promotion the legs sit in
        ``pending_entry`` forever and the WATCH-phase tick handler skips
        them (it only iterates armed legs), so the bracket never fires.

        Idempotent: :meth:`SoftwarePartialBracketEngine.on_parent_entry_filled`
        is a no-op for legs already in ``armed`` (or any other) state, so
        repeated ENTRY fills on the same ``pine_id`` (cumulative-partial
        progression) do not double-promote.
        """
        pine_id = event.pine_id
        if pine_id is None:
            return
        parent_trade = None
        for trade in self._position.open_trades:
            if trade.entry_id == pine_id:
                parent_trade = trade
                break
        if parent_trade is None:
            # Parent fill recorded but ``open_trades`` does not yet carry a
            # row for ``pine_id`` (e.g. the no-FIFO terminal branch already
            # closed it inside the same event). No live parent → no leg to
            # promote against; the cascade-cancel path handles cleanup.
            return
        parent_sign = parent_trade.sign
        if parent_sign > 0.0:
            parent_side = 'long'
        elif parent_sign < 0.0:
            parent_side = 'short'
        else:
            return
        fill_price = event.fill_price
        if fill_price is None or fill_price <= 0.0:
            # The engine resolves tick-offset legs against ``fill_price``;
            # without it we cannot compute an absolute level. Absolute-price
            # legs would still be promotable, but the on_parent_entry_filled
            # path needs a usable ``fill_price`` for the trail-activation /
            # tick-offset branches, so skip the whole promotion. A later
            # FILL with a populated price (or the stale-grace timer) drives
            # the recovery; meanwhile the WATCH phase keeps the legs
            # ``pending_entry`` rather than firing against a missing price.
            return
        resolver = TickOffsetResolver(self._mintick)
        self._partial_bracket_engine.on_parent_entry_filled(
            symbol=self._symbol,
            from_entry=pine_id,
            fill_price=float(fill_price),
            parent_side=parent_side,
            parent_qty=parent_trade.size if parent_trade.size >= 0.0 else -parent_trade.size,
            resolver=resolver,
        )

    def _sync_pine_exit_qty(self, bracket: ExitIntent, new_qty: float) -> None:
        """Mutate the Pine-side exit :class:`Order` to match the amended qty.

        Without this, the next :meth:`sync` rebuilds the ExitIntent from the
        unchanged ``pos.exit_orders[(exit_id, from_entry)]`` (whose ``size``
        still equals the original full qty), the diff engine sees a mismatch
        against the amended active intent, and emits a *second* ``modify_exit``
        back to the original qty — undoing the partial-fill cascade we just did.
        """
        exit_orders = getattr(self._position, 'exit_orders', None)
        if exit_orders is None:
            return
        order = exit_orders.get((bracket.pine_id, bracket.from_entry))
        if order is None:
            return
        sign = 1.0 if order.size >= 0.0 else -1.0
        order.size = sign * new_qty
        order.sign = sign if new_qty > 0.0 else 0.0

    def _emit_overfill_event(
            self,
            bracket: ExitIntent,
            entry: 'Intent | None',
            filled_qty: float,
    ) -> None:
        entry_qty = entry.qty if isinstance(entry, EntryIntent) else None
        self._emit_broker_event(LegRepairFailedEvent(
            pine_id=bracket.pine_id,
            from_entry=bracket.from_entry,
            leg='bracket',
            reason=(
                f"overfill detected: filled_qty={filled_qty} exceeds "
                f"entry qty={entry_qty}"
            ),
            action_taken='capped',
        ))

    def _emit_broker_event(self, event: BrokerEvent) -> None:
        """Forward a structured broker event to the registered sink, if any."""
        if self._broker_event_sink is None:
            _log.info("broker event (no sink): %r", event)
            return
        try:
            self._broker_event_sink(event)
        except Exception:  # pragma: no cover — defensive
            _log.exception("broker_event_sink raised for event %r", event)

    # === §2.6 / §2.6.7 broker-native fail-safe wiring =====================

    def set_native_bracket_dispatcher(
            self,
            dispatcher: 'Callable[[NativeBracketSnapshot], None] | None',
    ) -> None:
        """Inject the plugin-side hook that realises a desired native
        bracket snapshot on the broker (PUT /positions/{dealId} on
        Capital.com).

        The dispatcher receives one :class:`NativeBracketSnapshot` per
        call and must be *synchronous from the engine's view*: it either
        realises the snapshot and RETURNS (success) or RAISES (failure).
        :meth:`drive_native_failsafe` records the outcome on the
        dispatcher's behalf — ``record_put_success`` on a normal return,
        ``record_put_failure`` on any exception — both carrying the
        snapshot's ``generation`` and the drive's ``now_ms``. So the
        dispatcher must NOT call those record hooks itself (the engine is
        the single outcome owner, which avoids a double-decrement of the
        retry budget on the failure path). The independent
        :meth:`record_native_bracket_observed` recovery feed is still
        plugin-driven, from the reconcile snapshot.

        Without a registered dispatcher the §2.6 fail-safe state machine
        still runs (worst-SL recompute, ownership transitions, event
        emission) but no broker traffic is generated — the state-only
        model used by unit tests until the plugin opts in.
        """
        self._native_bracket_dispatcher = dispatcher

    def record_native_bracket_put_success(
            self,
            parent_entry_dispatch_ref: str,
            *,
            generation: int,
            now_ms: float,
    ) -> None:
        """Plugin-side hook: a PUT carrying ``generation`` returned 200."""
        self._native_failsafe_manager.record_put_success(
            parent_entry_dispatch_ref,
            generation=generation,
            now_ms=now_ms,
        )

    def record_native_bracket_put_failure(
            self,
            parent_entry_dispatch_ref: str,
            *,
            generation: int,
            reason: str,
            now_ms: float,
    ) -> None:
        """Plugin-side hook: a PUT carrying ``generation`` failed."""
        self._native_failsafe_manager.record_put_failure(
            parent_entry_dispatch_ref,
            generation=generation,
            reason=reason,
            now_ms=now_ms,
        )

    def record_native_bracket_observed(
            self,
            parent_entry_dispatch_ref: str,
            *,
            stop_level: float | None,
            profit_level: float | None,
            trailing_stop: float | None,
            now_ms: float,
    ) -> None:
        """Apply an observed broker-side bracket snapshot **on the main
        thread** for ``parent_entry_dispatch_ref``.

        Drives the §2.6.7 ``DEGRADING → HEALTHY`` recovery transition: a
        successful PUT only clears ``pending_put`` via
        :meth:`record_native_bracket_put_success`; the actual confirmation
        that the broker is now carrying the desired snapshot lands here.
        Without this hook a restart-replayed parent registered with
        ``pending_confirmation=True`` (so ``DEGRADING``) would stay
        ``DEGRADING`` until the stale-window timer escalates to
        ``DEGRADED``, blocking new entries / brackets until a manual
        ``reset_to_engine``.

        This mutates the manager's per-parent state, so it is NOT
        thread-safe: the reconcile pass runs on the broker event-loop
        thread and must use :meth:`enqueue_native_bracket_observed`
        instead, which hands the snapshot to the main thread via the
        observed queue (drained in :meth:`drive_native_failsafe`). This
        direct entry point is for main-thread callers and tests that
        control ``now_ms`` explicitly.
        """
        self._native_failsafe_manager.on_native_bracket_observed(
            parent_entry_dispatch_ref,
            stop_level=stop_level,
            profit_level=profit_level,
            trailing_stop=trailing_stop,
            now_ms=now_ms,
        )

    def enqueue_native_bracket_observed(
            self,
            parent_entry_dispatch_ref: str,
            *,
            stop_level: float | None,
            profit_level: float | None,
            trailing_stop: float | None,
    ) -> None:
        """Thread-safe recovery feed for the broker reconcile loop.

        The Capital.com plugin's ``_reconcile_snapshot`` runs on the broker
        event-loop thread and calls this (via the runner-installed
        ``native_failsafe_observed_sink``) once per live position. The
        snapshot is queued and applied on the MAIN thread in
        :meth:`drive_native_failsafe` — exactly the marshaling
        ``run_event_stream`` uses for async fills — so the manager's
        per-parent state is only ever mutated from one thread.

        No ``now_ms`` parameter: the confirmation is timestamped with the
        bar at which it is drained (the engine's current bar clock), which
        also keeps every fail-safe timestamp on a single clock. Snapshots
        for refs the manager does not track are dropped at drain time.
        """
        self._native_bracket_observed_queue.put((
            parent_entry_dispatch_ref, stop_level, profit_level, trailing_stop,
        ))

    def _drain_native_bracket_observed(self, *, now_ms: float) -> None:
        """Apply queued broker-observed snapshots (main thread).

        Called at the top of :meth:`drive_native_failsafe`, BEFORE
        ``tick_stale_window``, so a confirmation that landed since the last
        sync flips ``DEGRADING → HEALTHY`` and prevents an unnecessary
        escalation to ``DEGRADED`` — ``on_native_bracket_observed`` recovers
        only from ``DEGRADING``, never from ``DEGRADED``.

        The reconcile poll loop runs on the broker thread at its own cadence
        and may enqueue several observations for the same parent between two
        main-thread drives (the bar interval easily spans multiple polls).
        Only the LAST observation per parent describes the broker's current
        bracket; an earlier one carrying a pre-PUT level is stale. Applying
        them FIFO would let that stale mismatch flip
        ``ENGINE_FAILSAFE → UNKNOWN`` while the trailing match cannot restore
        ownership (:meth:`NativeFailsafeManager.on_native_bracket_observed`
        recovers ``DEGRADING → HEALTHY`` but never ``UNKNOWN → ENGINE_FAILSAFE``),
        stranding the parent until a manual reset. Coalesce per ref first and
        apply only the newest snapshot.
        """
        latest: dict[str, tuple[float | None, float | None, float | None]] = {}
        while True:
            try:
                ref, stop_level, profit_level, trailing_stop = (
                    self._native_bracket_observed_queue.get_nowait()
                )
            except queue.Empty:
                break
            latest[ref] = (stop_level, profit_level, trailing_stop)
        for ref, (stop_level, profit_level, trailing_stop) in latest.items():
            self._native_failsafe_manager.on_native_bracket_observed(
                ref,
                stop_level=stop_level,
                profit_level=profit_level,
                trailing_stop=trailing_stop,
                now_ms=now_ms,
            )

    def _rehydrate_native_failsafe_from_replayed_legs(self) -> None:
        """Seed :class:`NativeFailsafeManager` state from replayed legs.

        Scheduled by ``__init__`` (via
        ``_pending_native_failsafe_rehydrate``) and actually executed
        at the top of the first :meth:`sync` once ``_current_bar_ts_ms``
        has been set to a real bar timestamp. Walks the ledger,
        deduplicates per ``parent_entry_dispatch_ref``, registers each
        parent with the manager, and triggers the worst-SL recompute
        so the §2.6 broker-native fail-safe state machine has an
        accurate snapshot. ``parent_side`` is derived from the leg's
        ``side`` field — the leg's recorded side is the CLOSE side, so
        ``sell`` → parent LONG, ``buy`` → parent SHORT (see
        :class:`PartialBracketLeg` docstring).

        Without this seeding the §2.6.7 fail-safe path produces no
        ``NativeStopState`` for the parent, the recompute path returns
        early, and the broker-native stop snapshot stays absent until a
        new leg transition happens — which can be too late if the
        first post-restart bar already crosses a stop level.
        """
        seen: dict[str, tuple[str, str]] = {}
        # Track parents whose replayed legs include at least one armed SL /
        # trail contributor. TP-only brackets never seed a broker-native
        # worst-SL, so registering them with ``pending_confirmation=True``
        # would strand the state in ``DEGRADING`` forever: the subsequent
        # ``recompute_worst_sl`` would compare ``None`` desired against
        # the existing ``None`` desired and queue no PUT, leaving no
        # confirmation path to flip the state back to ``HEALTHY``.
        expects_native_sl: set[str] = set()
        for leg in self._partial_bracket_engine.iter_legs():
            ref = leg.parent_entry_dispatch_ref
            if not ref or ref in seen:
                continue
            if leg.side == 'sell':
                parent_side = 'long'
            elif leg.side == 'buy':
                parent_side = 'short'
            else:
                continue
            seen[ref] = (leg.symbol, parent_side)
        for leg in self._partial_bracket_engine.iter_legs():
            ref = leg.parent_entry_dispatch_ref
            if not ref or ref not in seen:
                continue
            if leg.leg_state not in LEG_STATE_ACTIVE:
                continue
            if leg.leg_kind == LEG_KIND_TP_PARTIAL:
                # TP legs never contribute to the worst-SL set.
                continue
            if leg.leg_state == LEG_STATE_PENDING_ENTRY:
                # Pre-fill SL/trail leg: no broker dealId exists yet, so the
                # recompute below intentionally skips this leg (mirrors
                # :meth:`_recompute_native_failsafe_for_parent`). Forcing
                # ``pending_confirmation=True`` here would arm
                # ``DEGRADING`` without queueing any PUT, so no
                # :meth:`on_native_bracket_observed` snapshot can ever
                # confirm the state back to ``HEALTHY`` — the stale
                # window then escalates the parent to ``DEGRADED`` and
                # later :meth:`on_parent_entry_filled` recomputes are
                # rejected by :meth:`recompute_worst_sl` (``DEGRADED``
                # early-returns), leaving the eventual fill without a
                # broker-native fail-safe and poisoning the symbol-level
                # entry gate. The leg's contribution is restored when
                # :meth:`SoftwarePartialBracketEngine.on_parent_entry_filled`
                # promotes it to ``armed`` and fires the listener; at
                # that point the live ``register_parent`` /
                # ``recompute_worst_sl`` path takes over.
                continue
            # Currently armed (or higher) SL/trail leg signals that a
            # broker-native SL is expected on this parent, so the
            # restart-replay state must enter ``DEGRADING`` until the
            # broker confirms the PUT.
            expects_native_sl.add(ref)
        if not seen:
            return
        # ``_current_bar_ts_ms`` is the first sync's bar timestamp:
        # ``__init__`` defers this call via
        # ``_pending_native_failsafe_rehydrate`` so the value below is a
        # real epoch-ms anchor, not zero. That keeps the stale-window
        # timer in :meth:`NativeFailsafeManager.tick_stale_window` from
        # measuring against epoch zero and immediately demoting the
        # freshly rehydrated state to ``DEGRADED``.
        now_ms = float(self._current_bar_ts_ms)
        for ref, (symbol, parent_side) in seen.items():
            # Restart-replay safety (§2.6.7): the previous process may
            # have transitioned this parent to ``DEGRADED`` / ``UNKNOWN``
            # before crashing, but health and owner are NOT persisted in
            # the leg rows — only the legs themselves are. Re-registering
            # with the default ``HEALTHY`` + ``ENGINE_FAILSAFE`` would
            # silently pass ``block_new_entry`` / ``block_new_partial_bracket``
            # on the first post-restart bar, letting Pine add exposure
            # before the next broker snapshot proves the native stop is
            # actually in place. ``pending_confirmation=True`` starts the
            # state in ``DEGRADING``: blocks for new entries / brackets
            # stay armed until :meth:`on_native_bracket_observed` confirms
            # the broker side, and the stale-window timer escalates to
            # ``DEGRADED`` if no confirmation arrives. The worst-SL
            # recompute below still queues a PUT so the broker stop is
            # re-attached on the first sync — that PUT's snapshot is
            # what eventually drives the ``DEGRADING → HEALTHY`` flip.
            # Only force ``DEGRADING`` for parents whose replayed legs
            # actually require a broker-native SL. A TP-only bracket has
            # no native fail-safe stop the engine needs to confirm, so
            # registering it as ``DEGRADING`` would permanently block
            # new entries / brackets without any path to recovery (the
            # recompute below queues no PUT, ``on_native_bracket_observed``
            # never matches a non-existent stop). Such parents start
            # ``HEALTHY`` immediately — the live path treats them the
            # same way (``register_parent`` is called there without
            # ``pending_confirmation``).
            self._native_failsafe_manager.register_parent(
                parent_entry_dispatch_ref=ref,
                symbol=symbol,
                parent_side=parent_side,
                mintick=self._mintick,
                minmove=self._minmove,
                pricescale=self._pricescale,
                pending_confirmation=ref in expects_native_sl,
                now_ms=now_ms,
            )
            sl_levels: list[float] = []
            for leg in self._partial_bracket_engine.iter_legs():
                if leg.parent_entry_dispatch_ref != ref:
                    continue
                if leg.leg_state not in LEG_STATE_ACTIVE:
                    continue
                if leg.leg_state == LEG_STATE_PENDING_ENTRY:
                    # Mirror the live-path guard in
                    # :meth:`_recompute_native_failsafe_for_parent`: a
                    # pending-entry leg has no broker dealId behind it
                    # yet, so seeding its absolute SL into the manager
                    # would queue a PUT for a phantom parent on the very
                    # first post-restart sync. The leg's contribution
                    # rejoins the set when ``on_parent_entry_filled``
                    # promotes it to ``armed`` and fires the listener.
                    continue
                if leg.leg_kind == LEG_KIND_TP_PARTIAL:
                    continue
                if leg.trigger_level is None:
                    # Pre-activation trail legs (§2.6.3): contribute the
                    # planned post-activation initial stop derived from
                    # ``trail_activation_level`` ± ``trigger_offset`` so
                    # the broker-native fail-safe is armed before the
                    # trail activates. Mirrors the live-path branch in
                    # :meth:`_recompute_native_failsafe_for_parent`.
                    if (leg.leg_kind == LEG_KIND_TRAIL_PARTIAL
                            and leg.trail_activation_level is not None
                            and leg.trigger_offset is not None):
                        parent_long = leg.side == 'sell'
                        planned = (leg.trail_activation_level - leg.trigger_offset) \
                            if parent_long \
                            else (leg.trail_activation_level + leg.trigger_offset)
                        sl_levels.append(planned)
                    continue
                sl_levels.append(leg.trigger_level)
            self._native_failsafe_manager.recompute_worst_sl(
                parent_entry_dispatch_ref=ref,
                active_sl_levels=sl_levels,
                now_ms=now_ms,
                trigger_kind='lifecycle',
            )

    # === Cancel-tentative state machine ====================================

    def _rehydrate_cancel_tentative_from_replayed_legs(self) -> None:
        """Rebuild :attr:`_cancel_disposition_pending` from replayed legs.

        Scheduled by ``__init__`` (via
        ``_pending_cancel_tentative_rehydrate``) and executed at the top
        of the startup :meth:`reconcile` BEFORE the first event drain so
        that replayed FILL / cancelled events for tentative parents can
        be resolved event-driven. (The first :meth:`sync` also calls
        this as a defensive fallback, but the gate flag makes it a
        no-op once reconcile has already rehydrated.) Walks the ledger
        for legs in :data:`LEG_STATE_CANCEL_TENTATIVE`, deduplicates per
        ``intent_key``, and re-anchors the stale-grace deadline from
        the persisted :data:`EXTRAS_KEY_CANCEL_TENTATIVE_SINCE_TS_MS`.

        Without this seeding the post-restart engine would observe the
        tentative legs in its in-memory ledger (via ``restart_replay``)
        but would never re-attempt the cancel disposition resolution —
        the legs would survive the stale-grace window unobserved, and
        the diff-loop adoption guard would never fire to defer a fresh
        reissue.
        """
        tentative_parents: set[str] = set()
        for leg in self._partial_bracket_engine.iter_legs():
            if leg.leg_state != LEG_STATE_CANCEL_TENTATIVE:
                continue
            parent_key = leg.from_entry
            if parent_key in self._cancel_disposition_pending:
                continue
            since_ts_ms = leg.extras.get(EXTRAS_KEY_CANCEL_TENTATIVE_SINCE_TS_MS)
            if since_ts_ms is None:
                # Persisted CANCEL_TENTATIVE leg without an anchor
                # timestamp — defensive fallback: stamp wall-clock now
                # so the stale-grace timer still has a deadline in the
                # same epoch as the comparison in
                # :meth:`_drive_cancel_tentative`.
                since_ts_ms = self._cancel_tentative_now_ms()
            reason = leg.extras.get(
                'cancel_tentative_reason', 'rehydrated_from_replay',
            )
            self._cancel_disposition_pending[parent_key] = _CancelTentativeMeta(
                since_ts_ms=int(since_ts_ms),
                reason=str(reason),
            )
            tentative_parents.add(parent_key)
        # Seed ``_order_mapping`` for the tentative parents BEFORE the
        # first event drain. The startup :meth:`reconcile` calls us
        # before draining replayed broker events; the FILL / CANCELLED
        # event-driven resolution paths in :meth:`_route_event` look the
        # tentative parent up via :meth:`_find_key_for_order_id`, which
        # iterates :attr:`_order_mapping`. Without this seeding the map
        # is still empty at drain time (``_verify_pending_dispatches``
        # — which would otherwise populate it from the open-orders poll
        # — only runs at the start of the first :meth:`sync`), so the
        # replayed event cannot be matched to its tentative parent and
        # the disposition resolution falls through to the
        # :meth:`_drive_cancel_tentative` retry path. Pulling the
        # broker order_ids from the persisted ``orders`` table (each
        # row carries ``intent_key`` and ``exchange_order_id``) gives
        # the routing helpers a live mapping immediately. Engine-trigger
        # partial legs are engine-internal and never have a broker-side
        # order, so a missing row for the parent is simply a no-op.
        if tentative_parents and self._store_ctx is not None:
            for row in self._store_ctx.iter_live_orders(symbol=self._symbol):
                if (row.intent_key is None
                        or row.intent_key not in tentative_parents
                        or not row.exchange_order_id):
                    continue
                current = self._order_mapping.setdefault(row.intent_key, [])
                if row.exchange_order_id not in current:
                    current.append(row.exchange_order_id)

    def _mark_intent_cancel_disposition_pending(
            self,
            intent_key: str,
            *,
            reason: str,
            now_ms: int,
    ) -> None:
        """Atomic two-level cancel-tentative entry.

        Sets the envelope/intent-level :attr:`_cancel_disposition_pending`
        flag AND flips every ``pending_entry`` leg of the trio to
        :data:`LEG_STATE_CANCEL_TENTATIVE` via
        :meth:`SoftwarePartialBracketEngine.mark_legs_cancel_tentative`.
        Idempotent: if the entry already exists, the call is a no-op
        (the original ``since_ts_ms`` is preserved so the stale-grace
        deadline does not slip).

        Emits :class:`PartialBracketCancelTentativeStartedEvent` for
        audit / observability.
        """
        if intent_key in self._cancel_disposition_pending:
            return
        self._cancel_disposition_pending[intent_key] = _CancelTentativeMeta(
            since_ts_ms=now_ms,
            reason=reason,
        )
        # ``intent_key`` here is the parent ``EntryIntent.intent_key``,
        # which equals ``leg.from_entry`` on every child partial bracket
        # leg — so the engine API's parent-scoped flip uses the same
        # value as the shadow-map key. The leg flip is a no-op when no
        # partial bracket exists for the parent (entry without partial
        # exits), which is the structural case where the shadow-map
        # entry alone provides the refuse-and-defer guard.
        self._partial_bracket_engine.mark_legs_cancel_tentative(
            intent_key, reason=reason, now_ms=now_ms,
        )
        self._emit_broker_event(PartialBracketCancelTentativeStartedEvent(
            intent_key=intent_key,
            reason=reason,
            since_ts_ms=now_ms,
        ))

    def _clear_intent_cancel_disposition_pending(
            self,
            intent_key: str,
            *,
            outcome: CancelDispositionOutcome,
            via: str,
            now_ms: int,
            is_partial_fill: bool = False,
    ) -> None:
        """Resolve a cancel-tentative entry forward or backward.

        Called from three sites:

        - :meth:`_drive_cancel_tentative` on a non-``UNKNOWN`` outcome.
        - :meth:`_on_order_event` FILL ag when a late parent fill
          arrives for a tentative key (``via='order_event'``).
        - :meth:`_on_order_event` cancelled ag when the broker pushes
          a CANCELLED confirmation for a tentative key.

        Acts according to the outcome:

        - :attr:`CancelDispositionOutcome.CANCEL_CONFIRMED`,
          :attr:`CancelDispositionOutcome.TOO_LATE_TO_CANCEL`,
          :attr:`CancelDispositionOutcome.STILL_OPEN`:
          confirm-cancel (legs → ``aborted_parent_never_arrived``;
          mapping, envelope, active_intents dropped).
        - :attr:`CancelDispositionOutcome.ALREADY_FILLED`:
          restore (legs → ``armed``; parent re-registered with
          ``NativeFailsafeManager`` ``pending_confirmation=True``;
          mapping retained, envelope retained, active_intents removed
          so the next diff-loop reissues / adopts).

        :attr:`CancelDispositionOutcome.UNKNOWN` is NOT a resolution —
        the caller handles it by bumping the retry counter and
        returning without invoking this method.
        """
        meta = self._cancel_disposition_pending.pop(intent_key, None)
        if meta is None:
            return
        duration_ms = max(0, now_ms - meta.since_ts_ms)
        if outcome is CancelDispositionOutcome.ALREADY_FILLED:
            # Restore to ``pending_entry`` rather than ``armed``: tentative
            # legs born from a pending-parent dispatch carry only
            # ``trigger_offset`` (tick distance), never an absolute
            # ``trigger_level``. Promoting them straight to ``armed`` here
            # would skip the trigger-offset → trigger-level resolution
            # that :meth:`SoftwarePartialBracketEngine.on_parent_entry_filled`
            # owns, leaving them armed with ``trigger_level=None`` and
            # therefore inert in the WATCH tick. Both reachable call paths
            # (order-event FILL routing AND reconcile retry) get the legs
            # promoted with a usable fill price below.
            legs = self._partial_bracket_engine.restore_legs_from_cancel_tentative(
                intent_key,
                parent_filled=False,
                reason='cancel_tentative_already_filled',
            )
            # Fire the OCA-cancel cascade BEFORE popping the active
            # intent. :meth:`_cascade_oca_cancel_for_key` looks up the
            # filled intent in :attr:`_active_intents` to discover its
            # OCA group; dropping the slot first would leave every
            # ``oca_type='cancel'`` sibling live exchange-side. The
            # downstream :meth:`_route_event` call to
            # :meth:`_cascade_oca_cancel` is a no-op via the in-sync
            # idempotency set (:attr:`_cancelled_oca_groups_this_sync`).
            # On the reconcile-retry path the slot may already be empty
            # (no FILL event in this process) and the helper returns
            # early — cheap, idempotent.
            #
            # Respect :data:`OcaPartialFillPolicy.FULL_FILL_ONLY` when
            # the resolution was triggered by a ``partial`` fill event:
            # the configured semantics keep OCA siblings live until a
            # full fill arrives. Skipping the proactive cascade here
            # (and not populating
            # :attr:`_cancelled_oca_groups_this_sync`) lets the
            # downstream :meth:`_cascade_oca_cancel(event)` enforce the
            # policy gate uniformly; the eventual full fill will
            # cascade through the normal path. ``is_partial_fill`` is
            # ``False`` on the reconcile-retry caller (no event is
            # available there, so we cannot distinguish partial from
            # full and conservatively keep the legacy cascade).
            skip_oca_cascade = (
                is_partial_fill
                and self._oca_partial_policy
                is OcaPartialFillPolicy.FULL_FILL_ONLY
            )
            if not skip_oca_cascade:
                self._cascade_oca_cancel_for_key(intent_key)
            # Drop ``_active_intents[intent_key]`` so the next sync's
            # diff-loop sees an empty slot and either reissues the script
            # intent or adopts the retained mapping via the post-restart
            # adoption branch (see :meth:`_diff_and_dispatch`). Without
            # this pop, a stale ``EntryIntent`` reached via the
            # cancel+reexecute modify path (see :class:`_PartialBracketModifyDeferred`)
            # would remain in ``_active_intents``, so a script-emitted
            # ``CloseIntent`` (or replacement) sharing the parent key
            # would be classified as a modify of the already-filled
            # entry instead of dispatching fresh. The retained mapping /
            # envelope continue to anchor the parent's deterministic
            # ``client_order_id`` for the upcoming dispatch.
            self._active_intents.pop(intent_key, None)
            # Order-event path (``via='order_event'``): the FILL routing
            # block in :meth:`_route_event` runs us BEFORE
            # :meth:`BrokerPosition.record_fill` updates ``open_trades``,
            # so the parent is not yet in :attr:`_position.open_trades`
            # here. The subsequent ``_promote_pending_partial_bracket_legs(event)``
            # call later in the same FILL routing handles promotion with
            # the event's ``fill_price``.
            #
            # Reconcile-retry path (``via='reconcile_retry'``): the parent
            # fill event was processed in a prior tick (otherwise the
            # broker would not be reporting ALREADY_FILLED), so the
            # parent IS in ``open_trades``. Promote here explicitly using
            # the trade's ``entry_price`` — there is no surrounding event
            # to drive ``_promote_pending_partial_bracket_legs`` for us.
            # If the lookup misses (race against an unprocessed FILL), the
            # legs stay in ``pending_entry`` and the next FILL event drives
            # the promotion naturally.
            parent_trade = next(
                (t for t in self._position.open_trades
                 if t.entry_id == intent_key),
                None,
            )
            if (parent_trade is not None
                    and parent_trade.entry_price > 0.0):
                parent_side_for_promotion = (
                    'long' if parent_trade.sign > 0.0
                    else 'short' if parent_trade.sign < 0.0
                    else None
                )
                if parent_side_for_promotion is not None:
                    self._partial_bracket_engine.on_parent_entry_filled(
                        symbol=self._symbol,
                        from_entry=intent_key,
                        fill_price=float(parent_trade.entry_price),
                        parent_side=parent_side_for_promotion,
                        parent_qty=(
                            parent_trade.size if parent_trade.size >= 0.0
                            else -parent_trade.size
                        ),
                        resolver=TickOffsetResolver(self._mintick),
                    )
            # Re-register the parent with the §2.6 manager:
            # ``pending_confirmation=True`` mirrors the restart-replay
            # path's ``DEGRADING`` start until a broker snapshot
            # confirms the desired SL is in place. Iterate the legs
            # we just restored (each carries the
            # :attr:`parent_entry_dispatch_ref` regardless of whether the
            # subsequent promotion fired); the ``register_parent`` call
            # is idempotent and the worst-SL recompute reads the
            # in-memory ledger so it picks up any legs the explicit
            # promotion above moved to ``armed``.
            for leg in legs:
                ref = leg.parent_entry_dispatch_ref
                if not ref:
                    continue
                parent_side = (
                    'long' if leg.side == 'sell'
                    else 'short' if leg.side == 'buy'
                    else None
                )
                if parent_side is None:
                    continue
                self._native_failsafe_manager.register_parent(
                    parent_entry_dispatch_ref=ref,
                    symbol=leg.symbol,
                    parent_side=parent_side,
                    mintick=self._mintick,
                    minmove=self._minmove,
                    pricescale=self._pricescale,
                    pending_confirmation=True,
                    now_ms=float(now_ms),
                )
                self._recompute_native_failsafe_for_parent(
                    ref, trigger_kind='lifecycle',
                )
        else:
            self._partial_bracket_engine.confirm_cancel_tentative(
                intent_key,
                reason=f'cancel_tentative_resolved_{outcome.value}',
            )
            self._order_mapping.pop(intent_key, None)
            self._drop_envelope(intent_key)
            self._active_intents.pop(intent_key, None)
            # Drop any child partial-exit ExitIntent slots whose
            # ``from_entry`` matches the resolved parent. Mirrors
            # :meth:`_retire_pending_partials_for_cancelled_entry` so a
            # later same-id entry re-dispatches its partial legs instead
            # of seeing the stale ExitIntent as "already active" and
            # short-circuiting :meth:`_dispatch_engine_trigger_partial_bracket`.
            # ``confirm_cancel_tentative`` above closed the leg rows; this
            # cleans up the envelope/intent-level shadow state the legs
            # alone do not cover.
            partial_exit_keys_to_drop = [
                key for key, active in self._active_intents.items()
                if isinstance(active, ExitIntent)
                and active.is_partial_qty_bracket
                and active.from_entry == intent_key
            ]
            for exit_key in partial_exit_keys_to_drop:
                self._active_intents.pop(exit_key, None)
                self._order_mapping.pop(exit_key, None)
                self._drop_envelope(exit_key)
        self._emit_broker_event(PartialBracketCancelTentativeResolvedEvent(
            intent_key=intent_key,
            outcome=outcome,
            via=via,
            duration_ms=duration_ms,
        ))

    @staticmethod
    def _cancel_tentative_now_ms() -> int:
        """Wall-clock anchor (ms) for the cancel-tentative stale-grace timer.

        ``CANCEL_TENTATIVE_STALE_GRACE_S`` is a seconds-based timeout
        and must therefore be measured against real elapsed time.
        Using ``_current_bar_ts_ms`` (the bar-open timestamp) would
        couple the deadline to the chart timeframe: on a 1-second
        chart the grace expires roughly when intended, but on a
        1-minute / 1-hour chart no retries inside the same bar age at
        all, then the very next bar can immediately overflow a 10s
        grace and halt after a single retry. ``time.time()`` matches
        the defensive-close grace pattern (see
        :meth:`_raise_if_stale_pending_defensive_close`) and survives
        across restarts because the persisted leg-extras anchor is
        compared in the same epoch.
        """
        return int(time.time() * 1000)

    def _drive_cancel_tentative(self, *, now_ms: int) -> None:
        """Idempotent cancel-retry loop driven by :meth:`reconcile`.

        For each tentative entry, either:

        - the stale-grace deadline has passed → promote to
          ``DEGRADED_HALT`` via
          :meth:`_handle_cancel_tentative_stale_grace_expiry`, OR
        - re-invoke :meth:`BrokerPlugin.execute_cancel_with_outcome`
          with the retained envelope and act on the outcome via
          :meth:`_clear_intent_cancel_disposition_pending` (or bump
          the retry counter on ``UNKNOWN``).

        Snapshot the keys before iterating because the loop mutates
        the shadow map on resolution. Skips envelopes that are no
        longer retained (defensive — should not happen because
        cancel-tentative explicitly preserves envelopes).

        :param now_ms: Wall-clock anchor (epoch ms) for the
            stale-grace comparison and ``last_retry_ts_ms`` bookkeeping;
            callers obtain it via :meth:`_cancel_tentative_now_ms`.
        """
        if not self._cancel_disposition_pending:
            return
        stale_grace_ms = int(self._cancel_tentative_stale_grace_s * 1000)
        for intent_key in list(self._cancel_disposition_pending):
            meta = self._cancel_disposition_pending.get(intent_key)
            if meta is None:
                continue
            if now_ms - meta.since_ts_ms >= stale_grace_ms:
                self._handle_cancel_tentative_stale_grace_expiry(
                    intent_key, meta, now_ms=now_ms,
                )
                continue
            envelope = self._envelopes.get(intent_key)
            # The cancel-retry loop must call ``execute_cancel_with_outcome``
            # with a ``CancelIntent`` envelope (plugins assert on the intent
            # type). The retained ``_envelopes[intent_key]`` slot deliberately
            # keeps the ORIGINAL parent intent envelope so a later
            # ALREADY_FILLED resolution can recover the parent's deterministic
            # entry ``client_order_id`` (e.g. for
            # :meth:`_dispatch_engine_trigger_partial_bracket`). Build a
            # short-lived ``CancelIntent`` envelope on the fly for each
            # retry instead of overwriting the slot; the ``intent_key`` of a
            # parent entry equals its ``pine_id``, which is all
            # ``CancelIntent`` needs together with the engine's symbol.
            # The restart-replay rehydrate path
            # (:meth:`_rehydrate_cancel_tentative_state`) follows the same
            # contract: it only repopulates ``_cancel_disposition_pending``
            # from leg extras and lets this rebuild produce a valid request.
            if envelope is None or not isinstance(envelope.intent, CancelIntent):
                cancel = CancelIntent(pine_id=intent_key, symbol=self._symbol)
                envelope = self._build_cancel_envelope(cancel)
            meta.retry_count += 1
            meta.last_retry_ts_ms = now_ms
            try:
                outcome = self._run_async(
                    self._broker.execute_cancel_with_outcome(envelope),
                )
            except BrokerManualInterventionError as e:
                self._record_halt(e)
                raise
            except Exception as e:  # noqa: BLE001
                _blog_warning(
                    "cancel-tentative retry for %r raised %s; treating as UNKNOWN",
                    intent_key, e,
                )
                meta.last_retry_outcome = CancelDispositionOutcome.UNKNOWN
                continue
            meta.last_retry_outcome = outcome
            if outcome is CancelDispositionOutcome.UNKNOWN:
                continue
            self._clear_intent_cancel_disposition_pending(
                intent_key,
                outcome=outcome,
                via='reconcile_retry',
                now_ms=now_ms,
            )

    def _handle_cancel_tentative_stale_grace_expiry(
            self,
            intent_key: str,
            meta: '_CancelTentativeMeta',
            *,
            now_ms: int,
    ) -> None:
        """Promote a stale cancel-tentative entry to ``DEGRADED_HALT``.

        Called by :meth:`_drive_cancel_tentative` when ``now_ms -
        since_ts_ms`` crosses the stale-grace window. Emits the
        degraded event, drops the shadow map entry, and records a
        :class:`BrokerManualInterventionError`-equivalent halt — the
        partial brackets cannot be safely re-armed against a parent
        whose live/dead status the engine cannot determine.
        """
        self._cancel_disposition_pending.pop(intent_key, None)
        stale_grace_ms = int(self._cancel_tentative_stale_grace_s * 1000)
        self._emit_broker_event(PartialBracketCancelTentativeDegradedEvent(
            intent_key=intent_key,
            symbol=self._symbol,
            since_ts_ms=meta.since_ts_ms,
            stale_grace_ms=stale_grace_ms,
        ))
        halt = BrokerManualInterventionError(
            reason='partial_bracket_cancel_disposition_unresolved',
            intent_key=intent_key,
            context={
                'since_ts_ms': meta.since_ts_ms,
                'now_ms': now_ms,
                'stale_grace_ms': stale_grace_ms,
                'retry_count': meta.retry_count,
                'last_retry_outcome': (
                    meta.last_retry_outcome.value
                    if meta.last_retry_outcome is not None
                    else None
                ),
            },
        )
        self._record_halt(halt)

    def _cancel_residual_after_partial_resolution(
            self,
            intent_key: str,
    ) -> None:
        """Best-effort cancel of the residual broker order after a
        ``partial`` event resolved cancel-tentative as ``ALREADY_FILLED``.

        The script issued a ``CancelIntent`` whose initial dispatch
        timed out, so the parent entered cancel-tentative. The
        subsequent ``partial`` event proves at least some of the entry
        landed (legs are now armed against the partial fill), but the
        broker still holds the unfilled residual qty live. The script
        will not reissue a fresh cancel — from its view the original
        cancel already left — so without this attempt the residual
        could fill later and grow the position past the just-armed
        leg protection.

        One-shot semantics: we deliberately do NOT re-enter
        cancel-tentative on timeout. Re-marking would flip the legs
        we just armed back to ``LEG_STATE_CANCEL_TENTATIVE`` (via
        :meth:`_mark_intent_cancel_disposition_pending`), undoing the
        protection. If the broker times out, log a warning so an
        operator can intervene; the next bar's ``reconcile`` will not
        retry (no cancel-tentative state to drive), but the
        deterministic ``client_order_id`` makes the call idempotent if
        the strategy ever reuses the same parent ``pine_id`` and the
        broker still honors the cancel.

        :param intent_key: Parent ``EntryIntent.intent_key`` whose
            residual order remains live in :attr:`_order_mapping` and
            now needs an explicit cancel.
        """
        if intent_key not in self._order_mapping:
            # Mapping already cleaned by a concurrent path (e.g. the
            # broker pushed ``cancelled`` for the residual between the
            # partial-event resolution and this helper).
            return
        cancel = CancelIntent(pine_id=intent_key, symbol=self._symbol)
        envelope = self._build_cancel_envelope(cancel)
        try:
            self._run_async(self._broker.execute_cancel(envelope))
        except OrderDispositionUnknownError as e:
            _blog_warning(
                "residual cancel for %r after partial-event resolution "
                "timed out (coid=%s); broker may still hold the unfilled "
                "qty live — operator intervention may be required if the "
                "residual fills against the partial position's leg "
                "protection: %s",
                intent_key, e.client_order_id, e,
            )
            return
        except BrokerManualInterventionError as e:
            self._record_halt(e)
            raise
        except Exception as e:  # noqa: BLE001
            _blog_warning(
                "residual cancel for %r after partial-event resolution "
                "raised %s; leaving mapping in place",
                intent_key, e,
            )
            return
        # Successful cancel — drop the order mapping so the next
        # diff does not adopt the stale residual order id. The
        # envelope / persisted anchor stay in place: the parent
        # entry is still partially filled (open position), and a
        # future ``strategy.exit`` attaching a fresh partial bracket
        # on the same ``from_entry`` needs the envelope/anchor to
        # rebuild ``parent_entry_dispatch_ref`` via the restart-path
        # fallback in :meth:`_dispatch_engine_trigger_partial_bracket`.
        # Calling :meth:`_drop_envelope` here would wipe both
        # :attr:`_envelopes` AND :attr:`_persisted_envelope_anchors`,
        # so the dispatch would later refuse with "no parent envelope
        # tracked" and the new bracket would not arm. The envelope
        # is retired via :meth:`_cleanup_closed_position` when the
        # parent finally closes, matching the normal post-fill
        # lifecycle.
        self._order_mapping.pop(intent_key, None)

    def _on_partial_bracket_leg_state_change(
            self,
            leg: PartialBracketLeg,
            old_state: str | None,
            new_state: str,
    ) -> None:
        """Listener fired by :class:`SoftwarePartialBracketEngine` on every
        leg-state mutation. Drives the §2.6 worst-SL recompute so the
        broker-native fail-safe state machine sees every armed / triggered
        / cancelled SL leg without coupling the partial-bracket engine to
        the manager.

        A call with ``old_state == new_state`` is a *level move* (trail
        leg ``trigger_level`` shifted while the leg stayed armed); those
        feed the §2.6.5 trail coalesce window. Genuine state transitions
        bypass the throttle.
        """
        parent_ref = leg.parent_entry_dispatch_ref
        if parent_ref is None:
            return
        trigger_kind = 'trail' if old_state == new_state else 'lifecycle'
        self._recompute_native_failsafe_for_parent(
            parent_ref, trigger_kind=trigger_kind,
        )

    def _recompute_native_failsafe_for_parent(
            self,
            parent_entry_dispatch_ref: str,
            *,
            trigger_kind: str = 'lifecycle',
    ) -> None:
        """Walk the active SL / trail legs for one parent and feed their
        current trigger levels to the failsafe manager.

        Only legs in :data:`LEG_STATE_ACTIVE` contribute — terminated
        legs (triggered / cancelled / aborted) drop out of the worst-SL
        set per §2.6.3. Trail legs contribute their current armed
        ``trigger_level`` (post-activation) or their pre-activation
        watch level (the latter is the *planned* worst, used so the
        broker-native bracket is in place before activation).
        """
        manager = self._native_failsafe_manager
        state = manager.get_state(parent_entry_dispatch_ref)
        if state is None:
            return
        sl_levels: list[float] = []
        for leg in self._partial_bracket_engine.iter_legs():
            if leg.parent_entry_dispatch_ref != parent_entry_dispatch_ref:
                continue
            if leg.leg_state not in LEG_STATE_ACTIVE:
                # Terminal legs (triggered / cancelled / aborted) drop out
                # of the worst-SL set per §2.6.3. The listener fires
                # from :meth:`SoftwarePartialBracketEngine._transition`
                # BEFORE the leg is evicted from the ledger, so the just-
                # terminated leg is still iterable here with a non-ACTIVE
                # state; including it would keep an obsolete broker-native
                # stop level on the parent.
                continue
            if leg.leg_state == LEG_STATE_PENDING_ENTRY:
                # The parent entry has not filled yet — there is no broker
                # dealId for ``NativeFailsafeManager`` to target. Including
                # the leg's absolute SL here would queue a PUT against a
                # phantom parent and the dispatcher would fail / retry /
                # degrade the state before the entry order even lands. The
                # leg's contribution is restored when
                # :meth:`SoftwarePartialBracketEngine.on_parent_entry_filled`
                # promotes it to ``armed`` and fires the listener again with
                # ``trigger_kind='lifecycle'``.
                continue
            if leg.leg_kind == LEG_KIND_TP_PARTIAL:
                continue  # TP not part of worst-SL set (§2.6.2)
            level = leg.trigger_level
            if level is None:
                # Pre-activation trail legs (§2.6.3): no moving stop has
                # been seeded yet, but the *planned* worst case is the
                # post-activation initial stop derived from the
                # activation level and the trailing offset
                # (``activation - offset`` for a long parent,
                # ``activation + offset`` for a short). Feeding that
                # planned level keeps the broker-native fail-safe armed
                # before the trail activates so a crash / network split
                # during the pre-activation window cannot leave the
                # parent unprotected. Once
                # :meth:`SoftwarePartialBracketEngine._maybe_advance_trail`
                # crosses the activation level it seeds ``trigger_level``
                # and fires the listener again — the recompute then
                # picks up the actual moving stop in this same loop.
                if (leg.leg_kind == LEG_KIND_TRAIL_PARTIAL
                        and leg.trail_activation_level is not None
                        and leg.trigger_offset is not None):
                    parent_long = leg.side == 'sell'
                    planned = (leg.trail_activation_level - leg.trigger_offset) \
                        if parent_long \
                        else (leg.trail_activation_level + leg.trigger_offset)
                    sl_levels.append(planned)
                continue
            sl_levels.append(level)
        manager.recompute_worst_sl(
            parent_entry_dispatch_ref=parent_entry_dispatch_ref,
            active_sl_levels=sl_levels,
            now_ms=float(self._current_bar_ts_ms),
            trigger_kind=trigger_kind,
        )

    def drive_native_failsafe(self, *, now_ms: float) -> None:
        """Drain the manager's pending dispatches into the plugin
        dispatcher, and tick the stale-window state machine.

        Called by the runner once per sync pass. Idempotent: a manager
        with no pending dispatches and no degrading state is a no-op.
        Released snapshots from the trail coalesce window (§2.6.5)
        are also dispatched here.
        """
        manager = self._native_failsafe_manager
        # Apply broker-observed confirmations queued from the reconcile
        # (broker-loop) thread BEFORE ticking the stale window, so a confirm
        # that landed since the last sync flips DEGRADING -> HEALTHY and
        # prevents an unnecessary escalation to DEGRADED.
        self._drain_native_bracket_observed(now_ms=now_ms)
        manager.tick_stale_window(now_ms=now_ms)
        manager.flush_coalesced_trails(now_ms)
        for snapshot in manager.pending_dispatch():
            if self._native_bracket_dispatcher is None:
                continue  # Slice A without plugin opt-in: state-only run
            manager.mark_dispatch_in_flight(
                snapshot.parent_entry_dispatch_ref, now_ms=now_ms,
            )
            try:
                self._native_bracket_dispatcher(snapshot)
            except BrokerManualInterventionError as halt:
                # A dispatcher raising the manual-intervention signal means
                # automated execution can no longer continue safely (e.g. the
                # plugin cannot resolve the parent/deal mapping). Treating it
                # as a retryable PUT failure would let the strategy keep
                # trading on an unsafe broker state, so record the terminal
                # halt and re-raise — the same uniform contract every other
                # dispatch path honours (see :meth:`run_event_stream`).
                self._record_halt(halt)
                raise
            except Exception as e:
                _blog_warning(
                    "native bracket dispatcher raised for %s: %s",
                    snapshot.parent_entry_dispatch_ref, e,
                )
                manager.record_put_failure(
                    snapshot.parent_entry_dispatch_ref,
                    generation=snapshot.generation,
                    reason=f'dispatcher-exception:{type(e).__name__}',
                    now_ms=now_ms,
                )
            else:
                # Normal return == the PUT (and its confirm) succeeded. The
                # engine is the single outcome owner (see
                # :meth:`set_native_bracket_dispatcher`): record success here
                # so the dispatcher stays a pure PUT-or-raise actuator and the
                # failure path cannot be double-counted. ``record_put_success``
                # only clears ``pending_put`` — the actual level confirmation
                # arrives later via :meth:`record_native_bracket_observed`.
                manager.record_put_success(
                    snapshot.parent_entry_dispatch_ref,
                    generation=snapshot.generation,
                    now_ms=now_ms,
                )

    # === Both-set entry STOP leg (software OCO) ===========================

    def _arm_entry_stop_watch(
            self, intent: EntryIntent, envelope: DispatchEnvelope,
    ) -> None:
        """Persist + register the software price-watch for a both-set entry's
        STOP leg.

        The native LIMIT leg has already been dispatched by the caller under
        the parent envelope's :data:`KIND_ENTRY` client-order-id; that id is
        recorded as the leg-scoped cancel target. The watch row uses the
        distinct :data:`KIND_ENTRY_STOP_WATCH` id (no exchange order — the
        engine owns it). Idempotent: a second arm for the same ``pine_id``
        (e.g. a re-dispatch of an unchanged both-set entry) is a no-op.
        """
        pine_id = intent.intent_key
        if self._entry_stop_engine.has_watch(pine_id):
            return
        if intent.stop is None:
            return
        watch_coid = envelope.client_order_id(KIND_ENTRY_STOP_WATCH)
        limit_coid = envelope.client_order_id(KIND_ENTRY)
        stop_level = float(intent.stop)
        if self._store_ctx is not None:
            create_entry_stop_watch_row(
                self._store_ctx,
                coid=watch_coid,
                symbol=intent.symbol,
                side=intent.side,
                qty=intent.qty,
                intent_key=pine_id,
                pine_entry_id=intent.pine_id,
                stop_level=stop_level,
                limit_coid=limit_coid,
            )
        self._entry_stop_engine.register_watch(EntryStopWatch(
            coid=watch_coid,
            symbol=intent.symbol,
            pine_id=intent.pine_id,
            side=intent.side,
            qty=intent.qty,
            stop_level=stop_level,
            limit_coid=limit_coid,
            state=ENTRY_STOP_STATE_ARMED,
        ))
        _blog_info(
            "armed entry-stop watch for %r: native LIMIT rests, software "
            "STOP @ %s fires a market on cross",
            pine_id, stop_level,
        )

    def _retire_entry_stop_watch_on_fill(self, event: OrderEvent) -> None:
        """Retire a both-set entry-stop watch when its native LIMIT leg fills.

        Only ``armed`` / ``cancel_pending`` watches are retired here: in those
        states an ENTRY fill for the ``pine_id`` can only be the native LIMIT
        leg (the stop-fired MARKET has not been placed yet), so the LIMIT won
        the OCO and the watch must stop watching — no market may ever fire.

        In ``stop_market_pending`` the fill is the stop-fired MARKET itself;
        :meth:`_fire_entry_stop_market` settles that via ``mark_stop_won``, so
        this method deliberately ignores it (otherwise a replayed market fill
        on restart would be mislabelled as a limit win).
        """
        pine_id = event.pine_id
        if pine_id is None:
            return
        watch = self._entry_stop_engine.get_watch(pine_id)
        if watch is None:
            return
        if watch.state in (
                ENTRY_STOP_STATE_ARMED, ENTRY_STOP_STATE_CANCEL_PENDING,
        ):
            self._entry_stop_engine.mark_limit_won(
                pine_id, reason='native_limit_filled',
            )

    def _drive_entry_stop_triggers(self, *, last_price: float | None) -> None:
        """Drive the both-set entry STOP leg state machine each sync.

        Runs from the tail of :meth:`sync` after :meth:`_diff_and_dispatch`
        (so a watch armed this cycle is visible) and after
        :meth:`_drive_partial_bracket_triggers`. For each live watch:

        * ``armed`` — when ``last_price`` has crossed the stop level, latch
          ``cancel_pending`` and resolve the cancel-then-fire path.
        * ``cancel_pending`` — re-drive the (idempotent) leg-scoped LIMIT
          cancel and the hard disposition gate. Reached on the same tick as a
          fresh cross, on a retry after an UNKNOWN cancel, or on restart.
        * ``stop_market_pending`` — re-dispatch the (idempotent) MARKET.
          Reached only on restart after the cancel was confirmed but the
          process died before the market settled.

        Price-independent states (``cancel_pending`` / ``stop_market_pending``)
        are driven even when ``last_price`` is ``None`` so restart recovery is
        not gated on a price context.
        """
        watches = [
            w for w in self._entry_stop_engine.iter_watches()
            if w.symbol == self._symbol
        ]
        if not watches:
            return
        for watch in watches:
            state = watch.state
            if state == ENTRY_STOP_STATE_ARMED:
                if last_price is None:
                    continue
                if not SoftwareEntryStopEngine.stop_crossed(watch, last_price):
                    continue
                _blog_info(
                    "entry-stop %r crossed (price=%s stop=%s): cancelling "
                    "native LIMIT before firing market",
                    watch.pine_id, last_price, watch.stop_level,
                )
                if self._entry_stop_engine.begin_cancel(watch.pine_id) is None:
                    continue
                self._resolve_entry_stop_cancel_then_fire(watch.pine_id)
            elif state == ENTRY_STOP_STATE_CANCEL_PENDING:
                self._resolve_entry_stop_cancel_then_fire(watch.pine_id)
            elif state == ENTRY_STOP_STATE_MARKET_PENDING:
                self._fire_entry_stop_market(watch.pine_id)

    def _resolve_entry_stop_cancel_then_fire(self, pine_id: str) -> None:
        """Leg-scoped cancel of the native LIMIT, then the hard fire gate.

        The cancel disposition is the gate (never a halt — forbidden for a
        live bot):

        * ``ALREADY_FILLED`` → the LIMIT won the OCO; ``mark_limit_won`` and
          the market NEVER fires.
        * ``UNKNOWN`` → stay ``cancel_pending`` and retry next tick; the
          market NEVER fires while the LIMIT's fate is unknown (a fire here
          could double-open against a LIMIT that actually filled).
        * ``CANCEL_CONFIRMED`` / ``TOO_LATE_TO_CANCEL`` / ``STILL_OPEN`` →
          the LIMIT is provably gone with no fill; fire the MARKET.
        """
        watch = self._entry_stop_engine.get_watch(pine_id)
        if watch is None:
            return
        cancel = CancelIntent(pine_id=pine_id, symbol=self._symbol)
        cancel_envelope = self._build_cancel_envelope(cancel)
        try:
            outcome = self._run_async(
                self._broker.execute_cancel_with_outcome(cancel_envelope),
            )
        except BrokerManualInterventionError as e:
            self._record_halt(e)
            raise
        except Exception as e:  # noqa: BLE001
            # Transient / ambiguous cancel failure — treat as UNKNOWN: stay
            # cancel_pending and retry on the next tick. The market must not
            # fire until the LIMIT is provably gone.
            _blog_warning(
                "entry-stop %r LIMIT cancel raised %s; staying cancel_pending, "
                "market withheld until disposition resolves",
                pine_id, type(e).__name__,
            )
            return
        if outcome is CancelDispositionOutcome.ALREADY_FILLED:
            _blog_info(
                "entry-stop %r: native LIMIT already filled — limit won the "
                "OCO, no market fired",
                pine_id,
            )
            self._entry_stop_engine.mark_limit_won(
                pine_id, reason='limit_filled_during_stop_cancel',
            )
            return
        if outcome is CancelDispositionOutcome.UNKNOWN:
            _blog_warning(
                "entry-stop %r: LIMIT cancel disposition UNKNOWN; staying "
                "cancel_pending, market withheld until next tick resolves",
                pine_id,
            )
            return
        # CANCEL_CONFIRMED / TOO_LATE_TO_CANCEL / STILL_OPEN: the LIMIT is
        # provably gone with no fill — safe to fire the market.
        self._fire_entry_stop_market(pine_id)

    def _fire_entry_stop_market(self, pine_id: str) -> None:
        """Dispatch the stop-fired MARKET order for a both-set entry.

        The deterministic :data:`KIND_ENTRY_STOP` client-order-id is persisted
        BEFORE the POST (via ``confirm_limit_cancelled_fire_market``) so a
        crash-restart re-dispatches the SAME id and the broker dedups it —
        never a double-open. On restart a watch already in
        ``stop_market_pending`` re-enters here and re-fires idempotently.
        """
        watch = self._entry_stop_engine.get_watch(pine_id)
        if watch is None:
            return
        active = self._active_intents.get(pine_id)
        if isinstance(active, EntryIntent):
            base = active
        else:
            # Restart before the first post-restart diff repopulated
            # ``_active_intents`` — reconstruct the market intent from the
            # persisted watch.
            base = EntryIntent(
                pine_id=watch.pine_id,
                symbol=watch.symbol,
                side=watch.side,
                qty=watch.qty,
                order_type=OrderType.MARKET,
            )
        market_intent = dataclasses.replace(
            base,
            order_type=OrderType.MARKET,
            limit=None,
            stop=None,
            stop_fired_market=True,
        )
        market_envelope = self._build_envelope(market_intent)
        market_coid = market_envelope.client_order_id(KIND_ENTRY_STOP)
        if watch.state == ENTRY_STOP_STATE_CANCEL_PENDING:
            # Persist the market identity BEFORE the POST (verify-before-resend
            # on restart). No-op when already stop_market_pending (retry path).
            self._entry_stop_engine.confirm_limit_cancelled_fire_market(
                pine_id, market_coid=market_coid,
            )
        try:
            orders = self._run_async(
                self._broker.execute_entry(market_envelope),
            )
        except OrderDispositionUnknownError as e:
            # The market POST timed out; it may or may not have landed. Stay
            # stop_market_pending — the next tick re-fires with the SAME
            # deterministic coid, which the broker dedups, so no double-open.
            _blog_warning(
                "entry-stop %r market dispatch unknown disposition (coid=%s); "
                "staying stop_market_pending for idempotent retry: %s",
                pine_id, e.client_order_id, e,
            )
            return
        except BrokerManualInterventionError as e:
            self._record_halt(e)
            raise
        except OrderSkippedByPlugin as e:
            # Plugin declined (e.g. below min-size). The OCO is already
            # committed to the stop side (LIMIT cancelled), so there is no
            # safe leg left — settle as stop_won to stop re-trying and let the
            # operator observe the skip. The position simply did not open.
            _blog_warning(
                "entry-stop %r market dispatch skipped by plugin (%s); no "
                "position opened",
                pine_id, e.reason,
            )
            self._entry_stop_engine.mark_stop_won(pine_id)
            return
        except Exception as e:  # noqa: BLE001
            _blog_error(
                "entry-stop %r market dispatch failed (%s: %s); staying "
                "stop_market_pending for retry",
                pine_id, type(e).__name__, e,
            )
            return
        self._order_mapping[pine_id] = [o.id for o in orders]
        self._entry_stop_engine.mark_stop_won(pine_id)
        _blog_info(
            "entry-stop %r: stop fired, market opened -> %s",
            pine_id, self._order_mapping.get(pine_id),
        )

    def _drive_partial_bracket_triggers(
            self, *, last_price: float | None,
    ) -> None:
        """Engine-trigger partial bracket WATCH phase + close dispatch.

        Drives the lifecycle step that owns:
        :class:`SoftwarePartialBracketEngine` armed legs →
        :meth:`SoftwarePartialBracketEngine.on_price_tick` →
        synthetic :class:`CloseIntent` →
        :meth:`_dispatch_new` →
        :meth:`SoftwarePartialBracketEngine.confirm_trigger_dispatched`
        (or the failed / unknown settling variant). Runs from the tail
        of :meth:`sync` after :meth:`_diff_and_dispatch` so the
        in-memory ledger reflects every register / cancel / restart
        replay mutation produced by this sync before any trigger
        evaluates.

        Skips when ``last_price`` is ``None`` (e.g. the runner has no
        price context yet — startup before the first bar ingest) or
        when the engine ledger is empty.

        The parent live snapshot is refreshed via the broker plugin
        on every call. :meth:`SoftwarePartialBracketEngine.on_price_tick`
        already short-circuits to an empty result when the snapshot
        comes back ``None`` so a transient connection blip surfaces
        as "no triggers this tick" rather than a halt.
        """
        if last_price is None:
            return
        if not any(True for _ in self._partial_bracket_engine.iter_legs()):
            return
        # Snapshot refresh failures (connection blip, exchange 5xx, plugin
        # timeout) must NOT propagate out of ``sync`` — every armed leg
        # would skip its trigger evaluation and the next price tick has
        # another chance. :meth:`SoftwarePartialBracketEngine.on_price_tick`
        # already short-circuits when ``parent_snapshot is None``, so
        # forward the failure as "no snapshot this tick" rather than
        # turning a transient blip into a live-runner halt.
        try:
            parent_snapshot = self._run_async(
                self._broker.get_position(self._symbol),
            )
        except Exception as e:
            _blog_warning(
                "partial-bracket trigger evaluation skipped: parent "
                "snapshot refresh failed (%s: %s); re-evaluating on the "
                "next price tick",
                type(e).__name__, e,
            )
            return
        # ``get_position`` returns ``None`` for a flat symbol on plugins that
        # do not allocate a row until the first open (Capital.com, others);
        # other plugins surface the same state as a concrete
        # :class:`ExchangePosition` with ``side='flat'`` and ``size=0``. Both
        # shapes mean "no live parent here" and must take the same cleanup
        # path, otherwise :meth:`on_price_tick` would only retire legs whose
        # triggers happen to cross this tick and the rest would linger.
        # Two distinct situations land here and require opposite handling:
        #
        # * Parent has at least one ``armed`` (or ``triggering`` /
        #   ``triggered_*``) leg → the parent was supposed to be open. A
        #   no-position snapshot then means the position vanished (external/
        #   manual close, broker-native fail-safe SL, …). Cascade-cancel
        #   every active leg under that parent so non-crossing armed legs
        #   cannot block a same-key bracket or fire against a reused entry
        #   later. Routing through :meth:`on_price_tick` with a synthesised
        #   flat snapshot would only retire legs whose triggers happened to
        #   cross this tick; the rest would linger indefinitely.
        # * Parent has only ``pending_entry`` or ``cancel_tentative`` legs →
        #   the bracket was dispatched against a still-unfilled entry order,
        #   or that entry's cancel disposition is still being resolved. In
        #   both cases the plugin has not yet allocated a row precisely
        #   BECAUSE the entry has not filled; killing the legs / dropping
        #   the parent/exit mappings here would either leave the position
        #   unprotected once the entry fills (pending) or strand the
        #   cancel-retry loop in :meth:`_drive_cancel_tentative` (tentative).
        #   Leave those parents alone — the pending→armed transition fires
        #   on the ``ENTRY`` order event, not on a price tick, and the
        #   cancel disposition resolution dissolves the tentative state.
        parent_is_flat = parent_snapshot is None or (
            parent_snapshot.side == 'flat' or parent_snapshot.size == 0
        )
        if parent_is_flat:
            # ``get_position`` lag vs. local ledger: when a parent ENTRY
            # event has just been drained on this same ``sync`` cycle the
            # pending legs were promoted to :data:`LEG_STATE_ARMED` by
            # :meth:`_promote_pending_partial_bracket_legs`, but the broker
            # ``/positions`` snapshot can still return ``None`` for the
            # newly-filled position (REST eventual consistency). Gating
            # the cascade purely on ``parent_snapshot is None`` would
            # then evict the freshly-armed bracket and leave the just-
            # opened parent unprotected. Use the local
            # :attr:`Position.open_trades` ledger — which is updated
            # synchronously from order events before this WATCH phase
            # runs — to filter the cascade to ``from_entry``s that
            # really are gone from BOTH sides; defer the rest until a
            # later sync sees a consistent snapshot.
            locally_open_entry_ids: set[str] = {
                trade.entry_id for trade in self._position.open_trades
            }
            from_entries_to_cascade: set[str] = set()
            for leg in self._partial_bracket_engine.iter_legs():
                if leg.symbol != self._symbol:
                    continue
                # ``PENDING_ENTRY`` legs are awaiting the parent ENTRY fill;
                # ``CANCEL_TENTATIVE`` legs sit alongside an entry whose
                # cancel disposition is still being resolved by
                # :meth:`_drive_cancel_tentative` (the parent has no
                # position yet either way). Cascading them here would drop
                # the parent/exit mappings before the cancel-retry loop
                # finishes and would not even close the legs, because
                # neither state is "active" in the engine's sense.
                if leg.leg_state in (
                    LEG_STATE_PENDING_ENTRY, LEG_STATE_CANCEL_TENTATIVE,
                ):
                    continue
                if leg.from_entry in locally_open_entry_ids:
                    # Local parent still alive → REST snapshot lag.
                    # Skip cascade for this ``from_entry`` and let the
                    # next sync re-evaluate against a fresh snapshot.
                    continue
                from_entries_to_cascade.add(leg.from_entry)
            # Route each truly-flat ``from_entry`` through the canonical
            # parent-flat cleanup. :meth:`_cleanup_position_tracking`
            # (a) retires :class:`NativeFailsafeManager` state for the
            # parent entry COID via :meth:`on_deal_id_disappeared` — a
            # plain manual leg cascade would skip this and let
            # :meth:`drive_native_failsafe` dispatch a clear PUT against
            # a disappeared deal (degrading/degraded), then block future
            # entries on an already-flat symbol; (b) cascade-cancels the
            # partial-bracket legs; (c) drops the entry intent + every
            # exit intent under this ``from_entry`` from
            # ``_active_intents`` / ``_order_mapping`` / ``_envelopes``;
            # (d) wipes the matching :attr:`Position.entry_orders` and
            # :attr:`Position.exit_orders` slots. Idempotent, so a
            # subsequent sync that sees the same flat snapshot is a no-op.
            for from_entry in from_entries_to_cascade:
                self._cleanup_position_tracking(
                    from_entry, cascade_reason='parent_flat_snapshot',
                )
            return
        triggering = self._partial_bracket_engine.on_price_tick(
            symbol=self._symbol,
            last_price=last_price,
            bid=None,
            ask=None,
            parent_snapshot=parent_snapshot,
        )
        for leg in triggering:
            self._dispatch_partial_bracket_close(leg)

    def _dispatch_partial_bracket_close(self, leg) -> None:
        """Synthesise + dispatch one partial-bracket :class:`CloseIntent`.

        Drives the :data:`LEG_STATE_TRIGGERING` leg through its
        terminal settling step. The synthesised ``pine_id`` carries
        the leg's identity coordinates so the dispatched intent
        cannot collide with the script's own :class:`CloseIntent`
        keys: ``__pyne_partial_trigger__{exit_pine_id}\0{from_entry}\0{leg_kind}``.
        The ``\0`` (NUL) delimiter mirrors :attr:`CloseIntent.intent_key`
        and is provably collision-free — Pine string literals (the only
        source of ``pine_id`` / ``from_entry``) cannot contain NUL.

        Outcome mapping (mirrors :meth:`_dispatch_new`'s exception
        contract):

        * Clean return AND the dispatch landed (``_order_mapping``
          populated for ``synth_pine_id``) →
          :meth:`SoftwarePartialBracketEngine.confirm_trigger_dispatched`
          flips the leg to :data:`LEG_STATE_TRIGGERED` and cascades the
          OCA siblings.
        * Clean return but the dispatch was parked by
          :meth:`_dispatch_new` (it swallows
          :class:`OrderDispositionUnknownError` and routes it to
          :meth:`_park_pending` without re-raising) →
          :meth:`SoftwarePartialBracketEngine.mark_trigger_dispatch_unknown`
          re-arms the leg so the next price tick re-evaluates against
          the live snapshot; the parked close still resolves through
          :meth:`_verify_pending_dispatches` if it actually landed.
        * :class:`OrderSkippedByPlugin` →
          :meth:`SoftwarePartialBracketEngine.mark_trigger_dispatch_failed`
          re-arms the leg with the skip reason on the audit row.
        * :class:`ExchangeOrderRejectedError` (including subclasses such
          as :class:`InsufficientMarginError`) →
          :meth:`SoftwarePartialBracketEngine.mark_trigger_dispatch_failed`
          re-arms the leg with the broker reject reason. Without this
          branch the generic ``Exception`` re-raise in
          :meth:`_dispatch_new` would propagate out of the trigger loop,
          aborting the current sync while the leg stays in
          :data:`LEG_STATE_TRIGGERING` and OCA siblings remain reserved.
        * :class:`BrokerManualInterventionError` propagates after
          marking the leg failed — the engine is going to halt.
        """
        synth_pine_id = (
            f"__pyne_partial_trigger__{leg.pine_id}\0"
            f"{leg.from_entry}\0{leg.leg_kind}"
        )
        close_intent = CloseIntent(
            pine_id=synth_pine_id,
            symbol=leg.symbol,
            side=leg.side,
            qty=leg.qty,
            immediately=True,
            comment=(
                f"engine-trigger partial bracket "
                f"({leg.leg_kind} for {leg.pine_id!r} "
                f"from_entry={leg.from_entry!r})"
            ),
        )
        try:
            self._dispatch_new(close_intent)
        except OrderSkippedByPlugin as e:
            # ``_dispatch_new`` built (and persisted via
            # :class:`BrokerStore`) the envelope for ``synth_pine_id`` before
            # the plugin declined. The dispatch never reached the broker, so
            # the COID anchor cannot be the dedup key for a real in-flight
            # order — leaving it in :attr:`_envelopes` would let a later
            # re-trigger for the same ``(pine_id, from_entry, leg_kind)``
            # rebuild the SAME ``client_order_id`` from a stale
            # ``bar_ts_ms`` / ``retry_seq`` anchor (and the persisted row
            # would survive a restart for the same collision). Drop the
            # synth envelope in lockstep with the failed settling so the
            # next dispatch starts with a fresh anchor.
            self._order_mapping.pop(synth_pine_id, None)
            self._drop_envelope(synth_pine_id)
            self._partial_bracket_engine.mark_trigger_dispatch_failed(
                leg.key, reason=f'plugin_skipped:{e.reason}',
            )
            return
        except BrokerManualInterventionError:
            self._partial_bracket_engine.mark_trigger_dispatch_failed(
                leg.key, reason='broker_manual_intervention',
            )
            raise
        except ExchangeOrderRejectedError as e:
            # The broker rejected the synthetic close (min-size, market-rule,
            # insufficient-margin, ...). ``_dispatch_new`` re-raises through
            # its generic ``Exception`` path, so without this branch the leg
            # would stay in :data:`LEG_STATE_TRIGGERING` and OCA siblings
            # would remain reserved while the current sync aborts. Settle
            # the leg as failed and let the next price tick re-arm against
            # a fresh parent snapshot. Drop the synth envelope for the same
            # reason as the ``OrderSkippedByPlugin`` branch — the broker
            # rejected the dispatch, so no live order owns the COID anchor;
            # a later re-trigger must mint a fresh one instead of reusing
            # the stale persisted anchor (and surviving a restart with it).
            self._order_mapping.pop(synth_pine_id, None)
            self._drop_envelope(synth_pine_id)
            self._partial_bracket_engine.mark_trigger_dispatch_failed(
                leg.key, reason=f'broker_rejected:{type(e).__name__}',
            )
            return
        except Exception as e:
            # Any other dispatch-side failure (e.g.
            # :class:`ExchangeConnectionError` raised by the broker plugin and
            # re-raised through ``_dispatch_new``'s generic ``Exception``
            # path) leaves the leg stuck in :data:`LEG_STATE_TRIGGERING` and
            # reserves its OCA group as in-flight. Subsequent ticks then skip
            # this leg and its siblings, so the bracket never retries until a
            # restart replays the state machine. Settle the leg as unknown
            # (the order may or may not have reached the exchange — the same
            # semantics as :class:`OrderDispositionUnknownError`) so the next
            # price tick re-evaluates against a fresh parent snapshot; the
            # leg's :meth:`_safety_check` caps against the live parent size
            # and prevents over-closing if the failed dispatch had in fact
            # landed. Drop the synth envelope so a re-trigger mints a fresh
            # idempotency anchor instead of reusing the stale one (and
            # surviving a restart with it).
            self._order_mapping.pop(synth_pine_id, None)
            self._drop_envelope(synth_pine_id)
            self._partial_bracket_engine.mark_trigger_dispatch_unknown(
                leg.key, reason=f'dispatch_failed:{type(e).__name__}',
            )
            raise
        # ``_dispatch_new`` catches :class:`OrderDispositionUnknownError`
        # internally and routes it to :meth:`_park_pending` without
        # re-raising. On the clean-return path the dispatch can therefore
        # be in one of two states: landed (``_order_mapping`` populated)
        # or parked (no ``_order_mapping`` entry, COID stashed in
        # ``_pending_verification``). Confirming the trigger on the
        # parked path would evict the leg + OCA-cancel siblings even
        # though the close was never acknowledged, leaving the parent
        # unprotected if the parked close eventually resolves
        # ``'rejected'``. Detect the park and re-arm the leg instead so
        # the next price tick re-evaluates and ``_verify_pending_dispatches``
        # can still reconcile the in-flight order if it did land.
        if synth_pine_id not in self._order_mapping:
            self._partial_bracket_engine.mark_trigger_dispatch_unknown(
                leg.key, reason='dispatch_parked',
            )
            return
        self._partial_bracket_engine.confirm_trigger_dispatched(
            leg.key, close_pine_id=synth_pine_id,
        )
        # Retire the synthetic close identity now that the leg owns the
        # dispatch. ``_order_mapping`` / ``_envelopes`` keyed on
        # ``synth_pine_id`` are not consulted by any later engine path
        # (the leg state machine tracks the close from here), but leaving
        # them populated would let :meth:`_build_envelope` recycle the
        # frozen ``client_order_id`` on the next trigger of the same
        # ``(pine_id, from_entry, leg_kind)`` — and the persisted
        # :class:`BrokerStore` rows would survive restart and collide
        # with a fresh close command for the same logical leg. Drop both
        # the live mapping and the persisted envelope/pending rows in
        # lockstep via :meth:`_drop_envelope` (which also calls
        # :meth:`storage.RunContext.record_complete`), and clear the
        # ``_order_mapping`` entry so a stray ``cancelled`` /
        # ``rejected`` event from the broker is treated as an external
        # observation rather than re-targeting the retired leg.
        self._order_mapping.pop(synth_pine_id, None)
        self._drop_envelope(synth_pine_id)
        # Retire the original partial ``ExitIntent`` once every active leg
        # under its ``intent_key`` is gone. ``confirm_trigger_dispatched``
        # flips the firing leg to :data:`LEG_STATE_TRIGGERED` and cascades
        # the OCA siblings to :data:`LEG_STATE_CASCADED_CANCEL`; for a
        # single-leg bracket (or any bracket whose siblings were already
        # evicted) no active leg remains for ``leg.intent_key``. Leaving
        # the parked ``ExitIntent`` in ``_active_intents`` / ``_order_mapping``
        # would let a subsequent script-side cancel or modify on the same
        # ``(pine_id, from_entry)`` slip past the engine-internal short-
        # circuit (gated on :meth:`has_active_legs_for_intent`) and reach
        # :meth:`_broker.execute_cancel` / ``execute_modify`` for an order
        # that was never submitted — the ``_order_mapping`` entry holds
        # the leg COIDs, not a real broker order id. Also drop the Pine
        # ``exit_orders`` slot so the next bar's intent-builder treats the
        # ``strategy.exit`` call as already-submitted (mirroring the
        # broker-FILL path :meth:`_remove_pine_order_for_intent`); without
        # that, an unchanged re-emission would re-arm a fresh bracket
        # against the now-reduced parent quantity.
        if not self._partial_bracket_engine.has_active_legs_for_intent(
                leg.intent_key,
        ):
            self._active_intents.pop(leg.intent_key, None)
            self._order_mapping.pop(leg.intent_key, None)
            self._drop_envelope(leg.intent_key)
            exit_orders = getattr(self._position, 'exit_orders', None)
            if exit_orders is not None:
                exit_orders.pop((leg.pine_id, leg.from_entry), None)

    def _record_halt(self, error: BrokerManualInterventionError) -> None:
        """Record a manual-intervention halt and emit the observability event.

        Idempotent — the first call latches the halt state and emits one
        :class:`ManualInterventionRequiredEvent`; subsequent calls (e.g. from
        a second dispatch path that also raised) are no-ops. After this the
        engine's :meth:`sync` returns early on every invocation until the
        strategy restarts.
        """
        if self._halted:
            return
        self._halted = True
        self._halted_reason = error.reason
        self._halted_intent_key = error.intent_key
        self._halted_context = dict(error.context)
        _blog_error(
            "sync engine halted by BrokerManualInterventionError: %s "
            "(intent_key=%s, context=%r)",
            error.reason, error.intent_key, error.context,
        )
        self._emit_broker_event(ManualInterventionRequiredEvent(
            reason=error.reason,
            intent_key=error.intent_key,
            context=dict(error.context),
        ))

    # === Tick resolution ===

    def _resolve_ticks(self, intent: Intent) -> Intent:
        if not isinstance(intent, ExitIntent) or not intent.has_unresolved_ticks:
            return intent
        entry_price, entry_sign = self._find_entry_reference(intent.from_entry)
        if entry_price is None:
            return intent
        return self._ticks_to_prices(intent, entry_price, entry_sign)

    def _find_entry_reference(
            self, from_entry: str,
    ) -> tuple[float | None, float]:
        for trade in self._position.open_trades:
            if trade.entry_id == from_entry:
                return trade.entry_price, trade.sign
        return None, 0.0

    def _ticks_to_prices(
            self, intent: ExitIntent, entry_price: float, entry_sign: float,
    ) -> ExitIntent:
        tp_price = intent.tp_price
        sl_price = intent.sl_price
        trail_price = intent.trail_price
        if intent.profit_ticks is not None:
            tp_price = entry_price + entry_sign * intent.profit_ticks * self._mintick
        if intent.loss_ticks is not None:
            sl_price = entry_price - entry_sign * intent.loss_ticks * self._mintick
        if intent.trail_points_ticks is not None:
            trail_price = (
                entry_price + entry_sign * intent.trail_points_ticks * self._mintick
            )
        return dataclasses.replace(
            intent,
            tp_price=tp_price,
            sl_price=sl_price,
            trail_price=trail_price,
            profit_ticks=None,
            loss_ticks=None,
            trail_points_ticks=None,
        )

    # === Interceptor chain ===

    def _apply_interceptors(self, intents: list[Intent]) -> list[Intent]:
        if not self._interceptors:
            return intents
        out: list[Intent] = []
        for intent in intents:
            current = intent
            rejected = False
            for fn in self._interceptors:
                result = fn(current)
                if result.rejected:
                    rejected = True
                    _log.info(
                        "intent %s rejected by interceptor: %s",
                        current.intent_key, result.reject_reason,
                    )
                    break
                current = self._apply_modifications(current, result)
            if not rejected:
                out.append(current)
        return out

    @staticmethod
    def _apply_modifications(
            intent: Intent, result: InterceptorResult,
    ) -> Intent:
        mods: dict[str, Any] = {}
        if result.modified_qty is not None:
            mods['qty'] = result.modified_qty
        if result.modified_limit is not None:
            if isinstance(intent, ExitIntent):
                mods['tp_price'] = result.modified_limit
            elif isinstance(intent, EntryIntent):
                mods['limit'] = result.modified_limit
        if result.modified_stop is not None:
            if isinstance(intent, ExitIntent):
                mods['sl_price'] = result.modified_stop
            elif isinstance(intent, EntryIntent):
                mods['stop'] = result.modified_stop
        return dataclasses.replace(intent, **mods) if mods else intent

    # === Diff + dispatch ===

    def _diff_and_dispatch(self, intents: list[Intent]) -> None:
        new_map: dict[str, Intent] = {i.intent_key: i for i in intents}

        # Pine semantic: when an entry intent fails to dispatch in this same
        # sync (e.g. plugin reports qty below venue minimum), a bracket exit
        # that references it via ``from_entry`` is a silent no-op — same as
        # Pine's own simulator returning at the missing-entry check
        # (``strategy/__init__.py``). We only short-circuit brackets whose
        # parent we just observed skipping; brackets that reference an
        # already-filled position (no entry intent in this sync) keep the
        # existing dispatch behaviour.
        skipped_entry_ids_this_sync: set[str] = set()

        # Cross-restart cleanup for orphan engine-trigger partial legs:
        # :meth:`SoftwarePartialBracketEngine.restart_replay` rebuilds
        # the leg ledger from persisted rows, but ``_active_intents``
        # and ``_order_mapping`` start empty after a restart. The
        # post-restart adoption branch below only re-anchors legs
        # whose ``intent_key`` is still emitted by Pine THIS sync. If
        # the strategy no longer issues that exit (script cancelled
        # the bracket between the crash and the restart, or simply
        # took a different branch), the legs remain armed in the
        # state-machine ledger forever — neither the cancellation diff
        # (it iterates ``_active_intents``) nor the new-intent diff
        # (it iterates ``new_map``) covers this case, and once the
        # price-tick WATCH wiring lands a script-cancelled bracket
        # could still fire. Cancel any orphan leg whose intent_key
        # appears in neither set BEFORE the diff loops run.
        if (self._partial_qty_bracket_exit_mode
                is CapabilityLevel.SOFTWARE):
            orphan_intent_keys: set[str] = set()
            for leg in self._partial_bracket_engine.iter_legs():
                ikey = leg.intent_key
                if ikey in self._active_intents:
                    continue
                if ikey in new_map:
                    continue
                orphan_intent_keys.add(ikey)
            for ikey in orphan_intent_keys:
                _blog_info(
                    "cancelling orphan engine-trigger partial legs for "
                    "intent %r (not emitted by Pine after restart)", ikey,
                )
                self._partial_bracket_engine.cancel_legs_for_intent(
                    ikey, reason='orphan_after_restart',
                )
                self._order_mapping.pop(ikey, None)
                self._drop_envelope(ikey)

            # Same-sync §12 #4 coexistence preflight: if THIS sync's
            # ``new_map`` contains both a partial-qty bracket and a
            # native whole-row exit on the same ``from_entry`` (distinct
            # ``intent_key`` values), the dispatch order would decide
            # which side wins — if the partial is iterated first, its
            # leg rows are persisted and registered before the later
            # whole-row dispatch raises, leaving the engine ledger in an
            # invalid combined-bracket state. Detect the conflict here
            # (before any leg row exists) and refuse BOTH conflicting
            # intents this sync; the script must drop one or the other
            # for the dispatch to proceed.
            # Multiple partial exits can legitimately share the same
            # ``from_entry`` (Pine scripts often issue several
            # ``strategy.exit(id_n, from_entry=parent)`` calls). Collect
            # every partial per ``from_entry`` so we cancel the entire
            # set when a conflicting whole-row exit appears — keeping
            # only the first via ``setdefault`` would let the remaining
            # partials slip through the preflight and violate the §12
            # #4 coexistence invariant.
            partials_by_from_entry: dict[str, list[ExitIntent]] = {}
            native_by_from_entry: dict[str, ExitIntent] = {}
            for cand in new_map.values():
                if not isinstance(cand, ExitIntent):
                    continue
                if cand.from_entry is None:
                    continue
                if cand.is_partial_qty_bracket:
                    partials_by_from_entry.setdefault(
                        cand.from_entry, [],
                    ).append(cand)
                else:
                    native_by_from_entry.setdefault(cand.from_entry, cand)
            conflicting_keys: set[str] = set()
            for from_entry, partial_intents in partials_by_from_entry.items():
                native_intent = native_by_from_entry.get(from_entry)
                if native_intent is None:
                    continue
                partial_keys = [p.intent_key for p in partial_intents]
                _blog_warning(
                    "engine-trigger partial brackets %r conflict with "
                    "native whole-row exit %r on the same from_entry "
                    "%r in this sync; refusing the conflicting "
                    "dispatches — the script must cancel one side "
                    "before the other can attach",
                    partial_keys,
                    native_intent.intent_key,
                    from_entry,
                )
                for p in partial_intents:
                    conflicting_keys.add(p.intent_key)
                conflicting_keys.add(native_intent.intent_key)
            # Only pop conflict keys that are NOT already in
            # ``_active_intents``. The cancellation loop below treats
            # ``key in _active_intents and key not in new_map`` as a
            # strategy-side cancel and dispatches ``_dispatch_cancel`` —
            # popping an already-active bracket here would tear down
            # the live broker-side protection the script never asked
            # to cancel. The intent of the preflight is to refuse the
            # NEW conflicting dispatch, not to flatten existing state.
            for ikey in conflicting_keys:
                if ikey in self._active_intents:
                    continue
                new_map.pop(ikey, None)

        for key in list(self._active_intents):
            if key not in new_map:
                if key not in self._active_intents:
                    # A prior iteration in this loop cancelled the parent
                    # EntryIntent and :meth:`_dispatch_cancel_strict`
                    # eagerly dropped its paired engine-trigger partial
                    # ``ExitIntent`` slot (see the cleanup at the end of
                    # that method). The exit is engine-internal — no
                    # broker-side order exists to cancel — so simply
                    # skip; the legs were retired in the same cleanup.
                    continue
                if key in self._pending_defensive_close:
                    # Parent entry has an in-flight defensive close —
                    # the marker deliberately keeps the entry in
                    # ``_active_intents`` so :meth:`reconcile` cannot
                    # treat the soon-to-be-flat broker snapshot as an
                    # external flatten. The strategy dropping the
                    # entry from its intent list this bar (e.g. after
                    # the script reverted the entry condition) MUST
                    # NOT trigger ``_dispatch_cancel`` — that would
                    # send a cancel for a position already being
                    # flattened by the synthetic close, fighting the
                    # marker's purpose and potentially racing the
                    # close FILL. Leave the intent in place; the
                    # close-fill settlement path
                    # (:meth:`_route_defensive_close_fill` →
                    # :meth:`_cleanup_position_tracking`) is the only
                    # legitimate remover while the marker is armed.
                    continue
                old = self._active_intents[key]
                if (isinstance(old, ExitIntent)
                        and old.from_entry in self._pending_defensive_close):
                    # Child exit of a parent entry whose defensive close
                    # is still in flight. ``ExitIntent.intent_key`` is
                    # ``f"{pine_id}\0{from_entry}"`` so the parent-key
                    # guard above does not match — but the same race
                    # logic applies: ``_handle_bracket_attach_after_fill_reject``
                    # already cancelled this exit's broker refs via
                    # :meth:`_cancel_bracket_reject_residuals`, the
                    # close-fill settlement path re-runs that cleanup,
                    # and dispatching another cancel here would either
                    # target already-cancelled refs or race the close
                    # FILL. Freeze the exit until the marker settles;
                    # :meth:`_cleanup_position_tracking` drops the
                    # entry's bracket intents at close-fill time.
                    continue
                if (isinstance(old, ExitIntent)
                        and old.is_partial_qty_bracket
                        and old.from_entry is not None
                        and old.from_entry in self._cancel_disposition_pending):
                    # Child engine-trigger partial-bracket ``ExitIntent``
                    # whose parent is cancel-tentative. Both
                    # :meth:`_mark_intent_cancel_disposition_pending` (parent
                    # tentative entry) and the late-fill / cancel-confirmed
                    # routing flip the parent's child legs to
                    # ``LEG_STATE_CANCEL_TENTATIVE``, which is NOT in
                    # :data:`LEG_STATE_ACTIVE`. The engine-internal cancel
                    # shortcut in :meth:`_dispatch_cancel_strict` keys off
                    # ``has_active_legs_for_intent`` and would therefore miss
                    # those rows and fall through to ``execute_cancel`` on a
                    # broker order that was never submitted. Defer the child
                    # cancel until the parent's disposition resolves —
                    # :meth:`_clear_intent_cancel_disposition_pending`
                    # already retires every child partial-exit slot under
                    # the resolved parent (CONFIRMED → child dropped;
                    # ALREADY_FILLED → legs restored alongside the parent).
                    _blog_warning(
                        "child partial-bracket exit %r cancel deferred — "
                        "parent %r is cancel-tentative; next reconcile "
                        "will resolve the parent and retire the child",
                        key, old.from_entry,
                    )
                    continue
                # Cross-key native-to-engine partial conversion: the script
                # removed a native whole-row exit AND added a partial-qty
                # bracket on the SAME ``from_entry`` under a DIFFERENT
                # ``strategy.exit`` id in this sync. The same-key conversion
                # in :meth:`_dispatch_modify` uses
                # :meth:`_dispatch_cancel_strict` so a timed-out cancel
                # defers the new dispatch; the default cancel path here
                # would instead swallow ``OrderDispositionUnknownError``,
                # drop the mapping/envelope, and let the new-intents loop
                # below arm engine-trigger legs while the broker may still
                # hold the whole-row bracket live — violating the §12 #4
                # coexistence invariant. Mirror the strict-cancel-with-
                # retry semantics here: on a timed-out cancel, restore the
                # OLD intent to ``_active_intents`` AND drop the new
                # partial from ``new_map`` so the next sync re-diffs and
                # retries the strict cancel (idempotent via the
                # deterministic ``client_order_id``).
                conflicting_partial_keys: list[str] = []
                if (isinstance(old, ExitIntent)
                        and not old.is_partial_qty_bracket
                        and old.from_entry is not None
                        and self._partial_qty_bracket_exit_mode
                        is CapabilityLevel.SOFTWARE):
                    conflicting_partial_keys = [
                        cand_key
                        for cand_key, cand in new_map.items()
                        if (
                            isinstance(cand, ExitIntent)
                            and cand.is_partial_qty_bracket
                            and cand.from_entry == old.from_entry
                        )
                    ]
                needs_strict_cancel = bool(conflicting_partial_keys)
                self._active_intents.pop(key)
                # A still-parked modify for this key (Pine cancels while
                # the previous amend's resolution is pending) is being
                # superseded — its rollback snapshot would target an
                # intent the strategy no longer wants. Drop the snapshot
                # so a late ``'rejected'`` resolution does not resurrect
                # a cancelled key into ``_active_intents``.
                self._modify_old_intents.pop(key, None)
                if needs_strict_cancel:
                    try:
                        self._dispatch_cancel_strict(old)
                    except OrderDispositionUnknownError as e:
                        _blog_warning(
                            "native-to-engine partial conversion deferred: "
                            "strict cancel of %s timed out (%s); restoring "
                            "old intent and dropping new partial bracket(s) "
                            "%r so the next sync retries the cancel before "
                            "engine-trigger legs are armed",
                            old, e, conflicting_partial_keys,
                        )
                        self._active_intents[key] = old
                        for partial_key in conflicting_partial_keys:
                            new_map.pop(partial_key, None)
                        continue
                else:
                    self._dispatch_cancel(old)

        for key, intent in new_map.items():
            if key not in self._active_intents:
                # Refuse-and-defer guard for cancel-tentative parents.
                # The shadow map is keyed by the parent
                # ``EntryIntent.intent_key``; an ``EntryIntent`` reissue
                # matches by ``key``, an ``ExitIntent`` on the same
                # parent matches by ``intent.from_entry``, and a
                # ``CloseIntent`` shares ``intent_key == pine_id`` with
                # the parent ``EntryIntent`` so it matches by ``key``.
                # While the disposition is unresolved we MUST NOT
                # dispatch a fresh intent (a parallel broker state would
                # double-life the parent) and MUST NOT adopt the
                # retained mapping (it belongs to the prior dispatch
                # under cancel-retry). The ``reconcile()`` cancel-retry-
                # loop will resolve the disposition within stale-grace;
                # until then the script's reissue defers and the audit
                # event records each skip.
                tentative_parent_key: str | None = None
                if isinstance(intent, (EntryIntent, CloseIntent)):
                    if key in self._cancel_disposition_pending:
                        tentative_parent_key = key
                elif (isinstance(intent, ExitIntent)
                        and intent.from_entry is not None
                        and intent.from_entry
                        in self._cancel_disposition_pending):
                    tentative_parent_key = intent.from_entry
                if tentative_parent_key is not None:
                    meta = self._cancel_disposition_pending[tentative_parent_key]
                    _blog_warning(
                        "intent %r deferred — prior cancel disposition on "
                        "parent %r is unresolved (since_ts_ms=%s); next "
                        "reconcile will attempt to resolve",
                        key, tentative_parent_key, meta.since_ts_ms,
                    )
                    self._emit_broker_event(
                        EntryDeferredCancelDispositionPendingEvent(
                            intent_key=key,
                            pine_id=intent.pine_id,
                            symbol=self._symbol,
                            since_ts_ms=meta.since_ts_ms,
                        ),
                    )
                    continue
                if key in self._order_mapping and (
                        not isinstance(intent, CloseIntent)
                        or key in self._recovered_close_anchor_keys):
                    # Cross-restart adoption: the persisted state recovered an
                    # exchange-side order for this intent (via
                    # _verify_pending_dispatches). Re-dispatching here would
                    # duplicate the order — instead, adopt the existing
                    # mapping and pin the envelope from the persisted anchor
                    # so subsequent modifies emit the same client_order_id.
                    #
                    # ``CloseIntent`` is excluded by default: its
                    # ``intent_key`` equals the parent's ``pine_id``, so a
                    # stale parent-entry mapping retained after an
                    # ``ALREADY_FILLED`` cancel-tentative resolution (see
                    # :meth:`_clear_intent_cancel_disposition_pending`)
                    # collides with the close's key. Adoption would silently
                    # pin the close to the filled parent's mapping without
                    # dispatching the market close, leaving the position open
                    # while the script believes it asked to flatten. The
                    # exception is :attr:`_recovered_close_anchor_keys`: when
                    # the mapping was seeded from a parked
                    # ``execute_close`` (:meth:`_verify_pending_dispatches`
                    # observed a :data:`KIND_CLOSE` COID), the mapping IS
                    # the close's broker order — adopting prevents the
                    # script's re-emitted ``CloseIntent`` from emitting a
                    # second market close that races (or doubles) the
                    # recovered one.
                    self._build_envelope(intent)
                    self._active_intents[key] = intent
                    if isinstance(intent, CloseIntent):
                        self._recovered_close_anchor_keys.discard(key)
                elif (isinstance(intent, ExitIntent)
                        and intent.is_partial_qty_bracket
                        and self._partial_qty_bracket_exit_mode
                        is CapabilityLevel.SOFTWARE
                        and self._partial_bracket_engine.has_active_legs_for_intent(
                            intent.intent_key,
                        )):
                    # Cross-restart adoption for engine-trigger partial brackets.
                    # ``restart_replay()`` repopulates the in-memory leg ledger
                    # from persisted rows, but ``_active_intents`` and
                    # ``_order_mapping`` start empty. Without this branch the
                    # first post-restart sync would re-enter
                    # :meth:`_dispatch_engine_trigger_partial_bracket`, where
                    # the duplicate guard raises on the still-tracked legs.
                    # Adopt the live legs instead: build the envelope, pin the
                    # intent, and restore ``_order_mapping`` from the leg
                    # ``coid``s so subsequent diffs treat the dispatch as live.
                    #
                    # Verify the persisted legs reference the CURRENT parent
                    # entry dispatch before adopting. If the previous parent
                    # closed while the bot was down and the script reuses the
                    # same ``strategy.exit(id, from_entry)`` for a new entry,
                    # the legs' ``parent_entry_dispatch_ref`` will not match
                    # the freshly rebuilt parent envelope's ``KIND_ENTRY``
                    # coid — adopting them would attach a stale partial close
                    # to the wrong position. Cancel the orphan legs and let
                    # the normal dispatch path arm a fresh bracket.
                    #
                    # Post-restart subtlety: ``_envelopes`` starts empty until
                    # an intent for the parent key is rebuilt this sync, but
                    # the common recovery path re-emits only the bracket
                    # ``ExitIntent`` for an already-open parent (the entry
                    # filled before the crash and the script no longer emits
                    # it). Fall back to ``_persisted_envelope_anchors`` —
                    # the anchor preserves ``bar_ts_ms`` / ``retry_seq`` and
                    # ``self._run_tag`` is deterministic, so the rebuilt
                    # ``KIND_ENTRY`` coid matches what the previous run
                    # stored on each leg. Without this fallback every leg
                    # would be classified stale even when the parent is the
                    # original one.
                    expected_parent_ref: str | None = None
                    if intent.from_entry is not None:
                        expected_parent_ref = self._resolve_parent_opening_ref(
                            intent.from_entry,
                        )
                    legs_for_intent = [
                        leg for leg
                        in self._partial_bracket_engine.iter_legs()
                        if leg.intent_key == intent.intent_key
                    ]
                    stale_parent = (
                        expected_parent_ref is None
                        or any(
                            leg.parent_entry_dispatch_ref != expected_parent_ref
                            for leg in legs_for_intent
                        )
                    )
                    if stale_parent:
                        # §2.6.7 preflight: if the NEW parent's native fail-safe
                        # is currently blocking new partial brackets
                        # (DEGRADING / DEGRADED / UNKNOWN owner), cancelling
                        # the replayed legs first would strip the only
                        # protection on the existing position while
                        # ``_dispatch_new`` raises :class:`OrderSkippedByPlugin`
                        # and arms nothing. Leave the replayed legs intact
                        # this sync; next sync re-enters this adoption branch
                        # and retries once the manager flips out of the
                        # blocking state (snapshot confirm / user reset).
                        if (expected_parent_ref is not None
                                and self._native_failsafe_manager
                                .is_new_partial_bracket_blocked(
                                    parent_entry_dispatch_ref=expected_parent_ref,
                                )):
                            _blog_warning(
                                "engine-trigger partial bracket adoption "
                                "deferred for intent %r: native fail-safe "
                                "for new parent %r is blocking; keeping "
                                "stale legs as residual protection until "
                                "the state recovers",
                                intent.intent_key,
                                expected_parent_ref,
                            )
                            continue
                        _blog_info(
                            "cancelling stale engine-trigger partial legs "
                            "for intent %r: parent dispatch ref mismatch "
                            "(expected %r) — script re-used from_entry %r "
                            "for a new parent",
                            intent.intent_key,
                            expected_parent_ref,
                            intent.from_entry,
                        )
                        self._partial_bracket_engine.cancel_legs_for_intent(
                            intent.intent_key,
                            reason='stale_parent_after_restart',
                        )
                        self._order_mapping.pop(key, None)
                        self._drop_envelope(key)
                        # Fall through to the normal dispatch path so a fresh
                        # bracket is armed against the new parent.
                        try:
                            self._dispatch_new(intent)
                        except OrderSkippedByPlugin as e:
                            _blog_warning("%s", e)
                            continue
                        self._active_intents[key] = intent
                        continue
                    # Completeness check: ``_dispatch_engine_trigger_partial_bracket``
                    # writes one leg row per declared leg in a loop. If the
                    # previous run crashed mid-loop, only some leg rows are
                    # durable — restart_replay() loads whatever exists, and
                    # adopting the partial set here would pin the intent with
                    # a missing sibling that no path ever re-creates. Compare
                    # the persisted ``leg_kind`` set against the kinds the
                    # current intent would emit; on mismatch, cancel the
                    # orphan legs and re-dispatch a fresh bracket so all
                    # declared legs are armed.
                    expected_specs = list(
                        self._enumerate_engine_trigger_legs(intent)
                    )
                    expected_leg_kinds = {spec.leg_kind for spec in expected_specs}
                    persisted_leg_kinds = {
                        leg.leg_kind for leg in legs_for_intent
                    }
                    if expected_leg_kinds != persisted_leg_kinds:
                        # §2.6.7 preflight: a blocking native fail-safe state
                        # (DEGRADING / DEGRADED / UNKNOWN owner) would make
                        # ``_dispatch_new`` raise :class:`OrderSkippedByPlugin`
                        # after the cancel — leaving the parent without any
                        # software protection. Keep the partial replayed
                        # leg set in place this sync; the next sync re-enters
                        # this branch once the state recovers and the fresh
                        # bracket can be armed cleanly.
                        if (expected_parent_ref is not None
                                and self._native_failsafe_manager
                                .is_new_partial_bracket_blocked(
                                    parent_entry_dispatch_ref=expected_parent_ref,
                                )):
                            _blog_warning(
                                "engine-trigger partial bracket adoption "
                                "deferred for intent %r: persisted leg set "
                                "%r differs from expected %r but native "
                                "fail-safe for parent %r is blocking; "
                                "keeping replayed legs as residual "
                                "protection until the state recovers",
                                intent.intent_key,
                                sorted(persisted_leg_kinds),
                                sorted(expected_leg_kinds),
                                expected_parent_ref,
                            )
                            continue
                        _blog_info(
                            "cancelling incomplete engine-trigger partial "
                            "legs for intent %r: persisted kinds %r do not "
                            "match expected %r — likely a crash between "
                            "leg-row writes; re-dispatching fresh bracket",
                            intent.intent_key,
                            sorted(persisted_leg_kinds),
                            sorted(expected_leg_kinds),
                        )
                        self._partial_bracket_engine.cancel_legs_for_intent(
                            intent.intent_key,
                            reason='incomplete_leg_set_after_restart',
                        )
                        self._order_mapping.pop(key, None)
                        self._drop_envelope(key)
                        try:
                            self._dispatch_new(intent)
                        except OrderSkippedByPlugin as e:
                            _blog_warning("%s", e)
                            continue
                        self._active_intents[key] = intent
                        continue
                    # Full-spec parity check: ``leg_kind`` parity alone does
                    # not catch a Pine script that re-emits the same
                    # ``intent_key`` with retuned TP/SL/trail levels or a
                    # different ``qty``. Without this guard the persisted
                    # legs (stale trigger levels, stale ``intent_partial_qty``)
                    # would be adopted while ``_active_intents[key] = intent``
                    # pins the NEW intent, so later diffs see no change and
                    # the engine permanently watches the stale parameters.
                    # Compare each persisted leg against its matching spec
                    # by leg kind:
                    # - TP/SL legs: ``trigger_level`` and ``trigger_offset``
                    #   are static once dispatched, so the persisted values
                    #   must equal the spec's.
                    # - Trail legs: the dynamic stop (``trigger_level``)
                    #   evolves at runtime, but ``trigger_offset`` (trail
                    #   distance from high-water) and ``trail_activation_offset``
                    #   (pre-fill activation distance) are stable across the
                    #   leg's lifetime — compare those. ``trail_activation_level``
                    #   is also compared, but only while the leg is still
                    #   pre-activation (``leg.trail_activation_level`` is not
                    #   ``None``) and the spec carries an absolute activation
                    #   price (``spec.trail_activation_level`` is not ``None``):
                    #   that is the only way to detect a Pine script that
                    #   retunes ``trail_price`` to a different absolute level
                    #   while keeping qty / offset / activation_offset stable.
                    #   Once the trail has activated (the engine clears
                    #   ``trail_activation_level`` after the activation tick),
                    #   the field is dropped from the comparison so adoption
                    #   does not spuriously cancel an in-flight trail.
                    # In all cases, ``leg.intent_partial_qty`` must equal
                    # the current ``intent.qty``.
                    specs_by_kind = {
                        spec.leg_kind: spec for spec in expected_specs
                    }
                    spec_mismatch = False
                    for leg in legs_for_intent:
                        spec = specs_by_kind.get(leg.leg_kind)
                        if spec is None:  # pragma: no cover — kind-set parity ensured above
                            spec_mismatch = True
                            break
                        if leg.intent_partial_qty != intent.qty:
                            spec_mismatch = True
                            break
                        if leg.leg_kind == LEG_KIND_TRAIL_PARTIAL:
                            if (leg.trigger_offset != spec.trigger_offset
                                    or leg.trail_activation_offset
                                    != spec.trail_activation_offset):
                                spec_mismatch = True
                                break
                            if (leg.trail_activation_level is not None
                                    and spec.trail_activation_level is not None
                                    and leg.trail_activation_level
                                    != spec.trail_activation_level):
                                spec_mismatch = True
                                break
                        else:
                            if (leg.trigger_level != spec.trigger_level
                                    or leg.trigger_offset != spec.trigger_offset):
                                spec_mismatch = True
                                break
                    if spec_mismatch:
                        # §2.6.7 preflight: a blocking native fail-safe state
                        # (DEGRADING / DEGRADED / UNKNOWN owner) would make
                        # ``_dispatch_new`` raise :class:`OrderSkippedByPlugin`
                        # after the cancel — leaving the parent without any
                        # software protection. Keep the stale replayed legs
                        # in place this sync; the next sync re-enters this
                        # branch once the state recovers and the fresh
                        # bracket can be armed cleanly.
                        if (expected_parent_ref is not None
                                and self._native_failsafe_manager
                                .is_new_partial_bracket_blocked(
                                    parent_entry_dispatch_ref=expected_parent_ref,
                                )):
                            _blog_warning(
                                "engine-trigger partial bracket adoption "
                                "deferred for intent %r: replayed leg spec "
                                "differs from current intent but native "
                                "fail-safe for parent %r is blocking; "
                                "keeping stale legs as residual protection "
                                "until the state recovers",
                                intent.intent_key,
                                expected_parent_ref,
                            )
                            continue
                        _blog_info(
                            "cancelling stale engine-trigger partial legs "
                            "for intent %r: persisted leg spec differs from "
                            "current intent (qty/trigger_level/trigger_offset/"
                            "trail_activation_offset) — script retuned the "
                            "bracket across the restart boundary; "
                            "re-dispatching fresh bracket",
                            intent.intent_key,
                        )
                        self._partial_bracket_engine.cancel_legs_for_intent(
                            intent.intent_key,
                            reason='stale_leg_spec_after_restart',
                        )
                        self._order_mapping.pop(key, None)
                        self._drop_envelope(key)
                        try:
                            self._dispatch_new(intent)
                        except OrderSkippedByPlugin as e:
                            _blog_warning("%s", e)
                            continue
                        self._active_intents[key] = intent
                        continue
                    self._build_envelope(intent)
                    self._active_intents[key] = intent
                    self._order_mapping[key] = [
                        leg.coid for leg in legs_for_intent
                    ]
                else:
                    if (isinstance(intent, ExitIntent)
                            and intent.from_entry is not None
                            and (intent.from_entry in skipped_entry_ids_this_sync
                                 or intent.from_entry
                                 in self._defensively_closed_entries_this_sync
                                 or intent.from_entry
                                 in self._pending_defensive_close)):
                        # Parent entry was just skipped (plugin declined),
                        # defensively closed earlier in this same loop via the
                        # bracket-attach-after-fill reject recovery, or has an
                        # in-flight defensive close still awaiting its FILL
                        # (cross-sync — ``_defensively_closed_entries_this_sync``
                        # was cleared at end-of-last-sync but the marker remains).
                        # Re-dispatching a bracket against a parent we are about
                        # to flatten would either trigger another
                        # ``BracketAttachAfterFillRejectedError`` or attach
                        # protection to a position that is already on its way out.
                        # Re-evaluated next bar.
                        continue
                    if (isinstance(intent, EntryIntent)
                            and (intent.pine_id
                                 in self._pending_defensive_close
                                 or self._replayed_defensive_close_entry_ids
                                 or any(
                                     m.fill_observed
                                     for m in self._pending_defensive_close.values()
                                 ))):
                        # Cross-restart scenario: process restarted with an
                        # unsettled ``defensive_close_pending`` marker (e.g.
                        # the defensive close already filled in the previous
                        # instance but the engine crashed before writing the
                        # ``'defensive_close_filled'`` audit event), so
                        # :meth:`_replay_pending_defensive_closes` re-armed
                        # the marker. ``_active_intents`` is empty after the
                        # restart, and a flat-branch strategy may emit a
                        # fresh :class:`EntryIntent` for the same
                        # ``pine_id``. Dispatching it now would re-open the
                        # position the defensive close just flattened,
                        # recreating the duplicate-entry race the marker is
                        # designed to prevent. Wait until residual cleanup
                        # settles the marker (or the stale-grace timer
                        # halts) before allowing a new entry on this key.
                        #
                        # Cross-restart broader guard: while ANY replayed
                        # defensive-close marker is unsettled
                        # (:attr:`_replayed_defensive_close_entry_ids`
                        # non-empty), block new entries on EVERY pine_id —
                        # not just the marker's own. :meth:`reconcile`
                        # deliberately skips startup size adoption in that
                        # state (broker still reports the pre-close
                        # position; adopting it would later double-apply
                        # the FILL), so :attr:`_position.size` stays at
                        # the engine default 0.0 while the broker is
                        # still exposed. A flat-branch strategy can then
                        # emit a fresh :class:`EntryIntent` under a
                        # DIFFERENT ``pine_id`` (e.g. a Long marker is
                        # pending and the strategy now wants Short),
                        # which the original same-key guard above would
                        # not match — letting the new entry dispatch
                        # alongside the unprotected pre-close position.
                        # Block until the FILL routes and clears the
                        # replay set.
                        #
                        # Runtime post-FILL residual-cleanup guard: when
                        # :meth:`_route_defensive_close_fill` observed the
                        # terminal FILL and flipped ``fill_observed=True`` but
                        # the final residual-cancel call raised a transient
                        # :class:`ExchangeConnectionError` /
                        # :class:`OrderDispositionUnknownError`, the marker
                        # stays armed (no audit, no drop) so reconcile /
                        # stale-grace can retry the cancel. ``_position.size``
                        # is already reduced (``record_fill`` ran before the
                        # routing helper at the call site in :meth:`_route_event`),
                        # so a flat-branch strategy can now emit a fresh
                        # :class:`EntryIntent` under a DIFFERENT ``pine_id``.
                        # The same-key check above does not match, and this
                        # is not a startup replay so
                        # ``_replayed_defensive_close_entry_ids`` is empty
                        # — letting the new entry race the still-live
                        # residual TP/SL orders attached to the just-closed
                        # parent. Block on ANY marker with ``fill_observed``
                        # set until the retry cleans up.
                        continue
                    if (isinstance(intent, CloseIntent)
                            and (intent.pine_id
                                 in self._pending_defensive_close
                                 or (intent.pine_id == ""
                                     and self._pending_defensive_close)
                                 or self._replayed_defensive_close_entry_ids
                                 or any(
                                     m.fill_observed
                                     for m in self._pending_defensive_close.values()
                                 ))):
                        # Same cross-restart window as the EntryIntent guard
                        # above, but for a script-emitted
                        # ``strategy.close("Long")`` that lands as a fresh
                        # :class:`CloseIntent` while ``_active_intents`` is
                        # still empty after replay. ``CloseIntent.intent_key``
                        # is ``pine_id`` (same as :class:`EntryIntent`), so
                        # this key only reaches the modify-branch
                        # ``CloseIntent`` guard (see below) when the parent
                        # ``EntryIntent`` survived in ``_active_intents``.
                        # Post-restart the strategy may re-emit the close
                        # against a still-tracked position before the
                        # synthetic defensive close settles — without this
                        # guard the close would fall through to
                        # :meth:`_dispatch_new` and send a *second* market
                        # close that races (or doubles) the in-flight
                        # synthetic flatten. The marker's FILL-settlement
                        # path is the only legitimate flatten while a
                        # defensive close is pending; next bar re-evaluates
                        # from a clean slate once the marker clears.
                        #
                        # ``strategy.close_all()`` builds a :class:`CloseIntent`
                        # with ``pine_id == ""`` (see
                        # :func:`build_close_intent`) — the empty key never
                        # appears in :attr:`_pending_defensive_close` (keyed by
                        # the real entry pine_id), so a script-side close_all
                        # issued during the pending window would otherwise fall
                        # through to :meth:`_dispatch_new` and race the
                        # synthetic flatten. Block it whenever ANY defensive
                        # close is in flight; once every marker settles, the
                        # next bar dispatches close_all normally.
                        continue
                    try:
                        self._dispatch_new(intent)
                    except OrderSkippedByPlugin as e:
                        # Plugin declined (e.g. qty below venue minimum) OR
                        # the bracket-attach-after-fill recovery path raised
                        # ``reason="bracket_reject_defensive_close"`` — the
                        # latter has already populated
                        # ``_defensively_closed_entries_this_sync`` so any
                        # remaining same-entry intents handled later in this
                        # loop short-circuit at the guard above.
                        _blog_warning("%s", e)
                        if isinstance(intent, EntryIntent):
                            skipped_entry_ids_this_sync.add(intent.pine_id)
                        continue
                    self._active_intents[key] = intent
            elif intent != self._active_intents[key]:
                if (isinstance(intent, EntryIntent)
                        and intent.pine_id
                        in self._pending_defensive_close):
                    # Parent entry has an in-flight defensive close
                    # (see the cancel branch above for the full
                    # rationale). Re-emitting the same key with
                    # different parameters this bar would otherwise
                    # call ``_dispatch_modify`` and send a modify_entry
                    # for an order on a position we are actively
                    # flattening — either no-ops at the broker (the
                    # entry order may already be filled and resting
                    # has been cancelled by ``_cancel_bracket_reject_residuals``)
                    # or, worse, attempts to resize a phantom resting
                    # entry. Leave the active intent untouched; once
                    # the close FILL settles the marker, the next
                    # bar re-evaluates from a clean slate.
                    continue
                if (isinstance(intent, ExitIntent)
                        and intent.from_entry is not None
                        and intent.from_entry
                        in self._pending_defensive_close):
                    # Same-entry already-active bracket re-emitted with
                    # changed parameters (e.g. SL/TP retuned) while a
                    # defensive close is in flight for the parent. The
                    # new-key branch above already short-circuits *fresh*
                    # bracket dispatches for this case; without an
                    # equivalent guard here, an existing active
                    # ``ExitIntent`` (one bracket attached before the
                    # sibling-attach reject triggered the defensive close)
                    # would fall through to ``_dispatch_modify`` and amend
                    # or recreate broker-side protection against a
                    # position the engine is actively flattening. Keep
                    # the active intent untouched; once the close FILL
                    # settles, :meth:`_cleanup_position_tracking` removes
                    # this slot and the next bar starts clean.
                    continue
                if (isinstance(intent, ExitIntent)
                        and intent.from_entry is not None
                        and intent.from_entry
                        in self._cancel_disposition_pending):
                    # Parent is mid cancel-tentative: its engine-trigger
                    # partial legs were moved to ``LEG_STATE_CANCEL_TENTATIVE``
                    # by :meth:`_dispatch_cancel`, so
                    # :meth:`SoftwarePartialBracketEngine.has_active_legs_for_intent`
                    # returns ``False`` and :meth:`_dispatch_modify`'s
                    # engine-internal branch misses. A re-emitted same-key
                    # bracket with retuned parameters would then fall through
                    # to broker ``modify_exit`` for a bracket that was never
                    # submitted (engine-trigger legs are engine-internal —
                    # the broker has no order to amend). Defer the modify
                    # until the cancel disposition resolves; the next sync
                    # re-diffs from a coherent state (legs re-armed on
                    # ``ALREADY_FILLED``, or slot dropped on confirm-cancel).
                    meta = self._cancel_disposition_pending[intent.from_entry]
                    _blog_warning(
                        "intent %r modify deferred — parent %r is mid "
                        "cancel-tentative (since_ts_ms=%s); next reconcile "
                        "will attempt to resolve",
                        key, intent.from_entry, meta.since_ts_ms,
                    )
                    self._emit_broker_event(
                        EntryDeferredCancelDispositionPendingEvent(
                            intent_key=key,
                            pine_id=intent.pine_id,
                            symbol=self._symbol,
                            since_ts_ms=meta.since_ts_ms,
                        ),
                    )
                    continue
                if (isinstance(intent, CloseIntent)
                        and (intent.pine_id
                             in self._pending_defensive_close
                             or (intent.pine_id == ""
                                 and self._pending_defensive_close))):
                    # Script-emitted ``strategy.close("Long")`` while the
                    # engine's own synthetic defensive close for the same
                    # parent ``pine_id`` is still in flight. ``CloseIntent``
                    # shares ``intent_key == pine_id`` with the parent
                    # ``EntryIntent`` that the marker deliberately keeps
                    # in ``_active_intents`` (see the EntryIntent guard
                    # above), so the diff drops here as a key-collision
                    # modify. Without this guard the modify branch falls
                    # through to ``_dispatch_modify`` which routes
                    # ``(EntryIntent, CloseIntent)`` through cancel + new
                    # — sending a *second* market close that races (or
                    # doubles) the synthetic defensive close already on
                    # the wire. The marker's FILL-settlement path is the
                    # only legitimate flatten while the close is pending;
                    # next bar re-evaluates from a clean slate.
                    #
                    # The ``pine_id == ""`` branch covers
                    # ``strategy.close_all()`` re-emissions for which a
                    # prior close_all already populated ``_active_intents``
                    # with the empty key — without it the modify branch
                    # would send a second flatten that races the synthetic
                    # defensive close on the wire.
                    continue
                try:
                    self._dispatch_modify(self._active_intents[key], intent)
                except OrderSkippedByPlugin as e:
                    # The cancel+re-execute fallback inside _dispatch_modify
                    # cancelled the old order before the plugin declined the
                    # new one. The exchange now has nothing for this key, so
                    # drop it from active too.
                    _blog_warning("%s", e)
                    self._active_intents.pop(key, None)
                    continue
                except _PartialBracketModifyDeferred:
                    # Engine-trigger partial-bracket modify was deferred by the
                    # §2.6.7 fail-safe gate inside ``_dispatch_modify`` — the
                    # OLD legs are still armed and no replacement was
                    # dispatched. Keep ``_active_intents[key]`` pointing at the
                    # OLD intent so the next sync re-diffs and retries the
                    # modify once the fail-safe recovers; promoting the slot to
                    # ``intent`` here would make Pine == active and the retry
                    # would never run.
                    continue
                except OrderDispositionUnknownError:
                    # Native-to-engine partial-bracket conversion path
                    # surfaces a timed-out strict cancel here. Leaving the
                    # slot pointing at the OLD native intent forces the
                    # next sync to re-diff and retry the conversion via
                    # ``_dispatch_modify`` — the deterministic
                    # ``client_order_id`` makes the broker-side cancel
                    # idempotent. Promoting to ``intent`` (the new partial)
                    # instead would mask the unfinished work because the
                    # next diff would observe Pine == active and skip
                    # arming the engine legs.
                    continue
                self._active_intents[key] = intent
            # else: unchanged — skip

    def _build_envelope(self, intent: Intent) -> DispatchEnvelope:
        """Wrap an intent in a :class:`DispatchEnvelope`.

        The first envelope for a given ``intent_key`` is pinned on creation
        (bar_ts_ms, retry_seq frozen). Subsequent modifies re-use the same
        anchor so the ``client_order_id`` stays stable across amend cycles —
        that stability is what lets the exchange recognise a retry as a
        duplicate rather than a new order.

        After a restart, the anchor for an existing ``intent_key`` is
        reconstructed from the persisted :class:`BrokerStore` / :class:`RunContext`
        instead of being recomputed from ``_current_bar_ts_ms`` — the latter
        would yield a new ``client_order_id`` and break exchange-side dedup.
        """
        existing = self._envelopes.get(intent.intent_key)
        if existing is not None:
            return DispatchEnvelope(
                intent=intent,
                run_tag=existing.run_tag,
                bar_ts_ms=existing.bar_ts_ms,
                retry_seq=existing.retry_seq,
            )
        anchor = self._persisted_envelope_anchors.pop(intent.intent_key, None)
        if anchor is not None:
            envelope = DispatchEnvelope(
                intent=intent,
                run_tag=self._run_tag,
                bar_ts_ms=anchor.bar_ts_ms,
                retry_seq=anchor.retry_seq,
            )
            self._envelopes[intent.intent_key] = envelope
            return envelope
        envelope = DispatchEnvelope(
            intent=intent,
            run_tag=self._run_tag,
            bar_ts_ms=self._current_bar_ts_ms,
            retry_seq=0,
        )
        self._envelopes[intent.intent_key] = envelope
        if self._store_ctx is not None:
            self._store_ctx.record_envelope(
                key=intent.intent_key,
                bar_ts_ms=envelope.bar_ts_ms,
                retry_seq=envelope.retry_seq,
            )
        return envelope

    def _build_cancel_envelope(self, cancel: CancelIntent) -> DispatchEnvelope:
        return DispatchEnvelope(
            intent=cancel,
            run_tag=self._run_tag,
            bar_ts_ms=self._current_bar_ts_ms,
            retry_seq=0,
        )

    def _park_pending(
            self, envelope: DispatchEnvelope, error: OrderDispositionUnknownError,
            *, kind: str = 'new', old_intent: Intent | None = None,
    ) -> None:
        """Stash a dispatch whose exchange disposition the plugin could not confirm.

        :meth:`_verify_pending_dispatches` reruns ``get_open_orders`` on each
        subsequent sync and promotes the envelope back to
        ``_order_mapping`` once the order shows up.

        :param kind: ``'new'`` for ``execute_*`` parks, ``'modify'`` for
            ``modify_*`` parks. Persisted on the pending row so a later
            ``'rejected'`` resolution can decide whether the original
            exchange order is still live (and only the amend failed).
        :param old_intent: Only set for ``kind='modify'``. The pre-modify
            ``_active_intents[key]`` snapshot, captured BEFORE
            ``_diff_and_dispatch`` promotes the slot to the new intent
            (line 1328). Stashed in :attr:`_modify_old_intents` so a
            later ``'rejected'`` resolution can restore the slot and
            force the next diff to re-emit the amend — without this the
            promoted-new active matches Pine and the diff stays silent
            even though the exchange still holds the OLD order
            unmodified.
        """
        coid = error.client_order_id
        self._pending_verification[coid] = envelope
        # Re-parking the same coid in this engine instance must reset
        # the in-memory ``_consumed_attached_coids`` dedup. Without
        # this, if an earlier 'attached' resolution already marked the
        # coid as consumed and the row has since been re-parked
        # (record_park resets ``resolution`` to NULL on conflict), a
        # later 'attached' write on the fresh park would be skipped by
        # :meth:`_consume_plugin_resolutions` — leaving
        # ``_pending_verification`` stuck for brokers whose orders
        # never appear in ``get_open_orders`` (e.g. Capital.com
        # position-attached brackets).
        self._consumed_attached_coids.discard(coid)
        if kind == 'modify' and old_intent is not None:
            # First-park-wins: a chained modify (Pine flips parameters
            # while a previous amend is still parked) overwrites the
            # NEW intent in ``_active_intents`` but the EXCHANGE may
            # still be on the OLDEST pre-park state if every park ends
            # up rejected. The earliest captured snapshot is the safest
            # restoration target. ``setdefault`` preserves it; an
            # ``'attached'`` resolution clears the entry so a later
            # genuinely-fresh modify gets a clean snapshot.
            self._modify_old_intents.setdefault(
                envelope.intent.intent_key, old_intent,
            )
        if self._store_ctx is not None:
            self._store_ctx.record_park(
                coid=coid,
                key=envelope.intent.intent_key,
                kind=kind,
                order_ids=self._order_mapping.get(
                    envelope.intent.intent_key, [],
                ),
            )
        _log.warning(
            "dispatch for %s ended with unknown disposition "
            "(client_order_id=%s, kind=%s); will verify on next sync: %s",
            envelope.intent.intent_key, coid, kind, error,
        )

    def _dispatch_new(self, intent: Intent) -> None:
        envelope = self._build_envelope(intent)
        _blog_info("dispatching %s", intent)
        try:
            if isinstance(intent, EntryIntent):
                # §2.6.7 gate: drop the entry signal when any parent on
                # this symbol holds a degrading / degraded native
                # fail-safe. The drop is permanent for this signal —
                # the engine does NOT queue / replay (a Pine signal is
                # valid for its emit bar/tick; rejaying later would
                # open under different conditions than the script
                # intended). Emits both the block event and the
                # skipped-signal companion so downstream observability
                # can distinguish "rejected" from "queued".
                if self._native_failsafe_manager.block_new_entry(
                        symbol=intent.symbol,
                        pine_id=intent.pine_id,
                        bar_ts_ms=self._current_bar_ts_ms,
                ):
                    # Drop the envelope we just built before raising. The
                    # ``_build_envelope`` call at method entry already
                    # persisted ``_envelopes[intent.intent_key]`` and the
                    # SQLite envelope row keyed by ``bar_ts_ms`` /
                    # ``retry_seq``. Without this cleanup the next sync
                    # (or a restart-replay) for the same ``pine_id``
                    # would re-use that anchor and rebuild the SAME
                    # ``client_order_id`` from a stale bar timestamp,
                    # contradicting the documented "no broker call, no
                    # state retained" drop semantics for §2.6.7
                    # block-new-entry skips. The intent never reached
                    # the broker, so no idempotency anchor is needed —
                    # a later re-emission must start with a fresh
                    # envelope.
                    self._drop_envelope(intent.intent_key)
                    # Raise the same skip exception the caller already
                    # handles for plugin-side declines so the entry is
                    # not registered in :attr:`_active_intents` and gets
                    # added to ``skipped_entry_ids_this_sync`` — both are
                    # required for the documented §2.6.7 drop semantics
                    # (later re-emissions of this signal must not be
                    # treated as already active, and any same-sync
                    # bracket pointing at this parent must be suppressed
                    # as a skipped-parent dependent). Do NOT seed
                    # ``_order_mapping[intent.intent_key] = []`` here: the
                    # cross-restart adoption branch in :meth:`_diff_and_dispatch`
                    # treats any key present in :attr:`_order_mapping` as
                    # "broker already has this", which would skip
                    # :meth:`_dispatch_new` for a later re-emission of the
                    # same signal and leave the strategy believing the
                    # entry is live while no broker order was ever sent.
                    raise OrderSkippedByPlugin(
                        f"Entry dispatch blocked by degraded native "
                        f"fail-safe on {intent.symbol} "
                        f"(pine_id={intent.pine_id!r}); signal dropped, "
                        f"no broker call.",
                        intent_key=intent.intent_key,
                        reason="native_failsafe_blocked_entry",
                        context={
                            'symbol': intent.symbol,
                            'pine_id': intent.pine_id,
                            'bar_ts_ms': self._current_bar_ts_ms,
                        },
                    )
                # A fresh entry dispatch for a previously neutralised
                # parent pine_id is the script's explicit signal that
                # the late-parent-ENTRY-fill window has closed (the
                # prior parent was already settled by a no-FIFO
                # defensive close, and the user script has knowingly
                # re-armed the same entry name). Without this clear,
                # every subsequent fill for the same pine id would be
                # dropped by :meth:`_is_neutralised_parent_entry_fill`
                # for the rest of the process, leaving :attr:`_position`
                # flat while the broker holds the new open trade. Apply
                # AFTER the §2.6.7 block gate so a blocked entry (no
                # broker call) leaves the marker in place — a late fill
                # from the prior parent must still be suppressed. The
                # COID-keyed entries in
                # :attr:`_neutralised_parent_entry_coids` are NOT
                # cleared here — each COID is unique to a specific
                # prior dispatch, so leaving them in place is a safe
                # no-op against any later event (every fresh dispatch
                # produces a new COID).
                if (intent.pine_id
                        and intent.pine_id
                        in self._neutralised_parent_entry_pine_ids):
                    self._neutralised_parent_entry_pine_ids.discard(
                        intent.pine_id,
                    )
                orders = self._run_async(self._broker.execute_entry(envelope))
                self._order_mapping[intent.intent_key] = [o.id for o in orders]
                # Both-set Pine entry: the dispatch above placed the native
                # LIMIT leg. Arm the software price-watch for the STOP leg so
                # a stop cross cancels the LIMIT and fires a MARKET order. The
                # synthetic stop-fired MARKET re-enters this branch with
                # ``stop_fired_market=True`` and must NOT re-arm.
                if (intent.limit is not None
                        and intent.stop is not None
                        and not intent.stop_fired_market):
                    self._arm_entry_stop_watch(intent, envelope)
            elif isinstance(intent, ExitIntent):
                # Route engine-trigger partial brackets through the
                # dedicated state-machine dispatch. The condition is
                # deliberately conjunctive (mode AND intent flag) so a
                # whole-row exit on a SOFTWARE plugin still uses the
                # native bracket dispatch — the engine only owns the
                # partial slice.
                if (intent.is_partial_qty_bracket
                        and self._partial_qty_bracket_exit_mode
                        is CapabilityLevel.SOFTWARE):
                    self._dispatch_engine_trigger_partial_bracket(intent)
                else:
                    # Whole-row bracket on a SOFTWARE broker
                    # while engine-trigger partial legs are already tracked
                    # for the same parent is the §12 #4 incompatibility:
                    # a native full-row bracket and engine-trigger partial
                    # legs cannot coexist on one position (the new state
                    # machine has no path to keep them in sync; either side
                    # filling would over-close or strand siblings). Refuse
                    # the dispatch — the caller must cancel the existing
                    # partial bracket first.
                    #
                    # Same-key conversion: when the persisted legs share
                    # the SAME ``intent_key`` as the incoming whole-row
                    # exit (cross-restart case where Pine swapped the
                    # partial bracket for a full-row exit under the same
                    # ``strategy.exit(id, from_entry)``), the orphan
                    # cleanup at the top of :meth:`_diff_and_dispatch`
                    # cannot drop those legs (key is present in
                    # ``new_map``) and the kind-set adoption branch
                    # skips this case (new intent is not
                    # ``is_partial_qty_bracket``). Cancel the persisted
                    # partial legs here so the conversion can land
                    # instead of stranding the position with no
                    # protection until manual cleanup.
                    if (intent.from_entry is not None
                            and self._partial_qty_bracket_exit_mode
                            is CapabilityLevel.SOFTWARE
                            and self._partial_bracket_engine
                                .has_active_partial_bracket(
                                    intent.symbol, intent.from_entry,
                                )):
                        if self._partial_bracket_engine.has_active_legs_for_intent(
                                intent.intent_key):
                            _blog_info(
                                "converting engine-trigger partial bracket "
                                "to native whole-row exit for intent %r: "
                                "cancelling persisted partial legs before "
                                "dispatching the replacement full-row exit",
                                intent.intent_key,
                            )
                            self._partial_bracket_engine.cancel_legs_for_intent(
                                intent.intent_key,
                                reason='partial_to_whole_row_conversion',
                            )
                            # Cancelling the partial legs fired the
                            # :meth:`_on_partial_bracket_leg_state_change`
                            # listener which calls
                            # :meth:`_recompute_native_failsafe_for_parent`.
                            # If no other partial legs remain on this parent
                            # the worst-SL recompute saw an empty set and
                            # queued a ``stop_level=None`` clear snapshot —
                            # which would, once the dispatcher is installed,
                            # be flushed by :meth:`drive_native_failsafe` and
                            # delete the native bracket we are about to
                            # attach below. Retire the failsafe state for the
                            # parent so the pending clear PUT is dropped and
                            # the native bracket lives untouched. When other
                            # partial intent_keys still hold legs on this
                            # parent, the listener's recompute already
                            # produced a non-empty snapshot and we must keep
                            # the engine in charge.
                            if not self._partial_bracket_engine \
                                    .has_active_partial_bracket(
                                        intent.symbol, intent.from_entry,
                                    ):
                                # Resolve the parent's deterministic
                                # ``KIND_ENTRY`` COID. On the restart path the
                                # parent's entry envelope is no longer in
                                # ``_envelopes`` (the original dispatch was
                                # retired when the entry filled in a previous
                                # run), but the persisted leg rows still
                                # reference the COID. Fall back to
                                # ``_persisted_envelope_anchors`` so the
                                # failsafe state for the replayed parent gets
                                # retired and the empty-set worst-SL recompute
                                # cannot leak a ``stop_level=None`` snapshot
                                # that would clear the native bracket we are
                                # about to attach.
                                parent_dispatch_ref: str | None = (
                                    self._resolve_parent_opening_ref(
                                        intent.from_entry,
                                    )
                                )
                                if parent_dispatch_ref is not None:
                                    self._native_failsafe_manager.unregister_parent(
                                        parent_dispatch_ref,
                                    )
                            self._order_mapping.pop(intent.intent_key, None)
                            self._drop_envelope(intent.intent_key)
                            envelope = self._build_envelope(intent)
                        else:
                            raise RuntimeError(
                                "whole-row exit refuses: engine-trigger partial "
                                "legs already tracked for "
                                f"{intent.symbol!r}/{intent.from_entry!r}; the "
                                "script must cancel the partial bracket before "
                                "attaching a full-row exit.",
                            )
                    elif (intent.from_entry is not None
                            and self._partial_qty_bracket_exit_mode
                            is CapabilityLevel.SOFTWARE):
                        # Cross-key partial → native conversion within ONE sync:
                        # the cancel loop above already evicted the persisted
                        # partial legs (old key ∉ ``new_map``), which fired the
                        # leg-cancel listener and queued a ``stop_level=None``
                        # PUT on the failsafe manager. The same-key conversion
                        # guard above did not run because no active legs remain
                        # under the NEW intent's key. Without retiring the
                        # stale failsafe state here, the next
                        # :meth:`drive_native_failsafe` would flush that clear
                        # PUT and delete the native bracket we are about to
                        # attach. Detect the leftover state by checking the
                        # parent's KIND_ENTRY COID and unregister it before
                        # ``execute_exit`` lands the replacement protection.
                        parent_dispatch_ref: str | None = (
                            self._resolve_parent_opening_ref(
                                intent.from_entry,
                            )
                        )
                        if (parent_dispatch_ref is not None
                                and self._native_failsafe_manager.get_state(
                                    parent_dispatch_ref,
                                ) is not None
                                and not self._partial_bracket_engine
                                .has_active_partial_bracket(
                                    intent.symbol, intent.from_entry,
                                )):
                            _blog_info(
                                "cross-key partial-to-native conversion: "
                                "retiring stale failsafe state for parent %r "
                                "(no remaining active legs) before attaching "
                                "replacement whole-row exit %r",
                                parent_dispatch_ref, intent.intent_key,
                            )
                            self._native_failsafe_manager.unregister_parent(
                                parent_dispatch_ref,
                            )
                    orders = self._run_async(self._broker.execute_exit(envelope))
                    self._order_mapping[intent.intent_key] = [o.id for o in orders]
            elif isinstance(intent, CloseIntent):
                order = self._run_async(self._broker.execute_close(envelope))
                self._order_mapping[intent.intent_key] = [order.id]
            _blog_info(
                "dispatched %s -> %s",
                intent, self._order_mapping.get(intent.intent_key),
            )
        except OrderDispositionUnknownError as e:
            _blog_warning(
                "dispatch parked (unknown disposition) for %s: %s",
                intent, e,
            )
            self._park_pending(envelope, e)
        except BracketAttachAfterFillRejectedError as e:
            self._handle_bracket_attach_after_fill_reject(intent, e)
        except BrokerManualInterventionError as e:
            _blog_error(
                "dispatch halted (manual intervention) for %s: %s",
                intent, e,
            )
            self._record_halt(e)
            raise
        except OrderSkippedByPlugin:
            # Caller (_diff_and_dispatch / _resolve_deferred_for_entry) is
            # responsible for the warning + active-intents bookkeeping —
            # don't mislabel this as a dispatch failure.
            raise
        except Exception as e:
            _blog_error(
                "dispatch failed for %s: %s: %s",
                intent, type(e).__name__, e,
            )
            raise

    def _dispatch_engine_trigger_partial_bracket(
            self, intent: ExitIntent,
    ) -> None:
        """PERSIST-FIRST dispatch for a partial-qty engine-trigger bracket.

        See the partial-qty bracket exit design dossier §3.1 for the
        full lifecycle. The dispatch is split across three steps:

        1. **Invariant guard.** A native full-row bracket on the same
           parent is incompatible with the engine-trigger partial
           legs we are about to register (§12 #4). If one is already
           tracked the dispatch refuses to proceed — the caller is
           expected to surface this as a script-level error rather
           than silently coexisting with a duplicate bracket.

        2. **PERSIST-FIRST.** One :func:`create_engine_trigger_partial_leg_row`
           per declared leg (TP / SL / trail). Tick-deferred legs go
           in as :data:`LEG_STATE_PENDING_ENTRY`; absolute-level legs
           as :data:`LEG_STATE_ARMED`. The synthetic OCA group name
           (``__partial_exit_{pine_id}_{from_entry}__``) is shared
           across the trio so the cascade cancels every sibling when
           any one fires.

        3. **In-memory ledger handoff.** The persisted legs are
           registered with :class:`SoftwarePartialBracketEngine`. The
           caller (``_dispatch_new``) marks the intent_key in
           ``_active_intents`` and ``_order_mapping`` so subsequent
           diff cycles see the dispatch as live; no exchange-side
           order is opened (the WATCH phase fires close intents later
           when the price tick crosses a trigger).

        Steps 4–5 of §3 (price-tick WATCH wiring + close-dispatch
        action emitter) run from
        :meth:`_drive_partial_bracket_triggers` at the tail of every
        :meth:`sync` cycle; the OCA cascade fires through
        :meth:`SoftwarePartialBracketEngine.cascade_cancel_oca`
        on the ``triggering → triggered`` settlement. The §2.6.7
        fail-safe ownership work remains pending — see the design
        dossier.
        """
        symbol = intent.symbol
        from_entry = intent.from_entry
        # Scope the duplicate-dispatch guard to ``intent_key`` —
        # ``(pine_id, from_entry)``. Two scale-out exits on the same
        # ``from_entry`` (TP1 / TP2 with different ``strategy.exit`` ids)
        # are valid concurrent intents; the parent-wide
        # :meth:`has_active_partial_bracket` is reserved for the
        # native-vs-engine-partial coexistence invariant (§12 #4),
        # checked elsewhere when a native bracket is about to attach.
        if self._partial_bracket_engine.has_active_legs_for_intent(
                intent.intent_key,
        ):
            raise RuntimeError(
                "engine-trigger partial bracket dispatch refuses: "
                f"active legs already tracked for intent "
                f"{intent.intent_key!r} ({symbol!r}, from_entry "
                f"{from_entry!r}); the diff path must cancel the "
                "prior intent before re-dispatching.",
            )

        # Reciprocal §12 #4 invariant: a whole-row / native ExitIntent on
        # the SAME ``from_entry`` (different ``pine_id``, distinct
        # ``intent_key``) already attached at the broker cannot coexist
        # with engine-trigger partial legs we are about to register —
        # the new state machine has no path to keep them in sync, and
        # either side filling would over-close or strand siblings. The
        # opposite ordering (partial registered first, native attempted
        # later) is caught at :meth:`_dispatch_new` via
        # :meth:`SoftwarePartialBracketEngine.has_active_partial_bracket`;
        # this branch closes the loop for partial-after-native.
        for active in self._active_intents.values():
            if not isinstance(active, ExitIntent):
                continue
            if active.from_entry != from_entry:
                continue
            if active.intent_key == intent.intent_key:
                # The intent-key collision is already covered by the
                # has_active_legs_for_intent guard above; a separate
                # active entry with the same key would be a diff-loop
                # bug, not a coexistence violation.
                continue
            if active.is_partial_qty_bracket:
                # Two scale-out partials under the same parent are a
                # supported Pine pattern (TP1 / TP2 with distinct exit
                # ids); only NATIVE whole-row exits are incompatible.
                continue
            raise RuntimeError(
                "engine-trigger partial bracket dispatch refuses: a "
                "native whole-row ExitIntent is already active for "
                f"{symbol!r}/{from_entry!r} (intent "
                f"{active.intent_key!r}); the script must cancel the "
                "full-row bracket before attaching a partial-qty "
                "bracket on the same parent.",
            )

        envelope = self._build_envelope(intent)
        # Key the fail-safe ownership + leg stamping on the COID the parent
        # position opened under (KIND_ENTRY normally, KIND_ENTRY_STOP when the
        # both-set STOP leg won the OCO), live envelope or restart-anchor
        # rebuild — so the broker reconcile feed, which reports the position
        # under that same COID, confirms the registered state.
        parent_dispatch_ref = self._resolve_parent_opening_ref(from_entry)
        if parent_dispatch_ref is None:
            raise RuntimeError(
                "engine-trigger partial bracket dispatch refuses: no "
                f"parent envelope tracked for {from_entry!r} — the "
                "parent entry must be dispatched before its bracket.",
            )

        # §2.6.7 gate: a degrading / degraded broker-native fail-safe on
        # this parent blocks *new* partial brackets. Engine close
        # dispatches and intermediate leg triggers continue to flow —
        # only new bracket attachments are refused, because attaching
        # one would seed a new worst-SL the broker layer cannot
        # currently realise (unbounded loss risk on the new leg).
        #
        # Soft-skip via :class:`OrderSkippedByPlugin` (the same vehicle
        # the §2.6.7 block_new_entry path uses, see :meth:`_dispatch_new`):
        # raising a raw :class:`RuntimeError` here would propagate through
        # ``_dispatch_new`` (generic Exception branch re-raises) and abort
        # ``_diff_and_dispatch`` for the whole symbol — a normal
        # degraded-fail-safe condition would take down the rest of the
        # sync loop. ``OrderSkippedByPlugin`` is caught at every
        # ``_dispatch_new`` call site so only this single intent is
        # dropped; later bars / a user reset re-evaluate the bracket
        # against fresh state.
        if self._native_failsafe_manager.block_new_partial_bracket(
                parent_entry_dispatch_ref=parent_dispatch_ref,
                symbol=symbol,
                pine_id=intent.pine_id,
                from_entry=from_entry,
        ):
            # Drop the envelope built above before raising. Same
            # rationale as the §2.6.7 block-new-entry path in
            # :meth:`_dispatch_new`: ``_build_envelope`` already
            # persisted an anchor (``_envelopes`` + SQLite envelope
            # row); a later re-emission of this partial bracket must
            # be treated as a fresh signal rather than re-using the
            # stale ``bar_ts_ms`` / ``retry_seq`` from a dispatch that
            # never reached the broker.
            self._drop_envelope(intent.intent_key)
            raise OrderSkippedByPlugin(
                "engine-trigger partial bracket dispatch refuses: native "
                f"fail-safe state for parent {parent_dispatch_ref!r} is "
                "degrading/degraded; user reset / set_risk is required "
                "before attaching a new partial bracket on this parent.",
                intent_key=intent.intent_key,
                reason="native_failsafe_blocked_partial_bracket",
                context={
                    'symbol': symbol,
                    'pine_id': intent.pine_id,
                    'from_entry': from_entry,
                    'parent_entry_dispatch_ref': parent_dispatch_ref,
                },
            )

        # §2.6.7 ownership flip: register this parent under engine-failsafe.
        # The desired worst-SL is seeded after the legs land in the
        # ledger (the manager needs to see the per-leg trigger_level set
        # in :meth:`SoftwarePartialBracketEngine.register_legs`).
        parent_side = 'long' if intent.side == 'sell' else 'short'
        self._native_failsafe_manager.register_parent(
            parent_entry_dispatch_ref=parent_dispatch_ref,
            symbol=symbol,
            parent_side=parent_side,
            mintick=self._mintick,
            minmove=self._minmove,
            pricescale=self._pricescale,
        )

        # The leg must stay in ``LEG_STATE_PENDING_ENTRY`` until the
        # parent entry has filled — not just until tick-based offsets
        # are resolved. With absolute ``limit`` / ``stop`` prices on a
        # partial exit dispatched against an unfilled parent, the leg
        # has no resolution dependency but the WATCH-phase safety check
        # requires a live parent snapshot: if ``on_price_tick`` fires
        # while the parent is still flat, ``_safety_check`` aborts the
        # leg with ``parent_flat`` and the later parent fill has no
        # bracket left to promote. Gate the pending/armed decision on
        # *both* tick resolution AND parent presence in ``open_trades``.
        parent_filled = any(
            trade.entry_id == from_entry
            for trade in self._position.open_trades
        )
        leg_state = (
            LEG_STATE_PENDING_ENTRY
            if (intent.has_unresolved_ticks or not parent_filled)
            else LEG_STATE_ARMED
        )
        # Engine-trigger partial bracket siblings (TP / SL / trail under one
        # ``strategy.exit``) are *intra-bracket* OCA: when one leg fires the
        # others must cancel. The Pine layer (`strategy/__init__.py`) defaults
        # the ``strategy.exit`` ``oca_type`` to ``reduce`` to control how this
        # exit interacts with *other* exits (cross-bracket reduce-only), but
        # that semantic does not apply between THIS bracket's own legs. With
        # ``reduce`` preserved here, :meth:`SoftwarePartialBracketEngine.
        # cascade_cancel_oca` would early-return on the triggered leg and
        # leave the losing sibling armed — the design dossier mandates
        # ``oca_type='cancel'`` for software partial brackets (§ partial-qty
        # plan, "OCA reduce vs cancel"). Force ``cancel`` for the bracket
        # legs regardless of the parent ``intent.oca_type``.
        #
        # The internal group must be private per ``(pine_id, from_entry)``:
        # adopting the user-supplied ``intent.oca_name`` here would lump the
        # engine-internal legs of two unrelated partial exits (any pair of
        # ``strategy.exit`` calls the script tagged with the same OCA name)
        # into one cascade-cancel group, so a TP fill on one bracket would
        # tear down the other bracket's still-armed SL/TP and leave the
        # remaining quantity unprotected. The user's cross-bracket OCA
        # intent is preserved at the intent layer (parked ``ExitIntent``
        # OCA name), not by aliasing the engine-internal group key.
        oca_group = f"__partial_exit_{intent.pine_id}_{from_entry}__"
        oca_type = OcaType.CANCEL.value

        legs: list[PartialBracketLeg] = []
        coids: list[str] = []
        for spec in self._enumerate_engine_trigger_legs(intent):
            coid = envelope.client_order_id(spec.kind_code)
            if self._store_ctx is not None:
                create_engine_trigger_partial_leg_row(
                    self._store_ctx,
                    coid=coid,
                    symbol=symbol,
                    side=intent.side,
                    qty=intent.qty,
                    intent_key=intent.intent_key,
                    pine_entry_id=intent.pine_id,
                    from_entry=from_entry,
                    leg_kind=spec.leg_kind,
                    leg_state=leg_state,
                    parent_pine_entry_id=from_entry,
                    parent_entry_dispatch_ref=parent_dispatch_ref,
                    intent_partial_qty=intent.qty,
                    trigger_level=spec.trigger_level,
                    trigger_offset=spec.trigger_offset,
                    trail_activation_level=spec.trail_activation_level,
                    trail_activation_offset=spec.trail_activation_offset,
                    oca_group=oca_group,
                    oca_type=oca_type,
                )
            legs.append(PartialBracketLeg(
                coid=coid,
                symbol=symbol,
                pine_id=intent.pine_id,
                from_entry=from_entry,
                leg_kind=spec.leg_kind,
                leg_state=leg_state,
                side=intent.side,
                qty=intent.qty,
                intent_key=intent.intent_key,
                parent_pine_entry_id=from_entry,
                parent_entry_dispatch_ref=parent_dispatch_ref,
                intent_partial_qty=intent.qty,
                trigger_level=spec.trigger_level,
                trigger_offset=spec.trigger_offset,
                trail_activation_level=spec.trail_activation_level,
                trail_activation_offset=spec.trail_activation_offset,
                oca_group=oca_group,
                oca_type=oca_type,
            ))
            coids.append(coid)

        if not legs:
            raise RuntimeError(
                "engine-trigger partial bracket dispatch refuses: "
                f"intent {intent.intent_key!r} carries no bracket leg "
                "(intent_builder set is_partial_qty_bracket=True with "
                "no tp/sl/trail field — likely a builder regression).",
            )

        self._partial_bracket_engine.register_legs(legs)
        self._order_mapping[intent.intent_key] = coids

    def _enumerate_engine_trigger_legs(
            self, intent: ExitIntent,
    ) -> list['_EngineTriggerLegSpec']:
        """Yield one :class:`_EngineTriggerLegSpec` per declared bracket leg.

        - TP leg uses :data:`KIND_EXIT_TP_PARTIAL` and either
          ``tp_price`` (absolute) or a deferred ``profit_ticks``
          distance.
        - SL leg uses :data:`KIND_EXIT_SL_PARTIAL` and either
          ``sl_price`` (absolute) or a deferred ``loss_ticks``
          distance.
        - Trail leg uses :data:`KIND_EXIT_TRAIL_PARTIAL` plus
          ``trail_price`` (activation level, absolute) /
          ``trail_points_ticks`` (activation distance from entry, in
          ticks — resolved at parent fill) and ``trail_offset``
          (trailing distance from high-water mark, in ticks).

        All tick-typed inputs (``profit_ticks`` / ``loss_ticks`` /
        ``trail_points_ticks`` / ``trail_offset``) are multiplied by
        ``self._mintick`` here so the leg row stores price units
        consistently — the pending→armed resolver and WATCH-phase
        recompute never need to re-multiply.

        For trail legs the spec separates the two distances Pine's
        ``strategy.exit(trail_price/trail_points, trail_offset)``
        encodes: the activation level (when the trail starts to
        follow) and the trailing offset (how far behind the high it
        sits afterwards). The activation distance lives in
        ``trail_activation_offset`` / ``trail_activation_level`` and
        ``trigger_offset`` carries only the post-activation stop
        offset, so a pre-activation tick cannot cross a stop seeded
        from the activation distance.
        """
        rows: list[_EngineTriggerLegSpec] = []
        if intent.tp_price is not None or intent.profit_ticks is not None:
            tp_offset = (
                float(intent.profit_ticks) * self._mintick
                if intent.profit_ticks is not None else None
            )
            rows.append(_EngineTriggerLegSpec(
                leg_kind=LEG_KIND_TP_PARTIAL,
                kind_code=KIND_EXIT_TP_PARTIAL,
                trigger_level=intent.tp_price,
                trigger_offset=tp_offset,
            ))
        if intent.sl_price is not None or intent.loss_ticks is not None:
            sl_offset = (
                float(intent.loss_ticks) * self._mintick
                if intent.loss_ticks is not None else None
            )
            rows.append(_EngineTriggerLegSpec(
                leg_kind=LEG_KIND_SL_PARTIAL,
                kind_code=KIND_EXIT_SL_PARTIAL,
                trigger_level=intent.sl_price,
                trigger_offset=sl_offset,
            ))
        # ``intent.trail_offset`` is the post-activation trailing
        # distance, in TICKS (see :attr:`pynecore.lib.strategy.Order.trail_offset`,
        # which is multiplied by ``syminfo.mintick`` at every comparison
        # in the Pine simulator). ``trail_price`` is the absolute
        # activation level (already in price units); ``trail_points_ticks``
        # is the pre-fill activation distance from entry (in ticks) and
        # is resolved into an activation level only after the parent
        # entry fills.
        trail_offset_ticks = intent.trail_offset
        trail_level = intent.trail_price
        trail_ticks = intent.trail_points_ticks
        if trail_level is not None or trail_offset_ticks is not None or trail_ticks is not None:
            resolved_trail_offset: float | None = (
                float(trail_offset_ticks) * self._mintick
                if trail_offset_ticks is not None else None
            )
            trail_activation_offset: float | None = (
                float(trail_ticks) * self._mintick
                if trail_ticks is not None else None
            )
            rows.append(_EngineTriggerLegSpec(
                leg_kind=LEG_KIND_TRAIL_PARTIAL,
                kind_code=KIND_EXIT_TRAIL_PARTIAL,
                # The moving stop is initialised by
                # :meth:`SoftwarePartialBracketEngine._maybe_advance_trail`
                # the moment the activation level is crossed — we
                # deliberately leave ``trigger_level`` unset here.
                trigger_level=None,
                trigger_offset=resolved_trail_offset,
                trail_activation_level=trail_level,
                trail_activation_offset=trail_activation_offset,
            ))
        return rows

    def _handle_bracket_attach_after_fill_reject(
            self, intent: Intent, e: BracketAttachAfterFillRejectedError,
    ) -> None:
        """Recover from a bracket attach reject after a parent fill committed.

        The plugin already rolled back the persisted bracket leg rows;
        the parent ENTRY/EXIT fill is on the exchange but has no
        protective TP/SL. Halting here would leave the unprotected
        position exposed indefinitely — instead synthesise a defensive
        :class:`CloseIntent` and dispatch it immediately to flatten the
        position.

        The recovery follows the *pending lifecycle* defined in
        ``docs/pynecore/plugin-system/broker/defensive-close-pending-lifecycle.md``:

        1. A :class:`BracketAttachRejectContext` is derived from the
           exception + triggering intent.
        2. A :class:`PendingDefensiveClose` marker is set BEFORE the
           defensive close dispatches — in-memory and persisted on the
           parent entry row (``extras['defensive_close_pending']``).
        3. The synthetic close is dispatched; on success the marker is
           updated with the broker-side close ref.
        4. The OCA-cancel cascade fires immediately (it does not wait
           on the close FILL).
        5. Residual broker-side orders enumerated by the plugin's
           :meth:`~pynecore.core.plugin.broker.BrokerPlugin.get_residual_orders_after_bracket_attach_reject`
           are cancelled via
           :meth:`~pynecore.core.plugin.broker.BrokerPlugin.cancel_broker_order_ref`
           with two-tier error handling.
        6. The original intent is surfaced as
           :class:`OrderSkippedByPlugin`.

        Crucially, the engine no longer drops the parent entry intent
        from ``_active_intents`` at this point — that cleanup is
        deferred until the close FILL event arrives, so a same-bar
        :meth:`reconcile` cannot misclassify the flat broker snapshot
        as an external flatten.

        If the defensive close itself fails in an unrecoverable way
        (anything other than transient park / plugin skip), the marker
        is cleared and we escalate to
        :class:`BrokerManualInterventionError` — at that point the
        position is open AND we couldn't auto-close it, an operator
        must intervene.
        """
        _blog_error(
            "bracket attach rejected after parent fill for %s; "
            "issuing defensive market close "
            "(deal_id=%s, side=%s, qty=%s): %s",
            intent, e.position_deal_id, e.position_side, e.qty, e,
        )
        close_side = "sell" if e.position_side == "buy" else "buy"
        defensive_pine_id = f"__pyne_defensive_close__{e.position_coid}"
        close_intent = CloseIntent(
            pine_id=defensive_pine_id,
            symbol=e.symbol,
            side=close_side,
            qty=e.qty,
            immediately=True,
            comment=f"defensive close after bracket attach reject: {e}",
        )

        context = BracketAttachRejectContext.from_exception(e, intent)
        entry_id = context.from_entry
        marker_armed = False
        if entry_id:
            # Pre-build the close envelope so we can stamp the canonical
            # ``client_order_id`` on the marker BEFORE the dispatch runs.
            # ``_dispatch_new`` will re-use this same envelope via
            # :meth:`_build_envelope`'s ``_envelopes`` cache, so the COID
            # we stamp here is exactly the one ``execute_close`` (or its
            # park path) submits. The COID is needed by
            # :meth:`_route_defensive_close_fill` to recognise FILLs
            # arriving without ``pine_id`` (polled-orders fallback) and
            # by the settlement path to drop the parked pending row.
            close_envelope = self._build_envelope(close_intent)
            marker = PendingDefensiveClose(
                entry_id=entry_id,
                close_intent_key=close_intent.intent_key,
                close_order_ref=None,
                pending_since=time.time(),
                reject_context=context,
                close_client_order_id=close_envelope.client_order_id(KIND_CLOSE),
                # Snapshot of :attr:`_position.size` BEFORE the
                # defensive close dispatches. Used by startup replay
                # (see :meth:`reconcile`'s replayed-marker branch) to
                # distinguish whether a post-restart broker snapshot
                # reflects the pre-close aggregate (close not yet
                # landed) or the post-close residual (close landed
                # but FILL was not observed by the prior process).
                # Without this, the no-FIFO settle branch cannot
                # safely apply the delayed FILL delta to a freshly
                # adopted broker view.
                pre_close_position_size=float(self._position.size),
            )
            self._set_pending_defensive_close(entry_id, marker)
            marker_armed = True

        try:
            self._dispatch_new(close_intent)
        except OrderDispositionUnknownError:
            # Defensive close parked (network-ambiguous) — the next
            # reconcile / next bar will resolve. The pending marker
            # stays armed so :meth:`reconcile` does not flip state
            # until the close ultimately settles (or the grace window
            # halts the run).
            pass
        except OrderSkippedByPlugin as skipped:
            # Plugin proactively declined to dispatch (e.g. close qty
            # below ``min_size``). Unlike a parked dispatch, NO order
            # was sent and none will ever fill — leaving the marker
            # armed would suppress retries until the grace window
            # expires while the unprotected position stays open.
            # Escalate to manual intervention immediately: drop the
            # marker, record the halt, and re-raise so the run halts.
            if marker_armed and entry_id:
                self._clear_pending_defensive_close(entry_id)
            halt = BrokerManualInterventionError(
                f"Defensive close after bracket attach reject was "
                f"skipped by plugin for {intent.intent_key} "
                f"(reason={skipped.reason!r}): position is open and "
                f"unprotected, no close order was dispatched — manual "
                f"intervention required: {skipped}",
                intent_key=intent.intent_key,
                context={
                    'position_deal_id': e.position_deal_id,
                    'position_coid': e.position_coid,
                    'symbol': e.symbol,
                    'qty': e.qty,
                    'cause': str(e),
                    'skipped_reason': skipped.reason,
                    'skipped_context': skipped.context,
                },
            )
            self._record_halt(halt)
            raise halt from skipped
        except BrokerManualInterventionError as halt:
            # Inner _dispatch_new already recorded the halt before
            # propagating — drop the marker (recovery is now manual)
            # and re-raise verbatim.
            if marker_armed and entry_id:
                self._clear_pending_defensive_close(entry_id)
            raise halt
        except Exception as nested:
            # Unexpected failure — escalate. We record the halt here
            # ourselves because this exception is being raised from a
            # call site (the outer _dispatch_new's
            # BracketAttachAfterFillRejectedError branch) that does
            # NOT pass back through the BrokerManualInterventionError
            # except in _dispatch_new, so _record_halt would otherwise
            # be skipped. Drop the marker — automated recovery has
            # already failed.
            if marker_armed and entry_id:
                self._clear_pending_defensive_close(entry_id)
            halt = BrokerManualInterventionError(
                f"Defensive close after bracket attach reject failed for "
                f"{intent.intent_key}: {nested}",
                intent_key=intent.intent_key,
                context={
                    'position_deal_id': e.position_deal_id,
                    'position_coid': e.position_coid,
                    'symbol': e.symbol,
                    'qty': e.qty,
                    'cause': str(e),
                },
            )
            self._record_halt(halt)
            raise halt from nested

        # Stamp ``natural_close_at`` on the parent entry row only when
        # the defensive close landed on the exchange synchronously
        # (``_order_mapping`` is populated on success but stays empty
        # when ``_dispatch_new`` parks the dispatch via
        # :meth:`_park_pending` on :class:`OrderDispositionUnknownError`).
        # When the close PARKED, the position may still be open —
        # stamping would mask a legitimately stuck position from the
        # plugin-side reconciler.
        #
        # On the success path the parent position will vanish from
        # the broker snapshot on the next poll, potentially BEFORE
        # the close activity record arrives. Without the stamp the
        # reconciler stamps ``missing_pending_since`` on the parent
        # entry row and raises a false :class:`UnexpectedCancelError`
        # once the grace window expires — halting the bot for a
        # position WE deliberately closed. ``natural_close_at`` opts
        # the row out of that accounting (same convention TP/SL
        # natural-close flows already use). The row is NOT physically
        # closed (``close_order`` would break ``find_by_ref`` lookups
        # when the close activity finally arrives via the broker's
        # history stream).
        dispatch_succeeded = close_intent.intent_key in self._order_mapping
        if dispatch_succeeded:
            if self._store_ctx is not None:
                row = self._store_ctx.get_order(e.position_coid)
                if row is not None:
                    extras = dict(row.extras or {})
                    extras['natural_close_at'] = time.time()
                    self._store_ctx.upsert_order(
                        e.position_coid, extras=extras,
                    )
            # Capture the broker-side close ref so the FILL handler
            # (and a possible startup replay) can match the close
            # event back to its marker even when the FILL routes
            # through the polled-orders path rather than the WS
            # stream (where ``OrderEvent.pine_id`` is reliable).
            if marker_armed and entry_id:
                refs = self._order_mapping.get(close_intent.intent_key, [])
                if refs:
                    self._set_pending_defensive_close(
                        entry_id,
                        dataclasses.replace(
                            self._pending_defensive_close[entry_id],
                            close_order_ref=refs[0],
                        ),
                    )
            if entry_id:
                # Fire the OCA-cancel cascade BEFORE removing the
                # parent ENTRY intent. We are running inside
                # :meth:`_resolve_deferred_for_entry`, which itself
                # runs from :meth:`_route_event` BEFORE the in-band
                # :meth:`_cascade_oca_cancel` call — dropping the
                # intent here without cascading first would leave
                # ``_active_intents`` empty when that later call runs,
                # silently skipping sibling cancellation for
                # ``oca_type='cancel'`` groups and leaving the
                # siblings live exchange-side after the defensive
                # close. The cascade is idempotent within a sync via
                # :attr:`_cancelled_oca_groups_this_sync`, so the
                # in-band follow-up is a no-op.
                self._cascade_oca_cancel_for_key(entry_id)
                # Residual broker-side orders the engine must cancel
                # explicitly — partial-fill entry remainders, separate
                # TP/SL order entities, etc. Default plugin returns
                # ``[]`` (Capital.com has nothing to cancel because
                # bracket attach is full-row + position-attached). When
                # a transient enumeration/cancel failure is swallowed,
                # stamp the marker so
                # :meth:`_retry_residual_cleanup_after_transient_fill`
                # replays before the close FILL lands — otherwise the
                # FILL-time path (which only fires after
                # ``_settled_defensive_close_pine_ids`` is seeded by the
                # FILL itself) would be the first retry, leaving live
                # residuals exchange-side during the entire dispatch →
                # FILL window.
                residual_clean = self._cancel_bracket_reject_residuals(context)
                if (not residual_clean
                        and marker_armed and entry_id):
                    existing_marker = self._pending_defensive_close.get(entry_id)
                    if (existing_marker is not None
                            and not existing_marker.residual_cleanup_pending):
                        self._set_pending_defensive_close(
                            entry_id,
                            dataclasses.replace(
                                existing_marker,
                                residual_cleanup_pending=True,
                            ),
                        )
                # Mark the entry as defensively closed so the in-flight
                # :meth:`_diff_and_dispatch` loop (when it is the caller)
                # short-circuits any subsequent intent in its precomputed
                # ``new_map`` that still references this ``from_entry``.
                # Without this guard a sibling bracket leg or an alternate
                # ``exit_id`` for the same parent — already missing from
                # ``_active_intents`` after the cleanup — would look fresh
                # and trigger another bracket dispatch against a position
                # we just flattened. The set is per-sync and harmless
                # outside the loop context (event-driven callers ignore it).
                self._defensively_closed_entries_this_sync.add(entry_id)

        raise OrderSkippedByPlugin(
            f"Bracket attach rejected after entry fill — parent position "
            f"closed defensively (deal_id={e.position_deal_id}); "
            f"intent re-evaluation deferred to next bar",
            intent_key=intent.intent_key,
            reason="bracket_reject_defensive_close",
            context={
                'position_deal_id': e.position_deal_id,
                'position_coid': e.position_coid,
                'symbol': e.symbol,
                'qty': e.qty,
            },
        ) from e

    def _cancel_bracket_reject_residuals(
            self, context: BracketAttachRejectContext,
            *, raise_on_transient: bool = False,
    ) -> bool:
        """Cancel broker-side residual orders enumerated by the plugin.

        Two-step flow: enumeration via the plugin's
        :meth:`~pynecore.core.plugin.broker.BrokerPlugin.get_residual_orders_after_bracket_attach_reject`
        then per-ref cancel via
        :meth:`~pynecore.core.plugin.broker.BrokerPlugin.cancel_broker_order_ref`.
        Both steps share the same error contract:

        - :class:`ExchangeConnectionError` /
          :class:`OrderDispositionUnknownError` → behaviour depends on
          ``raise_on_transient``:

          * Default (``False``) — transient warn; return ``False`` so the
            dispatch-time / replay caller can stamp
            ``residual_cleanup_pending=True`` on the pending marker and
            let :meth:`_retry_residual_cleanup_after_transient_fill`
            replay before the FILL ever lands. Used from
            :meth:`_handle_bracket_attach_after_fill_reject` and
            :meth:`_replay_pending_defensive_closes`.
          * ``True`` — re-raise so the caller can keep the marker armed
            instead of falling through to a marker drop. Used by
            :meth:`_route_defensive_close_fill`, which is the last
            in-process retry before the marker is cleared; swallowing a
            transient failure there — either in enumeration or in the
            per-ref cancel — would let the caller declare recovery
            complete while parent/TP/SL/partial-remainder orders are
            still live exchange-side.

          The plugin's idempotency contract guarantees a repeated cancel
          on the same ref is safe; enumeration is also documented as
          safe to call repeatedly with the same context.
        - :class:`BrokerManualInterventionError` (raised directly by the
          plugin in either step) → record the halt and propagate so the
          dispatch-time caller still emits the
          ``ManualInterventionRequiredEvent``.
        - Any other unexpected exception → treated as a manual-
          intervention condition: residuals are unknown / un-cancellable,
          so a still-live entry remainder could rebuild the exposure we
          just closed. The engine halts.

        :returns: ``True`` when enumeration + every per-ref cancel
            completed cleanly; ``False`` when a transient failure was
            swallowed (``raise_on_transient=False``) and the caller must
            schedule a retry. The non-transient branches either return
            ``True`` (success) or raise (halt-worthy failure).
        """
        try:
            residuals = self._broker.get_residual_orders_after_bracket_attach_reject(
                context,
            )
        except (ExchangeConnectionError, OrderDispositionUnknownError) as exc:
            # Transient enumeration failure: same contract as the
            # cancel-loop branch below. With ``raise_on_transient=True``
            # the caller (FILL-time
            # :meth:`_route_defensive_close_fill`) needs to keep the
            # marker armed for the next reconcile/restart retry —
            # silently returning here would let the caller write the
            # ``defensive_close_filled`` audit + drop the marker even
            # though residuals were never actually enumerated, leaving
            # any live parent/TP/SL/partial-remainder orders orphaned.
            _blog_warning(
                "plugin get_residual_orders_after_bracket_attach_reject "
                "transient failure for %s (%s: %s) — will retry on next "
                "reconcile/restart",
                context.intent_key, type(exc).__name__, exc,
            )
            if raise_on_transient:
                raise
            return False
        except BrokerManualInterventionError as halt:
            # Same propagation pattern as the cancel loop below: record
            # the halt explicitly so the in-flight ``_dispatch_new``
            # caller can still emit ``ManualInterventionRequiredEvent``
            # even though the exception type is re-raised through a
            # sibling ``try`` boundary that would not otherwise hit
            # ``_dispatch_new``'s ``except BrokerManualInterventionError``
            # clause.
            self._record_halt(halt)
            raise
        except Exception as exc:
            # Unexpected enumeration failure: residuals are unknown,
            # so we cannot safely declare the defensive close
            # recovered. Treat it the same way as an explicit broker
            # reject on a residual cancel — halt and require manual
            # intervention so a still-live parent / partial remainder
            # cannot rebuild the exposure we just closed.
            halt = BrokerManualInterventionError(
                f"plugin get_residual_orders_after_bracket_attach_reject "
                f"unexpectedly failed for {context.intent_key}: {exc}",
                intent_key=context.intent_key,
                context={
                    'position_coid': context.position_coid,
                    'symbol': context.symbol,
                    'cause': str(exc),
                },
            )
            self._record_halt(halt)
            raise halt from exc
        all_clean = True
        for ref in residuals:
            try:
                self._run_async(self._broker.cancel_broker_order_ref(ref))
            except (ExchangeConnectionError, OrderDispositionUnknownError) as exc:
                _blog_warning(
                    "residual cancel %s transient failure (%s): %s — will "
                    "retry on next reconcile/restart",
                    ref, type(exc).__name__, exc,
                )
                if raise_on_transient:
                    raise
                all_clean = False
            except BrokerManualInterventionError as halt:
                # Latch the halt before propagating. This helper is
                # reachable from :meth:`_handle_bracket_attach_after_fill_reject`
                # via the residual cancel loop, which itself is invoked
                # from inside :meth:`_dispatch_new`'s
                # ``except BracketAttachAfterFillRejectedError`` handler.
                # A bare ``raise`` here would propagate the
                # ``BrokerManualInterventionError`` out without re-entering
                # the sibling ``except BrokerManualInterventionError`` clause
                # in :meth:`_dispatch_new` (Python only honours one
                # handler per ``try``), bypassing :meth:`_record_halt` /
                # the ``ManualInterventionRequiredEvent`` emission and
                # leaving ``engine.halted`` ``False`` despite a
                # manual-intervention failure. Recording here is
                # idempotent — :meth:`_record_halt` keeps the first
                # halt's context if called twice.
                self._record_halt(halt)
                raise
            except Exception as exc:
                halt = BrokerManualInterventionError(
                    f"Residual cancel for {ref!r} explicitly rejected by "
                    f"broker after defensive close of {context.intent_key}: "
                    f"{exc}",
                    intent_key=context.intent_key,
                    context={
                        'residual_ref': ref,
                        'position_coid': context.position_coid,
                        'symbol': context.symbol,
                        'cause': str(exc),
                    },
                )
                self._record_halt(halt)
                raise halt from exc
        return all_clean

    def _set_pending_defensive_close(
            self, entry_id: str, marker: PendingDefensiveClose,
    ) -> None:
        """Install (or refresh) a defensive-close pending marker.

        In-memory map + parent entry row extras (``defensive_close_pending``
        key) are written atomically from the engine's point of view —
        the store layer's UPSERT runs under its own transaction.

        Pyramiding can fill multiple positions under the same Pine entry
        id, each with its own ``position_coid``. The in-memory map is
        keyed by ``entry_id`` only, so a second simultaneous bracket-
        attach reject for a different ``position_coid`` would silently
        overwrite the first marker — the earlier defensive close could
        no longer be matched, retried, audited, or cleared from extras.
        Detect this collision and halt: refresh of the SAME marker
        (matching ``position_coid``) stays in place, but a foreign-COID
        collision escalates to manual intervention rather than corrupt
        engine state.
        """
        existing = self._pending_defensive_close.get(entry_id)
        if (existing is not None
                and existing.reject_context.position_coid
                != marker.reject_context.position_coid):
            halt = BrokerManualInterventionError(
                f"Concurrent defensive-close markers for the same Pine "
                f"entry id {entry_id!r} are not supported: existing "
                f"marker has position_coid="
                f"{existing.reject_context.position_coid!r} "
                f"(close_intent_key={existing.close_intent_key!r}), "
                f"new marker has position_coid="
                f"{marker.reject_context.position_coid!r} "
                f"(close_intent_key={marker.close_intent_key!r}). "
                f"This can occur when pyramiding fills multiple "
                f"positions under the same entry id and more than one "
                f"hits a bracket-attach reject simultaneously — manual "
                f"intervention is required to settle both defensive "
                f"closes safely.",
                intent_key=marker.close_intent_key,
                context={
                    'entry_id': entry_id,
                    'existing_position_coid':
                        existing.reject_context.position_coid,
                    'existing_close_intent_key': existing.close_intent_key,
                    'new_position_coid': marker.reject_context.position_coid,
                    'new_close_intent_key': marker.close_intent_key,
                },
            )
            self._record_halt(halt)
            raise halt
        self._pending_defensive_close[entry_id] = marker
        if self._store_ctx is None:
            return
        position_coid = marker.reject_context.position_coid
        row = self._store_ctx.get_order(position_coid)
        if row is None:
            return
        extras = dict(row.extras or {})
        extras['defensive_close_pending'] = marker.to_extras_dict()
        self._store_ctx.upsert_order(position_coid, extras=extras)

    def _replay_pending_defensive_closes(self) -> None:
        """Re-arm or drop persisted defensive-close markers across restart.

        Iterates the BrokerStore's live orders looking for an
        ``extras['defensive_close_pending']`` payload, then either:

        - Drops the marker (in-memory + extras) when the matching
          ``'defensive_close_filled'`` audit event already exists for
          the run — the FILL settled before the crash, the in-memory
          marker is gone but the extras row survived.
        - Re-arms the in-memory marker AND replays the residual-cancel
          loop via
          :meth:`_cancel_bracket_reject_residuals`. The plugin's
          idempotency contract on
          :meth:`~pynecore.core.plugin.broker.BrokerPlugin.cancel_broker_order_ref`
          guarantees the replay is safe — already-cancelled refs are
          a no-op.

        Called once from
        :meth:`~pynecore.core.script_runner.ScriptRunner.start_broker`
        BEFORE the first :meth:`reconcile`, so the reconcile pass sees
        a coherent snapshot (markers re-armed for in-flight closes,
        already-settled markers cleared).

        No-op without persistence (``store_ctx is None``) — pure in-memory
        sessions cannot survive a restart by definition.
        """
        if self._store_ctx is None:
            return
        # First pass: reseed duplicate-fill caches from every persisted
        # ``defensive_close_filled`` audit event for this logical run.
        # The in-memory ``_settled_*`` sets are populated either by
        # :meth:`_mark_defensive_close_settled` at FILL time or by the
        # branches below that match a still-armed marker — but the
        # *normal* successful settlement path writes the audit and then
        # removes the ``defensive_close_pending`` extras payload, so a
        # restart after that cleanup would leave both the in-memory
        # caches and this loop's payload-gated branches empty. A delayed
        # WS reconnect / polled-orders resync that replays the same FILL
        # post-restart would then bypass
        # :meth:`_is_duplicate_defensive_close_fill` and reach
        # ``record_fill`` as a phantom opposite-side trade. Walking the
        # audit log first guarantees dedup state is restored regardless
        # of whether the marker payload survives.
        try:
            audit_iter = self._store_ctx.iter_events_by_kind_for_run_id(
                'defensive_close_filled',
            )
        except AttributeError:
            # Older store contexts without the helper — fall back to the
            # marker-payload-gated branches below.
            audit_iter = iter(())
        for intent_key, position_coid, exchange_order_id, audit_payload in audit_iter:
            if intent_key:
                self._settled_defensive_close_pine_ids.add(intent_key)
            if exchange_order_id:
                self._settled_defensive_close_order_refs.add(exchange_order_id)
            # The audit row's ``exchange_order_id`` column carries
            # ``marker.close_order_ref`` (the dispatch-time ref). When
            # the actual FILL arrived with a different ``event.order.id``
            # (polled-orders fallback / broker-rekey),
            # :meth:`_route_defensive_close_fill` stashes the alternate
            # id under ``fill_exchange_order_id`` in the audit payload.
            # Seeding the dedupe cache from both keys ensures a delayed
            # WS / polled-orders replay of the same FILL is rejected
            # regardless of which order id the replay carries.
            fill_exchange_order_id = audit_payload.get(
                'fill_exchange_order_id',
            )
            if (isinstance(fill_exchange_order_id, str)
                    and fill_exchange_order_id):
                self._settled_defensive_close_order_refs.add(
                    fill_exchange_order_id,
                )
            close_coid = audit_payload.get('close_client_order_id')
            if isinstance(close_coid, str) and close_coid:
                self._settled_defensive_close_client_order_ids.add(close_coid)
            # Late parent ENTRY fill neutralisation. When the prior process
            # wrote ``defensive_close_filled`` (so the close audit landed)
            # but the parent ENTRY ``filled`` / ``partial`` WS event was
            # not yet delivered before restart — and the marker row was
            # already cleaned (or the per-marker pass below cannot fire),
            # there is no longer any pending marker to gate
            # :meth:`_is_neutralised_parent_entry_fill`. Without seeding
            # the caches here, a delayed parent ENTRY fill arriving after
            # startup would pass :meth:`_route_event` and reach
            # :meth:`BrokerPosition.record_fill` on an already-flat (or
            # reduced) ``_position``, opening a phantom trade. Seed only
            # the unique ``position_coid`` cache from the audit row's
            # ``client_order_id`` column (where
            # :meth:`_route_defensive_close_fill` records
            # ``marker.reject_context.position_coid``).
            #
            # Pine ids (``entry_id``) are intentionally NOT seeded here:
            # they are user-script-controlled, reusable identifiers
            # (``"Long"``, ``"Short"``). After the prior process settled
            # the close and the user script re-dispatched the same pine
            # id, the in-memory set was cleared by :meth:`_dispatch_new`
            # — but the audit row still names the old pine id. Re-seeding
            # it on restart would silently drop a legitimate fill for the
            # re-dispatched entry (whose ``client_order_id`` / broker
            # ``order.id`` belong to the new dispatch, not the old
            # audit's ``position_coid``). The COID set is safe because
            # every dispatch mints a fresh COID — leaving stale COIDs
            # in the cache cannot collide with future legitimate
            # identifiers.
            if position_coid:
                self._neutralised_parent_entry_coids.add(position_coid)
        for row in self._store_ctx.iter_live_orders():
            payload = row.extras.get('defensive_close_pending')
            if payload is None:
                continue
            try:
                marker = PendingDefensiveClose.from_extras_dict(payload)
            except ValueError as exc:
                _blog_warning(
                    "defensive_close_pending replay: dropping malformed "
                    "marker on %s: %s",
                    row.client_order_id, exc,
                )
                extras = dict(row.extras)
                extras.pop('defensive_close_pending', None)
                self._store_ctx.upsert_order(
                    row.client_order_id, extras=extras,
                )
                continue
            # Settled-marker probe runs BEFORE the duplicate-entry-id
            # check. The prior instance can crash AFTER writing the
            # ``defensive_close_filled`` audit but BEFORE
            # :meth:`_clear_pending_defensive_close` removed the marker
            # row from ``order.extras``; if a sibling marker for the
            # same Pine id is also live (pyramiding under one entry id),
            # ``iter_live_orders`` row order would otherwise decide
            # between halting (the settled row arrives second and trips
            # the duplicate check) and a clean replay (the settled row
            # arrives first and gets cleared before the sibling is
            # observed). Drain any per-marker settled payloads first so
            # the duplicate check only sees genuinely-armed markers.
            settled = self._store_ctx.find_event_by_intent_key(
                marker.close_intent_key, 'defensive_close_filled',
            )
            if settled:
                # Marker survived the extras column but the FILL audit
                # event proves the cleanup already happened in a prior
                # instance — drop the persisted marker so reconcile
                # does not halt on a phantom pending close.
                _blog_info(
                    "defensive_close_pending replay: %s already settled "
                    "in prior run instance — clearing marker",
                    marker.entry_id,
                )
                # Seed the duplicate-fill cache with the synthetic
                # pine_id (and any known close_order_ref) so a duplicate
                # FILL replayed post-restart — e.g. the plugin's
                # ``get_open_orders`` resync racing a delayed WS replay —
                # is rejected by :meth:`_route_event` before it reaches
                # ``record_fill``.
                if marker.close_intent_key:
                    self._settled_defensive_close_pine_ids.add(
                        marker.close_intent_key,
                    )
                if marker.close_order_ref is not None:
                    self._settled_defensive_close_order_refs.add(
                        marker.close_order_ref,
                    )
                # Seed the ``client_order_id`` cache too — without it a
                # polled-orders FILL replay for a parked close (marker
                # carried ``close_order_ref=None``) would arrive with
                # ``pine_id=None`` and an ``order.id`` the engine never
                # saw before restart, slipping past
                # :meth:`_is_duplicate_defensive_close_fill` and reaching
                # ``record_fill``.
                if marker.close_client_order_id is not None:
                    self._settled_defensive_close_client_order_ids.add(
                        marker.close_client_order_id,
                    )
                # Finish the parked-close anchor cleanup the prior instance
                # may have crashed before completing. The audit event proves
                # the FILL landed, but :meth:`_route_defensive_close_fill`
                # writes the audit before ``record_unpark`` /
                # ``_clear_pending_defensive_close`` run, so a crash in that
                # window can leave the close ``client_order_id`` in
                # :attr:`_persisted_pending_anchors` (and the persisted
                # ``pending_verifications`` row). Without this drop,
                # :meth:`_verify_pending_dispatches` would keep polling
                # ``get_open_orders`` forever for a close that already
                # filled.
                close_coid = marker.close_client_order_id
                if close_coid is not None:
                    self._pending_verification.pop(close_coid, None)
                    self._persisted_pending_anchors.pop(close_coid, None)
                    self._store_ctx.record_unpark(close_coid)
                # Finish the idempotency-anchor cleanup the prior instance
                # may have crashed before completing. Mirror the in-flight
                # FILL settlement path (see ``_route_defensive_close_fill``):
                # without dropping the parent position tracking, the
                # synthetic close intent's ``_order_mapping`` slot, and its
                # envelope state, the stale anchors survive in
                # :attr:`envelopes` / :attr:`_persisted_envelope_anchors`
                # and the next same-key dispatch after restart would either
                # reuse the prior ``client_order_id`` (and get deduped by
                # the broker) or attach to the already-closed order via
                # :meth:`_find_key_for_order_id`.
                #
                # FIFO-aware cleanup, mirroring
                # :meth:`_route_defensive_close_fill` and the post-FILL
                # retry branch in
                # :meth:`_retry_residual_cleanup_after_transient_fill`.
                # ``marker.fifo_closed_entry_ids`` was persisted alongside
                # ``fill_observed`` by the prior instance's FILL handler;
                # it names the exact entries whose FIFO ``Trade`` rows the
                # defensive close consumed. In pyramiding (older + newer
                # same-side entries), FIFO closes the oldest first — so
                # cleaning ``marker.entry_id`` here would clear the wrong
                # entry's intents (the survivor's TP/SL protection) while
                # leaving the actually-closed entry's intents orphaned.
                # The post-FILL retry branch already applies the same
                # logic — keeping the two replay paths divergent would
                # let a crash between audit-write and cleanup corrupt
                # state, while a crash a few lines later would not. Fall
                # back to ``marker.entry_id`` only when the snapshot is
                # empty (no-FIFO / degenerate path, or replayed from an
                # older schema that did not persist the field).
                surviving_entry_ids = {
                    trade.entry_id
                    for trade in self._position.open_trades
                    if trade.entry_id is not None
                }
                cleanup_targets: list[str] = [
                    eid
                    for eid in marker.fifo_closed_entry_ids
                    if eid not in surviving_entry_ids
                ]
                if (not cleanup_targets
                        and not marker.fifo_closed_entry_ids):
                    cleanup_targets.append(marker.entry_id)
                for cleanup_id in cleanup_targets:
                    self._cleanup_position_tracking(cleanup_id)
                close_intent_key = marker.close_intent_key
                if close_intent_key:
                    self._order_mapping.pop(close_intent_key, None)
                    self._drop_envelope(close_intent_key)
                # Clear this marker's persisted ``defensive_close_pending``
                # extras inline rather than round-tripping through
                # :meth:`_clear_pending_defensive_close`. A sibling live
                # marker for the same Pine entry id (pyramiding under one
                # entry id, where each fill carries its own
                # ``position_coid``) may already be armed in
                # :attr:`_pending_defensive_close` from an earlier iteration
                # of this loop — ``iter_live_orders()`` row order is not
                # guaranteed. The set + clear pattern would overwrite that
                # live marker with the settled one and then pop the shared
                # entry-id key, losing both
                # :attr:`_pending_defensive_close` and
                # :attr:`_replayed_defensive_close_entry_ids` tracking for
                # the surviving live close. Only THIS marker's per-row
                # extras need clearing here; the live sibling's row keeps
                # its own ``defensive_close_pending`` payload under its
                # distinct ``position_coid``.
                settled_position_coid = (
                    marker.reject_context.position_coid
                )
                settled_row = self._store_ctx.get_order(
                    settled_position_coid,
                )
                if settled_row is not None:
                    settled_extras = dict(settled_row.extras or {})
                    if 'defensive_close_pending' in settled_extras:
                        settled_extras.pop('defensive_close_pending')
                        self._store_ctx.upsert_order(
                            settled_position_coid,
                            extras=settled_extras,
                        )
                continue
            existing = self._pending_defensive_close.get(marker.entry_id)
            if (existing is not None
                    and existing.reject_context.position_coid
                    != marker.reject_context.position_coid):
                # Same Pine entry id, different ``position_coid``: prior
                # instance persisted two concurrent defensive-close
                # markers (pyramiding fills under the same entry id, both
                # bracket-attach-rejected). The in-memory map is keyed by
                # ``entry_id`` alone, so re-arming would silently
                # overwrite the first marker — both closes would lose
                # tracking. Halt: manual intervention is required to
                # settle the duplicate pendings safely.
                halt = BrokerManualInterventionError(
                    f"defensive_close_pending replay: duplicate markers "
                    f"for Pine entry id {marker.entry_id!r} — existing "
                    f"position_coid="
                    f"{existing.reject_context.position_coid!r}, new "
                    f"position_coid="
                    f"{marker.reject_context.position_coid!r}. Cannot "
                    f"re-arm both with the current entry-id-keyed map; "
                    f"manual intervention required.",
                    intent_key=marker.close_intent_key,
                    context={
                        'entry_id': marker.entry_id,
                        'existing_position_coid':
                            existing.reject_context.position_coid,
                        'existing_close_intent_key':
                            existing.close_intent_key,
                        'new_position_coid':
                            marker.reject_context.position_coid,
                        'new_close_intent_key': marker.close_intent_key,
                    },
                )
                self._record_halt(halt)
                raise halt
            if marker.fill_observed:
                # Prior instance observed the defensive-close FILL (and
                # seeded the in-memory duplicate-fill caches via
                # :meth:`_mark_defensive_close_settled`) but crashed
                # BEFORE the audit event + cleanup landed — see the
                # ``fill_observed`` persistence in
                # :meth:`_route_defensive_close_fill`. Re-seed the
                # duplicate-fill caches so the post-restart polled /
                # WS replay of the same FILL is filtered out by
                # :meth:`_is_duplicate_defensive_close_fill`, then
                # re-arm the marker so
                # :meth:`_retry_residual_cleanup_after_transient_fill`
                # finishes the documented post-FILL lifecycle (residual
                # cancel + audit + marker drop) via its FILL-side
                # ``cache_seeded`` branch on the next reconcile. Without
                # this, the marker would fall through to the pre-FILL
                # re-arm path below and the engine would wait forever
                # for a FILL that already landed.
                _blog_warning(
                    "defensive_close_pending replay: %s observed FILL in "
                    "prior run instance but audit missing — re-seeding "
                    "duplicate-fill caches and re-arming marker for "
                    "post-FILL retry",
                    marker.entry_id,
                )
                if marker.close_intent_key:
                    self._settled_defensive_close_pine_ids.add(
                        marker.close_intent_key,
                    )
                if marker.close_order_ref is not None:
                    self._settled_defensive_close_order_refs.add(
                        marker.close_order_ref,
                    )
                if marker.close_client_order_id is not None:
                    self._settled_defensive_close_client_order_ids.add(
                        marker.close_client_order_id,
                    )
                # Seed the dedup cache from the actual observed FILL
                # ``order.id`` too. Polled-orders fallback / broker
                # rekey can deliver the FILL with an id different
                # from ``close_order_ref``; without this entry a
                # delayed replay of that same FILL after restart
                # would slip past
                # :meth:`_is_duplicate_defensive_close_fill` when its
                # ``order.client_order_id`` is also absent / unknown.
                if marker.fill_exchange_order_id is not None:
                    self._settled_defensive_close_order_refs.add(
                        marker.fill_exchange_order_id,
                    )
                # Re-seed the late-parent-ENTRY-fill neutralisation cache.
                # The prior instance's no-FIFO settle path called
                # :meth:`_mark_parent_entry_neutralised` so a delayed
                # parent ENTRY ``filled`` / ``partial`` arriving after the
                # close FILL would be discarded by
                # :meth:`_is_neutralised_parent_entry_fill` before reaching
                # ``record_fill``. That in-memory cache is lost across a
                # restart; the audit-driven seed at the top of
                # :meth:`_replay_pending_defensive_closes` only fires for
                # markers whose ``defensive_close_filled`` audit row
                # landed, but ``fill_observed=True`` proves the FILL was
                # observed in the prior instance precisely BEFORE the
                # audit was written. Without this seed, a delayed parent
                # ENTRY fill drained by the post-restart :meth:`reconcile`
                # (which runs before
                # :meth:`_retry_residual_cleanup_after_transient_fill`
                # clears the marker) slips past the
                # :meth:`_is_neutralised_parent_entry_fill` gate and
                # reaches ``record_fill``, opening a phantom trade
                # against the already-settled position.
                #
                # Pine ids are intentionally NOT seeded here — they are
                # user-script-controlled and reusable across dispatches.
                # The prior process could have already re-dispatched the
                # same pine id (clearing the in-memory set via
                # :meth:`_dispatch_new`) before crashing; re-seeding it
                # on restart would silently drop the new dispatch's
                # legitimate fill. Only the unique
                # ``position_coid`` is seeded — every dispatch mints a
                # fresh COID, so stale COIDs in the cache cannot collide
                # with future legitimate identifiers.
                parent_position_coid = marker.reject_context.position_coid
                if parent_position_coid:
                    self._neutralised_parent_entry_coids.add(parent_position_coid)
                self._pending_defensive_close[marker.entry_id] = marker
                self._replayed_defensive_close_entry_ids.add(marker.entry_id)
                continue
            _blog_warning(
                "defensive_close_pending replay: re-arming marker for "
                "entry %s (close_intent_key=%s, pending_since=%s)",
                marker.entry_id, marker.close_intent_key,
                marker.pending_since,
            )
            self._pending_defensive_close[marker.entry_id] = marker
            self._replayed_defensive_close_entry_ids.add(marker.entry_id)
            # Mirror the dispatch-time gate (see
            # :meth:`_handle_bracket_attach_after_fill_reject` — residual
            # cancel only fires on ``dispatch_succeeded == True``): if the
            # prior instance parked the synthetic close (``close_order_ref
            # is None``) and never recovered it, we have NO evidence the
            # close actually landed on the exchange. Cancelling residual
            # TP/SL/partial-remainder orders now would create an
            # unprotected-position window between this replay and the
            # eventual :meth:`_maybe_attach_defensive_close_ref` /
            # :meth:`_maybe_run_attached_defensive_close_cleanup` /
            # :meth:`_route_defensive_close_fill` callback that confirms
            # the close is live. Defer to those runtime paths — they own
            # the parked-recovery residual cleanup contract. The
            # ``residual_cleanup_pending`` escape hatch still re-runs the
            # cancel for markers a prior instance already proved residual
            # cleanup was due on (transient failure stamped the flag
            # AFTER ``close_order_ref`` was set).
            if (marker.close_order_ref is None
                    and not marker.residual_cleanup_pending):
                continue
            residual_clean = self._cancel_bracket_reject_residuals(
                marker.reject_context,
            )
            if not residual_clean and not marker.residual_cleanup_pending:
                # Persist the retry flag so a second restart before the
                # FILL still drives :meth:`_retry_residual_cleanup_after_transient_fill`
                # rather than waiting for the close to land.
                self._set_pending_defensive_close(
                    marker.entry_id,
                    dataclasses.replace(
                        marker, residual_cleanup_pending=True,
                    ),
                )
            elif residual_clean and marker.residual_cleanup_pending:
                # Replay cancelled the residuals cleanly on this instance —
                # clear the stale retry flag a prior crash left armed,
                # otherwise every subsequent reconcile before the FILL lands
                # would re-enter :meth:`_retry_residual_cleanup_after_transient_fill`
                # and redundantly re-cancel refs that are already gone.
                self._set_pending_defensive_close(
                    marker.entry_id,
                    dataclasses.replace(
                        marker, residual_cleanup_pending=False,
                    ),
                )

    def _retry_residual_cleanup_after_transient_fill(self) -> None:
        """Replay residual cleanup for markers stuck after a transient blip.

        Two distinct retry windows feed this method:

        - **Post-FILL retry.** :meth:`_route_defensive_close_fill` seeds
          the duplicate-fill cache BEFORE running the final residual
          cancel and returns without clearing the marker when the cancel
          raises a transient
          :class:`ExchangeConnectionError` /
          :class:`OrderDispositionUnknownError`. In that state the
          marker's ``close_intent_key`` is already in
          :attr:`_settled_defensive_close_pine_ids` (duplicate FILLs are
          ignored before re-entering the FILL routing path); the
          ``cache_seeded`` branch finishes the documented lifecycle on
          retry success.
        - **Pre-FILL retry.** Dispatch-time
          :meth:`_handle_bracket_attach_after_fill_reject` and replay-time
          :meth:`_replay_pending_defensive_closes` stamp
          ``residual_cleanup_pending=True`` when their non-raising
          :meth:`_cancel_bracket_reject_residuals` call swallowed a
          transient failure. Here the FILL has not landed yet, so the
          duplicate-fill caches are empty — without this branch the
          dispatch-time guard would leave parent/TP/SL/partial-remainder
          orders live for the entire dispatch → FILL window. The retry
          only clears the flag; the FILL-routing handler still owns the
          audit + marker-drop lifecycle.

        On either path another transient failure leaves the marker armed
        for the next reconcile (the plugin's
        :meth:`~pynecore.core.plugin.broker.BrokerPlugin.cancel_broker_order_ref`
        idempotency contract guarantees re-cancelling already-cancelled
        refs is safe). :class:`BrokerManualInterventionError` and any
        other exception propagate exactly like the FILL-routing path.
        """
        if not self._pending_defensive_close:
            return
        for entry_id in list(self._pending_defensive_close):
            marker = self._pending_defensive_close.get(entry_id)
            if marker is None:
                continue
            cache_seeded = (
                marker.close_intent_key in self._settled_defensive_close_pine_ids
                or (marker.close_order_ref is not None
                    and marker.close_order_ref
                    in self._settled_defensive_close_order_refs)
                or (marker.close_client_order_id is not None
                    and marker.close_client_order_id
                    in self._settled_defensive_close_client_order_ids)
            )
            if not cache_seeded and not marker.residual_cleanup_pending:
                continue
            try:
                self._cancel_bracket_reject_residuals(
                    marker.reject_context, raise_on_transient=True,
                )
            except (ExchangeConnectionError, OrderDispositionUnknownError) as exc:
                _blog_warning(
                    "residual cleanup retry still transiently failing for "
                    "entry %s (%s: %s) — marker stays armed for the next "
                    "reconcile / stale-grace halt",
                    marker.entry_id, type(exc).__name__, exc,
                )
                continue
            if not cache_seeded:
                # Pre-FILL retry succeeded. Clear the flag but keep the
                # marker armed — the FILL itself still has to arrive,
                # and ``_route_defensive_close_fill`` owns the audit +
                # marker-drop lifecycle. Skipping the flag clear would
                # cause every subsequent reconcile to redundantly
                # re-cancel the (now empty) residual set.
                self._set_pending_defensive_close(
                    entry_id,
                    dataclasses.replace(
                        marker, residual_cleanup_pending=False,
                    ),
                )
                continue
            # Cleanup finally succeeded — finish the documented lifecycle.
            if self._store_ctx is not None:
                # Mirror the FILL-time audit payload from
                # :meth:`_route_defensive_close_fill`. When the observed
                # FILL ``event.order.id`` differed from
                # ``marker.close_order_ref`` (polled-orders fallback /
                # broker rekey, or parked-close marker with
                # ``close_order_ref=None``), the FILL handler stashed the
                # observed id on the marker as
                # ``fill_exchange_order_id`` and would have persisted it
                # under the same payload key. The pre-FILL retry branch
                # leaves it unset (no FILL observed yet), but the
                # post-FILL retry branch — entered when the FILL routed
                # first, the residual cancel failed transiently, and the
                # marker survived to this reconcile — owns the durable
                # audit write. Dropping the key here means startup
                # replay only seeds the duplicate-fill cache from
                # ``marker.close_order_ref`` /
                # ``marker.close_client_order_id``; a delayed WS /
                # polled-orders replay of the same FILL that carries
                # only the observed ``order.id`` (and no client id)
                # would slip past
                # :meth:`_is_duplicate_defensive_close_fill` and reach
                # ``record_fill``, corrupting position state.
                audit_exchange_order_id = (
                    marker.close_order_ref
                    if marker.close_order_ref is not None
                    else marker.fill_exchange_order_id
                )
                payload: dict[str, object] = {
                    'entry_id': marker.entry_id,
                    'symbol': marker.reject_context.symbol,
                    'position_side': marker.reject_context.position_side,
                    'qty': marker.reject_context.qty,
                    'close_client_order_id': marker.close_client_order_id,
                }
                if (marker.fill_exchange_order_id is not None
                        and marker.fill_exchange_order_id
                        != marker.close_order_ref):
                    payload['fill_exchange_order_id'] = (
                        marker.fill_exchange_order_id
                    )
                self._store_ctx.log_event(
                    kind='defensive_close_filled',
                    intent_key=marker.close_intent_key,
                    client_order_id=marker.reject_context.position_coid,
                    exchange_order_id=audit_exchange_order_id,
                    payload=payload,
                )
            # FIFO-aware cleanup, mirroring the in-flight
            # :meth:`_route_defensive_close_fill` path. The marker's
            # ``fifo_closed_entry_ids`` snapshot was captured at FILL
            # time before any later :meth:`record_fill` overwrote the
            # live ``_last_fifo_closed_entry_ids`` cache, so it still
            # names the exact entries whose FIFO ``Trade`` rows this
            # defensive close consumed. In pyramiding (older + newer
            # same-side entries), FIFO closes the oldest first — so
            # cleaning ``marker.entry_id`` here would clear the wrong
            # entry's intents while leaving the actually-closed
            # entry's intents orphaned. Apply the same partial-close
            # survivor guard as the in-flight path: skip ids that
            # still appear in ``open_trades`` (residual after split).
            # Fall back to ``marker.entry_id`` only when the snapshot
            # is empty (no-FIFO / degenerate path, or replayed from an
            # older schema that did not persist the field).
            surviving_entry_ids = {
                trade.entry_id
                for trade in self._position.open_trades
                if trade.entry_id is not None
            }
            cleanup_targets: list[str] = [
                eid
                for eid in marker.fifo_closed_entry_ids
                if eid not in surviving_entry_ids
            ]
            if not cleanup_targets and not marker.fifo_closed_entry_ids:
                cleanup_targets.append(marker.entry_id)
            for cleanup_id in cleanup_targets:
                self._cleanup_position_tracking(cleanup_id)
            close_coid = marker.close_client_order_id
            if close_coid is not None:
                self._pending_verification.pop(close_coid, None)
                self._persisted_pending_anchors.pop(close_coid, None)
                if self._store_ctx is not None:
                    self._store_ctx.record_unpark(close_coid)
            # Mirror the synthetic-close mapping cleanup the in-flight
            # FILL settlement path runs (see ``_route_defensive_close_fill``):
            # without dropping the close intent's ``_order_mapping`` slot
            # and envelope, the filled defensive close would survive as a
            # live mapping (any later broker event for the same order ref
            # would route via :meth:`_find_key_for_order_id` as if a live
            # intent still existed) and as a persisted anchor that startup
            # replay would resurrect.
            close_intent_key = marker.close_intent_key
            if close_intent_key:
                self._order_mapping.pop(close_intent_key, None)
                self._drop_envelope(close_intent_key)
            self._clear_pending_defensive_close(entry_id)

    def _broker_matches_post_close_expectation(
            self,
            exch_pos: ExchangePosition | None,
            stale: list[PendingDefensiveClose],
    ) -> list[PendingDefensiveClose] | None:
        """Return the list of extra markers to settle alongside ``stale``
        when the broker snapshot is consistent with all ``stale``
        defensive closes having filled, or ``None`` when no consistent
        interpretation exists.

        Used by :meth:`_raise_if_stale_pending_defensive_close` in
        place of the original "broker must be flat" check, which
        false-halted pyramiding/multi-entry runs where a successful
        defensive close for one entry only *reduces* the aggregate
        netted position rather than flattening it.

        Two-branch contract:

        - **Single-entry / fully flat.** The broker reports no row or
          ``size == 0``. This is the original semantic and stays
          valid for the common case where only the to-be-closed
          entry was open.
        - **Multi-entry (pyramiding).** The engine's
          :attr:`_position.size` is the pre-close signed view
          (defensive-close FILLs have not yet been routed while these
          markers are still armed). For every stale marker, a filled
          defensive close changes that signed view by ``-qty`` when
          the open side is ``"buy"`` (long shrinks) or ``+qty`` when
          the open side is ``"sell"`` (short shrinks). The broker's
          signed position should land within a small tolerance of
          that expected value. If the stale-only expectation does
          NOT match, the method additionally enumerates every subset
          of non-stale pending markers (polled-orders / WS-gap brokers
          can reduce the snapshot by sibling closes whose FILL has not
          yet routed); a match against any such superset is accepted
          AND the matched non-stale markers are returned to the caller
          so they get settled in the same pass. Leaving them armed
          while the broker snapshot has already been adopted would
          double-subtract their qty on the next grace check and
          false-halt the run.

        Returns ``None`` on any inconsistency so the caller falls
        through to the manual-intervention halt — the conservative
        choice when neither branch can prove the closes settled.
        """
        # Decode broker snapshot to a signed scalar.
        if exch_pos is None:
            broker_signed = 0.0
            broker_flat = True
        else:
            broker_abs = float(exch_pos.size)
            broker_side = (exch_pos.side or "").lower()
            if broker_abs == 0.0 or broker_side == "flat":
                broker_signed = 0.0
                broker_flat = True
            elif broker_side == "long":
                broker_signed = broker_abs
                broker_flat = False
            elif broker_side == "short":
                broker_signed = -broker_abs
                broker_flat = False
            else:
                # Unrecognised side label with non-zero size — refuse
                # to settle, fall through to the explicit halt rather
                # than guess the sign.
                return None
        # Branch 1: original "broker is flat" contract.
        if broker_flat:
            # A flat broker snapshot proves every open position on the
            # symbol has closed — including any non-stale sibling
            # markers whose own grace window has not yet expired. If
            # we returned ``[]`` here, the caller would settle only the
            # stale markers and then ``_sync_position_size_to_broker``
            # would zero the engine view, leaving any non-stale
            # sibling markers armed for no reason — they would keep
            # blocking same-entry re-entry or ``close_all`` until
            # their own grace window fires. Surface them as extras so
            # the caller settles the whole batch atomically.
            stale_ids = {m.entry_id for m in stale}
            return [
                m for m in self._pending_defensive_close.values()
                if m.entry_id not in stale_ids
            ]
        # Branch 2: pyramiding multi-entry — match against engine's
        # expected post-close signed view.
        engine_signed = float(self._position.size)
        # Restart pre-adopt rescue: the stale-grace check runs in
        # :meth:`reconcile` BEFORE the startup adoption block (see
        # :meth:`_adopt_size_with_replayed_close`), so a fresh process
        # whose stale markers were re-armed by
        # :meth:`_replay_pending_defensive_closes` still has
        # :attr:`_position.size` at its zero-default here. Pyramiding
        # makes that fatal: with a prior process that posted a close
        # while the broker held size 2 and the close reduced (but did
        # not flatten) the aggregate to 1, no FILL/audit replay arrives
        # on a polled-orders broker — ``engine_signed = 0`` would yield
        # an expected post-close of ``-qty`` against a broker signed of
        # ``+1``, mismatching and false-halting. When every stale marker
        # carries replay provenance AND a persisted
        # :attr:`PendingDefensiveClose.pre_close_position_size` (i.e. the
        # prior process's engine view at marker creation), use the
        # max-abs pre-close anchor across all replayed markers (matching
        # the convention in :meth:`_adopt_size_with_replayed_close`) as
        # the pre-close baseline. Constrain to ``_position.size == 0``
        # so intra-process markers (where ``_position.size`` IS the
        # pre-close aggregate) keep the existing semantics.
        if (
            engine_signed == 0.0
            and stale
            and all(
                m.entry_id in self._replayed_defensive_close_entry_ids
                and m.pre_close_position_size is not None
                for m in stale
            )
        ):
            replayed_with_pre_close = [
                m for m in self._pending_defensive_close.values()
                if m.entry_id in self._replayed_defensive_close_entry_ids
                and m.pre_close_position_size is not None
            ]
            if replayed_with_pre_close:
                engine_signed = max(
                    (float(m.pre_close_position_size)  # type: ignore[arg-type]
                     for m in replayed_with_pre_close),
                    key=abs,
                )

        def _marker_delta(marker: PendingDefensiveClose) -> float | None:
            """Signed delta a filled close for ``marker`` would apply to
            :attr:`_position.size`. ``None`` for an unrecognised side.

            When a no-FIFO defensive close has already booked partial
            fills, :attr:`_position.size` was reduced by
            ``partial_filled_qty`` in the partial-fill branch and the
            marker records that cumulative slice. The terminal fill —
            still pending here — will only apply the REMAINING
            ``reject_context.qty - partial_filled_qty``. Subtracting the
            full close qty would double-count the partials and produce
            an expected post-close value that is off by exactly the
            already-booked slice, false-halting in non-flat pyramiding
            scenarios where the broker reports the correct post-close
            size. On single-shot closes ``partial_filled_qty`` is 0.0
            so the remaining qty equals the full reject context qty.
            """
            side = marker.reject_context.position_side
            qty = float(marker.reject_context.qty)
            partial = float(marker.partial_filled_qty)
            remaining = qty - partial
            if remaining < 0.0:
                remaining = 0.0
            if side == "buy":
                return -remaining
            if side == "sell":
                return remaining
            return None

        stale_delta = 0.0
        for marker in stale:
            delta = _marker_delta(marker)
            if delta is None:
                # Unknown side label — refuse to settle.
                return None
            stale_delta += delta
        expected_signed = engine_signed + stale_delta
        # Tolerance scales with the larger of the compared magnitudes
        # so float arithmetic does not trip the check on legitimate
        # matches, while still rejecting leftover qty greater than
        # ~1 lot.
        def _matches(expected: float) -> bool:
            scale = max(abs(expected), abs(broker_signed), 1.0)
            return abs(broker_signed - expected) <= scale * 1e-6 + 1e-9
        if _matches(expected_signed):
            return []
        # Non-stale pending markers may also have filled at the broker
        # even though their ``watch_orders`` FILL has not arrived yet
        # (polled-orders broker, transient WS gap). In that case the
        # broker snapshot has been reduced by both stale and non-stale
        # closes, so the stale-only expectation above under-shoots.
        # Try every subset of non-stale pending markers combined with
        # the full stale set; if any combined expectation matches the
        # broker, the snapshot is consistent with the stale closes
        # having filled — AND the matched non-stale markers must also
        # be settled, because the caller will adopt the broker snapshot
        # (which already reflects their fills). Leaving them armed with
        # ``_position.size`` reduced would cause the next grace-window
        # check to subtract their qty a second time and false-halt the
        # run.
        stale_ids = {m.entry_id for m in stale}
        non_stale: list[PendingDefensiveClose] = [
            m for m in self._pending_defensive_close.values()
            if m.entry_id not in stale_ids
        ]
        if non_stale:
            non_stale_deltas: list[float] = []
            for marker in non_stale:
                delta = _marker_delta(marker)
                if delta is None:
                    # Unknown side on a sibling marker — skip the
                    # combinatorial search and fall through to the
                    # restart-safe fallback / halt below.
                    non_stale_deltas = []
                    break
                non_stale_deltas.append(delta)
            # Cap combinatorial cost: with more than 12 sibling pending
            # markers the 2**N enumeration would explode, and that many
            # concurrent unsettled defensive closes is itself a strong
            # halt signal — fall through to manual intervention.
            if non_stale_deltas and len(non_stale_deltas) <= 12:
                base = expected_signed
                # Skip the empty subset (already covered by the
                # stale-only ``_matches`` above) — start from size 1.
                indices = range(len(non_stale_deltas))
                # Require a UNIQUE matching subset. With equal-size
                # pyramiding siblings, multiple sibling markers can
                # share the same signed delta, so two or more disjoint
                # subsets can produce the exact same candidate sum and
                # match the broker's aggregate. In that case the
                # aggregate size alone cannot identify which sibling
                # close actually filled — picking the first matching
                # combination would settle (and clean up the tracking
                # for) the wrong entry while leaving the truly filled
                # marker armed, eventually false-halting on the stale
                # check. Enumerate ALL combinations, then return a
                # match only when exactly one subset matches; on
                # ambiguity fall through to the restart-safe fallback
                # / manual intervention halt.
                matches: list[list[PendingDefensiveClose]] = []
                for r in range(1, len(non_stale_deltas) + 1):
                    for combo in itertools.combinations(indices, r):
                        candidate = base + sum(
                            non_stale_deltas[i] for i in combo
                        )
                        if _matches(candidate):
                            matches.append(
                                [non_stale[i] for i in combo]
                            )
                            if len(matches) > 1:
                                break
                    if len(matches) > 1:
                        break
                if len(matches) == 1:
                    return matches[0]
        # Restart-safe fallback: the docstring assumes ``_position.size``
        # is the pre-close aggregate, but a startup-adopted view can
        # already reflect the broker's post-close reduction (see
        # :meth:`reconcile` adopt branch). In that scenario the
        # subtraction above over-corrects: ``engine_signed`` itself
        # already equals ``broker_signed``.
        #
        # Critically, ``engine_signed == broker_signed`` is NOT by itself
        # proof the broker snapshot is post-close — it could equally mean
        # the close never landed and the engine adopted the broker's
        # pre-close size. Only :attr:`PendingDefensiveClose.fill_observed`
        # is evidence the FILL was actually seen by the prior process
        # (:meth:`_route_defensive_close_fill` flips and persists the flag
        # before the residual-cancel + audit lifecycle that may have been
        # interrupted by the crash). Require every stale marker to carry
        # both replay provenance (intra-process markers cannot have crossed
        # a restart-induced adopt) AND ``fill_observed=True`` (proof the
        # broker actually reduced); without both, leave the conservative
        # subtraction in place so a genuinely stuck close still halts.
        all_replayed_post_fill = stale and all(
            m.entry_id in self._replayed_defensive_close_entry_ids
            and m.fill_observed
            for m in stale
        )
        if all_replayed_post_fill and _matches(engine_signed):
            return []
        return None

    def _raise_if_stale_pending_defensive_close(self) -> bool:
        """Halt the run if any pending defensive-close marker has exceeded
        the resolution grace window.

        The grace value comes from the plugin's
        ``defensive_close_resolution_grace_s`` class attribute when set;
        otherwise the engine default
        (:data:`DEFENSIVE_CLOSE_RESOLUTION_GRACE_S`) applies.

        Before halting, the broker position is probed: a long restart
        gap or a polled-orders broker (no queued ``watch_orders`` FILL)
        can leave the exchange already flat with the settling FILL not
        yet routed. In that case the close is in fact complete — the
        only missing piece is the audit + marker drop — so each stale
        marker is settled via :meth:`_settle_marker_from_flat_broker`
        instead of halting. If :meth:`get_position` itself fails, we
        cannot prove the broker is flat and fall through to the
        original manual-intervention halt.

        :returns: ``True`` when settlement was deferred this pass
            because the final residual cancel still failed transiently
            (markers remain armed). The caller MUST skip adopting the
            broker snapshot into :attr:`_position.size` in that case
            — the broker view is already reduced by the (in fact
            filled) close, but the marker still represents that qty
            for the next grace probe; adopting now would shift
            ``engine_signed`` past the post-close expectation and
            false-halt the next reconcile via
            :meth:`_broker_matches_post_close_expectation`. ``False``
            otherwise (no markers, no stale markers, halted, or
            settled cleanly).
        """
        if not self._pending_defensive_close:
            return False
        grace = getattr(
            self._broker, 'defensive_close_resolution_grace_s', None,
        )
        if grace is None:
            grace = DEFENSIVE_CLOSE_RESOLUTION_GRACE_S
        now = time.time()
        stale = [
            m for m in self._pending_defensive_close.values()
            if now - m.pending_since > grace
        ]
        if not stale:
            return False
        # Probe the broker snapshot before halting. A poll-based broker
        # or a long restart gap can leave the position already flat
        # while no FILL event has been queued for the routing path —
        # halting in that state would label a successful close as
        # manual intervention.
        #
        # Pyramiding extension: with multiple open entries on the same
        # symbol (Capital.com aggregates rows into one netted position),
        # a successful defensive close for ONE entry reduces — but does
        # not flatten — the aggregate. Settling only on
        # ``size == 0`` would false-halt those scripts. We therefore
        # accept "broker has been reduced by exactly the stale markers'
        # net qty" as proof the closes filled, using
        # :attr:`_position.size` as the engine's pre-close signed view
        # (FILLs have not yet been routed, so it still includes the
        # to-be-closed qty).
        try:
            exch_pos = self._run_async(
                self._broker.get_position(self._symbol),
            )
        except Exception as exc:  # noqa: BLE001 — diagnostic fallback only
            _blog_warning(
                "stale-grace pre-check: get_position raised %s: %s — "
                "falling through to manual-intervention halt",
                type(exc).__name__, exc,
            )
            exch_pos = None
            extra_to_settle: list[PendingDefensiveClose] | None = None
        else:
            extra_to_settle = self._broker_matches_post_close_expectation(
                exch_pos, stale,
            )
        if extra_to_settle is not None:
            # ``extra_to_settle`` carries non-stale sibling markers whose
            # fills the broker snapshot already reflects (subset-match
            # branch). They must be settled in the same pass as the stale
            # set — leaving them armed while ``_sync_position_size_to_broker``
            # adopts the reduced broker size would let the next grace
            # check subtract their qty a second time and false-halt.
            settle_now: list[PendingDefensiveClose] = list(stale) + list(extra_to_settle)
            _blog_warning(
                "stale defensive-close marker(s) past grace=%.1fs but "
                "broker snapshot matches expected post-close state — "
                "settling %d marker(s) from snapshot instead of halting "
                "(stale=%d, sibling=%d)",
                grace, len(settle_now), len(stale), len(extra_to_settle),
            )
            # Before writing the settlement audit, run one final residual
            # cancel per marker. The transient-blip retry in
            # :meth:`_retry_residual_cleanup_after_transient_fill` may
            # have just swallowed a transient failure (leaving live
            # parent/TP/SL/partial-remainder orders) while the broker
            # snapshot independently reports flat — emitting the
            # ``defensive_close_filled`` audit in that state would mark
            # recovery complete and let any future restart skip residual
            # cleanup, leaving the residual orders live forever. We use
            # ``raise_on_transient=True`` so the marker stays armed for
            # the next reconcile (and the grace window keeps probing the
            # broker) when the residual cancel cannot be confirmed.
            #
            # Two-pass: first probe residual cleanup for every marker,
            # only commit settlement when ALL succeed. Settling a subset
            # while another marker stays armed would clear that subset's
            # FIFO ``open_trades`` rows and bump audits while leaving
            # :attr:`_position.size` at the pre-close aggregate (the
            # ``_sync_position_size_to_broker`` catch-up below is skipped
            # whenever any marker is still armed). The next reconcile
            # would then see only the still-armed marker as stale and
            # ``_broker_matches_post_close_expectation`` would subtract
            # only that marker's qty from ``engine_signed`` — falsely
            # halting because the broker snapshot has already been
            # reduced by the previously-settled marker too. Plugin
            # contract guarantees both enumeration and per-ref cancel
            # are idempotent, so re-running the already-succeeded
            # residual cleanup on the next reconcile is safe.
            still_armed: list[PendingDefensiveClose] = []
            for marker in settle_now:
                try:
                    self._cancel_bracket_reject_residuals(
                        marker.reject_context, raise_on_transient=True,
                    )
                except (ExchangeConnectionError, OrderDispositionUnknownError) as exc:
                    _blog_warning(
                        "flat-broker settlement skipped for entry %s — "
                        "residual cleanup still transiently failing "
                        "(%s: %s); marker stays armed for the next "
                        "reconcile",
                        marker.entry_id, type(exc).__name__, exc,
                    )
                    still_armed.append(marker)
            if still_armed:
                # At least one marker could not be probed cleanly. Defer
                # settling ANY marker so the next reconcile retries the
                # whole batch atomically and the engine view stays at the
                # pre-close aggregate (avoids the multi-marker subtraction
                # bug above). Signal the caller to skip startup-adoption
                # of the broker snapshot: the broker view is already
                # reduced by the (in fact filled) close, but the marker
                # still carries that qty for the next grace probe — if
                # ``reconcile`` adopted the reduced size into
                # :attr:`_position.size` now, the next
                # :meth:`_broker_matches_post_close_expectation` would
                # subtract the marker qty again and false-halt.
                return True
            for marker in settle_now:
                self._settle_marker_from_flat_broker(marker)
            # Sync the in-memory position size to the broker snapshot we
            # just used to prove settlement. ``_settle_marker_from_flat_broker``
            # cleans the marker / mapping / verification rows but does NOT
            # touch ``BrokerPosition.size``; for the single-entry "broker
            # flat" branch the engine view is usually already at zero, but
            # pyramiding (broker reduced but not flat) and any drift would
            # otherwise leave ``_position.size`` at the pre-close aggregate.
            # The periodic-reconcile adopt-mismatch branch only acts on
            # startup, so without this catch-up the size stays wrong until
            # restart.
            self._sync_position_size_to_broker(exch_pos)
            return False
        oldest = min(stale, key=lambda m: m.pending_since)
        halt = BrokerManualInterventionError(
            f"Defensive close FILL did not arrive within {grace}s for "
            f"entry {oldest.entry_id!r} "
            f"(close_intent_key={oldest.close_intent_key!r}, "
            f"pending_since={oldest.pending_since})",
            intent_key=oldest.close_intent_key,
            context={
                'entry_id': oldest.entry_id,
                'position_coid': oldest.reject_context.position_coid,
                'symbol': oldest.reject_context.symbol,
                'pending_since': oldest.pending_since,
                'grace_s': grace,
                'stale_count': len(stale),
            },
        )
        self._record_halt(halt)
        raise halt

    def _settle_marker_from_flat_broker(
            self, marker: PendingDefensiveClose,
    ) -> None:
        """Finish a defensive-close lifecycle when the broker snapshot
        proves the position is already flat.

        Mirrors the replay-settled branch in
        :meth:`_replay_pending_defensive_closes` for an in-process
        marker: seed every duplicate-fill cache (a late
        ``watch_orders`` replay or polled-orders re-report must not
        reach ``record_fill``), drop the parked-close anchor /
        verification rows, write the missing ``defensive_close_filled``
        audit event so any future restart's replay path also treats the
        marker as settled, clear position tracking + synthetic-close
        envelope state, and finally drop the marker.
        """
        if marker.close_intent_key:
            self._settled_defensive_close_pine_ids.add(
                marker.close_intent_key,
            )
        if marker.close_order_ref is not None:
            self._settled_defensive_close_order_refs.add(
                marker.close_order_ref,
            )
        if marker.close_client_order_id is not None:
            self._settled_defensive_close_client_order_ids.add(
                marker.close_client_order_id,
            )
        close_coid = marker.close_client_order_id
        if close_coid is not None:
            self._pending_verification.pop(close_coid, None)
            self._persisted_pending_anchors.pop(close_coid, None)
            if self._store_ctx is not None:
                self._store_ctx.record_unpark(close_coid)
        if self._store_ctx is not None:
            # Stamp the parent row's ``natural_close_at`` BEFORE writing
            # the durable audit. Mirrors the synchronous-success /
            # parked-recovery / attached-resolution path. The broker
            # snapshot has now proved this defensive close landed on the
            # exchange; without the stamp, the plugin-side missing-position
            # reconciler would see the unstamped parent row as an
            # unexpected manual close and halt the run. Parked closes that
            # never went through :meth:`_maybe_attach_defensive_close_ref`
            # or an attached resolution before settling here would
            # otherwise leave the parent row indistinguishable from a
            # genuine manual close.
            #
            # Ordering matters: if the audit row is written first and the
            # process crashes before the stamp lands,
            # :meth:`_replay_pending_defensive_closes` would see the audit
            # row, clear the marker / extras, and never re-run the stamp —
            # leaving the parent row unstamped and primed for a false
            # manual-close halt on the next reconcile.
            self._stamp_natural_close_at(
                marker.reject_context.position_coid,
            )
            # Without the audit, a subsequent restart's replay would
            # re-arm this marker from extras and the stale-grace path
            # would have to re-probe broker state every reconcile.
            #
            # Persist :attr:`PendingDefensiveClose.fill_exchange_order_id`
            # when the marker observed a FILL whose ``order.id`` differs
            # from ``marker.close_order_ref`` (polled-orders fallback /
            # broker rekey). Reachable when the FILL was observed in
            # the prior call, residual cleanup failed transiently, and
            # the stale-grace probe later settles from the flat broker
            # snapshot. Without this, startup replay only seeds the
            # duplicate-fill cache from the
            # ``exchange_order_id`` / ``close_client_order_id`` columns;
            # a delayed WS / polled-orders replay carrying only the
            # observed (alternate) order id would slip past
            # :meth:`_is_duplicate_defensive_close_fill` and reach
            # :meth:`record_fill`. Mirrors the FILL-path audit write in
            # :meth:`_route_defensive_close_fill` and the post-FILL
            # retry write in
            # :meth:`_retry_residual_cleanup_after_transient_fill`.
            audit_exchange_order_id: str | None = (
                marker.close_order_ref
                if marker.close_order_ref is not None
                else marker.fill_exchange_order_id
            )
            payload: dict[str, object] = {
                'entry_id': marker.entry_id,
                'symbol': marker.reject_context.symbol,
                'position_side': marker.reject_context.position_side,
                'qty': marker.reject_context.qty,
                'close_client_order_id': marker.close_client_order_id,
                'settled_via': 'flat_broker_snapshot',
            }
            if (marker.fill_exchange_order_id is not None
                    and marker.fill_exchange_order_id
                    != marker.close_order_ref):
                payload['fill_exchange_order_id'] = (
                    marker.fill_exchange_order_id
                )
            self._store_ctx.log_event(
                kind='defensive_close_filled',
                intent_key=marker.close_intent_key,
                client_order_id=marker.reject_context.position_coid,
                exchange_order_id=audit_exchange_order_id,
                payload=payload,
            )
        # Late parent-ENTRY-fill neutralisation. The broker snapshot proved
        # the post-close state already absorbs the parent ENTRY fill (the
        # close reduced the aggregate by exactly the marker qty — see
        # :meth:`_broker_matches_post_close_expectation`), so any delayed
        # parent ENTRY ``filled`` / ``partial`` event delivered after this
        # snapshot settlement must be dropped before
        # :meth:`BrokerPosition.record_fill` reaches it. Without seeding the
        # cache, the late event would either open a phantom fresh trade
        # against an already-flat broker (no-FIFO branch above) or open one
        # against a reduced-but-not-flat broker whose contributing entries
        # are already accounted for. The FILL-side no-FIFO branch in
        # :meth:`_route_event` runs the same call on its terminal ``filled``
        # path; here the settle is always terminal (the grace probe never
        # re-arms after settle), so no partial-vs-final guard is needed.
        self._mark_parent_entry_neutralised(marker)
        # Drop the closed entry's FIFO trade(s) before the caller adopts
        # the broker snapshot. ``_cleanup_position_tracking`` only clears
        # ``entry_orders`` / ``exit_orders`` / ``_active_intents`` rows;
        # it does NOT touch ``BrokerPosition.open_trades``. In a
        # pyramiding in-process run with FIFO trades populated, leaving
        # the closed entry's ``Trade`` in ``open_trades`` while
        # :meth:`_sync_position_size_to_broker` adopts the broker's
        # reduced ``size`` would let the next ``record_fill`` close the
        # wrong trade (FIFO walks in arrival order, blind to which
        # entry's defensive close just settled) and corrupt P&L.
        # When ``open_trades`` is empty (post-restart no-FIFO state)
        # this branch is a no-op.
        #
        # Pyramiding handling: Pine allows ``strategy.entry(id="long",
        # ...)`` to fire multiple times under ``pyramiding > 0``, so
        # ``open_trades`` can hold several rows (possibly sharing the
        # same ``entry_id``). The defensive close settles ONE tranche
        # worth of qty (``marker.reject_context.qty``); we must consume
        # exactly that quantity from ``open_trades`` and preserve the
        # rest. Trades carry no per-tranche ref the marker could
        # target, so the consumption walks ``open_trades`` strictly
        # FIFO (oldest first) — see the longer rationale further down.
        #
        # Partial-fill bookkeeping: when one or more ``partial`` defensive-
        # close events already routed through the FIFO path BEFORE we
        # reach this flat-broker settlement, those fills have ALREADY
        # reduced ``open_trades`` via :meth:`BrokerPosition.record_fill`
        # and the slice accumulated on ``marker.partial_filled_qty``.
        # The broker snapshot we are settling against likewise reflects
        # only the REMAINING ``reject_context.qty - partial_filled_qty``
        # (see :meth:`_broker_matches_post_close_expectation`). Removing
        # the full original ``reject_context.qty`` here would over-
        # consume open_trades by exactly the partial slice, dropping or
        # shrinking pyramided same-id tranches that were never part of
        # this defensive close — corrupting future FIFO walks and P&L.
        #
        # FIFO-order consumption: Pine / live :meth:`record_fill` walks
        # ``open_trades`` from the OLDEST element (index 0) regardless of
        # ``entry_id``. With pyramiding (LongA older, LongB newer,
        # bracket-attach reject → defensive close armed for LongB.qty),
        # a real FILL event would route through ``record_fill`` and
        # consume LongA's tranche first. Filtering by ``marker.entry_id``
        # here would instead remove LongB's trade while LongA stays —
        # the engine view then diverges from what the broker (and any
        # later in-process FIFO close) would have produced, leaving exit
        # routing / P&L attached to the wrong tranche. Consume strictly
        # FIFO so this snapshot-settlement path mirrors the FILL-event
        # path; track the entry ids of fully-consumed trades so the
        # downstream cleanup can target the same set that
        # :meth:`_route_defensive_close_fill` would clean.
        fifo_consumed_entry_ids: list[str] = []
        if self._position.open_trades:
            close_qty_remaining = (
                float(marker.reject_context.qty)
                - float(marker.partial_filled_qty)
            )
            if close_qty_remaining < 0.0:
                close_qty_remaining = 0.0
            surviving: list = []
            removed_any = False
            for t in self._position.open_trades:
                if close_qty_remaining <= 0.0:
                    surviving.append(t)
                    continue
                trade_qty = abs(float(t.size))
                if trade_qty <= close_qty_remaining + 1e-12:
                    # Whole tranche absorbed by the close — drop it.
                    close_qty_remaining -= trade_qty
                    removed_any = True
                    if t.entry_id is not None:
                        fifo_consumed_entry_ids.append(t.entry_id)
                else:
                    # Tranche larger than the residual close qty:
                    # shrink it so FIFO walks the remainder for future
                    # closes. This mirrors the partial-close behaviour
                    # that :meth:`Position.record_fill_close` applies in
                    # the FIFO path.
                    # Prorate commission alongside size — the closed piece
                    # carries ``commission * (closed_qty / trade_qty)`` and
                    # the survivor retains the remainder. ``record_fill_close``
                    # does this via ``trade.commission -= closed_piece.commission``;
                    # without it the survivor would keep the full original
                    # commission while ``open_commission`` (recomputed below)
                    # double-counts the closed slice, overstating equity until
                    # the next fill or restart.
                    new_size = trade_qty - close_qty_remaining
                    closed_commission = (
                        t.commission * (close_qty_remaining / trade_qty)
                    )
                    if t.sign < 0:
                        t.size = -new_size
                    else:
                        t.size = new_size
                    t.commission -= closed_commission
                    close_qty_remaining = 0.0
                    surviving.append(t)
                    removed_any = True
            if removed_any:
                self._position.open_trades = surviving
                # Recompute ``open_commission`` from the surviving trades so
                # the pyramiding (broker reduced but not flat) settlement
                # path does not leave the closed entry's commission summed
                # into the live equity view.
                # ``_sync_position_size_to_broker`` only resets
                # ``open_commission`` when the broker is fully flat
                # (post-close ``size == 0``); without this catch-up the
                # reduced-but-not-flat branch keeps reporting stale
                # commission / openprofit-derived equity until restart.
                # Same canonical pattern as
                # :meth:`Position.record_fill_close`.
                self._position.open_commission = sum(
                    t.commission for t in self._position.open_trades
                )
        # FIFO-aware cleanup targets. Mirror
        # :meth:`_route_defensive_close_fill`'s logic: clean intents for
        # every entry id whose trade was fully consumed AND no longer has
        # any surviving ``open_trades`` row (a partial-close split keeps
        # the residual under the same id). When the FIFO walk consumed
        # nothing (no open trades to begin with, or all consumption
        # happened earlier via FIFO ``partial`` events accumulated on
        # ``marker.partial_filled_qty`` so ``close_qty_remaining`` was
        # already zero on entry), fall back to ``marker.entry_id`` so the
        # marker's own pine-id is still cleared. Mirrors the FIFO route's
        # degenerate-path fallback (``not _last_fifo_closed_entry_ids``
        # → cleanup marker.entry_id).
        surviving_entry_ids = {
            trade.entry_id
            for trade in self._position.open_trades
            if trade.entry_id is not None
        }
        # Union the FIFO closures from prior partial events (persisted on
        # ``marker.fifo_closed_entry_ids`` via the merge in
        # :meth:`_route_event`) with the closures observed during this
        # snapshot walk. Building ``cleanup_targets`` from
        # ``fifo_consumed_entry_ids`` alone would leave intents / Pine
        # order rows for entries closed by earlier partials in place,
        # since their tranches were already removed from
        # ``open_trades`` before we ever reached this path. Mirrors the
        # FIFO route's source (``marker.fifo_closed_entry_ids``) in
        # :meth:`_route_defensive_close_fill`.
        candidate_closed_ids: list[str] = []
        seen_ids: set[str] = set()
        for entry_id in marker.fifo_closed_entry_ids:
            if entry_id not in seen_ids:
                seen_ids.add(entry_id)
                candidate_closed_ids.append(entry_id)
        for entry_id in fifo_consumed_entry_ids:
            if entry_id not in seen_ids:
                seen_ids.add(entry_id)
                candidate_closed_ids.append(entry_id)
        cleanup_targets: list[str] = [
            entry_id
            for entry_id in candidate_closed_ids
            if entry_id not in surviving_entry_ids
        ]
        if not cleanup_targets and not candidate_closed_ids:
            # Same-id survivor guard for the degenerate-cleanup fallback.
            # Pyramiding with the same ``entry_id`` (``strategy.entry(id=
            # "long", ...)`` firing under ``pyramiding > 0``) can leave
            # surviving tranches sharing ``marker.entry_id`` even when
            # the FIFO walk recorded nothing here (e.g. all consumption
            # happened via prior FIFO ``partial`` events, accumulated on
            # ``marker.partial_filled_qty``, so the remaining close qty
            # was already zero before this loop ran).
            # ``_cleanup_position_tracking`` would otherwise drop the
            # entry / exit intents for an id that still has live FIFO
            # trades, leaving the remaining broker exposure without
            # active intents or Pine exit-order rows. Skip cleanup when
            # ``marker.entry_id`` is still referenced by any remaining
            # ``open_trades`` row.
            any_surviving_marker_trade = any(
                trade.entry_id == marker.entry_id
                for trade in self._position.open_trades
            )
            if not any_surviving_marker_trade:
                cleanup_targets.append(marker.entry_id)
        for cleanup_id in cleanup_targets:
            self._cleanup_position_tracking(cleanup_id)
        close_intent_key = marker.close_intent_key
        if close_intent_key:
            self._order_mapping.pop(close_intent_key, None)
            self._drop_envelope(close_intent_key)
        self._clear_pending_defensive_close(marker.entry_id)

    def _sync_position_size_to_broker(
            self, exch_pos: ExchangePosition | None,
    ) -> None:
        """Align :attr:`_position.size` with the broker snapshot.

        Called from the stale-grace settlement path after one or more
        markers were settled from a verified broker snapshot. The
        per-marker :meth:`_settle_marker_from_flat_broker` cleans only the
        marker / mapping / verification rows; ``BrokerPosition.size`` is
        untouched. With pyramiding the broker can be REDUCED-but-not-flat
        after a defensive close fills, and without this catch-up the
        engine view stays at the pre-close aggregate (the periodic
        reconcile's adopt-mismatch branch only acts on startup, so the
        drift survives indefinitely otherwise).

        The broker snapshot is the source of truth here — we just proved
        it matches the expected post-close state via
        :meth:`_broker_matches_post_close_expectation`, so adopting its
        signed size is safe. ``open_trades`` are not reconstructed (the
        rows live in the persisted store and are not replayed here); a
        later record-fill against the empty FIFO would take the no-FIFO
        defensive-close branch above, which now also tracks signed deltas.
        """
        if exch_pos is None:
            new_signed = 0.0
        else:
            new_abs = float(exch_pos.size)
            side = (exch_pos.side or "").lower()
            if new_abs == 0.0 or side == "flat":
                new_signed = 0.0
            elif side == "long":
                new_signed = new_abs
            elif side == "short":
                new_signed = -new_abs
            else:
                # Unknown side label — leave size untouched rather than
                # guess. ``_broker_matches_post_close_expectation`` would
                # already have returned False for an unrecognised side,
                # so we should not reach this branch in practice.
                return
        if new_signed == self._position.size:
            return
        _blog_warning(
            "stale-grace settlement adopting broker size %s (was %s)",
            new_signed, self._position.size,
        )
        self._position.size = new_signed
        if new_signed == 0.0:
            self._position.sign = 0.0
            self._position.avg_price = na_float
            self._position.open_trades.clear()
            self._position.openprofit = 0.0
            self._position.open_commission = 0.0
        else:
            self._position.sign = 1.0 if new_signed > 0.0 else -1.0
            if exch_pos is not None and exch_pos.entry_price is not None:
                self._position.avg_price = exch_pos.entry_price

    def _stamp_natural_close_at(self, position_coid: str) -> None:
        """Stamp ``natural_close_at`` on a parent entry row.

        Mirrors the synchronous-success path in
        :meth:`_handle_bracket_attach_after_fill_reject` (line 3003 in the
        committed change): once a defensive close is proven to live on the
        exchange, the parent row must opt out of the plugin-side
        missing-position reconciler. Without the stamp, a broker that
        drops the position from its snapshot before the close activity is
        ingested would trigger a false :class:`UnexpectedCancelError`
        halt after the grace window.

        Idempotent: re-stamping an already-stamped row is harmless (the
        reconciler only cares that the key is present).
        """
        if self._store_ctx is None:
            return
        row = self._store_ctx.get_order(position_coid)
        if row is None:
            return
        extras = dict(row.extras or {})
        if 'natural_close_at' in extras:
            return
        extras['natural_close_at'] = time.time()
        self._store_ctx.upsert_order(position_coid, extras=extras)

    def _clear_pending_defensive_close(self, entry_id: str) -> None:
        """Drop a defensive-close pending marker from memory + extras.

        Called from the FILL event handler when the close settles, from
        :meth:`_handle_bracket_attach_after_fill_reject` when the
        dispatch path halts, and from
        :meth:`_replay_pending_defensive_closes` when startup replay
        detects an already-settled FILL.
        """
        marker = self._pending_defensive_close.pop(entry_id, None)
        self._replayed_defensive_close_entry_ids.discard(entry_id)
        if marker is None or self._store_ctx is None:
            return
        position_coid = marker.reject_context.position_coid
        row = self._store_ctx.get_order(position_coid)
        if row is None:
            return
        extras = dict(row.extras or {})
        if 'defensive_close_pending' in extras:
            extras.pop('defensive_close_pending')
            self._store_ctx.upsert_order(position_coid, extras=extras)

    def _dispatch_modify(self, old: Intent, new: Intent) -> None:
        # Engine-trigger partial bracket exits never reach the broker —
        # the leg rows are engine-internal. A modify must therefore
        # cancel the prior legs through the state machine and re-emit
        # the new intent as a fresh dispatch. The native ``modify_exit``
        # path would otherwise call into the plugin for an order that
        # was never sent. See the partial-qty bracket exit design
        # dossier §3.3.
        if (isinstance(old, ExitIntent)
                and self._partial_bracket_engine.has_active_legs_for_intent(
                    old.intent_key,
                )):
            # Preflight the §2.6.7 fail-safe gate on the parent before
            # evicting the currently armed legs: if the broker-native
            # fail-safe is degrading / degraded / owner-unknown,
            # :meth:`_dispatch_engine_trigger_partial_bracket` would
            # refuse the replacement and raise, leaving the parent with
            # no software TP/SL protection at all. The intended policy
            # (see dossier §2.6.7) is to keep servicing the existing
            # legs while only blocking *new* bracket attachments — the
            # already-armed legs are intermediate close dispatches and
            # must continue to flow. Skip the modify on a gate hit and
            # keep the prior legs in place; the next sync re-runs the
            # diff and retries once the fail-safe recovers.
            # Resolve the parent's deterministic ``KIND_ENTRY`` COID. On
            # the restart path the parent's entry envelope is no longer
            # in ``_envelopes`` (the original dispatch was retired when
            # the entry filled in a previous run), but the persisted
            # envelope anchor preserves ``bar_ts_ms`` / ``retry_seq`` so
            # the COID rebuilt from ``_persisted_envelope_anchors`` is
            # byte-identical to the previous run's. Without this
            # fallback the §2.6.7 preflight below would silently treat
            # every cross-restart modify as "no parent registered" and
            # skip the fail-safe gate — :meth:`cancel_legs_for_intent`
            # would then evict the existing protection while
            # :meth:`_dispatch_new` may reject the replacement, leaving
            # the parent without any software bracket.
            parent_dispatch_ref: str | None = None
            if new.from_entry is not None:
                parent_dispatch_ref = self._resolve_parent_opening_ref(
                    new.from_entry,
                )
            if (isinstance(new, ExitIntent)
                    and new.is_partial_qty_bracket
                    and self._partial_qty_bracket_exit_mode
                    is CapabilityLevel.SOFTWARE):
                if parent_dispatch_ref is not None:
                    if self._native_failsafe_manager.is_new_partial_bracket_blocked(
                            parent_entry_dispatch_ref=parent_dispatch_ref,
                    ):
                        _blog_warning(
                            "engine-trigger partial bracket modify deferred for "
                            "%s -> %s: fail-safe gate blocks the replacement; "
                            "preserving the existing legs until the next sync "
                            "(parent %r)",
                            old, new, parent_dispatch_ref,
                        )
                        # Raise the deferred-modify signal so
                        # :meth:`_diff_and_dispatch` keeps ``_active_intents[key]``
                        # pointing at the OLD intent. A plain ``return`` here would
                        # let the caller promote the slot to ``new``, causing the
                        # next sync to observe Pine == active and skip the retry
                        # forever (the engine would then watch stale TP/SL/qty
                        # levels until the strategy emits a different intent).
                        raise _PartialBracketModifyDeferred(
                            "engine-trigger partial bracket modify deferred by "
                            "§2.6.7 fail-safe gate"
                        )
            # Preflight the §12 #4 coexistence guard before evicting the
            # currently armed legs. A whole-row exit replacement on a
            # parent that still carries scale-out siblings under a
            # different ``intent_key`` (TP1 / TP2 with distinct
            # ``strategy.exit(id=...)``) would land in
            # :meth:`_dispatch_new` → ``else`` branch at L5128 and raise
            # ``RuntimeError`` because :meth:`has_active_partial_bracket`
            # is still True (the sibling legs remain). The exception
            # would propagate after :meth:`cancel_legs_for_intent`
            # already evicted ``old``'s legs and
            # :meth:`_drop_envelope` cleared ``old``'s envelope,
            # leaving ``_active_intents[old.intent_key]`` pointing at
            # the OLD intent while no software protection backs it.
            # Refuse the modify here so the existing legs stay armed
            # until the script either cancels the sibling brackets or
            # re-emits a compatible partial-qty replacement.
            if (isinstance(new, ExitIntent)
                    and not new.is_partial_qty_bracket
                    and new.from_entry is not None
                    and self._partial_bracket_engine
                        .has_sibling_active_legs(
                            old.symbol,
                            new.from_entry,
                            exclude_intent_key=old.intent_key,
                        )):
                raise RuntimeError(
                    "engine-trigger partial bracket modify refuses: "
                    f"whole-row replacement for {old.intent_key!r} would "
                    f"violate §12 #4 — scale-out sibling legs remain on "
                    f"{old.symbol!r}/{new.from_entry!r}; the script must "
                    "cancel the sibling partial brackets before promoting "
                    "this slot to a whole-row exit.",
                )
            _blog_info("modifying engine-trigger partial bracket %s -> %s",
                       old, new)
            self._partial_bracket_engine.cancel_legs_for_intent(
                old.intent_key, reason='intent_modified',
            )
            self._order_mapping.pop(old.intent_key, None)
            self._drop_envelope(old.intent_key)
            # When the replacement is itself an engine-trigger partial
            # bracket the new legs will re-register the worst-SL via
            # :meth:`_recompute_native_failsafe_for_parent`; let that
            # path own the failsafe state. But when no other partial
            # legs remain on this parent (e.g. the replacement is a
            # whole-row exit, a close, or a partial bracket with
            # different shape that leaves the parent's partial set
            # empty for one tick), the listener above already queued a
            # ``stop_level=None`` clear snapshot. The next
            # :meth:`drive_native_failsafe` drain would then send that
            # clear PUT and delete a native bracket that
            # :meth:`_dispatch_new(new)` may have just attached
            # (mirrors the same race the new-dispatch
            # partial-to-whole-row conversion handles around L5020).
            # Retire the failsafe state for this parent so the queued
            # clear is dropped before the dispatch runs; the new path
            # is then free to re-register if it arms partial legs.
            if (new.from_entry is not None
                    and parent_dispatch_ref is not None
                    and not self._partial_bracket_engine
                        .has_active_partial_bracket(
                            old.symbol, new.from_entry,
                        )):
                self._native_failsafe_manager.unregister_parent(
                    parent_dispatch_ref,
                )
            self._dispatch_new(new)
            return

        # Mirror the previous branch for the opposite transition:
        # a native whole-row bracket was active and the strategy
        # re-emitted the same ``intent_key`` as a partial-qty bracket
        # (e.g. user added a ``qty`` to a ``strategy.exit`` line). The
        # native ``modify_exit`` path would push the partial bracket
        # to the plugin even though the broker mode declared partial
        # brackets must be engine-triggered. Cancel the native bracket
        # at the broker and re-dispatch through
        # :meth:`_dispatch_engine_trigger_partial_bracket` instead.
        if (isinstance(old, ExitIntent)
                and isinstance(new, ExitIntent)
                and new.is_partial_qty_bracket
                and not old.is_partial_qty_bracket
                and self._partial_qty_bracket_exit_mode
                is CapabilityLevel.SOFTWARE):
            # Preflight the §12 #4 coexistence guard before evicting the
            # currently armed native bracket. The symmetric direction
            # (whole-row replacing scale-out siblings) already runs an
            # equivalent ``has_sibling_active_legs`` preflight above; this
            # branch needs the mirror for native siblings:
            # :meth:`_dispatch_engine_trigger_partial_bracket` rejects the
            # new partial bracket with ``RuntimeError`` when another
            # native whole-row ``ExitIntent`` for the same ``from_entry``
            # (different ``intent_key``) is still active in
            # ``_active_intents``. Without this preflight, the strict
            # cancel below would already have cancelled ``old`` at the
            # broker and dropped its ``_order_mapping`` / envelope, leaving
            # ``_active_intents[old.intent_key]`` pointing at a stale
            # intent with no live broker order until the next sync.
            # Refuse the conversion here so the existing native bracket
            # stays live until the script cancels the sibling native
            # exit and re-emits a compatible partial-qty replacement.
            if new.from_entry is not None:
                for active in self._active_intents.values():
                    if not isinstance(active, ExitIntent):
                        continue
                    if active.intent_key == old.intent_key:
                        continue
                    if active.from_entry != new.from_entry:
                        continue
                    if active.is_partial_qty_bracket:
                        continue
                    raise RuntimeError(
                        "native-to-engine partial bracket conversion "
                        f"refuses: a native whole-row ExitIntent is "
                        f"already active for {old.symbol!r}/"
                        f"{new.from_entry!r} (intent "
                        f"{active.intent_key!r}); the script must cancel "
                        "the sibling native exit before promoting "
                        f"{old.intent_key!r} to a partial-qty bracket.",
                    )
            _blog_info(
                "converting native bracket to engine-trigger partial "
                "bracket %s -> %s", old, new,
            )
            # Native cancel must complete cleanly before engine legs are
            # armed: a timed-out cancel (``OrderDispositionUnknownError``)
            # leaves the original whole-row bracket possibly live at the
            # broker. Arming engine-trigger partial legs against the same
            # parent in that state violates the §12 #4 coexistence
            # invariant — the native fill and the engine triggers could
            # both close size on the position. Drive the cancel through a
            # strict path that propagates the unknown-disposition error;
            # on timeout, do NOT park under the new engine-trigger
            # intent's envelope (the parked coid would belong to the
            # cancel, not the new partial dispatch — the verification
            # bookkeeping would be mismatched and engine legs would
            # never be armed even after the cancel resolves). Re-raise
            # so :meth:`_diff_and_dispatch` leaves the OLD native
            # intent in ``_active_intents``; the next sync re-runs the
            # diff which calls back into this branch and retries the
            # strict cancel (idempotent at the exchange via the
            # deterministic ``client_order_id``).
            try:
                self._dispatch_cancel_strict(old)
            except OrderDispositionUnknownError as e:
                _blog_warning(
                    "native-to-engine conversion deferred: cancel of %s "
                    "timed out (%s); engine legs will not be armed until "
                    "the next sync retries the cancel and observes the "
                    "native bracket cleared",
                    old, e,
                )
                raise
            self._dispatch_new(new)
            return
        old_env = self._build_envelope(old)
        new_env = self._build_envelope(new)
        _blog_info("modifying %s -> %s", old, new)
        try:
            if isinstance(new, EntryIntent) and isinstance(old, EntryIntent):
                orders = self._run_async(self._broker.modify_entry(old_env, new_env))
                self._order_mapping[new.intent_key] = [o.id for o in orders]
                # Keep the both-set entry's software STOP leg in step with the
                # amended native LIMIT. modify_entry preserves the LIMIT's
                # KIND_ENTRY coid (the envelope anchor is pinned across amend
                # cycles), so the watch's leg-scoped cancel target stays valid;
                # only the fire level / size / side change. amend_watch self-
                # guards (no watch -> no-op), so a plain entry modify is a no-op.
                if new.stop is not None:
                    if self._entry_stop_engine.has_watch(new.intent_key):
                        self._entry_stop_engine.amend_watch(
                            new.intent_key,
                            stop_level=float(new.stop),
                            qty=new.qty,
                            side=new.side,
                        )
                    else:
                        # limit-only -> both-set: the STOP leg is new this amend.
                        self._arm_entry_stop_watch(new, new_env)
                elif self._entry_stop_engine.has_watch(new.intent_key):
                    # both-set -> limit-only: the STOP leg is gone; retire the
                    # watch so no market ever fires (no-op once it has already
                    # committed to the stop side).
                    self._entry_stop_engine.mark_aborted(
                        new.intent_key, reason='entry_stop_removed_by_modify',
                    )
            elif isinstance(new, ExitIntent) and isinstance(old, ExitIntent):
                orders = self._run_async(self._broker.modify_exit(old_env, new_env))
                self._order_mapping[new.intent_key] = [o.id for o in orders]
            else:
                # CloseIntent or mismatched kinds — cancel + re-execute.
                self._dispatch_cancel(old)
                # If the cancel landed in cancel-tentative (default cancel
                # path swallowed an ``OrderDispositionUnknownError`` and
                # :meth:`_mark_intent_cancel_disposition_pending` was set),
                # MUST NOT dispatch the replacement in the same sync — the
                # original parent's disposition is still ambiguous and a
                # fresh dispatch would race the original entry on a later
                # ALREADY_FILLED resolution. Raise the deferred-modify
                # signal so :meth:`_diff_and_dispatch` keeps the OLD intent
                # in ``_active_intents`` and the next sync re-diffs after
                # the cancel-retry loop resolves the disposition.
                if old.intent_key in self._cancel_disposition_pending:
                    _blog_warning(
                        "modify %s -> %s deferred — cancel of %s left "
                        "disposition pending; the replacement will be "
                        "dispatched after the cancel-retry loop resolves",
                        old, new, old.intent_key,
                    )
                    raise _PartialBracketModifyDeferred(
                        "cancel+re-execute modify deferred — cancel "
                        "disposition pending"
                    )
                self._dispatch_new(new)
        except OrderDispositionUnknownError as e:
            _blog_warning(
                "modify parked (unknown disposition) for %s: %s", new, e,
            )
            self._park_pending(new_env, e, kind='modify', old_intent=old)
        except BrokerManualInterventionError as e:
            _blog_error(
                "modify halted (manual intervention) for %s: %s", new, e,
            )
            self._record_halt(e)
            raise
        except OrderSkippedByPlugin:
            # Inner _dispatch_new (the cancel+re-execute fallback) declined.
            # _diff_and_dispatch handles the active-intents pop + warning.
            raise
        except Exception as e:
            _blog_error(
                "modify failed for %s: %s: %s", new, type(e).__name__, e,
            )
            raise

    def _dispatch_cancel(self, old: Intent) -> None:
        # Default cancel path: ``OrderDispositionUnknownError`` is logged
        # and swallowed — the next ``reconcile()`` pass observes whether
        # the exchange-side order is still live, and a subsequent cancel
        # attempt hits the same deterministic id (idempotent at the
        # exchange). Callers that must NOT proceed when the cancel
        # disposition is unknown should invoke
        # :meth:`_dispatch_cancel_strict` instead.
        try:
            self._dispatch_cancel_strict(old)
        except OrderDispositionUnknownError as e:
            _blog_warning(
                "cancel dispatch for %s timed out (coid=%s); "
                "next reconcile will verify: %s",
                old.intent_key, e.client_order_id, e,
            )
            # Parent entry cancel with engine-trigger partial brackets:
            # enter cancel-tentative instead of the original eager retire.
            # The retained ``_order_mapping`` and envelope are needed by
            # the ``reconcile()`` cancel-retry-loop to re-invoke
            # ``execute_cancel_with_outcome``; the per-parent leg flip
            # excludes pending legs from worst-SL contribution / arming
            # until the disposition resolves. The diff-loop's refuse-and-
            # defer guard sees ``_cancel_disposition_pending[intent_key]``
            # and refuses to dispatch / adopt a fresh intent on the same
            # parent until ``_clear_intent_cancel_disposition_pending``
            # runs (or stale-grace promotes the parent to DEGRADED_HALT).
            # See the cancel-tentative state design dossier.
            if (isinstance(old, EntryIntent)
                    and self._partial_qty_bracket_exit_mode
                    is CapabilityLevel.SOFTWARE):
                # Keep the original EntryIntent envelope in ``_envelopes``.
                # A later ALREADY_FILLED resolution restores the parent and
                # downstream code (e.g.
                # :meth:`_dispatch_engine_trigger_partial_bracket`) derives
                # the deterministic ``KIND_ENTRY`` ``client_order_id`` from
                # ``_envelopes[from_entry]``; replacing the slot with a
                # ``CancelIntent`` envelope at the cancel bar's
                # ``bar_ts_ms`` / ``retry_seq=0`` would yield a different
                # COID than the original entry dispatch and attach
                # leg / fail-safe state to a non-existent parent reference.
                # The retry loop builds its own ``CancelIntent`` envelope
                # on the fly each iteration (see
                # :meth:`_drive_cancel_tentative`), so no persistent swap
                # is needed to satisfy the plugin contract.
                self._mark_intent_cancel_disposition_pending(
                    old.intent_key,
                    reason='parent_cancel_disposition_unknown',
                    now_ms=self._cancel_tentative_now_ms(),
                )
                return
            # Non-partial-bracket cancel (CloseIntent never reaches here;
            # whole-row ExitIntent native cancel, or any EntryIntent in
            # modes other than SOFTWARE partial bracket): keep the
            # original eager-retire semantics. The strict path raised
            # before reaching its post-cancel cleanup, so mirror the
            # mapping / envelope drop and the pending-partials cleanup
            # here — same as before this workstream.
            self._order_mapping.pop(old.intent_key, None)
            self._drop_envelope(old.intent_key)
            self._retire_pending_partials_for_cancelled_entry(
                old, reason='parent_cancelled_by_strategy_unknown',
            )

    def _dispatch_cancel_strict(self, old: Intent) -> None:
        """Cancel a dispatched intent and propagate unknown-disposition errors.

        Identical to :meth:`_dispatch_cancel` except a timed-out broker
        cancel (:class:`OrderDispositionUnknownError`) is re-raised
        rather than swallowed. Used by paths that arm replacement state
        only after the original is provably gone — e.g. the native-to-
        engine partial-bracket conversion, where leaving the native
        bracket possibly live while engine legs are armed would violate
        the §12 #4 coexistence invariant.
        """
        # Engine-trigger partial bracket exits are engine-internal: the
        # leg rows were never sent to the broker, so a plugin
        # ``execute_cancel`` call would target nothing. Cancel through
        # the state machine instead — the cascade flips every active
        # leg under this ``intent_key`` to
        # :data:`LEG_STATE_CASCADED_CANCEL` and clears the engine-side
        # mapping. See the partial-qty bracket exit design dossier §3.3.
        if (isinstance(old, ExitIntent)
                and self._partial_bracket_engine.has_active_legs_for_intent(
                    old.intent_key,
                )):
            _blog_info("cancelling engine-trigger partial bracket %s", old)
            self._partial_bracket_engine.cancel_legs_for_intent(
                old.intent_key, reason='intent_cancelled',
            )
            self._order_mapping.pop(old.intent_key, None)
            self._drop_envelope(old.intent_key)
            return
        if isinstance(old, EntryIntent):
            # A both-set entry's STOP leg is a software watch; the strategy
            # cancel below cancels the native LIMIT working order, so retire
            # the watch too. No-op when the entry was not both-set.
            self._entry_stop_engine.mark_aborted(
                old.intent_key, reason='strategy_cancel',
            )
            cancel = CancelIntent(pine_id=old.pine_id, symbol=self._symbol)
        elif isinstance(old, ExitIntent):
            cancel = CancelIntent(
                pine_id=old.pine_id,
                symbol=self._symbol,
                from_entry=old.from_entry,
            )
        else:
            # CloseIntent is immediate market — nothing to cancel.
            self._order_mapping.pop(old.intent_key, None)
            self._drop_envelope(old.intent_key)
            return
        cancel_envelope = self._build_cancel_envelope(cancel)
        _blog_info("cancelling %s", cancel)
        try:
            self._run_async(self._broker.execute_cancel(cancel_envelope))
        except BrokerManualInterventionError as e:
            self._record_halt(e)
            raise
        self._order_mapping.pop(old.intent_key, None)
        self._drop_envelope(old.intent_key)
        self._retire_pending_partials_for_cancelled_entry(
            old, reason='parent_cancelled_by_strategy',
        )

    def _retire_pending_partials_for_cancelled_entry(
            self, old: Intent, *, reason: str,
    ) -> None:
        """Drop pending partial-bracket state after a parent-entry cancel.

        Strategy-initiated parent-entry cancel: retire any engine-trigger
        partial-bracket :data:`LEG_STATE_PENDING_ENTRY` legs and their
        parked partial :class:`ExitIntent` slots tied to this entry. The
        async ``cancelled`` event path (:meth:`_on_order_event`) calls
        :meth:`_abort_pending_partial_legs_for_dead_entry`, but a
        synchronous ``execute_cancel`` success — or a swallowed
        :class:`OrderDispositionUnknownError` in
        :meth:`_dispatch_cancel` — may never produce a later event with
        a matching ``_order_mapping`` lookup (the mapping was just
        popped). Without this eager cleanup the legs survive in
        :data:`LEG_STATE_PENDING_ENTRY` and the still-active partial
        :class:`ExitIntent` slot can promote against a future entry
        reusing the same ``from_entry`` (attaching a stale partial close
        to the wrong position) or short-circuit the duplicate guard in
        :meth:`_dispatch_engine_trigger_partial_bracket`. Whole-row
        native exits are intentionally left alone here — they are
        independent broker-side orders that the outer cancel-diff loop
        cancels separately when the script drops them from ``new_map``.
        """
        if not isinstance(old, EntryIntent):
            return
        self._partial_bracket_engine.abort_pending_legs_for_parent_never_arrived(
            symbol=self._symbol,
            from_entry=old.intent_key,
            reason=reason,
        )
        partial_exit_keys_to_drop = [
            key for key, active in self._active_intents.items()
            if isinstance(active, ExitIntent)
            and active.is_partial_qty_bracket
            and active.from_entry == old.intent_key
        ]
        for exit_key in partial_exit_keys_to_drop:
            self._active_intents.pop(exit_key, None)
            self._order_mapping.pop(exit_key, None)
            self._drop_envelope(exit_key)

    # === Async bridge ===

    def _run_async(self, coro):
        """Run a broker coroutine synchronously from the engine's thread.

        In production the engine shares an event loop with the live
        provider; calls hop to that loop via ``run_coroutine_threadsafe``.
        In unit tests no loop is supplied — the coroutine is driven to
        completion by a transient ``asyncio.run``.
        """
        if self._loop is None:
            return asyncio.run(coro)
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(
            timeout=self._timeout,
        )
