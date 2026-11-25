"""One-way position emulation engine over a hedging-mode broker account.

A hedging account holds several open positions ("legs") per symbol; a Pine
strategy sees a single *one-way* position. The pure helpers in
:mod:`~pynecore.core.broker.emulator` decide *what* to do (net the legs, pick a
FIFO close plan, snap per-leg volumes); this engine *drives* it: it fans a
:class:`~pynecore.core.broker.models.CloseIntent` out across the legs through a
thin :class:`~pynecore.core.plugin.broker.PositionPort`, and owns the
persist-first ledger that lets an interrupted fan-out resume on restart without
re-closing a leg that already settled.

It is the close/reversal counterpart of
:class:`~pynecore.core.broker.software_stop_limit_engine.SoftwareStopLimitEngine`
and :class:`~pynecore.core.broker.software_partial_bracket_engine.SoftwarePartialBracketEngine`:
one instance per :class:`~pynecore.core.broker.sync_engine.OrderSyncEngine`, the
persisted close-leg rows in :mod:`~pynecore.core.broker.store_helpers` are the durable
representation (PERSIST-FIRST), and :meth:`OneWayEmulator.restart_replay` re-derives an
unfinished fan-out from them. The legs' close FILLs themselves flow through the engine's
ordinary natural-close path (mapped to ``LegType.CLOSE`` by the plugin), so this engine
never touches the fill router — only the dispatch + its crash/replay.

The engine carries no broker, clock, or exchange-model state. Each broker action
goes through the :class:`PositionPort`; aggregation / selection / quantization
are the pure functions in :mod:`~pynecore.core.broker.emulator`. That keeps it deterministic and
unit-testable in isolation, with ``store_ctx=None`` for the no-persistence path.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pynecore.core.broker.emulator import (
    LegClose,
    aggregate_positions,
    net_survivor_legs,
    plan_leg_close_volumes,
    plan_reversal,
    select_legs_for_close,
)
from pynecore.core.broker.exceptions import (
    BracketAttachAfterFillRejectedError,
    ExchangeConnectionError,
    ExchangeOrderRejectedError,
    ExchangeRateLimitError,
    MarketClosedError,
    OrderDispositionUnknownError,
    OrderSkippedByPlugin,
)
from pynecore.core.broker.idempotency import (
    CLIENT_ORDER_ID_MAX_LEN,
    KIND_CANCEL,
    KIND_CLOSE,
    KIND_ENTRY,
    KIND_ENTRY_TRIGGERED,
    KIND_EXIT_SL,
    parse_client_order_id,
    parse_wire_client_order_id,
)
from pynecore.core.broker.models import (
    CancelIntent,
    CloseIntent,
    DispatchEnvelope,
    EntryIntent,
    ExitIntent,
    OrderType,
    PositionLeg,
    format_intent_key,
)
from pynecore.core.plugin import ProviderError, is_retryable_provider_error
from pynecore.lib.log import broker_warning as _blog_warning
from pynecore.core.broker.store_helpers import (
    BRACKET_OWN_STATE_ACTIVE,
    BRACKET_OWN_STATE_CLEARING,
    BRACKET_OWN_STATE_RELEASED,
    CLOSE_LEG_STATE_DISPATCHED,
    ENTRY_KIND_POSITION,
    EXTRAS_KEY_BRACKET_OWN_ATTACH_COID,
    EXTRAS_KEY_BRACKET_OWN_CLEAR_COID,
    EXTRAS_KEY_BRACKET_OWN_LEG_ID,
    EXTRAS_KEY_BRACKET_OWN_SL,
    EXTRAS_KEY_BRACKET_OWN_STATE,
    EXTRAS_KEY_BRACKET_OWN_TP,
    EXTRAS_KEY_BRACKET_OWN_TRAIL_OFFSET,
    EXTRAS_KEY_CLOSE_LEG_ID,
    EXTRAS_KEY_CLOSE_LEG_VOLUME,
    EXTRAS_KEY_RESIDUAL_OPEN_BAR_TS_MS,
    EXTRAS_KEY_RESIDUAL_OPEN_ENTRY_COID,
    EXTRAS_KEY_RESIDUAL_OPEN_RETRY_SEQ,
    EXTRAS_KEY_RESIDUAL_OPEN_RUN_TAG,
    clear_residual_open_row,
    create_bracket_ownership_row,
    create_close_leg_row,
    create_residual_open_row,
    iter_active_bracket_ownerships,
    iter_active_close_legs,
    iter_active_residual_opens,
    update_bracket_ownership_state,
    update_close_leg_state,
)

#: Seconds an exit's incomplete-book deferral on one pending entry counts as
#: the routine wait for the entry's fill to drain. Past it the deferral is a
#: stall: the newest leg stays unprotected, so the caller warns. Measured in
#: time, not in syncs: the sync cadence is the engine's, not the venue's fill
#: latency (live: ctrader drained a pyramid fill in one or two syncs, bybit
#: cycle 438 took four syncs within the same second).
BOOK_INCOMPLETE_GRACE_S = 30.0

if TYPE_CHECKING:
    from collections.abc import Callable

    from pynecore.core.broker.storage import OrderRow, RunContext
    from pynecore.core.plugin.broker import PositionPort


async def _read_legs(port: 'PositionPort', symbol: str) -> list[PositionLeg]:
    """Read the raw legs of ``symbol``, translating a transient fault to the taxonomy.

    The fan-outs run inside the engine's order-dispatch bridge, which halts the
    bot on a raw transient fault because a WRITE that failed that way may have
    landed. A read has no venue side effect, so its transient fault is a plain
    :class:`ExchangeConnectionError` (park and retry) — the same verdict the
    engine's state-read bridge reaches. A non-retryable ``ProviderError``
    propagates unchanged.
    """
    try:
        return await port.fetch_raw_positions(symbol)
    except Exception as exc:
        raise _translate_port_read_fault(exc) from exc


async def _read_quantizer(port: 'PositionPort', symbol: str) -> 'Callable[[float], int]':
    """Read the volume quantizer of ``symbol`` with the :func:`_read_legs` fault mapping."""
    try:
        return await port.get_volume_quantizer(symbol)
    except Exception as exc:
        raise _translate_port_read_fault(exc) from exc


def _translate_port_read_fault(exc: Exception) -> Exception:
    """Map a transient port-read fault to :class:`ExchangeConnectionError`, else keep it.

    A venue throttle (:class:`ExchangeRateLimitError`) on a read is transient
    too: the read changed nothing, so the caller parks and retries on a later
    sync, the verdict the engine's own read bridge reaches for it.
    """
    if isinstance(exc, ExchangeRateLimitError):
        return ExchangeConnectionError(str(exc) or "exchange rate limit hit")
    if isinstance(exc, ProviderError):
        if is_retryable_provider_error(exc):
            return ExchangeConnectionError(str(exc) or "connection lost")
        return exc
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return ExchangeConnectionError(str(exc) or "connection lost")
    return exc


def close_leg_client_order_id(parent_coid: str, leg_id: str) -> str:
    """The client order id of one leg's close in a close fan.

    Composed from the fan's own client order id, so it names one dispatch's
    close of one leg: two closes of the same leg never share it.

    :param parent_coid: The client order id of the close being fanned out.
    :param leg_id: The :attr:`~pynecore.core.broker.models.PositionLeg.leg_id`
        of the leg it reduces.
    """
    return f"{parent_coid}:{leg_id}"


__all__ = [
    'close_leg_client_order_id',
    'CloseFanResult',
    'ReversalFanResult',
    'BracketFanResult',
    'BracketReplayRefusal',
    'ClearFanResult',
    'OneWayEmulator',
]


@dataclass(frozen=True)
class _LegLevels:
    """The protection one leg carries, or one exit asks for on it.

    :ivar tp_price: Pine-unit take-profit level, or ``None``.
    :ivar sl_price: Pine-unit stop-loss level, or ``None``.
    :ivar trail_offset: Trailing offset in Pine ticks (as the
        :class:`~pynecore.core.broker.models.ExitIntent` carries it), or ``None``.
    """
    tp_price: float | None = None
    sl_price: float | None = None
    trail_offset: float | None = None


@dataclass(frozen=True)
class BracketReplayRefusal:
    """A bracket the venue definitively refused to re-assert on a restart.

    One is recorded for every exit that owns the refused leg: the leg carries
    the merge of its owners' levels, so the refusal concerns each of them.

    :ivar intent_key: The exit's diff key.
    :ivar leg_id: The position leg the bracket was re-asserted on.
    :ivar tp_price: The take-profit level of the refused re-assert (the merge
        of the leg's owners).
    :ivar sl_price: The stop-loss level of the refused re-assert (the merge of
        the leg's owners).
    :ivar error: The venue's refusal.
    """
    intent_key: str
    leg_id: str
    tp_price: float | None
    sl_price: float | None
    error: ExchangeOrderRejectedError


@dataclass(frozen=True)
class CloseFanResult:
    """Outcome of one fanned-out close, for the caller's return shaping / audit.

    :ivar legs: ``(leg_id, broker-grid volume)`` pairs actually dispatched, in
        FIFO order. Empty when nothing was open or the close was skipped. The
        volumes are BROKER-grid integers (cTrader centi-units, Capital.com
        lot-step counts, ...) — never comparable with a Pine-unit fill
        quantity; use ``dispatched_qty`` for that.
    :ivar dispatched_qty: Pine-unit quantity the dispatched legs cover — the
        summed FIFO plan slices, in the SAME unit as
        :attr:`~OrderEvent.fill_qty`, so the
        caller can compare it with the fills that come back. ``0.0`` when
        nothing was dispatched.
    :ivar shortfall: Pine-unit quantity the open legs could not cover (the
        broker holds less than Pine believes). ``0.0`` in the normal case; a
        positive value is the caller's signal to log, not halt.
    :ivar skipped: ``True`` when the whole close quantized below the broker's
        volume grid, so no order was sent (a non-halting skip — the engine
        re-evaluates next bar). Distinct from an empty ``legs`` with
        ``skipped=False``, which means the symbol was already flat.
    """
    legs: tuple[tuple[str, int], ...]
    dispatched_qty: float
    shortfall: float
    skipped: bool


@dataclass(frozen=True)
class ReversalFanResult:
    """Outcome of one decomposed reversal / add, for the caller's return shaping.

    :ivar closes: ``(leg_id, broker-grid volume)`` pairs FIFO-closed to retire
        the opposing exposure. Empty for a pure add (no opposing legs).
    :ivar open_qty: Pine-unit size opened in the order's own direction after the
        closes. ``0.0`` for an exact flatten (close all, open nothing).
    :ivar opened_orders: The ``ExchangeOrder``(s) the residual open produced.
        Empty when ``open_qty`` was zero.
    """
    closes: tuple[tuple[str, int], ...]
    open_qty: float
    opened_orders: tuple


@dataclass(frozen=True)
class BracketFanResult:
    """Outcome of one exit-bracket replication across a position's legs.

    :ivar legs: Broker leg ids the bracket was replicated onto, in FIFO order.
    :ivar skipped: ``True`` when nothing was amended — the symbol was flat (no
        position-side legs to protect), the position book is incomplete (a
        protected-side entry's fill is not yet journaled), or the exit's
        ``from_entry`` holds no leg (``entry_without_leg``) — a non-halting
        no-op; the caller re-evaluates next bar.
    :ivar skip_reason: Human-readable cause for an incomplete-book or
        entry-without-leg skip, for the caller's skip message; ``None`` for the
        flat / flipped-side skips.
    :ivar stalled: ``True`` on the one incomplete-book skip where the wait for
        the pending entry's fill outgrew :data:`BOOK_INCOMPLETE_GRACE_S` — the
        deferral is no longer routine and the caller warns.
    :ivar entry_without_leg: ``True`` when the skip is because every survivor
        leg is attributed to an entry other than the exit's ``from_entry`` (see
        :meth:`~OneWayEmulator._target_legs`); already reported at WARNING once
        per exit by the emulator.
    """
    legs: tuple[str, ...]
    skipped: bool
    skip_reason: str | None = None
    stalled: bool = False
    entry_without_leg: bool = False


@dataclass(frozen=True)
class ClearFanResult:
    """Outcome of one ownership-scoped exit-bracket clear.

    :ivar legs: Broker leg ids of the ownership rows the clear released, one
        entry per row (a leg two retired exits shared appears twice) — the
        subset the cancelled exit owns, never the whole position side. Each was
        amended to the merged levels of the exits that still own it, and
        cleared only when none does. Empty when the exit owned nothing (already
        released, or no persisted ownership index).
    """
    legs: tuple[str, ...]


class OneWayEmulator:
    """Drives Pine one-way close/reversal over a hedging account's legs.

    One instance per :class:`~pynecore.core.broker.sync_engine.OrderSyncEngine`.
    The engine owns the persisted close-leg ledger and its restart replay; every
    broker-side action is performed through the :class:`PositionPort` the
    emulating plugin exposes.
    """

    def __init__(
            self,
            store_ctx: 'RunContext | None',
            *,
            mintick: float = 1.0,
            block_exposure_reopens: 'Callable[[], bool] | None' = None,
    ) -> None:
        self._store_ctx = store_ctx
        self._mintick = mintick
        #: Engine-supplied gate consulted before a residual-open re-dispatch
        #: (the reversal reopen leg). Wired to the engine's quarantine latch:
        #: under quarantine the owed FIFO closes still run (risk-reducing)
        #: but the reopen is withheld and its breadcrumb stays live, so the
        #: committed reversal finishes after the operator restart.
        self._block_exposure_reopens = block_exposure_reopens
        #: Brackets the venue refused to re-assert during the last restart
        #: replay; the engine takes them (:meth:`take_replay_refusals`) and
        #: re-arms the exits through its regular dispatch path.
        self._replay_refusals: list[BracketReplayRefusal] = []
        #: Exits (symbol, pine_id, from_entry) already told that their
        #: ``trail_price`` activation arms immediately per-leg. The limitation
        #: is a property of the exit, not of a dispatch: a trailing exit is
        #: re-dispatched on every bar it changes, so it is reported once.
        self._trail_activation_reported: set[tuple[str, str, str | None]] = set()
        #: Incomplete-book deferrals per (exit intent key, pending entry coid):
        #: the caller's ``now`` of the first one, and whether the stall was
        #: flagged; dropped once the exit's fan passes the gate.
        self._book_incomplete_deferrals: dict[tuple[object, str], tuple[float, bool]] = {}
        #: Exits (intent keys) already told that every survivor leg is
        #: attributed to another entry, so their ``from_entry`` holds no leg
        #: and the fan is skipped. Reported once per exit, until a fan of the
        #: exit finds a leg again.
        self._entry_without_leg_reported: set[str] = set()
        #: Legs (symbol, leg id, owning intent keys, conflict kind) already
        #: told that their owners ask for levels one leg cannot carry.
        self._level_conflicts_reported: set[tuple[str, str, frozenset[str], str]] = set()
        #: Legs (symbol, leg id, retiring intent keys, remaining owner keys)
        #: already told that the venue refused the retirement re-assert.
        self._retirement_refusals_reported: set[
            tuple[str, str, frozenset[str], frozenset[str]]] = set()
        #: The levels this process last put on each leg (symbol, leg id) with an
        #: amend the venue confirmed; see :meth:`_amend_leg`. An entry is dropped
        #: before every amend and only restored when the amend returns, so it is
        #: absent after any ambiguous, rejected or interrupted amend and after a
        #: restart.
        self._venue_levels: dict[tuple[str, str], _LegLevels] = {}

    def take_replay_refusals(self) -> list[BracketReplayRefusal]:
        """Hand over (and forget) the brackets the restart replay could not re-assert."""
        refusals, self._replay_refusals = self._replay_refusals, []
        return refusals

    def _trail_price_distance(self, trail_offset: float | None) -> float | None:
        """Convert Pine trailing-offset ticks to the PositionPort price unit."""
        if trail_offset is None:
            return None
        return trail_offset * self._mintick

    # === Close fan-out ====================================================

    async def run_close(
            self, envelope: 'DispatchEnvelope', port: 'PositionPort',
    ) -> CloseFanResult:
        """Fan a :class:`CloseIntent` out as per-leg FIFO closes.

        Reads the live legs, nets them, selects the oldest legs on the position
        side to cover ``intent.qty`` — a close bound to one entry
        (:meth:`_close_bound_entry`) takes that entry's attributed legs first,
        so the leg that physically closes is the one whose binding the fill
        settles, and a defensive close reduces only the leg it targets
        (``target_leg_id``; a target that already closed or shrank leaves the rest as
        ``shortfall``; the close's marker then resolves through the
        stale-grace broker probe) — snaps the plan to the broker grid, and
        dispatches one :meth:`PositionPort.close_leg` per leg — persisting each
        leg PERSIST-FIRST so a crash mid-fan resumes via :meth:`restart_replay`.

        Never raises for an undersized close: a whole-close-below-grid returns
        ``skipped=True`` and a broker-holds-less returns a positive
        ``shortfall`` — the bot keeps running and the caller decides how to
        surface it.
        """
        intent = envelope.intent
        assert isinstance(intent, CloseIntent)
        legs = await _read_legs(port, intent.symbol)
        pos = aggregate_positions(intent.symbol, legs)
        if pos is None or pos.side == 'flat':
            # Nothing open, or offsetting legs net to flat — benign no-op.
            return CloseFanResult(
                legs=(), dispatched_qty=0.0, shortfall=0.0, skipped=False,
            )
        # CloseIntent.side is the close direction ("sell" closes a long); the
        # legs to reduce sit on the opposite side of the book.
        leg_side = 'buy' if intent.side == 'sell' else 'sell'
        closes, shortfall = select_legs_for_close(
            intent.qty, legs, leg_side, entry_id=self._close_bound_entry(intent, legs),
            leg_id=intent.target_leg_id,
        )
        if not closes:
            return CloseFanResult(
                legs=(), dispatched_qty=0.0, shortfall=shortfall,
                skipped=False,
            )
        await self._reject_unsupported_partial_closes(
            closes, legs, symbol=intent.symbol,
            intent_key=intent.intent_key, port=port,
        )
        parent_coid = envelope.client_order_id(KIND_CLOSE)
        # A script close or an engine-trigger partial close may be declined
        # per leg without halting (the leg settled or is busy on the venue;
        # the book re-derives next tick). Marketable exits and defensive
        # closes keep the definitive-reject contract.
        dispatched = await self._fan_out_closes(
            closes, symbol=intent.symbol, side=intent.side,
            intent_key=intent.intent_key, pine_id=intent.pine_id,
            parent_coid=parent_coid, port=port,
            skippable=intent.synthetic_kind in (None, 'partial_trigger'),
        )
        if not dispatched:
            # Every slice rounded below the broker grid — skip, do not halt.
            return CloseFanResult(
                legs=(), dispatched_qty=0.0, shortfall=shortfall, skipped=True,
            )
        sent_legs = {leg_id for leg_id, _volume in dispatched}
        return CloseFanResult(
            legs=tuple(dispatched),
            # Pine units, NOT the broker-grid volumes above: the caller
            # compares this with the fill quantities the legs report back,
            # and those arrive in Pine units on every venue. Only the legs
            # that went on the wire count — a fan a plugin skip cut short
            # covers less than the plan.
            dispatched_qty=round(
                sum(close.qty for close in closes if close.leg_id in sent_legs), 12,
            ),
            shortfall=shortfall, skipped=False,
        )

    @staticmethod
    def _close_bound_entry(intent: CloseIntent, legs: list[PositionLeg]) -> str | None:
        """The entry whose bindings a close's fill settles, ``None`` when it is unbound.

        Mirrors the engine's fill-side attribution
        (:meth:`~pynecore.core.broker.sync_engine.OrderSyncEngine._close_fill_entry_id`):
        a script ``strategy.close(id)`` is bound to ``id`` (``strategy.close_all``
        carries an empty ``pine_id`` and is unbound), an engine close aimed at
        one entry (``partial_trigger``, ``marketable_exit``) to its
        ``target_entry_id``, a defensive close to the entry of the leg it
        targets (``target_leg_id``: the leg the venue refused to protect,
        whose entry's bindings the defensive-close marker settles); every
        other close walks the book FIFO.

        :param intent: The close being fanned out.
        :param legs: The symbol's open legs, read for the defensive close's
            target leg.
        """
        if intent.synthetic_kind is None:
            return intent.pine_id or None
        if intent.synthetic_kind in ('partial_trigger', 'marketable_exit'):
            return intent.target_entry_id
        if intent.target_leg_id is not None:
            for leg in legs:
                if leg.leg_id == intent.target_leg_id:
                    return leg.entry_id
        return None

    async def run_reversal(
            self, envelope: 'DispatchEnvelope', port: 'PositionPort',
    ) -> ReversalFanResult:
        """Decompose a Pine reversal/add entry over the hedging legs.

        Pine folds a reversal into one combined-size :class:`EntryIntent`
        assuming a netting auto-flip. On a hedging account that single order
        would open a separate opposing leg, so it is split: FIFO-close the
        opposing legs (up to the order size) and open only the residual in the
        order's own direction. Pure add (no opposing legs) opens the whole size;
        exact flatten (opposing exposure == order size) closes all and opens
        nothing.
        """
        intent = envelope.intent
        assert isinstance(intent, EntryIntent)
        if intent.order_type is not OrderType.MARKET:
            # A resting LIMIT / STOP entry is not Pine's combined-size market
            # auto-flip: it rests as a fresh working order and is decomposed
            # (if it is still a reversal when it triggers) only at FILL time.
            # Place it full-size now with NO opposing-leg close — retiring the
            # opposing exposure here would flatten the position before the entry
            # ever fires. (Fill-time decomposition of a resting reversal is a
            # follow-up.) ``place_leg`` runs the broker's own volume-bounds
            # pre-flight, so an out-of-range resting order still skips cleanly.
            opened = tuple(await port.place_leg(envelope, intent.qty))
            return ReversalFanResult(
                closes=(), open_qty=intent.qty, opened_orders=opened,
            )
        legs = await _read_legs(port, intent.symbol)
        plan = plan_reversal(intent.side, intent.qty, legs)
        # Pre-flight partial-leg capability BEFORE anything is persisted or
        # sent — same atomicity rationale as the volume-bounds check below.
        await self._reject_unsupported_partial_closes(
            plan.closes, legs, symbol=intent.symbol,
            intent_key=intent.intent_key, port=port,
        )
        # Pre-flight the broker volume bounds BEFORE any close lands: an
        # out-of-range order must raise the non-halting skip while it is still
        # true, otherwise the closes reduce the book yet the whole reversal is
        # reported skipped, desyncing the engine. A pure reduce (open_qty == 0)
        # validates the combined order size; a real reversal validates its
        # residual leg.
        preflight_qty = plan.open_qty if plan.open_qty > 0.0 else intent.qty
        await port.reject_out_of_range(envelope, preflight_qty)
        parent_coid = envelope.client_order_id(KIND_CLOSE)
        # Persist-first residual-open breadcrumb BEFORE the closes: a crash
        # after the closes land but before ``place_leg`` persists the residual
        # entry row would otherwise leave the book reduced with the residual
        # open lost — neither the close-leg replay nor the entry journal owns
        # it. ``restart_replay`` reconciles this against the residual entry row
        # and re-dispatches only if that row never landed. Only a genuine
        # reversal needs it: a pure add (no opposing legs, ``plan.closes``
        # empty) leaves the book intact, so re-opening it on restart would
        # bypass the next sync's Pine re-evaluation and resurrect an entry the
        # strategy may no longer want. The breadcrumb is gated on closes being
        # actually owed, never on a positive ``open_qty`` alone.
        residual_coid = f"{parent_coid}:residual"
        wrote_breadcrumb = bool(plan.closes) and plan.open_qty > 0.0
        if wrote_breadcrumb and self._store_ctx is not None:
            # The breadcrumb's entry coid must match the kind ``place_leg`` will
            # persist under: a triggered market entry dispatches as
            # ``KIND_ENTRY_TRIGGERED``, so anchoring on ``KIND_ENTRY`` there would
            # make the replay's row-existence check miss the landed entry row
            # and double-open the residual.
            entry_kind = KIND_ENTRY_TRIGGERED if intent.triggered_entry else KIND_ENTRY
            create_residual_open_row(
                self._store_ctx,
                coid=residual_coid,
                symbol=intent.symbol,
                side=intent.side,
                qty=plan.open_qty,
                intent_key=intent.intent_key,
                pine_entry_id=intent.pine_id,
                entry_coid=envelope.client_order_id(entry_kind),
                run_tag=envelope.run_tag,
                bar_ts_ms=envelope.bar_ts_ms,
                retry_seq=envelope.retry_seq,
            )
        dispatched: list[tuple[str, int]] = []
        if plan.closes:
            try:
                dispatched = await self._fan_out_closes(
                    plan.closes, symbol=intent.symbol, side=intent.side,
                    intent_key=intent.intent_key, pine_id=intent.pine_id,
                    parent_coid=parent_coid, port=port, skippable=False,
                )
            except ExchangeOrderRejectedError:
                # A close leg was DEFINITIVELY refused — the fan-out raised
                # before the residual ``place_leg`` ran, so the residual never
                # opened. ``_dispatch_new`` turns this reject into a non-halting
                # skip for the entry (signal dropped, bot continues), so the
                # breadcrumb must be discharged here exactly as the residual
                # reject path below does: leaving it live would let a later
                # restart replay re-close + re-open a reversal the exchange
                # already rejected and Pine never re-signalled. An ambiguous
                # ``OrderDispositionUnknownError`` (caught by the broader
                # ``BrokerError`` contract, not here) leaves the close's fate —
                # and thus a possibly-owed residual — unknown, so its breadcrumb
                # must survive for restart reconciliation and is left intact.
                if wrote_breadcrumb and self._store_ctx is not None:
                    clear_residual_open_row(self._store_ctx, residual_coid)
                raise
        opened_orders: tuple = ()
        if plan.open_qty > 0.0:
            try:
                opened_orders = tuple(await port.place_leg(envelope, plan.open_qty))
            except ExchangeOrderRejectedError:
                # The exchange definitively refused the residual entry — nothing
                # opened. The dispatch path turns this into a non-halting skip
                # (signal dropped, bot continues), so the breadcrumb must be
                # discharged here: leaving it live would let a later restart
                # replay re-open a residual the exchange already rejected and Pine
                # never re-signalled. Only a DEFINITIVE reject clears it — an
                # ambiguous ``OrderDispositionUnknownError`` / connection error
                # leaves the open's fate unknown, so its breadcrumb must survive
                # for restart reconciliation and is left intact (re-raised).
                if wrote_breadcrumb and self._store_ctx is not None:
                    clear_residual_open_row(self._store_ctx, residual_coid)
                raise
            if wrote_breadcrumb and self._store_ctx is not None:
                # The residual entry row is now persisted (persist-first inside
                # ``place_leg``) and dispatched — the breadcrumb is discharged.
                clear_residual_open_row(self._store_ctx, residual_coid)
        return ReversalFanResult(
            closes=tuple(dispatched), open_qty=plan.open_qty,
            opened_orders=opened_orders,
        )

    @staticmethod
    async def _reject_unsupported_partial_closes(
            closes: tuple[LegClose, ...], legs: list[PositionLeg], *,
            symbol: str, intent_key: str, port: 'PositionPort',
    ) -> None:
        """Atomically skip a close plan containing a partial leg slice on a
        port that cannot express one.

        A venue without a per-leg partial reduce (Capital.com's
        ``DELETE /positions/{dealId}`` is full-row only) declares
        ``supports_partial_leg_close = False`` on its port; an absent
        attribute means supported. The check runs BEFORE any close-leg row
        is persisted or dispatched, so the skip leaves no half-reduced book
        and nothing for restart replay to resume — raising from inside the
        fan (the partial tail is dispatched last, FIFO) would desync the
        engine exactly like the volume-bounds case documented in
        :meth:`run_reversal`. Volumes are compared on the broker grid so
        float noise in the Pine-unit plan cannot fake a partial.
        """
        if getattr(port, 'supports_partial_leg_close', True):
            return
        if not closes:
            return
        quantize = await _read_quantizer(port, symbol)
        live_by_id = {leg.leg_id: leg for leg in legs}
        for close in closes:
            live = live_by_id.get(close.leg_id)
            if live is None:
                continue
            if quantize(close.qty) < quantize(live.qty):
                raise OrderSkippedByPlugin(
                    f"Skipping {symbol} close: leg {close.leg_id} would be "
                    f"reduced partially ({close.qty} of {live.qty}) but this "
                    f"broker only supports whole-leg closes on a hedging "
                    f"account. No order sent; partial closes need a one-way "
                    f"(netting) account.",
                    intent_key=intent_key,
                    reason="partial_leg_close_unsupported",
                    context={
                        'symbol': symbol,
                        'leg_id': close.leg_id,
                        'close_qty': close.qty,
                        'leg_qty': live.qty,
                    },
                )

    async def _fan_out_closes(
            self, closes: tuple[LegClose, ...], *,
            symbol: str, side: str, intent_key: str, pine_id: str,
            parent_coid: str, port: 'PositionPort', skippable: bool,
    ) -> list[tuple[str, int]]:
        """Snap a FIFO close plan to the broker grid and dispatch it per leg.

        Shared by :meth:`run_close` and :meth:`run_reversal`. Each leg is
        PERSIST-FIRST: its row (coid ``f"{parent_coid}:{leg_id}"``, so a restart
        re-dispatch upserts the SAME row) is written ``pending`` BEFORE the wire
        call and finalised ``dispatched`` only after
        :meth:`PositionPort.close_leg` returns. Returns the
        ``(leg_id, broker-grid volume)`` pairs actually sent (empty when the
        whole plan rounded below the grid).

        A leg the plugin declines with
        :class:`~pynecore.core.broker.exceptions.OrderSkippedByPlugin` (the
        leg is already gone, or busy with another venue operation) never went
        on the wire, so its pending row is discharged before the skip is
        surfaced. With ``skippable`` the skip propagates as-is when no leg
        went on the wire yet — the caller's close is re-evaluated against the
        next snapshot without halting — and ends the fan as a partial success
        once earlier legs were dispatched, so their fills stay owned;
        otherwise it is re-raised as a definitive
        :class:`~pynecore.core.broker.exceptions.ExchangeOrderRejectedError`
        so a reversal's or a replay's leg close keeps its loud contract.
        """
        quantize = await _read_quantizer(port, symbol)
        planned = plan_leg_close_volumes(closes, quantize)
        dispatched: list[tuple[str, int]] = []
        for leg_id, volume in planned:
            leg_coid = close_leg_client_order_id(parent_coid, leg_id)
            if self._store_ctx is not None:
                create_close_leg_row(
                    self._store_ctx,
                    coid=leg_coid,
                    symbol=symbol,
                    side=side,
                    qty=float(volume),
                    intent_key=intent_key,
                    pine_entry_id=pine_id,
                    parent_close_coid=parent_coid,
                    leg_id=leg_id,
                    leg_volume=volume,
                )
            try:
                await port.close_leg(symbol, leg_id, volume, leg_coid)
            except OrderSkippedByPlugin as exc:
                if self._store_ctx is not None:
                    # Nothing reached the wire for this leg: a live pending
                    # row would make a restart replay re-close a leg the
                    # venue already settled, or double up with the caller's
                    # own re-evaluation.
                    self._store_ctx.close_order(leg_coid)
                if not skippable:
                    raise ExchangeOrderRejectedError(str(exc)) from exc
                if not dispatched:
                    raise
                # Earlier legs already went on the wire: their fills are
                # coming back and the caller must own them (order mapping,
                # fill ledger), so the fan ends here as a partial success
                # and the remainder re-derives from the next snapshot.
                _blog_warning(
                    "one-way close for %s stopped at leg %s (%s); %d leg(s) "
                    "already dispatched, the rest re-evaluates next tick",
                    format_intent_key(intent_key), leg_id, exc.reason,
                    len(dispatched),
                )
                break
            if self._store_ctx is not None:
                update_close_leg_state(
                    self._store_ctx, coid=leg_coid,
                    new_state=CLOSE_LEG_STATE_DISPATCHED, close_row=True,
                )
            dispatched.append((leg_id, volume))
        return dispatched

    # === Exit-bracket replication =========================================

    async def run_exit_bracket(
            self, envelope: 'DispatchEnvelope', port: 'PositionPort', now: float = 0.0,
    ) -> BracketFanResult:
        """Replicate a Pine exit's bracket onto the net-survivor legs it covers.

        Hedging venues carry protective levels per position, so a one-way bracket
        over a multi-leg position is delivered by amending TP/SL/trailing onto
        each leg the exit covers. On a mixed book (a manual hedge or a
        crash-interrupted reversal left opposing legs open) only the legs that
        survive virtual-FIFO netting are candidates (:func:`net_survivor_legs`),
        never the gross majority side — amending the whole majority side would
        close more than the net position when the stop fires and flip it to the
        minority side. Among them the exit covers the legs
        :meth:`_target_legs` picks from the legs' entry attribution: an exit with
        a ``from_entry`` covers that entry's own legs and every unattributed
        leg, so on a venue with one leg per entry each pyramided entry keeps its
        own exit's levels.

        A leg can be owned by several exits (a TP-only and an SL-only exit, or
        several exits on an unattributed leg), and a venue leg carries a single
        protection, so each covered leg is amended to the merge of ALL its
        active owners' levels (:meth:`_merge_owner_levels`), never to this
        exit's alone. Each leg is recorded PERSIST-FIRST in the ownership index
        (keyed by the exit's ``intent_key``, holding this exit's own levels)
        BEFORE its amend, so :meth:`run_exit_bracket_clear` later retires ONLY
        the legs this exit owns and :meth:`restart_replay` can re-assert or
        release them. A flat symbol — or a book whose net side is the opposite
        of the one this exit protects — is a non-halting skip.

        Re-running it (a modify, or a re-attach after pyramiding) upserts the
        same per-leg rows idempotently: the row key is STABLE per (exit
        identity, leg) — derived from the exit's pine_id + from_entry + leg_id,
        NOT the bar-varying dispatch coid — so a modify on a later bar updates
        the same rows instead of accreting stale ones. Two exits sharing a
        pine_id but differing in from_entry get disjoint rows even on a shared
        leg (the dispatch coid alone would collide — it encodes pine_id but not
        from_entry). Legs that have since closed are released by the restart /
        reconcile pass, not here.

        The fan is also deferred (``skipped=True`` with a ``skip_reason``)
        while any protected-side position entry's fill is still pending in the
        journal — see :meth:`_pending_position_entry` — so it always runs over
        a complete book and covers a same-sync pyramid entry's own leg. The
        wait is a stall once it outlasts :data:`BOOK_INCOMPLETE_GRACE_S`.

        A leg whose merged levels equal what this process last confirmed on it
        is not amended again (:meth:`_amend_leg`): attaching an exit every one
        of whose levels a co-owner already dominates changes nothing on the
        venue, and a re-send would reset a trailing stop the venue has moved.
        The ownership row is written exactly as if the amend had gone out.

        A definitive refusal of the amend depends on who owns the leg. When
        this exit is the leg's only owner the position is open and unprotected:
        its row is released and the refusal is raised as
        :class:`~pynecore.core.broker.exceptions.BracketAttachAfterFillRejectedError`
        (the engine closes the position defensively). When other exits own the
        leg too, the venue still holds their merge — the position is protected
        by them and only this exit's own levels are missing — so the leg's row
        goes back to what it was before this attach (released for a first
        attach, its previous levels for a modify; see
        :meth:`_refused_co_owned_attach`) and the venue's
        :class:`~pynecore.core.broker.exceptions.ExchangeOrderRejectedError`
        propagates unchanged: the engine treats it as a refused exit (one retry
        under a fresh id, then the crossed-level close or the defensive close),
        never as an immediate flatten over a co-owner's unchanged level.

        :param envelope: The exit's dispatch envelope.
        :param port: The venue's position port.
        :param now: The caller's monotonic seconds, timing an incomplete-book
            deferral (the engine itself keeps no clock).
        :return: The legs the bracket went onto, or the skip.
        """
        intent = envelope.intent
        assert isinstance(intent, ExitIntent)
        trail_key = (intent.symbol, intent.pine_id, intent.from_entry)
        if (intent.trail_price is not None and intent.trail_offset is not None
                and trail_key not in self._trail_activation_reported):
            # The PositionPort amend surface carries no ``trail_price`` — the
            # deferred-activation trailing of the direct paths is not
            # expressible per-leg, so the trailing arms IMMEDIATELY at
            # ``trail_offset``. Trades can exit earlier/tighter than the Pine
            # simulation; surface it (once per exit) instead of diverging
            # silently.
            self._trail_activation_reported.add(trail_key)
            _blog_warning(
                "one-way bracket replication: trail_price activation is not "
                "expressible per-leg — trailing on %s (exit %r) arms "
                "immediately at offset %s instead of after the activation "
                "level %s",
                intent.symbol, intent.pine_id, intent.trail_offset,
                intent.trail_price,
            )
        legs = await _read_legs(port, intent.symbol)
        side, on_side = net_survivor_legs(legs)
        # The net side must be the one this exit protects: ``intent.side`` is
        # the CLOSING direction, so a sell-side exit belongs to a long. When
        # the book has flipped since the exit was placed — the parent entry was
        # reversed away and the stale exit is re-dispatched against the new
        # opposite position — the levels are upside down for that side (a
        # long's take-profit sits ABOVE the price, a short's must sit below).
        # Venues reject that outright, which escalates into a defensive close
        # for a parent that no longer exists. The position this exit protects
        # is simply not there, so it is the same non-halting skip as a flat
        # book: no bracket, re-evaluated on the next bar.
        protected_side = 'buy' if intent.side == 'sell' else 'sell'
        if side == 'flat' or not on_side or side != protected_side:
            return BracketFanResult(legs=(), skipped=True)
        # Incomplete-book gate: a protected-side position entry whose FILL is
        # not yet journaled means the snapshot above cannot be tied to that
        # entry's venue leg yet (the leg id is only learned from the fill, and
        # on a same-sync pyramid the fill event has not been drained when this
        # fan runs). Fanning now would attach the bracket onto the OLDER legs
        # only, the intent would go active, and nothing ever re-fans — the
        # newest leg of a pyramid round would stay permanently unprotected. A
        # mixed book is worse: netting computed over the incomplete snapshot
        # can mis-drop a true survivor and the clear below would strip its live
        # bracket. Defer the WHOLE fan — the same non-halting skip as a flat
        # book — until every position-opening dispatch on this side has its
        # fill on the books; the re-run then covers all legs.
        if self._store_ctx is not None:
            pending_coid = self._pending_position_entry(protected_side)
            if pending_coid is not None:
                deferral_key = (intent.intent_key, pending_coid)
                since, flagged = self._book_incomplete_deferrals.get(deferral_key, (now, False))
                stalled = not flagged and now - since >= BOOK_INCOMPLETE_GRACE_S
                self._book_incomplete_deferrals[deferral_key] = (since, flagged or stalled)
                return BracketFanResult(
                    legs=(), skipped=True,
                    skip_reason=(
                        f"position book incomplete — entry dispatch "
                        f"{pending_coid!r} has not journaled its fill "
                        + (f"after {now - since:.0f}s; its leg stays unprotected "
                           f"until it does" if stalled else "yet")
                    ),
                    stalled=stalled,
                )
            for deferral_key in [k for k in self._book_incomplete_deferrals
                                 if k[0] == intent.intent_key]:
                del self._book_incomplete_deferrals[deferral_key]
        self._forget_vanished_legs(intent.symbol, {leg.leg_id for leg in legs})
        targets = self._target_legs(intent, on_side)
        # A re-attach can narrow the covered set: a manual hedge or an
        # interrupted reversal opens an opposing leg that virtually FIFO-consumes
        # the oldest majority leg, so :func:`net_survivor_legs` now drops a leg
        # this exit owned on a prior attach (or the leg's entry attribution now
        # assigns it to another entry). That leg is still PHYSICALLY open and
        # still carries this exit's levels from the earlier attach; its ownership
        # row would otherwise stay ``active`` and the restart pass would re-assert
        # the stale stop, which on a hit closes more than the net exposure and
        # flips the book. Retire those dropped legs FIRST: a survivor amend below
        # can raise :class:`OrderDispositionUnknownError`, which the dispatch path
        # parks while promoting the NEW intent into ``_active_intents`` — the next
        # diff then sees Pine == active and never re-runs this fan, so a
        # dropped-leg retirement left AFTER the survivor loop would never get
        # retried. The dropped set is disjoint from ``targets`` (the retirement
        # skips covered rows), so ordering it before the amend loop is otherwise
        # behaviour-neutral.
        await self._clear_dropped_survivor_legs(
            envelope, intent, targets={leg.leg_id for leg in targets}, port=port,
        )
        if not targets:
            # Every survivor leg belongs to another entry: the exit's own entry
            # holds no leg (:meth:`_target_legs`). Its rows on legs it covered
            # before were retired just above; nothing is amended, and the
            # caller re-evaluates on the next bar, like a flat book.
            return BracketFanResult(
                legs=(), skipped=True, entry_without_leg=True,
                skip_reason=(
                    f"entry {intent.from_entry!r} holds no open leg (every open "
                    f"leg belongs to another entry)"
                ),
            )
        # Bar-varying dispatch coid (matches the plugin's per-leg amend anchor);
        # one per bracket attach, reused across its legs.
        attach_coid = envelope.client_order_id(KIND_EXIT_SL)
        own_levels = _LegLevels(
            tp_price=intent.tp_price, sl_price=intent.sl_price,
            trail_offset=intent.trail_offset,
        )
        replicated: list[str] = []
        for leg in targets:
            own_coid = self._ownership_coid(intent.intent_key, leg.leg_id)
            # The other exits that own this leg keep their levels on it: the
            # amend carries the merge of every owner, this exit's new levels
            # standing in for its own earlier row.
            owners = [
                (key, levels)
                for key, levels in self._active_owner_levels(intent.symbol, leg.leg_id)
                if key != intent.intent_key
            ]
            co_owned = bool(owners)
            owners.append((intent.intent_key, own_levels))
            levels = self._merge_owner_levels(
                intent.symbol, leg.leg_id, intent.side, owners,
            )
            prior_row = self._own_active_row(own_coid)
            if self._store_ctx is not None:
                create_bracket_ownership_row(
                    self._store_ctx,
                    coid=own_coid,
                    symbol=intent.symbol,
                    side=intent.side,
                    qty=intent.qty,
                    intent_key=intent.intent_key,
                    pine_entry_id=intent.pine_id,
                    from_entry=intent.from_entry,
                    leg_id=leg.leg_id,
                    attach_coid=attach_coid,
                    tp_price=intent.tp_price,
                    sl_price=intent.sl_price,
                    trail_price=intent.trail_price,
                    trail_offset=intent.trail_offset,
                    oca_name=intent.oca_name,
                    oca_type=intent.oca_type,
                )
            try:
                await self._amend_leg(
                    port, intent.symbol, leg.leg_id, intent.side, levels, attach_coid,
                )
            except ExchangeOrderRejectedError as exc:
                if co_owned:
                    self._refused_co_owned_attach(intent, leg.leg_id, own_coid, prior_row, exc)
                    raise
                # The amend was DEFINITIVELY rejected on a leg of an OPEN
                # position (we hold ``on_side`` legs), so the position is now
                # open and UNPROTECTED. PERSIST-FIRST wrote this leg's ownership
                # row before the call, but a reject means we are still alive (not
                # a crash), so release that one never-attached row synchronously
                # — leaving it active would make a later ownership-scoped clear
                # or restart replay believe this exit protects a leg it does not.
                # The legs already amended earlier in this fan keep their active
                # rows: those brackets DO exist, and the defensive close the
                # engine issues for the wrapped error flattens the whole
                # position. (A leg that vanished mid-fan surfaces NOT as a reject
                # but as the port's benign no-op, so it never reaches here; an
                # ambiguous timeout is intentionally not caught — the amend may
                # have landed, so its row stays active for replay / reconcile.)
                if self._store_ctx is not None:
                    update_bracket_ownership_state(
                        self._store_ctx, coid=own_coid,
                        new_state=BRACKET_OWN_STATE_RELEASED, close_row=True,
                    )
                raise BracketAttachAfterFillRejectedError(
                    f"one-way bracket attach rejected after fill "
                    f"(leg={leg.leg_id}, from_entry={intent.from_entry!r}): {exc}",
                    position_coid=f"__pyne_orphan__{intent.symbol}__{intent.from_entry}",
                    symbol=intent.symbol,
                    position_side=('buy' if intent.side == 'sell' else 'sell'),
                    qty=intent.qty,
                    position_deal_id=leg.leg_id,
                    from_entry=intent.from_entry,
                    exit_id=intent.pine_id,
                    level_crossed=exc.level_crossed,
                ) from exc
            replicated.append(leg.leg_id)
        return BracketFanResult(legs=tuple(replicated), skipped=False)

    def _pending_position_entry(self, protected_side: str) -> str | None:
        """Client-order-id of a protected-side position entry awaiting its fill.

        A live journal row with ``extras['kind'] == ENTRY_KIND_POSITION`` opens
        a venue leg immediately (MARKET dispatch or an adopted position), so
        while its cumulative ``filled_qty`` is short of ``qty`` the book
        snapshot is incomplete for ``protected_side``. Working (LIMIT / STOP)
        rows are excluded — they hold no leg until their fill promotes them to
        ``ENTRY_KIND_POSITION`` — and so are entries opening the opposite side.

        :param protected_side: Position side the exit protects (``'buy'`` /
            ``'sell'``).
        :return: The first pending row's ``client_order_id``, or ``None`` when
            every protected-side position entry has its fill on the books.
        """
        assert self._store_ctx is not None
        for row in self._store_ctx.iter_live_orders():
            if (row.extras or {}).get('kind') != ENTRY_KIND_POSITION:
                continue
            if row.side != protected_side:
                continue
            if row.filled_qty < row.qty - 1e-9:
                return row.client_order_id
        return None

    def _target_legs(
            self, intent: ExitIntent, on_side: list[PositionLeg],
    ) -> list[PositionLeg]:
        """The net-survivor legs ``intent``'s bracket covers, in FIFO order.

        An exit without a ``from_entry`` covers every survivor leg. An exit
        bound to a ``from_entry`` covers the legs attributed to that entry
        (:attr:`~PositionLeg.entry_id`) and every
        unattributed leg: an unattributed leg cannot be told apart, so it is
        protected by every exit of its side, which keeps a plugin that never
        attributes on the replicate-onto-all behaviour.

        When every survivor leg is attributed to another entry, the exit covers
        nothing. Attribution only ever comes from the bot's own records, and a
        leg whose attribution is lost or uncertain is unattributed (``None``),
        never another entry's, so this means the entry holds no leg: typically
        its leg was just closed by its own take-profit or stop at the venue and
        the fill is not booked yet, while Pine re-emits the exit. Covering the
        other entries' legs then would put this exit's (possibly already
        touched) levels on them and close or flatten a position it does not
        protect. :meth:`run_exit_bracket` skips the fan instead (non-halting,
        re-evaluated on the next bar), reported once per exit at WARNING.

        :param intent: The exit being replicated.
        :param on_side: The net-survivor legs of the side it protects.
        :return: The covered legs; empty when the entry holds no leg.
        """
        if not intent.from_entry:
            return list(on_side)
        targeted = [
            leg for leg in on_side
            if leg.entry_id is None or leg.entry_id == intent.from_entry
        ]
        if targeted:
            self._entry_without_leg_reported.discard(intent.intent_key)
            return targeted
        if intent.intent_key not in self._entry_without_leg_reported:
            self._entry_without_leg_reported.add(intent.intent_key)
            _blog_warning(
                "one-way bracket replication: every open leg of %s (legs %s) is "
                "attributed to an entry other than %r, so exit %r has no leg to "
                "protect and is not attached; it is re-evaluated on every bar (a "
                "leg its entry opens, or the entry's booked close, settles it)",
                intent.symbol, ", ".join(leg.leg_id for leg in on_side),
                intent.from_entry, intent.pine_id,
            )
        return []

    def _active_owner_levels(
            self, symbol: str, leg_id: str,
    ) -> list[tuple[str, _LegLevels]]:
        """The levels of every exit that actively owns ``leg_id``.

        Only ``active`` ownership rows count: a ``clearing`` row belongs to an
        exit being retired from the leg, whose levels must not stay on it.

        :param symbol: The leg's symbol.
        :param leg_id: The broker leg id.
        :return: ``(intent_key, levels)`` per owning row, each row's own levels.
        """
        if self._store_ctx is None:
            return []
        owners: list[tuple[str, _LegLevels]] = []
        for row in iter_active_bracket_ownerships(self._store_ctx):
            extras = row.extras or {}
            if (row.symbol != symbol
                    or extras.get(EXTRAS_KEY_BRACKET_OWN_LEG_ID) != leg_id
                    or extras.get(EXTRAS_KEY_BRACKET_OWN_STATE) != BRACKET_OWN_STATE_ACTIVE):
                continue
            owners.append((row.intent_key or row.client_order_id, self._row_levels(row)))
        return owners

    @staticmethod
    def _row_levels(row: 'OrderRow') -> _LegLevels:
        """The owning exit's own levels persisted on an ownership row."""
        extras = row.extras or {}
        return _LegLevels(
            tp_price=extras.get(EXTRAS_KEY_BRACKET_OWN_TP),
            sl_price=extras.get(EXTRAS_KEY_BRACKET_OWN_SL),
            trail_offset=extras.get(EXTRAS_KEY_BRACKET_OWN_TRAIL_OFFSET),
        )

    def _merge_owner_levels(
            self, symbol: str, leg_id: str, exit_side: str,
            owners: list[tuple[str, _LegLevels]],
    ) -> _LegLevels:
        """The one protection a leg carries for all of its owning exits.

        A venue leg holds a single take-profit, stop-loss and trailing stop, so
        the owners' levels are merged per field, deterministically and
        independent of the order the exits attached in. A field only one owner
        sets is taken as is, so a TP-only and an SL-only exit union. Where
        several owners set the same field differently the leg cannot express
        them all; it gets the more protective level, the one the price reaches
        first — for a long (sell exit) the highest stop-loss and the lowest
        take-profit, for a short (buy exit) the lowest stop-loss and the highest
        take-profit — and the tightest trailing offset. That conflict is
        reported once per leg and owner set at WARNING. No owner at all yields
        the all-``None`` clear.

        A fixed stop-loss of one exit and a trailing stop of ANOTHER exit are
        not a "first reached" pair: the port takes ``sl_price`` and
        ``trail_offset`` together and venues read the pair differently
        (Capital.com: the trail replaces the stop; cTrader: the trail is
        anchored at the stop), so the leg carries neither owner's stop as the
        owner asked for it. Both fields are sent as merged, with no
        price-dependent choice, and the combination is reported once per leg
        and owner set at WARNING. One exit asking for both is its own request
        and is not reported.

        :param symbol: The leg's symbol, for the report.
        :param leg_id: The broker leg id, for the report.
        :param exit_side: The owners' exit side (``"sell"`` protects a long).
        :param owners: ``(intent_key, levels)`` of every owning exit.
        :return: The merged levels.
        """
        long_side = exit_side == 'sell'
        tps = {levels.tp_price for _key, levels in owners if levels.tp_price is not None}
        sls = {levels.sl_price for _key, levels in owners if levels.sl_price is not None}
        trails = {levels.trail_offset for _key, levels in owners
                  if levels.trail_offset is not None}
        merged = _LegLevels(
            tp_price=(min(tps) if long_side else max(tps)) if tps else None,
            sl_price=(max(sls) if long_side else min(sls)) if sls else None,
            trail_offset=min(trails) if trails else None,
        )
        keys = frozenset(key for key, _levels in owners)
        if len(tps) > 1 or len(sls) > 1 or len(trails) > 1:
            report_key = (symbol, leg_id, keys, 'conflict')
            if report_key not in self._level_conflicts_reported:
                self._level_conflicts_reported.add(report_key)
                _blog_warning(
                    "one-way bracket: exits %s all protect leg %s of %s with "
                    "different levels, which one leg cannot carry; it gets the "
                    "level the price reaches first (tp=%s sl=%s trail=%s)",
                    ", ".join(sorted(format_intent_key(key) for key in keys)),
                    leg_id, symbol, merged.tp_price, merged.sl_price,
                    merged.trail_offset,
                )
        stop_keys = {key for key, levels in owners
                     if levels.sl_price is not None or levels.trail_offset is not None}
        if sls and trails and len(stop_keys) > 1:
            report_key = (symbol, leg_id, keys, 'sl_trail')
            if report_key not in self._level_conflicts_reported:
                self._level_conflicts_reported.add(report_key)
                _blog_warning(
                    "one-way bracket: exits %s combine a fixed stop-loss and a "
                    "trailing stop of different exits on leg %s of %s; the venue "
                    "takes the pair as one stop and interprets it its own way (a "
                    "trail replacing the stop, or anchored at it), so neither "
                    "exit's stop is carried as asked (sl=%s trail=%s)",
                    ", ".join(sorted(format_intent_key(key) for key in keys)),
                    leg_id, symbol, merged.sl_price, merged.trail_offset,
                )
        return merged

    async def _amend_leg(
            self, port: 'PositionPort', symbol: str, leg_id: str, side: str,
            levels: _LegLevels, coid: str, *, reassert: bool = False,
    ) -> bool:
        """Send ``levels`` as the whole protection of one leg, unless it already holds them.

        The skip rule: the amend is NOT sent when ``levels`` equal the levels
        of the last amend THIS process sent to the leg and the venue confirmed
        (the call returned), and no amend of the leg was started since. The
        record (:attr:`~OneWayEmulator._venue_levels`) is dropped before every
        amend and written back only when it returns, so the venue state counts
        as known only after a confirmed amend: an amend that ended ambiguous
        (:class:`OrderDispositionUnknownError`), refused, rate-limited or
        interrupted, and every leg after a restart, is always re-sent. A
        retirement left ``clearing`` therefore re-sends whenever its own amend
        failed; a leg the retirement never reached still holds the confirmed
        levels and is compared against them. ``reassert`` sends
        unconditionally — the restart replay, whose purpose is to re-assert a
        venue state the process does not trust.

        A skipped amend is no venue call at all: a venue may reject an amend
        that changes nothing, and any re-send resets a trailing stop the venue
        has already moved. The caller's row bookkeeping (persist-first,
        release) proceeds exactly as if it had been sent.

        :return: ``True`` when the amend was sent, ``False`` when skipped.
        """
        key = (symbol, leg_id)
        if not reassert and self._venue_levels.get(key) == levels:
            return False
        self._venue_levels.pop(key, None)
        await port.amend_bracket(
            symbol, leg_id, side=side,
            tp_price=levels.tp_price, sl_price=levels.sl_price,
            trail_offset=self._trail_price_distance(levels.trail_offset),
            coid=coid,
        )
        self._venue_levels[key] = levels
        return True

    def _forget_vanished_legs(self, symbol: str, live_ids: set[str]) -> None:
        """Drop the confirmed-levels record of every ``symbol`` leg no longer open."""
        for key in [key for key in self._venue_levels
                    if key[0] == symbol and key[1] not in live_ids]:
            del self._venue_levels[key]

    def _own_active_row(self, coid: str) -> 'OrderRow | None':
        """The live, ``active`` ownership row ``coid``, or ``None``."""
        if self._store_ctx is None:
            return None
        row = self._store_ctx.get_order(coid)
        if (row is None or row.closed_ts_ms is not None
                or (row.extras or {}).get(EXTRAS_KEY_BRACKET_OWN_STATE)
                != BRACKET_OWN_STATE_ACTIVE):
            return None
        return row

    def _refused_co_owned_attach(
            self, intent: ExitIntent, leg_id: str, own_coid: str,
            prior_row: 'OrderRow | None', exc: ExchangeOrderRejectedError,
    ) -> None:
        """Undo this attach's ownership row on a co-owned leg the venue refused.

        The venue still holds the merge it held before this attach. A first
        attach owned nothing there, so its row is released; a modify owned the
        leg with its previous levels, which are still part of that merge, so
        the row is restored to them (still ``active``: a later retirement must
        take them off the leg). Logged at WARNING with the refusal.
        """
        if self._store_ctx is not None:
            if prior_row is None:
                update_bracket_ownership_state(
                    self._store_ctx, coid=own_coid,
                    new_state=BRACKET_OWN_STATE_RELEASED, close_row=True,
                )
            else:
                self._store_ctx.upsert_order(own_coid, extras=dict(prior_row.extras or {}))
        _blog_warning(
            "one-way bracket: the venue refused exit %r of entry %r on leg %s of "
            "%s, which other exits also protect (%s: %s); the leg keeps their "
            "levels%s, and the exit's own levels are not attached — the refusal "
            "goes to the refused-exit recovery",
            intent.pine_id, intent.from_entry, leg_id, intent.symbol,
            type(exc).__name__, exc,
            "" if prior_row is None else " and this exit's previous ones",
        )

    def _report_retirement_refusal(
            self, symbol: str, leg_id: str, retiring: list['OrderRow'], exc: Exception,
    ) -> None:
        """WARN once per (leg, owner set) that a retirement re-assert did not go through.

        :param symbol: The leg's symbol.
        :param leg_id: The broker leg id.
        :param retiring: The leg's ``clearing`` rows, which stay ``clearing``.
        :param exc: The venue's refusal or throttle.
        """
        remaining = self._active_owner_levels(symbol, leg_id)
        retiring_keys = frozenset(row.intent_key or row.client_order_id for row in retiring)
        remaining_keys = frozenset(key for key, _levels in remaining)
        report_key = (symbol, leg_id, retiring_keys, remaining_keys)
        if report_key in self._retirement_refusals_reported:
            return
        self._retirement_refusals_reported.add(report_key)
        _blog_warning(
            "one-way bracket: retiring exit(s) %s from leg %s of %s did not go "
            "through — the venue answered the re-assert of %s with %s: %s. The "
            "retired exit's levels may still be armed on the leg; the "
            "retirement is retried on every sync until it lands",
            ", ".join(sorted(format_intent_key(key) for key in retiring_keys)),
            leg_id, symbol,
            ("the remaining exits' levels ("
             + ", ".join(sorted(format_intent_key(key) for key in remaining_keys)) + ")"
             if remaining_keys else "an empty protection (no exit owns the leg any more)"),
            type(exc).__name__, exc,
        )

    async def _reassert_remaining_owners(
            self, port: 'PositionPort', symbol: str, leg_id: str, side: str,
            coid: str,
    ) -> None:
        """Amend one leg to the merged levels of its remaining active owners.

        The retirement step of every clear path: the retiring rows are already
        ``clearing`` (so they no longer count as owners), and the leg keeps the
        protection of the exits that still own it — cleared (all ``None``)
        only when none does. A merge equal to the leg's confirmed levels (a
        retired exit every one of whose levels was dominated) is not re-sent
        (:meth:`_amend_leg`). Raises whatever the amend raises; the caller's
        ``clearing`` rows then stay live for the retry.
        """
        levels = self._merge_owner_levels(
            symbol, leg_id, side, self._active_owner_levels(symbol, leg_id),
        )
        await self._amend_leg(port, symbol, leg_id, side, levels, coid)

    async def _clear_dropped_survivor_legs(
            self, envelope: 'DispatchEnvelope', intent: ExitIntent, *,
            targets: set[str], port: 'PositionPort',
    ) -> None:
        """Retire + release this exit's owned legs that fell out of its covered set.

        On a re-attach the covered set can shrink (an opposing leg now virtually
        FIFO-consumes a leg this exit protected before, or the leg's attribution
        now assigns it to another entry). Each such leg is still physically open
        and still carries this exit's levels, so its ownership row is marked
        ``clearing`` PERSIST-FIRST, the leg is amended to the merged levels of
        its remaining owners (cleared only when none is left) under a clear coid
        distinct from the attach coid, and the row is released — mirroring
        :meth:`run_exit_bracket_clear` for the implicit drop. A ``*_NOT_FOUND``
        race on a vanished leg is the port's benign no-op; the restart pass would
        otherwise re-assert the stale stop.

        An ambiguous retirement (:class:`OrderDispositionUnknownError`) on a
        dropped leg must NOT abort this call: :meth:`run_exit_bracket` runs the
        drop FIRST, before the survivor amend loop, so propagating here would
        skip the NEW protection on the surviving legs entirely — the dispatch
        path then parks the error while promoting the new intent into
        ``_active_intents``, so the next diff sees Pine == active, never re-runs
        the fan, and the survivors stay on STALE TP/SL indefinitely. Leaving the
        row ``clearing`` lets :meth:`drain_clearing_rows` (per sync) and
        :meth:`restart_replay` retry the idempotent re-amend; the survivor amend
        below proceeds with the new levels. This mirrors the unknown-disposition
        handling in :meth:`drain_clearing_rows`. A definitive refusal or a
        venue throttle of the retirement is handled the same way (the row stays
        ``clearing``, see :meth:`_report_retirement_refusal`): the retirement
        concerns a leg this exit no longer covers, and surfacing it as the
        exit's own refusal would send the engine into the refused-exit recovery
        of a bracket that was never refused.
        """
        if self._store_ctx is None:
            return
        clear_coid = envelope.client_order_id(KIND_CANCEL)
        dropped: dict[str, list['OrderRow']] = {}
        for row in iter_active_bracket_ownerships(
                self._store_ctx, from_entry=intent.from_entry):
            if (row.intent_key or '') != intent.intent_key:
                continue
            leg_id: str | None = (row.extras or {}).get(EXTRAS_KEY_BRACKET_OWN_LEG_ID)
            if leg_id is None or leg_id in targets:
                continue
            dropped.setdefault(leg_id, []).append(row)
        for leg_rows in dropped.values():
            for row in leg_rows:
                update_bracket_ownership_state(
                    self._store_ctx, coid=row.client_order_id,
                    new_state=BRACKET_OWN_STATE_CLEARING,
                    extras_patch={EXTRAS_KEY_BRACKET_OWN_CLEAR_COID: clear_coid},
                )
        for leg_id, leg_rows in dropped.items():
            try:
                await self._reassert_remaining_owners(
                    port, intent.symbol, leg_id, leg_rows[0].side, clear_coid,
                )
            except OrderDispositionUnknownError:
                # The retirement did not confirm; leave the row ``clearing`` so
                # ``drain_clearing_rows`` / ``restart_replay`` retry the
                # idempotent re-amend, and DO NOT abort the survivor amend
                # below — those legs need their new protection now.
                continue
            except ExchangeOrderRejectedError as exc:
                # The venue refused the remaining owners' levels; the row stays
                # ``clearing`` for the per-sync drain, and the survivor amend
                # below still runs.
                self._report_retirement_refusal(intent.symbol, leg_id, leg_rows, exc)
                continue
            except ExchangeRateLimitError as exc:
                # Throttled: stop retiring (every further call would be refused
                # too); the rows stay ``clearing`` for the per-sync drain.
                self._report_retirement_refusal(intent.symbol, leg_id, leg_rows, exc)
                break
            for row in leg_rows:
                update_bracket_ownership_state(
                    self._store_ctx, coid=row.client_order_id,
                    new_state=BRACKET_OWN_STATE_RELEASED, close_row=True,
                )

    async def drain_clearing_rows(self, symbol: str, port: 'PositionPort') -> set[str]:
        """Finish + release any ``clearing`` bracket-ownership rows in-session.

        :meth:`_clear_dropped_survivor_legs` (and :meth:`run_exit_bracket_clear`)
        mark a row ``clearing`` PERSIST-FIRST, then amend its leg to the merged
        levels of the leg's remaining owners. An ambiguous
        :class:`OrderDispositionUnknownError` on that amend leaves the row
        stranded in ``clearing``: the dispatch path parks the exit and promotes
        it into ``_active_intents``, so the next diff sees Pine == active and
        never re-runs the owning fan, and the one-way orphan sweep skips active
        keys — the stale levels stay armed until the next restart's
        :meth:`restart_replay`.

        This drain closes that window. The engine calls it once per sync (after
        the orphan sweep): every leg with a live ``clearing`` row is re-amended to
        the merged levels of its CURRENT ``active`` owners (cleared when none is
        left) under the persisted clear coid, and its ``clearing`` rows are
        released, exactly like :meth:`_replay_bracket_leg`'s clearing branch but
        per-sync rather than restart-only. A ``clearing`` row is by definition
        marked-for-removal, NEVER wanted protection, so dropping its levels is
        always safe even while the owning exit is still active; the leg's
        ``active`` owners keep theirs, since the amend carries them. A vanished
        leg's bracket is moot — release the rows. The amend is a full
        replacement, so a retirement that already landed is a broker no-op. A
        timeout, a definitive refusal or a venue throttle leaves the rows
        ``clearing`` for the next sync to retry rather than halting the bot; a
        refusal or throttle is reported once per leg and owner set
        (:meth:`_report_retirement_refusal`).

        Returns the set of ``intent_key`` values whose LAST surviving ownership row
        the drain released this call. The engine drops the in-memory
        envelope/mapping for any such key whose owning exit Pine no longer emits —
        the orphan sweep's direct-clear timeout (``continue`` before
        ``_drop_envelope``) leaves that engine state behind, and the drain
        releasing the row here is the second half of the same retirement. Keys that
        still carry an ``active`` row (a live exit's survivor legs) are excluded, so
        a live bracket never loses its anchor.
        """
        if self._store_ctx is None:
            return set()
        rows = [
            row
            for row in iter_active_bracket_ownerships(self._store_ctx)
            if (row.extras or {}).get(EXTRAS_KEY_BRACKET_OWN_STATE)
            == BRACKET_OWN_STATE_CLEARING
        ]
        if not rows:
            return set()
        legs = await _read_legs(port, symbol)
        live_ids = {leg.leg_id for leg in legs}
        self._forget_vanished_legs(symbol, live_ids)
        by_leg: dict[str | None, list['OrderRow']] = {}
        for row in rows:
            by_leg.setdefault(
                (row.extras or {}).get(EXTRAS_KEY_BRACKET_OWN_LEG_ID), [],
            ).append(row)
        released: set[str] = set()
        for leg_id, leg_rows in by_leg.items():
            if leg_id is not None and leg_id in live_ids:
                first = leg_rows[0]
                try:
                    await self._reassert_remaining_owners(
                        port, symbol, leg_id, first.side,
                        (first.extras or {}).get(EXTRAS_KEY_BRACKET_OWN_CLEAR_COID)
                        or first.client_order_id,
                    )
                except OrderDispositionUnknownError:
                    # The re-amend did not confirm; leave the rows ``clearing`` so
                    # the next sync retries it (idempotent full replacement).
                    # Halting here would strand the bot on a recoverable broker
                    # round-trip.
                    continue
                except ExchangeOrderRejectedError as exc:
                    # The venue refused the remaining owners' levels (a stop now
                    # inside the venue's minimum distance, a closed market). The
                    # rows stay ``clearing`` and the next sync retries; raising
                    # would end the run, and after a restart the same drain would
                    # raise again on every start.
                    self._report_retirement_refusal(symbol, leg_id, leg_rows, exc)
                    continue
                except ExchangeRateLimitError as exc:
                    # Throttled: stop draining (every further amend would be
                    # refused too) but return what was already released, like a
                    # dropped link below; the rest stays ``clearing``.
                    self._report_retirement_refusal(symbol, leg_id, leg_rows, exc)
                    break
                except ExchangeConnectionError:
                    # The connection dropped mid-drain. Earlier rows in this loop
                    # may already be RELEASED (durably closed in the store) and
                    # their keys collected in ``released``; if we let the exception
                    # escape, that partial progress is lost — the caller logs +
                    # retries but never runs the envelope/mapping retirement, so a
                    # released row's stale anchor survives and a later re-emission
                    # of the exit is built on it instead of on a fresh anchor.
                    # Stop draining (the link is down)
                    # but RETURN what was already released so the engine retires it;
                    # these rows stay ``clearing`` for the next sync to retry.
                    break
            for row in leg_rows:
                update_bracket_ownership_state(
                    self._store_ctx, coid=row.client_order_id,
                    new_state=BRACKET_OWN_STATE_RELEASED, close_row=True,
                )
                if row.intent_key is not None:
                    released.add(row.intent_key)
        # Only report a key whose every ownership row is now gone: a key still
        # carrying an ``active`` row (a live exit's survivor legs) must keep its
        # envelope, so the engine must not retire it.
        if not released:
            return released
        still_owned = {
            row.intent_key
            for row in iter_active_bracket_ownerships(self._store_ctx)
            if row.intent_key is not None
        }
        return released - still_owned

    async def drain_residual_opens(self, port: 'PositionPort') -> None:
        """Reconcile any live reversal residual-open breadcrumb per sync (in-session).

        The restart counterpart :meth:`_replay_residual_opens` reconciles these
        breadcrumbs once at startup, but the engine latches that replay after the
        first sync (``_one_way_replay_done``). Two paths leave a breadcrumb live
        mid-session: :meth:`run_reversal`'s residual ``place_leg`` raising
        :class:`OrderDispositionUnknownError` (the open's fate unknown, so the
        breadcrumb is intentionally NOT discharged), and :meth:`_replay_residual_one`
        hitting the same on its own re-dispatch. Without this drain such a breadcrumb
        would wait for the NEXT process restart before being reconciled.

        This closes that window. The engine calls it once per sync (alongside
        :meth:`drain_clearing_rows`), reconciling every live breadcrumb of
        the run through the SAME :meth:`_replay_residual_one` path restart replay
        uses: if the residual's persist-first entry row already landed the breadcrumb
        is simply discharged (the entry / recovery path owns the open), otherwise the
        residual is re-opened under its deterministic entry coid (``KIND_ENTRY``, or
        ``KIND_ENTRY_TRIGGERED`` for a triggered reversal) — a duplicate is impossible
        at the exchange dedup, so a per-sync re-dispatch cannot double-open. A still-ambiguous re-dispatch leaves the row live for the
        next sync to retry rather than halting the bot, mirroring
        :meth:`drain_clearing_rows`. A successful in-session reversal discharges its
        own breadcrumb synchronously, so this only ever sees genuinely unresolved
        rows.
        """
        if self._store_ctx is None:
            return
        for row in list(iter_active_residual_opens(self._store_ctx)):
            await self._replay_residual_one(row, port)

    @staticmethod
    def _ownership_coid(intent_key: str, leg_id: str) -> str:
        """Stable per-(exit, leg) bracket-ownership row key.

        Keyed on the exit's ``intent_key`` (pine_id + from_entry) and the broker
        leg id, with no bar timestamp, so a modify re-attach upserts the SAME row
        and two exits sharing a pine_id but differing in from_entry never collide
        on a shared leg. Distinct from the bar-varying dispatch coid.
        """
        return f"bo:{intent_key}:{leg_id}"

    async def run_exit_bracket_clear(
            self, envelope: 'DispatchEnvelope', port: 'PositionPort',
    ) -> ClearFanResult:
        """Retire the cancelled exit from ONLY the legs it owns.

        A broadcast clear amends every position-side leg, stripping a bracket a
        DIFFERENT exit set. This consults the persisted ownership index instead:
        it retires only the rows matching the cancel's ``intent_key`` (or, for a
        cancel-all-by-pine_id with ``from_entry`` None, every row whose exit
        shares that pine_id). Each leg those rows sit on is amended to the merged
        levels of the exits that STILL own it — so a co-owning exit (an SL-only
        exit next to the cancelled TP-only one, or another entry's exit on an
        unattributed leg) keeps its protection — and is cleared only when no
        owner is left; then the rows are released. Without a persisted index
        (``store_ctx`` None) there is nothing to clear — a benign empty result.

        A definitive refusal (:class:`ExchangeOrderRejectedError`) of a leg's
        re-assert never escapes: the leg's rows stay ``clearing``, the refusal
        is reported once per leg and owner set at WARNING
        (:meth:`_report_retirement_refusal`) and the remaining legs are still
        retired; :meth:`drain_clearing_rows` retries the leg on every sync. A
        venue throttle (:class:`ExchangeRateLimitError`) is reported the same
        way and stops the loop, leaving the legs not reached ``clearing`` too.
        For the engine the cancel is then settled, exactly as when the
        retirement's disposition is unknown: the drain owns the leftover rows.
        Any other amend error (an ambiguous timeout, a dropped link)
        propagates; the rows of that leg and of the legs not reached yet stay
        ``clearing`` for :meth:`drain_clearing_rows` / :meth:`restart_replay`
        to finish.
        """
        intent = envelope.intent
        assert isinstance(intent, CancelIntent)
        if self._store_ctx is None:
            return ClearFanResult(legs=())
        cancel_coid = envelope.client_order_id(KIND_CANCEL)
        owned: dict[str, list['OrderRow']] = {}
        for row in iter_active_bracket_ownerships(
                self._store_ctx, from_entry=intent.from_entry):
            if not self._owns(row, intent):
                continue
            leg_id: str | None = (row.extras or {}).get(EXTRAS_KEY_BRACKET_OWN_LEG_ID)
            if leg_id is None:
                continue
            owned.setdefault(leg_id, []).append(row)
        # PERSIST-FIRST: mark every retiring row ``clearing`` BEFORE any amend,
        # so an ambiguous (timed-out) amend leaves rows that :meth:`restart_replay`
        # / :meth:`drain_clearing_rows` finish rather than re-asserting the
        # retired levels — the close-leg ``pending`` -> ``dispatched`` two-phase
        # applied to the clear. Without it a timed-out clear stays ``active`` and
        # the next restart resurrects the bracket the script asked to cancel.
        # Marking them all first also takes them out of the owner set the merge
        # below reads, whichever leg is amended first. Record the clear coid too,
        # so the retry re-amends under the SAME clear coid, which correlates the
        # venue call with this retirement rather than with the attach.
        for leg_rows in owned.values():
            for row in leg_rows:
                update_bracket_ownership_state(
                    self._store_ctx, coid=row.client_order_id,
                    new_state=BRACKET_OWN_STATE_CLEARING,
                    extras_patch={EXTRAS_KEY_BRACKET_OWN_CLEAR_COID: cancel_coid},
                )
        cleared: list[str] = []
        for leg_id, leg_rows in owned.items():
            try:
                await self._reassert_remaining_owners(
                    port, intent.symbol, leg_id, leg_rows[0].side, cancel_coid,
                )
            except ExchangeOrderRejectedError as exc:
                self._report_retirement_refusal(intent.symbol, leg_id, leg_rows, exc)
                continue
            except ExchangeRateLimitError as exc:
                self._report_retirement_refusal(intent.symbol, leg_id, leg_rows, exc)
                break
            for row in leg_rows:
                update_bracket_ownership_state(
                    self._store_ctx, coid=row.client_order_id,
                    new_state=BRACKET_OWN_STATE_RELEASED, close_row=True,
                )
                cleared.append(leg_id)
        return ClearFanResult(legs=tuple(cleared))

    @staticmethod
    def _owns(row, intent: 'CancelIntent') -> bool:
        """Does ``row`` belong to the exit ``intent`` cancels?

        Exact ``intent_key`` match when the cancel targets one (pine_id,
        from_entry); an ``f"{pine_id}\\0"`` prefix match for a cancel-all
        (``from_entry`` None) that drops every exit sharing the pine_id.
        """
        row_key = row.intent_key or ''
        if intent.from_entry is not None:
            return row_key == intent.intent_key
        return row_key.startswith(f"{intent.pine_id}\0")

    # === Restart replay ===================================================

    async def restart_replay(self, port: 'PositionPort') -> bool:
        """Resume any close-leg fan-out or bracket replication a crash interrupted.

        Called at startup until it reports completion. Three ordered passes:
        pending close-leg rows are reconciled against the live legs and the
        residual re-dispatched (an already-settled leg is never re-closed); THEN
        reversal residual-open breadcrumbs are reconciled (the closes are now
        resolved, so a still-owed residual is safely re-opened) and the open
        re-dispatched only when its own entry row never landed; THEN each
        still-open leg with live bracket-ownership rows is amended once to the
        merged levels of its active owners (finishing any interrupted
        retirement on it), and the rows of a vanished leg are released.

        A bracket the venue refuses to re-assert never ends the run: a closed
        market leaves its row active and the replay reports itself incomplete,
        any other definitive refusal is recorded for the engine
        (:meth:`take_replay_refusals`), which re-arms the exit through its
        regular dispatch path.

        :return: ``True`` when every pass ran to completion. ``False`` when a
            re-dispatched close leg or a bracket re-assert was refused because
            the market is closed (or a close leg was refused definitively): the
            row stays as it is, the later passes are not reached, and the
            caller runs the replay again on its next sync.
        """
        if self._store_ctx is None:
            return True
        if not await self._replay_close_legs(port):
            return False
        await self._replay_residual_opens(port)
        return await self._replay_bracket_ownership(port)

    async def _replay_close_legs(self, port: 'PositionPort') -> bool:
        """Reconcile + re-dispatch any pending close-leg rows (first replay pass).

        For every persisted ``pending`` close-leg row, reconcile against the live
        broker legs before re-sending: if the leg is gone, the close already
        landed — finalise the row without dispatching; if it is still open,
        re-dispatch only the residual (capped at the live leg size, re-snapped to
        the grid) so an already-partly-closed leg is never over-reduced. The
        natural-close fill dedup guards the fill side.

        :return: ``True`` when every pending row was settled, ``False`` when the
            exchange refused at least one re-dispatch (that row stays pending
            for the next replay).
        """
        if self._store_ctx is None:
            return True
        pending = list(iter_active_close_legs(self._store_ctx))
        if not pending:
            return True
        by_symbol: dict[str, list] = {}
        for row in pending:
            by_symbol.setdefault(row.symbol, []).append(row)
        settled = True
        for symbol, rows in by_symbol.items():
            legs = await _read_legs(port, symbol)
            live_by_id = {leg.leg_id: leg for leg in legs}
            quantize = await _read_quantizer(port, symbol)
            for row in rows:
                if not await self._replay_one(row, symbol, live_by_id, quantize, port):
                    settled = False
        return settled

    async def _replay_one(
            self, row, symbol: str, live_by_id: dict[str, PositionLeg], quantize, port: 'PositionPort',
    ) -> bool:
        """Reconcile + (if needed) re-dispatch one pending close-leg row.

        :return: ``True`` when the row was settled (finalised, or re-dispatched
            and finalised), ``False`` when the exchange definitively refused the
            re-dispatch — the leg is still open and the row stays pending so the
            next replay retries it (a bot restarted into a closed market must not
            halt on the refusal; the close is still owed once trading resumes).
        """
        if self._store_ctx is None:
            return True
        extras = row.extras or {}
        leg_id: str | None = extras.get(EXTRAS_KEY_CLOSE_LEG_ID)
        live_leg = live_by_id.get(leg_id) if leg_id is not None else None
        if live_leg is None or leg_id is None:
            # Leg vanished — the close landed before the crash. Finalise only.
            update_close_leg_state(
                self._store_ctx, coid=row.client_order_id,
                new_state=CLOSE_LEG_STATE_DISPATCHED, close_row=True,
            )
            return True
        persisted = extras.get(EXTRAS_KEY_CLOSE_LEG_VOLUME) or 0
        residual = min(int(persisted), quantize(live_leg.qty))
        if residual > 0:
            try:
                await port.close_leg(symbol, leg_id, residual, row.client_order_id)
            except ExchangeOrderRejectedError as exc:
                _blog_warning(
                    "restart replay: close of leg %s (%s x%d) refused by the "
                    "exchange, kept pending for the next sync: %s",
                    leg_id, symbol, residual, exc,
                )
                return False
        update_close_leg_state(
            self._store_ctx, coid=row.client_order_id,
            new_state=CLOSE_LEG_STATE_DISPATCHED, close_row=True,
        )
        return True

    async def _replay_residual_opens(self, port: 'PositionPort') -> None:
        """Re-dispatch any reversal residual-open a crash left un-persisted.

        Runs AFTER :meth:`_replay_close_legs`, so the reversal's FIFO closes are
        already reconciled and re-opening the residual cannot race them. For each
        live breadcrumb: if the residual's own entry row already exists, the
        ``place_leg`` persist-first write landed and the entry journal / startup
        recovery own it — just clear the breadcrumb; otherwise rebuild the
        dispatch envelope and re-open the residual under the SAME deterministic
        entry coid (``KIND_ENTRY``, or ``KIND_ENTRY_TRIGGERED`` for a triggered
        reversal; a duplicate is therefore impossible at the exchange dedup),
        then clear it.
        """
        if self._store_ctx is None:
            return
        for row in list(iter_active_residual_opens(self._store_ctx)):
            await self._replay_residual_one(row, port)

    async def _replay_residual_one(self, row, port: 'PositionPort') -> None:
        """Reconcile + (if needed) re-dispatch one residual-open breadcrumb."""
        if self._store_ctx is None:
            return
        extras = row.extras or {}
        entry_coid: str | None = extras.get(EXTRAS_KEY_RESIDUAL_OPEN_ENTRY_COID)
        if (entry_coid is not None
                and self._store_ctx.get_order(entry_coid) is not None):
            # ``place_leg`` reached its persist-first entry-row write before the
            # crash — the open is durable and owned by the entry path. Discharge.
            clear_residual_open_row(self._store_ctx, row.client_order_id)
            return
        # The residual open never persisted (persist-first guarantees the entry
        # row precedes any wire send, so its absence proves no dispatch happened —
        # re-opening cannot double-open). The breadcrumb is written BEFORE the
        # FIFO closes, so a crash in that window can also leave the opposing legs
        # un-closed AND their close-leg rows un-persisted — ``_replay_close_legs``
        # then has nothing to re-dispatch. Re-opening the residual on top of those
        # still-live opposing legs would over-expose the book. Reconcile against
        # the live legs first: a genuine reversal (positive residual) fully
        # consumes the opposing exposure, so every still-live opposing leg is an
        # owed close — FIFO-close them under the SAME deterministic parent coid
        # (idempotent at the exchange dedup) before the residual open.
        run_tag = extras.get(EXTRAS_KEY_RESIDUAL_OPEN_RUN_TAG)
        bar_ts_ms = extras.get(EXTRAS_KEY_RESIDUAL_OPEN_BAR_TS_MS)
        retry_seq = extras.get(EXTRAS_KEY_RESIDUAL_OPEN_RETRY_SEQ)
        assert isinstance(run_tag, str)
        assert isinstance(bar_ts_ms, int)
        assert isinstance(retry_seq, int)
        opposing_side = 'sell' if row.side == 'buy' else 'buy'
        live_legs = await _read_legs(port, row.symbol)
        owed_closes = tuple(
            LegClose(leg_id=leg.leg_id, qty=leg.qty)
            for leg in live_legs
            if leg.side == opposing_side and leg.qty > 0.0
        )
        if owed_closes:
            parent_coid = row.client_order_id.removesuffix(':residual')
            await self._fan_out_closes(
                owed_closes, symbol=row.symbol, side=row.side,
                intent_key=row.intent_key or row.pine_entry_id or '',
                pine_id=row.pine_entry_id or '', parent_coid=parent_coid, port=port,
                skippable=False,
            )
        # Quarantine gate: the reopen leg is a new-exposure dispatch. The
        # owed FIFO closes above still ran (risk-reducing), but the reopen
        # is withheld and the breadcrumb stays live — the committed
        # reversal finishes on the replay after the operator restart.
        if (self._block_exposure_reopens is not None
                and self._block_exposure_reopens()):
            _blog_warning(
                "one-way residual reopen %r withheld by quarantine; "
                "breadcrumb stays pending until the operator restart",
                row.client_order_id,
            )
            return
        # Rebuild the intent with the ORIGINAL dispatch's coid kind: a
        # triggered entry's residual went out under ``KIND_ENTRY_TRIGGERED``
        # (the plugin picks the kind from ``triggered_entry``), so the
        # replayed ``place_leg`` must persist and dedup under that same coid —
        # otherwise a second ambiguous round-trip would land an entry row the
        # breadcrumb's ``entry_coid`` check never finds, re-opening the
        # residual again. The flag is recovered from the persisted coid's kind
        # code rather than a separate extras field — both forms carry it raw:
        # canonical in ``{k}{r}``, wire in the raw prefix. A wire-form coid's
        # length IS the minting venue's budget, so replaying it at
        # ``len(entry_coid)`` re-mints the byte-identical id; a canonical coid
        # replays under the default (identity) budget.
        parsed_entry = parsed_wire = None
        coid_max_len = CLIENT_ORDER_ID_MAX_LEN
        if entry_coid is not None:
            parsed_entry = parse_client_order_id(entry_coid)
            if parsed_entry is None:
                parsed_wire = parse_wire_client_order_id(entry_coid)
                if parsed_wire is not None:
                    coid_max_len = len(entry_coid)
        entry_kind = (
            parsed_entry.kind if parsed_entry is not None
            else parsed_wire.kind if parsed_wire is not None
            else None
        )
        triggered = entry_kind == KIND_ENTRY_TRIGGERED
        envelope = DispatchEnvelope(
            intent=EntryIntent(
                pine_id=row.pine_entry_id, symbol=row.symbol, side=row.side,
                qty=row.qty, order_type=OrderType.MARKET,
                triggered_entry=triggered,
            ),
            run_tag=run_tag, bar_ts_ms=bar_ts_ms, retry_seq=retry_seq,
            coid_max_len=coid_max_len,
        )
        try:
            await port.place_leg(envelope, row.qty)
        except ExchangeOrderRejectedError:
            # The exchange definitively refused the residual entry — nothing
            # opened. ``restart_replay`` runs inside the sync startup wrapper,
            # which only catches ``ExchangeConnectionError`` (sync_engine), so
            # propagating here would abort startup and the breadcrumb would
            # survive un-cleared, retrying the same rejected residual on every
            # later sync. Discharge the breadcrumb and stop: the residual the
            # exchange rejected and Pine never re-signalled must not re-open.
            clear_residual_open_row(self._store_ctx, row.client_order_id)
            return
        except OrderDispositionUnknownError:
            # The residual entry's fate is unknown (ambiguous round-trip). Leave
            # the breadcrumb live so the next sync's replay / drain reconciles it
            # against the then-known leg state, but do NOT propagate — that would
            # abort startup on a recoverable round-trip, mirroring the bracket
            # re-clear path above.
            return
        clear_residual_open_row(self._store_ctx, row.client_order_id)

    async def _replay_bracket_ownership(self, port: 'PositionPort') -> bool:
        """Re-assert / finish live bracket-ownership rows (second replay pass).

        Rows are handled per leg, because a leg carries ONE protection merged
        from all of its owners (:meth:`_merge_owner_levels`): re-asserting each
        row's own levels in turn would leave the leg on whichever row came last.
        See :meth:`_replay_bracket_leg`.

        :return: ``False`` when a re-assert met a closed market (its rows stay
            active for the next replay), ``True`` otherwise.
        """
        if self._store_ctx is None:
            return True
        rows = list(iter_active_bracket_ownerships(self._store_ctx))
        if not rows:
            return True
        by_symbol: dict[str, dict[str | None, list['OrderRow']]] = {}
        for row in rows:
            by_symbol.setdefault(row.symbol, {}).setdefault(
                (row.extras or {}).get(EXTRAS_KEY_BRACKET_OWN_LEG_ID), [],
            ).append(row)
        complete = True
        for symbol, by_leg in by_symbol.items():
            legs = await _read_legs(port, symbol)
            live_ids = {leg.leg_id for leg in legs}
            for leg_id, leg_rows in by_leg.items():
                if not await self._replay_bracket_leg(
                        leg_rows, symbol, leg_id, live_ids, port):
                    complete = False
        return complete

    async def _replay_bracket_leg(
            self, rows: list['OrderRow'], symbol: str, leg_id: str | None,
            live_ids: set[str], port: 'PositionPort',
    ) -> bool:
        """Re-assert / finish the ownership rows of one leg, or release a vanished leg.

        If the leg has vanished, the bracket it carried is moot — its rows are
        released so they leave the live index. Otherwise the leg is amended ONCE
        to the merged levels of its ``active`` rows:

        * With a ``clearing`` row on the leg (a retirement a crash or an
          ambiguous timeout interrupted), the amend finishes it under that row's
          persisted clear coid, so the venue call correlates with the
          retirement, and the ``clearing`` rows are released. The retired
          levels are NEVER re-asserted; a leg left without an active owner is
          cleared. A coid only correlates: the amend may carry levels other
          than the ones first sent under it (the owners changed since), which
          the :meth:`~PositionPort.amend_bracket`
          contract allows.
        * Otherwise the merge is re-asserted under the attach coid of the most
          recently written active row.

        The amend is always sent (``reassert``): the replay exists because the
        venue state is not trusted after a restart, so it is never skipped as
        unchanged.

        :return: ``False`` when the venue refused the re-assert because the
            market is closed or throttled the call, ``True`` otherwise.
        """
        if self._store_ctx is None:
            return True
        if leg_id is None or leg_id not in live_ids:
            if leg_id is not None:
                self._venue_levels.pop((symbol, leg_id), None)
            for row in rows:
                update_bracket_ownership_state(
                    self._store_ctx, coid=row.client_order_id,
                    new_state=BRACKET_OWN_STATE_RELEASED, close_row=True,
                )
            return True
        clearing = [
            row for row in rows
            if (row.extras or {}).get(EXTRAS_KEY_BRACKET_OWN_STATE)
            == BRACKET_OWN_STATE_CLEARING
        ]
        active = [row for row in rows if row not in clearing]
        side = rows[0].side
        levels = self._merge_owner_levels(
            symbol, leg_id, side,
            [(row.intent_key or row.client_order_id, self._row_levels(row))
             for row in active],
        )
        if clearing:
            first = clearing[0]
            try:
                await self._amend_leg(
                    port, symbol, leg_id, side, levels,
                    (first.extras or {}).get(EXTRAS_KEY_BRACKET_OWN_CLEAR_COID)
                    or first.client_order_id,
                    reassert=True,
                )
            except (OrderDispositionUnknownError, ExchangeOrderRejectedError,
                    ExchangeRateLimitError) as e:
                # The retirement did not confirm, or the venue refused it (a
                # closed market). ``restart_replay`` runs inside the sync startup
                # wrapper, which only catches ``ExchangeConnectionError``;
                # propagating would abort startup on a recoverable round-trip.
                # Leave the rows ``clearing`` (NOT released) so the next
                # ``drain_clearing_rows`` / replay retries the idempotent
                # re-amend, mirroring ``drain_clearing_rows``.
                if not isinstance(e, OrderDispositionUnknownError):
                    self._report_retirement_refusal(symbol, leg_id, clearing, e)
                return True
            for row in clearing:
                update_bracket_ownership_state(
                    self._store_ctx, coid=row.client_order_id,
                    new_state=BRACKET_OWN_STATE_RELEASED, close_row=True,
                )
            return True
        newest = max(active, key=lambda owner: (owner.updated_ts_ms, owner.client_order_id))
        try:
            await self._amend_leg(
                port, symbol, leg_id, side, levels,
                (newest.extras or {}).get(EXTRAS_KEY_BRACKET_OWN_ATTACH_COID)
                or newest.client_order_id,
                reassert=True,
            )
        except (MarketClosedError, ExchangeRateLimitError) as e:
            _blog_warning(
                "one-way bracket re-assert of %r on leg %s declined (%s: %s) — "
                "retried on the next sync", newest.client_order_id,
                leg_id, type(e).__name__, e,
            )
            return False
        except ExchangeOrderRejectedError as e:
            for row in active:
                self._replay_refusals.append(BracketReplayRefusal(
                    intent_key=row.intent_key or row.client_order_id,
                    leg_id=leg_id, tp_price=levels.tp_price,
                    sl_price=levels.sl_price, error=e,
                ))
        return True
