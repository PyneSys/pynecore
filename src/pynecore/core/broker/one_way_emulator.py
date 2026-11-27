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
unit-testable in isolation, with ``store_ctx=None`` for the no-persistence path. The
per-leg bracket ownership is kept either way: in the store when one is open (and
replayed after a restart), else in a
:class:`~pynecore.core.broker.ownership_journal.MemoryOwnershipJournal` for the
process lifetime. The close-leg rows follow the same split.

Concurrency. The engine's bridge can stop waiting for a port call while the call
keeps running, so a later sync can start a pass over the same venue entity. The
emulator serializes per entity, not with ad-hoc flags per path:

* **A leg close** (one close-leg row) is owned by one pass at a time
  (``_close_rows_busy``): the fan-out holds it while it sends, the settlement
  (:meth:`OneWayEmulator._settle_close_row`) while it reads the venue's evidence
  and re-sends. A row another pass holds is "not decided yet", never re-sent.
  Every send of a close is recorded on its row before it goes out, in the store
  or in process memory, so no path re-sends a close of unknown outcome under its
  id; every re-send is capped by what the caller's fresh plan says is safe.
* **A leg's protection** has one write at a time (``_unsettled_writes``). A
  repeat of the unsettled write bumps its generation; a venue read taken while
  the generation changed judges an earlier send and is discarded.
* **A symbol's settle pass** (:meth:`OneWayEmulator.settle_leg_writes`) runs
  once at a time (``_settling_symbols``).
* **A symbol's closes** are sent by one pass at a time (``_closing_symbols``),
  from the pass's fresh read to its last send: every close pass sizes its
  closes to the net exposure its own read shows, and a second pass reading
  before the first one's closes executed would spend that budget again (two
  passes each closing the net of a mixed book flip it). A dispatch that finds
  the symbol taken raises
  :class:`~pynecore.core.broker.exceptions.ExchangeConnectionError` before
  reading (nothing sent, retried on the next sync); a recovery pass skips the
  symbol and reports itself incomplete or leaves its row for the next sync.
  The claim ends when the pass's sends return, not when they execute, so
  every send also stays reserved against its leg until the read or the
  venue's record of the send shows it: a later close or market reversal
  plans on the read less those reservations
  (:meth:`OneWayEmulator._reserved_view`). The reservations are recorded on
  the close-leg rows, which stay live while one is held, so they outlast a
  restart.
* **A fill-time netting record** is planned by one pass at a time
  (``_netting_rows_busy``), from its read to its breadcrumb write, and only
  while none of its earlier halves is live.
* **A reversal's residual open** (one breadcrumb) is owned by one pass at a
  time (``_residual_rows_busy``). It follows its closes only through a fresh
  read that shows the opposing legs gone, never on a close's dispatch alone,
  and once a close went out the breadcrumb is its only owner.

Every claim is a try-acquire taken with no ``await`` between the check and
the claim, released in a ``finally`` of the coroutine that took it (so it is
held while the bridge no longer waits for that coroutine), and the claimed
entity is read again after the claim. The entry points that read the venue and
then write to it, and what each claims:

======================================  =========================================
Entry point                             Claimed entities
======================================  =========================================
``run_close`` (script, engine-trigger,  ``_closing_symbols``; per leg
defensive and software-guard closes)    ``_close_rows_busy``
``run_reversal`` (market)               ``_closing_symbols``; per leg
                                        ``_close_rows_busy``; the breadcrumb's
                                        ``_residual_rows_busy``
``run_reversal`` (resting LIMIT/STOP)   none: no read, one ``place_leg``
``drain_residual_opens`` and the        ``_residual_rows_busy`` and
restart's residual replay               ``_closing_symbols``; per leg
                                        ``_close_rows_busy``
``drain_nettings`` (a filled resting    planning: ``_netting_rows_busy`` and
entry's fill-time netting)              ``_closing_symbols``; its two halves
                                        then as the residual drain
restart close-leg replay                ``_closing_symbols``; per row
                                        ``_close_rows_busy``
``run_exit_bracket``,                   per leg ``_unsettled_writes`` (one
``run_exit_bracket_clear``,             write at a time, its generation
``drain_clearing_rows``, restart        judging the reads); the merged levels
bracket replay                          are taken with no ``await`` before
                                        the write's claim
``settle_leg_writes`` (share watcher,   ``_settling_symbols``; per leg
unknown-write settlement)               ``_unsettled_writes`` and
                                        ``_LegWrite.generation``
======================================  =========================================

A software guard firing is no entry point of its own: its close goes out
through ``run_close``.
"""
import hashlib
import sqlite3
from dataclasses import dataclass, field, replace
from time import time_ns
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from pynecore.core.broker.emulator import (
    LegClose,
    aggregate_positions,
    net_survivor_exposure,
    plan_leg_close_volumes,
    plan_reversal,
    select_legs_for_close,
)
from pynecore.core.broker.exceptions import (
    BracketAttachAfterFillRejectedError,
    ExchangeConnectionError,
    ExchangeOrderRejectedError,
    ExchangeRateLimitError,
    JournalWriteError,
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
from pynecore.core.broker.ownership_journal import MemoryOwnershipJournal, OwnershipJournal
from pynecore.core.broker.models import (
    CancelIntent,
    CapabilityLevel,
    CloseIntent,
    DispatchEnvelope,
    EntryIntent,
    ExitIntent,
    OrderType,
    PositionLeg,
    format_intent_key,
)
from pynecore.core.plugin import ProviderError, is_retryable_provider_error
from pynecore.lib.log import broker_error as _blog_error
from pynecore.lib.log import broker_info as _blog_info
from pynecore.lib.log import broker_warning as _blog_warning
from pynecore.core.broker.store_helpers import (
    BRACKET_OWN_STATE_ACTIVE,
    BRACKET_OWN_STATE_CLEARING,
    BRACKET_OWN_STATE_RELEASED,
    CLOSE_LEG_STATE_PENDING,
    ENTRY_KIND_POSITION,
    EXTRAS_KEY_BRACKET_OWN_ATTACH_COID,
    EXTRAS_KEY_BRACKET_OWN_CLEAR_COID,
    EXTRAS_KEY_BRACKET_OWN_GUARD_TRAIL_EXTREME,
    EXTRAS_KEY_BRACKET_OWN_GUARD_TRAIL_OFFSET,
    EXTRAS_KEY_BRACKET_OWN_LEG_ID,
    EXTRAS_KEY_BRACKET_OWN_SL,
    EXTRAS_KEY_BRACKET_OWN_SOFTWARE,
    EXTRAS_KEY_BRACKET_OWN_STATE,
    EXTRAS_KEY_BRACKET_OWN_TP,
    EXTRAS_KEY_BRACKET_OWN_TRAIL_OFFSET,
    EXTRAS_KEY_BRACKET_OWN_TRAIL_PRICE,
    EXTRAS_KEY_CLOSE_LEG_ATTEMPTS,
    EXTRAS_KEY_CLOSE_FAN_VOLUME,
    EXTRAS_KEY_CLOSE_LEG_ID,
    EXTRAS_KEY_CLOSE_LEG_PRE_VOLUME,
    EXTRAS_KEY_CLOSE_LEG_SINCE_MS,
    EXTRAS_KEY_CLOSE_LEG_VOLUME,
    EXTRAS_KEY_CLOSE_PARENT_COID,
    EXTRAS_KEY_CLOSE_REQUESTED_VOLUME,
    EXTRAS_KEY_RESIDUAL_OPEN_BAR_TS_MS,
    EXTRAS_KEY_RESIDUAL_OPEN_CLOSE_PLAN,
    EXTRAS_KEY_RESIDUAL_OPEN_COUNTED_CLOSES,
    EXTRAS_KEY_RESIDUAL_OPEN_ENTRY_COID,
    EXTRAS_KEY_RESIDUAL_OPEN_NETTING,
    EXTRAS_KEY_RESIDUAL_OPEN_RETRY_SEQ,
    EXTRAS_KEY_RESIDUAL_OPEN_RUN_TAG,
    EXTRAS_KEY_NETTING_CONTENDED,
    EXTRAS_KEY_NETTING_EXPECTED,
    EXTRAS_KEY_NETTING_PASSES,
    EXTRAS_KEY_NETTING_SETTLED_MS,
    EXTRAS_KEY_NETTING_SINCE_MS,
    STATE_CLOSE_LEG,
    STATE_NETTING,
    advance_residual_open_attempt,
    absorb_netting_fills,
    clear_residual_open_row,
    count_residual_open_close,
    create_bracket_ownership_row,
    create_close_leg_row,
    create_netting_row,
    create_residual_open_row,
    finish_close_leg_row,
    iter_active_bracket_ownerships,
    iter_active_close_legs,
    iter_active_nettings,
    iter_active_residual_opens,
    netting_absorbed_row_coid,
    netting_expected,
    iter_close_leg_sends,
    lower_close_leg_send_pre_volume,
    release_close_leg_send,
    reserve_close_leg_send,
    update_bracket_ownership_state,
    update_close_leg_state,
)

#: Verdicts of :meth:`OneWayEmulator._settle_close_row`: the row is settled
#: (finalised, any provably owed part sent), cannot be decided yet (nothing
#: sent; the row stays pending), or the exchange refused the owed part.
_CLOSE_ROW_SETTLED = 'settled'
_CLOSE_ROW_UNDECIDED = 'undecided'
_CLOSE_ROW_REFUSED = 'refused'

#: Seconds an exit's incomplete-book deferral on one pending entry counts as
#: the routine wait for the entry's fill to drain. Past it the deferral is a
#: stall: the newest leg stays unprotected, so the caller warns. Measured in
#: time, not in syncs: the sync cadence is the engine's, not the venue's fill
#: latency (live: ctrader drained a pyramid fill in one or two syncs, bybit
#: cycle 438 took four syncs within the same second).
BOOK_INCOMPLETE_GRACE_S = 30.0

#: Seconds a protection write whose outcome is unknown is still expected to
#: take effect at the venue. While the leg does not show the write and this
#: window (counted from the first settle attempt) is open, the write may still
#: land, so nothing else is written to the leg; once it has passed the write
#: counts as lost. The value is a core assumption, not a measured venue
#: property: no venue documents how long after a client-side timeout it can
#: still apply a request.
UNSETTLED_WRITE_WINDOW_S = 30.0

#: Pine-unit quantities below this are treated as nothing.
_QTY_EPS = 1e-9

#: The recorded protection mode of a leg whose switch between two modes did
#: not complete (:meth:`OneWayEmulator._protect_leg`): it differs from every
#: mode, so the next settle pass re-drives the leg.
_PROTECTION_PENDING = 'pending'

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

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


async def _read_bracket_match(
        port: 'PositionPort', symbol: str, leg_id: str, write: '_LegWrite', trail_distance: float | None,
) -> bool | None:
    """Ask the port whether ``leg_id`` holds ``write``, with the :func:`_read_legs` fault mapping.

    ``None`` is the port's "cannot tell" (:meth:`PositionPort.bracket_matches`).
    """
    try:
        return await port.bracket_matches(
            symbol, leg_id, side=write.side,
            tp_price=write.levels.tp_price, sl_price=write.levels.sl_price,
            trail_offset=trail_distance, qty=write.qty,
        )
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


def _more_favourable(long_side: bool, current: float | None, price: float | None) -> float | None:
    """The more favourable of two trail prices for a long (the higher) or a short (the lower).

    ``None`` stands for no price; the other one wins.
    """
    if current is None:
        return price
    if price is None:
        return current
    return max(current, price) if long_side else min(current, price)


def _shrink_overlap(credit: int, credited_pre: int, pre: int) -> int:
    """How much of a shrink credited to one close send lies in another send's window.

    A send's window starts at the leg size it recorded (``pre``). The shrink
    credited to a send that recorded ``credited_pre`` happened after that
    send went out; whatever of it the leg had already lost when the other
    send recorded a smaller size (``credited_pre - pre``) lies before the
    other send's window.

    :param credit: The broker-grid shrink credited to the first send.
    :param credited_pre: The leg size the first send recorded.
    :param pre: The leg size the other send recorded.
    """
    return max(0, credit - max(0, credited_pre - pre))


def close_leg_client_order_id(parent_coid: str, leg_id: str) -> str:
    """The client order id of one leg's close in a close fan.

    Composed from the fan's own client order id, so it names one dispatch's
    close of one leg: two closes of the same leg never share it.

    :param parent_coid: The client order id of the close being fanned out.
    :param leg_id: The :attr:`~PositionLeg.leg_id`
        of the leg it reduces.
    """
    return f"{parent_coid}:{leg_id}"


def close_leg_attempt_client_order_id(leg_coid: str, attempt: int) -> str:
    """The client order id of a restart replay's re-send of one leg's close.

    A re-send never reuses an id an earlier send of the same close used (a
    venue may reject the duplicate, or may not dedup it at all), so each
    attempt after the first gets its own deterministic suffix.

    :param leg_coid: The leg close's own id (:func:`close_leg_client_order_id`).
    :param attempt: How many sends of this close went out before this one.
    """
    return f"{leg_coid}#r{attempt}"


#: Width of the ``retry_seq`` block a residual open's re-attempts take
#: (:func:`_residual_attempt_retry_seq`): the engine bumps an entry's
#: ``retry_seq`` one by one per spent id on a bar, so it never reaches the
#: first block in practice.
_RESIDUAL_RETRY_BLOCK = 36 ** 3


def _residual_attempt_retry_seq(retry_seq: int) -> int:
    """The ``retry_seq`` of a residual open's next attempt after ``retry_seq``.

    The first re-attempt of a dispatch under ``retry_seq`` ``n`` (below
    :data:`_RESIDUAL_RETRY_BLOCK`) goes to the start of the dispatch's own
    block, ``(n + 1) * _RESIDUAL_RETRY_BLOCK``; every later one takes the next
    value of that block. The engine's own same-bar bumps stay below the first
    block, so no engine dispatch shares a residual attempt's coid, and the
    blocks of two dispatches never meet.

    :param retry_seq: The abandoned attempt's ``retry_seq``.
    """
    if retry_seq < _RESIDUAL_RETRY_BLOCK:
        return (retry_seq + 1) * _RESIDUAL_RETRY_BLOCK
    return retry_seq + 1


def close_leg_row_client_order_id(coid: str) -> str:
    """The leg close's own id behind a ``coid`` :meth:`PositionPort.close_leg` received.

    The close-leg row (with the close's intent key and Pine id) is stored under
    the leg close's own id; a replay re-send arrives under an attempt id
    (:func:`close_leg_attempt_client_order_id`), which this maps back.

    :param coid: A ``coid`` passed to :meth:`PositionPort.close_leg`.
    """
    head, sep, tail = coid.rpartition('#r')
    if sep and tail.isdigit():
        return head
    return coid


def _grid_qty(qty: float, quantize: 'Callable[[float], int]') -> float:
    """The Pine quantity ``qty`` comes to once snapped onto the broker grid.

    The smallest quantity the quantizer maps to the volume ``qty`` snaps to,
    found by bisection (a quantizer is monotone): what a close of ``qty``
    sends, and so what its fill reports.

    :param qty: A Pine-unit quantity.
    :param quantize: The port's Pine-units to broker-grid quantizer.
    """
    volume = quantize(qty)
    if volume <= 0:
        return 0.0
    low, high = 0.0, qty
    for _ in range(80):
        mid = (low + high) / 2.0
        if quantize(mid) >= volume:
            high = mid
        else:
            low = mid
    return high


__all__ = [
    'close_leg_client_order_id',
    'close_leg_attempt_client_order_id',
    'close_leg_row_client_order_id',
    'CloseFanResult',
    'ReversalFanResult',
    'BracketFanResult',
    'BracketReplayRefusal',
    'ClearFanResult',
    'SoftwareLegTrigger',
    'LegSettleResult',
    'OneWayEmulator',
    'UNSETTLED_WRITE_WINDOW_S',
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


@dataclass
class _CloseSend:
    """One reserved close send as its close-leg row records it.

    :ivar row_coid: The close-leg row the send belongs to.
    :ivar attempt_coid: The client order id the send went out under.
    :ivar leg_id: The closed leg (the row's current leg id).
    :ivar parent_coid: The close fan the send belongs to.
    :ivar volume: The broker-grid volume sent.
    :ivar pre_volume: The leg's broker-grid size the send is credited from.
    :ivar since_ms: Epoch milliseconds the venue is asked about the send from.
    :ivar sent_ns: Epoch nanoseconds of the send, its place in the send order.
    :ivar dropped: Set once the view drops the send.
    """
    row_coid: str
    attempt_coid: str
    leg_id: str
    parent_coid: str
    volume: int
    pre_volume: int
    since_ms: int
    sent_ns: int
    dropped: bool = False


@dataclass
class _LegWrite:
    """A protection write of one leg whose outcome is not known yet.

    :ivar levels: The levels the write carries.
    :ivar sized_qty: The quantity the write limits the protection to, or
        ``None`` for a protection of the whole venue position.
    :ivar qty: The quantity sent to the port with the write.
    :ivar side: The exit side sent with the write.
    :ivar coid: The correlation id sent with the write.
    :ivar finished: ``False`` while the port call, or an identical repeat of
        it, is still running (the engine's bridge stopped waiting for it),
        ``True`` once it ended without a verdict.
    :ivar since: The caller's monotonic seconds at the first settle attempt
        after the call (or a repeat that also ended without a verdict) ended,
        or ``None`` before it.
    :ivar resent: Whether an identical write was sent again while this one
        was unsettled. The leg showing the protection then no longer proves
        that THIS write landed, so only the window settles it.
    :ivar reported: Whether the hold on the leg was already logged.
    :ivar generation: How many sends of this write started; a repeat bumps it.
        A venue read taken while it changed judges an earlier send, so
        :meth:`OneWayEmulator.settle_leg_writes` discards it.
    """
    levels: _LegLevels
    sized_qty: float | None
    qty: float
    side: str
    coid: str
    finished: bool = False
    since: float | None = None
    resent: bool = False
    reported: bool = False
    generation: int = 0


@dataclass
class _SoftwareGuard:
    """The software protection of one leg the venue bracket cannot be sized for.

    :ivar exit_side: The exit side (``"sell"`` protects a long).
    :ivar levels: The merged levels of the leg's owning exits.
    :ivar qty: The Pine-unit quantity a trigger closes.
    :ivar trail_extreme: The most favourable price the trailing stop trails
        from (a long's highest, a short's lowest), as the leg's ownership rows
        hold it; ``None`` until the first price arms the trail. The stop lies
        the trailing distance behind it.
    :ivar unapplied_trail_price: Memory only: the most favourable price a
        tick observed whose persist-first write of a new extreme failed. The
        guard trails from it as if it had landed, and the next tick writes it
        first, so the observed extreme is never lost.
    :ivar firing: ``True`` between a trigger and the engine's verdict on the
        close it dispatched.
    """
    exit_side: str
    levels: _LegLevels
    qty: float
    trail_extreme: float | None = None
    unapplied_trail_price: float | None = None
    firing: bool = False


@dataclass(frozen=True)
class SoftwareLegTrigger:
    """A software leg guard whose level the price reached.

    :ivar symbol: The leg's symbol.
    :ivar leg_id: The broker leg to reduce.
    :ivar side: The close side (``"sell"`` closes a long).
    :ivar qty: The Pine-unit quantity to close: the protected quantity, never
        the whole venue position.
    :ivar reason: ``"take_profit"``, ``"stop_loss"`` or ``"trailing_stop"``.
    :ivar level: The level the price reached.
    :ivar from_entry: The entry every exit owning the leg protects, or
        ``None`` when they protect different entries.
    :ivar exit_ids: The Pine ids of the exits owning the leg, for the log.
    """
    symbol: str
    leg_id: str
    side: str
    qty: float
    reason: str
    level: float
    from_entry: str | None
    exit_ids: tuple[str, ...] = field(default=())


@dataclass(frozen=True)
class LegSettleResult:
    """Outcome of one pass over the legs with an unsettled protection write.

    :ivar confirmed: Intent keys of the exits every owned leg of which now
        carries a confirmed protection, among the exits owning a leg the pass
        re-drove. A dispatch of theirs parked on an unknown outcome is settled.
    :ivar refused: Intent keys of the exits owning a leg whose re-drive the
        venue definitively refused, with the refusal: the venue does not carry
        the protection their ownership rows ask for.
    """
    confirmed: frozenset[str] = frozenset()
    refused: tuple[tuple[str, ExchangeOrderRejectedError], ...] = ()


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
    :ivar opened_orders: The ``ExchangeOrder``(s) the residual open produced
        within this call. Empty when ``open_qty`` was zero, and also when the
        open is deferred: a reversal opens its residual only once a fresh read
        after the closes shows no opposing leg, so while one is still listed
        the residual-open breadcrumb stays live and
        :meth:`OneWayEmulator.drain_residual_opens` opens it on a later sync
        and returns its orders there. ``open_qty`` is the planned size either
        way.
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
        cleared only when none does. Empty when the exit owned nothing (never
        attached, or already released).
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
            sized_leg_bracket: CapabilityLevel = CapabilityLevel.UNSUPPORTED,
    ) -> None:
        self._store_ctx = store_ctx
        #: The per-leg bracket-ownership rows (which exit owns which leg, with
        #: the exit's own levels): the run's store when one is open, so a
        #: restart replays them, else a journal in process memory. Every
        #: ownership path reads and writes them through this one interface.
        self._ownership: OwnershipJournal = (
            store_ctx if store_ctx is not None else MemoryOwnershipJournal())
        #: The close-leg rows (one per leg close, with every coid it was sent
        #: under) and the reversals' residual-open breadcrumbs: the run's
        #: store when one is open, so a restart replays them, else a journal
        #: in process memory, so a close of unknown outcome is never re-sent
        #: under its id and an owed residual open is never lost within the
        #: process either.
        self._close_journal: OwnershipJournal = (
            store_ctx if store_ctx is not None else MemoryOwnershipJournal())
        #: Entry coids of the residual opens handed to ``place_leg`` on a run
        #: without a store: the in-memory counterpart of the store's
        #: persist-first entry row, so a breadcrumb whose open was already
        #: sent (its outcome maybe unknown) is discharged, never re-opened.
        self._residual_entries_sent: set[str] = set()
        #: The dispatch envelope of each reversal whose residual open
        #: :meth:`run_reversal` left to the drain, by breadcrumb coid: the
        #: drain of the same process opens the residual under the reversal's
        #: own envelope; only a restart rebuilds it from the breadcrumb.
        self._residual_envelopes: dict[str, 'DispatchEnvelope'] = {}
        #: Residual-open breadcrumbs (coids) a pass is reconciling or opening
        #: right now (:meth:`_replay_residual_one`): a drain the engine's
        #: bridge stopped waiting for may still run when the next starts, and
        #: two passes must never open the same residual.
        self._residual_rows_busy: set[str] = set()
        #: Fill-time netting records (coids) a pass is planning right now
        #: (:meth:`drain_nettings`): two passes must never plan the same
        #: record's closes twice.
        self._netting_rows_busy: set[str] = set()
        #: Netting records already told that the quarantine withholds them.
        self._netting_quarantine_reported: set[str] = set()
        #: Attempt moves (:meth:`_abandon_residual_attempt`) whose journal
        #: write failed, by breadcrumb coid, with the envelope and entry coid
        #: of the abandoned attempt. The next pass over the breadcrumb writes
        #: the move before it reads the breadcrumb, so the entry row the
        #: abandoned attempt left behind never reads as the residual's open.
        self._owed_attempt_moves: dict[str, tuple['DispatchEnvelope', str]] = {}
        #: ``(intent_key, orders)`` of every residual a recovery pass opened
        #: and the engine has not taken yet (:meth:`take_opened_residuals`).
        #: Kept here as soon as the open returns, so a later row failing the
        #: same pass cannot lose the order ids.
        self._opened_residuals: list[tuple[str, tuple]] = []
        self._mintick = mintick
        #: The port's declared
        #: :attr:`~pynecore.core.broker.models.ExchangeCapabilities.sized_leg_bracket`:
        #: only ``NATIVE`` lets a venue bracket protect less than the whole
        #: venue position; every other level routes such a leg to the software
        #: leg guard.
        self._sized_leg_bracket = sized_leg_bracket
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
        #: Close-leg rows a pass is settling or sending right now
        #: (:meth:`_settle_close_row`, :meth:`_fan_out_closes`): a pass the
        #: engine's bridge stopped waiting for may still run when the next
        #: starts, and two passes must never send the same close.
        self._close_rows_busy: set[str] = set()
        #: Symbols a :meth:`settle_leg_writes` pass is running for.
        self._settling_symbols: set[str] = set()
        #: Symbols a pass is closing legs of right now, from its fresh read to
        #: its last send (:meth:`run_close`, the market path of
        #: :meth:`run_reversal`, a close-leg replay pass of
        #: :meth:`_replay_close_legs`, a residual drain or replay pass of
        #: :meth:`_replay_residual_one`): every such pass sizes its closes to
        #: the net exposure of its own read, so a second pass must never spend
        #: that budget again from a read that does not show the first one's
        #: sends.
        self._closing_symbols: set[str] = set()
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
        #: The protection this process last put on each leg (symbol, leg id)
        #: with an amend the venue confirmed: the levels and the quantity the
        #: protection is limited to (``None`` for the whole venue position);
        #: see :meth:`_amend_leg`. An entry is dropped before every amend and
        #: only restored when the amend returns or :meth:`settle_leg_writes`
        #: finds it in effect, so it is absent after any ambiguous, rejected
        #: or interrupted amend and after a restart.
        self._venue_levels: dict[tuple[str, str], tuple[_LegLevels, float | None]] = {}
        #: The protection write of each leg (symbol, leg id) whose outcome is
        #: not known: still running, or ended without a verdict. While a leg
        #: has one, :meth:`_amend_leg` writes nothing to it;
        #: :meth:`settle_leg_writes` resolves it.
        self._unsettled_writes: dict[tuple[str, str], _LegWrite] = {}
        #: Legs (symbol, leg id) under the software leg guard; see
        #: :meth:`_protect_leg` and :meth:`software_guard_triggers`.
        self._software_guards: dict[tuple[str, str], _SoftwareGuard] = {}
        #: Legs (symbol, leg id) already reported as software-guarded.
        self._software_guard_reported: set[tuple[str, str]] = set()
        #: How :meth:`_protect_leg` last decided to protect each leg (symbol,
        #: leg id) its owners ask levels of: ``(mode, protected quantity)``
        #: (:meth:`_protection_mode`). :meth:`settle_leg_writes` re-reads the
        #: legs every sync while any is recorded and re-drives a leg whose
        #: decision no longer holds, e.g. one that started sharing its venue
        #: position with exposure the run does not own.
        self._protection_modes: dict[tuple[str, str], tuple[str, float]] = {}
        #: Legs (symbol, leg id) whose unsettled write was settled and whose
        #: protection :meth:`settle_leg_writes` still has to bring to what
        #: their ownership rows ask for.
        self._legs_to_redrive: set[tuple[str, str]] = set()

    def iter_bracket_ownerships(self) -> 'Iterator[OrderRow]':
        """The live (``active`` or ``clearing``) bracket-ownership rows of this run.

        Read from the store when one is open, else from the in-memory journal,
        so a caller sees exactly the ownership the emulator merges and retires.
        """
        return iter_active_bracket_ownerships(self._ownership)

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

        The close owns the symbol's closes from its read to its last send
        (:attr:`~OneWayEmulator._closing_symbols`): while another pass closes
        legs of the symbol nothing is read or sent, and
        :class:`~pynecore.core.broker.exceptions.ExchangeConnectionError` is
        raised, so the dispatch path retries it on the next sync from a read
        that shows the other pass's closes.
        """
        intent = envelope.intent
        assert isinstance(intent, CloseIntent)
        self._claim_closing_symbol(intent.symbol)
        try:
            return await self._run_close_owned(envelope, intent, port)
        finally:
            self._closing_symbols.discard(intent.symbol)

    def _claim_closing_symbol(self, symbol: str) -> None:
        """Make this pass the one closing legs of ``symbol``, or raise without sending.

        :raises ExchangeConnectionError: When another pass closes legs of the
            symbol right now; nothing was read or sent.
        """
        if symbol in self._closing_symbols:
            raise ExchangeConnectionError(
                f"{symbol}: another pass is closing legs of this symbol right now; "
                f"nothing was sent, the order is retried on the next sync")
        self._closing_symbols.add(symbol)

    async def _run_close_owned(
            self, envelope: 'DispatchEnvelope', intent: CloseIntent, port: 'PositionPort',
    ) -> CloseFanResult:
        """The body of :meth:`run_close`, run while the pass owns the symbol's closes."""
        parent_coid = envelope.client_order_id(KIND_CLOSE)
        raw_legs = await _read_legs(port, intent.symbol)
        legs = await self._reserved_view(port, intent.symbol, raw_legs, own_parent=parent_coid)
        pos = aggregate_positions(intent.symbol, legs)
        if pos is None or pos.side == 'flat':
            # Nothing open, or offsetting legs net to flat — benign no-op.
            return CloseFanResult(
                legs=(), dispatched_qty=0.0, shortfall=0.0, skipped=False,
            )
        # CloseIntent.side is the close direction ("sell" closes a long); the
        # legs to reduce sit on the opposite side of the book.
        leg_side = 'buy' if intent.side == 'sell' else 'sell'
        qty = intent.qty
        if intent.synthetic_kind == 'leg_guard':
            # The guard was sized when it was armed; an opposing leg may have
            # grown since. It closes at most what of its leg survives the
            # netting NOW, so it can never flip the net position.
            surviving = self._exposure_by_leg(legs).get(intent.target_leg_id or '')
            qty = min(qty, surviving[1] if surviving is not None else 0.0)
            if surviving is None or surviving[0].side != leg_side or qty <= _QTY_EPS:
                return CloseFanResult(
                    legs=(), dispatched_qty=0.0, shortfall=0.0, skipped=False,
                )
        over_net = 0.0
        if intent.target_leg_id is None:
            # A close reduces the net position and never flips it: on a mixed
            # book (an opposing leg is open) it closes at most the net
            # exposure of its side, whatever quantity it was asked for. The
            # rest is reported as a shortfall, like a book holding less.
            net = (sum(leg.qty for leg in legs if leg.side == leg_side)
                   - sum(leg.qty for leg in legs if leg.side != leg_side))
            if qty > net + _QTY_EPS:
                over_net = qty - max(net, 0.0)
                qty = max(net, 0.0)
            if qty <= _QTY_EPS:
                return CloseFanResult(
                    legs=(), dispatched_qty=0.0, shortfall=over_net, skipped=False,
                )
        closes, shortfall = select_legs_for_close(
            qty, legs, leg_side, entry_id=self._close_bound_entry(intent, legs),
            leg_id=intent.target_leg_id,
        )
        shortfall += over_net
        if not closes:
            return CloseFanResult(
                legs=(), dispatched_qty=0.0, shortfall=shortfall,
                skipped=False,
            )
        closes = await self._widen_partial_closes_to_whole_legs(
            closes, legs, symbol=intent.symbol, port=port, intent_key=intent.intent_key,
        )
        # A script close or an engine-trigger partial close may be declined
        # per leg without halting (the leg settled or is busy on the venue;
        # the book re-derives next tick). Marketable exits and defensive
        # closes keep the definitive-reject contract.
        dispatched, dispatched_qty = await self._fan_out_closes(
            closes, raw_legs, symbol=intent.symbol, side=intent.side,
            intent_key=intent.intent_key, pine_id=intent.pine_id,
            parent_coid=parent_coid, port=port,
            skippable=intent.synthetic_kind in (None, 'partial_trigger', 'leg_guard'),
            requested_qty=intent.qty,
        )
        if not dispatched:
            # Every slice rounded below the broker grid — skip, do not halt.
            return CloseFanResult(
                legs=(), dispatched_qty=0.0, shortfall=shortfall, skipped=True,
            )
        return CloseFanResult(
            legs=tuple(dispatched),
            # Pine units, NOT the broker-grid volumes above: the caller
            # compares this with the fill quantities the legs report back,
            # and those arrive in Pine units on every venue. Only what went
            # on the wire counts — a fan a plugin skip cut short, or a leg
            # whose earlier close already executed part of it, covers less
            # than the plan.
            dispatched_qty=dispatched_qty,
            shortfall=shortfall, skipped=False,
        )

    @staticmethod
    def _close_bound_entry(intent: CloseIntent, legs: list[PositionLeg]) -> str | None:
        """The entry whose bindings a close's fill settles, ``None`` when it is unbound.

        Mirrors the engine's fill-side attribution
        (:meth:`~pynecore.core.broker.sync_engine.OrderSyncEngine._close_fill_entry_id`):
        a script ``strategy.close(id)`` is bound to ``id`` (``strategy.close_all``
        carries an empty ``pine_id`` and is unbound), an engine close aimed at
        one entry (``partial_trigger``, ``marketable_exit``, a ``leg_guard``)
        to its ``target_entry_id``, a defensive close to the entry of the leg it
        targets (``target_leg_id``: the leg the venue refused to protect,
        whose entry's bindings the defensive-close marker settles); every
        other close walks the book FIFO.

        :param intent: The close being fanned out.
        :param legs: The symbol's open legs, read for the defensive close's
            target leg.
        """
        if intent.synthetic_kind is None:
            return intent.pine_id or None
        if intent.synthetic_kind in ('partial_trigger', 'marketable_exit', 'leg_guard'):
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

        A resting LIMIT / STOP entry is placed as it is, with no read and no
        close: its fill is netted against the opposing legs afterwards
        (:meth:`drain_nettings`).

        Pine folds a reversal into one combined-size :class:`EntryIntent`
        assuming a netting auto-flip. On a hedging account that single order
        would open a separate opposing leg, so it is split: FIFO-close the
        opposing legs (up to the order size) and open only the residual in the
        order's own direction. Pure add (no opposing legs) opens the whole size;
        exact flatten (opposing exposure == order size) closes all and opens
        nothing.

        :meth:`PositionPort.close_leg` only dispatches a close, and a close can
        still execute in part (a reduce-only IOC market close cancelled after
        a partial fill). So a real reversal opens its residual exactly as the
        per-sync drain and the restart replay do
        (:meth:`_replay_residual_one`): only once a fresh read after the
        closes shows no opposing leg. When the read still lists one, the
        residual-open breadcrumb stays live and the result carries no opened
        order; :meth:`drain_residual_opens` settles the closes from the
        venue's evidence (re-sending only the part provably not executed) and
        opens the residual on the first sync whose read shows them gone.

        Once a close reached the wire the reversal is committed to the
        breadcrumb: a later :class:`ExchangeConnectionError` (a close, the read
        after the closes or the residual open that never left the client) is
        absorbed, the result carries no opened order, and the drain finishes
        the closes and the open. Only a fault before any close went out is
        raised, for the dispatch path to retry the whole entry.

        A market reversal owns the symbol's closes from its read to its
        residual open (:attr:`~OneWayEmulator._closing_symbols`): while another
        pass closes legs of the symbol nothing is read or sent and
        :class:`~pynecore.core.broker.exceptions.ExchangeConnectionError` is
        raised, so the dispatch path retries the whole entry on the next sync.
        """
        intent = envelope.intent
        assert isinstance(intent, EntryIntent)
        if intent.order_type is not OrderType.MARKET:
            # A resting LIMIT / STOP entry is not Pine's combined-size market
            # auto-flip: it rests as a fresh working order and is netted
            # against the opposing legs only at FILL time
            # (:meth:`drain_nettings`). Place it full-size now with NO
            # opposing-leg close — retiring the opposing exposure here would
            # flatten the position before the entry ever fires. ``place_leg``
            # runs the broker's own volume-bounds pre-flight, so an
            # out-of-range resting order still skips cleanly.
            opened = tuple(await port.place_leg(envelope, intent.qty))
            return ReversalFanResult(
                closes=(), open_qty=intent.qty, opened_orders=opened,
            )
        self._claim_closing_symbol(intent.symbol)
        try:
            return await self._run_reversal_owned(envelope, intent, port)
        finally:
            self._closing_symbols.discard(intent.symbol)

    async def _run_reversal_owned(
            self, envelope: 'DispatchEnvelope', intent: EntryIntent, port: 'PositionPort',
    ) -> ReversalFanResult:
        """The market path of :meth:`run_reversal`, run while the pass owns the symbol's closes."""
        parent_coid = envelope.client_order_id(KIND_CLOSE)
        raw_legs = await _read_legs(port, intent.symbol)
        legs = await self._reserved_view(port, intent.symbol, raw_legs, own_parent=parent_coid)
        plan = plan_reversal(intent.side, intent.qty, legs)
        # A port without a partial leg reduce closes the whole leg instead; a
        # widened plan never opens a residual (only an order smaller than the
        # opposing exposure slices a leg).
        widened = await self._widen_partial_closes_to_whole_legs(
            plan.closes, legs, symbol=intent.symbol, port=port, intent_key=intent.intent_key,
        )
        if plan.open_qty > 0.0 and len(widened) != len(plan.closes):
            # An opposing leg that cannot be closed stays open: opening the
            # residual next to it would hedge the book instead of reversing it.
            raise OrderSkippedByPlugin(
                f"{intent.symbol} reversal: an opposing leg cannot be closed whole "
                f"on this broker, so the residual is not opened",
                intent_key=intent.intent_key, reason='whole_leg_close_overshoots',
                context={'symbol': intent.symbol},
            )
        plan = replace(plan, closes=widened)
        # Pre-flight the broker volume bounds BEFORE any close lands: an
        # out-of-range order must raise the non-halting skip while it is still
        # true, otherwise the closes reduce the book yet the whole reversal is
        # reported skipped, desyncing the engine. A pure reduce (open_qty == 0)
        # validates the combined order size; a real reversal validates its
        # residual leg.
        preflight_qty = plan.open_qty if plan.open_qty > 0.0 else intent.qty
        await port.reject_out_of_range(envelope, preflight_qty)
        # Persist-first residual-open breadcrumb BEFORE the closes. It is the
        # record that an open is owed until a fresh read shows the opposing
        # legs gone: the drain opens the residual from it in-session, and a
        # crash after the closes land but before ``place_leg`` persists the
        # residual entry row would otherwise leave the book reduced with the
        # residual open lost — ``restart_replay`` reconciles it against the
        # residual entry row and re-dispatches only if that row never landed.
        # It lives in the close journal (the run's store, or process memory
        # on a run without one). Only a genuine reversal needs it: a pure add
        # (no opposing legs, ``plan.closes`` empty) leaves the book intact, so
        # re-opening it later would bypass the next sync's Pine re-evaluation
        # and resurrect an entry the strategy may no longer want. The
        # breadcrumb is gated on closes being actually owed, never on a
        # positive ``open_qty`` alone.
        residual_coid = f"{parent_coid}:residual"
        # An opposing leg the read lists but the reserved view leaves out is
        # being closed by an earlier send whose outcome is not shown yet: the
        # open follows it like a reversal's residual, only once a fresh read
        # shows the opposing exposure gone, never next to it.
        opposing_listed = any(leg.side != intent.side and leg.qty > 0.0 for leg in raw_legs)
        if not plan.closes and not opposing_listed:
            # Pure add: nothing to retire, the whole size opens now.
            opened_orders: tuple = ()
            if plan.open_qty > 0.0:
                opened_orders = tuple(await port.place_leg(envelope, plan.open_qty))
            return ReversalFanResult(closes=(), open_qty=plan.open_qty, opened_orders=opened_orders)
        wrote_breadcrumb = plan.open_qty > 0.0
        journal = self._close_journal
        if wrote_breadcrumb:
            # The breadcrumb's entry coid must match the kind ``place_leg`` will
            # persist under: a triggered market entry dispatches as
            # ``KIND_ENTRY_TRIGGERED``, so anchoring on ``KIND_ENTRY`` there would
            # make the replay's row-existence check miss the landed entry row
            # and double-open the residual.
            entry_kind = KIND_ENTRY_TRIGGERED if intent.triggered_entry else KIND_ENTRY
            # The combined order size: each close the reversal goes out under
            # is counted off it (:func:`count_residual_open_close`), so after
            # the planned closes it reads ``plan.open_qty``, and a close a
            # later drain has to send itself still comes off the open.
            create_residual_open_row(
                journal,
                coid=residual_coid,
                symbol=intent.symbol,
                side=intent.side,
                qty=intent.qty,
                intent_key=intent.intent_key,
                pine_entry_id=intent.pine_id,
                entry_coid=envelope.client_order_id(entry_kind),
                run_tag=envelope.run_tag,
                bar_ts_ms=envelope.bar_ts_ms,
                retry_seq=envelope.retry_seq,
            )
        progress: list[tuple[str, int]] = []
        try:
            dispatched, _sent_qty = await self._fan_out_closes(
                plan.closes, raw_legs, symbol=intent.symbol, side=intent.side,
                intent_key=intent.intent_key, pine_id=intent.pine_id,
                parent_coid=parent_coid, port=port, skippable=False,
                progress=progress,
                residual_coid=residual_coid if wrote_breadcrumb else None,
            )
        except ExchangeConnectionError:
            if not progress:
                # No close reached the wire: the book is as the plan read it,
                # and the dispatch path retries the whole entry from a fresh
                # read.
                if wrote_breadcrumb:
                    clear_residual_open_row(journal, residual_coid)
                raise
            # Part of the closes went out, so the book is already reduced: a
            # retry of the combined entry would re-plan it from the reduced
            # book and open more than the residual (or turn a flatten into a
            # reversal). The breadcrumb owns the rest — the drain closes what
            # the read still shows and opens the residual once the opposing
            # legs are gone. A reduction or flatten (nothing to open) records
            # its own close plan: the drain finishes exactly that plan, never
            # every opposing leg, so a partial reduction is not turned into a
            # flatten.
            if not wrote_breadcrumb:
                try:
                    create_residual_open_row(
                        journal,
                        coid=residual_coid,
                        symbol=intent.symbol,
                        side=intent.side,
                        qty=0.0,
                        intent_key=intent.intent_key,
                        pine_entry_id=intent.pine_id,
                        entry_coid=envelope.client_order_id(
                            KIND_ENTRY_TRIGGERED if intent.triggered_entry else KIND_ENTRY),
                        run_tag=envelope.run_tag,
                        bar_ts_ms=envelope.bar_ts_ms,
                        retry_seq=envelope.retry_seq,
                        close_plan=[(close.leg_id, close.qty) for close in plan.closes],
                    )
                except sqlite3.Error as exc:
                    # Closes already went out: the failure must never read as
                    # "nothing sent", or the dispatch path would re-plan the
                    # combined entry from the reduced book.
                    raise OrderDispositionUnknownError(
                        f"{intent.symbol} reversal {format_intent_key(intent.intent_key)}: "
                        f"{len(progress)} close(s) went out but the record of the rest "
                        f"could not be written ({exc})",
                        client_order_id=residual_coid, cause=exc,
                    ) from exc
            self._residual_envelopes[residual_coid] = envelope
            _blog_warning(
                "%s reversal %s: a close could not be sent after %d leg(s) went out; "
                "the rest is finished by the next sync",
                intent.symbol, format_intent_key(intent.intent_key), len(progress),
            )
            return ReversalFanResult(
                closes=tuple(progress), open_qty=plan.open_qty, opened_orders=(),
            )
        except ExchangeOrderRejectedError:
            # A close leg was DEFINITIVELY refused, so the residual never
            # opened. ``_dispatch_new`` turns this reject into a non-halting
            # skip for the entry (signal dropped, bot continues), so the
            # breadcrumb must be discharged here exactly as the residual
            # reject path does: leaving it live would let a later drain or
            # restart replay re-close + re-open a reversal the exchange
            # already rejected and Pine never re-signalled. An ambiguous
            # ``OrderDispositionUnknownError`` leaves the close's fate — and
            # thus a possibly-owed residual — unknown, so its breadcrumb
            # survives for the drain and is left intact.
            if wrote_breadcrumb:
                clear_residual_open_row(journal, residual_coid)
            raise
        opened_orders = ()
        if wrote_breadcrumb:
            # ``close_leg`` only dispatched the closes: a reduce-only market
            # close can still fill in part and be cancelled. The residual
            # therefore opens through the breadcrumb's own path, which opens
            # it only when a fresh read shows no opposing leg; otherwise the
            # breadcrumb stays live and the per-sync drain settles the closes
            # from the venue's evidence and opens the residual once they are
            # gone.
            row = journal.get_order(residual_coid)
            assert row is not None
            try:
                opened_orders = await self._replay_residual_one(row, port, envelope=envelope)
            except ExchangeConnectionError as exc:
                # The read after the closes, or the open itself, never reached
                # the venue. The closes did, so the breadcrumb alone owns the
                # open from here: raising would let the dispatch path re-plan
                # the combined entry from the reduced book and open more than
                # the residual.
                self._residual_envelopes[residual_coid] = envelope
                _blog_warning(
                    "%s reversal %s: the residual open is left to the next sync (%s)",
                    intent.symbol, format_intent_key(intent.intent_key), exc,
                )
                opened_orders = ()
        return ReversalFanResult(
            closes=tuple(dispatched), open_qty=plan.open_qty,
            opened_orders=opened_orders,
        )

    def _reserve_close(
            self, row_coid: str, attempt_coid: str, volume: int, pre_volume: int,
    ) -> None:
        """Record a close send on its row right before it goes out (:meth:`_reserved_view`)."""
        now_ns = time_ns()
        reserve_close_leg_send(
            self._close_journal, row_coid, attempt_coid=attempt_coid, volume=int(volume),
            pre_volume=int(pre_volume), since_ms=now_ns // 1_000_000, sent_ns=now_ns,
        )

    def _close_sends(self, symbol: str) -> list[_CloseSend]:
        """The reserved close sends of ``symbol``'s live close-leg rows, in send order."""
        sends: list[_CloseSend] = []
        for row, sent in iter_close_leg_sends(self._close_journal, symbol):
            extras = row.extras or {}
            sends.append(_CloseSend(
                row_coid=row.client_order_id, attempt_coid=str(sent[0]),
                leg_id=str(extras.get(EXTRAS_KEY_CLOSE_LEG_ID) or ''),
                parent_coid=str(extras.get(EXTRAS_KEY_CLOSE_PARENT_COID) or ''),
                volume=int(sent[1]), pre_volume=int(sent[2]), since_ms=int(sent[3]),
                sent_ns=int(sent[4]),
            ))
        sends.sort(key=lambda send: send.sent_ns)
        return sends

    async def _reserved_view(
            self, port: 'PositionPort', symbol: str, legs: list[PositionLeg], *,
            own_parent: str | None,
    ) -> list[PositionLeg]:
        """The legs of a fresh read less what earlier close sends may still take from them.

        :meth:`PositionPort.close_leg` only dispatches a close; the position
        read can list a leg in full after a close of it went out. A new close
        plan sized on that read would spend the same net exposure again, and
        two closes that each take the net of a mixed book flip it. So every
        close send is reserved against its leg until the read shows it: its
        volume counts as gone from the leg until the leg shrank by it or
        vanished, or until the venue's record of the send
        (:meth:`PositionPort.close_leg_executed`) settles it, and then only
        what the send executed and the read does not show yet counts. While
        the venue cannot answer, the whole send counts.

        The reservations live on the close-leg rows
        (:data:`~pynecore.core.broker.store_helpers.EXTRAS_KEY_CLOSE_LEG_RESERVED`),
        recorded before each send goes out, and a row stays live while any of
        its sends is reserved: after a restart the view still holds back a
        send the read does not show yet.

        One shrink of a leg is credited to one send only. Several sends can
        be outstanding against the same leg, each having recorded the leg's
        size when it went out, and a shrink the read shows falls inside the
        window of every one of them. It is credited to the sends in the order
        they went out: a send gets only what the earlier ones did not take,
        and a send whose credit settles it hands that credit down to the
        others (:meth:`_drop_reservation`), so the next view never credits
        the same shrink again.

        The sends of ``own_parent`` (the fan being planned, re-planned after
        an unknown outcome) do not count: that fan settles its own earlier
        sends from the venue's evidence (:meth:`_settle_close_row`).

        :param port: The position port.
        :param symbol: The symbol the legs belong to.
        :param legs: The fresh read of the symbol's legs.
        :param own_parent: The parent close coid of the fan being planned, or
            ``None`` when every outstanding send counts.
        :return: The legs with every outstanding reservation taken off; a
            leg fully reserved is left out. The read itself when nothing is
            reserved.
        """
        sends = self._close_sends(symbol)
        if not sends:
            return legs
        quantize = await _read_quantizer(port, symbol)
        live_by_id = {leg.leg_id: leg for leg in legs}
        outstanding: dict[str, int] = {}
        # Per leg: (pre_volume, credit) of the sends credited so far and
        # still held, in send order.
        credited: dict[str, list[tuple[int, int]]] = {}
        for held in sends:
            if held.dropped:
                continue
            live = live_by_id.get(held.leg_id)
            if live is None:
                held.dropped = True
                release_close_leg_send(self._close_journal, held.row_coid, held.attempt_coid)
                continue
            earlier = credited.setdefault(held.leg_id, [])
            claimed = sum(_shrink_overlap(credit, pre, held.pre_volume)
                          for pre, credit in earlier)
            window = max(0, held.pre_volume - quantize(live.qty))
            credit = min(held.volume, max(0, window - claimed))
            if credit >= held.volume:
                self._drop_reservation(held, credit, sends)
                continue
            if held.parent_coid == own_parent:
                earlier.append((held.pre_volume, credit))
                continue
            try:
                executed = await port.close_leg_executed(
                    symbol, held.leg_id, (held.attempt_coid,), held.since_ms)
            except Exception as exc:
                raise _translate_port_read_fault(exc) from exc
            owed = held.volume if executed is None else min(int(executed), held.volume)
            left = owed - credit
            if left <= 0 and executed is not None:
                self._drop_reservation(held, credit, sends)
                continue
            earlier.append((held.pre_volume, credit))
            if left > 0:
                outstanding[held.leg_id] = outstanding.get(held.leg_id, 0) + left
        if not outstanding:
            return legs
        view: list[PositionLeg] = []
        for leg in legs:
            held_volume = outstanding.get(leg.leg_id, 0)
            if held_volume <= 0:
                view.append(leg)
                continue
            volume = quantize(leg.qty)
            if volume <= held_volume:
                continue
            gone = leg.qty * held_volume / volume
            view.append(replace(
                leg, qty=leg.qty - gone,
                venue_qty=None if leg.venue_qty is None else leg.venue_qty - gone,
            ))
        return view

    def _drop_reservation(self, dropped: _CloseSend, credit: int,
                          sends: list[_CloseSend]) -> None:
        """End a close reservation and hand its credited shrink down to the others.

        The shrink credited to the dropped send lies inside the window of
        every other send of the leg it overlaps (:func:`_shrink_overlap`):
        their recorded leg size is lowered by that overlap, on their rows as
        well, so a later view (after a restart too) credits them only with
        what the leg loses beyond it.

        :param dropped: The dropped send.
        :param credit: The broker-grid shrink credited to it; ``0`` when the
            venue's record settled it without any.
        :param sends: Every reserved send of the symbol, as the current view
            reads them.
        """
        dropped.dropped = True
        # The release and the hand-down are one write: a process stop between
        # them would let a restarted view credit the same shrink again.
        with self._close_journal.transaction():
            release_close_leg_send(self._close_journal, dropped.row_coid, dropped.attempt_coid)
            if credit <= 0:
                return
            for other in sends:
                if other.dropped or other is dropped or other.leg_id != dropped.leg_id:
                    continue
                overlap = _shrink_overlap(credit, dropped.pre_volume, other.pre_volume)
                if overlap > 0:
                    other.pre_volume = max(0, other.pre_volume - overlap)
                    lower_close_leg_send_pre_volume(
                        self._close_journal, other.row_coid, other.attempt_coid,
                        other.pre_volume)

    @staticmethod
    async def _widen_partial_closes_to_whole_legs(
            closes: tuple[LegClose, ...], legs: list[PositionLeg], *,
            symbol: str, port: 'PositionPort', intent_key: str,
    ) -> tuple[LegClose, ...]:
        """Turn every partial leg slice into a whole-leg close on a port that
        cannot reduce a leg.

        A venue without a per-leg partial reduce (Capital.com's
        ``DELETE /positions/{dealId}`` is full-row only) declares
        ``supports_partial_leg_close = False`` on its port; an absent
        attribute means supported. A close is a risk-reducing request, so
        where it cannot be met exactly it is met in full: the leg it would
        slice is closed whole, which leaves the position smaller than the
        script asked for, never larger. The widening is reported at WARNING
        with the asked and the closed quantity. Volumes are compared on the
        broker grid so float noise in the Pine-unit plan cannot fake a
        partial.

        A leg is only closed whole where that stays a reduction of the net
        position. On a mixed book (an opposing leg is open) the whole leg can
        be more than the net exposure of its side, and closing it would flip
        the net position to the other side; on a leg whose venue position also
        holds exposure the run does not own (:attr:`~PositionLeg.venue_qty`)
        the whole venue position is more than the run's share. Such a slice
        is dropped from the plan, reported at WARNING; when nothing is left
        the close is declined with
        :class:`~pynecore.core.broker.exceptions.OrderSkippedByPlugin`, so it
        is re-evaluated on the next bar without halting.

        :param closes: FIFO close plan.
        :param legs: The symbol's open legs the plan was built from.
        :param symbol: The traded symbol.
        :param port: The position port the plan is dispatched through.
        :param intent_key: The key of the intent the plan belongs to, for the
            skip.
        :return: The plan with every sliced leg closed whole; unchanged on a
            port that reduces a leg partially.
        :raises OrderSkippedByPlugin: When every slice of the plan was dropped.
        """
        if getattr(port, 'supports_partial_leg_close', True) or not closes:
            return closes
        quantize = await _read_quantizer(port, symbol)
        live_by_id = {leg.leg_id: leg for leg in legs}
        close_side = next(
            (live_by_id[close.leg_id].side for close in closes if close.leg_id in live_by_id),
            None,
        )
        same_side = sum(leg.qty for leg in legs if leg.side == close_side)
        opposing = sum(leg.qty for leg in legs if leg.side != close_side)
        # The most the plan may close on its side before the net position
        # crosses to the other side; never below what the plan itself asks.
        cap = max(same_side - opposing, sum(close.qty for close in closes))
        widened: list[LegClose] = []
        total = 0.0
        for close in closes:
            live = live_by_id.get(close.leg_id)
            venue_size = None if live is None else (
                live.venue_qty if live.venue_qty is not None else live.qty)
            if (live is not None and venue_size is not None
                    and quantize(close.qty) < quantize(venue_size)):
                if quantize(venue_size) > quantize(live.qty):
                    _blog_warning(
                        "%s close of %s on leg %s is not sent: this broker only "
                        "closes a whole position on a hedging account, and the "
                        "venue position (%s) also holds exposure this run does "
                        "not own",
                        symbol, close.qty, close.leg_id, venue_size,
                    )
                    continue
                if total + live.qty > cap + _QTY_EPS:
                    _blog_warning(
                        "%s close of %s on leg %s (%s open) is not sent: this "
                        "broker only closes a whole leg on a hedging account, and "
                        "closing the whole leg would flip the net position (an "
                        "opposing leg offsets the rest)",
                        symbol, close.qty, close.leg_id, live.qty,
                    )
                    continue
                _blog_warning(
                    "%s close asked for %s of leg %s (%s open): this broker only "
                    "closes a whole leg on a hedging account, so the whole leg is "
                    "closed. Partial closes need a one-way (netting) account.",
                    symbol, close.qty, close.leg_id, live.qty,
                )
                close = LegClose(leg_id=close.leg_id, qty=live.qty)
            widened.append(close)
            total += close.qty
        if not widened:
            raise OrderSkippedByPlugin(
                f"{symbol} close: no leg of the plan can be closed whole without "
                f"flipping the net position or closing exposure this run does not own",
                intent_key=intent_key, reason='whole_leg_close_overshoots',
                context={'symbol': symbol},
            )
        return tuple(widened)

    async def _fan_out_closes(
            self, closes: tuple[LegClose, ...], legs: Sequence[PositionLeg], *,
            symbol: str, side: str, intent_key: str, pine_id: str,
            parent_coid: str, port: 'PositionPort', skippable: bool,
            progress: list[tuple[str, int]] | None = None,
            residual_coid: str | None = None,
            requested_qty: float | None = None,
    ) -> tuple[list[tuple[str, int]], float]:
        """Snap a FIFO close plan to the broker grid and dispatch it per leg.

        Shared by :meth:`run_close` and :meth:`run_reversal`. Each leg is
        PERSIST-FIRST: its row (coid ``f"{parent_coid}:{leg_id}"``, so a restart
        re-dispatch upserts the SAME row) is written ``pending`` BEFORE the wire
        call and finalised ``dispatched`` only after
        :meth:`PositionPort.close_leg` returns. The rows live in the run's store,
        or in process memory on a run without one. Returns the
        ``(leg_id, broker-grid volume)`` pairs actually sent (empty when the
        whole plan rounded below the grid) and the Pine-unit quantity they
        carry. Each row also records the leg's broker-grid size from ``legs``
        (the snapshot the plan was made from), the measure
        :meth:`_settle_close_row` reads of what the close executed, and the
        broker-grid volume of the whole plan
        (:data:`~pynecore.core.broker.store_helpers.EXTRAS_KEY_CLOSE_FAN_VOLUME`),
        so a fan that stopped before its last leg is not read as complete,
        and with ``requested_qty`` (a :meth:`run_close`) the broker-grid
        volume the close was asked for
        (:data:`~pynecore.core.broker.store_helpers.EXTRAS_KEY_CLOSE_REQUESTED_VOLUME`),
        before the plan clipped it, dropped a leg it cannot close or widened it
        to whole legs: what the close executed is measured against that.
        A fan that is no netting half first marks the live netting records of
        the symbol contended on its side (:meth:`_mark_netting_contention`).

        A leg whose close id already has a row (a software guard re-firing
        after an unknown outcome, a reversal's residual drain re-planning its
        closes) is settled through :meth:`_settle_close_row` instead: only the
        part the earlier sends provably did not execute is sent, capped by
        this plan's volume for the leg, which the caller sized to what is safe
        to close now. A close an earlier process of the run already settled
        (its row closed there) sends nothing
        (:meth:`_close_settled_by_earlier_instance`).

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

        A leg whose ``close_leg`` raises
        :class:`~pynecore.core.broker.exceptions.ExchangeConnectionError`
        never left the client either: its row is discharged and the fault
        propagates. ``progress``, when given, is the list the sent legs are
        appended to as they go out, so a caller whose fan raised still knows
        whether any close reached the wire.

        ``residual_coid``, when given, names the breadcrumb of the reversal the
        closes belong to: each leg's close row is counted against the
        reversal's combined order (:func:`count_residual_open_close`), a new
        row in the same write as the row itself, an existing one before its
        settlement can send anything.
        """
        self._mark_netting_contention(symbol, side, parent_coid)
        quantize = await _read_quantizer(port, symbol)
        planned = plan_leg_close_volumes(closes, quantize)
        fan_volume = sum(volume for _leg_id, volume in planned)
        fan_payload: dict[str, int] = {EXTRAS_KEY_CLOSE_FAN_VOLUME: fan_volume}
        if requested_qty is not None:
            # What the close was asked for, before the plan clipped it to the
            # net exposure, dropped a leg it cannot close or widened it to
            # whole legs.
            fan_payload[EXTRAS_KEY_CLOSE_REQUESTED_VOLUME] = quantize(requested_qty)
        pre_volumes = {leg.leg_id: quantize(leg.qty) for leg in legs}
        live_by_id = {leg.leg_id: leg for leg in legs}
        pine_by_leg = {close.leg_id: close.qty for close in closes}
        journal = self._close_journal
        dispatched: list[tuple[str, int]] = progress if progress is not None else []
        sent_qty = 0.0
        for leg_id, volume in planned:
            leg_coid = close_leg_client_order_id(parent_coid, leg_id)
            existing = journal.get_order(leg_coid)
            if existing is not None and existing.state == STATE_CLOSE_LEG:
                # A repeat of a close this id already went out under: it may
                # have executed, so only the part it provably did not execute
                # is sent, never the plan again under the same id.
                if existing.closed_ts_ms is not None:
                    journal.reopen_order(leg_coid)
                if residual_coid is not None:
                    count_residual_open_close(
                        journal, residual_coid, close_coid=leg_coid,
                        qty=pine_by_leg.get(leg_id, 0.0))
                verdict, sent = await self._settle_close_row(
                    leg_coid, symbol, live_by_id, quantize, port, cap=volume,
                )
                if verdict == _CLOSE_ROW_REFUSED:
                    raise ExchangeOrderRejectedError(
                        f"{symbol} close of leg {leg_id}: the exchange refused the "
                        f"part of the earlier close {leg_coid} still owed")
                if verdict == _CLOSE_ROW_UNDECIDED:
                    raise OrderDispositionUnknownError(
                        f"{symbol} close of leg {leg_id}: what the earlier close "
                        f"{leg_coid} executed is not established yet; nothing is sent",
                        client_order_id=leg_coid,
                    )
                # The close of this leg is settled: what it executed is the
                # caller's to own, and only the remainder (maybe none) was sent.
                dispatched.append((leg_id, sent))
                sent_qty += pine_by_leg.get(leg_id, 0.0) * sent / volume
                continue
            if existing is None and self._close_settled_by_earlier_instance(leg_coid):
                # An earlier process of the run sent this close and settled it
                # (its row closed there, and a restart moves only live rows):
                # nothing of it is owed, and a send under its id again could
                # close the leg twice.
                dispatched.append((leg_id, 0))
                continue
            if leg_coid in self._close_rows_busy:
                raise OrderDispositionUnknownError(
                    f"{symbol} close of leg {leg_id}: the close {leg_coid} is being "
                    f"sent by another pass",
                    client_order_id=leg_coid,
                )
            # The row and its send's reservation are one write: a pending row
            # without its reservation would be re-sent by a settlement pass
            # although nothing of it went out. A failed write leaves no row
            # and sends nothing: it surfaces as a fault that never left the
            # client, which every caller already retries.
            try:
                with journal.transaction():
                    create_close_leg_row(
                        journal,
                        coid=leg_coid,
                        symbol=symbol,
                        side=side,
                        qty=float(volume),
                        intent_key=intent_key,
                        pine_entry_id=pine_id,
                        parent_close_coid=parent_coid,
                        leg_id=leg_id,
                        leg_volume=volume,
                        leg_pre_volume=pre_volumes.get(leg_id, volume),
                        extra_payload=fan_payload,
                    )
                    self._reserve_close(
                        leg_coid, leg_coid, volume, pre_volumes.get(leg_id, volume))
                    if residual_coid is not None:
                        count_residual_open_close(
                            journal, residual_coid, close_coid=leg_coid,
                            qty=pine_by_leg.get(leg_id, 0.0))
            except sqlite3.Error as exc:
                raise JournalWriteError(f"{symbol} close of leg {leg_id}", exc) from exc
            self._close_rows_busy.add(leg_coid)
            try:
                await port.close_leg(symbol, leg_id, volume, leg_coid)
            except OrderSkippedByPlugin as exc:
                # Nothing reached the wire for this leg: a live pending row
                # would make a restart replay re-close a leg the venue already
                # settled, or double up with the caller's own re-evaluation.
                # The release and the close are one write for the same reason.
                self._discharge_unsent_close(journal, symbol, leg_id, leg_coid)
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
            except ExchangeConnectionError:
                self._discharge_unsent_close(journal, symbol, leg_id, leg_coid)
                raise
            finally:
                self._close_rows_busy.discard(leg_coid)
            # The row stays live while the send is reserved (:meth:`_reserved_view`).
            try:
                finish_close_leg_row(journal, leg_coid)
            except sqlite3.Error as exc:
                # The close went out and its row still holds the reservation:
                # the dispatch parks, and the row's finish is owed, so the
                # running process closes the row once the send is accounted
                # for, not only a restart's settlement.
                self._owe_close_row_finish(leg_coid, [leg_coid])
                raise OrderDispositionUnknownError(
                    f"{symbol} close of leg {leg_id}: the close {leg_coid} was sent but "
                    f"its journal write failed ({exc}); settled from venue data",
                    client_order_id=leg_coid, cause=exc,
                ) from exc
            dispatched.append((leg_id, volume))
            sent_qty += pine_by_leg.get(leg_id, 0.0)
        return dispatched, round(sent_qty, 12)

    def _close_settled_by_earlier_instance(self, leg_coid: str) -> bool:
        """Whether an earlier process of the run settled the leg close ``leg_coid``.

        A restart moves only the live rows into the new process instance, so
        a close-leg row an earlier process closed is not found by this
        instance's lookups; the run's own record of it still is.

        :param leg_coid: The leg close's own id.
        """
        store = self._store_ctx
        if store is None:
            return False
        row = store.get_order_for_run_id(leg_coid)
        return row is not None and row.state == STATE_CLOSE_LEG and row.closed_ts_ms is not None

    def _owe_close_row_finish(self, coid: str, attempts: list[str]) -> None:
        """Owe the finish of a close-leg row whose send went out (storage contract).

        The send returned, so the row's ``dispatched`` mark is a fact; until it
        lands the row stays pending and keeps its reservation, the state a
        crash leaves for the restart settlement. The owed write reads nothing
        before it is queued, so a store that is still down only keeps it
        owed. It lands the mark only while the row still records exactly the
        sends it had when the finish failed: a later settlement pass that
        re-sent part of the close owns the row from then on, and finishes it
        itself.

        :param coid: The close-leg row's client order id.
        :param attempts: The send ids the row records once this send's
            reservation committed.
        """
        store = self._store_ctx
        if store is None:
            return
        journal = self._close_journal

        def finish() -> None:
            with journal.transaction():
                current = journal.get_order(coid)
                if (current is None or current.closed_ts_ms is not None
                        or [str(sent) for sent in (current.extras or {}).get(
                            EXTRAS_KEY_CLOSE_LEG_ATTEMPTS) or ()] != attempts):
                    return
                finish_close_leg_row(journal, coid)

        store.owe_write(f"finish close leg {coid}", finish)

    @staticmethod
    def _discharge_unsent_close(
            journal: OwnershipJournal, symbol: str, leg_id: str, leg_coid: str,
    ) -> None:
        """Release and close the row of a leg close the venue never received.

        :raises OrderDispositionUnknownError: When the write fails. The row then
            keeps its reservation and stays pending, so the settlement resolves
            the close from the venue's evidence, like a send of unknown outcome.
        """
        try:
            with journal.transaction():
                release_close_leg_send(journal, leg_coid, leg_coid)
                journal.close_order(leg_coid)
        except sqlite3.Error as exc:
            raise OrderDispositionUnknownError(
                f"{symbol} close of leg {leg_id}: the close {leg_coid} was not sent but "
                f"its journal write failed ({exc}); settled from venue data",
                client_order_id=leg_coid, cause=exc,
            ) from exc

    # === Exit-bracket replication =========================================

    async def run_exit_bracket(
            self, envelope: 'DispatchEnvelope', port: 'PositionPort', now: float = 0.0,
    ) -> BracketFanResult:
        """Replicate a Pine exit's bracket onto the net-survivor legs it covers.

        Hedging venues carry protective levels per position, so a one-way bracket
        over a multi-leg position is delivered by amending TP/SL/trailing onto
        each leg the exit covers. On a mixed book (a manual hedge or a
        crash-interrupted reversal left opposing legs open) only the legs that
        survive virtual-FIFO netting are candidates
        (:func:`~pynecore.core.broker.emulator.net_survivor_legs`),
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

        The protection of a leg covers the quantity of it that survives the
        netting, never more: where that is less than the leg's venue position
        (an opposing leg partly offsets it, or the venue position also holds
        exposure the run does not own), the leg gets a sized venue bracket or
        the software leg guard instead of a whole-position bracket
        (:meth:`_protect_leg`).

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
        side, exposure = net_survivor_exposure(legs)
        on_side = [leg for leg, _surviving in exposure]
        by_leg = self._exposure_by_leg(legs)
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
            by_leg=by_leg,
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
        unknown: OrderDispositionUnknownError | None = None
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
            # Judged before this attach writes its own row: a leg none of
            # whose rows records a venue bracket of this run holds none.
            venue_bracket_possible = self._leg_venue_bracket_possible(
                intent.symbol, leg.leg_id)
            create_bracket_ownership_row(
                self._ownership,
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
                extra_payload={
                    EXTRAS_KEY_BRACKET_OWN_SOFTWARE:
                        (co_owned or prior_row is not None)
                        and not venue_bracket_possible,
                    # A re-attach keeps the software guard's persisted trail.
                    **{key: value for key, value in (
                        (prior_row.extras or {}) if prior_row is not None else {}).items()
                       if key in (EXTRAS_KEY_BRACKET_OWN_GUARD_TRAIL_EXTREME,
                                  EXTRAS_KEY_BRACKET_OWN_GUARD_TRAIL_OFFSET)},
                },
            )
            try:
                await self._protect_leg(
                    port, leg, by_leg[leg.leg_id][1], intent.side, levels, attach_coid,
                    venue_bracket_possible=venue_bracket_possible,
                )
            except OrderDispositionUnknownError as exc:
                # The leg's write has no known outcome; its row stays active
                # and settle_leg_writes resolves it. The remaining legs are
                # separate venue positions and are still written: stopping
                # here would leave them without a row, so the settlement could
                # confirm the exit with those legs unprotected. The unknown
                # outcome is raised once every leg had its write.
                if unknown is None:
                    unknown = exc
                continue
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
                update_bracket_ownership_state(
                    self._ownership, coid=own_coid,
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
        if unknown is not None:
            raise unknown
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
        owners: list[tuple[str, _LegLevels]] = []
        for row in iter_active_bracket_ownerships(self._ownership):
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

    @staticmethod
    def _exposure_by_leg(legs: list[PositionLeg]) -> dict[str, tuple[PositionLeg, float]]:
        """Every open leg by id, paired with the quantity of it that survives the netting.

        The surviving quantity is the leg's share of the net one-way position
        (:func:`~pynecore.core.broker.emulator.net_survivor_exposure`), ``0.0``
        for a leg on the minority side or one the opposing legs offset whole.
        """
        by_leg = {leg.leg_id: (leg, 0.0) for leg in legs}
        for leg, surviving in net_survivor_exposure(legs)[1]:
            by_leg[leg.leg_id] = (leg, surviving)
        return by_leg

    async def _protect_leg(
            self, port: 'PositionPort', leg: PositionLeg, surviving: float, side: str,
            levels: _LegLevels, coid: str, *, reassert: bool = False,
            venue_bracket_possible: bool | None = None,
    ) -> None:
        """Make ``levels`` the whole protection of ``leg``, over its surviving quantity only.

        The protected quantity is ``surviving`` (the part of the leg the
        opposing legs do not offset), capped at the leg's own ``qty``. A venue
        bracket is a position attribute and fires on the whole venue position
        (:attr:`~PositionLeg.venue_qty`, else ``qty``), so three cases follow:

        * The protected quantity IS the whole venue position: the bracket is
          written with :meth:`_amend_leg`.
        * It is less, and the port declares ``sized_leg_bracket`` ``NATIVE``:
          the bracket is written limited to the protected quantity.
        * It is less, and the venue cannot size a bracket: a whole-position
          bracket would close exposure that is offset by an opposing leg (and
          flip the net position) or that the run does not own. The leg gets no
          venue bracket; a venue bracket this run may have written on it
          earlier is cleared first, then the leg is put under the software leg
          guard (:meth:`software_guard_triggers`), which closes exactly the
          protected quantity when the price reaches a level. That protection
          lives only while the bot runs, reported once per leg at WARNING.

        The choice is recorded per leg (:meth:`_protection_mode`) and taken
        again by :meth:`settle_leg_writes` whenever the leg's share of its
        venue position changes between syncs, without any bracket operation:
        a leg that starts sharing its venue position loses its whole-position
        bracket to the guard, and one that stops sharing it gets the venue
        bracket back. Neither switch leaves the run's exposure unprotected:
        the guard is armed before the venue bracket is cleared, and dropped
        only once the venue bracket that replaces it was written.

        A leg with nothing surviving has nothing to protect: a venue bracket
        of this run is cleared and no guard is armed.

        The guard closes a part of the leg, so on a port that cannot
        (``supports_partial_leg_close`` ``False``) a leg whose protected
        quantity is less than its venue position (partly offset, or sharing
        the venue position with exposure the run does not own) cannot be
        protected at all: it carries neither a venue bracket nor a guard,
        reported once per leg at ERROR, until the protected quantity is the
        whole venue position again.

        Whether the leg may hold a venue bracket of this run is read from its
        ownership rows (:meth:`_leg_venue_bracket_possible`) unless the
        confirmed record of this process says more; a caller that has just
        written a row of its own passes the verdict it took before
        (``venue_bracket_possible``). The rows' software flag is set only once
        the clear is confirmed and dropped before a venue bracket is written.

        Raises whatever :meth:`_amend_leg` raises.

        :param port: The venue's position port.
        :param leg: The leg, as read in this pass.
        :param surviving: The quantity of the leg that survives the netting.
        :param side: The exit side of the leg's owners.
        :param levels: The merged levels of the leg's active owners.
        :param coid: The correlation id of the write.
        :param reassert: Send a venue bracket even when it equals the
            confirmed one (the restart replay).
        :param venue_bracket_possible: The caller's own verdict, see above.
        """
        key = (leg.symbol, leg.leg_id)
        mode, protected = self._protection_mode(port, leg, surviving)
        previous = self._protection_modes.get(key)
        if levels == _LegLevels():
            self._protection_modes.pop(key, None)
        else:
            self._protection_modes[key] = (mode, protected)
        try:
            await self._write_leg_protection(
                port, leg, side, levels, coid, mode, protected,
                reassert=reassert, venue_bracket_possible=venue_bracket_possible,
            )
        except ExchangeOrderRejectedError:
            # A verdict: the refusal goes to the exit's own recovery, which
            # re-drives the leg; the share watcher must not repeat it.
            raise
        except BaseException:
            # The write did not complete (never sent, throttled, closed
            # market, or of unknown outcome). A failed switch between modes
            # may have left both protections in place (the guard armed, the
            # venue bracket not yet cleared, or the reverse): the leg is marked
            # pending, which differs from every mode, so the next settle pass
            # re-drives it to whatever its share asks for then, even when that
            # is its earlier mode again. Otherwise it keeps its earlier decision.
            if previous is None:
                self._protection_modes.pop(key, None)
            elif self._protection_changed(previous, (mode, protected)):
                self._protection_modes[key] = (_PROTECTION_PENDING, protected)
            else:
                self._protection_modes[key] = previous
            raise

    async def _write_leg_protection(
            self, port: 'PositionPort', leg: PositionLeg, side: str, levels: _LegLevels,
            coid: str, mode: str, protected: float, *, reassert: bool,
            venue_bracket_possible: bool | None,
    ) -> None:
        """Send the writes of :meth:`_protect_leg` for the decided ``mode`` over ``protected``."""
        key = (leg.symbol, leg.leg_id)
        venue_qty = leg.venue_qty if leg.venue_qty is not None else leg.qty
        if mode in ('whole', 'sized'):
            whole = mode == 'whole'
            self._flag_leg_rows_software(leg.symbol, leg.leg_id, False)
            # A guard armed while the leg was shared keeps protecting it until
            # the venue bracket is in place: dropped only once the write
            # returned, never while its outcome is unknown or refused.
            await self._amend_leg(
                port, leg.symbol, leg.leg_id, side, levels, coid,
                qty=venue_qty if whole else protected,
                sized_qty=None if whole else protected, reassert=reassert,
            )
            self._software_guards.pop(key, None)
            return
        if mode == 'software' and levels != _LegLevels():
            # Armed BEFORE the venue bracket is cleared, so the run's own
            # exposure is never left without either protection, even when the
            # clear below ends unknown or is refused.
            self._arm_software_guard(key, leg, side, levels, protected, venue_qty)
        else:
            self._software_guards.pop(key, None)
        confirmed = self._venue_levels.get(key)
        if confirmed is not None:
            venue_bracket_possible = confirmed[0] != _LegLevels()
        elif venue_bracket_possible is None:
            venue_bracket_possible = self._leg_venue_bracket_possible(leg.symbol, leg.leg_id)
        if venue_bracket_possible:
            await self._amend_leg(
                port, leg.symbol, leg.leg_id, side, _LegLevels(), coid,
                qty=venue_qty, sized_qty=None,
            )
        if protected <= _QTY_EPS or levels == _LegLevels():
            return
        self._flag_leg_rows_software(leg.symbol, leg.leg_id, True)
        if mode == 'unprotectable':
            # The guard's close would be a part of the venue position, which
            # this venue cannot send either: it would close the whole venue
            # position, offset or unowned exposure included.
            if key not in self._software_guard_reported:
                self._software_guard_reported.add(key)
                _blog_error(
                    "one-way bracket: leg %s of %s cannot be protected — %s of "
                    "its venue position of %s is this run's net exposure (%s), "
                    "and the venue can neither limit a bracket to that quantity "
                    "nor close a part of a position. A bracket or close of the "
                    "whole position would close more than that, so the leg "
                    "carries none (wanted: tp=%s sl=%s trail=%s)",
                    leg.leg_id, leg.symbol, protected, venue_qty,
                    ("an opposing leg offsets the rest" if leg.venue_qty is None
                     else "the rest is not this run's exposure"),
                    levels.tp_price, levels.sl_price, levels.trail_offset,
                )

    def _protection_mode(
            self, port: 'PositionPort', leg: PositionLeg, surviving: float,
    ) -> tuple[str, float]:
        """How :meth:`_protect_leg` protects ``leg``, and over which quantity.

        :return: ``(mode, protected)``: ``"whole"`` (a whole-position venue
            bracket), ``"sized"`` (a venue bracket limited to ``protected``),
            ``"software"`` (the software leg guard), ``"unprotectable"`` (no
            protection is possible, see :meth:`_protect_leg`) or ``"none"``
            (nothing survives the netting).
        """
        venue_qty = leg.venue_qty if leg.venue_qty is not None else leg.qty
        protected = min(surviving, leg.qty)
        if protected >= venue_qty - _QTY_EPS:
            return 'whole', protected
        if protected <= _QTY_EPS:
            return 'none', protected
        if self._sized_leg_bracket is CapabilityLevel.NATIVE:
            return 'sized', protected
        if not getattr(port, 'supports_partial_leg_close', True):
            return 'unprotectable', protected
        return 'software', protected

    @staticmethod
    def _protection_changed(recorded: tuple[str, float], current: tuple[str, float]) -> bool:
        """Does a leg need another protection than the one recorded on it?

        The mode decides; within ``"sized"`` and ``"software"`` the protected
        quantity does too, since the bracket or guard is limited to it. A
        whole-position bracket follows the venue position by itself.
        """
        if recorded[0] != current[0]:
            return True
        return (current[0] in ('sized', 'software')
                and abs(recorded[1] - current[1]) > _QTY_EPS)

    def _arm_software_guard(
            self, key: tuple[str, str], leg: PositionLeg, side: str, levels: _LegLevels,
            protected: float, venue_qty: float,
    ) -> None:
        """Put ``leg`` under the software leg guard over ``protected``, reported once per leg.

        A guard re-armed with the same side and trailing offset keeps its
        trail. A guard armed afresh (a restart, or a leg the guard left) resumes
        the trail its ownership rows persisted for that offset
        (:meth:`_persisted_guard_trail`), so its stop is never looser than the
        last one written; without one the trail arms at the next price.
        """
        previous = self._software_guards.get(key)
        keeps_trail = (previous is not None and previous.exit_side == side
                       and previous.levels.trail_offset == levels.trail_offset)
        if previous is not None and keeps_trail:
            extreme, unapplied = previous.trail_extreme, previous.unapplied_trail_price
        else:
            extreme = self._persisted_guard_trail(
                leg.symbol, leg.leg_id, side, levels.trail_offset)
            unapplied = None
        self._software_guards[key] = _SoftwareGuard(
            exit_side=side, levels=levels, qty=protected,
            trail_extreme=extreme, unapplied_trail_price=unapplied,
        )
        if key not in self._software_guard_reported:
            self._software_guard_reported.add(key)
            _blog_warning(
                "one-way bracket: leg %s of %s needs protection for %s of the %s "
                "its venue position holds (%s), and a venue bracket on it cannot "
                "be limited to that quantity; the leg carries no venue bracket "
                "and is guarded in software (tp=%s sl=%s trail=%s), which "
                "protects it only while the bot is running",
                leg.leg_id, leg.symbol, protected, venue_qty,
                ("an opposing leg offsets the rest" if leg.venue_qty is None
                 else "the rest is not this run's exposure"),
                levels.tp_price, levels.sl_price, levels.trail_offset,
            )

    def _leg_venue_bracket_possible(self, symbol: str, leg_id: str) -> bool:
        """May ``leg_id`` hold a venue bracket this run wrote, going by its ownership rows?

        ``True`` when a live ownership row of the leg (``active`` or
        ``clearing``) lacks the software flag: its attach wrote a venue
        bracket, or the flag was dropped ahead of one. A leg without any row
        was never written by this run.
        """
        for row in iter_active_bracket_ownerships(self._ownership):
            extras = row.extras or {}
            if (row.symbol == symbol
                    and extras.get(EXTRAS_KEY_BRACKET_OWN_LEG_ID) == leg_id
                    and not extras.get(EXTRAS_KEY_BRACKET_OWN_SOFTWARE)):
                return True
        return False

    def _flag_leg_rows_software(self, symbol: str, leg_id: str, software: bool) -> None:
        """Set the software flag of every live ownership row of ``leg_id`` to ``software``.

        Dropping the flag (the leg gets a venue bracket again) also drops the
        guard's persisted trail: a later guard of the leg starts a trail of its
        own. One write: a failure leaves every row as it was.
        """
        with self._ownership.transaction():
            for row in list(iter_active_bracket_ownerships(self._ownership)):
                extras = row.extras or {}
                if row.symbol != symbol or extras.get(EXTRAS_KEY_BRACKET_OWN_LEG_ID) != leg_id:
                    continue
                trail = not software and (
                    EXTRAS_KEY_BRACKET_OWN_GUARD_TRAIL_EXTREME in extras
                    or EXTRAS_KEY_BRACKET_OWN_GUARD_TRAIL_OFFSET in extras)
                if bool(extras.get(EXTRAS_KEY_BRACKET_OWN_SOFTWARE)) == software and not trail:
                    continue
                patched = {**extras, EXTRAS_KEY_BRACKET_OWN_SOFTWARE: software}
                if trail:
                    patched.pop(EXTRAS_KEY_BRACKET_OWN_GUARD_TRAIL_EXTREME, None)
                    patched.pop(EXTRAS_KEY_BRACKET_OWN_GUARD_TRAIL_OFFSET, None)
                self._ownership.upsert_order(row.client_order_id, extras=patched)

    def _persisted_guard_trail(
            self, symbol: str, leg_id: str, side: str, trail_offset: float | None,
    ) -> float | None:
        """The trail extreme the live ownership rows of ``leg_id`` hold for ``trail_offset``.

        The most favourable one for the protected side (an exit side ``"sell"``
        protects a long, whose extreme is the highest price) among the rows
        that persisted one at that offset; ``None`` when none did, or the
        guard does not trail.
        """
        if trail_offset is None:
            return None
        long_side = side == 'sell'
        extreme: float | None = None
        for row in self._leg_rows(symbol, leg_id):
            extras = row.extras or {}
            stored = extras.get(EXTRAS_KEY_BRACKET_OWN_GUARD_TRAIL_EXTREME)
            if (extras.get(EXTRAS_KEY_BRACKET_OWN_GUARD_TRAIL_OFFSET) != trail_offset
                    or not isinstance(stored, (int, float)) or isinstance(stored, bool)):
                continue
            extreme = _more_favourable(long_side, extreme, float(stored))
        return extreme

    def _persist_guard_trails(self, symbol: str, extremes: Mapping[str, float]) -> None:
        """Write the new trail extreme of each guarded leg onto its live ownership rows.

        PERSIST-FIRST: the guards take the extremes only once this returned.
        One write: a failure leaves every row as it was.

        :param symbol: The legs' symbol.
        :param extremes: The new extreme per leg id.
        :raises sqlite3.Error: When the store write fails.
        """
        with self._ownership.transaction():
            for leg_id, extreme in extremes.items():
                guard = self._software_guards[(symbol, leg_id)]
                for row in self._leg_rows(symbol, leg_id):
                    self._ownership.upsert_order(row.client_order_id, extras={
                        **(row.extras or {}),
                        EXTRAS_KEY_BRACKET_OWN_GUARD_TRAIL_EXTREME: extreme,
                        EXTRAS_KEY_BRACKET_OWN_GUARD_TRAIL_OFFSET: guard.levels.trail_offset,
                    })

    async def _amend_leg(
            self, port: 'PositionPort', symbol: str, leg_id: str, side: str,
            levels: _LegLevels, coid: str, *, qty: float,
            sized_qty: float | None, reassert: bool = False,
    ) -> bool:
        """Send ``levels`` as the whole venue bracket of one leg, unless it already holds them.

        **One write at a time.** A leg whose previous write has no known
        outcome is not written anything else: the earlier request may still
        reach the venue, and arriving after this one it would replace the
        newer protection with the older. Only a write IDENTICAL to the
        unsettled one (same levels, same quantity limit) may be repeated once
        that one has ended: whichever of the two arrives last, the leg holds
        the same protection, so a retry of a timed-out attach or retirement is
        never delayed. Any other write raises
        :class:`~pynecore.core.broker.exceptions.OrderDispositionUnknownError`
        without touching the venue, which every caller already treats as "not
        confirmed, retried later", and the leg is queued for
        :meth:`settle_leg_writes`, which learns the earlier write's fate (or
        waits for its call to end) and then brings the leg to the protection
        its ownership rows ask for. A write's outcome is known when the port call
        returns (confirmed) or raises a verdict that the venue did not apply
        it (:class:`ExchangeOrderRejectedError`,
        :class:`ExchangeRateLimitError`, :class:`OrderSkippedByPlugin`) or
        that it never left the client (:class:`ExchangeConnectionError`). It
        is unknown while the call is still running after the engine's bridge
        stopped waiting for it, and after it ended any other way.

        **The skip rule.** The amend is NOT sent when ``levels`` and
        ``sized_qty`` equal the protection of the last amend THIS process sent
        to the leg and the venue confirmed, and no amend of the leg was
        started since. The record (:attr:`~OneWayEmulator._venue_levels`) is
        dropped before every amend and written back only when it is confirmed,
        so the venue state counts as known only then: an amend that was
        refused or rate-limited, and every leg after a restart, is always
        re-sent. A retirement left ``clearing`` therefore re-sends whenever
        its own amend failed; a leg the retirement never reached still holds
        the confirmed levels and is compared against them. ``reassert`` sends
        unconditionally — the restart replay, whose purpose is to re-assert a
        venue state the process does not trust.

        A skipped amend is no venue call at all: a venue may reject an amend
        that changes nothing, and any re-send resets a trailing stop the venue
        has already moved. The caller's row bookkeeping (persist-first,
        release) proceeds exactly as if it had been sent.

        :param qty: The quantity the bracket protects, as the port takes it.
        :param sized_qty: ``qty`` when the bracket is limited to less than the
            venue position, ``None`` for a whole-position bracket.
        :return: ``True`` when the amend was sent, ``False`` when skipped.
        :raises OrderDispositionUnknownError: When the leg's previous write is
            unsettled (nothing is sent), or this write's outcome is unknown.
        """
        key = (symbol, leg_id)
        protection = (levels, sized_qty)
        unsettled = self._unsettled_writes.get(key)
        if unsettled is not None and (
                not unsettled.finished
                or protection != (unsettled.levels, unsettled.sized_qty)):
            if not unsettled.reported:
                unsettled.reported = True
                _blog_warning(
                    "one-way bracket: the write of leg %s of %s (tp=%s sl=%s "
                    "trail=%s) is held back — the leg's previous write (tp=%s "
                    "sl=%s trail=%s) %s, and it could still replace a newer "
                    "protection; the leg is written again once that is settled",
                    leg_id, symbol, levels.tp_price, levels.sl_price,
                    levels.trail_offset, unsettled.levels.tp_price,
                    unsettled.levels.sl_price, unsettled.levels.trail_offset,
                    ("ended without a verdict from the venue" if unsettled.finished
                     else "is still in flight"),
                )
            self._legs_to_redrive.add(key)
            raise OrderDispositionUnknownError(
                f"protection write on leg {leg_id} of {symbol} held back: the "
                f"leg's previous write has no known outcome yet",
                client_order_id=coid,
            )
        if not reassert and self._venue_levels.get(key) == protection:
            return False
        self._venue_levels.pop(key, None)
        if unsettled is None:
            write = _LegWrite(levels=levels, sized_qty=sized_qty, qty=qty, side=side, coid=coid)
            self._unsettled_writes[key] = write
        else:
            # A repeat of the unsettled write: that one stays unsettled
            # whatever this call returns, and a different write stays held.
            # The repeat is itself in flight until its call ends: arriving
            # late, it could otherwise replace a newer protection written once
            # the original write aged out. A repeat that ends without a verdict
            # restarts the settle window from its own end.
            write = unsettled
            write.resent = True
            write.finished = False
            write.generation += 1
        try:
            await port.amend_bracket(
                symbol, leg_id, side=side,
                tp_price=levels.tp_price, sl_price=levels.sl_price,
                trail_offset=self._trail_price_distance(levels.trail_offset),
                qty=qty, coid=coid,
            )
        except (ExchangeOrderRejectedError, ExchangeRateLimitError,
                ExchangeConnectionError, OrderSkippedByPlugin):
            # A verdict: the venue did not apply the write, or it was never sent.
            if unsettled is None:
                self._drop_write(key, write)
            else:
                write.finished = True
            raise
        except BaseException:
            write.finished = True
            if unsettled is not None:
                write.since = None
            raise
        if unsettled is not None:
            write.finished = True
        if unsettled is None:
            self._drop_write(key, write)
        self._venue_levels[key] = protection
        return True

    def _drop_write(self, key: tuple[str, str], write: _LegWrite) -> None:
        """Forget ``write`` as the unsettled write of its leg, if it still is."""
        if self._unsettled_writes.get(key) is write:
            del self._unsettled_writes[key]

    def _forget_vanished_legs(self, symbol: str, live_ids: set[str]) -> None:
        """Drop every per-leg record of the ``symbol`` legs no longer open.

        The confirmed-protection record, the software guard and a FINISHED
        unsettled write go: a write that lands on a closed position protects
        nothing. A write still in flight stays until its call ends.
        """
        def gone(leg_key: tuple[str, str]) -> bool:
            return leg_key[0] == symbol and leg_key[1] not in live_ids

        for key in [key for key in self._venue_levels if gone(key)]:
            del self._venue_levels[key]
        for key in [key for key in self._software_guards if gone(key)]:
            del self._software_guards[key]
            self._software_guard_reported.discard(key)
        for key in [key for key in self._protection_modes if gone(key)]:
            del self._protection_modes[key]
        for key in [key for key, write in self._unsettled_writes.items()
                    if gone(key) and write.finished]:
            del self._unsettled_writes[key]

    def _own_active_row(self, coid: str) -> 'OrderRow | None':
        """The live, ``active`` ownership row ``coid``, or ``None``."""
        row = self._ownership.get_order(coid)
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
        if prior_row is None:
            update_bracket_ownership_state(
                self._ownership, coid=own_coid,
                new_state=BRACKET_OWN_STATE_RELEASED, close_row=True,
            )
        else:
            self._ownership.upsert_order(own_coid, extras=dict(prior_row.extras or {}))
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
            self, port: 'PositionPort', leg: PositionLeg, surviving: float, side: str,
            coid: str,
    ) -> None:
        """Bring one leg to the merged levels of its remaining active owners.

        The retirement step of every clear path: the retiring rows are already
        ``clearing`` (so they no longer count as owners), and the leg keeps the
        protection of the exits that still own it — cleared (all ``None``)
        only when none does. A merge equal to the leg's confirmed levels (a
        retired exit every one of whose levels was dominated) is not re-sent
        (:meth:`_amend_leg`). The protection covers the leg's surviving
        quantity only (:meth:`_protect_leg`). Raises whatever the amend
        raises; the caller's ``clearing`` rows then stay live for the retry.
        """
        levels = self._merge_owner_levels(
            leg.symbol, leg.leg_id, side,
            self._active_owner_levels(leg.symbol, leg.leg_id),
        )
        await self._protect_leg(port, leg, surviving, side, levels, coid)

    async def _clear_dropped_survivor_legs(
            self, envelope: 'DispatchEnvelope', intent: ExitIntent, *,
            targets: set[str], port: 'PositionPort',
            by_leg: dict[str, tuple[PositionLeg, float]],
    ) -> None:
        """Retire + release this exit's owned legs that fell out of its covered set.

        On a re-attach the covered set can shrink (an opposing leg now virtually
        FIFO-consumes a leg this exit protected before, or the leg's attribution
        now assigns it to another entry). Each such leg is still physically open
        and still carries this exit's levels, so its ownership row is marked
        ``clearing`` PERSIST-FIRST, the leg is amended to the merged levels of
        its remaining owners (cleared only when none is left) under a clear coid
        distinct from the attach coid, and the row is released — mirroring
        :meth:`run_exit_bracket_clear` for the implicit drop. A leg that is no
        longer open (absent from ``by_leg``, the fan's own snapshot) carries no
        bracket: its rows are released without a venue call.

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
        clear_coid = envelope.client_order_id(KIND_CANCEL)
        dropped: dict[str, list['OrderRow']] = {}
        for row in iter_active_bracket_ownerships(
                self._ownership, from_entry=intent.from_entry):
            if (row.intent_key or '') != intent.intent_key:
                continue
            leg_id: str | None = (row.extras or {}).get(EXTRAS_KEY_BRACKET_OWN_LEG_ID)
            if leg_id is None or leg_id in targets:
                continue
            dropped.setdefault(leg_id, []).append(row)
        with self._ownership.transaction():
            for leg_rows in dropped.values():
                for row in leg_rows:
                    update_bracket_ownership_state(
                        self._ownership, coid=row.client_order_id,
                        new_state=BRACKET_OWN_STATE_CLEARING,
                        extras_patch={EXTRAS_KEY_BRACKET_OWN_CLEAR_COID: clear_coid},
                    )
        for leg_id, leg_rows in dropped.items():
            try:
                if leg_id in by_leg:
                    leg, surviving = by_leg[leg_id]
                    await self._reassert_remaining_owners(
                        port, leg, surviving, leg_rows[0].side, clear_coid,
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
            self._release_retired_rows(leg_rows)

    def _release_retired_rows(self, rows: list['OrderRow']) -> list['OrderRow']:
        """Release the rows a finished retirement retired, unless they were taken back.

        A retirement's venue call can outlast the engine's wait for it, and
        the script can place the same exit again meanwhile: the attach rewrites
        the exit's row for the leg (the row key is stable per exit and leg) as
        ``active``. The retirement, finishing late, must not release that row —
        it now records protection the exit asks for. Only a row still
        ``clearing`` is released.

        The rows are released in one write: a failure releases none of them,
        so a caller never loses track of a row released without its report.

        :param rows: The rows the retirement marked ``clearing`` before its call.
        :return: The rows released.
        """
        released: list['OrderRow'] = []
        with self._ownership.transaction():
            for row in rows:
                current = self._ownership.get_order(row.client_order_id)
                if (current is None or current.closed_ts_ms is not None
                        or (current.extras or {}).get(EXTRAS_KEY_BRACKET_OWN_STATE)
                        != BRACKET_OWN_STATE_CLEARING):
                    continue
                update_bracket_ownership_state(
                    self._ownership, coid=row.client_order_id,
                    new_state=BRACKET_OWN_STATE_RELEASED, close_row=True,
                )
                released.append(row)
        return released

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
        rows = [
            row
            for row in iter_active_bracket_ownerships(self._ownership)
            if (row.extras or {}).get(EXTRAS_KEY_BRACKET_OWN_STATE)
            == BRACKET_OWN_STATE_CLEARING
        ]
        if not rows:
            return set()
        legs = await _read_legs(port, symbol)
        live = self._exposure_by_leg(legs)
        self._forget_vanished_legs(symbol, set(live))
        by_leg: dict[str | None, list['OrderRow']] = {}
        for row in rows:
            by_leg.setdefault(
                (row.extras or {}).get(EXTRAS_KEY_BRACKET_OWN_LEG_ID), [],
            ).append(row)
        released: set[str] = set()
        for leg_id, leg_rows in by_leg.items():
            if leg_id is not None and leg_id in live:
                first = leg_rows[0]
                leg, surviving = live[leg_id]
                try:
                    await self._reassert_remaining_owners(
                        port, leg, surviving, first.side,
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
            try:
                retired = self._release_retired_rows(leg_rows)
            except sqlite3.Error as exc:
                if not released:
                    raise
                # Like a dropped link: the keys already released must reach
                # the engine, or their retirement is never finished. These
                # rows stay ``clearing`` for the next sync.
                _blog_warning(
                    "one-way bracket: releasing the retired rows of leg %s of %s "
                    "failed (%s); retried on the next sync", leg_id, symbol, exc,
                )
                break
            for row in retired:
                if row.intent_key is not None:
                    released.add(row.intent_key)
        # Only report a key whose every ownership row is now gone: a key still
        # carrying an ``active`` row (a live exit's survivor legs) must keep its
        # envelope, so the engine must not retire it.
        if not released:
            return released
        still_owned = {
            row.intent_key
            for row in iter_active_bracket_ownerships(self._ownership)
            if row.intent_key is not None
        }
        return released - still_owned

    def needs_settling(self) -> bool:
        """Is there a leg :meth:`settle_leg_writes` has work on?"""
        return bool(self._unsettled_writes) or bool(self._legs_to_redrive)

    def watches_leg_shares(self) -> bool:
        """Does :meth:`settle_leg_writes` have a protected leg whose share it must re-check?

        ``True`` while any leg carries a protection this process decided on
        (:meth:`_protection_mode`): the engine then runs the pass every sync,
        so a leg that starts or stops sharing its venue position, or that an
        opposing leg starts, stops or changes offsetting (an opposite fill, a
        leg the run did not open, a partial close of either leg), is re-driven
        in the first sync whose read shows the change.
        """
        return bool(self._protection_modes)

    async def settle_leg_writes(
            self, symbol: str, port: 'PositionPort', now: float,
    ) -> LegSettleResult:
        """Learn the fate of every unsettled protection write and re-drive its leg.

        The engine calls it once per sync while :meth:`needs_settling`. A
        write still running (the bridge stopped waiting for its call) is left
        alone: its own completion settles it or marks it finished, and a leg
        a write was held back on is re-driven once that happened. For a
        finished write of unknown outcome the venue is asked whether the leg
        holds exactly that protection (:meth:`PositionPort.bracket_matches`):

        * It does, and no identical write was repeated meanwhile: the write
          landed. It is recorded as the leg's confirmed protection.
        * :data:`UNSETTLED_WRITE_WINDOW_S` has passed since the first settle
          attempt: the write no longer counts as pending. Where the leg does
          not show it, it counts as lost (WARNING).
        * Otherwise the write may yet land (or, after a repeat, the leg
          showing the protection does not tell which of the two landed), so
          the leg stays held and is asked again on the next sync.
        * The port cannot tell (``None``: the venue does not report what
          would tell the write from the leg's earlier protection, such as a
          trailing distance). The write is neither confirmed nor counted as
          lost. Inside the window the leg stays held and nothing is repeated.
          Once the window has passed, the write is no longer pending and the
          leg is written once more with the merged levels of its owners —
          the same write when they did not change — and that re-sent write's
          own outcome settles the leg: a confirmed return records it, an
          unknown one holds the leg again. An identical repeat the venue
          already confirmed inside the window counts as that re-sent write.
          A re-sent trailing write carries the stop and distance the owners
          ask for, as every amend does: a trailing stop the venue has moved
          since the first write is written back to the level the write
          carries on a venue that takes the stop from the amend. How each
          venue treats the trail of a repeated amend is not measured here;
          that write-back is the price of an idempotent re-assertion over
          a confirmation the venue cannot give.

        A settled leg is then brought to the merged levels of its active
        owners through :meth:`_protect_leg` — the protection its ownership
        rows ask for now, which a held-back attach, modify or retirement may
        have changed since the write went out. A leg still held is re-driven
        too: when its rows ask for exactly the unsettled write, that write is
        repeated at once (:meth:`_amend_leg` allows the identical repeat), so
        a lost attach does not leave the leg unprotected for the window; a
        different protection stays held. A leg with a ``clearing`` row
        is left to :meth:`drain_clearing_rows`, which the engine runs next.
        That re-drive can itself end unknown (the leg is held again), be
        declined by a closed market or a throttle (retried on the next sync),
        or be refused definitively: the refusal is returned to the engine,
        which re-arms the exits through its regular dispatch path, where a
        refused exit has its full recovery.

        A read fault (:class:`ExchangeConnectionError`) propagates; nothing is
        lost, the next sync repeats the pass.

        One pass per symbol runs at a time: the engine's bridge can stop
        waiting for a slow pass while it keeps running, and a second pass would
        judge the same writes from venue reads the first one already acted on.
        A pass that finds one running returns an empty result; the next sync
        runs it again.

        :param symbol: The symbol to settle.
        :param port: The venue's position port.
        :param now: The caller's monotonic seconds.
        :return: The exits the pass confirmed and the ones the venue refused.
        """
        if symbol in self._settling_symbols:
            return LegSettleResult()
        self._settling_symbols.add(symbol)
        try:
            return await self._settle_leg_writes(symbol, port, now)
        finally:
            self._settling_symbols.discard(symbol)

    async def _settle_leg_writes(
            self, symbol: str, port: 'PositionPort', now: float,
    ) -> LegSettleResult:
        """The body of :meth:`settle_leg_writes`, run by one pass per symbol at a time."""
        finished = [
            (key, write) for key, write in self._unsettled_writes.items()
            if key[0] == symbol and write.finished
        ]
        redrive = {
            key for key in self._legs_to_redrive
            if key[0] == symbol and key not in self._unsettled_writes
        }
        protected_legs = any(key[0] == symbol and key not in self._unsettled_writes
                             for key in self._protection_modes)
        if not finished and not redrive and not protected_legs:
            return LegSettleResult()
        live = self._exposure_by_leg(await _read_legs(port, symbol))
        self._forget_vanished_legs(symbol, set(live))
        # A leg whose protection was decided on another share of its venue
        # position (it started or stopped sharing it with exposure the run
        # does not own, or an opposing leg now offsets more or less of it) is
        # re-driven: its venue bracket could otherwise close exposure that is
        # not the run's own, or a software guard would stand in for a bracket
        # the venue can carry again. Such a re-drive confirms no parked exit.
        reshaped: set[tuple[str, str]] = set()
        for key, recorded in list(self._protection_modes.items()):
            if key[0] != symbol or key[1] not in live or key in self._unsettled_writes:
                continue
            if not self._leg_rows(symbol, key[1]):
                # No exit owns the leg any more (a refused sole-owner attach
                # released its row): there is nothing to keep deciding.
                del self._protection_modes[key]
                continue
            leg, surviving = live[key[1]]
            if self._protection_changed(recorded, self._protection_mode(port, leg, surviving)):
                reshaped.add(key)
        reshaped -= redrive
        redrive |= reshaped
        for key, write in finished:
            if self._unsettled_writes.get(key) is not write:
                continue
            if write.since is None:
                write.since = now
            since = write.since
            generation = write.generation
            landed = await _read_bracket_match(
                port, symbol, key[1], write,
                self._trail_price_distance(write.levels.trail_offset),
            )
            if (self._unsettled_writes.get(key) is not write
                    or write.generation != generation or not write.finished
                    or write.since != since):
                # The write's own completion settled it while the venue was
                # read, a newer write holds the leg, or an identical repeat
                # started meanwhile (it may still be in flight, and it reset
                # the settle window): the read judges none of them.
                continue
            expired = now - since >= UNSETTLED_WRITE_WINDOW_S
            if landed is None:
                if not expired:
                    # Neither confirmed nor lost, and the leg is protected by
                    # something: no repeat inside the window, the one
                    # re-assertion comes after it.
                    continue
                del self._unsettled_writes[key]
                # An identical repeat the venue confirmed already re-asserted
                # the levels: the re-drive's skip rule then sends nothing.
                if self._venue_levels.get(key) != (write.levels, write.sized_qty):
                    _blog_warning(
                        "one-way bracket: the write of leg %s of %s (tp=%s sl=%s "
                        "trail=%s) got no verdict and the venue does not show "
                        "whether the leg holds it; after %.0fs the leg is "
                        "written once more to re-assert its levels",
                        key[1], symbol, write.levels.tp_price, write.levels.sl_price,
                        write.levels.trail_offset, now - since,
                    )
                self._legs_to_redrive.add(key)
                redrive.add(key)
                continue
            if landed and (expired or not write.resent):
                self._venue_levels[key] = (write.levels, write.sized_qty)
            elif not expired:
                # Still held; the re-drive below may repeat the identical
                # write (a lost attach is not left unprotected for the window).
                redrive.add(key)
                continue
            elif landed:
                pass
            else:
                _blog_warning(
                    "one-way bracket: the write of leg %s of %s (tp=%s sl=%s "
                    "trail=%s) got no verdict and the leg has not shown it for "
                    "%.0fs; it counts as lost and the leg is written again",
                    key[1], symbol, write.levels.tp_price, write.levels.sl_price,
                    write.levels.trail_offset, now - since,
                )
            del self._unsettled_writes[key]
            self._legs_to_redrive.add(key)
            redrive.add(key)
        redriven: set[str] = set()
        refused: list[tuple[str, ExchangeOrderRejectedError]] = []
        for key in sorted(redrive):
            leg_id = key[1]
            held = self._unsettled_writes.get(key)
            if held is not None and not held.finished:
                continue
            if leg_id not in live:
                self._legs_to_redrive.discard(key)
                continue
            rows = self._leg_rows(symbol, leg_id)
            active = [
                row for row in rows
                if (row.extras or {}).get(EXTRAS_KEY_BRACKET_OWN_STATE)
                == BRACKET_OWN_STATE_ACTIVE
            ]
            if len(active) != len(rows) or not active:
                # A retirement is pending on the leg (the drain re-drives it),
                # or no exit owns it any more and nothing is asked of it.
                self._legs_to_redrive.discard(key)
                continue
            leg, surviving = live[leg_id]
            newest = max(active, key=lambda owner: (owner.updated_ts_ms, owner.client_order_id))
            try:
                await self._reassert_remaining_owners(
                    port, leg, surviving, newest.side,
                    (newest.extras or {}).get(EXTRAS_KEY_BRACKET_OWN_ATTACH_COID)
                    or newest.client_order_id,
                )
            except OrderDispositionUnknownError:
                continue
            except (MarketClosedError, ExchangeRateLimitError) as exc:
                self._legs_to_redrive.add(key)
                _blog_warning(
                    "one-way bracket: re-driving leg %s of %s after its "
                    "unsettled write was declined (%s: %s) — retried on the "
                    "next sync", leg_id, symbol, type(exc).__name__, exc,
                )
                if isinstance(exc, ExchangeRateLimitError):
                    break
                continue
            except ExchangeOrderRejectedError as exc:
                self._legs_to_redrive.discard(key)
                refused.extend(
                    (row.intent_key, exc) for row in active if row.intent_key is not None)
                continue
            self._legs_to_redrive.discard(key)
            if key not in reshaped:
                redriven.update(row.intent_key for row in active if row.intent_key is not None)
        if not redriven:
            return LegSettleResult(confirmed=frozenset(redriven), refused=tuple(refused))
        unconfirmed: set[str] = set()
        for row in iter_active_bracket_ownerships(self._ownership):
            leg_id = (row.extras or {}).get(EXTRAS_KEY_BRACKET_OWN_LEG_ID)
            if row.symbol != symbol or row.intent_key is None or leg_id is None:
                continue
            key = (symbol, leg_id)
            if (key in self._unsettled_writes or key in self._legs_to_redrive
                    or (key not in self._venue_levels and key not in self._software_guards)):
                unconfirmed.add(row.intent_key)
        return LegSettleResult(
            confirmed=frozenset(redriven - unconfirmed), refused=tuple(refused))

    def _leg_rows(self, symbol: str, leg_id: str) -> list['OrderRow']:
        """Every live ownership row (``active`` or ``clearing``) of one leg."""
        return [
            row for row in iter_active_bracket_ownerships(self._ownership)
            if row.symbol == symbol
            and (row.extras or {}).get(EXTRAS_KEY_BRACKET_OWN_LEG_ID) == leg_id
        ]

    # === Software leg guard ===============================================

    def has_software_guards(self) -> bool:
        """Is any leg under the software leg guard?"""
        return bool(self._software_guards)

    def software_guard_triggers(
            self, symbol: str, last_price: float,
    ) -> list[SoftwareLegTrigger]:
        """Evaluate the software leg guards of ``symbol`` against ``last_price``.

        A guard stands in for the venue bracket of a leg whose protection must
        close less than the venue position (:meth:`_protect_leg`). A long's
        guard (exit side ``"sell"``) triggers when the price is at or above
        its take-profit, at or below its stop-loss, or at or below its
        trailing stop; a short's is the mirror. The trailing stop arms at the
        first price it sees, ``trail_offset`` away from it (the per-leg
        emulation has no deferred activation), and from then on only moves in
        the position's favour.

        The trail is persisted. Its extreme (the most favourable price it
        trails from; the stop lies ``trail_offset`` behind it) is written onto
        the leg's live ownership rows before the guard takes it, one write for
        every guard the price moved (:meth:`_persist_guard_trails`), so a
        restart resumes the trail instead of arming it again at the current,
        possibly less favourable price. A failed write raises the storage
        error; each guard keeps the price it was moved to in memory
        (:attr:`~_SoftwareGuard.unapplied_trail_price`) and trails from it as
        if it had landed, so the next evaluation writes it and the observed
        extreme is never lost. A pass on which a guard triggers writes
        nothing: the moved extremes wait in memory the same way, so a store
        failure can never hold back a trigger.

        A triggered guard is marked firing and is not evaluated again until
        the engine reports the fate of the close it dispatched
        (:meth:`software_guard_settled`).

        :param symbol: The symbol the price belongs to.
        :param last_price: The last observed price.
        :return: One trigger per guard whose level the price reached.
        :raises sqlite3.Error: When the trail write fails; no guard triggered.
        """
        hits: list[tuple[str, _SoftwareGuard, tuple[str, float]]] = []
        moves: dict[str, float] = {}
        for (guard_symbol, leg_id), guard in self._software_guards.items():
            if guard_symbol != symbol or guard.firing:
                continue
            long_side = guard.exit_side == 'sell'
            levels = guard.levels
            hit: tuple[str, float] | None = None
            distance = self._trail_price_distance(levels.trail_offset)
            if distance is not None:
                extreme = _more_favourable(
                    long_side, guard.trail_extreme, guard.unapplied_trail_price)
                if extreme is None:
                    moves[leg_id] = last_price
                else:
                    stop = extreme - distance if long_side else extreme + distance
                    if last_price <= stop if long_side else last_price >= stop:
                        hit = ('trailing_stop', stop)
                    else:
                        extreme = _more_favourable(long_side, extreme, last_price)
                    if extreme != guard.trail_extreme:
                        moves[leg_id] = extreme
            if levels.sl_price is not None and (
                    last_price <= levels.sl_price if long_side
                    else last_price >= levels.sl_price):
                hit = ('stop_loss', levels.sl_price)
            if levels.tp_price is not None and (
                    last_price >= levels.tp_price if long_side
                    else last_price <= levels.tp_price):
                hit = ('take_profit', levels.tp_price)
            if hit is not None:
                hits.append((leg_id, guard, hit))
        if moves and not hits:
            try:
                self._persist_guard_trails(symbol, moves)
            except BaseException:
                for leg_id, extreme in moves.items():
                    self._software_guards[(symbol, leg_id)].unapplied_trail_price = extreme
                raise
            for leg_id, extreme in moves.items():
                guard = self._software_guards[(symbol, leg_id)]
                guard.trail_extreme = extreme
                guard.unapplied_trail_price = None
        else:
            for leg_id, extreme in moves.items():
                self._software_guards[(symbol, leg_id)].unapplied_trail_price = extreme
        triggers: list[SoftwareLegTrigger] = []
        for leg_id, guard, (reason, level) in hits:
            guard.firing = True
            owners = [
                row for row in self._leg_rows(symbol, leg_id)
                if (row.extras or {}).get(EXTRAS_KEY_BRACKET_OWN_STATE)
                == BRACKET_OWN_STATE_ACTIVE
            ]
            entries = {row.from_entry for row in owners}
            triggers.append(SoftwareLegTrigger(
                symbol=symbol, leg_id=leg_id, side=guard.exit_side, qty=guard.qty,
                reason=reason, level=level,
                from_entry=next(iter(entries)) if len(entries) == 1 else None,
                exit_ids=tuple(sorted({row.pine_entry_id or '' for row in owners})),
            ))
        return triggers

    def software_guard_settled(self, symbol: str, leg_id: str, *, outcome: str) -> None:
        """Record the fate of the close a triggered guard asked for.

        :param symbol: The leg's symbol.
        :param leg_id: The guarded leg.
        :param outcome: ``'closed'``: the close went on the wire, the guard is
            done and removed. ``'retry'``: nothing went out, the guard is
            armed again and the next price re-evaluates it. ``'unknown'``: the
            close may or may not have reached the venue, so the quantity the
            guard would close is no longer certain: the guard is removed and
            the leg is queued for :meth:`settle_leg_writes`, which re-reads
            the leg and arms a guard over what then survives.
        """
        key = (symbol, leg_id)
        guard = self._software_guards.get(key)
        if guard is None:
            return
        if outcome == 'retry':
            guard.firing = False
            return
        del self._software_guards[key]
        if outcome != 'closed':
            self._legs_to_redrive.add(key)

    def owes_residual_open(self, intent_key: str) -> bool:
        """Whether a market reversal of ``intent_key`` still owes its residual open.

        The reversal is committed to its breadcrumb once a close went out: the
        drain opens the residual under the reversal's own envelope and hands
        its orders to the intent. Until then the entry has no venue order of
        its own, yet it is not free to be amended, replaced or cancelled: a
        second dispatch of the same entry would open next to the residual.

        :param intent_key: The entry's intent key.
        """
        return any(row.intent_key == intent_key and row.qty > 0.0
                   for row in iter_active_residual_opens(self._close_journal))

    async def drain_residual_opens(self, port: 'PositionPort') -> None:
        """Reconcile any live reversal residual-open breadcrumb per sync (in-session).

        The restart counterpart :meth:`_replay_residual_opens` reconciles these
        breadcrumbs once at startup, but the engine latches that replay after the
        first sync (``_one_way_replay_done``). Three paths leave a breadcrumb live
        mid-session: :meth:`run_reversal` whose fresh read after the closes
        still lists an opposing leg (the closes were dispatched, not yet seen
        executed), its residual ``place_leg`` raising
        :class:`OrderDispositionUnknownError` (the open's fate unknown, so the
        breadcrumb is intentionally NOT discharged), and :meth:`_replay_residual_one`
        hitting the same on its own re-dispatch.

        The engine calls this once per sync (alongside
        :meth:`drain_clearing_rows`), reconciling every live breadcrumb of
        the run through the SAME :meth:`_replay_residual_one` path
        :meth:`run_reversal` and the restart replay use: if the residual's open
        already went out (its persist-first entry row landed, or on a run
        without a store its ``place_leg`` was called) the breadcrumb is simply
        discharged (the entry / recovery path owns the open); while an
        opposing leg is still open its close is settled from the venue's
        evidence (only the part provably not executed is re-sent) and the
        breadcrumb waits; once the book shows none, the residual is opened
        under its deterministic entry coid (``KIND_ENTRY``, or
        ``KIND_ENTRY_TRIGGERED`` for a triggered reversal). A still-ambiguous
        open leaves the row live for the next sync rather than halting the
        bot, mirroring :meth:`drain_clearing_rows`.

        Every residual a recovery pass opens is kept for
        :meth:`take_opened_residuals` the moment its ``place_leg`` returns, so
        the engine maps its order ids to the reversal's entry intent even when
        a later row of the same pass raises.
        """
        for row in list(iter_active_residual_opens(self._close_journal)):
            await self._replay_residual_one(row, port)

    def take_opened_residuals(self) -> list[tuple[str, tuple]]:
        """Hand over (and forget) the residuals the recovery passes opened.

        :return: ``(intent_key, opened orders)`` per residual open, in the
            order they were opened.
        """
        opened, self._opened_residuals = self._opened_residuals, []
        return opened

    async def drain_nettings(self, port: 'PositionPort', *, fill_grace_s: float) -> None:
        """Net every fill-time netting record (:data:`~pynecore.core.broker.store_helpers.STATE_NETTING`).

        A resting LIMIT / STOP entry on a hedging account rests as a leg of
        its own (:meth:`run_reversal`). Once it fills against opposing legs,
        the venue holds both sides where a one-way account holds only the
        net; the sync engine then writes a netting record for the fill slice
        before anything is sent. A pass over a record reads the legs and
        nets ``min(record qty, opposing exposure, exposure on the fill's
        side)``: it writes two close-plan breadcrumbs in one transaction with
        the record's update (one closing the opposing legs FIFO, one closing
        the filled entry's legs, its own first, by the same quantity) and
        then sends them through :meth:`_replay_residual_one`, the path the
        residual drain and the restart replay settle every breadcrumb by
        (reserved views, unknown outcomes, re-sends only of what provably did
        not execute). The net position is the same before and after; the
        fills of these closes are expected by the engine as book-neutral up
        to the netted quantity (:data:`~pynecore.core.broker.store_helpers.EXTRAS_KEY_NETTING_EXPECTED`).

        A record is planned again only once its breadcrumbs are discharged,
        so a record whose fill side the read did not show in full yet is
        netted over several passes; a fill side that stays unlisted for
        :data:`BOOK_INCOMPLETE_GRACE_S` ends the record's planning with a
        warning. Once nothing is left to plan or close, the record waits for
        its expected close fills at most ``fill_grace_s`` and is then closed
        (with a warning when part of them never came).

        A port that closes only whole legs (``supports_partial_leg_close =
        False``) gets whole-leg plans (:meth:`_widen_netting_closes`).

        Under the quarantine latch no new pass is planned (the record stays
        live); halves already planned keep going through the residual drain,
        which never lets one half of a pair stand alone longer than needed.

        One pass at a time owns a record (``_netting_rows_busy``), and the
        planning pass owns the symbol's closes from its read to its
        breadcrumb write (``_closing_symbols``).

        :param port: The position port.
        :param fill_grace_s: Seconds a close fill may follow the venue's
            position change (the engine's fill-report grace).
        """
        for row in list(iter_active_nettings(self._close_journal)):
            halves = await self._plan_netting_one(row, port, fill_grace_s)
            for half in halves:
                await self._replay_residual_one(half, port)

    def record_netting(
            self, *, entry_coid: str, fill_key: str, symbol: str, side: str, qty: float,
            intent_key: str | None, entry_id: str | None, fill_grace_s: float,
    ) -> str | None:
        """Write the netting record of one fill slice of a resting entry (PERSIST-FIRST).

        Called by the sync engine before the fill moves the book and before
        anything is sent (:meth:`drain_nettings` nets it). The record's coid
        is the entry's coid and a digest of the fill's id, so the same fill
        maps onto the same record in every process: a fill whose record
        exists already (live or done) writes nothing. The record is looked up
        in every process instance of the run: a restart moves only the live
        rows, and a record an earlier process closed proves the slice already
        had its netting pass (a fill redelivered after restart, its receipt
        never written, must not net the two sides again).

        A close of the symbol that is no netting half and whose fills can
        still come back (:meth:`_ordinary_close_sides_in_flight`) marks the
        new record contended on its side
        (:data:`~pynecore.core.broker.store_helpers.EXTRAS_KEY_NETTING_CONTENDED`),
        exactly like a close fanned out while the record is live
        (:meth:`_mark_netting_contention`): its fill can come back after the
        record's own closes went out on the same side, and one that names no
        close of the run cannot be told from theirs.

        :param entry_coid: The filled entry order's client order id (its venue
            order id when the fill carries none).
        :param fill_key: The fill's id (its idempotency key).
        :param symbol: Exchange-side symbol.
        :param side: The fill's side.
        :param qty: The Pine quantity of the fill the venue must net.
        :param intent_key: The filled entry's diff key, when known.
        :param entry_id: The filled entry's Pine id.
        :param fill_grace_s: Seconds a close fill may follow the venue's
            position change (the engine's fill-report grace).
        :return: The record's coid, or ``None`` when it existed already.
        :raises sqlite3.Error: When the store cannot be written; nothing was
            written then.
        """
        coid = f"{entry_coid}:n{hashlib.sha256(fill_key.encode('utf-8')).hexdigest()[:8]}"
        if self._close_journal.get_order_for_run_id(coid) is not None:
            return None
        create_netting_row(
            self._close_journal, coid=coid, symbol=symbol, side=side, qty=qty,
            intent_key=intent_key, entry_id=entry_id, fill_id=fill_key,
            contended=self._ordinary_close_sides_in_flight(
                symbol, now_ms=time_ns() // 1_000_000, fill_grace_s=fill_grace_s),
        )
        return coid

    def _ordinary_close_sides_in_flight(
            self, symbol: str, *, now_ms: int, fill_grace_s: float,
    ) -> set[str]:
        """The sides of the closes of ``symbol`` that are no netting half and may still fill.

        A close-leg row stays live until its close is accounted for (every
        reserved send dropped), so a live row is a close whose fills can still
        come back. Its row closes once the position read (or the venue's
        record of the send) shows what the close took, and the fill report
        can come later than that read: up to the fill-report grace after the
        venue's position changed, as a netting record's own close fills
        (:meth:`_settle_netting_record`). A row closed less than
        ``fill_grace_s`` ago, in any process instance of the run, is
        therefore still in flight.

        :param symbol: Exchange-side symbol.
        :param now_ms: Epoch milliseconds now.
        :param fill_grace_s: Seconds a close fill may follow the venue's
            position change.
        :raises sqlite3.Error: When the store cannot be read.
        """
        journal = self._close_journal
        rows = list(journal.iter_live_orders())
        if fill_grace_s > 0.0:
            since_ms = now_ms - fill_grace_s * 1000.0
            rows.extend(row for row in journal.iter_orders_in_state_for_run_id(STATE_CLOSE_LEG)
                        if row.closed_ts_ms is not None and row.closed_ts_ms > since_ms)
        sides: set[str] = set()
        for row in rows:
            if row.state != STATE_CLOSE_LEG or row.symbol != symbol or not row.side:
                continue
            parent = str((row.extras or {}).get(EXTRAS_KEY_CLOSE_PARENT_COID) or '')
            if self._netting_record_of_close(parent) is None:
                sides.add(row.side)
        return sides

    def netting_expectations(self) -> list[tuple[str, dict[str, float]]]:
        """The close fills the live netting records still expect, oldest record first.

        :return: ``(record coid, {'buy': qty, 'sell': qty})`` per record with
            a positive expectation on a side.
        :raises sqlite3.Error: When the store cannot be read.
        """
        out: list[tuple[str, dict[str, float]]] = []
        for row in iter_active_nettings(self._close_journal):
            left = netting_expected(self._close_journal, row)
            if any(qty > _QTY_EPS for qty in left.values()):
                out.append((row.client_order_id, left))
        return out

    def absorb_netting_fills(self, absorbed: Mapping[str, Mapping[str, float]]) -> None:
        """Count close fills booked as netting on a run without a store.

        With a store the engine writes the count itself, in the transaction of
        the fills' receipts (:func:`~pynecore.core.broker.store_helpers.absorb_netting_fills`).

        :param absorbed: Per record coid, the quantity booked as netting by side.
        """
        absorb_netting_fills(self._close_journal, absorbed)

    def _netting_record_of_close(self, parent_close_coid: str) -> str | None:
        """The netting record a close fan belongs to, when it is a netting half.

        A netting pass fans its two halves under ``{record}:p{pass}{a|b}``
        (:meth:`_plan_netting_pass`).

        :param parent_close_coid: The fan's own client order id.
        :return: The record's coid, or ``None`` for any other close.
        """
        head, sep, tail = parent_close_coid.rpartition(':p')
        if not sep or len(tail) < 2 or tail[-1] not in 'ab' or not tail[:-1].isdigit():
            return None
        record = self._close_journal.get_order_for_run_id(head)
        return head if record is not None and record.state == STATE_NETTING else None

    def netting_close_owner(
            self, client_order_id: str | None, *, run_tag: str | None = None,
    ) -> tuple[bool, str | None]:
        """Which close of the run a close fill reports, by its client order id.

        A close fill that carries the id :meth:`PositionPort.close_leg`
        received (the leg close's own id or a re-send's attempt id) names the
        close-leg row it belongs to, and with it whether it is one of a
        netting record's closes. The row is looked up whether or not it is
        still live, in any process instance of the run (a restart moves only
        the live rows), so the answer depends neither on how long the fill
        took nor on a restart in between.

        A close fill whose id names any other order of the run is no netting
        close either: a netting close always goes out under its close-leg
        row's id. That is the id of another row of the run (an entry's own
        id, which a plugin reports for a close of the entry's position it
        cannot attribute to a leg close of the run, such as a manual one), or
        a client order id the run minted
        (:func:`~pynecore.core.broker.idempotency.parse_client_order_id`, or
        its wire form under the run's tag on a run without a store, where an
        entry has no row in this journal). A leg-close id never has either
        shape.

        :param client_order_id: The fill's client order id.
        :param run_tag: The run's tag, to recognise its wire-form ids.
        :return: ``(True, record coid)`` for a netting close, ``(True, None)``
            for any other order of the run, ``(False, None)`` when the id
            names nothing the run sent.
        :raises sqlite3.Error: When the store cannot be read.
        """
        if not client_order_id:
            return False, None
        row = self._close_journal.get_order_for_run_id(
            close_leg_row_client_order_id(client_order_id))
        if row is not None and row.state == STATE_CLOSE_LEG:
            parent = str((row.extras or {}).get(EXTRAS_KEY_CLOSE_PARENT_COID) or '')
            return True, self._netting_record_of_close(parent)
        if row is not None or parse_client_order_id(client_order_id) is not None:
            return True, None
        wire = parse_wire_client_order_id(client_order_id) if run_tag else None
        if wire is not None and wire.run_tag == run_tag:
            return True, None
        return False, None

    def netting_expectation_of(self, coid: str) -> dict[str, float]:
        """What of netting record ``coid``'s close fills is not booked as netting yet.

        Read whether or not the record is still live, in any process instance
        of the run: a fill identified as one of its closes
        (:meth:`netting_close_owner`) is the record's however late it comes,
        across a restart too.

        :return: ``{'buy': qty, 'sell': qty}``; empty when the record is gone.
        :raises sqlite3.Error: When the store cannot be read.
        """
        row = self._close_journal.get_order_for_run_id(coid)
        if row is None or row.state != STATE_NETTING:
            return {}
        return netting_expected(self._close_journal, row)

    def netting_contended_sides(self, coid: str) -> set[str]:
        """The close fill sides of netting record ``coid`` that another close shares.

        See :data:`~pynecore.core.broker.store_helpers.EXTRAS_KEY_NETTING_CONTENDED`.

        :raises sqlite3.Error: When the store cannot be read.
        """
        row = self._close_journal.get_order_for_run_id(coid)
        if row is None:
            return set()
        return {str(side) for side in (row.extras or {}).get(EXTRAS_KEY_NETTING_CONTENDED) or ()}

    def _mark_netting_contention(self, symbol: str, side: str, parent_coid: str) -> None:
        """Note on every live netting record of ``symbol`` that a close of ``side`` goes out.

        Called by :meth:`_fan_out_closes` before anything of a close that is
        no netting half is written or sent: its fills share a side with the
        records' own close fills, and a fill that names no close of the run
        could be either.

        :raises sqlite3.Error: When the store cannot be written; nothing was
            sent then.
        """
        if self._netting_record_of_close(parent_coid) is not None:
            return
        journal = self._close_journal
        rows = [row for row in iter_active_nettings(journal) if row.symbol == symbol]
        if not rows:
            return
        with journal.transaction():
            for row in rows:
                extras = dict(row.extras or {})
                sides = sorted({*(extras.get(EXTRAS_KEY_NETTING_CONTENDED) or ()), side})
                if sides != list(extras.get(EXTRAS_KEY_NETTING_CONTENDED) or ()):
                    extras[EXTRAS_KEY_NETTING_CONTENDED] = sides
                    journal.upsert_order(row.client_order_id, extras=extras)

    def netting_in_flight(self) -> bool:
        """Whether a fill-time netting still has closes to plan, send or receive fills of.

        A failed read answers ``True``: the caller then holds its verdict.
        """
        try:
            return any(True for _ in iter_active_nettings(self._close_journal))
        except sqlite3.Error:
            return True

    def _netting_halves(self, coid: str) -> list['OrderRow']:
        """The live close-plan breadcrumbs of netting record ``coid``."""
        return [row for row in iter_active_residual_opens(self._close_journal)
                if (row.extras or {}).get(EXTRAS_KEY_RESIDUAL_OPEN_NETTING) == coid]

    async def _plan_netting_one(
            self, row: 'OrderRow', port: 'PositionPort', fill_grace_s: float,
    ) -> list['OrderRow']:
        """Plan one netting pass of a record, or move the record on in its lifecycle.

        :return: The breadcrumbs the pass wrote, for the caller to send once
            the claims are released; empty when nothing was planned.
        """
        coid = row.client_order_id
        symbol = row.symbol
        if coid in self._netting_rows_busy or symbol in self._closing_symbols:
            return []
        self._netting_rows_busy.add(coid)
        self._closing_symbols.add(symbol)
        try:
            current = self._close_journal.get_order(coid)
            if current is None or current.closed_ts_ms is not None:
                return []
            if self._netting_halves(coid):
                return []
            now_ms = time_ns() // 1_000_000
            if current.qty <= _QTY_EPS:
                self._settle_netting_record(current, now_ms, fill_grace_s)
                return []
            if self._block_exposure_reopens is not None and self._block_exposure_reopens():
                if coid not in self._netting_quarantine_reported:
                    self._netting_quarantine_reported.add(coid)
                    _blog_warning(
                        "%s: netting %r of %s is withheld while the bot is quarantined; "
                        "both sides stay open until the operator resume",
                        symbol, coid, current.qty,
                    )
                return []
            self._netting_quarantine_reported.discard(coid)
            return await self._plan_netting_pass(current, port, now_ms)
        finally:
            self._netting_rows_busy.discard(coid)
            self._closing_symbols.discard(symbol)

    def _settle_netting_record(self, row: 'OrderRow', now_ms: int, fill_grace_s: float) -> None:
        """Close a netting record whose closes are done once its expected fills came.

        :param row: The record, with nothing left to plan and no live half.
        :param now_ms: Epoch milliseconds now.
        :param fill_grace_s: Seconds the expected close fills are waited for.
        """
        journal = self._close_journal
        coid = row.client_order_id
        left = netting_expected(journal, row)
        extras = dict(row.extras or {})
        if any(qty > _QTY_EPS for qty in left.values()):
            settled = extras.get(EXTRAS_KEY_NETTING_SETTLED_MS)
            if settled is None:
                extras[EXTRAS_KEY_NETTING_SETTLED_MS] = now_ms
                journal.upsert_order(coid, extras=extras)
                return
            if now_ms - int(settled) < fill_grace_s * 1000.0:
                return
            _blog_warning(
                "%s: netting %r ends with close fills of %s never reported; a later "
                "close fill that names one of its closes is still the netting's, any "
                "other moves the book as usual",
                row.symbol, coid, {side: qty for side, qty in left.items() if qty > _QTY_EPS},
            )
        with journal.transaction():
            journal.close_order(coid)
            journal.close_order(netting_absorbed_row_coid(coid))

    async def _plan_netting_pass(
            self, row: 'OrderRow', port: 'PositionPort', now_ms: int,
    ) -> list['OrderRow']:
        """Read the legs and write the two halves of one netting pass of ``row``.

        Run while the pass owns the record and the symbol's closes
        (:meth:`_plan_netting_one`).

        :return: The breadcrumbs written; empty when nothing could be netted.
        """
        journal = self._close_journal
        coid = row.client_order_id
        symbol = row.symbol
        fill_side = row.side
        opposing_side = 'sell' if fill_side == 'buy' else 'buy'
        extras = dict(row.extras or {})
        raw_legs = await _read_legs(port, symbol)
        view = await self._reserved_view(port, symbol, raw_legs, own_parent=None)
        opposing = sum(leg.qty for leg in view if leg.side == opposing_side and leg.qty > 0.0)
        own_side = sum(leg.qty for leg in view if leg.side == fill_side and leg.qty > 0.0)
        qty = min(row.qty, opposing, own_side)
        if qty <= _QTY_EPS:
            since = int(extras.get(EXTRAS_KEY_NETTING_SINCE_MS) or now_ms)
            if opposing > _QTY_EPS and now_ms - since < BOOK_INCOMPLETE_GRACE_S * 1000.0:
                # The fill's own leg is not in the read yet.
                return []
            if opposing > _QTY_EPS:
                _blog_warning(
                    "%s: netting %r stops with %s left: the read shows no exposure on "
                    "the fill's side (%s) %.0fs after the fill; the opposing legs stay",
                    symbol, coid, row.qty, fill_side, (now_ms - since) / 1000.0,
                )
            journal.upsert_order(coid, qty=0.0)
            return []
        opposing_closes, _ = select_legs_for_close(qty, view, opposing_side)
        own_closes, _ = select_legs_for_close(qty, view, fill_side, entry_id=row.pine_entry_id)
        quantize = await _read_quantizer(port, symbol)
        if not getattr(port, 'supports_partial_leg_close', True):
            widened = self._widen_netting_closes(
                opposing_closes, own_closes, view, quantize, symbol=symbol, coid=coid,
                netted=qty, opposing=opposing, own_side=own_side,
            )
            if widened is None:
                journal.upsert_order(coid, qty=0.0)
                return []
            opposing_closes, own_closes = widened
        # The fills come back on the broker grid: what each half nets is the
        # netted quantity as far as its snapped closes carry it.
        netted = {
            fill_side: min(qty, _grid_qty(sum(close.qty for close in opposing_closes), quantize)),
            opposing_side: min(qty, _grid_qty(sum(close.qty for close in own_closes), quantize)),
        }
        if min(netted.values()) <= _QTY_EPS:
            journal.upsert_order(coid, qty=0.0)
            return []
        passes = int(extras.get(EXTRAS_KEY_NETTING_PASSES) or 0) + 1
        expected = dict(extras.get(EXTRAS_KEY_NETTING_EXPECTED) or {})
        for side, part in netted.items():
            expected[side] = round(float(expected.get(side) or 0.0) + part, 12)
        extras[EXTRAS_KEY_NETTING_PASSES] = passes
        extras[EXTRAS_KEY_NETTING_EXPECTED] = expected
        extras.pop(EXTRAS_KEY_NETTING_SETTLED_MS, None)
        halves: list[str] = []
        intent_key = row.intent_key or coid
        with journal.transaction():
            # A close fanned while the legs were read may have marked the
            # record contended since.
            latest = journal.get_order(coid)
            contended = (latest.extras or {}).get(EXTRAS_KEY_NETTING_CONTENDED) \
                if latest is not None else None
            if contended:
                extras[EXTRAS_KEY_NETTING_CONTENDED] = contended
            for tag, side, closes in (('a', fill_side, opposing_closes),
                                      ('b', opposing_side, own_closes)):
                parent = f"{coid}:p{passes}{tag}"
                create_residual_open_row(
                    journal, coid=f"{parent}:residual", symbol=symbol, side=side, qty=0.0,
                    intent_key=intent_key, pine_entry_id=row.pine_entry_id or '',
                    entry_coid=parent, run_tag='', bar_ts_ms=0, retry_seq=0,
                    close_plan=[(close.leg_id, close.qty) for close in closes],
                    netting=coid,
                )
                halves.append(f"{parent}:residual")
            journal.upsert_order(
                coid, qty=round(max(0.0, row.qty - min(netted.values())), 12), extras=extras)
        _blog_info(
            "%s: netting %s of the filled %s entry %r against the opposing legs "
            "(closes %s and %s)",
            symbol, qty, fill_side, row.pine_entry_id,
            [(close.leg_id, close.qty) for close in opposing_closes],
            [(close.leg_id, close.qty) for close in own_closes],
        )
        written = [journal.get_order(half) for half in halves]
        return [half for half in written if half is not None]

    @staticmethod
    def _widen_netting_closes(
            opposing_closes: tuple[LegClose, ...], own_closes: tuple[LegClose, ...],
            legs: list[PositionLeg], quantize, *, symbol: str, coid: str, netted: float,
            opposing: float, own_side: float,
    ) -> tuple[tuple[LegClose, ...], tuple[LegClose, ...]] | None:
        """Turn a netting pass into whole-leg closes on a port that cannot reduce a leg.

        Every sliced leg of either half is closed whole: a close is a
        risk-reducing request, so where it cannot be met exactly it is met in
        full. The pass keeps the net position on its side and never larger
        (the venue may end smaller than the book, and the close fills beyond
        the netted quantity then move the book like any close); a widening
        that would leave a larger net, or the other side, is not sent and the
        record stops with a warning, leaving both sides open. A leg whose
        venue position also holds exposure the run does not own
        (:attr:`~PositionLeg.venue_qty`) cannot be closed whole either. A
        widened pass is reported in one warning with the legs it closes and
        the net it leaves.

        :return: The two halves, or ``None`` when the pass cannot be sent.
        """
        live_by_id = {leg.leg_id: leg for leg in legs}

        def whole(closes: tuple[LegClose, ...]) -> tuple[LegClose, ...] | None:
            out: list[LegClose] = []
            for close in closes:
                leg = live_by_id.get(close.leg_id)
                if leg is None:
                    return None
                venue_size = leg.venue_qty if leg.venue_qty is not None else leg.qty
                if quantize(venue_size) > quantize(leg.qty):
                    return None
                out.append(close if quantize(close.qty) >= quantize(leg.qty)
                           else LegClose(leg_id=close.leg_id, qty=leg.qty))
            return tuple(out)

        whole_opposing = whole(opposing_closes)
        whole_own = whole(own_closes)
        intended = opposing - own_side
        if whole_opposing is None or whole_own is None:
            _blog_warning(
                "%s: netting %r is not sent: this broker only closes a whole position "
                "on a hedging account, and a leg to net also holds exposure this run "
                "does not own; both sides stay open (net %s)",
                symbol, coid, intended,
            )
            return None
        closed_opposing = sum(close.qty for close in whole_opposing)
        closed_own = sum(close.qty for close in whole_own)
        after = intended - (closed_opposing - closed_own)
        tolerance = _QTY_EPS * max(1.0, abs(intended))
        if (abs(after) > abs(intended) + tolerance
                or (after * intended < 0.0 and abs(after) > tolerance)
                or (abs(intended) <= tolerance < abs(after))):
            _blog_warning(
                "%s: netting %r is not sent: this broker only closes a whole leg on a "
                "hedging account, and closing legs %s and %s whole would leave the net "
                "at %s instead of %s; both sides stay open",
                symbol, coid, [c.leg_id for c in whole_opposing],
                [c.leg_id for c in whole_own], after, intended,
            )
            return None
        if closed_opposing > netted + tolerance or closed_own > netted + tolerance:
            _blog_warning(
                "%s: netting %r closes legs %s and %s whole (%s and %s for the %s to "
                "net): this broker only closes a whole leg on a hedging account, so "
                "the net position ends at %s instead of %s",
                symbol, coid, [c.leg_id for c in whole_opposing],
                [c.leg_id for c in whole_own], closed_opposing, closed_own, netted,
                after, intended,
            )
        return whole_opposing, whole_own

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
        DIFFERENT exit set. This consults the ownership index instead:
        it retires only the rows matching the cancel's ``intent_key`` (or, for a
        cancel-all-by-pine_id with ``from_entry`` None, every row whose exit
        shares that pine_id). Each leg those rows sit on is amended to the merged
        levels of the exits that STILL own it — so a co-owning exit (an SL-only
        exit next to the cancelled TP-only one, or another entry's exit on an
        unattributed leg) keeps its protection — and is cleared only when no
        owner is left; then the rows are released. A leg that is no longer open
        carries no bracket, so its rows are released without a venue call.
        An exit that owns no row has nothing to clear — a benign empty result.

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
        cancel_coid = envelope.client_order_id(KIND_CANCEL)
        owned: dict[str, list['OrderRow']] = {}
        for row in iter_active_bracket_ownerships(
                self._ownership, from_entry=intent.from_entry):
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
        with self._ownership.transaction():
            for leg_rows in owned.values():
                for row in leg_rows:
                    update_bracket_ownership_state(
                        self._ownership, coid=row.client_order_id,
                        new_state=BRACKET_OWN_STATE_CLEARING,
                        extras_patch={EXTRAS_KEY_BRACKET_OWN_CLEAR_COID: cancel_coid},
                    )
        cleared: list[str] = []
        live = self._exposure_by_leg(await _read_legs(port, intent.symbol)) if owned else {}
        for leg_id, leg_rows in owned.items():
            try:
                if leg_id in live:
                    leg, surviving = live[leg_id]
                    await self._reassert_remaining_owners(
                        port, leg, surviving, leg_rows[0].side, cancel_coid,
                    )
            except ExchangeOrderRejectedError as exc:
                self._report_retirement_refusal(intent.symbol, leg_id, leg_rows, exc)
                continue
            except ExchangeRateLimitError as exc:
                self._report_retirement_refusal(intent.symbol, leg_id, leg_rows, exc)
                break
            cleared.extend(leg_id for _row in self._release_retired_rows(leg_rows))
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

    def retire_cancelled_owners(
            self, cancelled: 'Callable[[str], bool]', clear_coid: 'Callable[[str], str]',
    ) -> int:
        """Mark the ownership rows of exits a script cancelled ``clearing``, before the replay.

        For a cancel the script issued while the restart was unsettled
        (:attr:`~BrokerPosition.restart_cancels`):
        the exit's rows must not be re-asserted by :meth:`restart_replay`, which
        would put the cancelled levels back on the venue until the diff clears
        them. Marked ``clearing``, the replay retires them as it finishes an
        interrupted clear (:meth:`_replay_bracket_leg`): each leg is amended to
        the merged levels of its remaining owners, cleared when none is left,
        and the rows are released. Persist-first like
        :meth:`run_exit_bracket_clear`, so a store failure raises before
        anything is sent.

        :param cancelled: Whether the script cancelled the exit of this ownership
            row's intent key (exit id and entry joined by
            :data:`~pynecore.core.broker.models.INTENT_KEY_SEP`).
        :param clear_coid: The clear coid of a row's intent key.
        :return: How many rows were marked.
        """
        rows = [
            row for row in iter_active_bracket_ownerships(self._ownership)
            if (row.extras or {}).get(EXTRAS_KEY_BRACKET_OWN_STATE) != BRACKET_OWN_STATE_CLEARING
            and cancelled(row.intent_key or '')
        ]
        if not rows:
            return 0
        with self._ownership.transaction():
            for row in rows:
                update_bracket_ownership_state(
                    self._ownership, coid=row.client_order_id,
                    new_state=BRACKET_OWN_STATE_CLEARING,
                    extras_patch={EXTRAS_KEY_BRACKET_OWN_CLEAR_COID:
                                  clear_coid(row.intent_key or row.client_order_id)},
                )
        return len(rows)

    def retarget_replaced_owners(self, placed: Mapping[str, ExitIntent]) -> int:
        """Give the ownership rows of exits a script placed again its new levels, before the replay.

        An exit the script placed (again) while the restart was unsettled
        replaces the one the previous process persisted: :meth:`restart_replay`
        must assert the levels of the script's exit, never the replaced ones.
        Each ``active`` row of an exit in ``placed`` whose levels differ is
        rewritten to the levels of that exit, persist-first, so a store failure
        raises before anything is sent. The rows stay ``active``; the diff
        afterwards treats the exit like any other one it holds.

        :param placed: The exits the script holds, by intent key.
        :return: How many rows were rewritten.
        """
        rows = []
        for row in iter_active_bracket_ownerships(self._ownership):
            extras = row.extras or {}
            intent = placed.get(row.intent_key or '')
            if (intent is None
                    or extras.get(EXTRAS_KEY_BRACKET_OWN_STATE) != BRACKET_OWN_STATE_ACTIVE):
                continue
            levels = {
                EXTRAS_KEY_BRACKET_OWN_TP: intent.tp_price,
                EXTRAS_KEY_BRACKET_OWN_SL: intent.sl_price,
                EXTRAS_KEY_BRACKET_OWN_TRAIL_PRICE: intent.trail_price,
                EXTRAS_KEY_BRACKET_OWN_TRAIL_OFFSET: intent.trail_offset,
            }
            if any(extras.get(key) != value for key, value in levels.items()):
                rows.append((row, levels))
        if not rows:
            return 0
        with self._ownership.transaction():
            for row, levels in rows:
                update_bracket_ownership_state(
                    self._ownership, coid=row.client_order_id,
                    new_state=BRACKET_OWN_STATE_ACTIVE, extras_patch=levels,
                )
        return len(rows)

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

    async def settle_pending_close_legs(self, port: 'PositionPort') -> bool:
        """Settle every pending close-leg row now, in the running process.

        The first pass of :meth:`restart_replay` on its own: what each pending
        row's send executed is established from the live legs and the venue's
        answer, and only what is provably still open is re-sent
        (:meth:`_replay_close_legs`). The sync engine runs it for a held
        partial-bracket close whose leg send's answer was lost, so the bot
        settles it without a restart.

        :return: ``True`` when every pending row was settled.
        """
        if self._store_ctx is None:
            return True
        return await self._replay_close_legs(port)

    async def _replay_close_legs(self, port: 'PositionPort') -> bool:
        """Reconcile + re-dispatch any pending close-leg rows (first replay pass).

        For every persisted ``pending`` close-leg row, reconcile against the live
        broker legs before re-sending: if the leg is gone, the close already
        landed — finalise the row without dispatching; if it is still open,
        establish what the close already executed and re-send only the part
        still provably open (:meth:`_settle_close_row`). The natural-close fill dedup
        guards the fill side.

        A close reduces the net position and never flips it, so the re-sends
        of one symbol share a budget: the net exposure of the closed side in
        the fresh read, less what earlier rows of the pass re-sent. A close
        that is part of a reversal still owing its residual open (a live
        residual-open breadcrumb with a positive quantity under the same
        parent close) is exempt: the reversal closes the opposing exposure
        whole by design and opens the residual only once those closes are
        gone. So is a half of a fill-time netting pass
        (:meth:`drain_nettings`): its closes come in pairs that leave the net
        position as it is. A cut-short reduction or flatten owes no open and
        stays within the budget.

        :return: ``True`` when every pending row was settled, ``False`` when the
            exchange refused at least one re-dispatch or the evidence could not
            yet decide what a close executed (that row stays pending for the
            next replay).
        """
        if self._store_ctx is None:
            return True
        pending = list(iter_active_close_legs(self._store_ctx))
        if not pending:
            return True
        by_symbol: dict[str, list] = {}
        for row in pending:
            by_symbol.setdefault(row.symbol, []).append(row)
        reversals = {
            row.client_order_id.removesuffix(':residual')
            for row in iter_active_residual_opens(self._store_ctx)
            if row.qty > 0.0 or (row.extras or {}).get(EXTRAS_KEY_RESIDUAL_OPEN_NETTING)
        }
        settled = True
        for symbol, rows in by_symbol.items():
            if symbol in self._closing_symbols:
                # Another pass is spending this symbol's budget right now.
                settled = False
                continue
            self._closing_symbols.add(symbol)
            try:
                if not await self._replay_symbol_close_legs(symbol, rows, reversals, port):
                    settled = False
            finally:
                self._closing_symbols.discard(symbol)
        return settled

    async def _replay_symbol_close_legs(
            self, symbol: str, rows: list['OrderRow'], reversals: set[str], port: 'PositionPort',
    ) -> bool:
        """Settle the pending close-leg rows of one symbol under one budget.

        Run while the pass owns ``symbol`` (:meth:`_replay_close_legs`): the
        budget comes from one fresh read less the sends still reserved
        against it (:meth:`_reserved_view`), so a pass never spends what an
        earlier pass already sent, and is spent by this pass's sends alone.

        :return: ``True`` when every row of the symbol was settled.
        """
        settled = True
        legs = await _read_legs(port, symbol)
        live_by_id = {leg.leg_id: leg for leg in legs}
        quantize = await _read_quantizer(port, symbol)
        # The budget is the net exposure less every close an earlier pass
        # sent that the read does not show yet; the rows settle their own
        # sends from the raw read and the venue's evidence.
        unspent = await self._reserved_view(port, symbol, legs, own_parent=None)
        totals = {'buy': 0, 'sell': 0}
        for leg in unspent:
            totals[leg.side] = totals.get(leg.side, 0) + quantize(leg.qty)
        budget = {
            side: max(0, volume - sum(v for other, v in totals.items() if other != side))
            for side, volume in totals.items()
        }
        for row in rows:
            extras = row.extras or {}
            live_leg = live_by_id.get(extras.get(EXTRAS_KEY_CLOSE_LEG_ID) or '')
            capped = (live_leg is not None
                      and extras.get(EXTRAS_KEY_CLOSE_PARENT_COID) not in reversals)
            cap = budget.get(live_leg.side, 0) if capped and live_leg is not None else None
            verdict, sent = await self._settle_close_row(
                row.client_order_id, symbol, live_by_id, quantize, port, cap=cap,
            )
            if capped and live_leg is not None:
                budget[live_leg.side] = budget.get(live_leg.side, 0) - sent
            if verdict != _CLOSE_ROW_SETTLED:
                settled = False
        return settled

    async def _settle_close_row(
            self, coid: str, symbol: str, live_by_id: Mapping[str, PositionLeg], quantize,
            port: 'PositionPort', *, cap: int | None = None,
    ) -> tuple[str, int]:
        """Settle one close-leg row whose close may already have executed.

        A crash between the send and its ack, or a send that ended without a
        verdict, leaves the close's outcome unknown: it may have executed in
        full, in part, or not at all, and a venue need not dedup a re-send
        under the same client order id. So only what is provably still owed
        is re-sent. The executed part is the larger of two measures: the
        venue's own record of every send of this close
        (:meth:`PositionPort.close_leg_executed` over the row's attempt coids),
        and how much the leg lost since the dispatch recorded its size. Either
        can over-count (another reduction of the leg in the window), which
        leaves the leg less reduced for the reconcile to settle; neither can
        make a re-send close the leg twice. When the venue cannot answer yet
        and the leg's size does not already account for the whole close,
        nothing is sent and the row stays pending; the leg keeps its
        protection meanwhile. A re-send goes out under a new attempt coid,
        recorded on the row BEFORE the wire call, so the next settlement asks
        about it too. The venue is asked from the instant the first send was
        about to go out, never from a later reopen of the row.

        A re-send is never more than ``cap``: the evidence says how much of
        the close is still owed, but what is safe to close now (the net
        position may have shrunk since the dispatch) is the caller's plan.

        One pass at a time owns a row: a pass the engine's bridge stopped
        waiting for can still be running when the next one starts, and both
        would otherwise read the same attempts and send the same remainder.
        A row another pass holds is left alone (undecided); the row is read
        again once owned, so the attempts its earlier owner recorded count.

        :param coid: The row's client order id (the leg close's own id).
        :param symbol: The traded symbol.
        :param live_by_id: The symbol's open legs by leg id.
        :param quantize: The port's Pine-units to broker-grid quantizer.
        :param port: The position port.
        :param cap: The most broker-grid volume a re-send may carry, or
            ``None`` for no limit beyond the row and the live leg.
        :return: The verdict (settled, undecided, refused) and the broker-grid
            volume this call sent.
        """
        if coid in self._close_rows_busy:
            return _CLOSE_ROW_UNDECIDED, 0
        journal = self._close_journal
        self._close_rows_busy.add(coid)
        try:
            row = journal.get_order(coid)
            if row is None or row.closed_ts_ms is not None:
                return _CLOSE_ROW_SETTLED, 0
            extras = row.extras or {}
            leg_id: str | None = extras.get(EXTRAS_KEY_CLOSE_LEG_ID)
            live_leg = live_by_id.get(leg_id) if leg_id is not None else None
            if live_leg is None or leg_id is None:
                # Leg vanished — the close landed, and no send of it can take
                # anything more. Finalise only.
                finish_close_leg_row(journal, coid, release_sends=True)
                return _CLOSE_ROW_SETTLED, 0
            persisted = int(extras.get(EXTRAS_KEY_CLOSE_LEG_VOLUME) or 0)
            live_volume = quantize(live_leg.qty)
            attempts = [str(sent) for sent in extras.get(EXTRAS_KEY_CLOSE_LEG_ATTEMPTS) or ()]
            if not attempts:
                attempts = [coid]
            pre_volume = extras.get(EXTRAS_KEY_CLOSE_LEG_PRE_VOLUME)
            shrunk = max(0, int(pre_volume) - live_volume) if pre_volume is not None else 0
            since_ms = extras.get(EXTRAS_KEY_CLOSE_LEG_SINCE_MS)
            executed = await port.close_leg_executed(
                symbol, leg_id, tuple(attempts),
                int(since_ms) if since_ms is not None else row.created_ts_ms,
            )
            if executed is None:
                if shrunk < persisted:
                    _blog_warning(
                        "one-way close of leg %s (%s x%d): what it executed cannot "
                        "be established yet; nothing is re-sent, it is asked again "
                        "next sync",
                        leg_id, symbol, persisted,
                    )
                    return _CLOSE_ROW_UNDECIDED, 0
                executed = 0
            residual = min(persisted - max(executed, shrunk), live_volume)
            if cap is not None and residual > cap:
                _blog_warning(
                    "one-way close of leg %s (%s): %d of the earlier close is still "
                    "owed, only %d is safe to close now; the rest is not sent",
                    leg_id, symbol, residual, max(cap, 0),
                )
                residual = cap
            if residual > 0 and not getattr(port, 'supports_partial_leg_close', True):
                residual = self._whole_leg_resend(
                    residual, live_leg, live_volume, quantize, cap=cap,
                    symbol=symbol, leg_id=leg_id,
                )
            sent_attempts = list(attempts)
            if residual > 0:
                attempt_coid = close_leg_attempt_client_order_id(coid, len(attempts))
                sent_attempts.append(attempt_coid)
                try:
                    with journal.transaction():
                        update_close_leg_state(
                            journal, coid=coid,
                            new_state=CLOSE_LEG_STATE_PENDING,
                            extras_patch={EXTRAS_KEY_CLOSE_LEG_ATTEMPTS: sent_attempts},
                        )
                        self._reserve_close(coid, attempt_coid, residual, live_volume)
                except sqlite3.Error as exc:
                    raise JournalWriteError(f"{symbol} close re-send of leg {leg_id}",
                                            exc) from exc
                try:
                    await port.close_leg(symbol, leg_id, residual, attempt_coid)
                except ExchangeConnectionError:
                    self._release_unsent_attempt(journal, symbol, leg_id, coid, attempt_coid)
                    raise
                except ExchangeOrderRejectedError as exc:
                    self._release_unsent_attempt(journal, symbol, leg_id, coid, attempt_coid)
                    _blog_warning(
                        "one-way close of leg %s (%s x%d) refused by the exchange, "
                        "kept pending for the next sync: %s",
                        leg_id, symbol, residual, exc,
                    )
                    return _CLOSE_ROW_REFUSED, 0
            try:
                finish_close_leg_row(journal, coid)
            except sqlite3.Error as exc:
                if residual <= 0:
                    raise
                # The re-send went out and the row still holds its reservation:
                # the dispatch parks, and the row's finish is owed.
                self._owe_close_row_finish(coid, sent_attempts)
                raise OrderDispositionUnknownError(
                    f"{symbol} close of leg {leg_id}: the re-send of {coid} was sent but "
                    f"its journal write failed ({exc}); settled from venue data",
                    client_order_id=coid, cause=exc,
                ) from exc
            return _CLOSE_ROW_SETTLED, max(residual, 0)
        finally:
            self._close_rows_busy.discard(coid)

    @staticmethod
    def _release_unsent_attempt(
            journal: OwnershipJournal, symbol: str, leg_id: str, coid: str, attempt_coid: str,
    ) -> None:
        """Release the reservation of a close re-send the venue did not take.

        :raises OrderDispositionUnknownError: When the write fails. The attempt
            then stays reserved on the row, and the next settlement asks the
            venue about it like about any other send.
        """
        try:
            release_close_leg_send(journal, coid, attempt_coid)
        except sqlite3.Error as exc:
            raise OrderDispositionUnknownError(
                f"{symbol} close of leg {leg_id}: the re-send {attempt_coid} was not taken "
                f"but its journal write failed ({exc}); settled from venue data",
                client_order_id=coid, cause=exc,
            ) from exc

    @staticmethod
    def _whole_leg_resend(
            residual: int, live_leg: PositionLeg, live_volume: int, quantize, *,
            cap: int | None, symbol: str, leg_id: str,
    ) -> int:
        """Size a close re-send for a port that only closes a whole venue position.

        Such a port (``supports_partial_leg_close = False``) closes the whole
        venue position whatever volume it is given, so a re-send of less than
        the position is sent as the whole position, reported at WARNING, the
        same way :meth:`_widen_partial_closes_to_whole_legs` widens a planned
        slice. It is not sent at all where the whole position is more than is
        safe: more than ``cap`` (closing it would flip the net position) or
        more than the run owns of it (:attr:`~PositionLeg.venue_qty`).

        :param residual: The broker-grid volume the evidence says is owed.
        :param live_leg: The leg as the fresh read shows it.
        :param live_volume: The leg's run-owned size on the broker grid.
        :param quantize: The port's Pine-units to broker-grid quantizer.
        :param cap: The most a re-send may carry, or ``None`` for no limit.
        :param symbol: The traded symbol, for the warnings.
        :param leg_id: The broker leg id, for the warnings.
        :return: The volume to send; ``0`` for nothing.
        """
        venue_volume = (quantize(live_leg.venue_qty)
                        if live_leg.venue_qty is not None else live_volume)
        if residual >= venue_volume:
            return residual
        if venue_volume > live_volume:
            _blog_warning(
                "one-way close of leg %s (%s): %d is still owed, but this broker "
                "only closes a whole position on a hedging account, and the venue "
                "position (%d) also holds exposure this run does not own; nothing "
                "is sent",
                leg_id, symbol, residual, venue_volume,
            )
            return 0
        if cap is not None and venue_volume > cap:
            _blog_warning(
                "one-way close of leg %s (%s): %d is still owed, but this broker "
                "only closes a whole leg on a hedging account, and closing the "
                "whole leg (%d) would flip the net position; nothing is sent",
                leg_id, symbol, residual, venue_volume,
            )
            return 0
        _blog_warning(
            "one-way close of leg %s (%s): %d is still owed of a %d leg; this "
            "broker only closes a whole leg on a hedging account, so the whole "
            "leg is closed",
            leg_id, symbol, residual, venue_volume,
        )
        return venue_volume

    async def _replay_residual_opens(self, port: 'PositionPort') -> None:
        """Re-dispatch any reversal residual-open a crash left un-persisted.

        Runs AFTER :meth:`_replay_close_legs`, so the reversal's FIFO closes are
        already reconciled and re-opening the residual cannot race them. For each
        live breadcrumb: if the residual's own entry row already exists, the
        ``place_leg`` persist-first write landed and the entry journal / startup
        recovery own it — just clear the breadcrumb; otherwise close every
        opposing leg the fresh read still shows and leave the breadcrumb for
        the next pass, or, when none is left, rebuild the dispatch envelope and
        re-open the residual under the deterministic entry coid of its current
        attempt (``KIND_ENTRY``, or ``KIND_ENTRY_TRIGGERED`` for a triggered
        reversal; a duplicate is therefore impossible at the exchange dedup),
        then clear it.
        """
        if self._store_ctx is None:
            return
        for row in list(iter_active_residual_opens(self._store_ctx)):
            await self._replay_residual_one(row, port)

    def _residual_entry_sent(self, entry_coid: str | None) -> bool:
        """Whether the residual open of a breadcrumb already went out.

        With a store, ``place_leg``'s persist-first entry row of the current
        attempt proves it; on a run without one, the entry coid recorded right
        before ``place_leg``. An attempt whose ``place_leg`` failed before it
        left the client may leave its entry row behind, so the breadcrumb then
        moves on to the next attempt's coid
        (:func:`~pynecore.core.broker.store_helpers.advance_residual_open_attempt`):
        the stale row never reads as the open, before or after a restart.

        :param entry_coid: The current attempt's deterministic entry coid.
        """
        if entry_coid is None:
            return False
        if self._store_ctx is not None:
            return self._store_ctx.get_order(entry_coid) is not None
        return entry_coid in self._residual_entries_sent

    async def _replay_residual_one(
            self, row, port: 'PositionPort', *, envelope: 'DispatchEnvelope | None' = None,
    ) -> tuple:
        """Reconcile one residual-open breadcrumb and open the residual when it is due.

        The one path every residual open of a reversal goes through:
        :meth:`run_reversal` right after its closes (``envelope`` given),
        :meth:`drain_residual_opens` every sync and the restart replay. The
        residual is opened only by a call whose fresh read shows no opposing
        leg.

        One pass at a time owns a breadcrumb: a pass the engine's bridge
        stopped waiting for can still be running when the next one starts,
        and both would otherwise see the open as not sent yet. A breadcrumb
        another pass holds is left alone; the row is read again once owned. A
        recovery pass (no ``envelope``) closes the opposing legs its own read
        shows, so it also owns the symbol's closes
        (:attr:`~OneWayEmulator._closing_symbols`) for the whole pass; while
        another pass closes legs of the symbol, the breadcrumb is left to the
        next sync.

        :param row: The live breadcrumb row.
        :param port: The position port.
        :param envelope: The reversal's own dispatch envelope when called from
            :meth:`run_reversal`: a still-listed opposing leg is then left to
            the next drain (its close was dispatched a moment ago, so there
            is no evidence of its outcome yet), and a refused, unsent or
            ambiguous open is raised to the dispatching caller. Without it the
            envelope :meth:`run_reversal` kept is used, or after a restart one
            rebuilt from the breadcrumb, and the outcome of the open is
            absorbed (a refusal discharges the breadcrumb, an unsent or
            ambiguous open leaves it live).
        :return: The orders the residual open produced; empty when nothing was
            opened by this call.
        """
        coid = row.client_order_id
        if coid in self._residual_rows_busy:
            return ()
        # A recovery pass closes the opposing legs its own read shows, so it
        # owns the symbol's closes as well; :meth:`run_reversal` already does.
        symbol = row.symbol
        claim_symbol = envelope is None
        if claim_symbol and symbol in self._closing_symbols:
            return ()
        self._residual_rows_busy.add(coid)
        if claim_symbol:
            self._closing_symbols.add(symbol)
        try:
            owed_move = self._owed_attempt_moves.get(coid)
            if owed_move is not None:
                self._abandon_residual_attempt(coid, *owed_move)
            current = self._close_journal.get_order(coid)
            if current is None or current.closed_ts_ms is not None:
                return ()
            return await self._replay_residual_owned(current, port, envelope)
        finally:
            self._residual_rows_busy.discard(coid)
            if claim_symbol:
                self._closing_symbols.discard(symbol)

    async def _replay_residual_owned(
            self, row, port: 'PositionPort', envelope: 'DispatchEnvelope | None',
    ) -> tuple:
        """The body of :meth:`_replay_residual_one`, run while the pass owns ``row``."""
        extras = row.extras or {}
        entry_coid: str | None = extras.get(EXTRAS_KEY_RESIDUAL_OPEN_ENTRY_COID)
        if self._residual_entry_sent(entry_coid):
            # ``place_leg`` was already reached for this open — the open is
            # owned by the entry path. Discharge.
            self._discharge_residual(row.client_order_id)
            return ()
        # The residual open never went out (persist-first guarantees the entry
        # row precedes any wire send, so its absence proves no dispatch happened —
        # re-opening cannot double-open). The breadcrumb is written BEFORE the
        # FIFO closes, so a crash in that window can also leave the opposing legs
        # un-closed AND their close-leg rows un-persisted — ``_replay_close_legs``
        # then has nothing to re-dispatch. Re-opening the residual on top of those
        # still-live opposing legs would over-expose the book. Reconcile against
        # the live legs first: a genuine reversal (positive residual) fully
        # consumes the opposing exposure, so every still-live opposing leg is an
        # owed close — FIFO-close them under the SAME parent coid before the
        # residual open (a close already sent under its id is settled from the
        # venue's evidence, never re-sent as is).
        #
        # A close that returned is only dispatched, not executed: it may still
        # fill in part or not at all. So the residual is opened only by a pass
        # whose fresh read shows no opposing leg left; a pass that had closes to
        # send leaves the breadcrumb live, and the next sync's drain opens the
        # residual once the closes are visible in the book.
        run_tag = extras.get(EXTRAS_KEY_RESIDUAL_OPEN_RUN_TAG)
        bar_ts_ms = extras.get(EXTRAS_KEY_RESIDUAL_OPEN_BAR_TS_MS)
        retry_seq = extras.get(EXTRAS_KEY_RESIDUAL_OPEN_RETRY_SEQ)
        assert isinstance(run_tag, str)
        assert isinstance(bar_ts_ms, int)
        assert isinstance(retry_seq, int)
        opposing_side = 'sell' if row.side == 'buy' else 'buy'
        live_legs = await _read_legs(port, row.symbol)
        parent_coid = row.client_order_id.removesuffix(':residual')
        # The closes still to send are planned on the read less what other
        # closes' sends may still take (:meth:`_reserved_view`); the open
        # waits for the read itself to show the opposing exposure gone.
        unspent = await self._reserved_view(port, row.symbol, live_legs, own_parent=parent_coid)
        close_plan = extras.get(EXTRAS_KEY_RESIDUAL_OPEN_CLOSE_PLAN)
        planned: dict[str, float] | None = None
        counted: set[str] | None = None
        if close_plan is not None:
            # A cut-short reduction or flatten: only its own planned closes
            # are owed, each at most its planned quantity.
            plan_qty = {str(leg_id): float(qty) for leg_id, qty in close_plan}
            planned = plan_qty
            owed_closes = tuple(
                LegClose(leg_id=leg.leg_id, qty=min(leg.qty, plan_qty[leg.leg_id]))
                for leg in unspent
                if leg.side == opposing_side and leg.leg_id in plan_qty and leg.qty > 0.0
            )
        else:
            # A reversal: its own counted closes are settled whole (a re-send
            # takes only what they provably did not execute); any other
            # opposing exposure is closed FIFO up to what is left of the
            # combined order, and only the rest is opened.
            counted = {str(c) for c in extras.get(EXTRAS_KEY_RESIDUAL_OPEN_COUNTED_CLOSES) or ()}
            own = [leg for leg in unspent
                   if leg.side == opposing_side and leg.qty > 0.0
                   and close_leg_client_order_id(parent_coid, leg.leg_id) in counted]
            fresh = [leg for leg in unspent
                     if leg.side == opposing_side and leg.qty > 0.0
                     and close_leg_client_order_id(parent_coid, leg.leg_id) not in counted]
            fresh_closes: tuple[LegClose, ...] = ()
            if fresh and row.qty > 0.0:
                fresh_closes, _shortfall = select_legs_for_close(row.qty, fresh, opposing_side)
            owed_closes = (
                *(LegClose(leg_id=leg.leg_id, qty=leg.qty) for leg in own),
                *fresh_closes,
            )
        # Once a reversal's combined order is used up by its closes, only its
        # own closes' legs hold the breadcrumb: other opposing exposure is not
        # this order's to wait for.
        opposing_listed = any(
            leg.side == opposing_side and leg.qty > 0.0
            and (planned is None or leg.leg_id in planned)
            and (counted is None or row.qty > 0.0
                 or close_leg_client_order_id(parent_coid, leg.leg_id) in counted)
            for leg in live_legs
        )
        if opposing_listed and not owed_closes:
            # Every opposing leg the read lists is held by a close sent
            # earlier: nothing to send, and the open waits for the read.
            if envelope is not None:
                self._residual_envelopes[row.client_order_id] = envelope
            return ()
        if owed_closes:
            if envelope is not None:
                self._residual_envelopes[row.client_order_id] = envelope
                return ()
            intent_key = row.intent_key or row.pine_entry_id or ''
            # A leg the fresh read shows may be one the reversal never
            # planned (its venue position now shared, or a new one): it gets
            # the same whole-position checks as the reversal's own plan, so a
            # port that only closes whole positions never closes exposure the
            # run does not own. A leg that cannot be closed stays open, and the
            # residual with it.
            try:
                sendable = await self._widen_partial_closes_to_whole_legs(
                    owed_closes, unspent, symbol=row.symbol, port=port,
                    intent_key=intent_key,
                )
            except OrderSkippedByPlugin:
                sendable = ()
            if sendable:
                sent, _sent_qty = await self._fan_out_closes(
                    sendable, live_legs, symbol=row.symbol, side=row.side,
                    intent_key=intent_key, pine_id=row.pine_entry_id or '',
                    parent_coid=parent_coid, port=port, skippable=False,
                    residual_coid=row.client_order_id if close_plan is None else None,
                )
                if close_plan is not None and all(volume == 0 for _leg_id, volume in sent):
                    # Every planned close is settled from the venue's evidence
                    # (or rounds below the grid) and nothing more was sent:
                    # what the book still shows of those legs is exposure the
                    # plan leaves open.
                    self._discharge_residual(row.client_order_id)
            return ()
        if row.qty <= 0.0:
            # A flatten whose closes were cut short owes no open: once its
            # opposing legs are gone it is done.
            self._discharge_residual(row.client_order_id)
            return ()
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
            return ()
        if envelope is not None:
            return await self._open_residual(row, envelope, port, entry_coid, absorb=False)
        kept = self._residual_envelopes.get(row.client_order_id)
        if kept is not None:
            # The breadcrumb names the open's current attempt.
            kept = replace(kept, retry_seq=retry_seq)
            return await self._open_residual(row, kept, port, entry_coid, absorb=True)
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
        rebuilt = DispatchEnvelope(
            intent=EntryIntent(
                pine_id=row.pine_entry_id, symbol=row.symbol, side=row.side,
                qty=row.qty, order_type=OrderType.MARKET,
                triggered_entry=triggered,
            ),
            run_tag=run_tag, bar_ts_ms=bar_ts_ms, retry_seq=retry_seq,
            coid_max_len=coid_max_len,
        )
        return await self._open_residual(row, rebuilt, port, entry_coid, absorb=True)

    async def _open_residual(
            self, row, envelope: 'DispatchEnvelope', port: 'PositionPort',
            entry_coid: str | None, *, absorb: bool,
    ) -> tuple:
        """Send a breadcrumb's residual open and discharge the breadcrumb.

        :param row: The live breadcrumb row.
        :param envelope: The envelope the open is dispatched under.
        :param port: The position port.
        :param entry_coid: The open's deterministic entry coid from the
            breadcrumb.
        :param absorb: ``True`` for a recovery pass (drain, restart replay):
            a definitive refusal discharges the breadcrumb, an unsent
            (:class:`~pynecore.core.broker.exceptions.ExchangeConnectionError`)
            or ambiguous open leaves it live, none is raised, and the orders
            of a successful open are kept for :meth:`take_opened_residuals`.
            ``False`` for :meth:`run_reversal`, which raises each: the refusal
            discharges the breadcrumb first, the unsent and the ambiguous open
            leave it live (:meth:`run_reversal` absorbs the unsent one).
        :return: The orders the open produced; empty when it did not open.
        """
        if entry_coid is not None:
            if self._store_ctx is None:
                # The in-memory counterpart of ``place_leg``'s persist-first
                # entry row: from here on the open may be on the wire.
                self._residual_entries_sent.add(entry_coid)
        try:
            orders = tuple(await port.place_leg(envelope, row.qty))
        except ExchangeOrderRejectedError:
            # The exchange definitively refused the residual entry — nothing
            # opened. Discharge the breadcrumb: the residual the exchange
            # rejected and Pine never re-signalled must not re-open. A
            # recovery pass stops here (``restart_replay`` runs inside the sync
            # startup wrapper, which only catches ``ExchangeConnectionError``,
            # so propagating would abort startup); the in-session reversal
            # re-raises, and the dispatch path turns it into a non-halting
            # skip.
            self._discharge_residual(row.client_order_id)
            if self._store_ctx is None and entry_coid is not None:
                self._residual_entries_sent.discard(entry_coid)
            if absorb:
                return ()
            raise
        except ExchangeConnectionError:
            # The request never left the client: nothing opened, and the
            # breadcrumb stays live under its next attempt
            # (:meth:`_abandon_residual_attempt`): the drain retries it on the
            # next sync. The closes before it already reduced the book, so
            # the open is owned by the breadcrumb alone — the in-session
            # reversal (:meth:`run_reversal`) absorbs the fault rather than
            # letting the dispatch path re-plan the combined entry from the
            # reduced book.
            if entry_coid is not None:
                self._residual_entries_sent.discard(entry_coid)
                try:
                    self._abandon_residual_attempt(row.client_order_id, envelope, entry_coid)
                except sqlite3.Error as exc:
                    # The move stays owed (:attr:`~OneWayEmulator._owed_attempt_moves`)
                    # and is written by the next pass over the breadcrumb, before
                    # anything reads its attempt; the unsent open is handled as
                    # usual meanwhile.
                    _blog_warning(
                        "one-way residual open %r: moving the breadcrumb to its next "
                        "attempt failed (%s); written by the next pass",
                        row.client_order_id, exc,
                    )
            if absorb:
                return ()
            raise
        except OrderDispositionUnknownError:
            # The residual entry's fate is unknown (ambiguous round-trip). The
            # breadcrumb stays live so the next sync's drain reconciles it
            # (discharged once the open is known to have gone out). A recovery
            # pass does not propagate — that would abort startup on a
            # recoverable round-trip.
            if absorb:
                return ()
            raise
        self._discharge_residual(row.client_order_id)
        if absorb and orders and row.intent_key is not None:
            self._opened_residuals.append((row.intent_key, orders))
        return orders

    def _abandon_residual_attempt(
            self, coid: str, envelope: 'DispatchEnvelope', entry_coid: str,
    ) -> None:
        """Move a breadcrumb on to the next attempt after an open that never left the client.

        ``place_leg`` may have left its persist-first entry row under the
        attempt's coid behind. Re-sending under that coid would leave the row
        as the only evidence of the re-send, and a crash after the next
        attempt started could not tell the stale row from the re-send's own.
        So the next attempt goes out under a fresh deterministic coid, and
        only its row proves the open went out. The abandoned attempt never
        reached the venue, so the fresh coid cannot duplicate it. Without a
        left-behind row the attempt keeps its coid.

        The fresh coid must also be one the engine never mints for the same
        entry: the engine bumps an entry's ``retry_seq`` by one on the same
        bar whenever it spends its id (a refused or superseded entry, a
        re-issue over a consumed fill), and an engine dispatch under the
        breadcrumb's attempt coid would read as the residual's open. So the
        attempts of a breadcrumb take their own block of ``retry_seq``
        values (:func:`_residual_attempt_retry_seq`), far above what the
        engine's one-by-one bumps reach on one bar and disjoint from every
        other dispatch's block.

        Until the move is written it is held in
        :attr:`~OneWayEmulator._owed_attempt_moves`, so a failed write is
        redone by the next pass over the breadcrumb.

        :param coid: The breadcrumb's client-order-id.
        :param envelope: The envelope the abandoned attempt went out under.
        :param entry_coid: The abandoned attempt's entry coid.
        """
        self._owed_attempt_moves[coid] = (envelope, entry_coid)
        if self._store_ctx is None or self._store_ctx.get_order(entry_coid) is None:
            del self._owed_attempt_moves[coid]
            return
        parsed = parse_client_order_id(entry_coid)
        parsed_wire = parse_wire_client_order_id(entry_coid) if parsed is None else None
        kind = (parsed.kind if parsed is not None
                else parsed_wire.kind if parsed_wire is not None
                else KIND_ENTRY)
        retry_seq = _residual_attempt_retry_seq(envelope.retry_seq)
        advance_residual_open_attempt(
            self._close_journal, coid,
            entry_coid=replace(envelope, retry_seq=retry_seq).client_order_id(kind),
            retry_seq=retry_seq,
        )
        del self._owed_attempt_moves[coid]

    def _discharge_residual(self, coid: str) -> None:
        """Close a residual-open breadcrumb and forget its kept envelope."""
        clear_residual_open_row(self._close_journal, coid)
        self._residual_envelopes.pop(coid, None)
        self._owed_attempt_moves.pop(coid, None)

    async def _replay_bracket_ownership(self, port: 'PositionPort') -> bool:
        """Re-assert / finish live bracket-ownership rows (second replay pass).

        Rows are handled per leg, because a leg carries ONE protection merged
        from all of its owners (:meth:`_merge_owner_levels`): re-asserting each
        row's own levels in turn would leave the leg on whichever row came last.
        See :meth:`_replay_bracket_leg`.

        :return: ``False`` when a re-assert met a closed market (its rows stay
            active for the next replay), ``True`` otherwise.
        """
        rows = list(iter_active_bracket_ownerships(self._ownership))
        if not rows:
            return True
        by_symbol: dict[str, dict[str | None, list['OrderRow']]] = {}
        for row in rows:
            by_symbol.setdefault(row.symbol, {}).setdefault(
                (row.extras or {}).get(EXTRAS_KEY_BRACKET_OWN_LEG_ID), [],
            ).append(row)
        complete = True
        for symbol, by_leg in by_symbol.items():
            live = self._exposure_by_leg(await _read_legs(port, symbol))
            for leg_id, leg_rows in by_leg.items():
                if not await self._replay_bracket_leg(
                        leg_rows, symbol, leg_id, live, port):
                    complete = False
        return complete

    async def _replay_bracket_leg(
            self, rows: list['OrderRow'], symbol: str, leg_id: str | None,
            live: dict[str, tuple[PositionLeg, float]], port: 'PositionPort',
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
        unchanged. The protection covers the leg's surviving quantity only, so
        a leg the venue bracket cannot be sized for goes back under the
        software leg guard (:meth:`_protect_leg`); its trailing stop resumes
        from the extreme its rows persisted (:meth:`software_guard_triggers`).

        :return: ``False`` when the venue refused the re-assert because the
            market is closed or throttled the call, ``True`` otherwise.
        """
        if leg_id is None or leg_id not in live:
            if leg_id is not None:
                self._venue_levels.pop((symbol, leg_id), None)
            for row in rows:
                update_bracket_ownership_state(
                    self._ownership, coid=row.client_order_id,
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
        leg, surviving = live[leg_id]
        levels = self._merge_owner_levels(
            symbol, leg_id, side,
            [(row.intent_key or row.client_order_id, self._row_levels(row))
             for row in active],
        )
        if clearing:
            first = clearing[0]
            try:
                await self._protect_leg(
                    port, leg, surviving, side, levels,
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
            self._release_retired_rows(clearing)
            return True
        newest = max(active, key=lambda owner: (owner.updated_ts_ms, owner.client_order_id))
        try:
            await self._protect_leg(
                port, leg, surviving, side, levels,
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
