"""
Spot-venue inventory accounting: execution ledger, balance-invariant
reconciliation and position synthesis.

Spot venues expose no position object — the base-asset inventory *is*
the long exposure, and the venue balance pools the bot's inventory with
pre-existing holdings, manual trades and deposits. This module owns the
core bookkeeping a spot broker plugin needs:

- **Append-only execution ledger** (``spot_executions``): every venue
  fill as signed base/quote deltas in exact decimal arithmetic, deduped
  on the venue fill id, exempt from retention (an open position must
  stay reconstructible for as long as it is open).
- **Inventory epoch** (``spot_inventory_epoch``): the reconciliation
  baseline generation. The invariant is
  ``expected_total = foreign_baseline + bot_inventory(ledger)`` where
  ``foreign_baseline`` was frozen at epoch creation. A drift the execution
  history does not explain within the settlement grace waits while it may
  still be a fill of the bot's own live orders the venue has not indexed
  yet — for a bounded time, since a resting stop leg is live for the whole
  life of a position. Then it is judged against the bot's OWN inventory,
  not the baseline: base is fungible, so a balance still holding at least
  the bot's inventory left the bot's share intact whatever else moved
  (a deposit, a foreign buy, a sale out of the foreign holdings). That is
  someone else's separate activity: an ERROR and a durable
  ``spot_foreign_holdings_reanchored`` audit, the foreign baseline
  re-anchored on the balance, and the bot keeps trading; the span the
  re-anchor held is read again until a late own fill in it would have been
  indexed. A balance below the bot's inventory is someone taking the bot's
  own base (below). Either verdict needs a complete history: with an
  incomplete one the bot's own fill cannot be ruled out, so the conflict
  stays pending — an ERROR and a durable
  ``spot_inventory_shortfall_unverified`` or
  ``spot_foreign_activity_unverified`` audit, nothing booked, no
  quarantine — until a complete history decides it. A drift found at
  startup is judged the same way, by the runtime reconcile: the startup
  cannot wait out the venue indexing a fill of the bot's own behind its
  balance.
- **Asset-ownership lease** (``spot_asset_owner``): one active logical
  run per ``(plugin, account, base asset)`` within a broker store; a
  concurrent second run starts quarantined instead of double-booking.
- **Exactly-once ledger→engine handoff**: a live fill is recorded and
  outbox-flipped (``delivered``) in one transaction before the plugin
  emits its :class:`~pynecore.core.broker.models.OrderEvent`; on
  restart the startup adoption folds the ENTIRE ledger into the
  synthesized position the engine adopts, so every fill reaches the
  engine on exactly one path — never both, never neither.
- **Quarantine, not adoption**: external intervention in the bot's
  inventory is unsupported by contract. A confirmed runtime conflict that
  left the balance below the bot's own inventory — someone sold or
  withdrew what the bot holds — latches the external-intervention
  quarantine (process stays alive as an observer, no venue write), which
  only an operator resume lifts; the explicit ``halt`` policy exits
  instead. Its epoch is ``taken``: no restart touches the ledger while that
  quarantine holds, and the first start after the resume keeps the part of
  the bot's inventory the account still holds. The manager's own
  bookkeeping trouble (an inconclusive startup read, a measurement a
  baseline is frozen on that never settles, a negative ledger, a lost
  lease) latches the entry block instead; recovery is a restart, which
  retires the corrupt ledger stretch and re-anchors a fresh epoch on the
  venue's measured balance.

The manager takes explicit ``now_ms`` timestamps so reconciliation
timing is fully deterministic under test, mirroring
:class:`~pynecore.core.broker.disappearance.DisappearanceTracker`.
"""
import logging
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any, Final, NamedTuple, Protocol

from pynecore.core.broker.exceptions import SpotInventoryConflictError
from pynecore.core.broker.models import ExchangePosition, QuarantineKind
from pynecore.core.plugin import is_retryable_provider_error

if TYPE_CHECKING:
    from collections.abc import Callable
    from pynecore.core.broker.storage import (
        RunContext,
        SpotEpochRow,
        SpotExecutionRow,
    )

__all__ = [
    'INVENTORY_CONFLICT_POLICIES',
    'SpotExecution',
    'SpotExecutionBatch',
    'SpotInventoryPort',
    'InventoryFold',
    'SpotStartupResult',
    'SpotInventoryManager',
    'canonical_decimal',
    'fold_inventory',
]

logger = logging.getLogger(__name__)

#: Valid ``on_inventory_conflict`` policies. Deliberately narrower than
#: the ``on_unexpected_cancel`` set: ``re_place`` would buy back an
#: operator's withdrawal and ``ignore`` would trade on corrupt books, so
#: neither is applicable to an attribution conflict.
INVENTORY_CONFLICT_POLICIES: Final = ('quarantine', 'halt')

#: Valid cursor scopes a port may declare for its execution-history API.
CURSOR_SCOPES: Final = ('account', 'product', 'time')

#: Hard cap on catch-up pagination per invocation — a runaway venue
#: cursor (next_cursor never converging) must not loop forever.
_MAX_CATCHUP_PAGES: Final = 10_000

#: Balance reads a startup measurement may take before the history read
#: after it books nothing (:meth:`SpotInventoryManager._measure_settled`).
_MAX_SETTLE_BALANCE_READS: Final = 3

#: How long a fill of the bot's own may still be missing from the venue's
#: execution history after the balance showed it, beyond the settlement grace
#: (measured live on Bybit spot: the list returned a fill the balance had shown
#: 61 s earlier). An own live order whose fill has not reached the history by
#: then did not cause the drift: it bounds how long such orders hold a verdict
#: (:meth:`SpotInventoryManager._in_flight_orders_explaining`) and how long the
#: span a re-anchor held is read again (:meth:`SpotInventoryManager._reread_held_span`).
_HISTORY_LAG_CEILING_S: Final = 300.0


class _MeasureReasons(NamedTuple):
    """Quarantine reasons of a failed :meth:`SpotInventoryManager._measure_settled`."""
    inconclusive: str
    unavailable: str
    invalid: str
    unsettled: str


_STARTUP_MEASURE_REASONS: Final = _MeasureReasons(
    'spot_catchup_inconclusive', 'spot_startup_balance_unavailable',
    'spot_startup_balance_invalid', 'spot_startup_unsettled',
)
_RECOVERY_MEASURE_REASONS: Final = _MeasureReasons(
    'spot_recovery_catchup_inconclusive', 'spot_recovery_balance_unavailable',
    'spot_recovery_balance_invalid', 'spot_recovery_unsettled',
)


@dataclass(frozen=True)
class _SettledMeasurement:
    """A balance read and the ledger caught up to it."""
    balance: Decimal
    cursor: str | None
    recovered: int


class _ForeignLedgerRow(Exception):
    """Internal signal: a fill id is booked under another logical run.

    Raised INSIDE a ledger transaction (which rolls back) and converted
    to the quarantine + :class:`SpotInventoryConflictError` OUTSIDE it —
    the quarantine's own store writes must not ride the aborted span.
    """

    def __init__(self, fill_id: str, owner_run_id: str) -> None:
        super().__init__(fill_id)
        self.fill_id = fill_id
        self.owner_run_id = owner_run_id


def canonical_decimal(value: Decimal | int | str) -> str:
    """Serialize a decimal to its canonical ledger string.

    Canonical form: plain (exponent-free) notation, no trailing zeros,
    ``-0`` collapsed to ``0`` — one value, one string, so ledger rows
    compare and round-trip exactly. Rejects non-finite values and
    anything :class:`~decimal.Decimal` cannot parse exactly. Floats are
    deliberately not accepted: binary floats carry representation error
    that must not enter an exact ledger — the caller converts via
    ``Decimal(str(x))`` explicitly if a float source is unavoidable.

    The trailing-zero strip is TEXTUAL, not ``Decimal.normalize()``:
    ``normalize`` rounds to the ambient decimal context (28 significant
    digits by default), which would silently corrupt a higher-precision
    value. ``format(d, 'f')`` renders the exact stored digits with no
    context rounding, so the round trip stays exact at any precision.

    :raises ValueError: On non-finite or unparseable input.
    """
    if isinstance(value, float):
        raise ValueError(
            "canonical_decimal: float input is not accepted; convert "
            "explicitly (Decimal(str(x))) so the representation choice "
            "is the caller's"
        )
    try:
        d = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"canonical_decimal: unparseable value {value!r}") from exc
    if not d.is_finite():
        raise ValueError(f"canonical_decimal: non-finite value {value!r}")
    if d == 0:
        return '0'
    s = format(d, 'f')
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s


@dataclass(frozen=True)
class SpotExecution:
    """One venue execution (fill), in exact decimal arithmetic.

    The canonical delta equations the plugin must apply when building
    these from the venue's raw fill report:

    - ``base_delta`` = signed executed base quantity, minus any fee
      charged in the BASE currency (a base fee reduces what was actually
      received). Positive on a buy, negative on a sell.
    - ``quote_delta`` = the opposite-signed notional, minus any fee
      charged in the QUOTE currency. Negative on a buy (quote spent),
      positive on a sell (quote received).
    - A fee charged in a third currency is recorded (``fee_amount`` /
      ``fee_currency``) but does not touch either delta — it does not
      move the base invariant.

    Validation is fail-closed: a fill that cannot be represented
    exactly, or whose signs contradict its side, raises at construction
    instead of corrupting the ledger.

    :ivar venue_seq: The venue's monotonic execution-sequence number
        when it exposes one, else ``None``. Breaks fold-ordering ties
        within one millisecond; a venue whose fills can share a
        millisecond MUST provide it, or a same-ms buy/sell pair may
        replay reversed into a false oversell.
    """
    fill_id: str
    side: str  # "buy" | "sell"
    base_delta: Decimal
    quote_delta: Decimal
    price: Decimal
    fee_amount: Decimal
    fee_currency: str
    ts_ms: int
    exchange_order_id: str | None = None
    client_order_id: str | None = None
    venue_seq: int | None = None

    def __post_init__(self) -> None:
        if not self.fill_id:
            raise ValueError("SpotExecution: empty fill_id")
        if not self.client_order_id:
            # Attribution floor: the ledger tracks the BOT's inventory,
            # not the account's. The bot's OWN client_order_id is the only
            # reference it controls — a raw exchange_order_id is not proof
            # of bot ownership (a manual/web trade carries one too). The
            # port must attribute each fill to a bot order and set its
            # client_order_id (mapping via its own order records when the
            # venue does not echo it); a fill it cannot attribute belongs
            # to ``conclusive=False`` history, not the ledger. Booking an
            # unattributed fill would move balance AND ledger together, so
            # the invariant would never fire.
            raise ValueError(
                f"SpotExecution {self.fill_id!r}: needs the bot's own "
                f"client_order_id — an execution not attributable to a bot "
                f"dispatch must not enter the inventory ledger (an "
                f"exchange_order_id alone is not proof of bot ownership)"
            )
        if self.side not in ('buy', 'sell'):
            raise ValueError(
                f"SpotExecution {self.fill_id!r}: unknown side {self.side!r}"
            )
        for name in ('base_delta', 'quote_delta', 'price', 'fee_amount'):
            v = getattr(self, name)
            if not isinstance(v, Decimal) or not v.is_finite():
                raise ValueError(
                    f"SpotExecution {self.fill_id!r}: {name} must be a "
                    f"finite Decimal, got {v!r}"
                )
        if self.price <= 0:
            raise ValueError(
                f"SpotExecution {self.fill_id!r}: non-positive price "
                f"{self.price}"
            )
        if self.fee_amount < 0:
            raise ValueError(
                f"SpotExecution {self.fill_id!r}: negative fee "
                f"{self.fee_amount}"
            )
        if self.side == 'buy' and (self.base_delta <= 0 or self.quote_delta > 0):
            raise ValueError(
                f"SpotExecution {self.fill_id!r}: buy requires "
                f"base_delta > 0 and quote_delta <= 0, got "
                f"base={self.base_delta} quote={self.quote_delta}"
            )
        if self.side == 'sell' and (self.base_delta >= 0 or self.quote_delta < 0):
            raise ValueError(
                f"SpotExecution {self.fill_id!r}: sell requires "
                f"base_delta < 0 and quote_delta >= 0, got "
                f"base={self.base_delta} quote={self.quote_delta}"
            )
        if self.ts_ms <= 0:
            raise ValueError(
                f"SpotExecution {self.fill_id!r}: invalid ts_ms {self.ts_ms}"
            )


@dataclass(frozen=True)
class SpotExecutionBatch:
    """One page of a port's execution-history read.

    :ivar executions: The fills in this page (any order; the ledger
        sorts deterministically).
    :ivar next_cursor: Durable cursor to persist once every execution in
        this page is recorded. ``None`` keeps the previous cursor.
    :ivar has_more: ``True`` when another page must be fetched with
        ``next_cursor`` before catch-up is complete.
    :ivar conclusive: ``False`` when the venue could not provide an
        authoritative answer (endpoint degraded, history window
        truncated). An inconclusive catch-up fails closed at startup.
    """
    executions: tuple[SpotExecution, ...] = ()
    next_cursor: str | None = None
    has_more: bool = False
    conclusive: bool = True


class SpotInventoryPort(Protocol):
    """Surface a spot broker plugin exposes to the core inventory manager.

    Attributes are venue/product facts fixed for the run; the two
    methods are the venue reads the reconciliation loop drives. All
    quantities are exact :class:`~decimal.Decimal` — the port owns the
    parse from the venue's wire format.
    """

    product_id: str
    """Venue product identifier (e.g. ``"BTC-USD"``) — the ledger and
    epoch key. Not necessarily the Pine symbol."""

    base_asset: str
    """Base asset code (e.g. ``"BTC"``) — the exclusive position asset."""

    quote_asset: str
    """Quote asset code (e.g. ``"USD"``) — the shared cash asset."""

    cursor_scope: str
    """Scope of the execution-history cursor: ``'account'``,
    ``'product'`` or ``'time'``. Persisted with the cursor so a plugin
    upgrade that changes the scope invalidates the stored cursor instead
    of silently misreading it. Time-based APIs must fetch with an
    overlapping window (the ledger dedups on fill id) rather than assume
    a strict cursor."""

    base_tolerance: Decimal
    """Asset-specific quantization tolerance for the balance invariant —
    small and fixed (venue rounding of the total balance), NEVER a
    settlement-lag allowance. Settlement lag is handled as a *temporal*
    grace state, not a numeric widening."""

    settlement_grace_s: float
    """How long a balance-invariant mismatch may persist (armed as a
    pending conflict, re-checked with fresh executions + balance) before
    it is confirmed as a conflict. Venue settlement latency, not a
    tuning knob for hiding drift."""

    position_dust_threshold: Decimal
    """Positive base quantity below which the venue intentionally exposes
    no engine position. This is normally the product's minimum quantity
    increment. A zero value disables dust-to-flat reconciliation."""

    async def min_sellable_base(self) -> Decimal | None:
        """Smallest base quantity the venue would accept as a sell NOW.

        The bound folds every venue-side sell constraint together: the
        quantity grid step, the product's minimum order quantity and the
        minimum order notional converted at a current price. A net
        inventory strictly below this bound cannot be sold at all — not
        even in aggregate — so it can never represent a strategy
        position; the manager retires it into the epoch baseline as fee
        dust (:meth:`SpotInventoryManager.reconcile`).

        Return ``None`` when the bound cannot be computed right now
        (typically: no price observed yet) — the manager then simply
        skips compaction until a later pass. MUST err low, never high: an
        overestimated bound would compact away sellable inventory.
        """
        ...

    async def fetch_executions(self, cursor: str | None) -> SpotExecutionBatch:
        """Read the BOT's execution history from ``cursor``.

        MUST return only executions attributable to THIS bot — every
        :class:`SpotExecution` carries the bot's own ``client_order_id``
        (map it from the venue fill's order id via the plugin's order
        records when the venue does not echo it; a raw exchange order id
        is not proof of bot ownership). The ledger accounts for the bot's
        own inventory, not the account's: a manual or foreign trade folded
        in as if it were the bot's would move the balance AND the ledger
        together, so the invariant would never fire. A venue whose
        account-history endpoint cannot be filtered/attributed to bot
        orders MUST return ``conclusive=False`` (which fails closed)
        rather than dump raw account trades.

        ``cursor=None`` means "no history belongs to the bot yet" (first
        startup): return an EMPTY batch whose ``next_cursor`` anchors at
        the venue's current watermark — the account's prior history is
        foreign inventory and belongs to the epoch baseline, not the
        ledger. Raise a transient error (connection) to abort the read;
        return ``conclusive=False`` when the venue answered but cannot
        be trusted as complete.
        """
        ...

    async def fetch_base_balance(self) -> Decimal:
        """The account's TOTAL owned base-asset amount.

        Must include available balance PLUS amounts locked in open
        (sell) orders PLUS pending settlement — the invariant compares
        against total ownership; an available-only read would false-fire
        the moment a sell order rests. Raise on read failure (transient
        errors skip the check cycle; at startup they fail closed).
        """
        ...


@dataclass(frozen=True)
class InventoryFold:
    """Result of folding the ledger into net inventory + cost basis.

    ``net_base`` is exact (pure decimal addition). ``cost_quote`` is the
    quote actually spent on the current inventory (fees included via the
    canonical deltas), reduced proportionally on partial sells — plain
    VWAP cost-basis, matching the engine's realized-P&L expectations.
    ``violation`` reports an oversell: the ledger's sells exceed its
    buys at some point, which spot cannot legitimately do — bookkeeping
    corruption, handled fail-closed by the caller.
    """
    net_base: Decimal = Decimal(0)
    cost_quote: Decimal = Decimal(0)
    fill_count: int = 0
    violation: str | None = None

    @property
    def vwap(self) -> Decimal | None:
        if self.net_base <= 0 or self.cost_quote <= 0:
            return None
        return self.cost_quote / self.net_base


def fold_inventory(rows: 'list[SpotExecutionRow]') -> InventoryFold:
    """Fold ledger rows (oldest first) into net inventory and cost basis.

    Buys add their net base delta and their absolute quote delta to the
    cost basis; sells reduce the basis proportionally to the fraction of
    inventory sold. An exact flat resets the basis to zero. A sell
    exceeding the running inventory marks the fold ``violation`` (the
    arithmetic still completes so the caller can report the terminal
    state).
    """
    inv = Decimal(0)
    cost = Decimal(0)
    violation: str | None = None
    for row in rows:
        base_delta = Decimal(row.base_delta)
        quote_delta = Decimal(row.quote_delta)
        if base_delta > 0:
            inv += base_delta
            cost += -quote_delta
        else:
            sold = -base_delta
            if sold > inv:
                if violation is None:
                    violation = (
                        f"sell {row.fill_id!r} of {sold} exceeds running "
                        f"inventory {inv}"
                    )
                cost = Decimal(0)
            elif sold == inv:
                cost = Decimal(0)
            else:
                cost -= cost * (sold / inv)
            inv -= sold
    return InventoryFold(
        net_base=inv,
        cost_quote=cost,
        fill_count=len(rows),
        violation=violation,
    )


@dataclass(frozen=True)
class SpotStartupResult:
    """Outcome of :meth:`SpotInventoryManager.startup`."""
    quarantined: bool
    reason: str | None
    fold: InventoryFold
    epoch: 'SpotEpochRow | None'
    recovered_fills: int = 0
    adopted_fills: int = 0


class SpotInventoryManager:
    """Core spot-inventory bookkeeping for one ``(account, product)``.

    A spot plugin constructs one per run after authentication and drives
    three touchpoints:

    - :meth:`startup` once, before the engine's startup reconcile —
      lease claim, execution catch-up, epoch/invariant validation,
      adoption watermark. Fail-closed: any inconclusive read quarantines
      before a single dispatch; a drift is left to :meth:`reconcile`.
    - :meth:`record_live_fill` for every fill observed on the live
      stream, BEFORE emitting the corresponding
      :class:`~pynecore.core.broker.models.OrderEvent` (the return value
      says whether to emit — a dedup'd replay must not re-book).
    - :meth:`reconcile` per poll cycle — lease heartbeat + balance
      invariant with the persisted settlement-grace state machine. It
      returns any fills a runtime catch-up recovered (a stream gap) so
      the plugin can emit their events; the periodic engine reconcile
      ignores position increases, so these would otherwise stay
      invisible until the next restart.

    ``get_position()`` synthesis reads :meth:`synthesize_position`.

    Quarantine delivery mirrors the disappearance tracker: the
    ``request_quarantine`` hook latches the engine (process stays alive,
    ingestion keeps running); the ``halt`` policy — or a missing /
    raising hook — arms :attr:`pending_halt`, which the plugin's event
    stream raises so the run exits via the graceful
    manual-intervention path. Never fail-open.
    """

    def __init__(
            self,
            store_ctx: 'RunContext',
            port: SpotInventoryPort,
            *,
            account_id: str,
            symbol: str,
            request_quarantine: 'Callable[..., bool | None] | None' = None,
            on_inventory_conflict: str = 'quarantine',
    ) -> None:
        """
        :param store_ctx: Open run context of the unified broker store.
        :param port: The plugin's venue surface.
        :param account_id: The plugin's authenticated ``account_id`` —
            the ledger's fill-id uniqueness dimension.
        :param symbol: The Pine symbol the synthesized
            :class:`~pynecore.core.broker.models.ExchangePosition`
            carries (not necessarily ``port.product_id``).
        :param request_quarantine: The engine quarantine latch, normally
            the plugin's
            :attr:`~pynecore.core.plugin.broker.BrokerPlugin.quarantine_sink`:
            called as ``(reason, context)``, plus
            ``kind=QuarantineKind.EXTERNAL_INTERVENTION`` when the bot's own
            inventory was taken.
        :param on_inventory_conflict: ``'quarantine'`` (default) or
            ``'halt'``.
        """
        if on_inventory_conflict not in INVENTORY_CONFLICT_POLICIES:
            raise ValueError(
                f"on_inventory_conflict must be one of "
                f"{INVENTORY_CONFLICT_POLICIES}, got "
                f"{on_inventory_conflict!r}"
            )
        if port.cursor_scope not in CURSOR_SCOPES:
            raise ValueError(
                f"port.cursor_scope must be one of {CURSOR_SCOPES}, got "
                f"{port.cursor_scope!r}"
            )
        if not isinstance(port.base_tolerance, Decimal) \
                or not port.base_tolerance.is_finite() \
                or port.base_tolerance < 0:
            raise ValueError(
                f"port.base_tolerance must be a finite non-negative "
                f"Decimal, got {port.base_tolerance!r}"
            )
        dust_threshold = port.position_dust_threshold
        if not isinstance(dust_threshold, Decimal) \
                or not dust_threshold.is_finite() \
                or dust_threshold < 0:
            raise ValueError(
                f"port.position_dust_threshold must be a finite non-negative "
                f"Decimal, got {dust_threshold!r}"
            )
        grace = port.settlement_grace_s
        if isinstance(grace, bool) \
                or not isinstance(grace, (int, float)) \
                or not math.isfinite(grace) \
                or grace < 0:
            # A NaN/inf grace makes the expiry comparison never true, so
            # a confirmed conflict would stay pending forever while
            # trading continues — fail closed at construction instead.
            raise ValueError(
                f"port.settlement_grace_s must be a finite non-negative "
                f"real number, got {grace!r}"
            )
        self._store = store_ctx
        self._port = port
        self._account_id = account_id
        self._symbol = symbol
        self._request_quarantine = request_quarantine
        self._policy = on_inventory_conflict
        self._epoch: 'SpotEpochRow | None' = None
        self._fold = InventoryFold()
        self._quarantined = False
        self._quarantine_reason: str | None = None
        self._pending_halt: SpotInventoryConflictError | None = None
        self._started = False
        self._conflict_hold_coids: frozenset[str] = frozenset()
        # The drift of the pending conflict as this process last measured it,
        # and when it first measured that drift: the in-flight hold runs from
        # there (:meth:`_drift_seen_since`).
        self._drift_seen: tuple[Decimal, int] | None = None
        # ``pending_conflict_ts_ms`` of the pending conflict whose verdict an
        # incomplete history left open was reported (:meth:`_report_unverified_drift`).
        self._unverified_reported_since: int | None = None
        # Ledger rows this manager inserted, live or recovered: a reconcile pass
        # compares it around its balance read (:meth:`reconcile`).
        self._rows_booked = 0

    # --- Introspection ------------------------------------------------------

    @property
    def quarantined(self) -> bool:
        return self._quarantined

    @property
    def quarantine_reason(self) -> str | None:
        return self._quarantine_reason

    @property
    def pending_halt(self) -> SpotInventoryConflictError | None:
        """Armed process-exit signal (``halt`` policy or hook fallback).

        The plugin's event stream checks this each pass and raises the
        taken error so the engine's
        :class:`~pynecore.core.broker.exceptions.BrokerManualInterventionError`
        handling performs the graceful stop.
        """
        return self._pending_halt

    def consume_pending_halt(self) -> SpotInventoryConflictError | None:
        """Take (and clear) the armed halt — consume-once."""
        halt = self._pending_halt
        self._pending_halt = None
        return halt

    @property
    def fold(self) -> InventoryFold:
        """The current in-memory inventory fold (refreshed on writes)."""
        return self._fold

    # --- Startup ------------------------------------------------------------

    async def startup(self) -> SpotStartupResult:
        """Run the fail-closed startup sequence.

        Order (plugin calls this after connect + auth, before the
        engine's startup reconcile):

        1. **Lease claim** — a live foreign lease on the base asset
           means another logical run owns it: quarantine, touch nothing.
        2. **Persisted quarantine recovery** — a ``quarantined`` or
           ``taken`` epoch survives crashes mid-run, but a fresh startup
           can re-measure the one truth it protected (the venue's
           balance): :meth:`_recover_quarantined_epoch` re-anchors a
           fresh epoch on that balance. Only a failed measurement keeps
           the quarantine. While the store holds an external-intervention
           quarantine no operator resumed, nothing is recovered: the
           ledger stays as it is and the manager stays quarantined.
        3. **Execution catch-up** from the epoch's durable cursor (the
           crash window between a venue fill and its local persist is
           closed here). Paged; each page commits its fills and the
           advanced cursor in one transaction. Inconclusive → fail
           closed. Settled against the balance read
           (:meth:`_measure_settled`): the history is read again after
           the balance, so an own fill landing between the two reads is
           in both. Reads the bot's own fills keep unsettling fail closed
           only on the first epoch, whose baseline is frozen on them; an
           existing epoch defers the drift they leave (step 5).
        4. **Epoch load / first-epoch freeze** — on the very first
           startup the baseline is frozen as
           ``current_total − reconstructed bot inventory`` (NOT the raw
           total: a crash after fills but before the first epoch write
           must not launder those fills into the baseline).
        5. **Balance invariant** — a drift is not judged here
           (:meth:`_defer_startup_drift`): the execution cursor stays where
           the catch-up began, and the runtime reconcile decides it after
           the settlement grace like any drift.
        6. **Adoption watermark** — every ledger row is folded into the
           position the engine adopts, so all rows flip ``delivered``;
           none may later re-enter as a live event.
        """
        if self._started:
            raise RuntimeError("SpotInventoryManager.startup() already ran")
        self._started = True
        port = self._port

        # (1) Ownership lease. A live foreign lease on the base asset —
        # or a base-vs-quote overlap with another live run trading the
        # shared asset as cash — fails the claim: quarantine, touch
        # nothing.
        if not self._store.claim_spot_asset(
                self._account_id, port.base_asset, port.quote_asset,
        ):
            self._enter_quarantine(
                'spot_lease_conflict',
                {
                    'account_id': self._account_id,
                    'base_asset': port.base_asset,
                    'quote_asset': port.quote_asset,
                },
            )
            return self._startup_result()

        # (2) Persisted quarantine from a previous run: recover. A spot
        # run's own book is rebuilt from scratch on every startup, so
        # the only truth a quarantined ledger protected is measurable
        # right here — the venue's actual balance. Re-anchor on that
        # measurement; stay quarantined only when the measurement itself
        # fails. An external-intervention quarantine the operator has not
        # resumed freezes the book as it is.
        epoch = self._store.get_latest_spot_epoch(port.product_id)
        self._epoch = epoch
        if epoch is not None and epoch.state in ('quarantined', 'taken'):
            if self._external_quarantine_held():
                self._hold_quarantined_epoch(epoch)
                return self._startup_result()
            epoch = await self._recover_quarantined_epoch(epoch)
            if epoch is None:
                return self._startup_result()
            self._epoch = epoch
        if epoch is not None and epoch.cursor_scope != port.cursor_scope:
            # A plugin upgrade changed what the persisted cursor means —
            # trusting it could silently skip history. Fail closed.
            self._refresh_fold()
            self._enter_quarantine(
                'spot_cursor_scope_changed',
                {
                    'product_id': port.product_id,
                    'stored_scope': epoch.cursor_scope,
                    'port_scope': port.cursor_scope,
                },
            )
            return self._startup_result()

        # (3) Execution catch-up from the durable cursor, settled against
        # the balance read it is compared with. Only the first epoch's
        # baseline is frozen on it; an existing epoch leaves any drift to the
        # runtime verdict, so a read the bot's own fills keep unsettling is
        # deferred like any drift. A reread window a re-anchor opened
        # (:meth:`_reanchor_foreign_holdings`) starts the read from the span
        # it holds; the runtime reconcile decides when the window ends.
        if epoch is not None and epoch.reread_until_ms is not None \
                and epoch.reread_cursor is not None \
                and epoch.exec_cursor != epoch.reread_cursor:
            self._store.set_spot_epoch_cursor(
                port.product_id, epoch.epoch_seq, epoch.reread_cursor,
            )
            epoch = self._store.get_latest_spot_epoch(port.product_id)
            assert epoch is not None
            self._epoch = epoch
        cursor_before = epoch.exec_cursor if epoch is not None else None
        try:
            measured = await self._measure_settled(
                epoch.exec_cursor if epoch is not None else None,
                _STARTUP_MEASURE_REASONS, {'product_id': port.product_id},
                must_settle=epoch is None,
            )
        except SpotInventoryConflictError:
            # _catch_up already quarantined (foreign ledger row).
            return self._startup_result()
        if epoch is not None:
            # The catch-up advanced the persisted cursor page by page —
            # refresh the in-memory snapshot to match.
            epoch = self._store.get_latest_spot_epoch(port.product_id)
            self._epoch = epoch
        if measured is None:
            return self._startup_result()
        balance = measured.balance
        recovered = measured.recovered
        final_cursor = measured.cursor

        # (4) Fold + first-epoch baseline freeze.
        self._refresh_fold()
        if self._fold.violation is not None:
            self._enter_quarantine(
                'spot_ledger_negative_inventory',
                {'product_id': port.product_id,
                 'violation': self._fold.violation},
            )
            return self._startup_result()

        if epoch is None:
            baseline = balance - self._fold.net_base
            if baseline < -port.base_tolerance:
                # The account owns less base than the ledger says the bot
                # holds — a foreign withdrawal or external sale in the
                # crash window before the first epoch write. Freezing a
                # negative foreign baseline would make the invariant hold
                # by construction and synthesize inventory the account
                # cannot sell. Fail closed instead.
                self._enter_quarantine(
                    'spot_baseline_below_inventory',
                    {
                        'product_id': port.product_id,
                        'current_total': canonical_decimal(balance),
                        'bot_inventory': canonical_decimal(self._fold.net_base),
                        'implied_baseline': canonical_decimal(baseline),
                    },
                )
                return self._startup_result()
            with self._store.transaction():
                epoch = self._store.insert_spot_epoch(
                    account_id=self._account_id,
                    base_asset=port.base_asset,
                    product_id=port.product_id,
                    foreign_baseline=canonical_decimal(baseline),
                    cursor_scope=port.cursor_scope,
                    exec_cursor=final_cursor,
                    state='active',
                )
                self._store.log_event(
                    'spot_epoch_created',
                    payload={
                        'product_id': port.product_id,
                        'epoch_seq': epoch.epoch_seq,
                        'foreign_baseline': epoch.foreign_baseline,
                        'bot_inventory': canonical_decimal(self._fold.net_base),
                        'current_total': canonical_decimal(balance),
                    },
                )
            self._epoch = epoch

        # (5) Balance invariant: a drift is left to the runtime verdict.
        drift = self._invariant_drift(balance)
        drift_pending = abs(drift) > port.base_tolerance
        if drift_pending:
            self._defer_startup_drift(balance, drift, cursor_before)
        else:
            self._hold_reread_cursor()

        # (6) Adoption watermark: the synthesized position the engine is
        # about to adopt folds every row, so none may be re-delivered.
        assert self._epoch is not None
        with self._store.transaction():
            adopted = self._store.mark_spot_executions_delivered(
                self._account_id, port.product_id,
            )
            if self._epoch.pending_conflict_ts_ms is not None and not drift_pending:
                # The invariant holds again — the persisted grace state
                # from the previous run resolved itself (settlement
                # landed while we were down).
                self._store.set_spot_epoch_pending_conflict(
                    port.product_id, self._epoch.epoch_seq,
                    ts_ms=None, payload=None,
                )
                self._epoch = self._store.get_latest_spot_epoch(port.product_id)
            self._store.log_event(
                'spot_startup_adopted',
                payload={
                    'product_id': port.product_id,
                    'net_base': canonical_decimal(self._fold.net_base),
                    'cost_quote': canonical_decimal(self._fold.cost_quote),
                    'fill_count': self._fold.fill_count,
                    'recovered_fills': recovered,
                    'adopted_fills': adopted,
                },
            )
        return self._startup_result(recovered=recovered, adopted=adopted)

    def _startup_result(
            self, *, recovered: int = 0, adopted: int = 0,
    ) -> SpotStartupResult:
        return SpotStartupResult(
            quarantined=self._quarantined,
            reason=self._quarantine_reason,
            fold=self._fold,
            epoch=self._epoch,
            recovered_fills=recovered,
            adopted_fills=adopted,
        )

    def _defer_startup_drift(
            self, balance: Decimal, drift: Decimal, cursor_before: str | None,
    ) -> None:
        """Leave a drift found at startup to the runtime reconcile's verdict.

        A drift after a conclusive catch-up has the same causes at startup as
        at runtime: someone else's activity on the account, judged against the
        bot's own inventory, or a fill of the bot's own the venue has put in
        its balance but not yet in its execution history (it indexes the
        history behind the balance; an order resting while the bot was down
        can fill just before the start). The startup cannot wait for that
        history, so it judges nothing: the bot adopts its ledger, the execution
        cursor stays where the startup catch-up began so the next read returns
        such a fill, and the first runtime reconcile arms the settlement grace
        (one an earlier process armed keeps its start; the hold on the bot's
        in-flight orders runs from this process's first sight of the drift)
        and decides as for any drift: the foreign holdings re-anchored when
        the balance still holds the bot's inventory, the external-intervention
        quarantine when a complete history proves it taken.

        :param balance: The settled startup balance.
        :param drift: Its distance from the expected total.
        :param cursor_before: The epoch's execution cursor before the startup catch-up.
        """
        port = self._port
        epoch = self._epoch
        assert epoch is not None
        if cursor_before is not None and epoch.exec_cursor != cursor_before:
            self._store.set_spot_epoch_cursor(port.product_id, epoch.epoch_seq, cursor_before)
            self._epoch = self._store.get_latest_spot_epoch(port.product_id)
        context = self._conflict_context(balance, drift)
        self._store.log_event('spot_startup_drift_pending', payload=context)
        logger.warning(
            "spot inventory: at startup the %s balance of %r is %s, the bot's own inventory %s, "
            "the expected total off by %s — not judged at startup: the runtime reconcile "
            "decides it after the settlement grace (%.1fs), trading continues meanwhile",
            port.base_asset, port.product_id, context['current_total'],
            context['bot_inventory'], context['drift'], port.settlement_grace_s,
        )

    async def _measure_settled(
            self, cursor: str | None, reasons: _MeasureReasons, context: dict[str, Any],
            *, must_settle: bool = True,
    ) -> _SettledMeasurement | None:
        """Catch the ledger up and read the balance it is compared with.

        A fill of the bot's own that lands between a history read and the
        balance read is in the balance but not in the ledger; whatever is
        computed from the pair then counts it twice once the history books it
        (an own buy frozen into the foreign baseline, an own sell counted as
        taken, a drift the runtime reconcile then has to judge). So
        the history is read again after every balance read, and the balance
        again whenever that read booked a fill, until a read after the balance
        books nothing. A fill the venue indexes only later than its balance is
        beyond any read here; the runtime reconcile's settlement grace covers
        that.

        Fail-closed: an inconclusive history read, a failed or invalid
        balance read, or fills still arriving after
        :data:`_MAX_SETTLE_BALANCE_READS` balance reads latch the entry block
        under the matching ``reasons`` entry. A measurement nothing is frozen
        on (``must_settle=False``: an existing epoch's plain startup, whose
        drift the runtime verdict judges) returns its last balance read
        instead of latching on fills still arriving: a burst of the bot's own
        fills is no bookkeeping trouble.

        :param cursor: The durable execution cursor to read from.
        :param reasons: The quarantine reasons of this measurement.
        :param context: The quarantine context every failure carries.
        :param must_settle: Whether a baseline or a verdict is computed from
            the measurement, so fills still arriving after the last balance
            read must latch the entry block.
        :return: The settled measurement, or ``None`` once quarantined.
        :raises SpotInventoryConflictError: After quarantining on a
            foreign-owned ledger row (:meth:`_catch_up`).
        """
        port = self._port
        recovered_total = 0
        balance: Decimal | None = None
        reads = 0
        while True:
            recovered, conclusive, cursor = await self._catch_up(cursor)
            recovered_total += recovered
            if not conclusive:
                self._refresh_fold()
                self._enter_quarantine(reasons.inconclusive, dict(context))
                return None
            if balance is not None and recovered == 0:
                return _SettledMeasurement(balance, cursor, recovered_total)
            if reads == _MAX_SETTLE_BALANCE_READS:
                break
            reads += 1
            # noinspection PyBroadException
            try:
                read = await port.fetch_base_balance()
            except Exception as exc:
                # Fail-closed boundary: WHATEVER went wrong with the read,
                # nothing may be anchored on an unchecked book.
                logger.exception(
                    "spot inventory: base-balance read failed for %r (%s)",
                    port.product_id, reasons.unavailable,
                )
                self._refresh_fold()
                self._enter_quarantine(
                    reasons.unavailable, {**context, 'error': repr(exc)},
                )
                return None
            if not isinstance(read, Decimal) or not read.is_finite():
                self._refresh_fold()
                self._enter_quarantine(
                    reasons.invalid, {**context, 'balance': repr(read)},
                )
                return None
            balance = read
        if not must_settle and balance is not None:
            logger.warning(
                "spot inventory: fills of the bot's own on %r kept arriving through %d "
                "balance reads (%d booked); measured on the last read, any drift is left "
                "to the runtime reconcile", port.product_id, _MAX_SETTLE_BALANCE_READS,
                recovered_total,
            )
            return _SettledMeasurement(balance, cursor, recovered_total)
        self._refresh_fold()
        self._enter_quarantine(
            reasons.unsettled,
            {**context, 'balance_reads': _MAX_SETTLE_BALANCE_READS,
             'recovered_fills': recovered_total},
        )
        return None

    async def _catch_up(
            self, cursor: str | None,
    ) -> tuple[int, bool, str | None]:
        """Drain the venue's execution history from ``cursor``.

        Each CONCLUSIVE page commits atomically: all of its fills plus
        the advanced cursor. A crash mid-pagination therefore resumes
        exactly at the last committed page; the overlap a time-scoped API
        re-serves is absorbed by the fill-id dedup.

        An INCONCLUSIVE page's fills are still recorded (the fill-id
        dedup makes a re-fetch safe), but its ``next_cursor`` is NOT
        persisted and the returned cursor stays at the last conclusive
        position — persisting a cursor past history the venue could not
        vouch for would let a later retry or recovery skip the
        uncertain range and launder the omitted fills into a baseline.

        :return: ``(recovered_count, conclusive, final_cursor)`` —
            ``final_cursor`` is the last CONCLUSIVE cursor.
        :raises SpotInventoryConflictError: After quarantining on a
            foreign-owned ledger row.
        """
        port = self._port
        recovered = 0
        current = cursor
        for _ in range(_MAX_CATCHUP_PAGES):
            batch = await port.fetch_executions(current)
            try:
                with self._store.transaction():
                    for execution in batch.executions:
                        if self._record_execution(execution, delivered=False):
                            recovered += 1
                    if batch.conclusive \
                            and batch.next_cursor is not None \
                            and self._epoch is not None:
                        self._store.set_spot_epoch_cursor(
                            port.product_id, self._epoch.epoch_seq,
                            batch.next_cursor,
                        )
            except _ForeignLedgerRow as foreign:
                # The page's transaction rolled back; quarantine OUTSIDE
                # the aborted span so its own store writes survive.
                raise self._quarantine_foreign_row(foreign) from None
            if not batch.conclusive:
                # ``current`` is still the last conclusive cursor — do not
                # advance past a page the venue could not vouch for.
                return recovered, False, current
            if batch.next_cursor is not None:
                current = batch.next_cursor
            if not batch.has_more:
                return recovered, True, current
        logger.error(
            "spot inventory: catch-up exceeded %d pages for %r; "
            "treating as inconclusive", _MAX_CATCHUP_PAGES, port.product_id,
        )
        return recovered, False, current

    def _record_execution(
            self, execution: SpotExecution, *, delivered: bool,
    ) -> bool:
        """Insert one fill inside the caller's transaction.

        :return: ``True`` when the row was inserted, ``False`` on a
            benign own-run dedup.
        :raises _ForeignLedgerRow: When the fill id is already booked
            under ANOTHER logical run — the exclusivity contract is
            broken and neither run's books can be trusted. The caller
            converts this to a quarantine OUTSIDE the rolled-back span.
        """
        port = self._port
        inserted = self._store.record_spot_execution(
            self._account_id, port.product_id,
            fill_id=execution.fill_id,
            side=execution.side,
            base_delta=canonical_decimal(execution.base_delta),
            quote_delta=canonical_decimal(execution.quote_delta),
            price=canonical_decimal(execution.price),
            fee_amount=canonical_decimal(execution.fee_amount),
            fee_currency=execution.fee_currency,
            ts_ms=execution.ts_ms,
            venue_seq=execution.venue_seq,
            exchange_order_id=execution.exchange_order_id,
            client_order_id=execution.client_order_id,
            delivered=delivered,
        )
        if inserted:
            self._rows_booked += 1
            return True
        # A missing owner row can only mean our own insert raced the
        # dedup read — treat it as the benign own-run case.
        owner = self._store.spot_execution_owner(
            self._account_id, port.product_id, execution.fill_id,
        ) or self._store.run_id
        if owner == self._store.run_id:
            return False
        raise _ForeignLedgerRow(execution.fill_id, owner)

    def _quarantine_foreign_row(
            self, foreign: _ForeignLedgerRow,
    ) -> SpotInventoryConflictError:
        """Quarantine on a foreign-owned ledger row; build the halt error."""
        self._enter_quarantine(
            'spot_foreign_ledger_row',
            {
                'product_id': self._port.product_id,
                'fill_id': foreign.fill_id,
                'owner_run_id': foreign.owner_run_id,
            },
        )
        return SpotInventoryConflictError(
            f"spot fill {foreign.fill_id!r} already booked under "
            f"another logical run {foreign.owner_run_id!r}",
            context={
                'fill_id': foreign.fill_id,
                'owner_run_id': foreign.owner_run_id,
            },
        )

    # --- Live fills ---------------------------------------------------------

    def record_live_fill(self, execution: SpotExecution) -> bool:
        """Record a fill observed on the live stream — outbox pattern.

        The ledger row is inserted with ``delivered=1`` in one
        transaction; the caller emits the corresponding
        :class:`~pynecore.core.broker.models.OrderEvent` ONLY when this
        returns ``True``. A crash after the commit but before the emit
        loses the event, not the fill: the next startup folds the row
        into the adopted position (exactly-once, adoption path).

        :return: ``True`` → new fill, emit the event; ``False`` → replay
            dedup, do NOT emit.
        :raises SpotInventoryConflictError: When the fill is booked
            under another logical run (after quarantining).
        """
        if not self._started:
            raise RuntimeError(
                "record_live_fill before startup(): the adoption "
                "watermark is not established yet"
            )
        try:
            with self._store.transaction():
                inserted = self._record_execution(execution, delivered=True)
        except _ForeignLedgerRow as foreign:
            raise self._quarantine_foreign_row(foreign) from None
        if inserted:
            self._refresh_fold()
            if self._fold.violation is not None and not self._quarantined:
                self._enter_quarantine(
                    'spot_ledger_negative_inventory',
                    {'product_id': self._port.product_id,
                     'violation': self._fold.violation},
                )
        return inserted

    # --- Periodic reconcile ---------------------------------------------------

    async def reconcile(self, now_ms: int) -> 'list[SpotExecutionRow]':
        """Per-poll invariant check + lease heartbeat.

        Transient venue read failures skip the cycle (a live bot must
        not halt on a recoverable read). A mismatch beyond the numeric
        tolerance arms the persisted settlement-grace state and triggers
        a fresh execution catch-up (late fills are the innocent
        explanation). A mismatch still unexplained past
        ``port.settlement_grace_s`` is held while the verdict turns on fills
        of the bot's own live orders (:meth:`_in_flight_orders_explaining`),
        for at most :data:`_HISTORY_LAG_CEILING_S` more, counted from when
        this process first saw the drift at its present value
        (:meth:`_drift_seen_since`), and is otherwise
        judged against the bot's own inventory once a complete history read
        found no fill explaining it: a balance that still holds it
        re-anchors the foreign baseline (:meth:`_reanchor_foreign_holdings`,
        ERROR + audit, no quarantine), a balance below it latches the
        external-intervention quarantine. A pass that booked a fill after its
        balance read judges on a balance read again after it
        (:meth:`_reread_balance_for_verdict`). With an incomplete history the
        conflict stays pending, reported once with an ERROR and an audit
        (:meth:`_report_unverified_drift`).

        The lease heartbeat is fenced by the physical instance: if a
        replacement instance took the lease over while this (now zombie)
        process was silent, the heartbeat reports the loss and this run
        quarantines instead of trading on a lease it no longer holds.

        :return: The ledger rows recovered by a runtime catch-up this
            cycle, freshly flipped to ``delivered``. The caller (plugin)
            MUST emit the corresponding
            :class:`~pynecore.core.broker.models.OrderEvent` for each so
            the sync engine's position tracks the recovered fills — the
            periodic engine reconcile ignores position increases, so a
            stream-gap fill recovered here would otherwise stay invisible
            to the strategy until the next restart's adoption. Marking
            them ``delivered`` before emission is crash-safe: a crash
            before the emit loses only the event, and the next startup
            re-folds the whole ledger into the adopted position. Empty on
            a clean cycle.
        """
        lease_held = self._store.heartbeat_spot_asset(
            self._account_id, self._port.base_asset,
        )
        if not self._quarantined and self._epoch is not None and not lease_held:
            self._enter_quarantine(
                'spot_lease_lost',
                {
                    'product_id': self._port.product_id,
                    'base_asset': self._port.base_asset,
                },
            )
            return []
        if self._quarantined or self._epoch is None:
            return []
        port = self._port
        rows_before_balance = self._rows_booked
        # noinspection PyBroadException
        try:
            balance = await port.fetch_base_balance()
        except Exception as exc:
            # Transient read failure: skip the cycle, retry on the next
            # poll — a live bot must not halt on a recoverable read. A
            # retryable transport fault is one line (its message already
            # names the endpoint and errno); anything else keeps its
            # traceback.
            logger.warning(
                "spot inventory: base-balance read failed for %r; "
                "skipping this reconcile cycle: %s", port.product_id, exc,
                exc_info=not is_retryable_provider_error(exc),
            )
            return []
        if not isinstance(balance, Decimal) or not balance.is_finite():
            logger.warning(
                "spot inventory: invalid base balance %r for %r; "
                "skipping this reconcile cycle", balance, port.product_id,
            )
            return []

        drift = self._invariant_drift(balance)
        epoch = self._epoch
        if abs(drift) <= port.base_tolerance and epoch.reread_until_ms is None:
            if epoch.pending_conflict_ts_ms is not None:
                self._store.set_spot_epoch_pending_conflict(
                    port.product_id, epoch.epoch_seq,
                    ts_ms=None, payload=None,
                )
                self._epoch = self._store.get_latest_spot_epoch(port.product_id)
                self._store.log_event(
                    'spot_inventory_conflict_resolved',
                    payload={'product_id': port.product_id},
                )
            self._conflict_hold_coids = frozenset()
            self._unverified_reported_since = None
            self._drift_seen = None
            await self._maybe_compact_dust()
            return []

        # Mismatch, or a clean pass over the span a re-anchor held. Try the
        # innocent explanation first: fills we have not seen yet
        # (settlement / stream gap).
        cursor_before = epoch.exec_cursor
        try:
            recovered, conclusive, _ = await self._catch_up(epoch.exec_cursor)
        except SpotInventoryConflictError:
            return []  # already quarantined
        delivered: list['SpotExecutionRow'] = []
        if recovered:
            self._refresh_fold()
            if self._fold.violation is not None:
                self._enter_quarantine(
                    'spot_ledger_negative_inventory',
                    {'product_id': port.product_id,
                     'violation': self._fold.violation},
                )
                return []
            # The recovered fills must reach the sync engine — hand them
            # to the caller for emission and flip their outbox marker.
            delivered = self._deliver_recovered()
            drift = self._invariant_drift(balance)
        if abs(drift) <= port.base_tolerance:
            epoch = self._store.get_latest_spot_epoch(port.product_id)
            assert epoch is not None
            was_pending = epoch.pending_conflict_ts_ms is not None
            if was_pending:
                self._store.set_spot_epoch_pending_conflict(
                    port.product_id, epoch.epoch_seq,
                    ts_ms=None, payload=None,
                )
            self._epoch = self._store.get_latest_spot_epoch(port.product_id)
            if recovered or was_pending:
                self._store.log_event(
                    'spot_inventory_conflict_resolved',
                    payload={'product_id': port.product_id,
                             'recovered_fills': recovered},
                )
            self._conflict_hold_coids = frozenset()
            self._unverified_reported_since = None
            self._drift_seen = None
            self._reread_held_span(now_ms)
            return delivered
        self._epoch = self._store.get_latest_spot_epoch(port.product_id)
        epoch = self._epoch
        assert epoch is not None
        if cursor_before is not None and epoch.exec_cursor != cursor_before:
            # The drift is still unexplained, so the history the catch-up
            # just drained is NOT complete: the venue indexes its
            # execution list behind its wallet balance (measured live,
            # Bybit spot: the balance showed a fill the list did not
            # return 61 s later). Keeping the advanced cursor would put
            # the next read's overlapped start past that fill and lose
            # it for good; hold the cursor where it was until the drift
            # resolves — re-reading the span is free (fill-id dedup).
            self._store.set_spot_epoch_cursor(
                port.product_id, epoch.epoch_seq, cursor_before,
            )
            self._epoch = self._store.get_latest_spot_epoch(port.product_id)
            epoch = self._epoch
            assert epoch is not None

        pending_since = epoch.pending_conflict_ts_ms
        context = self._conflict_context(balance, drift)
        self._drift_seen_since(drift, now_ms)
        if pending_since is None:
            # First observation: arm the persisted grace state. The
            # timestamp survives crashes, so a restart loop cannot keep
            # resetting the window.
            self._store.set_spot_epoch_pending_conflict(
                port.product_id, epoch.epoch_seq,
                ts_ms=now_ms, payload=context,
            )
            self._epoch = self._store.get_latest_spot_epoch(port.product_id)
            self._store.log_event(
                'spot_inventory_conflict_pending',
                payload=context,
            )
            logger.warning(
                "spot inventory: balance invariant mismatch for %r "
                "(drift=%s); settlement grace armed (%.1fs)",
                port.product_id, context['drift'], port.settlement_grace_s,
            )
            return delivered
        if not conclusive:
            # Cannot re-verify against an authoritative history read —
            # keep the grace armed, do not extend or shorten it.
            logger.warning(
                "spot inventory: conflict re-check inconclusive for %r; "
                "grace stays armed", port.product_id,
            )
        grace_ms = port.settlement_grace_s * 1000.0
        if now_ms - pending_since < grace_ms:
            return delivered
        if self._rows_booked != rows_before_balance:
            # A fill booked since the balance read may have landed on the venue
            # after it: the balance does not hold it while the fold does, so the
            # pair would judge the bot's own buy as base taken from it, or its
            # own sell as foreign base gone. Judge on a balance read the ledger
            # was caught up to instead.
            remeasured = await self._reread_balance_for_verdict(delivered)
            if remeasured is None:
                return delivered
            balance, conclusive_again = remeasured
            conclusive = conclusive and conclusive_again
            drift = self._invariant_drift(balance)
            context = self._conflict_context(balance, drift)
        explaining = self._in_flight_orders_explaining(balance, drift)
        if explaining:
            hold_since = max(pending_since, self._drift_seen_since(drift, now_ms))
            if now_ms - hold_since < grace_ms + _HISTORY_LAG_CEILING_S * 1000.0:
                # The verdict turns on fills of the bot's own live orders:
                # their fill may have reached the balance but not the
                # venue's execution history yet. That is latency, not a
                # foreign actor — keep the grace armed and keep draining
                # the history; the verdict follows the moment the rows
                # resolve without a fill or the drift outgrows them, and
                # at the latest once a fill of theirs would have reached
                # the history.
                if explaining != self._conflict_hold_coids:
                    self._conflict_hold_coids = explaining
                    self._store.log_event(
                        'spot_inventory_conflict_held',
                        payload={**context, 'client_order_ids': sorted(explaining)},
                    )
                    logger.warning(
                        "spot inventory: balance invariant mismatch for %r "
                        "(drift=%s) may be the bot's in-flight order(s) %s; "
                        "holding the verdict until their fills reach the "
                        "execution history",
                        port.product_id, context['drift'], sorted(explaining),
                    )
                return delivered
            # Resting orders cover the drift, but no fill of theirs reached
            # the history within the lag a fill of the bot's own can have:
            # they did not cause it (a stop leg rests for the whole life of
            # the position).
            context['in_flight_orders_without_fill'] = sorted(explaining)
        if balance >= self._fold.net_base - port.base_tolerance and conclusive:
            # The account still holds at least the bot's own inventory:
            # whatever moved (a deposit, a foreign buy, a sale out of the
            # foreign holdings) left the bot's share intact. Someone else's
            # separate activity, not an intervention.
            self._reanchor_foreign_holdings(balance, context, now_ms=now_ms)
        elif conclusive:
            # The account holds less base than the bot's own inventory,
            # and neither a complete execution history nor an own order
            # explains it: someone else sold or withdrew what the bot
            # holds.
            self._enter_quarantine(
                'spot_inventory_taken', context,
                kind=QuarantineKind.EXTERNAL_INTERVENTION,
            )
        else:
            self._report_unverified_drift(
                context, pending_since,
                intact=balance >= self._fold.net_base - port.base_tolerance,
            )
        return delivered

    async def _reread_balance_for_verdict(
            self, delivered: 'list[SpotExecutionRow]',
    ) -> tuple[Decimal, bool] | None:
        """Read the balance again for a verdict whose pass booked fills after its balance read.

        The history is drained again after the new read, from the cursor the
        pending conflict holds (kept there: the drift is still unexplained).
        The verdict waits for a later pass when that read fails, when the
        drain books a fill again (handed to the caller through ``delivered``)
        or when the new balance explains the drift: the next pass's clean
        read resolves the conflict.

        :param delivered: This pass's recovered rows; rows the drain recovers
            are appended.
        :return: ``(balance, conclusive)`` to judge on, or ``None`` when the
            verdict waits.
        """
        port = self._port
        rows_before = self._rows_booked
        # noinspection PyBroadException
        try:
            balance = await port.fetch_base_balance()
        except Exception as exc:
            logger.warning(
                "spot inventory: base-balance re-read for the conflict verdict on %r "
                "failed; the verdict waits for the next pass: %s", port.product_id, exc,
                exc_info=not is_retryable_provider_error(exc),
            )
            return None
        if not isinstance(balance, Decimal) or not balance.is_finite():
            return None
        epoch = self._epoch
        assert epoch is not None
        cursor = epoch.exec_cursor
        try:
            _, conclusive, _ = await self._catch_up(cursor)
        except SpotInventoryConflictError:
            return None  # already quarantined
        self._epoch = self._store.get_latest_spot_epoch(port.product_id)
        epoch = self._epoch
        assert epoch is not None
        if cursor is not None and epoch.exec_cursor != cursor:
            self._store.set_spot_epoch_cursor(port.product_id, epoch.epoch_seq, cursor)
            self._epoch = self._store.get_latest_spot_epoch(port.product_id)
        if self._rows_booked != rows_before:
            self._refresh_fold()
            if self._fold.violation is not None:
                self._enter_quarantine(
                    'spot_ledger_negative_inventory',
                    {'product_id': port.product_id, 'violation': self._fold.violation},
                )
                return None
            delivered.extend(self._deliver_recovered())
            return None
        if abs(self._invariant_drift(balance)) <= port.base_tolerance:
            return None
        return balance, conclusive

    def _reread_held_span(self, now_ms: int) -> None:
        """Keep the cursor a re-anchor held on a clean pass, until the venue indexed the span.

        A re-anchor keeps the execution cursor where the unexplained drift
        held it (:meth:`_reanchor_foreign_holdings`): a fill of the bot's own
        the venue indexes late may be in that span, absorbed into the foreign
        holdings by the re-anchor. A clean pass alone never reads the history,
        so while the span may still miss such a fill every pass reads it again
        from the held cursor: the late fill is booked, the drift it leaves
        re-anchors the holdings back, and the bot never sells base it no
        longer holds. Once a fill of the bot's own would have reached the
        history (:data:`_HISTORY_LAG_CEILING_S`), the cursor the last read
        reached is kept. The window is persisted with the epoch, so a restart
        inside it reads the same span (:meth:`startup`).

        :param now_ms: This pass's timestamp.
        """
        epoch = self._epoch
        assert epoch is not None
        until = epoch.reread_until_ms
        if until is None:
            return
        if now_ms >= until:
            self._store.clear_spot_epoch_reread(self._port.product_id, epoch.epoch_seq)
            self._epoch = self._store.get_latest_spot_epoch(self._port.product_id)
            return
        self._hold_reread_cursor()

    def _hold_reread_cursor(self) -> None:
        """Put the execution cursor back on the span an open reread window holds."""
        epoch = self._epoch
        assert epoch is not None
        if epoch.reread_until_ms is None or epoch.reread_cursor is None:
            return
        if epoch.exec_cursor != epoch.reread_cursor:
            self._store.set_spot_epoch_cursor(
                self._port.product_id, epoch.epoch_seq, epoch.reread_cursor,
            )
            self._epoch = self._store.get_latest_spot_epoch(self._port.product_id)

    def _reanchor_foreign_holdings(
            self, balance: Decimal, context: dict[str, Any], *, now_ms: int,
    ) -> None:
        """Re-anchor the foreign baseline on a balance that still holds the bot's inventory.

        Someone else's activity on the account changed the base balance
        without touching the bot's share of it, and a complete history read
        found no fill of the bot's own explaining it. The bot's ledger stays
        as it is; the epoch rolls (old ``closed``, fresh ``active``) with the
        foreign baseline ``balance - bot inventory`` and the execution cursor
        where the reconcile held it, and that span is read again until a fill
        of the bot's own the venue indexes later would have reached the
        history (:meth:`_reread_held_span`). An ERROR and a durable
        ``spot_foreign_holdings_reanchored`` audit record it; no quarantine.

        :param balance: The venue's measured base balance.
        :param context: The conflict context of this measurement.
        :param now_ms: This pass's timestamp.
        """
        port = self._port
        epoch = self._epoch
        assert epoch is not None
        baseline = max(Decimal(0), balance - self._fold.net_base)
        with self._store.transaction():
            self._store.set_spot_epoch_state(
                port.product_id, epoch.epoch_seq, 'closed',
            )
            fresh = self._store.insert_spot_epoch(
                account_id=self._account_id,
                base_asset=port.base_asset,
                product_id=port.product_id,
                foreign_baseline=canonical_decimal(baseline),
                cursor_scope=epoch.cursor_scope,
                exec_cursor=epoch.exec_cursor,
                state='active',
                reread_until_ms=now_ms + int(_HISTORY_LAG_CEILING_S * 1000),
            )
            self._store.log_event(
                'spot_foreign_holdings_reanchored',
                payload={
                    **context,
                    'old_epoch_seq': epoch.epoch_seq,
                    'epoch_seq': fresh.epoch_seq,
                    'old_foreign_baseline': epoch.foreign_baseline,
                    'foreign_baseline': fresh.foreign_baseline,
                },
            )
        self._epoch = fresh
        self._conflict_hold_coids = frozenset()
        self._unverified_reported_since = None
        self._drift_seen = None
        logger.error(
            "spot inventory: the %s balance of %r moved by %s without a fill of the "
            "bot's (now %s, the bot's own inventory %s is intact) — someone else's "
            "activity on the account; foreign holdings re-anchored %s -> %s, no "
            "quarantine, trading continues",
            port.base_asset, port.product_id, context['drift'],
            context['current_total'], context['bot_inventory'],
            epoch.foreign_baseline, fresh.foreign_baseline,
        )

    def _drift_seen_since(self, drift: Decimal, now_ms: int) -> int:
        """When this process first measured the pending conflict's current drift.

        The persisted ``pending_conflict_ts_ms`` dates the conflict's first
        observation, possibly by an earlier process, and a conflict an
        incomplete history keeps open stays pending however long it takes.
        A fill of the bot's own that lands later changes the drift and needs
        the same time to reach the venue's execution history as any other,
        so the in-flight hold is measured from when this drift was first
        seen: by this process, at the drift it has now.

        :param drift: The drift measured now.
        :param now_ms: The time of the measurement.
        :return: The time the drift was first measured at its present value.
        """
        seen = self._drift_seen
        if seen is None or abs(drift - seen[0]) > self._port.base_tolerance:
            self._drift_seen = (drift, now_ms)
            return now_ms
        return seen[1]

    def _report_unverified_drift(
            self, context: dict[str, Any], pending_since: int, *, intact: bool,
    ) -> None:
        """Report, once per pending conflict, a drift an incomplete history cannot judge.

        The execution history could not be read completely, so a fill of the
        bot's own cannot be ruled out. The conflict stays pending and every
        reconcile re-reads the history: a complete read books the missing
        fill (the drift resolves) or decides it as usual.

        Nothing is booked meanwhile, whichever side of the bot's inventory the
        balance is on. Below it (``spot_inventory_shortfall_unverified``),
        booking the shortfall and then the own fill the history returns later
        would take it from the bot's inventory twice; an own buy hidden the
        same way would, once booked, show a shortfall the complete history
        then quarantines as taken. At or above it
        (``spot_foreign_activity_unverified``), re-anchoring the foreign
        holdings would absorb an own sell the history has not returned: the
        bot would go on to sell base it no longer holds, out of someone
        else's holdings. The bot keeps trading on its ledger: a spot venue
        never sells base the account does not hold.

        :param context: The conflict context of this measurement.
        :param pending_since: When the pending conflict was armed.
        :param intact: Whether the balance still holds the bot's own inventory.
        """
        if self._unverified_reported_since == pending_since:
            return
        self._unverified_reported_since = pending_since
        port = self._port
        if intact:
            self._store.log_event('spot_foreign_activity_unverified', payload=context)
            logger.error(
                "spot inventory: the %s balance of %r is %s, the bot's own inventory %s, "
                "the expected total off by %s, and the execution history cannot be read "
                "completely — a fill of the bot's own may explain it; the foreign "
                "holdings are not re-anchored, the conflict stays pending until a "
                "complete history decides",
                port.base_asset, port.product_id, context['current_total'],
                context['bot_inventory'], context['drift'],
            )
            return
        self._store.log_event('spot_inventory_shortfall_unverified', payload=context)
        logger.error(
            "spot inventory: the account holds %s %s of %r, less than the bot's own "
            "inventory %s, and the execution history cannot be read completely — a "
            "fill of the bot's own may explain it; no quarantine, the conflict stays "
            "pending until a complete history decides",
            context['current_total'], port.base_asset, port.product_id,
            context['bot_inventory'],
        )

    def _in_flight_orders_explaining(
            self, balance: Decimal, drift: Decimal,
    ) -> frozenset[str]:
        """The bot's live orders on the drift's side whose fills the verdict turns on.

        A sell moves the base balance by exactly its quantity, a buy by the
        quantity minus any base-denominated fee, so the quantity of the live
        rows on the drift's side not yet filled (neither in the journal nor in
        the ledger) bounds what fills of theirs, not yet in the execution
        history, can explain. The verdict turns on them when they cover the
        whole drift, and when the balance is below the bot's own inventory
        but would still hold it had they filled: someone selling the
        account's other base beside an own sell is not the bot's inventory
        taken.

        :param balance: The venue's measured base balance.
        :param drift: Its distance from the expected total.
        :return: The client order ids of those rows; empty when the verdict
            does not turn on them.
        """
        if drift == 0:
            return frozenset()
        side = 'sell' if drift < 0 else 'buy'
        booked: dict[str, Decimal] = {}
        for execution in self._store.iter_spot_executions(
                self._account_id, self._port.product_id,
        ):
            if execution.client_order_id:
                booked[execution.client_order_id] = (
                    booked.get(execution.client_order_id, Decimal(0))
                    + abs(Decimal(execution.base_delta))
                )
        coids: set[str] = set()
        unfilled = Decimal(0)
        for row in self._store.iter_live_orders(symbol=self._symbol):
            if row.side != side:
                continue
            filled = max(Decimal(str(row.filled_qty)),
                         booked.get(row.client_order_id, Decimal(0)))
            left = Decimal(str(row.qty)) - filled
            if left > 0:
                coids.add(row.client_order_id)
                unfilled += left
        if unfilled <= 0:
            return frozenset()
        tolerance = self._port.base_tolerance
        if abs(drift) <= unfilled + tolerance:
            return frozenset(coids)
        inventory = self._fold.net_base
        if drift < 0 and inventory - unfilled - tolerance <= balance < inventory - tolerance:
            return frozenset(coids)
        return frozenset()

    async def _maybe_compact_dust(self) -> None:
        """Retire an unsellable-dust ledger into the epoch baseline.

        The venue charges the buy-side fee in the BASE coin while the
        engine's close orders are sized from FIFO exposure, so every
        buy→sell round trip strands a sub-step residue in the ledger.
        The residues accumulate: after a handful of round trips the fold
        of an engine-flat book sits several grid steps above zero, which
        silently breaks every "flat means fold ≈ 0" consumer — the
        plugin's entry-row flat sweep never fires again, ``get_position``
        reports a perpetual micro-long, and the next startup adopts the
        dust as a position (measured on the Bybit spot lane: 0.0001 ETH
        of pure fee dust after ~20 round trips, cycle 12 MISMATCH).

        Compaction runs only on a clean-invariant pass and only when the
        ENTIRE fold is provably unsellable (strictly below the port's
        :meth:`~SpotInventoryPort.min_sellable_base`): such a net cannot
        be a strategy position by definition. The epoch rolls exactly
        like the quarantine recovery — old epoch ``closed``, rows
        retired, fresh ``active`` epoch — except the new baseline is
        ``old_baseline + fold.net_base`` (NOT the venue balance): the
        expected total is unchanged by construction, so a real drift can
        never be absorbed by a compaction. The dust's residual cost
        basis is dropped with the rows — an unsellable residue has no
        realizable P&L.
        """
        epoch = self._epoch
        fold = self._fold
        if (self._quarantined or epoch is None
                or epoch.pending_conflict_ts_ms is not None
                or fold.violation is not None
                or fold.net_base <= 0):
            return
        port = self._port
        # noinspection PyBroadException
        try:
            floor = await port.min_sellable_base()
        except Exception:
            # Transient venue/read failure — compaction is opportunistic,
            # the next clean pass retries.
            logger.warning(
                "spot inventory: min-sellable read failed for %r; "
                "dust compaction skipped this pass", port.product_id,
                exc_info=True,
            )
            return
        if (not isinstance(floor, Decimal) or not floor.is_finite()
                or floor <= 0 or fold.net_base >= floor):
            return
        rows = self._store.iter_spot_executions(
            self._account_id, port.product_id,
        )
        if any(not row.delivered for row in rows):
            # An undelivered recovered fill still owes its OrderEvent to
            # the engine; retiring would flip it ``delivered`` and
            # swallow the event. Let delivery finish first.
            return
        new_baseline = Decimal(epoch.foreign_baseline) + fold.net_base
        with self._store.transaction():
            self._store.set_spot_epoch_state(
                port.product_id, epoch.epoch_seq, 'closed',
            )
            retired = self._store.retire_spot_executions(
                self._account_id, port.product_id,
            )
            fresh = self._store.insert_spot_epoch(
                account_id=self._account_id,
                base_asset=port.base_asset,
                product_id=port.product_id,
                foreign_baseline=canonical_decimal(new_baseline),
                cursor_scope=epoch.cursor_scope,
                exec_cursor=epoch.exec_cursor,
                state='active',
            )
            self._store.log_event(
                'spot_dust_compacted',
                payload={
                    'product_id': port.product_id,
                    'old_epoch_seq': epoch.epoch_seq,
                    'epoch_seq': fresh.epoch_seq,
                    'retired_fills': retired,
                    'dust_net_base': canonical_decimal(fold.net_base),
                    'min_sellable_base': canonical_decimal(floor),
                    'foreign_baseline': fresh.foreign_baseline,
                },
            )
        self._epoch = fresh
        self._refresh_fold()
        logger.info(
            "spot inventory: %s unsellable fee dust on %r compacted into "
            "the epoch baseline (epoch %d -> %d, %d ledger row(s) retired)",
            canonical_decimal(fold.net_base), port.product_id,
            epoch.epoch_seq, fresh.epoch_seq, retired,
        )

    def _invariant_drift(self, balance: Decimal) -> Decimal:
        """``actual − (foreign_baseline + bot_inventory)``, exact."""
        assert self._epoch is not None
        baseline = Decimal(self._epoch.foreign_baseline)
        return balance - (baseline + self._fold.net_base)

    def _conflict_context(
            self, balance: Decimal, drift: Decimal,
    ) -> dict[str, Any]:
        assert self._epoch is not None
        return {
            'product_id': self._port.product_id,
            'base_asset': self._port.base_asset,
            'epoch_seq': self._epoch.epoch_seq,
            'foreign_baseline': self._epoch.foreign_baseline,
            'bot_inventory': canonical_decimal(self._fold.net_base),
            'current_total': canonical_decimal(balance),
            'drift': canonical_decimal(drift),
            'tolerance': canonical_decimal(self._port.base_tolerance),
        }

    # --- Position synthesis ---------------------------------------------------

    def synthesize_position(self, mark: float) -> ExchangePosition | None:
        """Build the :class:`ExchangePosition` the plugin's
        ``get_position()`` returns.

        ``None`` only when the bot's net inventory is genuinely flat —
        the engine reads ``None`` as an authoritative flat, so a spot
        plugin must never return it merely because the venue has no
        position object.
        """
        fold = self._fold
        if fold.net_base <= 0:
            return None
        vwap = fold.vwap
        entry_price = float(vwap) if vwap is not None else 0.0
        size = float(fold.net_base)
        unrealized = (mark - entry_price) * size if vwap is not None else 0.0
        return ExchangePosition(
            symbol=self._symbol,
            side='long',
            size=size,
            entry_price=entry_price,
            unrealized_pnl=unrealized,
            liquidation_price=None,
            leverage=1.0,
            margin_mode='cash',
        )

    # --- Rebaseline (operator recovery) ---------------------------------------

    def _external_quarantine_held(self) -> bool:
        """Whether the run holds an external-intervention quarantine no operator resumed.

        The engine restores that latch from the store before the plugin
        connects; an open latch with a resume request is being resumed by
        this process.
        """
        return any(
            row.kind == QuarantineKind.EXTERNAL_INTERVENTION
            and row.resume_requested_ts_ms is None
            for row in self._store.iter_open_quarantines()
        )

    def _hold_quarantined_epoch(self, epoch: 'SpotEpochRow') -> None:
        """Keep a quarantined epoch and its ledger as they are, under the held latch.

        The engine's restored external-intervention latch already stops
        every venue write. The book stays exactly as the bot left it, so the
        start after the operator resume measures the venue against the
        bot's own inventory, not against a baseline re-anchored meanwhile.
        """
        self._refresh_fold()
        self._quarantined = True
        self._quarantine_reason = 'spot_external_quarantine_held'
        logger.warning(
            "spot inventory: %s epoch %d for %r is kept as it is (bot inventory %s) "
            "while the external-intervention quarantine holds; it is recovered at "
            "the first start after the operator resume",
            epoch.state, epoch.epoch_seq, self._port.product_id,
            canonical_decimal(self._fold.net_base),
        )

    async def _recover_quarantined_epoch(
            self, epoch: 'SpotEpochRow',
    ) -> 'SpotEpochRow | None':
        """Re-anchor a quarantined product on the venue's actual balance.

        Whatever poisoned the old epoch — a negative fold, drift, a
        crash mid-conflict — the account's spot wallet is the ground
        truth, and a fresh startup can measure it before a single
        dispatch goes out. Sequence (fail-closed at every step):

        1. execution catch-up from the old epoch's cursor (a cursor
           whose scope no longer matches the port restarts from the
           beginning — the fill-id dedup absorbs the replay), settled
           against a fresh balance read (:meth:`_measure_settled`);
        2. ONE transaction: the old epoch flips ``closed``, every live
           ledger row retires (audit kept, fold excluded, adoption
           suppressed via ``delivered``), and a fresh ``active`` epoch
           freezes ``foreign_baseline = balance`` — the bot owns
           nothing, the account owns everything.

        A ``taken`` epoch (someone else sold or withdrew the bot's own
        inventory, and the operator resumed the run) keeps its ledger
        instead: the bot still owns what the account holds of its
        inventory, ``min(bot inventory, balance)``. The part the account no
        longer holds leaves the fold through one ``external-taken`` ledger
        row (a sell at the cost basis, so the remainder keeps its average
        price), and the fresh epoch freezes the rest of the balance as the
        foreign baseline. A ledger that is corrupt itself (a negative fold)
        retires as above.

        A fill of the bot's own that lands between a history read and the
        balance read is booked before anything is computed: the settled
        measurement reads the history again after the balance. A fill that
        lands after the balance read re-enters as a live event on the clean
        ledger.

        :return: The fresh active epoch, or ``None`` when a measurement
            failed and the quarantine latched instead.
        """
        port = self._port
        cursor = (
            epoch.exec_cursor
            if epoch.cursor_scope == port.cursor_scope else None
        )
        try:
            measured = await self._measure_settled(
                cursor, _RECOVERY_MEASURE_REASONS,
                {'product_id': port.product_id, 'epoch_seq': epoch.epoch_seq},
            )
        except SpotInventoryConflictError:
            # _catch_up already quarantined (foreign ledger row).
            return None
        if measured is None:
            return None
        balance = measured.balance
        recovered = measured.recovered
        final_cursor = measured.cursor
        self._refresh_fold()
        retired_fold = self._fold
        if epoch.state == 'taken' and retired_fold.violation is None:
            return self._recover_taken_epoch(
                epoch, balance, final_cursor=final_cursor, recovered=recovered,
            )
        with self._store.transaction():
            self._store.set_spot_epoch_state(
                port.product_id, epoch.epoch_seq, 'closed',
            )
            retired = self._store.retire_spot_executions(
                self._account_id, port.product_id,
            )
            fresh = self._store.insert_spot_epoch(
                account_id=self._account_id,
                base_asset=port.base_asset,
                product_id=port.product_id,
                foreign_baseline=canonical_decimal(balance),
                cursor_scope=port.cursor_scope,
                exec_cursor=final_cursor,
                state='active',
            )
            self._store.log_event(
                'spot_epoch_recovered',
                payload={
                    'product_id': port.product_id,
                    'old_epoch_seq': epoch.epoch_seq,
                    'old_pending_conflict': epoch.pending_conflict,
                    'epoch_seq': fresh.epoch_seq,
                    'foreign_baseline': fresh.foreign_baseline,
                    'retired_fills': retired,
                    'retired_net_base': canonical_decimal(
                        retired_fold.net_base),
                    'retired_violation': retired_fold.violation,
                    'recovered_fills': recovered,
                },
            )
        logger.warning(
            "spot inventory: quarantined epoch %d for %r recovered at "
            "startup — %d ledger row(s) retired (net %s, violation: %s), "
            "fresh baseline anchored on the venue balance %s",
            epoch.epoch_seq, port.product_id, retired,
            canonical_decimal(retired_fold.net_base),
            retired_fold.violation, canonical_decimal(balance),
        )
        self._refresh_fold()
        return fresh

    def _recover_taken_epoch(
            self, epoch: 'SpotEpochRow', balance: Decimal, *,
            final_cursor: str | None, recovered: int,
    ) -> 'SpotEpochRow':
        """Keep the part of the bot's inventory the account still holds.

        Someone else sold or withdrew the bot's own inventory and the
        operator resumed the run: the bot still owns its inventory less what
        was taken when the conflict latched (the bot's inventory over the
        balance the latch measured, :meth:`_taken_at_latch`), and never more
        than the balance holds now. Base deposited during the quarantine is
        not the bot's; its own fills since the latch are. The rest leaves the
        fold through one
        ``external-taken`` ledger row: a sell at the cost basis, so what is
        kept keeps its average price. It is written delivered (the engine
        adopts the folded position at startup and gets no event for it) and
        sorts after every ledger row. In ONE transaction the old epoch flips
        ``closed`` and a fresh ``active`` epoch freezes the rest of the
        balance, ``balance - kept``, as the foreign baseline, so the
        invariant holds by construction.

        ``balance`` is settled against the ledger (:meth:`_measure_settled`):
        an own order that filled while the history and the balance were read
        is already in the fold here, never counted as kept (an own buy) or as
        taken (an own sell) and then booked a second time.

        :param epoch: The ``taken`` epoch.
        :param balance: The venue's measured base balance.
        :param final_cursor: The execution cursor the catch-up reached.
        :param recovered: Fills the recovery's catch-up booked, for the audit.
        :return: The fresh active epoch.
        """
        port = self._port
        fold = self._fold
        kept = max(Decimal(0), min(fold.net_base - self._taken_at_latch(epoch), balance))
        taken = fold.net_base - kept
        rows = self._store.iter_spot_executions(self._account_id, port.product_id)
        with self._store.transaction():
            self._store.set_spot_epoch_state(
                port.product_id, epoch.epoch_seq, 'closed',
            )
            if taken > 0:
                cost_basis = fold.vwap or Decimal(0)
                self._store.record_spot_execution(
                    self._account_id, port.product_id,
                    fill_id=f"external-taken:{epoch.epoch_seq}",
                    side='sell',
                    base_delta=canonical_decimal(-taken),
                    quote_delta=canonical_decimal(fold.cost_quote * taken / fold.net_base),
                    price=canonical_decimal(cost_basis),
                    fee_amount='0',
                    fee_currency=port.quote_asset,
                    ts_ms=rows[-1].ts_ms + 1 if rows else 1,
                    delivered=True,
                )
            fresh = self._store.insert_spot_epoch(
                account_id=self._account_id,
                base_asset=port.base_asset,
                product_id=port.product_id,
                foreign_baseline=canonical_decimal(balance - kept),
                cursor_scope=port.cursor_scope,
                exec_cursor=final_cursor,
                state='active',
            )
            self._store.log_event(
                'spot_epoch_taken_recovered',
                payload={
                    'product_id': port.product_id,
                    'old_epoch_seq': epoch.epoch_seq,
                    'old_pending_conflict': epoch.pending_conflict,
                    'epoch_seq': fresh.epoch_seq,
                    'foreign_baseline': fresh.foreign_baseline,
                    'bot_inventory_before': canonical_decimal(fold.net_base),
                    'kept': canonical_decimal(kept),
                    'taken': canonical_decimal(taken),
                    'current_total': canonical_decimal(balance),
                    'recovered_fills': recovered,
                },
            )
        logger.warning(
            "spot inventory: taken epoch %d for %r recovered after the operator "
            "resume — the bot keeps %s of its %s (the account holds %s); %s left "
            "the bot's inventory, fresh baseline %s",
            epoch.epoch_seq, port.product_id, canonical_decimal(kept),
            canonical_decimal(fold.net_base), canonical_decimal(balance),
            canonical_decimal(taken), fresh.foreign_baseline,
        )
        self._refresh_fold()
        return fresh

    def _taken_at_latch(self, epoch: 'SpotEpochRow') -> Decimal:
        """The bot's inventory the account no longer held when ``epoch`` latched ``taken``.

        The latch recorded its measurement on the epoch (``bot_inventory`` and
        ``current_total``). Without a readable one only the balance at the
        recovery bounds what the bot keeps.

        :return: ``bot inventory - balance`` at the latch, at least ``0``.
        """
        conflict = epoch.pending_conflict or {}
        try:
            inventory = Decimal(str(conflict['bot_inventory']))
            total = Decimal(str(conflict['current_total']))
        except (KeyError, ArithmeticError, ValueError, TypeError):
            logger.warning(
                "spot inventory: taken epoch %d for %r carries no readable latch "
                "measurement (%r); the balance alone bounds what the bot keeps",
                epoch.epoch_seq, self._port.product_id, conflict,
            )
            return Decimal(0)
        if not (inventory.is_finite() and total.is_finite()):
            return Decimal(0)
        return max(Decimal(0), inventory - total)

    # --- Teardown -------------------------------------------------------------

    def close(self) -> None:
        """Release the asset lease on a clean shutdown."""
        self._store.release_spot_asset(
            self._account_id, self._port.base_asset,
        )

    # --- Internals --------------------------------------------------------------

    def _refresh_fold(self) -> None:
        rows = self._store.iter_spot_executions(
            self._account_id, self._port.product_id,
        )
        self._fold = fold_inventory(rows)

    def _deliver_recovered(self) -> 'list[SpotExecutionRow]':
        """Flip and return the undelivered rows a runtime catch-up left.

        Startup adoption and :meth:`record_live_fill` both leave the
        ledger fully delivered, so undelivered rows are exactly the
        catch-up recoveries of the current cycle. Flipping them here
        (before the caller emits) mirrors :meth:`record_live_fill`'s
        insert-delivered-then-emit ordering; the whole-ledger re-fold on
        the next startup is the crash-safety net for a lost emit.
        """
        rows = self._store.iter_spot_executions(
            self._account_id, self._port.product_id,
            undelivered_only=True,
        )
        if rows:
            self._store.mark_spot_executions_delivered(
                self._account_id, self._port.product_id,
                [row.fill_id for row in rows],
            )
        return rows

    def _enter_quarantine(
            self, reason: str, context: dict[str, Any], *,
            kind: QuarantineKind = QuarantineKind.ENTRY_BLOCK,
    ) -> None:
        """Latch the conflict — dual signal, mirroring the tracker.

        Persists the epoch's ``quarantined`` state (``taken`` for an
        external intervention; when an epoch exists, and never over a
        ``taken`` one), then delivers per policy: the ``request_quarantine``
        hook latches the engine and the process stays alive; ``halt`` —
        or a missing / raising hook — arms :attr:`pending_halt`. Never
        fail-open.

        :param kind: The quarantine the hook latches. ``ENTRY_BLOCK`` is the
            manager's own bookkeeping trouble; ``EXTERNAL_INTERVENTION``
            (someone took the bot's inventory) freezes every venue write
            until the operator resumes the run.
        """
        if self._quarantined:
            return
        self._quarantined = True
        self._quarantine_reason = reason
        epoch = self._epoch
        state = 'taken' if kind is QuarantineKind.EXTERNAL_INTERVENTION else 'quarantined'
        with self._store.transaction():
            # A ``taken`` epoch stays taken until its recovery: a measurement
            # failing on the way (a later ``quarantined`` latch) must not make
            # the next start retire the inventory the account still holds.
            if epoch is not None and epoch.state not in (state, 'taken'):
                self._store.set_spot_epoch_state(
                    self._port.product_id, epoch.epoch_seq, state,
                )
                if state == 'taken' and epoch.pending_conflict_ts_ms is not None:
                    # The measurement the latch judged: the recovery after the
                    # resume keeps only what the account held of the bot's
                    # inventory then (:meth:`_recover_taken_epoch`).
                    self._store.set_spot_epoch_pending_conflict(
                        self._port.product_id, epoch.epoch_seq,
                        ts_ms=epoch.pending_conflict_ts_ms, payload=context,
                    )
                self._epoch = self._store.get_latest_spot_epoch(
                    self._port.product_id,
                )
            self._store.log_event(
                'spot_inventory_quarantine',
                payload={'reason': reason, 'kind': kind.value, **context},
            )
        if kind is QuarantineKind.EXTERNAL_INTERVENTION:
            message = (
                f"spot inventory taken ({reason}): the account holds less "
                f"base than the bot's own inventory; every venue write stops "
                f"until the operator resumes the run"
            )
        else:
            message = (
                f"spot inventory conflict ({reason}): trading stops; "
                f"restart the run to recover on a fresh venue-balance baseline"
            )
        quarantined = False
        if self._policy == 'quarantine' and self._request_quarantine is not None:
            # noinspection PyBroadException
            try:
                if kind is QuarantineKind.ENTRY_BLOCK:
                    self._request_quarantine(message, dict(context))
                else:
                    self._request_quarantine(message, dict(context), kind=kind)
            except Exception:
                logger.exception(
                    "request_quarantine hook failed for %r; falling back "
                    "to the process-exiting halt", reason,
                )
            else:
                quarantined = True
        if not quarantined:
            # 'halt' policy, or a quarantining policy whose hook is
            # missing / raised: arm the process-exiting signal.
            self._pending_halt = SpotInventoryConflictError(
                message, context=dict(context),
            )
        logger.error(
            "spot inventory QUARANTINE (%s): %s", reason, context,
        )
