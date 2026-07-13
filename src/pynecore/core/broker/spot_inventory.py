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
  ``foreign_baseline`` was frozen at epoch creation. Any unexplainable
  drift in EITHER direction is an attribution conflict — a positive
  drift can mask an external sale netted against a deposit, so
  warn-and-continue is not an option (fail-closed both ways).
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
  inventory is unsupported by contract. A confirmed conflict stops
  trading via the engine quarantine latch (process stays alive as an
  observer); the explicit ``halt`` policy exits instead. Recovery is an
  operator ``rebaseline`` (new epoch, one transaction) plus restart.

The manager takes explicit ``now_ms`` timestamps so reconciliation
timing is fully deterministic under test, mirroring
:class:`~pynecore.core.broker.disappearance.DisappearanceTracker`.
"""
import logging
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any, Final, Protocol

from pynecore.core.broker.exceptions import SpotInventoryConflictError
from pynecore.core.broker.models import ExchangePosition
from pynecore.core.broker.store_helpers import PENDING_DISPATCH_STATES

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
      adoption watermark. Fail-closed: any inconclusive read or
      unexplainable drift quarantines before a single dispatch.
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
            request_quarantine: 'Callable[[str, dict[str, Any]], None] | None' = None,
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
            :attr:`~pynecore.core.plugin.broker.BrokerPlugin.quarantine_sink`.
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
        2. **Persisted quarantine check** — a ``quarantined`` epoch
           survives restarts; only an operator :meth:`rebaseline`
           clears it.
        3. **Execution catch-up** from the epoch's durable cursor (the
           crash window between a venue fill and its local persist is
           closed here). Paged; each page commits its fills and the
           advanced cursor in one transaction. Inconclusive → fail
           closed.
        4. **Epoch load / first-epoch freeze** — on the very first
           startup the baseline is frozen as
           ``current_total − reconstructed bot inventory`` (NOT the raw
           total: a crash after fills but before the first epoch write
           must not launder those fills into the baseline).
        5. **Balance invariant** — strict (the catch-up was conclusive,
           so no settlement grace applies at startup).
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

        # (2) Persisted quarantine from a previous run.
        epoch = self._store.get_latest_spot_epoch(port.product_id)
        self._epoch = epoch
        if epoch is not None and epoch.state == 'quarantined':
            self._refresh_fold()
            self._enter_quarantine(
                'spot_epoch_quarantined',
                {
                    'product_id': port.product_id,
                    'epoch_seq': epoch.epoch_seq,
                    'pending_conflict': epoch.pending_conflict,
                },
            )
            return self._startup_result()
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

        # (3) Execution catch-up from the durable cursor.
        try:
            recovered, conclusive, final_cursor = await self._catch_up(
                epoch.exec_cursor if epoch is not None else None,
            )
        except SpotInventoryConflictError:
            # _catch_up already quarantined (foreign ledger row).
            return self._startup_result()
        if epoch is not None:
            # The catch-up advanced the persisted cursor page by page —
            # refresh the in-memory snapshot to match.
            epoch = self._store.get_latest_spot_epoch(port.product_id)
            self._epoch = epoch
        if not conclusive:
            self._refresh_fold()
            self._enter_quarantine(
                'spot_catchup_inconclusive',
                {'product_id': port.product_id},
            )
            return self._startup_result()

        # (4) Fold + first-epoch baseline freeze / balance read.
        self._refresh_fold()
        if self._fold.violation is not None:
            self._enter_quarantine(
                'spot_ledger_negative_inventory',
                {'product_id': port.product_id,
                 'violation': self._fold.violation},
            )
            return self._startup_result()
        # noinspection PyBroadException
        try:
            balance = await port.fetch_base_balance()
        except Exception as exc:
            # Fail-closed boundary: WHATEVER went wrong with the read,
            # startup must not proceed to trading on an unchecked book.
            logger.exception(
                "spot inventory: startup base-balance read failed for %r",
                port.product_id,
            )
            self._enter_quarantine(
                'spot_startup_balance_unavailable',
                {'product_id': port.product_id, 'error': repr(exc)},
            )
            return self._startup_result()
        if not isinstance(balance, Decimal) or not balance.is_finite():
            self._enter_quarantine(
                'spot_startup_balance_invalid',
                {'product_id': port.product_id, 'balance': repr(balance)},
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

        # (5) Strict startup invariant.
        drift = self._invariant_drift(balance)
        if abs(drift) > port.base_tolerance:
            self._enter_quarantine(
                'spot_inventory_conflict',
                self._conflict_context(balance, drift),
            )
            return self._startup_result()

        # (6) Adoption watermark: the synthesized position the engine is
        # about to adopt folds every row, so none may be re-delivered.
        assert self._epoch is not None
        with self._store.transaction():
            adopted = self._store.mark_spot_executions_delivered(
                self._account_id, port.product_id,
            )
            if self._epoch.pending_conflict_ts_ms is not None:
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
        vouch for would let a later retry or rebaseline skip the
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
        explanation); a mismatch still unexplained past
        ``port.settlement_grace_s`` is a confirmed attribution conflict.

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
        # noinspection PyBroadException
        try:
            balance = await port.fetch_base_balance()
        except Exception:
            # Transient read failure: skip the cycle, retry on the next
            # poll — a live bot must not halt on a recoverable read.
            logger.warning(
                "spot inventory: base-balance read failed for %r; "
                "skipping this reconcile cycle", port.product_id,
                exc_info=True,
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
        if abs(drift) <= port.base_tolerance:
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
            return []

        # Mismatch. Try the innocent explanation first: fills we have
        # not seen yet (settlement / stream gap).
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
                if epoch.pending_conflict_ts_ms is not None:
                    self._store.set_spot_epoch_pending_conflict(
                        port.product_id, epoch.epoch_seq,
                        ts_ms=None, payload=None,
                    )
                self._epoch = self._store.get_latest_spot_epoch(port.product_id)
                self._store.log_event(
                    'spot_inventory_conflict_resolved',
                    payload={'product_id': port.product_id,
                             'recovered_fills': recovered},
                )
                return delivered
        self._epoch = self._store.get_latest_spot_epoch(port.product_id)
        epoch = self._epoch
        assert epoch is not None

        pending_since = epoch.pending_conflict_ts_ms
        context = self._conflict_context(balance, drift)
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
        if now_ms - pending_since >= port.settlement_grace_s * 1000.0:
            self._enter_quarantine('spot_inventory_conflict', context)
        return delivered

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

    async def rebaseline(self) -> 'SpotEpochRow':
        """Freeze a new baseline epoch after an operator intervened.

        Preconditions (all fail-closed, raising ``ValueError``):

        - dispatch is frozen — the run is quarantined (rebaselining a
          live-trading run would launder in-flight drift);
        - no unresolved dispatches: no parked verifications and no live
          order rows in a pending dispatch state (their eventual fills
          would land on the wrong side of the new baseline);
        - a FRESH conclusive execution catch-up and balance read succeed
          right here.

        The new epoch (bumped ``epoch_seq``, recomputed
        ``foreign_baseline``, ``state='active'``) and the old epoch's
        ``closed`` flip commit in ONE transaction. The engine's
        quarantine latch is in-memory by design — the operator restarts
        the run after a successful rebaseline; the fresh startup then
        finds the active epoch and trades again.

        :return: The freshly activated epoch row.
        """
        if not self._quarantined:
            raise ValueError(
                "rebaseline requires the run to be quarantined — "
                "dispatch must be frozen while the baseline moves"
            )
        port = self._port
        unresolved = self._unresolved_dispatches()
        if unresolved:
            raise ValueError(
                f"rebaseline blocked: unresolved dispatches remain "
                f"({', '.join(unresolved)}) — their eventual fills would "
                f"land on the wrong side of the new baseline"
            )
        old_epoch = self._store.get_latest_spot_epoch(port.product_id)
        recovered, conclusive, final_cursor = await self._catch_up(
            old_epoch.exec_cursor if old_epoch is not None else None,
        )
        if not conclusive:
            raise ValueError(
                "rebaseline blocked: execution catch-up was inconclusive"
            )
        self._refresh_fold()
        if self._fold.violation is not None:
            raise ValueError(
                f"rebaseline blocked: ledger corrupt "
                f"({self._fold.violation})"
            )
        balance = await port.fetch_base_balance()
        if not isinstance(balance, Decimal) or not balance.is_finite():
            raise ValueError(
                f"rebaseline blocked: invalid base balance {balance!r}"
            )
        baseline = balance - self._fold.net_base
        if baseline < -port.base_tolerance:
            # A negative foreign baseline would launder exactly the
            # withdrawal / external-sale conflict that forced the
            # quarantine: it would make the invariant hold while the
            # synthesized position exceeds what the account owns. Refuse
            # until the operator restores enough base holdings.
            raise ValueError(
                f"rebaseline blocked: base balance {balance} is below the "
                f"ledger's bot inventory {self._fold.net_base} — restore "
                f"the missing base holdings before rebaselining"
            )
        with self._store.transaction():
            if old_epoch is not None:
                self._store.set_spot_epoch_state(
                    port.product_id, old_epoch.epoch_seq, 'closed',
                )
            epoch = self._store.insert_spot_epoch(
                account_id=self._account_id,
                base_asset=port.base_asset,
                product_id=port.product_id,
                foreign_baseline=canonical_decimal(baseline),
                cursor_scope=port.cursor_scope,
                exec_cursor=final_cursor,
                state='active',
            )
            self._store.mark_spot_executions_delivered(
                self._account_id, port.product_id,
            )
            self._store.log_event(
                'spot_epoch_rebaselined',
                payload={
                    'product_id': port.product_id,
                    'old_epoch_seq': (
                        None if old_epoch is None else old_epoch.epoch_seq
                    ),
                    'epoch_seq': epoch.epoch_seq,
                    'foreign_baseline': epoch.foreign_baseline,
                    'bot_inventory': canonical_decimal(self._fold.net_base),
                    'current_total': canonical_decimal(balance),
                    'recovered_fills': recovered,
                },
            )
        self._epoch = epoch
        return epoch

    def _unresolved_dispatches(self) -> list[str]:
        """Names of dispatches whose outcome is still in flight."""
        unresolved: list[str] = []
        _envelopes, pending = self._store.replay()
        unresolved.extend(
            f"parked:{coid}" for coid in sorted(pending)
        )
        for row in self._store.iter_live_orders():
            if row.state in PENDING_DISPATCH_STATES:
                unresolved.append(f"order:{row.client_order_id}({row.state})")
        return unresolved

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

    def _enter_quarantine(self, reason: str, context: dict[str, Any]) -> None:
        """Latch the conflict — dual signal, mirroring the tracker.

        Persists the epoch's ``quarantined`` state (when an epoch
        exists), then delivers per policy: the ``request_quarantine``
        hook latches the engine and the process stays alive; ``halt`` —
        or a missing / raising hook — arms :attr:`pending_halt`. Never
        fail-open.
        """
        if self._quarantined:
            return
        self._quarantined = True
        self._quarantine_reason = reason
        epoch = self._epoch
        with self._store.transaction():
            if epoch is not None and epoch.state != 'quarantined':
                self._store.set_spot_epoch_state(
                    self._port.product_id, epoch.epoch_seq, 'quarantined',
                )
                self._epoch = self._store.get_latest_spot_epoch(
                    self._port.product_id,
                )
            self._store.log_event(
                'spot_inventory_quarantine',
                payload={'reason': reason, **context},
            )
        message = (
            f"spot inventory conflict ({reason}): trading stops; "
            f"operator rebaseline + restart required"
        )
        quarantined = False
        if self._policy == 'quarantine' and self._request_quarantine is not None:
            # noinspection PyBroadException
            try:
                self._request_quarantine(message, dict(context))
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
