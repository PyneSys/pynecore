"""
Data models for the broker plugin system.

These classes form the protocol between Pine Script, the Order Sync Engine,
and a concrete :class:`~pynecore.core.plugin.broker.BrokerPlugin`. Intent
objects describe what the script wants; Event objects describe what the
exchange actually did; Exchange* objects are snapshots of exchange state.

See ``docs/pynecore/plugin-system/broker-plugin-plan.md`` for the full design.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from pynecore.core.broker.idempotency import build_client_order_id

if TYPE_CHECKING:
    from pynecore.core.broker.exceptions import BracketAttachAfterFillRejectedError

__all__ = [
    'OrderStatus',
    'OrderType',
    'LegType',
    'OcaType',
    'OcaPartialFillPolicy',
    'CapabilityLevel',
    'PartialQtyBracketExitMode',
    'ExchangeOrder',
    'OrderEvent',
    'ExchangePosition',
    'ExchangeCapabilities',
    'EntryIntent',
    'ExitIntent',
    'CloseIntent',
    'CancelIntent',
    'BracketAttachRejectContext',
    'PendingDefensiveClose',
    'DispatchEnvelope',
    'ScriptRequirements',
    'InterceptorResult',
    'BrokerEvent',
    'AuthenticationFailedEvent',
    'BracketRegisteredEvent',
    'LegPartialRepairedEvent',
    'LegRepairFailedEvent',
    'BracketReconstructedEvent',
    'ManualInterventionRequiredEvent',
    'ProtectionDegradedEvent',
]


class OrderStatus(StrEnum):
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class LegType(StrEnum):
    ENTRY = "entry"
    TAKE_PROFIT = "tp"
    STOP_LOSS = "sl"
    TRAILING_STOP = "trail"
    CLOSE = "close"


class OcaType(StrEnum):
    """Canonical OCA semantics for the sync engine.

    The Pine-level literal values (``strategy.oca.cancel`` / ``.reduce`` /
    ``.none``) are plain strings for script compatibility; this enum is the
    single authority the sync engine and intent builder match against. Adding
    a new OCA semantic therefore requires exactly one source edit, not a
    scattered grep across the intent-builder / sync-engine / validator.
    """
    CANCEL = "cancel"
    REDUCE = "reduce"
    NONE = "none"


class OcaPartialFillPolicy(StrEnum):
    """How the sync engine treats a *partial* fill for OCA-cancel cascading.

    On a full fill the behaviour is unambiguous: sibling orders in the same
    OCA-cancel group must be cancelled. Partial fills are the grey zone — some
    exchanges re-fill the remainder at a better price (risking sibling fills
    too), others do not. The policy lets the user pick:

    - :data:`FILL_CANCELS` (default): a partial fill already commits the script
      to this side, so sibling cancel triggers immediately. Matches the
      Pine backtester, where the first touch on any leg wins.
    - :data:`FULL_FILL_ONLY`: wait until the leg is fully filled. Useful when
      the user prefers siblings to stay live in case the first leg partial is
      followed by a same-bar reversal that would otherwise lock in a
      sub-optimal entry.
    """
    FILL_CANCELS = "fill_cancels"
    FULL_FILL_ONLY = "full_fill_only"


class CapabilityLevel(StrEnum):
    """Tri-tone declaration of how a plugin upholds a single capability.

    The plugin advertises *what it can deliver end-to-end* (not raw exchange
    support). The four levels are an at-a-glance summary the runner, sync
    engine, validator and CLI can all consume:

    - :data:`UNSUPPORTED` — neither the exchange nor the plugin can uphold the
      semantics; :func:`~pynecore.core.broker.validation.validate_at_startup`
      rejects scripts that need it. Default so a missing field never silently
      advertises a capability.
    - :data:`SOFTWARE` — upheld in the plugin / sync engine without any
      exchange-side primitive (e.g. polling order state, software OCA cascade,
      netting-based reduce-only). Validation passes; latency / failure
      semantics are the plugin's responsibility.
    - :data:`PARTIAL_NATIVE` — the exchange supports a *subset* of the
      semantics natively, the plugin fills the rest in software. Used when a
      capability has axes the exchange covers some but not all of (e.g.
      Capital.com ``amend_order``: level / SL / TP amendable, ``size`` is not
      and needs cancel+recreate). Validation passes; PARTIAL_NATIVE is a
      diagnostic flag, not a stricter contract — the sync engine treats it
      the same as SOFTWARE for fallback decisions.
    - :data:`NATIVE` — single atomic exchange call delivers the full
      semantics (e.g. Bybit attached TP/SL, exchange-side OCA group, native
      reduce-only flag). The sync engine may suppress its software fallback
      paths on this level (see :class:`OrderSyncEngine`).

    The string values are stable and safe to log / persist (e.g. SQLite
    columns, ``pyne plugin info`` output).
    """
    UNSUPPORTED = "unsupported"
    SOFTWARE = "software"
    PARTIAL_NATIVE = "partial_native"
    NATIVE = "native"

    @property
    def is_supported(self) -> bool:
        """``True`` for every level except :data:`UNSUPPORTED`.

        The validator uses this to decide whether to reject a script — a
        SOFTWARE-level capability is just as valid as NATIVE, only the cost /
        latency / failure profile differs.
        """
        return self is not CapabilityLevel.UNSUPPORTED


class PartialQtyBracketExitMode(StrEnum):
    """Mechanism a plugin uses to deliver partial-qty exit brackets.

    A specialisation of :class:`CapabilityLevel` for the
    :attr:`ExchangeCapabilities.partial_qty_bracket_exit` field only, because
    the general :data:`~CapabilityLevel.SOFTWARE` value cannot disambiguate
    the two software paths the order sync engine has to dispatch differently:

    - :data:`UNSUPPORTED` — the plugin cannot attach a TP/SL/trailing leg to
      a fraction of a position. ``strategy.exit(qty=N, from_entry='L', ...)``
      with ``N < total_qty`` is rejected by
      :func:`~pynecore.core.broker.validation.validate_at_startup` rather
      than silently mis-hedging onto the whole row. Default for every
      plugin.
    - :data:`SOFTWARE_ENGINE_TRIGGER` — no parked broker order. The engine
      arms an in-memory leg state machine per
      ``strategy.exit`` call and dispatches a partial close via the plugin's
      existing ``execute_close`` route when the price crosses the trigger
      level. The plugin only needs the primitives it already has
      (``execute_close`` partial-close, ``get_position``, WS quote feed).
      Used by per-deal CFD brokers (Capital.com, IG, OANDA) where a
      reduce-only flag is either absent or not submit-time guaranteed.
    - :data:`SOFTWARE_REDUCE_ONLY_ORDER` — the plugin parks a reduce-only
      working order at the trigger level and the exchange guarantees it can
      only ever reduce the position, never open a new one. Used by
      aggregated-position brokers with documented reduce-only semantics
      (Bybit USDT-Perp, Binance Futures). Requires the three primitives
      ``submit_reduce_only_working_order`` /
      ``modify_reduce_only_working_order`` /
      ``cancel_reduce_only_working_order``.
    - :data:`NATIVE` — the exchange itself supports a partial-qty bracket
      within a single position as a first-class primitive. Placeholder for
      future plugins; no current broker delivers this.

    The order sync engine matches on this enum to pick the dispatch path
    (``_dispatch_engine_trigger_partial_bracket`` vs the reduce-only-order
    path), so the explicit ENUM value carries the contract — no implicit
    pairing with a companion field.
    """
    UNSUPPORTED = "unsupported"
    SOFTWARE_ENGINE_TRIGGER = "software_engine_trigger"
    SOFTWARE_REDUCE_ONLY_ORDER = "software_reduce_only_order"
    NATIVE = "native"

    @property
    def is_supported(self) -> bool:
        """``True`` for every mode except :data:`UNSUPPORTED`.

        Mirrors :attr:`CapabilityLevel.is_supported` so
        :func:`~pynecore.core.broker.validation.validate_at_startup` can
        treat this field with the same ``capability.is_supported`` check it
        uses for every other capability.
        """
        return self is not PartialQtyBracketExitMode.UNSUPPORTED


# === Exchange state snapshots ===

@dataclass
class ExchangeOrder:
    """An order as it exists on the exchange."""
    id: str
    symbol: str
    side: str  # "buy" | "sell"
    order_type: OrderType
    qty: float
    filled_qty: float
    remaining_qty: float
    price: float | None  # Limit price
    stop_price: float | None  # Trigger price
    average_fill_price: float | None
    status: OrderStatus
    timestamp: float  # Creation time (unix seconds)
    fee: float
    fee_currency: str
    reduce_only: bool = False
    # Exchange-side clientOrderId (our allocation, echoed back by the exchange).
    # Required for post-restart bracket reconstruction — without it, open TP/SL
    # legs left on the exchange cannot be mapped back to Pine identity.
    client_order_id: str | None = None


@dataclass
class OrderEvent:
    """
    A normalized fill/status event reported by a BrokerPlugin.

    The plugin is responsible for mapping exchange-level events back to
    Pine-level identity. A single Pine exit intent may become multiple
    exchange orders (e.g. Bybit Partial TP/SL pairs), and only the plugin
    knows the mapping.
    """
    order: ExchangeOrder
    event_type: str  # "created" | "filled" | "partial" | "cancelled" | "rejected"
    fill_price: float | None
    fill_qty: float | None
    timestamp: float
    # Pine-level identity (filled by the plugin, used by sync engine + BrokerPosition)
    pine_id: str | None = None  # Pine order ID (entry id or exit id)
    from_entry: str | None = None  # Which entry this fill belongs to (for exits)
    leg_type: LegType | None = None  # Which leg of a bracket filled
    fee: float = 0.0
    fee_currency: str = ""

    def __str__(self) -> str:
        parts = [
            self.event_type.upper(),
            f"id={self.order.id}",
            f"side={self.order.side}",
            f"qty={self.order.qty}",
            f"filled={self.order.filled_qty}",
        ]
        if self.fill_price is not None:
            parts.append(f"price={self.fill_price}")
        if self.pine_id:
            parts.append(f"pine={self.pine_id!r}")
        if self.from_entry:
            parts.append(f"from={self.from_entry!r}")
        if self.leg_type is not None:
            parts.append(f"leg={self.leg_type.value}")
        return " ".join(parts)


@dataclass
class ExchangePosition:
    """Current position on the exchange (futures/margin)."""
    symbol: str
    side: str  # "long" | "short" | "flat"
    size: float
    entry_price: float
    unrealized_pnl: float
    liquidation_price: float | None
    leverage: float
    margin_mode: str  # "cross" | "isolated"


@dataclass
class ExchangeCapabilities:
    """
    What the plugin can deliver end-to-end for the script, not raw exchange
    support. Declared once at startup. Each capability is a
    :class:`CapabilityLevel` — :data:`~CapabilityLevel.UNSUPPORTED` (default)
    rejects scripts that need it via
    :func:`~pynecore.core.broker.validation.validate_at_startup`; any other
    level passes the validator. The level distinction
    (NATIVE / PARTIAL_NATIVE / SOFTWARE) is a diagnostic — the sync engine
    treats NATIVE as "exchange owns this; suppress my software fallback"
    for the fields it explicitly checks (``oca_cancel``, ``tp_sl_bracket``),
    and everything else as "engine still runs the fallback path". Plugins
    that can guarantee end-to-end atomic delivery should declare NATIVE so
    the engine can skip its emulation.
    """
    # === Order types ===
    # NATIVE = exchange has the order type as a first-class primitive.
    # SOFTWARE = plugin emulates it (e.g. a poll loop that converts a
    # client-side trigger price into a market submit). UNSUPPORTED rejects
    # scripts that use the corresponding Pine parameter.
    stop_order: CapabilityLevel = CapabilityLevel.UNSUPPORTED
    stop_limit_order: CapabilityLevel = CapabilityLevel.UNSUPPORTED
    # NATIVE = server-side trailing stop (e.g. Capital.com
    # ``trailingStop=true, stopDistance``). SOFTWARE = plugin tracks the
    # last extreme and amends the SL each tick / bar.
    trailing_stop: CapabilityLevel = CapabilityLevel.UNSUPPORTED

    # === Exit bracket (TP+SL with OCA reduce semantics) ===
    # NATIVE = single atomic exchange call attaches both legs (Bybit V5
    # attached TP/SL, Capital.com position-attribute bracket). The sync
    # engine's partial-fill bracket-amend path is suppressed at this level.
    # PARTIAL_NATIVE = the exchange takes one leg natively but the other
    # requires a follow-up call (rare — e.g. SL on the position but TP only
    # as a separate working order). Engine fallback stays active.
    # SOFTWARE = the plugin issues two reduce-only orders and runs the
    # OCA-reduce / cascade-cancel logic itself.
    tp_sl_bracket: CapabilityLevel = CapabilityLevel.UNSUPPORTED

    # Mechanism the plugin uses to deliver a partial-qty exit bracket
    # (``strategy.exit(qty=N, from_entry="L", ...)`` with ``N`` less than
    # the full row qty entered under ``"L"``). Uses a dedicated
    # :class:`PartialQtyBracketExitMode` ENUM rather than
    # :class:`CapabilityLevel` because the order sync engine has to dispatch
    # the two software paths differently:
    #
    # - ``SOFTWARE_ENGINE_TRIGGER``: no parked broker order; the engine arms
    #   a per-leg state machine and triggers a partial close via the
    #   existing ``execute_close`` route on price cross. Capital.com / IG /
    #   OANDA path.
    # - ``SOFTWARE_REDUCE_ONLY_ORDER``: the plugin parks a reduce-only
    #   working order at the trigger level; broker guarantees it can only
    #   reduce, never open. Bybit / Binance path (future Slice C).
    # - ``NATIVE``: exchange supports partial-qty bracket inside a single
    #   position. Placeholder; no current broker delivers this.
    # - ``UNSUPPORTED``: validator rejects such scripts at startup rather
    #   than silently covering the whole row.
    partial_qty_bracket_exit: PartialQtyBracketExitMode = (
        PartialQtyBracketExitMode.UNSUPPORTED
    )

    # Companion to :attr:`partial_qty_bracket_exit`. ``True`` declares that
    # the plugin can route a partial-qty bracket correctly even when the
    # script's ``pyramiding > 1`` (i.e. multiple parent positions share the
    # Pine ``entry_id`` "L"). Aggregated-position brokers (Bybit netting,
    # Binance Futures) can flip this on once they implement the bracket
    # path; per-deal brokers (Capital.com, IG) leave it ``False`` until the
    # multi-row reduction-order routing question is answered empirically
    # (see ``partial-qty-bracket-exit-plan.md`` §9 #13). When ``False`` the
    # validator rejects ``pyramiding > 1`` scripts that need
    # partial-qty bracket support, regardless of the chosen mechanism.
    partial_qty_bracket_exit_supports_pyramiding: bool = False

    # === OCA cancel groups ===
    # NATIVE = the exchange tracks the OCA group and cancels siblings on
    # fill (Bybit bracket, OKX algo orders). The sync engine SUPPRESSES its
    # cascade-cancel logic — exchange is authoritative.
    # SOFTWARE = the engine emits CancelIntent dispatches itself when a
    # leg fills (the default for the vast majority of exchanges, including
    # Capital.com — its position-attribute bracket is a separate capability
    # under ``tp_sl_bracket``, not OCA).
    # UNSUPPORTED = the plugin cannot deliver OCA-cancel semantics at all
    # — scripts using ``oca_type='cancel'`` are rejected at startup.
    oca_cancel: CapabilityLevel = CapabilityLevel.UNSUPPORTED

    # === Order management ===
    # NATIVE = the exchange amends every parameter (price, size, SL/TP).
    # PARTIAL_NATIVE = some axes are amendable, others (commonly ``size``)
    # require cancel+recreate (Capital.com is exactly this case — level /
    # SL / TP fields amend in-place, ``size`` does not).
    # SOFTWARE = no in-place amend; plugin always cancel+recreate.
    amend_order: CapabilityLevel = CapabilityLevel.UNSUPPORTED

    # NATIVE = single batch call (e.g. Binance ``DELETE /openOrders``).
    # SOFTWARE = plugin iterates per-id cancel under the hood.
    cancel_all: CapabilityLevel = CapabilityLevel.UNSUPPORTED

    # NATIVE = exchange honours an explicit reduce-only flag on the order.
    # SOFTWARE = upheld via netting / one-way mode (plugin maps an
    # opposite-side order onto the existing position so it cannot flip).
    # UNSUPPORTED = scripts using ``strategy.exit`` / ``strategy.close`` are
    # rejected at startup.
    reduce_only: CapabilityLevel = CapabilityLevel.UNSUPPORTED

    # === Streaming & position ===
    # NATIVE = a real WebSocket order channel.
    # SOFTWARE = plugin emulates the stream by polling REST endpoints and
    # diffing snapshots (Capital.com ``GET /workingorders`` +
    # ``GET /positions`` cadence).
    watch_orders: CapabilityLevel = CapabilityLevel.UNSUPPORTED

    # NATIVE = a single REST/WS read returns the position state.
    # SOFTWARE = plugin reconstructs by aggregating fills locally.
    # The distinction is mostly informational — both deliver the same
    # contract.
    fetch_position: CapabilityLevel = CapabilityLevel.UNSUPPORTED

    # === Idempotency ===
    # NATIVE = the exchange accepts a client-supplied order id, echoes it
    # back on subsequent reads, AND rejects duplicate submissions of the
    # same id (Binance/Bybit/OKX). The sync engine and recovery path can
    # rely on the exchange to dedup retries after a timeout / restart.
    # PARTIAL_NATIVE = the exchange echoes the client id but does NOT dedup
    # duplicates; the plugin must dedup client-side before each dispatch
    # (Interactive Brokers, Deribit).
    # SOFTWARE = the exchange generates the id (Capital.com server-side
    # ``dealReference``); the plugin dedups in its own SQLite store using
    # :attr:`DispatchEnvelope.client_order_id` as the local key.
    # Restart-safe recovery still works, just without exchange-side
    # enforcement.
    # UNSUPPORTED = neither echo nor dedup — restart/timeout retries are
    # unsafe; live scripts are rejected at startup.
    idempotency: CapabilityLevel = CapabilityLevel.UNSUPPORTED


# === Pine Script intents ===

@dataclass(frozen=True)
class EntryIntent:
    """What the script wants: open or add to a position."""
    pine_id: str  # strategy.entry(id=...) or strategy.order(id=...)
    symbol: str
    side: str  # "buy" | "sell"
    qty: float
    order_type: OrderType
    limit: float | None = None  # Limit price
    stop: float | None = None  # Trigger/activation price
    oca_name: str | None = None  # OCA group name (strategy.entry/order oca_name param)
    oca_type: str | None = None  # "reduce" | "cancel" | "none" (strategy.oca.*)
    comment: str | None = None
    alert_message: str | None = None
    is_strategy_order: bool = False  # True if from strategy.order() (no pyramiding limit)

    @property
    def intent_key(self) -> str:
        """Stable diff key for the sync engine."""
        return self.pine_id

    def __str__(self) -> str:
        parts = [
            f"ENTRY {self.side.upper()} id={self.pine_id!r}",
            f"qty={self.qty}",
            f"type={self.order_type.value}",
        ]
        if self.limit is not None:
            parts.append(f"limit={self.limit}")
        if self.stop is not None:
            parts.append(f"stop={self.stop}")
        return " ".join(parts)


@dataclass(frozen=True)
class ExitIntent:
    """What the script wants: reduce/close a position via TP/SL bracket."""
    pine_id: str  # strategy.exit(id=...)
    from_entry: str  # strategy.exit(from_entry=...)
    symbol: str
    side: str  # Exit side ("sell" for long TP/SL, "buy" for short TP/SL)
    qty: float
    tp_price: float | None = None  # Take-profit limit price
    sl_price: float | None = None  # Stop-loss trigger price
    trail_price: float | None = None  # Trailing stop activation price
    trail_offset: float | None = None  # Trailing stop offset (price units)
    # Raw tick values — used when exit is against a pending (unfilled) entry,
    # so absolute prices cannot be calculated yet. The Order Sync Engine
    # converts these to tp_price/sl_price after the entry fills.
    profit_ticks: float | None = None
    loss_ticks: float | None = None
    trail_points_ticks: float | None = None
    oca_name: str | None = None  # OCA group name (auto: __exit_{id}_{from_entry}_oca__)
    oca_type: str | None = None  # "reduce" | "cancel" | "none" (default: reduce for exits)
    comment: str | None = None
    comment_profit: str | None = None
    comment_loss: str | None = None
    comment_trailing: str | None = None
    alert_message: str | None = None
    # One-way Pine semantics: every strategy.exit is reduce-only by definition.
    # A manual position close while the exit is pending must not flip the
    # book back to the other side. The plugin must pass this to the exchange
    # (Binance/Bybit/OKX ``reduceOnly``, Capital.com force-close, etc.).
    # ``False`` is rejected at construction — a future ``HedgeBrokerPlugin``
    # subclass will introduce a separate hedge-aware intent rather than flip
    # this flag.
    reduce_only: bool = True
    # ``True`` when the script asked for a TP/SL/trailing bracket on a
    # *fraction* of the parent entry's declared quantity
    # (``strategy.exit(qty=N, from_entry='L', tp=...)`` with ``N`` strictly
    # less than the declared total under ``'L'``). The intent builder
    # fills this from ``position.entry_orders[from_entry].size`` — the
    # script's declaration, not the broker's actual fill — falling back
    # to the ``open_trades`` aggregate when the entry Order has been
    # cancelled / cleared. The order sync engine dispatches on this flag
    # together with ``caps.partial_qty_bracket_exit`` to pick between the
    # native bracket path (NATIVE), the engine-side trigger state machine
    # (``SOFTWARE_ENGINE_TRIGGER``) and the parked reduce-only
    # working-order path (``SOFTWARE_REDUCE_ONLY_ORDER``). A bracket on
    # the whole row keeps this ``False`` and falls back to the existing
    # full-row path.
    #
    # ``compare=False`` deliberately excludes the flag from
    # :meth:`__eq__` / :meth:`__hash__`: it is a *route selector* derived
    # from the intent, not state that needs to be diff-synced to the
    # broker. The sync engine's qty-cap reconciliation
    # (``_sync_pine_exit_qty``) mutates ``exit_orders[(exit_id, from_entry
    # )].size`` after a partial entry fill so the next ``build_intents``
    # produces an ExitIntent with the capped qty; including the derived
    # flag in equality would let that cap mutation flip the flag and
    # trigger a spurious second ``modify_exit`` dispatch even though
    # nothing the broker needs to know has changed.
    is_partial_qty_bracket: bool = field(default=False, compare=False)

    def __post_init__(self) -> None:
        if self.reduce_only is not True:
            raise ValueError(
                "ExitIntent.reduce_only must be True — one-way Pine semantics. "
                "Hedge-mode intents belong on a future HedgeBrokerPlugin subclass."
            )

    @property
    def intent_key(self) -> str:
        """
        Stable diff key for the sync engine.

        Cannot be just pine_id — strategy.exit(id="TP") can create separate
        exit orders for different from_entry values (e.g. "Long" and "Short").
        The (pine_id, from_entry) tuple is the unique key.
        """
        return f"{self.pine_id}\0{self.from_entry}"

    @property
    def has_unresolved_ticks(self) -> bool:
        """True if tick-based prices need entry fill price to resolve."""
        return (self.profit_ticks is not None or self.loss_ticks is not None
                or self.trail_points_ticks is not None)

    def __str__(self) -> str:
        parts = [
            f"EXIT id={self.pine_id!r}",
            f"from={self.from_entry!r}",
            f"qty={self.qty}",
        ]
        if self.tp_price is not None:
            parts.append(f"tp={self.tp_price}")
        if self.sl_price is not None:
            parts.append(f"sl={self.sl_price}")
        if self.trail_price is not None or self.trail_offset is not None:
            parts.append(f"trail={self.trail_price}/{self.trail_offset}")
        return " ".join(parts)


@dataclass(frozen=True)
class CloseIntent:
    """
    What the script wants: close position with market order.

    The ``immediately`` flag mirrors TradingView backtest semantics: without
    it, a close waits for the next bar's open. With ``calc_on_every_tick``
    a non-immediate close can delay execution by an entire bar in live
    trading, which is why the flag exists.
    """
    pine_id: str  # strategy.close(id=...) or strategy.close_all()
    symbol: str
    side: str  # "sell" to close long, "buy" to close short
    qty: float
    immediately: bool = False
    comment: str | None = None
    alert_message: str | None = None
    # Same invariant as :attr:`ExitIntent.reduce_only` — a close can never
    # flip the book to the other side in one-way Pine mode.
    reduce_only: bool = True

    def __post_init__(self) -> None:
        if self.reduce_only is not True:
            raise ValueError(
                "CloseIntent.reduce_only must be True — one-way Pine semantics."
            )

    @property
    def intent_key(self) -> str:
        return self.pine_id

    def __str__(self) -> str:
        return (
            f"CLOSE id={self.pine_id!r} side={self.side} qty={self.qty}"
        )


@dataclass(frozen=True)
class CancelIntent:
    """
    What the script wants: cancel a pending order.

    ``strategy.cancel(id)`` cancels ALL orders matching that id. For exits
    this means every (pine_id, from_entry) pair with that pine_id. The
    Order Sync Engine resolves the affected intent_keys and may send
    multiple CancelIntents (one per from_entry), or a single one with
    ``from_entry=None`` meaning "cancel all exits with this pine_id".
    """
    pine_id: str
    symbol: str
    from_entry: str | None = None

    @property
    def intent_key(self) -> str:
        if self.from_entry is not None:
            return f"{self.pine_id}\0{self.from_entry}"
        return self.pine_id

    def __str__(self) -> str:
        tail = f" from={self.from_entry!r}" if self.from_entry else ""
        return f"CANCEL id={self.pine_id!r}{tail}"


# === Recovery contracts ===

@dataclass(frozen=True)
class BracketAttachRejectContext:
    """Recovery context for a bracket attach reject after a parent fill.

    Built by the sync engine when a plugin raises
    :class:`~pynecore.core.broker.exceptions.BracketAttachAfterFillRejectedError`
    and threaded into
    :meth:`~pynecore.core.plugin.broker.BrokerPlugin.get_residual_orders_after_bracket_attach_reject`
    so the plugin can enumerate any broker-side orders the engine must
    cancel as part of defensive recovery.

    Separation of concerns: the exception carries transport + debug info
    (chained cause, message), this context carries the recovery contract
    (what the engine needs to settle the defensive close and rebuild
    state on restart). Frozen + hashable so it can live inside a
    :class:`PendingDefensiveClose` marker that is persisted across
    restarts via the BrokerStore extras column.

    :ivar intent_key: Diff key of the original (rejected) intent — used
        to correlate the defensive close with its trigger in the audit log.
    :ivar position_coid: Client-order-id of the unprotected open position
        (the parent ENTRY row in BrokerStore). Universal recovery key
        across plugins.
    :ivar position_side: Side of the OPEN position (``"buy"`` for long,
        ``"sell"`` for short). The defensive close picks the opposite.
    :ivar qty: Quantity of the unprotected position the close must
        flatten.
    :ivar symbol: Trading symbol of the unprotected position.
    :ivar position_deal_id: Exchange-side identifier of the unprotected
        position when the plugin can supply one. Broker-specific
        (Capital.com deal id, IB permId, Bybit orderId, ...); recovery
        logic must not rely on it for correctness — use ``position_coid``
        for cross-broker lookups.
    :ivar from_entry: Pine ``strategy.entry`` id when the rejected
        bracket originated from a ``strategy.exit``. Populated by
        :meth:`from_exception` from the exception or — as a fallback —
        from the triggering intent.
    :ivar exit_id: Pine ``strategy.exit`` id when applicable.
    :ivar filled_qty: Quantity already filled on the parent at the time
        of the reject. ``None`` falls back to ``qty`` (conservative).
    :ivar error_code: Plugin-supplied exchange-side error code, if any.
    :ivar error_message: Plugin-supplied exchange-side error message.
    """
    intent_key: str
    position_coid: str
    position_side: str
    qty: float
    symbol: str
    position_deal_id: str | None = None
    from_entry: str | None = None
    exit_id: str | None = None
    filled_qty: float | None = None
    error_code: str | None = None
    error_message: str | None = None

    @classmethod
    def from_exception(
            cls,
            e: 'BracketAttachAfterFillRejectedError',
            intent: 'EntryIntent | ExitIntent | CloseIntent | CancelIntent',
    ) -> 'BracketAttachRejectContext':
        """Build a context from a raised exception + its triggering intent.

        ``from_entry`` is taken from the exception when present;
        otherwise it falls back to ``intent.from_entry`` for an
        :class:`ExitIntent` or ``intent.pine_id`` for an
        :class:`EntryIntent`. The optional plugin-supplied fields are
        read via :func:`getattr` so plugins can adopt them at their own
        pace without breaking the contract.
        """
        from_entry = e.from_entry
        if from_entry is None:
            if isinstance(intent, ExitIntent):
                from_entry = intent.from_entry
            elif isinstance(intent, EntryIntent):
                from_entry = intent.pine_id
        return cls(
            intent_key=intent.intent_key,
            position_coid=e.position_coid,
            position_side=e.position_side,
            qty=e.qty,
            symbol=e.symbol,
            position_deal_id=e.position_deal_id,
            from_entry=from_entry,
            exit_id=getattr(e, 'exit_id', None),
            filled_qty=getattr(e, 'filled_qty', None),
            error_code=getattr(e, 'error_code', None),
            error_message=getattr(e, 'error_message', None),
        )

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict for BrokerStore extras."""
        return {
            'intent_key': self.intent_key,
            'position_coid': self.position_coid,
            'position_side': self.position_side,
            'qty': self.qty,
            'symbol': self.symbol,
            'position_deal_id': self.position_deal_id,
            'from_entry': self.from_entry,
            'exit_id': self.exit_id,
            'filled_qty': self.filled_qty,
            'error_code': self.error_code,
            'error_message': self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'BracketAttachRejectContext':
        """Rebuild from a previously serialized ``to_dict`` payload.

        Raises :class:`ValueError` if required fields are missing or
        not of the expected primitive type — callers (startup replay)
        catch this and log+skip the malformed extras row instead of
        crashing the engine.
        """
        if not isinstance(data, dict):
            raise ValueError(
                f"BracketAttachRejectContext payload must be a dict, "
                f"got {type(data).__name__}"
            )
        try:
            intent_key = data['intent_key']
            position_coid = data['position_coid']
            position_side = data['position_side']
            qty = data['qty']
            symbol = data['symbol']
        except KeyError as exc:
            raise ValueError(
                f"BracketAttachRejectContext payload missing required "
                f"field {exc.args[0]!r}: {data!r}"
            ) from exc
        if not (isinstance(intent_key, str) and isinstance(position_coid, str)
                and isinstance(position_side, str) and isinstance(symbol, str)):
            raise ValueError(
                f"BracketAttachRejectContext required string fields have "
                f"wrong type: {data!r}"
            )
        if not isinstance(qty, (int, float)):
            raise ValueError(
                f"BracketAttachRejectContext.qty must be numeric: {data!r}"
            )
        filled_qty_raw = data.get('filled_qty')
        if filled_qty_raw is not None and not isinstance(filled_qty_raw, (int, float)):
            raise ValueError(
                f"BracketAttachRejectContext.filled_qty must be numeric or "
                f"None: {data!r}"
            )
        return cls(
            intent_key=intent_key,
            position_coid=position_coid,
            position_side=position_side,
            qty=float(qty),
            symbol=symbol,
            position_deal_id=data.get('position_deal_id'),
            from_entry=data.get('from_entry'),
            exit_id=data.get('exit_id'),
            filled_qty=(
                float(filled_qty_raw)
                if filled_qty_raw is not None else None
            ),
            error_code=data.get('error_code'),
            error_message=data.get('error_message'),
        )


@dataclass(frozen=True)
class PendingDefensiveClose:
    """Engine-side marker for an in-flight defensive close.

    Set by :meth:`OrderSyncEngine._handle_bracket_attach_after_fill_reject`
    BEFORE the defensive close dispatches; cleared by the engine's FILL
    handler when the close lands. While the marker lives, the entry
    intent stays in ``_active_intents`` so a same-bar
    :meth:`OrderSyncEngine.reconcile` cannot mistakenly classify the
    flat broker snapshot as an external flatten — see the
    ``defensive-close-pending-lifecycle`` design doc.

    Persisted to the parent entry's BrokerStore row under
    ``extras['defensive_close_pending']`` so a process restart between
    dispatch and FILL can replay the marker; the engine's startup
    replay then either drops the marker (FILL already recorded) or
    re-arms residual cancel handling.

    :ivar entry_id: Pine entry id this marker belongs to.
    :ivar close_intent_key: Diff key of the synthetic defensive close.
    :ivar close_order_ref: Broker-side order ref of the close, when the
        plugin returned one. ``None`` if dispatch ran but the plugin did
        not surface a ref (e.g. position-attached close on Capital.com).
    :ivar close_client_order_id: Canonical ``client_order_id`` of the
        synthetic close dispatch (derived from the close envelope). Set
        as soon as the dispatch is parked or returns — used to match the
        eventual FILL via :attr:`OrderEvent.order.client_order_id` when
        ``pine_id`` is absent and the parked order never appeared in
        :meth:`~pynecore.core.plugin.broker.BrokerPlugin.get_open_orders`
        (so :meth:`OrderSyncEngine._maybe_attach_defensive_close_ref`
        never backfilled :attr:`close_order_ref`). Also used to drop the
        parked pending row from
        :attr:`OrderSyncEngine._pending_verification` (and the persisted
        ``record_park`` row) once the close settles.
    :ivar pending_since: ``time.time()`` at marker creation. Drives the
        :meth:`OrderSyncEngine.reconcile` stale-pending grace check.
    :ivar reject_context: Frozen snapshot of the
        :class:`BracketAttachRejectContext` that produced the close —
        used by startup replay to re-derive residual orders without
        rebuilding the exception.
    :ivar residual_cleanup_pending: ``True`` when the dispatch-time
        residual-cancel call hit a transient
        :class:`ExchangeConnectionError` /
        :class:`OrderDispositionUnknownError` and could not finish
        cancelling parent/TP/SL/partial-remainder orders. The next
        :meth:`OrderSyncEngine.reconcile` replays
        :meth:`OrderSyncEngine._cancel_bracket_reject_residuals` so the
        residuals do not stay live until the FILL eventually arrives;
        cleared once the retry finishes (or on FILL-time settlement,
        whichever happens first).
    :ivar fill_observed: ``True`` once
        :meth:`OrderSyncEngine._route_defensive_close_fill` has matched
        the defensive-close FILL event (and seeded the in-memory
        duplicate-fill caches) but the final residual-cancel + audit
        sequence has not yet completed. Persisted so that a crash in
        the post-FILL settlement window does NOT lose the only record
        that the close already filled — startup replay re-seeds the
        duplicate-fill caches and routes the marker through the
        FILL-side branch of
        :meth:`OrderSyncEngine._retry_residual_cleanup_after_transient_fill`
        rather than waiting for a FILL that will never re-arrive.
    :ivar partial_filled_qty: Cumulative ``fill_qty`` already applied
        to :attr:`OrderSyncEngine._position.size` by no-FIFO
        defensive-close ``partial`` events. Set to zero on marker
        creation; accumulates as partials arrive. Used by the terminal
        ``filled`` event's missing/zero ``fill_qty`` fallback to derive
        the remaining close qty (``reject_context.qty -
        partial_filled_qty``) instead of re-applying the full marker
        qty (which would double-subtract the partial slices already
        accounted for).
    :ivar unapplied_partial_qty: Cumulative ``fill_qty`` from no-FIFO
        defensive-close ``partial`` events that the engine could NOT
        apply to :attr:`OrderSyncEngine._position.size` because the
        parent ENTRY fill had not yet arrived (``_position.size ==
        0.0``). Tracked separately from :attr:`partial_filled_qty` so
        the terminal computation does not believe the slice was
        already booked. When the parent ENTRY ``filled`` / ``partial``
        event finally routes through
        :meth:`OrderSyncEngine._route_event`, the engine subtracts the
        signed accumulated qty from :attr:`_position.size` and moves
        it onto :attr:`partial_filled_qty` so the terminal close FILL
        sees a consistent state. ``0.0`` on marker creation and after
        every drain.
    :ivar fill_exchange_order_id: Broker-side ``order.id`` of the
        actual FILL event observed by
        :meth:`OrderSyncEngine._route_defensive_close_fill`. Persisted
        alongside :attr:`fill_observed` so that startup replay can
        re-seed :attr:`OrderSyncEngine._settled_defensive_close_order_refs`
        with the fill id when the FILL arrived with an id different
        from :attr:`close_order_ref` (polled-orders fallback / broker
        rekey). Without persisting it, a delayed WS / polled-orders
        replay of the same FILL post-restart whose
        ``event.order.id`` differs from ``close_order_ref`` and does
        not echo ``close_client_order_id`` would slip past
        :meth:`OrderSyncEngine._is_duplicate_defensive_close_fill` and
        be re-applied to ``BrokerPosition``. ``None`` until the FILL
        is observed (or when the FILL event carried no ``order``).
    :ivar pre_close_position_size: Signed
        :attr:`OrderSyncEngine._position.size` captured at marker
        creation, before the defensive close dispatched. Used by
        startup replay to distinguish a broker snapshot that still
        reports the pre-close aggregate (close has not landed yet)
        from one that already reflects the close (close landed before
        crash but the FILL was not observed by the prior process —
        ``fill_observed`` would otherwise stay ``False``). Without it,
        the post-restart reconcile cannot safely adopt the broker's
        reduced-but-not-flat size into :attr:`_position.size` because
        the no-FIFO settle branch would treat the adopted size as
        pre-close and over-reduce when the delayed FILL routes.
        ``None`` only for markers persisted by an older schema; the
        replay path falls back to the conservative "skip adoption"
        behaviour in that case.
    :ivar fifo_closed_entry_ids: Snapshot of the entry ids whose FIFO
        ``Trade`` rows were closed by :meth:`record_fill` for this
        defensive close, accumulated across every ``partial`` /
        terminal ``filled`` event of the close sequence. The
        ``t == 'partial'`` branch in
        :meth:`OrderSyncEngine._route_event` merges each partial's
        FIFO closures here, and
        :meth:`OrderSyncEngine._route_defensive_close_fill` merges the
        terminal slice on top BEFORE the residual-cancel attempt — so
        in pyramiding (LongA closed by a partial, LongB closed by the
        terminal) both entries are persisted, not only the terminal's.
        Persisted so that
        :meth:`OrderSyncEngine._retry_residual_cleanup_after_transient_fill`
        can run the same FIFO-aware
        :meth:`OrderSyncEngine._cleanup_position_tracking` walk as the
        in-flight FILL path even when intervening
        :meth:`record_fill` calls have overwritten the live
        ``_last_fifo_closed_entry_ids`` snapshot. Empty when the FILL
        produced no FIFO closures (no-FIFO / degenerate path) or when
        the marker is replayed from an older schema; the retry path
        falls back to ``marker.entry_id`` in that case.
    """
    entry_id: str
    close_intent_key: str
    close_order_ref: str | None
    pending_since: float
    reject_context: BracketAttachRejectContext
    close_client_order_id: str | None = None
    residual_cleanup_pending: bool = False
    fill_observed: bool = False
    partial_filled_qty: float = 0.0
    fill_exchange_order_id: str | None = None
    pre_close_position_size: float | None = None
    fifo_closed_entry_ids: tuple[str, ...] = ()
    unapplied_partial_qty: float = 0.0

    def to_extras_dict(self) -> dict:
        """Serialize to a JSON-compatible dict for ``extras`` storage."""
        return {
            'entry_id': self.entry_id,
            'close_intent_key': self.close_intent_key,
            'close_order_ref': self.close_order_ref,
            'close_client_order_id': self.close_client_order_id,
            'pending_since': self.pending_since,
            'reject_context': self.reject_context.to_dict(),
            'residual_cleanup_pending': self.residual_cleanup_pending,
            'fill_observed': self.fill_observed,
            'partial_filled_qty': self.partial_filled_qty,
            'fill_exchange_order_id': self.fill_exchange_order_id,
            'pre_close_position_size': self.pre_close_position_size,
            'fifo_closed_entry_ids': list(self.fifo_closed_entry_ids),
            'unapplied_partial_qty': self.unapplied_partial_qty,
        }

    @classmethod
    def from_extras_dict(cls, data: dict) -> 'PendingDefensiveClose':
        """Rebuild from a previously serialized ``to_extras_dict`` payload.

        Raises :class:`ValueError` on malformed input — the startup
        replay catches this and logs + skips the row.
        """
        if not isinstance(data, dict):
            raise ValueError(
                f"PendingDefensiveClose payload must be a dict, "
                f"got {type(data).__name__}"
            )
        try:
            entry_id = data['entry_id']
            close_intent_key = data['close_intent_key']
            pending_since = data['pending_since']
            ctx_payload = data['reject_context']
        except KeyError as exc:
            raise ValueError(
                f"PendingDefensiveClose payload missing required "
                f"field {exc.args[0]!r}: {data!r}"
            ) from exc
        if not (isinstance(entry_id, str) and isinstance(close_intent_key, str)):
            raise ValueError(
                f"PendingDefensiveClose required string fields have "
                f"wrong type: {data!r}"
            )
        if not isinstance(pending_since, (int, float)):
            raise ValueError(
                f"PendingDefensiveClose.pending_since must be numeric: "
                f"{data!r}"
            )
        close_order_ref = data.get('close_order_ref')
        if close_order_ref is not None and not isinstance(close_order_ref, str):
            raise ValueError(
                f"PendingDefensiveClose.close_order_ref must be str or "
                f"None: {data!r}"
            )
        close_client_order_id = data.get('close_client_order_id')
        if (close_client_order_id is not None
                and not isinstance(close_client_order_id, str)):
            raise ValueError(
                f"PendingDefensiveClose.close_client_order_id must be str "
                f"or None: {data!r}"
            )
        residual_cleanup_pending = bool(
            data.get('residual_cleanup_pending', False),
        )
        fill_observed = bool(data.get('fill_observed', False))
        partial_filled_qty_raw = data.get('partial_filled_qty', 0.0)
        if not isinstance(partial_filled_qty_raw, (int, float)):
            raise ValueError(
                f"PendingDefensiveClose.partial_filled_qty must be "
                f"numeric: {data!r}"
            )
        partial_filled_qty = float(partial_filled_qty_raw)
        if partial_filled_qty < 0.0:
            raise ValueError(
                f"PendingDefensiveClose.partial_filled_qty must be "
                f"non-negative: {data!r}"
            )
        fill_exchange_order_id = data.get('fill_exchange_order_id')
        if (fill_exchange_order_id is not None
                and not isinstance(fill_exchange_order_id, str)):
            raise ValueError(
                f"PendingDefensiveClose.fill_exchange_order_id must be "
                f"str or None: {data!r}"
            )
        pre_close_position_size_raw = data.get('pre_close_position_size')
        if pre_close_position_size_raw is None:
            pre_close_position_size: float | None = None
        elif isinstance(pre_close_position_size_raw, (int, float)):
            pre_close_position_size = float(pre_close_position_size_raw)
        else:
            raise ValueError(
                f"PendingDefensiveClose.pre_close_position_size must be "
                f"numeric or None: {data!r}"
            )
        fifo_closed_entry_ids_raw = data.get('fifo_closed_entry_ids', ())
        if not isinstance(fifo_closed_entry_ids_raw, (list, tuple)):
            raise ValueError(
                f"PendingDefensiveClose.fifo_closed_entry_ids must be "
                f"a list or tuple: {data!r}"
            )
        fifo_closed_entry_ids_list: list[str] = []
        for entry in fifo_closed_entry_ids_raw:
            if not isinstance(entry, str):
                raise ValueError(
                    f"PendingDefensiveClose.fifo_closed_entry_ids items "
                    f"must be strings: {data!r}"
                )
            fifo_closed_entry_ids_list.append(entry)
        unapplied_partial_qty_raw = data.get('unapplied_partial_qty', 0.0)
        if not isinstance(unapplied_partial_qty_raw, (int, float)):
            raise ValueError(
                f"PendingDefensiveClose.unapplied_partial_qty must be "
                f"numeric: {data!r}"
            )
        unapplied_partial_qty = float(unapplied_partial_qty_raw)
        if unapplied_partial_qty < 0.0:
            raise ValueError(
                f"PendingDefensiveClose.unapplied_partial_qty must be "
                f"non-negative: {data!r}"
            )
        return cls(
            entry_id=entry_id,
            close_intent_key=close_intent_key,
            close_order_ref=close_order_ref,
            pending_since=float(pending_since),
            reject_context=BracketAttachRejectContext.from_dict(ctx_payload),
            close_client_order_id=close_client_order_id,
            residual_cleanup_pending=residual_cleanup_pending,
            fill_observed=fill_observed,
            partial_filled_qty=partial_filled_qty,
            fill_exchange_order_id=fill_exchange_order_id,
            pre_close_position_size=pre_close_position_size,
            fifo_closed_entry_ids=tuple(fifo_closed_entry_ids_list),
            unapplied_partial_qty=unapplied_partial_qty,
        )


# === Dispatch envelope ===

@dataclass(frozen=True)
class DispatchEnvelope:
    """Broker dispatch envelope — an intent plus idempotency metadata.

    The :class:`~pynecore.core.broker.sync_engine.OrderSyncEngine` wraps every
    intent in a fresh envelope before handing it to the :class:`BrokerPlugin`.
    Plugins call :meth:`client_order_id` for each exchange order they place;
    the result is deterministic, so a retry or restart regenerates the same id
    and the exchange dedups the duplicate.

    :ivar intent: The Pine-level intent this dispatch carries.
    :ivar run_tag: 4-char base36 session tag (see
        :meth:`~pynecore.core.broker.run_identity.RunIdentity.make_run_tag`).
    :ivar bar_ts_ms: Bar open timestamp (ms since Unix epoch).
    :ivar retry_seq: Bumped by the recovery path only when a prior attempt is
        deliberately abandoned — defaults to ``0``.
    """
    intent: 'EntryIntent | ExitIntent | CloseIntent | CancelIntent'
    run_tag: str
    bar_ts_ms: int
    retry_seq: int = 0

    def client_order_id(self, kind: str) -> str:
        """Allocate the canonical client-order-id for a given leg kind.

        :param kind: One of the ``KIND_*`` constants from
            :mod:`pynecore.core.broker.idempotency`.
        """
        return build_client_order_id(
            run_tag=self.run_tag,
            pine_id=self.intent.pine_id,
            bar_ts_ms=self.bar_ts_ms,
            kind=kind,
            retry_seq=self.retry_seq,
        )


# === Compile-time detected script requirements ===

@dataclass
class ScriptRequirements:
    """Broker capabilities needed by this script. Detected via AST analysis."""
    market_orders: bool = False
    limit_orders: bool = False
    stop_orders: bool = False
    stop_limit_orders: bool = False
    tp_sl_bracket: bool = False  # strategy.exit() with BOTH limit+stop or profit+loss
    trailing_stop: bool = False
    strategy_order: bool = False  # strategy.order() — no pyramiding limit
    # True if the script calls any of ``strategy.exit`` / ``strategy.close`` /
    # ``strategy.close_all``. Every such call requires the exchange to honour
    # reduce-only semantics — a manual position close otherwise lets the
    # still-pending exit flip the book the other way. The validator turns
    # this into a hard reject when ``caps.reduce_only=False``.
    exit_orders: bool = False
    # True if the script calls ``strategy.exit(qty=N, from_entry="L", ...)``
    # with ``N < total qty entered under "L"`` AND includes any bracket-leg
    # parameters (``limit=``/``stop=``/``profit=``/``loss=``/``trail_*=``).
    # The validator rejects the script at startup when
    # ``caps.partial_qty_bracket_exit=False`` — position-attribute bracket
    # exchanges (Capital.com) can only attach bracket to the whole row, not
    # a partial quantity, and silently covering the full qty would be a
    # safety violation.
    partial_qty_bracket_exit: bool = False


# === Interceptor (Order Sync Engine extension point) ===

# === Broker events (observability) =======================================

@dataclass
class BrokerEvent:
    """Base class for structured broker-side events.

    The plugin emits these via an injected callback so the runner can
    surface them in logs, metrics, and the user-facing event stream
    without the plugin coupling to any specific sink.
    """


@dataclass
class AuthenticationFailedEvent(BrokerEvent):
    """Emitted when the plugin's credentials are rejected by the exchange.

    ``reason`` is the short human-readable cause (``AuthenticationError.reason``);
    the runner surfaces the event to observability sinks and then performs a
    graceful stop — reconnect cannot gain access with wrong credentials.
    """
    reason: str


@dataclass
class BracketRegisteredEvent(BrokerEvent):
    pine_id: str
    from_entry: str
    tp_order_id: str | None
    sl_order_id: str | None


@dataclass
class LegPartialRepairedEvent(BrokerEvent):
    pine_id: str
    from_entry: str
    leg: str  # "tp" | "sl"
    generation: int
    old_qty: float
    new_qty: float


@dataclass
class LegRepairFailedEvent(BrokerEvent):
    pine_id: str
    from_entry: str
    leg: str  # "tp" | "sl"
    reason: str
    action_taken: str  # "degraded" | "retry" | ...


@dataclass
class BracketReconstructedEvent(BrokerEvent):
    pine_id: str
    from_entry: str
    source: str  # "open_orders" | "position_snapshot" | ...


@dataclass
class ProtectionDegradedEvent(BrokerEvent):
    """The bracket can no longer be maintained with OCA reduce semantics.

    ``reason`` is human-readable / diagnostic; ``policy_action`` names the
    manager's chosen follow-up (``"degraded"`` → bracket left in place but
    unsupervised; ``"terminal"`` → bracket closed out).
    """
    pine_id: str
    from_entry: str
    reason: str
    policy_action: str


@dataclass
class ManualInterventionRequiredEvent(BrokerEvent):
    """Emitted when the sync engine halts because the plugin raised
    :class:`~pynecore.core.broker.exceptions.BrokerManualInterventionError`.

    Surfaces the operator-actionable details (reason, optional intent key,
    plugin-supplied diagnostic context) so the observability bus, on-call
    alerting, and the user-facing event stream can page without the runner
    needing to reach into plugin internals. After this event fires the
    engine is halted — all subsequent :meth:`sync` calls return early until
    the strategy is restarted.
    """
    reason: str
    intent_key: str | None = None
    context: dict | None = None


@dataclass
class InterceptorResult:
    """
    Interceptor decision on an intent — modifiable before execution.

    Intent objects are frozen, so modifications are expressed as override
    fields on this result rather than by mutating the intent in place.
    """
    intent: EntryIntent | ExitIntent | CloseIntent | CancelIntent
    rejected: bool = False
    reject_reason: str = ""
    modified_qty: float | None = None
    modified_limit: float | None = None
    modified_stop: float | None = None
