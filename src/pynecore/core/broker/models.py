"""
Data models for the broker plugin system.

These classes form the protocol between Pine Script, the Order Sync Engine,
and a concrete :class:`~pynecore.core.plugin.broker.BrokerPlugin`. Intent
objects describe what the script wants; Event objects describe what the
exchange actually did; Exchange* objects are snapshots of exchange state.

See ``docs/pynecore/plugin-system/broker-plugin-plan.md`` for the full design.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    'OrderStatus',
    'OrderType',
    'LegType',
    'ExchangeOrder',
    'OrderEvent',
    'ExchangePosition',
    'ExchangeCapabilities',
    'EntryIntent',
    'ExitIntent',
    'CloseIntent',
    'CancelIntent',
    'ScriptRequirements',
    'InterceptorResult',
    'BrokerEvent',
    'BracketRegisteredEvent',
    'LegPartialRepairedEvent',
    'LegRepairFailedEvent',
    'BracketReconstructedEvent',
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
    support.  Declared once at startup.  A capability is ``True`` when the
    plugin can uphold its semantics on this exchange — natively (one atomic
    exchange call) or in software (e.g. two reduce-only orders + stream-driven
    repair with OCA reduce semantics).  If neither path can uphold the
    required semantics, declare ``False`` and
    :func:`~pynecore.core.broker.validation.validate_at_startup` rejects the
    script.
    """
    # Order types
    stop_order: bool = False
    stop_limit_order: bool = False
    trailing_stop: bool = False
    # Exit bracket (TP+SL with OCA reduce semantics).  ``tp_sl_bracket=True``
    # means the plugin delivers the bracket on this exchange; it does NOT
    # imply native support.  ``tp_sl_bracket_native=True`` additionally
    # promises a single atomic exchange call — useful for diagnostics,
    # latency budgeting, and per-exchange reconcile strategy.
    tp_sl_bracket: bool = False
    tp_sl_bracket_native: bool = False
    # Order management
    amend_order: bool = False
    cancel_all: bool = False
    reduce_only: bool = False
    # Streaming & position
    watch_orders: bool = False
    fetch_position: bool = False


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

    @property
    def intent_key(self) -> str:
        return self.pine_id


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
