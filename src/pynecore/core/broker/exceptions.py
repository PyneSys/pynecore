"""
Broker-related exception hierarchy.

All broker plugin and order-sync errors derive from :class:`BrokerError`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pynecore.core.broker.models import ExchangeOrder

__all__ = [
    'AuthenticationError',
    'BracketAttachAfterFillRejectedError',
    'BrokerError',
    'BrokerManualInterventionError',
    'ClientOrderIdSpentError',
    'ExchangeCapabilityError',
    'ExchangeConnectionError',
    'ExchangeOrderRejectedError',
    'ExchangeRateLimitError',
    'InsufficientMarginError',
    'OrderDispositionUnknownError',
    'OrderSkippedByPlugin',
    'OrderSyncError',
    'SpotInventoryConflictError',
    'UnexpectedCancelError',
]


class BrokerError(RuntimeError):
    """Base class for all broker-related errors."""


class AuthenticationError(BrokerError):
    """The exchange rejected the plugin's credentials.

    Raised when the exchange returns 401 / 403, reports an invalid API key,
    or bans the source IP. Semantics are **terminal** — reconnect cannot
    recover, the user must fix the credentials. The Script Runner treats
    this as a graceful-stop condition at startup and surfaces an
    :class:`~pynecore.core.broker.models.AuthenticationFailedEvent` so the
    observability layer can page.

    :ivar reason: Short human-readable cause (echoed on the event).
    """

    def __init__(self, message: str, reason: str = "") -> None:
        super().__init__(message)
        self.reason = reason or message


class ExchangeCapabilityError(BrokerError):
    """The exchange does not support a required feature.

    Raised by a BrokerPlugin when asked to do something its exchange cannot do
    (e.g. a TP+SL bracket with OCA reduce semantics on an exchange without
    native support for it). Treated as a graceful-stop condition at startup.
    """


class ExchangeConnectionError(BrokerError):
    """Connection to the exchange was lost.

    The Order Sync Engine is expected to reconnect and then reconcile state
    before resuming normal operation.
    """


class ExchangeOrderRejectedError(BrokerError):
    """The exchange rejected an order.

    :ivar order: The rejected order as it is known locally, or ``None`` if the
        order never made it far enough to have an exchange representation.
    """

    def __init__(self, message: str, order: 'ExchangeOrder | None' = None) -> None:
        super().__init__(message)
        self.order = order


class InsufficientMarginError(ExchangeOrderRejectedError):
    """The exchange rejected an order for insufficient margin / balance.

    A typed sub-class of :class:`ExchangeOrderRejectedError` so the risk
    layer can pattern-match the reason without string-parsing the exchange
    message. Intent-level reject — non-terminal, the runner keeps going and
    the strategy can respond (e.g. shrink size, back off).
    """


class ClientOrderIdSpentError(ExchangeOrderRejectedError):
    """A create was refused because the deterministic client order id is
    already consumed by a now-terminal order, on a venue that never allows
    client-id reuse (Bybit rejects re-creation under a cancelled order's
    ``orderLinkId``: spot retCode 170141, derivatives 110072 — measured
    live). The default cancel+recreate modify path re-sends the same
    pinned id, so on such venues the recreate collides with the id the
    cancel just spent.

    Contract for plugins: raise ONLY after verifying that nothing is live
    under the id (the duplicate lookup returned a terminal, dead order —
    not a fill) and after cleaning up any sibling orders the same dispatch
    attempt already placed. The engine responds by re-anchoring the
    intent's envelope on a bumped ``retry_seq`` and re-dispatching
    immediately, so the replacement lands under a fresh id instead of the
    dead order being silently adopted as live.
    """


class BracketAttachAfterFillRejectedError(ExchangeOrderRejectedError):
    """A protective TP/SL bracket attach was rejected AFTER the parent
    ENTRY/EXIT fill committed on the exchange.

    Distinct from a plain :class:`ExchangeOrderRejectedError` where no
    parent fill occurred (e.g. an entry rejected before contacting the
    exchange — nothing open, nothing to defend). Here the position is
    *open and unprotected*: the sync engine MUST NOT halt (which would
    leave the unprotected fill exposed indefinitely), instead it issues a
    defensive market close to take the position flat.

    The plugin raises this *after* rolling back the persisted leg rows
    (so the BrokerStore does not leak phantom protective legs). The sync
    engine's :meth:`_dispatch_new` recovery path catches it, builds a
    synthetic :class:`~pynecore.core.broker.models.CloseIntent` for the
    parent position, and dispatches that. The runner continues.

    Engine-side recovery uses a derived
    :class:`~pynecore.core.broker.models.BracketAttachRejectContext` as
    the formal hand-off to the plugin's
    :meth:`~pynecore.core.plugin.broker.BrokerPlugin.get_residual_orders_after_bracket_attach_reject`
    method — this exception is the *transport* (chained cause, message),
    the context is the *contract* (data the engine needs to settle the
    defensive close).

    :ivar position_coid: Client-order-id of the unprotected open position
        (the parent ENTRY row in BrokerStore). Universal recovery key
        across plugins — required.
    :ivar position_side: Side of the OPEN position (``"buy"`` for long,
        ``"sell"`` for short). The defensive close picks the opposite.
    :ivar qty: Quantity of the unprotected position (units the close
        intent must flatten). ``0`` is a *proven-flat* signal — the
        plugin measured that a racing sibling fill already consumed the
        entire bracket quantity; the engine then skips the defensive
        close (nothing remains to flatten) while still running the
        OCA-cancel / residual-cleanup cascade.
    :ivar symbol: Trading symbol of the unprotected position.
    :ivar position_deal_id: Exchange-side identifier of the unprotected
        open position, when the plugin can supply one. Broker-specific
        (Capital.com deal id, IB permId, Bybit orderId, ...); optional —
        recovery must not rely on it for correctness.
    :ivar from_entry: Pine ``strategy.entry`` id, when the bracket attach
        originated from a Pine ``strategy.exit``. Used for log
        correlation; not required for the close itself.
    :ivar exit_id: Pine ``strategy.exit`` id when applicable.
    :ivar filled_qty: Parent quantity already filled at the time of the
        reject. ``None`` falls back to ``qty`` (conservative).
    :ivar error_code: Exchange-side error code, if any.
    :ivar error_message: Exchange-side error message, if any.
    """

    def __init__(
            self,
            message: str,
            *,
            position_coid: str,
            symbol: str,
            position_side: str,
            qty: float,
            position_deal_id: str | None = None,
            from_entry: str | None = None,
            exit_id: str | None = None,
            filled_qty: float | None = None,
            error_code: str | None = None,
            error_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.position_coid = position_coid
        self.symbol = symbol
        self.position_side = position_side
        self.qty = qty
        self.position_deal_id = position_deal_id
        self.from_entry = from_entry
        self.exit_id = exit_id
        self.filled_qty = filled_qty
        self.error_code = error_code
        self.error_message = error_message


class ExchangeRateLimitError(BrokerError):
    """Exchange rate limit was hit.

    :ivar retry_after: Seconds the caller should wait before retrying.
    """

    def __init__(self, message: str, retry_after: float) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class OrderDispositionUnknownError(BrokerError):
    """A dispatch completed without a definitive accept-or-reject from the exchange.

    Raised by a BrokerPlugin when a submission times out mid-flight or the
    connection drops before the exchange acknowledges the order — the plugin
    genuinely does not know whether the order landed. Semantics are
    deliberately distinct from :class:`ExchangeConnectionError` (recoverable
    via reconnect) and :class:`ExchangeOrderRejectedError` (the order is known
    not to exist): the Order Sync Engine reacts by holding the dispatch in a
    pending-verification queue and matching against
    :meth:`~pynecore.core.plugin.broker.BrokerPlugin.get_open_orders` on the
    next sync, keyed by ``client_order_id``.

    :ivar client_order_id: The id the plugin attempted to submit with. The
        sync engine uses it to match the open-orders query back to the
        originating dispatch.
    :ivar cause: The underlying raw exception, if any, preserved for logging.
    :ivar predecessor_cancel_ids: For an ambiguous MODIFY dispatch, the
        modify shape the plugin executed before the disposition turned
        unknown. ``None`` (default) — shape undeclared: the sync engine
        assumes the plugin's default cancel + re-execute modify and treats a
        venue CANCELLED push for any currently mapped order id during the
        park as the engine-initiated predecessor confirmation. An empty
        tuple — atomic in-place amend, NO predecessor cancel was issued: a
        CANCELLED push during the park is a genuine external cancel and
        fires the ``on_unexpected_cancel`` policy. A non-empty tuple — the
        exchange order ids the plugin cancel-issued before the ambiguous
        replacement submission; exactly those pushes are consumed as
        engine-initiated. Meaningful only when raised from
        ``modify_entry`` / ``modify_exit``-shaped plugin calls.
    """

    def __init__(
            self,
            message: str,
            client_order_id: str,
            cause: Exception | None = None,
            predecessor_cancel_ids: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(message)
        self.client_order_id = client_order_id
        self.cause = cause
        self.predecessor_cancel_ids = predecessor_cancel_ids


class OrderSkippedByPlugin(BrokerError):
    """Plugin proactively declined to dispatch — not a failure, no order sent.

    Distinct from :class:`ExchangeOrderRejectedError` (the exchange said no)
    and :class:`OrderDispositionUnknownError` (network-ambiguous): the plugin
    short-circuited *before* contacting the exchange because the intent does
    not satisfy a known venue constraint (e.g. ``intent.qty`` below the
    instrument's ``min_size``). The Order Sync Engine logs a broker-warning
    and refuses to register the intent in ``_active_intents``, so the next
    bar re-evaluates the intent freely — a runtime sizing model that yields
    a placeable qty later just trades normally without recovery glue.

    :ivar intent_key: The intent's diff key, for engine-side lookups.
    :ivar reason: Short stable token (e.g. ``"below_min_size"``) for log
        filtering and programmatic policy.
    :ivar context: Plugin-supplied diagnostic dictionary (symbol, qty,
        min_size, …) for operator triage.
    """

    def __init__(
            self,
            message: str,
            *,
            intent_key: str,
            reason: str = "",
            context: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.intent_key = intent_key
        self.reason = reason
        self.context = context or {}


class OrderSyncError(BrokerError):
    """Exchange state diverged from the expected internal state."""


class BrokerManualInterventionError(BrokerError):
    """Automated execution cannot safely continue — a human must resolve
    broker-side ambiguity before the strategy runs again.

    Raised by a BrokerPlugin when reconcile or recovery logic encounters a
    state that cannot be resolved without risking an incorrect trade — e.g.
    a partial-close race whose reverse leg cannot be confidently identified,
    or a crash-recovery heuristic that finds multiple candidate deals for a
    single submitted intent. Semantics are **terminal**: the sync engine
    halts all further dispatches and the runner performs a graceful stop.
    Distinct from :class:`ExchangeOrderRejectedError` (the order is known
    not to exist) and :class:`OrderDispositionUnknownError` (the plugin
    parks and retries on next sync) — manual intervention signals that
    *automated* recovery cannot proceed safely.

    :ivar reason: Human-readable description of the ambiguity.
    :ivar intent_key: The intent's diff key, if the ambiguity relates to a
        specific in-flight intent. ``None`` for orphan/recovery cases.
    :ivar context: Plugin-supplied diagnostic dictionary (e.g. candidate
        deal ids, snapshot totals, timing) for operator triage.
    """

    def __init__(
            self,
            reason: str,
            *,
            intent_key: str | None = None,
            context: dict | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.intent_key = intent_key
        self.context = context or {}


class SpotInventoryConflictError(BrokerManualInterventionError):
    """The spot balance invariant broke or the bot's inventory ownership
    could not be established.

    Raised by the :class:`~pynecore.core.broker.spot_inventory.SpotInventoryManager`
    when the ``halt`` inventory-conflict policy is active (or as the
    fail-safe fallback when the quarantine hook is missing / raising):
    an unexplainable base-balance drift in either direction, a corrupt
    or foreign ledger row, an inconclusive execution catch-up, or a
    lost asset-ownership lease. External intervention in the bot's
    inventory is not supported — there is no adoption path, so the safe
    terminal signal is manual intervention followed by an operator
    ``rebaseline``.
    """


class UnexpectedCancelError(BrokerManualInterventionError):
    """A bot-owned order disappeared without the bot having cancelled it.

    Indicates external interference (manual user action, exchange-side
    maintenance, margin-induced cancel, etc.). Modelled as a manual-
    intervention error because automated recovery cannot reason about why
    the order vanished — the safe default is to halt and let a human
    inspect. The sync engine's :meth:`run_event_stream` and dispatch paths
    catch :class:`BrokerManualInterventionError` uniformly, so the same
    graceful-stop pipeline applies whether the trigger came from a polling
    snapshot or from an in-flight dispatch.
    """
