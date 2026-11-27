"""
Broker-related exception hierarchy.

All broker plugin and order-sync errors derive from :class:`BrokerError`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pynecore.core.broker.models import QuarantineKind

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
    'JournalWriteError',
    'MarketClosedError',
    'OrderDispositionUnknownError',
    'OrderSkippedByPlugin',
    'OrderSyncError',
    'SpotInventoryConflictError',
    'UnexpectedCancelError',
    'VenueWritesQuarantinedError',
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


class VenueWritesQuarantinedError(ExchangeConnectionError):
    """A venue write was refused locally: an external-intervention quarantine is latched.

    Raised before anything is sent, by the sync engine's write runners and by
    each plugin's transport chokepoint. It is an :class:`ExchangeConnectionError`
    on purpose: every write path already treats that as "the request never left
    the client — keep the state and retry later", which is exactly what a write
    deferred by the quarantine needs. After the operator resumes, the deferred
    work runs on the next sync.
    """

    def __init__(self, what: str = "venue write") -> None:
        super().__init__(
            f"{what} blocked: the bot is quarantined after an external "
            f"intervention and leaves the venue untouched until an operator "
            f"resume"
        )


class JournalWriteError(ExchangeConnectionError):
    """A journal write failed before anything was sent to the venue.

    Raised in place of the raw storage error (``sqlite3.Error``) by a
    persist-first write path when the write that must precede the venue call
    does not commit. The write is rolled back and the venue call is never
    made, so nothing is sent and nothing is advanced. It is an
    :class:`ExchangeConnectionError` on purpose, like
    :class:`VenueWritesQuarantinedError`: every write path already treats that
    as "the request never left the client — keep the state and retry later",
    and the sync loop retries the dispatch on the next sync. A storage failure
    AFTER the venue accepted a write is raised as
    :class:`OrderDispositionUnknownError` instead, so the dispatch parks and is
    resolved from venue data.

    :ivar cause: The underlying storage error.
    """

    def __init__(self, what: str, cause: BaseException) -> None:
        super().__init__(f"{what}: journal write failed before the venue call ({cause})")
        self.cause = cause


class ExchangeOrderRejectedError(BrokerError):
    """The exchange rejected an order.

    :ivar order: The rejected order as it is known locally, or ``None`` if the
        order never made it far enough to have an exchange representation.
    :ivar level_crossed: ``True`` when the venue refused a protective level
        (stop-loss / take-profit) because the market had already traded
        through it. For an exit the refusal is then Pine's immediate exit
        for a crossed level, not a venue that refuses to protect the
        position: the engine closes the parent at market without
        quarantining. For a single-level STOP or LIMIT entry it means the
        quote already crossed the entry level, which Pine fills at once: the
        engine sends the entry at market instead of dropping the signal.
    """

    def __init__(
            self,
            message: str,
            order: 'ExchangeOrder | None' = None,
            *,
            level_crossed: bool = False,
    ) -> None:
        super().__init__(message)
        self.order = order
        self.level_crossed = level_crossed


class InsufficientMarginError(ExchangeOrderRejectedError):
    """The exchange rejected an order for insufficient margin / balance.

    A typed sub-class of :class:`ExchangeOrderRejectedError` so the risk
    layer can pattern-match the reason without string-parsing the exchange
    message. Intent-level reject — non-terminal, the runner keeps going and
    the strategy can respond (e.g. shrink size, back off).
    """


class MarketClosedError(ExchangeOrderRejectedError):
    """The exchange refused an order because the instrument is not tradable now.

    A scheduled session pause, a maintenance break or a trading halt: the
    venue evaluated the request and declined it synchronously, so nothing
    landed and the position is untouched. Non-terminal for EVERY intent kind
    — the sync engine drops the dispatch and re-evaluates the intent once
    trading resumes, instead of the fatal contract a protective-order reject
    otherwise carries.
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
    engine's :meth:`~pynecore.core.broker.sync_engine.OrderSyncEngine._dispatch_new`
    recovery path catches it, builds a synthetic
    :class:`~pynecore.core.broker.models.CloseIntent` for the parent position, and
    dispatches that. The runner continues.

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

    ``level_crossed`` (see :class:`ExchangeOrderRejectedError`): the defensive
    close is then the exit Pine books for a crossed level, not a failed
    protection, and the engine logs it as such.
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
            level_crossed: bool = False,
    ) -> None:
        super().__init__(message, level_crossed=level_crossed)
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
    """The plugin cannot safely go on with what the engine asked of it.

    Raised by a BrokerPlugin when reconcile or recovery logic encounters a
    state that cannot be resolved without risking an incorrect trade — e.g.
    a partial-close race whose reverse leg cannot be confidently identified,
    or a crash-recovery heuristic that finds multiple candidate deals for a
    single submitted intent. The sync engine never stops the run on it: it
    latches a quarantine (:meth:`~pynecore.core.broker.sync_engine.OrderSyncEngine.record_quarantine`)
    and keeps the process alive. An :class:`UnexpectedCancelError` (someone
    else acted on the account) latches the ``external_intervention``
    quarantine; any other report latches an ``entry_block`` that ends by
    itself once a reconcile passes without a new report and the venue holds
    what the book holds. A write the error interrupted is treated as one of
    unknown outcome, never sent again blindly.
    Distinct from :class:`ExchangeOrderRejectedError` (the order is known
    not to exist) and :class:`OrderDispositionUnknownError` (the plugin
    parks and retries on next sync).

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
    as the fail-safe fallback when its quarantine hook is missing or raised:
    an unexplainable base-balance drift in either direction, a corrupt
    or foreign ledger row, an inconclusive execution catch-up, or a
    lost asset-ownership lease. The engine latches the quarantine the
    manager decided (:attr:`kind`) when the error reaches it.

    :ivar kind: The quarantine kind the conflict calls for: ``entry_block``
        for the bot's own inventory trouble, ``external_intervention`` when
        the bot's inventory was taken.
    """

    def __init__(
            self,
            reason: str,
            *,
            intent_key: str | None = None,
            context: dict | None = None,
            kind: QuarantineKind = QuarantineKind.ENTRY_BLOCK,
    ) -> None:
        super().__init__(reason, intent_key=intent_key, context=context)
        self.kind = kind


class UnexpectedCancelError(BrokerManualInterventionError):
    """A bot-owned order disappeared without the bot having cancelled it.

    Indicates external interference (manual user action, exchange-side
    maintenance, margin-induced cancel, etc.): someone else acted on the
    account. The sync engine latches the ``external_intervention``
    quarantine on it wherever it arrives (the event stream, a dispatch, a
    reconcile read): the process stays alive and the bot leaves the venue as
    it is until the operator resumes it. Raised only where the quarantine
    hook a policy needs is missing or failed.
    """
