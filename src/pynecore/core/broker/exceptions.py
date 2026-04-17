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
    'BrokerError',
    'ExchangeCapabilityError',
    'ExchangeConnectionError',
    'ExchangeOrderRejectedError',
    'ExchangeRateLimitError',
    'InsufficientMarginError',
    'OrderDispositionUnknownError',
    'OrderSyncError',
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
    """

    def __init__(
            self,
            message: str,
            client_order_id: str,
            cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.client_order_id = client_order_id
        self.cause = cause


class OrderSyncError(BrokerError):
    """Exchange state diverged from the expected internal state."""


class UnexpectedCancelError(BrokerError):
    """A bot-owned order disappeared without the bot having cancelled it.

    Indicates external interference (manual user action, exchange-side
    maintenance, margin-induced cancel, etc.). The default policy is a
    graceful stop.
    """
