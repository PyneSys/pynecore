"""
Broker-related exception hierarchy.

All broker plugin and order-sync errors derive from :class:`BrokerError`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pynecore.core.broker.models import ExchangeOrder

__all__ = [
    'BrokerError',
    'ExchangeCapabilityError',
    'ExchangeConnectionError',
    'ExchangeOrderRejectedError',
    'ExchangeRateLimitError',
    'OrderSyncError',
    'UnexpectedCancelError',
]


class BrokerError(RuntimeError):
    """Base class for all broker-related errors."""


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


class ExchangeRateLimitError(BrokerError):
    """Exchange rate limit was hit.

    :ivar retry_after: Seconds the caller should wait before retrying.
    """

    def __init__(self, message: str, retry_after: float) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class OrderSyncError(BrokerError):
    """Exchange state diverged from the expected internal state."""


class UnexpectedCancelError(BrokerError):
    """A bot-owned order disappeared without the bot having cancelled it.

    Indicates external interference (manual user action, exchange-side
    maintenance, margin-induced cancel, etc.). The default policy is a
    graceful stop.
    """
