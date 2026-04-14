"""
Broker plugin runtime support.

- :mod:`pynecore.core.broker.models` — intent, event, exchange-state,
  capability and requirement dataclasses.
- :mod:`pynecore.core.broker.exceptions` — broker error hierarchy.
- :mod:`pynecore.core.broker.position` — :class:`BrokerPosition` live
  position tracker (no simulation).
"""
from pynecore.core.broker.exceptions import (
    BrokerError,
    ExchangeCapabilityError,
    ExchangeConnectionError,
    ExchangeOrderRejectedError,
    ExchangeRateLimitError,
    OrderSyncError,
    UnexpectedCancelError,
)
from pynecore.core.broker.models import (
    OrderStatus,
    OrderType,
    LegType,
    ExchangeOrder,
    OrderEvent,
    ExchangePosition,
    ExchangeCapabilities,
    EntryIntent,
    ExitIntent,
    CloseIntent,
    CancelIntent,
    ScriptRequirements,
    InterceptorResult,
)
from pynecore.core.broker.position import BrokerPosition

__all__ = [
    'BrokerError',
    'ExchangeCapabilityError',
    'ExchangeConnectionError',
    'ExchangeOrderRejectedError',
    'ExchangeRateLimitError',
    'OrderSyncError',
    'UnexpectedCancelError',
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
    'BrokerPosition',
]
