"""ccxt exception -> PyneCore broker taxonomy mapping.

One place that knows which ccxt error means what, so ``broker.py`` wraps every
venue call in ``try/except`` and delegates here. Returns ``None`` for anything
unrecognised so the caller re-raises the original.
"""
from __future__ import annotations

from pynecore.core.broker.exceptions import (
    AuthenticationError,
    BrokerError,
    ExchangeConnectionError,
    ExchangeOrderRejectedError,
    ExchangeRateLimitError,
    InsufficientMarginError,
    OrderDispositionUnknownError,
)

__all__ = ['map_ccxt_exception', 'is_order_gone']


def map_ccxt_exception(raw: Exception, *, action: str,
                       client_order_id: str = "") -> BrokerError | None:
    """Classify a ccxt exception into the broker taxonomy.

    :param raw: The exception a ccxt call raised.
    :param action: Human label for the failing call ("place", "cancel", …).
    :param client_order_id: The write's client id — carried on
        :class:`OrderDispositionUnknownError` so the engine can match the
        parked write against ``get_open_orders()``.
    :return: A classified :class:`BrokerError`, or ``None`` to re-raise as-is.
    """
    import ccxt

    detail = f"binance {action}: {type(raw).__name__}: {raw}"
    if isinstance(raw, ccxt.AuthenticationError):
        return AuthenticationError(detail, reason=type(raw).__name__)
    if isinstance(raw, ccxt.InsufficientFunds):
        return InsufficientMarginError(detail)
    if isinstance(raw, ccxt.RateLimitExceeded):
        return ExchangeRateLimitError(detail, 1.0)
    if isinstance(raw, ccxt.RequestTimeout):
        # A write timeout is ambiguous — the order may have landed.
        return OrderDispositionUnknownError(detail, client_order_id=client_order_id)
    if isinstance(raw, ccxt.NetworkError):
        # DDoSProtection / ExchangeNotAvailable / OnMaintenance subclasses too.
        return ExchangeConnectionError(detail)
    if isinstance(raw, (ccxt.InvalidOrder, ccxt.BadRequest, ccxt.ExchangeError)):
        return ExchangeOrderRejectedError(detail)
    if isinstance(raw, ConnectionError):
        return ExchangeConnectionError(detail)
    return None


def is_trigger_immediate(raw: Exception) -> bool:
    """``True`` when the venue refused a stop because the market has already
    crossed the trigger (Binance -2010 "Stop price would trigger
    immediately."). Pine semantics for a crossed stop = fill at market, so
    the caller falls back to a MARKET order."""
    text = str(raw)
    return ('OrderImmediatelyFillable' in type(raw).__name__
            or 'OrderImmediatelyFillable' in text
            or 'would trigger immediately' in text)


def is_order_gone(raw: Exception) -> bool:
    """``True`` when the venue says the order does not exist (already
    terminal / unknown id) — a benign no-op for cancel idempotency."""
    import ccxt
    if isinstance(raw, ccxt.OrderNotFound):
        return True
    # Binance -2011 "Unknown order sent." surfaces as InvalidOrder on some
    # ccxt paths; match the code rather than the class.
    return '-2011' in str(raw)
