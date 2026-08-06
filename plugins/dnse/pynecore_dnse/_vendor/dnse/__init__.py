#!/usr/bin/env python3
from .api._version import __version__ as APIVersion
from .api.client import DNSEClient
from .websocket._version import __version__ as WSVersion
from .websocket.client import TradingClient
from .websocket.exceptions import (
    TradingWebSocketError,
    ConnectionError,
    ConnectionClosed,
    AuthenticationError,
    SubscriptionError,
    EncodingError,
)

__all__ = [
    "DNSEClient",
    "APIVersion",
    "WSVersion",
    "TradingClient",
    "TradingWebSocketError",
    "ConnectionError",
    "ConnectionClosed",
    "AuthenticationError",
    "SubscriptionError",
    "EncodingError"
]
