"""Access point for the vendored DNSE openapi-sdk.

The official DNSE SDK (github.com/dnse-tech/openapi-sdk, ``python/dnse``) is
*vendored* under ``_vendor/dnse`` (see ``_vendor/VENDOR_INFO.txt`` for the pinned
tag/commit) rather than pip-installed. Its modules use ``dnse.*`` imports, so
``_vendor`` is prepended to ``sys.path`` here (once, at import) so the vendored
copy always wins over any ``dnse`` that happens to be pip-installed.

Every plugin module imports the SDK through this shim::

    from ._sdk import DNSEClient

``DNSEClient`` is the REST client (stdlib ``urllib`` only). ``TradingClient`` is
the WebSocket trading stream (needs ``websockets``/``certifi``/``msgpack``).
"""
from __future__ import annotations

import os
import sys

_VENDOR = os.path.join(os.path.dirname(__file__), "_vendor")
if _VENDOR not in sys.path:
    # Prepend: the vendored copy must shadow any pip-installed ``dnse``.
    sys.path.insert(0, _VENDOR)

# Imported AFTER the sys.path shim above, hence noqa: E402.
from dnse import (  # noqa: E402
    DNSEClient,
    TradingClient,
    TradingWebSocketError,
    ConnectionClosed,
    AuthenticationError,
    SubscriptionError,
    EncodingError,
)

__all__ = [
    "DNSEClient",
    "TradingClient",
    "TradingWebSocketError",
    "ConnectionClosed",
    "AuthenticationError",
    "SubscriptionError",
    "EncodingError",
]
