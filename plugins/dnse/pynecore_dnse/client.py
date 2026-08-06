"""Plugin-side client over the vendored DNSE openapi-sdk.

The vendored ``DNSEClient`` (``_vendor/dnse``) is the official SDK, but it has
three sharp edges that this thin wrapper normalizes for plugin use — keeping the
vendored copy pristine (all fixes live here, not in the SDK):

1. **Pin API version to** ``2026-07-23`` (the published version). The SDK's
   ``cancel_order`` takes NO per-call ``version`` — it uses the client default —
   and a *floating* date (e.g. ``2026-08-06``) silently breaks conditional
   cancels: DNSE reads ``orderId`` as an integer and no-ops on the string id,
   returning ``200`` while cancelling nothing (proven live 2026-08-07). So the
   default is fixed here and there is deliberately **no override**.
2. **Parse response bodies.** The SDK returns ``(status, raw_string)``; every
   method here returns ``(status, parsed)`` where ``parsed`` is a ``dict``/``list``
   (or the raw text if it wasn't JSON, or ``None`` when empty) — the
   ``(status, body)`` contract the provider/broker are written against.
3. **Restore TLS verification.** The SDK ships ``cert_reqs=CERT_NONE`` +
   ``assert_hostname=False`` (unverified HTTPS — unacceptable for a live trading
   client). We swap in a verifying ``PoolManager`` backed by ``certifi``.
"""
from __future__ import annotations

import json

import certifi
import urllib3

from ._sdk import DNSEClient as _SdkClient

#: Published DNSE API version. Do NOT use a floating/future date — it silently
#: breaks the conditional-order cancel path (see module docstring).
API_VERSION = "2026-07-23"

DEFAULT_BASE_URL = "https://openapi.dnse.com.vn"


class DNSEClient:
    """Version-pinned, JSON-parsing, TLS-verifying facade over the vendored SDK.

    Public SDK methods (``get_ohlc``, ``post_order``, ``cancel_order``,
    ``get_orders``, ``get_security_definition``, ``send_email_otp`` …) are
    delegated through ``__getattr__`` and their ``(status, body)`` result is
    parsed. Callers use the same method names/signatures as the SDK.
    """

    def __init__(self, api_key: str, api_secret: str,
                 base_url: str = DEFAULT_BASE_URL):
        self._sdk = _SdkClient(api_key, api_secret, base_url=base_url,
                               api_version=API_VERSION)
        # Replace the SDK's unverified PoolManager with a verifying one.
        self._sdk._http = urllib3.PoolManager(
            num_pools=10, maxsize=10, block=False,
            timeout=urllib3.Timeout(connect=30.0, read=60.0),
            cert_reqs="CERT_REQUIRED", ca_certs=certifi.where(),
        )

    @staticmethod
    def _parse(result):
        """Turn the SDK's ``(status, raw)`` into ``(status, parsed)``."""
        if not (isinstance(result, tuple) and len(result) == 2):
            return result
        status, body = result
        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8", "replace")
        if isinstance(body, str) and body.strip():
            try:
                body = json.loads(body)
            except ValueError:
                pass  # leave non-JSON text as-is
        return status, body

    def __getattr__(self, name: str):
        # Only reached for names not defined on the wrapper (i.e. SDK methods).
        # ``_sdk`` lives in ``__dict__`` so it is found before this fires.
        if name.startswith("_"):
            raise AttributeError(name)
        attr = getattr(self._sdk, name)
        if not callable(attr):
            return attr

        def wrapped(*args, **kwargs):
            return self._parse(attr(*args, **kwargs))

        wrapped.__name__ = name
        return wrapped
