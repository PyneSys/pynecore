"""Pure transport-exception → sentinel taxonomy (#67, Phase A1).

The vendored SDK re-raises raw urllib3 exceptions on timeouts/connection
failures (``_request`` catches only ``HTTPError``-with-response). Left
unguarded they kill the run: a lost-reply POST unparked, a read blip past
every ``except ExchangeConnectionError``, the #55 cancel core crashing.

This module maps ANY ``urllib3.exceptions.HTTPError`` to the ``(0, body)``
"no response" sentinel that ``errors.classify`` already understands
(``status==0`` → write:DISPOSITION_UNKNOWN / read:CONNECTION). So the fix is
"catch at the chokepoint, return the sentinel" — the existing classify/raise
path does the rest, no engine change.

Two facts the #67 panel pinned:

- Failures arrive in TWO shapes: ``POST`` is not retried by urllib3
  (``POST ∉ Retry.DEFAULT_ALLOWED_METHODS``) so it raises the bare leaf
  (``ReadTimeoutError``/``NewConnectionError``), while GET/PUT/DELETE raise
  ``MaxRetryError`` wrapping the leaf in ``.reason``. Both must map.
- ``phase`` (connect vs sent) is METADATA ONLY — for A2's journal. It is
  NEVER a retry gate: ``ProtocolError`` is an ``HTTPError`` and is
  ambiguous, SSL is not pre-send, so every WRITE-transport failure parks.
"""
import urllib3

#: The client's "no response reached" status; ``errors.classify`` maps it to
#: the transient disposition (write→park+verify, read→reconnect).
SENTINEL_STATUS = 0

_CONNECT_PHASE = (
    urllib3.exceptions.NewConnectionError,
    urllib3.exceptions.ConnectTimeoutError,
    urllib3.exceptions.NameResolutionError,
    urllib3.exceptions.ProxyError,
)


def _leaf(exc: BaseException) -> BaseException:
    """Unwrap ``MaxRetryError`` (GET/PUT/DELETE) to the underlying reason."""
    reason = getattr(exc, "reason", None)
    return reason if reason is not None else exc


def _phase(exc: BaseException) -> str:
    """Best-effort phase for A2's journal (metadata, never a retry gate)."""
    leaf = _leaf(exc)
    if isinstance(leaf, _CONNECT_PHASE):
        return "connect"      # provably never reached the venue
    if isinstance(leaf, (urllib3.exceptions.ReadTimeoutError,
                         urllib3.exceptions.ProtocolError)):
        return "sent"         # may have reached — a WRITE must park+verify
    return "unknown"


def to_sentinel(exc: BaseException) -> "tuple[int, dict]":
    """Map a urllib3 transport exception to the ``(0, body)`` sentinel."""
    return SENTINEL_STATUS, {
        "code": "NO_RESPONSE",
        "phase": _phase(exc),
        "transport": type(_leaf(exc)).__name__,
    }


def guard(call):
    """Run ``call()``; convert any transport exception to the sentinel.

    ``call`` returns the client's ``(status, body)``. Only
    ``urllib3.exceptions.HTTPError`` (every transport failure is one) is
    converted; anything else propagates — a non-transport bug must not be
    silently swallowed as "no response".
    """
    try:
        return call()
    except urllib3.exceptions.HTTPError as exc:
        return to_sentinel(exc)
