"""DNSE error classification: map a ``(status, body)`` reply to one deliberate
action, so every write/read path reacts the same way instead of collapsing every
non-2xx into a blanket reject.

Pure and side-effect-free (no logging, raises nothing) so it is table-testable; the
broker turns a :class:`Classified` into the matching ``BrokerError`` and emits the
structured log line. Seeded from ``guide-error_codes.md`` plus codes observed live.
See ``docs/plan/dnse-error-handling.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

#: Fallback wait when a 429 carries no ``X-RateLimit-Reset`` we can read.
DEFAULT_RETRY_AFTER = 5.0


class Disposition(Enum):
    """What the plugin should do about a reply (the value is the log label)."""
    REJECT = "rejected"                     # -> ExchangeOrderRejectedError (terminal for this order)
    MARGIN = "rejected(margin)"             # -> InsufficientMarginError (non-terminal; strategy may resize)
    SESSION_REJECT = "degraded-protection"  # place refused by session -> reject + WARNING
    RATE_LIMIT = "rate-limit"               # -> ExchangeRateLimitError(retry_after)
    DISPOSITION_UNKNOWN = "park+verify"     # transient WRITE -> OrderDispositionUnknownError
    CONNECTION = "reconnect"                # transient READ / cancel-retry -> ExchangeConnectionError
    AUTH = "auth-fail"                      # -> AuthenticationError (terminal)
    AUTH_TOKEN = "token-reread"             # INVALID_TRADING_TOKEN -> re-read state file + retry once
    NOT_FOUND = "not-found"                 # order not in this book -> cancel probes the next / gone
    TERMINAL = "treated-gone"               # order already done -> a cancel of it is success


_LEVEL = {
    Disposition.REJECT: "error", Disposition.MARGIN: "error", Disposition.AUTH: "error",
    Disposition.SESSION_REJECT: "warning", Disposition.RATE_LIMIT: "warning",
    Disposition.DISPOSITION_UNKNOWN: "warning", Disposition.CONNECTION: "warning",
    Disposition.AUTH_TOKEN: "warning",
    Disposition.NOT_FOUND: "info", Disposition.TERMINAL: "info",
}


@dataclass(frozen=True)
class Classified:
    disposition: Disposition
    code: str
    http_status: int
    message: str = ""
    retry_after: float = 0.0

    @property
    def level(self) -> str:
        return _LEVEL[self.disposition]

    def log_message(self, action: str, ident: str) -> str:
        """The one structured line the broker emits (behind the ``[BROKER]`` tag)."""
        ra = f" retry_after={self.retry_after:.0f}s" if self.retry_after else ""
        msg = f' msg="{self.message[:160]}"' if self.message else ""
        return (f"{action} code={self.code} http={self.http_status} -> "
                f"{self.disposition.value} | order={ident}{ra}{msg}")


# --- code tables (docs/plan/dnse-error-handling.md §3) ---

_MARGIN = frozenset({
    "PURCHASING_POWER_NOT_ENOUGH", "PP0_EXCEED", "QMAX_EXCEED", "STOCK_NOT_ENOUGH",
    "VIOLATE_POOL_RULE", "VIOLATE_ROOM_RULE", "OUT_OF_MARGIN_BASKET",
})

#: Place refused by the session — the SAME request would work in another session.
#: Decision (docs/plan §5.1): reject + log, no parking.
_SESSION_PLACE = frozenset({
    "CAN_NOT_PLACE_ORDER_ON_THIS_SESSION",
    "CAN_NOT_PLACE_ORDER_WITH_THAT_ORDER_TYPE_ON_ATO_SESSION",
    "CAN_NOT_PLACE_ORDER_WITH_THAT_ORDER_TYPE_ON_ATC_SESSION",
    "INVALID_ORDER_TYPE_FOR_THIS_SESSION", "INVALID_TRADING_SESSION", "BATCH_IN_PROGRESS",
    "CANNOT_PLACE_ORDER_IN_THE_CLOSED_SESSION",  # observed live
    "CANNOT_PLACE_OPPOSITE_ORDER", "CANNOT_PLACE_OPPOSITE_ORDER_IN_THIS_SESSION",
    "CAN_NOT_PLACE_PLO_ORDER_WITHOUT_MATCHED",
})

#: Cancel/amend refused by the session — the order STILL RESTS, so this is not a
#: reject; a brief retry after the session flips is safe -> CONNECTION (transient).
_SESSION_CANCEL = frozenset({
    "CAN_NOT_CANCEL_THAT_ORDER_ON_THIS_SESSION",
    "CANNOT_CANCEL_THE_ORDER_IN_THE_ATO_SESSION",  # observed live
    "CAN_NOT_CANCEL_ATO_ORDER", "CAN_NOT_CANCEL_MARKET_ORDER",
    "CAN_NOT_CANCEL_PENDINGNEW_ORDER_IN_OPEN_SESSION",
    "CAN_NOT_REPLACE_PLO_ORDER", "CAN_NOT_REPLACE_THAT_ORDER_ON_THIS_SESSION",
})

#: Not in the queried book (or never existed): a cancel probes the next book / is gone.
NOT_FOUND_CODES = frozenset({"RESOURCE_NOT_FOUND", "INVALID_ORDER_ID"})

#: Found but already finished: a cancel of it is success (nothing left to do).
TERMINAL_CODES = frozenset({
    "ORDER_IS_DONE", "ORDER_STATUS_REJECTED",
    "CO-ORD-013",  # observed: conditional already Activated (it fired -> a NORMAL order)
})

_AUTH = frozenset({"OA-401", "OA-403", "FORBIDDEN", "INVALID_OTP", "UNAUTHORIZED"})

_TRANSIENT = frozenset({
    "OA-500", "OA-503", "SYSTEM_ERROR", "REMOTE_SERVER_ERROR", "THIRD_PARTY_ERROR",
    "TIMEOUT", "SERVICE_UNAVAILABLE",
})

_RATE_LIMIT = frozenset({"OA-429", "RATE_LIMIT_EXCEEDED", "TOO_MANY_REQUESTS"})


def code_of(body) -> str:
    """The DNSE error ``code`` from a parsed body, or ``""``."""
    return str(body.get("code") or "") if isinstance(body, dict) else ""


def _message_of(body) -> str:
    if isinstance(body, dict):
        return str(body.get("message") or body.get("error") or "")
    if isinstance(body, str):
        return body.strip()[:200]
    return ""


def _retry_after(body) -> float:
    if isinstance(body, dict):
        for key in ("retryAfter", "retry_after", "X-RateLimit-Reset"):
            val = body.get(key)
            if val:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    pass
    return DEFAULT_RETRY_AFTER


def classify(status, body, *, is_write: bool) -> "Classified | None":
    """Map ``(status, body)`` to a :class:`Classified`, or ``None`` on success.

    ``is_write`` selects the transient shape: a WRITE that faults ambiguously is
    ``DISPOSITION_UNKNOWN`` (the engine parks + verifies — never blind-retry a place);
    a READ is ``CONNECTION`` (the engine reconnects).
    """
    if status in (200, 201, 202, 204):
        return None
    code = code_of(body)
    msg = _message_of(body)
    transient = Disposition.DISPOSITION_UNKNOWN if is_write else Disposition.CONNECTION

    # status==0 is our client's "no response reached" sentinel (network/timeout).
    if status == 0:
        return Classified(transient, code or "NO_RESPONSE", status,
                          msg or "no response from DNSE")
    # Classify on the code first, then fall back to HTTP status.
    if code == "INVALID_TRADING_TOKEN":
        return Classified(Disposition.AUTH_TOKEN, code, status, msg)
    if code in _AUTH or status in (401, 403):
        return Classified(Disposition.AUTH, code or f"HTTP-{status}", status, msg)
    if code in _RATE_LIMIT or status == 429:
        return Classified(Disposition.RATE_LIMIT, code or "OA-429", status, msg,
                          _retry_after(body))
    if code in TERMINAL_CODES:
        return Classified(Disposition.TERMINAL, code, status, msg)
    if code in NOT_FOUND_CODES:
        return Classified(Disposition.NOT_FOUND, code, status, msg)
    if code in _MARGIN:
        return Classified(Disposition.MARGIN, code, status, msg)
    if code in _SESSION_PLACE:
        return Classified(Disposition.SESSION_REJECT, code, status, msg)
    if code in _SESSION_CANCEL:
        return Classified(Disposition.CONNECTION, code, status, msg)
    if code in _TRANSIENT or status in (500, 502, 503, 504):
        return Classified(transient, code or f"HTTP-{status}", status, msg)
    # Default: a definitive reject (validation, bad symbol, unknown code, 400/404/422).
    return Classified(Disposition.REJECT, code or f"HTTP-{status}", status, msg)
