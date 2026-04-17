"""
Startup-time validation of script :class:`ScriptRequirements` against a
plugin's :class:`ExchangeCapabilities`.

Pure function — the Script Runner calls this at broker-mode startup (future
phase) and, on a non-empty error list, refuses to start trading.
"""
from __future__ import annotations

from pynecore.core.broker.models import ScriptRequirements, ExchangeCapabilities

__all__ = ['validate_at_startup']


def validate_at_startup(
        reqs: ScriptRequirements,
        caps: ExchangeCapabilities,
) -> list[str]:
    """
    Return a list of human-readable error strings — empty if all requirements
    are satisfied by the exchange capabilities.

    The rule is simple: if the script uses a Pine parameter, the exchange
    must support the corresponding capability. No runtime "softening" — a
    syntactically-present ``stop=`` keyword means stop orders are required,
    even if the runtime value would end up being ``na`` on every bar.
    Safety-first: better to refuse to start than to fail on the first
    unexpected bar in live trading.
    """
    errors: list[str] = []
    if reqs.stop_orders and not caps.stop_order:
        errors.append(
            "Script uses stop orders, but the exchange doesn't support them."
        )
    if reqs.stop_limit_orders and not caps.stop_limit_order:
        errors.append(
            "Script uses stop-limit orders, but the exchange doesn't support them."
        )
    if reqs.tp_sl_bracket and not caps.tp_sl_bracket:
        errors.append(
            "Script uses TP+SL exit brackets (OCA reduce), but the exchange "
            "plugin doesn't support them. Use a plugin that emulates this, "
            "or modify the script."
        )
    if reqs.trailing_stop and not caps.trailing_stop:
        errors.append(
            "Script uses trailing stops, but the exchange doesn't support them."
        )
    if reqs.exit_orders and not caps.reduce_only:
        errors.append(
            "Script uses strategy.exit / strategy.close, but the exchange "
            "doesn't support reduce-only orders. A later-arriving exit "
            "could flip the book to the other side once the position is "
            "already closed — refuse to start."
        )
    return errors
