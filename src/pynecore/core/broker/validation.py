"""
Startup-time validation of script :class:`ScriptRequirements` against a
plugin's :class:`ExchangeCapabilities`.

Pure function — the Script Runner calls this at broker-mode startup and, on a
non-empty error list, refuses to start trading.
"""
from pynecore.core.broker.models import ExchangeCapabilities, ScriptRequirements

__all__ = ['validate_at_startup']


def validate_at_startup(
        reqs: ScriptRequirements,
        caps: ExchangeCapabilities,
) -> list[str]:
    """
    Return a list of human-readable error strings — empty if all requirements
    are satisfied by the exchange capabilities.

    The rule is simple: if the script uses a Pine parameter, the exchange
    must support the corresponding capability at *any* level (SOFTWARE,
    PARTIAL_NATIVE or NATIVE). Only :data:`~pynecore.core.broker.models.
    CapabilityLevel.UNSUPPORTED` fails — the level distinction is a
    diagnostic, not a stricter contract, because :class:`ScriptRequirements`
    has no channel today to declare a "must be native" requirement.
    Safety-first: better to refuse to start than to fail on the first
    unexpected bar in live trading.
    """
    errors: list[str] = []
    if reqs.stop_orders and not caps.stop_order.is_supported:
        errors.append(
            "Script uses stop orders, but the exchange doesn't support them."
        )
    if reqs.stop_limit_orders and not caps.stop_limit_order.is_supported:
        errors.append(
            "Script uses stop-limit orders, but the exchange doesn't support them."
        )
    if reqs.tp_sl_bracket and not caps.tp_sl_bracket.is_supported:
        errors.append(
            "Script uses TP+SL exit brackets (OCA reduce), but the exchange "
            "plugin doesn't support them. Use a plugin that emulates this, "
            "or modify the script."
        )
    if reqs.trailing_stop and not caps.trailing_stop.is_supported:
        errors.append(
            "Script uses trailing stops, but the exchange doesn't support them."
        )
    if reqs.exit_orders and not caps.reduce_only.is_supported:
        errors.append(
            "Script uses strategy.exit / strategy.close, but the exchange "
            "doesn't support reduce-only orders. A later-arriving exit "
            "could flip the book to the other side once the position is "
            "already closed — refuse to start."
        )
    if reqs.partial_qty_bracket_exit and not caps.partial_qty_bracket_exit.is_supported:
        errors.append(
            "Script calls strategy.exit(qty=N, from_entry='L', ...) with a "
            "bracket parameter (limit/stop/profit/loss/trail_*) where N is "
            "less than the total qty entered under 'L', but the exchange "
            "only supports full-row position-attribute brackets. The plugin "
            "cannot attach TP/SL to a partial quantity, and silently "
            "covering the full row would mis-hedge the strategy. Either "
            "split into (a) strategy.exit(qty=N) without bracket + "
            "(b) strategy.exit with bracket on the full row, or use a "
            "different broker."
        )
    return errors
