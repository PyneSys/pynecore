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
        pyramiding: int = 1,
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

    :param pyramiding: ``strategy(pyramiding=...)`` from the running script;
        ``1`` (the default) means single-row, where the partial-qty bracket
        path is always safe — *unless* the script also calls
        ``strategy.order()``, which is exempt from the pyramiding cap and
        can open multiple same-id rows on its own. ``pyramiding > 1`` or
        ``reqs.strategy_order=True`` activates the
        :attr:`ExchangeCapabilities.partial_qty_bracket_exit_pyramiding`
        gate: the intent builder's
        ``entry_orders[from_entry]`` lookup keys on a single Pine entry id
        and would silently use the latest row's quantity if multiple rows
        share that id, so the validator refuses to start until the plugin
        explicitly opts the multi-row path in.
    """
    errors: list[str] = []
    if reqs.stop_orders and not caps.stop_order.is_supported:
        errors.append(
            "Script uses stop orders, but the exchange doesn't support them."
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
    if (
        reqs.partial_qty_bracket_exit
        and caps.partial_qty_bracket_exit.is_supported
        and (pyramiding > 1 or reqs.strategy_order)
        and not caps.partial_qty_bracket_exit_pyramiding.is_supported
    ):
        if reqs.strategy_order and pyramiding <= 1:
            trigger = (
                "Script uses strategy.order() (which is exempt from the "
                "pyramiding cap and can open multiple same-id rows) "
            )
        else:
            trigger = "Script combines strategy(pyramiding>1) "
        errors.append(
            trigger +
            "with a partial-qty exit bracket "
            "(strategy.exit(qty=N, from_entry='L', ...) where N is "
            "less than the row total). This exchange plugin's partial-qty "
            "bracket path is single-row only — multiple parent entries "
            "sharing one Pine entry id would be routed against just the "
            "latest row's declared quantity, silently mis-hedging the "
            "older rows. Use strategy(pyramiding=1) without "
            "strategy.order() on this broker, or switch to a plugin that "
            "opts into partial-qty bracket pyramiding support."
        )
    return errors
