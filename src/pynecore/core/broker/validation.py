"""
Startup-time validation for broker mode.

- :func:`validate_at_startup` — script :class:`ScriptRequirements` against a
  plugin's :class:`ExchangeCapabilities`.
- :func:`validate_plugin_contract` — the plugin itself against the
  :class:`~pynecore.core.plugin.broker.BrokerPlugin` authoring contract
  (override pairs, capability-declaration consistency, lifecycle state).

Pure functions — the ``pyne run --broker`` startup path calls them and, on a
non-empty error list, refuses to start trading.
"""
import math
from dataclasses import fields

from pynecore.core.broker.idempotency import WIRE_CLIENT_ORDER_ID_MIN_LEN
from pynecore.core.broker.models import (
    CapabilityLevel,
    ExchangeCapabilities,
    ScriptRequirements,
)
from pynecore.core.plugin.broker import BrokerPlugin

__all__ = ['validate_at_startup', 'validate_plugin_contract']

#: Methods a :class:`~pynecore.core.plugin.broker.PositionPort` implementation
#: must provide. Mirrors the Protocol surface — kept here as data so the
#: contract probe can enumerate it without runtime Protocol introspection.
_POSITION_PORT_METHODS = (
    'fetch_raw_positions',
    'get_volume_quantizer',
    'close_leg',
    'reject_out_of_range',
    'place_leg',
    'amend_bracket',
)

#: Surface a :class:`~pynecore.core.broker.spot_inventory.SpotInventoryPort`
#: implementation must provide — same enumerate-as-data approach as above.
_SPOT_INVENTORY_PORT_METHODS = (
    'fetch_executions',
    'fetch_base_balance',
)
_SPOT_INVENTORY_PORT_ATTRS = (
    'product_id',
    'base_asset',
    'quote_asset',
    'cursor_scope',
    'base_tolerance',
    'settlement_grace_s',
    'position_dust_threshold',
)


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

    :param reqs: The running script's :class:`ScriptRequirements`.
    :param caps: The plugin's advertised :class:`ExchangeCapabilities`.
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
    if reqs.may_go_short and not caps.short_selling.is_supported:
        errors.append(
            "Script passes a constant strategy.short direction to "
            "strategy.entry / strategy.order, but the exchange doesn't "
            "support short selling (spot venue — a negative base position "
            "cannot exist). Remove the short side, or trade on a "
            "margin-capable broker."
        )
    return errors


def validate_plugin_contract(
        plugin: BrokerPlugin,
        *,
        require_account_id: bool = False,
) -> tuple[list[str], list[str]]:
    """
    Probe a broker plugin against the enforceable parts of the
    :class:`~pynecore.core.plugin.broker.BrokerPlugin` authoring contract.

    The abstract ``execute_*`` surface is small, but the real contract lives
    in docstring prose that a new plugin author can silently miss. This probe
    turns the machine-checkable subset into fail-fast startup errors:

    - **Override pairs** — a plugin that overrides
      :meth:`~pynecore.core.plugin.broker.BrokerPlugin.get_residual_orders_after_bracket_attach_reject`
      returns broker refs the engine hands back to
      :meth:`~pynecore.core.plugin.broker.BrokerPlugin.cancel_broker_order_ref`,
      whose default raises :class:`NotImplementedError` — the defensive-close
      recovery loop would crash exactly when it is needed.
    - **Capability declaration consistency** — every
      :class:`ExchangeCapabilities` field must be a :class:`CapabilityLevel`
      (a ``True``/``False`` slips through type checkers on untyped call
      sites); a supported ``watch_orders`` needs the method actually
      overridden; a NATIVE / PARTIAL_NATIVE ``amend_order`` claim needs at
      least one of ``modify_entry`` / ``modify_exit`` overridden (the
      inherited defaults are cancel+recreate, which the declaration denies).
    - **Idempotency floor** — ``idempotency=UNSUPPORTED`` means restart /
      timeout retries can double-fill; live trading is refused
      (:class:`CapabilityLevel` documents this rejection, this is where it
      is enforced).
    - **Client-id budget** — ``client_order_id_max_len`` must be an int of
      at least :data:`~pynecore.core.broker.idempotency.WIRE_CLIENT_ORDER_ID_MIN_LEN`;
      the wire-form client-order-id cannot stay deterministic-and-recognisable
      below that.
    - **Lifecycle** — with ``require_account_id=True`` the plugin must have
      populated ``_account_id`` during authentication *before* the broker
      storage derives the run identity from it; a silent ``"default"``
      would collide every run of the account.
    - **PositionPort surface** — a non-``None``
      :attr:`~pynecore.core.plugin.broker.BrokerPlugin.position_port` must
      carry the full port surface the core
      :class:`~pynecore.core.broker.one_way_emulator.OneWayEmulator` drives.

    Deliberately NOT checked: ``cancel_all`` capability vs
    ``execute_cancel_all`` override. A ``SOFTWARE`` ``cancel_all`` is
    legitimately delivered through the sync engine's diff loop as per-intent
    cancels with the default ``execute_cancel_all`` untouched (Capital.com
    does exactly this).

    Warnings flag legal but degraded setups the author should confirm are
    intentional; they must not block startup.

    :param plugin: The instantiated broker plugin to probe.
    :param require_account_id: ``True`` on the production ``--broker`` path,
        where authentication has already run and the broker storage is about
        to derive the run identity from :attr:`BrokerPlugin.account_id`.
        Leave ``False`` for paths that never open broker storage.
    :return: ``(errors, warnings)`` — human-readable strings; empty lists
        when the plugin conforms.
    """
    errors: list[str] = []
    warnings: list[str] = []
    cls = type(plugin)
    name = cls.__name__

    def overridden(method: str) -> bool:
        return getattr(cls, method) is not getattr(BrokerPlugin, method)

    # --- Capability declaration ---
    caps = plugin.get_capabilities()
    bad_fields: set[str] = set()
    for f in fields(ExchangeCapabilities):
        value = getattr(caps, f.name)
        if not isinstance(value, CapabilityLevel):
            bad_fields.add(f.name)
            errors.append(
                f"{name}.get_capabilities().{f.name} is {value!r} "
                f"({type(value).__name__}) — every capability field must be "
                f"a CapabilityLevel, never a bool or plain string."
            )

    if 'idempotency' not in bad_fields and not caps.idempotency.is_supported:
        errors.append(
            f"{name} declares idempotency=UNSUPPORTED — without client-id "
            f"echo or dedup, restart/timeout retries can double-fill. Live "
            f"trading is refused; declare SOFTWARE and dedup locally (see "
            f"the Capital.com plugin) if the exchange offers nothing."
        )

    if 'watch_orders' not in bad_fields:
        if caps.watch_orders.is_supported and not overridden('watch_orders'):
            errors.append(
                f"{name} declares watch_orders={caps.watch_orders.name} but "
                f"does not override watch_orders() — the base method raises "
                f"NotImplementedError, so the declared order stream cannot "
                f"exist. Either implement the stream or declare UNSUPPORTED."
            )
        elif not overridden('watch_orders'):
            warnings.append(
                f"{name} has no watch_orders() stream: the engine falls back "
                f"to reconcile() polling for fills, and there is NO channel "
                f"for bot-owned-order disappearance detection (manual closes, "
                f"broker liquidations and silent cancels stay invisible). "
                f"Confirm this is acceptable for the venue."
            )

    if ('amend_order' not in bad_fields
            and caps.amend_order in (CapabilityLevel.NATIVE, CapabilityLevel.PARTIAL_NATIVE)
            and not overridden('modify_entry')
            and not overridden('modify_exit')):
        errors.append(
            f"{name} declares amend_order={caps.amend_order.name} but "
            f"overrides neither modify_entry() nor modify_exit() — the "
            f"inherited defaults are cancel+recreate (an unprotected window "
            f"the declaration claims not to have). Override at least one "
            f"with the exchange's in-place amend, or declare SOFTWARE."
        )

    # --- Client-id budget ---
    max_len = plugin.client_order_id_max_len
    if not isinstance(max_len, int) or isinstance(max_len, bool):
        errors.append(
            f"{name}.client_order_id_max_len is {max_len!r} "
            f"({type(max_len).__name__}) — must be an int (the venue's "
            f"client-order-id length limit in characters)."
        )
    elif max_len < WIRE_CLIENT_ORDER_ID_MIN_LEN:
        errors.append(
            f"{name}.client_order_id_max_len={max_len} is below the wire "
            f"floor ({WIRE_CLIENT_ORDER_ID_MIN_LEN}): the deterministic "
            f"wire-form client-order-id needs 14 raw prefix characters plus "
            f"a >=6-character hash tail to keep restart adoption sound. "
            f"Venues with shorter client-id fields are not supportable."
        )

    # --- Override pairs ---
    if (overridden('get_residual_orders_after_bracket_attach_reject')
            and not overridden('cancel_broker_order_ref')):
        errors.append(
            f"{name} overrides get_residual_orders_after_bracket_attach_reject() "
            f"but not cancel_broker_order_ref() — the defensive-close recovery "
            f"loop passes every returned ref to cancel_broker_order_ref(), "
            f"whose default raises NotImplementedError. Override both."
        )

    if not overridden('execute_cancel_with_outcome'):
        warnings.append(
            f"{name} does not override execute_cancel_with_outcome(): every "
            f"cancel disposition collapses to UNKNOWN, so a cancel-tentative "
            f"order can only resolve through a broker-pushed FILL/CANCEL "
            f"event. Override it to classify the exchange's post-cancel "
            f"disposition when the venue makes it readable."
        )

    # --- PositionPort surface ---
    port = plugin.position_port
    if port is not None:
        missing = [m for m in _POSITION_PORT_METHODS
                   if not callable(getattr(port, m, None))]
        if missing:
            errors.append(
                f"{name}.position_port opts into core one-way emulation but "
                f"is missing PositionPort method(s): {', '.join(missing)}. "
                f"The OneWayEmulator drives the plugin purely through this "
                f"surface — implement all of them."
            )

    # --- SpotInventoryPort surface ---
    spot_port = plugin.spot_inventory_port
    if spot_port is not None:
        missing: list[str] = [
            m for m in _SPOT_INVENTORY_PORT_METHODS
            if not callable(getattr(spot_port, m, None))
        ]
        missing.extend(
            a for a in _SPOT_INVENTORY_PORT_ATTRS
            if getattr(spot_port, a, None) is None
        )
        if missing:
            errors.append(
                f"{name}.spot_inventory_port opts into core spot inventory "
                f"but is missing SpotInventoryPort member(s): "
                f"{', '.join(missing)}. The SpotInventoryManager drives the "
                f"venue purely through this surface — implement all of them."
            )
        if plugin.on_inventory_conflict not in ('quarantine', 'halt'):
            errors.append(
                f"{name}.on_inventory_conflict is "
                f"{plugin.on_inventory_conflict!r} — must be 'quarantine' "
                f"or 'halt' (the inventory-conflict policy set is narrower "
                f"than on_unexpected_cancel by design)."
            )
        grace = getattr(spot_port, 'settlement_grace_s', None)
        if (isinstance(grace, bool)
                or not isinstance(grace, (int, float))
                or not math.isfinite(grace)
                or grace < 0):
            errors.append(
                f"{name}.spot_inventory_port.settlement_grace_s is "
                f"{grace!r} — must be a finite non-negative real number "
                f"(a NaN/inf grace would let a confirmed inventory "
                f"conflict stay pending forever while trading continues)."
            )
        if ('short_selling' not in bad_fields
                and caps.short_selling.is_supported):
            errors.append(
                f"{name} declares a spot_inventory_port AND a supported "
                f"short_selling capability — the two are mutually "
                f"exclusive. The spot ledger models long-only exposure "
                f"(a negative base position cannot exist on a spot "
                f"venue); a short-capable venue must not opt into core "
                f"spot inventory."
            )

    # --- Lifecycle ---
    if require_account_id and plugin.account_id == "default":
        errors.append(
            f"{name}.account_id is still the \"default\" sentinel after "
            f"authentication — connect()/session setup must populate "
            f"self._account_id BEFORE broker storage derives the run "
            f"identity, otherwise every run of every account of this "
            f"plugin collides on one identity."
        )

    return errors, warnings
