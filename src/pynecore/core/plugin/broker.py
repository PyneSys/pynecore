"""
:class:`BrokerPlugin` — high-level order execution layer.

A broker plugin receives Pine Script *intents* (entry, exit bracket, close,
cancel) and translates them to exchange-specific orders.  The plugin author
decides HOW: native brackets, separate orders with software monitoring,
``reduce_only`` flags, editOrder vs cancel-and-replace, etc.

Intents carry full Pine Script identity (``pine_id``, ``from_entry``,
``oca_name``) so the plugin can track order lifecycle and the sync engine
can route :class:`OrderEvent` fills back to the correct Pine trade.

See ``docs/pynecore/plugin-system/broker-plugin-plan.md`` for the full
design, in particular the rationale for the high-level intent API.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Protocol, TypeVar

from pynecore.core.plugin.live_provider import LiveProviderConfig, LiveProviderPlugin
from pynecore.core.broker.exceptions import (
    BrokerError,
    ExchangeCapabilityError,
    ExchangeConnectionError,
    OrderDispositionUnknownError,
)
from pynecore.core.broker.idempotency import CLIENT_ORDER_ID_MAX_LEN
from pynecore.core.broker.models import (
    CancelDispositionOutcome,
    CancelIntent,
    DispatchEnvelope,
)

if TYPE_CHECKING:
    from pynecore.core.broker.storage import RunContext
    from pynecore.core.broker.models import (
        BracketAttachRejectContext,
        ExchangeOrder,
        ExchangePosition,
        ExchangeCapabilities,
        OrderEvent,
        PositionLeg,
    )

__all__ = ['BrokerPlugin', 'PositionPort']


class PositionPort(Protocol):
    """Thin transport a one-way-emulating broker plugin exposes to core.

    A hedging-mode account holds several broker positions ("legs") per symbol;
    Pine sees a single *one-way* position. A plugin that opts into core one-way
    emulation sets :attr:`BrokerPlugin.position_port` to an object (usually
    ``self``) implementing these primitives. The core
    :class:`~pynecore.core.broker.one_way_emulator.OneWayEmulator` owns all
    netting / FIFO / crash-replay logic and drives the plugin purely through
    this surface — each method sends or reads exactly ONE broker entity.
    Plugins that do not emulate (netting-native, or hedging-rejecting) leave
    ``position_port`` ``None`` and the engine uses the regular ``execute_*``
    path unchanged.

    The surface grows per emulation feature (close, then reversal, then bracket
    replication); only the methods a wired feature needs are required.

    Optional capability attribute — read via ``getattr``, absence means
    ``True``:

    * ``supports_partial_leg_close`` — ``False`` when the venue cannot reduce
      a single leg by a partial volume (e.g. Capital.com, whose position
      DELETE is full-row only). The emulator then pre-flights every close /
      reversal plan and atomically skips
      (:class:`~pynecore.core.broker.exceptions.OrderSkippedByPlugin`) any
      plan containing a partial leg slice, before persisting or dispatching
      anything.
    """

    async def fetch_raw_positions(self, symbol: str) -> list[PositionLeg]:
        """All open legs for ``symbol`` (any direction), oldest first. No
        aggregation — the core emulator nets them."""
        ...

    async def get_volume_quantizer(self, symbol: str) -> Callable[[float], int]:
        """A sync ``Pine-units -> broker-grid-int`` quantizer for ``symbol``.

        Returned as a closure so the emulator can snap per-leg volumes in a
        tight loop without an await each call; the plugin owns the broker unit
        (e.g. cTrader centi-units snapped to ``stepVolume``).
        """
        ...

    async def close_leg(
            self, symbol: str, leg_id: str, volume: int, coid: str,
    ) -> None:
        """Send ONE reduce/close of ``volume`` (broker-grid units) on broker
        leg ``leg_id`` under client-order-id ``coid``. The resulting fill
        arrives on the regular order-event stream; this just dispatches."""
        ...

    async def reject_out_of_range(
            self, envelope: 'DispatchEnvelope', qty: float,
    ) -> None:
        """Raise the broker's non-halting volume-bounds skip
        (:class:`~pynecore.core.broker.exceptions.OrderSkippedByPlugin`) when
        ``qty`` (Pine units) is below the minimum or above the maximum tradable
        size; return ``None`` when it is in range.

        The core emulator calls this BEFORE a reversal's leg closes so an
        out-of-range residual skips the whole reversal while that is still true,
        never leaving the book half-reduced.
        """
        ...

    async def place_leg(
            self, envelope: 'DispatchEnvelope', qty: float,
    ) -> list[ExchangeOrder]:
        """Open ONE order of ``qty`` (Pine units) for the envelope's
        :class:`EntryIntent` — the residual leg of a reversal, or a plain add.
        Returns the resulting :class:`ExchangeOrder`(s)."""
        ...

    async def amend_bracket(
            self, symbol: str, leg_id: str, *,
            side: str,
            tp_price: float | None,
            sl_price: float | None,
            trail_offset: float | None,
            coid: str,
    ) -> None:
        """Replicate (or clear) a protective bracket on ONE broker leg.

        Sets the take-profit / stop-loss / trailing levels of the exit's bracket
        on broker leg ``leg_id`` under client-order-id ``coid``. Levels are in
        PINE UNITS (absolute prices for ``tp_price`` / ``sl_price``, a price
        distance for ``trail_offset``), exactly as the :class:`ExitIntent`
        carries them — the plugin owns the conversion to its broker grid and any
        broker-specific reduction (e.g. cTrader has no numeric trailing offset,
        so the plugin seeds an absolute trailing anchor and a trailing flag).
        ``side`` is the exit side, supplied because a trail-only bracket needs it
        to place the initial anchor on the correct side.

        Passing ``tp_price`` / ``sl_price`` / ``trail_offset`` all ``None``
        CLEARS the bracket on that one leg: on venues like cTrader the protection
        is a single position attribute that an amend overwrites wholesale, so an
        empty amend wipes it. The core emulator drives this per leg from its
        ownership index, so a clear touches only the legs the cancelled exit owns
        (never the whole position side).

        Must be idempotent on ``coid`` (a restart re-amend with the same levels
        is a broker no-op) and must NOT halt the bot on a leg-already-gone race —
        a ``*_NOT_FOUND`` response normalises to a benign return. A genuine
        rejection propagates as
        :class:`~pynecore.core.broker.exceptions.ExchangeOrderRejectedError` for
        the caller to surface.
        """
        ...


BrokerConfigT = TypeVar('BrokerConfigT', bound=LiveProviderConfig)


class BrokerPlugin(LiveProviderPlugin[BrokerConfigT], ABC):
    """
    High-level order execution layer.

    Subclasses implement the ``execute_*`` methods in whatever way their
    exchange supports. The Order Sync Engine only calls these methods and
    routes back the :class:`OrderEvent` objects they produce — it never
    reaches into exchange-specific APIs itself.

    **Bot-owned-order disappearance detection is a plugin responsibility.**
    The engine deliberately does not diff its in-memory order mapping
    against :meth:`get_open_orders`, because the resource namespaces are
    broker-specific: a Pine entry may live as a working order, an open
    position, or a position-attached bracket on different exchanges, and
    ``get_open_orders`` only covers one of those namespaces. The plugin must
    therefore detect manual closes / broker-side liquidations / silent
    cancels itself (typically a per-poll snapshot of *all* relevant
    namespaces, with a small grace window to absorb in-flight races) and
    report disappearance through :meth:`watch_orders` — either by emitting
    a synthesised ``cancelled`` :class:`OrderEvent` (the engine's event
    router cleans it out of internal tracking) or by raising
    :class:`~pynecore.core.broker.exceptions.UnexpectedCancelError`
    according to the configured :attr:`on_unexpected_cancel` policy. See
    the Capital.com plugin's ``_reconcile_snapshot`` +
    ``_emit_unexpected_cancellations`` for the reference implementation.

    **Transient-fault contract.** A plugin classifies connectivity faults into
    the broker taxonomy: :class:`~pynecore.core.broker.exceptions.ExchangeConnectionError`
    for a drop the engine recovers from by reconnecting, and
    :class:`~pynecore.core.broker.exceptions.OrderDispositionUnknownError` for a
    write whose acknowledgement was lost in flight (the engine parks it and
    matches against :meth:`get_open_orders`). The Order Sync Engine adds a central
    safety net for a plugin that lets a *raw* transient escape — a retryable
    :class:`~pynecore.core.plugin.ProviderError` (transient wire faults should
    subclass it) or a stdlib :class:`ConnectionError` / :class:`TimeoutError`. The
    net behaves differently on a read than on a write:

    - On a per-bar state read (:meth:`get_position` / :meth:`get_open_orders`)
      the untranslated transient — retryable ``ProviderError``,
      ``ConnectionError`` or ``TimeoutError`` — is treated as
      ``ExchangeConnectionError`` and parked: reads are idempotent, so the
      retry-next-bar is always safe. The one-shot startup :meth:`get_balance`
      auth probe is outside this net (it runs before the per-bar engine loop, so
      there is no cycle to park onto); a raw transient there aborts startup.
    - On a direct order write (``execute_*`` / ``modify_*`` /
      :meth:`cancel_broker_order_ref`) it is unrecoverable centrally — the engine
      cannot tell a never-sent request from a landed one — so an untranslated
      retryable ``ProviderError`` or ``ConnectionError`` raises
      :class:`~pynecore.core.broker.exceptions.BrokerManualInterventionError` (a
      controlled halt) rather than risk a duplicate fill. A raw write-side
      ``TimeoutError`` is the exception: the dispatch bridge cannot cancel the
      still-running coroutine, so the order may yet land and the timeout stays
      fatal rather than being latched as a halt.

    The net only catches contract violations; a plugin SHOULD still classify
    explicitly, which keeps the recoverable read path parking and the ambiguous
    write path parked-for-verification instead of halting.
    """

    client_order_id_max_len: int = CLIENT_ORDER_ID_MAX_LEN
    """
    The venue's client-order-id length budget, in characters.

    The default (30) is the canonical
    :mod:`~pynecore.core.broker.idempotency` width and fits every currently
    supported exchange. A plugin for a venue with a shorter client-id field
    (some FIX implementations cap ``ClOrdID`` at 20) overrides this with the
    venue's actual limit; the engine then mints deterministic wire-form ids
    of exactly this length (see the idempotency module docstring). Must be
    at least ``WIRE_CLIENT_ORDER_ID_MIN_LEN`` (20) — the startup contract
    probe (:func:`~pynecore.core.broker.validation.validate_plugin_contract`)
    rejects anything lower.
    """

    defensive_close_resolution_grace_s: float | None = None
    """
    Grace window (seconds) for a defensive close FILL to settle after
    :meth:`~pynecore.core.broker.sync_engine.OrderSyncEngine._handle_bracket_attach_after_fill_reject`
    dispatches it. The engine halts the run when a pending marker
    survives this window — the missing FILL means an operator must
    intervene.

    ``None`` (default) uses the engine's built-in default
    (``DEFENSIVE_CLOSE_RESOLUTION_GRACE_S``, currently 30 seconds).
    Plugins on slow venues with multi-minute post-trade reporting
    latency may override.
    """

    on_unexpected_cancel: str = "stop"
    """
    Policy for bot-owned orders that disappear without the bot cancelling them.

    One of ``"stop"`` (graceful stop, default), ``"stop_and_cancel"``
    (stop + cancel remaining bot orders), ``"re_place"`` (auto-replace
    protective orders), or ``"ignore"`` (continue).  The class-level
    default is the graceful-stop fallback used by test paths that
    construct plugins without the CLI.  In production the value comes
    from the cross-broker ``workdir/config/brokers.toml`` via
    :class:`~pynecore.core.broker.defaults.BrokerDefaults` — the
    ``pyne run --broker`` entry point loads it once and assigns it as
    an instance attribute on the plugin before the script runner
    starts.  The policy is broker-agnostic by design, so plugins do
    not declare it in their own ``Config`` dataclasses.
    """

    _account_id: str | None = None
    """Plugin-qualified broker account identifier.

    Subclasses populate this during the authentication flow
    (``connect()`` / ``create_session()``), e.g.
    ``self._account_id = f"capitalcom-demo-{preferred_account}"``.  The
    :meth:`account_id` property reads from here and falls back to
    ``"default"`` when the plugin has not been authenticated — keeps tests
    and backtest-only paths working without mandating an identity.

    The value is used by the :class:`~pynecore.core.broker.run_identity.RunIdentity`
    to derive the ``run_id`` and ``run_tag``.  It is fixed at run-creation
    time — if the user switches accounts at the broker UI mid-run, the bot
    must be restarted so a new ``run_instance_id`` is allocated.  That is
    the intended safety boundary, not a limitation.
    """

    @property
    def account_id(self) -> str:
        """Plugin-qualified broker account identifier.

        Sync property (not async) by design: the identity must be fixed
        before the broker storage opens a run, and the storage layer is
        sync.  Authentication (which **is** network I/O) populates
        ``self._account_id`` once; the property reads it back without
        making any further calls.

        :return: The populated identifier, or ``"default"`` when the plugin
            has not yet authenticated.  Returning a sentinel rather than
            raising keeps test paths (mock brokers, backtests) simple.
        """
        return self._account_id or "default"

    store_ctx: 'RunContext | None' = None
    """Optional per-run :class:`RunContext` from the unified broker storage.

    The :class:`~pynecore.core.script_runner.ScriptRunner` sets this on the
    plugin after :meth:`~pynecore.core.broker.storage.BrokerStore.open_run`
    returns, so plugin code can perform ``add_ref`` / ``find_by_ref`` /
    ``upsert_order`` / ``log_event`` calls without receiving the context
    as a parameter on every broker API.  ``None`` during test paths or
    when persistence is off; plugin methods should guard accordingly.
    """

    native_failsafe_observed_sink: Callable[..., None] | None = None
    """Optional §2.6.7 broker-native fail-safe recovery feed.

    The :class:`~pynecore.core.script_runner.ScriptRunner` installs a closure
    that routes one broker-observed bracket triple into the Order Sync Engine's
    :meth:`~pynecore.core.broker.sync_engine.OrderSyncEngine.record_native_bracket_observed`
    (it supplies the engine's bar-clock ``now_ms``, so every fail-safe timestamp
    stays on one clock). A plugin's reconcile pass calls this once per live
    position so a parent stuck in ``DEGRADING`` (restart-replayed, or after a
    PUT retry whose success report could not be confirmed directly) flips back
    to ``HEALTHY`` once the broker is observed carrying the desired worst-SL.
    Without the feed the stale-window timer escalates ``DEGRADING -> DEGRADED``,
    blocking new entries / partial brackets until a manual ``reset_to_engine``.

    The callee is keyed by ``parent_entry_dispatch_ref`` (the entry order's
    ``client_order_id``) and the engine no-ops for refs it does not track, so
    the plugin may feed every live position blindly without consulting the
    fail-safe state. ``None`` when persistence is off or the runner has not
    wired it (state-only test paths); the plugin must guard accordingly.

    Signature: ``sink(parent_entry_dispatch_ref, *, stop_level, profit_level,
    trailing_stop)`` — pass ``None`` for fields the broker is not carrying.
    """

    position_port: 'PositionPort | None' = None
    """Optional one-way emulation transport (hedging-mode accounts).

    When non-``None``, the Order Sync Engine routes reducing / closing and
    reversing intents for this plugin through the core
    :class:`~pynecore.core.broker.one_way_emulator.OneWayEmulator` instead of
    the plugin's ``execute_close`` / ``execute_entry``, so the per-leg FIFO
    fan-out and its persist-first crash/replay live in core. The plugin sets
    this (typically to ``self``) once it knows the account is hedging-mode; it
    stays ``None`` for netting accounts and for plugins that do not emulate,
    leaving their existing dispatch path untouched.
    """

    # === High-level order intents ===
    #
    # Every execute_* method takes a :class:`DispatchEnvelope` rather than a
    # bare intent.  The envelope carries the idempotency metadata the plugin
    # needs to allocate stable ``client_order_id`` values via
    # :meth:`DispatchEnvelope.client_order_id`.  The wrapped intent is on
    # ``envelope.intent`` with its original Pine-level fields intact.

    @abstractmethod
    async def execute_entry(self, envelope: 'DispatchEnvelope') -> list['ExchangeOrder']:
        """
        Open or add to a position.

        Maps to ``strategy.entry()`` and ``strategy.order()``. ``envelope.intent``
        is the :class:`EntryIntent`. Use ``envelope.client_order_id(KIND_ENTRY)``
        for the exchange-side client id.

        | Pine params         | order_type   | limit    | stop     |
        |---------------------|--------------|----------|----------|
        | no limit, no stop   | MARKET       | None     | None     |
        | limit only          | LIMIT        | price    | None     |
        | stop only           | STOP         | None     | trigger  |

        A both-set Pine entry (``limit`` AND ``stop``) is not a stop-limit
        order — Pine has none. The sync engine splits it into two OCO legs
        before dispatch: the LIMIT leg arrives here as ``order_type=LIMIT``,
        and if the stop side triggers the engine sends a separate
        ``order_type=MARKET`` entry. The plugin therefore never receives a
        single both-set order — only plain MARKET / LIMIT / STOP.
        """

    @abstractmethod
    async def execute_exit(self, envelope: 'DispatchEnvelope') -> list['ExchangeOrder']:
        """
        Exit (reduce) a position.  OCA REDUCE semantics expected.

        Maps to ``strategy.exit()``. ``envelope.intent`` is the
        :class:`ExitIntent`. Allocate per-leg client ids via
        ``envelope.client_order_id(KIND_EXIT_TP)`` and
        ``envelope.client_order_id(KIND_EXIT_SL)``.

        The plugin decides HOW to implement the TP+SL bracket on its exchange:
        native bracket orders, separate orders with monitoring, etc.

        MUST handle: partial fill on one leg adjusts the other.  If the
        exchange cannot support a required combination, raise
        :class:`ExchangeCapabilityError`.
        """

    @abstractmethod
    async def execute_close(self, envelope: 'DispatchEnvelope') -> 'ExchangeOrder':
        """
        Close a position with a market order.

        Maps to ``strategy.close()`` / ``strategy.close_all()``. Use
        ``envelope.client_order_id(KIND_CLOSE)`` for the exchange-side id.
        """

    @abstractmethod
    async def execute_cancel(self, envelope: 'DispatchEnvelope') -> bool:
        """
        Cancel pending order(s).  Returns ``True`` if cancelled.

        ``envelope.intent`` is the :class:`CancelIntent`. The canonical cancel
        id (``envelope.client_order_id(KIND_CANCEL)``) is primarily useful for
        audit and retry correlation — the actual exchange call typically
        references the existing order by its exchange-side id.
        """

    async def execute_cancel_with_outcome(
            self, envelope: 'DispatchEnvelope',
    ) -> CancelDispositionOutcome:
        """
        Cancel pending order(s) and return a normalized disposition outcome.

        Used by the sync engine's cancel-tentative state machine to drive
        ``reconcile()``'s idempotent cancel-retry loop forward without
        making broker-specific assumptions (e.g., what HTTP 404 means).
        Each plugin override classifies its exchange-side responses into
        the five :class:`CancelDispositionOutcome` categories.

        The default implementation maps the existing :meth:`execute_cancel`
        contract conservatively to
        :attr:`CancelDispositionOutcome.UNKNOWN`:

        - ``execute_cancel`` returns ``True`` →
          :attr:`CancelDispositionOutcome.UNKNOWN`
        - ``execute_cancel`` returns ``False`` →
          :attr:`CancelDispositionOutcome.UNKNOWN`
        - ``execute_cancel`` raises
          :class:`~pynecore.core.broker.exceptions.OrderDispositionUnknownError` →
          :attr:`CancelDispositionOutcome.UNKNOWN`
        - Any other exception propagates.

        Both boolean outcomes collapse to ``UNKNOWN`` because the bool-only
        :meth:`execute_cancel` contract cannot disambiguate the four terminal
        cancel dispositions safely:

        - ``True`` can mean a confirmed cancel OR a benign no-op (no live row
          matched: the parent may have already filled, leaving an open
          position the engine must keep tracking). Mapping ``True`` to
          :attr:`CancelDispositionOutcome.CANCEL_CONFIRMED` would let the
          sync engine abort the partial-bracket legs of a freshly filled
          parent — exactly the opposite of the intended safety behaviour.
        - ``False`` means *"cancel did not land"* (still pending), the
          opposite of the ``STILL_OPEN`` enum semantic, which the sync
          engine treats like a confirmed cancel (legs aborted, mapping
          dropped).

        ``UNKNOWN`` keeps the cancel-tentative armed; the engine retries on
        the next ``reconcile()`` (or resolves via a broker-pushed FILL /
        CANCEL event in the meantime), so progress still happens.

        ``CANCEL_CONFIRMED``, ``ALREADY_FILLED``, ``TOO_LATE_TO_CANCEL`` and
        ``STILL_OPEN`` outcomes are reachable ONLY via plugin override — the
        default cannot disambiguate them from the bool-only contract.
        Plugins that can read the exchange's post-cancel disposition
        (activity history, position diff) SHOULD override and emit the more
        precise outcome.

        :param envelope: Dispatch envelope around the
            :class:`CancelIntent`; ``envelope.intent`` is the cancel
            request, ``envelope.client_order_id(KIND_CANCEL)`` is the
            audit-correlation id.
        :return: Normalized cancel disposition outcome.
        """
        try:
            await self.execute_cancel(envelope)
        except OrderDispositionUnknownError:
            return CancelDispositionOutcome.UNKNOWN
        return CancelDispositionOutcome.UNKNOWN

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    async def execute_cancel_all(self, symbol: str | None = None) -> int:
        """Cancel all open orders.  Returns the count cancelled."""
        raise ExchangeCapabilityError("Bulk cancel not supported")

    # === Defensive-close recovery (bracket attach reject) ===

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def get_residual_orders_after_bracket_attach_reject(
            self, context: 'BracketAttachRejectContext',
    ) -> list[str]:
        """Enumerate broker-side orders the engine must cancel as part of
        defensive recovery after a bracket attach reject.

        Called by the engine's defensive-close path (see
        :class:`~pynecore.core.broker.exceptions.BracketAttachAfterFillRejectedError`
        and the ``defensive-close-pending-lifecycle`` design dossier)
        after the engine has dispatched the defensive close. The returned
        broker-side refs are passed back one-by-one to
        :meth:`cancel_broker_order_ref`.

        Default implementation returns an empty list. Plugins that
        raise :class:`BracketAttachAfterFillRejectedError` MUST override
        this method when their execution model can leave residual
        cancellable orders, namely:

        - Residual unfilled portion of a partial-fill parent entry that
          is NOT auto-cancelled by the exchange when the bracket attach
          fails.
        - Separate TP/SL order entities (when the exchange does not use
          position-attached protective levels).

        Implementations MUST be safe to call repeatedly with the same
        context — startup replay invokes it again so terminal orders may
        legitimately drop out of the list between calls.

        :param context: Recovery context built from the raised
            :class:`BracketAttachAfterFillRejectedError`.
        :return: List of exchange-native order refs to cancel. Empty by
            default; the engine treats an empty list as "no residual
            cleanup required".
        """
        return []

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    async def cancel_broker_order_ref(self, ref: str) -> None:
        """Cancel a broker-side order by its raw exchange ref.

        Used by engine-internal recovery flows that operate on refs
        outside the Pine intent system — currently the defensive-close
        residual-cancel loop after a
        :class:`~pynecore.core.broker.exceptions.BracketAttachAfterFillRejectedError`.

        **Idempotency contract** (MUST be honoured by every plugin that
        overrides this method):

        - "not found" / "already cancelled" / "already filled" responses
          from the exchange MUST be normalized to a successful no-op
          return (NO exception).
        - Transient connectivity failures MUST raise
          :class:`~pynecore.core.broker.exceptions.ExchangeConnectionError`
          or
          :class:`~pynecore.core.broker.exceptions.OrderDispositionUnknownError`
          so the engine can retry on the next reconcile / restart.
        - Any other exchange-side rejection (e.g.
          :class:`~pynecore.core.broker.exceptions.ExchangeOrderRejectedError`)
          MUST propagate — the engine treats it as an operator-attention
          condition and halts.

        The default implementation raises :class:`NotImplementedError`
        — plugins that return non-empty lists from
        :meth:`get_residual_orders_after_bracket_attach_reject` MUST
        also override this method.
        """
        raise NotImplementedError(
            "cancel_broker_order_ref() must be overridden by plugins that "
            "return non-empty lists from "
            "get_residual_orders_after_bracket_attach_reject()"
        )

    # === Modify (upsert/replace) ===

    async def modify_entry(
            self, old: 'DispatchEnvelope', new: 'DispatchEnvelope',
    ) -> list['ExchangeOrder']:
        """
        Modify an existing entry order (price/qty changed).

        Default implementation: cancel + execute.  Plugin authors SHOULD
        override with an atomic amend when the exchange supports it.
        """
        cancel_envelope = DispatchEnvelope(
            intent=CancelIntent(
                pine_id=old.intent.pine_id,
                symbol=old.intent.symbol,
            ),
            run_tag=new.run_tag,
            bar_ts_ms=new.bar_ts_ms,
            retry_seq=new.retry_seq,
            coid_max_len=new.coid_max_len,
        )
        await self.execute_cancel(cancel_envelope)
        return await self.execute_entry(new)

    async def modify_exit(
            self, old: 'DispatchEnvelope', new: 'DispatchEnvelope',
    ) -> list['ExchangeOrder']:
        """
        Modify an existing exit bracket (TP/SL price changed).

        Default: cancel + new.  This opens a window without protection —
        plugin authors SHOULD override with an atomic amend when the
        exchange supports it (``editOrder``, Bybit amend, etc.).
        """
        old_exit = old.intent
        cancel_envelope = DispatchEnvelope(
            intent=CancelIntent(
                pine_id=old_exit.pine_id,
                symbol=old_exit.symbol,
                from_entry=getattr(old_exit, 'from_entry', None),
            ),
            run_tag=new.run_tag,
            bar_ts_ms=new.bar_ts_ms,
            retry_seq=new.retry_seq,
            coid_max_len=new.coid_max_len,
        )
        await self.execute_cancel(cancel_envelope)
        return await self.execute_exit(new)

    # === State queries ===

    @abstractmethod
    async def get_open_orders(self, symbol: str | None = None) -> list['ExchangeOrder']:
        """Fetch all open orders from the exchange."""

    @abstractmethod
    async def get_position(self, symbol: str) -> 'ExchangePosition | None':
        """Fetch current position.  Returns ``None`` for spot markets."""

    @abstractmethod
    async def get_balance(self) -> dict[str, float]:
        """Get available balance per currency."""

    # === Live order stream ===

    def watch_orders(self) -> AsyncIterator['OrderEvent']:
        """
        Stream order status updates.

        If not implemented, the framework polls :meth:`get_open_orders` on
        each bar.  Return an async iterator of :class:`OrderEvent` objects;
        the plugin is responsible for filling in the Pine identity fields
        (``pine_id``, ``from_entry``, ``leg_type``) on each event.

        The stream is the only channel through which the engine learns
        about broker-side state transitions, and the plugin is free to
        emit *synthesised* events alongside any native exchange feed:

        - A ``cancelled`` event for a bot-owned order the plugin itself
          observed disappearing (e.g. via an internal ``/positions`` +
          ``/workingorders`` poll past a grace window) — the engine's
          event router pops the matching ``_order_mapping`` entry exactly
          as it would for a native cancel.
        - Recovery events that backfill state missed during a network
          drop or restart, before the native stream catches up.

        Plugins that decide a disappearance warrants a graceful halt
        instead of a soft cancel should raise
        :class:`~pynecore.core.broker.exceptions.UnexpectedCancelError`
        from the same generator — the engine catches it on the broker
        thread and latches a halt flag so the next runner tick exits via
        its ``finally`` block.
        """
        raise NotImplementedError

    # === Exception mapping ===

    # noinspection PyMethodMayBeStatic
    def _map_exception(self, raw: Exception) -> BrokerError | None:
        """Translate a raw exchange-SDK exception into the broker taxonomy.

        Utility hook for plugin authors — **not** called by the sync engine
        directly. Plugin ``execute_*`` implementations wrap their SDK calls in
        try/except and delegate classification here so exchange-specific
        knowledge stays in one place. Default implementation only handles
        stdlib exceptions common to every plugin: a concrete plugin should
        override to layer in its SDK's error types (``ccxt.AuthenticationError``,
        IB ``errorCode``, etc.) and return ``None`` for anything it doesn't
        recognise so the caller re-raises the original.

        :returns: A :class:`BrokerError` subclass instance if ``raw`` can be
            classified, or ``None`` if the plugin should re-raise as-is.
        """
        if isinstance(raw, ConnectionError):
            return ExchangeConnectionError(str(raw) or "Connection lost")
        return None

    # === Capabilities ===

    @abstractmethod
    def get_capabilities(self) -> 'ExchangeCapabilities':
        """
        Declare what the exchange supports.

        Called once at startup for validation against script requirements
        (see :func:`~pynecore.core.broker.validation.validate_at_startup`).

        Each field is a :class:`~pynecore.core.broker.models.CapabilityLevel`:
        ``UNSUPPORTED`` (rejects scripts that need it),
        ``SOFTWARE`` / ``PARTIAL_NATIVE`` / ``NATIVE`` (all pass validation,
        the level is a diagnostic). The sync engine reads ``NATIVE`` as
        "exchange is authoritative; suppress my fallback" for ``oca_cancel``
        and ``tp_sl_bracket``.
        """
