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
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from pynecore.core.plugin import ConfigT
from pynecore.core.plugin.live_provider import LiveProviderPlugin
from pynecore.core.broker.exceptions import (
    BrokerError,
    ExchangeCapabilityError,
    ExchangeConnectionError,
)
from pynecore.core.broker.models import CancelIntent, DispatchEnvelope

if TYPE_CHECKING:
    from pynecore.core.broker.storage import RunContext
    from pynecore.core.broker.models import (
        ExchangeOrder,
        ExchangePosition,
        ExchangeCapabilities,
        OrderEvent,
    )

__all__ = ['BrokerPlugin']


class BrokerPlugin(LiveProviderPlugin[ConfigT], ABC):
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
    """

    on_unexpected_cancel: str = "stop"
    """
    Policy for bot-owned orders that disappear without the bot cancelling them.

    One of ``"stop"`` (graceful stop, default), ``"stop_and_cancel"``
    (stop + cancel remaining bot orders), ``"re_place"`` (auto-replace
    protective orders), or ``"ignore"`` (continue).  Plugin authors may
    override the default; users may further override via plugin config.
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
        | limit + stop        | STOP_LIMIT   | price    | trigger  |
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

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    async def execute_cancel_all(self, symbol: str | None = None) -> int:
        """Cancel all open orders.  Returns the count cancelled."""
        raise ExchangeCapabilityError("Bulk cancel not supported")

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
