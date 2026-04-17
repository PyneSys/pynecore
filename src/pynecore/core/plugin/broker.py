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
    """

    on_unexpected_cancel: str = "stop"
    """
    Policy for bot-owned orders that disappear without the bot cancelling them.

    One of ``"stop"`` (graceful stop, default), ``"stop_and_cancel"``
    (stop + cancel remaining bot orders), ``"re_place"`` (auto-replace
    protective orders), or ``"ignore"`` (continue).  Plugin authors may
    override the default; users may further override via plugin config.
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
        Stream order status updates via WebSocket.

        If not implemented, the framework polls :meth:`get_open_orders` on
        each bar.  Return an async iterator of :class:`OrderEvent` objects;
        the plugin is responsible for filling in the Pine identity fields
        (``pine_id``, ``from_entry``, ``leg_type``) on each event.
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
        """
