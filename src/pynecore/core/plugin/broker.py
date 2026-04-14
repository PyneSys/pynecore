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
from pynecore.core.broker.exceptions import ExchangeCapabilityError
from pynecore.core.broker.models import CancelIntent

if TYPE_CHECKING:
    from pynecore.core.broker.models import (
        EntryIntent,
        ExitIntent,
        CloseIntent,
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

    @abstractmethod
    async def execute_entry(self, intent: 'EntryIntent') -> list['ExchangeOrder']:
        """
        Open or add to a position.

        Maps to ``strategy.entry()`` and ``strategy.order()``.

        | Pine params         | order_type   | limit    | stop     |
        |---------------------|--------------|----------|----------|
        | no limit, no stop   | MARKET       | None     | None     |
        | limit only          | LIMIT        | price    | None     |
        | stop only           | STOP         | None     | trigger  |
        | limit + stop        | STOP_LIMIT   | price    | trigger  |
        """

    @abstractmethod
    async def execute_exit(self, intent: 'ExitIntent') -> list['ExchangeOrder']:
        """
        Exit (reduce) a position.  OCA REDUCE semantics expected.

        Maps to ``strategy.exit()``.

        The plugin decides HOW to implement the TP+SL bracket on its exchange:
        native bracket orders, separate orders with monitoring, etc.

        MUST handle: partial fill on one leg adjusts the other.  If the
        exchange cannot support a required combination, raise
        :class:`ExchangeCapabilityError`.
        """

    @abstractmethod
    async def execute_close(self, intent: 'CloseIntent') -> 'ExchangeOrder':
        """
        Close a position with a market order.

        Maps to ``strategy.close()`` / ``strategy.close_all()``.
        """

    @abstractmethod
    async def execute_cancel(self, intent: 'CancelIntent') -> bool:
        """
        Cancel pending order(s).  Returns ``True`` if cancelled.
        """

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    async def execute_cancel_all(self, symbol: str | None = None) -> int:
        """Cancel all open orders.  Returns the count cancelled."""
        raise ExchangeCapabilityError("Bulk cancel not supported")

    # === Modify (upsert/replace) ===

    async def modify_entry(
            self, old_intent: 'EntryIntent', new_intent: 'EntryIntent',
    ) -> list['ExchangeOrder']:
        """
        Modify an existing entry order (price/qty changed).

        Default implementation: cancel + execute.  Plugin authors SHOULD
        override with an atomic amend when the exchange supports it.
        """
        await self.execute_cancel(CancelIntent(
            pine_id=old_intent.pine_id,
            symbol=old_intent.symbol,
        ))
        return await self.execute_entry(new_intent)

    async def modify_exit(
            self, old_intent: 'ExitIntent', new_intent: 'ExitIntent',
    ) -> list['ExchangeOrder']:
        """
        Modify an existing exit bracket (TP/SL price changed).

        Default: cancel + new.  This opens a window without protection —
        plugin authors SHOULD override with an atomic amend when the
        exchange supports it (``editOrder``, Bybit amend, etc.).
        """
        await self.execute_cancel(CancelIntent(
            pine_id=old_intent.pine_id,
            symbol=old_intent.symbol,
            from_entry=old_intent.from_entry,
        ))
        return await self.execute_exit(new_intent)

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

    # === Capabilities ===

    @abstractmethod
    def get_capabilities(self) -> 'ExchangeCapabilities':
        """
        Declare what the exchange supports.

        Called once at startup for validation against script requirements
        (see :func:`~pynecore.core.broker.validation.validate_at_startup`).
        """
