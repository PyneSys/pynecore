from __future__ import annotations

from abc import abstractmethod, ABCMeta
from dataclasses import dataclass

from pynecore.types.ohlcv import OHLCV

from . import ConfigT
from .provider import ProviderPlugin


@dataclass
class BarUpdate:
    """A single bar update from a live data source."""

    ohlcv: OHLCV
    """The OHLCV data for this update."""

    is_closed: bool
    """True if the bar is final (closed), False for intra-bar updates."""


class LiveProviderPlugin(ProviderPlugin[ConfigT], metaclass=ABCMeta):
    """
    WebSocket/streaming data provider extending :class:`ProviderPlugin`.

    Adds real-time data streaming to the offline OHLCV download capability.
    Subclasses must implement connection lifecycle and data streaming methods
    in addition to the :class:`ProviderPlugin` abstract methods.

    The async methods run in a background thread; the framework bridges them
    to the synchronous :class:`ScriptRunner` via a :class:`queue.Queue`.
    """

    reconnect_delay: float = 1.0
    """Initial delay in seconds before reconnection attempt."""

    max_reconnect_attempts: int = 10
    """Maximum number of consecutive reconnection attempts."""

    # --- Connection lifecycle ---

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data source."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection cleanly."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the connection is currently active."""

    # --- Data streaming ---

    @abstractmethod
    async def watch_ohlcv(self, symbol: str, timeframe: str) -> BarUpdate:
        """
        Wait for and return the next OHLCV update.

        Called in a loop by the framework.  Each call blocks (awaits)
        until new data arrives from the data source.

        :param symbol: The symbol in provider-specific format.
        :param timeframe: Timeframe in TradingView format (e.g. ``"1D"``, ``"1"``, ``"4H"``).
        :return: A :class:`BarUpdate` with the OHLCV data and closed/open status.
        """

    # --- Reconnection hooks (override for custom behavior) ---

    async def on_disconnect(self) -> None:
        """Called when the connection is unexpectedly lost."""

    async def on_reconnect(self) -> None:
        """Called after a successful reconnection."""
