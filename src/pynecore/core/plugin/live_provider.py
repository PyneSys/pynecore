from __future__ import annotations

from abc import abstractmethod, ABCMeta
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, TypeVar

from pynecore.types.ohlcv import OHLCV

from .provider import ProviderPlugin

if TYPE_CHECKING:
    from pathlib import Path
    from ..syminfo import SymInfo


@dataclass
class LiveProviderConfig:
    """Common base for all live-provider plugin configs.

    Provides the symbol-translation map every live plugin shares so a Pine
    script can keep using TradingView-style symbols (e.g. ``"FX:EURUSD"``)
    even when the running provider exposes a different native identifier
    (e.g. Capital.com's ``"EURUSD"`` epic). Subclasses inherit this field
    via dataclass inheritance and add their own credentials/tunables on top.

    Generated TOML example (a commented default appears in every live
    plugin's config file)::

        # Optional translation map. Keys are TradingView-style symbols as
        # written in your Pine script (``request.security("FX:EURUSD", ...)``);
        # values are the native identifier the plugin sends to the
        # exchange API. Example for Capital.com::
        #
        #     [symbol_map]
        #     "FX:EURUSD" = "EURUSD"
        #     "OANDA:XAUUSD" = "GOLD"
        #
        # When a Pine symbol is not in the map the plugin falls back to
        # ``normalize_symbol()`` (provider-specific prefix-strip etc.).
        #symbol_map = {}
    """

    symbol_map: dict[str, str] = field(default_factory=dict)
    """Optional translation map. Keys are TradingView-style symbols as
    written in Pine scripts (e.g. ``"FX:EURUSD"``); values are the native
    identifier the plugin sends to the exchange API (e.g. ``"EURUSD"``).
    Missing keys fall back to ``ProviderPlugin.normalize_symbol``."""


LiveProviderConfigT = TypeVar('LiveProviderConfigT', bound=LiveProviderConfig)


@dataclass(frozen=True)
class PluginSymbol:
    """Live-mode descriptor for a :func:`request.security` data source.

    Replaces the historical ``.ohlcv`` file path in live runs: the security
    subprocess instantiates ``provider_name``'s :class:`LiveProviderPlugin`
    itself, downloads warmup history in-memory, then streams live bars.
    No intermediate ``.ohlcv`` file is written.

    :ivar provider_name: Entry-point name for :func:`load_plugin`
                         (e.g. ``"capitalcom"``).
    :ivar symbol: Plugin-native symbol (e.g. ``"EURUSD"``); already passed
                  through :meth:`ProviderPlugin.resolve_symbol`.
    :ivar timeframe: Timeframe in TradingView format (e.g. ``"1D"``).
    :ivar config: Pre-loaded plugin config dataclass instance. The chart
                  process runs :func:`ensure_config` once and hands the
                  resulting instance to every subprocess via spawn args —
                  subprocesses must not touch the TOML file themselves
                  because :func:`ensure_config` is not process-safe.
    :ivar time_from: Optional warmup-window start. Defaults to ``None``,
                  which lets the subprocess fall back to its built-in
                  ``_DEFAULT_WARMUP_BARS`` heuristic; the chart passes
                  its own ``--from`` here so security contexts inherit
                  the same look-back range.
    :ivar syminfo: Optional pre-fetched :class:`SymInfo`. The chart
                  process can call ``provider.update_symbol_info()`` once
                  and pass the result here, so the subprocess does not
                  have to repeat the REST round-trip on startup. ``None``
                  means the subprocess will fetch syminfo itself.
    :ivar is_rate_source: When ``True`` the subprocess runs in a stripped
                  "close-only" loop instead of importing and executing the
                  Pine script. Used by the auto-spawned rate-source
                  contexts created from a ``request.security(..., currency=X)``
                  request when no explicit context already covers the
                  required currency pair.
    :ivar ohlcv_dir: Optional path to the chart process's OHLCV data dir.
                  Forwarded to the subprocess so the provider can locate
                  workdir-side resources that live next to the data dir —
                  most notably per-exchange config overrides in
                  ``<workdir>/config/plugins/<provider>.toml`` (e.g. the
                  ``[binance]`` section of ``ccxt.toml``). Without it the
                  subprocess provider runs with default config while the
                  chart side runs with the override, breaking auth and
                  market-type selection for cross-symbol security feeds.
                  ``None`` means the subprocess provider is constructed
                  without an OHLCV directory (no file is written either way).
    """

    provider_name: str
    symbol: str
    timeframe: str
    config: LiveProviderConfig | None = None
    time_from: datetime | None = None
    syminfo: 'SymInfo | None' = None
    is_rate_source: bool = False
    ohlcv_dir: Path | None = None


class LiveProviderPlugin(ProviderPlugin[LiveProviderConfigT], metaclass=ABCMeta):
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
    async def watch_ohlcv(self, symbol: str, timeframe: str) -> OHLCV:
        """
        Wait for and return the next OHLCV update.

        Called in a loop by the framework.  Each call blocks (awaits)
        until new data arrives from the data source.

        :param symbol: The symbol in provider-specific format.
        :param timeframe: Timeframe in TradingView format (e.g. ``"1D"``, ``"1"``, ``"4H"``).
        :return: An :class:`OHLCV` with ``is_closed=True`` for a final bar, ``False`` for intra-bar updates.
        """

    # --- Reconnection hooks (override for custom behavior) ---

    async def on_disconnect(self) -> None:
        """Called when the connection is unexpectedly lost."""

    async def on_reconnect(self) -> None:
        """Called after a successful reconnection."""

    # --- Shutdown hooks ---

    # noinspection PyMethodMayBeStatic
    async def can_shutdown(self) -> bool:
        """
        Whether the provider is ready to shut down.

        Override to delay shutdown while cleanup is in progress
        (e.g. waiting for open orders to fill or positions to close).
        Called every second during the graceful shutdown phase.

        :return: True if ready to shut down, False to keep waiting.
        """
        return True
