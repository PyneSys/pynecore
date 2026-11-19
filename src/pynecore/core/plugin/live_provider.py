from abc import abstractmethod, ABCMeta
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from pynecore.types.ohlcv import OHLCV

from .provider import ProviderPlugin, canonical_timeframe

if TYPE_CHECKING:
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

    :ivar provider_name: Entry-point name for :func:`~pynecore.core.plugin.load_plugin`
                         (e.g. ``"capitalcom"``).
    :ivar symbol: Plugin-native symbol (e.g. ``"EURUSD"``); already passed
                  through :meth:`ProviderPlugin.resolve_symbol`.
    :ivar timeframe: Timeframe in TradingView format (e.g. ``"1D"``), stored with
                     its multiplier (``"D"`` becomes ``"1D"``).
    :ivar config: Pre-loaded plugin config dataclass instance. The chart
                  process runs :func:`~pynecore.core.config.ensure_config` once and hands the
                  resulting instance to every subprocess via spawn args —
                  subprocesses must not touch the TOML file themselves, so
                  every context runs with the configuration the chart read.
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

    def __post_init__(self) -> None:
        # Pine accepts a bare unit (``"D"``), the providers' timeframe tables do not
        if self.timeframe:
            object.__setattr__(self, 'timeframe', canonical_timeframe(self.timeframe))


class LiveProviderPlugin(ProviderPlugin[LiveProviderConfigT], metaclass=ABCMeta):
    """
    WebSocket/streaming data provider extending :class:`ProviderPlugin`.

    Adds real-time data streaming to the offline OHLCV download capability.
    Subclasses must implement connection lifecycle and data streaming methods
    in addition to the :class:`ProviderPlugin` abstract methods.

    The async methods run in a background thread; the framework bridges them
    to the synchronous :class:`~pynecore.core.script_runner.ScriptRunner` via a
    :class:`queue.Queue`.

    Every blocking call these methods hand to a worker thread (``asyncio.to_thread``
    and any other default-executor use) must carry its own enforceable timeout.
    Shutdown cancels the awaiting *task*, never the worker running the call, and
    such a worker is not a daemon thread: one blocking SDK call that never returns
    keeps the interpreter from exiting even after the runner's own thread is gone.
    An SDK without a timeout parameter therefore belongs behind something that can
    be terminated from outside, not in the event loop's executor.
    """

    initial_connect_timeout: float = 30.0
    """Total seconds allowed for the initial connection and its transient retries.

    Providers whose authenticated startup can legitimately include a longer
    venue-requested backoff may raise this bound. A single ``connect()`` attempt
    is also cancelled at the remaining deadline so broker startup cannot proceed
    while the provider is still half-connected.
    """

    reconnect_delay: float = 1.0
    """Initial delay in seconds before a reconnection attempt. Doubles per
    consecutive failure (exponential backoff) up to :attr:`max_reconnect_delay`."""

    max_reconnect_delay: float = 60.0
    """Ceiling in seconds for the exponential reconnect backoff.

    Reconnection is retried indefinitely — a live session must survive
    arbitrarily long network / provider outages and resume on its own once
    connectivity returns — so there is no attempt limit; the backoff simply
    saturates at this delay.
    """

    feed_timeout_bars: int | None = 3
    """Feed-liveness watchdog threshold, in timeframe periods.

    When the provider reports a healthy connection but :meth:`watch_ohlcv`
    has produced no update (closed bar or intra-bar) for this many timeframe
    periods (floored at 90 s) during an open trading session, the framework
    assumes the data feed is dead even though the transport looks alive
    (half-open socket, lost server-side subscription) and drives a full
    ``disconnect()`` → ``connect()`` cycle. ``None`` disables the watchdog —
    only justified for providers whose feed legitimately stays silent for
    long in-session stretches AND whose own liveness machinery already
    covers the dead-feed case.
    """

    history_has_idle_bars: bool = True
    """Whether the venue's history has a bar for a slot without trades.

    Decides what the boundary watchdog does with a slot that got neither a close
    nor a single update by its deadline. ``True``: the history returns a
    zero-volume bar for such a slot, so the stream emits the same bar (a
    zero-volume filler at the previous close). ``False``: the history has no bar
    there, so the stream emits none either and ``bar_index`` does not step, the
    same series a restart would replay from the history.
    """

    history_publish_lag: float = 10.0
    """Seconds after a bar's close until :meth:`backfill_closed_bars` returns it.

    The framework asks the history for a close the stream missed no earlier than
    this long after the scheduled close, so a quiet market costs a few queries per
    bar instead of one per poll. It is also the line between "not yet" and "none":
    an empty answer means the venue has no bar for a slot only when the query was
    asked at least this long after the slot's close. A slot without a forming bar
    therefore waits up to this long for the history; one with a forming bar is
    finalized from it at the close's deadline. Declare the longest delay the venue
    shows, not the typical one: a lag set too short turns a bar the venue publishes
    late into a hole.
    """

    history_revises_before_lag: bool = False
    """Whether the history returns a fresh bar before its final values.

    ``True`` for a venue whose history answers with a slot right at its close and
    revises the bar's values afterwards, within :attr:`history_publish_lag`. The
    framework then takes no bar from an answer asked earlier than the lag past the
    bar's close: it asks again once the lag is over. A bar taken this way is asked
    for one more time, 30 seconds after its close (twice the lag when that is
    longer), and a bar the venue changed after the script processed it is reported
    in the log; the script's values are not recomputed. ``False`` (the default): a bar
    the history returns is final whenever it is asked, so a later slot's bar can also
    prove an earlier one empty before the lag ends.
    """

    stream_closes_match_history: bool = True
    """Whether the stream's closed bar of a slot carries the history's values.

    ``False`` for a venue whose socket candle is built from another price feed than
    its history (a throttled quote stream, for example), so a close from the socket
    can differ from the bar a restart replays from :meth:`backfill_closed_bars`. The
    framework then takes every closed bar from the history, asked once
    :attr:`history_publish_lag` has passed, and treats the socket's close as the
    slot's last forming update. The stream's boundary watchdog hands that update on
    as the close when the history has not delivered the bar by the close's
    deadline. Without a history query the socket's closes go on as they are.
    """

    # --- Connection lifecycle ---

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data source.

        Called for the initial connection AND again on every reconnect (the
        framework drives ``disconnect()`` → ``connect()`` after a connection
        error or a stale feed). Implementations must therefore fully
        re-initialize every piece of connection-scoped state here —
        authentication, server-side subscriptions, partially accumulated
        quote state — rather than assuming a first-call-only environment.
        """

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

    # --- Closed-bar gap recovery ---

    async def backfill_closed_bars(
            self, symbol: str, timeframe: str, since_ms: int,
    ) -> list[OHLCV]:
        """Return closed bars the venue produced after ``since_ms``.

        Historical warmup data is downloaded *before* the live subscription is
        established, so every bar that closes while the subscription is being
        set up falls into a hole: the download no longer covers it and the
        stream does not reach back to it. The framework calls this right after
        the subscription comes up and splices the result between the warmup and
        the stream. A slow handshake is enough to lose a bar — measured on a
        live cTrader lane, a 20 s subscribe silently dropped one M1 bar while
        the same run on Bybit (1 s subscribe) lost none. It is also called
        after every reconnect and for a close the socket missed, and a consumer
        may request it at any time, so it must be a pure query: any number of
        calls, in any order, leave the provider's state as it was. A provider
        therefore recovers no missed close inside its own :meth:`watch_ohlcv`
        or reconnect handling: the framework does it, and one slot is not
        recovered twice.

        Implementations should return only bars that are genuinely closed,
        strictly after ``since_ms``, in ascending timestamp order. Returning
        the still-forming bar would hand the strategy a partial bar as final.
        The bars must come in the venue's own order of publishing: a later bar
        returned while an earlier slot is missing tells the framework the
        earlier slot has no bar. An empty list is the venue's answer that it has
        no such bar yet; the framework takes it as final for a slot only once
        :attr:`history_publish_lag` has passed since the slot's close. A query
        the venue did not answer must raise instead. The framework retries a
        raised query and does not go on over the gap meanwhile.

        The default returns nothing, which keeps the historical behaviour for
        providers that cannot query history on the live connection.

        :param symbol: The symbol in provider-specific format.
        :param timeframe: Timeframe in TradingView format.
        :param since_ms: Opening timestamp of the last bar the framework
            already has; only later bars are wanted.
        :return: Closed bars in ascending order, possibly empty.
        """
        return []

    # --- Reconnection hooks (override for custom behavior) ---

    async def on_disconnect(self) -> None:
        """Called when the connection is unexpectedly lost."""

    async def on_reconnect(self) -> None:
        """Called after a successful reconnection."""

    # --- Shutdown hooks ---

    async def can_shutdown(self) -> bool:
        """
        Whether the provider is ready to shut down.

        Override to delay shutdown while cleanup is in progress
        (e.g. waiting for open orders to fill or positions to close).
        Called every second during the graceful shutdown phase.

        :return: True if ready to shut down, False to keep waiting.
        """
        return True
