"""
Async/sync bridge for live data streaming.

Runs a LiveProviderPlugin's async watch_ohlcv() in a background thread
and yields OHLCV objects to the synchronous ScriptRunner via queue.Queue.

Serial startup: ``live_ohlcv_generator()`` starts the background thread
**and blocks until ``provider.connect()`` succeeds** before returning.
This guarantees that by the time the caller starts consuming the warmup
(local OHLCV file), the WS subscription is already active — any bar that
closes while warmup is running goes into ``bar_queue`` and cannot be
lost. A parallel-start design is appealing on paper but in practice
``provider.connect()`` (REST session + activity-cursor recovery + WS
handshake + subscribe) is slow enough that warmup of a local file
finishes first; the resulting gap means the very next bar close on the
exchange happens with no listener, and that bar is gone forever.

Catch-up: when the consumer first pulls from the live iterator (right
after the file iterator has been exhausted), any closed bars already
sitting in the queue are drained as additional warmup (the script_runner
historical loop processes them with ``barstate.ishistory=True`` and the
strategy still suppressed). Intra-bar updates queued during warmup are
dropped — they would otherwise inflate ``bar_index`` against a still-open
bar that the historical loop is not equipped to dedup. Once the queue
empties, ``LIVE_TRANSITION`` is yielded inline; the script_runner flips
to live mode and every subsequent bar runs against an unsuppressed
strategy.

Dedup: the duplicate-filter for ``last_historical_timestamp`` uses
**strict** less-than for both closed bars and intra-bar updates. The
plugin contract is that ``download_ohlcv`` returns fully-closed bars
only (the Capital.com plugin, for instance, drops the still-forming
last bar in its REST response), so under normal operation the
``ts == last_historical`` case does not arise. The strict-less filter
is kept as defense-in-depth: if a provider violates the contract or a
race serves an in-progress bar from REST, the script_runner live loop
treats the same-timestamp first live update as a continuation of the
last warmup bar — it seeds ``last_bar_timestamp`` from
``last_warmup_timestamp``, so ``bar_index`` does not double-bump and
the bar simply gets one more execution with the refined OHLC values.
"""
import asyncio
import logging
import time
import threading
from collections.abc import Coroutine, Generator
from datetime import datetime
from queue import Queue, Empty, Full
from typing import Any
from zoneinfo import ZoneInfo

from pynecore.core.syminfo import SymInfo
from pynecore.types.ohlcv import OHLCV
from pynecore.core.plugin import is_retryable_provider_error
from pynecore.core.plugin.live_provider import LiveProviderPlugin
from pynecore.core.script_runner import LIVE_TRANSITION
from pynecore.lib.log import broker_info, broker_warning
from pynecore.lib.session import _is_in_session, _is_point_in_session
from pynecore.lib.timeframe import in_seconds

__all__ = ['live_ohlcv_generator', 'download_warmup_in_memory', 'LiveBarStreamer']


class LiveBarStreamer:
    """Non-blocking, thread-safe closed-bar source for security subprocesses.

    Wraps :func:`live_ohlcv_generator` in a background thread that drains its
    output into an internal queue. The security subprocess polls
    :meth:`pop_new_closed_bars` once per chart advance — receiving zero or
    more freshly-closed bars without ever blocking the chart-driven flow.

    Closed bars go to the queue; the latest still-forming (intra-bar) bar is
    kept in a single slot read via :meth:`peek_developing_bar` (overwritten in
    place, never queued). Cross-symbol :func:`request.security` ignores the
    forming slot — it exposes closed bars only — while live
    :func:`request.security_lower_tf` carries the forming bar as the developing
    last element of its intrabar window. The warmup→live transition sentinel is
    dropped (the subprocess does not switch modes mid-run).
    """

    def __init__(self, provider: LiveProviderPlugin, symbol: str, timeframe: str,
                 *, syminfo: SymInfo | None = None,
                 last_historical_timestamp: int | None = None):
        self._provider = provider
        self._symbol = symbol
        self._timeframe = timeframe
        self._syminfo = syminfo
        self._last_historical_timestamp = last_historical_timestamp
        self._queue: Queue[OHLCV] = Queue()
        # Latest still-forming bar, overwritten in place by the drain thread and
        # read by the subprocess via ``peek_developing_bar``. A single reference
        # assignment/read is atomic under the GIL, so no lock is needed.
        self._developing: OHLCV | None = None
        self._stopped = threading.Event()
        self._gen: Generator[OHLCV, None, None] | None = None
        self._thread: threading.Thread | None = None
        # Captures an exception raised by the upstream generator so the
        # next ``pop_new_closed_bars()`` call can surface it to the
        # security subprocess. Without this, a dead WS would just stop
        # delivering bars and the security loop would silently emit ``na``
        # forever while the chart-side liveness check never notices.
        self._drain_error: BaseException | None = None

    def start(self) -> None:
        """Start the background drain thread."""
        if self._thread is not None:
            return
        self._gen = live_ohlcv_generator(
            provider=self._provider,
            symbol=self._symbol,
            timeframe=self._timeframe,
            syminfo=self._syminfo,
            last_historical_timestamp=self._last_historical_timestamp,
        )
        self._thread = threading.Thread(
            target=self._drain, daemon=True, name=f"sec-stream-{self._symbol}",
        )
        self._thread.start()

    def _drain(self) -> None:
        assert self._gen is not None
        try:
            for bar in self._gen:
                if self._stopped.is_set():
                    break
                if bar is LIVE_TRANSITION:
                    continue
                if getattr(bar, 'is_closed', True):
                    # A close supersedes the forming snapshot for its slot, but
                    # only when it is not older than the tracked forming bar.
                    # Providers that close a slot on the next slot's timestamp
                    # (cTrader) — or simple stream reordering — can deliver the
                    # forming tick of slot N+1 ahead of the late close of slot N;
                    # an unconditional clear would erase the newer forming
                    # snapshot and make ``peek_developing_bar`` read ``None``
                    # until the next tick. Timestamp-guarded exactly like the
                    # generator's ``last_forming_bar`` handling.
                    dev = self._developing
                    if dev is None or bar.timestamp >= dev.timestamp:
                        self._developing = None
                    self._queue.put(bar)
                else:
                    # Latest forming (intra-bar) snapshot for the open slot.
                    self._developing = bar
        except Exception as exc:  # noqa: BLE001
            logger.exception("LiveBarStreamer drain raised")
            if not self._stopped.is_set():
                self._drain_error = exc

    def pop_new_closed_bars(self) -> list[OHLCV]:
        """Drain all currently-available closed bars (non-blocking).

        Re-raises any exception captured by the upstream drain thread once
        the queue is empty, so the security subprocess can fail loudly and
        the chart-side liveness check (``proc.is_alive()`` polling) catches
        the dead process instead of silently turning every bar into ``na``.
        """
        out: list[OHLCV] = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except Empty:
                break
        if not out and self._drain_error is not None:
            err = self._drain_error
            # Single-shot raise so a re-tried call (e.g. after restart)
            # would not keep raising forever; ``_drain_error`` stays set
            # only as a flag for ``stop()`` cleanup.
            self._drain_error = None
            raise err
        return out

    def peek_developing_bar(self) -> 'OHLCV | None':
        """Return the latest still-forming bar without consuming it.

        The forming bar is never queued — the drain thread overwrites this slot
        in place as new intra-bar ticks arrive and clears it when the bar
        closes. Live ``request.security_lower_tf`` reads it each chart tick to
        carry the developing intrabar as the live last element of its window;
        the eventual closed bar (delivered via :meth:`pop_new_closed_bars`)
        finalizes that tail.
        """
        return self._developing

    def wait_for_bars(self, timeout: float) -> list[OHLCV]:
        """Block up to ``timeout`` seconds for at least one closed bar.

        Used by the cross-symbol live HTF security loop when the chart
        process has signaled a confirmed period whose bar has not yet been
        published by the upstream WS feed (different exchange / network
        delay). Blocking briefly on the streamer queue avoids advancing the
        chart's ``last_confirmed`` watermark past the security's actual bar
        and emitting ``na`` for that period.

        Drains any further bars that have already accumulated after the
        first one arrives so a single call still returns the full batch.
        Re-raises a captured drain error using the same single-shot
        semantics as :meth:`pop_new_closed_bars`.
        """
        out: list[OHLCV] = []
        try:
            first = self._queue.get(timeout=max(0.0, timeout))
        except Empty:
            if self._drain_error is not None:
                err = self._drain_error
                self._drain_error = None
                raise err
            return out
        out.append(first)
        while True:
            try:
                out.append(self._queue.get_nowait())
            except Empty:
                break
        return out

    def stop(self) -> None:
        """Signal the drain thread to exit and close the upstream generator.

        The drain thread is typically parked inside ``next(self._gen)`` at
        this point. Calling ``self._gen.close()`` from a *different* thread
        while the generator is suspended on a ``yield`` is supported, but
        if the runtime considers the generator to be executing on the drain
        side (race with the moment a bar is being yielded), CPython raises
        ``ValueError: generator already executing``. The drain thread will
        eventually return on its own once ``self._stopped`` is observed, so
        swallow the race here and rely on the ``join`` below to finish
        cleanup.
        """
        self._stopped.set()
        if self._gen is not None:
            try:
                self._gen.close()
            except (RuntimeError, ValueError, GeneratorExit):
                pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)


def download_warmup_in_memory(
        provider: LiveProviderPlugin,
        time_from: datetime,
        time_to: datetime,
) -> list[OHLCV]:
    """
    Download historical OHLCV data into an in-memory list (no file written).

    Used by the security subprocess to fetch warmup bars without creating
    any ``.ohlcv`` file. Captures the records by temporarily redirecting
    :meth:`ProviderPlugin.save_ohlcv_data` to an internal list — the
    plugin's own ``download_ohlcv`` implementation is otherwise unchanged.

    :param provider: A live provider instance (``ohlcv_dir`` may be ``None``).
    :param time_from: Naive UTC start of the warmup window.
    :param time_to: Naive UTC end of the warmup window.
    :return: Fully-closed warmup bars in chronological order.
    """
    captured: list[OHLCV] = []

    original_save = provider.save_ohlcv_data

    def _capture(data):
        if isinstance(data, OHLCV):
            captured.append(data)
        else:
            captured.extend(data)

    provider.save_ohlcv_data = _capture  # type: ignore[method-assign]
    try:
        tf = time_from.replace(tzinfo=None) if time_from.tzinfo else time_from
        tt = time_to.replace(tzinfo=None) if time_to.tzinfo else time_to
        provider.download_ohlcv(tf, tt)
    finally:
        provider.save_ohlcv_data = original_save  # type: ignore[method-assign]

    return captured

logger = logging.getLogger(__name__)


class _Sentinel(BaseException):
    """Marker signaling end of the live stream."""


_SENTINEL = _Sentinel()

# Soft cap for intra-bar updates queued ahead of the consumer. Closed
# bars are never dropped (they go through the blocking ``put`` path),
# but intra-bar updates are advisory and must not accumulate without
# bound when the consumer falls behind — otherwise stale ticks would
# sit ahead of newer closed bars and delay live transition.
_INTRA_BAR_SOFT_CAP = 32

# Sleep cadence during known-closed windows. Long enough to keep the
# TimeoutError handler from spinning at ``effective_timeout`` (~50ms)
# while the boundary deadline is stale, short enough that the next
# session open is noticed within ~30s. Exposed at module scope so tests
# can shrink it without spending the full 30s per gated timeout.
_CLOSED_WINDOW_SLEEP_S = 30.0

# Floor for the feed-staleness threshold (``feed_timeout_bars`` periods,
# but never less than this). Keeps tiny timeframes from flapping into a
# reconnect on a few quiet seconds. Exposed at module scope so tests can
# shrink it.
_FEED_STALE_FLOOR_S = 90.0

# Cadence of WARNING-level idle-synth reminders within one idle streak:
# the first synth of a streak warns, then every Nth; the rest are DEBUG.
_SYNTH_WARN_EVERY = 10


def _warn_this_attempt(attempts: int) -> bool:
    """Reconnect-log rate policy: WARNING for the first attempts of an
    outage, then one WARNING per ten attempts (the rest log at DEBUG).
    With the backoff saturated at ``max_reconnect_delay`` (default 60 s)
    this works out to roughly one console line every ten minutes during
    a long outage instead of one per attempt.
    """
    return attempts <= 3 or attempts % 10 == 0


def _is_transient_connect_error(exc: BaseException) -> bool:
    """Whether a failed initial ``provider.connect()`` is worth retrying.

    Extends the provider-error classification (:func:`is_retryable_provider_error`)
    to the raw socket/TLS layer: a transient ``OSError``-derived fault — most
    notably ``ConnectionResetError`` ([Errno 54]) raised straight out of
    ``asyncio.open_connection`` during the TLS handshake — is a network blip, not
    a user-actionable misconfiguration, so the startup connect should ride it out
    on the backoff path rather than die before the handshake. The ``__cause__`` /
    ``__context__`` chain is walked so a transient still classifies after being
    re-wrapped. Permanent failures (bad symbol / credentials / account mode)
    surface as a non-retryable :class:`ProviderError` and correctly return
    ``False`` here so they keep failing fast.

    :param exc: The exception raised by ``provider.connect()``.
    :return: ``True`` if a retry could plausibly succeed.
    """
    if is_retryable_provider_error(exc):
        return True
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OSError):
            return True
        current = current.__cause__ or current.__context__
    return False


def live_ohlcv_generator(
        provider: LiveProviderPlugin,
        symbol: str,
        timeframe: str,
        syminfo: SymInfo | None = None,
        *,
        last_historical_timestamp: int | None = None,
        shutdown_timeout: float = 120.0,
        event_loop: asyncio.AbstractEventLoop | None = None,
        engine_event_stream: Coroutine[Any, Any, Any] | None = None,
        raise_on_connect_failure: bool = False,
) -> Generator[OHLCV, None, None]:
    """
    Bridge async watch_ohlcv() to a sync Generator[OHLCV, None, None].

    Spawns a background thread running asyncio, collects OHLCV objects
    via queue.Queue, and yields them including intra-bar updates.

    The background thread is started **eagerly** at call time, not on
    first ``next()`` — so the WS subscription is open during warmup and
    no bar is lost in the gap between the REST historical download
    finishing and the consumer reaching the first live update.

    The first batch of bars yielded after the consumer starts pulling is
    the catch-up: closed bars that landed in the queue while the local
    warmup loop was running. ``LIVE_TRANSITION`` is yielded inline once
    the queue empties, so the script_runner can flip to live mode at the
    correct point regardless of how long warmup took.

    :param provider: A LiveProviderPlugin instance (already configured).
    :param symbol: Symbol in provider-specific format.
    :param timeframe: Timeframe in TradingView format.
    :param syminfo: Optional symbol metadata used to gate idle-bar
                    synthesis and reconnect attempts on the trading-
                    session calendar. ``None`` (the default) or an
                    empty ``opening_hours`` preserves the legacy 24/7
                    behaviour where idle synth and reconnect fire on
                    every timeout.
    :param last_historical_timestamp: Timestamp of the last historical bar to avoid duplicates.
    :param shutdown_timeout: Max seconds to wait for graceful shutdown. 0 = wait forever.
    :param event_loop: Optional externally-owned event loop. When supplied, the background
                       thread runs the async loop on it via ``run_until_complete`` instead
                       of ``asyncio.run``. Required for broker mode so that the Order Sync
                       Engine can submit coroutines to the same loop.
    :param engine_event_stream: Optional coroutine (typically
                                ``OrderSyncEngine.run_event_stream()``) to run as a
                                long-lived task alongside the OHLCV watcher. The engine
                                receives its :class:`OrderEvent` stream this way.
    :param raise_on_connect_failure: When True, a ``provider.connect()`` that
                                fails fast during warmup is re-raised here, from
                                the construction call, instead of being buffered
                                for the first bar pull. Broker mode sets this so
                                the real connect error surfaces before
                                ``start_broker()`` can mask it with a generic
                                "live connection not established" reconcile
                                failure. Data-only callers (and the security
                                ``LiveBarStreamer``) leave it False and keep the
                                surface-through-the-iterator behaviour.
    :return: Iterator yielding OHLCV objects (both closed and intra-bar) interleaved
             with a single ``LIVE_TRANSITION`` sentinel marking the warmup→live boundary.
    """
    # Unbounded on the closed-bar path: a bounded queue would block
    # ``bar_queue.put`` when the consumer (``script_runner`` live path)
    # falls behind during heavy fill processing. Since ``_async_loop``
    # runs as an asyncio task on the same event loop as ``_listen_loop``
    # (broker mode pumps both via ``run_coroutine_threadsafe`` onto the
    # main loop), a sync ``Queue.put`` parking on ``not_full`` would
    # stall the entire event loop — listener stops draining the WS,
    # quote ticks pile up in the kernel buffer, ``_tick_volume`` stops
    # advancing, and the watchdog ends up synthesising V=0 bars on a
    # market that is actually trading. Intra-bar updates are bounded
    # by a soft ``qsize`` cap at the put site (see
    # ``_INTRA_BAR_SOFT_CAP`` below) so advisory ticks cannot pile up
    # ahead of closed bars when the consumer lags. The closed-bar
    # ``broker_warning`` in the put path below flags any consumer lag.
    bar_queue: Queue[OHLCV | BaseException] = Queue()
    stop_event = threading.Event()
    # Loop-side mirror of ``stop_event`` so an ``await`` running on the broker
    # loop (the reconnect ``connect()`` / ``on_reconnect()`` handshake) can be
    # abandoned the instant a shutdown is requested from the consumer thread.
    # ``stop_event`` is a ``threading.Event`` — awaiting it from the loop would
    # need a poll; the ``asyncio.Event`` here is *set* on the loop via
    # ``call_soon_threadsafe`` (see ``_consumer``), keeping teardown event-driven
    # and bounded even when a reconnect is in flight. Populated by ``_async_loop``
    # with its running loop + event once it starts.
    shutdown_signal: dict[str, Any] = {}
    # Signalled by ``_async_loop`` once ``provider.connect()`` has either
    # succeeded (WS subscribed, ready to receive bars) or failed (with
    # the exception already pushed into ``bar_queue``). Used to make
    # ``live_ohlcv_generator`` block until the WS is up before returning
    # — see module docstring for why warmup must follow connect.
    connected_event = threading.Event()
    connect_timeout_seconds = 30.0
    # Holds the exception from a fast-failing ``provider.connect()``. Appended
    # before ``connected_event`` is set (so the construction-site read below
    # synchronises on the event and observes it), letting the caller re-raise
    # the real cause instead of returning a generator whose buffered error
    # only surfaces on the first pull — too late, by which point
    # ``start_broker()`` has masked it. See ``raise_on_connect_failure``.
    connect_failure: list[BaseException] = []

    async def _graceful_shutdown():
        """Poll can_shutdown(), then disconnect. Respects shutdown_timeout."""
        logger.info("Graceful shutdown started, polling can_shutdown()...")

        if shutdown_timeout > 0:
            deadline = time.monotonic() + shutdown_timeout
        else:
            deadline = None

        while True:
            try:
                if await provider.can_shutdown():
                    logger.info("Provider ready to shut down")
                    break
            except Exception as e:
                logger.warning("can_shutdown() raised: %s", e)
                break

            if deadline is not None and time.monotonic() >= deadline:
                logger.warning("Shutdown timeout (%.0fs reached), forcing disconnect",
                               shutdown_timeout)
                break

            await asyncio.sleep(1.0)

        try:
            await provider.disconnect()
        except (OSError, RuntimeError):
            pass

    # Idle-bar synthesis: providers whose WS feed only emits ohlc.event when
    # a bar contained at least one tick (Capital.com is the documented
    # case; Bybit/Binance/IB exhibit the same on illiquid moments) leave
    # bar_index frozen across idle TF intervals. The REST history
    # endpoint, however, returns those zero-volume bars on the same
    # calendar boundaries — so on a future restart the historical replay
    # would step through bars the live run never saw, and strategy
    # decisions on the same minute diverge between live and replay.
    # The watchdog below synthesises one zero-volume CLOSED bar
    # (O=H=L=C=last close, V=0) per missed TF boundary so the live
    # stream matches what REST will later return. This lives in the
    # framework, not in each plugin, so providers stay free of bar-rhythm
    # bookkeeping — they only emit when their feed pushes a real bar.
    tf_seconds = max(1, int(in_seconds(timeframe)))
    # Grace past close before declaring a bar missed. The minimum is set
    # to 15s so plugins that recover dropped WS bars via REST have a
    # realistic window to fetch and inject the missing bar before this
    # synth fires — exchanges typically publish a closed bar to REST 5-10s
    # past close, and the plugin watchdog needs a few extra seconds to
    # detect, fetch and inject. The cap (30s) keeps longer TFs from
    # sitting on idle gaps too long.
    bar_grace = max(15.0, min(tf_seconds * 0.5, 30.0))

    # Feed-liveness watchdog threshold: how long ``watch_ohlcv`` may stay
    # silent during an open session before the feed is declared dead and a
    # reconnect is forced even though ``is_connected`` still reports True
    # (half-open socket, lost server-side subscription). ``None`` disables.
    feed_stale_after: float | None = None
    if provider.feed_timeout_bars:
        feed_stale_after = max(
            provider.feed_timeout_bars * tf_seconds, _FEED_STALE_FLOOR_S
        )

    # Resolve the symbol timezone once. ``syminfo.opening_hours`` times are
    # expressed in this zone; epoch timestamps must be converted before
    # session checks.
    _sym_tz: ZoneInfo | None = None
    if syminfo is not None and syminfo.opening_hours:
        try:
            _sym_tz = ZoneInfo(syminfo.timezone)
        except Exception:  # noqa: BLE001
            logger.warning("Unknown syminfo.timezone=%r; session-gate disabled",
                           syminfo.timezone)
    _has_calendar = (
        syminfo is not None
        and bool(syminfo.opening_hours)
        and _sym_tz is not None
    )

    def _market_open_at(epoch_ts: float) -> bool:
        """Slot-aware "is this bar slot in-session?" check.

        Returns True iff the candle ``[epoch_ts, epoch_ts+tf)`` overlaps
        any ``opening_hours`` interval (or unconditionally True for the
        24/7 fallback when the symbol has no calendar). Use this for
        bar-synth decisions and any other slot-aware logic — for the
        point-in-time "is the market open right now?" question use
        :func:`_market_open_now` instead, which does not extend the
        instant by one timeframe.
        """
        if not _has_calendar:
            return True
        assert syminfo is not None and _sym_tz is not None
        local_dt = datetime.fromtimestamp(epoch_ts, tz=_sym_tz)
        return _is_in_session(syminfo.opening_hours, local_dt, tf_seconds)

    def _market_open_now() -> bool:
        """Point-in-time "is the market open right now?" check.

        Does not extend wall-clock by one timeframe, so it does not
        report a session as open one timeframe before its real start.
        Used by the reconnect gate so a long timeframe (e.g. 1h, 1D)
        does not spend the final timeframe of a closed window churning
        through pointless reconnect cycles.
        """
        if not _has_calendar:
            return True
        assert syminfo is not None and _sym_tz is not None
        local_dt = datetime.fromtimestamp(time.time(), tz=_sym_tz)
        return _is_point_in_session(syminfo.opening_hours, local_dt)

    async def _async_loop():
        # Loop-side shutdown signal: ``_consumer`` sets it (cross-thread, via
        # ``call_soon_threadsafe``) alongside ``stop_event`` so a reconnect
        # handshake awaiting ``provider.connect()`` here is abandoned at once
        # rather than blocking teardown for the whole connect timeout.
        stop_aevent = asyncio.Event()
        shutdown_signal['loop'] = asyncio.get_running_loop()
        shutdown_signal['event'] = stop_aevent

        async def _await_or_stop(awaitable) -> bool:
            """Await ``awaitable`` but abandon it if a shutdown is requested.

            Races the awaitable against ``stop_aevent``. Returns ``True`` when
            the shutdown signal won (the awaitable was cancelled — the caller
            must stop reconnecting and let teardown proceed); ``False`` when the
            awaitable finished on its own. Any exception it raised is
            re-propagated so a genuine connect / reconnect failure still drives
            the backoff-retry path.
            """
            task = asyncio.ensure_future(awaitable)
            stop_waiter = asyncio.ensure_future(stop_aevent.wait())
            try:
                await asyncio.wait(
                    {task, stop_waiter}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                stop_waiter.cancel()
                if not stop_waiter.done():
                    try:
                        await stop_waiter
                    except asyncio.CancelledError:
                        pass
            if task.done():
                task.result()  # re-raise a connect/reconnect failure
                return False
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001 - teardown must not surface it
                pass
            return True

        # Declared before ``try`` so the ``finally`` branch can reference
        # it even when ``provider.connect()`` raises before assignment.
        engine_task: asyncio.Task | None = None
        # Last CLOSED bar seen (real or synthesised). The boundary
        # watchdog only arms after the first real bar so we never
        # fabricate state without a baseline.
        last_closed_bar: OHLCV | None = None
        # Latest FORMING (intra-bar) update for the currently open slot.
        # Tracked independently of the intra-bar queue soft-cap so the
        # boundary watchdog can finalise it with its real accumulated
        # OHLCV when no close event arrives — providers that close a bar
        # only on the next bar's timestamp (cTrader) never emit a closing
        # event across a session boundary, so without this the real last
        # bar of a session would be lost and replaced by a frozen V=0
        # synth. Cleared when a closed bar supersedes its slot or once it
        # has been finalised.
        last_forming_bar: OHLCV | None = None
        # Tracks whether the last observed market state was open. Flips to
        # False on the first synth-skip in a closed period (emits a single
        # INFO log) and back to True when a real bar arrives.
        market_open_state = True
        # Wall-clock of the last REAL ``watch_ohlcv`` update (closed bar or
        # intra-bar tick). The feed-liveness watchdog measures staleness
        # against this — ``last_closed_bar`` is unusable for that because
        # idle-bar synthesis keeps rolling it forward on a dead feed.
        # First stamped once ``connect()`` succeeds, then rebased on every
        # reconnect and while the market is known-closed, so the staleness
        # clock only runs against a live, in-session feed.
        last_real_update: float
        # Wall-clock when the current outage began (first failed attempt
        # of a reconnect streak); ``None`` while connected. Lets the
        # rate-limited reconnect logs and the recovery line report how
        # long the feed was actually down.
        outage_started: float | None = None
        # Consecutive idle-synth bars since the last real closed bar.
        # Drives the rate-limited synth warning: first of a streak warns,
        # then every ``_SYNTH_WARN_EVERY``th, the rest log at DEBUG.
        synth_streak = 0
        async def _connect_with_initial_backoff() -> None:
            """Open the first provider connection, riding out transient faults.

            A transient socket/TLS fault on the very first connect — a broker
            edge reset (``ConnectionResetError``), a half-open network path —
            must not kill the live run before the handshake. The same
            classification the in-loop reconnect path applies to a mid-session
            drop is extended here so the startup connect retries with capped
            exponential backoff instead of propagating a raw traceback and
            exiting. Retries are bounded by ``connect_timeout_seconds`` (the
            window the constructing thread waits on ``connected_event``) so a
            genuinely-down venue still surfaces a clean error inside that window
            rather than looping forever, and a permanent misconfiguration
            (non-retryable :class:`ProviderError`) is re-raised on the first
            attempt so it keeps failing fast. The inter-attempt wait is
            event-driven — raced against ``stop_aevent`` — so a shutdown request
            abandons it at once without polling.
            """
            deadline = time.monotonic() + connect_timeout_seconds
            attempt = 0
            delay = provider.reconnect_delay
            while True:
                try:
                    await provider.connect()
                    return
                except BaseException as exc:
                    if not _is_transient_connect_error(exc):
                        raise
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise
                    attempt += 1
                    wait = min(delay, remaining)
                    log = logger.warning if _warn_this_attempt(attempt) else logger.debug
                    log("Initial connect failed (attempt %d): %s; "
                        "retrying in %.1fs", attempt, exc, wait)
                    try:
                        await asyncio.wait_for(stop_aevent.wait(), timeout=wait)
                        # Shutdown requested mid-backoff: surface the last
                        # connect error so teardown runs and the caller unblocks.
                        raise
                    except asyncio.TimeoutError:
                        pass
                    delay = min(delay * 2.0, provider.max_reconnect_delay)

        try:
            broker_info("WS connect starting (warmup blocks until subscribed)")
            try:
                await _connect_with_initial_backoff()
            except BaseException as connect_exc:
                # Record the real cause, then unblock the caller waiting on
                # ``connected_event``. The append happens-before ``set()``,
                # which the caller synchronises on via ``.wait()``, so the
                # construction-site re-raise (when ``raise_on_connect_failure``)
                # observes it. The outer ``except`` still pushes it onto
                # ``bar_queue`` so a post-warmup failure surfaces through the
                # iterator instead of hanging the wait for the full timeout.
                connect_failure.append(connect_exc)
                connected_event.set()
                raise
            watch_symbol = provider.normalize_symbol(symbol)
            broker_info("WS connected and subscribed: %s %s@%s",
                        type(provider).__name__, symbol, timeframe)
            connected_event.set()
            last_real_update = time.time()

            # Broker mode: attach the Order Sync Engine's event stream as
            # a background task so OrderEvents land in its queue without
            # blocking the OHLCV reader.
            if engine_event_stream is not None:
                engine_task = asyncio.create_task(engine_event_stream)

            reconnect_attempts = 0

            async def _handle_connection_error(
                    err: BaseException,
                    attempts: int,
            ) -> tuple[int, bool]:
                """Drive the closed-market wait / reconnect sequence.

                Returns ``(attempts, should_break)``. Used by the
                ``except Exception`` branch below and by the
                synth-skip session-gate when a dead WS is detected
                inside the ``except asyncio.TimeoutError`` handler —
                ``raise`` from one ``except`` handler does not enter
                its sibling handlers, so the original ``raise
                ConnectionError`` pattern would escape past
                ``except Exception`` and kill the live iterator.
                """
                nonlocal market_open_state, last_real_update, outage_started
                # Session-gate: when the market is in a known-closed
                # window (e.g. FX weekend), do not churn through
                # reconnect cycles on a connection error. We sleep ~30s
                # and re-enter the loop without incrementing
                # ``reconnect_attempts`` so the backoff (and the
                # rate-limited logging keyed on the attempt count) does
                # not run away across a long closed window.
                # Logs the connection error once on the closed→still-
                # closed transition for post-mortem visibility.
                # Uses the point-in-time helper (not the slot-aware
                # ``_market_open_at``) so a long timeframe cannot report
                # the market as already open one TF before the real
                # session start.
                if not _market_open_now():
                    if market_open_state:
                        broker_info(
                            "market closed: pausing reconnect attempts "
                            "until next session open "
                            "(last error: %s)",
                            err,
                        )
                        market_open_state = False
                    slept = 0.0
                    while slept < 30.0 and not stop_event.is_set():
                        await asyncio.sleep(min(1.0, 30.0 - slept))
                        slept += 1.0
                    return attempts, stop_event.is_set()
                # No attempt limit: a live session must ride out an
                # arbitrarily long outage (router restart, ISP drop,
                # provider maintenance) and resume on its own. The
                # exponential backoff saturates at
                # ``provider.max_reconnect_delay`` and the per-attempt
                # logging is rate-limited so a multi-hour outage costs
                # a handful of log lines, not one per attempt.
                attempts += 1
                if attempts == 1:
                    outage_started = time.time()
                offline_s = time.time() - (outage_started or time.time())
                log = logger.warning if _warn_this_attempt(attempts) else logger.debug
                log(
                    "Connection error (attempt %d, offline %.0fs): %s",
                    attempts, offline_s, err,
                )
                if await _await_or_stop(provider.on_disconnect()):
                    return attempts, True
                # The exponent is clamped so the power stays a small int;
                # the delay saturates at ``max_reconnect_delay`` anyway.
                delay = min(
                    provider.reconnect_delay * (2 ** min(attempts - 1, 16)),
                    provider.max_reconnect_delay,
                )
                slept = 0.0
                while slept < delay and not stop_event.is_set():
                    await asyncio.sleep(min(0.5, delay - slept))
                    slept += 0.5
                if stop_event.is_set():
                    return attempts, True
                try:
                    if await _await_or_stop(provider.disconnect()):
                        return attempts, True
                except Exception as disc_err:
                    logger.debug(
                        "disconnect() before reconnect raised: %s",
                        disc_err,
                    )
                try:
                    if await _await_or_stop(provider.connect()):
                        return attempts, True
                    if await _await_or_stop(provider.on_reconnect()):
                        return attempts, True
                    # Give the freshly (re)subscribed feed a full
                    # staleness window to deliver before the liveness
                    # watchdog may declare it dead again.
                    last_real_update = time.time()
                    logger.info("Reconnected successfully (attempt %d)", attempts)
                except Exception as reconn_err:
                    log = (logger.warning if _warn_this_attempt(attempts)
                           else logger.debug)
                    log(
                        "Reconnect failed (attempt %d, offline %.0fs): %s",
                        attempts, time.time() - (outage_started or time.time()),
                        reconn_err,
                    )
                return attempts, False

            # Latched dead-WS signal from inside the
            # ``except asyncio.TimeoutError`` handler: ``raise`` from one
            # ``except`` does not enter sibling handlers of the same
            # ``try``, so we cannot trigger ``_handle_connection_error``
            # via ``raise``. The handler sets this flag instead and the
            # top-of-loop dispatch at the next iteration consumes it.
            pending_connection_error: BaseException | None = None

            while not stop_event.is_set():
                # Dispatch any deferred dead-WS signal from the previous
                # iteration's ``except asyncio.TimeoutError`` handler
                # (see ``pending_connection_error`` notes above).
                if pending_connection_error is not None:
                    err = pending_connection_error
                    pending_connection_error = None
                    reconnect_attempts, should_break = (
                        await _handle_connection_error(err, reconnect_attempts)
                    )
                    if should_break:
                        break
                    continue

                # Cap the per-iteration wait at 2 s for the existing
                # is_connected healthcheck cadence; if a missed-bar
                # deadline falls sooner, shorten the wait so synthesis
                # fires promptly when the WS goes idle.
                if last_closed_bar is not None:
                    boundary_deadline = (
                        last_closed_bar.timestamp + 2 * tf_seconds + bar_grace
                    )
                    boundary_remaining = boundary_deadline - time.time()
                else:
                    boundary_remaining = float("inf")
                effective_timeout = min(2.0, max(0.05, boundary_remaining))

                try:
                    bar_update = await asyncio.wait_for(
                        provider.watch_ohlcv(watch_symbol, timeframe),
                        timeout=effective_timeout,
                    )
                    last_real_update = time.time()
                    if reconnect_attempts:
                        # Data-level recovery marker: ``Reconnected
                        # successfully`` above only proves the socket came
                        # back, not that data flows again (a reconnect can
                        # succeed onto a feed that stays silent). This is
                        # the line that closes an outage in the log, so it
                        # is WARNING like the failure lines it answers.
                        logger.warning(
                            "Live feed restored after %d reconnect attempt(s)"
                            " (offline %.0fs)",
                            reconnect_attempts,
                            time.time() - (outage_started or time.time()),
                        )
                        outage_started = None
                    reconnect_attempts = 0

                    # Filter duplicates from the historical phase. Strict
                    # ``<`` for closed bars too: see module docstring on
                    # the open-bar overlap (Capital.com et al.) — the
                    # equal-timestamp case must reach the script_runner
                    # so it can refine the partial last-warmup bar with
                    # the true close, not be silently swallowed here.
                    if last_historical_timestamp is not None:
                        ts = bar_update.timestamp
                        if ts < last_historical_timestamp:
                            continue

                    # In-stream dedup against the boundary watchdog:
                    # if a real ``ohlc.event`` arrives late (past
                    # ``bar_grace`` past close) for a slot we already
                    # synthesised, the synth is already in the queue
                    # and may have been consumed downstream — drop the
                    # late real bar so the consumer never sees two
                    # closed bars on the same TF boundary. Only applies
                    # to ``is_closed=True`` because providers' intra-bar
                    # updates legitimately reuse the last closed bar's
                    # timestamp until the next close arrives.
                    if (bar_update.is_closed
                            and last_closed_bar is not None
                            and bar_update.timestamp <= last_closed_bar.timestamp):
                        continue

                    if bar_update.is_closed:
                        last_closed_bar = bar_update
                        synth_streak = 0
                        # A real close for this slot (or a newer one)
                        # supersedes the tracked forming bar — drop it so
                        # the watchdog never re-finalises an already-closed
                        # slot. Timestamp-guarded so a late/duplicate older
                        # closed bar cannot wipe the current open slot's
                        # forming state.
                        if (last_forming_bar is not None
                                and bar_update.timestamp
                                >= last_forming_bar.timestamp):
                            last_forming_bar = None
                        if not market_open_state:
                            broker_info(
                                "market reopened: resuming live stream "
                                "(first real bar ts=%d)",
                                bar_update.timestamp,
                            )
                            market_open_state = True
                        # Backpressure probe: a non-trivial qsize or put
                        # latency here is the smoking gun for consumer-side
                        # lag stalling the asyncio loop. The queue is
                        # unbounded so ``put`` cannot actually block on
                        # capacity, but cross-thread handoff + GIL pressure
                        # can still take meaningful time when the consumer
                        # is busy. Warn so the live log preserves the
                        # evidence next time a synth-laden run happens.
                        qsize_before = bar_queue.qsize()
                        put_t0 = time.monotonic()
                        bar_queue.put(bar_update)
                        put_ms = (time.monotonic() - put_t0) * 1000.0
                        if qsize_before > 50 or put_ms > 100.0:
                            broker_warning(
                                "bar_queue backpressure: qsize_before=%d "
                                "put_ms=%.1f ts=%d",
                                qsize_before, put_ms, bar_update.timestamp,
                            )
                    else:
                        # Remember the latest forming bar BEFORE the queue
                        # soft-cap. Finalisation state must not depend on
                        # queue admission: when the consumer lags the cap
                        # drops the queued update, but the watchdog must
                        # still be able to close this slot with its real
                        # accumulated OHLCV at the session boundary.
                        last_forming_bar = bar_update
                        # Intra-bar updates are advisory — closed bars
                        # carry authoritative state. With an unbounded
                        # queue, ``put_nowait`` would never raise
                        # ``Full``, so a lagging consumer could let stale
                        # intra-bar items pile up ahead of newer closed
                        # bars and delay live-mode transition or fill
                        # processing. Apply a soft cap based on
                        # ``qsize`` so the closed-bar path retains its
                        # unbounded guarantee while intra-bar growth
                        # stays bounded.
                        if bar_queue.qsize() < _INTRA_BAR_SOFT_CAP:
                            try:
                                bar_queue.put_nowait(bar_update)
                            except Full:
                                pass

                except asyncio.TimeoutError:
                    # Boundary watchdog: if the next-bar close has passed
                    # by more than ``bar_grace`` without a real WS bar,
                    # synthesise a zero-volume filler. Emits exactly one
                    # missed bar per timeout; the next iteration re-checks
                    # against ``time.time()`` and either fills the next
                    # gap or waits for a real push.
                    if (last_closed_bar is not None
                            and time.time()
                            >= last_closed_bar.timestamp
                            + 2 * tf_seconds + bar_grace):
                        synth_ts = last_closed_bar.timestamp + tf_seconds
                        # Session-gate: never synthesise a bar for a slot
                        # that the symbol's opening_hours calendar marks
                        # as closed. Without this gate the framework would
                        # emit V=0 bars across the whole weekend on an FX
                        # symbol whose feed quite legitimately goes silent
                        # at session close. Pine remains paused;
                        # ``bar_index`` does not advance until a real bar
                        # arrives after the next session opens.
                        if not _market_open_at(synth_ts):
                            if market_open_state:
                                broker_info(
                                    "market closed: pausing live stream "
                                    "until next session open "
                                    "(skipped synth ts=%d)",
                                    synth_ts,
                                )
                                market_open_state = False
                            # Dead WS surfaces via the
                            # ``pending_connection_error`` flag, not by
                            # ``raise``: raising from inside this
                            # ``except asyncio.TimeoutError`` handler
                            # escapes past the sibling
                            # ``except Exception`` reconnect block and
                            # would kill the live iterator. WS alive
                            # uses a coarse sleep instead of an
                            # immediate ``continue`` so the loop does
                            # not race ``wait_for`` at the 50ms floor
                            # for the whole closed window
                            # (``boundary_remaining`` is far past, so
                            # ``effective_timeout`` would otherwise pin
                            # to 0.05s → ~20 watch_ohlcv calls/s for
                            # the entire weekend).
                            if not provider.is_connected:
                                pending_connection_error = ConnectionError(
                                    "Provider reports disconnected state"
                                )
                            elif not _market_open_now():
                                # The staleness clock must not run while the
                                # market is closed — a weekend of legitimate
                                # feed silence would otherwise trip the
                                # liveness watchdog right at session open.
                                # Keyed on the CURRENT session state, not the
                                # slot calendar: ``synth_ts`` never advances
                                # in this branch, so after the session
                                # reopens on a dead feed the slot stays
                                # pinned at the pre-close boundary and an
                                # unconditional rebase here would disarm the
                                # watchdog forever.
                                last_real_update = time.time()
                                slept = 0.0
                                while (slept < _CLOSED_WINDOW_SLEEP_S
                                       and not stop_event.is_set()):
                                    step = min(
                                        1.0,
                                        _CLOSED_WINDOW_SLEEP_S - slept,
                                    )
                                    await asyncio.sleep(step)
                                    slept += step
                                if stop_event.is_set():
                                    break
                            else:
                                # Slot pinned in a closed window while the
                                # session is open NOW: no real bar has
                                # advanced the boundary since the close —
                                # e.g. Monday morning on a feed that died
                                # over the weekend. The staleness clock was
                                # last rebased during the closed window, so
                                # it measures in-session silence since the
                                # reopen: trip the liveness watchdog once it
                                # expires, otherwise wait coarsely for the
                                # feed's first post-open bar.
                                if (feed_stale_after is not None
                                        and time.time() - last_real_update
                                        >= feed_stale_after):
                                    pending_connection_error = ConnectionError(
                                        f"feed stale: no data from provider "
                                        f"for "
                                        f"{time.time() - last_real_update:.0f}s "
                                        f"during open session"
                                    )
                                else:
                                    slept = 0.0
                                    while (slept < _CLOSED_WINDOW_SLEEP_S
                                           and not stop_event.is_set()):
                                        step = min(
                                            1.0,
                                            _CLOSED_WINDOW_SLEEP_S - slept,
                                        )
                                        await asyncio.sleep(step)
                                        slept += step
                                    if stop_event.is_set():
                                        break
                            # Skip synth in all branches — we are waiting
                            # out the closed window (WS alive), or we
                            # deferred a connection / stale-feed error so
                            # the next iteration's top-of-loop dispatch
                            # drives the reconnect path.
                            continue
                        # Real-data finalisation: if the provider already
                        # delivered a forming bar for exactly this slot,
                        # close it with its accumulated OHLCV instead of
                        # fabricating a frozen V=0 filler. Providers that
                        # close a bar only when the next bar's timestamp
                        # arrives (cTrader) never emit a closing event for
                        # the last bar before a session boundary or a feed
                        # gap — the close event simply never comes. Without
                        # this the real last-session bar would be discarded
                        # and replaced by a frozen synth (O=H=L=C=prev
                        # close, V=0), shifting the strategy's view of the
                        # close. The finalised bar is REAL data, so it ends
                        # the idle streak rather than counting as synth, and
                        # keeps its own ``extra_fields`` (ask/spread).
                        if (last_forming_bar is not None
                                and last_forming_bar.timestamp == synth_ts):
                            finalized = last_forming_bar._replace(is_closed=True)
                            last_closed_bar = finalized
                            last_forming_bar = None
                            synth_streak = 0
                            broker_info(
                                "idle-bar finalized forming bar: ts=%d "
                                "close=%s vol=%s (real accumulated data; no "
                                "close event arrived before boundary)",
                                finalized.timestamp, finalized.close,
                                finalized.volume,
                            )
                            bar_queue.put(finalized)
                            continue
                        # Dead feed during an open session: reconnect instead
                        # of synthesising a frozen idle bar. Idle-bar synth is
                        # for a live-but-quiet feed; manufacturing bars on a
                        # dead socket would run the strategy on stale prices
                        # while the real market keeps moving, and the steady
                        # synth cadence keeps the boundary deadline perpetually
                        # "just passed" so the dead-WS check below never fires.
                        # ``is_connected`` only sees the transport, so the
                        # feed-liveness watchdog additionally treats a healthy-
                        # looking connection with no real ``watch_ohlcv`` data
                        # for a whole staleness window as dead (half-open
                        # socket, lost server-side subscription). The
                        # ``_market_open_now()`` guard covers the slot-vs-now
                        # mismatch: a backlog slot can still be in-session
                        # right after the market closed, and the staleness
                        # clock only counts in-session silence.
                        # Defer via the flag — a ``raise`` here would escape the
                        # sibling ``except Exception`` reconnect block (same
                        # reason as the session-gated branch above).
                        if not provider.is_connected:
                            pending_connection_error = ConnectionError(
                                "Provider reports disconnected state"
                            )
                            continue
                        if (feed_stale_after is not None
                                and time.time() - last_real_update
                                >= feed_stale_after
                                and _market_open_now()):
                            pending_connection_error = ConnectionError(
                                f"feed stale: no data from provider for "
                                f"{time.time() - last_real_update:.0f}s "
                                f"during open session"
                            )
                            continue
                        last_close = last_closed_bar.close
                        synth = OHLCV(
                            timestamp=synth_ts,
                            open=last_close,
                            high=last_close,
                            low=last_close,
                            close=last_close,
                            volume=0.0,
                            extra_fields=last_closed_bar.extra_fields,
                            is_closed=True,
                        )
                        last_closed_bar = synth
                        # Explicit log marker so the V=0 in the next
                        # OHLCV line is unambiguously framework synth,
                        # not a provider-side closed bar whose
                        # ``_tick_volume`` happened to be zero. Carries
                        # the synth timestamp + frozen close so a
                        # post-mortem can see exactly which TF slot the
                        # watchdog filled and at what price. Rate-limited
                        # within an idle streak — a legitimately quiet
                        # market can idle for hours and one WARNING per
                        # bar would flood the console; the first synth of
                        # a streak and every ``_SYNTH_WARN_EVERY``th warn,
                        # the rest log at DEBUG.
                        synth_streak += 1
                        synth_log = (
                            broker_warning
                            if (synth_streak == 1
                                or synth_streak % _SYNTH_WARN_EVERY == 0)
                            else logger.debug
                        )
                        synth_log(
                            "idle-bar synth emitted: ts=%d close=%s "
                            "(%d consecutive; no real ohlc.event for "
                            ">= 2*tf+grace)",
                            synth_ts, last_close, synth_streak,
                        )
                        bar_queue.put(synth)
                        continue
                    if not provider.is_connected:
                        # Defer to the top-of-loop dispatch rather than
                        # raising here: a ``raise`` from inside this
                        # ``except asyncio.TimeoutError`` handler escapes past
                        # the sibling ``except Exception`` reconnect block and
                        # would kill the live iterator (same reason as the
                        # session-gated branch above).
                        pending_connection_error = ConnectionError(
                            "Provider reports disconnected state"
                        )
                    elif feed_stale_after is not None:
                        # Feed-liveness watchdog on the regular polling
                        # path (covers the pre-first-bar case too, where
                        # the boundary watchdog is not armed yet). The
                        # staleness clock pauses while the market is
                        # closed: silence is legitimate there, and at
                        # reopen the feed gets a fresh window.
                        if not _market_open_now():
                            last_real_update = time.time()
                        elif (time.time() - last_real_update
                              >= feed_stale_after):
                            pending_connection_error = ConnectionError(
                                f"feed stale: no data from provider for "
                                f"{time.time() - last_real_update:.0f}s "
                                f"during open session"
                            )
                    continue
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    reconnect_attempts, should_break = (
                        await _handle_connection_error(e, reconnect_attempts)
                    )
                    if should_break:
                        break
                    continue

        except Exception as e:
            bar_queue.put(e)
        finally:
            if engine_task is not None and not engine_task.done():
                engine_task.cancel()
                try:
                    await engine_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            await _graceful_shutdown()
            bar_queue.put(_SENTINEL)

    def _thread_target():
        if event_loop is not None:
            if event_loop.is_running():
                # Loop is already driven elsewhere (e.g. the CLI broker
                # event-loop pump). Submit our async worker onto it and
                # block this thread on the resulting future instead of
                # trying to start the loop a second time.
                future = asyncio.run_coroutine_threadsafe(_async_loop(), event_loop)
                future.result()
            else:
                asyncio.set_event_loop(event_loop)
                try:
                    event_loop.run_until_complete(_async_loop())
                finally:
                    # The caller owns the loop; don't close it here.
                    pass
        else:
            asyncio.run(_async_loop())

    # Start the provider thread, then block until ``provider.connect()``
    # has completed (or timed out / failed). The warmup that follows
    # therefore runs against an already-listening WS — bars that close
    # mid-warmup land in ``bar_queue`` and are drained as catch-up at
    # the warmup→live boundary. Without this barrier the eager-start
    # design is racy: a slow connect() can finish AFTER warmup, leaving
    # an unsubscribed window in which the next bar close is lost.
    thread = threading.Thread(target=_thread_target, daemon=True, name="live-provider")
    thread.start()
    if not connected_event.wait(timeout=connect_timeout_seconds):
        broker_info(
            "WS connect did not confirm within %.0fs — proceeding; "
            "any underlying error will surface on the first bar pull",
            connect_timeout_seconds,
        )
    elif connect_failure and raise_on_connect_failure:
        # connect() failed fast during warmup. Surface the REAL cause to the
        # caller now, before it reaches start_broker() (whose reconcile would
        # otherwise mask it). ``connected_event`` is set inside the connect
        # except BEFORE ``_async_loop``'s finally has run, so the producer
        # thread may still be draining ``_graceful_shutdown()`` (can_shutdown
        # poll + disconnect) on the shared broker loop. ``_consumer`` is never
        # created on this path, so nothing else will signal ``stop_event`` or
        # join the thread — and the caller's teardown closes the broker loop
        # right away, which would interrupt that pending shutdown and leak the
        # half-open connection. Signal and join here so the teardown finishes
        # while the loop is still alive.
        stop_event.set()
        join_timeout = (shutdown_timeout + 5.0) if shutdown_timeout > 0 else None
        thread.join(timeout=join_timeout)
        raise connect_failure[0]

    def _consumer() -> Generator[OHLCV, None, None]:
        in_warmup_catchup = True

        try:
            while True:
                if in_warmup_catchup:
                    # Drain bars buffered during warmup without blocking.
                    # An empty queue means warmup catch-up is over: emit
                    # the transition sentinel and switch to live polling.
                    try:
                        item = bar_queue.get_nowait()
                    except Empty:
                        yield LIVE_TRANSITION
                        in_warmup_catchup = False
                        continue

                    if item is _SENTINEL:
                        # Stream ended before any live bar arrived. Still
                        # emit the transition so callers that gate on it
                        # observe a clean warmup→live boundary.
                        yield LIVE_TRANSITION
                        break
                    if isinstance(item, BaseException):
                        raise item

                    # Intra-bar updates queued during warmup are dropped:
                    # the historical loop in script_runner has no intra-bar
                    # path and would treat each tick as a fresh bar, so the
                    # bar_index would inflate against a still-open bar.
                    # The latest tick state is preserved by the provider's
                    # internal _last_bar_ohlcv; new ticks after transition
                    # flow through the live path normally.
                    if not item.is_closed:
                        continue

                    yield item
                else:
                    try:
                        item = bar_queue.get(timeout=1.0)
                    except Empty:
                        if not thread.is_alive():
                            break
                        continue

                    if item is _SENTINEL:
                        break
                    if isinstance(item, BaseException):
                        raise item

                    yield item

        except KeyboardInterrupt:
            logger.info("Live streaming interrupted by user")
        finally:
            stop_event.set()
            # Wake a reconnect handshake blocked on ``provider.connect()`` on the
            # broker loop: setting the threading Event alone is invisible to an
            # in-flight ``await``, so mirror it onto the loop-side asyncio Event
            # (see ``_async_loop._await_or_stop``) or teardown waits out the whole
            # connect timeout before the producer thread can exit.
            _sd_loop = shutdown_signal.get('loop')
            _sd_event = shutdown_signal.get('event')
            if _sd_loop is not None and _sd_event is not None:
                try:
                    _sd_loop.call_soon_threadsafe(_sd_event.set)
                except RuntimeError:
                    # Loop already closed — nothing left to wake.
                    pass
            join_timeout = (shutdown_timeout + 5.0) if shutdown_timeout > 0 else None
            thread.join(timeout=join_timeout)

    return _consumer()
