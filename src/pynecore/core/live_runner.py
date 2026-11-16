"""
Async/sync bridge for live data streaming.

Runs a LiveProviderPlugin's async watch_ohlcv() in a background thread
and yields OHLCV objects to the synchronous ScriptRunner via queue.SimpleQueue.

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

Gap fill: closed bars a provider returns from its history
(:meth:`LiveProviderPlugin.backfill_closed_bars`) enter the stream through the
same closed-bar cursor as the socket's closes — a closed bar is handed on only
when it is newer than every closed bar handed on before it. The startup gap
between the warmup history and the subscription is filled this way, and so is
any later fill a consumer requests through :class:`ClosedBarBackfill`, so a
history bar and the socket's close of the same or a neighbouring slot reach the
consumer once and in order, whichever arrives first. A provider with
:attr:`LiveProviderPlugin.core_stream_recovery` has the gap after every reconnect
and a close its socket missed filled the same way. A history query that fails is
retried; the stream does not hand on a later bar over a gap the venue has not
answered for.

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
import sys
import time
import threading
from collections.abc import Callable, Coroutine, Generator
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime
from queue import Queue, SimpleQueue, Empty
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pynecore.core.syminfo import SymInfo
from pynecore.types.ohlcv import OHLCV, AskOHLC
from pynecore.core.plugin import is_retryable_provider_error
from pynecore.core.plugin.live_provider import LiveProviderPlugin
from pynecore.core.script_runner import LIVE_TRANSITION
from pynecore.core.live_timing import close_grace_seconds, is_catch_up
from pynecore.lib.log import broker_info, broker_warning
from pynecore.core.session import is_in_session, is_point_in_session
from pynecore.lib.timeframe import _in_seconds

__all__ = ['live_ohlcv_generator', 'download_warmup_in_memory', 'LiveBarStreamer', 'FeedStatus',
           'LiveStop', 'ClosedBarBackfill']


class LiveStop:
    """A stop request for a live stream, settable from any thread.

    :func:`live_ohlcv_generator` waits for the next update without a deadline, so a
    front end that cannot interrupt the consuming thread with a signal (an IDE bridge
    reading its stop command on another thread) hands one of these to the stream.
    :meth:`set` wakes the waiting consumer at once; the iterator then ends like an
    exhausted one and tears the stream down.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._is_set = False
        self._wakers: list[Callable[[], None]] = []

    @property
    def is_set(self) -> bool:
        """Whether the stop was requested."""
        return self._is_set

    def set(self) -> None:
        """Request the stop and wake every stream waiting on it."""
        with self._lock:
            if self._is_set:
                return
            self._is_set = True
            wakers = list(self._wakers)
        for wake in wakers:
            wake()

    def add_waker(self, wake: Callable[[], None]) -> None:
        """Call ``wake`` on :meth:`set`, at once when the stop was already requested."""
        with self._lock:
            if not self._is_set:
                self._wakers.append(wake)
                return
        wake()

    def remove_waker(self, wake: Callable[[], None]) -> None:
        """Stop calling ``wake`` on :meth:`set`."""
        with self._lock:
            if wake in self._wakers:
                self._wakers.remove(wake)


class ClosedBarBackfill:
    """A request line into a running live stream's closed-bar gap fill.

    :func:`live_ohlcv_generator` fills the gap between the warmup history and the
    subscription from the provider's history at startup. A consumer that misses a
    close later — a ``request.security`` round whose target bar the venue publishes
    late — asks for the same fill again through this object, from any thread. The
    stream runs the query on its own event loop and splices the result through its
    closed-bar cursor, so the bars reach the stream's own output once and in order
    next to the socket's closes; the request only reports how many of them passed.

    The stream binds the object while it runs (:meth:`attach`) and releases it when it
    ends (:meth:`detach`); between the two :meth:`request` returns ``None``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._fill: Callable[[int | None], Coroutine[Any, Any, int]] | None = None

    def attach(self, loop: asyncio.AbstractEventLoop,
               fill: Callable[[int | None], Coroutine[Any, Any, int]]) -> None:
        """Bind the stream that serves the requests.

        :param loop: The event loop the stream's provider runs on.
        :param fill: ``(since_ms) -> accepted count``: the stream's gap fill.
        """
        with self._lock:
            self._loop = loop
            self._fill = fill

    def detach(self) -> None:
        """Release the stream; later requests return ``None``."""
        with self._lock:
            self._loop = None
            self._fill = None

    def request(self, since_ms: int | None) -> 'Future[int] | None':
        """Ask the stream for the closed bars after ``since_ms``.

        :param since_ms: Open of the newest closed bar the consumer holds, ``None``
            when it holds none. The query starts after the newer of this and the
            stream's own cursor.
        :return: The pending fill, resolving to the number of bars the stream handed
            on; ``None`` while no stream is bound.
        """
        with self._lock:
            loop, fill = self._loop, self._fill
        if loop is None or fill is None:
            return None
        query = fill(since_ms)
        try:
            return asyncio.run_coroutine_threadsafe(query, loop)
        except RuntimeError:
            # The loop closed between the bind check and the submission
            query.close()
            return None


@dataclass(frozen=True)
class FeedStatus:
    """A state change of a live feed, reported by :func:`live_ohlcv_generator`.

    :ivar state: ``connecting`` before the first connection attempt, ``live`` once data
        flows (after the subscription, a recovered outage or a reopened market),
        ``reconnecting`` on every reconnect attempt of an outage, ``market_closed``
        while the symbol's session calendar says the market is closed, and
        ``stopped`` when the stream has ended.
    :ivar attempt: The reconnect attempt, counted from 1 within one outage; 0 otherwise.
    :ivar offline_seconds: Seconds since the outage began, on ``reconnecting``.
    :ivar error: The error that started or continues the outage, or closed the market
        wait, when there is one.
    """
    state: Literal['connecting', 'live', 'reconnecting', 'market_closed', 'stopped']
    attempt: int = 0
    offline_seconds: float = 0.0
    error: str | None = None


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

    A close the stream is late with can be asked for from the provider's history
    (:meth:`request_backfill`); the bars it recovers come through the same queue,
    behind the stream's closed-bar cursor.
    """

    def __init__(self, provider: LiveProviderPlugin, symbol: str, timeframe: str,
                 *, syminfo: SymInfo | None = None,
                 last_historical_timestamp: int | None = None,
                 boundary_watchdog: bool = True):
        """
        :param provider: The stream's provider instance.
        :param symbol: Symbol in provider-specific format.
        :param timeframe: Timeframe in TradingView format.
        :param syminfo: The symbol's metadata, for the session-gated parts of the stream.
        :param last_historical_timestamp: Open of the newest bar the caller already holds.
        :param boundary_watchdog: Let the stream close a slot it received no close for
            (:func:`live_ohlcv_generator`); ``False`` when the caller asks the history
            for a late close (:meth:`request_backfill`) instead.
        """
        self._provider = provider
        self._symbol = symbol
        self._timeframe = timeframe
        self._syminfo = syminfo
        self._last_historical_timestamp = last_historical_timestamp
        self._boundary_watchdog = boundary_watchdog
        # Closed bars, and ``None`` as a wake-up without a bar (a finished gap
        # fill): it ends a ``wait_for_bars`` early with an empty batch.
        self._queue: Queue[OHLCV | None] = Queue()
        self._backfill = ClosedBarBackfill()
        # Latest still-forming bar, overwritten in place by the drain thread and
        # read by the subprocess via ``peek_developing_bar``. A single reference
        # assignment/read is atomic under the GIL, so no lock is needed.
        self._developing: OHLCV | None = None
        self._stopped = threading.Event()
        # Wakes the stream's consumer, which the drain thread is parked in, so
        # :meth:`stop` ends the stream instead of racing the running generator.
        self._live_stop = LiveStop()
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
            stop=self._live_stop,
            backfill=self._backfill,
            boundary_watchdog=self._boundary_watchdog,
        )
        thread = threading.Thread(
            target=self._drain, daemon=True, name=f"sec-stream-{self._symbol}",
        )
        self._thread = thread
        thread.start()

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
                bar = self._queue.get_nowait()
            except Empty:
                break
            if bar is not None:
                out.append(bar)
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

        A finished :meth:`request_backfill` ends the wait early, possibly with an
        empty batch, so an empty result does not mean the timeout passed.
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
        if first is not None:
            out.append(first)
        while True:
            try:
                bar = self._queue.get_nowait()
            except Empty:
                break
            if bar is not None:
                out.append(bar)
        return out

    def request_backfill(self, since_ms: int | None) -> 'Future[int] | None':
        """Ask the provider's history for the closed bars after ``since_ms``.

        The recovered bars arrive through :meth:`wait_for_bars` like the stream's
        own closes; a bar the stream has already delivered, or delivers first, is
        not delivered twice. The fill's completion wakes a pending
        :meth:`wait_for_bars`.

        :param since_ms: Open of the newest closed bar the caller holds, ``None``
            when it holds none.
        :return: The pending fill, resolving to the number of bars it delivered;
            ``None`` while the stream is not connected.
        """
        future = self._backfill.request(since_ms)
        if future is not None:
            future.add_done_callback(self._wake_on_fill)
        return future

    def _wake_on_fill(self, _future: 'Future[int]') -> None:
        self._queue.put(None)

    def stop(self) -> None:
        """Signal the drain thread to exit and close the upstream generator.

        The drain thread is typically parked inside ``next(self._gen)``, in the
        stream's wait for its next update, where ``self._gen.close()`` from this
        thread only raises ``ValueError: generator already executing``. The
        :class:`LiveStop` wakes that wait, the stream ends and tears itself down,
        and the drain loop returns. The close is kept for a generator suspended
        on a ``yield``; its race is swallowed and the ``join`` below finishes
        the cleanup.
        """
        self._stopped.set()
        self._live_stop.set()
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


class _StopRequest(BaseException):
    """Marker a :class:`LiveStop` puts on the queue to wake the consumer."""


_STOP_REQUEST = _StopRequest()

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

# How long teardown keeps the provider loop alive for an abandoned
# ``connect()`` to land (see ``_graceful_shutdown``). Bounded on purpose: a
# handshake that never finishes must not hold the shutdown. Exposed at module
# scope so tests can shrink it instead of spending the full window.
_LATE_CONNECT_GRACE_S = 5.0

# How long the reconnect path waits for a CANCELLED provider hook (``connect``
# / ``disconnect`` / ``on_disconnect`` / ``on_reconnect``) to actually finish
# before detaching it. Without a bound a hook that swallows its
# ``CancelledError`` pins ``_async_loop`` before it can reach
# ``_graceful_shutdown``, and with ``shutdown_timeout=0`` (wait forever) the
# consumer's join never returns. Exposed at module scope so tests can shrink it.
_HOOK_CANCEL_GRACE_S = 5.0

# How long the runner-owned loop's exit sweep waits for whatever is still
# pending after ``_graceful_shutdown`` has run (see :func:`_close_owned_loop`).
# Bounded for the same reason as the two windows above: a task that never
# finishes cancelling must not keep the provider thread — and with it the
# consumer's join — alive. Exposed at module scope so tests can shrink it.
_LOOP_SWEEP_GRACE_S = 5.0

# History queries for one missed close, spread between the provider's history
# publish lag and the close's deadline. A quiet market, where the history has
# nothing to add, pays at most this many queries per bar.
_MISSED_CLOSE_PROBES = 3

# Consecutive closes delivered only by the history, for slots the socket showed
# no update of although they had trades, after which the subscription counts as
# dead and is reconnected.
_STALE_HISTORY_CLOSES = 2


def _backoff_delay(provider: LiveProviderPlugin, attempt: int) -> float:
    """The provider's reconnect backoff before retry ``attempt``.

    Starts at :attr:`LiveProviderPlugin.reconnect_delay`, doubles per attempt and
    saturates at :attr:`LiveProviderPlugin.max_reconnect_delay`.

    :param attempt: The retry, counted from 1.
    :return: The delay in seconds.
    """
    # The exponent is clamped so the power stays a small int
    return min(provider.reconnect_delay * (2 ** min(attempt - 1, 16)),
               provider.max_reconnect_delay)


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
    surface as a non-retryable :class:`~pynecore.core.plugin.ProviderError` and correctly return
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


def _discard_task_outcome(task: 'asyncio.Task') -> None:
    """Consume a detached task's result so asyncio never logs it as unretrieved.

    Used for a ``provider.connect()`` task abandoned at its deadline: the
    cancellation may take arbitrarily long (or never complete) in a provider that
    swallows it, so the caller must not await it — but the eventual exception
    still has to be retrieved somewhere.
    """
    if not task.cancelled():
        task.exception()


class _AbandonMarker:
    """Runner-owned "this handshake was given up on" flag.

    Abandonment cannot be read back off the task itself: suppressing a
    ``CancelledError`` and then calling :meth:`asyncio.Task.uncancel` is the
    documented way for a coroutine to adopt a cancellation, and it zeroes the
    :meth:`asyncio.Task.cancelling` counter — a handshake doing that would look
    like an ordinary success and keep its connection open. So the runner records
    the decision itself, before it cancels.
    """

    __slots__ = ('abandoned',)

    def __init__(self) -> None:
        self.abandoned = False


async def _connect_closing_if_abandoned(provider: LiveProviderPlugin,
                                        opening: Coroutine[Any, Any, None],
                                        marker: _AbandonMarker) -> None:
    """Run a connection-opening handshake, closing it again if it was abandoned.

    The runner never awaits a ``connect()`` it gave up on — that unbounded wait
    is exactly what the deadline race exists to avoid — so the cancellation it
    sends instead is best-effort: a provider that swallows it goes on and
    finishes the handshake after teardown has already disconnected, leaving a
    live venue connection behind a run that reported failure. Closing that from
    a done-callback is not enough, because the callback fires while the loop is
    already being torn down (the exit-time task sweep runs right before the loop
    is closed) and nothing awaits what the callback schedules. Doing it INSIDE
    the connect task keeps the close part of the very task that sweep already
    waits for, so it survives however late the handshake lands.

    :param provider: Provider whose connection has to be closed again.
    :param opening: The ``connect()`` coroutine to run.
    :param marker: Flag the runner sets before cancelling this handshake.
    """
    await opening
    if not marker.abandoned:
        return
    try:
        await provider.disconnect()
    except (OSError, RuntimeError):
        pass


def _close_owned_loop(loop: 'asyncio.AbstractEventLoop') -> None:
    """Close the runner-owned provider loop, sweeping leftovers with a bound.

    ``asyncio.run`` does this for us on the way out, but its sweep gathers every
    remaining task with no deadline: a single detached hook that keeps
    suppressing its ``CancelledError`` (see :data:`_HOOK_CANCEL_GRACE_S`) parks
    the provider thread there indefinitely, and since ``shutdown_timeout=0``
    ("wait forever") leaves the consumer's join unbounded too, the whole shutdown
    hangs on that provider's goodwill. Everything that matters — ``can_shutdown``
    polling, ``disconnect()``, the window for late handshakes to close themselves
    — has already run in ``_graceful_shutdown`` by the time we get here, so what
    is still pending gets a grace window and then the loop closes without it.

    :param loop: The loop this runner created and therefore owns.
    """
    try:
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        if pending:
            for task in pending:
                task.cancel()
            loop.run_until_complete(
                asyncio.wait(pending, timeout=_LOOP_SWEEP_GRACE_S))
            for task in pending:
                if task.done() and not task.cancelled():
                    task.exception()  # consume, never logged as unretrieved
            stuck = sum(1 for t in pending if not t.done())
            if stuck:
                logger.warning(
                    "%d provider task(s) ignored their cancellation; closing the "
                    "provider loop without them", stuck)
        # The two wind-down steps ``asyncio.run`` also performs — bounded here,
        # because provider code reaches both. ``shutdown_asyncgens`` runs the
        # ``aclose()`` finalisation of the provider's own feed generators, and
        # ``shutdown_default_executor`` JOINS the default executor's workers, so
        # a hook that put its blocking SDK call there (``asyncio.to_thread``)
        # pins this thread for as long as that call runs — cancelling the task
        # does not stop the worker. CPython bounds the join too (``asyncio.run``
        # passes ``THREAD_JOIN_TIMEOUT``); this is the same idea on the runner's
        # own, much shorter window.
        def _bounded(coro: Coroutine[Any, Any, None], what: str) -> None:
            try:
                loop.run_until_complete(
                    asyncio.wait_for(coro, _LOOP_SWEEP_GRACE_S))
            except TimeoutError:
                logger.warning(
                    "%s did not finish within %.1fs; closing the provider loop "
                    "without it", what, _LOOP_SWEEP_GRACE_S)

        _bounded(loop.shutdown_asyncgens(), "Provider async generator shutdown")
        # Only from 3.12 does the executor step survive being cut short: there
        # the join sits in an ``else`` branch that a cancellation skips, while
        # 3.11 runs it from a ``finally`` no timeout can escape — wrapping THAT
        # in ``wait_for`` would still block right here. On 3.11 the step is
        # therefore skipped outright; ``loop.close()`` below ends in the very
        # same non-blocking ``executor.shutdown(wait=False)`` the bounded path
        # falls back to, it just forgoes the grace window for work that is, by
        # this point, orphaned anyway (its awaiting task is already gone).
        if sys.version_info >= (3, 12):
            _bounded(loop.shutdown_default_executor(), "Executor thread shutdown")
    finally:
        loop.close()


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
        on_interrupt: Callable[[], None] | None = None,
        on_feed_status: Callable[[FeedStatus], None] | None = None,
        stop: LiveStop | None = None,
        backfill: ClosedBarBackfill | None = None,
        boundary_watchdog: bool = True,
) -> Generator[OHLCV, None, None]:
    """
    Bridge async watch_ohlcv() to a sync Generator[OHLCV, None, None].

    Spawns a background thread running asyncio, collects OHLCV objects
    via queue.SimpleQueue, and yields them including intra-bar updates.

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
                                receives its :class:`~pynecore.core.broker.models.OrderEvent`
                                stream this way.
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
    :param on_interrupt: Called when a :class:`KeyboardInterrupt` (a signalled
                         stop) lands inside the iterator, BEFORE the teardown
                         disconnects the provider and ends the producer loop.
                         Broker mode ingests the fills of its in-flight orders
                         here: when the provider is also the broker, those
                         fills and any follow-up write travel on the very
                         connection the teardown closes.
    :param on_feed_status: Called with a :class:`FeedStatus` on every state change of the
                           feed, so a front end can show it without parsing the log. It
                           runs on the provider's background thread, between two awaits of
                           the provider loop: it must return quickly and must not block. An
                           exception it raises is logged and swallowed.
    :param stop: Ends the iterator when set, from any thread, the same way the stream
                 ends on its own: no :class:`KeyboardInterrupt`, the teardown runs. The
                 updates still queued are dropped.
    :param backfill: Bound to the stream while it is connected, so a consumer can ask for
                     the closed bars after its newest one from the provider's history
                     (:meth:`LiveProviderPlugin.backfill_closed_bars`). They are spliced in
                     through the same closed-bar cursor as the startup gap and the socket's
                     closes.
    :param boundary_watchdog: Close a slot the stream received no close for once its
                              grace is over: the forming bar is finalized, an idle slot
                              gets a zero-volume filler, or no bar at all when the
                              venue's history has none for an idle slot
                              (:attr:`LiveProviderPlugin.history_has_idle_bars`). With
                              :attr:`LiveProviderPlugin.core_stream_recovery` the
                              history is asked for the missed close first, and a slot
                              whose query has not been answered waits for the answer.
                              ``False`` for a consumer that recovers a missed close
                              itself through ``backfill`` (a live ``request.security``
                              context): there a close that is only late must not be
                              replaced by a filler before the history has it, and the
                              stream recovers nothing in-stream. The feed-liveness
                              checks run either way.
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
    # A ``SimpleQueue``, because the consumer's wait is where a Ctrl-C lands: its
    # ``get`` blocks in C, so a KeyboardInterrupt, or a repeated one raised while
    # the first is still unwinding, leaves it at once. ``Queue.get`` waits on a
    # ``threading.Condition`` written in Python, and an interrupt between its
    # release and reacquire of the queue's lock ends the ``with`` block on a lock
    # it no longer holds: ``RuntimeError: release unlocked lock`` replaces the
    # KeyboardInterrupt and the run dies with a traceback instead of stopping.
    bar_queue: SimpleQueue[OHLCV | BaseException] = SimpleQueue()
    stop_event = threading.Event()
    # Loop-side mirror of ``stop_event`` so an ``await`` running on the broker
    # loop (the reconnect ``connect()`` / ``on_reconnect()`` handshake) can be
    # abandoned the instant a shutdown is requested from the consumer thread.
    # ``stop_event`` is a ``threading.Event`` — awaiting it from the loop would
    # need a poll; the ``asyncio.Event`` here is *set* on the loop via
    # ``call_soon_threadsafe`` (see ``_consumer``), keeping teardown event-driven
    # and bounded even when a reconnect is in flight. Populated by ``_async_loop``
    # with its running loop + event once it starts.
    shutdown_loop: asyncio.AbstractEventLoop | None = None
    shutdown_event: asyncio.Event | None = None
    # Signalled by ``_async_loop`` once ``provider.connect()`` has either
    # succeeded (WS subscribed, ready to receive bars) or failed (with
    # the exception already pushed into ``bar_queue``). Used to make
    # ``live_ohlcv_generator`` block until the WS is up before returning
    # — see module docstring for why warmup must follow connect.
    connected_event = threading.Event()
    connect_timeout_seconds = max(
        0.1,
        float(getattr(provider, "initial_connect_timeout", 30.0)),
    )
    # How long the construction site reaps the producer thread when STARTUP
    # fails (connect timed out or failed fast). Deliberately independent of
    # ``shutdown_timeout``: that budget belongs to an operator-requested
    # shutdown of a RUNNING bot that may have work to wind down, while here the
    # run never started and there is nothing to drain. The fault being handled
    # is also precisely a producer that may never stop — a ``connect()`` that
    # swallows its cancellation keeps the loop's exit-time task sweep, and with
    # it the thread, alive for the whole grace window — so an unbounded (or
    # two-minute) join would only delay handing the caller the real error.
    startup_join_timeout = min(max(connect_timeout_seconds, 1.0), 10.0)
    # Holds the exception from a fast-failing ``provider.connect()``. Appended
    # before ``connected_event`` is set (so the construction-site read below
    # synchronises on the event and observes it), letting the caller re-raise
    # the real cause instead of returning a generator whose buffered error
    # only surfaces on the first pull — too late, by which point
    # ``start_broker()`` has masked it. See ``raise_on_connect_failure``.
    connect_failure: list[BaseException] = []
    # Connection-opening handshakes abandoned when they outlived their bound:
    # the startup deadline (``_connect_with_initial_backoff``) or a shutdown
    # request during a reconnect (``_await_or_stop``). Each one runs wrapped in
    # :func:`_connect_closing_if_abandoned`, so a provider that swallows its
    # cancellation and finishes the handshake anyway closes it again from inside
    # the task itself. They are never awaited unbounded — that wait is exactly
    # what the deadline race exists to avoid — but teardown gives them a bounded
    # window to run that cleanup while the loop is still ours
    # (:func:`_graceful_shutdown`).
    abandoned_connects: list[asyncio.Task] = []

    # The last state handed to ``on_feed_status``
    reported_state: list[str | None] = [None]

    def _report(state: Literal['connecting', 'live', 'reconnecting', 'market_closed', 'stopped'],
                attempt: int = 0, offline_seconds: float = 0.0,
                error: BaseException | None = None) -> None:
        """Hand a feed state change to ``on_feed_status``, never letting it break the feed.

        A state equal to the last one reported is a repeat, not a change, except for
        ``reconnecting``, whose every attempt carries news.
        """
        if on_feed_status is None:
            return
        if state == reported_state[0] and state != 'reconnecting':
            return
        reported_state[0] = state
        # A front-end hook must not stop the feed, whatever it raises
        # noinspection PyBroadException
        try:
            on_feed_status(FeedStatus(state, attempt, offline_seconds,
                                      None if error is None else str(error)))
        except Exception:
            logger.exception("on_feed_status raised")

    def _signal_loop_shutdown() -> None:
        """Mirror ``stop_event`` onto the loop-side ``asyncio.Event``.

        Setting the threading Event alone is invisible to an in-flight ``await``,
        so a handshake blocked on ``provider.connect()`` would keep the producer
        thread alive for the whole connect timeout. Called from the consumer
        thread — the set itself is scheduled onto the provider loop.
        """
        if shutdown_loop is None or shutdown_event is None:
            return

        def set_shutdown_event(event: asyncio.Event) -> None:
            event.set()

        try:
            shutdown_loop.call_soon_threadsafe(set_shutdown_event, shutdown_event)
        except RuntimeError:
            # Loop already closed — nothing left to wake.
            pass

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

        # Snapshot the abandoned handshakes that are STILL in flight before
        # disconnecting: a provider that swallowed its cancellation can finish
        # the handshake after the disconnect below and re-open what it just
        # closed. Each of them carries its own closing ``disconnect()``
        # (:func:`_connect_closing_if_abandoned`) — what is missing is loop time
        # to run it, which only this coroutine can still hand out.
        in_flight = [t for t in abandoned_connects if not t.done()]
        abandoned_connects.clear()

        try:
            await provider.disconnect()
        except (OSError, RuntimeError):
            pass

        # This coroutine is the last thing that runs while the provider loop is
        # still driven by us: on the ``event_loop=None`` path the loop is closed
        # right after its exit-time task sweep. So give a handshake that is still
        # cleaning up a bounded window to finish here, where it is awaited
        # properly. Whatever exceeds the window keeps its self-closing cleanup
        # and is left to that sweep — a task the sweep gathers (for its own
        # bounded window), unlike anything a done-callback could schedule then.
        if in_flight:
            await asyncio.wait(in_flight, timeout=_LATE_CONNECT_GRACE_S)

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
    tf_seconds = max(1, _in_seconds(timeframe))
    # OHLCV timestamps are Unix milliseconds, so every deadline compared
    # against a bar timestamp is computed on the millisecond axis.
    tf_ms = tf_seconds * 1000
    # Grace past close before declaring a bar missed
    bar_grace_ms = close_grace_seconds(tf_seconds) * 1000.0

    # In-stream recovery of missed closes from the venue's history. Only on a
    # stream that closes its own slots: without the boundary watchdog the
    # consumer recovers a missed close itself (``backfill``). Read with defaults
    # so a duck-typed provider keeps the plain behaviour.
    core_recovery = boundary_watchdog and bool(
        getattr(provider, "core_stream_recovery", False))
    history_has_idle_bars = bool(getattr(provider, "history_has_idle_bars", True))
    publish_lag_ms = float(getattr(provider, "history_publish_lag", 10.0)) * 1000.0

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
        except (ValueError, ZoneInfoNotFoundError):
            logger.warning("Unknown syminfo.timezone=%r; session-gate disabled",
                           syminfo.timezone)
    _has_calendar = (
        syminfo is not None
        and bool(syminfo.opening_hours)
        and _sym_tz is not None
    )

    def _market_open_at(epoch_ms: float) -> bool:
        """Slot-aware "is this bar slot in-session?" check.

        Returns True iff the candle ``[epoch_ms, epoch_ms+tf)`` overlaps
        any ``opening_hours`` interval (or unconditionally True for the
        24/7 fallback when the symbol has no calendar). Use this for
        bar-synth decisions and any other slot-aware logic — for the
        point-in-time "is the market open right now?" question use
        ``_market_open_now()`` instead, which does not extend the
        instant by one timeframe.

        :param epoch_ms: Bar slot start time in Unix milliseconds, the unit
                         every OHLCV timestamp carries.
        """
        if not _has_calendar:
            return True
        assert syminfo is not None and _sym_tz is not None
        local_dt = datetime.fromtimestamp(epoch_ms / 1000.0, tz=_sym_tz)
        return is_in_session(syminfo.opening_hours, local_dt, tf_seconds,
                             syminfo.session_corrections)

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
        return is_point_in_session(syminfo.opening_hours, local_dt,
                                   syminfo.session_corrections)

    async def _async_loop():
        nonlocal shutdown_loop, shutdown_event
        # Loop-side shutdown signal: ``_consumer`` sets it (cross-thread, via
        # ``call_soon_threadsafe``) alongside ``stop_event`` so a reconnect
        # handshake awaiting ``provider.connect()`` here is abandoned at once
        # rather than blocking teardown for the whole connect timeout.
        stop_aevent = asyncio.Event()
        shutdown_loop = asyncio.get_running_loop()
        shutdown_event = stop_aevent

        async def _await_or_stop(awaitable: Coroutine[Any, Any, None], *,
                                 opens_connection: bool = False) -> bool:
            """Await ``awaitable`` but abandon it if a shutdown is requested.

            Races the awaitable against ``stop_aevent``. Returns ``True`` when
            the shutdown signal won (the awaitable was cancelled — the caller
            must stop reconnecting and let teardown proceed); ``False`` when the
            awaitable finished on its own. Any exception it raised is
            re-propagated so a genuine connect / reconnect failure still drives
            the backoff-retry path.

            The wait on the cancelled hook is bounded: cancellation is a
            request, not a guarantee, and a provider hook that swallows its
            ``CancelledError`` would otherwise pin teardown here — before
            ``_graceful_shutdown`` ever runs — for as long as it likes, which
            with ``shutdown_timeout=0`` ("wait forever") means the consumer's
            join never returns either.

            :param awaitable: The provider hook to run.
            :param opens_connection: Whether the hook can leave a live venue
                                     connection behind, so an abandoned one has
                                     to close itself and be handed to teardown.
            """
            marker: _AbandonMarker | None = None
            if opens_connection:
                marker = _AbandonMarker()
                awaitable = _connect_closing_if_abandoned(
                    provider, awaitable, marker)
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
            if marker is not None:
                marker.abandoned = True
            task.cancel()
            done, _ = await asyncio.wait({task}, timeout=_HOOK_CANCEL_GRACE_S)
            if done:
                await asyncio.gather(task, return_exceptions=True)
            else:
                # Detached: its outcome still has to be consumed, and if it can
                # open a connection teardown keeps holding the loop open for the
                # self-closing cleanup it carries.
                task.add_done_callback(_discard_task_outcome)
                if opens_connection:
                    abandoned_connects.append(task)
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
        # Newest slot the framework closed without the stream's close event —
        # the boundary watchdog (from the forming bar or as a V=0 synth) or the
        # history gap fill — and the newest one whose late intra-bar updates
        # were already reported. A feed that stalls past the grace and then
        # flushes its backlog delivers that slot's ticks after it was settled
        # (live: bybit cycle 402, 2026-11-12, ~90 s of silence on ETHUSDT.P):
        # they are dropped here as late, not handed to the consumer as a
        # provider mis-stamping its forming bar.
        settled_ts: int | None = None
        late_ticks_reported_ts: int | None = None
        # Tracks whether the last observed market state was open. Flips to
        # False on the first synth-skip in a closed period (emits a single
        # INFO log) and back to True when a real bar arrives.
        market_open_state = True
        # Backpressure episode: ``True`` from the first lag report until the
        # queue drains, so a lagging consumer is reported once per episode
        # rather than once per bar; the peak depth goes into the drain line.
        backpressure_reported = False
        backpressure_peak = 0
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
        # Newest slot the boundary watchdog decided has no bar, on a venue whose
        # history has none for a slot without trades: the watchdog moves on to
        # the next slot while the closed-bar cursor stays where it is.
        empty_slot_ts: int | None = None
        # Missed-close probe of the slot whose close the stream waits for: when
        # the next history query is due, and whether one has been answered (an
        # empty answer included) or how many failed since the slot became due.
        probe_slot: int | None = None
        probe_next_ms = float("inf")
        probe_answered = False
        probe_failures = 0
        # Newest slot the socket delivered any update for, and the run of
        # consecutive closes only the history delivered although the slot had
        # trades (see ``_STALE_HISTORY_CLOSES``).
        socket_seen_ts: int | None = None
        history_close_streak = 0
        # Socket updates of a slot after the awaited one, held while the awaited
        # slot is not settled: handing one on would move the closed-bar cursor (or
        # the consumer's forming bar) past the awaited slot, and its close could
        # then never be spliced in. Replayed in order once the slot is settled.
        # Closed bars are all kept, a forming slot only by its newest update.
        held_updates: list[OHLCV] = []

        async def _stop_within(seconds: float) -> bool:
            """Wait up to ``seconds`` for a shutdown request.

            :param seconds: The longest wait.
            :return: ``True`` when the shutdown was requested.
            """
            try:
                await asyncio.wait_for(stop_aevent.wait(), timeout=seconds)
                return True
            except asyncio.TimeoutError:
                return False

        def _awaited_slot() -> int | None:
            """Open of the slot whose close the stream waits for.

            :return: The slot after the newest closed or empty-decided one, ``None``
                before the first close.
            """
            if last_closed_bar is None:
                return None
            base = last_closed_bar.timestamp
            if empty_slot_ts is not None and empty_slot_ts > base:
                base = empty_slot_ts
            return base + tf_ms

        def _probe_times(slot: int) -> list[float]:
            """When the history is asked for the close of ``slot``, in epoch ms.

            The first query waits for the provider's publish lag past the close,
            the rest are spread evenly up to the close's deadline; none is later
            than the deadline.

            :param slot: Open of the slot.
            :return: Ascending query times.
            """
            close_ms = slot + tf_ms
            deadline_ms = close_ms + bar_grace_ms
            first_ms = min(close_ms + publish_lag_ms, deadline_ms)
            step_ms = (deadline_ms - first_ms) / _MISSED_CLOSE_PROBES
            if step_ms <= 0.0:
                return [first_ms]
            return [first_ms + k * step_ms for k in range(_MISSED_CLOSE_PROBES)]

        def _sync_probe(slot: int | None) -> None:
            """Start the probe schedule of ``slot`` when the awaited slot changed."""
            nonlocal probe_slot, probe_next_ms, probe_answered, probe_failures
            if slot == probe_slot:
                return
            probe_slot = slot
            probe_answered = False
            probe_failures = 0
            probe_next_ms = float("inf") if slot is None else _probe_times(slot)[0]

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
            (non-retryable :class:`~pynecore.core.plugin.ProviderError`) is re-raised on the first
            attempt so it keeps failing fast. The inter-attempt wait is
            event-driven — raced against ``stop_aevent`` — so a shutdown request
            abandons it at once without polling.
            """
            deadline = time.monotonic() + connect_timeout_seconds
            attempt = 0
            delay = provider.reconnect_delay
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "initial provider connection exceeded its timeout"
                    )
                # The deadline is enforced by racing the connect task, NOT by
                # ``asyncio.wait_for``: that helper decides on the cancellation
                # it injected, so a ``connect()`` that swallows the
                # ``CancelledError`` and returns normally is reported as a
                # SUCCESS past the deadline (and one that never finishes
                # cancelling hangs the startup barrier outright). Deciding on
                # "did it finish in time" instead makes the bound hard: the
                # abandoned task is cancelled best-effort and detached, never
                # awaited. The handshake runs wrapped so that an attempt lost
                # this way closes whatever it still manages to open.
                connect_marker = _AbandonMarker()
                connect_task = asyncio.ensure_future(
                    _connect_closing_if_abandoned(
                        provider, provider.connect(), connect_marker))
                try:
                    done, _ = await asyncio.wait({connect_task}, timeout=remaining)
                except BaseException:
                    connect_marker.abandoned = True
                    connect_task.cancel()
                    connect_task.add_done_callback(_discard_task_outcome)
                    abandoned_connects.append(connect_task)
                    raise
                if not done:
                    connect_marker.abandoned = True
                    connect_task.cancel()
                    connect_task.add_done_callback(_discard_task_outcome)
                    # The cancel is best-effort, so this handshake may still
                    # land; teardown holds the loop open for its self-close.
                    abandoned_connects.append(connect_task)
                    raise TimeoutError(
                        "initial provider connection exceeded its timeout"
                    )
                try:
                    connect_task.result()
                    return
                except BaseException as connect_error:
                    if not _is_transient_connect_error(connect_error):
                        raise
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise
                    attempt += 1
                    wait = min(delay, remaining)
                    log = logger.warning if _warn_this_attempt(attempt) else logger.debug
                    log("Initial connect failed (attempt %d): %s; "
                        "retrying in %.1fs", attempt, connect_error, wait)
                    try:
                        await asyncio.wait_for(stop_aevent.wait(), timeout=wait)
                        # Shutdown requested mid-backoff: surface the last
                        # connect error so teardown runs and the caller unblocks.
                        raise
                    except asyncio.TimeoutError:
                        pass
                    delay = min(delay * 2.0, provider.max_reconnect_delay)

        try:
            _report('connecting')
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
            _report('live')
            last_real_update = time.time()

            history_query = getattr(provider, "backfill_closed_bars", None)

            def _splice_closed(bars: list[OHLCV], since_ms: int) -> list[OHLCV]:
                """Hand on the closed history bars the stream has not, in order.

                The one entry point of history bars into the stream, and the
                closed-bar cursor the socket's closes are checked against too: a
                bar passes only when it is closed, newer than ``since_ms``, the
                warmup history and every closed bar handed on before it. A
                history bar and the socket's close of the same slot can come in
                either order — whichever arrives first is the slot's close, the
                other is dropped — and a slot the cursor already passed is never
                handed on behind a newer one. A slot settled this way also ends
                its forming bar: late ticks of it are dropped like after the
                boundary watchdog. Runs without an ``await``, so the socket loop
                cannot interleave with it.

                :param bars: The provider's answer, in any order.
                :param since_ms: The bar open the query asked after.
                :return: The bars handed on, ascending.
                """
                nonlocal last_closed_bar, last_forming_bar, settled_ts, synth_streak
                accepted: list[OHLCV] = []
                for history_bar in sorted(bars, key=lambda b: b.timestamp):
                    if not history_bar.is_closed or history_bar.timestamp <= since_ms:
                        continue
                    if (last_historical_timestamp is not None
                            and history_bar.timestamp <= last_historical_timestamp):
                        continue
                    if (last_closed_bar is not None
                            and history_bar.timestamp <= last_closed_bar.timestamp):
                        continue
                    last_closed_bar = history_bar
                    settled_ts = history_bar.timestamp
                    synth_streak = 0
                    if (last_forming_bar is not None
                            and history_bar.timestamp >= last_forming_bar.timestamp):
                        last_forming_bar = None
                    bar_queue.put(history_bar)
                    accepted.append(history_bar)
                return accepted

            async def _query_history(since_ms: int | None) -> list[OHLCV] | None:
                """Query the provider for the closed bars after the cursor and splice them.

                The query starts after the newest of ``since_ms``, the warmup
                history and the stream's last closed bar: everything up to that
                is already on its way to the consumer. Raced against the
                shutdown signal, so a stalled history query cannot keep a stop
                waiting. A provider failure propagates to the caller.

                :param since_ms: Open of the newest closed bar the consumer holds,
                    ``None`` when it holds none.
                :return: The bars handed on, ``None`` when a shutdown ended the query.
                """
                query = history_query
                if query is None:
                    return []
                floors = [stamp for stamp in (
                    since_ms, last_historical_timestamp,
                    None if last_closed_bar is None else last_closed_bar.timestamp,
                ) if stamp is not None]
                if not floors:
                    return []
                floor = max(floors)
                recovered: list[OHLCV] = []

                async def _run_query(
                        run: Callable[[str, str, int], Coroutine[Any, Any, list[OHLCV]]],
                ) -> None:
                    recovered.extend(await run(watch_symbol, timeframe, floor))

                if await _await_or_stop(_run_query(query)):
                    return None
                return _splice_closed(recovered, floor)

            async def _fill_from_history(since_ms: int | None) -> int:
                """Splice the closed history bars after ``since_ms`` and the cursor.

                The fill :class:`ClosedBarBackfill` serves; a provider failure
                propagates to the requester.

                :param since_ms: Open of the newest closed bar the consumer holds,
                    ``None`` when it holds none.
                :return: The number of bars handed on.
                """
                accepted = await _query_history(since_ms)
                return 0 if accepted is None else len(accepted)

            async def _probe_missed_close(slot: int) -> bool:
                """Ask the history for the close of ``slot`` the socket has not delivered.

                A bar the history has goes on through the closed-bar cursor, the
                slot's forming bar with it. A failed query is not an answer: the
                slot keeps waiting for one, past the schedule at the provider's
                backoff. The next query is scheduled either way.

                :param slot: Open of the awaited slot.
                :return: ``True`` when the subscription counts as dead: consecutive
                    closes with trades came from the history and not the socket.
                """
                nonlocal probe_next_ms, probe_answered, probe_failures, history_close_streak
                try:
                    accepted = await _query_history(None)
                except Exception as exc:
                    probe_failures += 1
                    log = logger.warning if _warn_this_attempt(probe_failures) else logger.debug
                    log("missed close of ts=%d: history query failed (attempt %d): %s",
                        slot, probe_failures, exc)
                    accepted = []
                else:
                    if accepted is None:
                        return False
                    probe_answered = True
                now_ms = time.time() * 1000.0
                upcoming = [at for at in _probe_times(slot) if at > now_ms]
                if upcoming:
                    probe_next_ms = upcoming[0]
                elif probe_answered:
                    probe_next_ms = float("inf")
                else:
                    probe_next_ms = now_ms + _backoff_delay(provider, probe_failures) * 1000.0
                if not accepted:
                    return False
                logger.info("missed close: %d closed bar(s) from history, newest ts=%d",
                            len(accepted), accepted[-1].timestamp)
                for history_bar in accepted:
                    if socket_seen_ts is not None and socket_seen_ts >= history_bar.timestamp:
                        # The socket was alive for this slot, its close was only late
                        history_close_streak = 0
                    elif history_bar.volume > 0.0:
                        history_close_streak += 1
                if history_close_streak < _STALE_HISTORY_CLOSES:
                    return False
                history_close_streak = 0
                return True

            # Startup gap: the warmup download stopped at
            # ``last_historical_timestamp`` and the stream only delivers from
            # now on, so every bar that closed while the subscription was
            # coming up belongs to neither side. Ask the provider for them
            # before the first live bar, so the strategy sees one unbroken
            # series. A failed query is retried: no live bar is handed on over
            # the hole. Providers that cannot query history return nothing.
            if (last_historical_timestamp is not None and tf_ms > 0
                    and history_query is not None):
                elapsed_ms = time.time() * 1000.0 - last_historical_timestamp
                # A whole bar must have closed past the warmup before anything
                # can be missing.
                if elapsed_ms >= 2 * tf_ms:
                    gap_attempts = 0
                    while True:
                        try:
                            gap_bars = await _query_history(None)
                            break
                        except Exception as gap_error:
                            gap_attempts += 1
                            gap_delay = _backoff_delay(provider, gap_attempts)
                            gap_log = (broker_warning if _warn_this_attempt(gap_attempts)
                                       else logger.debug)
                            gap_log("startup gap recovery failed (attempt %d): %s — retrying"
                                    " in %.1fs, the live stream waits for the gap",
                                    gap_attempts, gap_error, gap_delay)
                            if await _stop_within(gap_delay):
                                return
                    if gap_bars is None:
                        return
                    if gap_bars:
                        broker_info(
                            "startup gap: recovered %d closed bar(s) between "
                            "the warmup history and the live stream",
                            len(gap_bars),
                        )

            # Later fills on a consumer's request run through the same splice
            if backfill is not None:
                backfill.attach(asyncio.get_running_loop(), _fill_from_history)

            # Broker mode: attach the Order Sync Engine's event stream as
            # a background task so OrderEvents land in its queue without
            # blocking the OHLCV reader.
            if engine_event_stream is not None:
                engine_task = asyncio.create_task(engine_event_stream)

            reconnect_attempts = 0

            async def _handle_connection_error(
                    connection_error: BaseException,
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
                nonlocal history_close_streak
                while not stop_event.is_set():
                    # Session-gate: when the market is in a known-closed
                    # window (e.g. FX weekend), do not churn through
                    # reconnect cycles on a connection error. Wait one closed-
                    # window cadence and re-enter without incrementing
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
                                connection_error,
                            )
                            market_open_state = False
                            _report('market_closed', error=connection_error)
                        try:
                            await asyncio.wait_for(
                                stop_aevent.wait(),
                                timeout=_CLOSED_WINDOW_SLEEP_S,
                            )
                            return attempts, True
                        except asyncio.TimeoutError:
                            continue
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
                    log = (
                        logger.warning
                        if _warn_this_attempt(attempts)
                        else logger.debug
                    )
                    log(
                        "Connection error (attempt %d, offline %.0fs): %s",
                        attempts, offline_s, connection_error,
                    )
                    _report('reconnecting', attempts, offline_s, connection_error)
                    if await _await_or_stop(provider.on_disconnect()):
                        return attempts, True
                    delay = _backoff_delay(provider, attempts)
                    if delay > 0.0:
                        try:
                            await asyncio.wait_for(
                                stop_aevent.wait(),
                                timeout=delay,
                            )
                            return attempts, True
                        except asyncio.TimeoutError:
                            pass
                    try:
                        if await _await_or_stop(provider.disconnect()):
                            return attempts, True
                    except Exception as disc_err:
                        logger.debug(
                            "disconnect() before reconnect raised: %s",
                            disc_err,
                        )
                    try:
                        if await _await_or_stop(provider.connect(),
                                                opens_connection=True):
                            return attempts, True
                        if await _await_or_stop(provider.on_reconnect()):
                            return attempts, True
                    except Exception as reconn_err:
                        log = (
                            logger.warning
                            if _warn_this_attempt(attempts)
                            else logger.debug
                        )
                        log(
                            "Reconnect failed (attempt %d, offline %.0fs): %s",
                            attempts,
                            time.time() - (outage_started or time.time()),
                            reconn_err,
                        )
                        connection_error = reconn_err
                        continue
                    # Reconnect gap: the bars that closed while the socket was
                    # down go on before any socket bar of the new subscription.
                    # A query that fails leaves the gap unanswered, so the
                    # stream must not go on over it: another reconnect round.
                    if core_recovery:
                        try:
                            recovered = await _query_history(None)
                        except Exception as fill_err:
                            log = (
                                logger.warning
                                if _warn_this_attempt(attempts)
                                else logger.debug
                            )
                            log(
                                "Reconnect gap fill failed (attempt %d, offline %.0fs): %s",
                                attempts,
                                time.time() - (outage_started or time.time()),
                                fill_err,
                            )
                            connection_error = fill_err
                            continue
                        if recovered is None:
                            return attempts, True
                        if recovered:
                            broker_info(
                                "reconnect gap: recovered %d closed bar(s) missed "
                                "while the feed was down",
                                len(recovered),
                            )
                        history_close_streak = 0
                    # Give the freshly (re)subscribed feed a full
                    # staleness window to deliver before the liveness
                    # watchdog may declare it dead again.
                    last_real_update = time.time()
                    logger.info("Reconnected successfully (attempt %d)", attempts)
                    return attempts, False
                return attempts, True

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

                # Missed-close probe: ask the history for the awaited slot's
                # close once its query is due. Only while the query can run
                # (connected, slot in session), so a probe that cannot run does
                # not shorten the wait below.
                probe_due_ms = float("inf")
                awaited_slot = _awaited_slot() if boundary_watchdog else None
                if core_recovery:
                    _sync_probe(awaited_slot)
                    if (awaited_slot is not None and provider.is_connected
                            and _market_open_at(awaited_slot)):
                        probe_due_ms = probe_next_ms
                    if awaited_slot is not None and time.time() * 1000.0 >= probe_due_ms:
                        if await _probe_missed_close(awaited_slot):
                            pending_connection_error = ConnectionError(
                                f"stale subscription: {_STALE_HISTORY_CLOSES} consecutive "
                                f"closes came from the history, none from the socket"
                            )
                        continue

                # Cap the per-iteration wait at 2 s for the existing
                # is_connected healthcheck cadence; if a missed-bar
                # deadline or a probe falls sooner, shorten the wait so the
                # watchdog fires promptly when the WS goes idle. A slot that
                # waits for its history answer leaves the deadline to the probe.
                next_due_ms = probe_due_ms
                if (awaited_slot is not None
                        and not (core_recovery and not probe_answered)):
                    next_due_ms = min(next_due_ms, awaited_slot + tf_ms + bar_grace_ms)
                effective_timeout = min(
                    2.0, max(0.05, (next_due_ms - time.time() * 1000.0) / 1000.0))

                # A held update goes on once the slot it waited behind is settled,
                # or once the awaited slot is in a closed session, which is never
                # settled: the same bypass a fresh socket update gets below.
                replay: OHLCV | None = None
                if (held_updates and awaited_slot is not None
                        and (held_updates[0].timestamp <= awaited_slot
                             or not _market_open_at(awaited_slot))):
                    replay = held_updates.pop(0)
                # While updates are held, a busy socket must not starve the
                # watchdog, which only runs when a read times out: once the
                # history has answered for the awaited slot and its deadline
                # passed, the slot is settled before the socket is read again.
                settle_now = (
                    replay is None and bool(held_updates) and probe_answered
                    and awaited_slot is not None
                    and time.time() * 1000.0 >= awaited_slot + tf_ms + bar_grace_ms)

                try:
                    if settle_now:
                        raise asyncio.TimeoutError
                    if replay is not None:
                        bar_update = replay
                    else:
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
                            _report('live')
                        reconnect_attempts = 0
                        if socket_seen_ts is None or bar_update.timestamp > socket_seen_ts:
                            socket_seen_ts = bar_update.timestamp

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

                    # An update of a later slot while the awaited slot is not
                    # settled waits behind it (see ``held_updates``). A slot in a
                    # closed session is never settled: the first bar after the
                    # reopen goes on as before.
                    if (core_recovery and awaited_slot is not None
                            and bar_update.timestamp > awaited_slot
                            and _market_open_at(awaited_slot)):
                        if (not bar_update.is_closed and held_updates
                                and not held_updates[-1].is_closed
                                and held_updates[-1].timestamp == bar_update.timestamp):
                            held_updates[-1] = bar_update
                        else:
                            held_updates.append(bar_update)
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
                        history_close_streak = 0
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
                            _report('live')
                        # Backpressure probe: a non-trivial qsize or put
                        # latency here is the smoking gun for consumer-side
                        # lag stalling the asyncio loop. The queue is
                        # unbounded so ``put`` cannot actually block on
                        # capacity, but cross-thread handoff + GIL pressure
                        # can still take meaningful time when the consumer
                        # is busy. Warn so the live log preserves the
                        # evidence next time a synth-laden run happens.
                        # A bar that closed more than a missed-bar deadline
                        # ago is catch-up (the provider re-delivering the
                        # bars of an outage, live: ctrader cycle 358,
                        # 2026-11-07, ~130 bars after a 2h venue outage):
                        # the queue is deep because the bars arrive in one
                        # burst, not because the consumer lags.
                        qsize_before = bar_queue.qsize()
                        put_t0 = time.monotonic()
                        bar_queue.put(bar_update)
                        put_ms = (time.monotonic() - put_t0) * 1000.0
                        catching_up = is_catch_up(bar_update.timestamp, tf_seconds,
                                                  time.time() * 1000.0)
                        lagging = put_ms > 100.0 or (
                            qsize_before > 50 and not catching_up
                        )
                        if lagging:
                            backpressure_peak = max(
                                backpressure_peak, qsize_before,
                            )
                            if not backpressure_reported:
                                broker_warning(
                                    "bar_queue backpressure: qsize_before=%d "
                                    "put_ms=%.1f ts=%d — the strategy is "
                                    "falling behind the live feed",
                                    qsize_before, put_ms, bar_update.timestamp,
                                )
                                backpressure_reported = True
                        elif backpressure_reported and qsize_before <= 50:
                            broker_info(
                                "bar_queue backpressure cleared (peak "
                                "qsize=%d)", backpressure_peak,
                            )
                            backpressure_reported = False
                            backpressure_peak = 0
                    elif (settled_ts is not None
                            and bar_update.timestamp <= settled_ts):
                        if late_ticks_reported_ts != settled_ts:
                            late_ticks_reported_ts = settled_ts
                            broker_info(
                                "late intra-bar updates for ts=%d after the "
                                "framework closed it — dropped (the feed "
                                "delivered them past the bar's close)",
                                bar_update.timestamp,
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
                            bar_queue.put_nowait(bar_update)

                except asyncio.TimeoutError:
                    # Boundary watchdog: if the awaited slot's close has
                    # passed by more than the grace without a real WS bar,
                    # settle the slot: finalize its forming bar, or emit a
                    # zero-volume filler, or none when the venue's history
                    # has no bar for an idle slot. Settles exactly one slot
                    # per timeout; the next iteration re-checks against
                    # ``time.time()`` and either settles the next slot or
                    # waits for a real push.
                    if (awaited_slot is not None
                            and time.time() * 1000.0
                            >= awaited_slot + tf_ms + bar_grace_ms):
                        assert last_closed_bar is not None
                        synth_ts = awaited_slot
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
                                _report('market_closed')
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
                        # The history has not answered for this slot yet (every
                        # query failed, or none could run): a slot the venue may
                        # well have a bar for is neither finalized nor filled.
                        # The probe goes on at the provider's backoff, and a
                        # dead feed still reconnects, whose gap fill answers.
                        if core_recovery and (probe_slot != synth_ts or not probe_answered):
                            if not provider.is_connected:
                                pending_connection_error = ConnectionError(
                                    "Provider reports disconnected state"
                                )
                            elif (feed_stale_after is not None
                                    and time.time() - last_real_update
                                    >= feed_stale_after
                                    and _market_open_now()):
                                pending_connection_error = ConnectionError(
                                    f"feed stale: no data from provider for "
                                    f"{time.time() - last_real_update:.0f}s "
                                    f"during open session"
                                )
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
                        # keeps its own ask side and ``extra_fields``.
                        if (last_forming_bar is not None
                                and last_forming_bar.timestamp == synth_ts):
                            finalized = last_forming_bar._replace(is_closed=True)
                            last_closed_bar = finalized
                            settled_ts = finalized.timestamp
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
                        # An idle slot on a venue whose history has no bar for
                        # it: no bar here either, so the stream is the series a
                        # restart replays from the history. The watchdog moves
                        # on to the next slot; the cursor and ``bar_index`` stay.
                        if not history_has_idle_bars:
                            empty_slot_ts = synth_ts
                            logger.debug("idle slot ts=%d left empty: the venue's history "
                                         "has no bar for a slot without trades", synth_ts)
                            continue
                        last_close = last_closed_bar.close
                        # The ask side freezes the same way as the bid side
                        last_ask = last_closed_bar.ask
                        synth = OHLCV(
                            timestamp=synth_ts,
                            open=last_close,
                            high=last_close,
                            low=last_close,
                            close=last_close,
                            volume=0.0,
                            extra_fields=last_closed_bar.extra_fields,
                            is_closed=True,
                            ask=None if last_ask is None else AskOHLC(
                                last_ask.close, last_ask.close, last_ask.close,
                                last_ask.close),
                        )
                        last_closed_bar = synth
                        settled_ts = synth.timestamp
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
                        # the boundary watchdog is not armed yet, and a
                        # stream without the boundary watchdog). The
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
            if backfill is not None:
                backfill.detach()
            if engine_task is not None and not engine_task.done():
                engine_task.cancel()
                try:
                    await engine_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            await _graceful_shutdown()
            _report('stopped')
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
            # Deliberately NOT ``asyncio.run``: its exit-time sweep gathers the
            # leftover tasks without a deadline, so one hook that never finishes
            # cancelling keeps this thread — and the consumer's join — alive
            # forever. :func:`_close_owned_loop` does the same sweep on a bound.
            own_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(own_loop)
            try:
                own_loop.run_until_complete(_async_loop())
            finally:
                try:
                    _close_owned_loop(own_loop)
                finally:
                    asyncio.set_event_loop(None)

    # Start the provider thread, then block until ``provider.connect()``
    # has completed (or timed out / failed). The warmup that follows
    # therefore runs against an already-listening WS — bars that close
    # mid-warmup land in ``bar_queue`` and are drained as catch-up at
    # the warmup→live boundary. Without this barrier the eager-start
    # design is racy: a slow connect() can finish AFTER warmup, leaving
    # an unsubscribed window in which the next bar close is lost.
    thread = threading.Thread(target=_thread_target, daemon=True, name="live-provider")
    thread.start()
    if not connected_event.wait(timeout=connect_timeout_seconds + 1.0):
        if raise_on_connect_failure:
            # ``_async_loop`` never reported back — a ``connect()`` that ignores
            # its cancellation is still running on the provider loop. ``_consumer``
            # is not created on this path, so nothing else would signal the
            # loop-side shutdown or reap the thread: tear down here, with a BOUNDED
            # join (the very fault being handled is a producer that will not stop).
            stop_event.set()
            _signal_loop_shutdown()
            thread.join(timeout=startup_join_timeout)
            raise TimeoutError(
                "provider connection did not report completion within its timeout"
            )
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
        thread.join(timeout=startup_join_timeout)
        raise connect_failure[0]

    def _wake_on_stop() -> None:
        bar_queue.put(_STOP_REQUEST)

    def _consumer() -> Generator[OHLCV, None, None]:
        in_warmup_catchup = True

        if stop is not None:
            stop.add_waker(_wake_on_stop)
        try:
            while True:
                if stop is not None and stop.is_set:
                    break
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

                    if item is _STOP_REQUEST:
                        break
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

                    if item is _SENTINEL or item is _STOP_REQUEST:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    if stop is not None and stop.is_set:
                        break

                    yield item

        except KeyboardInterrupt:
            # Propagate after the teardown below so the run is reported as
            # interrupted, the same as an interrupt landing outside the
            # iterator; swallowing it here made a signalled stop look like
            # a completed run.
            logger.info("Live streaming interrupted by user")
            if on_interrupt is not None:
                on_interrupt()
            raise
        finally:
            if stop is not None:
                stop.remove_waker(_wake_on_stop)
            stop_event.set()
            # Wake a reconnect handshake blocked on ``provider.connect()`` on the
            # broker loop (see ``_async_loop._await_or_stop``), or teardown waits
            # out the whole connect timeout before the producer thread can exit.
            _signal_loop_shutdown()
            consumer_join_timeout = (
                shutdown_timeout + 5.0
                if shutdown_timeout > 0
                else None
            )
            thread.join(timeout=consumer_join_timeout)

    return _consumer()
