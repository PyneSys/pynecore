"""
Provider-mode runs: download a provider's history, then stream it live.

The machinery ``pyne run`` uses for a provider string (``ccxt:BYBIT:BTC/USDT:USDT@1``),
kept free of any terminal output so other front ends (an IDE bridge, a service) run
providers the same way. Progress and warnings go to a :class:`DownloadListener`,
failures are raised.
"""
import os
import tempfile
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import replace as dc_replace
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from pynecore.core.plugin import (load_plugin, ProviderError, ProviderPlugin,
                                  is_retryable_provider_error)
from pynecore.core.plugin.live_provider import LiveProviderPlugin
from pynecore.core.config import ensure_config
from pynecore.core.live_runner import FeedStatus, LiveStop, live_ohlcv_generator
from pynecore.lib.timeframe import in_seconds
from pynecore.core._file_io import exclusive_file_lock, replace_file
from pynecore.core.ohlcv import OHLCVReader, OHLCVWriter
from pynecore.core.provider_string import ProviderString, parse_provider_string
from pynecore.core.session import is_in_session
from pynecore.core.syminfo import SymInfo
from pynecore.types.ohlcv import OHLCV

if TYPE_CHECKING:
    import asyncio

__all__ = ['ProviderHistory', 'ProviderSetupError', 'DownloadListener', 'parse_time_value',
           'fetch_provider_symbol_info', 'download_provider_history',
           'download_provider_history_resilient', 'DownloadStopped', 'require_live_provider',
           'open_live_stream']


class ProviderSetupError(Exception):
    """A provider run cannot start: the plugin cannot serve what the run asks of it."""


class DownloadStopped(Exception):
    """A provider download was stopped through its :class:`~pynecore.core.live_runner.LiveStop`."""


class DownloadListener:
    """Receives the progress and the warnings of a provider download.

    Every method is a no-op here; a front end overrides the ones it shows. The
    ``*_finished`` calls always come, also when the step failed.
    """

    def symbol_info_started(self) -> None:
        """The symbol information is being fetched from the venue."""

    def symbol_info_finished(self, ok: bool) -> None:
        """The symbol information fetch ended.

        :param ok: Whether it succeeded.
        """

    def download_started(self, time_from: datetime) -> None:
        """The OHLCV download begins.

        :param time_from: The requested start.
        """

    def download_attempt(self, time_from: datetime, total_seconds: int) -> None:
        """One download pass begins; a ``-N bars`` download may need several.

        :param time_from: The start of this pass (naive UTC).
        :param total_seconds: The length of the pass's window.
        """

    def download_progress(self, elapsed_seconds: int) -> None:
        """The pass reached this far into its window.

        :param elapsed_seconds: Seconds from the pass's start.
        """

    def download_finished(self, ok: bool) -> None:
        """The OHLCV download ended.

        :param ok: Whether it succeeded.
        """

    def warning(self, message: str) -> None:
        """Something the user should know about, the run goes on.

        :param message: The full message.
        """

    def retry_wait(self, error: BaseException, delay: float, attempt: int) -> None:
        """A transient provider failure: the download is retried after a wait.

        :param error: The failure.
        :param delay: Seconds before the retry.
        :param attempt: The retry about to happen, counted from 1.
        """


def parse_time_value(value: str | None, *, allow_bars: bool = False) -> datetime | int | None:
    """
    Parse a ``--from`` or ``--to`` style time value.

    A date without an explicit UTC offset is taken as UTC, never as the host's local time.

    :param value: The raw string value: a date, a number of days back, or with
        ``allow_bars`` a negative bar count.
    :param allow_bars: If True, allow negative numbers as bar counts.
    :return: A timezone-aware datetime, a negative int (bar count), or None.
    :raises ValueError: If the value is none of the accepted forms.
    """
    if value is None:
        return None
    value: str = value.strip()

    # Negative number = bar count (only for --from in provider mode)
    if allow_bars and value.startswith('-'):
        try:
            bars = int(value)
            return bars
        except ValueError:
            pass

    # Positive number = days back
    try:
        days: int | None = int(value)
    except ValueError:
        days = None
    if days is not None:
        if days < 0:
            raise ValueError("Days cannot be negative (use negative numbers only with provider "
                             "mode for bar count)")
        return (datetime.now(UTC) - timedelta(days=days)).replace(second=0, microsecond=0)

    # Date string
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(f"Invalid date or number: '{value}'") from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


class ProviderHistory:
    """A provider's downloaded history: the chart data of a provider-mode run.

    :ivar ohlcv_path: The OHLCV file the history was written to.
    :ivar syminfo: The symbol information, fresh from the venue.
    :ivar parsed_string: The parsed provider string.
    :ivar provider_instance: The provider plugin instance, already authenticated; a live
        run streams through the same instance.
    :ivar time_from_ts: Exact start timestamp (Unix milliseconds) that yields exactly the
        requested bar count in ``-N bars`` mode; ``None`` when the file's natural start is
        the window start (date/days mode, no bar target).
    :ivar as_of_ts: Unix milliseconds no later than the moment the bars were downloaded: the
        start of the download pass that wrote the file. A bar whose period had not ended by
        then may hold partial data.
    """

    def __init__(self, ohlcv_path: Path, syminfo: SymInfo, parsed_string: ProviderString,
                 provider_instance: ProviderPlugin, time_from_ts: int | None = None,
                 as_of_ts: int | None = None):
        self.ohlcv_path = ohlcv_path
        self.syminfo = syminfo
        self.provider_instance = provider_instance
        self.parsed_string: ProviderString = parsed_string
        self.time_from_ts = time_from_ts
        self.as_of_ts = as_of_ts


def _missing_slots(real_ts: list[int], start_ts: int, end_ts: int,
                   tf_seconds: int) -> list[int]:
    """Return the aligned slot start-timestamps in ``[start_ts, end_ts)`` that
    have no real bar.

    The expected grid is inferred from the real bars themselves — anchored on
    the newest real bar and stepped back by ``tf_seconds`` — so the reported
    slots share the feed's phase and only genuinely-absent slots surface. With
    no real bars the grid phase is unknown and an empty list is returned.

    :param real_ts: Real (non-gap-fill) bar timestamps; need not be sorted.
    :param start_ts: Inclusive window start (epoch seconds).
    :param end_ts: Exclusive window end (epoch seconds).
    :param tf_seconds: Timeframe length in seconds.
    :return: Sorted list of missing slot start-timestamps.
    """
    if tf_seconds <= 0 or not real_ts:
        return []
    present = set(real_ts)
    anchor = max(real_ts)
    missing: list[int] = []
    slot = anchor
    # Walk the grid back from the newest real bar to ``start_ts``.
    while slot >= start_ts:
        if start_ts <= slot < end_ts and slot not in present:
            missing.append(slot)
        slot -= tf_seconds
    missing.reverse()
    return missing


def _classify_missing_slots(missing: list[int], syminfo: 'SymInfo',
                            tf_seconds: int) -> tuple[list[int], list[int]]:
    """Split missing slots into in-session anomalies vs closed-session gaps.

    A slot that falls inside a trading session but has no bar is an *anomaly*
    (incomplete pagination or a venue-omitted tick), while a slot in a
    known-closed window (weekend, out-of-hours) is an *expected* venue gap.
    When the symbol carries no ``opening_hours`` (24/7 instrument) every slot
    is treated as in-session, so any hole is reported as an anomaly.

    :param missing: Missing slot start-timestamps (epoch seconds).
    :param syminfo: Symbol calendar (``opening_hours``, ``timezone``).
    :param tf_seconds: Timeframe length in seconds.
    :return: ``(in_session, closed)`` timestamp lists.
    """
    opening_hours = syminfo.opening_hours
    if not opening_hours:
        return list(missing), []
    # noinspection PyBroadException
    try:
        tz = ZoneInfo(syminfo.timezone)
    except Exception:  # noqa: BLE001
        return list(missing), []
    in_session: list[int] = []
    closed: list[int] = []
    for ts in missing:
        local_dt = datetime.fromtimestamp(ts, tz)
        if is_in_session(opening_hours, local_dt, tf_seconds,
                         syminfo.session_corrections):
            in_session.append(ts)
        else:
            closed.append(ts)
    return in_session, closed


def _merge_intervals(slots: list[int], tf_seconds: int) -> list[tuple[int, int]]:
    """Coalesce consecutive grid slots into ``[start, end)`` epoch intervals.

    :param slots: Sorted slot start-timestamps (epoch seconds).
    :param tf_seconds: Timeframe length in seconds.
    :return: List of ``(start_ts, end_ts)`` half-open intervals.
    """
    if not slots:
        return []
    intervals: list[tuple[int, int]] = []
    run_start = slots[0]
    prev = slots[0]
    for ts in slots[1:]:
        if ts == prev + tf_seconds:
            prev = ts
            continue
        intervals.append((run_start, prev + tf_seconds))
        run_start = ts
        prev = ts
    intervals.append((run_start, prev + tf_seconds))
    return intervals


def _format_missing_report(bar_count: int, real_bars: int, oldest_ts: int | None,
                           in_session: list[int], closed: list[int],
                           tf_seconds: int, horizon_reached: bool) -> str:
    """Build a precise, actionable coverage-shortfall message.

    Distinguishes a venue history-horizon limit (extending ``from`` yields no
    older bars) from in-session holes (incomplete pagination / omitted ticks),
    and lists the exact missing intervals for each class instead of a vague
    sparse-coverage count.
    """
    def fmt(ts: int) -> str:
        return datetime.fromtimestamp(ts, UTC).strftime('%Y-%m-%d %H:%M')

    def fmt_intervals(slots: list[int]) -> str:
        parts = [f"{fmt(a)}..{fmt(b)}Z" for a, b in _merge_intervals(slots, tf_seconds)]
        return ', '.join(parts)

    lines = [f"Warning: requested {bar_count} bars, got {real_bars} real bars."]
    if horizon_reached and oldest_ts is not None:
        lines.append(
            f"  Venue history horizon reached: oldest bar the feed serves is "
            f"{fmt(oldest_ts)}Z. The remaining {bar_count - real_bars} bar(s) "
            f"predate the venue's available history (not a pagination gap)."
        )
    if in_session:
        lines.append(
            f"  {len(in_session)} in-session slot(s) missing (incomplete "
            f"pagination or venue-omitted ticks): {fmt_intervals(in_session)}"
        )
    if closed:
        lines.append(
            f"  {len(closed)} slot(s) fall in known-closed sessions "
            f"(expected venue gap): {fmt_intervals(closed)}"
        )
    if not horizon_reached and not in_session and not closed:
        lines.append("  Cause could not be localized from the downloaded range.")
    return '\n'.join(lines)


@contextmanager
def _ohlcv_download_lock(ohlcv_path: Path | None):
    """Serialize the shared-OHLCV download/rewrite across concurrent processes.

    Two runs on the same ``(provider, symbol, timeframe)`` share one
    ``.ohlcv`` file. The warmup download truncates and rewrites it; a second
    process reading (or truncating) that file at the same instant sees a
    half-written or empty file. An OS advisory lock on a sidecar ``.lock``
    next to the ``.ohlcv`` makes the second process block in the kernel until
    the first finishes, then read a complete file.

    POSIX uses ``fcntl.flock`` and Windows uses ``LockFileEx``. Both block in the
    kernel rather than polling. A process-local lock also serializes separate file
    handles opened by threads in the same process.

    :param ohlcv_path: The target ``.ohlcv`` path, or ``None`` (live/in-memory
        feeds with no shared file — nothing to serialize).
    """
    if ohlcv_path is None:
        yield
        return

    lock_path = ohlcv_path.with_suffix(ohlcv_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(lock_path):
        yield


@contextmanager
def _atomic_ohlcv_download_target(provider: ProviderPlugin):
    """Write one provider download privately, then atomically publish it.

    The sidecar lock serializes publishers, while the temporary file keeps the
    canonical path complete for readers during the entire rewrite. The
    provider's original path and unopened writer are restored before return.

    The private writer declares the same period as the provider's own writer,
    so the published file carries the timeframe the provider was built for.
    ``truncate=True`` turns the empty file ``mkstemp`` reserved into a valid
    empty OHLCV file instead of an unwritable zero-byte one.
    """
    final_path = provider.ohlcv_path
    if final_path is None:
        yield
        return
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp_path = tempfile.mkstemp(
        prefix=f".{final_path.name}.", suffix=".tmp", dir=final_path.parent,
    )
    os.close(fd)
    temp_path = Path(raw_temp_path)
    original_writer = provider.ohlcv_file
    # A provider built with an ohlcv_path always has its writer set.
    assert original_writer is not None
    try:
        with _ohlcv_download_lock(final_path):
            provider.ohlcv_path = temp_path
            provider.ohlcv_file = OHLCVWriter(
                temp_path, original_writer.period, truncate=True,
            )
            try:
                yield
                replace_file(temp_path, final_path)
            finally:
                provider.ohlcv_path = final_path
                provider.ohlcv_file = original_writer
    finally:
        temp_path.unlink(missing_ok=True)


def fetch_provider_symbol_info(provider: ProviderPlugin,
                               listener: DownloadListener | None = None) -> SymInfo:
    """Load the symbol metadata for a provider-mode run, fresh from the venue.

    Provider mode re-downloads the bar data on every run, and the metadata
    must be just as fresh: the venue publishes dated session exceptions
    (holidays, trading breaks) only days ahead, so a cached sidecar from an
    earlier run does not know about them and the live session calendar
    would treat a closed market as open. When the venue cannot be reached
    and a cached sidecar exists, the run continues on the cached metadata
    with a warning rather than failing before the data download gets its
    own say about the connection.

    :param provider: The provider plugin instance.
    :param listener: Receives the warning of a failed refresh.
    :return: The symbol information.
    """
    cached = provider.is_symbol_info_exists()
    try:
        return provider.get_symbol_info(force_update=True)
    except Exception as e:  # plugin exception families differ per venue
        if not cached:
            raise
        (listener or DownloadListener()).warning(
            f"Warning: symbol info refresh failed ({e}); using the cached symbol info")
        return provider.get_symbol_info()


def download_provider_history(provider_str: str, time_from_str: str | None, *,
                              data_dir: Path, config_dir: Path,
                              listener: DownloadListener | None = None) -> ProviderHistory:
    """
    Download the history of a provider string into its OHLCV file.

    :param provider_str: Provider string (e.g. "ccxt:BYBIT:BTC/USDT:USDT@1D").
    :param time_from_str: The start (date, days, or -bars); ``None`` means -500 bars.
    :param data_dir: The workdir's ``data`` directory, where the OHLCV file is written.
    :param config_dir: The workdir's ``config`` directory, holding the plugin configs.
    :param listener: Receives the progress and the warnings of the download.
    :return: The downloaded history.
    :raises ProviderSetupError: If the plugin is not a data provider.
    :raises ValueError: If the start is not a valid time value.
    """
    report: DownloadListener = listener or DownloadListener()

    # Load the provider plugin first so we know whether it is multi-broker,
    # which controls how the provider string is split (broker vs. symbol).
    provider_name = provider_str.split(':', 1)[0].lower()
    provider_class = load_plugin(provider_name)
    if not issubclass(provider_class, ProviderPlugin):
        raise ProviderSetupError(f"Plugin '{provider_name}' is not a data provider.")

    ps = parse_provider_string(provider_str, require_timeframe=True,
                               multi_broker=provider_class.multi_broker)
    # Store the normalized (lowercased) provider name so that later
    # case-sensitive ``load_plugin()`` lookups for security/auto-rate
    # contexts succeed even when the user typed the provider in a
    # different case (e.g. ``CCXT:...``).
    ps = dc_replace(ps, provider=provider_name)

    # Default to -500 bars if --from not specified in provider mode
    if not time_from_str:
        time_from_str = "-500"

    time_from_value = parse_time_value(time_from_str, allow_bars=True)
    time_to_dt = datetime.now(UTC).replace(second=0, microsecond=0)

    # Convert bar count to time range. ``bar_count`` being set signals
    # the "-N bars" mode — we then guarantee at least N *real* bars
    # (exchange-provided, non-gap-fill) after download, extending the
    # from-timestamp on miss. Date/days ranges are left untouched.
    tf_seconds = in_seconds(ps.timeframe)
    bar_count: int | None = None
    if isinstance(time_from_value, int) and time_from_value < 0:
        bc = abs(time_from_value)
        bar_count = bc
        # Pad the request by one bar to absorb the still-forming current
        # bar that closed-bars-only providers (e.g. Capital.com) filter
        # out of history responses. Without this, every ``-N`` run
        # against a now-aligned end-time would burn a wasted retry pass
        # (``real_bars == N - 1`` on first attempt → retry → success).
        time_from_dt = time_to_dt - timedelta(seconds=tf_seconds * (bc + 1))
    else:
        assert isinstance(time_from_value, datetime)
        time_from_dt = time_from_value

    # Load config
    config = None
    config_cls: type | None = getattr(provider_class, 'Config', None)
    if config_cls is not None:
        config = ensure_config(config_cls, config_dir / 'plugins' / f'{provider_name}.toml')

    # Create provider instance. ``provider_symbol`` re-folds the broker into
    # the symbol for multi-broker providers, which split it off internally.
    provider_instance: ProviderPlugin = provider_class(
        symbol=ps.provider_symbol, timeframe=ps.timeframe,
        ohlcv_dir=data_dir, config=config
    )

    # Fetch symbol info
    report.symbol_info_started()
    fetched = False
    try:
        syminfo = fetch_provider_symbol_info(provider_instance, report)
        fetched = True
    finally:
        report.symbol_info_finished(fetched)

    # Download OHLCV data (always fresh in provider mode). In bar-count
    # mode we may re-download with an extended ``from`` until we hit the
    # target — some feeds omit minutes with no ticks (CFD quiet hours,
    # illiquid futures), and ``--from -500`` must mean 500 real bars.
    # The progress lives outside the retry loop so a gap-driven second pass
    # updates the same display instead of starting a second one.
    max_retries = 4
    report.download_started(time_from_dt)
    downloaded = False
    try:
        # Oldest real bar the venue served on the previous attempt. When a
        # further-extended ``from`` fails to pull any older bar, the feed's
        # history horizon is reached and extending again is pointless — we
        # stop and surface that precise reason instead of burning retries.
        prev_oldest_ts: int | None = None
        # Serialize the truncate-and-rewrite (and the verification reads) against
        # any concurrent run sharing this ``.ohlcv`` file: a second process waits
        # on the kernel flock and then reads a complete file, never a
        # half-truncated one. The lock spans the whole retry loop and the
        # bar-count pin below so no other process can rewrite the file between
        # our download and our reads.
        exact_from_ts: int | None = None
        as_of_ts: int | None = None
        with _atomic_ohlcv_download_target(provider_instance):
            for attempt in range(max_retries + 1):
                # Every pass rewrites the whole file, the last bar included
                as_of_ts = int(datetime.now(UTC).timestamp() * 1000)
                with provider_instance as ohlcv_writer:
                    ohlcv_writer.truncate()

                    # Providers take naive UTC datetimes.
                    time_from_dl = time_from_dt.astimezone(UTC).replace(tzinfo=None)
                    time_to_dl = time_to_dt.astimezone(UTC).replace(tzinfo=None)

                    total_seconds = int((time_to_dl - time_from_dl).total_seconds())

                    report.download_attempt(time_from_dl, total_seconds)

                    def cb_progress(current_time: datetime):
                        report.download_progress(
                            int((current_time - time_from_dl).total_seconds()))

                    # One batch for the whole download: a provider that saves bar by
                    # bar would otherwise pay two fsyncs per bar.
                    with ohlcv_writer.batched():
                        provider_instance.download_ohlcv(
                            time_from_dl, time_to_dl, on_progress=cb_progress,
                        )

                if bar_count is None:
                    break

                # Collect the real bar timestamps in the requested range. The
                # reader speaks Unix milliseconds and serves only real bars, so
                # the count is the venue's own coverage.
                window_from_ts = int(time_from_dt.timestamp() * 1000)
                window_to_ts = int(time_to_dt.timestamp() * 1000)
                with OHLCVReader(provider_instance.ohlcv_path) as r:  # type: ignore[arg-type]
                    real_ts = [b.timestamp for b in r.read_from(
                        window_from_ts, window_to_ts,
                    )]
                real_bars = len(real_ts)
                oldest_ts = min(real_ts) if real_ts else None

                if real_bars >= bar_count:
                    break

                # History-horizon check: if extending ``from`` earlier did not
                # surface any older bar than the previous attempt, the venue has
                # no more history — retrying cannot help, so report precisely now.
                horizon_reached = (
                    prev_oldest_ts is not None and oldest_ts is not None
                    and oldest_ts >= prev_oldest_ts
                )

                if horizon_reached or attempt == max_retries:
                    # The coverage report walks the session calendar, whose
                    # resolution is one second (``is_in_session``), so the
                    # millisecond bar timestamps are stepped down here — the
                    # single place where the two units meet.
                    grid_ts = [ts // 1000 for ts in real_ts]
                    missing_slots = _missing_slots(
                        grid_ts, window_from_ts // 1000, window_to_ts // 1000,
                        int(tf_seconds),
                    )
                    in_session, closed = _classify_missing_slots(
                        missing_slots, syminfo, int(tf_seconds),
                    )
                    report.warning(
                        _format_missing_report(
                            bar_count, real_bars,
                            None if oldest_ts is None else oldest_ts // 1000,
                            in_session, closed, int(tf_seconds), horizon_reached,
                        )
                    )
                    break

                prev_oldest_ts = oldest_ts

                # Extend the range, anchoring on the oldest bar actually served so
                # the next pass reaches strictly-older history: request the missing
                # count of slots before it, plus a multi-day buffer that clears any
                # single weekend / session gap in one jump. Without the buffer a
                # weekend between ``from`` and the oldest bar would stall the
                # oldest cursor and be misread as the venue history horizon. The
                # over-fetch is harmless — the range is pinned to exactly N real
                # bars below.
                anchor_ts = oldest_ts if oldest_ts is not None else window_from_ts
                missing = bar_count - real_bars
                candidate = (
                    datetime.fromtimestamp(anchor_ts / 1000, UTC)
                    - timedelta(seconds=tf_seconds * (missing + 10))
                    - timedelta(days=3)
                )
                # Only ever move the window start earlier.
                time_from_dt = min(time_from_dt, candidate)

            assert provider_instance.ohlcv_path is not None

            # For bar-count mode, pin the start timestamp to the N-th last real
            # bar so the reader serves *exactly* N bars, not the over-fetched
            # surplus we used to guarantee coverage through gaps.
            if bar_count is not None:
                with OHLCVReader(provider_instance.ohlcv_path) as r:  # type: ignore[arg-type]
                    real_ts = [b.timestamp for b in r.read_from(
                        0, int(time_to_dt.timestamp() * 1000),
                    )]
                if len(real_ts) >= bar_count:
                    exact_from_ts = real_ts[-bar_count]
        downloaded = True
    finally:
        report.download_finished(downloaded)

    return ProviderHistory(
        ohlcv_path=provider_instance.ohlcv_path,
        syminfo=syminfo,
        provider_instance=provider_instance,
        parsed_string=ps,
        time_from_ts=exact_from_ts,
        as_of_ts=as_of_ts,
    )


#: Startup provider-download retry backoff (seconds): the first wait and the
#: ceiling the exponential growth saturates at. Internal tunables — deliberately
#: not user config: a live bot should keep trying until the broker returns, and
#: ~1 minute is a sane poll cadence through a multi-hour maintenance window.
_PROVIDER_RETRY_BASE_DELAY = 2.0
_PROVIDER_RETRY_MAX_DELAY = 60.0


def _wait_before_retry(delay: float, stop: LiveStop | None = None) -> None:
    """Block for ``delay`` seconds interruptibly, without a bare sleep.

    Waits on a :class:`threading.Event`, an event-driven, deadline-bounded
    primitive: a ``Ctrl-C`` (``KeyboardInterrupt`` in the main thread) or a set
    ``stop`` breaks out of the backoff at once instead of stalling for the full
    delay, and there is no busy-poll.

    :param delay: Maximum seconds to wait.
    :param stop: Ends the wait when set, from any thread.
    :raises DownloadStopped: If ``stop`` was set.
    """
    woken = threading.Event()
    if stop is not None:
        stop.add_waker(woken.set)
    try:
        woken.wait(timeout=delay)
    finally:
        if stop is not None:
            stop.remove_waker(woken.set)
    if stop is not None and stop.is_set:
        raise DownloadStopped()


def download_provider_history_resilient(
        provider_str: str, time_from_str: str | None, *, data_dir: Path, config_dir: Path,
        retry_transient: bool, listener: DownloadListener | None = None,
        stop: LiveStop | None = None,
) -> ProviderHistory:
    """Download provider data, riding out transient broker outages.

    For a long-running run (``--broker`` / ``--live``) a *transient* provider
    failure — broker maintenance, a lost route, a dropped connection — must not
    strand the bot: we wait with capped exponential backoff and keep retrying
    until the feed returns or the operator interrupts (``Ctrl-C``). A
    *permanent* failure (unknown symbol, bad credentials, wrong account mode),
    and every failure of a one-shot backtest, is raised at once.

    Retrying is intentionally unbounded for the transient class: a bot is meant
    to run until told to stop, so giving up after N attempts would re-introduce
    the very crash we are fixing for any outage longer than the backoff budget.
    The safeguard against looping on a misclassified error is that the transient
    set is narrow and source-specific (see ``is_retryable_provider_error`` and
    the cTrader ``_CONNECTION_CLASS_CODES``), and every wait is logged.

    :param provider_str: The provider string (e.g. ``ctrader:pepperstoneuk:BTCUSD@1``).
    :param time_from_str: The start (date, days, or ``-bars``).
    :param data_dir: See :func:`download_provider_history`.
    :param config_dir: See :func:`download_provider_history`.
    :param retry_transient: Whether to wait-and-retry on transient errors —
        ``True`` only for long-running ``--broker`` / ``--live`` runs.
    :param listener: Receives the progress, the warnings and every retry wait.
    :param stop: Ends a retry wait at once when set from another thread (a front end
        without ``Ctrl-C``); a download pass already running is not interrupted.
    :return: The downloaded history.
    :raises ProviderError: On a permanent failure, or any failure without ``retry_transient``.
    :raises DownloadStopped: If ``stop`` was set before a retry.
    """
    listener = listener or DownloadListener()

    delay = _PROVIDER_RETRY_BASE_DELAY
    attempt = 0
    while True:
        if stop is not None and stop.is_set:
            raise DownloadStopped()
        try:
            return download_provider_history(provider_str, time_from_str, data_dir=data_dir,
                                             config_dir=config_dir, listener=listener)
        except ProviderError as e:
            if not (retry_transient and is_retryable_provider_error(e)):
                raise
            attempt += 1
            listener.retry_wait(e, delay, attempt)
            _wait_before_retry(delay, stop)
            delay = min(delay * 2.0, _PROVIDER_RETRY_MAX_DELAY)


def require_live_provider(history: ProviderHistory) -> LiveProviderPlugin:
    """
    The provider of a downloaded history, checked to stream live data.

    :param history: The downloaded history.
    :return: Its provider instance, as a live provider.
    :raises ProviderSetupError: If the provider has no live data.
    """
    provider = history.provider_instance
    if not isinstance(provider, LiveProviderPlugin):
        raise ProviderSetupError(
            f"Plugin '{history.parsed_string.provider}' does not support live data.")
    return provider


def open_live_stream(history: ProviderHistory, *, last_historical_timestamp: int,
                     shutdown_timeout: float = 120.0,
                     event_loop: 'asyncio.AbstractEventLoop | None' = None,
                     raise_on_connect_failure: bool = False,
                     on_interrupt: Callable[[], None] | None = None,
                     on_feed_status: Callable[[FeedStatus], None] | None = None,
                     stop: LiveStop | None = None,
                     ) -> Generator[OHLCV, None, None]:
    """
    Start the live stream of a downloaded provider history.

    The returned iterator yields the closed bars that arrived during the warmup, then
    :data:`~pynecore.core.script_runner.LIVE_TRANSITION`, then the live updates: chained
    after the history bars it is the ``ohlcv_iter`` of a live
    :class:`~pynecore.core.script_runner.ScriptRunner`. The connection is opened at once,
    before this returns (see :func:`~pynecore.core.live_runner.live_ohlcv_generator`).

    :param history: The downloaded history; its provider instance streams the updates.
    :param last_historical_timestamp: Timestamp (ms) where the history the run executes
        ends, so the stream does not repeat earlier bars.
    :param shutdown_timeout: Max seconds for the graceful shutdown; 0 waits forever.
    :param event_loop: Externally owned event loop, for broker mode.
    :param raise_on_connect_failure: Re-raise a failed first connection here.
    :param on_interrupt: Called when a stop lands in the stream, before its teardown.
    :param on_feed_status: Receives every state change of the feed.
    :param stop: Ends the iterator when set from another thread, with the normal teardown.
    :return: The live iterator.
    :raises ProviderSetupError: If the provider has no live data.
    """
    provider = require_live_provider(history)
    timeframe = history.parsed_string.timeframe
    assert timeframe is not None
    return live_ohlcv_generator(
        provider=provider,
        symbol=history.parsed_string.symbol,
        timeframe=timeframe,
        syminfo=history.syminfo,
        last_historical_timestamp=last_historical_timestamp,
        shutdown_timeout=shutdown_timeout,
        event_loop=event_loop,
        raise_on_connect_failure=raise_on_connect_failure,
        on_interrupt=on_interrupt,
        on_feed_status=on_feed_status,
        stop=stop,
    )
