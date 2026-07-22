"""
Shared OHLCV download core.

The ``.ohlcv`` download flow (start-date resolution, range clamping, truncate
handling, progress accounting, symbol-info persistence) is driven by several
front-ends: the ``pyne data download`` CLI, the symbol-browser TUI and the
IDE bridge's provider service. They only differ in how they render progress
and how they answer the "the file would have to be truncated" question, so
everything else lives here.

This module is part of the non-interactive ``core`` layer: it never prompts,
never prints and knows nothing about typer/rich. Front-ends pass callbacks.
"""
from __future__ import annotations

from typing import Callable, Literal, TypeAlias
from dataclasses import dataclass
from datetime import datetime, timedelta, UTC
from pathlib import Path

from .download_info import write_download_provider
from .ohlcv_file import OHLCVWriter
from .plugin.provider import ProviderPlugin
from .syminfo import SymInfo

__all__ = [
    'ConflictAction', 'DownloadConflict', 'DownloadPlan', 'DownloadProgress',
    'DownloadResult', 'DownloadError', 'DownloadConflictError',
    'InvalidTimeRangeError', 'download_to_file',
]

ConflictAction: TypeAlias = Literal['truncate', 'abort']
"""What to do when the requested start date precedes the first bar of the
existing file: drop the file's content or raise. Appending before the existing
first bar is not possible, the writer only ever appends at the end."""


@dataclass(frozen=True)
class DownloadConflict:
    """The requested start date is before the first bar of the existing file."""
    ohlcv_path: Path
    time_from: datetime
    existing_start: datetime


@dataclass(frozen=True)
class DownloadPlan:
    """The resolved download range, handed to ``on_start`` before the first
    provider request so front-ends can size their progress display."""
    ohlcv_path: Path
    time_from: datetime
    time_to: datetime
    fetch_all: bool
    total_seconds: int
    """Length of the range in seconds, 0 for a ``fetch_all`` download (whose
    end is unknown, so no proportional progress can be shown)."""


@dataclass(frozen=True)
class DownloadProgress:
    """One progress tick. Progress is time-proportional: the provider reports
    the timestamp it has reached, not a bar count."""
    current: datetime
    elapsed_seconds: int
    total_seconds: int


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of a finished download."""
    ohlcv_path: Path
    time_from: datetime
    time_to: datetime
    fetch_all: bool
    bars_written: int
    """Number of records the file grew by. Bars that overwrote already present
    records do not count, so this is a lower bound of the fetched bars."""
    syminfo: SymInfo | None
    """The symbol info belonging to the file. When the caller passed one in, it
    is the very same object, whose ``mincontract`` may have been refined."""


class DownloadError(Exception):
    """Base class of the errors raised by :func:`download_to_file`."""


class InvalidTimeRangeError(DownloadError):
    """The resolved end date is before the resolved start date."""


class DownloadConflictError(DownloadError):
    """The download would have truncated an existing file and the caller's
    ``on_conflict`` policy was ``'abort'``."""

    def __init__(self, conflict: DownloadConflict):
        super().__init__(
            f"The start date (from: {conflict.time_from}) is before the start of the "
            f"existing file ({conflict.existing_start}); "
            f"downloading it would truncate the file."
        )
        self.conflict = conflict


def download_to_file(
        provider: ProviderPlugin, *,
        time_from: datetime | str,
        time_to: datetime,
        truncate: bool = False,
        chunk_size: int | None = None,
        extra_data: bool = False,
        syminfo: SymInfo | None = None,
        on_start: Callable[[DownloadPlan], None] | None = None,
        on_progress: Callable[[DownloadProgress], None] | None = None,
        on_conflict: "ConflictAction | Callable[[DownloadConflict], ConflictAction]" = 'abort',
        provider_string: str | None = None,
) -> DownloadResult:
    """
    Download OHLCV data into the provider's ``.ohlcv`` file.

    The provider must already be bound to a symbol (built with one, so its
    ``ohlcv_path`` and writer are set); every front-end constructs it that way,
    which for multi-broker providers keeps the broker in the filename.

    :param provider: The provider instance to download with.
    :param time_from: Start date, or the ``"continue"`` sentinel to resume the
        existing file (falling back to all available data for
        ``fetch_all_by_default`` providers, otherwise one year back). Aware
        datetimes are converted to UTC, naive ones are taken as UTC.
    :param time_to: End date (same timezone handling as ``time_from``); clamped
        to "now", as the future takes forever to download.
    :param truncate: Drop the existing file content before downloading.
    :param chunk_size: Override the provider's per-request bar count.
    :param extra_data: Also fetch the provider's extra fields (``.extra.csv``).
    :param syminfo: Already-loaded symbol info. When None it is fetched (and
        persisted) *before* the download, best-effort, so an interrupted
        download still leaves a resumable file.
    :param on_start: Called once with the resolved :class:`DownloadPlan`, before
        the first OHLCV request.
    :param on_progress: Called with :class:`DownloadProgress` as the download
        advances. Never called for a ``fetch_all`` download.
    :param on_conflict: Policy or callback deciding what to do when the start
        date precedes the existing file's first bar. A callback may prompt (the
        CLI does) and must return one of the :data:`ConflictAction` values.
    :param provider_string: Canonical provider string to persist in the
        ``[download]`` section of the syminfo TOML, written *before* the
        download so the file stays re-downloadable by path even if the download
        is cut short. None skips it.
    :return: The :class:`DownloadResult` of the finished download.
    :raises InvalidTimeRangeError: If the end date is before the start date.
    :raises DownloadConflictError: If a truncating conflict was answered with
        ``'abort'``.
    """
    assert provider.ohlcv_path is not None
    ohlcv_path = provider.ohlcv_path

    with provider as ohlcv_writer:
        if truncate:
            ohlcv_writer.seek(0)
            ohlcv_writer.truncate()

        resolved_from, fetch_all = _resolve_from(provider, time_from, ohlcv_writer)

        # The rest of the flow works with naive UTC datetimes
        resolved_from = _to_naive_utc(resolved_from)
        resolved_to = _to_naive_utc(time_to)

        # We cannot download data from the future otherwise it would take very long
        now_naive = datetime.now(UTC).replace(tzinfo=None)
        if resolved_to > now_naive:
            resolved_to = now_naive

        if not fetch_all and resolved_to < resolved_from:
            raise InvalidTimeRangeError(
                "End date (to) must be greater than start date (from)!")

        if ohlcv_writer.start_timestamp and not fetch_all:
            existing_start = ohlcv_writer.start_datetime.replace(tzinfo=None)
            if resolved_from < existing_start:
                conflict = DownloadConflict(ohlcv_path=ohlcv_path, time_from=resolved_from,
                                            existing_start=existing_start)
                action = on_conflict(conflict) if callable(on_conflict) else on_conflict
                if action == 'abort':
                    raise DownloadConflictError(conflict)
                if action != 'truncate':
                    raise ValueError(f"Invalid conflict action: {action!r}")
                ohlcv_writer.seek(0)
                ohlcv_writer.truncate()

        # Persist the symbol info and the originating provider string BEFORE the
        # (potentially long) download, so an interrupted or user-aborted download
        # still leaves a resumable file: the [download] section lets it be
        # continued by path, and "continue" picks up from the last written bar.
        if syminfo is None:
            # noinspection PyBroadException
            try:
                syminfo = provider.get_symbol_info()  # save_toml() side effect
            except Exception:
                syminfo = None  # Symbol info is best-effort, don't block the download
        if provider_string is not None:
            # No-op until the syminfo TOML exists; get_symbol_info() above (or the
            # caller) is what creates it.
            write_download_provider(ohlcv_path.with_suffix('.toml'), provider_string)

        total_seconds = 0 if fetch_all else max(0, int((resolved_to - resolved_from).total_seconds()))
        if on_start is not None:
            on_start(DownloadPlan(ohlcv_path=ohlcv_path, time_from=resolved_from,
                                  time_to=resolved_to, fetch_all=fetch_all,
                                  total_seconds=total_seconds))

        progress_cb = on_progress

        def cb_progress(current_time: datetime) -> None:
            """ Callback to report time-proportional progress """
            assert progress_cb is not None
            elapsed = min(max(int((current_time - resolved_from).total_seconds()), 0), total_seconds)
            progress_cb(DownloadProgress(current=current_time, elapsed_seconds=elapsed,
                                         total_seconds=total_seconds))

        cb = cb_progress if (progress_cb is not None and not fetch_all) else None

        size_before = ohlcv_writer.size
        provider.download_ohlcv(resolved_from, resolved_to, on_progress=cb,
                                limit=chunk_size, with_extra=extra_data)
        bars_written = ohlcv_writer.size - size_before

        # Refine the heuristic mincontract from the downloaded volume data
        # (only when the provider had no exchange value for it). save_toml()
        # preserves the [download] section written before the download.
        if syminfo is not None and provider.mincontract_estimated:
            qty_step = ohlcv_writer.analyzed_qty_step or 0.0
            if qty_step > 0.0 and qty_step != syminfo.mincontract:
                syminfo.mincontract = qty_step
                syminfo.save_toml(ohlcv_path.with_suffix('.toml'))

    return DownloadResult(ohlcv_path=ohlcv_path, time_from=resolved_from, time_to=resolved_to,
                          fetch_all=fetch_all, bars_written=bars_written, syminfo=syminfo)


def _resolve_from(provider: ProviderPlugin, time_from: datetime | str,
                  ohlcv_writer: OHLCVWriter) -> tuple[datetime, bool]:
    """
    Resolve the ``"continue"`` start-date sentinel against the existing file.

    :param provider: The provider instance (its ``fetch_all_by_default``
        decides what "no data yet" means).
    :param time_from: Start date or the ``"continue"`` sentinel.
    :param ohlcv_writer: The already opened writer of the target file.
    :return: ``(start_date, fetch_all)``.
    """
    if isinstance(time_from, datetime):
        return time_from, False

    assert time_from == "continue", f"Unexpected from value: {time_from!r}"

    end_ts = ohlcv_writer.end_timestamp
    interval = ohlcv_writer.interval
    if end_ts and interval:  # Resume from last download
        # One interval ahead, otherwise the last bar would be downloaded again
        return datetime.fromtimestamp(end_ts, UTC) + timedelta(seconds=interval), False
    if provider.fetch_all_by_default:
        return datetime.fromtimestamp(0, UTC), True
    return datetime.now(UTC) - timedelta(days=365), False


def _to_naive_utc(dt: datetime) -> datetime:
    """
    Convert a datetime to naive UTC, the form the download flow works with.

    :param dt: Aware or naive (already UTC) datetime.
    :return: Naive UTC datetime.
    """
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(UTC).replace(tzinfo=None)
