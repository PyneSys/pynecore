"""
OHLCV timeframe aggregation — converts lower timeframe data to higher timeframes.

Uses Resampler.get_bar_time() for correct bar boundary alignment across all
timeframe types including weekly and monthly. Multi-period targets (nD/nW/nM,
n > 1) live on the year-reset scheduled grid (see ``resampler`` module docs);
'observed' symbols (exchange-listed) count the actual trading days seen in the
source stream, which reproduces TradingView's holiday-aware grid.
"""
from datetime import timezone as dt_timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .ohlcv_file import OHLCVReader, OHLCVWriter
from .resampler import (
    Resampler, ObservedDayCounter, grid_mode, overnight_opens, trading_day,
    first_monday,
)
from ..lib.timeframe import in_seconds, _process_tf
from ..types.ohlcv import OHLCV


def validate_aggregation(source_tf: str, target_tf: str) -> None:
    """
    Validate that aggregation from source to target timeframe is possible.

    :param source_tf: Source timeframe string (e.g., '5', '1D')
    :param target_tf: Target timeframe string (e.g., '60', '1W')
    :raises ValueError: If timeframes are incompatible
    """
    source_sec = in_seconds(source_tf)
    target_sec = in_seconds(target_tf)

    if target_sec <= source_sec:
        raise ValueError(
            f"Target timeframe ({target_tf}) must be larger than "
            f"source timeframe ({source_tf})"
        )

    if target_sec % source_sec != 0:
        raise ValueError(
            f"Target timeframe ({target_tf}) must be evenly divisible by "
            f"source timeframe ({source_tf})"
        )


def _merge_candles(candles: list[OHLCV], bar_time: int) -> OHLCV:
    """
    Merge a window of candles into a single aggregated candle.

    :param candles: Non-empty list of OHLCV candles belonging to the same bar
    :param bar_time: Aligned bar opening timestamp in seconds
    :return: Aggregated OHLCV candle
    """
    return OHLCV(
        timestamp=bar_time,
        open=candles[0].open,
        high=max(c.high for c in candles),
        low=min(c.low for c in candles),
        close=candles[-1].close,
        volume=sum(c.volume for c in candles),
    )


def aggregate_ohlcv(
        source_path: Path,
        target_path: Path,
        target_tf: str,
        tz: ZoneInfo | dt_timezone | None = None,
        session_starts: list | None = None,
        opening_hours: list | None = None,
        sym_type: str | None = None,
        source_tf: str | None = None,
) -> tuple[int, int]:
    """
    Aggregate OHLCV data from a lower timeframe file to a higher timeframe file.

    :param source_path: Path to source .ohlcv file
    :param target_path: Path to target .ohlcv file (will be overwritten)
    :param target_tf: Target timeframe string (e.g., '60', '1W')
    :param tz: Timezone for day/week/month boundary alignment.
               Should match the data's timezone (from TOML metadata).
    :param session_starts: Per-trading-day primary opens for intraday session
               anchoring and multi-period grids. When given, intraday bars align
               to the session open (TradingView behaviour) instead of the UTC
               clock; ``None`` keeps the pure clock-floor.
               See :meth:`Resampler.get_bar_time`.
    :param opening_hours: ``SymInfo.opening_hours`` — trading-day roll source
               for multi-period (nD/nW/nM) grids.
    :param sym_type: ``SymInfo.type`` for :func:`grid_mode` classification of
               multi-period grids. Exchange-listed symbols ('observed' mode)
               count the actual trading days present in the source data; for an
               exact TradingView-matching grid their source data should reach
               back to the year's first trading day (the in-year counter is
               otherwise approximated from the weekday grid).
    :param source_tf: Source timeframe string. On multi-period (nD/nW/nM)
               targets an intraday source bar belongs to the trading day its
               *last* instant falls into — the bar containing a session open
               starts the new day even when its timestamp precedes the open
               (see the ``resampler`` module docs).
    :return: Tuple of (source_candles_read, target_candles_written)
    """
    # noinspection PyProtectedMember
    modifier, multiplier = _process_tf(target_tf)
    mode = grid_mode(sym_type, opening_hours)

    src_off = 0
    if modifier in ('D', 'W', 'M') and multiplier > 1 and source_tf:
        # noinspection PyProtectedMember
        src_mod, _ = _process_tf(source_tf)
        if src_mod in ('', 'S'):
            src_off = in_seconds(source_tf) - 1

    if modifier in ('D', 'W', 'M') and multiplier > 1 and mode == 'observed':
        return _aggregate_observed(
            source_path, target_path, modifier, multiplier, tz,
            session_starts, opening_hours, src_off)

    resampler = Resampler.get_resampler(target_tf)

    source_count = 0
    target_count = 0

    with OHLCVReader(source_path) as reader:
        with OHLCVWriter(target_path, truncate=True) as writer:
            window: list[OHLCV] = []
            current_bar_time: int | None = None

            start_ts = reader.start_timestamp
            if start_ts is None:
                if reader.size == 1:
                    # A single-record source has no derivable interval (the reader
                    # needs two timestamps to infer one), so ``start_timestamp`` is
                    # None and ``read_from`` yields nothing — yet that lone bar IS a
                    # whole target period and must be emitted. Floor its timestamp
                    # onto the target grid (exactly as the loop below does) so HTF
                    # confirmation (the ``bar_opens`` clamp) and
                    # ``request.security(.., time)`` see the period boundary, not the
                    # raw sub-bar instant.
                    only = reader.read(0)
                    only_bar_time = resampler.get_bar_time(
                        (only.timestamp + src_off) * 1000, tz=tz,
                        session_starts=session_starts,
                        opening_hours=opening_hours, mode=mode) // 1000
                    writer.write(_merge_candles([only], only_bar_time))
                    return 1, 1
                return 0, 0

            for candle in reader.read_from(start_ts):
                source_count += 1

                # Resampler works in ms, OHLCV timestamps are in seconds.
                # src_off resolves multi-period bars by their last instant.
                bar_time_ms = resampler.get_bar_time(
                    (candle.timestamp + src_off) * 1000, tz=tz,
                    session_starts=session_starts,
                    opening_hours=opening_hours, mode=mode)
                bar_time = bar_time_ms // 1000

                if current_bar_time is not None and bar_time != current_bar_time:
                    # New bar boundary — flush the window
                    writer.write(_merge_candles(window, current_bar_time))
                    target_count += 1
                    window = []

                current_bar_time = bar_time
                window.append(candle)

            # Flush last window
            if window and current_bar_time is not None:
                writer.write(_merge_candles(window, current_bar_time))
                target_count += 1

    return source_count, target_count


def _aggregate_observed(
        source_path: Path,
        target_path: Path,
        modifier: str,
        multiplier: int,
        tz: ZoneInfo | dt_timezone | None,
        session_starts: list | None,
        opening_hours: list | None,
        src_off: int = 0,
) -> tuple[int, int]:
    """
    Multi-period aggregation for 'observed' symbols (exchange-listed).

    TradingView's grid on these symbols counts its holiday calendar's scheduled
    trading days; the actual daily data realizes that calendar, so counting the
    trading days present in the source stream reproduces the grid (year-reset
    counter, verified 100% on CME 2022+). Periods are stamped with their first
    source candle — TradingView's stamp on these symbols is the period's first
    actual session.

    The first (partial) year's counter is seeded from the weekday grid: the
    trading days between Jan 1 and the data start are not observable, so the
    phase there is approximate. Data reaching back to a year start is exact
    from that year on.

    :param source_path: Path to source .ohlcv file
    :param target_path: Path to target .ohlcv file (will be overwritten)
    :param modifier: 'D', 'W' or 'M' (from ``_process_tf``)
    :param multiplier: Period multiplier (> 1)
    :param tz: Exchange timezone
    :param session_starts: ``SymInfo.session_starts`` template
    :param opening_hours: ``SymInfo.opening_hours`` template
    :param src_off: Source bar open -> last instant offset in seconds; a bar
               belongs to the trading day its last instant falls into
    :return: Tuple of (source_candles_read, target_candles_written)
    """
    on = overnight_opens(opening_hours, session_starts)

    source_count = 0
    target_count = 0

    with OHLCVReader(source_path) as reader:
        with OHLCVWriter(target_path, truncate=True) as writer:
            window: list[OHLCV] = []
            window_start: int | None = None
            group_key: tuple | None = None
            counter = ObservedDayCounter()

            start_ts = reader.start_timestamp
            if start_ts is None:
                return 0, 0

            for candle in reader.read_from(start_ts):
                source_count += 1
                td = trading_day(candle.timestamp + src_off, tz, on)

                if modifier == 'D':
                    # Count observed trading days, counter resets each year
                    key = (td.year, counter.ordinal(td) // multiplier)
                elif modifier == 'W':
                    # Weeks are always realized — the scheduled week grid is
                    # exact; only the stamp comes from the data
                    monday = td - timedelta(days=td.weekday())
                    weeks = (monday - first_monday(monday.year)).days // 7
                    key = (monday.year, weeks // multiplier)
                else:  # 'M'
                    key = (td.year, (td.month - 1) // multiplier)

                if group_key is not None and key != group_key:
                    writer.write(_merge_candles(window, window_start))
                    target_count += 1
                    window = []
                    window_start = None

                group_key = key
                if window_start is None:
                    window_start = candle.timestamp
                window.append(candle)

            if window and window_start is not None:
                writer.write(_merge_candles(window, window_start))
                target_count += 1

    return source_count, target_count
