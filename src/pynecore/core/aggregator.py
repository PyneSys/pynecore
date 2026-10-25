"""
OHLCV timeframe aggregation — converts lower timeframe data to higher timeframes.

Uses Resampler.get_bar_time() for correct bar boundary alignment across all
timeframe types including weekly and monthly. Multi-period targets (nD/nW/nM,
n > 1) live on the year-reset scheduled grid (see ``resampler`` module docs);
'observed' symbols (exchange-listed) count the actual trading days seen in the
source stream, which reproduces TradingView's holiday-aware grid.
"""
import logging
from datetime import date, datetime, timezone as dt_timezone
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from .ohlcv import OHLCVWriter
from .ohlcv import OHLCVReader
from .ohlcv import restore_f32_volume
from .resampler import (
    Resampler, ObservedDayCounter, grid_mode, overnight_opens, trading_day,
)
# noinspection PyProtectedMember
from .security import _day_session_ends, _exclusive_session_end, _match_session_end
# noinspection PyProtectedMember
from ..lib.timeframe import _in_seconds, _process_tf
from ..types.ohlcv import OHLCV

logger = logging.getLogger(__name__)


def validate_aggregation(source_tf: str, target_tf: str) -> None:
    """
    Validate that aggregation from source to target timeframe is possible.

    :param source_tf: Source timeframe string (e.g., '5', '1D')
    :param target_tf: Target timeframe string (e.g., '60', '1W')
    :raises ValueError: If timeframes are incompatible
    """
    source_sec = _in_seconds(source_tf)
    target_sec = _in_seconds(target_tf)

    if target_sec <= source_sec:
        raise ValueError(
            f"Target timeframe ({target_tf}) must be larger than "
            f"source timeframe ({source_tf})"
        )

    target_mod, target_mult = _process_tf(target_tf)
    if target_mod in ('W', 'M'):
        # Weeks and months are calendar periods without a fixed length in seconds, so
        # a source fits when every one of its bars lies inside a single period: an
        # intraday bar that tiles the day, a single day, or the same calendar unit
        # with a multiplier that divides the target's (both grids reset each year)
        source_mod, source_mult = _process_tf(source_tf)
        if source_mod == target_mod:
            fits = target_mult % source_mult == 0
        elif source_mod == 'D':
            fits = source_mult == 1
        elif source_mod in ('', 'S'):
            fits = 86400 % source_sec == 0
        else:
            fits = False
        if not fits:
            raise ValueError(
                f"Source timeframe ({source_tf}) bars do not fit evenly into "
                f"target timeframe ({target_tf}) periods"
            )
        return

    if target_sec % source_sec != 0:
        raise ValueError(
            f"Target timeframe ({target_tf}) must be evenly divisible by "
            f"source timeframe ({source_tf})"
        )


def _merge_candles(candles: list[OHLCV], bar_time: int,
                   restore_volume: bool = False) -> OHLCV:
    """
    Merge a window of candles into a single aggregated candle.

    :param candles: Non-empty list of OHLCV candles belonging to the same bar
    :param bar_time: Aligned bar opening timestamp in milliseconds
    :param restore_volume: Whether the source volumes carry float32 storage
               error that has to be undone before summing
    :return: Aggregated OHLCV candle
    """
    # The target file stores volume as an absolute f64, so its values are read
    # back as written — a consumer has no clean-up left to do. A legacy source
    # therefore has to be restored HERE, per source bar (exactly what the chart
    # does before summing), or the sum carries the float32 dust forever.
    volumes = (restore_f32_volume(c.volume) for c in candles) if restore_volume \
        else (c.volume for c in candles)
    # TradingView aggregates volume server-side as the exact DECIMAL sum of the feed's
    # volumes. A binary float sum drifts from it by an ulp on many bars, and the drift
    # carries into every volume-weighted series computed on the aggregated context.
    return OHLCV(
        timestamp=bar_time,
        open=candles[0].open,
        high=max(c.high for c in candles),
        low=min(c.low for c in candles),
        close=candles[-1].close,
        volume=float(sum(Decimal(repr(v)) for v in volumes)),
    )


class _SourceBarEnd:
    """
    Last instant of an intraday source bar, the instant that decides its D/W/M period.

    A source bar belongs to the period its *last* instant falls into: the bar containing
    a session open starts the new trading day even when its own timestamp precedes the
    open. The last instant is the bar's scheduled close minus one millisecond, and a
    session end inside the bar's nominal span cuts that close short -- the same rule
    the runtime applies to chart bars (``lib._chart_span_off_ms``). Without opening
    hours the nominal span stands.
    """

    __slots__ = ('_span_ms', '_tz', '_hours', '_day', '_ends')

    def __init__(self, source_tf: str, tz: ZoneInfo | dt_timezone,
                 opening_hours: list | None) -> None:
        """
        :param source_tf: Intraday source timeframe string
        :param tz: Exchange timezone of the source data
        :param opening_hours: ``SymInfo.opening_hours`` of the source data
        """
        self._span_ms = _in_seconds(source_tf) * 1000
        self._tz = tz
        self._hours = list(opening_hours) if opening_hours else []
        self._day: date | None = None
        self._ends: tuple = ()

    def last_ms(self, open_ms: int) -> int:
        """
        :param open_ms: Source bar open in epoch milliseconds
        :return: The bar's last instant in epoch milliseconds
        """
        close_ms = open_ms + self._span_ms
        if self._hours:
            open_dt = datetime.fromtimestamp(open_ms / 1000, tz=self._tz)
            day = open_dt.date()
            if day != self._day:
                # The candidate session ends depend on the calendar date alone
                self._day = day
                self._ends = _day_session_ends(day, self._tz, self._hours)
            session_end = _match_session_end(open_dt.time(), self._ends)
            if session_end is not None and session_end < close_ms:
                session_end = _exclusive_session_end(session_end, self._tz)
                close_ms = min(close_ms, session_end)
        return close_ms - 1


def aggregate_ohlcv(
        source_path: Path,
        target_path: Path,
        target_tf: str,
        tz: ZoneInfo | dt_timezone = dt_timezone.utc,
        session_starts: list | None = None,
        opening_hours: list | None = None,
        sym_type: str | None = None,
        source_tf: str | None = None,
        prefix: str | None = None,
) -> tuple[int, int]:
    """
    Aggregate OHLCV data from a lower timeframe file to a higher timeframe file.

    :param source_path: Path to source .ohlcv file
    :param target_path: Path to target .ohlcv file (will be overwritten)
    :param target_tf: Target timeframe string (e.g., '60', '1W')
    :param tz: Timezone for day/week/month boundary alignment.
               Should match the data's timezone (from TOML metadata).
               Defaults to UTC, never the host's local zone, so the same
               files aggregate identically on every machine.
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
    :param source_tf: Source timeframe string. On D/W/M targets an intraday
               source bar belongs to the trading day its *last* instant falls
               into — the bar containing a session open starts the new day even
               when its timestamp precedes the open (see :class:`_SourceBarEnd`).
               A multi-day source on an 'observed' nD target counts as that many
               trading days.
    :param prefix: ``SymInfo.prefix``, the feed the data comes from; a CFD feed
               has no exchange calendar and stays on the weekday grid.
    :return: Tuple of (source_candles_read, target_candles_written)
    :raises ValueError: If ``target_path`` is the source file (also via a symlink
               or hard link) — truncating it would destroy the input before it is read
    """
    if target_path.exists() and source_path.samefile(target_path):
        raise ValueError(f"Target {target_path} is the source file; aggregate to another path")

    # noinspection PyProtectedMember
    modifier, multiplier = _process_tf(target_tf)
    # The written file declares its period with an explicit multiplier ('D' -> '1D').
    target_period = f"{multiplier}{modifier}" if modifier else str(multiplier)
    mode = grid_mode(sym_type, opening_hours, prefix)

    bar_end: _SourceBarEnd | None = None
    day_step = 1
    if modifier in ('D', 'W', 'M') and source_tf:
        # noinspection PyProtectedMember
        src_mod, src_mult = _process_tf(source_tf)
        if src_mod in ('', 'S'):
            bar_end = _SourceBarEnd(source_tf, tz, opening_hours)
        elif src_mod == 'D':
            # Each nD source bar spans n trading days of the observed day count
            day_step = src_mult

    if modifier in ('D', 'W', 'M') and multiplier > 1 and mode == 'observed':
        if not opening_hours and not session_starts:
            logger.warning(
                "%s: no opening_hours or session_starts in the symbol info; %s%s periods "
                "are grouped by calendar date, so overnight sessions (e.g. a Sunday "
                "evening futures open) count as their own trading day",
                source_path.name, multiplier, modifier)
        return _aggregate_observed(
            source_path, target_path, modifier, multiplier, tz,
            session_starts, opening_hours, bar_end, day_step)

    resampler = Resampler.get_resampler(target_tf)

    source_count = 0
    target_count = 0

    with OHLCVReader(source_path) as reader:
        with OHLCVWriter(target_path, target_period, truncate=True,
                         minmove=reader.minmove,
                         pricescale=reader.pricescale) as writer, writer.batched():
            window: list[OHLCV] = []
            current_bar_time: int | None = None
            restore_volume = not reader.lossless_volume

            start_ts = reader.start_timestamp
            if start_ts is None:
                return 0, 0

            for candle in reader.read_from(start_ts):
                source_count += 1

                # An intraday source bar resolves to its D/W/M period by its last instant
                bar_time = resampler.get_bar_time(
                    candle.timestamp if bar_end is None else bar_end.last_ms(candle.timestamp),
                    tz=tz,
                    session_starts=session_starts,
                    opening_hours=opening_hours, mode=mode)

                if current_bar_time is not None and bar_time != current_bar_time:
                    # New bar boundary — flush the window
                    writer.write(_merge_candles(window, current_bar_time, restore_volume))
                    target_count += 1
                    window = []

                current_bar_time = bar_time
                window.append(candle)

            # Flush last window
            if window and current_bar_time is not None:
                writer.write(_merge_candles(window, current_bar_time, restore_volume))
                target_count += 1

    return source_count, target_count


def _aggregate_observed(
        source_path: Path,
        target_path: Path,
        modifier: str,
        multiplier: int,
        tz: ZoneInfo | dt_timezone,
        session_starts: list | None,
        opening_hours: list | None,
        bar_end: _SourceBarEnd | None = None,
        day_step: int = 1,
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
    :param bar_end: Last-instant resolver of an intraday source; a bar belongs to
               the trading day its last instant falls into, and the bar ends drive
               the holiday half-day fold (see :class:`ObservedDayCounter`). ``None``
               for a daily or coarser source, which is already folded.
    :param day_step: Trading days one source bar spans (the ``n`` of an nD source)
    :return: Tuple of (source_candles_read, target_candles_written)
    """
    on = overnight_opens(opening_hours, session_starts)

    source_count = 0
    target_count = 0

    with OHLCVReader(source_path) as reader:
        with OHLCVWriter(target_path, f"{multiplier}{modifier}", truncate=True,
                         minmove=reader.minmove,
                         pricescale=reader.pricescale) as writer, writer.batched():
            window: list[OHLCV] = []
            window_start: int | None = None
            group_key: tuple | None = None
            counter = ObservedDayCounter(tz, opening_hours, fold=bar_end is not None,
                                         step=day_step)
            restore_volume = not reader.lossless_volume

            start_ts = reader.start_timestamp
            if start_ts is None:
                return 0, 0

            for candle in reader.read_from(start_ts):
                source_count += 1
                # The trading-day calendar works in seconds, OHLCV timestamps in ms.
                if bar_end is None:
                    td = trading_day(candle.timestamp // 1000, tz, on)
                    counter.ordinal(td)
                else:
                    last_sec = bar_end.last_ms(candle.timestamp) // 1000
                    td = trading_day(last_sec, tz, on)
                    counter.ordinal(td, last_sec + 1)
                key = counter.key(modifier, multiplier)

                if group_key is not None and key != group_key:
                    # A non-None group key means at least one bar was windowed.
                    assert window_start is not None
                    writer.write(_merge_candles(window, window_start, restore_volume))
                    target_count += 1
                    window = []
                    window_start = None

                group_key = key
                if window_start is None:
                    window_start = candle.timestamp
                window.append(candle)

            if window and window_start is not None:
                writer.write(_merge_candles(window, window_start, restore_volume))
                target_count += 1

    return source_count, target_count
