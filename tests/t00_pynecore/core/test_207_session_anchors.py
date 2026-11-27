"""
TradingView session anchors: the origin of the intraday bar grid and the daily stamp.

MEASURED (TradingView, 2026-11-27, 29 symbols): every intraday grid starts at
TradingView's own session start of the trading day. A provider's reopening after a
short pause is not one: CAPITALCOM:US30 reopens at 17:05 New York, its TradingView
session -- 2-minute grid and daily stamp -- starts at 17:00. These tests cover the
resolver (explicit ``session_anchors`` and the quarter-hour fallback), the resampler,
the aggregator against a real TradingView 2-minute reference, and the TOML field.
"""
import csv
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from pynecore.core.aggregator import aggregate_ohlcv
from pynecore.core.ohlcv import OHLCVReader, OHLCVWriter
from pynecore.core.resampler import Resampler, overnight_opens
from pynecore.core.syminfo import (
    SymInfo, SymInfoAnchor, SymInfoInterval, SymInfoSession, resolve_session_anchors,
)
from pynecore.types.ohlcv import OHLCV

__test_helper_NY = ZoneInfo('America/New_York')
__test_helper_HK = ZoneInfo('Asia/Hong_Kong')
__test_helper_DATA = Path(__file__).parent / 'data'


def __test_helper_ms(tz: ZoneInfo, y: int, mo: int, d: int, h: int = 0, mi: int = 0) -> int:
    return int(datetime(y, mo, d, h, mi, tzinfo=tz).timestamp() * 1000)


def __test_helper_capital_index():
    """
    A Capital.com US index CFD in New York time: a 17:00-17:05 pause Monday to Thursday,
    the Sunday open at 18:00 (the template ``update_symbol_info`` writes, split at the
    UTC midnight).
    """
    hours = [SymInfoInterval(6, time(18), time(18, 59, 59)),
             SymInfoInterval(6, time(19), time(23, 59, 59))]
    for d in range(4):
        hours += [SymInfoInterval(d, time(0), time(17)),
                  SymInfoInterval(d, time(17, 5), time(18, 59, 59)),
                  SymInfoInterval(d, time(19), time(23, 59, 59))]
    hours.append(SymInfoInterval(4, time(0), time(17)))
    starts = [SymInfoSession(6, time(18))] + [SymInfoSession(d, time(17, 5)) for d in range(4)]
    return starts, hours


# --- resolver ---

def __test_reopening_after_pause_resolves_to_quarter_hour__():
    """17:05 after a 17:00 close -> 17:00; the 18:00 Sunday open stays."""
    starts, hours = __test_helper_capital_index()
    opens, grid = resolve_session_anchors(None, starts, hours)
    assert opens == grid
    assert grid == sorted([SymInfoSession(6, time(18))]
                          + [SymInfoSession(d, time(17)) for d in range(4)])


def __test_quarter_hour_opens_stay__():
    """Opens on the quarter hour are TradingView's session starts already."""
    for open_time in (time(9, 30), time(9, 15), time(18)):
        starts = [SymInfoSession(d, open_time) for d in range(5)]
        hours = [SymInfoInterval(d, open_time, time(16)) for d in range(5)]
        assert resolve_session_anchors(None, starts, hours)[1] == starts


def __test_reopening_without_pause_stays__():
    """An open the market trades right up to is not moved (no pause before it)."""
    starts = [SymInfoSession(0, time(17, 5))]
    hours = [SymInfoInterval(0, time(9), time(17, 5)), SymInfoInterval(0, time(17, 5), time(20))]
    assert resolve_session_anchors(None, starts, hours)[1] == starts


def __test_overnight_interval_counts_as_trading__():
    """The next-day part of an overnight interval is trading time on its own day."""
    starts = [SymInfoSession(1, time(0, 5))]
    hours = [SymInfoInterval(0, time(18), time(0, 2)), SymInfoInterval(1, time(0, 5), time(16))]
    assert resolve_session_anchors(None, starts, hours)[1] == starts
    hours = [SymInfoInterval(0, time(18), time(0, 0)), SymInfoInterval(1, time(0, 5), time(16))]
    assert resolve_session_anchors(None, starts, hours)[1] == [SymInfoSession(1, time(0))]


def __test_explicit_anchors_win__():
    """Explicit anchors replace the derived ones; ``align`` moves only the grid."""
    starts, hours = __test_helper_capital_index()
    explicit = [SymInfoAnchor(0, time(9, 30), time(9)), SymInfoAnchor(1, time(9, 30))]
    opens, grid = resolve_session_anchors(explicit, starts, hours)
    assert opens == [SymInfoSession(0, time(9, 30)), SymInfoSession(1, time(9, 30))]
    assert grid == [SymInfoSession(0, time(9)), SymInfoSession(1, time(9, 30))]


def __test_resolving_is_idempotent__():
    """Resolved anchors resolve to themselves (consumers may resolve their input again)."""
    starts, hours = __test_helper_capital_index()
    grid = resolve_session_anchors(None, starts, hours)[1]
    assert resolve_session_anchors(None, grid, hours)[1] == grid
    explicit_grid = [SymInfoSession(d, time(17, 20)) for d in range(4)]
    assert resolve_session_anchors(None, explicit_grid, hours)[1] == explicit_grid


def __test_no_template__():
    """Without opening hours the session starts are the anchors; without starts, none."""
    starts = [SymInfoSession(0, time(17, 5))]
    assert resolve_session_anchors(None, starts, None) == (starts, starts)
    assert resolve_session_anchors(None, None, None) == ([], [])


# --- resampler ---

def __test_two_minute_grid_on_tv_session_start__():
    """The 2-minute bar holding the 17:05 reopening opens at 17:04, not 17:05."""
    starts, hours = __test_helper_capital_index()
    r = Resampler.get_resampler('2')
    ny = __test_helper_NY
    for mi, want in ((5, 4), (6, 6), (7, 6)):
        t = __test_helper_ms(ny, 2026, 11, 17, 17, mi)
        assert r.get_bar_time(t, ny, starts, hours) == __test_helper_ms(ny, 2026, 11, 17, 17, want)


def __test_240_minute_grid_on_tv_session_start__():
    """240-minute bars at 17:00, 21:00, 01:00 (TradingView), not 17:05, 21:05."""
    starts, hours = __test_helper_capital_index()
    r = Resampler.get_resampler('240')
    ny = __test_helper_NY
    assert r.get_bar_time(__test_helper_ms(ny, 2026, 11, 17, 21, 0), ny, starts, hours) \
        == __test_helper_ms(ny, 2026, 11, 17, 21, 0)
    assert r.get_bar_time(__test_helper_ms(ny, 2026, 11, 18, 0, 59), ny, starts, hours) \
        == __test_helper_ms(ny, 2026, 11, 17, 21, 0)


def __test_daily_bar_stamped_at_tv_session_start__():
    """MEASURED (CAPITALCOM:US30 1D): daily bars open 17:00 New York, Sunday 18:00."""
    starts, hours = __test_helper_capital_index()
    r = Resampler.get_resampler('1D')
    ny = __test_helper_NY
    assert r.get_bar_time(__test_helper_ms(ny, 2026, 11, 18, 12), ny, starts, hours) \
        == __test_helper_ms(ny, 2026, 11, 17, 17)
    assert r.get_bar_time(__test_helper_ms(ny, 2026, 11, 16, 12), ny, starts, hours) \
        == __test_helper_ms(ny, 2026, 11, 15, 18)


def __test_aligned_grid_stamps_first_bar_at_open__():
    """
    MEASURED (HKEX:700): open 09:30 on a grid from 09:00 (``A0900``): 60-minute bars
    09:30, 10:00, 11:00, 13:00; 45-minute bars 09:30, 09:45, ..., 12:45.
    """
    hk = __test_helper_HK
    grid = [SymInfoSession(d, time(9)) for d in range(5)]
    opens = [SymInfoSession(d, time(9, 30)) for d in range(5)]
    r60 = Resampler.get_resampler('60')
    r45 = Resampler.get_resampler('45')
    got60 = {r60.get_bar_time(__test_helper_ms(hk, 2026, 11, 26, h, m), hk, grid,
                              session_opens=opens)
             for h, m in ((9, 30), (9, 59), (10, 0), (11, 59), (13, 0), (13, 30))}
    assert got60 == {__test_helper_ms(hk, 2026, 11, 26, h, m)
                     for h, m in ((9, 30), (10, 0), (11, 0), (13, 0))}
    assert r45.get_bar_time(__test_helper_ms(hk, 2026, 11, 26, 9, 40), hk, grid,
                            session_opens=opens) == __test_helper_ms(hk, 2026, 11, 26, 9, 30)
    assert r45.get_bar_time(__test_helper_ms(hk, 2026, 11, 26, 13, 10), hk, grid,
                            session_opens=opens) == __test_helper_ms(hk, 2026, 11, 26, 12, 45)


def __test_overnight_open_moves_to_anchor__():
    """The trading day of an overnight 17:05 reopening rolls at its 17:00 anchor."""
    hours = [SymInfoInterval(d, time(17, 5), time(17)) for d in (6, 0, 1, 2, 3)]
    raw = [SymInfoSession(d, time(17, 5)) for d in (6, 0, 1, 2, 3)]
    anchors = [SymInfoSession(d, time(17)) for d in (6, 0, 1, 2, 3)]
    assert overnight_opens(hours, raw) == {d: time(17, 5) for d in (6, 0, 1, 2, 3)}
    assert overnight_opens(hours, anchors) == {d: time(17) for d in (6, 0, 1, 2, 3)}


# --- aggregator against TradingView ---

def __test_helper_read_csv(name: str) -> list[list[str]]:
    with open(__test_helper_DATA / name, newline='') as f:
        return list(csv.reader(f))[1:]


def __test_aggregate_us30_matches_tradingview_2m__(tmp_path):
    """
    Capital.com US30 1-minute data aggregated to 2 minutes lands on TradingView's
    CAPITALCOM:US30 2-minute bars (Sunday 2026-11-15 17:30 to Tuesday 06:00 New York,
    across the Sunday open and the Monday 17:05 reopening): every timestamp and every
    OHLC value of the 1078 bars.
    """
    source = tmp_path / 'us30_1.ohlcv'
    with OHLCVWriter(source, '1', truncate=True, minmove=1, pricescale=10) as w:
        for t, o, h, lo, c, v in __test_helper_read_csv('session_anchor_us30_1m.csv'):
            w.write(OHLCV(timestamp=int(t), open=float(o), high=float(h), low=float(lo),
                          close=float(c), volume=float(v)))
    starts, hours = __test_helper_capital_index()
    target = tmp_path / 'us30_2.ohlcv'
    aggregate_ohlcv(source, target, '2', tz=__test_helper_NY, session_starts=starts,
                    opening_hours=hours, sym_type='index', source_tf='1', prefix='CAPITALCOM')
    with OHLCVReader(target) as r:
        got = {c.timestamp: (c.open, c.high, c.low, c.close)
               for c in r.read_from(r.start_timestamp)}
    tv = {int(t): (float(o), float(h), float(lo), float(c))
          for t, o, h, lo, c in __test_helper_read_csv('session_anchor_us30_tv_2m.csv')}
    assert len(tv) == 1078
    assert got == tv


def __test_aggregate_explicit_anchor_overrides__(tmp_path):
    """Explicit anchors are the grid origin even where the derived one differs."""
    source = tmp_path / 'src_1.ohlcv'
    ny = __test_helper_NY
    with OHLCVWriter(source, '1', truncate=True) as w:
        for mi in range(50, 70):
            w.write(OHLCV(timestamp=__test_helper_ms(ny, 2026, 11, 17, 17, 0) + mi * 60_000,
                          open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0))
    hours = [SymInfoInterval(1, time(17, 50), time(23, 59, 59))]
    starts = [SymInfoSession(1, time(17, 50))]
    target = tmp_path / 'dst.ohlcv'
    aggregate_ohlcv(source, target, '30', tz=ny, session_starts=starts, opening_hours=hours,
                    session_anchors=[SymInfoAnchor(1, time(17, 30))])
    with OHLCVReader(target) as r:
        stamps = [c.timestamp for c in r.read_from(r.start_timestamp)]
    assert stamps == [__test_helper_ms(ny, 2026, 11, 17, 17, 30),
                      __test_helper_ms(ny, 2026, 11, 17, 18, 0)]


# --- TOML ---

def __test_session_anchors_toml_round_trip__(tmp_path):
    """``[[session_anchors]]`` with and without ``align`` survive save and load."""
    starts, hours = __test_helper_capital_index()
    si = SymInfo(prefix='HKEX', description='x', ticker='700', currency='HKD', period='1',
                 type='stock', mintick=0.2, pricescale=5, pointvalue=1.0, mincontract=1.0,
                 opening_hours=hours, session_starts=starts, session_ends=[],
                 session_anchors=[SymInfoAnchor(0, time(9, 30), time(9)),
                                  SymInfoAnchor(1, time(9, 30))],
                 timezone='Asia/Hong_Kong')
    path = tmp_path / 'sym.toml'
    si.save_toml(path)
    loaded = SymInfo.load_toml(path)
    assert loaded.session_anchors == si.session_anchors
    si.session_anchors = []
    si.save_toml(path)
    assert SymInfo.load_toml(path).session_anchors == []
    assert '# [[session_anchors]]' in path.read_text()


def __test_security_aligned_grid_stamps_first_bar_at_open__():
    """
    A ``request.security`` context on HKEX:700 with explicit ``A0900`` anchors: the
    09:00 grid lies on the UTC hour, yet the 60-minute context anchors, and its opening
    bar -- in the resampler and in the developing-bar aggregator -- opens at 09:30.
    """
    from pynecore.core.htf_aggregator import HTFAggregator
    from pynecore.core.security import resolve_session_anchor
    hk = __test_helper_HK
    hours = []
    for d in range(5):
        hours += [SymInfoInterval(d, time(9, 30), time(12)),
                  SymInfoInterval(d, time(13), time(16))]
    si = SymInfo(prefix='HKEX', description='x', ticker='700', currency='HKD', period='1',
                 type='stock', mintick=0.2, pricescale=5, pointvalue=1.0, mincontract=1.0,
                 opening_hours=hours,
                 session_starts=[SymInfoSession(d, time(9, 30)) for d in range(5)],
                 session_ends=[],
                 session_anchors=[SymInfoAnchor(d, time(9, 30), time(9)) for d in range(5)],
                 timezone='Asia/Hong_Kong')
    starts, tz, oh = resolve_session_anchor(si, '60', hk)
    assert starts == [SymInfoSession(d, time(9)) for d in range(5)]
    assert tz is not None and oh == hours
    r = Resampler.get_resampler('60')
    assert r.get_bar_time(__test_helper_ms(hk, 2026, 11, 26, 9, 45), tz, starts, oh) \
        == __test_helper_ms(hk, 2026, 11, 26, 9, 30)
    assert r.get_bar_time(__test_helper_ms(hk, 2026, 11, 26, 10, 15), tz, starts, oh) \
        == __test_helper_ms(hk, 2026, 11, 26, 10)
    # An empty ``session_opens`` gives the plain grid slot (the time_close helper)
    assert r.get_bar_time(__test_helper_ms(hk, 2026, 11, 26, 9, 30), tz, starts, oh,
                          session_opens=()) == __test_helper_ms(hk, 2026, 11, 26, 9)
    agg = HTFAggregator('60', tz, session_starts=starts, chart_span_ms=15 * 60_000,
                        opening_hours=oh)
    _, dev, _ = agg.update(__test_helper_ms(hk, 2026, 11, 26, 9, 30), 1, 1, 1, 1, 1)
    assert dev is not None and dev.period_start == __test_helper_ms(hk, 2026, 11, 26, 9, 30)
    # A 30-minute context needs no anchoring: the 09:30 open lies on its clock grid
    assert resolve_session_anchor(si, '30', hk) == (None, None, None)


def __test_helper_hkex_syminfo() -> SymInfo:
    """HKEX:700 with TradingView's ``A0900`` anchors: open 09:30, grid from 09:00."""
    hours = []
    for d in range(5):
        hours += [SymInfoInterval(d, time(9, 30), time(12)),
                  SymInfoInterval(d, time(13), time(16))]
    return SymInfo(prefix='HKEX', description='x', ticker='700', currency='HKD', period='1',
                   type='stock', mintick=0.2, pricescale=5, pointvalue=1.0, mincontract=1.0,
                   opening_hours=hours,
                   session_starts=[SymInfoSession(d, time(9, 30)) for d in range(5)],
                   session_ends=[],
                   session_anchors=[SymInfoAnchor(d, time(9, 30), time(9)) for d in range(5)],
                   timezone='Asia/Hong_Kong')


def __test_aligned_opening_bar_closes_at_grid_slot_end__():
    """
    The 09:30 opening bar of an HKEX 60-minute grid from 09:00 closes at 10:00 even
    with no successor known (the script-timeframe path, a feed's last bar).
    """
    from pynecore.core.security import actual_bar_close, chart_bar_calendar
    hk = __test_helper_HK
    cal = chart_bar_calendar(__test_helper_hkex_syminfo(), hk)
    assert actual_bar_close(__test_helper_ms(hk, 2026, 11, 26, 9, 30), 0, cal, '60') \
        == __test_helper_ms(hk, 2026, 11, 26, 10)
    assert actual_bar_close(__test_helper_ms(hk, 2026, 11, 26, 10), 0, cal, '60') \
        == __test_helper_ms(hk, 2026, 11, 26, 11)
    assert actual_bar_close(__test_helper_ms(hk, 2026, 11, 26, 11), 0, cal, '60') \
        == __test_helper_ms(hk, 2026, 11, 26, 12)


def __test_off_grid_provider_bar_keeps_its_period__():
    """
    A provider's hourly feed stamped 10:30 and 11:30 is not on the HKEX grid from 09:00:
    the alignment shortens only the clipped 09:30 open, so the 10:30 bar still closes
    at 11:30 and cannot be confirmed before trades it holds.
    """
    from pynecore.core.security import actual_bar_close, chart_bar_calendar
    hk = __test_helper_HK
    cal = chart_bar_calendar(__test_helper_hkex_syminfo(), hk)
    assert actual_bar_close(__test_helper_ms(hk, 2026, 11, 26, 10, 30), 0, cal, '60') \
        == __test_helper_ms(hk, 2026, 11, 26, 11, 30)
    assert actual_bar_close(__test_helper_ms(hk, 2026, 11, 26, 9, 30), 0, cal, '60') \
        == __test_helper_ms(hk, 2026, 11, 26, 10)


def __test_lunch_bucket_holding_the_reopening_closes_with_it__():
    """
    A 45-minute HKEX grid from 09:00 stamps the bucket holding the 13:00 reopening at
    12:45; with a short 13:00-13:15 afternoon it closes at 13:15, not at 13:30. A
    30-minute lunch bucket from 12:00 that ends before the reopening has no session.
    """
    from pynecore.core.security import BarCalendar, _session_bar_closes, actual_bar_close
    hk = __test_helper_HK
    hours = [SymInfoInterval(3, time(9, 30), time(12)), SymInfoInterval(3, time(13), time(13, 15))]
    grid = [SymInfoSession(3, time(9))]
    stamp = __test_helper_ms(hk, 2026, 11, 26, 12, 45)
    assert _session_bar_closes([stamp], hk, hours, 45 * 60_000, anchors=grid) \
        == [__test_helper_ms(hk, 2026, 11, 26, 13, 15)]
    cal = BarCalendar(tz=hk, opening_hours=tuple(hours),
                      session_starts=(SymInfoSession(3, time(9, 30)),),
                      session_grid=tuple(grid))
    assert actual_bar_close(stamp, 0, cal, '45') == __test_helper_ms(hk, 2026, 11, 26, 13, 15)
    assert _session_bar_closes([__test_helper_ms(hk, 2026, 11, 26, 12)], hk, hours,
                               30 * 60_000, anchors=grid) is None


def __test_pre_open_stamp_closes_with_its_session__():
    """
    A bar stamped on the grid before its session's open closes with that session: the
    7-minute bar at 17:07 of a grid derived from a 17:13 reopening (anchor 17:00), and
    the 60-minute bar of an explicit 16:45 anchor for a 17:05-17:20 session.
    """
    from pynecore.core.security import BarCalendar, _session_bar_closes, actual_bar_close
    ny = __test_helper_NY
    hours = [SymInfoInterval(1, time(0), time(17)), SymInfoInterval(1, time(17, 13), time(17, 20))]
    _, grid = resolve_session_anchors(None, [SymInfoSession(1, time(17, 13))], hours)
    assert grid == [SymInfoSession(1, time(17))]
    stamp = __test_helper_ms(ny, 2026, 11, 17, 17, 7)
    assert _session_bar_closes([stamp], ny, hours, 7 * 60_000, anchors=grid) \
        == [__test_helper_ms(ny, 2026, 11, 17, 17, 14)]
    # Without an anchor before it a stamp outside every session stays unmatched
    assert _session_bar_closes([stamp], ny, hours, 7 * 60_000) is None
    cal = BarCalendar(tz=ny, opening_hours=(SymInfoInterval(1, time(17, 5), time(17, 20)),),
                      session_starts=(SymInfoSession(1, time(17, 5)),),
                      session_grid=(SymInfoSession(1, time(16, 45)),))
    assert actual_bar_close(__test_helper_ms(ny, 2026, 11, 17, 16, 45), 0, cal, '60') \
        == __test_helper_ms(ny, 2026, 11, 17, 17, 20)


def __test_bucket_across_midnight_closes_with_the_reopening__():
    """
    A 60-minute grid from Monday 22:30 stamps the bucket holding Tuesday's 00:05
    reopening at Monday 23:30; it closes at that session's 00:20 end, not at 00:30.
    """
    from pynecore.core.security import BarCalendar, _session_bar_closes, actual_bar_close
    hk = __test_helper_HK
    hours = [SymInfoInterval(0, time(18), time(23)), SymInfoInterval(1, time(0, 5), time(0, 20))]
    grid = [SymInfoSession(0, time(22, 30))]
    stamp = __test_helper_ms(hk, 2026, 11, 23, 23, 30)
    close = __test_helper_ms(hk, 2026, 11, 24, 0, 20)
    assert _session_bar_closes([stamp], hk, hours, 60 * 60_000, anchors=grid) == [close]
    cal = BarCalendar(tz=hk, opening_hours=tuple(hours),
                      session_starts=(SymInfoSession(0, time(18)),),
                      session_grid=tuple(grid))
    assert actual_bar_close(stamp, 0, cal, '60') == close


def __test_bucket_across_lunch_closes_after_its_afternoon_trades__():
    """
    A 150-minute HKEX grid from 09:00 aggregates the 11:45 and 13:59 candles into its
    11:30 bar; the bar closes at 14:00 with its afternoon trades, not at the noon break.
    """
    from pynecore.core.security import _session_bar_closes, actual_bar_close, chart_bar_calendar
    hk = __test_helper_HK
    si = __test_helper_hkex_syminfo()
    opens, grid = si.resolved_session_anchors()
    r = Resampler.get_resampler('150')
    bar = __test_helper_ms(hk, 2026, 11, 26, 11, 30)
    for h, mi in ((11, 45), (13, 59)):
        assert r.get_bar_time(__test_helper_ms(hk, 2026, 11, 26, h, mi), hk, grid,
                              si.opening_hours) == bar
    close = __test_helper_ms(hk, 2026, 11, 26, 14)
    assert _session_bar_closes([bar], hk, si.opening_hours, 150 * 60_000, anchors=grid) \
        == [close]
    assert actual_bar_close(bar, 0, chart_bar_calendar(si, hk), '150') == close
    # The next grid origin ends the bucket: the 14:00 bar closes at the 16:00 session end
    assert actual_bar_close(close, 0, chart_bar_calendar(si, hk), '150') \
        == __test_helper_ms(hk, 2026, 11, 26, 16)


def __test_provider_opening_bar_keeps_its_period_by_its_successor__():
    """
    A provider's hourly feed stamped 09:30 and 10:30 shares its opening stamp with the
    clipped HKEX 09:30 bar; the off-grid 10:30 successor shows the provider's cadence,
    so the 09:30 bar closes at 10:30. A successor on the grid keeps the 10:00 close.
    """
    from pynecore.core.security import actual_bar_close, chart_bar_calendar
    hk = __test_helper_HK
    cal = chart_bar_calendar(__test_helper_hkex_syminfo(), hk)
    stamp = __test_helper_ms(hk, 2026, 11, 26, 9, 30)
    assert actual_bar_close(stamp, __test_helper_ms(hk, 2026, 11, 26, 10, 30), cal, '60') \
        == __test_helper_ms(hk, 2026, 11, 26, 10, 30)
    assert actual_bar_close(stamp, __test_helper_ms(hk, 2026, 11, 26, 10), cal, '60') \
        == __test_helper_ms(hk, 2026, 11, 26, 10)
    assert actual_bar_close(stamp, __test_helper_ms(hk, 2026, 11, 26, 11), cal, '60') \
        == __test_helper_ms(hk, 2026, 11, 26, 10)


def __test_grid_origin_inside_a_session_caps_the_close__():
    """
    An explicit 12:05 grid origin inside continuous 09:00-16:00 trading starts a new
    bar: the 12:00 hourly bar holds 12:04 only and closes at 12:05, not at 13:00, on
    both close paths, with no successor known.
    """
    from pynecore.core.security import BarCalendar, _session_bar_closes, actual_bar_close
    ny = __test_helper_NY
    hours = [SymInfoInterval(3, time(9), time(16))]
    grid = [SymInfoSession(3, time(9)), SymInfoSession(3, time(12, 5))]
    r = Resampler.get_resampler('60')
    bar = __test_helper_ms(ny, 2026, 11, 26, 12)
    assert r.get_bar_time(__test_helper_ms(ny, 2026, 11, 26, 12, 4), ny, grid, hours) == bar
    assert r.get_bar_time(__test_helper_ms(ny, 2026, 11, 26, 12, 5), ny, grid, hours) \
        == __test_helper_ms(ny, 2026, 11, 26, 12, 5)
    close = __test_helper_ms(ny, 2026, 11, 26, 12, 5)
    assert _session_bar_closes([bar], ny, hours, 60 * 60_000, anchors=grid) == [close]
    cal = BarCalendar(tz=ny, opening_hours=tuple(hours),
                      session_starts=(SymInfoSession(3, time(9)),),
                      session_grid=tuple(grid))
    assert actual_bar_close(bar, 0, cal, '60') == close


def __test_clipped_opening_bucket_ignores_the_next_slot_session__():
    """
    The 09:30 opening bar of an HKEX 240-minute grid from 09:00 holds the morning
    session only: its slot ends at 13:00, where the afternoon session opens a new bar,
    so it closes at the 12:00 morning close on both close paths, with or without the
    13:00 successor, not at 13:00 because the afternoon session lies within a full
    period of the stamp.
    """
    from pynecore.core.security import (
        _bar_bucket_ends, _session_bar_closes, actual_bar_close, calendar_session_anchors,
        chart_bar_calendar,
    )
    hk = __test_helper_HK
    si = __test_helper_hkex_syminfo()
    cal = chart_bar_calendar(si, hk)
    r = Resampler.get_resampler('240')
    stamp = __test_helper_ms(hk, 2026, 11, 26, 9, 30)
    afternoon = __test_helper_ms(hk, 2026, 11, 26, 13)
    grid = list(calendar_session_anchors(si)['session_grid'])
    assert r.get_bar_time(__test_helper_ms(hk, 2026, 11, 26, 13, 5), hk, grid,
                          si.opening_hours, session_opens=()) == afternoon
    noon = __test_helper_ms(hk, 2026, 11, 26, 12)
    assert actual_bar_close(stamp, 0, cal, '240') == noon
    assert actual_bar_close(stamp, afternoon, cal, '240') == noon
    opens = [stamp, afternoon]
    period_ms = 240 * 60_000
    closes = _session_bar_closes(opens, hk, si.opening_hours, period_ms,
                                 anchors=grid,
                                 bucket_ends=_bar_bucket_ends(opens, cal, '240', period_ms))
    assert closes == [noon, __test_helper_ms(hk, 2026, 11, 26, 16)]


def __test_chart_time_close_keeps_provider_cadence_by_its_successor__():
    """
    Bare ``time_close()`` on the 09:30 bar of an HKEX hourly chart follows the chart's
    next bar the way security does: a 10:30 successor shows the provider's cadence and
    closes the bar at 10:30, a 10:00 one keeps the clipped 10:00 close, and without a
    successor the stamp alone gives TradingView's 10:00.
    """
    from pynecore import lib
    from pynecore.lib import syminfo as lib_syminfo
    hk = __test_helper_HK
    si = __test_helper_hkex_syminfo()
    stamp = __test_helper_ms(hk, 2026, 11, 26, 9, 30)
    keys = ('timezone', 'period', '_opening_hours', '_session_starts', 'session_anchors',
            'session')
    saved = {k: getattr(lib_syminfo, k, None) for k in keys}
    saved_lib = (lib._time, lib._next_time, lib._datetime, lib._script_timeframe,
                 lib._main_timeframe)
    try:
        lib_syminfo.timezone = si.timezone
        lib_syminfo.period = '60'
        lib_syminfo._opening_hours = si.opening_hours
        lib_syminfo._session_starts = si.session_starts
        lib_syminfo.session_anchors = si.session_anchors
        lib_syminfo.session = si.session
        lib._script_timeframe = None
        lib._main_timeframe = None
        lib._time = stamp
        lib._datetime = datetime.fromtimestamp(stamp / 1000, tz=hk)
        for nxt, expected in ((__test_helper_ms(hk, 2026, 11, 26, 10, 30), (10, 30)),
                              (__test_helper_ms(hk, 2026, 11, 26, 10), (10, 0)),
                              (0, (10, 0))):
            lib._next_time = nxt
            assert lib.time_close() == __test_helper_ms(hk, 2026, 11, 26, *expected)
    finally:
        for k, v in saved.items():
            setattr(lib_syminfo, k, v)
        (lib._time, lib._next_time, lib._datetime, lib._script_timeframe,
         lib._main_timeframe) = saved_lib
