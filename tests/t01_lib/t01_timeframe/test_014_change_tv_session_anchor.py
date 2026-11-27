"""
@pyne
"""
from pynecore.lib import script, timeframe, plot, time


@script.indicator(title="Timeframe Change TV Session Anchor", shorttitle="tf_change_tv_anchor")
def main():
    plot(time("45"), "t45")
    plot(time("60"), "t60")
    plot(time("240"), "t240")
    plot(1 if timeframe.change("45") else 0, "c45")
    plot(1 if timeframe.change("60") else 0, "c60")
    plot(1 if timeframe.change("240") else 0, "c240")


def __test_helper_compare(runner, fixture, syminfo_override):
    """
    Run the script on TradingView's own 1-minute bar times and count the bars where
    time() or timeframe.change of "45", "60" and "240" differ from TradingView's.

    MEASURED (TradingView, 2026-11-27): the fixtures hold, per 1-minute bar,
    time("45"/"60"/"240") and timeframe.change("45"/"60"/"240") as TradingView
    computed them.
    """
    import csv
    from pathlib import Path
    from pynecore.types.ohlcv import OHLCV

    with open(Path(__file__).parent / 'data' / fixture, newline='') as f:
        rows = list(csv.DictReader(f))
    bars = [OHLCV(timestamp=int(r['time']), open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0)
            for r in rows]
    diffs: dict[str, list[int]] = {}
    compared = 0
    for (candle, plots), r in zip(runner(bars, syminfo_override).run_iter(), rows):
        compared += 1
        for key in ('t45', 't60', 't240', 'c45', 'c60', 'c240'):
            got = int(plots[key]) if key[0] == 't' else int(bool(plots[key]))
            if got != int(r[key]):
                diffs.setdefault(key, []).append(candle.timestamp)
    assert compared == len(rows) > 0
    # The first bar never reports a change (no previous bar to compare with)
    for key in ('c45', 'c60', 'c240'):
        if diffs.get(key) == [bars[0].timestamp]:
            del diffs[key]
    return diffs


def __test_change_aapl_off_hour_open__(runner):
    """ NASDAQ:AAPL 09:30 open: 60-minute changes at 10:30, 11:30 on a 1-minute chart """
    from datetime import time as dt_time
    from pynecore.core.syminfo import SymInfoInterval, SymInfoSession

    days = range(5)
    override = dict(
        prefix="NASDAQ", ticker="AAPL", period="1", type="stock", timezone="America/New_York",
        opening_hours=[SymInfoInterval(d, dt_time(9, 30), dt_time(16)) for d in days],
        session_starts=[SymInfoSession(d, dt_time(9, 30)) for d in days],
        session_ends=[SymInfoSession(d, dt_time(16)) for d in days],
        session_anchors=[],
    )
    assert __test_helper_compare(runner, 'session_anchor_aapl_tv_1m_htf.csv', override) == {}


def __test_change_capitalcom_reopening__(runner):
    """
    Capital.com's US30 template (17:05 reopening after a 17:00 close) gives
    TradingView's CAPITALCOM:US30 grid, which starts at 17:00
    """
    from datetime import time as dt_time
    from pynecore.core.syminfo import SymInfoInterval, SymInfoSession

    hours = [SymInfoInterval(6, dt_time(18), dt_time(18, 59, 59)),
             SymInfoInterval(6, dt_time(19), dt_time(23, 59, 59))]
    for d in range(4):
        hours += [SymInfoInterval(d, dt_time(0), dt_time(17)),
                  SymInfoInterval(d, dt_time(17, 5), dt_time(18, 59, 59)),
                  SymInfoInterval(d, dt_time(19), dt_time(23, 59, 59))]
    hours.append(SymInfoInterval(4, dt_time(0), dt_time(17)))
    override = dict(
        prefix="CAPITALCOM", ticker="US30", period="1", type="index", timezone="US/Eastern",
        opening_hours=hours,
        session_starts=[SymInfoSession(6, dt_time(18))]
                       + [SymInfoSession(d, dt_time(17, 5)) for d in range(4)],
        session_ends=[SymInfoSession(d, dt_time(17)) for d in range(5)],
        session_anchors=[],
    )
    assert __test_helper_compare(runner, 'session_anchor_us30_tv_1m_htf.csv', override) == {}


def __test_change_hkex_aligned_grid__(runner):
    """ HKEX:700 opens 09:30 on a grid aligned at 09:00 (the ``A0900`` session token) """
    from datetime import time as dt_time
    from pynecore.core.syminfo import SymInfoAnchor, SymInfoInterval, SymInfoSession

    days = range(5)
    hours = []
    for d in days:
        hours += [SymInfoInterval(d, dt_time(9, 30), dt_time(12)),
                  SymInfoInterval(d, dt_time(13), dt_time(16))]
    override = dict(
        prefix="HKEX", ticker="700", period="1", type="stock", timezone="Asia/Hong_Kong",
        opening_hours=hours,
        session_starts=[SymInfoSession(d, dt_time(9, 30)) for d in days]
                       + [SymInfoSession(d, dt_time(13)) for d in days],
        session_ends=[SymInfoSession(d, dt_time(16)) for d in days],
        session_anchors=[SymInfoAnchor(d, dt_time(9, 30), dt_time(9)) for d in days],
    )
    assert __test_helper_compare(runner, 'session_anchor_hkex_tv_1m_htf.csv', override) == {}
