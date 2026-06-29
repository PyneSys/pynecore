"""
@pyne
"""
from pynecore.lib import (
    barmerge, close, high, low, open, plot, request, script, syminfo,
)
from pynecore.types import Series


@script.indicator(title="Intraday Multi-Period Confirm", shorttitle="IMPC")
def main():
    # A finer (hourly) base feed queried at D / 3D / 5D. Each must expose the
    # last confirmed period's AGGREGATED bar (open=first, high=max, low=min,
    # close=last), held across the developing period — never a single raw
    # sub-bar and never the developing close.
    o3: Series[float] = request.security(
        syminfo.tickerid, "3D", open, lookahead=barmerge.lookahead_off)
    h3: Series[float] = request.security(
        syminfo.tickerid, "3D", high, lookahead=barmerge.lookahead_off)
    l3: Series[float] = request.security(
        syminfo.tickerid, "3D", low, lookahead=barmerge.lookahead_off)
    c3: Series[float] = request.security(
        syminfo.tickerid, "3D", close, lookahead=barmerge.lookahead_off)
    cd: Series[float] = request.security(
        syminfo.tickerid, "D", close, lookahead=barmerge.lookahead_off)
    c5: Series[float] = request.security(
        syminfo.tickerid, "5D", close, lookahead=barmerge.lookahead_off)
    plot(o3, "o3")
    plot(h3, "h3")
    plot(l3, "l3")
    plot(c3, "c3")
    plot(cd, "cd")
    plot(c5, "c5")


# A dense hourly 24/7 feed. ``base = (day + 1) * 1000 + hour``; open == close ==
# base, the HIGH spikes +5000 at hour 12 and the LOW dips -5000 at hour 6, so a
# period's true high/low fall on NON-last bars — distinguishing a real aggregate
# (max/min over the period) from a single exposed sub-bar. Calendar grid, year
# reset from 2025-01-01: 3D -> [0,1,2][3,4,5][6,7,8]; 5D -> [0..4][5..8].
_T0 = 1735689600  # 2025-01-01T00:00:00 UTC, aligned to the day grid
_N_DAYS = 9


def _base(day, hour):
    return float((day + 1) * 1000 + hour)


def _ohlc(day, hour):
    b = _base(day, hour)
    hi = b + (5000.0 if hour == 12 else 0.0)
    lo = b - (5000.0 if hour == 6 else 0.0)
    return b, hi, lo, b  # open, high, low, close


def _write_feed(tmp_dir):
    """Write the hourly 24/7 feed (+ a 24/7 UTC ``.toml``); return the path."""
    from datetime import time
    from pynecore.core.ohlcv_file import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / "FEED.ohlcv"
    with OHLCVWriter(path) as w:
        for day in range(_N_DAYS):
            for hour in range(24):
                o, hi, lo, c = _ohlc(day, hour)
                w.write(OHLCV(timestamp=_T0 + (day * 24 + hour) * 3600,
                              open=o, high=hi, low=lo, close=c, volume=1.0))
    SymInfo(
        prefix="EXCH", description="Multi-period", ticker="IMPC",
        currency="USD", period="60", type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))
    return str(path)


def _chart_bars():
    from pynecore.types.ohlcv import OHLCV
    out = []
    for day in range(_N_DAYS):
        for hour in range(24):
            o, hi, lo, c = _ohlc(day, hour)
            out.append(OHLCV(timestamp=_T0 + (day * 24 + hour) * 3600,
                             open=o, high=hi, low=lo, close=c, volume=1.0))
    return out


def __test_intraday_multiperiod_aggregates_and_holds__(runner, log):
    """Multi-day ``request.security`` exposes the last confirmed period AGGREGATE.

    Regression for issue #70 (and the same latent value bug on single-period
    ``D``): with a base feed finer than the security timeframe, the native
    ``--security`` child used to expose a single raw sub-bar of the period — the
    developing period for multi-day (the visible #70 symptom), or the period's
    first sub-bar after the timing was corrected. The full fix pre-resamples the
    finer feed to the security timeframe so the child reads ONE aggregated bar per
    period: ``open`` = the period's first open, ``high`` = its max, ``low`` = its
    min, ``close`` = its last close — held across the developing period and
    matching TradingView, exactly as it already does for a period-resolution feed.
    """
    import sys
    import tempfile
    from pathlib import Path
    from pynecore.types.na import NA

    sys.modules.pop(Path(__file__).stem, None)

    rows = {}
    with tempfile.TemporaryDirectory() as td:
        feed = _write_feed(Path(td))
        r = runner(_chart_bars(), syminfo_override={"period": "60"}, security_data={"D": feed, "3D": feed, "5D": feed})
        for candle, pv in r.run_iter():
            mins = (candle.timestamp - _T0) // 60
            rows[(mins // (60 * 24), (mins // 60) % 24)] = (
                candle.close, pv.get("o3"), pv.get("h3"),
                pv.get("l3"), pv.get("c3"), pv.get("cd"), pv.get("c5"))

    def cell(day, hour, idx):
        return rows[(day, hour)][idx]

    # On an HOURLY chart, ``lookahead_off`` confirms a higher-timeframe bar on
    # its LAST hourly bar — the chart bar whose close instant (h23 -> next
    # midnight) reaches the period end — then HOLDS that aggregate across the
    # whole next developing period. The confirmed value is therefore the just-
    # closed period, never a developing sub-bar and never the developing close.

    # ── 3D, periods [0,1,2][3,4,5][6,7,8] ──────────────────────────────────
    # Each aggregate is (open=first, high=max, low=min, close=last); the spikes
    # at h12 (+5000) and dips at h6 (-5000) make all four distinct, so a single
    # exposed sub-bar (o==h==l==c) could never pass.
    agg012 = (_base(0, 0), _base(2, 12) + 5000.0, _base(0, 6) - 5000.0, _base(2, 23))  # 1000 8012 -3994 3023
    agg345 = (_base(3, 0), _base(5, 12) + 5000.0, _base(3, 6) - 5000.0, _base(5, 23))  # 4000 11012 -994 6023
    agg678 = (_base(6, 0), _base(8, 12) + 5000.0, _base(6, 6) - 5000.0, _base(8, 23))  # 7000 14012 2006 9023

    def expected_3d(day, hour):
        t = day * 24 + hour
        if t < 2 * 24 + 23:        # before day2 23:00 -> period [0,1,2] developing
            return None
        if t < 5 * 24 + 23:        # day2 23:00 .. day5 22:00 -> hold [0,1,2]
            return agg012
        if t < 8 * 24 + 23:        # day5 23:00 .. day8 22:00 -> hold [3,4,5]
            return agg345
        return agg678              # day8 23:00 -> hold [6,7,8]

    for day in range(_N_DAYS):
        for hour in range(24):
            ohlc = (cell(day, hour, 1), cell(day, hour, 2),
                    cell(day, hour, 3), cell(day, hour, 4))
            exp = expected_3d(day, hour)
            if exp is None:
                assert isinstance(ohlc[3], NA), \
                    f"3D day {day} h{hour}: c3={ohlc[3]} should be na (period developing)"
            else:
                assert ohlc == exp, f"3D day {day} h{hour}: ohlc={ohlc} != {exp}"

    # ── Single-period D: the daily CLOSE (last bar), not the day's first bar.
    # The daily bar for day d confirms on its own h23; before that the prior
    # day's close holds. Day 0 has no prior daily bar, so it stays na until h23.
    def expected_d(day, hour):
        if day == 0 and hour < 23:
            return None
        if hour == 23:
            return _base(day, 23)       # today's daily close confirms on the last hour
        return _base(day - 1, 23)       # hold previous day's close

    for day in range(_N_DAYS):
        for hour in range(24):
            cd = cell(day, hour, 5)
            exp_cd = expected_d(day, hour)
            if exp_cd is None:
                assert isinstance(cd, NA), f"D day {day} h{hour}: cd={cd} should be na"
            else:
                assert cd == exp_cd, f"D day {day} h{hour}: cd={cd} != {exp_cd}"

    # ── 5D, period [0..4]: na while it develops, then hold its close (confirmed
    # on day4 h23). The next 5D period [5..9] never completes (the feed ends on
    # day8), so c5 holds the [0..4] close through to the end.
    def expected_5d(day, hour):
        t = day * 24 + hour
        if t < 4 * 24 + 23:        # before day4 23:00 -> period [0..4] developing
            return None
        return _base(4, 23)        # day4 23:00 onward -> hold [0..4] close

    for day in range(_N_DAYS):
        for hour in range(24):
            c5 = cell(day, hour, 6)
            exp_c5 = expected_5d(day, hour)
            if exp_c5 is None:
                assert isinstance(c5, NA), f"5D day {day} h{hour}: c5={c5} should be na (developing)"
            else:
                assert c5 == exp_c5, f"5D day {day} h{hour}: c5={c5} != {exp_c5}"

    # The original #70 signature: the multi-day value tracking the developing
    # chart close. On every HOLD bar the frozen aggregate must differ from the
    # moving chart close; only on a confirmation bar (the period's last hour)
    # does the period close legitimately equal the chart bar's own close.
    confirm_3d = {(2, 23), (5, 23), (8, 23)}
    confirm_5d = {(4, 23)}
    for (day, hour), vals in rows.items():
        for idx, name, confirm in ((4, "3D", confirm_3d), (6, "5D", confirm_5d)):
            v = vals[idx]
            if not isinstance(v, NA) and (day, hour) not in confirm:
                assert v != vals[0], f"{name} day {day} h{hour}: tracks developing close {vals[0]}"

    log.info("intraday 3D/5D/D request.security exposes the last confirmed period aggregate")


def __test_resample_finer_security_feed_aggregates_to_period_bars__(log):
    """``_resample_finer_security_feed`` turns a finer feed into one aggregate bar per period.

    The spawn-time helper resamples a base feed finer than the security timeframe
    to that timeframe; an at/above-resolution feed is returned unchanged (no temp
    file). This pins the aggregation that makes the #70 fix work end to end.
    """
    import shutil
    import tempfile
    from pathlib import Path
    from datetime import time
    from pynecore.core.ohlcv_file import OHLCVReader, OHLCVWriter
    from pynecore.core.script_runner import _resample_finer_security_feed
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        path = tmp / "FEED.ohlcv"
        with OHLCVWriter(path) as w:
            for day in range(9):
                for hour in range(24):
                    o, hi, lo, c = _ohlc(day, hour)
                    w.write(OHLCV(timestamp=_T0 + (day * 24 + hour) * 3600,
                                  open=o, high=hi, low=lo, close=c, volume=1.0))
        SymInfo(prefix="E", description="d", ticker="MP", currency="USD", period="60",
                type="crypto", mintick=0.01, pricescale=100, minmove=1, pointvalue=1,
                mincontract=0.0001, timezone="UTC", volumetype="base",
                opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                               for i in range(7)],
                session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
                session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
                ).save_toml(path.with_suffix(".toml"))

        holder: list[str] = []
        out_path = _resample_finer_security_feed(str(path), "3D", holder)
        assert out_path != str(path), "a finer feed must be resampled to a temp file"
        assert holder, "a temp dir must be created for the resampled feed"

        with OHLCVReader(out_path) as reader:
            bars = list(reader.read_from(reader.start_timestamp))
        # 9 days -> three 3D periods [0,1,2] [3,4,5] [6,7,8].
        assert len(bars) == 3, f"expected 3 aggregated 3D bars, got {len(bars)}"
        b0 = bars[0]
        assert b0.timestamp == _T0, f"period [0,1,2] opens at day 0 ({b0.timestamp} != {_T0})"
        assert b0.open == _base(0, 0), f"open={b0.open} != {_base(0, 0)}"
        assert b0.high == _base(2, 12) + 5000.0, f"high={b0.high} != {_base(2, 12) + 5000.0}"
        assert b0.low == _base(0, 6) - 5000.0, f"low={b0.low} != {_base(0, 6) - 5000.0}"
        assert b0.close == _base(2, 23), f"close={b0.close} != {_base(2, 23)}"

        # The resampled sidecar carries the security timeframe.
        out_si = SymInfo.load_toml(Path(out_path).with_suffix(".toml"))
        assert out_si.period == "3D", f"resampled sidecar period={out_si.period} != 3D"

        # A feed already at the security resolution is a no-op (returned unchanged).
        noop = _resample_finer_security_feed(str(path), "60", holder)
        assert noop == str(path), "an at-resolution feed must be returned unchanged"

        for d in holder:
            shutil.rmtree(d, ignore_errors=True)

    log.info("_resample_finer_security_feed aggregates a finer feed to one bar per period")
