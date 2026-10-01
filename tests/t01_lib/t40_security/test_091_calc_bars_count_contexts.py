"""
@pyne
"""
from pynecore.lib import array, bar_index, plot, request, script


@script.indicator(title="Calc Bars Count Contexts", shorttitle="CBCC", calc_bars_count=10)
def main():
    plot(request.security("EXCH:SYM", "60", bar_index), "h_bi")
    ltf = request.security_lower_tf("EXCH:SYM", "15", bar_index)
    plot(array.size(ltf), "ltf_n")
    plot(array.first(ltf) if array.size(ltf) > 0 else -1, "ltf_first")


# Every timestamp here is Unix MILLISECONDS; the chart is 30 bars of 30 minutes
# from 2025-01-01 00:00 UTC, so its last 10 (calculated) bars open 10:00 .. 14:30.
__test_helper_ts0 = 1_735_689_600_000
__test_helper_minute = 60_000


def __test_helper_write_feed(tmp_dir, period, minutes, count):
    """Write a 24/7 UTC ``EXCH:SYM`` feed of ``count`` bars of ``minutes`` from ts0."""
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / f"EXCH_SYM_{period}.ohlcv"
    with OHLCVWriter(path, period) as w:
        for i in range(count):
            c = 100.0 + i
            w.write(OHLCV(timestamp=__test_helper_ts0 + i * minutes * __test_helper_minute,
                          open=c, high=c, low=c, close=c, volume=1.0))
    SymInfo(
        prefix="EXCH", description="Calc Window Symbol", ticker="SYM",
        currency="USD", period=period, type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base", taker_fee=0.1, maker_fee=0.1,
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))
    return str(path)


def __test_calc_bars_count_cuts_every_context__(runner, log):
    """Each security context calculates only its own last
    ``max(1, calc_bars_count * chart_tf // context_tf)`` bars

    MEASURED on TradingView (BINANCE:BTCUSDT@30 and NASDAQ:AAPL@30 with
    ``calc_bars_count`` 500, every context plotting its own ``bar_index``): "60"
    holds 250 bars and "15" 1000, so the finer context opens one bar BEFORE the
    chart's window, ``request.security_lower_tf`` included. Here: 10 chart bars
    at 30 minutes give the "60" context 5 bars (10:00 .. 14:00) and the "15" one
    20 bars (09:45 .. 14:30).
    """
    import os
    import tempfile
    from pathlib import Path
    from pynecore.types.ohlcv import OHLCV
    from pynecore.types.na import NA

    os.environ['PYNE_SAVE_SCRIPT_TOML'] = '0'
    chart = [OHLCV(timestamp=__test_helper_ts0 + i * 30 * __test_helper_minute,
                   open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0)
             for i in range(30)]
    with tempfile.TemporaryDirectory() as td:
        hourly = __test_helper_write_feed(Path(td), "60", 60, 15)
        quarter = __test_helper_write_feed(Path(td), "15", 15, 60)
        r = runner(chart, syminfo_override={"period": "30"}, last_bar_index=len(chart) - 1,
                   last_bar_time=chart[-1].timestamp,
                   security_data={"EXCH:SYM:60": hourly, "EXCH:SYM:15": quarter})
        rows = [dict(pv) for _candle, pv in r.run_iter()]

    assert len(rows) == 10
    # The 10:00 hourly bar is the context's first: it confirms on the 10:30 chart bar
    h_bi = [None if isinstance(row["h_bi"], NA) else row["h_bi"] for row in rows]
    assert h_bi == [None, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0]
    # The 09:45 intrabar is the LTF context's bar 0, so the 10:00 chart bar holds 1 and 2
    assert [row["ltf_n"] for row in rows] == [2.0] * 10
    assert [row["ltf_first"] for row in rows] == [float(1 + 2 * i) for i in range(10)]
    log.info("60 context starts at the window, 15 context one intrabar before it")
