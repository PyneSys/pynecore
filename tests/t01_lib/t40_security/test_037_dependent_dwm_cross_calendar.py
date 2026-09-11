"""
@pyne
"""
from pynecore.lib import close, plot, request, script
from pynecore.types import Series


@script.indicator(title="DWM Cross Calendar", shorttitle="DXC")
def main():
    # A 24/7 WEEKLY producer read from an equity daily chart. The two calendars
    # differ, so the as-of calendar extension does not apply: the consumer never
    # reaches past its own 16:00 close, and the crypto week (closing Monday
    # 00:00 UTC) becomes visible on the first equity bar whose close reaches it.
    w: Series[float] = request.security("EXCH:CRYPTO", "W", close)
    dep: Series[float] = request.security("NYSE:PEER", "D", w + 0.5)
    plot(w, "w")
    plot(dep, "dep")


# Every timestamp here is Unix MILLISECONDS. 2025-01-06 is a Monday.
_T0_EQ = 1_736_173_800_000  # 2025-01-06T14:30:00Z == 09:30 New York (winter, UTC-5)
_T0_UTC = 1_736_121_600_000  # 2025-01-06T00:00:00Z, the crypto week's open
_DAY = 86_400_000
_WEEK = 7 * _DAY
_N_WEEKS = 5
# The Monday of this week is an equity market holiday: the chart has no bar for
# it, so the crypto week that closed at its midnight arrives a bar late.
_HOLIDAY_WEEK = 3


def __test_helper_crypto_weekly_close(week):
    return 7000.0 + week


def __test_helper_equity_syminfo(period):
    from datetime import time
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    return SymInfo(
        prefix="NYSE", description="Equity", ticker="DXC",
        currency="USD", period=period, type="stock",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=1,
        timezone="America/New_York", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(9, 30), end=time(16, 0))
                       for i in range(5)],
        session_starts=[SymInfoSession(day=i, time=time(9, 30)) for i in range(5)],
        session_ends=[SymInfoSession(day=i, time=time(16, 0)) for i in range(5)],
    )


def __test_helper_write_crypto_weekly(tmp_dir):
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / "CRYPTOW.ohlcv"
    with OHLCVWriter(path, "1W") as w:
        for week in range(_N_WEEKS):
            c = __test_helper_crypto_weekly_close(week)
            w.write(OHLCV(timestamp=_T0_UTC + week * _WEEK, open=c, high=c, low=c,
                          close=c, volume=1.0))
    SymInfo(
        prefix="EXCH", description="Crypto", ticker="CRYPTO",
        currency="USD", period="1W", type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))
    return str(path)


def __test_helper_write_peer_daily(tmp_dir):
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / "PEER.ohlcv"
    with OHLCVWriter(path, "1D") as w:
        for week, weekday in __test_helper_chart_days():
            c = 100.0 + week * 5 + weekday
            w.write(OHLCV(timestamp=__test_helper_bar_ms(week, weekday), open=c,
                          high=c, low=c, close=c, volume=1.0))
    si = __test_helper_equity_syminfo("1D")
    si.ticker = "PEER"
    si.save_toml(path.with_suffix(".toml"))
    return str(path)


def __test_helper_chart_days():
    """``(week, weekday)`` pairs the equity chart has a bar for."""
    out = []
    for week in range(_N_WEEKS):
        for weekday in range(5):
            if week == _HOLIDAY_WEEK and weekday == 0:
                continue  # holiday Monday
            out.append((week, weekday))
    return out


def __test_helper_bar_ms(week, weekday):
    return _T0_EQ + (week * 7 + weekday) * _DAY


def __test_helper_run_with_timeout(fn, seconds=120):
    """Run ``fn`` on a daemon thread; a deadlock fails the test instead of hanging."""
    import threading
    box = {}

    def target():
        try:
            box['result'] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            box['error'] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise AssertionError(f"deadlock: the run did not finish within {seconds}s")
    if 'error' in box:
        raise box['error']
    return box.get('result')


def __test_cross_calendar_weekly_peer_never_leaks_and_never_deadlocks__(runner, log):
    """A differently-scheduled weekly peer is paired on closes only, with a holiday lag.

    The consumer keeps the equity calendar, the producer a 24/7 one, so the
    calendar extension is off: the crypto week closing Monday 00:00 UTC first
    shows up on the Monday equity bar (closing 16:00 New York). When that Monday
    is a market holiday the value arrives on Tuesday — one bar late, never a bar
    early. The expectation below is derived from the closes alone, so any
    pairing that reached past the consumer bar's close would fail it.
    """
    import sys
    import tempfile
    from pathlib import Path
    from pynecore.types.na import NA
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(Path(__file__).stem, None)

    si = __test_helper_equity_syminfo("1D")

    def scenario():
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            crypto = __test_helper_write_crypto_weekly(tmp)
            peer = __test_helper_write_peer_daily(tmp)
            bars = [OHLCV(timestamp=__test_helper_bar_ms(week, weekday), open=1.0,
                          high=1.0, low=1.0, close=1.0, volume=1.0)
                    for week, weekday in __test_helper_chart_days()]
            r = runner(bars, syminfo_override=dict(
                prefix="NYSE", ticker="DXC", type="stock",
                timezone="America/New_York", period="1D",
                opening_hours=si.opening_hours, session_starts=si.session_starts,
                session_ends=si.session_ends),
                security_data={"EXCH:CRYPTO:W": crypto, "NYSE:PEER:D": peer})
            return {c.timestamp: (pv.get("w"), pv.get("dep")) for c, pv in r.run_iter()}

    rows = __test_helper_run_with_timeout(scenario)

    # The crypto week ``k`` closes at ``_T0_UTC + (k + 1) * _WEEK``; the equity
    # bar of (week, weekday) closes at 16:00 New York, i.e. 21:00 UTC in winter.
    def equity_close_ms(week, weekday):
        return __test_helper_bar_ms(week, weekday) + 6 * 3_600_000 + 30 * 60_000

    def expected_week(week, weekday):
        deadline = equity_close_ms(week, weekday)
        best = None
        for k in range(_N_WEEKS):
            if _T0_UTC + (k + 1) * _WEEK <= deadline:
                best = k
        return best

    checked = 0
    for week, weekday in __test_helper_chart_days():
        value = rows[__test_helper_bar_ms(week, weekday)][0]
        k = expected_week(week, weekday)
        if k is None:
            assert isinstance(value, NA) or value != value, \
                f"week {week} day {weekday}: got {value}, expected na"
            continue
        assert value == __test_helper_crypto_weekly_close(k), \
            (f"week {week} day {weekday}: w={value} != crypto week {k} "
             f"({__test_helper_crypto_weekly_close(k)})")
        checked += 1

    assert checked > 15, f"only {checked} bars compared"

    # The holiday Monday is gone from the chart, so week (_HOLIDAY_WEEK - 1)'s
    # crypto bar — which closed at that Monday's midnight — arrives on Tuesday.
    tuesday = rows[__test_helper_bar_ms(_HOLIDAY_WEEK, 1)][0]
    assert tuesday == __test_helper_crypto_weekly_close(_HOLIDAY_WEEK - 1), \
        f"Tuesday after the holiday Monday: w={tuesday}"
    friday_before = rows[__test_helper_bar_ms(_HOLIDAY_WEEK - 1, 4)][0]
    assert friday_before != __test_helper_crypto_weekly_close(_HOLIDAY_WEEK - 1), \
        "the Friday before the crypto week closed already saw it (lookahead!)"

    # The dependent context is a SEPARATE equity-daily symbol keeping the chart's
    # schedule, so its child-side pairing has to land on the same weekly bar the
    # chart itself picked.
    for ts, (direct, dependent) in rows.items():
        if direct is None or isinstance(direct, NA):
            continue
        assert dependent == direct + 0.5, \
            f"ts={ts}: dependent={dependent} != direct+0.5"

    log.info("cross-calendar weekly peer: %d bars paired on closes only", checked)
