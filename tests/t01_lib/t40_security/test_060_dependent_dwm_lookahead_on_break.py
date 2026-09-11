"""
@pyne
"""
from pynecore.lib import barmerge, close, plot, request, script, syminfo
from pynecore.types import Series


@script.indicator(title="DWM Lookahead Break", shorttitle="DLB")
def main():
    # A weekly ``lookahead_on`` producer and a monthly consumer of it, on a
    # daily equity chart. Every chart bar closes at 16:00 — inside the
    # scheduled break — so a D/W/M peer on the chart's own calendar is asked
    # through to the next session open. The weekly context answers with a
    # DEVELOPING round on almost every bar, and that round has to publish as
    # far as the consumer asks or the chain never resolves.
    w: Series[float] = request.security(syminfo.tickerid, "W", close,
                                        lookahead=barmerge.lookahead_on)
    m: Series[float] = request.security(syminfo.tickerid, "M", w,
                                        lookahead=barmerge.lookahead_on)
    plot(w, "w")
    plot(m, "m")


# Every timestamp here is Unix MILLISECONDS. 2025-01-06 is a Monday.
_T0 = 1_736_173_800_000  # 2025-01-06T14:30:00Z == 09:30 New York (winter, UTC-5)
_DAY = 86_400_000
_N_DAYS = 30  # six Mon-Fri weeks: 2025-01-06 .. 2025-02-14


def __test_helper_weekday_days():
    """Calendar-day offsets from ``_T0`` that are Mon-Fri."""
    out = []
    for day in range(_N_DAYS + 10):
        if day % 7 < 5:
            out.append(day)
        if len(out) == _N_DAYS:
            break
    return out


def __test_helper_equity_syminfo(ticker, period):
    from datetime import time
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    return SymInfo(
        prefix="NYSE", description="Equity", ticker=ticker,
        currency="USD", period=period, type="stock",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=1,
        timezone="America/New_York", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(9, 30), end=time(16, 0))
                       for i in range(5)],
        session_starts=[SymInfoSession(day=i, time=time(9, 30)) for i in range(5)],
        session_ends=[SymInfoSession(day=i, time=time(16, 0)) for i in range(5)],
    )


def __test_helper_weekly_close(week):
    return 7000.0 + week


def __test_helper_write_weekly(tmp_dir):
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / "DLBW.ohlcv"
    with OHLCVWriter(path, "1W") as w:
        for week in range(_N_DAYS // 5):
            c = __test_helper_weekly_close(week)
            w.write(OHLCV(timestamp=_T0 + week * 7 * _DAY, open=c, high=c, low=c,
                          close=c, volume=1.0))
    __test_helper_equity_syminfo("DLB", "1W").save_toml(path.with_suffix(".toml"))
    return str(path)


def __test_helper_write_monthly(tmp_dir):
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / "DLBM.ohlcv"
    with OHLCVWriter(path, "1M") as w:
        # January opens on the 2nd (four calendar days before ``_T0``),
        # February on Monday the 3rd (offset 28).
        for offset, c in ((-4, 500.0), (28, 600.0)):
            w.write(OHLCV(timestamp=_T0 + offset * _DAY, open=c, high=c, low=c,
                          close=c, volume=1.0))
    __test_helper_equity_syminfo("DLB", "1M").save_toml(path.with_suffix(".toml"))
    return str(path)


def __test_helper_chart_bars():
    from pynecore.types.ohlcv import OHLCV
    out = []
    for i, day in enumerate(__test_helper_weekday_days()):
        c = 100.0 + i
        out.append(OHLCV(timestamp=_T0 + day * _DAY, open=c, high=c, low=c,
                         close=c, volume=1.0))
    return out


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


def __test_developing_dwm_round_publishes_through_the_session_break__(runner, log):
    """A developing D/W/M round must reach the as-of its consumers derive for it.

    The chart bar of a daily equity chart closes at 16:00, in the scheduled
    break, so both the chart (``chart_asof``) and a peer child (``_peer_asof``)
    extend their as-of for a D/W/M context on that same calendar to the NEXT
    session open. A developing round that published only up to its tick instant
    (16:00) would leave the monthly consumer waiting on 09:30 of the following
    day — an instant no round of the weekly context ever reaches, since the next
    chart bar brings a new tick. The chart is parked on that consumer, so the
    run freezes rather than producing a wrong value.
    """
    import sys
    import tempfile
    from pathlib import Path

    from pynecore.types.na import NA

    sys.modules.pop(Path(__file__).stem, None)

    si = __test_helper_equity_syminfo("DLB", "1D")

    def scenario():
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            weekly = __test_helper_write_weekly(tmp)
            monthly = __test_helper_write_monthly(tmp)
            r = runner(__test_helper_chart_bars(), syminfo_override=dict(
                prefix="NYSE", ticker="DLB", type="stock",
                timezone="America/New_York", period="1D",
                opening_hours=si.opening_hours, session_starts=si.session_starts,
                session_ends=si.session_ends),
                security_data={"W": weekly, "M": monthly})
            return [(c.timestamp, pv.get("w"), pv.get("m")) for c, pv in r.run_iter()]

    rows = __test_helper_run_with_timeout(scenario)

    assert len(rows) == _N_DAYS, \
        f"bar count {len(rows)} != {_N_DAYS} — the run stalled on a dependent D/W/M read"

    # The chain really resolved: the monthly consumer carries weekly values, not
    # a run that merely survived because nothing was ever paired.
    carried = 0
    for _ts, _w, m in rows:
        if isinstance(m, NA) or m != m:
            continue
        carried += 1
    assert carried > 0, "the monthly consumer never received a value from the weekly peer"
    log.info("the dependent weekly->monthly chain resolved on %d of %d bars",
             carried, len(rows))
