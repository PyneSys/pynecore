"""
@pyne
"""
from pynecore.lib import close, high, plot, request, script, syminfo, ta
from pynecore.types import Series


@script.indicator(title="Dependent Chain And Unused", shorttitle="DCU")
def main():
    # A three-link chain in ONE hourly context plus a context nothing consumes.
    # The unused one must still produce its own value (and allocate no ring),
    # while every link of the chain sees the producer's per-bar history.
    h: Series[float] = request.security(syminfo.tickerid, "60", close)
    sma3: Series[float] = request.security(syminfo.tickerid, "60", ta.sma(h, 3))
    sma2: Series[float] = request.security(syminfo.tickerid, "60", ta.sma(sma3, 2))
    unused: Series[float] = request.security(syminfo.tickerid, "60", high)
    plot(h, "h")
    plot(sma3, "sma3")
    plot(sma2, "sma2")
    plot(unused, "unused")


# Every timestamp here is Unix MILLISECONDS.
_T0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, aligned to the 1h and 5m grids
_HOUR = 3_600_000
_CHART_STEP = 300_000  # 5 minutes
_N_HOURS = 9


def __test_helper_htf_close(hour):
    return 100.0 + hour


def __test_helper_htf_high(hour):
    return 1000.0 + hour


def __test_helper_write_feed(tmp_dir):
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / "HTF60.ohlcv"
    with OHLCVWriter(path, "60") as w:
        for hour in range(_N_HOURS):
            c = __test_helper_htf_close(hour)
            w.write(OHLCV(timestamp=_T0 + hour * _HOUR, open=c,
                          high=__test_helper_htf_high(hour), low=c, close=c, volume=1.0))
    SymInfo(
        prefix="EXCH", description="Chain HTF", ticker="DCU",
        currency="USD", period="60", type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))
    return str(path)


def __test_helper_chart_bars():
    from pynecore.types.ohlcv import OHLCV
    return [OHLCV(timestamp=_T0 + i * _CHART_STEP, open=1.0, high=1.0, low=1.0,
                  close=1.0, volume=1.0)
            for i in range(_N_HOURS * 12)]


def __test_dependent_chain_and_unused_context__(runner, log):
    """A three-link dependent chain resolves per producer bar; an unused context still runs.

    ``sma3`` consumes ``h`` and ``sma2`` consumes ``sma3``, all in the same
    hourly context, so each link needs the producer's value on EVERY hourly bar
    — not just the last one of the warmup round. ``unused`` has no consumer at
    all: it allocates no ring, and its value must be unaffected by that.
    """
    import sys
    import tempfile
    from pathlib import Path
    from pynecore.types.na import NA

    sys.modules.pop(Path(__file__).stem, None)

    rows = {}
    with tempfile.TemporaryDirectory() as td:
        feed = __test_helper_write_feed(Path(td))
        r = runner(__test_helper_chart_bars(), security_data={"60": feed})
        for candle, pv in r.run_iter():
            minute = (candle.timestamp - _T0) // 60_000
            rows[(minute // 60, minute % 60)] = (
                pv.get("h"), pv.get("sma3"), pv.get("sma2"), pv.get("unused"))

    for hour in range(4, _N_HOURS):
        h, sma3, sma2, unused = rows[(hour, 0)]
        # The last CLOSED hourly bar is hour-1.
        j = hour - 1
        assert h == __test_helper_htf_close(j), f"hour {hour}: h={h}"
        assert unused == __test_helper_htf_high(j), f"hour {hour}: unused={unused}"
        # sma3 over hourly closes j-2..j, whose mean is close(j) - 1.
        assert not isinstance(sma3, NA) and abs(sma3 - (__test_helper_htf_close(j) - 1.0)) < 1e-9, \
            f"hour {hour}: sma3={sma3} != {__test_helper_htf_close(j) - 1.0}"
        # sma2 over the last two sma3 values, i.e. close(j) - 1.5.
        assert not isinstance(sma2, NA) and abs(sma2 - (__test_helper_htf_close(j) - 1.5)) < 1e-9, \
            f"hour {hour}: sma2={sma2} != {__test_helper_htf_close(j) - 1.5}"

    log.info("dependent chain and unused context both correct")
