"""
@pyne
"""
from pynecore.lib import close, plot, request, script, syminfo
from pynecore.types import Series


@script.indicator(title="Live forming tail", shorttitle="LFT")
def main():
    # The downloaded history of a live run ends with the bar the live feed goes on
    # forming. Here that bar is the last chart bar of a 240-minute AND of a daily
    # period, so confirming it would close both.
    h4: Series[float] = request.security(syminfo.tickerid, "240", close)
    d1: Series[float] = request.security(syminfo.tickerid, "D", close)
    plot(h4, "h4")
    plot(d1, "d1")


# Every timestamp here is Unix MILLISECONDS.
__test_helper_T0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, aligned to 1h/4h/1D
__test_helper_HOUR = 3_600_000
# The history ends with hour 47 (23:00 on day two), downloaded half way through it.
__test_helper_TAIL = 47
__test_helper_N_HOURS = __test_helper_TAIL + 6


def __test_helper_close(hour):
    return 100.0 + hour


def __test_helper_bar(hour, is_closed=True):
    from pynecore.types.ohlcv import OHLCV
    price = __test_helper_close(hour)
    return OHLCV(timestamp=__test_helper_T0 + hour * __test_helper_HOUR, open=price,
                 high=price, low=price, close=price, volume=1.0, is_closed=is_closed)


def __test_helper_write_feed(tmp_dir):
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo

    path = tmp_dir / "FEED.ohlcv"
    with OHLCVWriter(path, "60") as w:
        for hour in range(__test_helper_N_HOURS):
            w.write(__test_helper_bar(hour))
    SymInfo(
        prefix="EXCH", description="Live forming tail", ticker="LFT",
        currency="USD", period="60", type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        # No trading schedule: a session ending at 23:59:59 would turn the bar's last
        # second into a break, which a daily context already reads as the day's close.
        opening_hours=[], session_starts=[], session_ends=[],
    ).save_toml(path.with_suffix(".toml"))
    return str(path)


def __test_helper_last_closed(asof_hour, span_hours):
    """Close of the context's last bar closed by ``asof_hour``, or ``None``."""
    k = asof_hour // span_hours - 1
    if k < 0:
        return None
    return __test_helper_close((k + 1) * span_hours - 1)


def __test_forming_history_tail_confirms_no_context_bar__(
        script_path, module_key, syminfo, log):
    """The still-forming last history bar of a live run closes no ``request.security`` bar.

    Its period has not ended when the history is downloaded, so a context read on it
    sees the bars closed by then -- on the history run of that bar and on every
    realtime tick continuing it -- and the period it would close only closes with
    the bar's real close from the live feed. MEASURED live on BINANCE:BTC/USDT@60
    before this was enforced: a lookahead_off ``"240"`` context exposed the forming
    12:00-16:00 bar, with its download-time close, from 15:00 on.
    """
    import itertools
    import sys
    import tempfile
    from pathlib import Path

    from pynecore.core.script_runner import ScriptRunner, LIVE_TRANSITION

    sys.modules.pop(Path(__file__).stem, None)
    sys.modules.pop(module_key, None)

    tail = __test_helper_TAIL
    tail_ts = __test_helper_T0 + tail * __test_helper_HOUR
    syminfo.period = "60"
    with tempfile.TemporaryDirectory() as td:
        feed = __test_helper_write_feed(Path(td))
        historical = [__test_helper_bar(h) for h in range(tail + 1)]
        live = [__test_helper_bar(tail, is_closed=False),
                __test_helper_bar(tail, is_closed=False)]
        live += [__test_helper_bar(h) for h in range(tail, __test_helper_N_HOURS)]
        runner = ScriptRunner(
            script_path, itertools.chain(historical, [LIVE_TRANSITION], live),
            syminfo, security_data={"240": feed, "D": feed}, live=True,
            last_bar_index=tail, last_bar_time=tail_ts,
            history_as_of=tail_ts + __test_helper_HOUR // 2)
        rows = [(c.timestamp, c.is_closed, pv.get("h4"), pv.get("d1"))
                for c, pv in runner.run_iter(yield_ticks=True)]

    tail_indexes = [i for i, r in enumerate(rows) if r[0] == tail_ts]
    # The history run of the tail, two realtime ticks, then its close
    assert len(tail_indexes) == 4, f"tail rows: {[rows[i] for i in tail_indexes]}"
    for i, (ts, confirmed, h4, d1) in enumerate(rows):
        hour = (ts - __test_helper_T0) // __test_helper_HOUR
        if hour < tail - 4:
            continue
        # Only the live feed's own close confirms the tail, never its history run
        confirmed_close = confirmed and i != tail_indexes[0]
        asof_hour = hour + 1 if confirmed_close else hour
        expected_h4 = __test_helper_last_closed(asof_hour, 4)
        expected_d1 = __test_helper_last_closed(asof_hour, 24)
        assert h4 == expected_h4, \
            f"hour {hour} (closed={confirmed_close}): 240 read {h4}, expected {expected_h4}"
        assert d1 == expected_d1, \
            f"hour {hour} (closed={confirmed_close}): D read {d1}, expected {expected_d1}"
    log.info("the forming history tail confirmed no context bar")
