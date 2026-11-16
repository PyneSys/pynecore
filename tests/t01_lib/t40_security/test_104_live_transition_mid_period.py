"""
@pyne
"""
from pynecore.lib import close, plot, request, script, syminfo
from pynecore.types import Series


@script.indicator(title="Live transition mid period", shorttitle="LTMP")
def main():
    # Same-symbol higher-timeframe contexts, read on every bar from the history into
    # the live phase. The live transition lands INSIDE an open period of both.
    h4: Series[float] = request.security(syminfo.tickerid, "240", close)
    d1: Series[float] = request.security(syminfo.tickerid, "D", close)
    plot(h4, "h4")
    plot(d1, "d1")


# Every timestamp here is Unix MILLISECONDS.
__test_helper_T0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, aligned to 1h/4h/1D
__test_helper_HOUR = 3_600_000
# History ends at hour 29 (05:00 on day two); the first live bar, hour 30, lies inside
# the 240-minute period [28, 32) and the daily period [24, 48).
__test_helper_N_HIST = 30
__test_helper_N_LIVE = 8
__test_helper_N_HOURS = __test_helper_N_HIST + __test_helper_N_LIVE


def __test_helper_close(hour):
    return 100.0 + hour


def __test_helper_bar(hour, is_closed=True):
    from pynecore.types.ohlcv import OHLCV
    price = __test_helper_close(hour)
    return OHLCV(timestamp=__test_helper_T0 + hour * __test_helper_HOUR, open=price,
                 high=price, low=price, close=price, volume=1.0, is_closed=is_closed)


def __test_helper_write_feed(tmp_dir):
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession

    path = tmp_dir / "FEED.ohlcv"
    with OHLCVWriter(path, "60") as w:
        for hour in range(__test_helper_N_HOURS):
            w.write(__test_helper_bar(hour))
    SymInfo(
        prefix="EXCH", description="Live transition mid period", ticker="LTMP",
        currency="USD", period="60", type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))
    return str(path)


def __test_helper_last_closed(hour, span_hours, confirmed):
    """Close of the context's last bar closed at the chart bar's as-of, or ``None``.

    A confirmed chart bar of hour ``h`` is read as of its close (``h + 1``), a
    developing one as of an instant inside it: a context bar ``k`` covers hours
    ``k * span .. (k + 1) * span - 1`` and is closed once the as-of reaches
    ``(k + 1) * span``.
    """
    asof_hour = hour + 1 if confirmed else hour
    k = asof_hour // span_hours - 1
    if k < 0:
        return None
    return __test_helper_close((k + 1) * span_hours - 1)


def __test_live_transition_mid_period_keeps_the_confirmed_value__(
        script_path, module_key, syminfo, log):
    """A lookahead_off HTF context keeps its last confirmed value across the live transition.

    The transition switches the context from the closed-only flow to the live HTF
    transport, whose one-time prefill used to re-run on top of what the closed-only
    flow had already read. Inside an open period that prefill had nothing left to
    read, and the empty round published ``na``, which every realtime tick of the
    period then read: MEASURED live on BINANCE:BTC/USDT@60, where the daily close of
    the day before turned into ``na`` from the first realtime tick on. The
    developing ticks of the first live bar and every live bar after it must read the
    same last closed bar as the history does.
    """
    import itertools
    import sys
    import tempfile
    from pathlib import Path

    from pynecore.core.script_runner import ScriptRunner, LIVE_TRANSITION

    sys.modules.pop(Path(__file__).stem, None)
    sys.modules.pop(module_key, None)

    syminfo.period = "60"
    with tempfile.TemporaryDirectory() as td:
        feed = __test_helper_write_feed(Path(td))
        historical = [__test_helper_bar(h) for h in range(__test_helper_N_HIST)]
        first_live = __test_helper_N_HIST
        live = [__test_helper_bar(first_live, is_closed=False),
                __test_helper_bar(first_live, is_closed=False)]
        live += [__test_helper_bar(h) for h in range(first_live, __test_helper_N_HOURS)]
        runner = ScriptRunner(
            script_path, itertools.chain(historical, [LIVE_TRANSITION], live),
            syminfo, security_data={"240": feed, "D": feed}, live=True)
        rows = [(c.timestamp, c.is_closed, pv.get("h4"), pv.get("d1"))
                for c, pv in runner.run_iter(yield_ticks=True)]

    checked = 0
    for ts, confirmed, h4, d1 in rows:
        hour = (ts - __test_helper_T0) // __test_helper_HOUR
        if hour < __test_helper_N_HIST - 4:
            continue
        expected_h4 = __test_helper_last_closed(hour, 4, confirmed)
        expected_d1 = __test_helper_last_closed(hour, 24, confirmed)
        assert h4 == expected_h4, \
            f"hour {hour} (confirmed={confirmed}): 240 context read {h4}, expected {expected_h4}"
        assert d1 == expected_d1, \
            f"hour {hour} (confirmed={confirmed}): D context read {d1}, expected {expected_d1}"
        checked += 1
    assert checked >= 4 + 2 + __test_helper_N_LIVE, f"only {checked} bars checked"
    log.info("HTF contexts kept their confirmed value across the live transition on %d bars",
             checked)
