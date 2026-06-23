"""
@pyne
"""
from pynecore.lib import array, close, plot, request, script


@script.indicator(title="LTF Weekly/Monthly Chart Test", shorttitle="LWM")
def main():
    # ``n`` is the per-chart-bar intrabar count, ``sum`` the sum of the intrabar
    # closes — together they fully characterise the LTF array for that bar.
    vals = request.security_lower_tf("EXCH:LTFSYM", "1D", close)
    plot(array.size(vals), "n")
    plot(array.sum(vals), "sum")


# --- Geometry -----------------------------------------------------------------
# Weekly and monthly CHARTS (single-period civil ``1W`` / ``1M``) with a daily
# (``1D``) LTF feed, 24/7 UTC. Both proves the civil period end is calendar-built
# (not a fixed/nominal span): a week is exactly 7 days, and January is 31 days —
# ``in_seconds("1M")`` is only a nominal ~30.4-day month, so an arithmetic target
# would leak Jan 31 into February.
_DAY = 86400
_LTF_TF = "1D"


def _write_ltf(tmp_dir, bars):
    """Write a 1-day LTF feed from ``(timestamp_s, close)`` pairs (+ a 24/7 UTC
    ``.toml`` sidecar the subprocess loads on startup); return the path.
    """
    from datetime import time
    from pynecore.core.ohlcv_file import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / "EXCH_LTFSYM_1D.ohlcv"
    with OHLCVWriter(path) as w:
        for ts, c in bars:
            c = float(c)
            w.write(OHLCV(timestamp=ts, open=c, high=c, low=c, close=c, volume=1.0))

    SymInfo(
        prefix="EXCH", description="LTF Test Symbol", ticker="LTFSYM",
        currency="USD", period="1D", type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base", taker_fee=0.1, maker_fee=0.1,
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))

    return str(path)


def _chart_bars(timestamps):
    """Chart bars at the given epoch-second opens; OHLCV body is irrelevant to
    the LTF result.
    """
    from pynecore.types.ohlcv import OHLCV
    return [
        OHLCV(timestamp=ts, open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0)
        for ts in timestamps
    ]


def _run(runner, period, chart_opens, ltf_bars):
    """One end-to-end run on a ``1W``/``1M`` UTC chart; returns the per-bar
    ``(n, sum)`` plot rows. Drops the stem-key import cache first so each geometry
    re-imports a clean module (one runner-using test per module).
    """
    import sys
    import tempfile
    from pathlib import Path

    sys.modules.pop(Path(__file__).stem, None)
    rows = []
    with tempfile.TemporaryDirectory() as td:
        ltf_path = _write_ltf(Path(td), ltf_bars)
        r = runner(
            _chart_bars(chart_opens),
            syminfo_override={"period": period, "timezone": "UTC"},
            security_data={f"EXCH:LTFSYM:{_LTF_TF}": ltf_path},
        )
        for _candle, pv in r.run_iter():
            rows.append((pv["n"], pv["sum"]))
    return rows


def _assert(rows, expected_size, expected_sum):
    import math
    from pynecore.types.na import NA

    assert len(rows) == len(expected_size), f"bar count {len(rows)} != {len(expected_size)}"
    for i, (n, s) in enumerate(rows):
        assert int(n) == expected_size[i], f"bar {i}: size {n} != {expected_size[i]}"
        exp = expected_sum[i]
        if exp is None:
            assert isinstance(s, NA), f"bar {i}: expected na, got {s!r}"
        else:
            assert not isinstance(s, NA), f"bar {i}: expected {exp}, got na"
            assert math.isclose(float(s), exp, abs_tol=1e-9), f"bar {i}: sum {s} != {exp}"


def __test_ltf_weekly_monthly_chart_window__(runner, log):
    """The LTF window on single-period civil weekly and monthly charts (UTC). A
    single test hosts both geometries (one runner-using test per module); ``_run``
    re-imports a clean script between them.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    utc = ZoneInfo("UTC")

    # --- Weekly chart: each weekly bar (Monday-anchored) collects its own 7-day
    # window. 2025-01-06 is a Monday. Feed is 10 contiguous daily bars (closes
    # 1..10): week 0 [Mon Jan 6, Mon Jan 13) -> days 1..7 (sum 28), week 1 the
    # 3-day tail Jan 13..15 (sum 27).
    mon = int(datetime(2025, 1, 6, tzinfo=utc).timestamp())
    chart_opens = [mon, mon + 7 * _DAY]
    ltf_bars = [(mon + d * _DAY, d + 1) for d in range(10)]
    rows = _run(runner, "1W", chart_opens, ltf_bars)
    _assert(rows, [7, 3], [28, 27])
    log.info("LTF weekly-chart window correct")

    # --- Monthly chart: each monthly bar collects its own calendar month. Feed is
    # 34 contiguous daily bars from Jan 1 (closes 1..34): January has 31 days, so
    # the Jan bar holds all 31 (sum 496) and February the 3-day tail Feb 1..3
    # (sum 99). A nominal-month (~30.4d) target would wrongly leak Jan 31.
    jan1 = int(datetime(2025, 1, 1, tzinfo=utc).timestamp())
    feb1 = int(datetime(2025, 2, 1, tzinfo=utc).timestamp())
    chart_opens = [jan1, feb1]
    ltf_bars = [(jan1 + d * _DAY, d + 1) for d in range(34)]
    rows = _run(runner, "1M", chart_opens, ltf_bars)
    _assert(rows, [31, 3], [496, 99])
    log.info("LTF monthly-chart window correct")
