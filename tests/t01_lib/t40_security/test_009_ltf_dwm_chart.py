"""
@pyne
"""
from pynecore.lib import array, close, plot, request, script


@script.indicator(title="LTF D/W/M Chart Test", shorttitle="LDC")
def main():
    # ``n`` is the per-chart-bar intrabar count, ``sum`` the sum of the intrabar
    # closes — together they fully characterise the LTF array for that bar.
    vals = request.security_lower_tf("EXCH:LTFSYM", "60", close)
    plot(array.size(vals), "n")
    plot(array.sum(vals), "sum")


# --- Geometry -----------------------------------------------------------------
# Daily CHART (single-period civil ``1D``) with an hourly (``60``) LTF feed. On a
# daily chart ``chart_off`` is 0, so Phase 1's ``chart_time + chart_off`` target
# would degrade to the bar open and re-collect the previous day. Phase 2 targets
# the chart bar's civil period end (next civil open - 1), windowing each chart
# bar to its OWN calendar day ``[T, next_day_open)``. The 24/7 UTC case is the
# primary proof; the America/New_York case proves the civil next-open is built in
# the exchange timezone (correct across the spring-forward DST transition, where
# the local day is only 23 hours).
_HOUR = 3600
_LTF_TF = "60"


def _write_ltf(tmp_dir, bars):
    """Write a 1-hour LTF feed from ``(timestamp_s, close)`` pairs (+ a 24/7 UTC
    ``.toml`` sidecar the subprocess loads on startup); return the path. The LTF
    feed timezone is irrelevant to the window (the parent drives ``target_time``);
    only the bar opens matter.
    """
    from datetime import time
    from pynecore.core.ohlcv_file import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / "EXCH_LTFSYM_60.ohlcv"
    with OHLCVWriter(path) as w:
        for ts, c in bars:
            c = float(c)
            w.write(OHLCV(timestamp=ts, open=c, high=c, low=c, close=c, volume=1.0))

    SymInfo(
        prefix="EXCH", description="LTF Test Symbol", ticker="LTFSYM",
        currency="USD", period="60", type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base", taker_fee=0.1, maker_fee=0.1,
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))

    return str(path)


def _chart_bars(timestamps):
    """Daily chart bars at the given epoch-second opens; OHLCV body is irrelevant
    to the LTF result.
    """
    from pynecore.types.ohlcv import OHLCV
    return [
        OHLCV(timestamp=ts, open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0)
        for ts in timestamps
    ]


def _run(runner, timezone, chart_opens, ltf_bars):
    """One end-to-end run on a ``1D`` chart in ``timezone``; returns the per-bar
    ``(n, sum)`` plot rows.

    Drops the script's stem-key import cache first so each geometry re-imports a
    clean module (the ``runner`` fixture only deletes pytest's dotted key once,
    and a single module may host only one runner-using test).
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
            syminfo_override={"period": "1D", "timezone": timezone},
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


def __test_ltf_dwm_chart_window__(runner, log):
    """The LTF window on a single-period civil daily chart, UTC and across a DST
    transition. A single test hosts both geometries because the ``runner``
    fixture may be used by only one test per module; ``_run`` re-imports a clean
    script between geometries.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    # --- 24/7 UTC daily chart: each daily bar collects its own calendar day's
    # hourly intrabars. Feed is 53 contiguous hourly bars (closes 1..53): day 0
    # gets hours 0..23 (24 bars, sum 300), day 1 hours 24..47 (24 bars, sum 876),
    # day 2 the partial tail hours 48..52 (5 bars, sum 255).
    utc = ZoneInfo("UTC")
    ts0 = int(datetime(2025, 1, 1, tzinfo=utc).timestamp())
    chart_opens = [ts0 + d * 24 * _HOUR for d in range(3)]
    ltf_bars = [(ts0 + h * _HOUR, h + 1) for h in range(53)]
    rows = _run(runner, "UTC", chart_opens, ltf_bars)
    _assert(rows, [24, 24, 5], [300, 876, 255])
    log.info("LTF daily-chart UTC window correct")

    # --- America/New_York daily chart across the 2025-03-09 spring-forward. The
    # civil next-open is built in NY local time, so the boundary lands on each
    # NY-local midnight (05:00 UTC in EST, 04:00 UTC in EDT). Mar 9's local day is
    # only 23 hours, so its bar holds 23 hourly intrabars, not 24 — a fixed-delta
    # (T + 86400) target would mis-window here. Feed is 51 contiguous UTC-hourly
    # bars from Mar 8 00:00 NY (closes 1..51): Mar 8 -> 24 (sum 300), Mar 9 -> 23
    # (sum 828), Mar 10 -> the 4-bar tail (sum 198).
    ny = ZoneInfo("America/New_York")
    chart_opens = [int(datetime(2025, 3, d, tzinfo=ny).timestamp()) for d in (8, 9, 10)]
    ltf_start = chart_opens[0]
    ltf_bars = [(ltf_start + h * _HOUR, h + 1) for h in range(51)]
    rows = _run(runner, "America/New_York", chart_opens, ltf_bars)
    _assert(rows, [24, 23, 4], [300, 828, 198])
    log.info("LTF daily-chart DST (spring-forward) window correct")
