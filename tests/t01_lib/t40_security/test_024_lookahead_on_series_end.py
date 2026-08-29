"""
@pyne
"""
from pynecore.lib import (barmerge, last_bar_index, last_bar_time, plot, request,
                          script, syminfo)


@script.indicator(title="Lookahead On Series End", shorttitle="loSE")
def main():
    htf_lbt = request.security(syminfo.tickerid, "D", last_bar_time,
                               lookahead=barmerge.lookahead_on)
    htf_lbi = request.security(syminfo.tickerid, "D", last_bar_index,
                               lookahead=barmerge.lookahead_on)
    plot(htf_lbt, "htf_lbt")
    plot(htf_lbi, "htf_lbi")


# Every timestamp here is Unix MILLISECONDS.
_T0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, aligned to the day grid
_HOUR = 3_600_000
_N_DAYS = 6


def _bars():
    from pynecore.types.ohlcv import OHLCV
    return [
        OHLCV(timestamp=_T0 + (day * 24 + hour) * _HOUR,
              open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0)
        for day in range(_N_DAYS) for hour in range(24)
    ]


def _write_feed(tmp_dir):
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession

    path = tmp_dir / "FEED.ohlcv"
    with OHLCVWriter(path, "60") as w:
        for bar in _bars():
            w.write(bar)
    SymInfo(
        prefix="EXCH", description="Lookahead series end", ticker="loSE",
        currency="USD", period="60", type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))
    return str(path)


def __test_developing_transport_keeps_the_series_end__(runner, log):
    """The ``lookahead_on`` developing transport must not walk the series end back.

    That transport steps the child one period at a time through the MIDDLE of a
    historical run, so the period it replays is not the series' last bar — the
    child file's final bar is. Anchoring ``last_bar_time`` to the period being
    replayed makes it recede on every chart bar, and with it
    ``chart.right_visible_bar_time``: a ``if time == chart.right_visible_bar_time``
    block then fires on EVERY bar instead of once. MEASURED on the wild
    ``Support Resistance Classification (VR) [LuxAlgo]``, which indexed an empty
    array on the child's first chart bar and killed the subprocess.
    """
    import sys
    import tempfile
    from pathlib import Path
    from pynecore.types.na import NA

    sys.modules.pop(Path(__file__).stem, None)

    last_day_ms = _T0 + (_N_DAYS - 1) * 24 * _HOUR
    with tempfile.TemporaryDirectory() as td:
        feed = _write_feed(Path(td))
        times, indices = [], []
        r = runner(_bars(), syminfo_override={"period": "60"},
                   security_data={"D": feed})
        for _candle, pv in r.run_iter():
            times.append(pv.get("htf_lbt"))
            indices.append(pv.get("htf_lbi"))

    seen_t = {int(v) for v in times if v is not None and not isinstance(v, NA)}
    seen_i = {int(v) for v in indices if v is not None and not isinstance(v, NA)}
    assert seen_t, "security never returned a value"
    assert seen_t == {last_day_ms}, \
        f"expected constant {last_day_ms}, got {sorted(seen_t)}"
    assert seen_i == {_N_DAYS - 1}, \
        f"expected constant {_N_DAYS - 1}, got {sorted(seen_i)}"
    log.info("lookahead_on child keeps last_bar_time/last_bar_index at the file end")
