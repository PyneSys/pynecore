"""
@pyne
"""
from pynecore.lib import last_bar_time, plot, request, script


@script.indicator(title="Security Last Bar Time", shorttitle="slbt")
def main():
    htf_lbt = request.security("EXCH:HTFSYM", "15", last_bar_time)
    plot(htf_lbt, "htf_lbt")


_TS0 = 1735689600  # 2025-01-01T00:00:00 UTC, aligned to the 300s and 900s grids
_CHART_STEP = 300
_HTF_STEP = 900


def _write_htf(tmp_dir, timestamps):
    """Write a 15-minute security feed plus its ``.toml`` sidecar; return the path."""
    from datetime import time
    from pynecore.core.ohlcv_file import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / "EXCH_HTFSYM_15.ohlcv"
    with OHLCVWriter(path) as w:
        for ts in timestamps:
            w.write(OHLCV(timestamp=ts, open=1.0, high=2.0, low=0.5, close=1.5, volume=1.0))

    SymInfo(
        prefix="EXCH", description="HTF Test Symbol", ticker="HTFSYM",
        currency="USD", period="15", type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base", taker_fee=0.1, maker_fee=0.1,
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))

    return str(path)


def _chart_bars(n):
    from pynecore.types.ohlcv import OHLCV
    return [
        OHLCV(timestamp=_TS0 + i * _CHART_STEP,
              open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0)
        for i in range(n)
    ]


def __test_security_last_bar_time_anchored__(runner, log):
    """The security child anchors ``last_bar_time`` to its own file's final bar
    (Pine fixes it on historical bars), so the expression value is that constant
    on every chart bar — not the child's per-bar current time."""
    import tempfile
    from pathlib import Path
    from pynecore.types.na import NA

    with tempfile.TemporaryDirectory() as td:
        htf_path = _write_htf(Path(td), [_TS0, _TS0 + _HTF_STEP])
        file_final_ms = (_TS0 + _HTF_STEP) * 1000

        rows = []
        r = runner(_chart_bars(8), security_data={"EXCH:HTFSYM:15": htf_path})
        for _candle, pv in r.run_iter():
            rows.append(pv["htf_lbt"])

    values = [v for v in rows if not isinstance(v, NA)]
    assert values, "security never returned a value"
    assert all(int(v) == file_final_ms for v in values), \
        f"expected constant {file_final_ms}, got {sorted(set(values))}"
    log.info("security child last_bar_time anchored to its file end")
