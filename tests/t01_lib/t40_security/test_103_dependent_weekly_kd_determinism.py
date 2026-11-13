"""
@pyne
"""
from pynecore.lib import close, plot, request, script, syminfo, ta


@script.indicator("Weekly K/D", shorttitle="WKD")
def main():
    # ``k`` inside ``d``'s weekly context is the same week's ``k``: ``d`` is the
    # 3-week mean of weekly ``k``. Both contexts stand at the top level.
    k = request.security(syminfo.tickerid, "W", ta.sma(close, 3))
    d = request.security(syminfo.tickerid, "W", ta.sma(k, 3))
    plot(k, "k")
    plot(d, "d")


# The read context lives in a helper: its round starts when the chart calls it.
_READ_IN_FUNCTION = '''"""
@pyne
"""
from pynecore.lib import close, plot, request, script, syminfo, ta


def htf(src):
    return request.security(syminfo.tickerid, "W", ta.sma(src, 3))


@script.indicator("Weekly helper", shorttitle="WH")
def main():
    k = htf(close)
    d = request.security(syminfo.tickerid, "W", ta.sma(k, 3))
    plot(k, "k")
    plot(d, "d")
'''

# The reading context lives in a helper and reads its parameter.
_READER_IN_FUNCTION = '''"""
@pyne
"""
from pynecore.lib import close, plot, request, script, syminfo, ta


def htf(src):
    return request.security(syminfo.tickerid, "W", ta.sma(src, 3))


@script.indicator("Weekly reader", shorttitle="WR")
def main():
    k = request.security(syminfo.tickerid, "W", ta.sma(close, 3))
    d = htf(k)
    plot(k, "k")
    plot(d, "d")
'''

# Daily 24/7 bars from a Monday, ``close = day + 1``: week ``w`` covers days
# ``7w .. 7w+6`` and closes at ``7w + 7``. Every timestamp here is Unix MILLISECONDS.
_T0 = 1_736_121_600_000  # 2025-01-06T00:00:00 UTC, a Monday
_DAY = 86_400_000
_N_DAYS = 300
_N_RUNS = 8


def _bars():
    from pynecore.types.ohlcv import OHLCV
    return [OHLCV(timestamp=_T0 + i * _DAY, open=float(i + 1), high=float(i + 1),
                  low=float(i + 1), close=float(i + 1), volume=1.0)
            for i in range(_N_DAYS)]


def _write_feed(tmp_dir):
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession

    path = tmp_dir / "FEED.ohlcv"
    with OHLCVWriter(path, "1D") as w:
        for bar in _bars():
            w.write(bar)
    SymInfo(
        prefix="EXCH", description="Weekly K/D", ticker="WKD",
        currency="USD", period="1D", type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))
    return path


def _rows(run_iter):
    """``(k, d)`` per chart day of one run, ``na`` as None so that runs compare."""
    from pynecore.types.na import isna_num

    rows = {}
    for candle, pv in run_iter:
        k, d = pv.get("k"), pv.get("d")
        rows[(candle.timestamp - _T0) // _DAY] = (None if isna_num(k) else k,
                                                  None if isna_num(d) else d)
    assert len(rows) == _N_DAYS
    return rows


def _check(rows, shape):
    """Every ``k`` is the 3-week mean of weekly closes and every ``d`` the 3-week
    mean of ``k``, each landing on the chart bar that closes the week."""
    for day in range(_N_DAYS):
        k, d = rows[day]
        weeks = (day + 1) // 7  # weeks closed by the end of this day
        w = weeks - 1  # the latest closed week
        # k_w = mean(7w+7, 7w, 7w-7) = 7w from the third closed week on
        if weeks < 3:
            assert k is None, f"{shape} day {day}: k={k} should be na"
        else:
            assert k == 7 * w, f"{shape} day {day}: k={k} != {7 * w}"
        # d_w = mean(k_w, k_w-1, k_w-2) = 7w - 7 from the fifth closed week on
        if weeks < 5:
            assert d is None, f"{shape} day {day}: d={d} should be na"
        else:
            assert d == 7 * w - 7, f"{shape} day {day}: d={d} != {7 * w - 7} (3-week mean of k)"


def __test_weekly_kd_is_the_same_on_every_run_in_all_three_shapes__(runner, log):
    """A context reading another context's result is deterministic and exact.

    Each context runs in its own process and the chart used to start every
    context's round of a bar at once; a child read a peer's latest written value
    without waiting, so ``d`` read whatever week ``k``'s process had reached and
    runs of one script on the same bars differed (about half of them wrong). The
    same race hit a read context living in a helper, whose round only starts
    when the chart calls the helper, mid-bar.

    Three shapes, each run eight times on the same 300 daily bars with weekly
    data: both contexts at the top level, the read context in a helper
    (``k = htf(close)``), the reader in a helper (``d = htf(k)``). Every run must
    be identical, and every ``d`` the 3-week mean of weekly ``k``.
    """
    import sys
    import tempfile
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner
    from pynecore.core.syminfo import SymInfo

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        feed = _write_feed(tmp_dir)
        security_data = {"W": str(feed)}

        # Shape 1: this module, both contexts at the top level
        runs = []
        for _ in range(_N_RUNS):
            sys.modules.pop(Path(__file__).stem, None)
            r = runner(_bars(), syminfo_override={"period": "1D"}, security_data=security_data)
            runs.append(_rows(r.run_iter()))
        assert all(run == runs[0] for run in runs), "top level: runs differ"
        _check(runs[0], "top level")

        # Shapes 2 and 3: the helper scripts
        syminfo = SymInfo.load_toml(feed.with_suffix(".toml"))
        sys.path.insert(0, str(tmp_dir))
        try:
            for name, source in (("read_in_function", _READ_IN_FUNCTION),
                                 ("reader_in_function", _READER_IN_FUNCTION)):
                path = tmp_dir / f"weekly_{name}.py"
                path.write_text(source)
                runs = []
                for _ in range(_N_RUNS):
                    sys.modules.pop(path.stem, None)
                    sr = ScriptRunner(path, iter(_bars()), syminfo, security_data=security_data)
                    runs.append(_rows(sr.run_iter()))
                sys.modules.pop(path.stem, None)
                assert all(run == runs[0] for run in runs), f"{name}: runs differ"
                _check(runs[0], name)
        finally:
            sys.path.remove(str(tmp_dir))

    log.info("weekly K/D: %d identical runs in each of three shapes, d = 3-week mean of k",
             _N_RUNS)
