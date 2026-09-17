"""
@pyne
"""
from pynecore.lib import close, input, plot, request, script, syminfo
from pynecore.types import Series


@script.indicator(title="Conditional Expression Missing Feed", shorttitle="CEMF")
def main(htf=input.timeframe(defval="60", title="HTF")):
    # A runtime-resolved security standing in a ternary branch. Pine evaluates
    # both branches of a ternary on every bar, so the context signals every bar
    # even while the branch is not taken — but it is only READ on the bars where
    # the condition holds, and only there does it need a feed.
    x: Series[float] = request.security(syminfo.tickerid, htf, close) if close > 50.0 else close
    plot(x, "x")


# Every timestamp here is Unix MILLISECONDS.
_T0 = 1_735_689_600_000  # 2025-01-01T00:00:00 UTC, aligned to the 1h and 5m grids
_HOUR = 3_600_000
_CHART_STEP = 300_000  # 5 minutes
_N_HOURS = 12
_SWITCH_BAR = 30  # chart bar from which the condition holds

_STATEMENT_SCRIPT = '''"""
@pyne
"""
from pynecore.lib import close, input, plot, request, script, syminfo
from pynecore.types import Series


@script.indicator(title="Statement Form", shorttitle="SF")
def main(htf=input.timeframe(defval="60", title="HTF")):
    x: Series[float] = close
    if close > 50.0:
        x = request.security(syminfo.tickerid, htf, close)
    plot(x, "x")
'''


def __test_helper_write_feed(tmp_dir):
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / "HTF60.ohlcv"
    with OHLCVWriter(path, "60") as w:
        for hour in range(_N_HOURS):
            c = 100.0 + hour
            w.write(OHLCV(timestamp=_T0 + hour * _HOUR, open=c, high=c, low=c,
                          close=c, volume=1.0))
    SymInfo(
        prefix="PYTEST", description="Conditional HTF", ticker="TEST",
        currency="USD", period="60", type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))
    return str(path)


def __test_helper_chart_bars(closes):
    from pynecore.types.ohlcv import OHLCV
    return [OHLCV(timestamp=_T0 + i * _CHART_STEP, open=c, high=c, low=c,
                  close=c, volume=1.0)
            for i, c in enumerate(closes)]


def __test_helper_closes(value):
    return [value] * (_N_HOURS * 12)


def __test_helper_switching_closes():
    bars = _N_HOURS * 12
    return [1.0 if i < _SWITCH_BAR else 100.0 for i in range(bars)]


def __test_helper_run_with_timeout(fn, seconds):
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


def __test_untaken_branch_tolerates_the_missing_feed__(runner, caplog, log):
    """An unread runtime-resolved context with no feed does not fail the run

    Its ``__sec_signal__`` fires on every bar because a ternary evaluates both
    branches, so resolution happens whether or not the branch is taken. With no
    data behind the resolved symbol the context simply stays unprovisioned —
    named once in a warning — and the else branch answers every bar.
    """
    import logging
    import sys
    from pathlib import Path

    sys.modules.pop(Path(__file__).stem, None)

    def scenario():
        rows = {}
        r = runner(__test_helper_chart_bars(__test_helper_closes(1.0)))
        for i, (_candle, pv) in enumerate(r.run_iter()):
            rows[i] = dict(pv)
        return rows

    with caplog.at_level(logging.WARNING, logger="pyne_core_logger"):
        rows = __test_helper_run_with_timeout(scenario, seconds=60)

    assert len(rows) == _N_HOURS * 12
    assert all(v["x"] == 1.0 for v in rows.values()), \
        "the untaken branch did not answer with the chart close"

    warnings = [rec.getMessage() for rec in caplog.records
                if rec.levelno == logging.WARNING
                and "Unprovisioned security context" in rec.getMessage()]
    assert len(warnings) == 1, f"expected exactly one warning, got: {warnings}"
    assert "PYTEST:TEST" in warnings[0] and "'60'" in warnings[0], \
        f"the warning does not name the resolved context: {warnings[0]}"

    log.info("an untaken conditional branch tolerates the missing feed")


def __test_taken_branch_raises_at_the_reading_bar__(runner, log):
    """The missing feed is reported at the first bar that actually reads it

    Tolerating the missing data at resolve time must not swallow it: the bar
    whose condition holds reads the context, and that read has no value to
    return, so it raises there — not at startup.
    """
    import sys
    from pathlib import Path

    sys.modules.pop(Path(__file__).stem, None)

    def scenario():
        rows = {}
        error = None
        r = runner(__test_helper_chart_bars(__test_helper_switching_closes()))
        try:
            for i, (_candle, pv) in enumerate(r.run_iter()):
                rows[i] = dict(pv)
        except ValueError as exc:
            error = exc
        return rows, error

    rows, error = __test_helper_run_with_timeout(scenario, seconds=60)

    assert error is not None, "the read of an unprovisioned context did not raise"
    assert "No OHLCV data found for security context" in str(error), \
        f"unexpected error message: {error}"
    assert "PYTEST:TEST" in str(error), f"the error does not name the symbol: {error}"
    assert len(rows) == _SWITCH_BAR, \
        f"raised at bar {len(rows)} instead of the first reading bar {_SWITCH_BAR}"
    assert all(rows[i]["x"] == 1.0 for i in range(_SWITCH_BAR)), \
        "the bars before the reading one did not answer with the chart close"

    log.info("the missing feed is reported at the first bar that reads the context")


def __test_taken_branch_matches_the_statement_form__(runner, log):
    """With the feed present the ternary matches the equivalent if-statement

    The condition holds on every bar here, so both shapes read the context on
    every bar and their values have to agree exactly — the tolerant resolution
    changes nothing on the provisioned path.
    """
    import os
    import sys
    import tempfile
    from pathlib import Path
    from pynecore.core.script_runner import ScriptRunner

    sys.modules.pop(Path(__file__).stem, None)

    def scenario():
        os.environ['PYNE_SAVE_SCRIPT_TOML'] = '0'
        with tempfile.TemporaryDirectory() as td:
            tmp_dir = Path(td)
            feed = __test_helper_write_feed(tmp_dir)
            bars = __test_helper_chart_bars(__test_helper_closes(100.0))

            ternary = {}
            r = runner(list(bars), security_data={"60": feed})
            for i, (_candle, pv) in enumerate(r.run_iter()):
                ternary[i] = pv["x"]

            statement_path = tmp_dir / "statement_form_reference.py"
            statement_path.write_text(_STATEMENT_SCRIPT)
            sys.modules.pop(statement_path.stem, None)
            sys.path.insert(0, str(tmp_dir))
            try:
                sr = ScriptRunner(statement_path, iter(list(bars)), r.syminfo,
                                  security_data={"60": feed})
                statement = {}
                for i, (_candle, pv) in enumerate(sr.run_iter()):
                    statement[i] = pv["x"]
            finally:
                sys.path.remove(str(tmp_dir))
                sys.modules.pop(statement_path.stem, None)
        return ternary, statement

    ternary, statement = __test_helper_run_with_timeout(scenario, seconds=120)

    assert len(ternary) == _N_HOURS * 12
    assert len(statement) == len(ternary)
    for i in range(len(ternary)):
        assert ternary[i] == statement[i] or (
            __test_helper_is_na(ternary[i]) and __test_helper_is_na(statement[i])), \
            f"bar {i}: ternary {ternary[i]!r} != statement {statement[i]!r}"
    assert any(not __test_helper_is_na(v) for v in ternary.values()), \
        "the provisioned context never produced a value"

    log.info("the ternary and the statement form agree bar for bar")


def __test_helper_is_na(value):
    from pynecore.types.na import NA
    return isinstance(value, NA)
