"""
@pyne
"""
from pynecore.lib import close, plot, request, script, syminfo, ta
from pynecore.types import Series


def pick_symbol(use_chart: bool) -> str:
    """A user function standing between the script and the requested symbol.

    Its result is not a "simple" chain, so the transformer cannot hoist the
    context's signal: the symbol only exists once ``main()`` reaches this call,
    and the chart resolves (and spawns) that context inline, well after the
    consumer's child was spawned.
    """
    if use_chart:
        return syminfo.tickerid
    return syminfo.ticker


@script.indicator(title="Runtime Resolved Peer", shorttitle="RRP")
def main():
    # Runtime-resolved producer: inline signal, resolved at this statement.
    sym = pick_symbol(True)
    weekly: Series[float] = request.security(sym, "W", close)
    # Consumer of it. Its own symbol IS a simple chain, so its signal is
    # hoisted to the top of main() — it is signalled, and its child starts
    # waiting on the peer, BEFORE the peer above even has a symbol.
    dep: Series[float] = request.security(syminfo.tickerid, "W", ta.sma(weekly, 3))
    # Reference: the same thing with nothing runtime-resolved in the way.
    nested: Series[float] = request.security(syminfo.tickerid, "W", ta.sma(close, 3))
    plot(weekly, "weekly")
    plot(dep, "dep")
    plot(nested, "nested")


_T0 = 1_262_304_000_000  # 2010-01-01T00:00:00 UTC, on the day grid
_DAY = 86_400_000
_N_DAYS = 900
_CHART_START = 400
# Wall-clock bound for the whole run. A consumer waiting for a record the chart
# never sends would hang forever, which is the failure this test is for.
_RUN_TIMEOUT_SECONDS = 180.0


def __test_helper_bars():
    from pynecore.types.ohlcv import OHLCV
    import math
    out = []
    for i in range(_N_DAYS):
        c = 100.0 + 20.0 * math.sin(i / 11.0) + 5.0 * math.sin(i / 2.7)
        out.append(OHLCV(timestamp=_T0 + i * _DAY, open=c, high=c + 1.0,
                         low=c - 1.0, close=c, volume=1.0))
    return out


def __test_helper_write_feed(tmp_dir):
    from datetime import time
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession

    path = tmp_dir / "FEED.ohlcv"
    with OHLCVWriter(path, "1D") as w:
        for bar in __test_helper_bars():
            w.write(bar)
    SymInfo(
        prefix="EXCH", description="Runtime resolved peer", ticker="RRP",
        currency="USD", period="1D", type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))
    return str(path)


def __test_dependent_runtime_resolved_peer__(runner, log):
    """A dependency whose symbol only exists at runtime still pairs, and never hangs.

    The consumer's signal is hoisted, so its child is already running — and
    already reading the peer — while the chart has not yet executed the peer's
    inline ``__sec_signal__``. The child cannot have the peer's record in its
    spawn snapshot; it blocks on the registry pipe until the chart resolves the
    context and pushes the record down. The run is bounded so a missing record
    fails the test instead of hanging the suite.
    """
    import sys
    import threading
    import tempfile
    from pathlib import Path
    from pynecore.types.na import NA

    sys.modules.pop(Path(__file__).stem, None)

    rows: list = []
    failure: list = []

    def _run() -> None:
        try:
            with tempfile.TemporaryDirectory() as td:
                feed = __test_helper_write_feed(Path(td))
                r = runner(__test_helper_bars()[_CHART_START:],
                           syminfo_override={"period": "1D"},
                           security_data={"W": feed})
                for candle, pv in r.run_iter():
                    rows.append((candle.timestamp, pv.get("dep"), pv.get("nested"),
                                 pv.get("weekly")))
        except BaseException as exc:  # surfaced on the main thread below
            failure.append(exc)

    worker = threading.Thread(target=_run, name="rrp-run", daemon=True)
    worker.start()
    worker.join(_RUN_TIMEOUT_SECONDS)
    assert not worker.is_alive(), (
        f"the run did not finish in {_RUN_TIMEOUT_SECONDS:.0f}s — a consumer is "
        f"most likely still waiting for a runtime-resolved peer's record"
    )
    if failure:
        raise failure[0]

    non_na = 0
    seen_weekly = set()
    for ts, dep, nested, weekly in rows:
        if weekly is not None and not isinstance(weekly, NA) and weekly == weekly:
            seen_weekly.add(round(float(weekly), 6))
        dep_na = dep is None or isinstance(dep, NA) or dep != dep
        nested_na = nested is None or isinstance(nested, NA) or nested != nested
        assert dep_na == nested_na, f"ts={ts}: na mismatch dep={dep} nested={nested}"
        if not dep_na:
            non_na += 1
            assert dep == nested, f"ts={ts}: dep={dep!r} != nested={nested!r}"
    assert non_na > 400, f"only {non_na} comparable bars — the fixture is too short"
    assert len(seen_weekly) > 10, \
        f"the weekly close takes only {len(seen_weekly)} distinct values — degenerate"
    log.info("runtime-resolved peer paired on %d bars", non_na)
