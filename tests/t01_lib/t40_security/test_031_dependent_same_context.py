"""
@pyne
"""
from pynecore.lib import close, plot, request, script, syminfo, ta
from pynecore.types import Series


@script.indicator(title="Dependent Same Context", shorttitle="DSC")
def main():
    # Dependent form: the second context's expression READS the first one's
    # value. TradingView evaluates it in the requested context, so the inner
    # ``ta.sma`` runs on the monthly RSI stream.
    rsi_m: Series[float] = request.security(syminfo.tickerid, "M", ta.rsi(close, 14))
    sma_dep: Series[float] = request.security(syminfo.tickerid, "M", ta.sma(rsi_m, 14))
    # Fully nested reference form: one context, everything inside.
    sma_nested: Series[float] = request.security(
        syminfo.tickerid, "M", ta.sma(ta.rsi(close, 14), 14))
    plot(rsi_m, "rsi_m")
    plot(sma_dep, "sma_dep")
    plot(sma_nested, "sma_nested")


_T0 = 1_262_304_000_000  # 2010-01-01T00:00:00 UTC, a Friday, on the day grid
_DAY = 86_400_000
_N_DAYS = 2200
# The chart starts deep inside the feed, so the child's first round replays
# ~50 monthly bars in ONE batch — the warmup the whole ring machinery exists for.
_CHART_START = 1500


def __test_helper_bars():
    from pynecore.types.ohlcv import OHLCV
    import math
    out = []
    for i in range(_N_DAYS):
        # A deterministic, non-monotonic price so RSI/SMA are non-degenerate.
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
        prefix="EXCH", description="Dependent same context", ticker="DSC",
        currency="USD", period="1D", type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))
    return str(path)


def __test_dependent_same_context_equals_nested__(runner, log):
    """A dependent same-context chain equals the fully nested form on every bar.

    ``ta.sma(rsiMonthly, 14)`` requested in the same ``"M"`` context must be
    bit-identical to ``ta.sma(ta.rsi(close, 14), 14)`` requested as one nested
    expression — including the warmup, where the historical batch replays many
    monthly bars in a single round. Without a per-bar ring the consumer would
    read only the round's LAST monthly value and the whole warmup would drift.
    """
    import sys
    import tempfile
    from pathlib import Path
    from pynecore.types.na import NA

    sys.modules.pop(Path(__file__).stem, None)

    rows = []
    with tempfile.TemporaryDirectory() as td:
        feed = __test_helper_write_feed(Path(td))
        r = runner(__test_helper_bars()[_CHART_START:],
                   syminfo_override={"period": "1D"},
                   security_data={"M": feed})
        for candle, pv in r.run_iter():
            rows.append((candle.timestamp, pv.get("sma_dep"), pv.get("sma_nested"),
                         pv.get("rsi_m")))

    non_na = 0
    seen_rsi = set()
    for ts, dep, nested, rsi in rows:
        if rsi is not None and not isinstance(rsi, NA) and rsi == rsi:
            seen_rsi.add(round(float(rsi), 6))
        dep_na = dep is None or isinstance(dep, NA) or dep != dep
        nested_na = nested is None or isinstance(nested, NA) or nested != nested
        assert dep_na == nested_na, f"ts={ts}: na mismatch dep={dep} nested={nested}"
        if not dep_na:
            non_na += 1
            assert dep == nested, f"ts={ts}: dep={dep!r} != nested={nested!r}"
    assert non_na > 600, f"only {non_na} comparable bars — the fixture is too short"
    assert len(seen_rsi) > 10, \
        f"the monthly RSI takes only {len(seen_rsi)} distinct values — degenerate fixture"
    log.info("dependent monthly chain matches the nested form on %d bars", non_na)
