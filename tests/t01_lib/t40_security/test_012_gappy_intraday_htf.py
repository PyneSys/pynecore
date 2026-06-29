"""
@pyne
"""
from pynecore.lib import barmerge, high, plot, request, script, syminfo, ta
from pynecore.types import Series


@script.indicator(title="Gappy Intraday HTF", shorttitle="GIH")
def main():
    # ``h60`` is the confirmed (lookahead_off) HTF high; ``hh`` is a bar-count
    # history read (highest over the last 3 HTF bars). Both must see the HTF
    # series as REAL bars only — never the writer's V=-1 gap fills.
    h60: Series[float] = request.security(
        syminfo.tickerid, "60", high, lookahead=barmerge.lookahead_off)
    hh: Series[float] = request.security(
        syminfo.tickerid, "60", ta.highest(high, 3), lookahead=barmerge.lookahead_off)
    plot(h60, "h60")
    plot(hh, "hh")


# A 60-minute HTF feed with a 2-hour gap. ``OHLCVWriter`` forward-fills the gap
# (hours 4 and 5) with flat V=-1 bars, so the stored feed is continuous even
# though only hours 0,1,2,3,6 carry real bars. ``high`` is distinctive per bar
# so both the confirmed value and the highest-of-3 lookback are unambiguous.
_T0 = 1735689600  # 2025-01-01T00:00:00 UTC, aligned to the 3600s and 300s grids
_REAL = {0: (100, 9), 1: (10, 9), 2: (11, 10), 3: (50, 20), 6: (13, 13)}  # hour -> (high, close)


def _write_htf(tmp_dir):
    """Write the gappy 60-minute HTF feed (+ a 24/7 UTC ``.toml``); return the path."""
    from datetime import time
    from pynecore.core.ohlcv_file import OHLCVWriter
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / "HTF60.ohlcv"
    with OHLCVWriter(path) as w:
        for hour, (hi, cl) in _REAL.items():
            w.write(OHLCV(timestamp=_T0 + hour * 3600,
                          open=float(hi), high=float(hi), low=float(cl), close=float(cl),
                          volume=1.0))

    SymInfo(
        prefix="EXCH", description="Gappy HTF", ticker="GHTF",
        currency="USD", period="60", type="crypto",
        mintick=0.01, pricescale=100, minmove=1, pointvalue=1, mincontract=0.0001,
        timezone="UTC", volumetype="base",
        opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                       for i in range(7)],
        session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
        session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
    ).save_toml(path.with_suffix(".toml"))
    return str(path)


def _chart_bars(n):
    """``n`` dense 5-minute chart bars from ``_T0`` (OHLC body is irrelevant)."""
    from pynecore.types.ohlcv import OHLCV
    return [OHLCV(timestamp=_T0 + i * 300, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0)
            for i in range(n)]


def __test_gappy_intraday_htf_skips_fills_and_holds_over_gap__(runner, log):
    """A gappy intraday HTF feed: the child reads only real bars and holds over the gap.

    Regression for the session-gapped intraday ``request.security()`` HTF chain:

    * The security child must read only the REAL HTF bars, never the V=-1 gap
      fills the OHLCV file stores for the missing hours 4 and 5. A bar-count
      history read (``ta.highest(high, 3)``) therefore spans 3 real HTF bars,
      not 3 feed rows padded with fills — so at hour 7 it reaches the hour-3
      high (50), not the fills' carried close (20).
    * ``lookahead_off`` confirmation must hold the last CLOSED fixed-span bar
      across the gap. Hour 3's bar closes at hour 4, so hours 4, 5 and 6 all
      report it (high 50) — never a stale earlier bar and never ``na``.

    Both break without the gap-compacted child read + the real-open clamp:
    without compaction hour 7's ``hh`` would be 20; without the clamp the gap
    hours would read the hour-2 bar (11) or ``na``.
    """
    import sys
    import tempfile
    from pathlib import Path
    from pynecore.types.na import NA

    sys.modules.pop(Path(__file__).stem, None)

    # Per chart bar, keyed by (hour, minute) for unambiguous assertions.
    rows = {}
    with tempfile.TemporaryDirectory() as td:
        htf_path = _write_htf(Path(td))
        # Hours 0..7 inclusive at 5-minute steps (8 * 12 bars); hour 7 lets the
        # hour-6 bar close so it can be confirmed.
        r = runner(_chart_bars(8 * 12), security_data={"60": htf_path})
        for candle, pv in r.run_iter():
            minute_of_day = (candle.timestamp - _T0) // 60
            rows[(minute_of_day // 60, minute_of_day % 60)] = (pv.get("h60"), pv.get("hh"))

    def h60(hr):
        return rows[(hr, 0)][0]

    def hh(hr):
        return rows[(hr, 0)][1]

    # Confirmed HTF high (lookahead_off): each fixed-span bar is reported once it
    # has closed, then held until the next real bar closes.
    assert h60(2) == 10, f"hour 2: h60={h60(2)} != 10 (hour-1 bar)"
    assert h60(3) == 11, f"hour 3: h60={h60(3)} != 11 (hour-2 bar)"
    # Hours 4, 5, 6: the hour-3 bar (closed at hour 4) is held across the gap and
    # while the hour-6 bar is still developing. Without the real-open clamp the
    # gap hours read a stale earlier bar (hour 2 = 11) or na.
    for hour in (4, 5, 6):
        assert h60(hour) == 50, f"hour {hour}: h60={h60(hour)} != 50 (hour-3 bar held over gap)"
    assert h60(7) == 13, f"hour 7: h60={h60(7)} != 13 (hour-6 bar)"

    # No spurious na once the HTF series has started.
    for hour in range(2, 8):
        assert not isinstance(h60(hour), NA), f"hour {hour}: h60 unexpectedly na"

    # Bar-count history read over real bars only: at hour 3 the window covers the
    # hour-0..2 bars (max 100); at hour 7 it covers the hour-2/3/6 bars (max 50) —
    # the gap fills are NOT counted (otherwise the window would stop at 20).
    assert hh(3) == 100, f"hour 3: hh={hh(3)} != 100 (max of hour 0..2 highs)"
    assert hh(7) == 50, f"hour 7: hh={hh(7)} != 50 (max of real hour 2/3/6 highs, fills excluded)"

    log.info("gappy intraday HTF: child skips V=-1 fills, holds the closed bar across the gap")


def __test_load_htf_bar_opens_gappy_vs_dense_intraday__(log):
    """``load_htf_bar_opens`` arms the real-open clamp for a gappy intraday feed only.

    A gappy intraday HTF feed (real bars sparser than the stored, gap-filled
    grid) gets ``bar_opens`` set to the REAL opens with single-period
    (clamp) semantics. A dense feed and an LTF context stay no-ops — the cheaper
    arithmetic grid / the LTF intrabar machinery handle them.
    """
    import tempfile
    from pathlib import Path
    from zoneinfo import ZoneInfo
    from datetime import time
    from pynecore.core.ohlcv_file import OHLCVWriter
    from pynecore.core.resampler import Resampler
    from pynecore.core.security import SecurityState, load_htf_bar_opens
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    from pynecore.types.ohlcv import OHLCV

    def _state(is_ltf=False):
        return SecurityState(
            sec_id="s", timeframe="60", gaps_on=False, same_timeframe=False,
            resampler=Resampler.get_resampler("60"), tz=ZoneInfo("UTC"), is_ltf=is_ltf,
        )

    def _feed(tmp_dir, name, hours):
        path = tmp_dir / f"{name}.ohlcv"
        with OHLCVWriter(path) as w:
            for hour in hours:
                v = float(hour + 1)
                w.write(OHLCV(timestamp=_T0 + hour * 3600,
                              open=v, high=v, low=v, close=v, volume=1.0))
        SymInfo(prefix="E", description="d", ticker="GHTF", currency="USD", period="60",
                type="crypto", mintick=0.01, pricescale=100, minmove=1, pointvalue=1,
                mincontract=0.0001, timezone="UTC", volumetype="base",
                opening_hours=[SymInfoInterval(day=i, start=time(0, 0), end=time(23, 59, 59))
                               for i in range(7)],
                session_starts=[SymInfoSession(day=i, time=time(0, 0)) for i in range(7)],
                session_ends=[SymInfoSession(day=i, time=time(23, 59, 59)) for i in range(7)],
                ).save_toml(path.with_suffix(".toml"))
        return str(path)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # Gappy: real bars at hours 0,1,2,3,6 → fills at 4,5 in the stored file.
        gappy = _state()
        load_htf_bar_opens(gappy, _feed(tmp, "gappy", [0, 1, 2, 3, 6]))
        assert gappy.bar_opens == [_T0 * 1000 + h * 3600_000 for h in (0, 1, 2, 3, 6)], \
            f"gappy bar_opens={gappy.bar_opens} (must be the 5 real opens, fills excluded)"
        assert gappy.bar_opens_multiperiod is False, "gappy intraday must use the single-period clamp"

        # Dense: a bar every hour → no fills → no-op (arithmetic grid stays correct).
        dense = _state()
        load_htf_bar_opens(dense, _feed(tmp, "dense", list(range(8))))
        assert dense.bar_opens is None, f"dense feed must stay a no-op (bar_opens={dense.bar_opens})"

        # LTF context: never an HTF-confirmation concern.
        ltf = _state(is_ltf=True)
        load_htf_bar_opens(ltf, _feed(tmp, "ltf", [0, 1, 2, 3, 6]))
        assert ltf.bar_opens is None, f"LTF must stay a no-op (bar_opens={ltf.bar_opens})"

    log.info("load_htf_bar_opens arms the clamp for gappy intraday only")
