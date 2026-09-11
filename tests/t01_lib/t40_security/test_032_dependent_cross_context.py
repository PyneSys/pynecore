"""
@pyne
"""
from pynecore.lib import close, plot, request, script, syminfo, ta
from pynecore.types import Series


@script.indicator(title="Dependent Cross Context", shorttitle="DCC")
def main():
    # The three TradingView-measured pairings, in one script (probe
    # ``cross_ctx.pine``, CAPITALCOM:EURUSD @ 240):
    #   * a WEEKLY consumer of a DAILY producer,
    #   * a DAILY consumer of a WEEKLY producer,
    #   * an EURUSD-daily consumer of a GBPUSD-daily producer.
    # All three reduce to ONE rule: the producer's last bar whose scheduled
    # close is at or before the consumer bar's scheduled close.
    rsiDaily: Series[float] = request.security(syminfo.tickerid, "D", ta.rsi(close, 14))
    smaWofD: Series[float] = request.security(syminfo.tickerid, "W", ta.sma(rsiDaily, 3))
    rsiWeek: Series[float] = request.security(syminfo.tickerid, "W", ta.rsi(close, 14))
    smaDofW: Series[float] = request.security(syminfo.tickerid, "D", ta.sma(rsiWeek, 3))
    rsiGbpD: Series[float] = request.security("CAPITALCOM:GBPUSD", "D", ta.rsi(close, 14))
    smaEurOfGbp: Series[float] = request.security(
        "CAPITALCOM:EURUSD", "D", ta.sma(rsiGbpD, 3))
    plot(rsiDaily, "rsiDaily")
    plot(smaWofD, "smaWofD")
    plot(rsiWeek, "rsiWeek")
    plot(smaDofW, "smaDofW")
    plot(rsiGbpD, "rsiGbpD")
    plot(smaEurOfGbp, "smaEurOfGbp")


# TradingView writes ``na`` as 1e100 in the exported CSV.
__test_helper_TV_NA = 1e99
# The producers are RMA-based (infinite memory), so the trimmed feeds converge
# on TradingView's state rather than starting from it. Measured, the residual is
# below 1e-9 across the whole window — the values are effectively bit-identical,
# and a pairing error would be whole producer bars wide.
__test_helper_TOL = 1e-9
__test_helper_PLOTS = ("rsiDaily", "smaWofD", "rsiWeek", "smaDofW",
                       "rsiGbpD", "smaEurOfGbp")


def __test_helper_fx_syminfo_kwargs(ticker, base, period):
    from datetime import time
    from pynecore.core.syminfo import SymInfoInterval, SymInfoSession
    # Forex week: sessions OPEN Sun-Thu 17:00 New York and each runs 24h, so the
    # week's last session ends Friday 17:00 New York — the instant TradingView
    # confirms the weekly bar on.
    open_days = (6, 0, 1, 2, 3)
    end_days = (0, 1, 2, 3, 4)
    return dict(
        prefix="CAPITALCOM", description=f"{base} / US Dollar", ticker=ticker,
        currency="USD", basecurrency=base, period=period, type="forex",
        mintick=0.00001, pricescale=100000, minmove=1, pointvalue=1,
        timezone="America/New_York", volumetype="tick",
        opening_hours=[SymInfoInterval(day=d, start=time(17, 0), end=time(17, 0))
                       for d in open_days],
        session_starts=[SymInfoSession(day=d, time=time(17, 0)) for d in open_days],
        session_ends=[SymInfoSession(day=d, time=time(17, 0)) for d in end_days],
    )


def __test_dependent_cross_context_matches_tradingview__(runner, log):
    """Cross-context dependent security values match the TradingView reference.

    The probe ``cross_ctx.pine`` was run on CAPITALCOM:EURUSD @ 240 and its six
    plots exported; the fixture holds the last 400 chart bars plus the daily and
    weekly producer feeds. TradingView's three pairings — weekly consumer of a
    daily producer (last daily bar of the week), daily consumer of a weekly
    producer (last weekly bar closed by the day's close) and same-timeframe
    cross-symbol (time-aligned) — are all reproduced by the single
    ``close_A <= asof`` rule, so this asserts them together on real data.
    """
    import csv
    import sys
    from pathlib import Path
    from pynecore.types.na import NA
    from pynecore.types.ohlcv import OHLCV

    sys.modules.pop(Path(__file__).stem, None)

    data_dir = Path(__file__).parent / "data"
    # The export's FINAL bar of every series (chart, daily, weekly) is still
    # forming at fetch time, so the last chart bar is dropped: its producers'
    # last bars are partial and would compare a closed value against a snapshot.
    rows = list(csv.DictReader(open(data_dir / "dependent_cross_context.csv")))[:-1]

    bars = [OHLCV(timestamp=int(float(r["time"])) * 1000, open=float(r["open"]),
                  high=float(r["high"]), low=float(r["low"]),
                  close=float(r["close"]), volume=float(r["Volume"]))
            for r in rows]

    security_data = {
        "CAPITALCOM:EURUSD:D": str(data_dir / "dep_cross_EURUSD_D"),
        "CAPITALCOM:EURUSD:W": str(data_dir / "dep_cross_EURUSD_W"),
        "CAPITALCOM:GBPUSD:D": str(data_dir / "dep_cross_GBPUSD_D"),
    }

    r = runner(bars,
               syminfo_override=__test_helper_fx_syminfo_kwargs("EURUSD", "EUR", "240"),
               security_data=security_data)

    compared = {name: 0 for name in __test_helper_PLOTS}
    bad = []
    for i, (candle, pv) in enumerate(r.run_iter()):
        ref = rows[i]
        assert int(float(ref["time"])) * 1000 == candle.timestamp, \
            f"bar {i}: fixture/chart timestamp drift"
        for name in __test_helper_PLOTS:
            expected = float(ref[name])
            if expected != expected or expected >= __test_helper_TV_NA:
                continue
            got = pv.get(name)
            if got is None or isinstance(got, NA) or got != got:
                bad.append((i, name, expected, "na"))
                continue
            compared[name] += 1
            if abs(float(got) - expected) > __test_helper_TOL:
                bad.append((i, name, expected, float(got)))

    if bad:
        head = "; ".join(f"bar {i} {n}: tv={e} pyne={g}" for i, n, e, g in bad[:8])
        raise AssertionError(f"{len(bad)} mismatches vs TradingView: {head}")
    for name in __test_helper_PLOTS:
        assert compared[name] > 300, \
            f"{name}: only {compared[name]} comparable bars — fixture/wiring problem"
    log.info("cross-context dependent security matches TradingView on %d bars", len(rows))
