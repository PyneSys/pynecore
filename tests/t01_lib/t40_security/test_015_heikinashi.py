"""
@pyne
"""
from pynecore.lib import close, open, high, low, format, plot, request, script, syminfo, timeframe, ticker


@script.indicator(title="Heikin Ashi Test", shorttitle="HA", format=format.price, precision=8)
def main():
    ha = ticker.heikinashi(syminfo.tickerid)
    plot(request.security(ha, timeframe.period, close), title="haClose")
    plot(request.security(ha, timeframe.period, open), title="haOpen")
    plot(request.security(ha, timeframe.period, high), title="haHigh")
    plot(request.security(ha, timeframe.period, low), title="haLow")


def __test_heikinashi_same_symbol__(runner, syminfo, tmp_path, log):
    """request.security(ticker.heikinashi(...)) returns TV-standard Heikin Ashi bars.

    A same-symbol Heikin Ashi request must route to a security subprocess reading
    a transformed feed (never the same-context inline path). The transform is
    checked against the hand-computed TradingView recurrence.
    """
    import math
    from datetime import datetime, UTC
    from pynecore import lib
    from pynecore.types.na import NA
    from pynecore.types.ohlcv import OHLCV
    from pynecore.core.ohlcv_file import OHLCVWriter

    # Deterministic 5-minute feed (matches the syminfo fixture period)
    base_ts = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp())
    ohlc = [
        (100.0, 105.0, 99.0, 104.0),
        (104.0, 108.0, 103.0, 106.0),
        (106.0, 107.0, 101.0, 102.0),
        (102.0, 103.0, 98.0, 100.0),
        (100.0, 110.0, 100.0, 109.0),
        (109.0, 112.0, 107.0, 108.0),
        (108.0, 109.0, 104.0, 105.0),
        (105.0, 106.0, 100.0, 101.0),
    ]
    bars = [OHLCV(timestamp=base_ts + i * 300, open=o, high=h, low=lo, close=c, volume=1000.0)
            for i, (o, h, lo, c) in enumerate(ohlc)]

    # Write the source feed + syminfo sidecar the security child loads
    src = tmp_path / "ha_src.ohlcv"
    with OHLCVWriter(src) as w:
        for b in bars:
            w.write(b)
    syminfo.save_toml(src.with_suffix('.toml'))

    # Hand-compute the TradingView Heikin Ashi recurrence
    exp_close, exp_open, exp_high, exp_low = [], [], [], []
    prev_open = prev_close = None
    for b in bars:
        hc = (b.open + b.high + b.low + b.close) / 4.0
        ho = (b.open + b.close) / 2.0 if prev_open is None else (prev_open + prev_close) / 2.0
        exp_close.append(hc)
        exp_open.append(ho)
        exp_high.append(max(b.high, ho, hc))
        exp_low.append(min(b.low, ho, hc))
        prev_open, prev_close = ho, hc

    # Same symbol (PYTEST:TEST) mapped to the source file → the base symbol
    # matches, the Heikin Ashi marker routes it to a transformed subprocess feed.
    tickerid = f"{syminfo.prefix}:{syminfo.ticker}"
    r = runner(
        iter(bars),
        security_data={tickerid: str(src.with_suffix(''))},
    )

    checked = 0
    for i, (_candle, plot_values) in enumerate(r.run_iter()):
        hc = plot_values.get('haClose')
        if isinstance(hc, NA) or hc is None:
            continue
        assert math.isclose(hc, exp_close[i], rel_tol=1e-6, abs_tol=1e-6), \
            f"bar {i}: haClose={hc} != expected {exp_close[i]}"
        assert math.isclose(plot_values['haOpen'], exp_open[i], rel_tol=1e-6, abs_tol=1e-6), \
            f"bar {i}: haOpen={plot_values['haOpen']} != expected {exp_open[i]}"
        assert math.isclose(plot_values['haHigh'], exp_high[i], rel_tol=1e-6, abs_tol=1e-6), \
            f"bar {i}: haHigh={plot_values['haHigh']} != expected {exp_high[i]}"
        assert math.isclose(plot_values['haLow'], exp_low[i], rel_tol=1e-6, abs_tol=1e-6), \
            f"bar {i}: haLow={plot_values['haLow']} != expected {exp_low[i]}"
        checked += 1

    assert checked >= len(bars) - 1, \
        f"expected Heikin Ashi values on nearly every bar, got {checked}/{len(bars)}"
    log.info(f"Heikin Ashi same-symbol test passed — {checked}/{len(bars)} bars matched")
