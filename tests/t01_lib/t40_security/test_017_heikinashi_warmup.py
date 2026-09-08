"""
@pyne
"""
from pynecore.lib import open, format, plot, request, script, syminfo, ticker


@script.indicator(title="Heikin Ashi Warmup Test", shorttitle="HAW",
                  format=format.price, precision=8)
def main():
    ha = ticker.heikinashi(syminfo.tickerid)
    plot(request.security(ha, "15", open), title="haOpen15")


def __test_helper_ha_open(bars):
    """The Heikin Ashi open of the bar FOLLOWING this sequence."""
    prev_open = prev_close = None
    for b in bars:
        ha_close = (b.open + b.high + b.low + b.close) / 4.0
        ha_open = ((b.open + b.close) / 2.0 if prev_open is None
                   else (prev_open + prev_close) / 2.0)
        prev_open, prev_close = ha_open, ha_close
    return (prev_open + prev_close) / 2.0


def __test_heikinashi_htf_warmup_seed__(runner, syminfo, tmp_path, log):
    """A mapped feed warms an HTF Heikin Ashi context's recurrence, without feeding it.

    TradingView aggregates a same-symbol INTRADAY context from the chart's bars —
    measured on BINANCE:BTCUSDT@30, a ``"120"`` Heikin Ashi context opens at
    ``bar_index`` 0 on the chart's first bar, exactly like the plain one — yet its
    first ``open`` is the Heikin Ashi recurrence carried over from the context
    feed's EARLIER bars. So a ``--security`` mapping for such a context is a seed
    source, not a data source: the bars before the chart advance the transform,
    the chart's own aggregation stays the series.
    """
    import math
    from datetime import datetime, UTC
    from pynecore.types.na import NA
    from pynecore.types.ohlcv import OHLCV
    from pynecore.core.ohlcv import OHLCVWriter

    base_ts = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp()) * 1000
    warm_ohlc = [
        (50.0, 56.0, 49.0, 55.0),
        (55.0, 58.0, 52.0, 53.0),
        (53.0, 61.0, 53.0, 60.0),
        (60.0, 64.0, 58.0, 59.0),
    ]
    chart_ohlc = [
        (100.0, 105.0, 99.0, 104.0),
        (104.0, 108.0, 103.0, 106.0),
        (106.0, 107.0, 101.0, 102.0),
        (102.0, 103.0, 98.0, 100.0),
        (100.0, 110.0, 100.0, 109.0),
        (109.0, 112.0, 107.0, 108.0),
        (108.0, 109.0, 104.0, 105.0),
        (105.0, 106.0, 100.0, 101.0),
        (101.0, 104.0, 100.0, 103.0),
    ]
    # The chart's own 5-minute bars start where the 15-minute warmup feed ends.
    chart_start = base_ts + len(warm_ohlc) * 900_000
    chart_bars = [OHLCV(timestamp=chart_start + i * 300_000,
                        open=o, high=h, low=lo, close=c, volume=1000.0)
                  for i, (o, h, lo, c) in enumerate(chart_ohlc)]
    warm_bars = [OHLCV(timestamp=base_ts + i * 900_000,
                       open=o, high=h, low=lo, close=c, volume=3000.0)
                 for i, (o, h, lo, c) in enumerate(warm_ohlc)]

    chart_path = tmp_path / "ha_warm_chart.ohlcv"
    with OHLCVWriter(chart_path, syminfo.period) as w:
        for b in chart_bars:
            w.write(b)
    syminfo.save_toml(chart_path.with_suffix('.toml'))

    warm_path = tmp_path / "ha_warm_ctx.ohlcv"
    with OHLCVWriter(warm_path, "15") as w:
        for b in warm_bars:
            w.write(b)
    syminfo.save_toml(warm_path.with_suffix('.toml'))

    # The context's first bar aggregates the chart's first three 5-minute bars.
    first_ctx_bar = OHLCV(
        timestamp=chart_start, open=chart_ohlc[0][0],
        high=max(b[1] for b in chart_ohlc[:3]), low=min(b[2] for b in chart_ohlc[:3]),
        close=chart_ohlc[2][3], volume=3000.0)
    warm_open = __test_helper_ha_open(warm_bars)
    cold_open = (first_ctx_bar.open + first_ctx_bar.close) / 2.0
    assert not math.isclose(warm_open, cold_open, rel_tol=1e-9), \
        "the fixture must separate the warm seed from the cold one"

    tickerid = f"{syminfo.prefix}:{syminfo.ticker}"

    def first_ha_open(security_data):
        r = runner(iter(chart_bars), security_data=security_data,
                   chart_data_path=chart_path)
        for _candle, plot_values in r.run_iter():
            value = plot_values.get('haOpen15')
            if value is not None and not isinstance(value, NA) and value == value:
                return value
        return None

    seeded = first_ha_open({f"{tickerid}:15": str(warm_path.with_suffix(''))})
    cold = first_ha_open(None)

    assert cold is not None and math.isclose(cold, cold_open, rel_tol=1e-9), \
        f"unmapped context must start the recurrence cold, got {cold} != {cold_open}"
    assert seeded is not None and math.isclose(seeded, warm_open, rel_tol=1e-9), \
        f"mapped context must carry the warmup recurrence, got {seeded} != {warm_open}"
    log.info(f"Heikin Ashi HTF warmup seed test passed — cold {cold}, seeded {seeded}")
