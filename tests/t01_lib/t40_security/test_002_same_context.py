"""
@pyne
"""
import math

from pynecore.lib import close, high, format, plot, request, script, syminfo, timeframe, ta


@script.indicator(title="Same Context Test", shorttitle="SCT", format=format.price, precision=6)
def main():
    # Same symbol + same timeframe → should use chart data directly (no process)
    sec_close = request.security(syminfo.ticker, timeframe.period, close)
    sec_high = request.security(syminfo.ticker, timeframe.period, high)
    sec_sma = request.security(syminfo.ticker, timeframe.period, ta.sma(close, 5))

    plot(sec_close, title="SecClose")
    plot(sec_high, title="SecHigh")
    plot(sec_sma, title="SecSMA")


def __test_same_context__(csv_reader, runner, log, monkeypatch):
    """Same symbol + same TF uses chart data directly without spawning a process."""
    from pynecore import lib
    from pynecore.core import security
    from pynecore.types.na import NA

    cleanup_calls = 0
    cleanup_shared_memory = security.cleanup_shared_memory

    def count_cleanup(sync_block, result_blocks):
        nonlocal cleanup_calls
        cleanup_calls += 1
        cleanup_shared_memory(sync_block, result_blocks)

    monkeypatch.setattr(security, "cleanup_shared_memory", count_cleanup)
    sma_values = []
    with csv_reader('advance_decline_ratio.csv', subdir="data") as cr:
        r = runner(
            cr,
            syminfo_override=dict(timezone="US/Eastern"),
        )

        for i, (candle, plot_values) in enumerate(r.run_iter()):
            sma_values.append(lib.close)

            # sec_close must be exactly close
            if not isinstance(plot_values.get('SecClose'), NA):
                assert plot_values['SecClose'] == lib.close, \
                    f"bar {i}: SecClose={plot_values['SecClose']} != close={lib.close}"

            # sec_high must be exactly high
            if not isinstance(plot_values.get('SecHigh'), NA):
                assert plot_values['SecHigh'] == lib.high, \
                    f"bar {i}: SecHigh={plot_values['SecHigh']} != high={lib.high}"

            # sec_sma should match chart-side SMA(close, 5)
            if i >= 4:
                expected_sma = sum(sma_values[-5:]) / 5
                sec_sma_val = plot_values.get('SecSMA')
                if sec_sma_val is not None and not isinstance(sec_sma_val, NA):
                    assert math.isclose(sec_sma_val, expected_sma, rel_tol=1e-10), \
                        f"bar {i}: SecSMA={sec_sma_val} != expected={expected_sma}"

    assert cleanup_calls == 1
    log.info("Same context test passed — values match chart data")
