"""
@pyne
"""
from pynecore.lib import script, log, bar_index, ta, na


@script.indicator(title="EMA RMA Exact", shorttitle="ema_rma_exact")
def main():
    # TradingView parity for the two smoothing machines. Every reference below
    # is the full-precision value TradingView produced for these exact series
    # (probe m558, 2026-08-03); the log rounds to three decimals, so each value
    # is amplified against its reference instead, where one ulp is ~14 units.
    a = 100.0 + (bar_index % 7) / 3.0
    # na bars are skipped whole: the output is na there and the state does not
    # advance, so the window is the last `length` non-na values.
    g = na if bar_index % 3 == 0 else a

    k1 = ta.ema(a, 3)
    k2 = ta.ema(a, 5)
    # rma is Wilder's own machine, not ema with alpha = 1 / length.
    k3 = ta.rma(a, 3)
    k4 = ta.rma(a, 5)
    k5 = ta.ema(g, 4)
    k6 = ta.rma(g, 4)
    # Length 1 is the identity on both.
    k7 = ta.ema(a, 1)

    r1 = 100.8125 if bar_index == 10 else 101.07291666666666
    r2 = 100.79423868312757 if bar_index == 10 else 100.97393689986282
    r3 = 100.80724483056444 if bar_index == 10 else 100.98260766482075
    r4 = 100.79716266666667 if bar_index == 10 else 100.9043968
    r5 = 100.696 if bar_index == 10 else 100.95093333333332
    r6 = 100.734375 if bar_index == 10 else 100.88411458333333
    r7 = 101.0 if bar_index == 10 else 101.33333333333333

    if bar_index == 10 or bar_index == 11:
        log.info("{0}|{1}|{2}|{3}|{4}|{5}|{6}", (k1 - r1) * 1e15, (k2 - r2) * 1e15,
                 (k3 - r3) * 1e15, (k4 - r4) * 1e15, (k5 - r5) * 1e15, (k6 - r6) * 1e15,
                 (k7 - r7) * 1e15)


def __test_ema_rma_exact__(runner, dummy_ohlcv_iter, file_reader, log_comparator):
    """ EMA and RMA TradingView-exact machines """
    tv_log_out = file_reader(subdir="data", suffix=".txt")
    run_iter = runner(dummy_ohlcv_iter).run_iter()
    with log_comparator(tv_log_out):
        for _ in range(14):
            next(run_iter)
