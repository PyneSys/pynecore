"""
@pyne
"""
from pynecore.lib import script, ta, close


@script.indicator(title="BB Test", shorttitle="BB_T", overlay=True)
def main():
    # This also tests the stdev and variance functions
    base, upper, lower = ta.kc(close, 20, 2)
    return {
        "ta.kc(close, 20, 2).base": base,
        "ta.kc(close, 20, 2).upper": upper,
        "ta.kc(close, 20, 2).lower": lower,
        "ta.kcw(close, 20, 2.1, useTrueRange=false)": ta.kcw(close, 20, 2.1, useTrueRange=False),
    }


def __test_kc__(csv_reader, runner, dict_comparator, log):
    """ KC / KCW """
    with csv_reader('kc.csv', subdir="data") as cr:
        for i, (candle, plot) in enumerate(runner(cr).run_iter()):
            dict_comparator(plot, candle.extra_fields, abs_tol=1e-10, rel_tol=1e-7)
            if i > 150:
                break
