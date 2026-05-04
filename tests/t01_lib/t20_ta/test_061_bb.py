"""
@pyne
"""
from pynecore.lib import script, ta, close


@script.indicator(title="BB Test", shorttitle="BB_T", overlay=True)
def main():
    # This also tests the stdev and variance functions
    base, upper, lower = ta.bb(close, 20, 2)
    return {
        "ta.bb(close, 20, 2).base": base,
        "ta.bb(close, 20, 2).upper": upper,
        "ta.bb(close, 20, 2).lower": lower,
        "ta.bbw(close, 20, 2.1)": ta.bbw(close, 20, 2.1),
    }


def __test_bb__(csv_reader, runner, dict_comparator, log):
    """ BB / BBW """
    with csv_reader('bb.csv', subdir="data") as cr:
        for i, (candle, plot) in enumerate(runner(cr).run_iter()):
            dict_comparator(plot, candle.extra_fields, abs_tol=1e-10, rel_tol=1e-7)
            if i > 150:
                break
