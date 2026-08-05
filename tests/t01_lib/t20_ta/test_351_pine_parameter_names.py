"""
@pyne
"""
from pynecore.lib import script, ta, close, bar_index


@script.indicator(title="TA Pine parameter names", shorttitle="ta_pine_names")
def main():
    # Every parameter below is named the way Pine names it, so a compiled named
    # argument binds instead of raising TypeError. A rising source keeps the
    # oscillators out of their na branch on the constant dummy feed.
    src = close + bar_index * 0.1

    alma_kw = ta.alma(series=src, length=5, offset=0.85, sigma=6.0, floor=False)
    bb_kw_middle, bb_kw_upper, bb_kw_lower = ta.bb(series=src, length=5, mult=2.0)
    bbw_kw = ta.bbw(series=src, length=5, mult=2.0)
    cmo_kw = ta.cmo(series=src, length=5)
    mfi_kw = ta.mfi(series=src, length=5)
    st_kw, st_dir_kw = ta.supertrend(factor=3.0, atrPeriod=5)

    alma_pos = ta.alma(src, 5, 0.85, 6.0, False)
    bb_pos_middle, bb_pos_upper, bb_pos_lower = ta.bb(src, 5, 2.0)
    bbw_pos = ta.bbw(src, 5, 2.0)
    cmo_pos = ta.cmo(src, 5)
    mfi_pos = ta.mfi(src, 5)
    st_pos, st_dir_pos = ta.supertrend(3.0, 5)

    if bar_index >= 10:
        assert alma_kw == alma_pos
        assert bb_kw_middle == bb_pos_middle
        assert bb_kw_upper == bb_pos_upper
        assert bb_kw_lower == bb_pos_lower
        assert bbw_kw == bbw_pos
        assert cmo_kw == cmo_pos
        assert mfi_kw == mfi_pos
        assert st_kw == st_pos
        assert st_dir_kw == st_dir_pos


def __test_ta_pine_parameter_names__(runner, dummy_ohlcv_iter):
    """ Pine parameter names bind as keyword arguments """
    run_iter = runner(dummy_ohlcv_iter).run_iter()
    for _ in range(20):
        next(run_iter)
