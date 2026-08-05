"""
@pyne

ta.percentile_* with an na percentage.

The shared selectors used to raise on an na percentage, so a warmup-derived
argument halted the script -- while the array face of the same function quietly
returned na. Measured on TradingView (FX:EURUSD 240), the two faces agree: the
nearest rank answers an na percentage exactly like 0, tracking ta.lowest bar by
bar, and the interpolation form answers na.
"""
from pynecore.lib import script, close, ta
from pynecore.types.na import NA


@script.indicator(title="ta percentile na percentage")
def main():
    return {
        "rank_na": ta.percentile_nearest_rank(close, 5, NA(float)),
        "rank_zero": ta.percentile_nearest_rank(close, 5, 0),
        "lowest": ta.lowest(close, 5),
        "lin_na": ta.percentile_linear_interpolation(close, 5, NA(float)),
        "lin_zero": ta.percentile_linear_interpolation(close, 5, 0),
    }


def __test_percentile_na_percentage__(csv_reader, runner):
    """An na percentage is 0 for the nearest rank and na for the interpolation"""
    checked = 0
    with csv_reader('ma.csv', subdir="data") as cr:
        for i, (_, plot) in enumerate(runner(cr).run_iter()):
            rank_na = plot["rank_na"]
            lin_na = plot["lin_na"]
            assert isinstance(lin_na, NA) or lin_na != lin_na, \
                f"bar {i}: interpolation with an na percentage must be na, got {lin_na!r}"
            if i < 4:
                # Warm-up: the window is not full yet, everything is na
                assert isinstance(rank_na, NA) or rank_na != rank_na
            else:
                assert rank_na == plot["rank_zero"] == plot["lowest"], \
                    f"bar {i}: {rank_na!r} != {plot['rank_zero']!r} / {plot['lowest']!r}"
                # The 0 percentage itself is unaffected by the na branch
                assert plot["lin_zero"] == plot["lowest"]
                checked += 1
            if i > 20:
                break
    assert checked > 10, "too few bars compared"
