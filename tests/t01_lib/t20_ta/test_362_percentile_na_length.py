"""
@pyne

ta percentile family with an na length.

An unset ``array.new_int()`` slot handed straight to a rolling ta call is a
real-world shape: TradingView's own runtime answers na for the percentile
machines instead of raising, while every other rolling function raises
RE10003 ("must not be na"). Measured on BINANCE:BTCUSDT 30m across
percentile_nearest_rank, percentile_linear_interpolation, median, mode and
fourteen other ta calls; a length of exactly 0 raises RE10001 in this family
too, so only na is tolerant.
"""
from pynecore.lib import script, close, ta
from pynecore.types.na import NA


@script.indicator(title="ta percentile na length")
def main():
    na_len = NA(int)
    return {
        "rank_na_len": ta.percentile_nearest_rank(close, na_len, 75),
        "lin_na_len": ta.percentile_linear_interpolation(close, na_len, 75),
        "median_na_len": ta.median(close, na_len),
        "mode_na_len": ta.mode(close, na_len),
        "rank_ref": ta.percentile_nearest_rank(close, 5, 75),
    }


def __test_percentile_na_length__(csv_reader, runner):
    """An na length yields na without halting the script"""
    checked = 0
    with csv_reader('ma.csv', subdir="data") as cr:
        for i, (_, plot) in enumerate(runner(cr).run_iter()):
            for key in ("rank_na_len", "lin_na_len", "median_na_len", "mode_na_len"):
                value = plot[key]
                assert isinstance(value, NA) or value != value, \
                    f"bar {i}: {key} must be na with an na length, got {value!r}"
            if i >= 4:
                # The na length must not disturb a live window next to it
                ref = plot["rank_ref"]
                assert not isinstance(ref, NA) and ref == ref, \
                    f"bar {i}: the reference percentile must still compute, got {ref!r}"
                checked += 1
            if i > 20:
                break
    assert checked > 10, "too few bars compared"
