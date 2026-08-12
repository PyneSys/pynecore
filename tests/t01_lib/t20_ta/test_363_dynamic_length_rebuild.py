"""
@pyne

Rolling ta windows must survive a series length that changes mid-run.

``median`` and the two percentile machines keep an incremental window, so a
changed length has to rebuild it from the source history: a shrinking length
otherwise evicts only one value per bar and the machine keeps answering from
the older, wider window forever, and a length that was na for a while comes
back with a window that is short by exactly the na bars. A length that grows
again needs history the smaller length never used, so the source buffer must
not be shrunk to the current length, and ``median`` drops na sources from its
window, so its rebuild has to read a window that drops them too.

Both scenarios are compared against a reference that computes the same thing
without the incremental state: the na-recovery ones against a constant-length
call, and the shrinking ``median`` against ``percentile_linear_interpolation``
at 50%, which is the median for an odd and an even window alike.
"""
from pynecore.lib import script, close, ta, bar_index
from pynecore.types.na import NA


@script.indicator(title="ta dynamic length rebuild")
def main():
    na_then_3 = NA(int) if bar_index < 5 else 3  # na prefix, then a valid length
    gap_3 = NA(int) if bar_index == 8 else 3  # one na bar between two equal lengths
    five_then_2 = 5 if bar_index < 6 else 2  # shrinking length
    two_then_5 = 2 if bar_index < 8 else 5  # growing length
    five_1_five = 1 if bar_index == 12 else 5  # one bar on the length == 1 shortcut
    gapped = NA(float) if bar_index == 7 else close  # an na hole in the source
    five_then_3 = 5 if bar_index < 8 else 3  # shrinking length over the na hole
    return {
        "rank_na": ta.percentile_nearest_rank(close, na_then_3, 100),
        "lin_na": ta.percentile_linear_interpolation(close, na_then_3, 50),
        "median_na": ta.median(close, na_then_3),
        "rank_gap": ta.percentile_nearest_rank(close, gap_3, 100),
        "median_gap": ta.median(close, gap_3),
        "rank_ref": ta.percentile_nearest_rank(close, 3, 100),
        "lin_ref": ta.percentile_linear_interpolation(close, 3, 50),
        "median_ref": ta.median(close, 3),
        "median_shrink": ta.median(close, five_then_2),
        "shrink_ref": ta.percentile_linear_interpolation(close, five_then_2, 50),
        "rank_grow": ta.percentile_nearest_rank(close, two_then_5, 100),
        "lin_grow": ta.percentile_linear_interpolation(close, two_then_5, 50),
        "median_grow": ta.median(close, two_then_5),
        "rank_one": ta.percentile_nearest_rank(close, five_1_five, 100),
        "lin_one": ta.percentile_linear_interpolation(close, five_1_five, 50),
        "median_one": ta.median(close, five_1_five),
        "rank_ref5": ta.percentile_nearest_rank(close, 5, 100),
        "lin_ref5": ta.percentile_linear_interpolation(close, 5, 50),
        "median_ref5": ta.median(close, 5),
        "median_srcgap": ta.median(gapped, five_then_3),
        "median_srcgap_ref": ta.median(gapped, 3),
    }


def __test_dynamic_length_rebuild__(runner):
    """A length change rebuilds the rolling window instead of keeping a stale one"""
    from datetime import datetime, UTC
    from pynecore.types.ohlcv import OHLCV

    base_ts = int(datetime.fromisoformat("2025-01-01T00:00:00").replace(tzinfo=UTC).timestamp())
    rows = []
    for bar in range(24):
        price = float(bar + 1)  # a strict ramp: every window has a distinct median
        rows.append(OHLCV(timestamp=base_ts + bar * 1800, open=price, high=price,
                          low=price, close=price, volume=10.0))

    def is_na(value):
        return isinstance(value, NA) or value != value

    # (dynamic, reference, first bar the two must agree on)
    pairs = (("rank_na", "rank_ref", 5), ("lin_na", "lin_ref", 5), ("median_na", "median_ref", 5),
             ("rank_gap", "rank_ref", 9), ("median_gap", "median_ref", 9),
             ("median_shrink", "shrink_ref", 6),
             ("rank_grow", "rank_ref5", 8), ("lin_grow", "lin_ref5", 8),
             ("median_grow", "median_ref5", 8),
             ("rank_one", "rank_ref5", 13), ("lin_one", "lin_ref5", 13),
             ("median_one", "median_ref5", 13),
             ("median_srcgap", "median_srcgap_ref", 8))
    compared = 0
    for i, (_candle, plot) in enumerate(runner(iter(rows)).run_iter()):
        for dyn_key, ref_key, first_bar in pairs:
            if i < first_bar:
                continue
            dyn, ref = plot[dyn_key], plot[ref_key]
            assert not is_na(dyn), f"bar {i}: {dyn_key} is na, the window was not rebuilt"
            assert dyn == ref, f"bar {i}: {dyn_key} is {dyn}, expected {ref_key} = {ref}"
            compared += 1

    assert compared > 200, f"too few comparisons: {compared}"
