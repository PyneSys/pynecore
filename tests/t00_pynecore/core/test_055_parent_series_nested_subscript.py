"""
@pyne
"""
from pynecore import Series
from pynecore.lib import script, close, high, low, na, plot


@script.indicator("Parent Series Nested Subscript", shorttitle="PSNS")
def main():
    # A user-declared parent-scope ``Series[float]`` that is history-indexed
    # ONLY from within a nested function must keep its series buffer. The
    # UnusedSeriesDetector used to track subscripts per scope only, so a parent
    # series indexed just in a child scope looked "unused", got demoted to a
    # plain scalar, and the nested ``s[k]`` read crashed with
    # ``TypeError: 'float' object is not subscriptable`` (wild BigBeluga
    # "Comprehensive Trading Toolkit" ``rsi[lookbackRight]`` inside a UDF).
    s: Series[float] = (high + low) / 2.0
    # ``oracle`` is indexed directly in main, so it is always kept — the ground
    # truth the scope-chain-resolved nested read must reproduce.
    oracle: Series[float] = (high + low) / 2.0

    def read_s(offset: int) -> float:
        return s[offset]

    plot(read_s(2), "nested")
    plot(oracle[2], "inline")
    plot(close, "close")  # anchor a builtin so the run has a stable baseline


def __test_parent_series_nested_subscript__(csv_reader, runner, log):
    """A parent-scope user series indexed only inside a nested function reads
    the parent's history through the scope chain (wild BigBeluga fix)."""
    bars = 0
    checked = 0
    with csv_reader('series_if_for.csv', subdir="data") as cr:
        for i, (_candle, pv) in enumerate(runner(cr).run_iter()):
            nested = pv.get('nested')
            inline = pv.get('inline')
            nested_na = nested is None or na(nested)
            inline_na = inline is None or na(inline)
            assert nested_na == inline_na, \
                f"bar {i}: nested s[2]={nested} != inline oracle[2]={inline}"
            if not nested_na:
                assert nested == inline, \
                    f"bar {i}: nested s[2]={nested} != inline oracle[2]={inline}"
                checked += 1
            bars += 1

    assert bars > 10, f"expected the data to drive more than 10 bars, got {bars}"
    assert checked > 5, f"expected more than 5 non-na comparisons, got {checked}"
    log.info("parent series nested subscript matched inline over %d bars", bars)
