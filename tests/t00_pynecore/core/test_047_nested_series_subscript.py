"""
@pyne
"""
from pynecore.lib import bar_index, close, plot, script


@script.indicator("Nested Series Subscript", shorttitle="NSS")
def main():
    # Issue #67: a builtin price series subscript (``close[k]``) inside a nested
    # function that is called a bar-VARYING number of times must read the GLOBAL
    # close history, exactly like the inlined equivalent. Before the fix the
    # ClosureArgumentsTransformer value-passed ``close`` into the nested function,
    # which then kept its OWN history buffer advancing once per CALL (not once
    # per bar). Since the call count varies per bar, ``close[k]`` resolved to the
    # wrong offset and the nested count diverged from the inline count.
    bound = 5

    def below_via_nested(offset: int) -> bool:
        return close[offset] < close[0]

    nested_count = 0
    k = 1
    while k <= bar_index and k <= bound:
        if below_via_nested(k):
            nested_count += 1
        k += 1

    inline_count = 0
    k = 1
    while k <= bar_index and k <= bound:
        if close[k] < close[0]:
            inline_count += 1
        k += 1

    plot(nested_count, "nested")
    plot(inline_count, "inline")


def __test_nested_series_subscript__(csv_reader, runner, log):
    """A close[k] subscript in a bar-varyingly-called nested function matches
    the inlined equivalent (issue #67)."""
    bars = 0
    with csv_reader('series_if_for.csv', subdir="data") as cr:
        for i, (_candle, pv) in enumerate(runner(cr).run_iter()):
            nested = pv.get('nested')
            inline = pv.get('inline')
            assert nested == inline, \
                f"bar {i}: nested close[k] count={nested} != inline count={inline}"
            bars += 1

    assert bars > 10, f"expected the data to drive more than 10 bars, got {bars}"
    log.info("nested close[k] subscript stays consistent with inline over %d bars", bars)
