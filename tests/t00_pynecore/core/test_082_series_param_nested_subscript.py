"""
@pyne
"""
from pynecore.lib import bar_index, close, input, plot, script
from pynecore.types import Series


@script.indicator("Series Param Nested Subscript", shorttitle="SPNS")
def main(src: Series[float] = input(close, "Source")):
    # A ``Series[T]`` PARAMETER of main (what an ``input(close, ...)`` source
    # compiles to) subscripted inside a nested function must read the parent's
    # history through the scope chain, exactly like a ``s: Series`` declared in
    # main's body. The ClosureVariableCollector recorded annotations only for
    # annotated ASSIGNMENTS, so a Series parameter was invisible to
    # ``_drop_series_closures`` and got value-passed into the nested function —
    # there the name was a plain float and ``src[i]`` raised
    # "'float' object is not subscriptable" (same class of defect as issue #67).
    bound = 5

    def sum_via_nested(count: int) -> float:
        total = 0.0
        for i in range(count):
            total += src[i]
        return total

    n = bound if bar_index >= bound else bar_index + 1

    inline_total = 0.0
    for i in range(n):
        inline_total += src[i]

    plot(sum_via_nested(n), "nested")
    plot(inline_total, "inline")


def __test_series_param_nested_subscript__(csv_reader, runner, log):
    """A Series input parameter stays subscriptable inside a nested function."""
    bars = 0
    with csv_reader('series_if_for.csv', subdir="data") as cr:
        for i, (_candle, pv) in enumerate(runner(cr).run_iter()):
            nested = pv.get('nested')
            inline = pv.get('inline')
            assert nested == inline, \
                f"bar {i}: nested src[i] sum={nested} != inline sum={inline}"
            bars += 1

    assert bars > 10, f"expected the data to drive more than 10 bars, got {bars}"
    log.info("Series parameter subscript stays consistent with inline over %d bars", bars)
