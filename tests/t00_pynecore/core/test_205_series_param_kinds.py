"""
@pyne

Every parameter kind keeps its ``Series`` history, not only the plain positional one.

A keyword-only (``*, x``) and a positional-only (``x, /``) parameter annotated
``Series[float]`` read ``x[1]`` from the parameter's own history, as a plain one
does; so does one whose qualifier marker the ``SeriesQualifier`` step moved under
``Series``. A ``*args`` name is a local of its function: it hides a series of the
same name in the enclosing scope. Pyne Edge forbids these parameter kinds, so this
script is plain ``@pyne``.
"""
from pynecore.lib import close, script
from pynecore.types import PyneFloat, Series, Simple


@script.indicator("series parameter kinds")
def main():
    src: Series[float] = close

    def plain_prev(x: Series[PyneFloat]):
        return x[1]

    def keyword_prev(*, x: Series[PyneFloat]):
        return x[1]

    def positional_prev(x: Series[PyneFloat], /):
        return x[1]

    def marked_keyword_prev(*, x: Simple[Series[PyneFloat]]):
        return x[1]

    # The shadowing is what this checks: ``src`` here is the tuple, not the series
    # noinspection PyShadowingNames
    def count(*src):
        return len(src)

    return {
        "plain": plain_prev(close),
        "keyword": keyword_prev(x=close),
        "positional": positional_prev(close),
        "marked_keyword": marked_keyword_prev(x=close),
        "starred": count(src, src),
    }


def __test_every_parameter_kind_reads_its_history__(csv_reader, runner):
    """``x[1]`` is the previous bar's argument for every parameter kind"""
    previous = None
    with csv_reader('series_if_for.csv', subdir="data") as cr:
        for i, (candle, plot) in enumerate(runner(cr).run_iter()):
            if previous is not None:
                assert plot["plain"] == previous
                assert plot["keyword"] == previous
                assert plot["positional"] == previous
                assert plot["marked_keyword"] == previous
            assert plot["starred"] == 2
            previous = candle.close
            if i >= 5:
                break
