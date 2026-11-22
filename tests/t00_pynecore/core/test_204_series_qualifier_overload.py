"""
@pyne edge

A qualifier marker on a parameter whose history is read selects like any other.

Pine allows ``g(simple float x) => x[1] + 100`` next to ``g(float x) => x``. The
first parameter's history is read, so its annotation is ``Series[float]``, and the
marker goes around it: PyneComp writes ``Simple[Series[float]]``, Pyne code may also
write ``Series[Simple[float]]``. The ``SeriesQualifier`` step turns the first into
the second, so the series lowering keeps the parameter's history and the type pass
reads the marker under ``Series``: a const or simple argument runs the marked
implementation, a series one the unmarked one. The element type is spelled
``PyneFloat`` here, so the IDE accepts the ``[1]`` history read.
"""
from pynecore.core.overload import overload
from pynecore.lib import close, input, script
from pynecore.types import Const, PyneFloat, Series, Simple


@script.indicator("series qualifier overload")
def main(
        level=input.float(2.5)
):
    @overload
    def outer(x: Simple[Series[PyneFloat]]):
        return x[1] + 100

    @overload
    def outer(x: float):
        return x

    @overload
    def inner(x: Series[Simple[PyneFloat]]):
        return x[1] + 100

    @overload
    def inner(x: float):
        return x

    @overload
    def rank(x: Const[Series[PyneFloat]]):
        return x[1] + 1000

    @overload
    def rank(x: Simple[Series[PyneFloat]]):
        return x[1] + 100

    @overload
    def rank(x: float):
        return x

    return {
        "const_outer": outer(2.5),
        "simple_outer": outer(level),
        "series_outer": outer(close),
        "const_inner": inner(2.5),
        "simple_inner": inner(level),
        "series_inner": inner(close),
        "const_rank": rank(2.5),
        "simple_rank": rank(level),
        "series_rank": rank(close),
    }


def __test_the_argument_qualifier_selects_the_implementation__(csv_reader, runner):
    """Each call runs the implementation Pine selects, and the marked one keeps its history"""
    with csv_reader('series_if_for.csv', subdir="data") as cr:
        for i, (candle, plot) in enumerate(runner(cr).run_iter()):
            assert plot["series_outer"] == candle.close
            assert plot["series_inner"] == candle.close
            assert plot["series_rank"] == candle.close
            if i >= 1:
                assert plot["const_outer"] == 102.5
                assert plot["simple_outer"] == 102.5
                assert plot["const_inner"] == 102.5
                assert plot["simple_inner"] == 102.5
                assert plot["const_rank"] == 1002.5
                assert plot["simple_rank"] == 102.5
            if i >= 3:
                break


def __test_every_site_is_pinned_by_index__():
    """Every call carries an indexed pin, and none is reported undecided"""
    import ast
    from pathlib import Path
    from pynecore.transformers.pine_type_rules import PIN_INDEX, get_pin
    from ..pine_analysis import analyse_module

    analysed = analyse_module(Path(__file__))
    assert analysed is not None
    tree, table, _ = analysed
    pins = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func) in ('outer', 'inner', 'rank'):
            pins.append((ast.unparse(node.func), ast.unparse(node.args[0]), str(get_pin(node))))
    pins.sort()
    assert all(PIN_INDEX in pin for _, _, pin in pins), pins
    assert pins == [
        ('inner', '2.5', 'f=0'), ('inner', 'level', 'f=0'), ('inner', 'lib.close', 'f=1'),
        ('outer', '2.5', 'f=0'), ('outer', 'level', 'f=0'), ('outer', 'lib.close', 'f=1'),
        ('rank', '2.5', 'f=0'), ('rank', 'level', 'f=1'), ('rank', 'lib.close', 'f=2'),
    ], pins
    found = [(diag.code, diag.args) for diag in table.diags if diag.code in ('E0071', 'E0072')]
    assert found == []


def __test_the_marker_goes_under_series__():
    """``Simple[Series[T]]`` and ``Const[Series[T]]`` become ``Series[Simple[T]]``, ``Series[Const[T]]``"""
    import ast
    from pynecore.transformers.series_qualifier import normalize_series_qualifiers

    tree = ast.parse(
        "def f(a: Simple[Series[float]], b: Const[Series[int]], c: Series[Simple[str]],\n"
        "      d: Simple[float], e: Series[float], *, k: types.Simple[Series[bool]]):\n"
        "    def g(x: Const[Series[float]]): pass\n")
    marked = tree.body[0].args.args[0].annotation
    normalize_series_qualifiers(tree)
    assert ast.unparse(tree.body[0].args) == (
        "a: Series[Simple[float]], b: Series[Const[int]], c: Series[Simple[str]], "
        "d: Simple[float], e: Series[float], *, k: Series[types.Simple[bool]]")
    assert ast.unparse(tree.body[0].body[0].args) == "x: Series[Const[float]]"
    assert tree.body[0].args.args[0].annotation is marked
