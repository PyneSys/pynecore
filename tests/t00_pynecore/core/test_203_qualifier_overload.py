"""
@pyne edge

Overloads that differ only by a Pine qualifier select by the argument's qualifier.

Pine allows ``f(simple string s)`` next to ``f(string s)``; in Python both are
``f(s: str)``, so PyneComp marks the qualified parameter ``Simple[str]`` (or
``Const[str]``). The marker is plain ``str`` at run time; the type pass reads it
and pins each call to the implementation Pine selects: a const argument runs the
``Const`` one before the ``Simple`` one before the unmarked one, a simple argument
the ``Simple`` one before the unmarked one, a series argument the unmarked one.
"""
from pynecore.core.overload import overload
from pynecore.lib import close, input, script, string
from pynecore.types import Const, Simple


@script.indicator("qualifier overload")
def main(
        name=input.string("abc")
):
    @overload
    def tag(this: Simple[str]):
        return 1

    @overload
    def tag(this: str):
        return 2

    @overload
    def rank(this: Const[str]):
        return 10

    @overload
    def rank(this: Simple[str]):
        return 20

    @overload
    def rank(this: str):
        return 30

    @overload
    def label(this: Simple[str], size: int = 1, place: str = "top"):
        return 100 + size

    @overload
    def label(this: str, size: int = 1, place: str = "top"):
        return 200 + size

    series_text = string.tostring(close)
    return {
        "const_tag": tag("lit"),
        "simple_tag": tag(name),
        "series_tag": tag(series_text),
        "const_rank": rank("lit"),
        "simple_rank": rank(name),
        "series_rank": rank(series_text),
        "simple_label": label(name, size=2, place="bottom"),
        "series_label": label(series_text, place="bottom"),
        "keyword_receiver": label(this=series_text, size=3),
    }


def __test_the_argument_qualifier_selects_the_implementation__(csv_reader, runner):
    """Each call runs the implementation Pine selects for its argument's qualifier"""
    with csv_reader('series_if_for.csv', subdir="data") as cr:
        for i, (_candle, plot) in enumerate(runner(cr).run_iter()):
            assert plot["const_tag"] == 1
            assert plot["simple_tag"] == 1
            assert plot["series_tag"] == 2
            assert plot["const_rank"] == 10
            assert plot["simple_rank"] == 20
            assert plot["series_rank"] == 30
            assert plot["simple_label"] == 102
            assert plot["series_label"] == 201
            assert plot["keyword_receiver"] == 203
            if i >= 2:
                break


def __test_every_qualifier_site_is_pinned_by_index__():
    """Every call carries an indexed pin, and none is reported undecided"""
    import ast
    from pathlib import Path
    from pynecore.transformers.pine_type_rules import PIN_INDEX, get_pin
    from ..pine_analysis import analyse_module

    analysed = analyse_module(Path(__file__))
    assert analysed is not None
    tree, table, _ = analysed
    pins = []
    labels = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if ast.unparse(node.func) in ('tag', 'rank'):
            pins.append((ast.unparse(node.args[0]), ast.unparse(node.func), str(get_pin(node))))
        elif ast.unparse(node.func) == 'label':
            labels.append((ast.unparse(node), str(get_pin(node))))
    pins.sort()
    labels.sort()
    assert all(PIN_INDEX in pin for _, _, pin in pins), pins
    assert all(PIN_INDEX in pin for _, pin in labels), labels
    assert pins == [("'lit'", 'rank', 's=0'), ("'lit'", 'tag', 's=0'),
                    ('name', 'rank', 's=1'), ('name', 'tag', 's=0'),
                    ('series_text', 'rank', 's=2'), ('series_text', 'tag', 's=1')]
    found = [(diag.code, diag.args) for diag in table.diags if diag.code in ('E0071', 'E0072')]
    assert found == []


def __test_a_marker_is_its_payload_at_run_time__():
    """``Simple[T]`` and ``Const[T]`` are ``T`` itself, nothing wraps the value"""
    assert Simple[str] is str
    assert Const[float] is float
