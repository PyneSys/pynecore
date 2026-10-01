"""
The type pass picks the machine of a window call from the qualifier of its length.

``ta.highest``/``ta.lowest``/``ta.highestbars``/``ta.lowestbars``/``ta.linreg`` run
a ``length + 1`` ring for a const, input or simple length and a forward-filled
history for a series one -- by the QUALIFIER, not the value (``pine_qualifier``
holds the measured rules). Hand-written Pyne code has no compiler in front of it,
so the type pass derives the qualifier from the code itself and the isolation
pass passes ``_series_length`` to the call.
"""
import ast

import pytest

from pynecore.transformers.function_isolation import FunctionIsolationTransformer
from pynecore.transformers.import_normalizer import ImportNormalizerTransformer
from pynecore.transformers.persistent import PersistentTransformer
from pynecore.transformers.pine_qualifier import get_series_len, get_series_lens
from pynecore.transformers.pine_type_infer import infer_module
from pynecore.transformers.pine_type_transformer import PineTypeTransformer
from pynecore.transformers.series import SeriesTransformer
from pynecore.transformers.slot_layout import ModuleLayout, apply_layout

#: A script entry; ``{body}`` is indented into ``main``
SCRIPT = '''
from pynecore.lib import script, ta, input, syminfo, low, bar_index, na, math
from pynecore.types import Persistent


@script.indicator(title="q")
def main(inp=input.int(8)):
{body}
'''


def _script(body: str) -> str:
    """Wrap statements into the script entry."""
    return SCRIPT.format(body='\n'.join('    ' + line for line in body.strip().splitlines()))


def _parse(source: str) -> ast.Module:
    """Parse a script and normalize its imports into the ``lib.*`` chains the type pass reads."""
    return ImportNormalizerTransformer().visit(ast.parse(source))


def _verdicts(body: str, qualify_windows: bool = True) -> dict[str, bool | None]:
    """
    Infer a script and return the verdict of each window call by its target name.

    :param body: The statements of ``main``; every window call is assigned to a name
    :param qualify_windows: Passed to the type pass
    :return: target name -> True (series machine), False (ring) or None (unstamped)
    """
    tree = _parse(_script(body))
    infer_module(tree, 'test', qualify_windows=qualify_windows)
    out: dict[str, bool | None] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = get_series_len(node.value)
    return out


def _lowered(source: str) -> str:
    """Run the slot mini pipeline with the qualifying type pass, return the source."""
    tree = _parse(source)
    layout = ModuleLayout()
    tree = PineTypeTransformer(None, qualify_windows=True).visit(tree)
    tree = SeriesTransformer(layout).visit(tree)
    tree = PersistentTransformer(layout).visit(tree)
    tree = FunctionIsolationTransformer(layout).visit(tree)
    tree = apply_layout(tree, layout)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


# The measured rules (probes ``qual``/``qual2``, BINANCE:BTCUSDT 30m)
@pytest.mark.parametrize("preamble,length,series", [
    # Not series: a literal, an input, the symbol, a never- or unconditionally
    # reassigned variable
    ('', '8', False),
    ('', 'inp', False),
    ('n = 8 if syminfo.mintick > 0 else 9', 'n', False),
    ('n = 8', 'n', False),
    ('n = 8\nn = 9', 'n', False),
    ('n = math.max(8, 2)', 'n', False),
    # Series: anything reading the bar clock, even when its value never changes
    ('n = 8 if bar_index >= 0 else 9', 'n', True),
    ('n = math.max(8, 0 * bar_index)', 'n', True),
    ('n = inp + 0 * bar_index', 'n', True),
    # ... a reassignment under a series condition
    ('n = 8\nif bar_index < 0:\n    n = 9', 'n', True),
    # ... and a ``var`` reassigned anywhere, even under a never-true condition
    ('n: Persistent[int] = 8\nif bar_index < 0:\n    n = 9', 'n', True),
])
def __test_length_qualifier_selects_the_machine__(preamble: str, length: str, series: bool):
    """Each length form gets the machine TradingView runs for it"""
    verdicts = _verdicts(f'{preamble}\nr = ta.lowest(low, {length})')
    assert verdicts['r'] is series


@pytest.mark.parametrize("call", [
    'ta.highest(low, {n})', 'ta.lowest(low, {n})', 'ta.highestbars(low, {n})',
    'ta.lowestbars(low, {n})', 'ta.highest({n})', 'ta.linreg(low, {n}, 0)',
    'ta.linreg(low, length={n}, offset=0)',
])
def __test_every_window_call_reads_its_length__(call: str):
    """The length is found in every spelling of every window call"""
    verdicts = _verdicts(f"s = 8 if bar_index >= 0 else 9\n"
                         f"a = {call.format(n='8')}\n"
                         f"b = {call.format(n='s')}")
    assert verdicts['a'] is False
    assert verdicts['b'] is True


@pytest.mark.parametrize("guard,series", [
    ('bar_index % 2 == 0', True),
    ('syminfo.mintick > 0', False),
])
@pytest.mark.parametrize("early", ['return 8', 'return'])
def __test_a_return_takes_the_condition_it_stands_under__(guard: str, early: str, series: bool):
    """Constant returns chosen by a series test give a series result"""
    verdicts = _verdicts(f'''
def f():
    if {guard}:
        {early}
    return 9
r = ta.lowest(low, f())
''')
    assert verdicts['r'] is series


def __test_a_raise_travels_a_chain_of_any_length__():
    """A series reassignment reaches a read through every link of an assignment chain"""
    verdicts = _verdicts('''
a = inp
b = inp
c = inp
d = inp
e = inp
r = ta.lowest(low, a)
a = b
b = c
c = d
d = e
if bar_index > 0:
    e = 9
''')
    assert verdicts['r'] is True


def __test_an_explicit_keyword_is_left_alone__():
    """A call that names the machine itself keeps it"""
    assert _verdicts('r = ta.lowest(low, 8, _series_length=True)')['r'] is None


def __test_lib_modules_are_not_qualified__():
    """Without ``qualify_windows`` (a lib module) no call is stamped"""
    assert _verdicts('r = ta.lowest(low, 8 if bar_index >= 0 else 9)',
                     qualify_windows=False)['r'] is None


def __test_a_helper_takes_the_qualifier_of_its_argument__():
    """
    TradingView specializes a function per call: the same body runs the ring for
    a const argument and the history for a series one, so the window site inside
    carries one verdict per context.
    """
    tree = _parse(_script('''
def f(n):
    return ta.lowest(low, n)
s = 8 if bar_index >= 0 else 9
a = f(8)
b = f(s)
'''))
    infer_module(tree, 'test', qualify_windows=True)
    window = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute) and node.func.attr == 'lowest')
    assert get_series_len(window) is None
    assert sorted(get_series_lens(window).values()) == [False, True]


def __test_one_context_helper_gets_a_constant__():
    """A helper only ever called with a series length gets a plain verdict"""
    tree = _parse(_script('''
def f(n):
    return ta.lowest(low, n)
a = f(8 if bar_index >= 0 else 9)
'''))
    infer_module(tree, 'test', qualify_windows=True)
    window = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute) and node.func.attr == 'lowest')
    assert get_series_len(window) is True
    assert get_series_lens(window) is None


def __test_the_keyword_reaches_the_call__():
    """
    The isolation pass passes the verdict. A site the contexts disagree on reads it
    from its instance vector, and each call of the helper hands over its own.
    """
    source = _lowered(_script('''
def f(n):
    return ta.lowest(low, n)
s = 8 if bar_index >= 0 else 9
a = ta.lowest(low, 8)
b = ta.lowest(low, s)
c = f(8)
d = f(s)
'''))
    lines = {line.strip().split(' ', 1)[0]: line for line in source.splitlines()
             if line.strip().startswith(('return', 'a =', 'b =', 'c =', 'd ='))}
    assert '_series_length=__state__[0][0]' in lines['return']
    assert '_series_length' not in lines['a']
    assert '_series_length=True' in lines['b']
    assert '(False,)' in lines['c']
    assert '(True,)' in lines['d']
