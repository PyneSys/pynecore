"""
Behavior tests for the lazy-context ``inline_series`` hoist.

Pine evaluates a history-referenced expression (``expr[n]``) on every bar its
statement executes, even when the ``[n]`` sits in a ternary branch or a
short-circuited ``and``/``or`` operand the bar's values skip (verified against
TradingView v6 bar-by-bar); only statements inside conditionally executed
BLOCKS keep the compressed "gap" history. The hoist pass moves lazy-position
``inline_series`` calls to temp assignments before the enclosing statement so
their buffers advance whenever the statement runs.

The mini pipeline mirrors the import hook's order: the hoist runs before the
slot transforms, so the hoisted statements are the anchorable call sites.
"""
import ast
from contextlib import contextmanager

from pynecore import lib
from pynecore.core.instance_state import _make_state
from pynecore.transformers.function_isolation import FunctionIsolationTransformer
from pynecore.transformers.inline_series_hoist import InlineSeriesHoistTransformer
from pynecore.transformers.persistent import PersistentTransformer
from pynecore.transformers.series import SeriesTransformer
from pynecore.transformers.slot_layout import ModuleLayout, apply_layout
from pynecore.types.na import NA


def _transform(source: str) -> tuple[dict, str]:
    """Run the hoist + slot mini pipeline on a source string.

    :param source: Pyne-style module source.
    :return: (exec'd module namespace, unparsed transformed source)
    """
    tree = ast.parse(source)
    tree = InlineSeriesHoistTransformer().visit(tree)
    layout = ModuleLayout()
    tree = SeriesTransformer(layout).visit(tree)
    tree = PersistentTransformer(layout).visit(tree)
    tree = FunctionIsolationTransformer(layout).visit(tree)
    tree = apply_layout(tree, layout)
    ast.fix_missing_locations(tree)
    ns: dict = {}
    exec(compile(tree, '<inline-series-hoist-test>', 'exec'), ns)  # noqa: S102
    return ns, ast.unparse(tree)


@contextmanager
def _bars():
    """Drive ``lib.bar_index`` for the duration of a test; yields a
    ``next_bar()`` callable and restores the original value afterwards."""
    old = lib.bar_index
    lib.bar_index = 0

    def next_bar():
        lib.bar_index += 1

    try:
        yield next_bar
    finally:
        lib.bar_index = old


def __test_hoist_ternary_fresh_history__():
    """ A gated ternary's inline history advances every bar: the bar after a
    skipped gate still sees the true previous-bar value (TV v6 parity) """
    ns, dump = _transform('''
from pynecore.core.series import inline_series

def main(gate, v):
    return inline_series(v * 1.0, 1) if gate else -1.0
''')
    # the call moved to a statement-level temp assignment
    assert '__hist_0__ = ' in dump
    assert 'if gate else -1.0' in dump
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    with _bars() as next_bar:
        assert isinstance(ns['main'](state, True, 10), NA)  # no history yet
        next_bar()
        assert ns['main'](state, False, 20) == -1.0         # gate skips
        next_bar()
        # previous BAR's value (20), not the last gated one (10)
        assert ns['main'](state, True, 30) == 20.0


def __test_hoist_boolop_fresh_history__():
    """ An ``and`` right operand's inline history advances every bar """
    ns, dump = _transform('''
from pynecore.core.series import inline_series

def main(gate, v):
    return gate and inline_series(v * 1.0, 1) == v - 1
''')
    assert '__hist_0__ = ' in dump
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    with _bars() as next_bar:
        ns['main'](state, True, 10)
        next_bar()
        assert ns['main'](state, False, 11) is False        # gate skips
        next_bar()
        assert ns['main'](state, True, 12) is True          # [1] == 11, fresh


def __test_if_block_keeps_gap_semantics__():
    """ Inside an ``if`` BLOCK a lazy call is hoisted only within the block,
    so skipped bars still leave a gap (compressed history, TV parity) """
    ns, dump = _transform('''
from pynecore.core.series import inline_series

def main(gate, lazy_gate, v):
    r = -1.0
    if gate:
        r = inline_series(v * 1.0, 1) if lazy_gate else -2.0
    return r
''')
    # hoisted, but INSIDE the if body (gap semantics of the block preserved)
    tree = ast.parse(dump)
    fn = next(s for s in tree.body if isinstance(s, ast.FunctionDef))
    if_stmt = next(s for s in ast.walk(fn) if isinstance(s, ast.If))
    assert any('__hist_0__' in ast.unparse(s) for s in if_stmt.body)
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    with _bars() as next_bar:
        ns['main'](state, True, True, 10)
        next_bar()
        assert ns['main'](state, False, True, 20) == -1.0   # block skipped
        next_bar()
        # gap semantics: [1] is the last EXECUTED bar's value (10), not 20
        assert ns['main'](state, True, True, 30) == 10.0


def __test_nested_lazy_inside_hoisted_args__():
    """ A lazy inline_series nested in a hoisted call's argument is hoisted
    too, definition-before-use (hoist-only dump: the slot pipeline would
    rewrite the calls to the uniform route) """
    tree = InlineSeriesHoistTransformer().visit(ast.parse('''
def main(gate, v):
    return gate and inline_series(inline_series(v, 1) if gate else v, 1)
'''))
    dump = ast.unparse(tree)
    inner = dump.index('__hist_0__ = inline_series(v, 1)')
    outer = dump.index('__hist_1__ = inline_series(__hist_0__ if gate else v, 1)')
    assert inner < outer


def __test_while_test_untouched__():
    """ A ``while`` test re-evaluates per iteration — no hoist from there """
    _, dump = _transform('''
from pynecore.core.series import inline_series

def main(gate, v):
    n = 0
    while gate and inline_series(v, 1) != v:
        n += 1
        gate = False
    return n
''')
    assert '__hist_' not in dump
    assert 'while gate and' in dump


def __test_eager_positions_untouched__():
    """ Calls already in eager positions stay in place (hoist-only dump) """
    tree = InlineSeriesHoistTransformer().visit(ast.parse('''
def main(gate, v):
    a = inline_series(v, 1)
    b = (inline_series(v, 2) if True else 0) if gate else -1.0
    return a, b
'''))
    dump = ast.unparse(tree)
    # `a`'s call is eager — not hoisted; only the lazy one moved
    assert 'a = inline_series(v, 1)' in dump
    assert '__hist_0__ = inline_series(v, 2)' in dump
