"""
Behavior tests for inline_series on the slot scheme.

``inline_series`` is the call PyneComp emits for series indexing of
expressions (``(a + b)[1]``); its name and signature are a fixed surface.
The function publishes a ``__pyne_bind__`` factory, so the slot transform
classifies it UNIFORM and every call site anchors an instance with its own
buffer — two ``expr[n]`` rewrites in one scope stay independent (issue #61),
and a loop site keeps one buffer per iteration.

The real ``lib.bar_index`` is advanced between calls because
``SeriesImpl.add`` consults it to convert same-bar adds into ``set``.
"""
import ast
from contextlib import contextmanager

from pynecore import lib
from pynecore.core.instance_state import _make_state
from pynecore.core.series import inline_series
from pynecore.transformers.function_isolation import FunctionIsolationTransformer
from pynecore.transformers.persistent import PersistentTransformer
from pynecore.transformers.series import SeriesTransformer
from pynecore.transformers.slot_layout import ModuleLayout, apply_layout
from pynecore.types.na import NA


def _transform(source: str) -> tuple[dict, str]:
    """Run the slot mini pipeline on a source string.

    :param source: Pyne-style module source.
    :return: (exec'd module namespace, unparsed transformed source)
    """
    tree = ast.parse(source)
    layout = ModuleLayout()
    tree = SeriesTransformer(layout).visit(tree)
    tree = PersistentTransformer(layout).visit(tree)
    tree = FunctionIsolationTransformer(layout).visit(tree)
    tree = apply_layout(tree, layout)
    ast.fix_missing_locations(tree)
    ns: dict = {}
    exec(compile(tree, '<slot-inline-series-test>', 'exec'), ns)  # noqa: S102
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


def __test_inline_series_sites_independent__():
    """ Two call sites in one scope keep independent buffers """
    ns, dump = _transform('''
from pynecore.core.series import inline_series

def main(c, o):
    a = inline_series(c, 1)
    b = inline_series(o, 1)
    return a, b
''')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    with _bars() as next_bar:
        a, b = ns['main'](state, 10.0, 1.0)
        assert isinstance(a, NA) and isinstance(b, NA)  # no previous bar yet
        next_bar()
        a, b = ns['main'](state, 20.0, 2.0)
        assert a == 10.0  # c[1] — the legacy shared buffer returned o[1] here
        assert b == 1.0   # o[1]
    # __pyne_bind__ routes the sites uniform, one anchor each
    assert '__bind_any__(__state__, 0, inline_series)' in dump
    assert '__bind_any__(__state__, 1, inline_series)' in dump


def __test_inline_series_loop_site__():
    """ A loop site keeps one buffer per iteration """
    ns, dump = _transform('''
from pynecore.core.series import inline_series

def main(v):
    out = []
    for k in range(2):
        out.append(inline_series(v + k, 1))
    return out
''')
    state = _make_state(ns['__pyne_slot_layout__']['main'])
    with _bars() as next_bar:
        first = ns['main'](state, 0.0)
        assert all(isinstance(x, NA) for x in first)
        next_bar()
        assert ns['main'](state, 10.0) == [0.0, 1.0]  # each iteration sees its own [1]
    assert '__bind_any_loop__(__chl_0__, __i__, inline_series)' in dump


def __test_inline_series_direct_call_shared__():
    """ Anchorless direct calls fall back to one shared module-lifetime
    buffer (the legacy module-global semantics) """
    with _bars() as next_bar:
        inline_series(1.0, 0)
        next_bar()
        assert inline_series(2.0, 1) == 1.0  # the shared buffer persisted


def __test_inline_series_bind_factory__():
    """ The __pyne_bind__ factory hands out independent instances """
    factory = getattr(inline_series, '__pyne_bind__')
    one, two = factory(), factory()
    with _bars() as next_bar:
        assert one(1.0, 0) == 1.0
        next_bar()
        assert one(2.0, 1) == 1.0
        assert isinstance(two(5.0, 1), NA)  # untouched by one's history
