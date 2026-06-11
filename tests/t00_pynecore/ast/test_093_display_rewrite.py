"""
Tests for the dump display rewrite (named index constants in debug output).
"""
import ast

from pynecore.transformers.display_rewrite import display_dump
from pynecore.transformers.function_isolation import FunctionIsolationTransformer
from pynecore.transformers.persistent import PersistentTransformer
from pynecore.transformers.series import SeriesTransformer
from pynecore.transformers.slot_layout import ModuleLayout, apply_layout


def _display(source: str) -> str:
    """Run the slot mini pipeline and return the display dump."""
    tree = ast.parse(source)
    layout = ModuleLayout()
    tree = SeriesTransformer(layout).visit(tree)
    tree = PersistentTransformer(layout).visit(tree)
    tree = FunctionIsolationTransformer(layout).visit(tree)
    tree = apply_layout(tree, layout)
    ast.fix_missing_locations(tree)
    return display_dump(tree, layout)


def __test_display_named_indexes__():
    """ Literal slot indexes become named constants in the dump """
    out = _display('''
from pynecore import Persistent

def t1():
    p: Persistent[int] = 0
    p += 1
    return p

def main():
    q: Persistent[float] = 0.0
    q = (q := q + 1.0) * 2.0
    return t1(), q
''')
    assert '__slot·t1·p__ = 0' in out
    assert '__state__[__slot·t1·p__] += 1' in out
    assert '__state__[__slot·main·q__]' in out
    # fast-path helper index argument is renamed too
    assert '__resolve_slot__(__state__, __slot·main·t1·0__, t1)' in out
    # the walrus-write __setitem__ index as well
    assert '__setitem__(__slot·main·q__' in out
    # no bare literal state indexes remain
    assert '__state__[0]' not in out and '__state__[1]' not in out


def __test_display_scope_qualified_param__():
    """ The scope-qualified state parameter resolves to its scope's names """
    out = _display('''
from pynecore import Persistent

def main():
    p: Persistent[int] = 0
    def bump():
        return p + 1
    p = bump()
    return p
''')
    assert '__state·main__[__slot·main·p__]' in out


def __test_display_no_layout_passthrough__():
    """ A module without slots dumps unchanged (no constant block) """
    out = _display('''
def helper(x):
    return x * 2

def main(x):
    return helper(x)
''')
    assert '__slot·' not in out
    assert 'def main(x):' in out
