"""
The Pine type stamp survives the whole lowering chain.

The type travels ON the node (``node._pine_ty``), which the later passes get
for free wherever they reuse the node object — and lose wherever they build a
replacement. That loss is silent: the tree still compiles, the script still
runs, and only the overload pin and the AOT front end notice that the value
they were about to type is untyped.

The invariant that catches it is upward closure: **no unstamped expression may
contain a stamped one**. A pass that wraps typed operands in fresh plumbing
violates it the moment it forgets ``stamp_lowering``/``inherit_ty``, and it
violates it at the wrapper — which is exactly the node that needed the stamp.

The check runs over the REAL pipeline, not a mini one: the point is to notice
when a pass that nobody thought about starts building wrappers.
"""
import ast
from pathlib import Path

import pytest

import pynecore

from pynecore.core.import_hook import PyneLoader
from pynecore.transformers.pine_type_rules import (
    BOOL, FLOAT, INT, OBJECT, get_ty, stamp_lowering,
)

SCRIPT = '''"""
@pyne
"""
from pynecore import Series, Persistent
from pynecore.lib import close, high, low, ta, math, plot, input, nz


length = input.int(14, "Length")


def half(value: float) -> float:
    return value / 2


def main():
    d = length / 8
    fast: Series[float] = ta.sma(close, 5)
    total: Persistent[float] = 0.0
    total += close - low
    hot = close > high
    counted = int(nz(d)) + len("abc")
    picked = half(close) if hot else math.max(d, 1)
    rolling = [i for i in range(3)]
    a, b = close, low
    plot(fast + total + counted + picked + a - b + rolling[0])
'''


def _transformed(source: str, path: str, monkeypatch) -> ast.Module:
    """Run the real pipeline and return the tree it hands to the compiler.

    ``fix_locations`` is the last pass of the transform, so intercepting it
    captures the finished tree without the loader having to give one up. The
    path matters: the loader picks the script or the lib profile from whether
    it points inside the pynecore package.
    """
    import pynecore.transformers.locations as locations

    captured: list[ast.Module] = []
    original = locations.fix_locations

    def capture(tree):
        captured.append(tree)
        return original(tree)

    monkeypatch.setattr(locations, 'fix_locations', capture)
    PyneLoader(Path(path).stem, path).source_to_code(source, path)
    assert captured, "the pipeline never reached fix_locations"
    return captured[-1]


def unstamped_wrappers(tree: ast.AST) -> list[ast.expr]:
    """Every expression that carries no type but contains one that does."""
    broken = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.expr) or hasattr(node, '_pine_ty'):
            continue
        if any(child is not node and isinstance(child, ast.expr) and hasattr(child, '_pine_ty')
               for child in ast.walk(node)):
            broken.append(node)
    return broken


def _stamped_count(tree: ast.AST) -> int:
    """How many expressions came out typed -- a guard against a vacuous pass."""
    return sum(1 for node in ast.walk(tree)
               if isinstance(node, ast.expr) and hasattr(node, '_pine_ty'))


def _report(broken: list[ast.expr]) -> str:
    """Name the offending wrappers the way a fix needs them."""
    return '\n'.join(f'  {type(node).__name__}: {ast.unparse(node)[:100]}'
                     for node in broken)


def __test_lowering_keeps_every_type__(monkeypatch):
    """No pass of the pipeline drops a type by wrapping a typed node"""
    tree = _transformed(SCRIPT, 'pine_ty_coverage.py', monkeypatch)
    assert _stamped_count(tree) > 50, "the type pass did not run"
    broken = unstamped_wrappers(tree)
    assert not broken, (
        f"{len(broken)} lowered wrapper(s) lost their Pine type -- the pass that "
        f"builds them needs stamp_lowering()/inherit_ty():\n{_report(broken)}")


def __test_lib_modules_keep_every_type__(monkeypatch):
    """The lib's own profile of the pipeline drops no type either

    A lib module skips the fold, the truthiness and the tolerance rewrites and
    is series-compacted instead, so it exercises passes the script profile
    never reaches.
    """
    source = Path(pynecore.__file__).parent / 'lib' / '_math_stateful.py'
    tree = _transformed(source.read_text(), str(source), monkeypatch)
    assert _stamped_count(tree) > 50, "the type pass did not run"
    broken = unstamped_wrappers(tree)
    assert not broken, (
        f"{len(broken)} lowered wrapper(s) lost their Pine type in a lib "
        f"module:\n{_report(broken)}")


def __test_the_load_bearing_nodes_are_typed__(monkeypatch):
    """The lowered division, comparison and call sites carry the right types"""
    tree = _transformed(SCRIPT, 'pine_ty_nodes.py', monkeypatch)
    kinds: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            kinds.setdefault(node.func.attr, set()).add(get_ty(node))
    # ``length / 8`` is int-typed on TradingView and stays so through the
    # safe-division wrapper -- the divergence this whole pass exists for
    assert kinds['safe_div'] == {INT, FLOAT}, kinds['safe_div']
    # ``int(...)`` keeps the cast's type across the na-safe rewrite
    assert kinds['safe_int'] == {INT}, kinds['safe_int']
    # ``close > high`` is a bool whatever shape the tolerance rewrite took
    tolerance = [node for node in ast.walk(tree)
                 if isinstance(node, ast.IfExp) and get_ty(node) == BOOL]
    assert tolerance, "the tolerant comparison was not typed"


def __test_the_guard_sees_a_dropped_stamp__():
    """A wrapper built without the stamp is what the invariant reports"""
    tree = ast.parse('x = 1 + 2')
    inner = tree.body[0].value  # type: ignore[attr-defined]
    setattr(inner, '_pine_ty', INT)
    assert not unstamped_wrappers(tree)
    tree.body[0].value = ast.Call(  # type: ignore[attr-defined]
        func=ast.Name(id='wrap', ctx=ast.Load()), args=[inner], keywords=[])
    assert [type(node).__name__ for node in unstamped_wrappers(tree)] == ['Call']


@pytest.mark.parametrize("source,expected", [
    # The plumbing types itself from the operands it wraps
    ('wrap(a - b)', {'BinOp': FLOAT}),
    ('wrap(a is not None and b is not None)', {'BoolOp': BOOL, 'Compare': BOOL}),
    ('wrap((t := a))', {'NamedExpr': FLOAT}),
    ('wrap(a if b else a)', {'IfExp': FLOAT}),
    ('wrap(state[0].add)', {'Attribute': OBJECT, 'Subscript': OBJECT}),
])
def __test_stamp_lowering_types_the_plumbing__(source: str, expected: dict[str, str]):
    """Emitted machinery is typed mechanically from its typed operands"""
    tree = ast.parse(source, mode='eval')
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in ('a', 'b'):
            setattr(node, '_pine_ty', FLOAT)
    stamp_lowering(tree.body, OBJECT)
    for node in ast.walk(tree.body):
        want = expected.get(type(node).__name__)
        if want is not None:
            assert get_ty(node) == want, ast.dump(node)


def __test_a_preserved_operand_keeps_its_own_type__():
    """``stamp_lowering`` never overwrites the stamp of what it wrapped"""
    tree = ast.parse('f(d)', mode='eval')
    argument = tree.body.args[0]  # type: ignore[attr-defined]
    setattr(argument, '_pine_ty', INT)
    stamp_lowering(tree.body, FLOAT)
    assert get_ty(argument) == INT
    assert get_ty(tree.body) == FLOAT
