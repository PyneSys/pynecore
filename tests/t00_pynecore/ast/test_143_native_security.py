"""
The native ``Security`` step (``rust/crates/pyne-transform/src/steps/security/``) against
``transformers/security.py``, and the native ``SecuritySlice`` and ``FinalizeCloneDefaults``
steps (``rust/crates/pyne-transform/src/steps/security_slice/``) against
``transformers/security_slice.py``.

The step runs over modules reaching its rewrite (top-block, inline and deferred signals,
hoisted bindings, tuple and lower-timeframe reads, nested calls), the re-run that keeps
a deferred producer in the top block, the cross-function lift, the dependency analysis
and the groups, and over every security source of the security test modules; the
results are compared by ``native.describe``, a rejection by its verdict. The slice steps
run over the security sources of the slice suites and the ``@pyne`` modules of the
security runtime suite, typed (after every earlier analysis step) and untyped (after the
security split alone), with the slicing on and off.
"""
import ast
from pathlib import Path

import pytest

from pynecore.core import import_hook
from pynecore.transformers import native
from pynecore.transformers.pipeline import ANALYSIS, PipelineContext
from pynecore.transformers.security import SecurityTransformer
from pynecore.transformers.security_slice import (
    CLONE_PREFIX, SecuritySliceTransformer, finalize_clone_defaults,
)

from .test_137_native_transform import __test_helper_require_native

#: A module reaching every part of the step
__test_helper_MODULE = '''
@lib.script.indicator("security parity")
def main(tf=lib.input.timeframe("D"), length=lib.input.int(5)):
    pinned = "240"
    symbol = lib.ticker.heikinashi(lib.syminfo.tickerid)
    a = lib.request.security(symbol, pinned, lib.ta.sma(lib.close, length))
    b = lib.request.security(symbol, pinned, lib.ta.sma(lib.close, length))
    c = lib.request.security(lib.syminfo.tickerid, "D", lib.close[1],
                             lookahead=lib.barmerge.lookahead_on)
    gated = lib.na
    if c > 0:
        gated = lib.request.security(lib.syminfo.tickerid, "180", lib.close)
    o, h = lib.request.security(lib.syminfo.tickerid, "W", (lib.open, lib.high),
                                gaps=lib.barmerge.gaps_on, currency="USD",
                                ignore_invalid_symbol=True)
    pair = lib.request.security(lib.syminfo.tickerid, "M", helper_pair())
    lo, hi = lib.request.security_lower_tf(lib.syminfo.tickerid, "1", (lib.low, lib.high))
    x, y = lib.request.security_lower_tf(lib.syminfo.tickerid, "5", helper_pair())
    v = lib.na
    if lib.bar_index > 10:
        v = lib.request.security(lib.syminfo.tickerid, tf, lib.close)
    w = lib.request.security(lib.syminfo.tickerid, "W", v + lib.close)
    nested = lib.request.security(lib.syminfo.tickerid, "D",
                                  lib.request.security(lib.syminfo.tickerid, "W", lib.hl2))
    store = lib.array.new_float(0)
    for i in range(3):
        lib.array.push(store, lib.request.security(lib.syminfo.tickerid, "60", lib.volume))
    late = lib.request.security(lib.syminfo.tickerid, "120", lib.array.get(store, 0))
    lifted = wrapper(tf)
    flag = lib.na
    if lib.bar_index > 1:
        flag = lib.request.security(lib.syminfo.tickerid, "D" if length > 3 else "W",
                                    lib.close, lookahead=lib.barmerge.lookahead_on
                                    if length > 4 else lib.barmerge.lookahead_off)
    return a + b + c + gated + o + h + pair + v + w + nested + late + lifted + flag


def helper_pair():
    return lib.close, lib.open


def wrapper(tf):
    k = 2
    return lib.request.security(lib.syminfo.tickerid, tf, lib.close * k)


class Holder:
    def method(self):
        return lib.request.security(lib.syminfo.tickerid, "D", lib.close)
'''

#: A module whose analysis falls back on an unmodelled construct, with a record class
__test_helper_FALLBACK = '''
@udt
class Point:
    x: float = 0.0
    y: float = field(default_factory=lambda: lib.na)


@lib.script.strategy("fallback")
def main():
    a = lib.request.security(lib.syminfo.tickerid, "D", lib.close)
    try:
        b = lib.request.security(lib.syminfo.tickerid, "W", a + 1)
    except ValueError:
        b = 0
    with ctx():
        c = [lib.request.security(lib.syminfo.tickerid, "M", lib.open) for _ in range(2)]
    while lib.bar_index < 0:
        if a > b:
            break
        d = lib.request.security(lib.syminfo.tickerid, "60", b)
    match a:
        case 1 if b > 0:
            e = lib.request.security(lib.syminfo.tickerid, "120", a)
        case _:
            pass
    return a, b, c
'''

#: The test modules whose security sources the step is also compared over
__test_helper_SUITES = (
    'test_080_security_transformer.py', 'test_082_security_slice.py',
    'test_083_security_closed_shift_check.py', 'test_113_typing_coverage.py',
    'test_115_tuples.py',
)


def __test_helper_sources() -> list[str]:
    """Every string constant of the security test modules that calls
    ``request.security[_lower_tf]`` and parses as a module."""
    here = Path(__file__).resolve().parent
    found = []
    for name in __test_helper_SUITES:
        for node in ast.walk(ast.parse((here / name).read_text())):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and 'request.security' in node.value):
                continue
            try:
                ast.parse(node.value)
            except SyntaxError:
                continue
            found.append(node.value)
    return found


def __test_helper_step(source: str, path: str | None = None):
    """The module after the Python step and after the native step: each a module, the
    native one None when it declines, and ``'rejected'`` when a step rejects it."""
    from pynecore.transformers import _native

    tree = ast.parse(source)
    if path is not None:
        tree._module_file_path = path  # type: ignore[attr-defined]
    context = {'path': path or 'security_parity.py', 'pyne_mode': None, 'source': source,
               'user_code': True, 'host': native._Host()}
    try:
        native_tree = _native.run(tree, ['Security'], context)
    except _native.Unsupported:
        native_tree = None
    except _native.Rejected:
        native_tree = 'rejected'
    try:
        python_tree = SecurityTransformer().visit(tree)
    except SyntaxError:
        python_tree = 'rejected'
    return python_tree, native_tree


@pytest.mark.parametrize('name', ['module', 'fallback'])
def __test_native_security_step_matches_python__(name):
    """The native step lowers, analyses and groups exactly what the Python step does"""
    __test_helper_require_native()
    source = {'module': __test_helper_MODULE, 'fallback': __test_helper_FALLBACK}[name]
    path = str(Path('/srv/scripts/security parity é.py'))
    python_tree, native_tree = __test_helper_step(source, path)
    assert isinstance(native_tree, ast.Module), 'the native step declined the module'
    assert isinstance(python_tree, ast.Module)
    assert native.describe(native_tree) == native.describe(python_tree)
    contexts = python_tree.body[-1]
    assert isinstance(contexts, ast.Assign)
    keys = [key.value for key in contexts.value.keys]  # type: ignore[attr-defined]
    assert keys and all(key.startswith('sec·') for key in keys)


def __test_native_security_matches_python_over_the_security_tests__():
    """Every security source of the security test modules lowers natively to Python's
    tree, or is rejected by both"""
    __test_helper_require_native()
    sources = __test_helper_sources()
    assert len(sources) > 150
    for source in sources:
        python_tree, native_tree = __test_helper_step(source)
        assert native_tree is not None, source
        if python_tree == 'rejected' or native_tree == 'rejected':
            assert python_tree == native_tree, source
            continue
        assert native.describe(native_tree) == native.describe(python_tree), source


def __test_native_security_rejects_strategy_state__():
    """A strategy state read in a security expression is rejected natively too"""
    __test_helper_require_native()
    source = ('def main():\n'
              '    return lib.request.security(lib.syminfo.tickerid, "D",'
              ' lib.strategy.position_size)\n')
    assert __test_helper_step(source) == ('rejected', 'rejected')


#: The test modules whose security sources the slice step is compared over, besides the
#: ``@pyne`` modules of the security runtime suite
__test_helper_SLICE_SUITES = (
    'test_080_security_transformer.py', 'test_082_security_slice.py',
    'test_083_security_closed_shift_check.py',
)


def __test_helper_slice_sources() -> list[tuple[str, str]]:
    """``(name, source)`` of every security source of the slice suites, and of every
    ``@pyne`` module of the security runtime suite."""
    here = Path(__file__).resolve().parent
    found = []
    for name in __test_helper_SLICE_SUITES:
        for node in ast.walk(ast.parse((here / name).read_text())):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and 'request.security' in node.value):
                continue
            try:
                ast.parse(node.value)
            except SyntaxError:
                continue
            found.append((f'{name}:{node.lineno}', node.value))
    for path in sorted((here.parents[1] / 't01_lib' / 't40_security').glob('test_*.py')):
        text = path.read_text()
        if '@pyne' in text[:400] and 'request.security' in text:
            found.append((path.name, text))
    return found


def __test_helper_before_slice(source: str, name: str, typed: bool):
    """The module as SecuritySlice meets it, and its pipeline context: after every
    earlier analysis step (typed), or after the security split alone (untyped)."""
    tree = ast.parse(source)
    mode = import_hook._module_mode(tree)[1]
    ctx = PipelineContext(path=Path(name.split(':')[0]).resolve(), pyne_mode=mode,
                          source=source)
    if not typed:
        return SecurityTransformer().visit(tree), ctx
    for step in ANALYSIS:
        if step.name == 'SecuritySlice':
            break
        tree = step.run(tree, ctx)
    return tree, ctx


@pytest.mark.parametrize('typed', [True, False])
@pytest.mark.parametrize('disabled', [False, True])
def __test_native_security_slice_matches_python__(typed, disabled, monkeypatch):
    """The native SecuritySlice emits the clones, forces the guarded writes and clears
    ``closed_shift`` exactly as the Python step does, with the slicing on and off, and
    the native FinalizeCloneDefaults makes the clone defaults inert as Python does"""
    __test_helper_require_native()
    from pynecore.transformers import _native

    if disabled:
        monkeypatch.setenv('PYNE_NO_SECURITY_SLICE', '1')
    else:
        monkeypatch.delenv('PYNE_NO_SECURITY_SLICE', raising=False)
    sources = __test_helper_slice_sources()
    assert len(sources) > 300
    compared = cloned = forced = finalized = 0
    for name, source in sources:
        try:
            tree, ctx = __test_helper_before_slice(source, name, typed)
        except SyntaxError:
            continue
        context = native._context(ctx)
        assert context['security_slice_disabled'] is disabled
        native_tree = _native.run(tree, ['SecuritySlice'], context)
        python_tree = SecuritySliceTransformer().visit(tree)
        assert native.describe(native_tree) == native.describe(python_tree), name
        compared += 1
        clones = [node for node in python_tree.body if isinstance(node, ast.FunctionDef)
                  and node.name.startswith(CLONE_PREFIX)]
        cloned += bool(clones)
        forced += any(isinstance(stmt, ast.FunctionDef) for clone in clones for stmt in clone.body)
        native_final = _native.run(python_tree, ['FinalizeCloneDefaults'], context)
        finalize_clone_defaults(python_tree)
        assert native.describe(native_final) == native.describe(python_tree), name
        finalized += any(clone.args.defaults for clone in clones)
    # Switched off, only the contexts whose write is guarded get a clone, a forced one
    assert compared > 250 and cloned > (10 if disabled else 50) and forced > 3
    assert finalized > 0 or not typed
