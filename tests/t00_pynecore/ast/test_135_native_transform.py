"""The native transform (``rust/``) against the Python pipeline it ports.

The native wheel gate (``PYNE_REQUIRE_NATIVE_TRANSFORM=1``) requires the extension
and every port to be current. A source install without the extension skips the
native checks.
"""
import ast
import os
import sys
from pathlib import Path

import pytest

from pynecore.core import import_hook
from pynecore.transformers import native, pipeline

#: A module exercising every branch of the ported steps
__test_helper_SOURCE = '''"""
@pyne
"""
from pynecore import lib
from pynecore.core.pine_udt import udt
from pynecore.lib import close, hl2, na, plot, request, syminfo, timeframe
from pynecore.types import Matrix, Series


@udt
class Flag:
    f: bool = na(bool)


@udt
class Chain:
    value: float = 0.0
    next: 'Chain' = lib._na_none
    top: Flag = lib._na_none


def smoothed(source: float = hl2, length: int = 2):
    """Averaged"""
    from pynecore.lib import ta
    return ta.sma(source, length)


def helper(x):
    from pynecore.lib import ta
    from pynecore.lib import ta
    from pynecore.lib.math import abs as absolute

    def inner():
        from pynecore.lib import math
        return math.max(1, 2)

    return ta.sma(x, 2) + absolute(x) + inner()


def htf(tf, symbol=syminfo.tickerid):
    return request.security(symbol, tf, close)


def htf_trend(tf):
    return htf(tf) - htf(timeframe.period)


async def ignored():
    from pynecore.lib import ta
    return ta


def label(value=lib._na_none) -> str:
    return value


def pick(v=lib._na_none):
    return Flag() if lib.bar_index < 0 else v


def bag(v=lib._na_none) -> list[float]:
    return v


def put(a, v):
    lib.array.push(a, v)


def pair():
    return close > hl2, close


def main():
    from pynecore.lib import ta as indicators
    a: list[float] = lib.array.new_float(0)
    lib.array.push(a, 2)
    lib.array.push(a, -3)
    n = lib.bar_index
    lib.array.set(a, 0, n)
    lib.array.concat(a, lib.array.from_items(5, 6))
    m: Matrix[float] = lib.matrix.new(1, 1, 0.0)
    lib.matrix.concat(m, lib.matrix.new(1, 1, 3))
    s: str = lib._na_none
    x: float = lib._na_none
    p: Flag = lib._na_none
    b: list[float] = lib._na_none
    put(lib.array.new_float(0), 1)
    put(lib.array.new_int(0), 1)
    plot(close, title=lib._na_none)
    t = label()
    q = pick()
    r = bag()
    trend = htf_trend("60") + htf_trend(tf="D") + htf("W", symbol=syminfo.prefix)
    rising = request.security(syminfo.tickerid, "D", close > hl2)
    above, level = request.security(syminfo.tickerid, "W", pair())
    flat: Series[float] = close
    prev: Series[float] = close
    shift = prev[1]
    return helper(close) + indicators.ema(close, 3) + smoothed() + Flag().f + trend
'''

#: A module the type pass stamps every per-context and per-instance attribute on: an
#: overload pin that differs per context, an int put into a float array in one
#: context and an int array in another, a window length const in one and series in
#: the other
__test_helper_STAMPS = '''"""
@pyne
"""
from pynecore import lib
from pynecore.lib import script, ta, low, bar_index
from pynecore.core.overload import overload


@overload
def g(x: int) -> str:
    return 'int-impl'


@overload
def g(x: float) -> str:
    return 'float-impl'


def wrapper(v):
    return g(v)


def put(a, v):
    lib.array.push(a, v)


def lowest(n):
    return ta.lowest(low, n)


@script.indicator(title="stamps")
def main():
    wrapper(1)
    wrapper(2.0)
    put(lib.array.new_float(0), 1)
    put(lib.array.new_int(0), 1)
    s = 8 if bar_index >= 0 else 9
    a = lowest(8)
    b = lowest(s)
'''

#: A library whose exports read enum members proven from the sources of a package that
#: must not execute, through a relative and an absolute import, and a per-bar scalar
#: retained in its library slot
__test_helper_LIBRARY = '''"""
@pyne
"""
from pynecore.lib import script, high, low, math
from pynecore.core.pine_export import export
from .definitions import Mode
import capture_parity.definitions as definitions

fixed = Mode.fixed


@script.library("parity")
def main():
    spread = math.abs(high - low)
    hexadecimal = definitions.Mode.hexadecimal

    @export
    def get_mode() -> str:
        return fixed

    @export
    def get_spread() -> float:
        return spread + 1 if hexadecimal == fixed else spread
'''

#: A library whose ``main`` defines its export surface
__test_helper_EXPORTS = '''"""
@pyne
"""
from pynecore.lib import script, close
from pynecore.core.pine_export import export


@script.library("exports")
def main():
    """Exports"""
    @export
    def last() -> float:
        return close

    @export
    def twice(x: float) -> float:
        return x * 2
'''

#: The lowering steps the helper modules exercise natively, each changing the tree
__test_helper_LOWERING_STEPS = ('SecurityDefault', 'ExportOnce', 'UnusedSeriesDetector')

#: The analysis steps after which the tree carries what the type pass stamped
__test_helper_TYPED_STEPS = ('PineType', 'ImplicitConversion', 'SecuritySlice')

_TESTS = Path(__file__).resolve().parents[2]
_LIB = Path(import_hook.__file__).resolve().parent.parent / 'lib'


def __test_helper_require_native():
    if not native.available():
        if os.environ.get('PYNE_REQUIRE_NATIVE_TRANSFORM'):
            pytest.fail('the native transform extension is not installed')
        pytest.skip('native transform extension not built')


def __test_helper_inputs() -> list[Path]:
    """Pyne modules of the test suite and the lib, for the parity checks."""
    paths = sorted((_TESTS / 't00_pynecore' / 'ast').glob('test_*.py'))
    paths += sorted(path for path in _LIB.rglob('*.py') if '@pyne' in path.read_text()[:400])
    return paths


def __test_native_ports_are_current__():
    """Every native step was ported from the Python source as it is now."""
    __test_helper_require_native()
    from pynecore.transformers import _native

    assert native.current_steps() == frozenset(_native.steps())


def __test_native_parse_matches_cpython__():
    """The native parser builds CPython's tree, locations included."""
    __test_helper_require_native()
    from pynecore.transformers import _native

    sources = [__test_helper_SOURCE] + [path.read_text() for path in __test_helper_inputs()]
    for source in sources:
        try:
            tree = _native.parse(source)
        except _native.Unsupported:
            continue
        assert (ast.dump(tree, include_attributes=True)
                == ast.dump(ast.parse(source), include_attributes=True))


def __test_native_dump_matches_cpython__():
    """The native ``ast.dump`` writes what CPython writes."""
    __test_helper_require_native()
    from pynecore.transformers import _native

    for path in __test_helper_inputs():
        tree = ast.parse(path.read_text())
        assert _native.dump(tree, True) == ast.dump(tree, include_attributes=True), path


@pytest.mark.parametrize('source', ['helper', 'suite'])
def __test_native_steps_match_python__(source, monkeypatch):
    """Under the verify backend the whole pipeline runs, every native step checked."""
    __test_helper_require_native()
    monkeypatch.setenv('PYNE_AST_BACKEND', 'verify')
    if source == 'helper':
        items = [(Path('native_parity.py'), __test_helper_SOURCE)]
    else:
        items = [(path, path.read_text()) for path in __test_helper_inputs()]
    for path, text in items:
        tree = ast.parse(text)
        user, mode = import_hook._module_mode(tree)
        if not user:
            continue
        analysed = import_hook._analyse_tree(tree, text, path.resolve(), mode)
        import_hook._lower_tree(analysed, path.resolve(), mode)


def __test_native_export_capture_reads_enum_sources__(tmp_path, monkeypatch):
    """A library proving enum members from imported sources runs the native
    ExportCapture, which reads the same files as Python and stamps the same deps."""
    __test_helper_require_native()
    package = tmp_path / 'capture_parity'
    package.mkdir()
    (package / '__init__.py').write_text('raise RuntimeError("must not execute")\n')
    definitions = package / 'definitions.py'
    definitions.write_text("from enum import StrEnum\n\n\nclass Mode(StrEnum):\n"
                           "    fixed = 'FIXED'\n    hexadecimal = 'HEX'\n")
    path = package / 'library.py'
    path.write_text(__test_helper_LIBRARY)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv('PYNE_AST_BACKEND', 'verify')
    answers = []
    run = native.run

    def recording(names, tree, ctx):
        answer = run(names, tree, ctx)
        if names == ['ExportCapture']:
            answers.append(answer)
        return answer

    monkeypatch.setattr(native, 'run', recording)
    tree = ast.parse(__test_helper_LIBRARY)
    mode = import_hook._module_mode(tree)[1]
    analysed = import_hook._analyse_tree(tree, __test_helper_LIBRARY, path, mode)
    import_hook._lower_tree(analysed, path, mode)
    assert len(answers) == 1 and isinstance(answers[0], ast.Module)
    stat = definitions.stat()
    assert answers[0]._export_capture_deps == (
        (str(definitions.resolve()), stat.st_mtime_ns, stat.st_size),)
    assert 'capture_parity' not in sys.modules


def __test_helper_extras(tree: ast.AST) -> list[tuple[str, object]]:
    """Every value stamped on a tree as an attribute of its own, containers and the
    values inside them included, each node visited once, in a stable order. The
    operator and context objects are left out, as ``native.describe`` leaves them:
    one instance stands for every occurrence, so what is stamped on it means nothing."""
    out: list[tuple[str, object]] = []
    seen: set[int] = set()
    fieldless = (ast.expr_context, ast.operator, ast.unaryop, ast.cmpop, ast.boolop)

    def value(name: str, item: object) -> None:
        out.append((name, item))
        if isinstance(item, ast.AST):
            node(item)
        elif isinstance(item, (list, tuple)) and type(item) in (list, tuple):
            for element in item:
                value(name, element)
        elif isinstance(item, dict) and type(item) is dict:
            for key, element in item.items():
                value(name, key)
                value(name, element)

    def node(item: ast.AST) -> None:
        if id(item) in seen or isinstance(item, fieldless):
            return
        seen.add(id(item))
        for child in ast.iter_child_nodes(item):
            node(child)
        for name, item_value in sorted(vars(item).items()):
            if name not in item._fields and name not in item._attributes:
                value(name, item_value)

    node(tree)
    return out


def __test_helper_check_roundtrip(tree: ast.Module, where: str) -> set[str]:
    """Round-trip a tree through the native bridge and check what came back.

    :return: The names of the stamped attributes the tree carries.
    """
    from pynecore.transformers import _native

    back = _native.roundtrip(tree)
    assert native.describe(back) == native.describe(tree), where
    before = __test_helper_extras(tree)
    after = __test_helper_extras(back)
    assert [name for name, _ in after] == [name for name, _ in before], where
    for (name, old), (_, new) in zip(before, after):
        if isinstance(old, ast.AST) or type(old) in (list, dict):
            # Converted into the arena and built anew
            assert new is not old, (where, name)
        elif old is not None and type(old) not in (bool, int, float, str, tuple):
            # Handed through: the very object, not a copy
            assert new is old, (where, name, type(old).__name__)
    assert back._pine_types is tree._pine_types, where
    return {name for name, _ in before}


def __test_typed_tree_crosses_the_bridge__(monkeypatch):
    """
    What the type pass stamps survives the native bridge: plain values and node
    references as values, the ``PineTypeTable`` as the very same object -- after the
    type pass and after the later analysis steps, for every module of the corpus.
    """
    __test_helper_require_native()
    monkeypatch.setenv('PYNE_AST_BACKEND', 'python')
    checked: dict[str, int] = dict.fromkeys(__test_helper_TYPED_STEPS, 0)
    names: dict[str, set[str]] = {}
    run = pipeline._run

    def checking(name, ctx, fn, *args):
        tree = run(name, ctx, fn, *args)
        if name in checked:
            found = __test_helper_check_roundtrip(tree, f'{ctx.path.name} after {name}')
            names.setdefault(ctx.path.name, set()).update(found)
            checked[name] += 1
        return tree

    monkeypatch.setattr(pipeline, '_run', checking)
    items = [(Path('native_parity.py'), __test_helper_SOURCE),
             (Path('native_stamps.py'), __test_helper_STAMPS)]
    items += [(path, path.read_text()) for path in __test_helper_inputs()]
    modules = 0
    for path, text in items:
        tree = ast.parse(text)
        user, mode = import_hook._module_mode(tree)
        if not user:
            continue
        modules += 1
        import_hook._analyse_tree(tree, text, path.resolve(), mode)
    assert checked['PineType'] == checked['SecuritySlice'] == modules
    assert checked['ImplicitConversion'] > 0
    assert names['native_stamps.py'] >= {
        '_pine_types', '_pine_pin', '_pine_pins', '_pine_varying', '_pine_vector',
        '_pine_widen_site', '_pine_widens', '_pine_series_lens',
    }


def __test_helper_emission(backend: str, monkeypatch) -> str:
    """The lowered emission of the helper module under a backend."""
    monkeypatch.setenv('PYNE_AST_BACKEND', backend)
    path = Path('native_parity.py').resolve()
    tree = ast.parse(__test_helper_SOURCE)
    mode = import_hook._module_mode(tree)[1]
    analysed = import_hook._analyse_tree(tree, __test_helper_SOURCE, path, mode)
    lowered, _ = import_hook._lower_tree(analysed, path, mode)
    return ast.dump(lowered, include_attributes=True)


def __test_native_backend_emits_what_python_emits__(monkeypatch):
    """The ``native`` backend's emission is the Python backend's, text and locations."""
    __test_helper_require_native()
    assert (__test_helper_emission('native', monkeypatch)
            == __test_helper_emission('python', monkeypatch))


def __test_typed_steps_run_natively__(monkeypatch):
    """The type pass and the ported steps after it handle the helper module natively, in
    one native call, as does the batch holding the security lowering, rather than
    declining it to Python."""
    __test_helper_require_native()
    answers: dict[str, object] = {}
    run = native.run

    def recording(names, tree, ctx):
        result = run(names, tree, ctx)
        answers['+'.join(names)] = result
        return result

    monkeypatch.setattr(native, 'run', recording)
    __test_helper_emission('native', monkeypatch)
    batch = next(key for key in answers if 'ImplicitConversion' in key.split('+'))
    assert 'PineType' in batch.split('+')
    assert isinstance(answers[batch], ast.Module)
    assert isinstance(answers['+'.join(__test_helper_LOWERING_STEPS)], ast.Module)
    security = [answer for names, answer in answers.items() if 'Security' in names.split('+')]
    assert len(security) == 1 and isinstance(security[0], ast.Module)


def __test_lowering_steps_run_natively__(monkeypatch):
    """Each ported lowering step handles a helper module natively and changes it, so
    the verify backend compares a rewrite rather than an untouched tree."""
    __test_helper_require_native()
    monkeypatch.setenv('PYNE_AST_BACKEND', 'verify')
    changed: dict[str, bool] = {}
    run = native.run

    def recording(names, tree, ctx):
        before = native.describe(tree)
        result = run(names, tree, ctx)
        if names[0] in __test_helper_LOWERING_STEPS:
            assert isinstance(result, ast.Module), names
            changed[names[0]] = changed.get(names[0], False) or native.describe(result) != before
        return result

    monkeypatch.setattr(native, 'run', recording)
    for name, text in (('native_parity.py', __test_helper_SOURCE),
                       ('native_exports.py', __test_helper_EXPORTS)):
        path = Path(name).resolve()
        tree = ast.parse(text)
        mode = import_hook._module_mode(tree)[1]
        analysed = import_hook._analyse_tree(tree, text, path, mode)
        import_hook._lower_tree(analysed, path, mode)
    assert changed == dict.fromkeys(__test_helper_LOWERING_STEPS, True)


def __test_stale_port_runs_in_python__(monkeypatch):
    """A port whose Python source changed is out of use, and the result does not change."""
    __test_helper_require_native()
    expected = __test_helper_emission('python', monkeypatch)
    monkeypatch.setattr(native, '_current', None)
    monkeypatch.setattr(native, 'source_digest', lambda module: 'edited')
    assert native.current_steps() == frozenset()
    assert __test_helper_emission('native', monkeypatch) == expected


@pytest.mark.parametrize('backend', ['native', 'verify'])
def __test_native_rejection_matches_python__(backend, monkeypatch):
    """A module the Python step rejects is rejected natively too, with Python's error."""
    __test_helper_require_native()
    monkeypatch.setenv('PYNE_AST_BACKEND', backend)
    source = '"""\n@pyne\n"""\nfrom pynecore.lib import plot, close\n\n\ndef main():\n' \
             '    if close > 1:\n        plot(close)\n'
    tree = ast.parse(source)
    mode = import_hook._module_mode(tree)[1]
    with pytest.raises(SyntaxError, match='plot'):
        import_hook._analyse_tree(tree, source, Path('rejected.py').resolve(), mode)
