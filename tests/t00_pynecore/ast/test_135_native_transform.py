"""The native transform (``rust/``) against the Python pipeline it ports.

The native wheel gate (``PYNE_REQUIRE_NATIVE_TRANSFORM=1``) requires the extension
and every port to be current. A source install without the extension skips the
native checks.
"""
import ast
import functools
import os
import sys
from pathlib import Path

import pytest

from pynecore.core import import_hook
from pynecore.transformers import native, pipeline
from pynecore.transformers.function_isolation import NON_TRANSFORMABLE_FUNCTIONS
from pynecore.transformers.slot_layout import ModuleLayout

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

#: A script whose series, persistents and security contexts exercise the slot steps:
#: series parameters, declarations, writes and tuple unpacking, literal, lazy, varip and
#: na persistents, a walrus write, a parent's slots read from a nested definition, and
#: a ``closed_shift`` candidate whose lowered write indexes a list
__test_helper_SLOTS = '''"""
@pyne
"""
from pynecore import lib
from pynecore.lib import close, high, low, na
from pynecore.types import IBPersistent, Persistent, Series


def pair():
    return high, low


def counter(src: Series[float], step: int = 1):
    """Counts"""
    total: Persistent[int] = 0
    peak: Persistent[float] = -1
    seen: IBPersistent[int] = +2
    start: Persistent[float] = src
    last: Persistent[float] = na
    total += step
    if (last := src) > peak:
        peak = last
    acc: Series[float] = 0.0
    acc += src
    acc = acc * 0.5
    return total + peak + seen + start + acc[1] + src[2]


def outer():
    level: Persistent[float] = 0.0
    hist: Series[float] = close

    def inner():
        return hist[1] + level

    return inner()


@lib.script.indicator("slots")
def main():
    data = [lib.close, lib.close]
    a = lib.request.security(lib.syminfo.tickerid, "D", data[1],
                             lookahead=lib.barmerge.lookahead_on)
    data: Series[float] = lib.close
    b = lib.request.security(lib.syminfo.tickerid, "240", data[1],
                             lookahead=lib.barmerge.lookahead_on)
    hi: Series[float]
    hi, lo = pair()
    lib.max_bars_back(data, 100)
    lib.plot(a + b + counter(close) + outer() + hi[1] + lo)
'''

#: A strategy with source and value inputs, divisions, a remainder, a cast and
#: comparisons, lowered as a three-state bool module, and inlined lib wrappers: a
#: literal-decided guard, a forced argument order, an array read in a ``pine_range``
#: loop
__test_helper_STRATEGY = '''"""
@pyne
"""
from pynecore import lib, pine_range
from pynecore.lib import script, strategy, input, close, high, low, hl2


@script.strategy("parity strategy", overlay=True)
def main(src=input.source(close, "Source"), length=input.int(5, "Length"),
         level=input(hl2, "Level")):
    total = 0.0
    for i in range(length):
        total += close[i] / length
    ratio = high / low
    ratio /= 2.0
    half = int(total) % 3
    levels = lib.array.from_items(high, low)
    for k in pine_range(0, 1):
        total += lib.array.get(levels, k) + lib.math.pow(ratio, 2)
    total += lib.math.max(level, lib.math.abs(close - src)) + lib.array.size(levels)
    if ratio > 1.01 and close >= src:
        strategy.entry("long", strategy.long, limit=low)
    if ratio < 0.99 or level != close:
        strategy.entry("short", strategy.short)
    strategy.exit("x", "long", qty=1, profit=10, loss=5, trail_points=3)
    if close[half] == level:
        strategy.close_all()
'''

#: The helper modules the lowering checks run: file name, source, three-state bool
__test_helper_LOWERED = (('native_parity.py', __test_helper_SOURCE, False),
                         ('native_exports.py', __test_helper_EXPORTS, False),
                         ('native_slots.py', __test_helper_SLOTS, False),
                         ('native_strategy.py', __test_helper_STRATEGY, True))

#: The lowering passes the helper modules exercise natively, each changing the tree
__test_helper_LOWERING_STEPS = ('SecurityDefault', 'ExportOnce', 'UnusedSeriesDetector', 'Series',
                                'VerifyClosedShift', 'Persistent', 'CallInline',
                                'FunctionIsolation', 'ScriptRequirements', 'Input', 'Expression',
                                'ApplyLayout', 'FixLocations')

#: The same in sequential mode, where the rules of the phase run one by one
__test_helper_SEQUENTIAL_STEPS = ('SecurityDefault', 'ExportOnce', 'UnusedSeriesDetector',
                                  'Series', 'VerifyClosedShift', 'Persistent', 'CallInline',
                                  'FunctionIsolation', 'ScriptRequirements', 'Input',
                                  'SafeConvert', 'SafeDivision', 'FloatTolerance', 'ApplyLayout',
                                  'FixLocations')

#: The native calls of the lowering half under the native backend: all of it, in one
__test_helper_LOWERING_BATCHES = (
    'SecurityDefault+ExportOnce+UnusedSeriesDetector+Series+VerifyClosedShift+Persistent'
    '+CallInline+FunctionIsolation+ScriptRequirements+Input+Expression+FinalizeCloneDefaults'
    '+ApplyLayout+FixLocations',)

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

    def recording(names, tree, ctx, layout=None, table=False):
        answer = run(names, tree, ctx, layout, table)
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


def __test_helper_emission(backend: str, monkeypatch, name: str = 'native_parity.py',
                           text: str = __test_helper_SOURCE, na_bool: bool = False) -> str:
    """The lowered emission of a helper module under a backend."""
    monkeypatch.setenv('PYNE_AST_BACKEND', backend)
    path = Path(name).resolve()
    tree = ast.parse(text)
    mode = import_hook._module_mode(tree)[1]
    analysed = import_hook._analyse_tree(tree, text, path, mode)
    lowered, _ = import_hook._lower_tree(analysed, path, mode, na_bool=na_bool)
    return ast.dump(lowered, include_attributes=True)


@pytest.mark.parametrize('sequential', [False, True])
def __test_native_backend_emits_what_python_emits__(sequential, monkeypatch):
    """The ``native`` backend's emission is the Python backend's, text and locations."""
    __test_helper_require_native()
    monkeypatch.setattr(pipeline, '_SEQUENTIAL', sequential)
    for name, text, na_bool in __test_helper_LOWERED:
        assert (__test_helper_emission('native', monkeypatch, name, text, na_bool)
                == __test_helper_emission('python', monkeypatch, name, text, na_bool)), name


def __test_typed_steps_run_natively__(monkeypatch):
    """The type pass and the ported steps after it handle the helper module natively, in
    one native call, as does the batch holding the security lowering, rather than
    declining it to Python; the expression phase runs as one unit."""
    __test_helper_require_native()
    monkeypatch.setattr(pipeline, '_SEQUENTIAL', False)
    answers: dict[str, object] = {}
    run = native.run

    def recording(names, tree, ctx, layout=None, table=False):
        result = run(names, tree, ctx, layout, table)
        answers['+'.join(names)] = result
        return result

    monkeypatch.setattr(native, 'run', recording)
    __test_helper_emission('native', monkeypatch)
    batch = next(key for key in answers if 'ImplicitConversion' in key.split('+'))
    assert {'PineType', 'SecuritySlice'} <= set(batch.split('+'))
    assert isinstance(answers[batch], ast.Module)
    security = [answer for names, answer in answers.items() if 'Security' in names.split('+')]
    assert len(security) == 1 and isinstance(security[0], ast.Module)
    for batch in __test_helper_LOWERING_BATCHES:
        assert isinstance(answers[batch], ast.Module), batch


@pytest.mark.parametrize('sequential', [False, True])
def __test_lowering_steps_run_natively__(sequential, monkeypatch):
    """Each ported lowering step handles a helper module natively and changes it, so
    the verify backend compares a rewrite rather than an untouched tree: the phase as
    one unit, and in sequential mode each of its rules."""
    __test_helper_require_native()
    monkeypatch.setenv('PYNE_AST_BACKEND', 'verify')
    monkeypatch.setattr(pipeline, '_SEQUENTIAL', sequential)
    steps = __test_helper_SEQUENTIAL_STEPS if sequential else __test_helper_LOWERING_STEPS
    changed: dict[str, bool] = {}
    run = native.run

    def recording(names, tree, ctx, layout=None, table=False):
        before = native.describe(tree)
        result = run(names, tree, ctx, layout, table)
        if names[0] in steps:
            assert isinstance(result, ast.Module), names
            changed[names[0]] = changed.get(names[0], False) or native.describe(result) != before
        return result

    monkeypatch.setattr(native, 'run', recording)
    for name, text, na_bool in __test_helper_LOWERED:
        path = Path(name).resolve()
        tree = ast.parse(text)
        mode = import_hook._module_mode(tree)[1]
        analysed = import_hook._analyse_tree(tree, text, path, mode)
        import_hook._lower_tree(analysed, path, mode, na_bool=na_bool)
    assert changed == dict.fromkeys(steps, True)


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


#: The steps that read or fill the slot layout, in pipeline order
__test_helper_SLOT_STEPS = ('Series', 'VerifyClosedShift', 'Persistent', 'FunctionIsolation',
                            'ApplyLayout')

#: Modules for the slot steps alone, as the lowering half meets them, each reaching
#: branches the pipeline helpers do not: Series-annotated parameters of every kind and
#: a dynamic-default guard, bare and repeated declarations, shadowing, overloads, an
#: async definition; Persistent spelled every way, nonlocal stripping, walrus and tuple
#: targets; the closed-shift verdicts over hand-lowered writes; a lib module, whose
#: int persistents stay ints
__test_helper_SLOT_SOURCES = {
    'series': '''
from pynecore.types import Series, Persistent
from pynecore import Series as Other, lib
from os import path


def f(a: Series[float], b: Series, c: Series[str], *, d: Series[int] = 0, e=__dyn_default__):
    """Docstring"""
    if e is __dyn_default__:
        e = a
    if a is __dyn_default__:
        a = lib.close
    x: Series[int] = a
    y: Series[float]
    y, z = a, b
    [y, w] = (1, 2)
    p, q.r = 1, 2
    x = x + 1
    x += 2
    z = x[1] + y[2] + a[1] + c[0]
    lib.max_bars_back(x, 10)
    lib.max_bars_back(unknown, 10)
    lib.other(x, 10)
    v: float = x[1]
    t: int

    def g():
        x = 1
        return a[1] + x

    def h():
        return x[1] + y[1]

    async def k():
        u: Series[float] = 1.0
        return u[1]

    return x


def ov(x: Series[float]):
    return x[1]


def ov(x: Series[int]):
    s: Series[bool] = x > 0
    return s[1]
''',
    'persistent': '''
from pynecore.types import Persistent, IBPersistent, IBPersistentSeries
from pynecore import types, Persistent


def main(seed):
    a: Persistent[int] = 0
    b: Persistent[float] = -3
    c: Persistent[int] = +4
    d: types.Persistent = seed
    e: IBPersistent[int] = -seed
    f: Persistent[bool] = na
    g: Persistent[str]
    h: IBPersistentSeries[float] = 1.5
    a += 1
    b = a + b
    y = (c := c + 1) * 2
    a, b = b, a
    k: float = a
    a.attr = 1

    def inner():
        nonlocal a, x
        a = 1
        return b + c

    def inner2():
        nonlocal b
        b += 1
        return b

    for a in range(3):
        pass
    return a + b + c + d + e + f + g + h
''',
    'closed_shift': '''
from pynecore.core.series import inline_series

__security_contexts__ = {
    's0': {'closed_shift': True, 'group': 1}, 's1': {'closed_shift': True, 'group': 1},
    's2': {'closed_shift': True}, 's3': {'closed_shift': True, 'group': 2},
    's4': {'closed_shift': True}, 's5': {'closed_shift': True}, 's6': {'closed_shift': False},
    's7': {'closed_shift': True}, 's8': {'closed_shift': True, 'group': 3}, 7: {}, 's9': 3,
    's10': {'closed_shift': True, 'group': 3},
}


def main(data):
    __hist_0__ = inline_series(lib.close, 1)
    __sec_write__('s0', __state__[0][1])
    __sec_write__('s1', data[1])
    __sec_write__('s2', (__state__[0][1], inline_series(x, 2)))
    __sec_write__('s3', __hist_0__)
    __sec_write__('s4', [])
    __sec_write__('s7', inline_series(x, 0))
    __sec_write__('s8', lib.ta.sma(lib.close, 3))
    __sec_write__('s10', __state__[0][2])
    s: Series[float] = lib.close


def other():
    __sec_write__('s2', __state·other__[3][1])

    def nested():
        return 1
''',
    'shadowed': '''
__security_contexts__ = {'s0': {'closed_shift': True}}


def inline_series(value, offset):
    return value


def main():
    __sec_write__('s0', inline_series(x, 1))
''',
}


@pytest.mark.parametrize('name', [*__test_helper_SLOT_SOURCES, 'lib'])
def __test_slot_steps_match_python__(name):
    """
    The slot steps, each run natively on the input its Python step gets, leave the
    tree and the slot layout Python leaves; and run as one native batch from the
    first one's input, they hand back what the three Python steps leave together,
    the layout written into the pipeline's object field for field.
    """
    __test_helper_require_native()
    assert native.layout_steps() == frozenset(__test_helper_SLOT_STEPS)
    source = __test_helper_SLOT_SOURCES['persistent' if name == 'lib' else name]
    steps = [step for step in pipeline.LOWERING
             if isinstance(step, pipeline.Step) and step.name in __test_helper_SLOT_STEPS]

    def context() -> pipeline.PipelineContext:
        return pipeline.PipelineContext(
            path=Path(f'slot_{name}.py').resolve(), pyne_mode='lib' if name == 'lib' else None,
            source=source, slot_layout=ModuleLayout(compacted_series=name == 'lib'))

    ctx = context()
    tree = ast.parse(source)
    for step in steps:
        layout = ModuleLayout()
        answer = native.run([step.name], tree, ctx, layout)
        assert isinstance(answer, ast.Module), step.name
        tree = step.run(tree, ctx)
        native.verify(step.name, tree, answer, ctx, layout)

    batch = context()
    answer = native.run([step.name for step in steps], ast.parse(source), batch, batch.slot_layout)
    assert isinstance(answer, ast.Module)
    assert native.describe(answer, batch.slot_layout) == native.describe(tree, ctx.slot_layout)


def __test_isolation_calls_the_same_callees_plainly__():
    """The native isolation step's list of the callees it calls plainly is Python's."""
    __test_helper_require_native()
    assert native._native is not None
    assert sorted(native._native.plain_callees()) == sorted(NON_TRANSFORMABLE_FUNCTIONS)


def __test_slot_steps_reject_what_python_rejects__():
    """A Series or Persistent declaration outside a function is rejected natively too."""
    __test_helper_require_native()
    for step, source in (('Series', 'from pynecore.types import Series\ns: Series[float] = 1.0\n'),
                         ('Persistent', 'p: Persistent[int] = 0\n')):
        ctx = pipeline.PipelineContext(path=Path('rejected.py'), pyne_mode=None, source=source)
        assert isinstance(native.run([step], ast.parse(source), ctx, ModuleLayout()),
                          native.Rejection)


def __test_slot_steps_decline_what_they_cannot_model__():
    """A persistent int literal beyond 64 bits is converted to float by Python's own
    arithmetic; the native step declines it, and the pipeline runs the Python step."""
    __test_helper_require_native()
    source = 'def main():\n    p: Persistent[int] = 99999999999999999999\n    return p\n'
    ctx = pipeline.PipelineContext(path=Path('big.py'), pyne_mode=None, source=source)
    layout = ModuleLayout()
    assert native.run(['Persistent'], ast.parse(source), ctx, layout) is None
    assert not layout.scopes


#: Modules for the rules of the expression phase. Every name listed beside a module is
#: typed as given; every other int or float literal, arithmetic expression and
#: ``int()`` cast is typed a Pine int. A comparison whose left operand is ``exact``
#: carries ``pine_exact``, one whose left operand is ``inexact`` carries it False
__test_helper_RULE_SOURCES: tuple[tuple[str, dict[str, str]], ...] = (
    ('''"""Indexes"""
from __future__ import annotations
import os


def f(lst, n, a, b, c, y, key, d, m, i, j):
    x = lst[n] + lst[n + 1] + lst[int(y)] + lst[int(y) + 1] + lst[2.0] + lst[3]
    x = lst[a:b:c] + lst[:b] + d[key] + m[i, j] + 'abc'[2.0] + lst[-0.0]
    lst[n] = 1.0
    z = __state__[3][n] + __state·f__[0][n]
    for i in range(n):
        x = lst[i]
        for i in range(n, 2, *a):
            x = lst[i]
    y = lst[i] + float(n) + int(n) + float() + range(n, step=1)
    for (k, v) in range(n):
        x = lst[k]
    for k in other(n):
        x = lst[k]
''', {'n': 'i', 'a': 'i', 'b': 'i', 'c': 'i', 'i': 'i', 'j': 'i', 'key': 's'}),
    ('''from pynecore.lib.ta import range as range


def f(n, lst):
    for i in range(n):
        x = lst[i]
''', {'n': 'i', 'i': 'i'}),
    ('''def range(n):
    return n


def f(n, lst):
    for i in range(n):
        x = lst[i]
    y = int(n)
''', {'n': 'i', 'i': 'i'}),
    ('''import pynecore.core.safe_convert as safe_convert
x = int(a) + a / b
''', {}),
    ('''from pynecore.core.safe_convert import native_int as safe_convert
x = int(a) + a / b
''', {}),
    ('''from ..core import safe_convert
from pynecore.core import safe_convert as sc
x = int(a) + a / b
''', {}),
    ('''"""Divisions"""
x = a / b
x /= 2
obj.attr /= n
lst[i] /= n
lst[i:j] /= n
lst[f()] %= g()
lst['k'] /= n
3[n] /= 2


class C:
    y = a / b
    obj.attr /= n

    def m(self, a, b):
        return a / b


def f(a, b, obj, lst, s, n):
    c = a / b + 1 / 2 + 1 / 0 + a / 0 + a / 2.0 + a / 1e999 + a / 0.0 + a / True + a / 'x'
    d = (a + 1) / (b + 2) + a / b + 1 / b + a / (b + 1) + (a + 1) / b + a / s
    e = lambda: a / b
    g = [a / b for a in lst] + {a / b for a in lst} + [x for x in (a / b for a in lst)]
    h = a % b + 'x %s' % a + s % b + f'{a}' % b + {a: a / b for a in lst}
    obj.attr /= b
    lst[a] %= b
    lst[f(a)][g(b)] /= n
    a /= b
    a %= b
    a += b

    class Inner:
        z = a / b

        def deep(self):
            return a / b

    @decorate(a / b)
    def nested(p=a / b) -> a / b:
        return p / a
''', {'s': 's', 'a': 'i', 'b': 'f', 'n': 'i'}),
    ('''"""Comparisons"""
from __future__ import annotations


def f(a, b, c, s, obj):
    r = a < b, a <= b, a > b >= c, a == b, a != b, g(a) == h(b) != c < obj.x
    r = a == 'x', a is b, a in b, 1 < a <= 2.5, a < b == c, obj.x != g(), True == a
    r = exact < a, inexact < a, a < 1, 1 == 2, a != 1.5, obj.x <= obj.y, g() >= a
    r = a < b < c < g(), [x == y for x in a], lambda q: q != a
''', {'a': 'f', 'b': 'f', 'c': 'i'}),
)


def __test_helper_stamped(source: str, names: dict[str, str]) -> ast.Module:
    """A rule module parsed and stamped (see ``__test_helper_RULE_SOURCES``)."""
    from pynecore.transformers.pine_type_rules import INT, set_ty

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in names:
            set_ty(node, names[node.id])
        elif isinstance(node, ast.BinOp) or isinstance(node, ast.Call) \
                and isinstance(node.func, ast.Name) and node.func.id == 'int':
            set_ty(node, INT)
        elif isinstance(node, ast.Constant) and type(node.value) in (int, float):
            set_ty(node, INT)
        elif isinstance(node, ast.Compare) and isinstance(node.left, ast.Name):
            if node.left.id == 'exact':
                setattr(node, 'pine_exact', True)
            elif node.left.id == 'inexact':
                setattr(node, 'pine_exact', False)
    return tree


def __test_helper_check_native(name: str, tree: ast.Module, ctx: pipeline.PipelineContext,
                               python) -> None:
    """Run a native unit and its Python reference on the same input: the native one
    must handle it and produce Python's tree."""
    answer = native.run([name], tree, ctx)
    assert isinstance(answer, ast.Module), (name, answer)
    native.verify(name, python(tree, ctx), answer, ctx)


@pytest.mark.parametrize('sequential', [False, True])
def __test_expression_rules_match_python__(sequential):
    """Each rule of the expression phase, and the phase, natively equal Python's over
    modules reaching every branch, under every context the rules read."""
    __test_helper_require_native()
    phase = next(step for step in pipeline.LOWERING if isinstance(step, pipeline.Phase))
    for source, names in __test_helper_RULE_SOURCES:
        for pyne_mode in (None, 'lib'):
            for user_code in (True, False):
                for na_bool in (False, True):
                    ctx = pipeline.PipelineContext(
                        path=Path('rules.py'), pyne_mode=pyne_mode, user_code=user_code,
                        na_bool=na_bool, sequential=sequential)
                    for rule in phase.rules:
                        __test_helper_check_native(
                            rule.name, __test_helper_stamped(source, names), ctx,
                            lambda tree, c, rule=rule: rule.make(c).visit(tree))
                    units = pipeline._units(phase, ctx)
                    __test_helper_check_native(
                        phase.name, __test_helper_stamped(source, names), ctx,
                        lambda tree, c: functools.reduce(
                            lambda module, unit: unit[1](module, c), units, tree))


#: Modules for the strategy requirements and the input pass
__test_helper_STEP_SOURCES = (
    '''"""Strategy"""
from pynecore.lib import script, strategy


@script.strategy('S', _broker_requirements=None, overlay=True)
def main():
    strategy.entry('a', strategy.long, limit=1, stop=2)
    strategy.entry('b', strategy.direction.short)
    strategy.entry('c', direction=lib.strategy.short)
    strategy.entry('d', strategy.short, direction=strategy.long)
    strategy.entry(*args)
    lib.strategy.order('e', strategy.long, stop=1)
    strategy.order('f', direction=lib.strategy.direction.short, limit=1)
    strategy.exit('x', limit=1, stop=2, qty=1)
    strategy.exit('y', profit=1, trail_offset=2, **extra)
    strategy.exit('z', loss=1, trail_price=1)
    strategy.close('a')
    lib.strategy.close_all()
    strategy.cancel('a')
    other.entry('q')
    x.strategy.exit('q', limit=1)

    @lib.script.strategy('inner')
    def inner():
        pass
''',
    '''from pynecore.core.broker.models import ScriptRequirements
from pynecore import lib


@lib.script.strategy('S')
async def main():
    lib.strategy.exit('x', limit=1)
''',
    '''from pynecore.lib import script


@script.indicator('I')
def main():
    strategy.entry('a')
''',
    '''@script.strategy('S')
def main():
    pass
''',
    '''"""Inputs"""
from __future__ import annotations
from pynecore import lib


def main(a, src=input.source(close), x=lib.input(lib.close), y=lib.input.float(1.0),
         z=input(hl2), w=input(5), v=lib.input.source('high'), u=input.source(defval=lib.low),
         t=input.source(''), s=foo.input(1), r=x.y.input(1), q=input.int(defval=close),
         p=input.source(0), o=input(lib.na), n=input(defval=volume), m=lib.input(lib.time)):
    def inner(p=input.source(open)):
        pass

    f = lambda k=input.source(close): k

    async def coro(m=input.source(close)):
        pass


def main(a=input.source(volume), *, kw=input.source(close)):
    pass


def posonly(a=input.source(close), /, b=input.source(high)):
    pass


g = lambda k=input.source(close): k


class C:
    def method(self, s=lib.input.source(lib.close)):
        pass
''',
    '''def main(length=input.int(5)):
    pass
''',
)


def __test_requirements_and_inputs_match_python__():
    """ScriptRequirements and Input natively equal Python's over modules reaching every
    branch."""
    __test_helper_require_native()
    steps = {step.name: step for step in pipeline.LOWERING if isinstance(step, pipeline.Step)}
    ctx = pipeline.PipelineContext(path=Path('steps.py'), pyne_mode=None)
    for source in __test_helper_STEP_SOURCES:
        for name in ('ScriptRequirements', 'Input', 'FixLocations'):
            __test_helper_check_native(name, ast.parse(source), ctx, steps[name].run)
