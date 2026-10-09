"""
A dependency reached through another module's import is a dependency too.

The lowering routes a call on the object an import binds, wherever that object
lives: a script calling ``mid.g(x)`` is routed on ``leaf.g`` when ``mid`` does
``from leaf import g``. The types only consulted ``mid``, which neither exports
``g`` nor changes when ``g`` starts keeping state, so the script's cache has to
depend on ``leaf`` through ``mid``'s own record of what it imports.

Every load here starts from a cold registry and with the libraries dropped from
``sys.modules``, the way a new process starts (see
``test_158_dependency_routes``).
"""
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from pynecore.core.import_hook import PyneLoader, _baked_deps, _cache_from_source
from pynecore.transformers import module_interface

__test_helper_STATELESS = 'x * 2.0'
__test_helper_STATEFUL = 'ta.sma(x, 3)'


@contextmanager
def __test_helper_bytecode_writing():
    """Allow ``.pyc`` writing, which the test suite turns off globally."""
    saved = sys.dont_write_bytecode
    sys.dont_write_bytecode = False
    try:
        yield
    finally:
        sys.dont_write_bytecode = saved


def __test_helper_write_leaf(path: Path, body: str) -> None:
    """The library that declares ``g``, returning ``body``."""
    path.write_text(f'''"""
@pyne
"""
from pynecore.lib import script, ta


@script.library("{path.stem}")
def main():
    pass


def g(x: float) -> float:
    return {body}
''')


def __test_helper_write_mid(path: Path, spelling: str) -> None:
    """A library that only imports the leaf, with ``spelling``, and calls nothing."""
    path.write_text(f'''"""
@pyne
"""
from pynecore.lib import script
{spelling}


@script.library("{path.stem}")
def main():
    pass
''')


def __test_helper_write_app(path: Path, imports: str, call: str) -> None:
    """A script plotting ``call`` from its ``main``."""
    path.write_text(f'''"""
@pyne
"""
from pynecore.lib import script, close, plot
{imports}


@script.indicator("{path.stem}")
def main():
    plot({call}, "y")
''')


def __test_helper_load(path: Path, monkeypatch, *modules: str):
    """Load a module as a new process would: cold registry, nothing imported yet.

    :return: The code object, and how many times its source was transformed.
    """
    module_interface._registry.clear()
    module_interface._analysing.clear()
    for name in (path.stem,) + modules:
        monkeypatch.delitem(sys.modules, name, raising=False)
    counts = {'transform': 0}
    real = PyneLoader.source_to_code

    def counted(self, data, source_path, *, _optimize: int = -1):
        if Path(source_path).resolve() == path.resolve():
            counts['transform'] += 1
        return real(self, data, source_path, _optimize=_optimize)

    with monkeypatch.context() as patch, __test_helper_bytecode_writing():
        patch.setattr(PyneLoader, 'source_to_code', counted)
        code = PyneLoader(path.stem, str(path)).get_code(path.stem)
    return code, counts['transform']


def __test_helper_names(code) -> set[str]:
    """Every global name a module's code, nested code objects included, reads."""
    names = set(code.co_names)
    for const in code.co_consts:
        if hasattr(const, 'co_names'):
            names |= __test_helper_names(const)
    return names


def __test_helper_paths(code) -> set[str]:
    """The sources a module's bytecode records a dependency on."""
    return {record.path for record in _baked_deps(code)}


@pytest.mark.parametrize('tag, spelling, call', [
    ('from', 'from {leaf} import g', '{mid}.g(close)'),
    ('alias', 'from {leaf} import g as h', '{mid}.h(close)'),
    ('star', 'from {leaf} import *', '{mid}.g(close)'),
    ('namespace', 'import {leaf}', '{mid}.{leaf}.g(close)'),
])
def __test_a_reexported_function_that_starts_keeping_state_drops_the_cache__(
        tmp_path, monkeypatch, tag, spelling, call):
    """The script reaches the leaf only through the middle library's import"""
    monkeypatch.syspath_prepend(tmp_path)
    leaf = tmp_path / f'reexp_{tag}_leaf.py'
    mid = tmp_path / f'reexp_{tag}_mid.py'
    app = tmp_path / f'reexp_{tag}_app.py'
    names = {'leaf': leaf.stem, 'mid': mid.stem}
    __test_helper_write_leaf(leaf, __test_helper_STATELESS)
    __test_helper_write_mid(mid, spelling.format(**names))
    __test_helper_write_app(app, f'import {mid.stem}', call.format(**names))

    code, _ = __test_helper_load(app, monkeypatch, mid.stem, leaf.stem)
    assert str(leaf.resolve()) in __test_helper_paths(code), 'the reexported module is untracked'
    assert '__resolve_slot·__' not in __test_helper_names(code)

    __test_helper_write_leaf(leaf, __test_helper_STATEFUL)
    code, transforms = __test_helper_load(app, monkeypatch, mid.stem, leaf.stem)

    assert transforms == 1, 'the bytecode calling g without its state was kept'
    assert '__resolve_slot·__' in __test_helper_names(code), 'g is not called on the state route'


def __test_a_relative_reexport_is_tracked__(tmp_path, monkeypatch):
    """A relative import names a file next to the importer, with no package context"""
    monkeypatch.syspath_prepend(tmp_path)
    package = tmp_path / 'reexp_pkg'
    package.mkdir()
    (package / '__init__.py').write_text('')
    leaf = package / 'leaf.py'
    mid = package / 'mid.py'
    app = tmp_path / 'reexp_rel_app.py'
    __test_helper_write_leaf(leaf, __test_helper_STATELESS)
    __test_helper_write_mid(mid, 'from .leaf import g')
    __test_helper_write_app(app, 'from reexp_pkg import mid', 'mid.g(close)')
    modules = ('reexp_pkg', 'reexp_pkg.mid', 'reexp_pkg.leaf')

    code, _ = __test_helper_load(app, monkeypatch, *modules)
    assert str(leaf.resolve()) in __test_helper_paths(code)

    __test_helper_write_leaf(leaf, __test_helper_STATEFUL)
    code, transforms = __test_helper_load(app, monkeypatch, *modules)

    assert transforms == 1, 'the bytecode calling g without its state was kept'
    assert '__resolve_slot·__' in __test_helper_names(code)


def __test_a_reexported_body_edit_that_keeps_the_routes_keeps_the_cache__(
        tmp_path, monkeypatch):
    """Tracking the import must not make every edit behind it rebuild the script"""
    monkeypatch.syspath_prepend(tmp_path)
    leaf = tmp_path / 'reexp_keep_leaf.py'
    mid = tmp_path / 'reexp_keep_mid.py'
    app = tmp_path / 'reexp_keep_app.py'
    __test_helper_write_leaf(leaf, __test_helper_STATELESS)
    __test_helper_write_mid(mid, f'from {leaf.stem} import g')
    __test_helper_write_app(app, f'import {mid.stem}', f'{mid.stem}.g(close)')
    __test_helper_load(app, monkeypatch, mid.stem, leaf.stem)
    mtime = _cache_from_source(app).stat().st_mtime_ns

    __test_helper_write_leaf(leaf, 'x * 3.25 + 1.0')
    _code, transforms = __test_helper_load(app, monkeypatch, mid.stem, leaf.stem)

    assert transforms == 0, 'a body edit that keeps every route rebuilt the dependent'
    assert _cache_from_source(app).stat().st_mtime_ns == mtime


def __test_an_import_of_plain_python_records_nothing__(tmp_path, monkeypatch):
    """A module that is not Pyne code publishes no interface to depend on"""
    monkeypatch.syspath_prepend(tmp_path)
    plain = tmp_path / 'reexp_plain.py'
    plain.write_text('VALUE = 2.0\n')
    leaf = tmp_path / 'reexp_plain_leaf.py'
    app = tmp_path / 'reexp_plain_app.py'
    __test_helper_write_leaf(leaf, __test_helper_STATELESS)
    __test_helper_write_app(app, f'import {plain.stem}\nimport {leaf.stem}',
                            f'{leaf.stem}.g(close) * {plain.stem}.VALUE')

    code, transforms = __test_helper_load(app, monkeypatch, plain.stem, leaf.stem)

    assert transforms == 1
    assert __test_helper_paths(code) == {str(leaf.resolve())}


@pytest.mark.parametrize('tag, imports, call', [
    ('module', 'import {wrap}', '{wrap}.g(close)'),
    ('from', 'from {wrap} import g', 'g(close)'),
])
def __test_a_function_reexported_by_plain_python_is_tracked__(tmp_path, monkeypatch, tag,
                                                              imports, call):
    """A plain Python module between the script and the leaf does not hide the leaf"""
    monkeypatch.syspath_prepend(tmp_path)
    leaf = tmp_path / f'reexp_wrap_{tag}_leaf.py'
    wrap = tmp_path / f'reexp_wrap_{tag}.py'
    app = tmp_path / f'reexp_wrap_{tag}_app.py'
    names = {'wrap': wrap.stem}
    __test_helper_write_leaf(leaf, __test_helper_STATELESS)
    wrap.write_text(f'from {leaf.stem} import g\n')
    __test_helper_write_app(app, imports.format(**names), call.format(**names))

    code, _ = __test_helper_load(app, monkeypatch, wrap.stem, leaf.stem)
    assert str(leaf.resolve()) in __test_helper_paths(code), 'the defining module is untracked'
    assert '__resolve_slot·__' not in __test_helper_names(code)

    __test_helper_write_leaf(leaf, __test_helper_STATEFUL)
    code, transforms = __test_helper_load(app, monkeypatch, wrap.stem, leaf.stem)

    assert transforms == 1, 'the bytecode calling g without its state was kept'
    assert '__resolve_slot·__' in __test_helper_names(code), 'g is not called on the state route'


def __test_a_plain_python_rebinding_drops_the_cache__(tmp_path, monkeypatch):
    """What the plain module binds decides the route: an edit to it rebuilds the script"""
    monkeypatch.syspath_prepend(tmp_path)
    stateless = tmp_path / 'reexp_bind_stateless.py'
    stateful = tmp_path / 'reexp_bind_stateful.py'
    wrap = tmp_path / 'reexp_bind_wrap.py'
    app = tmp_path / 'reexp_bind_app.py'
    __test_helper_write_leaf(stateless, __test_helper_STATELESS)
    __test_helper_write_leaf(stateful, __test_helper_STATEFUL)
    wrap.write_text(f'from {stateless.stem} import g\n')
    __test_helper_write_app(app, f'import {wrap.stem}', f'{wrap.stem}.g(close)')
    modules = (wrap.stem, stateless.stem, stateful.stem)

    code, _ = __test_helper_load(app, monkeypatch, *modules)
    assert '__resolve_slot·__' not in __test_helper_names(code)

    wrap.write_text(f'from {stateful.stem} import g  # rebound\n')
    code, transforms = __test_helper_load(app, monkeypatch, *modules)

    assert transforms == 1, 'the bytecode routed on the old binding was kept'
    assert '__resolve_slot·__' in __test_helper_names(code)


def __test_a_plain_python_rebinding_behind_a_library_drops_the_callers_cache__(
        tmp_path, monkeypatch):
    """The plain module's record reaches the script through the library calling into it"""
    monkeypatch.syspath_prepend(tmp_path)
    stateless = tmp_path / 'reexp_deep_stateless.py'
    stateful = tmp_path / 'reexp_deep_stateful.py'
    wrap = tmp_path / 'reexp_deep_wrap.py'
    mid = tmp_path / 'reexp_deep_mid.py'
    app = tmp_path / 'reexp_deep_app.py'
    __test_helper_write_leaf(stateless, __test_helper_STATELESS)
    __test_helper_write_leaf(stateful, __test_helper_STATEFUL)
    wrap.write_text(f'from {stateless.stem} import g\n')
    __test_helper_write_mid(mid, f'import {wrap.stem}')
    with mid.open('a') as file:
        file.write(f'\n\ndef f(x: float) -> float:\n    return {wrap.stem}.g(x)\n')
    __test_helper_write_app(app, f'import {mid.stem}', f'{mid.stem}.f(close)')
    modules = (mid.stem, wrap.stem, stateless.stem, stateful.stem)

    code, _ = __test_helper_load(app, monkeypatch, *modules)
    assert str(wrap.resolve()) in __test_helper_paths(code), 'the plain module is untracked'
    assert '__resolve_slot·__' not in __test_helper_names(code)

    wrap.write_text(f'from {stateful.stem} import g  # rebound\n')
    code, transforms = __test_helper_load(app, monkeypatch, *modules)

    assert transforms == 1, 'the bytecode calling f without its state was kept'
    assert '__resolve_slot·__' in __test_helper_names(code), 'f is not called on the state route'
