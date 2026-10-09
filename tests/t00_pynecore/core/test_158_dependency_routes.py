"""
A dependency record covers how the dependent's calls into the dependency are routed.

The lowering emits a call into a state-carrying function with the hidden state
argument in front, and a call into a stateless one plainly -- and which of the
two a library function is can change with nothing but its body. Its interface
digest is blind to bodies on purpose, so the dependency record also carries the
dependency's ROUTES digest: an edit that changes whether an export keeps state
has to drop the dependent's cache, one that does not has to keep it.

Every load here starts from a cold registry and with the dependencies dropped
from ``sys.modules``, the way a new process starts: the lowering of the
dependent imports the libraries it calls into, and a module object left over
from an earlier load would answer for a source that is no longer on disk.
"""
import sys
from contextlib import contextmanager
from pathlib import Path

from pynecore.core.import_hook import (
    PyneLoader, _PYNE_INTERFACE, _baked_deps, _cache_from_source,
)
from pynecore.transformers import module_interface
from pynecore.transformers.pine_type_table import DepRecord

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


def __test_helper_write_lib(path: Path, body: str) -> None:
    """A library whose one export returns ``body``."""
    path.write_text(f'''"""
@pyne
"""
from pynecore.lib import script, ta


@script.library("{path.stem}")
def main():
    pass


def f(x: float) -> float:
    return {body}
''')


def __test_helper_write_app(path: Path, lib: str) -> None:
    """A script calling ``<lib>.f`` from its ``main``."""
    path.write_text(f'''"""
@pyne
"""
from pynecore.lib import script, close, plot
import {lib}


@script.indicator("{path.stem}")
def main():
    plot({lib}.f(close), "y")
''')


def __test_helper_load(path: Path, monkeypatch, *modules: str):
    """Load a module as a new process would: cold registry, nothing imported yet.

    :return: The code object, and how many times a source was transformed.
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


def __test_helper_pair(tmp_path: Path, monkeypatch, prefix: str,
                       body: str) -> tuple[Path, Path, DepRecord]:
    """Write a library and a script calling it, and build the script once.

    :return: The library, the script, and the record the script was baked with.
    """
    monkeypatch.syspath_prepend(tmp_path)
    lib = tmp_path / f'{prefix}_lib.py'
    app = tmp_path / f'{prefix}_app.py'
    __test_helper_write_lib(lib, body)
    __test_helper_write_app(app, lib.stem)
    code, transforms = __test_helper_load(app, monkeypatch, lib.stem)
    assert transforms == 1
    records = _baked_deps(code)
    assert [record.path for record in records] == [str(lib.resolve())]
    assert records[0].routes, 'the record was baked without the routes it was built against'
    return lib, app, records[0]


def __test_a_dependency_that_starts_keeping_state_drops_the_cache__(tmp_path, monkeypatch):
    """A stateless export turning stateful has to be called with a state argument"""
    lib, app, before = __test_helper_pair(tmp_path, monkeypatch, 'routes_gain',
                                          __test_helper_STATELESS)

    __test_helper_write_lib(lib, __test_helper_STATEFUL)
    code, transforms = __test_helper_load(app, monkeypatch, lib.stem)

    assert transforms == 1, 'the bytecode calling f without its state was kept'
    after = _baked_deps(code)[0]
    assert after.digest == before.digest, 'the signature did not move, only the routes'
    assert after.routes != before.routes
    assert '__resolve_slot·__' in __test_helper_names(code), 'f is not called on the state route'


def __test_a_dependency_that_stops_keeping_state_drops_the_cache__(tmp_path, monkeypatch):
    """A stateful export turning stateless must not be handed a state argument"""
    lib, app, before = __test_helper_pair(tmp_path, monkeypatch, 'routes_lose',
                                          __test_helper_STATEFUL)

    __test_helper_write_lib(lib, __test_helper_STATELESS)
    code, transforms = __test_helper_load(app, monkeypatch, lib.stem)

    assert transforms == 1, 'the bytecode handing f a state argument was kept'
    after = _baked_deps(code)[0]
    assert after.digest == before.digest
    assert after.routes != before.routes
    assert '__resolve_slot·__' not in __test_helper_names(code), 'f is still given a state'


def __test_a_body_edit_that_keeps_the_routes_keeps_the_cache__(tmp_path, monkeypatch):
    """Neither the signature nor whether f keeps state moved, so nothing is rebuilt"""
    lib, app, before = __test_helper_pair(tmp_path, monkeypatch, 'routes_keep',
                                          __test_helper_STATELESS)
    mtime = _cache_from_source(app).stat().st_mtime_ns

    __test_helper_write_lib(lib, 'x * 3.25 + 1.0')
    code, transforms = __test_helper_load(app, monkeypatch, lib.stem)

    assert transforms == 0, 'a body edit that keeps every route rebuilt the dependent'
    assert _cache_from_source(app).stat().st_mtime_ns == mtime
    assert _baked_deps(code)[0] == before


def __test_a_dependent_built_against_a_cached_dependency_keeps_its_cache__(
        tmp_path, monkeypatch):
    """Routes are settled even when the dependency itself came from its .pyc"""
    lib, app, _ = __test_helper_pair(tmp_path, monkeypatch, 'routes_cached',
                                     __test_helper_STATELESS)

    # The script is edited, its library is not: the library loads from its cache
    # while the script is transformed, and the record still has to carry routes
    app.write_text(app.read_text() + '\n# edited\n')
    code, transforms = __test_helper_load(app, monkeypatch, lib.stem)
    assert transforms == 1
    assert _baked_deps(code)[0].routes
    mtime = _cache_from_source(app).stat().st_mtime_ns

    __test_helper_write_lib(lib, 'x * 3.25 + 1.0')
    _code, transforms = __test_helper_load(app, monkeypatch, lib.stem)

    assert transforms == 0, 'a body edit that keeps every route rebuilt the dependent'
    assert _cache_from_source(app).stat().st_mtime_ns == mtime


def __test_a_route_change_two_hops_away_drops_the_cache__(tmp_path, monkeypatch):
    """A middle function calling a leaf that starts keeping state keeps state itself"""
    monkeypatch.syspath_prepend(tmp_path)
    leaf = tmp_path / 'routes_chain_leaf.py'
    middle = tmp_path / 'routes_chain_mid.py'
    app = tmp_path / 'routes_chain_app.py'
    __test_helper_write_lib(leaf, __test_helper_STATELESS)
    middle.write_text(f'''"""
@pyne
"""
from pynecore.lib import script
import {leaf.stem}


@script.library("{middle.stem}")
def main():
    pass


def f(x: float) -> float:
    return {leaf.stem}.f(x) + 1.0
''')
    __test_helper_write_app(app, middle.stem)
    code, _ = __test_helper_load(app, monkeypatch, middle.stem, leaf.stem)
    assert sorted(record.path for record in _baked_deps(code)) == \
        sorted([str(leaf.resolve()), str(middle.resolve())])
    assert '__resolve_slot·__' not in __test_helper_names(code)
    middle_bytes = middle.read_bytes()

    __test_helper_write_lib(leaf, __test_helper_STATEFUL)
    code, transforms = __test_helper_load(app, monkeypatch, middle.stem, leaf.stem)

    assert middle.read_bytes() == middle_bytes, 'the middle module was not touched'
    assert transforms == 1, 'the bytecode calling the middle function plainly was kept'
    assert '__resolve_slot·__' in __test_helper_names(code)


def __test_a_dependency_is_transformed_once_per_cold_load__(tmp_path, monkeypatch):
    """The transform the type pass asks for is the one the dependency's import reuses"""
    monkeypatch.syspath_prepend(tmp_path)
    lib = tmp_path / 'routes_once_lib.py'
    unused = tmp_path / 'routes_once_unused.py'
    app = tmp_path / 'routes_once_app.py'
    __test_helper_write_lib(lib, __test_helper_STATELESS)
    __test_helper_write_lib(unused, __test_helper_STATEFUL)
    __test_helper_write_app(app, lib.stem)
    app.write_text(app.read_text().replace(f'import {lib.stem}\n',
                                           f'import {lib.stem}\nimport {unused.stem}\n'))
    transforms: list[str] = []
    real = PyneLoader.source_to_code

    def counted(self, data, source_path, *, _optimize: int = -1):
        transforms.append(Path(source_path).stem)
        return real(self, data, source_path, _optimize=_optimize)

    monkeypatch.setattr(PyneLoader, 'source_to_code', counted)
    code, _ = __test_helper_load(app, monkeypatch, lib.stem, unused.stem)
    # The called library is imported by the script's lowering, the unused one
    # only consulted by its type pass: each is transformed exactly once
    assert sorted(transforms) == sorted([app.stem, lib.stem, unused.stem])
    assert {record.path: bool(record.routes) for record in _baked_deps(code)} == \
        {str(lib.resolve()): True, str(unused.resolve()): True}

    # Executing the script imports both: each finds its .pyc written, and
    # carries the interface it published in it
    with __test_helper_bytecode_writing():
        for module in (lib, unused):
            dependency = PyneLoader(module.stem, str(module)).get_code(module.stem)
            assert any(isinstance(const, tuple) and const and const[0] == _PYNE_INTERFACE
                       for const in dependency.co_consts)
    assert sorted(transforms) == sorted([app.stem, lib.stem, unused.stem])


def __test_a_moved_dependency_is_transformed_once_into_its_pyc__(tmp_path, monkeypatch):
    """The transform a moved dependency costs is left in its .pyc for its import and the next process"""
    lib, app, _ = __test_helper_pair(tmp_path, monkeypatch, 'routes_pyc',
                                     __test_helper_STATELESS)

    __test_helper_write_lib(lib, 'x * 3.25 + 1.0')
    transforms: list[str] = []
    real = PyneLoader.source_to_code

    def counted(self, data, source_path, *, _optimize: int = -1):
        transforms.append(str(Path(source_path).resolve()))
        return real(self, data, source_path, _optimize=_optimize)

    monkeypatch.setattr(PyneLoader, 'source_to_code', counted)
    _code, app_transforms = __test_helper_load(app, monkeypatch, lib.stem)
    assert app_transforms == 0, 'a body edit that keeps every route rebuilt the dependent'
    assert transforms == [str(lib.resolve())]

    # The library's own import reuses that transform
    with __test_helper_bytecode_writing():
        PyneLoader(lib.stem, str(lib)).get_code(lib.stem)
    assert transforms == [str(lib.resolve())], 'the import transformed the library again'

    # A new process reads the routes off the library's .pyc, and transforms nothing
    __test_helper_load(app, monkeypatch, lib.stem)
    assert transforms == [str(lib.resolve())], 'the transform was paid for twice'
