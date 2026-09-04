"""
What a module publishes, and how the next process finds it again.

Three things are under test here, and they are one mechanism: the INTERFACE a
module exports, the digest that says whether it changed, and the artifact plus
registry that let a dependent read it without importing anything.

The interface is what makes the dependency check worth having. It is derived
from the signatures alone, so editing a body leaves every dependent's cached
bytecode valid, while changing a return annotation invalidates exactly the
dependents that could care.
"""
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from pynecore.core.import_hook import (
    PIPELINE_DIGEST, PyneLoader, _get_transform_pipeline_hash, analyse_source,
)
from pynecore.transformers import pine_type_artifact
from pynecore.transformers.pine_type_artifact import (
    ARTIFACT_ENV, ARTIFACT_VERSION, NO_FINGERPRINT, artifact_enabled, artifact_path,
    build_interface, interface_digest, lookup, read_artifact, register, registered,
    source_digest, stable_source,
)
from pynecore.transformers.pine_type_rules import FLOAT, INT, UNKNOWN
from pynecore.transformers.pine_type_table import ModuleInterface


@contextmanager
def _bytecode_writing_enabled():
    """Temporarily allow ``.pyc`` writing (the test suite disables it globally)."""
    saved = sys.dont_write_bytecode
    sys.dont_write_bytecode = False
    try:
        yield
    finally:
        sys.dont_write_bytecode = saved


@pytest.fixture(autouse=True)
def _clean_registry():
    """Keep the process-wide registry from leaking between tests."""
    pine_type_artifact._registry.clear()
    pine_type_artifact._analysing.clear()
    yield
    pine_type_artifact._registry.clear()
    pine_type_artifact._analysing.clear()


def _module(tmp_path: Path, name: str, source: str) -> Path:
    """Write a Pyne module and hand back its path."""
    path = tmp_path / f'{name}.py'
    path.write_text(source, encoding='utf-8')
    return path


def _interface(tmp_path: Path, name: str, source: str) -> ModuleInterface:
    """Analyse a source and build the interface it publishes."""
    path = _module(tmp_path, name, source)
    analysed = analyse_source(str(path))
    assert analysed is not None, 'the module was not recognized as Pyne code'
    tree, table, fingerprint = analysed
    return build_interface(tree, table, str(path.resolve()), fingerprint)


def _loader(path: Path) -> PyneLoader:
    """The loader an import of one module would use."""
    return PyneLoader(path.stem, str(path))


def _compile(path: Path):
    """Run the real pipeline over a module, the way an import would."""
    return _loader(path).source_to_code(path.read_bytes(), str(path))


# --- the interface --------------------------------------------------------


PLAIN = '''"""
@pyne
"""
__all__ = ['typed', 'plain']

WIDTH = 3


class Holder:
    value: int = 1


def typed(length: int, source: float) -> float:
    return source / length


def plain(x):
    return x
'''


def __test_module_level_defs_are_the_exports__(tmp_path):
    """A module-level def is an export; a class and a variable are not"""
    interface = _interface(tmp_path, 'plain_mod', PLAIN)

    assert set(interface.exports) == {'typed', 'plain'}
    assert interface.all == ('typed', 'plain')


def __test_an_annotated_def_publishes_its_shape__(tmp_path):
    """Parameter types, arity and return type come out of the annotations"""
    sig = _interface(tmp_path, 'shape_mod', PLAIN).exports['typed']

    assert sig.kind == 'function'
    assert sig.params == (INT, FLOAT)
    assert sig.required == 2
    assert sig.open_ended is False
    assert sig.ret == FLOAT
    assert sig.annotated is True


def __test_an_unannotated_def_says_so__(tmp_path):
    """A parameter with no readable annotation makes the export unannotated"""
    sig = _interface(tmp_path, 'unannotated_mod', PLAIN).exports['plain']

    assert sig.annotated is False
    assert sig.params == (UNKNOWN,)


def __test_the_last_definition_wins_the_name__(tmp_path):
    """Two defs of one name publish the second, the way Python binds it"""
    interface = _interface(tmp_path, 'rebound_mod', '''"""
@pyne
"""


def pick(x: int) -> int:
    return x


def pick(x: int, y: int) -> float:
    return x + y + 0.5
''')

    sig = interface.exports['pick']
    assert sig.params == (INT, INT)
    assert sig.ret == FLOAT


@pytest.mark.parametrize('name,rebinding', [
    ('assignment', 'pick = other'),
    ('import', 'from math import pick'),
    ('class', 'class pick:\n    pass'),
])
def __test_a_replaced_definition_is_not_published__(tmp_path, name: str, rebinding: str):
    """An importer receives what the NAME is bound to, never the def that lost it"""
    # Publishing the definition would have every dependent type -- and pin --
    # against a function the import never hands them.
    interface = _interface(tmp_path, f'replaced_{name}', f'''"""
@pyne
"""


def other(x: int) -> float:
    return x + 0.5


def pick(x: int) -> int:
    return x


{rebinding}
''')

    assert 'pick' not in interface.exports
    assert 'other' in interface.exports


def __test_a_conditional_rebinding_unpublishes_too__(tmp_path):
    """A binding in a branch is a binding: whether it ran is not a static question"""
    interface = _interface(tmp_path, 'replaced_branch_mod', '''"""
@pyne
"""


def pick(x: int) -> int:
    return x


if True:
    pick = None
''')

    assert interface.exports == {}


def __test_a_definition_that_wins_the_name_back_is_published__(tmp_path):
    """The last binding decides, so a def BELOW the assignment is the export"""
    # Order matters and an order-blind "this name is bound elsewhere too"
    # loses a perfectly good signature here.
    interface = _interface(tmp_path, 'reclaimed_mod', '''"""
@pyne
"""


def other(x: int) -> float:
    return x + 0.5


pick = other


def pick(x: int) -> int:
    return x
''')

    assert interface.exports['pick'].ret == INT
    assert interface.exports['pick'].params == (INT,)


def __test_exclusive_branch_definitions_are_not_published__(tmp_path):
    """Neither def binds the name for certain, so neither may be published"""
    # Both are "not rebound" -- a def is not a rebinding -- and the lexically
    # last one is no more the runtime binding than the first.
    interface = _interface(tmp_path, 'branch_defs_mod', '''"""
@pyne
"""
FLAG = True

if FLAG:

    def pick(x: int) -> int:
        return x

else:

    def pick(x: int) -> float:
        return x + 0.5
''')

    assert 'pick' not in interface.exports


def __test_an_unconditional_definition_closes_a_branched_one__(tmp_path):
    """A def no branch guards binds the name whatever the branch did"""
    interface = _interface(tmp_path, 'closed_branch_mod', '''"""
@pyne
"""
FLAG = True

if FLAG:

    def pick(x: int) -> int:
        return x


def pick(x: int) -> float:
    return x + 0.5
''')

    assert interface.exports['pick'].ret == FLOAT


def __test_a_branched_overload_group_is_not_published__(tmp_path):
    """A group publishes every implementation, so one in a branch conditions the set"""
    interface = _interface(tmp_path, 'branch_group_mod', '''"""
@pyne
"""
from pynecore.core.overload import overload

FLAG = True

if FLAG:

    @overload
    def pick(x: int) -> int:
        return x


@overload
def pick(x: float) -> float:
    return x
''')

    assert 'pick' not in interface.exports


GROUP = '''"""
@pyne
"""
from pynecore.core.overload import overload


@overload
def pick(x: int) -> int:
    return x


@overload
def pick(x: float) -> float:
    return x
'''


def __test_an_overload_group_publishes_its_implementations__(tmp_path):
    """A group is one export carrying every implementation's own shape"""
    sig = _interface(tmp_path, 'group_mod', GROUP).exports['pick']

    assert sig.kind == 'group'
    assert [impl.params for impl in sig.impls] == [(INT,), (FLOAT,)]
    assert [impl.ret for impl in sig.impls] == [INT, FLOAT]
    # The implementations disagree, so the group itself has no single type
    assert sig.ret == UNKNOWN
    assert sig.annotated is True


EXPORTED = '''"""
@pyne edge
"""
from pynecore.core.pine_export import Exported, export
from pynecore.lib import script
from typing import Protocol, Any


__all__ = ['scaled']


class _ProtocolScaled(Protocol):
    def __call__(self, length: int) -> Any: ...


scaled: _ProtocolScaled = Exported()


@script.library('probe')
def main():
    def helper(length: int) -> int:
        return length * 2

    @export
    def scaled(length: int):
        return helper(length) + 1
'''


def __test_an_exported_proxy_takes_the_nested_signature__(tmp_path):
    """``X = Exported()`` publishes the ``@export def X`` nested in ``main``"""
    interface = _interface(tmp_path, 'exported_mod', EXPORTED)

    assert set(interface.exports) == {'main', 'scaled'}
    sig = interface.exports['scaled']
    assert sig.kind == 'function'
    assert sig.params == (INT,)
    # The nested def spells no return annotation, so the type is the inferred one
    assert sig.ret == INT
    # ``helper`` is nested but carries no ``@export``, so it publishes nothing
    assert 'helper' not in interface.exports


EXPORTED_GROUP = '''"""
@pyne edge
"""
from pynecore.core.overload import overload
from pynecore.core.pine_export import Exported, export
from pynecore.lib import script
from typing import Protocol, Any


__all__ = ['at']


class _ProtocolAt(Protocol):
    def __call__(self, length: int) -> Any: ...


at: _ProtocolAt = Exported()


@script.library('probe')
def main():
    @export
    @overload
    def at(length: int, flag: bool = True):
        return length + (1 if flag else 0)

    @export
    @overload
    def at(length: int, name: str = ''):
        return length + len(name)
'''


def __test_an_exported_proxy_can_be_a_group__(tmp_path):
    """A compiled library stacks ``@export`` on ``@overload``; that is one group"""
    sig = _interface(tmp_path, 'exported_group_mod', EXPORTED_GROUP).exports['at']

    assert sig.kind == 'group'
    assert len(sig.impls) == 2
    assert [impl.params for impl in sig.impls] == [(INT, 'b'), (INT, 's')]
    # Both implementations return an int, so the group does too
    assert [impl.ret for impl in sig.impls] == [INT, INT]
    assert sig.ret == INT


# --- the digest -----------------------------------------------------------


DIGEST_BASE = '''"""
@pyne
"""
__all__ = ['area']


def area(width: int, height: int) -> int:
    return width * height
'''


def __test_a_body_change_keeps_the_digest__(tmp_path):
    """The digest is over the signatures; a body is not part of the contract"""
    first = _interface(tmp_path, 'digest_a', DIGEST_BASE)
    second = _interface(tmp_path, 'digest_b', DIGEST_BASE.replace(
        'return width * height', 'total: int = width * height\n    return total'))

    assert first.digest == second.digest


@pytest.mark.parametrize('name,changed', [
    ('return_type', DIGEST_BASE.replace('-> int:', '-> float:')),
    ('param_annotation', DIGEST_BASE.replace('height: int', 'height: float')),
    ('added_export', DIGEST_BASE + '\n\ndef perimeter(side: int) -> int:\n    return 4 * side\n'),
    ('changed_all', DIGEST_BASE.replace("__all__ = ['area']", "__all__ = ['area', 'other']")),
])
def __test_an_interface_change_moves_the_digest__(tmp_path, name: str, changed: str):
    """Everything a dependent could resolve a call against is in the digest"""
    base = _interface(tmp_path, f'base_{name}', DIGEST_BASE)
    other = _interface(tmp_path, f'other_{name}', changed)

    assert base.digest != other.digest


def __test_the_digest_is_a_short_hex_string__(tmp_path):
    """The digest rides in a code constant and a file name, so it stays short"""
    interface = _interface(tmp_path, 'short_mod', DIGEST_BASE)

    assert len(interface.digest) == 16
    assert interface_digest(interface) == interface.digest
    int(interface.digest, 16)  # raises if it is not hex


# --- the artifact ---------------------------------------------------------


ARTIFACT_SOURCE = '''"""
@pyne edge
"""
__all__ = ['halve']

LENGTH: int = 14
STEP = LENGTH / 2


def halve(length: int) -> int:
    return length / 2
'''


def _write_artifact_for(tmp_path: Path, name: str, source: str, monkeypatch,
                        env: str | None) -> Path:
    """Transform a module with the artifact switch in a known state."""
    if env is None:
        monkeypatch.delenv(ARTIFACT_ENV, raising=False)
    else:
        monkeypatch.setenv(ARTIFACT_ENV, env)
    path = _module(tmp_path, name, source)
    with _bytecode_writing_enabled():
        _compile(path)
    return path


def __test_the_artifact_records_the_whole_analysis__(tmp_path, monkeypatch):
    """A transformed module leaves its types beside its bytecode"""
    path = _write_artifact_for(tmp_path, 'artifact_mod', ARTIFACT_SOURCE, monkeypatch, '1')
    target = artifact_path(path, PIPELINE_DIGEST)

    assert target.exists()
    assert target.name == f'artifact_mod.{PIPELINE_DIGEST[:12]}.pynetypes.json'
    data = json.loads(target.read_text(encoding='utf-8'))
    assert data['v'] == ARTIFACT_VERSION
    assert data['pipeline'] == PIPELINE_DIGEST
    assert data['src'] == source_digest(path.read_bytes())
    # ``LENGTH / 2`` is int-typed on TradingView, which is the whole point of
    # the pass whose answers this file carries
    assert any(entry[1] == INT for entry in data['exprs'])
    assert data['funcs']['halve']['ret'] == INT
    assert data['interface']['all'] == ['halve']
    assert data['interface']['exports']['halve']['params'] == [INT]
    assert data['deps'] == {}


PINNED_SOURCE = '''"""
@pyne
"""
from pynecore.core.overload import overload


@overload
def pick(x: int) -> str:
    return 'int-impl'


@overload
def pick(x: float) -> str:
    return 'float-impl'


def wrapper(v):
    return pick(v)


def main():
    return wrapper(1), wrapper(2.0)
'''


def __test_the_serialized_pins_name_a_call_that_exists__(tmp_path, monkeypatch):
    """A context's pins are written out by POSITION, which is what survives"""
    # The node ids the inference hands out describe the tree it walked; the
    # lowering builds new call nodes over it and the expression list is
    # renumbered over the FINAL tree, so an id written here would name
    # nothing a reader -- of this same file -- could find.
    path = _write_artifact_for(tmp_path, 'pinned_mod', PINNED_SOURCE, monkeypatch, '1')
    data = json.loads(artifact_path(path, PIPELINE_DIGEST).read_text(encoding='utf-8'))

    entries = [entry for context in data['contexts'] for entry in context['pins']]
    assert entries, 'the module has pinned call sites, and none was written out'
    # The two contexts disagree, which is the case the per-context map exists
    # for -- and the reason a single pin on the node is not enough
    assert {entry['pin'] for entry in entries} == {'i', None}

    sites = {(call['line'], call['col']) for call in data['calls']}
    assert sites
    for entry in entries:
        assert (entry['line'], entry['col']) in sites, f'{entry} names no call site'


def __test_a_foreign_pipeline_artifact_is_not_read__(tmp_path, monkeypatch):
    """An artifact of another pipeline names another file and is never read"""
    path = _write_artifact_for(tmp_path, 'foreign_mod', ARTIFACT_SOURCE, monkeypatch, '1')

    assert read_artifact(path, PIPELINE_DIGEST) is not None
    assert read_artifact(path, '0000oldpipeline0') is None


def __test_the_artifact_switch_decides_the_write__(tmp_path, monkeypatch):
    """``PYNE_TYPE_ARTIFACT`` overrides the mode in both directions"""
    off = _write_artifact_for(tmp_path, 'off_mod', ARTIFACT_SOURCE, monkeypatch, '0')
    assert not artifact_path(off, PIPELINE_DIGEST).exists()

    edge = _write_artifact_for(tmp_path, 'edge_mod', ARTIFACT_SOURCE, monkeypatch, None)
    assert artifact_path(edge, PIPELINE_DIGEST).exists()

    script = _write_artifact_for(
        tmp_path, 'script_mod', ARTIFACT_SOURCE.replace('@pyne edge', '@pyne'),
        monkeypatch, None)
    assert not artifact_path(script, PIPELINE_DIGEST).exists()


def __test_the_switch_is_read_the_same_way_everywhere__(monkeypatch):
    """The mode only decides where the environment says nothing"""
    monkeypatch.delenv(ARTIFACT_ENV, raising=False)
    assert artifact_enabled('edge') is True
    assert artifact_enabled('lib') is False
    assert artifact_enabled(None) is False

    monkeypatch.setenv(ARTIFACT_ENV, '1')
    assert artifact_enabled(None) is True
    monkeypatch.setenv(ARTIFACT_ENV, '0')
    assert artifact_enabled('edge') is False


def __test_no_bytecode_means_no_artifact__(tmp_path, monkeypatch):
    """The artifact is a cache, so it follows the bytecode switch"""
    monkeypatch.setenv(ARTIFACT_ENV, '1')
    path = _module(tmp_path, 'nocache_mod', ARTIFACT_SOURCE)
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    _compile(path)

    assert not artifact_path(path, PIPELINE_DIGEST).exists()


# --- the lookup order -----------------------------------------------------


def __test_the_transform_publishes_the_bytes_its_fingerprint_belongs_to__(tmp_path, monkeypatch):
    """The loader read the source before handing it over; the file may have moved since"""
    # ``get_code`` reads the bytes and only then calls in here, so an atomic
    # replace in between would otherwise pair the OLD bytes with the NEW
    # file's stat -- a state that never existed, which every later fingerprint
    # check then certifies as current.
    monkeypatch.setenv(ARTIFACT_ENV, '0')
    path = _module(tmp_path, 'race_mod', ARTIFACT_SOURCE)
    superseded = ARTIFACT_SOURCE.replace('-> int:', '-> float:').encode('utf-8')

    _loader(path).source_to_code(superseded, str(path))

    interface = registered(str(path.resolve()))
    on_disk = path.stat()
    assert interface is not None
    assert interface.exports['halve'].ret == INT, 'the superseded bytes were published'
    assert (interface.mtime_ns, interface.size) == (on_disk.st_mtime_ns, on_disk.st_size)
    # The fingerprint is now one a lookup may hand out, because it describes
    # the file the signatures were read from
    assert lookup(str(path), None, PIPELINE_DIGEST) is interface


def __test_bytes_with_no_file_behind_them_publish_no_fingerprint__(tmp_path, monkeypatch):
    """A source that cannot be read alongside its stat gets a pairing nothing matches"""
    monkeypatch.setenv(ARTIFACT_ENV, '1')
    path = tmp_path / 'absent_mod.py'

    with _bytecode_writing_enabled():
        _loader(path).source_to_code(ARTIFACT_SOURCE, str(path))

    interface = registered(str(path))
    assert interface is not None, 'the transform itself still has to work'
    assert (interface.mtime_ns, interface.size) == NO_FINGERPRINT
    assert not artifact_path(path, PIPELINE_DIGEST).exists(), 'an untrustworthy artifact'

    # A file appearing under that name does not make the sentinel match, so
    # the entry is evicted rather than handed out
    path.write_text(ARTIFACT_SOURCE, encoding='utf-8')
    assert lookup(str(path), analyse_source, PIPELINE_DIGEST) is not interface


def __test_a_registered_interface_answers_without_a_file__(tmp_path, monkeypatch):
    """The registry is the first answer, and it reads nothing"""
    interface = _interface(tmp_path, 'registry_mod', DIGEST_BASE)
    register(interface)

    def forbidden(*_args, **_kwargs):
        raise AssertionError('the registry hit should not reach the artifact')

    monkeypatch.setattr(pine_type_artifact, 'read_artifact', forbidden)
    assert lookup(interface.path, None, PIPELINE_DIGEST) is interface
    assert registered(interface.path) is interface


def __test_a_valid_artifact_answers_without_an_analysis__(tmp_path, monkeypatch):
    """Across processes the artifact stands in for the analysis"""
    path = _write_artifact_for(tmp_path, 'crossproc_mod', ARTIFACT_SOURCE, monkeypatch, '1')
    expected = registered(str(path.resolve()))
    assert expected is not None
    pine_type_artifact._registry.clear()

    calls: list[str] = []

    def spy(target: str):
        calls.append(target)
        return analyse_source(target)

    interface = lookup(str(path), spy, PIPELINE_DIGEST)

    assert calls == []
    assert interface is not None
    assert interface.digest == expected.digest
    assert set(interface.exports) == set(expected.exports)


class _LyingStat:
    """A stat describing no file on disk, for the preliminary-stat check.

    ``lookup`` takes one stat BEFORE either fallback reads a byte, so it can
    only ever answer "has this entry's file moved". Standing in for it with a
    pair no real file has is what proves nothing is CONSTRUCTED from it.
    """
    st_mtime_ns = -1
    st_size = -2


def __test_the_analysis_hands_back_the_fingerprint_it_read__(tmp_path):
    """A tree comes back with the pairing its own bytes were read under"""
    path = _module(tmp_path, 'analyse_pair_mod', ARTIFACT_SOURCE)

    analysed = analyse_source(str(path))
    read = stable_source(path)

    assert analysed is not None and read is not None
    assert analysed[2] == read[1]
    assert source_digest(read[0]) == source_digest(path.read_bytes())


def __test_the_artifact_publishes_the_fingerprint_of_the_bytes_it_matched__(
        tmp_path, monkeypatch):
    """The digest matched one version's bytes; the fingerprint has to be theirs"""
    path = _write_artifact_for(tmp_path, 'artifact_pair_mod', ARTIFACT_SOURCE, monkeypatch, '1')
    on_disk = path.stat()
    pine_type_artifact._registry.clear()
    monkeypatch.setattr(pine_type_artifact, '_stat', lambda _path: _LyingStat())

    interface = lookup(str(path), None, PIPELINE_DIGEST)

    assert interface is not None, 'the artifact was not read'
    assert (interface.mtime_ns, interface.size) == (on_disk.st_mtime_ns, on_disk.st_size)


def __test_the_analysis_publishes_the_fingerprint_of_the_bytes_it_parsed__(
        tmp_path, monkeypatch):
    """Same for the last resort: the signatures and the stat are one read's"""
    path = _module(tmp_path, 'analyse_lookup_mod', ARTIFACT_SOURCE)
    on_disk = path.stat()
    monkeypatch.setattr(pine_type_artifact, 'read_artifact', lambda *_args: None)
    monkeypatch.setattr(pine_type_artifact, '_stat', lambda _path: _LyingStat())

    interface = lookup(str(path), analyse_source, PIPELINE_DIGEST)

    assert interface is not None, 'the analysis was not reached'
    assert (interface.mtime_ns, interface.size) == (on_disk.st_mtime_ns, on_disk.st_size)


def __test_a_stale_artifact_falls_back_to_the_analysis__(tmp_path, monkeypatch):
    """An artifact whose source moved on is ignored, not trusted"""
    path = _write_artifact_for(tmp_path, 'stale_mod', ARTIFACT_SOURCE, monkeypatch, '1')
    pine_type_artifact._registry.clear()
    path.write_text(ARTIFACT_SOURCE.replace('-> int:', '-> float:'), encoding='utf-8')

    calls: list[str] = []

    def spy(target: str):
        calls.append(target)
        return analyse_source(target)

    interface = lookup(str(path), spy, PIPELINE_DIGEST)

    assert calls == [str(path.resolve())]
    assert interface is not None
    assert interface.exports['halve'].ret == FLOAT


def __test_a_changed_source_evicts_the_registry_entry__(tmp_path, monkeypatch):
    """A registry entry answers for the file it was derived from, and no other"""
    # Without the eviction the stale entry would answer forever, and
    # ``dep_record`` would then pair the file's FRESH stat with the OLD digest
    # -- a state that never existed, which the cheap stat check believes.
    path = _module(tmp_path, 'evict_mod', ARTIFACT_SOURCE)
    monkeypatch.setattr(pine_type_artifact, 'read_artifact', lambda *_args: None)
    first = lookup(str(path), analyse_source, PIPELINE_DIGEST)
    assert first is not None and first.exports['halve'].ret == INT

    path.write_text(ARTIFACT_SOURCE.replace('-> int:', '-> float:'), encoding='utf-8')
    second = lookup(str(path), analyse_source, PIPELINE_DIGEST)

    assert second is not None
    assert second.exports['halve'].ret == FLOAT
    assert second.digest != first.digest
    assert registered(str(path)) is second


def __test_the_dependency_record_is_the_interfaces_own_fingerprint__(tmp_path):
    """The record pairs the digest with the stat that digest was derived from"""
    path = _module(tmp_path, 'fingerprint_mod', ARTIFACT_SOURCE)
    interface = _interface(tmp_path, 'fingerprint_mod', ARTIFACT_SOURCE)
    before = path.stat()

    changed = ARTIFACT_SOURCE.replace('-> int:', '-> float:')
    path.write_text(changed, encoding='utf-8')
    record = pine_type_artifact.dep_record(interface)

    assert (record.mtime_ns, record.size) == (before.st_mtime_ns, before.st_size)
    assert record.size != path.stat().st_size
    assert record.digest == interface.digest


# --- the interface's own dependencies --------------------------------------
#
# An export can be INFERRED from a third module: annotated parameters and no
# return annotation take the return from whatever the body calls. Its module's
# source then says nothing about a signature that moved, so the interface has
# to carry the closure it was derived under and a reader has to re-check it.


CHAIN_LEAF = '''"""
@pyne
"""


def cval(x: int) -> {ret}:
    return x{tail}
'''

CHAIN_MIDDLE = '''"""
@pyne
"""
from chain_leaf import cval


def bval(x: int):
    return cval(x)
'''


def __test_the_interface_carries_its_own_dependency_closure__(tmp_path, monkeypatch):
    """What a module was derived from travels with what it publishes"""
    monkeypatch.syspath_prepend(tmp_path)
    _write_artifact_for(tmp_path, 'chain_leaf',
                        CHAIN_LEAF.format(ret='float', tail=' + 0.5'), monkeypatch, '1')
    middle = _write_artifact_for(tmp_path, 'chain_middle', CHAIN_MIDDLE, monkeypatch, '1')
    leaf = str((tmp_path / 'chain_leaf.py').resolve())

    interface = registered(str(middle.resolve()))
    assert interface is not None
    assert interface.exports['bval'].ret == FLOAT
    assert list(interface.deps) == [leaf]
    data = json.loads(artifact_path(middle, PIPELINE_DIGEST).read_text(encoding='utf-8'))
    assert list(data['deps']) == [leaf]


def __test_an_artifact_with_a_stale_dependency_is_not_accepted__(tmp_path, monkeypatch):
    """The middle module's own bytes are untouched; its published return is not"""
    # Its own module names: a finder that still remembers another test's
    # directory would resolve the import there and never see the edit
    monkeypatch.syspath_prepend(tmp_path)
    leaf = _write_artifact_for(tmp_path, 'stale_leaf',
                               CHAIN_LEAF.format(ret='float', tail=' + 0.5'), monkeypatch, '1')
    middle = _write_artifact_for(tmp_path, 'stale_middle',
                                 CHAIN_MIDDLE.replace('chain_leaf', 'stale_leaf'),
                                 monkeypatch, '1')
    before = middle.read_bytes()

    leaf.write_text(CHAIN_LEAF.format(ret='int', tail=' * 2'), encoding='utf-8')
    pine_type_artifact._registry.clear()

    calls: list[str] = []

    def spy(target: str):
        calls.append(target)
        return analyse_source(target)

    interface = lookup(str(middle), spy, PIPELINE_DIGEST)

    assert middle.read_bytes() == before, 'the middle module was not touched'
    assert interface is not None
    assert interface.exports['bval'].ret == INT, 'the stale artifact was believed'
    assert str(middle.resolve()) in calls, 'the middle module was not re-analysed'


REGISTRY_LEAF = '''"""
@pyne
"""


def rval(x: int) -> {ret}:
    return x{tail}
'''

REGISTRY_MIDDLE = '''"""
@pyne
"""
from registry_leaf import rval


def rmid(x: int):
    return rval(x)
'''

REGISTRY_TOP = '''"""
@pyne
"""
from registry_middle import rmid


def rtop(x: int):
    return rmid(x)
'''


def __test_a_registry_hit_revalidates_its_closure__(tmp_path, monkeypatch):
    """The middle module's own bytes never move; the return it INFERRED does"""
    # Its own fingerprint answers for its own source and nothing else. A
    # registry entry handed out on that check alone would keep publishing a
    # return type the leaf stopped having, for the rest of the process.
    monkeypatch.syspath_prepend(tmp_path)
    leaf = _write_artifact_for(tmp_path, 'registry_leaf',
                               REGISTRY_LEAF.format(ret='float', tail=' + 0.5'),
                               monkeypatch, '0')
    middle = _write_artifact_for(tmp_path, 'registry_middle', REGISTRY_MIDDLE, monkeypatch, '0')
    stale = registered(str(middle.resolve()))
    assert stale is not None and stale.exports['rmid'].ret == FLOAT
    before = middle.read_bytes()

    leaf.write_text(REGISTRY_LEAF.format(ret='int', tail=' * 2'), encoding='utf-8')
    fresh = lookup(str(middle), analyse_source, PIPELINE_DIGEST)

    assert middle.read_bytes() == before, 'the middle module was not touched'
    assert fresh is not None
    assert fresh is not stale, 'the registry answered from its own fingerprint alone'
    assert fresh.exports['rmid'].ret == INT
    assert fresh.digest != stale.digest

    # And a module transformed NOW records the interface that is true now
    top = _module(tmp_path, 'registry_top', REGISTRY_TOP)
    _compile(top)
    dependent = registered(str(top.resolve()))
    assert dependent is not None
    assert dependent.deps[str(middle.resolve())].digest == fresh.digest


def __test_a_dependency_cycle_in_the_closure_terminates__(tmp_path, monkeypatch):
    """A module under validation is not current, which is what ends the recursion"""
    monkeypatch.syspath_prepend(tmp_path)
    first = _write_artifact_for(tmp_path, 'cyc_first', '''"""
@pyne
"""
from cyc_second import beta


def alpha(x: int) -> int:
    return beta(x)
''', monkeypatch, '1')
    second = _write_artifact_for(tmp_path, 'cyc_second', '''"""
@pyne
"""
from cyc_first import alpha


def beta(x: int) -> int:
    return x


def gamma(x: int) -> int:
    return alpha(x)
''', monkeypatch, '1')
    # The isolation pass imports the callee's module, so the second module is
    # compiled in its own right and names the first one back: a real cycle
    first_interface = registered(str(first.resolve()))
    second_interface = registered(str(second.resolve()))
    assert first_interface is not None and second_interface is not None
    assert list(first_interface.deps) == [str(second.resolve())]
    assert list(second_interface.deps) == [str(first.resolve())]

    # Both files move without their CONTENT moving, which is the shape that
    # recurses: no stat short-circuit can answer, every artifact still matches
    # its source, so each one's validation reaches the other's
    for path in (first, second):
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns + 10 ** 9, stat.st_mtime_ns + 10 ** 9))
    pine_type_artifact._registry.clear()

    interface = lookup(str(first), analyse_source, PIPELINE_DIGEST)

    assert interface is not None
    assert interface.exports['alpha'].ret == INT


def __test_a_module_being_analysed_answers_nothing__(tmp_path):
    """The guard that makes an import cycle terminate instead of recursing"""
    path = _module(tmp_path, 'cycle_mod', DIGEST_BASE)
    key = str(path.resolve())
    pine_type_artifact._analysing.add(key)

    def forbidden(_target: str):
        raise AssertionError('a module being analysed must not be analysed again')

    try:
        assert lookup(key, forbidden, PIPELINE_DIGEST) is None
    finally:
        pine_type_artifact._analysing.discard(key)


# --- the analyser ---------------------------------------------------------


def __test_analyse_source_derives_what_the_transform_registers__(tmp_path, monkeypatch):
    """The analysis-only half answers exactly what the full pipeline would"""
    path = _write_artifact_for(tmp_path, 'agree_mod', ARTIFACT_SOURCE, monkeypatch, '0')
    compiled = registered(str(path.resolve()))
    assert compiled is not None

    analysed = analyse_source(str(path))
    assert analysed is not None
    tree, table, fingerprint = analysed
    derived = build_interface(tree, table, str(path.resolve()), fingerprint)

    assert derived.digest == compiled.digest
    assert {key: sig.ret for key, sig in table.funcs.items()}['halve'] == INT


def __test_the_published_shape_is_the_modules_own__(tmp_path, monkeypatch):
    """What is published is the module's signature, never the emission's

    The isolation pass prepends a state parameter to every script function and
    the series pass rewrites the parameter annotations, so an interface read off
    the lowered tree describes a function no caller ever writes: one argument
    too many, and every annotation unreadable.
    """
    path = _write_artifact_for(tmp_path, 'shape_edge_mod', EXPORTED, monkeypatch, '0')
    compiled = registered(str(path.resolve()))
    assert compiled is not None

    sig = compiled.exports['scaled']
    assert sig.params == (INT,)
    assert sig.required == 1
    assert sig.annotated is True


def __test_analyse_source_declines_what_is_not_pyne_code__(tmp_path):
    """A plain module publishes nothing, and neither does a missing one"""
    plain = tmp_path / 'not_pyne.py'
    plain.write_text('def f(x):\n    return x\n', encoding='utf-8')

    assert analyse_source(str(plain)) is None
    assert analyse_source(str(tmp_path / 'absent.py')) is None


# --- the pipeline digest --------------------------------------------------


def __test_the_pipeline_digest_is_the_public_face_of_the_hash__():
    """PyneAOT bakes this into its bundle to notice a pipeline it was not built by"""
    assert PIPELINE_DIGEST == _get_transform_pipeline_hash()
    assert len(PIPELINE_DIGEST) == 16
    int(PIPELINE_DIGEST, 16)  # raises if it is not hex
