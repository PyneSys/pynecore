"""
What one module publishes, and how another one finds it again.

The inference is per-module, but a call into an imported module needs that
module's signatures -- and re-deriving them on every import would cost a full
parse per dependency per process. So the answer is published three times over,
each cheaper than the last to reach: a process-wide registry, a JSON artifact
next to the ``.pyc``, and a re-analysis from source as the last resort.

The INTERFACE is what a dependent is allowed to depend on: the exported
signatures, the classes a dependent may annotate with and the module's
``__all__``, and nothing about any body. That is
what makes the dependency check cheap AND precise -- editing a function's body
leaves every dependent's cached bytecode valid, while changing its return
annotation invalidates exactly the dependents that call it.

Two things travel WITH an interface without being part of it: the fingerprint
of the source it was derived from, and the dependency closure the derivation
consulted. Neither is signature -- neither moves the digest -- and both are
what a cached answer has to be checked against before it may be handed out,
the registry's answers included.

Nothing here imports the import hook. The analyser and the pipeline digest are
passed IN, so the analysis stays usable without a loader -- and so the hook can
keep importing this module instead of the other way round.
"""
import ast
import hashlib
import importlib.util
import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from .node_ids import assign_node_ids, node_id
from .pine_type_rules import TY_ATTR, UNKNOWN, ImplSig, annotation_type, impl_sig
from .pine_type_table import (
    Analyser, ClassSig, DepRecord, ExportSig, ModuleInterface, PineTypeTable, Unknown,
    qualify,
)

__all__ = [
    'ARTIFACT_VERSION', 'NO_FINGERPRINT', 'build_interface', 'interface_digest',
    'source_digest', 'stable_source', 'register', 'registered', 'lookup',
    'analysing', 'analysing_scope',
    'dep_record', 'dep_current', 'artifact_path', 'artifact_enabled',
    'write_artifact', 'read_artifact', 'table_json',
]

#: Bumped whenever the JSON shape changes. An artifact of a different version
#: is not read, the same way one of a different pipeline is not.
ARTIFACT_VERSION = 8

#: Environment override for the artifact. ``'1'`` writes one for every Pyne
#: module, ``'0'`` for none; unset leaves it to the module's mode.
ARTIFACT_ENV = 'PYNE_TYPE_ARTIFACT'

#: Length of every digest this module produces. Short on purpose: it names a
#: file and rides in a code constant, and a collision only costs a needless
#: retransform.
_DIGEST_LEN = 16

#: The fingerprint of a source whose bytes no stat could be paired with -- an
#: unreadable file, or one that kept changing under the read. No real file's
#: stat matches it, so an interface carrying it is never handed out by a later
#: fingerprint check; it is the same pairing ``dep_record`` gives an
#: unstat-able dependency.
NO_FINGERPRINT = (0, -1)

#: How many times a stable read is retried before it gives up. A file being
#: rewritten under the reader settles within a rename or two; anything past
#: that is churn no number of retries would outlast.
_STABLE_READ_ATTEMPTS = 3

#: Resolved path -> the interface that module publishes, for this process.
_registry: dict[str, ModuleInterface] = {}

#: Paths whose analysis has not returned yet. An import cycle A -> B -> A
#: reaches ``lookup`` for A while A is still being analysed; answering None
#: there is what terminates it.
_analysing: set[str] = set()


def _key(path: str) -> str:
    """
    The identity a module is registered and looked up under.

    :param path: Any spelling of the source path
    :return: Its resolved form
    """
    return str(Path(path).resolve())


# --- the interface --------------------------------------------------------


def build_interface(tree: ast.Module, table: PineTypeTable, path: str,
                    fingerprint: tuple[int, int] | None = None) -> ModuleInterface:
    """
    Everything a module publishes, derived from its analysed tree.

    Three shapes count as an export, and nothing else does -- a class, a
    ``@udt`` and a module-level variable are not callable contracts:

    * a module-level ``def``, whose LAST definition wins the name the way
      Python's own binding does;
    * a module-level ``@overload`` group, which publishes its implementations;
    * a compiled library's ``X = Exported()`` proxy, whose signature lives on
      the ``@export`` definition nested inside ``main`` -- and which is a group
      of its own when those definitions are ``@overload`` too.

    Which of them a name actually ENDS UP bound to is the table's answer, not
    this one's: ``table.exportable`` is the set whose last module-level binding
    is a definition no branch guards. ``def f`` followed by ``f = other``,
    ``from m import f`` or ``class f`` is not in it -- the importer gets
    whatever the binding put there, and publishing the definition would have
    every dependent type, and pin, against a function they never reach --
    while the same two lines the other way round are. The one exception is the
    proxy, where the assignment IS the export.

    :param tree: The module, after the type pass has stamped it
    :param table: The table that pass produced
    :param path: Resolved source path of the module
    :param fingerprint: The (mtime_ns, size) the analysed bytes were read
                        under, as one indivisible pair; None stats the file
                        now, and ``NO_FINGERPRINT`` says the pairing could not
                        be had at all
    :return: The module's interface, digest included
    """
    exports: dict[str, ExportSig] = {}

    proxies = _exported_proxies(tree)
    definitions: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, str, str]] = []
    _collect_defs(tree, '', definitions)

    # Module level first, so a proxy's nested definition overrules a same-named
    # module-level one -- the proxy IS what the name is bound to at import time
    for node, key, scope in definitions:
        if scope == '' and node.name in table.exportable:
            sig = _export_sig(node, key, table)
            if sig is not None:
                exports[node.name] = sig
    for node, key, scope in definitions:
        if scope != '' and node.name in proxies and _is_exported(node):
            sig = _export_sig(node, key, table)
            if sig is not None:
                exports[node.name] = sig

    if fingerprint is None:
        fingerprint = _fingerprint(path)
    all_names = _module_all(tree)
    interface = ModuleInterface(path=path, exports=exports, all=all_names,
                                classes=_module_classes(tree, all_names, table),
                                extensions={cid: dict(methods)
                                            for cid, methods in table.extensions.items()},
                                digest='', deps=dict(table.deps),
                                mtime_ns=fingerprint[0], size=fingerprint[1],
                                suppressed=table.pins_suppressed.message
                                if table.pins_suppressed is not None else '')
    return replace(interface, digest=interface_digest(interface))


def _stat(path: str) -> os.stat_result | None:
    """
    The source stat, or nothing when the file cannot be reached.

    :param path: Source path of the module
    :return: Its stat, or None
    """
    try:
        return os.stat(path)
    except OSError:
        return None


def _fingerprint(path: str) -> tuple[int, int]:
    """
    The (mtime_ns, size) pair a source is currently at.

    :param path: Source path of the module
    :return: Its fingerprint, ``NO_FINGERPRINT`` when it cannot be stat'd
    """
    stat = _stat(path)
    return NO_FINGERPRINT if stat is None else (stat.st_mtime_ns, stat.st_size)


def stable_source(path: Path) -> tuple[bytes, tuple[int, int]] | None:
    """
    A source's bytes and the fingerprint they belong to, as one pair.

    The fingerprint is what every later check compares against, so it may only
    ever be paired with the bytes it actually describes. Stat'ing around a read
    does NOT give that: an atomic replace landing between the read and the stat
    pairs one version's bytes with another version's fingerprint, and an
    interface built from that pairing is stale in a way the fingerprint check
    then certifies as fresh. Every reader here goes through this -- the loader
    transforming a module, the artifact validating its own source digest, the
    analyser re-deriving a dependency -- because they all publish a fingerprint
    alongside signatures they read.

    The pairing is taken through ONE open file: ``fstat`` before the read and
    again after it, both on the same descriptor, so a replacement is either
    wholly outside the pair (the descriptor keeps reading the file it was
    opened on, whose fingerprint is the one returned) or shows up as a
    difference between the two stats, which is retried. A pair that never
    settles, like a file that cannot be opened at all, has no fingerprint to
    give -- see ``NO_FINGERPRINT``, which no real file's stat matches.

    :param path: Path to the source file
    :return: Its bytes and their ``(mtime_ns, size)``, or None when no such
             pair could be had
    """
    for _ in range(_STABLE_READ_ATTEMPTS):
        try:
            with open(path, 'rb') as handle:
                before = os.fstat(handle.fileno())
                data = handle.read()
                after = os.fstat(handle.fileno())
        except OSError:
            return None
        if (before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size):
            return data, (before.st_mtime_ns, before.st_size)
    return None


def _export_sig(node: ast.FunctionDef | ast.AsyncFunctionDef, key: str,
                table: PineTypeTable) -> ExportSig | None:
    """
    The published signature of one definition.

    :param node: The definition
    :param key: Its scope-qualified id in the table
    :param table: The module's type table
    :return: The signature, or None when the inference never saw the definition
    """
    func = table.funcs.get(key)
    if func is None:
        return None
    impls = table.groups.get(key, ())
    if impls:
        first = impls[0]
        return ExportSig(
            name=node.name, kind='group', params=first.params, required=first.required,
            open_ended=first.open_ended, ret=func.ret,
            annotated=all(UNKNOWN not in impl.params for impl in impls),
            impls=impls, line=getattr(node, 'lineno', 0), names=first.names)
    sig = impl_sig(node, func.ret, classes=table.classes)
    positional = list(node.args.posonlyargs) + list(node.args.args)
    return ExportSig(
        name=node.name, kind='function', params=sig.params, required=sig.required,
        open_ended=sig.open_ended, ret=sig.ret,
        annotated=all(annotation_type(arg.annotation, table.classes) != UNKNOWN
                      for arg in positional),
        line=getattr(node, 'lineno', 0), names=sig.names)


def _collect_defs(node: ast.AST, scope: str,
                  out: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, str, str]]) -> None:
    """
    Every definition of a tree, with the scope-qualified id the table keys it by.

    :param node: The node to descend into
    :param scope: Scope id its children live in, empty at module level
    :param out: Collects (definition, its id, the scope it was declared in), in
                source order
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            key = qualify(scope, child.name)
            out.append((child, key, scope))
            _collect_defs(child, key, out)
        elif isinstance(child, ast.ClassDef):
            _collect_defs(child, qualify(scope, child.name), out)
        else:
            _collect_defs(child, scope, out)


def _exported_proxies(tree: ast.Module) -> set[str]:
    """
    The names a compiled library binds an ``Exported()`` proxy to.

    :param tree: The module
    :return: Every module-level name assigned an ``Exported()`` call
    """
    names: set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            targets, value = stmt.targets, stmt.value
        elif isinstance(stmt, ast.AnnAssign):
            targets, value = [stmt.target], stmt.value
        else:
            continue
        if not isinstance(value, ast.Call):
            continue
        func = value.func
        called = func.id if isinstance(func, ast.Name) else \
            (func.attr if isinstance(func, ast.Attribute) else '')
        if called != 'Exported':
            continue
        names.update(target.id for target in targets if isinstance(target, ast.Name))
    return names


def _is_exported(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """
    Whether a definition carries the ``@export`` decorator.

    :param node: The definition to inspect
    :return: True when one of its decorators is named ``export``
    """
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name) and target.id == 'export':
            return True
        if isinstance(target, ast.Attribute) and target.attr == 'export':
            return True
    return False


def _module_all(tree: ast.Module) -> tuple[str, ...] | None:
    """
    The module's literal ``__all__``, when it spells one.

    :param tree: The module
    :return: The names it lists, or None when there is no literal ``__all__``
    """
    found: tuple[str, ...] | None = None
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            targets, value = stmt.targets, stmt.value
        elif isinstance(stmt, ast.AnnAssign):
            targets, value = [stmt.target], stmt.value
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id == '__all__' for t in targets):
            continue
        if not isinstance(value, (ast.List, ast.Tuple)):
            continue
        found = tuple(element.value for element in value.elts
                      if isinstance(element, ast.Constant) and isinstance(element.value, str))
    return found


def _module_classes(tree: ast.Module, all_names: tuple[str, ...] | None,
                    table: PineTypeTable) -> dict[str, ClassSig]:
    """
    The classes a module publishes, in source order, with what they hold.

    Module level only: a class nested in a function is not reachable through
    the import, so no dependent can name it in an annotation. ``__all__``
    filters them for the same reason it filters the exports -- a namespace
    import reads the module through it.

    What travels is the whole class: its id, its field types and the methods
    declared on it. A dependent reading ``pivot.price`` needs the field's
    type, and it can only get it from here -- the class is the contract, not
    just its name.

    :param tree: The module
    :param all_names: Its literal ``__all__``, or None when it spells none
    :param table: The table the type pass produced, which holds the classes
    :return: Class name -> what it declares
    """
    published = None if all_names is None else set(all_names)
    out: dict[str, ClassSig] = {}
    for stmt in tree.body:
        if not isinstance(stmt, ast.ClassDef):
            continue
        if published is not None and stmt.name not in published:
            continue
        sig = table.class_sigs.get(table.classes.get(stmt.name, ''))
        if sig is not None:
            out[stmt.name] = sig
    return out


def interface_digest(interface: ModuleInterface) -> str:
    """
    Digest of what a module publishes, blind to how it publishes it.

    The line numbers are left out on purpose: a body edit moves every
    definition below it, and a dependent has no business being invalidated by
    that. What IS in here is every signature, every implementation of every
    group, ``__all__``, the published classes and the methods this module adds
    to another module's class -- adding or removing one changes what a
    dependent's annotations, and its method calls, resolve to.

    :param interface: The interface to digest
    :return: A short hex digest
    """
    payload = {
        'all': list(interface.all) if interface.all is not None else None,
        # The class id is left out on the same grounds the line numbers are:
        # its module half IS this interface's own path, so it says nothing a
        # dependent could be invalidated by. The FIELDS are the contract --
        # a field whose type moves changes what every reader of it resolves to
        # The fields as an ORDERED list: a constructor binds them by position
        'classes': {name: {'fields': list(sig.fields.items()), 'required': sig.required,
                           'methods': {method: _unlined(_export_json(published))
                                       for method, published in sig.methods.items()}}
                    for name, sig in interface.classes.items()},
        # An extension is keyed by the FOREIGN class id, whose module half is
        # another module's path -- that is the identity, and a dependent
        # resolving a method on that class does depend on it
        'extensions': {cid: {name: _unlined(_export_json(published))
                             for name, published in methods.items()}
                       for cid, methods in interface.extensions.items()},
        'exports': {name: _unlined(_export_json(sig))
                    for name, sig in interface.exports.items()},
        # Whether the module's pins were given up is part of what a dependent
        # resolves to: its own pins follow
        'suppressed': interface.suppressed,
    }
    return _digest(_canonical(payload).encode('utf-8'))


def _unlined(payload: dict) -> dict:
    """
    One signature with its line number taken out.

    A body edit moves every definition below it, and a dependent has no
    business being invalidated by that.

    :param payload: The serialized signature
    :return: The same, without ``line``
    """
    return {field: value for field, value in payload.items() if field != 'line'}


def source_digest(source: bytes) -> str:
    """
    Digest of a module's source bytes.

    :param source: The raw file contents
    :return: A short hex digest
    """
    return _digest(source)


def _digest(data: bytes) -> str:
    """
    The one hash every digest here is built with.

    :param data: The bytes to digest
    :return: A short hex digest
    """
    return hashlib.sha256(data).hexdigest()[:_DIGEST_LEN]


def _canonical(payload: object) -> str:
    """
    The one JSON spelling a digest may be taken over.

    :param payload: The structure to dump
    :return: Sorted, whitespace-free JSON
    """
    return json.dumps(payload, sort_keys=True, separators=(',', ':'))


# --- the registry ---------------------------------------------------------


def register(interface: ModuleInterface) -> None:
    """
    Publish a module's interface for the rest of this process.

    :param interface: The interface to register
    """
    _registry[_key(interface.path)] = interface


def registered(path: str) -> ModuleInterface | None:
    """
    The interface a module already published in this process.

    :param path: Source path of the module
    :return: Its interface, or None
    """
    return _registry.get(_key(path))


def lookup(path: str, analyse: Analyser | None, pipeline_hash: str) -> ModuleInterface | None:
    """
    A module's interface, from wherever it is cheapest to get.

    The registry answers within a process, the artifact across processes, and
    a re-analysis when neither does. A failed analysis is NOT remembered: the
    file may be written, fixed or restored a moment later, and a cached "no"
    would outlive the reason for it.

    Both cached answers are checked against the file before they are handed
    out, and both are checked the same way: the module's own fingerprint --
    one ``os.stat``, against the one the interface itself carries -- AND its
    whole dependency closure, because an inferred signature can move without a
    single byte of this module changing. A registry entry is not exempt from
    the second half: a module registered early in a process keeps answering
    for the rest of it, and an edit to a module its exports were INFERRED from
    lands nowhere near its own source. Whichever check fails, the entry is
    evicted and the answer re-derived rather than handed out stale.

    :param path: Source path of the module
    :param analyse: Re-derives the tree and table from source; None to look no
                    further than the artifact
    :param pipeline_hash: Digest of the pipeline an artifact must come from
    :return: The interface, or None when it cannot be had
    """
    key = _key(path)
    # A module still being analysed cannot answer for itself; saying so is
    # what makes an import cycle terminate instead of recursing
    if key in _analysing:
        return None

    # This stat decides EVICTION and nothing else. It is taken before either
    # fallback reads a byte, so it describes the file as it was then -- fine
    # for "has the entry's file moved", and never a fingerprint to publish:
    # each fallback pairs its own read with its own stat instead
    stat = _stat(key)
    hit = _registry.get(key)
    if hit is not None:
        if stat is not None and (hit.mtime_ns, hit.size) == (stat.st_mtime_ns, stat.st_size) \
                and _closure_current(key, hit, analyse, pipeline_hash):
            return hit
        del _registry[key]
    if stat is None:
        return None

    # Everything below may reach back here through a dependency's own
    # validation; the mark is what makes a cycle in the closure answer None
    # -- a module under analysis is NOT current -- instead of recursing
    with analysing_scope(key):
        interface = _from_artifact(key, analyse, pipeline_hash)
        if interface is None and analyse is not None:
            analysed = analyse(key)
            if analysed is not None:
                tree, table, fingerprint = analysed
                interface = build_interface(
                    tree, table, key,
                    NO_FINGERPRINT if fingerprint is None else fingerprint)
    if interface is None:
        return None
    register(interface)
    return interface


def _from_artifact(key: str, analyse: Analyser | None,
                   pipeline_hash: str) -> ModuleInterface | None:
    """
    The interface an artifact carries, when it still describes the world.

    Two things have to hold, and the second is the one a source digest cannot
    see: the module's own bytes must be the ones the artifact was written
    from, AND every module the analysis consulted must still say what it said
    then. An export whose return was INFERRED from a call into a third module
    moves when that module's signature moves, while this file stays untouched.

    The source is read through ``stable_source``, so the fingerprint the
    interface goes out under is the one belonging to the very bytes whose
    digest just matched -- not to whatever the file was before the read.

    :param key: Resolved source path of the module
    :param analyse: Re-derives a dependency that the stat check cannot clear
    :param pipeline_hash: Digest of the pipeline the artifact must come from
    :return: The interface, or None when the artifact may not be trusted
    """
    data = read_artifact(Path(key), pipeline_hash)
    if data is None:
        return None
    stable = stable_source(Path(key))
    if stable is None:
        return None
    source, fingerprint = stable
    if data.get('src') != source_digest(source):
        return None
    interface = _interface_from_json(key, data, fingerprint)
    if not _closure_current(key, interface, analyse, pipeline_hash):
        return None
    return interface


def _closure_current(key: str, interface: ModuleInterface, analyse: Analyser | None,
                     pipeline_hash: str) -> bool:
    """
    Whether every module an interface was derived from still says the same thing.

    One ``os.stat`` per closure member, and nothing more for the members that
    did not move -- which is all of them on an ordinary import. The closure is
    transitive, so this is the whole question: a third module's signature
    moving is a record of its own here, not a change hiding behind an
    untouched file.

    The check runs under the module's OWN analysing mark, because validating a
    dependency may reach back here for this very module. A module under
    analysis answers None, which makes it not current -- the conservative end
    of a cycle, and the one that terminates.

    :param key: Resolved source path of the module the interface belongs to
    :param interface: The interface whose closure is being checked
    :param analyse: Re-derives a dependency the stat check cannot clear
    :param pipeline_hash: Digest of the pipeline an artifact must come from
    :return: True while every dependency still matches what was recorded
    """
    if not interface.deps:
        return True
    with analysing_scope(key):
        return all(dep_current(record, analyse, pipeline_hash)
                   for record in interface.deps.values())


def analysing(path: str) -> bool:
    """
    Whether a module's own analysis is on the stack right now.

    This is what tells an import CYCLE apart from a module that simply has no
    interface to give. Both make ``lookup`` answer None, and only one of them
    is worth telling the user about.

    :param path: Source path of the module, empty when it has none
    :return: True while it is being analysed
    """
    return bool(path) and _key(path) in _analysing


@contextmanager
def analysing_scope(path: str) -> Iterator[None]:
    """
    Mark a module as being analysed for the duration of a walk.

    The walk itself enters here, not just ``lookup``: a module reached through
    a cycle must answer None even when its analysis was started by the loader
    rather than by another module's lookup. Re-entering an already marked path
    is a no-op, so the outer scope stays the one that unmarks it.

    :param path: Source path of the module being analysed, empty when it has
                 none -- an in-memory tree marks nothing
    """
    if not path:
        yield
        return
    key = _key(path)
    if key in _analysing:
        yield
        return
    _analysing.add(key)
    try:
        yield
    finally:
        _analysing.discard(key)


# --- the dependency records -----------------------------------------------


def dep_record(interface: ModuleInterface) -> DepRecord:
    """
    The state of one dependency, as the dependent's bytecode remembers it.

    The fingerprint is the interface's OWN, never a fresh stat. Stat'ing again
    here would pair the file as it is now with a digest derived from the file
    as it was: the dependent would then remember a state that never existed,
    and the cheap stat check would keep accepting the stale signatures for as
    long as nobody touched the file again.

    :param interface: The interface the dependent was built against
    :return: The record to bake into the dependent
    """
    return DepRecord(path=_key(interface.path), mtime_ns=interface.mtime_ns,
                     size=interface.size, digest=interface.digest)


def dep_current(record: DepRecord, analyse: Analyser | None, pipeline_hash: str) -> bool:
    """
    Whether a dependency still means what the dependent was built against.

    The stat pair is checked first and answers on its own: an untouched file
    costs one ``os.stat`` and no parsing at all, which is the case every
    ordinary import is. Only a file that moved is worth re-deriving an
    interface for -- and a body edit lands here and still says yes.

    That short-circuit is only sound because the closure is TRANSITIVE: a
    dependent records its dependencies' dependencies too, so a third module's
    signature moving is a record of its own here rather than a change hiding
    behind an untouched file.

    :param record: What the dependent remembers
    :param analyse: Re-derives the dependency when the artifact cannot
    :param pipeline_hash: Digest of the pipeline an artifact must come from
    :return: True while the dependent's bytecode is still valid
    """
    try:
        stat = os.stat(record.path)
    except OSError:
        return False
    if stat.st_mtime_ns == record.mtime_ns and stat.st_size == record.size:
        return True
    interface = lookup(record.path, analyse, pipeline_hash)
    return interface is not None and interface.digest == record.digest


# --- the artifact ---------------------------------------------------------


def artifact_enabled(pyne_mode: str | None) -> bool:
    """
    Whether this module's types are worth writing out.

    ``@pyne edge`` is the compiler's own output, which is what the AOT front
    end consumes, so that is the default. The environment overrides it either
    way -- writing one for a hand-written script is how the types are
    inspected during development.

    :param pyne_mode: The module's mode word, None for a hand-written script
    :return: Whether to write the artifact
    """
    override = os.environ.get(ARTIFACT_ENV)
    if override == '1':
        return True
    if override == '0':
        return False
    return pyne_mode == 'edge'


def artifact_path(source_path: Path, pipeline_hash: str) -> Path:
    """
    Where a module's type artifact lives.

    Beside the ``.pyc``, and named after the pipeline that produced it, so an
    artifact of a different pipeline is a different file and never has to be
    invalidated. The cache directory comes from ``importlib`` itself, so a
    ``sys.pycache_prefix`` tree is honoured exactly as it is for bytecode.

    The path is resolved first: the writer is handed whatever spelling the
    loader was given, the reader a resolved dependency path, and both have to
    name the same file.

    :param source_path: Path to the ``.py`` source
    :param pipeline_hash: Digest of the transform pipeline
    :return: Path to the artifact file
    """
    resolved = source_path.resolve()
    cache = Path(importlib.util.cache_from_source(str(resolved)))
    return cache.parent / f'{resolved.stem}.{pipeline_hash[:12]}.pynetypes.json'


def write_artifact(tree: ast.Module, table: PineTypeTable, interface: ModuleInterface,
                   source: bytes, path: Path, pipeline_hash: str) -> None:
    """
    Write a module's types next to its bytecode, best effort.

    Best effort is the whole contract: the artifact is a cache, and a
    read-only install, a full disk or a racing process may all deny it. None
    of that may fail an import, so every write error is swallowed -- the types
    are simply re-derived next time.

    :param tree: The finished tree, which is renumbered before it is listed
    :param table: The type table of the module
    :param interface: What the module publishes
    :param source: The module's raw source bytes
    :param path: Path to the ``.py`` source
    :param pipeline_hash: Digest of the transform pipeline
    """
    if sys.dont_write_bytecode:
        return
    target = artifact_path(path, pipeline_hash)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_canonical(
            table_json(tree, table, interface, source, pipeline_hash)), encoding='utf-8')
    except OSError:
        pass


def read_artifact(path: Path, pipeline_hash: str) -> dict | None:
    """
    Read back a module's artifact, when there is a usable one.

    :param path: Path to the ``.py`` source
    :param pipeline_hash: Digest of the transform pipeline it must come from
    :return: The parsed artifact, or None
    """
    try:
        data = json.loads(artifact_path(path, pipeline_hash).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get('v') != ARTIFACT_VERSION or data.get('pipeline') != pipeline_hash:
        return None
    return data


def table_json(tree: ast.Module, table: PineTypeTable, interface: ModuleInterface,
               source: bytes, pipeline_hash: str) -> dict:
    """
    The whole analysis of one module, as the artifact spells it.

    The expression list is taken over the FINAL tree and renumbered right
    here: the ids the inference handed out describe a tree the lowering has
    since rewritten, so they would name nothing a reader could find. The fresh
    pre-order numbering is what makes two artifacts of the same source
    diffable.

    :param tree: The finished tree
    :param table: The type table of the module
    :param interface: What the module publishes
    :param source: The module's raw source bytes
    :param pipeline_hash: Digest of the transform pipeline
    :return: The JSON-ready structure
    """
    assign_node_ids(tree)
    exprs = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.expr):
            continue
        ty = getattr(node, TY_ATTR, None)
        if ty is None:
            continue
        exprs.append([node_id(node), ty, getattr(node, 'lineno', 0),
                      getattr(node, 'col_offset', 0)])
    exprs.sort(key=lambda entry: entry[0])

    return {
        'v': ARTIFACT_VERSION,
        'module': interface.path or table.module_path,
        'src': source_digest(source),
        'pipeline': pipeline_hash,
        'exprs': exprs,
        'bindings': {scope: {name: _binding_json(binding)
                             for name, binding in names.items()}
                     for scope, names in table.bindings.items()},
        'funcs': {key: {'params': list(func.params), 'ret': func.ret, 'line': func.line}
                  for key, func in table.funcs.items()},
        'contexts': [{'cid': result.cid, 'key': result.key, 'params': list(result.params),
                      'ret': result.ret, 'pins': _pins_json(result.pins, table)}
                     for result in table.contexts.values()],
        'calls': [{'callee': call.callee, 'line': call.line, 'col': call.col,
                   'argc': call.argc, 'ty': call.ty, 'pin': call.pin}
                  for call in table.calls],
        'diags': [{'message': diag.message, 'line': diag.line, 'col': diag.col,
                   'origin': _unknown_json(diag.origin), 'fix': diag.fix,
                   'end_line': diag.end_line, 'end_col': diag.end_col}
                  for diag in table.diags],
        # The module's dependency closure, which is also the interface's: a
        # reader validates it before it may trust the published signatures
        'deps': {path: {'mtime_ns': record.mtime_ns, 'size': record.size,
                        'digest': record.digest}
                 for path, record in table.deps.items()},
        'interface': {
            'all': list(interface.all) if interface.all is not None else None,
            'classes': {name: _class_json(sig) for name, sig in interface.classes.items()},
            'extensions': {cid: {name: _export_json(published)
                                 for name, published in methods.items()}
                           for cid, methods in interface.extensions.items()},
            'exports': {name: _export_json(sig) for name, sig in interface.exports.items()},
            'suppressed': interface.suppressed,
        },
    }


def _pins_json(pins: dict[int, str | None], table: PineTypeTable) -> list[dict]:
    """
    One context's overload pins, keyed by where the call STANDS.

    The node ids the inference hands out describe the tree it walked, and the
    lowering builds new call nodes over it -- so an id written out here would
    name nothing a reader could find, not even in this same artifact, whose
    expression list is renumbered over the FINAL tree. The source position is
    the one identity both trees agree on, and it is what ``calls`` is listed
    by too.

    :param pins: Call node id -> the pin this context justified there
    :param table: The module's type table, for the positions
    :return: One entry per pinned call, in source order
    """
    entries = [{'line': table.call_pos[nid][0], 'col': table.call_pos[nid][1], 'pin': pin}
               for nid, pin in pins.items() if nid in table.call_pos]
    entries.sort(key=lambda entry: (entry['line'], entry['col']))
    return entries


def _binding_json(binding) -> dict:
    """
    One binding, with its provenance when it has one.

    :param binding: The binding to serialize
    :return: Its JSON form
    """
    out: dict = {'ty': binding.ty, 'line': binding.line}
    if binding.unknown is not None:
        out['unknown'] = _unknown_json(binding.unknown)
    if binding.series:
        out['series'] = True
    return out


def _unknown_json(unknown: Unknown | None) -> dict | None:
    """
    Where a type was lost, when that is recorded.

    :param unknown: The provenance, or None
    :return: Its JSON form, or None
    """
    if unknown is None:
        return None
    return {'reason': unknown.reason, 'line': unknown.line, 'col': unknown.col,
            'detail': unknown.detail}


def _export_json(sig: ExportSig) -> dict:
    """
    One published signature, in the artifact's shape.

    :param sig: The signature to serialize
    :return: Its JSON form
    """
    return {
        'kind': sig.kind, 'params': list(sig.params), 'required': sig.required,
        'open_ended': sig.open_ended, 'ret': sig.ret, 'annotated': sig.annotated,
        'impls': [{'params': list(impl.params), 'required': impl.required,
                   'open_ended': impl.open_ended, 'ret': impl.ret, 'fits': impl.fits,
                   'names': list(impl.names)}
                  for impl in sig.impls],
        'line': sig.line, 'names': list(sig.names),
    }


def _interface_from_json(path: str, data: dict,
                         fingerprint: tuple[int, int]) -> ModuleInterface:
    """
    Rebuild an interface an artifact carries.

    The digest is recomputed rather than read back, so a hand-edited or
    truncated artifact cannot make a dependency look current. The fingerprint
    is NOT read back either: it is the one paired with the bytes the caller
    just matched against the artifact, which is the only pairing of digest and
    stat that ever held on this machine.

    :param path: Resolved source path of the module
    :param data: The whole artifact -- the published section and the
                 dependency closure it was derived under
    :param fingerprint: Of the source bytes the artifact was just validated
                        against
    :return: The interface it describes
    """
    published = data.get('interface') or {}
    all_names = published.get('all')
    exports = {name: _export_from_json(name, sig)
               for name, sig in (published.get('exports') or {}).items()}
    deps = {}
    for dep_path, record in (data.get('deps') or {}).items():
        deps[dep_path] = DepRecord(path=dep_path, mtime_ns=record.get('mtime_ns', 0),
                                   size=record.get('size', -1),
                                   digest=record.get('digest', ''))
    classes = {name: _class_from_json(name, sig)
               for name, sig in (published.get('classes') or {}).items()}
    extensions = {cid: {name: _export_from_json(name, method)
                        for name, method in (methods or {}).items()}
                  for cid, methods in (published.get('extensions') or {}).items()}
    interface = ModuleInterface(
        path=path, exports=exports,
        all=tuple(all_names) if all_names is not None else None,
        classes=classes, extensions=extensions, digest='', deps=deps,
        mtime_ns=fingerprint[0], size=fingerprint[1],
        suppressed=str(published.get('suppressed') or ''))
    return replace(interface, digest=interface_digest(interface))


def _export_from_json(name: str, data: dict) -> ExportSig:
    """
    Rebuild one published signature an artifact carries.

    :param name: The exported name
    :param data: Its JSON form
    :return: The signature
    """
    return ExportSig(
        name=name, kind=data.get('kind', 'function'),
        params=tuple(data.get('params') or ()), required=data.get('required', 0),
        open_ended=bool(data.get('open_ended')), ret=data.get('ret', UNKNOWN),
        annotated=bool(data.get('annotated')),
        impls=tuple(ImplSig(params=tuple(impl.get('params') or ()),
                            required=impl.get('required', 0),
                            open_ended=bool(impl.get('open_ended')),
                            ret=impl.get('ret', UNKNOWN), fits=impl.get('fits', ''),
                            names=tuple(impl.get('names') or ()))
                    for impl in (data.get('impls') or ())),
        line=data.get('line', 0), names=tuple(data.get('names') or ()))


def _class_json(sig: ClassSig) -> dict:
    """
    One published class, in the artifact's shape.

    :param sig: The class to serialize
    :return: Its JSON form
    """
    return {
        'id': sig.id,
        'fields': dict(sig.fields),
        # The artifact is written with sorted keys, so the declaration order
        # a constructor binds positional arguments by travels on its own
        'order': list(sig.fields),
        'required': sig.required,
        'methods': {name: _export_json(method) for name, method in sig.methods.items()},
    }


def _class_from_json(name: str, data: dict) -> ClassSig:
    """
    Rebuild one published class an artifact carries.

    :param name: The class name
    :param data: Its JSON form
    :return: The class
    """
    fields = dict(data.get('fields') or {})
    ordered = {field_name: fields[field_name]
               for field_name in (data.get('order') or ()) if field_name in fields}
    ordered.update(fields)
    return ClassSig(
        name=name, id=data.get('id', ''), fields=ordered,
        required=data.get('required', 0),
        methods={method: _export_from_json(method, sig)
                 for method, sig in (data.get('methods') or {}).items()})
