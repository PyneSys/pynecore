"""
What one module publishes, and how another one finds it again.

The inference is per-module, but a call into an imported module needs that
module's signatures -- and re-deriving them on every import would cost a full
transform per dependency per process. So the answer is published twice over,
the cheaper one first: a process-wide registry, and a constant baked into the
module's own ``.pyc``, which the import hook reads back without executing
anything. A module that has neither is transformed into its ``.pyc`` the way
importing it would, once, and answers from that transform.

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

A dependent's bytecode depends on one more thing, which is not a type and so
has a digest of its own: how each call into the module is ROUTED. The
lowering emits a call into a state-carrying function with the hidden state
argument and a call into a stateless one plainly, and which one a function is
can change with nothing but its body. The routes digest is settled by the
lowering, so an interface published off the analysis alone carries none; a
dependency record compares it beside the interface digest wherever the record
is checked for a dependent's bytecode.

Nothing here imports the import hook. What reads a ``.pyc`` and what
transforms a module into one is passed IN (an ``Analyser``), so the analysis
stays usable without a loader -- and so the hook can keep importing this
module instead of the other way round.
"""
import ast
import hashlib
import json
import os
import zlib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

from . import ast_walk
from .pine_type_rules import UNKNOWN, ImplSig, annotation_type, impl_sig
from .pine_type_table import (
    Analyser, ClassSig, DepRecord, ExportSig, ModuleInterface, PineTypeTable, qualify,
)
from .source_read import stable_source

__all__ = [
    'NO_FINGERPRINT', 'build_interface', 'interface_digest',
    'routes_shape', 'routes_digest', 'CallPlan', 'call_plan', 'call_routes', 'STAR',
    'StaticInterface', 'static_interface', 'read_static_interface',
    'ROUTE_SKIP', 'ROUTE_DIRECT', 'ROUTE_FAST', 'ROUTE_FAST_SHARED', 'ROUTE_UNIFORM',
    'source_digest', 'register', 'registered', 'lookup',
    'closes_cycle', 'analysing_scope', 'Compile', 'compiling', 'reached_back',
    'SOURCE_DIGEST', 'dep_record', 'source_record', 'add_dep', 'dep_current', 'settle',
    'interface_payload', 'interface_from_payload',
]

#: Length of every digest this module produces. Short on purpose: it rides in a
#: code constant, and a collision only costs a needless retransform.
_DIGEST_LEN = 16

#: The fingerprint of a source whose bytes no stat could be paired with -- an
#: unreadable file, or one that kept changing under the read. No real file's
#: stat matches it, so an interface carrying it is never handed out by a later
#: fingerprint check; it is the same pairing ``dep_record`` gives an
#: unstat-able dependency.
NO_FINGERPRINT = (0, -1)

#: The digest of a dependency record that stands for a plain Python module: it
#: publishes no interface to re-derive, so the record holds exactly as long as
#: the file's stat does. No derived digest is spelled like it.
SOURCE_DIGEST = 'source'

#: Resolved path -> the interface that module publishes, for this process.
_registry: dict[str, ModuleInterface] = {}

#: Paths whose analysis has not returned yet. An import cycle A -> B -> A
#: reaches ``lookup`` for A while A is still being analysed; answering None
#: there is what terminates it.
_analysing: set[str] = set()


@dataclass(slots=True)
class Compile:
    """
    One transform of a dependency that a lookup started, while it runs.

    Such a transform runs in the middle of ANOTHER module's: typically inside
    the type pass of the module that imports it. It is only the transform
    importing the dependency would produce while it reaches none of the
    modules under analysis around it. One that does -- an import cycle back
    into them -- gets no interface where the real import, which runs once
    their analysis is done, gets one; and its lowering would import what the
    module calls, among them a module whose own analysis has not returned.
    So it stops once its analysis is done (see ``reached_back``): its types
    are still the best answer there is, the rest is left to the import.
    """
    #: Resolved source path of the module being transformed
    path: str
    #: The modules whose analysis was on the stack when the transform began
    outer: frozenset[str]
    #: Whether the transform has reached one of them since
    cyclic: bool = False


#: The dependency transforms in progress, innermost last
_compiles: list[Compile] = []


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
    :return: The module's interface, digest included; its routes are left to
             the lowering, which is the only thing that can settle them
    """
    exports: dict[str, ExportSig] = {}

    proxies = _exported_proxies(tree)
    definitions: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, str, str]] = []
    _collect_defs(tree, '', definitions)

    # Whether each export ends up bound to its proxy: an ``@export`` definition
    # hands itself to the proxy and binds the proxy back, any other module-level
    # definition binds the plain function over it
    proxied: dict[str, bool] = {}
    # Module level first, so a proxy's nested definition overrules a same-named
    # module-level one -- the proxy IS what the name is bound to at import time
    for node, key, scope in definitions:
        if scope == '' and node.name in table.exportable:
            sig = _export_sig(node, key, table)
            if sig is not None:
                exports[node.name] = sig
                proxied[node.name] = node.name in proxies and _is_exported(node)
    for node, key, scope in definitions:
        if scope != '' and node.name in proxies and _is_exported(node):
            sig = _export_sig(node, key, table)
            if sig is not None:
                exports[node.name] = sig
                proxied[node.name] = True

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
                                if table.pins_suppressed is not None else '',
                                proxies=tuple(sorted(name for name, bound in proxied.items()
                                                     if bound)))
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
    # A definition is a statement, and no expression holds one
    for child in ast_walk.iter_child_statements(node):
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
    group, ``__all__``, the published classes, the methods this module adds
    to another module's class and the exports a method call can reach by name
    -- adding or removing one changes what a dependent's annotations, and its
    method calls, resolve to.

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
                                       for method, published in sig.methods.items()},
                           'members': {member: _unlined(_export_json(published))
                                       for member, published in sig.members.items()}}
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
        # Which library answers a method called by name on a receiver without a
        # class is decided among these
        'proxies': list(interface.proxies),
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


# --- the routes -----------------------------------------------------------


def routes_shape(tree: ast.Module) -> str:
    """
    The module-level structure a call into the module is routed by, bodies left out.

    A dependent's lowering classifies a call into this module by what the
    imported name IS when the module has run: a plain function, an overload
    dispatcher, an ``Exported`` proxy, a module property, a class, or whatever
    an assignment or an import bound it to. Every one of those is decided by a
    statement outside any function body -- a definition's decorators, an
    assignment, an import, a class body -- so the shape is the module with
    every function body left out, and an edit inside one never moves it.
    Docstrings and source positions are left out too: neither routes a call.

    It is taken off the ANALYSED tree, before the lowering adds plumbing of
    its own, which follows the bodies.

    :param tree: The analysed module
    :return: Its shape, as canonical text
    """
    return _canonical([_shape(stmt) for stmt in tree.body if not _is_docstring(stmt)])


#: The nodes a statement list holds, which hold statement lists of their own
_STATEMENT_LIKE = (ast.stmt, ast.excepthandler, ast.match_case)


def _shape(node: ast.AST) -> object:
    """
    One statement of the routes shape.

    :param node: A statement, ``except`` handler or ``match`` case
    :return: Its JSON-ready shape
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        # What a call reaches is the decorators' return value; the body and
        # the parameters (the interface's business) do not decide that
        return ['def', node.name, [ast.dump(decorator) for decorator in node.decorator_list]]
    shaped: list[object] = [type(node).__name__]
    for name, value in ast.iter_fields(node):
        if isinstance(value, list):
            shaped.append([name, [_shape(item) if isinstance(item, _STATEMENT_LIKE)
                                  else ast.dump(item) if isinstance(item, ast.AST) else item
                                  for item in value if not _is_docstring(item)]])
        elif isinstance(value, ast.AST):
            shaped.append([name, ast.dump(value)])
        else:
            shaped.append([name, value])
    return shaped


def _is_docstring(node: object) -> bool:
    """
    Whether a statement is a bare string literal.

    :param node: The statement
    :return: True for a docstring, or any other string standing as a statement
    """
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
        and isinstance(node.value.value, str)


class _Layout(Protocol):
    """What :func:`routes_digest` reads of a ``slot_layout.ModuleLayout``.

    Spelled out here rather than imported, so the type pass, which imports this
    module, does not depend on the lowering's slot allocator.
    """

    def scope_segment(self, node: ast.FunctionDef) -> str: ...

    def state_carrying(self, scope_id: str) -> bool: ...


def routes_digest(shape: str, tree: ast.Module, layout: _Layout) -> str:
    """
    Digest of how every call into a module is routed, once its lowering settled it.

    The shape says what each module-level name is; the lowering adds the one
    fact no statement spells: which definitions carry state. A dependent calls
    such a function on the fast route, with the hidden state argument in
    front, and one without plainly -- and neither emission runs against the
    other kind of function. Statefulness is only settled by the lowering, as
    it follows calls: a function that calls a state-carrying one carries
    state itself.

    The definitions counted are the ones an importer can reach, the same the
    shape lists: module-level ones, and those in a class body or a
    module-level branch. A definition nested in a function is out of every
    importer's reach, and so is whether it keeps state.

    :param shape: ``routes_shape`` of the analysed tree
    :param tree: The same tree after the lowering
    :param layout: The slot layout that lowering allocated
    :return: A short hex digest
    """
    # Only a plain ``def`` is given a state parameter; an ``async def`` never is
    carriers = [[node.name, isinstance(node, ast.FunctionDef)
                 and layout.state_carrying(layout.scope_segment(node))]
                for node in _reachable_defs(tree)]
    return _digest(_canonical([shape, carriers]).encode('utf-8'))


def _reachable_defs(node: ast.AST) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef]:
    """
    The definitions of a module that do not stand inside another definition.

    :param node: The node to descend into
    :return: Every such definition, in source order
    """
    for child in ast_walk.iter_child_statements(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield child
        else:
            yield from _reachable_defs(child)


# --- the call routes ------------------------------------------------------

#: The routes a dependent's lowering emits a call on, as ``function_isolation``
#: spells them
ROUTE_SKIP = 'skip'
ROUTE_DIRECT = 'direct'
ROUTE_FAST = 'fast'
ROUTE_FAST_SHARED = 'fast-shared'
ROUTE_UNIFORM = 'uniform'

#: Where the ``udt`` decorator lives: the class it decorates is a dataclass that
#: gains a ``new`` classmethod and a ``copy`` function of that module
_UDT = (0, 'pynecore.core.pine_udt', 'udt')

#: The marker of a state-carrying function whose one instance every call site in a
#: loop shares without the same-bar rollback (``function_isolation._FAST_SHARED``)
_SHARED_MARKER = '__pyne_shared_call_site__'

#: The reexport key of the star import a name the module does not bind after it
#: may come from
STAR = '*'


@dataclass(slots=True)
class _Binding:
    """One statement binding a name in a module or class body."""
    #: Position of the body statement it stands in
    index: int
    #: The definition or class it binds, None for any other binding
    node: ast.AST | None
    #: Whether it may not run (it stands in a compound statement, a walrus, ...)
    conditional: bool
    #: For an import, where it leads: (level, module, attribute)
    target: tuple[int, str, str] | None = None


@dataclass(slots=True)
class _AttributeStore:
    """One store (or ``del``) of an attribute on a name of the body."""
    index: int
    attr: str
    #: The value of a plain ``name.attr = value``, None for any other store
    value: ast.expr | None
    conditional: bool


class _Bindings:
    """What the statements of one body bind, in order: the names, the attributes
    set on them, and the star imports."""

    def __init__(self, body: list[ast.stmt]):
        self.names: dict[str, list[_Binding]] = {}
        self.attributes: dict[str, list[_AttributeStore]] = {}
        #: (index, level, module, conditional) of every star import
        self.stars: list[tuple[int, int, str, bool]] = []
        for index, stmt in enumerate(body):
            self._statement(stmt, index)

    def _bind(self, name: str, binding: _Binding) -> None:
        self.names.setdefault(name, []).append(binding)

    def _statement(self, stmt: ast.stmt, index: int) -> None:
        """A body statement: what it binds itself is unconditional, anything in it
        that binds besides is not."""
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self._bind(stmt.name, _Binding(index, stmt, False))
            self._header(stmt, index)
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            self._import(stmt, index, False)
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                self._target(target, index, stmt.value if len(stmt.targets) == 1 else None)
            self._scan(stmt.value, index)
        elif isinstance(stmt, ast.AnnAssign):
            if stmt.value is not None:
                self._target(stmt.target, index, stmt.value)
                self._scan(stmt.value, index)
            self._scan(stmt.annotation, index)
        elif isinstance(stmt, (ast.AugAssign, ast.Delete)):
            for target in stmt.targets if isinstance(stmt, ast.Delete) else [stmt.target]:
                self._target(target, index, None)
            if isinstance(stmt, ast.AugAssign):
                self._scan(stmt.value, index)
        else:
            # A compound statement runs its parts maybe, an expression binds by walrus
            self._scan(stmt, index)

    def _target(self, target: ast.expr, index: int, value: ast.expr | None) -> None:
        """An assignment target of a body statement."""
        if isinstance(target, ast.Name):
            self._bind(target.id, _Binding(index, None, False))
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._target(element, index, None)
        elif isinstance(target, ast.Starred):
            self._target(target.value, index, None)
        elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            self.attributes.setdefault(target.value.id, []).append(
                _AttributeStore(index, target.attr, value, False))
        else:
            self._scan(target, index)

    def _import(self, stmt: ast.Import | ast.ImportFrom, index: int, conditional: bool) -> None:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                # ``import a.b`` binds ``a``; ``import a.b as x`` binds ``a.b`` itself
                module = alias.name if alias.asname else alias.name.split('.')[0]
                self._bind(alias.asname or module,
                           _Binding(index, stmt, conditional, (0, module, '')))
            return
        for alias in stmt.names:
            if alias.name == '*':
                self.stars.append((index, stmt.level, stmt.module or '', conditional))
            else:
                self._bind(alias.asname or alias.name,
                           _Binding(index, stmt, conditional,
                                    (stmt.level, stmt.module or '', alias.name)))

    def _header(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
                index: int) -> None:
        """What a definition evaluates in the body's scope: decorators, defaults,
        annotations, bases."""
        parts: list[ast.AST] = list(node.decorator_list)
        if isinstance(node, ast.ClassDef):
            parts.extend(node.bases)
            parts.extend(keyword.value for keyword in node.keywords)
        else:
            parts.append(node.args)
            if node.returns is not None:
                parts.append(node.returns)
        for part in parts:
            self._scan(part, index)

    def _scan(self, node: ast.AST, index: int) -> None:
        """Every binding below a node that lands in the body's scope, as conditional."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self._bind(node.name, _Binding(index, node, True))
            self._header(node, index)
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            self._import(node, index, True)
            return
        if isinstance(node, ast.Lambda):
            self._scan(node.args, index)
            return
        if isinstance(node, ast.comprehension):
            # Its target is the comprehension's own
            self._scan(node.iter, index)
            for test in node.ifs:
                self._scan(test, index)
            return
        if isinstance(node, ast.Name):
            if not isinstance(node.ctx, ast.Load):
                self._bind(node.id, _Binding(index, None, True))
            return
        if isinstance(node, ast.Attribute) and not isinstance(node.ctx, ast.Load) \
                and isinstance(node.value, ast.Name):
            self.attributes.setdefault(node.value.id, []).append(
                _AttributeStore(index, node.attr, None, True))
            return
        if isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
            self._bind(node.name, _Binding(index, None, True))
        elif isinstance(node, ast.MatchMapping) and node.rest:
            self._bind(node.rest, _Binding(index, None, True))
        for child in ast_walk.iter_child_nodes(node):
            self._scan(child, index)


@dataclass(slots=True)
class CallPlan:
    """
    How a call through each module-level name of a module is routed, as far as
    its analysed tree says: the lowering settles the rest, which definitions keep
    state (see :func:`call_routes`).
    """
    #: Dotted path -> the route the tree alone proves
    fixed: dict[str, str] = field(default_factory=dict)
    #: Dotted path -> the plain function a call reaches, and whether it is
    #: marked as sharing its call site
    defs: dict[str, tuple[ast.FunctionDef | ast.AsyncFunctionDef, bool]] = \
        field(default_factory=dict)
    #: ``ModuleInterface.reexports``
    reexports: dict[str, tuple[int, str, str]] = field(default_factory=dict)


def call_plan(tree: ast.Module) -> CallPlan:
    """
    How a call through each module-level name is routed, from the analysed tree.

    A dependent's lowering routes a call by the object an import of the module
    binds, without importing it: this is the object, told from the statements
    that bind it. A name is only given a route when the statements prove it --
    the last binding of the name is a statement that always runs, and nothing
    the module does afterwards can rebind it -- and every other name takes the
    uniform route, which is correct for any object, only slower:

    * a plain ``def`` is called on the direct route, or the fast one when the
      lowering gives it state (:func:`call_routes`); a decorated one is whatever
      its decorators return, so it has no route here;
    * a class is constructed raw, and so is a ``@udt`` class's ``new``; a
      method a dependent reaches through the class is routed like a plain
      function, a ``classmethod`` is not isolated;
    * a name an import binds is where the import leads (``reexports``), and a
      name only a star import can bind leads into that module.

    Read off the ANALYSED tree, like :func:`routes_shape`: what the lowering
    adds at module level is plumbing no dependent calls through. Everything it
    reads is in the routes shape, and the carriers the lowering adds are in the
    routes digest, so the routes digest moves whenever a route here does.

    :param tree: The analysed module
    :return: Its plan
    """
    found = _Bindings(tree.body)
    plan = CallPlan()
    last_star = max((star[0] for star in found.stars), default=-1)
    # The one star import a name the module does not bind after it may come from;
    # with more than one, or one that may not run, such a name has no route
    star = found.stars[0] if len(found.stars) == 1 and not found.stars[0][3] else None
    if star is not None:
        plan.reexports[STAR] = (star[1], star[2], STAR)
    for name, bindings in found.names.items():
        final = bindings[-1]
        if final.index < last_star:
            # A star import after it may rebind the name
            continue
        node = final.node
        if any(binding.conditional for binding in bindings):
            pass
        elif final.target is not None:
            plan.reexports[name] = final.target
            continue
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            shared = _shared(found.attributes.get(name, ()), final.index)
            if not node.decorator_list and shared is not None:
                plan.defs[name] = (node, shared)
                continue
        elif isinstance(node, ast.ClassDef):
            decorators = node.decorator_list
            udt = len(decorators) == 1 and isinstance(decorators[0], ast.Name) \
                and _bound_to(found, decorators[0].id, _UDT, final.index)
            if not decorators or udt:
                _plan_class(plan, found, name, node, final.index, udt)
                continue
        if star is not None:
            # Bound after the star import, so the star import does not answer for it
            plan.fixed[name] = ROUTE_UNIFORM
    return plan


def _shared(stores: Sequence[_AttributeStore], index: int) -> bool | None:
    """
    Whether a function is marked as sharing its call site, from the attribute stores
    on its name.

    :param stores: Every attribute store on the name, in order
    :param index: Position of the definition the name ends up bound to
    :return: The marker's value; None when a store after the definition makes the
             route unprovable (any attribute but the marker, set to a constant)
    """
    shared = False
    for store in stores:
        if store.index < index:
            # It went to an object the definition replaced
            continue
        if store.conditional or store.attr != _SHARED_MARKER \
                or not isinstance(store.value, ast.Constant):
            return None
        shared = bool(store.value.value)
    return shared


def _bound_to(found: _Bindings, name: str, target: tuple[int, str, str], index: int) -> bool:
    """
    Whether a name is an import of one particular object at a position of the body.

    :param found: The body's bindings
    :param name: The name
    :param target: Where the import has to lead
    :param index: The position: every binding of the name precedes it
    :return: True when the name's one binding is that import, which always runs
    """
    bindings = found.names.get(name, ())
    return len(bindings) == 1 and not bindings[0].conditional and bindings[0].index < index \
        and bindings[0].target == target and not any(star[0] < index for star in found.stars)


def _plan_class(plan: CallPlan, found: _Bindings, name: str, node: ast.ClassDef, index: int,
                udt: bool) -> None:
    """
    The routes through a class: the class itself, and the methods reached through it.

    :param plan: Extended in place
    :param found: The module's bindings
    :param name: The name the class is bound to
    :param node: The class
    :param index: Its position in the module
    :param udt: Whether ``@udt`` decorates it
    """
    plan.fixed[name] = ROUTE_SKIP
    # An attribute the module sets on the class afterwards is no longer what the
    # body bound
    replaced = {store.attr for store in found.attributes.get(name, ()) if store.index > index}
    if udt:
        # ``udt`` sets ``new`` (a classmethod) and ``copy`` (a plain function of its
        # own module, which no route is proven for) over whatever the body bound
        if 'new' not in replaced:
            plan.fixed[f'{name}.new'] = ROUTE_SKIP
        replaced |= {'new', 'copy'}
    body = _Bindings(node.body)
    # The decorators a method is told apart by, unless something rebinds them
    builtins_free = not found.stars and not any(
        decorator in found.names or decorator in body.names
        for decorator in ('staticmethod', 'classmethod'))
    for attr, bindings in body.names.items():
        method = bindings[-1].node
        if attr in replaced or attr in body.attributes \
                or any(binding.conditional for binding in bindings) \
                or not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decorators = [decorator.id if isinstance(decorator, ast.Name) else None
                      for decorator in method.decorator_list]
        if decorators and not builtins_free:
            continue
        if decorators == ['classmethod']:
            plan.fixed[f'{name}.{attr}'] = ROUTE_SKIP
        elif decorators in ([], ['staticmethod']):
            plan.defs[f'{name}.{attr}'] = (method, False)


def call_routes(plan: CallPlan, layout: _Layout) -> dict[str, str]:
    """
    The call routes of a module, once its lowering settled which definitions keep state.

    :param plan: :func:`call_plan` of the analysed tree
    :param layout: The slot layout the lowering allocated
    :return: ``ModuleInterface.call_routes``
    """
    routes = dict(plan.fixed)
    for path, (node, shared) in plan.defs.items():
        # Only a plain ``def`` is given a state parameter, as in ``routes_digest``
        if isinstance(node, ast.FunctionDef) and layout.state_carrying(layout.scope_segment(node)):
            routes[path] = ROUTE_FAST_SHARED if shared else ROUTE_FAST
        else:
            routes[path] = ROUTE_DIRECT
    return routes


@dataclass(frozen=True, slots=True)
class StaticInterface:
    """
    What a call through a module-level name of a module reaches, read off the
    module's source without running it: the routes of a module that publishes no
    settled interface (see :func:`static_interface`).
    """
    #: Dotted path -> its route, as ``ModuleInterface.call_routes``
    call_routes: dict[str, str]
    #: ``ModuleInterface.reexports``
    reexports: dict[str, tuple[int, str, str]]
    #: The literal ``__all__``, None when the module spells none
    all: tuple[str, ...] | None
    #: Every name a statement of the module may bind
    bound: frozenset[str]
    #: Whether a name no statement binds may still be served: by a star import or
    #: by a module ``__getattr__``
    open: bool


def static_interface(tree: ast.Module) -> StaticInterface:
    """
    The call interface of a module that publishes no settled one, from its tree alone.

    :func:`call_plan` proves what each module-level name binds: a class is
    constructed raw, an import leads on into the module it names. A function
    takes the uniform route: whether it carries state is not in its source -- a
    Pyne module's lowering gives it state, and a plain module may attach a
    ``__pyne_bind__`` to it at run time, as ``core/series.py`` does to
    ``inline_series`` -- and the uniform route binds whatever it turns out to be.

    :param tree: The module as parsed
    :return: Its static interface
    """
    plan = call_plan(tree)
    routes = dict(plan.fixed)
    for path in plan.defs:
        routes[path] = ROUTE_UNIFORM
    found = _Bindings(tree.body)
    return StaticInterface(call_routes=routes, reexports=dict(plan.reexports),
                           all=_module_all(tree), bound=frozenset(found.names),
                           open=bool(found.stars) or '__getattr__' in found.names)


def read_static_interface(path: str) -> StaticInterface | None:
    """
    The static interface of a module source file.

    :param path: The ``.py`` source
    :return: Its static interface, None when the file cannot be read or parsed
    """
    pair = stable_source(Path(path))
    if pair is None:
        return None
    try:
        tree = ast.parse(pair[0], filename=path)
    except (SyntaxError, ValueError):
        return None
    return static_interface(tree)


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


def lookup(path: str, analyse: Analyser | None, pipeline_hash: str,
           settled: bool = False) -> ModuleInterface | None:
    """
    A module's interface, from wherever it is cheapest to get.

    The registry answers within a process. When it cannot, the analyser does:
    it reads the interface the module's own ``.pyc`` carries, and when there
    is no such ``.pyc`` that still holds, it transforms the module into one --
    the very transform importing the module would run, which the import then
    finds done. A failed answer is NOT remembered: the file may be written,
    fixed or restored a moment later, and a cached "no" would outlive the
    reason for it.

    A caller that needs the ROUTES asks for a SETTLED interface, and gets one
    whose routes are settled or none at all: a registry entry the analysis
    alone published -- a module the lowering of which has not run yet -- is
    then not an answer, and the analyser is asked instead. A caller that does
    not gets what the types need, settled or not.

    A registry entry is checked against the file before it is handed out: the
    module's own fingerprint -- one ``os.stat``, against the one the interface
    itself carries -- AND its whole dependency closure, because an inferred
    signature can move without a single byte of this module changing. A
    module registered early in a process keeps answering for the rest of it,
    and an edit to a module its exports were INFERRED from lands nowhere near
    its own source. Whichever check fails, the entry is evicted and the answer
    re-derived rather than handed out stale. The analyser holds a ``.pyc`` to
    the same two checks.

    :param path: Source path of the module
    :param analyse: Reads or transforms the module when the registry cannot
                    answer; None to look no further than the registry
    :param pipeline_hash: Digest of the pipeline a ``.pyc`` must come from
    :param settled: Whether the caller needs the routes too
    :return: The interface, or None when it cannot be had
    """
    key = _key(path)
    # A module still being analysed cannot answer for itself; saying so is
    # what makes an import cycle terminate instead of recursing
    if _closes_cycle(key):
        return None

    # This stat decides EVICTION and nothing else. It is taken before the
    # analyser reads a byte, so it describes the file as it was then -- fine
    # for "has the entry's file moved", and never a fingerprint to publish:
    # the analyser pairs its own read with its own stat instead
    stat = _stat(key)
    hit = _registry.get(key)
    if hit is not None:
        if stat is not None and (hit.mtime_ns, hit.size) == (stat.st_mtime_ns, stat.st_size) \
                and _closure_current(key, hit, analyse, pipeline_hash, settled):
            # Current, but an entry without routes cannot answer a caller that
            # needs them; the analyser below replaces it
            if not settled or hit.routes:
                return hit
        else:
            _registry.pop(key, None)
    if stat is None or analyse is None:
        return None

    interface = analyse(key, pipeline_hash)
    if interface is None:
        return None
    register(interface)
    return interface if not settled or interface.routes else None


def _closure_current(key: str, interface: ModuleInterface, analyse: Analyser | None,
                     pipeline_hash: str, settled: bool) -> bool:
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

    The routes are part of the question when the caller needs them: the
    routes of an interface follow its dependencies' routes -- a function that
    calls a function which started keeping state keeps state itself -- so an
    entry whose closure moved its routes is as stale as one whose closure
    moved a signature.

    :param key: Resolved source path of the module the interface belongs to
    :param interface: The interface whose closure is being checked
    :param analyse: Reads or transforms a dependency the stat check cannot clear
    :param pipeline_hash: Digest of the pipeline a ``.pyc`` must come from
    :param settled: Whether the routes have to hold too
    :return: True while every dependency still matches what was recorded
    """
    if not interface.deps:
        return True
    with analysing_scope(key):
        return all(dep_current(record, analyse, pipeline_hash, settled)
                   for record in interface.deps.values())


def _closes_cycle(key: str) -> bool:
    """
    Whether reaching a module closes an import cycle, noted where it matters.

    A module closes one while its own analysis is on the stack. Every
    dependency transform in progress that began while it already was has
    just reached back into it, and is marked for it (see ``Compile``).

    :param key: Resolved source path of the module reached
    :return: True while it is being analysed
    """
    if key not in _analysing:
        return False
    for compile_ in _compiles:
        if key in compile_.outer:
            compile_.cyclic = True
    return True


def closes_cycle(path: str) -> bool:
    """
    Whether reaching a module closes an import cycle: its analysis is on the stack.

    This is what tells an import CYCLE apart from a module that simply has no
    interface to give. Both make ``lookup`` answer None, and only one of them
    is worth telling the user about. Asking is reaching the module, which a
    dependency transform in progress is marked for (see ``Compile``).

    :param path: Source path of the module, empty when it has none
    :return: True while it is being analysed
    """
    return bool(path) and _closes_cycle(_key(path))


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


@contextmanager
def compiling(path: str) -> Iterator[Compile]:
    """
    Track one dependency transform a lookup started, for as long as it runs.

    :param path: Source path of the module being transformed
    :return: The transform's record
    """
    compile_ = Compile(_key(path), frozenset(_analysing))
    _compiles.append(compile_)
    try:
        yield compile_
    finally:
        _compiles.pop()


def reached_back(path: str) -> bool:
    """
    Whether a module's dependency transform reached back into a module under analysis.

    Asked by the transform once the module's analysis is done, which is when
    the answer is complete: the type pass looks up every Pyne module the
    module imports, so whatever its lowering would import has been reached by
    then -- and when one of them reached back, the lowering must not run (see
    ``Compile``). A transform no lookup started, an import's own, is never one.

    :param path: Source path of the module being transformed
    :return: True when its transform has to stop at the analysis
    """
    return bool(_compiles) and _compiles[-1].cyclic and _compiles[-1].path == _key(path)


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
                     size=interface.size, digest=interface.digest, routes=interface.routes)


def source_record(path: str) -> DepRecord | None:
    """
    The record of a plain Python module a dependent's routes were read through.

    Such a module publishes no interface: a route read off the object it binds
    holds while it binds the same object, and nothing short of its source
    staying as it was says so. The record travels in the dependency closure
    like any other, so it reaches the dependents of the dependent too.

    :param path: Resolved source path of the module
    :return: The record, or None when the file cannot be stat'ed
    """
    stat = _stat(path)
    if stat is None:
        return None
    return DepRecord(path=_key(path), mtime_ns=stat.st_mtime_ns, size=stat.st_size,
                     digest=SOURCE_DIGEST, routes=SOURCE_DIGEST)


def add_dep(deps: dict[str, DepRecord], interface: ModuleInterface, own_path: str) -> None:
    """
    Record the dependency on one module, and the dependencies it carries.

    Its dependencies are the dependent's too. An export whose return was
    INFERRED from a call one module further out moves when THAT module's
    signature moves, and nothing about the dependent or its direct dependency
    changes when it does -- so the closure has to be carried, not just the edge.

    :param deps: The dependent's records, keyed by path; extended in place
    :param interface: The interface the dependent now depends on
    :param own_path: The dependent's own source path
    """
    record = dep_record(interface)
    deps[record.path] = record
    for inherited in interface.deps.values():
        if inherited.path == own_path:
            # A cyclic pair names the dependent in the other's closure; its
            # own source is not something it can be invalidated by
            continue
        deps.setdefault(inherited.path, inherited)


def dep_current(record: DepRecord, analyse: Analyser | None, pipeline_hash: str,
                settled: bool = False) -> bool:
    """
    Whether a dependency still means what the dependent was built against.

    The stat pair is checked first and answers on its own: an untouched file
    costs one ``os.stat`` and no parsing at all, which is the case every
    ordinary import is. Only a file that moved is worth re-deriving an
    interface for -- and a body edit lands here and still says yes, unless it
    changed whether a function keeps state.

    That short-circuit is only sound because the closure is TRANSITIVE: a
    dependent records its dependencies' dependencies too, so a third module's
    signature -- or routes -- moving is a record of its own here rather than a
    change hiding behind an untouched file.

    The routes are compared when the caller asks for them, which is what a
    check of a dependent's BYTECODE does: its emission routed every call into
    the dependency. A check that only vouches for types has no routes to
    compare. A plain module's record (``SOURCE_DIGEST``) has nothing to
    re-derive, so for it the stat pair is the whole answer.

    :param record: What the dependent remembers
    :param analyse: Reads or transforms the dependency once its file moved
    :param pipeline_hash: Digest of the pipeline a ``.pyc`` must come from
    :param settled: Whether the routes have to match too
    :return: True while the dependent's bytecode is still valid
    """
    try:
        stat = os.stat(record.path)
    except OSError:
        return False
    if stat.st_mtime_ns == record.mtime_ns and stat.st_size == record.size:
        return True
    if record.digest == SOURCE_DIGEST:
        return False
    interface = lookup(record.path, analyse, pipeline_hash, settled)
    if interface is None or interface.digest != record.digest:
        return False
    return not settled or interface.routes == record.routes


def settle(record: DepRecord, analyse: Analyser, pipeline_hash: str) -> DepRecord:
    """
    A dependency record with the routes it was made without, when they can be had.

    The type pass records a dependency off whatever interface it found, and
    one the analysis alone published -- a module whose lowering had not run
    yet -- has no routes. Before the record is baked into the dependent it is
    settled -- usually for free, from the registry entry the dependency's own
    transform replaced it with. The routes are only taken from an interface at
    the record's own fingerprint and digest: a file that moved in between
    leaves the record unsettled, which no later check accepts once the file
    moves, so it costs at most a retransform.

    :param record: The record the type pass made
    :param analyse: Reads or transforms the dependency when the registry cannot
                    settle it
    :param pipeline_hash: Digest of the pipeline a ``.pyc`` must come from
    :return: The record with its routes, or the record itself
    """
    if record.routes:
        return record
    interface = lookup(record.path, analyse, pipeline_hash, settled=True)
    if interface is None or (interface.mtime_ns, interface.size) != (record.mtime_ns, record.size) \
            or interface.digest != record.digest:
        return record
    return replace(record, routes=interface.routes)


# --- the payload ----------------------------------------------------------


def interface_payload(interface: ModuleInterface) -> bytes:
    """
    An interface as the one constant its module's ``.pyc`` carries it in.

    Everything a dependent reads off the interface, the routes included --
    and nothing that already rides in the ``.pyc`` elsewhere or that is not
    the module's to say: the dependency closure is the module's own baked
    dependency records, and the fingerprint is paired with the payload by
    whoever bakes it, from the read the transform was given.

    Compressed, because it rides in every ``.pyc`` and stays referenced by the
    module for as long as it lives: the JSON of a large library's interface
    runs to tens of kilobytes -- a quarter of its bytecode -- and repeats the
    same keys for every signature, which compresses some twentyfold.

    :param interface: The interface to serialize
    :return: Compressed canonical JSON
    """
    return zlib.compress(_canonical({
        'all': list(interface.all) if interface.all is not None else None,
        'classes': {name: _class_json(sig) for name, sig in interface.classes.items()},
        'extensions': {cid: {name: _export_json(published)
                             for name, published in methods.items()}
                       for cid, methods in interface.extensions.items()},
        'exports': {name: _export_json(sig) for name, sig in interface.exports.items()},
        'suppressed': interface.suppressed,
        'routes': interface.routes,
        'proxies': list(interface.proxies),
        'call_routes': interface.call_routes,
        'reexports': {name: list(target) for name, target in interface.reexports.items()},
    }).encode('utf-8'))


def interface_from_payload(path: str, payload: bytes, deps: dict[str, DepRecord],
                           fingerprint: tuple[int, int]) -> ModuleInterface | None:
    """
    Rebuild the interface a ``.pyc`` carries.

    The digest is recomputed rather than read back, so what a dependent is
    checked against is always a digest of what it was actually handed. The
    routes have to be read back: they are the lowering's verdict, and nothing
    short of lowering the module again could recompute them.

    :param path: Resolved source path of the module
    :param payload: What ``interface_payload`` made of the interface
    :param deps: The dependency closure the module was transformed under
    :param fingerprint: Of the source the ``.pyc`` was just validated against
    :return: The interface it describes, or None for a payload that is no
             interface
    """
    try:
        published = json.loads(zlib.decompress(payload))
    except (zlib.error, ValueError, TypeError):
        return None
    if not isinstance(published, dict):
        return None
    all_names = published.get('all')
    exports = {name: _export_from_json(name, sig)
               for name, sig in (published.get('exports') or {}).items()}
    classes = {name: _class_from_json(name, sig)
               for name, sig in (published.get('classes') or {}).items()}
    extensions = {cid: {name: _export_from_json(name, method)
                        for name, method in (methods or {}).items()}
                  for cid, methods in (published.get('extensions') or {}).items()}
    interface = ModuleInterface(
        path=path, exports=exports,
        all=tuple(all_names) if all_names is not None else None,
        classes=classes, extensions=extensions, digest='', deps=dict(deps),
        mtime_ns=fingerprint[0], size=fingerprint[1],
        suppressed=str(published.get('suppressed') or ''),
        routes=str(published.get('routes') or ''),
        proxies=tuple(published.get('proxies') or ()),
        call_routes={str(path): str(route)
                     for path, route in (published.get('call_routes') or {}).items()},
        reexports={str(name): (int(target[0]), str(target[1]), str(target[2]))
                   for name, target in (published.get('reexports') or {}).items()})
    return replace(interface, digest=interface_digest(interface))


def _export_json(sig: ExportSig) -> dict:
    """
    One published signature, in the payload's shape.

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


def _export_from_json(name: str, data: dict) -> ExportSig:
    """
    Rebuild one published signature a payload carries.

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
    One published class, in the payload's shape.

    :param sig: The class to serialize
    :return: Its JSON form
    """
    return {
        'id': sig.id,
        'fields': dict(sig.fields),
        # The payload is written with sorted keys, so the declaration order
        # a constructor binds positional arguments by travels on its own
        'order': list(sig.fields),
        'required': sig.required,
        'methods': {name: _export_json(method) for name, method in sig.methods.items()},
        'members': {name: _export_json(member) for name, member in sig.members.items()},
    }


def _class_from_json(name: str, data: dict) -> ClassSig:
    """
    Rebuild one published class a payload carries.

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
                 for method, sig in (data.get('methods') or {}).items()},
        members={member: _export_from_json(member, sig)
                 for member, sig in (data.get('members') or {}).items()})
