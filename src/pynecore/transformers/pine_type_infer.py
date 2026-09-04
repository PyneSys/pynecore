"""
The Pine type inference engine.

Walks a module and gives every expression a Pine type, which it stamps on the
node itself (``node._pine_ty``). Later passes reuse the node objects, so the
stamp travels with the tree into the lowered form the AOT compiler consumes;
the passes that BUILD a wrapper node have to carry it over explicitly, which
is what ``inherit_ty`` is for.

Completeness on the Pine-expressible subset is the point, not best effort.
Int-ness has a CLOSED set of origins -- an int literal, an ``int``-ish
annotation, an int-returning lib name, an ``int()`` cast -- and travels over a
closed set of operators, so anything still UNKNOWN afterwards has genuinely
left the Pine world.

An unannotated user-function parameter is typed PER CALL SITE, which is what
TradingView measurably does: the parameter's type is JOIN(type of its default,
type of the argument), and the body behaves as if instantiated once per
distinct parameter-type tuple. There are no clones here -- one body is
analysed once per context, and the results are kept apart in the type table.
A node reached in several contexts carries the JOIN of what they found, and a
call site whose overload pin the contexts disagree on carries the per-context
pins (``node._pine_pins``) instead of a single one, for a later pass to turn
into per-instance data.

A call into an IMPORTED module is the one place that rule stops. Such a callee
is typed from its DECLARED signature alone -- the interface its module
publishes -- and never from the argument types at the call site: the callee's
body belongs to another module, which is analysed once, on its own. An
overload group is still pinned there, because a pin selects among signatures
and needs no body. Every interface consulted is recorded in ``table.deps``, so
the loader can tell when a dependency's signatures moved.

The other bounded fixpoint that remains is the loop one: a loop-carried
variable only reaches its type on the second walk of the body.

This module is analysis-only: it rewrites nothing, clones nothing and pins
nothing. That keeps it testable one rule at a time, and keeps the rules
(``pine_type_rules``) separable from the walking done here.
"""
import ast
import importlib.util
import json
from bisect import bisect_right
from collections.abc import Callable, Container, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from ..utils.stdlib_checker import is_stdlib
from . import pine_type_artifact
from .dynamic_default import is_script_entry
from .node_ids import assign_node_ids, node_id
from .pine_type_report import unknown_diags
from .pine_type_rules import (
    INT, FLOAT, BOOL, STR, TYPELESS, UNKNOWN, VOID, OBJECT, NUMERIC,
    join, binop_type, unaryop_type, compare_type, annotation_type, bare_wrapper,
    LIB_TYPE_OVERRIDES, OVERRIDE_PARAM_NAMES, BUILTIN_CALL_TYPES, BUILTIN_NAME_TYPES,
    TY_ATTR, get_ty, set_ty, inherit_ty,
    constant_type, pin_for, get_pins, set_pin, set_pins, set_vector, set_varying,
    overload_result, ImplSig, overload_pick, default_fit, FIT_REQUIRED, FactoryFields,
    impl_sig, _param_defaults, _dotted, _DYN_DEFAULT,
    CLASS_SEP, LIB_MODULE, SCALARS, PINE_LOOP, array_of, builtin_class_id, class_id, class_of,
    element_of, elements_of, head, is_array,
    is_map, is_matrix, is_tuple, key_of, map_of, matrix_of, namespace_of,
    object_ty, render_ty, shape_conflict, tuple_of, value_of,
)
from .pine_type_table import (
    Analyser, Binding, CallSite, ClassSig, ContextKey, ContextResult, Diag, ExportSig,
    FuncSig, ModuleInterface, PineTypeTable, Unknown, qualify,
)

__all__ = ['infer_module', 'lib_types', 'lib_classes', 'lib_namespaces',
           'TY_ATTR', 'get_ty', 'set_ty', 'inherit_ty']

#: How many times a loop body is re-inferred before the types are declared
#: stable. The lattice is two high (int -> float -> unknown), so a binding can
#: only move twice; a third pass exists to OBSERVE that nothing moved.
_MAX_LOOP_PASSES = 3

#: Ceiling on how many per-call-site contexts one module may produce. Contexts
#: follow the call GRAPH, so a module that fans out through many layers of
#: generic helpers could in principle produce a path-shaped explosion. Real
#: scripts stay two orders of magnitude below this; the limit exists so a
#: pathological one degrades to UNKNOWN instead of hanging the compiler.
_MAX_CONTEXTS = 500

#: The lib namespaces ``core.pine_method.method_call`` dispatches a builtin
#: method to, by the receiver's runtime class. A method name that reaches only
#: one of them, or one they all answer the same way, is answerable statically.
_METHOD_NAMESPACES: Final = ('array', 'matrix', 'map', 'line', 'box', 'label',
                             'table', 'linefill', 'polyline')

#: How each shape-reading override form derives its answer from the type of
#: the argument it names. Each one DECLINES -- answers UNKNOWN or a bare
#: object, which the caller turns into "no answer" -- where the shape it
#: needs is not there, so the call falls through to the lib's annotation.
_SHAPE_FORMS: Final[dict[str, Callable[[str], str]]] = {
    'array_of_map_keys': lambda ty: array_of(key_of(ty)),
    'array_of_map_values': lambda ty: array_of(value_of(ty)),
    'array_of_elem': lambda ty: array_of(element_of(ty)),
    'array_of_arg': array_of,
    'matrix_of_arg': matrix_of,
    'map_key': key_of,
    'map_value': value_of,
    'elem': element_of,
    # A container comes back as itself, but only a container of the KIND the
    # callee is for: ``array.copy`` of a tuple is a list at run time, whose
    # positions the tuple's types say nothing about
    'same_array': lambda ty: ty if is_array(ty) else UNKNOWN,
    'same_matrix': lambda ty: ty if is_matrix(ty) else UNKNOWN,
    'same_map': lambda ty: ty if is_map(ty) else UNKNOWN,
}

#: A node that opens a lexical scope of its own.
_Scope = ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | ast.ClassDef
_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)

#: Where one position stands in a module body's control flow: the chain of
#: (compound statement, branch label) pairs leading down to it from the body.
#: Two positions are mutually exclusive when the first branch they take apart
#: is one no single pass runs both of -- see ``_mutually_exclusive``.
_BranchPath = tuple[tuple[ast.stmt, str], ...]

_LIB_TYPES_PATH = Path(__file__).parent / 'lib_types.json'
_LIB_TYPES: dict[str, Any] = {}
_LIB_CLASSES: dict[str, 'ClassSig'] = {}
_LIB_SCALARS: dict[str, str] = {}
_LIB_NAMESPACES: set[str] = set()


def lib_types() -> dict[str, Any]:
    """
    The generated lib registry, loaded once.

    Read from JSON rather than by importing the lib: this module is imported
    by the import hook while it transforms pynecore's own lib modules, and an
    import there would re-enter a half-initialized package (the same reason
    ``const_fold`` defers its lib import).

    :return: name -> entry mapping
    """
    if not _LIB_TYPES:
        _LIB_TYPES.update(json.loads(_LIB_TYPES_PATH.read_text())['names'])
    return _LIB_TYPES


def lib_classes() -> dict[str, ClassSig]:
    """
    The classes the lib publishes, with the fields they declare, loaded once.

    A ``chart.point`` knows its class like any other object, so ``p.price``
    is the field's declared type -- but a builtin class says what it holds in
    the type package rather than in a module interface, which is why the
    generated registry carries it.

    :return: Class id -> what that class declares
    """
    if not _LIB_CLASSES:
        published = json.loads(_LIB_TYPES_PATH.read_text())['classes']
        for name, fields in published.items():
            cid = builtin_class_id(name)
            _LIB_CLASSES[cid] = ClassSig(name=name, id=cid, fields=dict(fields), methods={})
    return _LIB_CLASSES


def lib_scalar_classes() -> dict[str, str]:
    """
    The lib classes that ARE a scalar, loaded once.

    ``format.percent`` is a ``Format``, and a ``Format`` is a string: Pine
    spells such a constant as a ``const string``, and the parameter that takes
    it is annotated ``str``. The registry records which classes derive from a
    scalar so the value fits the parameter without losing its own identity.

    :return: Class id -> the scalar it is
    """
    if not _LIB_SCALARS:
        published = json.loads(_LIB_TYPES_PATH.read_text()).get('scalar_classes') or {}
        _LIB_SCALARS.update({object_ty(builtin_class_id(name)): scalar
                             for name, scalar in published.items()})
    return _LIB_SCALARS


def lib_namespaces() -> set[str]:
    """
    Every dotted path the registry's names hang off, derived once.

    A namespace is not a name the registry lists -- ``ta`` has entries under
    it, not one of its own -- yet a script does pass the module object itself
    around (``shadowed_namespace(lib_alias, lib.ta)``). Such a reference is a
    known non-scalar, and telling it from a misspelled lib name needs the set
    of prefixes.

    :return: The dotted namespace paths
    """
    if not _LIB_NAMESPACES:
        for key in lib_types():
            parts = key.split('.')
            for depth in range(1, len(parts)):
                _LIB_NAMESPACES.add('.'.join(parts[:depth]))
    return _LIB_NAMESPACES


def infer_module(tree: ast.Module, module_path: str = '', *,
                 analyse: Analyser | None = None,
                 pipeline_hash: str = '') -> PineTypeTable:
    """
    Infer and stamp the Pine types of a whole module.

    One walk is enough. A call to a helper defined further down used to read
    UNKNOWN and needed the whole module re-walked until the return types
    stopped moving; now the call itself drives the analysis of the callee --
    the definitions are collected up front, so definition order says nothing
    about what a call site can know.

    :param tree: The module to walk; it is stamped in place
    :param module_path: Absolute source path, for diagnostics
    :param analyse: Re-derives an imported module's table from its source path,
                    for resolving a call into another module
    :param pipeline_hash: Digest of the transform pipeline, which an imported
                          module's cached interface has to have been built by
    :return: The derived type table
    """
    assign_node_ids(tree)
    engine = _Inference(module_path, analyse=analyse, pipeline_hash=pipeline_hash)
    # The module answers None for itself while it is being walked, which is
    # what terminates an import cycle -- and marking it HERE rather than only
    # in ``lookup`` is what makes the cycle visible to the module that has it,
    # instead of to a throwaway re-analysis one level further down
    with pine_type_artifact.analysing_scope(module_path):
        engine.run(tree)
    return engine.table


@dataclass(slots=True)
class _Frame:
    """One live lexical scope of the walk: its id and the names it binds."""
    #: Scope id, the same identity ``table.funcs`` and ``table.bindings`` use
    scope: str
    #: Names bound in THIS context of the scope, kept apart from every other
    names: dict[str, Binding] = field(default_factory=dict)
    #: Names the scope declared ``global``: they are not bound here at all,
    #: they read and write the MODULE's binding
    declared_global: set[str] = field(default_factory=set)
    #: Names it declared ``nonlocal``: the same, against the nearest
    #: enclosing function scope that has them
    declared_nonlocal: set[str] = field(default_factory=set)
    #: Names a bare ``x: Series`` declared before anything was assigned to them
    declared_series: set[str] = field(default_factory=set)
    #: Name of a ``pine_loop(...)`` counter object -> the type its counter
    #: has, the join of the bounds it was built and stepped with
    loop_counters: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class _Import:
    """
    A module-level name an import binds, and the module it reaches.

    ``attrs`` is what the import spelling already consumed: ``from m import f``
    binds ``f`` with ``('f',)`` still to resolve against module ``m``, while
    ``import a.b as x`` binds ``x`` to module ``a.b`` with nothing left over.
    """
    #: Dotted module the import statement names
    module: str
    #: Attribute path the spelling consumed, module-relative
    attrs: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class _Shadowed:
    """
    A name bound to a Pine library merged over the builtin namespace it shadows.

    MEASURED: ``shadowed_namespace`` resolves MEMBER BY MEMBER -- what the
    library exports comes from the library, every other member from the
    builtin namespace -- so the binding has to carry both halves.
    """
    #: The import the merged namespace wraps
    source: _Import
    #: Registry prefix of the builtin namespace, e.g. ``'ta'``
    namespace: str


class _Inference:
    """The walker. One instance per module."""

    def __init__(self, module_path: str, *, analyse: Analyser | None = None,
                 pipeline_hash: str = ''):
        self.table = PineTypeTable(module_path=module_path)
        #: How an imported module's table is re-derived, when one is needed.
        #: Not ``_analyse`` -- that name is the walker's own body analysis
        self._analyser = analyse
        #: Which pipeline an imported module's cached interface must come from
        self._pipeline_hash = pipeline_hash
        #: Live lexical scopes, outermost first; the module scope is ``''``
        self._frames: list[_Frame] = [_Frame('')]
        self.table.bindings[''] = self._frames[0].names
        #: Names bound by the enclosing lib import, e.g. ``lib``
        self._lib_aliases: set[str] = set()
        #: Module-level name -> the import that binds it. Module level only: a
        #: function-level import is a local of that scope and stays opaque,
        #: the same shapes the isolation pass declines to resolve
        self._imports: dict[str, _Import] = {}
        #: Module-level name -> the merged namespace ``shadowed_namespace``
        #: binds it to
        self._shadowed: dict[str, _Shadowed] = {}
        #: Name -> the positions its OWN import (or its ``shadowed_namespace``
        #: assignment) binds it at. Every other module-level binding of that
        #: name is a rebinding, and makes the import unusable
        self._import_positions: dict[str, set[tuple[int, int]]] = {}
        #: Names more than one import binds. ``_imports`` holds one entry per
        #: name, so such a name has no single answer to give -- which module
        #: it reaches depends on which statement ran last, and two of them in
        #: exclusive branches make even that unanswerable
        self._multi_imports: set[str] = set()
        #: Dotted module name -> its source path, resolved without importing
        #: the module itself
        self._module_paths: dict[str, str | None] = {}
        #: Source path -> the interface that module publishes. One lookup per
        #: module for the whole walk, however many call sites reach it
        self._interfaces: dict[str, ModuleInterface | None] = {}
        #: Whether each function spelled its return type out
        self._annotated_returns: dict[str, bool] = {}
        #: Scope-qualified ids of the module's ``@overload`` groups -- the only
        #: user callees where a call site has anything to choose between
        self._overload_groups: set[str] = set()
        #: Every implementation's return type per group, in declaration order.
        #: A group's own type is what they all agree on (``overload_result``),
        #: so the entries have to stay apart until the last one is known
        self._group_returns: dict[str, list[str]] = {}
        #: Def node id -> its slot in its group's ``_group_returns`` list. Keyed
        #: by the NODE, so a body analysed out of declaration order -- or twice
        #: -- still fills in its own return and never another's
        self._group_slot: dict[int, int] = {}
        #: Every definition, by scope-qualified id. A list, because an overload
        #: group spells several implementations under one id
        self._defs: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
        #: Scope id -> the names that scope binds by something OTHER than a
        #: ``def``. Such a name is not the definition's any more, whatever the
        #: definition is called
        self._rebound: dict[str, set[str]] = {}
        #: Def node ids that already have at least one context
        self._walked: set[int] = set()
        #: Context ids, innermost last; 0 is the module body itself
        self._contexts: list[int] = [0]
        self._next_context = 1
        #: (function id, parameter tuple, def node id) whose analysis has not
        #: returned yet
        self._in_progress: set[tuple[str, tuple[str, ...], int | None]] = set()
        #: Def node id -> the enclosing-scope names its body reads, computed
        #: once per definition: the lexically resolved ones and the ones a
        #: ``global`` declaration sends straight to the module scope
        self._free: dict[int, tuple[tuple[str, ...], tuple[str, ...]]] = {}
        #: One call site's analysis identity -- everything but the
        #: free-variable types of a memo key, plus the call node -> the memo
        #: key that currently answers it. What makes a re-analysis of the SAME
        #: site under widened enclosing types supersede the stale one instead
        #: of joining it, while leaving a different site its own context
        self._anchors: dict[tuple, ContextKey] = {}
        #: Node id -> type THIS walk gave it. A body walked in two contexts
        #: gives its nodes two types; the stamp on the node is their join, so
        #: the walk needs its own view to derive a context's own answers from
        self._ty: dict[int, str] = {}
        #: Node id -> context id -> the pin that context justified
        self._pins: dict[int, dict[int, str | None]] = {}
        #: Where the running context collects its inner pins; None at module
        #: level, which owns no context result
        self._pin_sink: list[dict[int, str | None] | None] = [None]
        #: Definitions whose body walk is still owed, per live walk
        self._pending: list[list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]] = [[]]
        #: Function ids whose signature already absorbed one context
        self._signed: set[str] = set()
        #: Call node ids already reported as pin-disagreeing
        self._pin_diags: set[int] = set()
        #: Call node id -> calling context id -> the context the callee was
        #: instantiated in there. What ``_pins`` is for an overload site, this
        #: is for a call to a generic user function: the per-instance channel
        #: needs to know WHICH instantiation each caller's instance reaches
        self._call_ctx: dict[int, dict[int, int]] = {}
        #: Call node id -> scope-qualified id of the generic user function it
        #: calls, for the same reason
        self._callee_key: dict[int, str] = {}
        #: Module-scope name -> every position it is bound at by something
        #: other than a ``def``, each with the branch path it stands on. Module
        #: bindings are SEQUENTIAL, so a call above all of them still reaches
        #: the definition -- and one in a branch the call cannot share reaches
        #: it wherever it stands
        self._module_rebinds: dict[str, list[tuple[tuple[int, int], _BranchPath]]] = {}
        #: The branch path of every module statement, for the same question
        self._branches = _BranchIndex(())
        #: The loop statements the walk is inside, outermost first. A loop
        #: body runs again after its own later statements, so a rebinding
        #: anywhere in one reaches the calls above it on the next iteration
        self._loop_stack: list[ast.stmt] = []
        #: Every class name an annotation of this module may name -> its class
        #: id, its own and its imports' alike; filled in before anything reads
        #: an annotation
        self._classes: dict[str, str] = {}
        #: Class id -> what that class declares. Its own classes are here, and
        #: so is every class of every interface this module consulted: a value
        #: reaching this module from another one carries its class id, and a
        #: field read of it needs the fields behind that id
        self._class_sigs: dict[str, ClassSig] = dict(lib_classes())
        #: Class id -> method name -> its definition, for the classes this
        #: module DECLARES. A Pine method is a free function whose first
        #: parameter is annotated with the class, so what a receiver reaches is
        #: resolved through the ordinary scope search on that name; the node is
        #: what the published signature is derived from
        self._class_methods: dict[str, dict[str, ast.FunctionDef]] = {}
        #: Nodes a shape complaint was already reported for. A body is walked
        #: once per context and again per loop pass, and one mismatch is one
        #: complaint however many walks find it
        self._shape_diags: set[int] = set()
        #: The ``field(default_factory=...)`` calls that stand as UDT field defaults
        self._factory_fields: set[int] = set()
        self._factory = FactoryFields(ast.Module(body=[], type_ignores=[]))
        #: The diagnostic that took every pin away from the module, once one has
        self._pins_suppressed: Diag | None = None
        #: Call nodes whose shape or argument types the callee rejected
        self._bad_calls: set[int] = set()
        #: Security id -> the expressions ``__sec_write__`` publishes under it.
        #: What a ``__sec_read__`` of the same id evaluates to
        self._sec_writes: dict[str, list[ast.expr]] = {}

    # --- scope plumbing --------------------------------------------------

    @property
    def _scope(self) -> str:
        return self._frames[-1].scope

    @property
    def _context(self) -> int:
        return self._contexts[-1]

    @staticmethod
    def _qualify(scope: str, name: str) -> str:
        """The scope-qualified identity of a name declared in one scope."""
        return qualify(scope, name)

    def _resolve_func(self, name: str, node: ast.AST) -> tuple[str, int, bool] | None:
        """
        The signature a call name resolves to, searching outward.

        Function signatures are keyed the same way bindings are, because a bare
        ``helper()`` means a DIFFERENT function in each enclosing scope that
        defines one; keying by the bare name alone let two same-named nested
        helpers overwrite each other's return type.

        The frame index comes back with the key: the callee's lexical parents
        are exactly the frames up to and including that one, which is the
        environment its body has to be walked in.

        A name a scope on the way also BINDS is reported as shadowed, and a
        shadowed name must be treated as unresolvable. Its runtime value is
        whatever was assigned to it, so everything derived from the definition
        -- its contexts, its pin, its instance vector -- would describe a
        function the call never reaches; the vector is the dangerous one, since
        it is handed to the callee that DOES run and indexes that one's slots.
        The isolation pass routes such a name through its uniform path for the
        same reason, so the two passes agree on which names are opaque.

        Where the shadowing is decided differs by scope, though, because
        Python's own binding rules do -- see ``_reaches_def``.

        :param name: The name as the call spells it
        :param node: The call node, for the module scope's ordering rule
        :return: (key in ``table.funcs``, declaring frame index, shadowed), or
                 None when no definition of that name is live at all
        """
        shadowed = False
        for index in range(len(self._frames) - 1, -1, -1):
            scope = self._frames[index].scope
            if name in self._rebound.get(scope, ()) \
                    and not self._reaches_def(scope, name, node):
                shadowed = True
            key = self._qualify(scope, name)
            if key in self.table.funcs:
                return key, index, shadowed
        return None

    def _reaches_def(self, scope: str, name: str, node: ast.AST) -> bool:
        """
        Whether a call reaches the definition despite a rebinding of its name.

        MODULE scope binds sequentially: ``result = f(1 / 2)`` written above
        ``f = other`` calls the definition, because the rebinding has not run
        yet when the call does. A FUNCTION body has no such order -- a name
        assigned anywhere in it is local to the whole of it, so a call above
        the assignment raises ``UnboundLocalError`` rather than reaching an
        outer definition -- which is why this only ever answers for the module,
        and only for a call the module body itself makes.

        A loop up there does not suspend that order, it only adds a back edge:
        a module-level loop COMPLETES before the statement below it runs, so
        ``for _ in range(1): result = f(1 / 2)`` followed by ``f = other``
        calls the definition on every iteration. What the back edge does add is
        the loop's OWN later statements -- a rebinding standing after the call
        inside the same loop body has run by the second iteration -- so a
        rebinding shadows the call when it stands above it, or anywhere inside
        a loop that encloses it.

        Order is not the whole of it either, because BRANCHES break it: with
        ``if flag: f = other`` and the call in the ``else``, the rebinding
        stands above the call and no run of the module executes both, so the
        call always reaches the definition. Reading position alone lost the pin
        there. A rebinding a branch keeps away from the call is therefore
        skipped -- unless a loop puts it back in reach, since the next
        iteration is free to take the branch this one did not.

        :param scope: The scope the rebinding was found in
        :param name: The name being called
        :param node: The call node
        :return: True when the definition is what the call reaches
        """
        if scope or len(self._frames) > 1:
            return False
        rebinds = self._module_rebinds.get(name)
        if not rebinds:
            return False
        at = (_line(node), _col(node))
        path = self._branches.of(at)
        spans = [_span(loop) for loop in self._loop_stack]
        for position, rebind_path in rebinds:
            if any(start <= position <= end for start, end in spans):
                return False
            if _mutually_exclusive(rebind_path, path):
                continue
            if position < at:
                return False
        return True

    def _bindings(self) -> dict[str, Binding]:
        return self._frames[-1].names

    def _declared_home(self, name: str) -> dict[str, Binding] | None:
        """
        Where a ``global``/``nonlocal`` declaration sends one name.

        The declaration moves the name WHOLE, reads and writes alike, so the
        walk that types a body and the memo key that decides whether it has to
        run again look at the same binding. ``global`` lands on the module
        however many same-named locals stand in between; ``nonlocal`` lands on
        the nearest enclosing function scope that has the name, and a
        ``nonlocal`` no scope answers is not valid Python at all -- the local
        frame is what is left of it.

        :param name: The name being bound or read
        :return: The name map to use, or None for an ordinary local
        """
        current = self._frames[-1]
        if name in current.declared_global:
            return self._frames[0].names
        if name in current.declared_nonlocal:
            for frame in reversed(self._frames[1:-1]):
                if name in frame.names:
                    return frame.names
            return current.names
        return None

    def _lookup(self, name: str) -> Binding | None:
        """Find a name in the innermost live scope that has it."""
        declared = self._declared_home(name)
        if declared is not None:
            return declared.get(name)
        for frame in reversed(self._frames):
            found = frame.names.get(name)
            if found is not None:
                return found
        return None

    def _bind(self, name: str, ty: str, node: ast.AST, unknown: Unknown | None = None,
              series: bool = False) -> None:
        """
        Record an assignment.

        Re-assigning a name JOINS with what it already had: Pine's variables
        are single-typed, and a branch that stores a float into an int-typed
        variable widens it for every later read.

        A ``global``/``nonlocal`` name is written where it LIVES, not here --
        otherwise the scope that widens it and the closure that reads it would
        be looking at two different bindings.
        """
        bindings = self._declared_home(name)
        if bindings is None:
            bindings = self._bindings()
        existing = bindings.get(name)
        line = getattr(node, 'lineno', 0)
        series = series or name in self._frames[-1].declared_series
        if existing is None:
            bindings[name] = Binding(name=name, ty=ty, line=line, unknown=unknown, series=series)
            return
        # A series declaration holds for the name, whatever is assigned later
        existing.series = existing.series or series
        before = existing.ty
        joined = self._joined(before, ty, node, f"'{name}'")
        existing.ty = joined
        if joined == UNKNOWN and existing.unknown is None:
            existing.unknown = self._shape_unknown(before, ty, node) or unknown \
                or self._unknown('joined-branches', node, name)

    def _joined(self, left: str, right: str, node: ast.AST, what: str) -> str:
        """
        Join two types one expression may have, reporting a SHAPE conflict.

        Two branches producing different shapes is a Pine compile error, not
        an untypable expression: ``array<int>`` and ``array<float>`` are
        different types and the language rejects a variable that holds either.
        So the join is UNKNOWN -- widening them to "some object" would throw
        away exactly the element type the next read needs -- and the walker
        says so, because ``join`` is a pure function on strings and has no
        node to point at.

        :param left: The type established so far
        :param right: The type the new branch produces
        :param node: Where the two meet
        :param what: How the joined thing is named in the message
        :return: The joined type
        """
        joined = join(left, right)
        conflict = shape_conflict(left, right)
        if conflict is None:
            return joined
        nid = node_id(node)
        if nid is not None and nid in self._shape_diags:
            return joined
        if nid is not None:
            self._shape_diags.add(nid)
        # A conflict inside a tuple leaves the other positions typed, which
        # is what the unpack that follows needs; the DIAGNOSTIC is what marks
        # the program wrong, and it points at the pair that disagrees
        self._diag(
            f'{what} gets both {render_ty(conflict[0])} and {render_ty(conflict[1])}, '
            f'which are different types', node,
            self._shape_unknown(conflict[0], conflict[1], node),
            fix='make both branches the same type')
        return joined

    @staticmethod
    def _shape_unknown(left: str, right: str, node: ast.AST) -> Unknown | None:
        """
        The provenance of a type lost to a shape conflict.

        :param left: One type
        :param right: The other type
        :param node: Where they meet
        :return: The provenance, or None when the conflict is not one of shape
        """
        if shape_conflict(left, right) is None:
            return None
        return Unknown(reason='shape-mismatch', line=getattr(node, 'lineno', 0),
                       col=getattr(node, 'col_offset', 0),
                       detail=f'{render_ty(left)} vs {render_ty(right)}')

    def _unknown(self, reason: str, node: ast.AST, detail: str = '') -> Unknown:
        return Unknown(reason=reason, line=getattr(node, 'lineno', 0),
                       col=getattr(node, 'col_offset', 0), detail=detail)

    def _diag(self, message: str, node: ast.AST, origin: Unknown | None = None,
              fix: str = '') -> None:
        self.table.diags.append(Diag(
            message=message, line=getattr(node, 'lineno', 0),
            col=getattr(node, 'col_offset', 0), origin=origin, fix=fix))

    # --- the per-node type view ------------------------------------------

    def _ty_of(self, node: ast.AST) -> str:
        """
        The type THIS walk gave a node.

        Not the stamp: the stamp is the join over every context that reached
        the node, and a context has to derive its own answers -- its return
        type, its overload pins -- from what IT found, or an int context that
        shares a body with a float one would read the float back.

        :param node: A node this walk has already visited
        :return: Its type in the running context
        """
        nid = node_id(node)
        if nid is not None and nid in self._ty:
            return self._ty[nid]
        return get_ty(node)

    def _stamp(self, node: ast.expr, ty: str) -> str:
        """
        Record a node's type in this walk, and JOIN it into the node's stamp.

        The stamp is what the later passes read, and they see one tree, not one
        per context -- so it has to be the type that is true for every context
        the node is reached in. Joining is also what makes the loop fixpoint
        safe: a binding only ever moves upward, so re-walking a body can widen
        a stamp but never silently narrow one.

        :param node: The node being typed
        :param ty: Its type in the running context
        :return: ``ty``
        """
        nid = node_id(node)
        if nid is not None:
            self._ty[nid] = ty
        existing = getattr(node, TY_ATTR, None)
        set_ty(node, ty if existing is None else join(existing, ty))
        return ty

    # --- entry point -----------------------------------------------------

    def run(self, tree: ast.Module) -> None:
        """Walk a module: the lib aliases, the imports, the definitions, the body."""
        self._factory = FactoryFields(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == 'pynecore':
                self._lib_aliases.update(a.asname or a.name for a in node.names)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith('pynecore.lib'):
                        self._lib_aliases.add(alias.asname or alias.name.split('.')[0])
        # A module with no explicit import still spells lib references
        # ``lib.<name>`` after normalization
        self._lib_aliases.add('lib')

        self._branches = _BranchIndex(tree.body)
        self._module_rebinds = {
            name: [(position, self._branches.of(position)) for position in positions]
            for name, positions in _bound_positions(tree.body).items()}
        self._rebound[''] = set(self._module_rebinds)
        self._collect_imports(tree)
        self._collect_classes(tree)
        self._sec_writes = _security_writes(tree)
        self._collect(tree.body, '')
        self._body(tree.body)
        self._flush_pending()
        self._stamp_instance_vectors(tree)
        self._suppress_pins(tree)
        # Where the types ran out, once per cause: the coverage meter of a
        # hand-written script, the first error of an edge module
        self.table.diags.extend(unknown_diags(tree, self.table))
        # The implementations are only final once every body has been walked:
        # an unannotated return only gets its type from the walk. Published
        # here so a consumer -- the module interface, and through it another
        # module's call sites -- reads the same shapes the selection does.
        for key in self._overload_groups:
            self.table.groups[key] = tuple(self._impl_sigs(key))
        # What the module interface publishes. Computed here rather than from
        # the shapes alone, because the answer is POSITIONAL: which binding a
        # name ends the module's run under is what an importer receives, and
        # the groups are only known once every definition has been collected
        self.table.exportable = _exportable_names(
            tree.body, self._module_rebinds, self._overload_groups)
        self._publish_methods()

    def _publish_methods(self) -> None:
        """
        Give every class the signatures of the methods declared on it.

        Last, because a method's return type is only final once its body has
        been walked. What this publishes is what a DEPENDENT needs: a receiver
        of an imported class reaches the method through the class, not through
        the module's own name for it.

        A method on a class this module does NOT declare goes to the
        extensions instead: it belongs to no class of ours, and a dependent
        finds it by searching the modules it imports, the way the runtime does.
        """
        keys = {id(node): key for key, defs in self._defs.items() for node in defs}
        for cid, methods in self._class_methods.items():
            sig = self.table.class_sigs.get(cid)
            for name, node in methods.items():
                func = self.table.funcs.get(keys.get(id(node), ''))
                if func is None:
                    continue
                shape = impl_sig(node, func.ret, self._ty_of, self._classes)
                positional = list(node.args.posonlyargs) + list(node.args.args)
                published = ExportSig(
                    name=name, kind='function', params=shape.params,
                    required=shape.required, open_ended=shape.open_ended, ret=shape.ret,
                    annotated=all(annotation_type(arg.annotation, self._classes) != UNKNOWN
                                  for arg in positional),
                    line=_line(node), names=shape.names)
                if sig is not None:
                    sig.methods[name] = published
                else:
                    # A method on a class another module declares: it belongs
                    # to no class of ours, and travels as an extension
                    self.table.extensions.setdefault(cid, {})[name] = published

    def _collect_imports(self, tree: ast.Module) -> None:
        """
        Record what the module's own imports bind, before anything is typed.

        Module level only, and for the same reason the isolation pass reads
        only those: a function-level import binds a local of that scope, and a
        relative one names a package this pass has no anchor for. ``pynecore``
        itself is left out because the lib registry already owns those names,
        and the standard library because it publishes no Pine interface.

        The ``shadowed_namespace`` bindings come second: what they merge is one
        of the imports, so the import map has to be complete first.

        One name, one import. A name two statements bind is recorded as
        unusable instead: the map holds a single entry per name, and picking
        either of two imports would type the call against a module it may
        never reach -- two spellings in exclusive branches have no answer at
        all, and a later import simply replaces the earlier one.

        :param tree: The module being walked
        """
        statements = list(_module_statements(tree.body))
        for stmt in statements:
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    if alias.asname:
                        self._add_import(alias.asname, _Import(alias.name, ()), alias)
                    else:
                        # ``import a.b.c`` binds ``a``, and the rest of the
                        # path is spelled out again at every use
                        head = alias.name.split('.')[0]
                        self._add_import(head, _Import(head, ()), alias)
            elif isinstance(stmt, ast.ImportFrom) and stmt.module and not stmt.level:
                for alias in stmt.names:
                    self._add_import(alias.asname or alias.name,
                                     _Import(stmt.module, (alias.name,)), alias)
        for stmt in statements:
            self._add_shadowed(stmt)

    def _collect_classes(self, tree: ast.Module) -> None:
        """
        Every class an annotation of this module may name, with what it holds.

        A UDT is a type, so a parameter annotated with one is annotated --
        reading such a name as unknown made the parameter behave like an
        unannotated one and took the whole export out of the typed world for
        its callers. It is a type with FIELDS, too, so reading it as an
        anonymous object lost every ``obj.field`` behind it. Which names ARE
        classes cannot be answered from the annotation alone: ``Pivot`` is a
        class here and a stray name there, so the map has to be built before
        anything reads an annotation.

        A class is identified by (module, name) and never by the bare name:
        two libraries each publishing a ``Settings`` publish two different
        types, and a value of one must not answer a field of the other.

        Two passes, because a field annotation may name a class declared
        further down or the class itself: the NAMES are resolved first, then
        every field is typed against the complete map.

        The module's own classes come from the tree, whatever scope they are
        declared in -- a forward reference names a class that stands further
        down, and a nested one is nameable from the body it lives in. An
        imported one is resolved through the same interface the calls go
        through, and only for a name some annotation actually spells: an
        import nothing annotates with is not worth an interface lookup.

        The map is over SPELLINGS: an imported class is keyed by the path
        that reaches it (``Settings`` for ``from m import Settings``,
        ``m.Settings`` for a namespace import), because identity is (module,
        name) and two libraries' same-named classes are two different types.
        Keying by the leaf made the second one unreachable -- both spellings
        resolved to whichever interface was consulted first, and every field
        read behind the loser was answered with the winner's type. The
        module's OWN classes stay under their bare names, which is the only
        spelling that reaches them, and a class declared inside a function is
        nameable from every scope. What IS excluded is a class name the module
        binds again at module scope (``class Amount: ...`` then ``Amount =
        int``): the name is that binding's, and reading an annotation on it as
        an object would type against a class nothing holds.

        :param tree: The module being walked
        """
        own = self._own_class_nodes(tree)
        module_key = self.table.module_path or ''
        names = {name: class_id(module_key, name) for name in own}
        imported: dict[str, ClassSig] = {}
        for spelled, node in _annotation_names(tree).items():
            parts = spelled.split('.')
            if spelled in names or spelled in imported:
                continue
            if parts[0] in self._multi_imports:
                continue
            binding = self._imports.get(parts[0])
            if binding is None:
                continue
            if len(parts) == 1:
                # ``from m import C``: the spelling consumed the name already
                if not binding.attrs:
                    continue
                hops, wanted = binding.attrs[:-1], binding.attrs[-1]
            else:
                hops, wanted = binding.attrs + tuple(parts[1:-1]), parts[-1]
            interface = self._interface_of(binding.module, hops, node)
            found = None if interface is None else interface.classes.get(wanted)
            if found is not None:
                imported[spelled] = found
        self._classes = {**{name: sig.id for name, sig in imported.items()}, **names}
        self.table.classes = self._classes
        for sig in imported.values():
            self._class_sigs.setdefault(sig.id, sig)
        self._declare_classes(tree, own, names)

    def _declare_classes(self, tree: ast.Module, own: dict[str, ast.ClassDef],
                         names: dict[str, str]) -> None:
        """
        Type the fields of this module's own classes, and collect its methods.

        Every ``@method`` the module declares is kept, whether it attaches to
        one of these classes or to an imported one -- what tells the two apart
        is where the finished signature is published.

        :param tree: The module being walked
        :param own: Class name -> its declaration
        :param names: Class name -> class id, for this module's own classes
        """
        methods = self._declared_methods(tree)
        for name, node in own.items():
            cid = names[name]
            fields: dict[str, str] = {}
            required = 0
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    fields[stmt.target.id] = annotation_type(stmt.annotation, self._classes)
                    if stmt.value is None:
                        required += 1
            self.table.class_sigs[cid] = ClassSig(
                name=name, id=cid, fields=fields, required=required, methods={})
        self._class_methods = methods
        self._class_sigs.update(self.table.class_sigs)

    def _declared_methods(self, tree: ast.Module) -> dict[str, dict[str, ast.FunctionDef]]:
        """
        The Pine methods this module declares, by the class they attach to.

        A method is a free function carrying the ``@method`` decorator whose
        FIRST parameter is annotated with the class -- ``def bump(self: Pivot,
        amt: int)``, wherever it stands, since a compiled script declares its
        methods inside ``main``. What is recorded is the bare NAME: which
        definition a receiver reaches is a scope question, and the ordinary
        outward search is what answers it.

        The receiver's class needs no relation to this module: Pine lets one
        library declare a method on another library's UDT, and the runtime
        finds it by searching the modules the script imports. Such a method is
        an EXTENSION, and it is told apart where the results are published.

        :param tree: The module being walked
        :return: class id -> method name -> its definition
        """
        out: dict[str, dict[str, ast.FunctionDef]] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not _is_method(node):
                continue
            positional = list(node.args.posonlyargs) + list(node.args.args)
            if not positional:
                continue
            cid = class_of(annotation_type(positional[0].annotation, self._classes))
            if cid is not None:
                out.setdefault(cid, {})[node.name] = node
        return out

    def _own_class_nodes(self, tree: ast.Module) -> dict[str, ast.ClassDef]:
        """
        The classes this module declares and still holds, by name.

        A name the module binds at module scope by anything but its own
        ``class`` statement is not one: what stands under it at import time is
        the assignment's value, so an annotation naming it describes an object
        that never exists. The comparison is by POSITION, the same way an
        import's own binding is told from a rebinding of it -- the class
        statement is itself a module-scope binding, so nothing else would tell
        the two apart.

        :param tree: The module being walked
        :return: Class name -> its declaration, the LAST one under that name
        """
        declared: dict[str, list[ast.ClassDef]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                declared.setdefault(node.name, []).append(node)
        out: dict[str, ast.ClassDef] = {}
        for name, nodes in declared.items():
            positions = {(_line(node), _col(node)) for node in nodes}
            if all(position in positions
                   for position, _ in self._module_rebinds.get(name, ())):
                out[name] = nodes[-1]
        return out

    def _add_import(self, name: str, binding: _Import, node: ast.AST) -> None:
        """
        Record one import binding, unless the module behind it is not ours.

        :param name: The name the import binds
        :param binding: What it names
        :param node: The ``alias`` node, whose position identifies the binding
        """
        if binding.module == 'pynecore' or binding.module.startswith('pynecore.') \
                or is_stdlib(binding.module):
            return
        if name in self._imports:
            self._multi_imports.add(name)
        self._imports[name] = binding
        self._import_positions.setdefault(name, set()).add((_line(node), _col(node)))

    def _add_shadowed(self, stmt: ast.stmt) -> None:
        """
        Record a ``N = shadowed_namespace(<import>, lib.<ns>)`` binding.

        :param stmt: A module-level statement, which may be that assignment
        """
        if isinstance(stmt, ast.Assign):
            targets, value = stmt.targets, stmt.value
        elif isinstance(stmt, ast.AnnAssign):
            targets, value = [stmt.target], stmt.value
        else:
            return
        if not isinstance(value, ast.Call) or len(value.args) != 2:
            return
        called = _dotted(value.func)
        if called is None or called.split('.')[-1] != 'shadowed_namespace':
            return
        library = value.args[0]
        if not isinstance(library, ast.Name):
            return
        source = self._imports.get(library.id)
        namespace = self._lib_name(value.args[1])
        if source is None or namespace is None:
            return
        for target in targets:
            if isinstance(target, ast.Name):
                if target.id in self._imports or target.id in self._shadowed:
                    self._multi_imports.add(target.id)
                self._shadowed[target.id] = _Shadowed(source=source, namespace=namespace)
                self._import_positions.setdefault(target.id, set()).add(
                    (_line(target), _col(target)))

    def _collect(self, body: list[ast.stmt], scope: str) -> None:
        """
        Record every definition and its annotated signature, before any walk.

        Every nested statement list is descended into, not just the plain ones:
        a ``def`` inside an ``if`` or a ``try`` is a definition like any other,
        and skipping those left such a helper with no signature at all, so
        every call to it read UNKNOWN.

        :param body: The statements to scan
        :param scope: Scope id the definitions live in, empty at module level
        """
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                key = self._qualify(scope, stmt.name)
                self._defs.setdefault(key, []).append(stmt)
                params = [annotation_type(a.annotation, self._classes)
                          for a in list(stmt.args.posonlyargs) + list(stmt.args.args)]
                declared = annotation_type(stmt.returns, self._classes)
                annotated = declared != UNKNOWN
                if _is_overload(stmt):
                    # One name, several implementations: each contributes its
                    # own return, and the group's type is what they agree on
                    self._overload_groups.add(key)
                    returns = self._group_returns.setdefault(key, [])
                    nid = node_id(stmt)
                    if nid is not None:
                        self._group_slot[nid] = len(returns)
                    returns.append(declared)
                    declared = overload_result(returns)
                    annotated = annotated and self._annotated_returns.get(key, True)
                self.table.funcs[key] = FuncSig(
                    name=stmt.name, params=params, ret=declared, line=_line(stmt))
                self._annotated_returns[key] = annotated
                # A parameter binds its name in the body's own scope, so it
                # shadows an outer definition of that name like an assignment
                own = self._rebound.setdefault(key, set())
                own.update(arg.arg for arg in _every_param(stmt))
                own.update(arg.arg for arg in (stmt.args.vararg, stmt.args.kwarg)
                           if arg is not None)
                own.update(_bound_names(stmt.body))
                self._collect(stmt.body, key)
            else:
                for nested in _statement_lists(stmt):
                    self._collect(nested, scope)

    # --- statements ------------------------------------------------------

    def _body(self, body: list[ast.stmt]) -> None:
        for stmt in body:
            self._stmt(stmt)

    def _stmt(self, stmt: ast.stmt) -> None:
        match stmt:
            case ast.FunctionDef() | ast.AsyncFunctionDef():
                self._definition(stmt)
            case ast.Assign():
                ty = self._expr(stmt.value)
                for target in stmt.targets:
                    self._store(target, ty, stmt.value)
                    if ty == PINE_LOOP and isinstance(target, ast.Name) \
                            and isinstance(stmt.value, ast.Call):
                        self._frames[-1].loop_counters[target.id] = \
                            self._loop_bound(stmt.value, 'pine_loop', stmt.value.args, None)
            case ast.AnnAssign():
                declared: str | None = annotation_type(stmt.annotation, self._classes)
                value_ty = self._expr(stmt.value) if stmt.value is not None else None
                unknown = None
                if bare_wrapper(stmt.annotation):
                    # ``x: Series = expr`` and ``var x = expr``: the wrapper says
                    # how the variable lives, the VALUE says what it holds. A
                    # bare declaration without a value binds nothing yet -- the
                    # assignment that follows is what gives the name its type
                    declared = value_ty
                elif declared != UNKNOWN and value_ty is not None \
                        and value_ty not in _UNCHECKED and not _fits(declared, value_ty):
                    # An explicit annotation is a DECLARATION, the way Pine's
                    # ``int x = ...`` is -- and Pine rejects a value that does
                    # not fit it. Trusting the annotation over the value would
                    # carry the lie into every pin downstream
                    spelled = ast.unparse(stmt.target)
                    self._node_diag(
                        f"'{spelled}' is declared {render_ty(declared)} and assigned "
                        f"{render_ty(value_ty)}", stmt.target, 'type-mismatch', spelled,
                        fix=f"declare it {render_ty(value_ty)}, or assign {render_ty(declared)}")
                    unknown = self._unknown('type-mismatch', stmt.target, spelled)
                    declared = UNKNOWN
                if declared is not None:
                    self._store(stmt.target, declared,
                                stmt.value if stmt.value is not None else stmt,
                                _spells_series(stmt.annotation), unknown)
                elif isinstance(stmt.target, ast.Name) and _spells_series(stmt.annotation):
                    # ``x: Series`` with no value declares how the name lives
                    # before the assignment that gives it its type
                    self._frames[-1].declared_series.add(stmt.target.id)
            case ast.AugAssign():
                value_ty = self._expr(stmt.value)
                current = self._target_type(stmt.target)
                self._store(stmt.target, binop_type(stmt.op, current, value_ty), stmt)
            case ast.Return():
                if stmt.value is not None:
                    self._expr(stmt.value)
            case ast.If():
                self._expr(stmt.test)
                self._body(stmt.body)
                self._body(stmt.orelse)
            case ast.While():
                self._loop(stmt, lambda: (self._expr(stmt.test), self._body(stmt.body)))
            case ast.For() | ast.AsyncFor():
                iter_ty = self._expr(stmt.iter)
                if iter_ty in SCALARS:
                    self._node_diag(
                        f"'{ast.unparse(stmt.iter)[:40]}' is a {render_ty(iter_ty)}; a for "
                        f"loop iterates a range or an array", stmt.iter, 'not-iterable',
                        render_ty(iter_ty), fix='loop over range(...) or an array')
                    iter_ty = UNKNOWN
                self._store_iteration(stmt, iter_ty)
                self._loop(stmt, lambda: self._body(stmt.body))
                self._body(stmt.orelse)
            case ast.Expr():
                self._expr(stmt.value)
            case ast.With() | ast.AsyncWith():
                for item in stmt.items:
                    self._expr(item.context_expr)
                self._body(stmt.body)
            case ast.Try():
                self._body(stmt.body)
                for handler in stmt.handlers:
                    self._body(handler.body)
                self._body(stmt.orelse)
                self._body(stmt.finalbody)
            case ast.Match():
                self._expr(stmt.subject)
                for branch in stmt.cases:
                    if branch.guard is not None:
                        self._expr(branch.guard)
                    self._body(branch.body)
            case ast.ClassDef():
                self._class_body(stmt)
            case _:
                # Import, Pass, Break, Continue, Global, Nonlocal, Delete:
                # nothing to type
                for child in ast.iter_child_nodes(stmt):
                    if isinstance(child, ast.expr):
                        self._expr(child)

    def _class_body(self, stmt: ast.ClassDef) -> None:
        """
        Type a class body without letting its fields bind names outside it.

        ``price: float = 0.0`` in a class declares a FIELD, and a field is
        reached through an instance -- it is not a variable of the scope the
        class stands in. Walking the body in the enclosing frame bound it as
        one, so a script that also has a ``price`` of its own met the field's
        type in it: a widening at best, and for two different shapes a
        conflict the program does not have.

        The frame keeps the enclosing scope's id, so what the body's
        expressions record -- a call site, a pin -- still belongs where it is
        written.

        :param stmt: The class statement
        """
        for value in self._factory.of(stmt):
            self._factory_fields.add(id(value))
        self._frames.append(_Frame(self._frames[-1].scope))
        try:
            self._body(stmt.body)
        finally:
            self._frames.pop()

    def _loop(self, stmt: ast.stmt, walk) -> None:
        """
        Run a loop body until its bindings stop moving.

        A loop-carried variable is the one place a single forward pass is not
        enough: ``total = 0`` then ``total := total + close`` inside the body
        reads as int on the first pass and only becomes float on the second.
        The lattice is two high, so three passes are provably enough -- the
        third exists to confirm the second changed nothing.

        The loop STATEMENT is kept, not just a depth count: its span is what
        tells ``_reaches_def`` which rebindings the back edge puts ahead of a
        call written inside it.

        :param stmt: The loop statement being walked
        :param walk: Runs one pass over it
        """
        self._loop_stack.append(stmt)
        try:
            for _ in range(_MAX_LOOP_PASSES):
                before = self._snapshot()
                walk()
                if self._snapshot() == before:
                    return
        finally:
            self._loop_stack.pop()

    def _snapshot(self) -> list[dict[str, str]]:
        """The bindings every live scope holds right now."""
        return [{name: b.ty for name, b in frame.names.items()} for frame in self._frames]

    # --- definitions and contexts ----------------------------------------

    def _definition(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """
        Type a definition's defaults, and decide when its body is walked.

        The defaults belong to the ENCLOSING scope and are typed right here.
        The body is another matter: walking it now, with its unannotated
        parameters bound to UNKNOWN, would put an UNKNOWN into the join of
        every node a caller is about to type properly. So an ordinary function
        waits -- if nothing ever calls it, the pending flush walks it at the
        end of this scope, which is the only case where UNKNOWN is the truth.
        """
        for default in node.args.defaults:
            self._expr(default)
        for kw_default in node.args.kw_defaults:
            if kw_default is not None:
                self._expr(kw_default)

        key = self._qualify(self._scope, node.name)
        if key in self._overload_groups or is_script_entry(node):
            # An implementation of an overload group is already per-signature
            # -- the pin selects among them, so a call site has nothing to
            # instantiate. An entry point has exactly one context: the runner
            # calls it with no arguments, so its defaults ARE its parameters.
            self._declaration_context(key, node)
            return
        self._pending[-1].append((key, node))

    def _flush_pending(self) -> None:
        """
        Walk the definitions of this scope that no call site ever typed.

        Order decides whether that set is right. A helper whose only caller is
        another definition waiting here must not be walked first: the walk
        would bind its parameters to UNKNOWN, and that answer then JOINS into
        every type its real call sites are about to establish. So a definition
        goes last if anything still waiting calls it -- callers first, and by
        the time a called helper comes up it already has its contexts and is
        skipped. A cycle has no such order and takes the first one left.
        """
        remaining = [(key, node, _called_names(node)) for key, node in self._pending[-1]]
        self._pending[-1].clear()
        while remaining:
            picked = next(
                (index for index, (_, node, _) in enumerate(remaining)
                 if not any(other != index and node.name in names
                            for other, (_, _, names) in enumerate(remaining))),
                0)
            key, node, _ = remaining.pop(picked)
            nid = node_id(node)
            if nid is None or nid not in self._walked:
                self._declaration_context(key, node)

    def _declared_params(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
        """
        The parameter types a definition states on its own, with no caller.

        An annotation states one outright. A script entry's default states one
        too, because the runner passes no arguments -- see ``_param_defaults``.
        Anything else is UNKNOWN, which is the honest answer for a function
        nothing calls.

        :param node: The definition to read
        :return: One type per parameter, positional first then keyword-only
        """
        defaults = _param_defaults(node) if is_script_entry(node) else {}
        out: list[str] = []
        for arg in _every_param(node):
            ty = annotation_type(arg.annotation, self._classes)
            if ty == UNKNOWN:
                default = defaults.get(arg.arg)
                ty = UNKNOWN if default is None else self._ty_of(default)
            out.append(ty)
        return tuple(out)

    def _declaration_context(self, key: str,
                             node: ast.FunctionDef | ast.AsyncFunctionDef) -> ContextResult | None:
        """Analyse a body in the only context its own definition describes."""
        return self._analyse(key, node, self._declared_params(node))

    def _analyse(self, key: str, node: ast.FunctionDef | ast.AsyncFunctionDef,
                 params: tuple[str, ...], parents: list[_Frame] | None = None,
                 site: ast.Call | None = None) -> ContextResult | None:
        """
        Walk one function body under one parameter-type tuple.

        The result is memoized on (function id, parameter tuple, CALLING
        context, DEFINITION, enclosing types). Everything past the parameters
        is there because a body reads more than its own arguments:

        * The calling context: the enclosing scopes hold different types in
          different contexts of the enclosing function -- ``main -> helper ->
          inner`` is the shape where dropping it would let one ``inner`` answer
          stand for two different ``helper`` instantiations.
        * The definition: an overload group spells SEVERAL implementations
          under one id, and they are analysed under the parameter types they
          declare -- two of them whose annotations collapse onto the same Pine
          characters (``list[int]`` and ``list[float]`` are both objects) would
          otherwise share one context and only the first body would be walked.
        * The types of the enclosing bindings the body READS: the calling
          context's id does not move while the loop fixpoint widens a
          loop-carried variable, so a callee closing over one would be analysed
          once, under the type of the first pass, and would keep the pin that
          pass justified.

        :param key: Scope-qualified id of the function
        :param node: Its definition
        :param params: Type of each parameter, positional first then kw-only
        :param parents: The callee's lexical parent frames; the live ones when
                        the definition is being analysed where it stands
        :param site: The call node this analysis was started from, when one was
        :return: The context, or None when it is re-entrant or over budget
        """
        # A module-level function's only enclosing scope is the module, and the
        # module has exactly one context -- so who calls it adds nothing to its
        # environment, and every caller shares one analysis. Only a NESTED
        # function reads a scope that differs per context, and that is the case
        # the calling context is in the key for.
        outer = self._frames
        env = list(outer if parents is None else parents)
        nid = node_id(node)
        origin = self._context if '·' in key else 0
        memo: ContextKey = (key, params, origin, nid, self._free_env(node, env))
        found = self.table.contexts.get(memo)
        if found is not None:
            return found
        guard = (key, params, nid)
        if guard in self._in_progress or len(self.table.contexts) >= _MAX_CONTEXTS:
            return None

        cid = self._supersede((key, params, origin, nid, node_id(site)), memo)
        result = ContextResult(cid=cid, key=key, params=params)
        self.table.contexts[memo] = result
        self._in_progress.add(guard)
        declared_global, declared_nonlocal = _declared_names(node.body)
        self._frames = env + [_Frame(key, declared_global=declared_global,
                                     declared_nonlocal=declared_nonlocal)]
        self._contexts.append(cid)
        self._pin_sink.append(result.pins)
        self._pending.append([])
        try:
            self._bind_params(node, params)
            self._body(node.body)
            declared = annotation_type(node.returns, self._classes)
            result.ret = declared if declared != UNKNOWN else self._return_type(node)
            self._flush_pending()
            bound = self._frames[-1].names
        finally:
            self._pending.pop()
            self._pin_sink.pop()
            self._contexts.pop()
            self._frames = outer
            self._in_progress.discard(guard)

        if nid is not None:
            self._walked.add(nid)
        self._merge_bindings(key, bound)
        self._record_signature(key, node, result)
        return result

    def _supersede(self, anchor: tuple, memo: ContextKey) -> int:
        """
        The context id a fresh analysis of ONE call site takes over.

        A memo entry that differs from this one only in the types of the
        enclosing bindings is not a second instantiation, it is the same one
        answered before those types settled -- the loop fixpoint re-walking a
        body after a loop-carried variable widened. Its verdict has to
        DISAPPEAR, not stand alongside the new one: the two together would look
        like a body reached in two contexts, the call sites inside it would be
        marked instance-varying over a difference no instance ever sees, and
        the stale pin would be the one a per-instance channel handed out.

        Which is why the anchor carries the ORIGINATING CALL NODE. Only the
        same site re-analysed says "the answer I gave has gone stale"; a
        different site under a different environment is a second instantiation
        and must keep its own context, or one nested closure called once
        before and once after its captured variable widens would collapse into
        whichever call the walk reached last.

        Taking the stale context's id over is what erases it. Every pin the
        stale walk justified is filed under that id, and the new walk visits
        the same sites in the same body, so ``_record_pin``'s same-context
        overwrite replaces each of them -- the node ends up carrying what the
        last walk stands behind, exactly as it does for a call site the
        fixpoint re-walks directly.

        :param anchor: The memo key with the call site in place of the
                       enclosing types
        :param memo: The full memo key of the analysis about to run
        :return: The context id to run under
        """
        previous = self._anchors.get(anchor)
        self._anchors[anchor] = memo
        if previous is None:
            cid = self._next_context
            self._next_context += 1
            return cid
        return self.table.contexts.pop(previous).cid

    def _free_env(self, node: ast.FunctionDef | ast.AsyncFunctionDef,
                  frames: list[_Frame]) -> tuple[tuple[str, str], ...]:
        """
        The types the enclosing scopes hold for the names a body reads.

        :param node: The definition about to be analysed
        :param frames: The lexical parent frames its body will see
        :return: (name, type) for every free name that resolves, sorted
        """
        out: list[tuple[str, str]] = []
        lexical, module_level = self._free_names(node)
        for name in lexical:
            for frame in reversed(frames):
                found = frame.names.get(name)
                if found is not None:
                    out.append((name, found.ty))
                    break
        # A ``global`` name resolves to the MODULE's binding however many
        # scopes with a same-named local stand between, so the search that
        # walks outward would answer with the wrong one
        module = frames[0] if frames else None
        for name in module_level:
            found = None if module is None else module.names.get(name)
            if found is not None:
                out.append((name, found.ty))
        out.sort()
        return tuple(out)

    def _free_names(self, node: ast.FunctionDef | ast.AsyncFunctionDef
                    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """
        The enclosing-scope names one definition's body reads, computed once.

        Nested definitions count as part of the body: they read the same
        enclosing scopes, and a call to the outer definition is what makes them
        run at all. What they do NOT do is BIND for it, which is why the answer
        is computed one lexical scope at a time (``_scope_free``) rather than
        over one flattened walk.

        :param node: The definition to scan
        :return: (names resolved lexically, ``global`` names), each sorted
        """
        nid = node_id(node)
        cached = None if nid is None else self._free.get(nid)
        if cached is not None:
            return cached
        # The definition's own name is bound where the definition stands, so a
        # body that calls itself is reading its own scope, not the one above
        lexical, module_level = _scope_free(node)
        free = (tuple(sorted(lexical - {node.name})),
                tuple(sorted(module_level - {node.name})))
        if nid is not None:
            self._free[nid] = free
        return free

    def _bind_params(self, node: ast.FunctionDef | ast.AsyncFunctionDef,
                     params: tuple[str, ...]) -> None:
        """Bind a context's parameters in the freshly pushed frame."""
        bindings = self._bindings()
        for arg, ty in zip(_every_param(node), params):
            unknown = None
            if ty == UNKNOWN:
                unknown = self._unknown('unannotated-param', arg, arg.arg)
            bindings[arg.arg] = Binding(name=arg.arg, ty=ty, line=_line(arg), unknown=unknown,
                                        series=_spells_series(arg.annotation))

    def _merge_bindings(self, scope: str, names: dict[str, Binding]) -> None:
        """
        Fold a finished context's bindings into the scope's public view.

        ``table.bindings`` is one entry per SCOPE, so what it can report about
        a name analysed twice is the join of the two answers -- the same thing
        the node stamps carry, and for the same reason. The contexts themselves
        stay apart in ``table.contexts``.
        """
        target = self.table.bindings.setdefault(scope, {})
        for name, binding in names.items():
            existing = target.get(name)
            if existing is None:
                target[name] = Binding(name=name, ty=binding.ty, line=binding.line,
                                       unknown=binding.unknown, series=binding.series)
                continue
            existing.series = existing.series or binding.series
            joined = join(existing.ty, binding.ty)
            existing.ty = joined
            if joined == UNKNOWN and existing.unknown is None:
                existing.unknown = binding.unknown

    def _record_signature(self, key: str, node: ast.FunctionDef | ast.AsyncFunctionDef,
                          result: ContextResult) -> None:
        """Fold a context's parameter and return types into the signature."""
        signature = self.table.funcs.get(key)
        if signature is None:
            return
        positional = len(node.args.posonlyargs) + len(node.args.args)
        observed = list(result.params[:positional])
        if key in self._signed and len(signature.params) == len(observed):
            signature.params = [join(a, b) for a, b in zip(signature.params, observed)]
        else:
            signature.params = observed

        if key in self._overload_groups:
            # One implementation of a group just finished: fill in its own
            # return if it declared none, then re-derive what they agree on
            returns = self._group_returns.get(key, [])
            slot = self._group_slot.get(node_id(node) or -1)
            if slot is not None and slot < len(returns) and returns[slot] == UNKNOWN:
                returns[slot] = result.ret
            signature.ret = overload_result(returns)
        elif self._annotated_returns.get(key):
            signature.ret = result.ret
        elif key in self._signed:
            signature.ret = join(signature.ret, result.ret)
        else:
            signature.ret = result.ret
        self._signed.add(key)

    def _return_type(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """
        Join every ``return`` in a function body, ignoring nested functions.

        The join is the reporting one: two returns of different SHAPE -- a
        tuple against a scalar, or two tuples of different length -- are a
        Pine error rather than something to widen, and the message points at
        the return that disagreed.
        """
        result: str | None = None
        for stmt in _walk_own_scope(node):
            if isinstance(stmt, ast.Return):
                ty = VOID if stmt.value is None else self._ty_of(stmt.value)
                result = ty if result is None else \
                    self._joined(result, ty, stmt, 'the function')
        return VOID if result is None else result

    def _store(self, target: ast.expr, ty: str, source: ast.AST, series: bool = False,
               unknown: Unknown | None = None) -> None:
        """
        Bind an assignment target, recursing into tuple/list targets.

        :param target: The target expression
        :param ty: The type stored
        :param source: The node the binding is attributed to
        :param series: Whether the declaration spells a series
        :param unknown: The provenance to record for an UNKNOWN store, when
                        the caller has a more precise one than "the value"
        """
        if isinstance(target, ast.Name):
            self._stamp(target, ty)
            if ty == UNKNOWN and unknown is None:
                unknown = self._unknown('unknown-value', source)
            self._bind(target.id, ty, source, unknown, series)
        elif isinstance(target, (ast.Tuple, ast.List)):
            self._stamp(target, ty if is_tuple(ty) else OBJECT)
            self._distribute(target, ty, source)
        elif isinstance(target, ast.Attribute):
            self._field_store(self._expr(target.value), target, ty)
            self._stamp(target, ty)
        elif isinstance(target, ast.Subscript):
            self._expr(target.value)
            self._stamp(target, ty)

    def _field_store(self, base: str, node: ast.Attribute, ty: str) -> None:
        """
        Check a ``obj.field = value`` against the receiver's class.

        A store is the read's mirror: the class has to declare the field, and
        what is stored has to fit what the field holds. A scalar has no
        fields at all, and a bare object cannot be checked, which is a
        complaint about the receiver's typing.

        :param base: The receiver's type
        :param node: The attribute target
        :param ty: The type being stored
        """
        cid = class_of(base)
        if cid is not None:
            sig = self._class_sig(cid, node)
            if sig is None:
                return
            found = sig.fields.get(node.attr)
            if found is None:
                self._node_diag(
                    f"'{sig.name}' has no field '{node.attr}'", node,
                    'unknown-field', f'{sig.name}.{node.attr}',
                    fix=f"declare '{node.attr}' on '{sig.name}', or assign a field it has")
            elif ty not in _UNCHECKED and not _fits(found, ty):
                self._node_diag(
                    f"'{sig.name}.{node.attr}' holds {render_ty(found)} and is assigned "
                    f"{render_ty(ty)}", node, 'type-mismatch', f'{sig.name}.{node.attr}',
                    fix=f'assign {render_ty(found)}')
            return
        if base in SCALARS:
            self._node_diag(
                f"a {render_ty(base)} has no field '{node.attr}' to assign", node,
                'unknown-field', node.attr, fix='assign a field of a @udt object')
        elif base == OBJECT and not self._is_reference(node.value):
            self._node_diag(
                f"the class of '{_dotted(node.value) or ast.unparse(node.value)}' is not "
                f"known here, so its field '{node.attr}' cannot be checked", node,
                'unknown-class', node.attr, fix='annotate the value with the type it holds')

    def _distribute(self, target: ast.Tuple | ast.List, ty: str, source: ast.AST) -> None:
        """
        Bind the names of an unpack, one element type each.

        ``[a, b] = f()`` is where a tuple pays off: each name gets the type of
        ITS position, which is what puts an int-typed half of a pair back into
        the typed world. A value that is not a tuple says nothing about any of
        them, and one of the WRONG arity is a Pine error, not an approximation
        to make -- Pine matches the arity of an unpack against the type, so
        both names are unknown and the mismatch is reported where it stands.

        :param target: The tuple or list target
        :param ty: Type of the value being unpacked
        :param source: The node the binding is attributed to
        """
        names = [element.value if isinstance(element, ast.Starred) else element
                 for element in target.elts]
        if any(isinstance(element, ast.Starred) for element in target.elts):
            self._diag(
                'a starred target takes an unknown number of elements, which Pine '
                'has no form for', target,
                self._unknown('shape-mismatch', target, render_ty(ty)),
                fix='unpack exactly as many names as the tuple has elements')
            for name in names:
                self._store(name, UNKNOWN, source)
            return
        elements = elements_of(ty)
        if elements and len(elements) != len(names):
            self._diag(
                f'{render_ty(ty)} has {len(elements)} elements, and this unpacks '
                f'{len(names)}', target,
                self._unknown('shape-mismatch', target,
                              f'{len(elements)} vs {len(names)}'),
                fix=f'unpack {len(elements)} names, as many as the tuple has')
            elements = ()
        for index, name in enumerate(names):
            self._store(name, elements[index] if elements else UNKNOWN, source)

    def _target_type(self, target: ast.expr) -> str:
        """Current type of an augmented-assignment target."""
        if isinstance(target, ast.Name):
            found = self._lookup(target.id)
            return found.ty if found is not None else UNKNOWN
        if isinstance(target, ast.Attribute):
            # ``point.price += x`` reads the field before it writes it
            return self._e_Attribute(target)
        return UNKNOWN

    def _element_type(self, iter_node: ast.expr, iter_ty: str) -> str:
        """
        Type of a ``for`` loop variable.

        MEASURED: TradingView does NOT truncate a Pine ``for``. With
        ``R = input.int(14)``, ``for i = R / 8 to R / 4`` iterates i = 1.75 and
        2.75, so the counter is an int-TYPED variable carrying a fractional
        value -- exactly the law this whole pass exists for. The type is
        therefore the join of the bounds, and a native ``range`` over int
        arguments yields an int.

        Iterating a CONTAINER is the other half: ``for x in arr`` gives ``x``
        the array's element type, which is the shape's whole point.
        """
        if isinstance(iter_node, ast.Call):
            callee = _dotted(iter_node.func)
            if callee in ('range', 'pine_range', 'lib.pine_range'):
                bounds = [self._ty_of(a) for a in iter_node.args]
                if bounds and all(b in NUMERIC for b in bounds):
                    return INT if all(b == INT for b in bounds) else FLOAT
                return UNKNOWN
        return element_of(iter_ty)

    def _store_iteration(self, stmt: ast.For | ast.AsyncFor, iter_ty: str) -> None:
        """
        Bind what one pass of a ``for`` puts under its target.

        ``for [i, x] in arr`` compiles to ``for i, x in enumerate(arr)``, whose
        two halves are typed apart: the index is an int and the value is the
        array's element. Storing the pair as one opaque tuple lost both.

        :param stmt: The loop statement
        :param iter_ty: Type of the expression being iterated
        """
        target = stmt.target
        indexed = _enumerated(stmt.iter)
        if isinstance(target, (ast.Tuple, ast.List)) and indexed is not None \
                and len(target.elts) == 2:
            element = self._element_type(indexed, self._ty_of(indexed))
            self._stamp(target, OBJECT)
            self._store(target.elts[0], INT, stmt)
            self._store(target.elts[1], element, stmt)
            return
        self._store(target, self._element_type(stmt.iter, iter_ty), stmt)

    # --- expressions -----------------------------------------------------

    def _expr(self, node: ast.expr) -> str:
        """Type one expression, stamping it and everything under it."""
        method = getattr(self, f'_e_{type(node).__name__}', None)
        if method is None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    self._expr(child)
            ty = UNKNOWN
        else:
            ty = method(node)
        return self._stamp(node, ty)

    # Each ``_e_*`` returns the type; the caller stamps it.

    def _e_Constant(self, node: ast.Constant) -> str:
        # A stamp already on the node wins: the constant folder replaces a
        # Pine-typed subtree with its literal and records what the TYPE was,
        # which the Python literal alone can no longer tell (``14 / 8`` folds
        # to 1.75, indistinguishable from a float literal)
        existing = getattr(node, TY_ATTR, None)
        if existing is not None:
            return existing
        return constant_type(node.value)

    def _e_Name(self, node: ast.Name) -> str:
        """
        Type a bare name read.

        The searches run in the order the RUNTIME resolves the name: a live
        binding first, then the lib registry, then the plumbing the pipeline
        binds around a script. What is left is a name that denotes a TYPE --
        the ``float`` of the truthiness test's ``.__class__ is float``, the
        ``int`` of a ``na(int)``, a UDT named in a constructor -- and a class
        object is a known non-scalar, not a typing failure.

        :param node: The name node
        :return: The type of a read of it
        """
        if node.id == _DYN_DEFAULT:
            # ``DynamicDefaultTransformer``'s "no argument passed" marker: the
            # parameter holds nothing yet, and the prologue that tests for it
            # assigns the real default -- typeless, so that assignment decides
            return TYPELESS
        found = self._lookup(node.id)
        if found is not None:
            return found.ty
        entry = lib_types().get(node.id)
        if entry is not None:
            return entry['ty'] if entry['kind'] == 'value' else OBJECT
        builtin = BUILTIN_NAME_TYPES.get(node.id)
        if builtin is not None:
            return builtin
        if node.id in self._lib_aliases or node.id in lib_namespaces():
            # The head of a normalized lib reference: the module object itself
            return OBJECT
        return OBJECT if self._names_type(node) else UNKNOWN

    def _names_class(self, node: ast.expr) -> str | None:
        """
        The class a qualified spelling in value position names, if any.

        ``zigzag.Pivot`` is the class when ``zigzag`` is the namespace import
        that reaches it, resolved through the library's interface the way a
        constructor's callee is -- an annotation need not have spelled it
        first. A script value of the head's name makes the same spelling a
        field read, which is the other answer's business.

        :param node: The expression in value position
        :return: The class id, or None when the spelling names no class
        """
        if not isinstance(node, ast.Attribute):
            return None
        spelled = _dotted(node)
        return None if spelled is None else self._class_named(tuple(spelled.split('.')), node)

    def _names_type(self, node: ast.expr) -> bool:
        """
        Whether an expression in value position NAMES a type rather than a value.

        The compiled form spells a declared ``na`` as ``na(int)`` / ``NA(Line)``
        and the truthiness test as ``... .__class__ is float``, so a type name
        reaches value position. It is only a type name where nothing binds it:
        a script's own ``float = 3`` makes it an ordinary variable, and the
        binding search has already answered for that case.

        :param node: The expression to test
        :return: True when it resolves to a type
        """
        if isinstance(node, ast.Name) and self._lookup(node.id) is not None:
            return False
        return annotation_type(node, self._classes) != UNKNOWN

    def _e_Attribute(self, node: ast.Attribute) -> str:
        base = self._expr(node.value)
        if node.attr == '__class__':
            # ``PineTruthinessTransformer``'s float test reads the Python class
            # of a bound temporary; a class object is not a Pine value
            return OBJECT
        name = self._lib_name(node)
        if name is None:
            if self._names_class(node) is not None:
                # A class named through its module, ``zigzag.Pivot`` in a
                # ``na(zigzag.Pivot)``: the class object, as a bare class name
                return OBJECT
            if node.attr == 'new' and self._names_class(node.value) is not None:
                # ``zigzag.Settings.new``: the constructor Pine spells on a
                # type, a callable reference; what it builds is the call's
                return OBJECT
            return self._field_read(base, node)
        override = LIB_TYPE_OVERRIDES.get(name)
        if isinstance(override, str) and len(override) == 1:
            return override
        entry = lib_types().get(name)
        if entry is None:
            # A namespace, not a name in it: ``shadowed_namespace(x, lib.ta)``
            # passes the module object itself
            return OBJECT if name in lib_namespaces() else UNKNOWN
        if entry['kind'] == 'value':
            return entry['ty']
        if name == 'na':
            # In value position ``na`` is Pine's typeless absence marker, not
            # the predicate it names: ``module_property`` rewrites such a read
            # to ``lib._na_none``, but a view that has not run that pass yet
            # -- and every reader of this pass' own output -- still sees it
            return TYPELESS
        # A bare reference to a lib function is the function itself
        return OBJECT

    def _field_read(self, base: str, node: ast.Attribute) -> str:
        """
        Type a ``obj.field`` read from the receiver's class.

        A Pine object knows its class, so the field's DECLARED type is the
        answer -- that is the whole reason a shaped type carries a class id.
        Two failures are worth telling apart, and both get a diagnostic of
        their own: a field the class does not declare, and a receiver whose
        class was lost upstream (a bare object), which is where the fix has to
        go. A receiver that is not even known to be an object says nothing new
        -- its own UNKNOWN already carries the provenance.

        :param base: The receiver's type
        :param node: The attribute node
        :return: The field's type, or UNKNOWN
        """
        if base == PINE_LOOP:
            return self._loop_field(node)
        cid = class_of(base)
        if cid is not None:
            sig = self._class_sig(cid, node)
            if sig is None:
                return UNKNOWN
            found = sig.fields.get(node.attr)
            if found is not None:
                return found
            if node.attr in sig.methods or node.attr in self._class_methods.get(cid, ()):
                # ``p.bump`` under ``p.bump(2)``: a bound method, which is an
                # object like any other callable reference. What the CALL
                # evaluates to is the call's question, not the reference's
                return OBJECT
            self._node_diag(
                f"'{sig.name}' has no field '{node.attr}'", node,
                'unknown-field', f'{sig.name}.{node.attr}',
                fix=f"declare '{node.attr}' on '{sig.name}', or read a field it has")
            return UNKNOWN
        if base == OBJECT and not self._is_reference(node.value):
            self._node_diag(
                f"the class of '{_dotted(node.value) or ast.unparse(node.value)}' is not "
                f"known here, so its field '{node.attr}' has no type", node,
                'unknown-class', node.attr,
                fix='annotate the value with the type it holds')
        return UNKNOWN

    def _loop_field(self, node: ast.Attribute) -> str:
        """
        Type a read of the compiled loop counter object.

        ``__loop_1__ = pine_loop(0, 1)`` / ``while __loop_1__.step(10)`` /
        ``i = __loop_1__.value`` is what a Pine ``for`` with a re-read bound
        compiles to, and ``i`` has to be typed the way the ``for`` variable
        is: from the bounds. The object has exactly the ``value`` field and
        the ``step`` method; ``value`` is the counter type recorded for the
        name, joined with every bound ``step`` was given so far.

        :param node: The attribute node
        :return: The counter type for ``value``, OBJECT for ``step``
        """
        if node.attr == 'step':
            return OBJECT
        if node.attr != 'value':
            self._node_diag(f"the loop counter has no field '{node.attr}'", node, 'unknown-field',
                            f'PineLoop.{node.attr}', fix="read 'value', or call 'step'")
            return UNKNOWN
        counter = self._loop_counter(node.value)
        if counter is None or counter == UNKNOWN:
            self._node_diag(
                f"the bounds of '{ast.unparse(node.value)}' are not known here, so its counter "
                f'has no type', node, 'unknown-value', 'PineLoop.value',
                fix='give the loop bounds known numeric types')
            return UNKNOWN
        return counter

    def _loop_counter(self, receiver: ast.expr) -> str | None:
        """
        The recorded counter type of a loop object, by the name that holds it.

        :param receiver: The receiver expression
        :return: The counter type, or None when the name holds no loop object
        """
        if not isinstance(receiver, ast.Name):
            return None
        for frame in reversed(self._frames):
            found = frame.loop_counters.get(receiver.id)
            if found is not None:
                return found
        return None

    def _loop_bound(self, node: ast.Call, what: str, bounds: Sequence[ast.expr],
                    counter: str | None) -> str:
        """
        Join a loop counter with the bounds one call passes.

        A Pine ``for`` variable is an int when every bound is, a float when
        any is; a bound that is not a number is not a bound at all and is
        reported at the call. An unknown bound leaves the counter unknown,
        with its own cause reported where it was lost.

        :param node: The call passing the bounds
        :param what: The callee, for the message
        :param bounds: The bound expressions
        :param counter: The counter so far, or None when the object is built
        :return: The joined counter type
        """
        result = counter if counter is not None else INT
        for bound in bounds:
            ty = self._ty_of(bound)
            if ty in NUMERIC:
                result = join(result, ty)
            elif ty in _UNCHECKED:
                result = UNKNOWN
            else:
                self._node_diag(
                    f"'{what}' takes a number for a bound, {render_ty(ty)} passed", node,
                    'bad-call', what, fix='pass int or float bounds')
                result = UNKNOWN
        return result

    def _loop_step(self, node: ast.Call, func: ast.Attribute) -> str:
        """
        Type a ``loop.step(to)`` call: it advances the counter and says whether
        the loop goes on, so it is a bool, and the bound it passes widens the
        counter the way the ``for`` variable widens with every bound.

        :param node: The call node
        :param func: Its ``loop.step`` callee
        :return: BOOL
        """
        if func.attr != 'step':
            return UNKNOWN
        if len(node.args) != 1 or node.keywords:
            self._node_diag(f"'step' takes 1 argument(s), {_call_arity(node)} passed", node,
                            'bad-call', 'PineLoop.step', fix="pass the bound: 'step(to)'")
            return BOOL
        counter = self._loop_counter(func.value)
        if counter is not None and isinstance(func.value, ast.Name):
            joined = self._loop_bound(node, 'step', node.args, counter)
            for frame in reversed(self._frames):
                if func.value.id in frame.loop_counters:
                    frame.loop_counters[func.value.id] = joined
                    break
        return BOOL

    def _is_reference(self, node: ast.expr) -> bool:
        """
        Whether an attribute's base is a MODULE or namespace rather than a value.

        ``math.floor`` and ``lib.ta`` read a member of something imported, not
        a field of a Pine object, so a missing class there is no complaint to
        make. A bare name nothing binds is exactly that case: every Pine value
        the walk can see is bound somewhere, or is not a bare name at all. An
        IMPORTED name is the other case -- a Pine library and a namespace
        merged over a builtin one (``shadowed_namespace``) both bind a name
        whose members are functions, not fields.

        :param node: The attribute's base expression
        :return: True when the base names a module or a namespace
        """
        if not isinstance(node, ast.Name):
            return False
        return node.id in self._imports or node.id in self._shadowed \
            or self._lookup(node.id) is None

    def _node_diag(self, message: str, node: ast.AST, reason: str, detail: str,
                   fix: str = '') -> None:
        """
        Report a node once, however many walks reach it.

        :param message: What happened
        :param node: The attribute node
        :param reason: The provenance reason
        :param detail: The provenance detail
        :param fix: The concrete remedy
        """
        nid = node_id(node)
        if nid is not None:
            if nid in self._shape_diags:
                return
            self._shape_diags.add(nid)
        self._diag(message, node, self._unknown(reason, node, detail), fix=fix)

    def _e_BinOp(self, node: ast.BinOp) -> str:
        return binop_type(node.op, self._expr(node.left), self._expr(node.right))

    def _e_UnaryOp(self, node: ast.UnaryOp) -> str:
        return unaryop_type(node.op, self._expr(node.operand))

    def _e_BoolOp(self, node: ast.BoolOp) -> str:
        for value in node.values:
            self._expr(value)
        return BOOL

    def _e_Compare(self, node: ast.Compare) -> str:
        left = self._expr(node.left)
        right = left
        for comparator in node.comparators:
            right = self._expr(comparator)
        return compare_type(left, right)

    def _e_IfExp(self, node: ast.IfExp) -> str:
        self._expr(node.test)
        # MEASURED: ``d > 1 ? d : R`` is int, ``d > 1 ? d : 1.0`` is float --
        # the arms join, they do not widen unconditionally
        return self._joined(self._expr(node.body), self._expr(node.orelse),
                            node, 'the ternary')

    def _e_NamedExpr(self, node: ast.NamedExpr) -> str:
        ty = self._expr(node.value)
        self._store(node.target, ty, node)
        return ty

    def _e_Subscript(self, node: ast.Subscript) -> str:
        base = self._expr(node.value)
        self._expr(node.slice)
        if is_tuple(base):
            # A tuple has no history: the index picks a POSITION, and only a
            # constant one is known before the program runs
            elements = elements_of(base)
            index = node.slice.value if isinstance(node.slice, ast.Constant) else None
            if isinstance(index, int) and not isinstance(index, bool) \
                    and 0 <= index < len(elements):
                return elements[index]
            return UNKNOWN
        if is_array(base) or is_matrix(base) or is_map(base):
            # An element read is ``array.get``: a Python index on the
            # container is not the Pine form
            self._node_diag(
                f'indexing {render_ty(base)} is not Pine', node, 'not-pine', 'Subscript',
                fix='read elements with array.get / matrix.get / map.get')
            return UNKNOWN
        if base in SCALARS and not self._is_series(node.value):
            # The history operator reads a SERIES: a plain scalar name has no
            # bars behind it, and a call's result is a value, not a series
            self._node_diag(
                f"'{ast.unparse(node.value)[:40]}' is a plain {render_ty(base)}, not a "
                f"series", node, 'not-series', render_ty(base),
                fix=f'declare it Series[{render_ty(base)}] to read its history')
            return UNKNOWN
        # MEASURED: ``d[1]`` on an int-typed ``d`` is int -- the history index
        # is type-preserving, it reads the same series one bar back
        return base

    def _is_series(self, node: ast.expr) -> bool:
        """
        Whether an expression reads a series, whose history ``[n]`` exists.

        A name declared ``Series[...]`` is one, and so is a lib value
        (``close``); a name assigned without the declaration, a field and a
        call's result are plain values at run time.

        :param node: The subscripted expression
        :return: True for a series
        """
        if isinstance(node, ast.Name):
            found = self._lookup(node.id)
            if found is not None:
                return found.series or node.id in self._frames[-1].declared_series
            return node.id in lib_types()
        return self._lib_name(node) is not None

    def _e_Tuple(self, node: ast.Tuple) -> str:
        """
        A tuple literal is a tuple of what its elements are.

        Pine's ``[a, b]`` is a value of type ``[<a>, <b>]``, and PyneComp emits
        it as this literal -- as the value of a tuple-returning function, as a
        block result, and as the ``na`` filler a security read defaults to.
        Reading it as an anonymous object lost every element on the way to the
        unpack that follows.

        :param node: The tuple (or list) literal
        :return: The tuple shape
        """
        return tuple_of([self._expr(element) for element in node.elts])

    # PyneComp emits a Pine tuple as either of the two Python sequence
    # literals; the empty one -- the array a security read defaults to -- has
    # no elements and stays an object
    _e_List = _e_Tuple

    def _e_Set(self, node: ast.Set) -> str:
        """A set is not a Pine type; its elements are still typed."""
        for element in node.elts:
            self._expr(element)
        return OBJECT

    def _e_Dict(self, node: ast.Dict) -> str:
        for key in node.keys:
            if key is not None:
                self._expr(key)
        for value in node.values:
            self._expr(value)
        return OBJECT

    def _e_ListComp(self, node) -> str:
        """
        Walk a comprehension so nothing under it is left untyped.

        Comprehensions are outside Pine and the ``edge`` gate rejects them, but
        the lib's own code uses them, and an unvisited subtree would leave
        typed literals sitting under untyped nodes. The loop variables are
        bound in the ENCLOSING scope here rather than a scope of their own --
        an approximation Python does not make, and one that only matters for a
        name the comprehension shadows.
        """
        for generator in node.generators:
            iter_ty = self._expr(generator.iter)
            self._store(generator.target, self._element_type(generator.iter, iter_ty), node)
            for condition in generator.ifs:
                self._expr(condition)
        if isinstance(node, ast.DictComp):
            self._expr(node.key)
            self._expr(node.value)
        else:
            self._expr(node.elt)
        return OBJECT

    _e_SetComp = _e_ListComp
    _e_GeneratorExp = _e_ListComp
    _e_DictComp = _e_ListComp

    def _e_Call(self, node: ast.Call) -> str:
        for arg in node.args:
            self._expr(arg)
        for keyword in node.keywords:
            self._expr(keyword.value)

        callee = self._lib_name(node.func)
        if callee is None:
            self._expr(node.func)
            return self._user_call(node)
        return self._lib_call(node, callee)

    def _lib_call(self, node: ast.Call, callee: str) -> str:
        """
        Type a call to a lib name and record its call site.

        :param node: The call node, whose arguments are already typed
        :param callee: The registry key the callee resolves to
        :return: The type the call evaluates to
        """
        argc = _call_arity(node)
        entry = lib_types().get(callee)
        if entry is None and LIB_TYPE_OVERRIDES.get(callee) is None:
            if callee in lib_namespaces():
                self._node_diag(f"'{callee}' is a lib namespace, not a function", node,
                                'unknown-lib', callee, fix=f"call a name in '{callee}'")
            else:
                self._node_diag(f"'{callee}' is not a lib name", node, 'unknown-lib', callee,
                                fix='call a lib name that exists')
        if isinstance(entry, dict) and argc is not None \
                and not self._lib_call_fits(callee, entry, node):
            self.table.calls.append(CallSite(
                callee=callee, line=_line(node), col=_col(node), argc=argc, ty=UNKNOWN, pin=None))
            return UNKNOWN
        pin = self._pin(node, isinstance(entry, dict) and entry.get('kind') == 'overloads')
        ty = self._lib_call_type(callee, node.args, node.keywords, argc, pin)
        self.table.calls.append(CallSite(
            callee=callee, line=_line(node), col=_col(node), argc=argc, ty=ty, pin=pin))
        return ty

    def _lib_call_fits(self, callee: str, entry: dict[str, Any], node: ast.Call,
                       args: Sequence[ast.expr] | None = None) -> bool:
        """
        Whether a lib call has the shape and the argument types its callee takes.

        The registry records each function's positional parameters, their
        types, how many have defaults and whether it is open-ended; an
        overload group records that per implementation. A call that fits no
        shape, names a parameter that does not exist, or passes a value the
        parameter's type does not take is a Pine error, and is reported once,
        here, whatever becomes of its result.

        :param callee: The registry key
        :param entry: Its registry entry
        :param node: The call node, whose arguments are typed already
        :param args: The positional arguments, when they are not the node's
                     own (a method call binds the receiver first)
        :return: True when the call is well-formed
        """
        positional_args = list(node.args if args is None else args)
        kind = entry.get('kind')
        if kind == 'overloads':
            impls = entry['impls']
        elif kind == 'function':
            impls = [entry]
        elif kind == 'value' and entry.get('ty') in SCALARS and not entry.get('callable'):
            # ``close(1)``: a scalar is nothing a call can reach (a module
            # property may be called, and reads the same)
            self._node_diag(f"'{callee}' is a lib value, not a function", node, 'unknown-lib',
                            callee, fix=f"read '{callee}' without calling it")
            return False
        else:
            return True
        argc = len(positional_args) + len(node.keywords)
        if not any(_arity_fits(impl, argc) for impl in impls):
            counts = sorted({len(impl['params']) - impl['defaults'] for impl in impls})
            self._bad_call(node, callee, f"'{callee}' does not take {argc} argument(s) "
                                         f"(at least {counts[0]} needed)")
            return False
        for keyword in node.keywords:
            if not any(keyword.arg in impl['names'] or impl.get('kwarg') for impl in impls):
                self._bad_call(node, callee, f"'{callee}' has no parameter '{keyword.arg}'")
                return False
        if kind != 'function':
            return True
        params, names = entry['params'], entry['names']
        given: list[tuple[str, str, ast.expr]] = []
        for index, arg in enumerate(positional_args[:len(params)]):
            given.append((names[index] if index < len(names) else str(index),
                          params[index], arg))
        for keyword in node.keywords:
            if keyword.arg in names:
                given.append((keyword.arg, params[names.index(keyword.arg)], keyword.value))
        for pname, declared, arg in given:
            if declared not in SCALARS:
                # The registry spells a container parameter the way the
                # Python signature does (``Array[float]``), while Pine's law
                # is generic in the element: only a scalar parameter is a
                # type the call has to meet
                continue
            passed = get_ty(arg)
            if passed in _UNCHECKED or _fits(declared, passed):
                continue
            self._bad_call(node, callee, f"'{callee}' takes {render_ty(declared)} for "
                                         f"'{pname}', {render_ty(passed)} passed")
            return False
        return True

    def _pin(self, node: ast.Call, is_group: bool) -> str | None:
        """
        The overload pin this call site justifies in the running context.

        Only a proven overload group is worth pinning: everything else has a
        single implementation and nothing to choose between. A keyword or an
        unpacked argument is declined rather than guessed -- the pin is
        positional, and the runtime declines the same shapes anyway.

        :param node: The call node
        :param is_group: Whether the callee is a proven overload group
        :return: The pin this context justifies, or None
        """
        pin = None
        if is_group and not node.keywords and not any(
                isinstance(a, ast.Starred) for a in node.args):
            pin = pin_for([self._ty_of(a) for a in node.args])
        return self._record_pin(node, pin)

    def _record_pin(self, node: ast.Call, pin: str | None) -> str | None:
        """
        Merge this context's verdict into the node's pin and stamp both forms.

        ``set_pin`` always stamps, the None included, and that has to keep
        working through two different kinds of repeat visit:

        * The LOOP fixpoint re-walks a body in the SAME context. The second
          walk is the verdict -- ``total`` starts int and widens to float --
          so it OVERWRITES this context's entry, and a pin the first walk
          justified disappears from the node with it.
        * A second CONTEXT is a different instantiation, not a correction. Its
          verdict is recorded ALONGSIDE the first one. While they agree the
          node keeps the pin, because every instance would resolve the same
          way; the moment they disagree the single pin is erased and the
          per-context map takes its place, for a later pass to hand the right
          character to each instance.

        Both are the same rule: the node carries what holds for every context,
        and nothing else.

        :param node: The call node
        :param pin: What the running context justified
        :return: ``pin``
        """
        nid = node_id(node)
        if nid is None:
            set_pin(node, pin)
            set_pins(node, None)
            return pin
        self.table.call_pos[nid] = (_line(node), _col(node))
        seen = self._pins.setdefault(nid, {})
        seen[self._context] = pin
        sink = self._pin_sink[-1]
        if sink is not None:
            sink[nid] = pin
        agreed = len(set(seen.values())) == 1
        set_pin(node, pin if agreed else None)
        set_pins(node, None if agreed else dict(seen))
        if not agreed and nid not in self._pin_diags:
            self._pin_diags.add(nid)
            self._diag(
                f"overload pin differs per call context: {sorted(set(seen.values()), key=str)}",
                node, self._unknown('context-dependent-pin', node, _dotted(node.func) or ''),
                fix='annotate the parameters so every call site agrees')
        return pin

    def _lib_call_type(self, callee: str, args: Sequence[ast.expr],
                       keywords: Sequence[ast.keyword], argc: int | None,
                       pin: str | None = None) -> str:
        """
        Result type of a call to a lib name.

        The measured override wins over the annotation: the lib annotates
        ``math.round`` as a float because that is what Python returns, while
        TradingView types the one-argument form as an int.

        The arguments are passed in rather than read off a node, because one
        call shape does not spell them where its own node has them: a
        ``method_call('get', arr, i)`` is an ``array.get(arr, i)`` whose
        receiver stands in the SECOND slot of a different call.

        :param callee: The registry key the call resolves to
        :param args: The positional arguments, in order
        :param keywords: The keyword arguments
        :param argc: How many arguments it passes, or None when an unpacking
                     hides the count
        :param pin: The overload pin this call site justified, when it has one
        :return: The type the call evaluates to
        """
        entry = lib_types().get(callee)
        override = LIB_TYPE_OVERRIDES.get(callee)
        if override is not None:
            names = entry.get('names') if isinstance(entry, dict) else None
            if names is None:
                names = OVERRIDE_PARAM_NAMES.get(callee)
            resolved = self._apply_override(override, args, keywords, argc, names)
            if resolved is not None:
                return resolved

        if entry is None:
            return UNKNOWN
        if entry['kind'] == 'value':
            # A module property read that the property pass turned into a call
            return entry['ty']
        if entry['kind'] == 'function':
            return entry['ret']
        # An overload group: a pin names one implementation outright, and
        # otherwise only the implementations this arity can reach speak, with
        # the call's type being what they agree on
        if argc is None:
            return UNKNOWN
        if pin is not None:
            picked = overload_pick([_lib_impl_sig(impl) for impl in entry['impls']], pin)
            if picked is not None:
                return picked
        return overload_result([impl['ret'] for impl in entry['impls']
                                if _arity_fits(impl, argc)])

    def _apply_override(self, override: Any, args: Sequence[ast.expr],
                        keywords: Sequence[ast.keyword], argc: int | None,
                        param_names: list[str] | None = None) -> str | None:
        """
        Resolve one entry of the measured override table.

        ``param_names`` is the callee's declared parameter order, which is what
        turns a keyword spelling back into a position; without it only the
        positional arguments can be addressed.

        A LIST is a chain: each form is tried in turn and the first one with an
        answer wins. That is what lets one name carry two faces -- ``na(int)``
        builds an na OF a type while ``na(close)`` tests one -- without the
        arity split being able to tell them apart.

        The SHAPE forms decline instead of answering UNKNOWN. ``array.avg`` of
        an array whose element type is not known here is exactly the case the
        lib's own annotation still types, and claiming an unknown would be a
        regression dressed up as honesty.
        """
        if isinstance(override, list):
            for candidate in override:
                resolved = self._apply_override(candidate, args, keywords, argc, param_names)
                if resolved is not None:
                    return resolved
            return None
        if isinstance(override, dict):
            if argc is None:
                return UNKNOWN
            picked = override.get(argc)
            return None if picked is None else self._apply_override(
                picked, args, keywords, argc, param_names)
        if not isinstance(override, str):
            return None
        if override == 'na_arg':
            # The typed-na constructors: the argument NAMES the type, it is not
            # a value of it. A call that passes a value has nothing to say here
            argument = _bound_arg(args, keywords, 0, param_names)
            if argument is None:
                return None
            built = self._names_class(argument)
            if built is not None:
                return object_ty(built)
            if not self._names_type(argument):
                return None
            return annotation_type(argument, self._classes)
        if override == 'join_args':
            # Every argument widens the result: MEASURED, ``nz(int, float)`` is
            # a float on TradingView while ``nz(int, int)`` is an int
            passed = self._passed(args, keywords, argc)
            if passed is None:
                return UNKNOWN
            if not passed:
                return None
            result = passed[0]
            for ty in passed[1:]:
                result = join(result, ty)
            return result
        if override == 'all_int':
            # Every argument counts, however it was spelled: ``math.max`` is
            # int-typed exactly when all of them are -- and an unpacking hides
            # some of them, so there is nothing to decide on
            passed = self._passed(args, keywords, argc)
            if passed is None:
                return UNKNOWN
            if not passed or any(t not in NUMERIC for t in passed):
                return UNKNOWN if passed else None
            return INT if all(t == INT for t in passed) else FLOAT
        if override == 'array_of_join_args':
            # ``array.from(x, y, ...)``: MEASURED, ``array.from(int, float)``
            # reads back as a float, so the elements JOIN
            passed = self._passed(args, keywords, argc)
            if not passed:
                return None
            result = passed[0]
            for ty in passed[1:]:
                result = join(result, ty)
            return array_of(result) if result != UNKNOWN else None
        if override.startswith('merge_') or override.startswith('put_'):
            return self._merge_override(override, args, keywords, param_names)
        if override == 'matrix_mult':
            # ``matrix.mult`` follows its SECOND operand: a matrix by a matrix
            # or a scalar is a matrix, a matrix by an array is an array. Which
            # one it is has to be KNOWN: an unknown or classless second operand
            # may be either at run time, and answering the matrix shape there
            # would hand the next read an element type of the wrong container
            base = self._arg_ty(args, keywords, 0, param_names)
            other = self._arg_ty(args, keywords, 1, param_names)
            if element_of(base) == UNKNOWN:
                return None
            if is_array(other):
                return array_of(element_of(base))
            if is_matrix(other) or other in SCALARS:
                return base
            return None
        return self._shape_override(override, args, keywords, param_names)

    def _merge_override(self, form: str, args: Sequence[ast.expr],
                        keywords: Sequence[ast.keyword],
                        param_names: list[str] | None) -> str | None:
        """
        Resolve a container operation that takes a SECOND container.

        ``array.concat(a, b)`` appends ``b``'s elements to ``a`` and hands
        ``a`` back, so ``a`` keeps its type only while ``b`` is an array whose
        elements fit ITS element type -- Pine takes an array of the same type
        there and nothing else. A second operand of another kind, of an
        element type that does not fit, or of a type this pass does not know
        may have put anything into ``a``: the call is reported, and ``a``'s
        binding is INVALIDATED, because the array is mutated in place and a
        later read through the old type would be a confident wrong answer
        about a value that is no longer there.

        :param form: ``merge_array``, ``merge_matrix`` or
                     ``merge_matrix_or_scalar``
        :param args: The positional arguments, in order
        :param keywords: The keyword arguments
        :param param_names: The callee's declared parameter order, when known
        :return: The receiver's type, UNKNOWN once it was invalidated, or None
                 when the receiver is not the container the form is for
        """
        putting = form.startswith('put_')
        put_kind = form[4:].partition(':')[0] if putting else ''
        if putting:
            kind = {'array': is_array, 'map': is_map,
                    'matrix': is_matrix, 'matrix_array': is_matrix}[put_kind]
        else:
            kind = is_array if form == 'merge_array' else is_matrix
        base = self._arg_ty(args, keywords, 0, param_names)
        if not kind(base):
            return None
        # ``array.set(a, i, v)`` puts its THIRD argument in: the form says which
        operand_at = int(form.partition(':')[2]) if putting else 1
        if putting and _bound_arg(args, keywords, operand_at, param_names) is None:
            # ``matrix.add_row(m)`` appends an empty row: nothing goes in
            return VOID
        other = self._arg_ty(args, keywords, operand_at, param_names)
        element = value_of(base) if put_kind == 'map' else element_of(base)
        if putting:
            # One element in: it fits when the container's element type takes
            # it (an int into ``array<float>``); a typeless na, a bare object
            # or a void value goes in unchecked, the way a declaration takes
            # them, while an UNKNOWN stays a report; a row or column goes in
            # as an array of fitting elements; a map's key has to fit its key
            # type as well
            unchecked = other != UNKNOWN and other in _UNCHECKED
            if put_kind == 'matrix_array':
                fits = unchecked or (is_array(other) and element_of(other) != UNKNOWN
                                     and _fits(element, element_of(other)))
            else:
                fits = unchecked or (other != UNKNOWN and _fits(element, other))
            if fits and put_kind == 'map':
                key = self._arg_ty(args, keywords, 1, param_names)
                fits = (key != UNKNOWN and key in _UNCHECKED) \
                    or (key != UNKNOWN and _fits(key_of(base), key))
                if not fits:
                    operand_at, other = 1, key
        elif kind(other):
            fits = element_of(other) != UNKNOWN \
                and shape_conflict(element, element_of(other)) is None \
                and join(element, element_of(other)) == element
        elif form == 'merge_matrix_or_scalar' and other in NUMERIC:
            fits = join(element, other) == element
        else:
            fits = False
        if fits:
            if putting:
                # ``map.put`` hands back what the key held before
                return value_of(base) if put_kind == 'map' else VOID
            return base
        receiver = _bound_arg(args, keywords, 0, param_names)
        operand = _bound_arg(args, keywords, operand_at, param_names)
        where = operand if operand is not None else receiver
        assert where is not None
        nid = node_id(where)
        already = nid is not None and nid in self._shape_diags
        if already:
            # The operand is reported already, by whatever failed in it; the
            # merge's own diagnostic would only repeat that it is unknown
            unknown = self._unknown('unknown-value', where, render_ty(base))
        else:
            if nid is not None:
                self._shape_diags.add(nid)
            if other == UNKNOWN or (kind(other) and element_of(other) == UNKNOWN):
                unknown = self._unknown('unknown-value', where, render_ty(base))
                self._diag(
                    f'{render_ty(base)} takes an operand whose type is not known '
                    f'here, so what it holds afterwards is not known either', where,
                    unknown, fix='give the operand a known type')
            else:
                unknown = Unknown(reason='shape-mismatch',
                                  line=getattr(where, 'lineno', 0),
                                  col=getattr(where, 'col_offset', 0),
                                  detail=f'{render_ty(base)} vs {render_ty(other)}')
                self._diag(
                    f'{render_ty(base)} cannot take {render_ty(other)}: the '
                    f'elements are different types', where, unknown,
                    fix=f'pass {render_ty(base)}')
        # Every failed merge takes the pins away, reported here or before
        if self._pins_suppressed is None:
            self._pins_suppressed = self._diag_at(where) or self.table.diags[-1]
        if isinstance(receiver, ast.Name):
            found = self._lookup(receiver.id)
            if found is not None and found.ty != UNKNOWN:
                found.ty = UNKNOWN
                found.unknown = unknown
        return UNKNOWN

    def _diag_at(self, node: ast.AST) -> Diag | None:
        """The diagnostic already standing at a node's position, if any."""
        line, col = _line(node), _col(node)
        return next((diag for diag in self.table.diags
                     if diag.line == line and diag.col == col), None)

    def _suppress_pins(self, tree: ast.Module) -> None:
        """
        Take every overload pin away from a module whose containers cannot be trusted.

        A merge that failed put something into a container that its type
        does not hold, and invalidating the name it was passed under is not
        enough: ``alias = ai`` names the same list, a field or a parameter
        does too, and a ``var`` container read EARLIER in source order holds
        the mutated contents on the next bar, because the script runs again
        from the top. None of those aliases is tracked, so any type derived
        from a container downstream may be a confident wrong answer, and a
        pin built on it would drive the wrong overload. Without a pin the
        runtime dispatches on the values it actually sees, which is what it
        does today for every unpinned site. The program is reported either
        way -- the diagnostic is what marks it wrong.

        :param tree: The walked module, whose stamps are cleared in place
        """
        if self._pins_suppressed is None:
            return
        self.table.pins_suppressed = self._pins_suppressed
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                set_pin(node, None)
                set_pins(node, None)
                set_vector(node, None)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                set_varying(node, None)
        for site in self.table.calls:
            site.pin = None

    def _shape_override(self, override: str, args: Sequence[ast.expr],
                        keywords: Sequence[ast.keyword],
                        param_names: list[str] | None) -> str | None:
        """
        Resolve the forms that read ONE argument, shape or not.

        :param override: The form, ending in the argument position it reads
        :param args: The positional arguments, in order
        :param keywords: The keyword arguments
        :param param_names: The callee's declared parameter order, when known
        :return: The type, or None when the form has no answer here
        """
        if override.startswith('arg') and override[3:].isdigit():
            argument = _bound_arg(args, keywords, int(override[3:]), param_names)
            return UNKNOWN if argument is None else self._ty_of(argument)
        for form, read in _SHAPE_FORMS.items():
            if override.startswith(form) and override[len(form):].isdigit():
                ty = self._arg_ty(args, keywords, int(override[len(form):]), param_names)
                answer = read(ty)
                return None if answer in (UNKNOWN, OBJECT) else answer
        # Not a form at all: the entry IS the type, a whole shape included
        return override

    def _arg_ty(self, args: Sequence[ast.expr], keywords: Sequence[ast.keyword],
                index: int, param_names: list[str] | None) -> str:
        """
        The type of the argument bound to one declared position.

        :param args: The positional arguments, in order
        :param keywords: The keyword arguments
        :param index: Declared parameter position, 0-based
        :param param_names: The callee's declared parameter order, when known
        :return: Its type, UNKNOWN when the position is not filled
        """
        argument = _bound_arg(args, keywords, index, param_names)
        return UNKNOWN if argument is None else self._ty_of(argument)

    def _passed(self, args: Sequence[ast.expr], keywords: Sequence[ast.keyword],
                argc: int | None) -> list[str] | None:
        """
        The type of every argument a call passes, however it spelled them.

        :param args: The positional arguments, in order
        :param keywords: The keyword arguments
        :param argc: The argument count, None when an unpacking hides it
        :return: The types, or None when the call cannot be read
        """
        if argc is None:
            return None
        passed = [self._ty_of(a) for a in args]
        passed += [self._ty_of(k.value) for k in keywords if k.arg is not None]
        return passed

    def _user_call(self, node: ast.Call) -> str:
        """
        Result type of a call to a function this module defines or imports.

        A definition of this module wins, with its own shadowing rules. Only
        when there is none does the import map speak -- the imported callee is
        then typed from its DECLARED signature, never from this call site.

        Two shapes are answered before either: the plumbing an earlier pass
        emitted, which is not a call a script wrote at all, and a CONSTRUCTOR,
        which names a class rather than a function and so resolves through
        neither map.
        """
        name = _dotted(node.func) or ''
        if name == '__sec_read__':
            return self._security_read(node)
        if name == 'method_call':
            dispatched = self._method_call(node)
            return dispatched if dispatched is not None else self._unknown_method(node)
        if id(node) in self._factory_fields:
            # ``field(default_factory=...)`` as a UDT field's default: the
            # dataclass machinery builds it, the annotation types the field
            return OBJECT
        resolved = self._resolve_func(name, node)
        if resolved is None:
            constructed = self._constructor_call(name, node)
            if constructed is not None:
                return constructed
            invoked = self._method_of(node)
            if invoked is not None:
                return invoked
            imported = self._imported_call(node, name)
            if imported is not None:
                return imported
            # A module function shadows the builtin of the same name, so the
            # builtins are only consulted once the module has no such name
            builtin = BUILTIN_CALL_TYPES.get(name)
            if builtin is not None and isinstance(node.func, ast.Name):
                fallback = self._apply_override(
                    builtin, node.args, node.keywords, _call_arity(node))
                if fallback is not None:
                    return fallback
                self._node_diag(f"the call to '{name}' has no known type", node,
                                'unknown-call', name, fix=f"call '{name}' the way Pine spells it")
                return UNKNOWN
            # Nothing answers for the callee: not a definition, not a class,
            # not a method, not an import, not a builtin. That is the case
            # whatever becomes of the value, so it is told here, not by the
            # reader of the result
            self._node_diag(
                f"'{name}' is not a function this module defines or imports, nor a lib name"
                if name else 'the callee is not a name', node, 'unknown-call', name,
                fix='define it, import it or call a lib name')
            return UNKNOWN

        key, frame, shadowed = resolved
        if shadowed:
            self._diag(
                f"'{name}' is assigned as well as defined, so what it calls is unknown",
                node, self._unknown('rebound-name', node, name),
                fix=f"call '{name}' under a name nothing assigns to")
            return UNKNOWN
        is_group = key in self._overload_groups
        if is_group:
            # A group is per-signature already; the pin is what selects among
            # its implementations, so there is no context to instantiate --
            # but a call no implementation takes is an error before any pin
            self._ensure_group(key, frame)
            if not self._group_call_fits(key, name, node):
                self.table.calls.append(CallSite(
                    callee=name, line=_line(node), col=_col(node),
                    argc=_call_arity(node), ty=UNKNOWN, pin=None))
                return UNKNOWN
        pin = self._pin(node, is_group)
        if is_group:
            ty = self._group_type(key, pin)
        else:
            ty = self._call_context(key, node, frame)
        self.table.calls.append(CallSite(
            callee=name, line=_line(node), col=_col(node),
            argc=_call_arity(node), ty=ty, pin=pin))
        return ty

    def _group_call_fits(self, key: str, name: str, node: ast.Call,
                         args: Sequence[ast.expr] | None = None) -> bool:
        """
        Whether some implementation of a user group takes a call's arguments.

        Each implementation is measured the way a plain definition is; the
        call is an error only when NONE fits. An open-ended implementation or
        an unpacked argument is a shape this analysis does not describe, and
        is let through to the runtime dispatch.

        :param key: Scope-qualified id of the group
        :param name: The callee as written
        :param node: The call node
        :param args: The positional arguments, when they are not the node's
                     own (a method call binds the receiver first)
        :return: True when the call may run
        """
        positional = node.args if args is None else args
        if any(isinstance(arg, ast.Starred) for arg in node.args) \
                or any(keyword.arg is None for keyword in node.keywords):
            return True
        impls = self._defs.get(key, ())
        if not impls:
            return True
        for impl in impls:
            if impl.args.vararg is not None or impl.args.kwarg is not None:
                return True
            if self._call_params(impl, node, positional, report=False) is not None:
                return True
        self._bad_call(node, name, f"no overload of '{name}' takes these arguments")
        return False

    def _constructor_call(self, name: str, node: ast.Call) -> str | None:
        """
        Type a call that builds an instance of a class.

        A UDT is constructed either directly (``SessionInfo(...)``, which is
        what a dataclass compiles to) or through the ``new`` classmethod Pine
        spells (``Settings.new(...)``, ``zigzag.Settings.new(...)``), and both
        evaluate to the object. Neither resolves through the function map: the
        callee is a CLASS, and a class is not one of the definitions the walk
        collected.

        :param name: The callee as written, dotted
        :param node: The call node
        :return: The object type it builds, or None when it builds none
        """
        parts = tuple(part for part in name.split('.') if part)
        if not parts:
            return None
        built = None
        if len(parts) > 1 and parts[-1] == 'new':
            built = self._class_named(parts[:-1], node)
        if built is None:
            built = self._class_named(parts, node)
        if built is None:
            return None
        if not self._constructor_fits(built, name, node):
            return UNKNOWN
        return object_ty(built)

    def _constructor_fits(self, cid: str, name: str, node: ast.Call) -> bool:
        """
        Whether a constructor call matches the class it builds.

        A UDT is a dataclass: its fields are the parameters, in declaration
        order, and the ones without a default have to be passed. What is
        passed has to fit what the field holds.

        :param cid: The class id
        :param name: The callee as written
        :param node: The call node
        :return: True when the call builds the object
        """
        if any(isinstance(arg, ast.Starred) for arg in node.args) \
                or any(keyword.arg is None for keyword in node.keywords):
            return True
        sig = self._class_sig(cid, node)
        if sig is None:
            return True
        fields = list(sig.fields)
        if len(node.args) > len(fields):
            self._bad_call(node, name, f"'{sig.name}' has {len(fields)} field(s), "
                                       f"{len(node.args)} argument(s) passed")
            return False
        bound = dict(zip(fields, node.args))
        for keyword in node.keywords:
            if keyword.arg not in sig.fields:
                self._bad_call(node, name, f"'{sig.name}' has no field '{keyword.arg}'")
                return False
            if keyword.arg in bound:
                self._bad_call(node, name, f"'{keyword.arg}' is passed to '{sig.name}' twice")
                return False
            bound[keyword.arg] = keyword.value
        for index, field_name in enumerate(fields):
            passed = bound.get(field_name)
            if passed is None:
                if index < sig.required:
                    self._bad_call(node, name, f"'{sig.name}' needs a value for '{field_name}'")
                    return False
                continue
            declared = sig.fields[field_name]
            given = self._ty_of(passed)
            if declared != UNKNOWN and given not in _UNCHECKED and not _fits(declared, given):
                self._bad_call(node, name, f"'{sig.name}.{field_name}' holds "
                                           f"{render_ty(declared)}, {render_ty(given)} passed")
                return False
        return True

    def _class_named(self, parts: tuple[str, ...], node: ast.AST) -> str | None:
        """
        The class a dotted path names.

        The module's own classes are known outright. An imported one is
        resolved through the same interface the imported CALLS go through, so
        ``zigzag.Settings`` is answered by the library's published classes --
        and answered with the library's own class id, which is what keeps two
        libraries' same-named types apart. A qualified spelling is answered by
        that interface or not at all: its leaf never stands for a class of
        this module. A value bound under the head's name shadows every class
        behind it.

        :param parts: The dotted path, split
        :param node: The node the question is asked for, for the cycle diagnostic
        :return: The class id, or None when the path names no class
        """
        if not parts or self._lookup(parts[0]) is not None:
            # A value of that name stands in front of the class: ``Pivot(1)``
            # under a parameter ``Pivot`` calls the value, ``bogus.Pivot.new``
            # reads a field of it
            return None
        leaf = parts[-1]
        if len(parts) == 1:
            own = self._classes.get(leaf)
            if own is not None:
                return own
            # ``from lib.m import Settings`` binds the class itself
            binding = self._imports.get(leaf)
            if binding is None or not binding.attrs or leaf in self._multi_imports \
                    or self._nearer_binding(leaf):
                return None
            interface = self._interface_of(binding.module, binding.attrs[:-1], node)
            if interface is None:
                return None
            bound = interface.classes.get(binding.attrs[-1])
            return None if bound is None else bound.id
        binding = self._imports.get(parts[0])
        if binding is None or parts[0] in self._multi_imports \
                or self._nearer_binding(parts[0]):
            return None
        interface = self._interface_of(binding.module, binding.attrs + parts[1:-1], node)
        if interface is None:
            return None
        found = interface.classes.get(leaf)
        return None if found is None else found.id

    def _method_call(self, node: ast.Call) -> str | None:
        """
        Type a ``method_call(name, obj, ...)`` the method transform emitted.

        The receiver's class is what selects the implementation at run time,
        and now the receiver's TYPE carries it: an ``array<int>`` sends
        ``'get'`` to ``array.get`` and answers int, a ``Box`` sends
        ``'get_top'`` to ``box.get_top``, and a UDT sends the name to the
        method its own class declares. The receiver decides, exactly as the
        runtime's own dispatch does -- and in its order, which puts the BUILTIN
        namespace the receiver's shape reaches ahead of everything a script
        declared (``core/pine_method.method_call`` calls
        ``_get_builtin_method`` first, in both of its branches).

        Two shapes reach here. The compiled form spells a USER method by the
        function itself (``method_call(bump, x, 1)``) and a builtin one by its
        name (``method_call('get', a, 0)``); a name is only a user method's
        when the receiver's class declares it.

        Where the receiver's class is not known the old rule still stands: the
        answer is claimed only where every builtin namespace the name could
        reach agrees on it, and only while no user function of that name is in
        scope to shadow them.

        :param node: The ``method_call`` node
        :return: The type the call evaluates to, or None
        """
        if len(node.args) < 2:
            return None
        selector, receiver = node.args[0], node.args[1]
        args = node.args[1:]
        if isinstance(selector, (ast.Name, ast.Attribute)):
            return self._named_method(selector, node, args)
        if not isinstance(selector, ast.Constant) or not isinstance(selector.value, str):
            return None
        name = selector.value
        receiver_ty = self._ty_of(receiver)
        namespace = namespace_of(receiver_ty)
        if namespace is not None and f'{namespace}.{name}' in lib_types():
            # The builtin the receiver's shape reaches, which the runtime
            # resolves BEFORE anything a script declared: a user method
            # annotated with the BUILTIN class itself does not displace it
            return self._builtin_method(f'{namespace}.{name}', node, args)
        cid = class_of(receiver_ty)
        if cid is not None:
            found = self._method_result(cid, name, node, args)
            if found is not None:
                return found
        if self._resolve_func(name, node) is not None:
            return None
        returns: list[str] = []
        for candidate in _METHOD_NAMESPACES:
            entry = lib_types().get(f'{candidate}.{name}')
            if entry is None:
                continue
            if entry['kind'] != 'function':
                return None
            returns.append(entry['ret'])
        if not returns:
            return None
        agreed = overload_result(returns)
        return None if agreed == UNKNOWN else agreed

    def _builtin_method(self, callee: str, node: ast.Call, args: Sequence[ast.expr]) -> str:
        """
        Type a builtin reached through the method spelling, held to its shape.

        ``method_call('get', a, 0)`` IS ``array.get(a, 0)``: the receiver is
        the first argument, and the call has to meet the registry's shape the
        way the direct spelling does.

        :param callee: The registry key the receiver's namespace reaches
        :param node: The ``method_call`` node
        :param args: The receiver and the arguments after it
        :return: The type the call evaluates to
        """
        entry = lib_types().get(callee)
        if isinstance(entry, dict) and not self._lib_call_fits(callee, entry, node, args):
            return UNKNOWN
        return self._lib_call_type(callee, args, node.keywords, _arity(args, node.keywords))

    def _named_method(self, selector: ast.expr, node: ast.Call,
                      args: Sequence[ast.expr]) -> str | None:
        """
        Type a ``method_call(<the method itself>, obj, ...)``.

        Naming the function does NOT make it the one that runs: the runtime
        tries ``_get_builtin_method(method.__name__, var)`` first and only
        calls the function it was handed when the receiver's shape reaches no
        builtin of that name (see ``core/pine_method.method_call``). So a user
        method called ``delete`` on a ``Box`` receiver is ``box.delete``.

        Past that check the function IS the answer -- ``_bound_method(method)``
        calls the one it was handed, whatever the receiver's class declares --
        so what is typed here is the definition the name resolves to, in this
        module or in the one that exports it, and never a search over the
        receiver's other methods.

        The name a builtin is looked up by is the FUNCTION's own, which is the
        spelling's leaf wherever a script can write it down.

        :param selector: The expression naming the method function
        :param node: The ``method_call`` node
        :param args: The receiver and the arguments after it
        :return: The type the call evaluates to, or None
        """
        spelled = _dotted(selector)
        if spelled is None:
            return None
        imported = self._imported_selector(spelled)
        name = spelled.rpartition('.')[2] if imported is None else imported[1][-1]
        receiver_ty = self._ty_of(args[0]) if args else UNKNOWN
        namespace = namespace_of(receiver_ty)
        if namespace is not None and f'{namespace}.{name}' in lib_types():
            return self._builtin_method(f'{namespace}.{name}', node, args)
        resolved = self._resolve_func(spelled, node)
        if resolved is None:
            if imported is None:
                return None
            binding, path = imported
            return self._imported_method(binding, path, node, args)
        key, frame, shadowed = resolved
        if shadowed:
            self._diag(
                f"'{spelled}' is assigned as well as defined, so what it calls is unknown",
                node, self._unknown('rebound-name', node, spelled),
                fix=f"call '{spelled}' under a name nothing assigns to")
            return UNKNOWN
        if key in self._overload_groups:
            # A group is per-signature already: the receiver and the arguments
            # select the implementation the way the direct spelling's pin
            # does. Nothing is STAMPED, though -- the node's callee is the
            # plumbing, whose runtime dispatch selects on the values itself
            self._ensure_group(key, frame)
            if not self._group_call_fits(key, name, node, args):
                return UNKNOWN
            pin = None if node.keywords else pin_for([self._ty_of(a) for a in args])
            return self._group_type(key, pin)
        return self._call_context(key, node, frame, args, record=False)

    def _unknown_method(self, node: ast.Call) -> str:
        """
        Report a ``method_call`` that reaches no method, naming the METHOD.

        The plumbing itself is always there; what is missing is a method of
        the receiver, and that is what the message has to say. A receiver of
        unknown type says nothing new -- its own UNKNOWN carries the
        provenance -- while one whose class was lost upstream is told where
        the fix has to go, like a field read on it.

        :param node: The ``method_call`` node
        :return: UNKNOWN
        """
        if len(node.args) < 2:
            self._node_diag("'method_call' takes a method and a receiver", node, 'bad-call',
                            'method_call', fix="spell it 'method_call(method, receiver, ...)'")
            return UNKNOWN
        selector, receiver = node.args[0], node.args[1]
        if isinstance(selector, ast.Constant) and isinstance(selector.value, str):
            spelled = selector.value
        else:
            spelled = _dotted(selector) or ast.unparse(selector)
        receiver_ty = self._ty_of(receiver)
        if receiver_ty == UNKNOWN:
            return UNKNOWN
        if receiver_ty == OBJECT:
            self._node_diag(
                f"the class of '{_dotted(receiver) or ast.unparse(receiver)}' is not known "
                f"here, so its method '{spelled}' has no type", node, 'unknown-class', spelled,
                fix='annotate the receiver with the type it holds')
            return UNKNOWN
        self._node_diag(
            f"'{spelled}' is not a method of {render_ty(receiver_ty)} here", node,
            'unknown-method', spelled,
            fix=f"declare '{spelled}' on the receiver's class, or call a method its shape has")
        return UNKNOWN

    def _imported_selector(self, spelled: str) -> tuple[_Import, tuple[str, ...]] | None:
        """
        The import a named selector comes from, and the path to it inside it.

        An alias is another NAME for a function, not another function: the
        runtime looks the builtin up by ``method.__name__``, which is the name
        the declaring module gave it, so ``from ext import delete as erase``
        still runs ``box.delete`` on a box. The last segment of the path is
        that declared name.

        A name the module also assigns to, or one two imports bind, holds
        something else by the time the call runs -- the same reason a plain
        call through it is unknowable.

        :param spelled: The selector as the call spells it, dotted
        :return: (the import binding, the path inside it), or None when the
                 selector is not a usable imported name
        """
        head, _, rest = spelled.partition('.')
        binding = self._imports.get(head)
        if binding is None or head in self._multi_imports \
                or self._import_rebound(head) or self._nearer_binding(head):
            return None
        path = binding.attrs + tuple(part for part in rest.split('.') if part)
        return None if not path else (binding, path)

    def _imported_method(self, binding: _Import, path: tuple[str, ...],
                         node: ast.Call, args: Sequence[ast.expr]) -> str | None:
        """
        Type a named method the call site reached through an import.

        The function handed to ``method_call`` is the one that runs, and one
        an import binds is no different: it is typed from what its module
        publishes, exactly as a direct call to it would be.

        :param binding: The import the selector comes from
        :param path: The path to the function inside that import
        :param node: The ``method_call`` node, for the dependency it creates
        :param args: The receiver and the arguments after it
        :return: The type the call evaluates to, or None
        """
        interface = self._interface_of(binding.module, path[:-1], node)
        if interface is None:
            return None
        sig = interface.exports.get(path[-1])
        if sig is None:
            return None
        return self._published_result(sig, path[-1], node, args)

    def _method_of(self, node: ast.Call) -> str | None:
        """
        Type an ``obj.method(...)`` call on a receiver whose class is known.

        :param node: The call node
        :return: The type the call evaluates to, or None when this is not one
        """
        func = node.func
        if not isinstance(func, ast.Attribute):
            return None
        receiver = self._ty_of(func.value)
        if receiver == PINE_LOOP:
            return self._loop_step(node, func)
        cid = class_of(receiver)
        if cid is None:
            return None
        return self._method_result(cid, func.attr, node, [func.value] + list(node.args))

    def _class_sig(self, cid: str, node: ast.AST) -> ClassSig | None:
        """
        What a class id declares, loading the module that declares it if needed.

        An interface installs the classes IT declares, and a value's class does
        not have to be one of them: a wrapper exporting ``get() -> base.Pivot``
        hands out a class of a module the caller never names, and reading a
        field of it needs that module's own interface. The class id carries
        where to find it -- its module half is the declaring module's source
        path -- and the load goes through the same machinery an import does, so
        the dependency is recorded and the cycle guard applies.

        :param cid: The class id
        :param node: The node the class is wanted for
        :return: The class signature, or None when it cannot be had
        """
        sig = self._class_sigs.get(cid)
        if sig is not None:
            return sig
        source = cid.rpartition(CLASS_SEP)[0]
        if not source or source == LIB_MODULE:
            # The lib's own classes are the one family whose module key is not
            # a path; they are loaded from the registry up front
            return None
        interface = self._interface_at(source, source, node)
        if interface is None:
            return None
        for published in interface.classes.values():
            self._class_sigs.setdefault(published.id, published)
        return self._class_sigs.get(cid)

    def _method_result(self, cid: str, name: str, node: ast.Call,
                       args: Sequence[ast.expr]) -> str | None:
        """
        Type one call to a method of a known class.

        A method this module declares is analysed like any other call, with
        the RECEIVER bound to its first parameter -- which is what makes
        ``this.price`` inside it read the class's own field type. One a
        dependency declares is typed from what that module publishes, the same
        way an imported function is.

        Three modules may hold a method of this name, and the order between
        them is the runtime's (``core/pine_method.method_call``): the module
        that DECLARES the receiver's class first, then the calling module
        itself, then every library the caller imports -- a library may declare
        a method on another library's UDT, so the search does not stop at the
        class's own module.

        :param cid: The receiver's class id
        :param name: The method name
        :param node: The call node, for its keywords and its position
        :param args: The receiver followed by the arguments after it
        :return: The type the call evaluates to, or None when there is none
        """
        sig = self._class_sig(cid, node)
        published = None if sig is None else sig.methods.get(name)
        if published is not None and cid not in self.table.class_sigs:
            # Declared by the class's own module, which the runtime reaches
            # before it looks at anything of ours
            return self._published_result(published, name, node, args)
        if name in self._class_methods.get(cid, ()):
            resolved = self._resolve_func(name, node)
            if resolved is not None:
                key, frame, shadowed = resolved
                if not shadowed and key not in self._overload_groups:
                    return self._call_context(key, node, frame, args, record=False)
        if published is not None:
            return self._published_result(published, name, node, args)
        extension = self._imported_extension(cid, name, node)
        if extension is None:
            return None
        return self._published_result(extension, name, node, args)

    def _published_result(self, sig: ExportSig, name: str, node: ast.Call,
                          args: Sequence[ast.expr]) -> str:
        """
        Type a call to a method another module publishes.

        The receiver is the first positional argument of the published
        signature, and the call is held to that signature the way any call
        into another module is; the declared return is the answer only
        where the parameters are annotated.

        :param sig: The published signature
        :param name: The method name
        :param node: The call node, for its keywords and position
        :param args: The receiver followed by the arguments after it
        :return: The type the call evaluates to
        """
        if not self._export_call_fits(node, name, sig, args):
            return UNKNOWN
        return sig.ret if sig.annotated else UNKNOWN

    def _imported_extension(self, cid: str, name: str, node: ast.AST) -> ExportSig | None:
        """
        A method one of the imported modules declares on another's class.

        The last place the runtime looks: after the receiver's own module and
        the caller's globals it scans the library MODULES it finds in those
        globals, because one library extending another's UDT is a shape Pine
        allows.

        Which modules those are is what the import spelling decides, and it is
        not every module named in one: ``import ext`` and ``from pkg import
        mod`` put a module object there, while ``from ext import helper`` puts
        one name there and leaves the rest of ``ext`` out of reach. A binding
        of that second kind answers only for the selector it is bound as.

        And it is the globals as they STAND that are searched: a name the
        module assigns to holds that value instead of the import. A local of
        the calling scope is another matter -- the search is over
        ``f_globals``, which a local never reaches into.

        :param cid: The receiver's class id
        :param name: The method name
        :param node: The node the question is asked for, for the cycle diagnostic
        :return: The published signature, or None when no import declares one
        """
        for bound, binding in self._import_sources():
            if bound in self._multi_imports or self._import_rebound(bound):
                # The globals hold whatever was stored last, and what a second
                # import or an assignment stored is not this library -- an
                # alias assigned over is not even a module any more, so the
                # runtime's scan passes it by
                continue
            interface = self._interface_of(binding.module, binding.attrs, node)
            if interface is not None:
                # ``import m``, ``import m as x`` and ``from pkg import mod``
                # all put a MODULE in the caller's globals, which is what the
                # runtime scans for a method it has not found anywhere else
                found = interface.extensions.get(cid, {}).get(name)
                if found is not None:
                    return found
                continue
            if not binding.attrs or bound != name:
                # ``from m import helper`` binds a NAME, not the module: the
                # rest of what ``m`` declares never reaches these globals, so
                # such a binding answers only for the selector it is bound as
                continue
            interface = self._interface_of(binding.module, binding.attrs[:-1], node)
            if interface is None:
                continue
            # The alias is what the selector matched; the extension is keyed by
            # the name it was declared under
            found = interface.extensions.get(cid, {}).get(binding.attrs[-1])
            if found is not None:
                return found
        return None

    def _import_sources(self) -> list[tuple[str, _Import]]:
        """
        Every import of this module, with the name it binds, once each.

        The NAME matters as much as the module: what an import puts in the
        caller's globals is either the module object or a single name out of
        it, and the runtime's method search treats the two differently.

        A namespace merged over a builtin one (``shadowed_namespace``) is an
        import like any other -- what it wraps is the module whose methods
        matter here.

        :return: (the name it binds, what it names), in the order recorded
        """
        out: list[tuple[str, _Import]] = []
        seen: set[tuple[str, str, tuple[str, ...]]] = set()
        for name, binding in list(self._imports.items()) + \
                [(name, shadow.source) for name, shadow in self._shadowed.items()]:
            key = (name, binding.module, binding.attrs)
            if key in seen:
                continue
            seen.add(key)
            out.append((name, binding))
        return out

    def _security_read(self, node: ast.Call) -> str:
        """
        Type a ``__sec_read__('id', default)`` the security transform emitted.

        What the read yields is what the matching ``__sec_write__('id', expr)``
        put there, so the write's expression IS the type -- the two halves are
        one expression the pass split in two, and reading them apart is what
        left every security-fed variable untyped. The DEFAULT joins in as a
        second possible value, except where it is the interned typeless ``na``:
        that one says "no value on this bar" and has no type to contribute.

        The write's node is read, not a type recorded when it was walked, so
        the answer improves on its own if a re-walk widens the expression.
        What it does need is for the write to have been walked already, which
        the emitted form guarantees: the transform puts the guarded write
        above the read of the same id, in the same body.

        :param node: The ``__sec_read__`` call node
        :return: The type the read evaluates to
        """
        sec_id = node.args[0] if node.args else None
        parts: list[str] = []
        if isinstance(sec_id, ast.Constant) and isinstance(sec_id.value, str):
            parts += [self._ty_of(written)
                      for written in self._sec_writes.get(sec_id.value, ())]
        if len(node.args) > 1 and _dotted(node.args[1]) != 'lib._na_none':
            parts.append(self._ty_of(node.args[1]))
        if not parts:
            return UNKNOWN
        result = parts[0]
        for ty in parts[1:]:
            result = join(result, ty)
        return result

    def _ensure_group(self, key: str, frame: int) -> None:
        """Walk any implementation of a group the call got ahead of."""
        parents = self._frames[:frame + 1]
        for impl in self._defs.get(key, ()):
            nid = node_id(impl)
            if nid is None or nid not in self._walked:
                self._analyse(key, impl, self._declared_params(impl), parents)

    def _group_type(self, key: str, pin: str | None) -> str:
        """
        Type of a call to a user overload group.

        The group's own type is what its implementations AGREE on, which is
        nothing where they return different types -- but a pinned call site
        has already settled which implementation runs, and that one's return
        IS the call's type. Without this a composed group (``g(h(1))``) loses
        the pin the inner call earned: the outer site would read UNKNOWN and
        dispatch on the value ``h`` happens to produce.

        :param key: Scope-qualified id of the group
        :param pin: The pin this call site justified, when it has one
        :return: The type the call evaluates to
        """
        if pin is not None:
            picked = overload_pick(self._impl_sigs(key), pin)
            if picked is not None:
                return picked
        return self.table.funcs[key].ret

    def _impl_sigs(self, key: str) -> list[ImplSig]:
        """
        Every implementation of a user group, as the static selection reads it.

        An implementation with a keyword-only parameter that has no default is
        left out: a pin describes positional arguments only, so such a
        signature cannot be the one a pinned site binds to.

        :param key: Scope-qualified id of the group
        :return: One entry per implementation, in declaration order
        """
        returns = self._group_returns.get(key, [])
        out: list[ImplSig] = []
        for impl in self._defs.get(key, ()):
            if any(default is None for default in impl.args.kw_defaults):
                continue
            slot = self._group_slot.get(node_id(impl) or -1)
            ret = returns[slot] if slot is not None and slot < len(returns) else UNKNOWN
            out.append(impl_sig(impl, ret, self._ty_of, self._classes))
        return out

    def _call_context(self, key: str, node: ast.Call, frame: int,
                      args: Sequence[ast.expr] | None = None, record: bool = True) -> str:
        """
        Analyse the callee in the context this call site describes, and type it.

        The positional arguments are passed in where the call node does not
        spell them itself: a method call binds the RECEIVER to the first
        parameter, and its own node has the receiver in another slot.

        :param key: Scope-qualified id of the callee
        :param node: The call node
        :param frame: Index of the frame whose scope declares the callee
        :param args: The positional arguments, when they are not the node's own
        :param record: Whether the site takes part in the per-instance channel.
                       A method call does not: the isolation pass leaves
                       ``method_call`` raw and anchors nothing there, so a
                       vector stamped on it would configure nobody
        :return: The type this call evaluates to
        """
        definitions = self._defs.get(key)
        if not definitions:
            return self.table.funcs[key].ret
        # A redefined name is the LAST definition, the way Python binds it
        target = definitions[-1]
        parents = self._frames[:frame + 1]
        params = self._call_params(target, node, node.args if args is None else args)
        if params is None:
            if id(node) in self._bad_calls:
                # The call was reported: what it would evaluate to is moot
                return UNKNOWN
            # A shape this analysis does not describe -- an unpacking or a
            # starred argument. The callee still gets the one context its own
            # definition states, so the call is typed by whatever the body
            # says with UNKNOWN parameters
            settled = self._analyse(key, target, self._declared_params(target), parents)
            return UNKNOWN if settled is None else settled.ret
        if (key, params, node_id(target)) in self._in_progress:
            self._diag(f"'{_dotted(node.func)}' is re-entrant, so its result type is unknown",
                       node, self._unknown('recursion', node, key))
            return UNKNOWN
        result = self._analyse(key, target, params, parents, node)
        if result is None:
            self._diag(f"'{_dotted(node.func)}' hit the per-module context limit", node,
                       self._unknown('context-budget', node, key))
            return UNKNOWN
        nid = node_id(node)
        if nid is not None and record:
            # Only the RESOLVED path is recorded. The fallbacks above hand the
            # callee its own declaration context instead of the one this site
            # describes, so a vector derived from them would configure an
            # instance the call site never actually establishes
            self._call_ctx.setdefault(nid, {})[self._context] = result.cid
            self._callee_key[nid] = key
        return result.ret

    def _call_params(self, target: ast.FunctionDef | ast.AsyncFunctionDef,
                     node: ast.Call, args: Sequence[ast.expr],
                     report: bool = True) -> tuple[str, ...] | None:
        """
        The parameter types one call site instantiates the callee with.

        MEASURED on TradingView: the type of a parameter at a call site is
        JOIN(type of its default, type of the argument) -- a float argument to
        ``f(x = 0)`` makes ``x`` float, and an int argument to ``h(x = 0.0)``
        makes it float too. An annotation still outranks both, and an omitted
        argument leaves the default alone as the value that IS passed.

        :param target: The callee's definition
        :param node: The call node
        :param args: The positional arguments the call binds, in order
        :return: One type per parameter, or None when the shape is unresolvable
        """
        declared = target.args
        name = target.name

        def reject(message: str) -> None:
            if report:
                self._bad_call(node, name, message)

        if declared.vararg is not None or declared.kwarg is not None:
            return None
        if any(isinstance(a, ast.Starred) for a in args):
            return None
        if any(k.arg is None for k in node.keywords):
            return None
        positional = list(declared.posonlyargs) + list(declared.args)
        if len(args) > len(positional):
            reject(f"'{name}' takes {len(positional)} positional "
                                       f"argument(s), {len(args)} passed")
            return None

        bound: dict[str, ast.expr] = {}
        for arg, value in zip(positional, args):
            bound[arg.arg] = value
        declared_names = {a.arg for a in positional + list(declared.kwonlyargs)}
        for keyword in node.keywords:
            if keyword.arg not in declared_names:
                reject(f"'{name}' has no parameter '{keyword.arg}'")
                return None
            if keyword.arg in bound:
                reject(f"'{keyword.arg}' is passed to '{name}' twice")
                return None
            bound[keyword.arg] = keyword.value

        defaults = _param_defaults(target)
        out: list[str] = []
        for arg in _every_param(target):
            passed = bound.get(arg.arg)
            default = defaults.get(arg.arg)
            if passed is None and default is None:
                reject(f"'{name}' needs an argument for '{arg.arg}'")
                return None
            annotated = annotation_type(arg.annotation, self._classes)
            if annotated != UNKNOWN:
                if passed is not None:
                    given = self._ty_of(passed)
                    if given not in _UNCHECKED and not _fits(annotated, given):
                        reject(f"'{name}' takes {render_ty(annotated)} for "
                                        f"'{arg.arg}', {render_ty(given)} passed")
                        return None
                out.append(annotated)
            elif passed is None:
                out.append(self._ty_of(default))
            elif default is None:
                out.append(self._ty_of(passed))
            else:
                out.append(join(self._ty_of(default), self._ty_of(passed)))
        return tuple(out)

    def _bad_call(self, node: ast.Call, name: str, message: str) -> None:
        """A call whose shape or argument types the callee does not take."""
        self._bad_calls.add(id(node))
        self._node_diag(message, node, 'bad-call', name,
                        fix=f"call '{name}' the way it is declared")

    # --- calls into another module ---------------------------------------

    def _imported_call(self, node: ast.Call, name: str) -> str | None:
        """
        Type a call whose callee comes from another module.

        MEASURED: an imported function is typed from what it DECLARES and
        nothing else. TradingView compiles a library on its own, so a caller's
        argument types cannot reach into it the way they reach a helper of the
        same script -- an unannotated library parameter stays unannotated
        however the call spells its arguments.

        :param node: The call node
        :param name: The callee as written, dotted
        :return: The call's type, or None when the callee is not imported
        """
        head, _, rest = name.partition('.')
        parts = tuple(part for part in rest.split('.') if part)
        shadow = self._shadowed.get(head)
        binding = shadow.source if shadow is not None else self._imports.get(head)
        if binding is None:
            return None
        if self._nearer_binding(head):
            # A parameter or a local of some scope in between holds the name;
            # what it holds is whatever was assigned to it, not the import
            return UNKNOWN
        if head in self._multi_imports:
            self._diag(
                f"'{head}' is imported more than once, so what it calls is unknown",
                node, self._unknown('rebound-name', node, head),
                fix=f"import '{head}' once, under a name nothing else binds")
            return UNKNOWN
        if self._import_rebound(head):
            self._diag(
                f"'{head}' is assigned as well as imported, so what it calls is unknown",
                node, self._unknown('rebound-name', node, head),
                fix=f"call '{head}' under a name nothing assigns to")
            return UNKNOWN
        if shadow is not None:
            return self._shadowed_call(node, name, shadow, parts)

        path = binding.attrs + parts
        if not path:
            # The module object itself is being called, which types nothing
            return UNKNOWN
        interface = self._interface_of(binding.module, path[:-1], node)
        if interface is None:
            return UNKNOWN
        sig = interface.exports.get(path[-1])
        return UNKNOWN if sig is None else self._export_call(node, name, sig, interface)

    def _shadowed_call(self, node: ast.Call, name: str, shadow: _Shadowed,
                       parts: tuple[str, ...]) -> str:
        """
        Type a call through an alias that merges a library over a builtin namespace.

        The merge is per MEMBER, so the answer is too: the library's own
        exports come from its interface, and every other member from the
        builtin namespace the alias shadows. A library that cannot be resolved
        at all decides neither half, so nothing is claimed for it.

        :param node: The call node
        :param name: The callee as written
        :param shadow: The merged namespace the head is bound to
        :param parts: The attribute path after the head
        :return: The type the call evaluates to
        """
        if len(parts) != 1:
            return UNKNOWN
        member = parts[0]
        interface = self._interface_of(shadow.source.module, shadow.source.attrs, node)
        if interface is None:
            return UNKNOWN
        published = interface.all if interface.all is not None else tuple(interface.exports)
        if member not in published:
            return self._lib_call(node, f'{shadow.namespace}.{member}')
        sig = interface.exports.get(member)
        return UNKNOWN if sig is None else self._export_call(node, name, sig, interface)

    def _export_call(self, node: ast.Call, name: str, sig: ExportSig,
                     interface: ModuleInterface) -> str:
        """
        Type one call against the signature its module publishes.

        A group is pinnable exactly as a same-module one is: the pin selects
        among SIGNATURES, and needs no body to do it. A plain function has
        nothing to select, so its declared return is the whole answer -- and
        where its parameters carry no annotations there is no answer at all,
        because the module was analysed without this call site and the types it
        would have needed are the ones it never got.

        :param node: The call node
        :param name: The callee as written, for the call site record
        :param sig: The published signature
        :param interface: The module that publishes it
        :return: The type the call evaluates to
        """
        is_group = sig.kind == 'group'
        if not self._export_call_fits(node, name, sig):
            self.table.calls.append(CallSite(
                callee=name, line=_line(node), col=_col(node),
                argc=_call_arity(node), ty=UNKNOWN, pin=None))
            return UNKNOWN
        pin = self._pin(node, is_group)
        if is_group:
            picked = None if pin is None else overload_pick(list(sig.impls), pin)
            ty = sig.ret if picked is None else picked
        elif sig.annotated:
            ty = sig.ret
        else:
            ty = UNKNOWN
            where = f'{interface.path}:{sig.line}'
            self._diag(
                f"'{sig.name}' is imported from {where}, where its parameters carry no "
                f"annotations, so what it returns is unknown", node,
                self._unknown('unannotated-import', node, sig.name),
                fix=f"annotate the parameters of '{sig.name}' in {where}")
        self.table.calls.append(CallSite(
            callee=name, line=_line(node), col=_col(node),
            argc=_call_arity(node), ty=ty, pin=pin))
        return ty

    def _export_call_fits(self, node: ast.Call, name: str, sig: ExportSig,
                          args: Sequence[ast.expr] | None = None) -> bool:
        """
        Whether a call matches the signature its module publishes.

        The interface carries each export's parameter types, names and
        arity -- for a group, one shape per implementation -- which is all a
        call site needs to be held to it, the same way a same-module call is.

        :param node: The call node
        :param name: The callee as written
        :param sig: The published signature
        :param args: The positional arguments, when they are not the node's
                     own (a method call binds the receiver first)
        :return: True when the call may run
        """
        positional_args = list(node.args if args is None else args)
        if any(isinstance(arg, ast.Starred) for arg in positional_args) \
                or any(keyword.arg is None for keyword in node.keywords):
            return True
        shapes: list[ExportSig | ImplSig] = list(sig.impls) if sig.kind == 'group' else [sig]
        argc = len(positional_args) + len(node.keywords)
        if not any(shape.required <= argc and (shape.open_ended or argc <= len(shape.params))
                   for shape in shapes):
            self._bad_call(node, name, f"'{name}' does not take {argc} argument(s)")
            return False
        for keyword in node.keywords:
            if not any(keyword.arg in shape.names for shape in shapes):
                self._bad_call(node, name, f"'{name}' has no parameter '{keyword.arg}'")
                return False
        if sig.kind == 'group':
            return True
        given: list[tuple[str, str, ast.expr]] = []
        for index, arg in enumerate(positional_args[:len(sig.params)]):
            given.append((sig.names[index] if index < len(sig.names) else str(index),
                          sig.params[index], arg))
        positional = set(sig.names[:len(positional_args)])
        for keyword in node.keywords:
            if keyword.arg in positional:
                self._bad_call(node, name, f"'{keyword.arg}' is passed to '{name}' twice")
                return False
            if keyword.arg in sig.names:
                given.append((keyword.arg, sig.params[sig.names.index(keyword.arg)],
                              keyword.value))
        for pname, declared, arg in given:
            passed = self._ty_of(arg)
            if declared == UNKNOWN or passed in _UNCHECKED or _fits(declared, passed):
                continue
            self._bad_call(node, name, f"'{name}' takes {render_ty(declared)} for '{pname}', "
                                       f"{render_ty(passed)} passed")
            return False
        return True

    def _interface_of(self, module: str, hops: tuple[str, ...],
                      node: ast.AST) -> ModuleInterface | None:
        """
        The interface of the module a dotted callee reaches, and the dependency on it.

        Every segment before the export name has to be a MODULE. One that is
        not is an attribute path -- a UDT's method, an object's field -- and
        this pass does not follow those, so the callee is simply unknown.

        Consulting an interface is what creates a dependency: the record goes
        into the table, the loader bakes it into the bytecode and re-checks it
        before accepting the cache, so a signature that moves is noticed.

        :param module: Dotted module the import names
        :param hops: Attribute path leading to the module that owns the export
        :param node: The node the interface is wanted for, for the cycle
                     diagnostic
        :return: The interface, or None when there is none to be had
        """
        dotted = '.'.join((module,) + hops)
        path = self._module_source(dotted)
        if path is None:
            return None
        return self._interface_at(path, dotted, node)

    def _interface_at(self, path: str, dotted: str,
                      node: ast.AST) -> ModuleInterface | None:
        """
        The interface of one module SOURCE, and the dependency on it.

        Taking the source path rather than the import spelling is what lets a
        class id be followed home: its module half IS a resolvable source path
        by contract (see ``class_id``), so a value whose class no import of
        this module names can still be asked what it declares.

        :param path: Resolved source path of the module
        :param dotted: What names it, for the cycle diagnostic
        :param node: The node the interface is wanted for
        :return: The interface, or None when there is none to be had
        """
        if pine_type_artifact.analysing(path):
            self._node_diag(
                f"'{dotted}' imports this module back, so its signatures are not "
                f"available yet", node, 'import-cycle', dotted,
                fix=f'break the import cycle between {self.table.module_path} and {path}')
            return None
        if path not in self._interfaces:
            interface = pine_type_artifact.lookup(path, self._analyser, self._pipeline_hash)
            self._interfaces[path] = interface
            if interface is not None:
                if interface.suppressed:
                    # What it publishes may rest on a container it cannot
                    # trust, and neither can a pin built here on one of its
                    # return types: this module gives its pins up with it
                    self._diag(
                        f"'{dotted}' is reported, and no type it publishes may drive "
                        f"a dispatch here: {interface.suppressed}", node,
                        self._unknown('suppressed-import', node, dotted),
                        fix=f"fix '{dotted}' first")
                    if self._pins_suppressed is None:
                        self._pins_suppressed = self.table.diags[-1]
                # Every class the module publishes, whether or not an
                # annotation here names one: a value of such a class reaches
                # this module through a CALL as readily as through a
                # declaration, and a field read of it needs the fields behind
                # the class id that value carries
                for sig in interface.classes.values():
                    self._class_sigs.setdefault(sig.id, sig)
                record = pine_type_artifact.dep_record(interface)
                self.table.deps[record.path] = record
                # Its dependencies are this module's too. An export whose
                # return was INFERRED from a call one module further out moves
                # when THAT module's signature moves, and nothing about this
                # module or its direct dependency changes when it does -- so
                # the closure has to be carried, not just the edge
                for inherited in interface.deps.values():
                    if inherited.path == self.table.module_path:
                        # A cyclic pair names this module in the other's
                        # closure; its own source is not something it can be
                        # invalidated by
                        continue
                    self.table.deps.setdefault(inherited.path, inherited)
        return self._interfaces[path]

    def _module_source(self, dotted: str) -> str | None:
        """
        Where a module's source lives, without importing the module itself.

        ``find_spec`` is what keeps this free of side effects: the module whose
        signatures are wanted is never executed -- the analysis reads its
        source -- and a segment that is not a module has no spec at all, which
        is exactly the question the caller is asking.

        :param dotted: Dotted module name
        :return: Its source path, or None when it is not a readable Python module
        """
        if dotted in self._module_paths:
            return self._module_paths[dotted]
        origin: str | None = None
        try:
            spec = importlib.util.find_spec(dotted)
        except Exception:  # noqa: any resolution failure means "not a module here"
            spec = None
        if spec is not None and spec.origin and spec.origin.endswith('.py'):
            origin = spec.origin
        self._module_paths[dotted] = origin
        return origin

    def _nearer_binding(self, name: str) -> bool:
        """
        Whether a scope between the call and the module binds the imported name.

        :param name: The head of the callee
        :return: True when some enclosing function scope binds it
        """
        return any(name in self._rebound.get(frame.scope, ())
                   for frame in self._frames[1:])

    def _import_rebound(self, name: str) -> bool:
        """
        Whether the module binds an imported name by anything but the import.

        Order says nothing here, unlike for a definition: an import is a VALUE
        binding like every other one, so two of them are simply two stores to
        the same name and neither is the one a call reaches.

        :param name: The head of the callee
        :return: True when a module-level binding other than the import has it
        """
        own = self._import_positions.get(name, frozenset())
        return any(position not in own
                   for position, _ in self._module_rebinds.get(name, ()))

    # --- the per-instance pin channel ------------------------------------

    def _stamp_instance_vectors(self, tree: ast.Module) -> None:
        """
        Work out what a shared body resolves PER INSTANCE, and stamp the wire.

        One body is shared by every context it was instantiated in, so a site
        inside it whose answer differs between those contexts cannot be
        emitted as a constant. Such a site is *instance-varying*: an overload
        site whose pin differs (``get_pins`` is present), or a call to a
        generic callee whose own instance vector differs. The two are the same
        question one level apart, so the definition is recursive and settled by
        a fixpoint -- making a callee's site vary can make its caller's site
        vary too.

        A function's VECTOR under one context is one entry per varying site of
        that function, in source order: the pin character for an overload site,
        the callee's own vector (nested) for a generic one, and None where this
        context reached the site with nothing to say. None means "configure
        nothing" all the way down -- the callee then keeps the all-None default
        its layout carries, which is the value dispatch the site had before
        there was a channel.

        Two stamps come out, and they are the whole contract with
        :mod:`~pynecore.transformers.function_isolation`:

        * ``set_varying`` on a definition -- its varying sites in source order.
          The list index IS the site's index in every vector of that function,
          and the isolation pass reserves one state slot to hold the vector.
        * ``set_vector`` on a call site -- the vector to hand the callee, where
          every context reaching the site agrees on one. A site that is itself
          varying is left unstamped on purpose: it reads its vector out of its
          caller's slot instead of carrying a constant.

        An overload GROUP is left out entirely. Its implementations are
        per-signature already, they are reached through a dispatcher that
        creates their state vectors itself (``overload._anchored``), and no
        call site of the group can hand that dispatcher anything.

        :param tree: The module being stamped
        """
        targets = {key: defs[-1] for key, defs in self._defs.items()
                   if key not in self._overload_groups and defs}
        cids: dict[str, list[int]] = {}
        for result in self.table.contexts.values():
            cids.setdefault(result.key, []).append(result.cid)
        own = {key: _own_calls(node) for key, node in targets.items()}
        owner: dict[int, str] = {}
        for key, calls in own.items():
            for call in calls:
                nid = node_id(call)
                if nid is not None:
                    owner[nid] = key
        varying: dict[str, list[ast.Call]] = {
            key: [call for call in calls if get_pins(call) is not None]
            for key, calls in own.items()}
        memo: dict[tuple[str, int], tuple] = {}

        def vector_of(key: str, cid: int, stack: frozenset) -> tuple:
            sites = varying.get(key)
            if not sites:
                return ()
            token = (key, cid)
            found = memo.get(token)
            if found is not None:
                return found
            if token in stack:
                # A context cycle cannot be unfolded; the callee keeps its
                # default, which is the dispatch it had before
                return (None,) * len(sites)
            built = tuple(entry_of(site, cid, stack | {token}) for site in sites)
            memo[token] = built
            return built

        def entry_of(site: ast.Call, cid: int, stack: frozenset) -> Any:
            nid = node_id(site)
            if nid is None:
                return None
            if get_pins(site) is not None:
                return self._pins.get(nid, {}).get(cid)
            callee = self._callee_key.get(nid)
            target = self._call_ctx.get(nid, {}).get(cid)
            if callee is None or target is None:
                return None
            return vector_of(callee, target, stack)

        changed = True
        while changed:
            changed = False
            for key, calls in own.items():
                found: list[ast.Call] = []
                for call in calls:
                    if get_pins(call) is not None:
                        found.append(call)
                    elif len({entry_of(call, cid, frozenset())
                              for cid in cids.get(key, ())}) > 1:
                        found.append(call)
                if found != varying[key]:
                    varying[key] = found
                    memo.clear()
                    changed = True

        for key, sites in varying.items():
            set_varying(targets[key], sites or None)
        per_instance = {id(call) for sites in varying.values() for call in sites}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or id(node) in per_instance:
                continue
            nid = node_id(node)
            callee = self._callee_key.get(nid) if nid is not None else None
            if callee is None or not varying.get(callee):
                continue
            key = owner.get(nid)
            callers = cids.get(key, ()) if key is not None else (0,)
            seen = {entry_of(node, cid, frozenset()) for cid in callers}
            if len(seen) != 1:
                continue
            vector = seen.pop()
            if vector is not None and vector != (None,) * len(varying[callee]):
                set_vector(node, vector)

    def _lib_name(self, node: ast.expr) -> str | None:
        """
        The registry key a lib reference resolves to.

        After import normalization every lib reference is spelled
        ``lib.<dotted path>``, so the key is that path with the alias stripped.

        :param node: The referenced expression
        :return: The dotted key, or None when this is not a lib reference
        """
        dotted = _dotted(node)
        if dotted is None:
            return None
        head, _, rest = dotted.partition('.')
        if head in self._lib_aliases and rest:
            return rest
        return None


def _security_writes(tree: ast.Module) -> dict[str, list[ast.expr]]:
    """
    What every security id publishes, by id.

    ``SecurityTransformer`` splits one ``request.security(...)`` expression
    into a guarded ``__sec_write__('id', expr)`` and a ``__sec_read__('id',
    default)`` that stands where the expression did. Collected up front, in one
    walk, because the read is what a script's value flows through and it has to
    find the write however the two are laid out.

    :param tree: The module being walked
    :return: security id -> the expressions written under it
    """
    out: dict[str, list[ast.expr]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) < 2:
            continue
        if _dotted(node.func) != '__sec_write__':
            continue
        sec_id = node.args[0]
        if isinstance(sec_id, ast.Constant) and isinstance(sec_id.value, str):
            out.setdefault(sec_id.value, []).append(node.args[1])
    return out


def _every_param(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.arg]:
    """Every named parameter of a definition, positional first then kw-only."""
    return list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)


def _bound_positions(body: Sequence[ast.AST]) -> dict[str, list[tuple[int, int]]]:
    """
    Every name one scope's statements bind by something other than a ``def``,
    with the position of EVERY binding of each.

    An assignment, an augmented assignment, a walrus, a loop target, a ``with``
    target, an import, an ``except ... as``, a ``match`` capture and a class
    name all put a VALUE under the name; a definition does not appear here,
    because the definition is what these are measured against. Nested
    definitions are not descended into -- their bindings belong to their own
    scope -- while an ``if`` or a ``for`` body is part of this one, and a
    binding nested in one counts where it stands.

    A ``global``/``nonlocal`` STATEMENT is not one of them. It stores nothing;
    it only says where the name lives, and a scope that declares ``global g``
    without ever assigning to it reads the very binding the declaration names.
    An actual store through the declaration is an assignment like any other and
    is recorded as one. ``_declared_names`` is what tracks the declarations.

    The positions are what makes MODULE scope answerable: binding is
    sequential up there, so a call written above the rebinding still reaches
    the definition -- and every position is kept, not just the earliest,
    because a rebinding inside a LOOP shadows the calls of that loop's body
    however many earlier ones stand below them (see ``_reaches_def``).

    :param body: The scope's own statements
    :return: bound name -> the positions it is bound at, in source order
    """
    out: dict[str, list[tuple[int, int]]] = {}

    def record(name: str, at: ast.AST) -> None:
        out.setdefault(name, []).append((_line(at), _col(at)))

    stack: list[ast.AST] = list(body)
    while stack:
        current = stack.pop()
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(current, ast.ClassDef):
            record(current.name, current)
            continue
        if isinstance(current, ast.Name):
            if not isinstance(current.ctx, ast.Load):
                record(current.id, current)
        elif isinstance(current, ast.alias):
            record((current.asname or current.name).split('.')[0], current)
        elif isinstance(current, ast.ExceptHandler) and current.name is not None:
            record(current.name, current)
        elif isinstance(current, (ast.MatchAs, ast.MatchStar)) and current.name is not None:
            record(current.name, current)
        elif isinstance(current, ast.MatchMapping) and current.rest is not None:
            record(current.rest, current)
        stack.extend(ast.iter_child_nodes(current))
    # The walk is depth-first over a LIFO stack, which visits the siblings
    # back to front; source order is what the positions are compared in
    return {name: sorted(positions) for name, positions in out.items()}


def _module_defs(body: Sequence[ast.stmt]) -> dict[str, list[tuple[tuple[int, int], bool]]]:
    """
    Every ``def`` the MODULE scope declares, with where it stands and whether
    anything guards it.

    Guarded means nested in a statement at all -- an ``if``, a ``for``, a
    ``try``, a ``with``, a ``match``. Only a definition directly in the
    module's own body is certain to have run by the time the module is
    imported; every other one is a definition that MAY have bound the name.
    Nested scopes are not descended into: a method or an inner helper binds a
    name in its own scope, and no importer ever reaches it.

    :param body: The module's own statements
    :return: name -> (position, unguarded) per definition of it, in source order
    """
    out: dict[str, list[tuple[tuple[int, int], bool]]] = {}

    def collect(stmts: Sequence[ast.stmt], unguarded: bool) -> None:
        for stmt in stmts:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.setdefault(stmt.name, []).append(((_line(stmt), _col(stmt)), unguarded))
            elif isinstance(stmt, ast.ClassDef):
                continue
            else:
                for nested in _statement_lists(stmt):
                    collect(nested, False)

    collect(body, True)
    return out


def _exportable_names(body: Sequence[ast.stmt],
                      rebinds: dict[str, list[tuple[tuple[int, int], _BranchPath]]],
                      groups: Container[str]) -> frozenset[str]:
    """
    The module-level names an importer is guaranteed to find a definition under.

    A module binds sequentially, so what an importer receives is whatever the
    LAST binding of a name put there -- and that is a question of POSITION, not
    of which shapes the module happens to contain. ``pick = other`` above
    ``def pick`` leaves the definition bound and is perfectly exportable; the
    same two lines the other way round are not, and an order-blind "the name is
    bound by something else too" answers neither.

    So the rule is the runtime one: the last module-level binding of the name
    has to be a ``def``, and that def has to be one no branch guards. Two
    definitions in exclusive branches (``if X: def f`` / ``else: def f``) fail
    it -- publishing either would name a function half the runs never bind --
    while a guarded definition CLOSED by a later unguarded one passes, because
    the later one binds the name whatever the branch did.

    An ``@overload`` group is held to more than its last member: a group
    publishes every implementation, so one of them standing in a branch makes
    the published set itself conditional.

    :param body: The module's own statements
    :param rebinds: Module-scope name -> where anything OTHER than a ``def``
                    binds it, with the branch path of each
    :param groups: The scope-qualified ids of the module's ``@overload``
                   groups; a module-level group's id is its bare name
    :return: The names the interface may publish a definition for
    """
    exportable: set[str] = set()
    for name, positions in _module_defs(body).items():
        last, unguarded = max(positions)
        if not unguarded:
            continue
        if any(position > last for position, _ in rebinds.get(name, ())):
            continue
        if name in groups and not all(flag for _, flag in positions):
            continue
        exportable.add(name)
    return frozenset(exportable)


def _bound_names(body: Sequence[ast.AST]) -> set[str]:
    """
    Every name one scope's statements bind by something other than a ``def``.

    :param body: The scope's own statements
    :return: The names it binds
    """
    return set(_bound_positions(body))


def _declared_names(body: Sequence[ast.AST]) -> tuple[set[str], set[str]]:
    """
    The names one scope declares ``global`` or ``nonlocal``.

    Such a declaration is not a binding at all: it says the name lives in
    ANOTHER scope, so every read and every write goes there. What makes a
    ``global helper`` scope's ``helper()`` unresolvable is the STORE the
    declaration enables, not the declaration -- and the store is an ordinary
    assignment, which ``_bound_positions`` records where it stands. This is
    what tells the free-name walk to take such a name back out of the locals.

    Nested definitions are not descended into: their declarations belong to
    their own scope.

    :param body: The scope's own statements
    :return: (``global`` names, ``nonlocal`` names)
    """
    globals_: set[str] = set()
    nonlocals: set[str] = set()
    stack: list[ast.AST] = list(body)
    while stack:
        current = stack.pop()
        if isinstance(current, _SCOPE_NODES):
            continue
        if isinstance(current, ast.Global):
            globals_.update(current.names)
        elif isinstance(current, ast.Nonlocal):
            nonlocals.update(current.names)
        stack.extend(ast.iter_child_nodes(current))
    return globals_, nonlocals


def _scope_free(node: _Scope) -> tuple[set[str], set[str]]:
    """
    The names one lexical scope reads from OUTSIDE itself.

    Scope by scope, not one flat walk over the subtree. A nested definition
    binds its own parameters and its own locals, so ``def inner(total)`` says
    nothing about the ``total`` the enclosing body reads -- flattening the two
    let that parameter cancel the enclosing read, and the read is exactly what
    tells the inference a memoized analysis has gone stale: the enclosing type
    could then widen without any memo key moving with it.

    What a nested scope contributes instead is its OWN free names, minus
    whatever this scope binds -- a capture the enclosing body does not define
    is a read of the scope above it, and travels outward until some scope does.

    A ``global``/``nonlocal`` declaration is the case where the name is bound
    here and free all the same: it names a binding somewhere else, so a body
    that says ``nonlocal total`` and reads ``total`` is reading the enclosing
    scope's value, and the memo key has to move when that value's type does.
    Counting the declaration as a local binding hid exactly that, and the pin
    a stale context had justified stayed on the call site.

    The two halves are kept apart because they resolve differently: a
    ``global`` name skips every intermediate scope and lands on the module's
    binding, so an enclosing scope's same-named local must NOT cancel it.

    :param node: The scope to scan
    :return: (names resolved lexically, names ``global`` forces to the module)
    """
    body: list[ast.AST] = list(node.body) if isinstance(node.body, list) else [node.body]
    declared_global, declared_nonlocal = _declared_names(body)
    bound = _bound_names(body) - declared_global - declared_nonlocal
    args = getattr(node, 'args', None)
    if isinstance(args, ast.arguments):
        bound.update(arg.arg for arg in
                     args.posonlyargs + args.args + args.kwonlyargs)
        bound.update(arg.arg for arg in (args.vararg, args.kwarg) if arg is not None)

    loaded: set[str] = set()
    nested: list[_Scope] = []
    stack: list[ast.AST] = list(body)
    while stack:
        current = stack.pop()
        if isinstance(current, _SCOPE_NODES):
            nested.append(current)
            if not isinstance(current, ast.Lambda) and current.name not in declared_global:
                bound.add(current.name)
            # A nested definition's decorators, defaults and annotations are
            # evaluated where the definition STANDS, not inside it
            stack.extend(_scope_header(current))
            continue
        if isinstance(current, ast.Name) and isinstance(current.ctx, ast.Load):
            loaded.add(current.id)
        stack.extend(ast.iter_child_nodes(current))

    free = (loaded - bound) - declared_global
    module_free = loaded & declared_global
    for scope in nested:
        nested_free, nested_module = _scope_free(scope)
        free |= nested_free - bound
        module_free |= nested_module
    return free, module_free


def _scope_header(node: _Scope) -> list[ast.expr]:
    """
    The parts of a nested scope that are evaluated where it STANDS.

    :param node: The nested definition, class or lambda
    :return: The expressions belonging to the enclosing scope
    """
    out: list[ast.expr] = list(getattr(node, 'decorator_list', ()))
    if isinstance(node, ast.ClassDef):
        out += list(node.bases) + [keyword.value for keyword in node.keywords]
        return out
    args = node.args
    out += list(args.defaults)
    out += [default for default in args.kw_defaults if default is not None]
    out += [arg.annotation
            for arg in args.posonlyargs + args.args + args.kwonlyargs
            if arg.annotation is not None]
    if not isinstance(node, ast.Lambda) and node.returns is not None:
        out.append(node.returns)
    return out


def _called_names(node: ast.AST) -> set[str]:
    """
    Every bare name a subtree calls, its nested definitions included.

    Nested ones count: a module-level helper whose only caller sits three
    scopes down inside another definition is still called by that definition
    as far as walk ORDER is concerned.

    :param node: The subtree to scan
    :return: The bare callee names it mentions
    """
    return {child.func.id for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)}


def _labelled_lists(stmt: ast.stmt) -> Iterator[tuple[str, list[ast.stmt]]]:
    """
    Every statement list nested directly in one statement, with its label.

    The label names WHICH branch of the statement the list is: ``'body'``,
    ``'orelse'`` or ``'finalbody'`` of an ``if``, a loop or a ``try``,
    ``'handler:<i>'`` of one ``except`` clause, ``'case:<i>'`` of one ``match``
    case. That is what tells two nested positions apart -- see
    ``_mutually_exclusive``.

    :param stmt: The statement to open up
    :return: (branch label, statements) pairs
    """
    for label in ('body', 'orelse', 'finalbody'):
        nested = getattr(stmt, label, None)
        if isinstance(nested, list):
            yield label, nested
    for index, handler in enumerate(getattr(stmt, 'handlers', None) or ()):
        yield f'handler:{index}', handler.body
    for index, case in enumerate(getattr(stmt, 'cases', None) or ()):
        yield f'case:{index}', case.body


def _module_statements(body: Sequence[ast.stmt]) -> Iterator[ast.stmt]:
    """
    Every statement that belongs to the MODULE scope, nested ones included.

    An ``if`` or a ``try`` at module level opens no scope, so an import inside
    one binds a module-level name like any other -- which is how a guarded
    import (``try: import x``) is spelled. A ``def`` or a ``class`` does open
    one and is not descended into.

    Source order, which is the order the module RUNS in: a reader of this
    sequence is deciding what a name is bound to, and the last statement to
    bind it is the one that decides.

    :param body: The module's own statements
    :return: Its statements, in source order
    """
    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield stmt
        for nested in _statement_lists(stmt):
            yield from _module_statements(nested)


def _statement_lists(stmt: ast.stmt) -> Iterator[list[ast.stmt]]:
    """
    Every statement list nested directly in one statement.

    A ``def`` can live inside an ``if``, a ``for``, a ``try`` or a class body,
    and it is a definition wherever it lives. Walking only the plain top-level
    lists left those functions with no signature at all.

    :param stmt: The statement to open up
    :return: Its nested statement lists
    """
    for _, nested in _labelled_lists(stmt):
        yield nested


def _mutually_exclusive(first: _BranchPath, second: _BranchPath) -> bool:
    """
    Whether one pass of the module body can never reach both positions.

    Source order alone cannot answer that: ``if flag: g = other`` with the call
    in the ``else`` puts the rebinding ABOVE the call and yet no run of the
    module executes them both. What decides it is where the two paths first
    take different branches of the SAME statement: an ``if`` runs one of its
    two branches and a ``match`` one of its cases, so those separate them.

    Nothing else does. A ``try`` body and one of its handlers both run when the
    body raises halfway through -- the body may well have rebound the name
    before it did -- and a loop's ``body`` and its ``orelse`` both run in the
    ordinary case. ``elif`` needs no rule of its own: it is an ``if`` nested in
    the outer one's ``orelse``, which the first-divergence test already covers.

    Paths that never diverge -- one a prefix of the other, or a divergence into
    two different sibling statements -- are sequential, not exclusive.

    :param first: One position's branch path
    :param second: The other position's branch path
    :return: True when no single pass reaches both
    """
    for (stmt, label), (other, other_label) in zip(first, second):
        if stmt is not other:
            return False
        if label != other_label:
            return isinstance(stmt, (ast.If, ast.Match))
    return False


class _BranchIndex:
    """
    Where every statement of one module body stands in its branch structure.

    Built once per module: the branch path of a position is then a lookup, not
    another walk of the tree. Function bodies are left out -- they are other
    scopes, and ``_reaches_def`` answers for the module body alone.

    An UNGUARDED ``match`` case's PATTERN is indexed on the case's own path
    too, not only its body. A capture there binds exactly when that case is
    the one taken, so ``case [g]:`` above ``case _: result = g(1 / 2)`` is as
    exclusive with the call as an assignment in the first case's body would be
    -- the pattern that did not match bound nothing.

    A GUARDED case's pattern is not, and neither is the guard itself. A
    pattern that matched leaves its captures bound even when the guard then
    fails, and the subject goes on to the NEXT case carrying them, so both the
    capture and the guard's walrus fall back to the ``match`` statement's own
    path, which no case is exclusive with.
    """

    __slots__ = ('_entries', '_starts')

    def __init__(self, body: Sequence[ast.stmt]) -> None:
        entries: list[tuple[tuple[int, int], tuple[int, int], _BranchPath]] = []

        def collect(stmts: Sequence[ast.stmt], path: _BranchPath) -> None:
            for stmt in stmts:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                start, end = _span(stmt)
                entries.append((start, end, path))
                if isinstance(stmt, ast.Match):
                    for index, case in enumerate(stmt.cases):
                        if case.guard is not None:
                            continue
                        pattern_start, pattern_end = _span(case.pattern)
                        entries.append((pattern_start, pattern_end,
                                        path + ((stmt, f'case:{index}'),)))
                for label, nested in _labelled_lists(stmt):
                    collect(nested, path + ((stmt, label),))

        collect(body, ())
        entries.sort(key=lambda entry: entry[0])
        self._entries = entries
        self._starts = [entry[0] for entry in entries]

    def of(self, position: tuple[int, int]) -> _BranchPath:
        """
        The branch path of the innermost statement covering one position.

        A statement starts before everything nested in it, so the innermost
        cover is the LAST one to start at or before the position -- the ones
        walked past on the way are siblings that already ended above it.

        :param position: A (line, column) pair
        :return: Its branch path, empty when the module body does not cover it
        """
        for index in range(bisect_right(self._starts, position) - 1, -1, -1):
            _, end, path = self._entries[index]
            if position <= end:
                return path
        return ()


def _is_overload(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """
    Whether a definition is one implementation of an ``@overload`` group.

    The decorator is matched by its bare name, however it is spelled: a Pyne
    script imports it as ``overload``, the compiled form may qualify it. A
    ``typing.overload`` stub matches too, and harmlessly -- such a name
    publishes no ``__pyne_bind__`` factory, so the pin never reaches a
    binding.

    :param node: The definition to inspect
    :return: True when it carries an ``overload`` decorator
    """
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Attribute) and target.attr == 'overload':
            return True
        if isinstance(target, ast.Name) and target.id == 'overload':
            return True
    return False


def _is_method(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """
    Whether a definition is a Pine method.

    The decorator is matched by its bare name, however it is spelled: a
    compiled script imports it as ``method`` and may qualify it.

    :param node: The definition to inspect
    :return: True when it carries a ``method`` decorator
    """
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Attribute) and target.attr == 'method':
            return True
        if isinstance(target, ast.Name) and target.id == 'method':
            return True
    return False


def _annotation_names(tree: ast.Module) -> dict[str, ast.expr]:
    """
    Every dotted name a module's annotations spell, with where it stands.

    Only the annotations, not the whole tree: this is what decides which
    imports are worth resolving an interface for, and an import a call reaches
    is resolved by the call itself. A stringized annotation is parsed and read
    the same way, since that is how a forward reference is spelled.

    :param tree: The module to scan
    :return: dotted spelling -> the annotation node it was found in
    """
    out: dict[str, ast.expr] = {}

    def take(node: ast.expr, where: ast.expr) -> None:
        for child in ast.walk(node):
            if isinstance(child, (ast.Name, ast.Attribute)):
                spelled = _dotted(child)
                if spelled is not None:
                    out.setdefault(spelled, where)
            elif isinstance(child, ast.Constant) and isinstance(child.value, str):
                try:
                    take(ast.parse(child.value, mode='eval').body, where)
                except SyntaxError:
                    continue

    for node in ast.walk(tree):
        if isinstance(node, (ast.arg, ast.AnnAssign)):
            annotation = node.annotation
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotation = node.returns
        else:
            continue
        if annotation is not None:
            take(annotation, annotation)
    return out


def _line(node: ast.AST) -> int:
    """
    A node's line, tolerating the synthetic ones.

    Earlier passes emit nodes without positions; the pipeline only fills them
    in at the very end (``transformers/locations.py``), so anything read here
    has to survive their absence.
    """
    return getattr(node, 'lineno', 0)


def _col(node: ast.AST) -> int:
    """A node's column, tolerating the synthetic ones."""
    return getattr(node, 'col_offset', 0)


def _span(node: ast.AST) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    The source range one statement covers, as comparable positions.

    A synthetic node has no end either, and an empty span is the honest answer
    for it: nothing can be found inside a statement the parser never saw.

    :param node: The statement to measure
    :return: (start, end), each a (line, column) pair
    """
    start = (_line(node), _col(node))
    end = (getattr(node, 'end_lineno', None) or start[0],
           getattr(node, 'end_col_offset', None) or start[1])
    return start, end


#: Values no fit check has a say about: a lost type, a typeless na, a bare
#: object, and ``None`` -- the na of every object and reference type
_UNCHECKED = frozenset({UNKNOWN, TYPELESS, OBJECT, VOID})


def _fits(declared: str, passed: str) -> bool:
    """
    Whether a value of one type may stand where another is declared.

    An int fits a float slot, a float does not fit an int slot, and two
    shapes fit only when they are the same: the declared type has to be what
    the join comes back as.

    :param declared: The declared type
    :param passed: The value's type
    :return: True when the value fits
    """
    if declared in SCALARS and passed.startswith('o:'):
        # A lib constant that IS a scalar (``format.percent`` is a string)
        passed = lib_scalar_classes().get(passed, passed)
    return shape_conflict(declared, passed) is None and join(declared, passed) == declared


#: Annotation heads that give a name a HISTORY: ``x[n]`` reads a series
_SERIES_HEADS = frozenset({'Series', 'PersistentSeries', 'IBPersistentSeries'})


def _spells_series(annotation: ast.expr | None) -> bool:
    """Whether an annotation wraps its type in a series head."""
    if annotation is None:
        return False
    return any((isinstance(sub, ast.Name) and sub.id in _SERIES_HEADS)
               or (isinstance(sub, ast.Attribute) and sub.attr in _SERIES_HEADS)
               for sub in ast.walk(annotation))


def _call_arity(node: ast.Call) -> int | None:
    """
    How many arguments a call actually passes.

    A keyword argument IS an argument: ``math.round(x, precision=2)`` is the
    two-argument -- float-typed -- form, and counting only ``node.args`` would
    resolve it to the one-argument int overload. An unpacking (``*seq``,
    ``**kw``) makes the count unknowable, which is what None says; the
    overrides and the overload groups then decline to pick rather than pick
    wrong.

    :param node: The call node
    :return: The argument count, or None when an unpacking hides it
    """
    return _arity(node.args, node.keywords)


def _arity(args: Sequence[ast.expr], keywords: Sequence[ast.keyword]) -> int | None:
    """
    How many arguments one argument list passes.

    :param args: The positional arguments
    :param keywords: The keyword arguments
    :return: The count, or None when an unpacking hides it
    """
    if any(isinstance(a, ast.Starred) for a in args):
        return None
    if any(k.arg is None for k in keywords):
        return None
    return len(args) + len(keywords)


def _bound_arg(args: Sequence[ast.expr], keywords: Sequence[ast.keyword], index: int,
               param_names: list[str] | None) -> ast.expr | None:
    """
    The expression bound to one declared parameter position.

    A type-preserving override names the parameter it copies from, and Python
    lets the caller spell that parameter either way, so the keywords have to be
    bound back to their declared position before the position can be read. An
    unpacking hides which position an argument landed on, and is unresolvable.

    :param args: The positional arguments, in order
    :param keywords: The keyword arguments
    :param index: Declared parameter position, 0-based
    :param param_names: The callee's declared parameter order, when it is known
    :return: The bound expression, or None when it cannot be determined
    """
    if any(isinstance(a, ast.Starred) for a in args):
        return None
    if index < len(args):
        return args[index]
    if param_names is None or index >= len(param_names):
        return None
    wanted = param_names[index]
    for keyword in keywords:
        if keyword.arg == wanted:
            return keyword.value
    return None


def _lib_impl_sig(impl: dict[str, Any]) -> ImplSig:
    """
    One registry implementation, as the static selection reads it.

    The registry has no keyword-only parameters -- ``lib_type_collector``
    records the positional ones only -- so the fits string is exactly as long
    as ``params``.

    ``default_none_ok`` is absent unless the implementation has a literal
    ``None`` default, which is the only default whose fit the annotation's
    None-acceptance decides; where it is absent the flag is never read.

    :param impl: The generated entry of one implementation
    :return: Its shape
    """
    params = tuple(impl['params'])
    required = len(params) - impl['defaults']
    default_ty = impl.get('default_ty', ())
    none_ok = impl.get('default_none_ok') or [False] * len(default_ty)
    fits = [FIT_REQUIRED] * required
    fits += [default_fit(declared, ty, takes_none)
             for declared, ty, takes_none in zip(params[required:], default_ty, none_ok)]
    return ImplSig(params=params, required=required,
                   open_ended=impl.get('vararg') is not None, ret=impl['ret'],
                   fits=''.join(fits), names=tuple(impl.get('names') or ()))


def _arity_fits(impl: dict[str, Any], argc: int) -> bool:
    """Whether an overload implementation can take this many positional arguments."""
    if impl.get('vararg') is not None:
        return True
    params = impl['params']
    if impl.get('kwarg'):
        return len(params) - impl['defaults'] <= argc
    return len(params) - impl['defaults'] <= argc <= len(params)


def _own_calls(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.Call]:
    """
    Every call site of a function's own body, in source order.

    A nested definition owns its own sites, so the walk stops there; a lambda
    does NOT own one, because the isolation pass anchors a lambda's sites in
    the enclosing function's state vector.

    :param node: The definition to walk
    :return: The call nodes, ordered by their pre-order node id
    """
    calls: list[ast.Call] = []
    stack: list[ast.AST] = list(node.body)
    while stack:
        current = stack.pop()
        if isinstance(current, ast.Call):
            calls.append(current)
        if not isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stack.extend(ast.iter_child_nodes(current))
    calls.sort(key=lambda call: node_id(call) or 0)
    return calls


def _walk_own_scope(node: ast.AST):
    """Walk a function body without descending into nested function scopes."""
    stack = list(ast.iter_child_nodes(node))
    while stack:
        current = stack.pop()
        yield current
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(current))


def _enumerated(node: ast.expr) -> ast.expr | None:
    """
    What an ``enumerate(x)`` call iterates, when the expression is one.

    ``for [i, v] in arr`` is the Pine spelling PyneComp emits as
    ``for i, v in enumerate(arr)``, and the element type lives on ``arr``.

    :param node: The expression being iterated
    :return: The enumerated expression, or None
    """
    if not isinstance(node, ast.Call) or len(node.args) != 1:
        return None
    return node.args[0] if _dotted(node.func) == 'enumerate' else None
