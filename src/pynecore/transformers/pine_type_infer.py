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
from collections.abc import Container, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils.stdlib_checker import is_stdlib
from . import pine_type_artifact
from .dynamic_default import is_script_entry
from .node_ids import assign_node_ids, node_id
from .pine_type_rules import (
    INT, FLOAT, BOOL, STR, UNKNOWN, VOID, OBJECT, NUMERIC,
    join, binop_type, unaryop_type, compare_type, annotation_type,
    LIB_TYPE_OVERRIDES, BUILTIN_CALL_TYPES, TY_ATTR, get_ty, set_ty, inherit_ty,
    constant_type, pin_for, get_pins, set_pin, set_pins, set_vector, set_varying,
    overload_result, ImplSig, overload_pick, default_fit, FIT_REQUIRED,
    impl_sig, _param_defaults, _dotted,
)
from .pine_type_table import (
    Analyser, Binding, CallSite, ContextKey, ContextResult, Diag, ExportSig, FuncSig,
    ModuleInterface, PineTypeTable, Unknown, qualify,
)

__all__ = ['infer_module', 'lib_types', 'TY_ATTR', 'get_ty', 'set_ty', 'inherit_ty']

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
        #: Every class name an annotation of this module may name, its own and
        #: its imports' alike -- filled in before anything reads an annotation
        self._classes: frozenset[str] = frozenset()

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

    def _bind(self, name: str, ty: str, node: ast.AST, unknown: Unknown | None = None) -> None:
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
        if existing is None:
            bindings[name] = Binding(name=name, ty=ty, line=line, unknown=unknown)
            return
        joined = join(existing.ty, ty)
        existing.ty = joined
        if joined == UNKNOWN and existing.unknown is None:
            existing.unknown = unknown or self._unknown('joined-branches', node, name)

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
        self._collect(tree.body, '')
        self._body(tree.body)
        self._flush_pending()
        self._stamp_instance_vectors(tree)
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
        Every class name an annotation of this module may name.

        A UDT is a type, so a parameter annotated with one is annotated --
        reading such a name as unknown made the parameter behave like an
        unannotated one and took the whole export out of the typed world for
        its callers. Which names ARE classes cannot be answered from the
        annotation alone: ``Pivot`` is a class here and a stray name there, so
        the set has to be collected before anything reads an annotation.

        The module's own classes come from the tree, whatever scope they are
        declared in -- a forward reference names a class that stands further
        down, and a nested one is nameable from the body it lives in. An
        imported one is resolved through the same interface the calls go
        through, and only for a name some annotation actually spells: an
        import nothing annotates with is not worth an interface lookup.

        The set is FLAT, and deliberately so on two counts. A dotted spelling
        is matched by its leaf, the way ``annotation_type`` matches the
        builtin type names, so ``a.Settings`` and ``b.Settings`` are one name
        here -- two libraries publishing the same class name is not a shape
        Pine produces. And a class declared inside a function is nameable from
        every scope, not only from the body it lives in; scoping it would mean
        threading a per-scope class set through every annotation read, and an
        annotation naming a nested class it cannot see is not a shape Pine
        produces either. What IS excluded is a class name the module binds
        again at module scope (``class Amount: ...`` then ``Amount = int``):
        the name is that binding's, and reading an annotation on it as an
        object would type against a class nothing holds.

        :param tree: The module being walked
        """
        own = self._own_classes(tree)
        imported: set[str] = set()
        for spelled, node in _annotation_names(tree).items():
            parts = spelled.split('.')
            if parts[-1] in own or parts[-1] in imported:
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
            if interface is not None and wanted in interface.classes:
                imported.add(parts[-1])
        self._classes = frozenset(own | imported)
        self.table.classes = self._classes

    def _own_classes(self, tree: ast.Module) -> set[str]:
        """
        The class names this module declares and still holds.

        A name the module binds at module scope by anything but its own
        ``class`` statement is not one: what stands under it at import time is
        the assignment's value, so an annotation naming it describes an object
        that never exists. The comparison is by POSITION, the same way an
        import's own binding is told from a rebinding of it -- the class
        statement is itself a module-scope binding, so nothing else would tell
        the two apart.

        :param tree: The module being walked
        :return: The class names an annotation of this module may name
        """
        declared: dict[str, set[tuple[int, int]]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                declared.setdefault(node.name, set()).add((_line(node), _col(node)))
        return {name for name, positions in declared.items()
                if all(position in positions
                       for position, _ in self._module_rebinds.get(name, ()))}

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
            case ast.AnnAssign():
                declared = annotation_type(stmt.annotation, self._classes)
                if stmt.value is not None:
                    self._expr(stmt.value)
                # An explicit annotation is a DECLARATION: it wins over what
                # the initializer happens to be, the way Pine's `int x = ...`
                # does. That is the whole point of writing one.
                self._store(stmt.target, declared, stmt)
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
                self._store(stmt.target, self._element_type(stmt.iter, iter_ty), stmt)
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
                self._body(stmt.body)
            case _:
                # Import, Pass, Break, Continue, Global, Nonlocal, Delete:
                # nothing to type
                for child in ast.iter_child_nodes(stmt):
                    if isinstance(child, ast.expr):
                        self._expr(child)

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
            bindings[arg.arg] = Binding(name=arg.arg, ty=ty, line=_line(arg), unknown=unknown)

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
                                       unknown=binding.unknown)
                continue
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
        """Join every ``return`` in a function body, ignoring nested functions."""
        result: str | None = None
        for stmt in _walk_own_scope(node):
            if isinstance(stmt, ast.Return):
                ty = VOID if stmt.value is None else self._ty_of(stmt.value)
                result = ty if result is None else join(result, ty)
        return VOID if result is None else result

    def _store(self, target: ast.expr, ty: str, source: ast.AST) -> None:
        """Bind an assignment target, recursing into tuple/list targets."""
        if isinstance(target, ast.Name):
            self._stamp(target, ty)
            unknown = self._unknown('unknown-value', source) if ty == UNKNOWN else None
            self._bind(target.id, ty, source, unknown)
        elif isinstance(target, (ast.Tuple, ast.List)):
            self._stamp(target, OBJECT)
            for element in target.elts:
                # A destructured element's own type is not modeled: the tuple
                # shapes Pine has (``request.security`` tuples) are opaque here
                self._store(element, UNKNOWN, source)
        elif isinstance(target, (ast.Attribute, ast.Subscript)):
            self._expr(target.value)
            self._stamp(target, ty)

    def _target_type(self, target: ast.expr) -> str:
        """Current type of an augmented-assignment target."""
        if isinstance(target, ast.Name):
            found = self._lookup(target.id)
            return found.ty if found is not None else UNKNOWN
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
        """
        if isinstance(iter_node, ast.Call):
            callee = _dotted(iter_node.func)
            if callee in ('range', 'pine_range', 'lib.pine_range'):
                bounds = [self._ty_of(a) for a in iter_node.args]
                if bounds and all(b in NUMERIC for b in bounds):
                    return INT if all(b == INT for b in bounds) else FLOAT
                return UNKNOWN
        return UNKNOWN if iter_ty != OBJECT else UNKNOWN

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
        found = self._lookup(node.id)
        if found is not None:
            return found.ty
        entry = lib_types().get(node.id)
        if entry is not None and entry.get('kind') == 'value':
            return entry['ty']
        return UNKNOWN

    def _e_Attribute(self, node: ast.Attribute) -> str:
        self._expr(node.value)
        name = self._lib_name(node)
        if name is None:
            return UNKNOWN
        override = LIB_TYPE_OVERRIDES.get(name)
        if isinstance(override, str) and len(override) == 1:
            return override
        entry = lib_types().get(name)
        if entry is None:
            return UNKNOWN
        if entry['kind'] == 'value':
            return entry['ty']
        # A bare reference to a lib function is the function itself
        return OBJECT

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
        return join(self._expr(node.body), self._expr(node.orelse))

    def _e_NamedExpr(self, node: ast.NamedExpr) -> str:
        ty = self._expr(node.value)
        self._store(node.target, ty, node)
        return ty

    def _e_Subscript(self, node: ast.Subscript) -> str:
        base = self._expr(node.value)
        self._expr(node.slice)
        # MEASURED: ``d[1]`` on an int-typed ``d`` is int -- the history index
        # is type-preserving, it reads the same series one bar back
        return base

    def _e_Tuple(self, node: ast.Tuple) -> str:
        for element in node.elts:
            self._expr(element)
        return OBJECT

    _e_List = _e_Tuple
    _e_Set = _e_Tuple

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
        pin = self._pin(node, isinstance(entry, dict) and entry.get('kind') == 'overloads')
        ty = self._lib_call_type(callee, node, argc, pin)
        self.table.calls.append(CallSite(
            callee=callee, line=_line(node), col=_col(node), argc=argc, ty=ty, pin=pin))
        return ty

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

    def _lib_call_type(self, callee: str, node: ast.Call, argc: int | None,
                       pin: str | None = None) -> str:
        """
        Result type of a call to a lib name.

        The measured override wins over the annotation: the lib annotates
        ``math.round`` as a float because that is what Python returns, while
        TradingView types the one-argument form as an int.

        :param callee: The registry key the call resolves to
        :param node: The call node
        :param argc: How many arguments it passes, or None when an unpacking
                     hides the count
        :param pin: The overload pin this call site justified, when it has one
        :return: The type the call evaluates to
        """
        entry = lib_types().get(callee)
        override = LIB_TYPE_OVERRIDES.get(callee)
        if override is not None:
            names = entry.get('names') if isinstance(entry, dict) else None
            resolved = self._apply_override(override, node, argc, names)
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

    def _apply_override(self, override: Any, node: ast.Call, argc: int | None,
                        param_names: list[str] | None = None) -> str | None:
        """
        Resolve one entry of the measured override table.

        ``param_names`` is the callee's declared parameter order, which is what
        turns a keyword spelling back into a position; without it only the
        positional arguments can be addressed.
        """
        if isinstance(override, dict):
            if argc is None:
                return UNKNOWN
            picked = override.get(argc)
            return None if picked is None else self._apply_override(
                picked, node, argc, param_names)
        if not isinstance(override, str):
            return None
        if override == 'all_int':
            # Every argument counts, however it was spelled: ``math.max`` is
            # int-typed exactly when all of them are -- and an unpacking hides
            # some of them, so there is nothing to decide on
            if argc is None:
                return UNKNOWN
            passed = [self._ty_of(a) for a in node.args]
            passed += [self._ty_of(k.value) for k in node.keywords if k.arg is not None]
            if not passed or any(t not in NUMERIC for t in passed):
                return UNKNOWN if passed else None
            return INT if all(t == INT for t in passed) else FLOAT
        if override.startswith('arg') and override[3:].isdigit():
            argument = _bound_arg(node, int(override[3:]), param_names)
            return UNKNOWN if argument is None else self._ty_of(argument)
        return override

    def _user_call(self, node: ast.Call) -> str:
        """
        Result type of a call to a function this module defines or imports.

        A definition of this module wins, with its own shadowing rules. Only
        when there is none does the import map speak -- the imported callee is
        then typed from its DECLARED signature, never from this call site.
        """
        name = _dotted(node.func) or ''
        resolved = self._resolve_func(name, node)
        if resolved is None:
            imported = self._imported_call(node, name)
            if imported is not None:
                return imported
            # A module function shadows the builtin of the same name, so the
            # builtins are only consulted once the module has no such name
            builtin = BUILTIN_CALL_TYPES.get(name)
            if builtin is not None and isinstance(node.func, ast.Name):
                fallback = self._apply_override(builtin, node, _call_arity(node))
                if fallback is not None:
                    return fallback
            return UNKNOWN

        key, frame, shadowed = resolved
        if shadowed:
            self._diag(
                f"'{name}' is assigned as well as defined, so what it calls is unknown",
                node, self._unknown('rebound-name', node, name),
                fix=f"call '{name}' under a name nothing assigns to")
            return UNKNOWN
        is_group = key in self._overload_groups
        pin = self._pin(node, is_group)
        if is_group:
            # A group is per-signature already; the pin is what selects among
            # its implementations, so there is no context to instantiate
            self._ensure_group(key, frame)
            ty = self._group_type(key, pin)
        else:
            ty = self._call_context(key, node, frame)
        self.table.calls.append(CallSite(
            callee=name, line=_line(node), col=_col(node),
            argc=_call_arity(node), ty=ty, pin=pin))
        return ty

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

    def _call_context(self, key: str, node: ast.Call, frame: int) -> str:
        """
        Analyse the callee in the context this call site describes, and type it.

        :param key: Scope-qualified id of the callee
        :param node: The call node
        :param frame: Index of the frame whose scope declares the callee
        :return: The type this call evaluates to
        """
        definitions = self._defs.get(key)
        if not definitions:
            return self.table.funcs[key].ret
        # A redefined name is the LAST definition, the way Python binds it
        target = definitions[-1]
        parents = self._frames[:frame + 1]
        params = self._call_params(target, node)
        if params is None:
            # A shape this analysis does not describe -- an unpacking, an
            # unknown keyword, an arity that does not fit. The callee still
            # gets the one context its own definition states, so the call is
            # typed by whatever the body says with UNKNOWN parameters
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
        if nid is not None:
            # Only the RESOLVED path is recorded. The fallbacks above hand the
            # callee its own declaration context instead of the one this site
            # describes, so a vector derived from them would configure an
            # instance the call site never actually establishes
            self._call_ctx.setdefault(nid, {})[self._context] = result.cid
            self._callee_key[nid] = key
        return result.ret

    def _call_params(self, target: ast.FunctionDef | ast.AsyncFunctionDef,
                     node: ast.Call) -> tuple[str, ...] | None:
        """
        The parameter types one call site instantiates the callee with.

        MEASURED on TradingView: the type of a parameter at a call site is
        JOIN(type of its default, type of the argument) -- a float argument to
        ``f(x = 0)`` makes ``x`` float, and an int argument to ``h(x = 0.0)``
        makes it float too. An annotation still outranks both, and an omitted
        argument leaves the default alone as the value that IS passed.

        :param target: The callee's definition
        :param node: The call node
        :return: One type per parameter, or None when the shape is unresolvable
        """
        args = target.args
        if args.vararg is not None or args.kwarg is not None:
            return None
        if any(isinstance(a, ast.Starred) for a in node.args):
            return None
        if any(k.arg is None for k in node.keywords):
            return None
        positional = list(args.posonlyargs) + list(args.args)
        if len(node.args) > len(positional):
            return None

        bound: dict[str, ast.expr] = {}
        for arg, value in zip(positional, node.args):
            bound[arg.arg] = value
        declared_names = {a.arg for a in positional + list(args.kwonlyargs)}
        for keyword in node.keywords:
            if keyword.arg not in declared_names or keyword.arg in bound:
                return None
            bound[keyword.arg] = keyword.value

        defaults = _param_defaults(target)
        out: list[str] = []
        for arg in _every_param(target):
            passed = bound.get(arg.arg)
            default = defaults.get(arg.arg)
            if passed is None and default is None:
                return None
            annotated = annotation_type(arg.annotation, self._classes)
            if annotated != UNKNOWN:
                out.append(annotated)
            elif passed is None:
                out.append(self._ty_of(default))
            elif default is None:
                out.append(self._ty_of(passed))
            else:
                out.append(join(self._ty_of(default), self._ty_of(passed)))
        return tuple(out)

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
        if pine_type_artifact.analysing(path):
            self._diag(
                f"'{dotted}' imports this module back, so its signatures are not "
                f"available yet", node, self._unknown('import-cycle', node, dotted),
                fix=f'break the import cycle between {self.table.module_path} and {path}')
            return None
        if path not in self._interfaces:
            interface = pine_type_artifact.lookup(path, self._analyser, self._pipeline_hash)
            self._interfaces[path] = interface
            if interface is not None:
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
    if any(isinstance(a, ast.Starred) for a in node.args):
        return None
    if any(k.arg is None for k in node.keywords):
        return None
    return len(node.args) + len(node.keywords)


def _bound_arg(node: ast.Call, index: int, param_names: list[str] | None) -> ast.expr | None:
    """
    The expression bound to one declared parameter position.

    A type-preserving override names the parameter it copies from, and Python
    lets the caller spell that parameter either way, so the keywords have to be
    bound back to their declared position before the position can be read. An
    unpacking hides which position an argument landed on, and is unresolvable.

    :param node: The call node
    :param index: Declared parameter position, 0-based
    :param param_names: The callee's declared parameter order, when it is known
    :return: The bound expression, or None when it cannot be determined
    """
    if any(isinstance(a, ast.Starred) for a in node.args):
        return None
    if index < len(node.args):
        return node.args[index]
    if param_names is None or index >= len(param_names):
        return None
    wanted = param_names[index]
    for keyword in node.keywords:
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
                   fits=''.join(fits))


def _arity_fits(impl: dict[str, Any], argc: int) -> bool:
    """Whether an overload implementation can take this many positional arguments."""
    if impl.get('vararg') is not None:
        return True
    params = impl['params']
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
