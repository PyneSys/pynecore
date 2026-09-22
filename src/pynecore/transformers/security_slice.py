"""Per-context backward slicing of ``main()`` for ``request.security`` children.

A security child process re-runs the script's whole ``main()`` on every bar of
its own feed, even though only one ``__sec_write__`` block of it produces the
value the chart waits for. This pass answers the obvious question — which of
``main()``'s statements can that write possibly depend on? — and emits the
answer as an ordinary module-level function, one per static security context::

    def __sec_main_0__():
        <the backward slice of main() for sec·<hash>·0>

Contexts the security transformer put in the same ``group`` — those resolving to
one and the same feed — share ONE clone: it keeps every member's write block and
every member's reads, so a single child can serve all of them. The number in the
clone's name is the position of the context it was built for, and for a group
the position of its first member; a clone name therefore belongs to exactly one
unit.

The clone is recorded in ``__security_contexts__[sid]['slice_main']`` of every
context it serves; the child runs it instead of ``main()`` (see
``core/security_process.py``). Chart and child
load the SAME bytecode, so the clones must live in the same module as ``main``
— a child-only transform is impossible.

Placement in the pipeline: after :class:`~pynecore.transformers.security.
SecurityTransformer` (which assigns the positional sec ids, so nothing may be
dropped before it) and before the lowering half, so the clones are laid out by
the series / persistent / isolation passes as ordinary functions with their own
slot layout.

The slice is a conservative over-approximation: every uncertainty KEEPS a
statement, and a construct the pass cannot classify at all drops the whole
optimization for the module (the child then runs ``main()`` as before).
``PYNE_NO_SECURITY_SLICE=1`` switches the slicing off; the flag is mixed into
the transform pipeline digest, so chart and child can never load bytecode built
under the other setting.

A clone is also where a guarded write is made unconditional
(:func:`_force_conditional_writes`). That is a matter of what the child
computes, not of how much it may skip, so a context the slice does not serve —
an LTF one, one of a module the analysis gave up on, any of them while the
slicing is switched off — still gets a clone whenever its write is guarded: one
of the whole ``main()``, with nothing dropped.
"""
import ast
import copy
from collections.abc import Callable

from ..core.import_hook import PYNE_RESERVED_NAME_CHAR, security_slice_disabled
from .dynamic_default import is_script_entry
from .persistent import VARIP_TYPES
from .pine_type_rules import OBJECT, STR, stamp_lowering
from .security import (
    _COLLECTION_NAMESPACES, _COLLECTION_READERS, _DependencyAnalyzer, _UNMODELLED_NODES,
)

__all__ = ['SecuritySliceTransformer', 'CLONE_PREFIX', 'CLONE_SUFFIX']

#: Name shape of an emitted clone. The security protocol's own injected names
#: (``__sec_write__``, ``__sec_read__``, ...) share this namespace; a script
#: spelling one of them would already be broken today.
CLONE_PREFIX = '__sec_main_'
CLONE_SUFFIX = '__'

#: Pseudo-name standing for state the pass cannot see: another module's
#: globals, an unresolvable call. A statement that touches it both reads and
#: writes it, so keeping any one of them keeps them all.
_EXTERNAL = '\x00external'

#: Statements that can end ``main()`` before a later statement is reached. One
#: of them is kept unconditionally: dropping it would let the write block run
#: on a bar where ``main()`` returned ahead of it.
_EXIT_NODES: tuple[type[ast.AST], ...] = (ast.Return,)

#: Statements that can keep a later write from being performed. A ``raise`` is
#: one of them next to the slicer's exits: whether it fires is decided by the
#: developing bar exactly like an early ``return``.
_WRITE_SKIPPING_NODES: tuple[type[ast.AST], ...] = (*_EXIT_NODES, ast.Raise)

#: The protocol calls the security split injects. None of them touches a name
#: of the script, so they carry no data dependency of their own.
_PROTOCOL_CALLS = frozenset({
    '__sec_read__', '__sec_write__', '__sec_signal__', '__sec_wait__',
    '__ltf_unzip__',
})

#: Calls the earlier passes emit that stand for a plain expression: they run
#: no script code of their own, so an unresolved one is not opaque.
_MARKER_CALLS = frozenset({'inline_series'})

#: Fixpoint safety net for the call-graph summaries; the lattice is finite and
#: monotone, so the loop always converges well below this.
_MAX_ROUNDS = 1000


class _SliceScopes(_DependencyAnalyzer):
    """Scope tree and name resolution of the dependency analyzer, alone.

    Only the lexical half of :class:`_DependencyAnalyzer` is wanted here — the
    scope tree, the qualification of a bare name to the scope that binds it and
    the resolution of a call to the function it names. Subclassing gets all
    three without duplicating them and without running (or touching) the taint
    fixpoint.
    """

    def __init__(self, module: ast.Module):
        super().__init__(module, [])
        self._funcs = {
            n.name: n for n in ast.walk(module)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self._build_scopes()

    def qualify(self, name: str, scope_key: str) -> str:
        """Qualify ``name`` with the key of the scope that binds it."""
        return self._q(name, scope_key)

    def func_key(self, name: str, scope_key: str) -> str | None:
        """Scope key of the user function ``name`` resolves to in ``scope_key``."""
        self._scope = scope_key
        return self._lookup_func(name)

    def scope_key_of(self, node: ast.AST) -> str | None:
        """Scope key of a function definition node."""
        return self._key_of.get(id(node))

    def scope_body(self, key: str) -> list[ast.stmt]:
        """Statement list of the scope."""
        return self._scopes[key].body

    def scope_node(self, key: str) -> ast.AST:
        """Defining node of the scope."""
        return self._scopes[key].node

    @staticmethod
    def root_name(node: ast.expr) -> str | None:
        """Base ``Name`` of a place expression (``a``, ``a.b``, ``a[i].c``)."""
        return _DependencyAnalyzer._root_name(node)


class _StmtIndex:
    """Per top-level statement of ``main()``, the facts every slice needs.

    All of it is independent of the context being sliced for, so it is built
    once and every context's fixpoint reads it instead of re-walking ``main()``.

    :ivar reads: alias classes the statement reads
    :ivar defs: alias classes it writes or may mutate
    :ivar read_sids: sids whose ``__sec_read__`` the statement holds
    :ivar write_sids: sids whose ``__sec_write__`` the statement holds
    :ivar mandatory: a definition or an early exit — kept in every slice
    """

    __slots__ = ('reads', 'defs', 'read_sids', 'write_sids', 'mandatory')

    def __init__(self) -> None:
        self.reads: set[str] = set()
        self.defs: set[str] = set()
        self.read_sids: set[str] = set()
        self.write_sids: set[str] = set()
        self.mandatory: bool = False


class _Facts:
    """What one statement (or one function body) does to the names around it.

    :ivar reads: qualified names whose VALUE the code reads
    :ivar writes: qualified names the code rebinds
    :ivar mutates: qualified names whose object the code may mutate in place
    :ivar calls: scope keys of the user functions the code calls
    :ivar read_sids: sids the code reads through ``__sec_read__``
    :ivar write_sids: sids the code writes through ``__sec_write__``
    """

    __slots__ = ('reads', 'writes', 'mutates', 'calls', 'read_sids', 'write_sids')

    def __init__(self) -> None:
        self.reads: set[str] = set()
        self.writes: set[str] = set()
        self.mutates: set[str] = set()
        self.calls: set[str] = set()
        self.read_sids: set[str] = set()
        self.write_sids: set[str] = set()

    def absorb(self, other: '_Facts') -> bool:
        """Merge another summary's name and sid sets in; True when anything grew."""
        before = (len(self.reads), len(self.writes), len(self.mutates),
                  len(self.read_sids), len(self.write_sids))
        self.reads |= other.reads
        self.writes |= other.writes
        self.mutates |= other.mutates
        self.read_sids |= other.read_sids
        self.write_sids |= other.write_sids
        return before != (len(self.reads), len(self.writes), len(self.mutates),
                          len(self.read_sids), len(self.write_sids))

    def external(self) -> None:
        """Mark the code as touching state the pass cannot model."""
        self.reads.add(_EXTERNAL)
        self.writes.add(_EXTERNAL)
        self.mutates.add(_EXTERNAL)


class _Aliases:
    """Union-find over qualified names that may denote the SAME object.

    ``b = a`` (and every call that hands one of its arguments back) puts the
    two names in one class, so a mutation written through either alias counts
    as a mutation of both.
    """

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, name: str) -> str:
        root = name
        while self._parent.get(root, root) != root:
            root = self._parent[root]
        while self._parent.get(name, name) != name:
            self._parent[name], name = root, self._parent[name]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra

    def classes(self, names: set[str]) -> set[str]:
        """The alias classes the names belong to."""
        return {self.find(name) for name in names}


def _is_lib_chain(node: ast.expr) -> bool:
    """Whether an expression is an attribute chain rooted at ``lib``."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name) and node.id == 'lib'


def _collection_call(func: ast.Attribute) -> tuple[str, str] | None:
    """``(namespace, function)`` of a ``lib.<array|matrix|map>.<fn>`` callee."""
    owner = func.value
    if (isinstance(owner, ast.Attribute)
            and owner.attr in _COLLECTION_NAMESPACES
            and isinstance(owner.value, ast.Name)
            and owner.value.id == 'lib'):
        return owner.attr, func.attr
    return None


def _collection_target(node: ast.Call) -> ast.expr | None:
    """The collection a ``lib.<array|matrix|map>.<fn>`` call acts on.

    Every collection builtin takes it as its first parameter, named ``id``, so
    a keyword call names the very same object a positional one passes. An
    unpacked positional (``array.set(*args)``) names nothing at all.
    """
    if node.args:
        first = node.args[0]
        return None if isinstance(first, ast.Starred) else first
    for kw in node.keywords:
        if kw.arg == 'id':
            return kw.value
    return None


def _protocol_sid(node: ast.AST, name: str) -> str | None:
    """Sid argument of a ``__sec_read__`` / ``__sec_write__`` call node."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == name and node.args):
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def _walk_own(node: ast.AST):
    """Every descendant of ``node``, not entering nested function definitions."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        yield child
        yield from _walk_own(child)


class _Collector:
    """Builds :class:`_Facts` and alias classes for the statements of a scope."""

    def __init__(self, scopes: _SliceScopes, aliases: _Aliases,
                 import_names: frozenset[str]):
        self._scopes = scopes
        self._aliases = aliases
        self._imports = import_names
        self._returns: dict[str, list[str]] = {}
        self._returning: set[str] = set()
        # Set when a mutation was seen that joins no alias class, so no slice of
        # the module may drop statements (see :attr:`module_skip`).
        self.unresolved: str | None = None

    # --- aliasing ---

    def alias_roots(self, expr: ast.expr | None, scope: str) -> list[str]:
        """Qualified names ``expr`` may evaluate to an alias of."""
        if expr is None:
            return []
        if isinstance(expr, (ast.Name, ast.Attribute, ast.Subscript)):
            root = self._scopes.root_name(expr)
            if root is None or root == 'lib':
                return []
            return [self._scopes.qualify(root, scope)]
        if isinstance(expr, ast.IfExp):
            return (self.alias_roots(expr.body, scope)
                    + self.alias_roots(expr.orelse, scope))
        if isinstance(expr, ast.BoolOp):
            roots: list[str] = []
            for value in expr.values:
                roots.extend(self.alias_roots(value, scope))
            return roots
        if not isinstance(expr, ast.Call):
            return []
        func = expr.func
        if isinstance(func, ast.Attribute):
            collection = _collection_call(func)
            if collection is not None:
                # A reader does not hand the collection itself back, but it may
                # hand back an element of it (``array.get`` of a UDT) or a view
                # sharing its storage, and a mutation written through that
                # element is a mutation of the collection. A constructor keeps
                # the objects it was seeded with (``array.new(size, item)``,
                # ``array.from_items(a, b)``), so its result shares their
                # storage too. Every argument is therefore taken as alias
                # provenance: a scalar index or size contributes nothing, and
                # widening a class only ever keeps more statements.
                roots = []
                for arg in list(expr.args) + [kw.value for kw in expr.keywords]:
                    roots.extend(self.alias_roots(arg, scope))
                return roots
            if _is_lib_chain(func):
                # Every other ``lib`` builtin returns a fresh value.
                return []
        # A user function or an unknown callee may return one of its arguments.
        roots = []
        for arg in list(expr.args) + [kw.value for kw in expr.keywords]:
            roots.extend(self.alias_roots(arg, scope))
        # It may also return an object of an enclosing scope it never got as an
        # argument (a module-level collection built once and handed out by a
        # getter), which no argument of the call names.
        if isinstance(func, ast.Name):
            key = self._scopes.func_key(func.id, scope)
            if key is not None:
                roots.extend(self._returned_roots(key))
        return roots

    def _returned_roots(self, key: str) -> list[str]:
        """Qualified names the user function ``key`` may hand back.

        The callee's ``return`` expressions are resolved in the callee's own
        scope, so a name it closes over or reads from module scope qualifies to
        the very object the caller's other statements mutate. A recursive
        callee is answered with :data:`_EXTERNAL`: the recursion cannot be
        unrolled here, and joining the external class keeps every statement
        that could reach the object.
        """
        cached = self._returns.get(key)
        if cached is not None:
            return cached
        if key in self._returning:
            return [_EXTERNAL]
        self._returning.add(key)
        roots: list[str] = []
        for sub in _walk_own(self._scopes.scope_node(key)):
            if isinstance(sub, ast.Return) and sub.value is not None:
                roots.extend(self.alias_roots(sub.value, key))
        self._returning.discard(key)
        self._returns[key] = roots
        return roots

    def record_aliases(self, scope: str) -> None:
        """Union every ``target = <alias expression>`` of the scope."""
        node = self._scopes.scope_node(scope)
        for sub in [node, *_walk_own(node)]:
            if isinstance(sub, ast.Call):
                self._record_call_aliases(sub, scope)
            targets: list[ast.expr] = []
            value: ast.expr | None = None
            if isinstance(sub, ast.Assign):
                targets, value = list(sub.targets), sub.value
            elif isinstance(sub, ast.AnnAssign) and sub.value is not None:
                targets, value = [sub.target], sub.value
            elif isinstance(sub, (ast.For, ast.AsyncFor)):
                targets, value = [sub.target], sub.iter
            if value is None:
                continue
            roots = self.alias_roots(value, scope)
            if not roots:
                continue
            for target in targets:
                name = self._scopes.root_name(target)
                if name is None or name == 'lib':
                    continue
                qualified = self._scopes.qualify(name, scope)
                for root in roots:
                    self._aliases.union(qualified, root)

    def _record_call_aliases(self, node: ast.Call, scope: str) -> None:
        """Join the objects a call may store into one another.

        ``array.push(store, item)`` puts ``item`` inside ``store`` without
        binding any name, so a later ``array.get(store, 0).acc = x`` mutates
        ``item`` as well. A user or unknown callee may keep one argument inside
        another the same way, and a method call may keep an argument inside its
        receiver. Joining the argument classes is conservative: a wider class
        only ever keeps more statements.
        """
        func = node.func
        args: list[ast.expr] = list(node.args) + [kw.value for kw in node.keywords]
        if isinstance(func, ast.Attribute):
            collection = _collection_call(func)
            if collection is not None:
                namespace, fname = collection
                if fname in _COLLECTION_READERS[namespace]:
                    return
            elif _is_lib_chain(func):
                # Every other ``lib`` builtin keeps nothing of its arguments.
                return
            else:
                root = self._scopes.root_name(func)
                if root is not None and root != 'lib':
                    args.append(func.value)
        roots: list[str] = []
        for arg in args:
            roots.extend(self.alias_roots(arg, scope))
        for other in roots[1:]:
            self._aliases.union(roots[0], other)

    # --- facts ---

    def facts(self, nodes: list[ast.AST], scope: str) -> _Facts:
        """Collect what the given nodes do, not entering nested definitions."""
        facts = _Facts()
        for top in nodes:
            for sub in [top, *_walk_own(top)]:
                self._visit(sub, scope, facts)
        return facts

    def _visit(self, node: ast.AST, scope: str, facts: _Facts) -> None:
        if isinstance(node, ast.Name):
            qualified = self._scopes.qualify(node.id, scope)
            if node.id == 'lib':
                return
            if isinstance(node.ctx, ast.Store):
                facts.writes.add(qualified)
            elif isinstance(node.ctx, ast.Del):
                facts.writes.add(qualified)
            else:
                facts.reads.add(qualified)
            return
        if isinstance(node, (ast.Attribute, ast.Subscript)):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                root = self._scopes.root_name(node)
                if root is not None and root != 'lib':
                    qualified = self._scopes.qualify(root, scope)
                    facts.reads.add(qualified)
                    facts.mutates.add(qualified)
                elif root == 'lib':
                    # Writing onto the library namespace is global state.
                    facts.external()
                else:
                    # The object written to is produced by an expression
                    # (``array.get(store, 0).acc = x``): the store reaches
                    # whatever that expression is an alias of.
                    base = node
                    while isinstance(base, (ast.Attribute, ast.Subscript)):
                        base = base.value
                    roots = self.alias_roots(base, scope)
                    if roots:
                        facts.reads.update(roots)
                        facts.mutates.update(roots)
                    else:
                        self.unresolved = f'unresolved_store:{scope or "<module>"}'
            return
        if isinstance(node, ast.AugAssign):
            root = self._scopes.root_name(node.target)
            if root is not None and root != 'lib':
                facts.reads.add(self._scopes.qualify(root, scope))
            return
        if isinstance(node, ast.Call):
            self._visit_call(node, scope, facts)
            return
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                facts.writes.add(self._scopes.qualify(name, scope))
            return

    def _visit_call(self, node: ast.Call, scope: str, facts: _Facts) -> None:
        func = node.func
        args = list(node.args) + [kw.value for kw in node.keywords]
        if isinstance(func, ast.Name):
            key = self._scopes.func_key(func.id, scope)
            if key is not None:
                facts.calls.add(key)
                for arg in args:
                    facts.mutates.update(self.alias_roots(arg, scope))
                return
            if func.id in _PROTOCOL_CALLS:
                # A protocol call belongs to the context named by its first
                # argument, not to any name of the script.
                sid = _protocol_sid(node, '__sec_read__')
                if sid is not None:
                    facts.read_sids.add(sid)
                sid = _protocol_sid(node, '__sec_write__')
                if sid is not None:
                    facts.write_sids.add(sid)
                return
            if func.id in self._imports:
                facts.external()
                return
            # An unresolvable callee: assume it does everything.
            facts.external()
            for arg in args:
                facts.mutates.update(self.alias_roots(arg, scope))
            return
        if isinstance(func, ast.Attribute):
            collection = _collection_call(func)
            if collection is not None:
                namespace, fname = collection
                if fname not in _COLLECTION_READERS[namespace]:
                    target = _collection_target(node)
                    # A mutator whose collection argument cannot be named
                    # (unpacking) is not modelled at all: the module was given
                    # up before this point (see :func:`_unresolved_collection`).
                    if target is not None:
                        facts.mutates.update(self.alias_roots(target, scope))
                return
            if _is_lib_chain(func):
                return
            root = self._scopes.root_name(func)
            if root is not None and root in self._imports:
                # A call into another module: its globals are invisible here.
                facts.external()
                for arg in args:
                    facts.mutates.update(self.alias_roots(arg, scope))
                return
            if root is not None:
                # A method call on an object of the script's own.
                facts.mutates.add(self._scopes.qualify(root, scope))
                return
        facts.external()
        for arg in args:
            facts.mutates.update(self.alias_roots(arg, scope))


class SecuritySliceTransformer(ast.NodeTransformer):
    """Emit a backward-sliced ``main()`` clone per security context.

    :ivar module_skip: why no context of the module could be sliced, if so
    :ivar skipped: per sid, why that context got no clone at all
    """

    def __init__(self) -> None:
        self.module_skip: str | None = None
        self.skipped: dict[str, str] = {}

    def visit_Module(self, node: ast.Module) -> ast.Module:
        sliced = self._emit_clones(node)
        contexts = _find_contexts(node)
        if contexts is not None:
            _finalize_closed_shift(node, contexts, sliced)
        return node

    def _emit_clones(self, node: ast.Module) -> dict[str, list[ast.stmt]]:
        """Emit the clones and record what each served context's child runs.

        :param node: the lowered module
        :return: per sid, the statements of the clone its child runs (absent for
            a context that got no clone — its child runs the whole ``main()``)
        """
        contexts = _find_contexts(node)
        main = _find_main(node)
        if contexts is None or main is None:
            self.module_skip = 'no_contexts' if main is not None else 'no_main'
            return {}
        entries = list(enumerate(_iter_contexts(contexts)))

        # A sliced clone per unit the analysis can serve, then a clone of the
        # WHOLE ``main()`` for every other unit with a guarded write: forcing
        # the write is a matter of what the child computes, not of how much of
        # ``main()`` it may skip, so it cannot depend on the slice succeeding.
        plans = self._sliced_units(node, main, entries)
        covered = {sid for unit, _kept in plans for _index, sid, _ctx in unit}
        whole = list(range(len(main.body)))
        plans.extend((unit, None) for unit in _whole_units(entries, covered))

        clones: list[ast.stmt] = []
        sliced: dict[str, list[ast.stmt]] = {}
        for unit, kept in plans:
            # The unit's first member names the clone: its index is its own,
            # so no other unit can pick the same name.
            name = f'{CLONE_PREFIX}{unit[0][0]}{CLONE_SUFFIX}'
            clone = _build_clone(node, main, whole if kept is None else kept, name)
            guarded = {sid for _index, sid, ctx in unit
                       if _ctx_get(ctx, 'ohlcv_fields') is None
                       and not _write_reached(clone.body, sid)}
            forced = bool(guarded) and _force_conditional_writes(node, clone, guarded)
            if kept is None and not forced:
                # Nothing to gain: the clone would be ``main()`` itself.
                continue
            clones.append(clone)
            clones.extend(_bind_defaults(main.name, name))
            for _index, _sid, ctx in unit:
                sliced[_sid] = clone.body
                if kept is None:
                    self.skipped.pop(_sid, None)
                # The type pass has already run, so the two new literals are
                # stamped here — an unstamped node inside a stamped subtree
                # loses the type of everything above it.
                ctx.keys.append(stamp_lowering(ast.Constant(value='slice_main'), STR))
                ctx.values.append(stamp_lowering(ast.Constant(value=name), STR))

        if clones:
            insert_at = node.body.index(main) + 1
            node.body[insert_at:insert_at] = clones
        return sliced

    def _sliced_units(self, node: ast.Module, main: ast.FunctionDef,
                      entries: list[tuple[int, tuple[str, ast.Dict]]],
                      ) -> list[tuple[list[tuple[int, str, ast.Dict]], list[int] | None]]:
        """The units a backward slice serves, each with the statements it keeps.

        :param node: the lowered module
        :param main: the script's ``main()``
        :param entries: ``(index, (sid, context dict))`` in context order
        :return: ``(unit, kept indexes)`` per sliced unit; empty when the module
            cannot be sliced at all (see :attr:`module_skip`)
        """
        if security_slice_disabled():
            self.module_skip = 'disabled'
            return []
        scopes = _SliceScopes(node)
        main_key = scopes.scope_key_of(main)
        if main_key is None:
            self.module_skip = 'no_main_scope'
            return []
        reachable = _reachable_scopes(scopes, main_key)
        for key in reachable:
            found = _unmodelled_kind(scopes.scope_node(key))
            if found is not None:
                self.module_skip = f'unmodelled:{found}:{key or "<module>"}'
                return []
            if _unresolved_collection(scopes.scope_node(key)):
                self.module_skip = f'unresolved_collection:{key or "<module>"}'
                return []

        import_names = _module_imports(node)
        aliases = _Aliases()
        collector = _Collector(scopes, aliases, import_names)
        # Module level binds names too (``store = array.new_float(1)`` followed
        # by ``alias = store``), and both names denote the same object at
        # runtime, so its bindings join the alias classes as well.
        for key in ['', *reachable]:
            collector.record_aliases(key)
        summaries = _function_summaries(collector, scopes, reachable, main_key)

        index_of_stmt = _build_index(main.body, collector, aliases, summaries, main_key)
        if collector.unresolved is not None:
            self.module_skip = collector.unresolved
            return []
        written_sids: set[str] = set()
        for entry in index_of_stmt:
            written_sids |= entry.write_sids

        plans: list[tuple[list[tuple[int, str, ast.Dict]], list[int] | None]] = []
        for unit in _slice_units(entries, written_sids, self.skipped):
            keep_reads: set[str] = set()
            for _index, sid, ctx in unit:
                keep_reads |= {sid} | _sid_list(ctx, 'depends') | _sid_list(ctx, 'late_reads')
            write_sids = {sid for _index, sid, _ctx in unit}
            kept = _slice(index_of_stmt, write_sids, keep_reads)
            if kept is None:
                for _index, sid, _ctx in unit:
                    self.skipped[sid] = 'nothing_dropped'
                continue
            plans.append((unit, kept))
        return plans


# --- module inspection ---


def _find_main(module: ast.Module) -> ast.FunctionDef | None:
    """The script's ``main`` entry point, if the module has one."""
    for stmt in module.body:
        if (isinstance(stmt, ast.FunctionDef) and stmt.name == 'main'
                and is_script_entry(stmt)):
            return stmt
    return None


def _find_contexts(module: ast.Module) -> ast.Dict | None:
    """The ``__security_contexts__`` dict literal, if the module has one."""
    for stmt in module.body:
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.value, ast.Dict)):
            continue
        target = stmt.targets[0]
        if isinstance(target, ast.Name) and target.id == '__security_contexts__':
            return stmt.value
    return None


def _iter_contexts(contexts: ast.Dict):
    """Yield ``(sid, context dict literal)`` pairs of ``__security_contexts__``."""
    for key, value in zip(contexts.keys, contexts.values):
        if (isinstance(key, ast.Constant) and isinstance(key.value, str)
                and isinstance(value, ast.Dict)):
            yield key.value, value


def _ctx_get(ctx: ast.Dict, name: str) -> ast.expr | None:
    """One entry of a context dict literal."""
    for key, value in zip(ctx.keys, ctx.values):
        if isinstance(key, ast.Constant) and key.value == name:
            return value
    return None


def _sid_list(ctx: ast.Dict, name: str) -> set[str]:
    """A context's ``depends`` / ``late_reads`` sid list."""
    value = _ctx_get(ctx, name)
    if not isinstance(value, ast.List):
        return set()
    sids: set[str] = set()
    for elt in value.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            sids.add(elt.value)
    return sids


def _module_imports(module: ast.Module) -> frozenset[str]:
    """Names bound by the module's imports (every call through one is opaque)."""
    names: set[str] = set()
    for stmt in ast.walk(module):
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for alias in stmt.names:
                names.add(alias.asname or alias.name.split('.')[0])
    names.discard('lib')
    return frozenset(names)


def _unmodelled_kind(node: ast.AST) -> str | None:
    """Node type of the first construct in a scope's own body the pass cannot
    classify, or None when the scope is fully modelled."""
    for sub in _walk_own(node):
        if isinstance(sub, _UNMODELLED_NODES):
            return type(sub).__name__
    return None


def _unresolved_collection(node: ast.AST) -> bool:
    """Whether a scope's own body calls a collection builtin on an unnameable id.

    ``lib.array.set(**args)`` mutates a collection the pass cannot name, so the
    mutation joins no alias class and the statement holding it would be dropped
    from a slice that reads the very collection it writes. The whole module's
    optimization is given up instead.
    """
    for sub in _walk_own(node):
        if not (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)):
            continue
        collection = _collection_call(sub.func)
        if collection is None:
            continue
        namespace, fname = collection
        if fname in _COLLECTION_READERS[namespace]:
            continue
        if _collection_target(sub) is None:
            return True
    return False


def _reachable_scopes(scopes: _SliceScopes, main_key: str) -> list[str]:
    """Scope keys of ``main`` and every user function it can reach."""
    seen = {main_key}
    work = [main_key]
    while work:
        key = work.pop()
        node = scopes.scope_node(key)
        for sub in _walk_own(node):
            if not isinstance(sub, ast.Call) or not isinstance(sub.func, ast.Name):
                continue
            callee = scopes.func_key(sub.func.id, key)
            if callee is not None and callee not in seen:
                seen.add(callee)
                work.append(callee)
        # A nested definition is reachable through any call of it; its own body
        # is scanned when the call is found, but the definition's decorators and
        # defaults belong to the enclosing scope and are covered there.
    return sorted(seen)


# --- summaries and slicing ---


def _function_summaries(collector: _Collector, scopes: _SliceScopes,
                        reachable: list[str], main_key: str) -> dict[str, _Facts]:
    """Transitive read / write / mutate summary of every reachable function."""
    summaries: dict[str, _Facts] = {}
    for key in reachable:
        if key == main_key:
            continue
        body = scopes.scope_body(key)
        summaries[key] = collector.facts(list(body), key)
    for _ in range(_MAX_ROUNDS):
        changed = False
        for key, facts in summaries.items():
            for callee in list(facts.calls):
                other = summaries.get(callee)
                if other is not None and facts.absorb(other):
                    changed = True
                elif other is None and _EXTERNAL not in facts.reads:
                    facts.external()
                    changed = True
        if not changed:
            break
    return summaries


def _expand(facts: _Facts, summaries: dict[str, _Facts]) -> _Facts:
    """Fold the summaries of the functions a statement calls into its facts."""
    seen: set[str] = set()
    work = list(facts.calls)
    while work:
        key = work.pop()
        if key in seen:
            continue
        seen.add(key)
        other = summaries.get(key)
        if other is None:
            facts.external()
            continue
        facts.absorb(other)
        work.extend(other.calls)
    return facts


def _skip_reason(sid: str, ctx: ast.Dict, written_sids: set[str]) -> str | None:
    """Why this context may not get a clone at all, or None when it may.

    Excluded: ``request.security_lower_tf`` contexts, the plain-OHLCV fast path
    (which already skips ``main()`` entirely) and a write block that does not
    sit in ``main()`` itself.

    A context whose symbol or timeframe is only resolved at runtime is NOT
    excluded: the slice is of ``main()``'s statements, which are the same
    whatever the resolved feed turns out to be, and the child picks its clone
    from the context meta the chart hands it when the process is spawned —
    after the deferred resolution, on the very same path as a static context.
    """
    if _ctx_get(ctx, 'is_ltf') is not None:
        return 'lower_tf'
    if _ctx_get(ctx, 'ohlcv_fields') is not None:
        return 'ohlcv_passthrough'
    if sid not in written_sids:
        return 'write_block_outside_main'
    return None


def _ctx_group(ctx: ast.Dict) -> int | None:
    """A context's compile-time group ordinal, if it has one."""
    value = _ctx_get(ctx, 'group')
    if isinstance(value, ast.Constant) and isinstance(value.value, int):
        return value.value
    return None


#: Library functions whose own state is per EXECUTION rather than per bar — a
#: ``varip`` slot the child's re-tick rollback leaves alone, so a period whose
#: developing rounds are skipped would reach them a different number of times.
#: ``ta.valuewhen`` pushes its occurrence ring once per execution (see
#: ``lib/ta.py``); ``_math_stateful``'s only ``varip`` is a memo of a walk the
#: rolled-back state reproduces, so its value does not depend on the count.
_PER_EXECUTION_LIB_CALLS = frozenset({'valuewhen'})

#: Pure Python builtins a call may name without a body to read: none of them
#: touches state of the script, so an unresolved one is not opaque. The list is
#: explicit rather than ``dir(builtins)`` — a builtin that runs arbitrary code
#: (``eval``, ``exec``, ``getattr``) must stay unresolvable.
_SAFE_BUILTIN_CALLS = frozenset({
    'abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'filter', 'float', 'int',
    'len', 'list', 'map', 'max', 'min', 'range', 'reversed', 'round', 'set',
    'sorted', 'str', 'sum', 'tuple', 'zip',
})


def _is_varip_annotation(annotation: ast.expr) -> bool:
    """Whether an annotation declares a ``varip`` (``IBPersistent``) variable."""
    if isinstance(annotation, ast.Name):
        return annotation.id in VARIP_TYPES
    if isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name):
        return annotation.value.id in VARIP_TYPES
    if isinstance(annotation, ast.Attribute):
        return annotation.attr in VARIP_TYPES
    return False


def _pynecore_imported_names(module: ast.Module) -> set[str]:
    """Names an import of the PYNECORE package binds.

    Such a name stands for a library function, and the library's per-execution
    state is enumerated (:data:`_PER_EXECUTION_LIB_CALLS`), so a call through it
    needs no body — the same trust the ``lib``-rooted attribute spelling gets.
    An import of any other module binds a body this module does not hold and is
    not in here (:func:`_imported_names`).

    :param module: The lowered module.
    :return: The bound names.
    """
    names: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom):
            if _is_pynecore_module(node.module):
                for alias in node.names:
                    names.add(alias.asname or alias.name.split('.')[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _is_pynecore_module(alias.name):
                    names.add(alias.asname or alias.name.split('.')[0])
    return names


def _is_pynecore_module(module: str | None) -> bool:
    """Whether a module path names the PyneCore package itself."""
    return module is not None and (module == 'pynecore'
                                   or module.startswith('pynecore.'))


def _declares_varip(stmts: list[ast.stmt], funcs: dict[str, list[ast.AST]],
                    rebound: set[str], pyne_imported: set[str]) -> bool:
    """Whether the code these statements RUN carries ``varip`` state.

    The child's re-tick rollback restores the state VECTOR; ``varip``
    (``IBPersistent``) slots are deliberately outside it
    (``core/instance_state.py``), so a period whose developing rounds are
    skipped would end on a different number. Ordinary Python storage is outside
    the rollback too, and is not searched for here: the language rule forbids it
    (:mod:`~pynecore.transformers.outer_write` — an object created outside a
    function cannot be modified inside one), so the only state left is the one
    this scan enumerates.

    The scan has to be COMPLETE over the code the child can execute, so a call
    is followed only when its callee is statically visible:

    - a ``lib``-rooted (or pynecore-import-rooted) attribute chain whose
      attribute is not in :data:`_PER_EXECUTION_LIB_CALLS` — a library function,
      whose own per-execution state is enumerated and which never calls back
      into script code,
    - a bare ``Name`` the module defines with a ``def``, whose body — EVERY
      definition of that name — is then scanned with the same rules,
    - a bare ``Name`` bound by a pynecore import, one of the protocol / marker
      calls the earlier passes emit (:data:`_PROTOCOL_CALLS`,
      :data:`_MARKER_CALLS`), or one of the pure builtins in
      :data:`_SAFE_BUILTIN_CALLS`.

    A name an assignment, a parameter or a non-pynecore import also binds
    (``rebound``) is never resolved: it may shadow any of the above. Every other
    callee shape is unresolvable, and an unresolvable callee gives the skip up.

    :param stmts: The statements to scan.
    :param funcs: The module's function bodies by name (nested ones included).
    :param rebound: Names an assignment, parameter or foreign import binds.
    :param pyne_imported: Names an import of the pynecore package binds.
    :return: Whether state outside the rollback is reachable.
    """
    seen: set[int] = set()
    work: list[ast.AST] = []
    for stmt in stmts:
        work.extend(_stmt_nodes(stmt))
    while work:
        node = work.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        for sub in [node, *_walk_own(node)]:
            if isinstance(sub, ast.AnnAssign) and _is_varip_annotation(sub.annotation):
                return True
            if isinstance(sub, (ast.Global, ast.Nonlocal)):
                return True
            if not isinstance(sub, ast.Call):
                continue
            func = sub.func
            if isinstance(func, ast.Name):
                if func.id in rebound or func.id in _PER_EXECUTION_LIB_CALLS:
                    return True
                bodies = funcs.get(func.id)
                if bodies:
                    work.extend(bodies)
                elif func.id not in pyne_imported and func.id not in _PROTOCOL_CALLS \
                        and func.id not in _MARKER_CALLS \
                        and func.id not in _SAFE_BUILTIN_CALLS:
                    return True
            elif isinstance(func, ast.Attribute):
                if func.attr in _PER_EXECUTION_LIB_CALLS:
                    return True
                root = _chain_root_name(func)
                if root is None or root in rebound \
                        or not (root == 'lib' or root in pyne_imported):
                    return True
            else:
                # Neither a name nor an attribute chain: nothing to resolve.
                return True
    return False


def _chain_root_name(node: ast.expr) -> str | None:
    """The base ``Name`` id of an attribute chain, or None when it has none."""
    root: ast.expr = node
    while isinstance(root, ast.Attribute):
        root = root.value
    return root.id if isinstance(root, ast.Name) else None


def _imported_names(module: ast.Module) -> set[str]:
    """Names an import of a NON-pynecore module binds.

    A subset of :func:`_rebound_names` narrow enough to judge an ARGUMENT by:
    such a name stands for a user callable this module does not hold, so handing
    it to a call (``map(counter, ...)``) runs a body whose ``varip`` state the
    scan cannot see. An ordinary variable is not in here, so a value argument
    never trips the check.

    :param module: The lowered module.
    :return: The bound names.
    """
    names: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom):
            if not _is_pynecore_module(node.module):
                for alias in node.names:
                    names.add(alias.asname or alias.name.split('.')[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if not _is_pynecore_module(alias.name):
                    names.add(alias.asname or alias.name.split('.')[0])
    return names


def _rebound_names(module: ast.Module) -> set[str]:
    """Names an assignment, a parameter, or a ``for``/``with``/comprehension
    target binds.

    A call through one of them is not resolvable by name, so
    :func:`_declares_varip` gives its context up rather than guess. Parameters
    belong here even though the R1 rule of :func:`_declares_varip` already
    rejects an unresolvable Name call: a parameter may SHADOW a resolvable name — a
    callable passed as ``abs`` shadows the builtin, a helper's ``src`` shadows
    a module-level ``def src`` — and the shadowed spelling would otherwise
    answer for the value actually passed in.

    An import of a NON-pynecore module belongs here for the same reason: ``from
    helper import abs`` and ``import helper as abs`` both bind an opaque user
    callable to a spelling :data:`_SAFE_BUILTIN_CALLS` would otherwise trust as
    a builtin, and the attribute root of ``helper.counter()`` is the same name.
    """
    names: set[str] = _imported_names(module)
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            args = node.args
            for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs,
                        args.vararg, args.kwarg]:
                if arg is not None:
                    names.add(arg.arg)
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            targets = [node.target]
        elif isinstance(node, ast.withitem):
            targets = [node.optional_vars] if node.optional_vars is not None else []
        for target in targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
    return names


#: Test deciding whether a node IS the target whose reachability is judged: the
#: sid's ``__sec_write__`` call (:func:`_sid_write`) or, while the static call
#: chain is followed, one exact call node (:func:`_same_node`).
_Hit = Callable[[ast.AST], bool]


def _sid_write(sid: str) -> _Hit:
    """A target test matching any ``__sec_write__`` call of ``sid``."""

    def hit(node: ast.AST) -> bool:
        return _protocol_sid(node, '__sec_write__') == sid

    return hit


def _same_node(target: ast.AST) -> _Hit:
    """A target test matching one exact node."""

    def hit(node: ast.AST) -> bool:
        return node is target

    return hit


def _contains_write(node: ast.AST, hit: _Hit) -> bool:
    """Whether ``node`` itself or one of its own descendants is the target."""
    return any(hit(sub) for sub in [node, *_walk_own(node)])


def _has_exit(stmt: ast.stmt) -> bool:
    """Whether ``stmt`` can end the round before the next statement."""
    return any(isinstance(sub, _WRITE_SKIPPING_NODES) for sub in [stmt, *_walk_own(stmt)])


def _expr_unguarded(node: ast.AST, hit: _Hit) -> bool:
    """Whether the target stands under no expression-level branch inside ``node``.

    The two expression-level branches are ``IfExp`` and the short-circuit
    ``BoolOp``: a target in a branch of the first or in a later operand of the
    second is evaluated only for some values of what precedes it, so it is
    guarded. The ``IfExp``'s test and the ``BoolOp``'s first operand always
    run. A target standing in a call argument or in an
    operand of a plain operator is not guarded at all. A lambda or a
    comprehension answers False as well: when its body runs is not visible here.

    :param node: The node holding the target.
    :param hit: The target test.
    :return: Whether the target is free of expression-level guards.
    """
    if hit(node):
        return True
    if isinstance(node, ast.IfExp):
        # Only the test itself runs unconditionally.
        return (_contains_write(node.test, hit)
                and not any(_contains_write(branch, hit)
                            for branch in (node.body, node.orelse))
                and _expr_unguarded(node.test, hit))
    if isinstance(node, ast.BoolOp):
        # Only the first operand runs whatever the others evaluate to.
        first = node.values[0]
        return (_contains_write(first, hit)
                and not any(_contains_write(value, hit) for value in node.values[1:])
                and _expr_unguarded(first, hit))
    if isinstance(node, (ast.Lambda, ast.ListComp, ast.SetComp,
                         ast.DictComp, ast.GeneratorExp)):
        return False
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if _contains_write(child, hit) and not _expr_unguarded(child, hit):
            return False
    return True


def _write_unconditional(stmts: list[ast.stmt], hit: _Hit) -> bool:
    """Whether the target runs on EVERY execution of ``stmts``.

    ``closed_shift`` says the context's value is the same on every chart bar of
    one HTF period, so the period's later developing rounds may be skipped —
    which also means their WRITE is skipped. That is only unobservable while the
    write is unconditional: a ``if close <= open: return`` ahead of it (an exit
    node the slicer keeps on purpose) is decided by the child's DEVELOPING bar,
    so it can turn from skipping the write to performing it inside one period,
    and the skipped round would leave the previous period's value standing.

    The rule is therefore purely structural: the target must stand in a TOP-LEVEL
    statement of ``stmts``, with no early exit (:data:`_WRITE_SKIPPING_NODES`) reachable
    ahead of it, and with no branch of any kind around it — no ``if``, no
    ``IfExp``, no short-circuit ``BoolOp``, no loop, no ``with``, no ``try``.
    The only ``if`` that is not a user branch is the protocol's own
    ``__active_security__`` dispatch (:func:`_is_dispatch_test`): the child
    always has its own sid active, so that body runs on every round. A target the
    scan does not find at all answers False.

    This scan never enters a nested ``def``; a write standing in a helper body
    is judged by :func:`_write_reached`, which follows the call chain.

    :param stmts: The statements the child runs, in order.
    :param hit: The target test.
    :return: Whether the target is unconditional.
    """
    for stmt in stmts:
        # A nested ``def`` only binds a name here (:func:`_stmt_nodes`), so
        # neither its write nor its ``return`` belongs to this statement list.
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(hit(sub) for sub in _walk_own(stmt)):
            return _stmt_unconditional(stmt, hit)
        if _has_exit(stmt):
            return False
    return False


def _stmt_unconditional(stmt: ast.stmt, hit: _Hit) -> bool:
    """Whether the statement holding the target performs it on every run.

    :param stmt: The statement holding the target.
    :param hit: The target test.
    :return: Whether the target is unconditional within ``stmt``.
    """
    if _holds_write(stmt, hit):
        return _expr_unguarded(stmt, hit)
    if not isinstance(stmt, ast.If):
        return False
    if _contains_write(stmt.test, hit):
        # A target inside the test itself runs before the branch is taken.
        return _expr_unguarded(stmt.test, hit)
    if any(_contains_write(sub, hit) for sub in stmt.orelse):
        return False
    if any(_contains_write(sub, hit) for sub in stmt.body):
        if not (_is_dispatch_test(stmt.test) and not stmt.orelse):
            return False
        return _write_unconditional(stmt.body, hit)
    return True


def _is_dispatch_test(test: ast.expr) -> bool:
    """Whether ``test`` is the protocol's own ``__active_security__`` dispatch."""
    return any(isinstance(sub, ast.Name) and sub.id == '__active_security__'
               for sub in ast.walk(test))


def _holds_write(stmt: ast.stmt, hit: _Hit) -> bool:
    """Whether a simple statement holds the target itself."""
    if not isinstance(stmt, (ast.Expr, ast.Assign, ast.AnnAssign, ast.AugAssign,
                             ast.Return)):
        return False
    return any(hit(sub) for sub in _walk_own(stmt))


def _holds_node(stmt: ast.stmt, target: ast.AST) -> bool:
    """Whether ``target`` stands anywhere inside ``stmt``, nested defs included."""
    return any(sub is target for sub in ast.walk(stmt))


def _def_chain(body: list[ast.stmt],
               target: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef] | None:
    """The nested ``def``s standing between ``body`` and ``target``.

    An empty list means the target stands in ``body`` itself; ``None`` means the
    chain is not one this pass models — the target is not in ``body`` at all, or
    it is reached only through a ``def`` that is not a top-level statement of its
    scope (a ``def`` inside an ``if`` binds its name conditionally).

    :param body: The scope's statement list.
    :param target: The node to reach.
    :return: The defs from the outermost to the innermost, or ``None``.
    """
    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _holds_node(stmt, target):
                continue
            inner = _def_chain(stmt.body, target)
            if inner is None:
                # In a decorator or a default, or behind a conditional ``def``.
                return None
            return [stmt, *inner]
        if _holds_node(stmt, target):
            return []
    return None


def _def_called_unconditionally(func: ast.FunctionDef | ast.AsyncFunctionDef,
                                body: list[ast.stmt],
                                seen: frozenset[str]) -> bool:
    """Whether ``func``'s body runs on every execution of ``body``.

    The callee resolution is the closed R1 rule of the ``varip`` scan
    (:func:`_declares_varip`) narrowed to the one shape this pass can follow: a
    bare ``Name`` that resolves to this very ``def``. Anything else about the
    name makes it unresolvable and the answer False — a second ``def`` or an
    assignment binding it, a parameter shadowing it, or a reference outside
    callee position (``cb = helper``), which would let the body run from a place
    the scan does not see. Every call site must then run on every execution of
    ``body`` itself (:func:`_node_reached`), so one conditional call cannot
    weaken the others; recursion answers False through ``seen``.

    :param func: The definition whose body holds the target.
    :param body: The scope the definition stands in.
    :param seen: Names already being resolved on this chain.
    :return: Whether the body is entered on every execution of ``body``.
    """
    name = func.name
    if name in seen:
        return False
    seen = seen | {name}
    nodes = [sub for stmt in body for sub in ast.walk(stmt)]
    for sub in nodes:
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and sub.name == name and sub is not func:
            return False
        if isinstance(sub, ast.arg) and sub.arg == name:
            return False
    exempt = {id(sub.func) for sub in nodes if isinstance(sub, ast.Call)}
    for sub in nodes:
        if isinstance(sub, ast.Name) and sub.id == name and id(sub) not in exempt:
            return False
    calls = [sub for sub in nodes if isinstance(sub, ast.Call)
             and isinstance(sub.func, ast.Name) and sub.func.id == name]
    if not calls:
        return False
    return all(_node_reached(call, body, seen) for call in calls)


def _node_reached(target: ast.AST, body: list[ast.stmt],
                  seen: frozenset[str]) -> bool:
    """Whether ``target`` runs on every execution of ``body``.

    Either the target stands in ``body`` itself, and the statement-level rule
    decides (:func:`_write_unconditional`), or it stands in a nested ``def``,
    and both links have to hold: the target must run on every execution of that
    def's body, and the def must be entered on every execution of ``body``
    (:func:`_def_called_unconditionally`). Applied recursively, this walks up to
    the clone's top level.

    :param target: The node to reach.
    :param body: The scope's statement list.
    :param seen: Names already being resolved on this chain.
    :return: Whether the target runs on every execution of ``body``.
    """
    chain = _def_chain(body, target)
    if chain is None:
        return False
    if not chain:
        return _write_unconditional(body, _same_node(target))
    outer = chain[0]
    return (_node_reached(target, outer.body, seen)
            and _def_called_unconditionally(outer, body, seen))


def _write_reached(stmts: list[ast.stmt], sid: str) -> bool:
    """Whether the sid's write runs on every execution of ``stmts``.

    The statement-level rule answers first, for a write standing in ``stmts``
    itself. A write inside a helper body — what the lowered scripts actually
    emit, since :mod:`.security` puts the protocol block where the call is
    written — is then judged along the static call chain
    (:func:`_node_reached`). One sid belongs to ONE call site there: the
    isolation pass gives every call of a helper its own copy of the def
    (``get_ma_htf__pyne_inst<N>``) and the security pass numbers the sids per
    copy, so the write node is unique and the chain is unambiguous. Two write
    nodes for one sid would not be, and answer False.

    :param stmts: The statements the child runs, in order.
    :param sid: The context's id.
    :return: Whether the write is unconditional.
    """
    hit = _sid_write(sid)
    if _write_unconditional(stmts, hit):
        return True
    targets = [sub for stmt in stmts for sub in ast.walk(stmt) if hit(sub)]
    if len(targets) != 1:
        return False
    return _node_reached(targets[0], stmts, frozenset())


def _finalize_closed_shift(module: ast.Module, contexts: ast.Dict,
                           sliced: dict[str, list[ast.stmt]]) -> None:
    """Clear ``closed_shift`` wherever the child's rounds are observable.

    Two reasons, both of them about the CODE the child runs and not about the
    expression's value: ``varip`` state, and a write that does not run on every
    round (:func:`_write_unconditional`).

    The flag the security transformer emits says the context's VALUE cannot
    change inside an HTF period. What makes skipping the period's developing
    rounds unobservable is the child's re-tick rollback — and ``varip`` state is
    deliberately outside it (``core/instance_state.py``), so a counter in the
    code the child runs would end the period on a different number. The decision
    belongs here because only the slicer knows WHAT the child runs: its own
    clone's kept statements, or the whole ``main()`` when it got no clone.

    Ordinary Python storage at module level is outside the rollback as well, and
    is not analysed here: the language rule enforced by
    :mod:`~pynecore.transformers.outer_write` makes writing such an object from
    inside a function a compile error, so a script that reaches this point has
    none. What a write through an alias or through a parameter does is
    documented as unpredictable, not proven safe.

    :param module: The lowered module.
    :param contexts: The ``__security_contexts__`` dict literal.
    :param sliced: Per sid, the statements its clone kept (see
        :meth:`SecuritySliceTransformer._emit_clones`).
    """
    candidates = [(sid, ctx) for sid, ctx in _iter_contexts(contexts)
                  if _ctx_closed_shift(ctx) is not None]
    if not candidates:
        return
    funcs: dict[str, list[ast.AST]] = {}
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.setdefault(node.name, []).append(node)
    rebound = _rebound_names(module)
    pyne_imported = _pynecore_imported_names(module)
    main = _find_main(module)
    whole: bool | None = None
    for sid, ctx in candidates:
        stmts = sliced.get(sid)
        if stmts is None:
            if main is None:
                observable = True
            else:
                if whole is None:
                    whole = _declares_varip(list(main.body), funcs, rebound,
                                            pyne_imported)
                observable = whole or not _write_reached(list(main.body), sid)
        else:
            observable = (_declares_varip(stmts, funcs, rebound, pyne_imported)
                          or not _write_reached(stmts, sid))
        if observable:
            value = _ctx_closed_shift(ctx)
            assert value is not None
            value.value = False


def _ctx_closed_shift(ctx: ast.Dict) -> ast.Constant | None:
    """The context's ``closed_shift`` literal, only while it says True."""
    value = _ctx_get(ctx, 'closed_shift')
    if isinstance(value, ast.Constant) and value.value is True:
        return value
    return None


def _slice_units(entries: list[tuple[int, tuple[str, ast.Dict]]],
                 written_sids: set[str],
                 skipped: dict[str, str]) -> list[list[tuple[int, str, ast.Dict]]]:
    """Split the module's contexts into the units one clone is built for.

    A context without a group is a unit of its own, as before. The contexts of
    one group share a single clone: they resolve to the same feed, so one child
    can serve all of them, and the clone must then hold every member's write
    block and every member's reads.

    A member the pass cannot slice at all (an LTF context, a write block
    outside ``main()``) takes the whole group down with it — the group's child
    would be missing that member's value. A plain-OHLCV member does NOT: its
    write block is a single trivial statement the group clone simply keeps. A
    group of nothing but plain-OHLCV members gets no clone, because its child
    never runs ``main()`` in the first place.

    :param entries: ``(index, (sid, context dict))`` in context order
    :param written_sids: sids whose write block stands in ``main()``
    :param skipped: filled with the per-sid reason wherever no clone is built
    :return: the units, each a list of ``(index, sid, context dict)``
    """
    reasons: dict[str, str | None] = {}
    groups: dict[int, list[tuple[int, str, ast.Dict]]] = {}
    ordered: list[tuple[int, str, ast.Dict, int | None]] = []
    for index, (sid, ctx) in entries:
        reasons[sid] = _skip_reason(sid, ctx, written_sids)
        group = _ctx_group(ctx)
        ordered.append((index, sid, ctx, group))
        if group is not None:
            groups.setdefault(group, []).append((index, sid, ctx))

    units: list[list[tuple[int, str, ast.Dict]]] = []
    emitted: set[int] = set()
    for index, sid, ctx, group in ordered:
        if group is None:
            reason = reasons[sid]
            if reason is not None:
                skipped[sid] = reason
            else:
                units.append([(index, sid, ctx)])
            continue
        if group in emitted:
            continue
        emitted.add(group)
        members = groups[group]
        blocking = [s for _i, s, _c in members
                    if reasons[s] is not None and reasons[s] != 'ohlcv_passthrough']
        sliceable = [s for _i, s, _c in members if reasons[s] is None]
        if blocking or not sliceable:
            for _i, member, _c in members:
                skipped[member] = reasons[member] or 'group_unsliceable'
            continue
        units.append(members)
    return units


def _whole_units(entries: list[tuple[int, tuple[str, ast.Dict]]],
                 covered: set[str]) -> list[list[tuple[int, str, ast.Dict]]]:
    """The units no sliced clone serves, grouped the way their children are.

    Nothing blocks a unit here: a clone of the whole ``main()`` holds every
    write block ``main()`` does, whatever kind of context it belongs to.

    :param entries: ``(index, (sid, context dict))`` in context order
    :param covered: sids a sliced clone already serves
    :return: the remaining units, each a list of ``(index, sid, context dict)``
    """
    units: list[list[tuple[int, str, ast.Dict]]] = []
    by_group: dict[int, list[tuple[int, str, ast.Dict]]] = {}
    for index, (sid, ctx) in entries:
        if sid in covered:
            continue
        group = _ctx_group(ctx)
        if group is None:
            units.append([(index, sid, ctx)])
        elif group in by_group:
            by_group[group].append((index, sid, ctx))
        else:
            by_group[group] = [(index, sid, ctx)]
            units.append(by_group[group])
    return units


def _stmt_nodes(stmt: ast.stmt) -> list[ast.AST]:
    """The nodes of a top-level statement that RUN when the statement runs.

    A nested ``def`` only binds a name here: what its body reads and writes
    materializes at the call sites, through the function summaries.
    """
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        spec = stmt.args
        defaults: list[ast.AST] = [d for d in spec.defaults]
        defaults += [d for d in spec.kw_defaults if d is not None]
        return list(stmt.decorator_list) + defaults
    return [stmt]


def _build_index(body: list[ast.stmt], collector: _Collector, aliases: _Aliases,
                 summaries: dict[str, _Facts], main_key: str) -> list[_StmtIndex]:
    """Build the context-independent :class:`_StmtIndex` of ``main()``'s body."""
    index: list[_StmtIndex] = []
    for stmt in body:
        entry = _StmtIndex()
        facts = _expand(collector.facts(_stmt_nodes(stmt), main_key), summaries)
        entry.reads = aliases.classes(facts.reads)
        entry.defs = aliases.classes(facts.writes) | aliases.classes(facts.mutates)
        # The sids come from the same expansion as the names: a protocol call
        # inside a helper belongs to the STATEMENT THAT CALLS the helper, not to
        # the ``def`` that only binds it — dropping the call site would leave
        # the write block unreached and the context silently unwritten.
        entry.read_sids = facts.read_sids
        entry.write_sids = facts.write_sids
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # A definition costs nothing per bar and may be called by anything
            # the slice keeps.
            entry.mandatory = True
        elif any(isinstance(sub, _EXIT_NODES) for sub in [stmt, *_walk_own(stmt)]):
            # An early exit decides whether a later write block is reached.
            entry.mandatory = True
        index.append(entry)
    return index


def _slice(index: list[_StmtIndex], sids: set[str],
           keep_reads: set[str]) -> list[int] | None:
    """Backward slice of ``main()``'s top-level statements for one unit.

    :param index: the per-statement index of ``main()``'s body
    :param sids: the contexts the clone must write (one, or a whole group)
    :param keep_reads: sids whose ``__sec_read__`` must run in this child
    :return: indexes of the kept statements, or None when nothing was dropped
    """
    kept: set[int] = set()
    for position, entry in enumerate(index):
        if (entry.mandatory or entry.write_sids & sids
                or entry.read_sids & keep_reads):
            kept.add(position)

    for _ in range(len(index) + 1):
        needed: set[str] = set()
        for position in kept:
            needed |= index[position].reads
        grown = False
        for position, entry in enumerate(index):
            if position not in kept and entry.defs & needed:
                kept.add(position)
                grown = True
        if not grown:
            break

    if len(kept) == len(index):
        return None
    return sorted(kept)


def _reaching_defs(defs: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]],
                   hit: _Hit) -> set[str]:
    """Names of the ``def``s whose call can perform the target.

    A direct writer holds the target in its own body; an indirect one calls a
    writer by name. The closure is taken over the call graph the code spells
    out, so an unresolvable callee (an attribute, a value passed in) simply
    leaves its caller out — that path is then not forced.

    :param defs: The candidate definitions, by name.
    :param hit: The target test.
    :return: The names whose call reaches the target.
    """
    reaching = {name for name, nodes in defs.items()
                if any(_contains_write(stmt, hit)
                       for node in nodes for stmt in node.body
                       if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)))}
    callees = {
        name: {sub.func.id for node in nodes for sub in _walk_own(node)
               if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)}
        for name, nodes in defs.items()
    }
    for _ in range(len(defs) + 1):
        grown = False
        for name, called in callees.items():
            if name not in reaching and called & reaching:
                reaching.add(name)
                grown = True
        if not grown:
            break
    return reaching


def _nested_defs(func: ast.FunctionDef) -> dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Every ``def`` standing anywhere inside ``func``, by name."""
    defs: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for node in ast.walk(func):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node is not func):
            defs.setdefault(node.name, []).append(node)
    return defs


def _stmt_reaches(stmt: ast.stmt, test: _Hit) -> bool:
    """Whether running ``stmt`` performs the target (nested ``def``s excluded)."""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    return _contains_write(stmt, test)


#: Expression shapes whose body runs at a time the forcing pass does not see.
_DEFERRED_EXPRS: tuple[type[ast.AST], ...] = (
    ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)

#: Simple statements whose expressions are rewritten in place.
_FORCED_SIMPLE: tuple[type[ast.AST], ...] = (
    ast.Expr, ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return)


class _Forcer:
    """Rewrites a child's code so the targets run on every execution.

    Two kinds of guard can stand between a scope's body and a target: a user
    ``if`` statement, and an expression-level branch (``IfExp``, short-circuit
    ``BoolOp``) around the CALL of a helper that performs the target. The
    ``__sec_write__`` block is a statement of its own and always precedes the
    statement that held the ``request.security()`` call, so an expression-level
    guard only ever stands around a helper call.

    A user ``if`` stays where it is: only the statements that reach a target
    are LIFTED in front of it, each behind copies of the plain assignments of
    the same branch it reads. Everything else the branch does — drawings, the
    script's own conditionally advancing state — keeps running under its guard,
    as it does in the context TradingView compiles: the request is what is
    unconditional there, not the code around it.

    A loop, a ``with``, a ``try`` and a ``match`` are left as they are, and so
    is a lambda or a comprehension: how often their body runs is not a guard
    this pass can remove.

    :ivar changed: whether anything was rewritten
    """

    def __init__(self, test: _Hit, aliases: _Aliases) -> None:
        self.test = test
        self.aliases = aliases
        self.changed = False
        self._temps = 0
        self._locals: set[str] = set()
        self._certain: set[str] = set()

    def func_body(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
        """One function's body, forced, with its local bindings tracked.

        A lift moves reads in front of the guard they were written under, so a
        local the guard itself is what binds — ``if flag: original = close``
        read from a second ``if flag:`` — would be read unbound on a pass that
        takes neither branch. Telling such a name from a module-level one needs
        the function: a name bound nowhere in it is a global, an import or a
        builtin and is always available, while every name the function binds is
        a local that has to be proven bound before the guard.

        :param node: The function whose body to force.
        :return: The forced body.
        """
        saved_locals, saved_certain = self._locals, self._certain
        args = node.args
        params = {arg.arg for arg in
                  [*args.posonlyargs, *args.args, *args.kwonlyargs,
                   *([args.vararg] if args.vararg else []),
                   *([args.kwarg] if args.kwarg else [])]}
        self._locals = params.union(*(_bound_names(stmt) for stmt in node.body)) \
            if node.body else params
        self._certain = set(params)
        try:
            return self.body(node.body)
        finally:
            self._locals, self._certain = saved_locals, saved_certain

    def body(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        """The statement list with every reaching statement forced."""
        out: list[ast.stmt] = []
        for stmt in stmts:
            if _stmt_reaches(stmt, self.test):
                out.extend(self.stmt(stmt))
            else:
                out.append(stmt)
            bound, removed = _definite_bindings(stmt)
            self._certain = (self._certain - removed) | bound
        return out

    def stmt(self, stmt: ast.stmt) -> list[ast.stmt]:
        """One reaching statement, as the statements that replace it."""
        if isinstance(stmt, ast.If):
            return self._if(stmt)
        if not isinstance(stmt, _FORCED_SIMPLE):
            return [stmt]
        hoisted: list[ast.expr] = []
        for field, value in ast.iter_fields(stmt):
            if isinstance(value, ast.expr):
                setattr(stmt, field, self.expr(value, hoisted))
        return [*_as_statements(hoisted, stmt), stmt]

    def _if(self, stmt: ast.If) -> list[ast.stmt]:
        hoisted: list[ast.expr] = []
        in_test = _contains_write(stmt.test, self.test)
        if in_test:
            stmt.test = self.expr(stmt.test, hoisted)
        out = _as_statements(hoisted, stmt)
        in_body = any(_stmt_reaches(sub, self.test) for sub in stmt.body)
        in_else = any(_stmt_reaches(sub, self.test) for sub in stmt.orelse)
        # What is certainly bound in front of the ``if`` is what a lift out of
        # either branch may read; a branch's own bindings are undone for the
        # other branch and for everything after the statement.
        outer = set(self._certain)
        if _is_dispatch_test(stmt.test) or not (in_body or in_else):
            # The protocol's own dispatch stays: the child always has its sid
            # active. A target in the test alone already runs on every pass.
            stmt.body = self.body(stmt.body)
            self._certain = set(outer)
            stmt.orelse = self.body(stmt.orelse)
            self._certain = outer
            return [*out, stmt]
        # Both branches may hold a target — one context in each. TradingView
        # computes both series on every bar, so the child lifts both.
        for branch in (stmt.body, stmt.orelse):
            self._certain = set(outer)
            forced = self.body(branch)
            self._certain = outer
            kept, lifted = self._lift(forced, outer)
            if not lifted:
                continue
            self.changed = True
            out.extend(lifted)
            branch[:] = kept or ([ast.copy_location(ast.Pass(), stmt)] if branch else [])
        return [*out, stmt]

    def _lift(self, stmts: list[ast.stmt],
              certain: set[str]) -> tuple[list[ast.stmt], list[ast.stmt]]:
        """Split a branch into what stays guarded and what runs ahead of the guard.

        A reaching statement is lifted only when everything the branch does
        before it can run ahead of the guard unchanged: plain assignments of
        side-effect-free expressions, which the lift copies. A name bound any
        other way — a nested conditional, an augmented assignment, an
        assignment whose value calls something — keeps the statement reading
        it where it is, and so does a script value an earlier effect of the
        branch may have mutated: a lift would read an unbound name, publish a
        stale value, or run the effect twice.

        Every local the lifted group reads must also be certainly bound in
        front of the guard, or bound by a copy of the group itself: a local the
        branch alone binds is unbound on a pass that never takes it.

        :param stmts: The branch, its own nested guards already handled.
        :param certain: The locals certainly bound in front of the guard.
        :return: The statements left in the branch, and the reaching ones in
                 source order, each preceded by the assignments it reads.
        """
        kept: list[ast.stmt] = []
        lifted: list[ast.stmt] = []
        touched: set[str] = set()
        for stmt in stmts:
            if not _stmt_reaches(stmt, self.test):
                kept.append(stmt)
                if not _is_copyable_assignment(stmt):
                    touched |= _operand_names(stmt)
                continue
            feeding = _feeding_copies(stmt, kept)
            reads = _operand_names(stmt)
            loaded = _loaded_names(stmt)
            provided = set(certain)
            for copied in feeding or ():
                reads |= _operand_names(copied)
                loaded |= _loaded_names(copied)
                provided |= _bound_names(copied)
            if feeding is not None and (loaded & self._locals) - provided:
                # A local only the branch binds would be read unbound ahead of
                # the guard.
                feeding = None
            if feeding is not None and self.aliases.classes(touched) & self.aliases.classes(reads):
                # An earlier effect of the branch — a collection mutation, a
                # call on a script value — may be what the statement reads.
                feeding = None
            if feeding is None:
                kept.append(stmt)
                touched |= _operand_names(stmt)
                continue
            self._detach(feeding, stmt)
            lifted.extend(feeding)
            lifted.append(stmt)
        return kept, lifted

    def _detach(self, feeding: list[ast.stmt], stmt: ast.stmt) -> None:
        """Rebind the copied dependencies onto private names.

        The originals stay under the guard, so a copy keeping its own target
        would write that name ahead of the ``if`` — turning a test that reads
        it false when it was true, and leaving the branch's value behind on a
        pass that never took the branch. The copies therefore bind names of
        their own, and only the lifted statement follows them.

        :param feeding: The copies, in source order; renamed in place.
        :param stmt: The lifted statement; its reads are rewritten in place.
        """
        renamed: dict[str, str] = {}
        for copied in feeding:
            for name in sorted(_bound_names(copied)):
                if name not in renamed:
                    renamed[name] = (f'__sec_dep{PYNE_RESERVED_NAME_CHAR}'
                                     f'{self._temps}__')
                    self._temps += 1
        if not renamed:
            return
        # A namespace root is never bound by a copy, so it is never renamed
        for copied in feeding:
            for node in ast.walk(copied):
                if isinstance(node, ast.Name) and node.id in renamed:
                    node.id = renamed[node.id]
        for node in ast.walk(stmt):
            if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                    and node.id in renamed):
                node.id = renamed[node.id]

    def expr(self, node: ast.expr, hoisted: list[ast.expr]) -> ast.expr:
        """``node`` with every guard around a reaching call removed.

        An ``IfExp`` becomes the branch that reaches a target, and a guarded
        ``BoolOp`` operand is taken out of the chain. Whatever reaches a target
        but cannot stay in the expression — the second branch of an ``IfExp``
        holding one in each, the operand taken out — is handed to ``hoisted``
        and runs as a statement of its own ahead of the rewritten one.

        :param node: The expression.
        :param hoisted: Collects the expressions that must run separately.
        :return: The expression to put in ``node``'s place.
        """
        if isinstance(node, _DEFERRED_EXPRS) or not _contains_write(node, self.test):
            return node
        if isinstance(node, ast.IfExp):
            node.test = self.expr(node.test, hoisted)
            in_body = _contains_write(node.body, self.test)
            in_else = _contains_write(node.orelse, self.test)
            if not (in_body or in_else):
                return node
            self.changed = True
            body = self.expr(node.body, hoisted)
            orelse = self.expr(node.orelse, hoisted)
            if in_body and in_else:
                hoisted.append(orelse)
            return body if in_body else orelse
        if isinstance(node, ast.BoolOp):
            values = [self.expr(value, hoisted) for value in node.values]
            kept = [values[0]]
            for value in values[1:]:
                if _contains_write(value, self.test):
                    hoisted.append(value)
                    self.changed = True
                else:
                    kept.append(value)
            if len(kept) == 1:
                return kept[0]
            node.values = kept
            return node
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.expr):
                setattr(node, field, self.expr(value, hoisted))
            elif isinstance(value, list):
                setattr(node, field, [self.expr(item, hoisted) if isinstance(item, ast.expr)
                                      else item for item in value])
        for keyword in getattr(node, 'keywords', ()):
            keyword.value = self.expr(keyword.value, hoisted)
        return node


#: Annotations of a declaration that owns per-call-site state: running a copy of
#: one would give the script a second, independently advancing variable.
_STATEFUL_ANNOTATIONS = frozenset({'Persistent', 'Series', *VARIP_TYPES})


def _loaded_names(node: ast.AST) -> set[str]:
    """Every name the node reads."""
    return {sub.id for sub in ast.walk(node)
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load)}


#: Statements whose bindings only happen on some paths through them.
_CONDITIONAL_STMT = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try,
                     ast.TryStar, ast.With, ast.AsyncWith, ast.Match)


def _definite_bindings(stmt: ast.stmt) -> tuple[set[str], set[str]]:
    """What the statement certainly gives a value to, and what it may unbind.

    This is deliberately not ``_bound_names``, which answers the other
    question — which names are locals of the scope at all. A name is local
    even when the statement that makes it one gives it no value
    (``original: float``) or takes its value away (``del original``), and
    neither of those may let a lift read it ahead of a guard.

    What binds is therefore listed by statement kind, and only outside a
    compound statement: a body may not run, may run zero times, or may leave
    through an exception halfway. A statement kind that is not listed adds
    nothing, which also keeps an assignment evaluated inside a short-circuiting
    expression — ``flag and (original := close)`` — out. Deletions are
    subtracted wherever they sit, nested ones included, since a compound
    statement that binds nothing certain can still take a binding away. An
    ``except E as name`` handler counts as such a deletion: Python unbinds
    ``name`` when the handler leaves, whatever value it held before.

    :param stmt: The statement.
    :return: The names certainly bound, and the names possibly unbound.
    """
    nodes = [stmt, *_walk_own(stmt)]
    removed = {sub.id for sub in nodes
               if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Del)}
    removed |= {sub.name for sub in nodes
                if isinstance(sub, ast.ExceptHandler) and sub.name is not None}
    if isinstance(stmt, _CONDITIONAL_STMT):
        return set(), removed
    bound: set[str] = set()
    targets: list[ast.expr] = []
    if isinstance(stmt, ast.Assign):
        targets = list(stmt.targets)
    elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
        targets = [stmt.target]
    elif isinstance(stmt, ast.AugAssign):
        targets = [stmt.target]
    elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        bound.add(stmt.name)
    elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
        bound |= {(alias.asname or alias.name).split('.')[0] for alias in stmt.names}
    for target in targets:
        bound |= {sub.id for sub in [target, *ast.walk(target)]
                  if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store)}
    return bound - removed, removed


def _bound_names(stmt: ast.stmt) -> set[str]:
    """Every local name the statement may bind, nested statements included.

    Nested function definitions bind only their own name here: what their body
    binds is local to them.
    """
    names: set[str] = set()
    for node in [stmt, *_walk_own(stmt)]:
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split('.')[0])
        elif isinstance(node, ast.ExceptHandler) and node.name is not None:
            names.add(node.name)
    for node in ast.iter_child_nodes(stmt):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


def _is_repeatable(node: ast.expr) -> bool:
    """Whether the expression may be evaluated ahead of the guard it sits under.

    This is an allowlist: only expression kinds that provably neither advance
    state nor raise are accepted, because the copy runs on every pass while the
    original stays guarded. Accepted are a constant, a bare name and an
    attribute chain rooted in the compiler-owned ``lib`` binding, which always
    exists and whose members are the library namespaces.

    Everything else is rejected, and the reason is the same for all of it: the
    guard is usually what makes the expression valid at all. A call may advance
    state (``array.pop``) or draw; a subscript needs the index the guard
    establishes — ``if array.size(store) > 0: x = store[0]``; an attribute of a
    script value needs the owner the guard establishes — ``if item is not None:
    x = item.value``. Operators are no different: ``if item != 0: x = 10 //
    item`` raises ``ZeroDivisionError`` outside the guard, and ``if item is not
    None: x = item + 1`` raises ``TypeError``. Since neither the operand types
    nor their values are known here, no operator can be shown safe, so none is
    accepted.
    """
    for sub in [node, *ast.walk(node)]:
        if isinstance(sub, ast.Attribute):
            if not _is_lib_attribute(sub):
                return False
        elif not isinstance(sub, (ast.Constant, ast.Name, ast.Load)):
            return False
    return True


def _is_lib_attribute(node: ast.Attribute) -> bool:
    """Whether the attribute chain is rooted in the compiler's ``lib`` binding."""
    base: ast.expr = node.value
    while isinstance(base, ast.Attribute):
        base = base.value
    return isinstance(base, ast.Name) and base.id == 'lib'


def _plain_aliases(nodes: list[ast.AST]) -> _Aliases:
    """Union-find over the bare names a scope may make denote one object.

    Only ``b = a`` is read here — annotated (``b: list[float] = a``) as much as
    plain, since the annotation changes nothing about the object the second
    name denotes: it is the spelling that hands a collection or an object to a
    second name, after which a mutation written through either one is a
    mutation of both. The names are taken unqualified, so two scopes spelling
    the same name share a class — that only makes the lift more conservative,
    never less.

    :param nodes: The roots to scan, nested functions included.
    :return: The alias classes.
    """
    aliases = _Aliases()
    for root in nodes:
        for node in ast.walk(root):
            if isinstance(node, ast.Assign):
                targets: list[ast.expr] = list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            if not isinstance(node.value, ast.Name):
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    aliases.union(target.id, node.value.id)
    return aliases


def _operand_names(node: ast.AST) -> set[str]:
    """The script values the node touches, the compiler namespace left out.

    Only ``lib`` is dropped: it is the compiler-owned binding of the library
    namespaces — ``lib.array``, ``lib.close`` — and holds nothing a statement
    of the branch can mutate. Any other attribute owner is a value of the
    script itself, and a method called on it — ``store.append(3)`` — mutates
    it, so its root stays an operand and two statements naming it are treated
    as dependent.
    """
    roots: set[int] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute):
            base = sub.value
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id == 'lib':
                roots.add(id(base))
    return {sub.id for sub in ast.walk(node)
            if isinstance(sub, ast.Name) and id(sub) not in roots}


def _is_copyable_assignment(stmt: ast.stmt) -> bool:
    """Whether the statement may run a second time ahead of its own guard.

    Only a plain assignment of a repeatable expression qualifies. Anything
    else either binds a name in a way a copy cannot reproduce or performs an
    effect — a drawing, a collection mutation, an order — that must happen
    exactly once, in its original place.
    """
    targets = _plain_assignment_targets(stmt)
    value = stmt.value if isinstance(stmt, (ast.Assign, ast.AnnAssign)) else None
    if not targets or value is None or not _is_repeatable(value):
        return False
    return not _bound_names(stmt) - targets


def _feeding_copies(stmt: ast.stmt, kept: list[ast.stmt]) -> list[ast.stmt] | None:
    """Copies of the branch statements the reaching statement reads.

    :param stmt: The reaching statement about to be lifted.
    :param kept: The branch statements before it that stay guarded.
    :return: The copies in source order, or ``None`` when a dependency cannot
             run ahead of the guard and the statement must stay where it is.
    """
    needed = _loaded_names(stmt)
    feeding: list[ast.stmt] = []
    for earlier in reversed(kept):
        bound = _bound_names(earlier)
        if not bound & needed:
            continue
        if not _is_copyable_assignment(earlier):
            return None
        feeding.append(copy.deepcopy(earlier))
        needed |= _loaded_names(earlier)
    feeding.reverse()
    # The originals stay under the guard and run again after the copies, so a
    # copy must not read a name the group itself binds at or after it:
    # ``length = length + 1`` would advance twice, and ``a = b`` followed by
    # ``b = a + 1`` would re-read the already advanced ``b``.
    for index, earlier in enumerate(feeding):
        later_targets: set[str] = set()
        for other in feeding[index:]:
            later_targets |= _bound_names(other)
        if later_targets & _loaded_names(earlier):
            return None
    return feeding


def _plain_assignment_targets(stmt: ast.stmt) -> set[str]:
    """The local names a plain assignment binds, or nothing for anything else.

    Only such a statement may be copied ahead of a guard: an augmented
    assignment or a persistent / series declaration advances state, so a lifted
    statement reads whatever value that state has under its own guard.
    """
    if isinstance(stmt, ast.Assign):
        targets = stmt.targets
    elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
        if any(isinstance(node, ast.Name) and node.id in _STATEFUL_ANNOTATIONS
               for node in ast.walk(stmt.annotation)):
            return set()
        targets = [stmt.target]
    else:
        return set()
    names: set[str] = set()
    for target in targets:
        # Only a bare name: an attribute or a subscript target writes through
        # an object, and a tuple / list / starred one unpacks — ``x, y =
        # items`` raises ``ValueError`` as soon as the guard above it is what
        # gives ``items`` the matching length.
        if not isinstance(target, ast.Name):
            return set()
        names.add(target.id)
    return names


def _as_statements(exprs: list[ast.expr], where: ast.stmt) -> list[ast.stmt]:
    """Expression statements running ``exprs``, located at ``where``."""
    return [ast.copy_location(ast.Expr(value=value), where) for value in exprs]


def _any_sid_write(sids: set[str]) -> _Hit:
    """A target test matching a ``__sec_write__`` call of any of ``sids``."""

    def hit(node: ast.AST) -> bool:
        return _protocol_sid(node, '__sec_write__') in sids

    return hit


def _force_conditional_writes(module: ast.Module, clone: ast.FunctionDef,
                              sids: set[str]) -> bool:
    """Make the writes of ``sids`` run on every bar the child computes.

    TradingView HOISTS a ``request.security()`` call to global scope: the branch
    it is written in decides where its RESULT lands on the chart, never whether
    the requested context's series is computed. The security split leaves the
    ``__sec_write__`` block where the call stood, so a call site inside a branch
    has the CHILD re-evaluate that test against ITS OWN bars and publish on only
    some of them — the chart then reads the previous period's value.
    MEASURED on a 90-minute context of a 30-minute chart gated by
    ``bar_index % 3 != 1``, against an ungated call of the same expression:
    TradingView delivers the same series on every bar for a call in an ``if``,
    in either branch of an ``if`` / ``else`` holding one context each, in a
    ternary, behind ``and``, through a helper in each of those positions, and
    for ``request.security_lower_tf``.

    The clone is the child's own code, so the path is forced here: every write,
    and every call of a helper performing one, is lifted in front of the guards
    standing between it and the clone's body (:class:`_Forcer`), inside the
    helper functions first and then in the clone's own body. Branches the
    EXPRESSION depends on are untouched — they are part of the subtree
    TradingView compiles into the security context.

    A writing helper defined at module level is shared with ``main()``, so the
    clone gets forced copies of the module-level definitions on the call chain
    as nested ``def``s, which shadow the originals inside the clone only.

    :param module: The lowered module.
    :param clone: The emitted clone, modified in place.
    :param sids: The contexts whose write must become unconditional.
    :return: Whether anything was rewritten.
    """
    hit = _any_sid_write(sids)
    outer: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for node in module.body:
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not is_script_entry(node) and not node.name.startswith(CLONE_PREFIX)):
            outer.setdefault(node.name, []).append(node)
    nested = _nested_defs(clone)
    shared = {name: nodes for name, nodes in outer.items() if name not in nested}
    reaching_shared = _reaching_defs(shared, hit)
    memo: dict[int, object] = {id(module): module}
    copies = [copy.deepcopy(node, memo)
              for name in sorted(reaching_shared) for node in shared[name]]

    defs = dict(nested)
    for node in copies:
        defs.setdefault(node.name, []).append(node)
        for name, nodes in _nested_defs(node).items():
            defs.setdefault(name, []).extend(nodes)
    reaching = _reaching_defs(defs, hit)

    def test(node: ast.AST) -> bool:
        if hit(node):
            return True
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in reaching)

    shared_forcer = _Forcer(test, _plain_aliases(list(copies)))
    for node in copies:
        for sub in [node, *[n for nodes in _nested_defs(node).values() for n in nodes]]:
            sub.body = shared_forcer.func_body(sub)
    forcer = _Forcer(test, _plain_aliases([clone]))
    for nodes in nested.values():
        for node in nodes:
            node.body = forcer.func_body(node)
    clone.body = forcer.func_body(clone)
    if shared_forcer.changed:
        clone.body[0:0] = copies
    ast.fix_missing_locations(clone)
    return forcer.changed or shared_forcer.changed


def _build_clone(module: ast.Module, main: ast.FunctionDef, kept: list[int],
                 name: str) -> ast.FunctionDef:
    """Build the clone function from ``main``'s signature and the kept body.

    The clone carries no ``@script.*`` decorator — it is not an entry point the
    runner may discover — but it keeps ``main``'s parameter list so the child
    can call it with no arguments exactly like ``main``. The defaults the clone
    RUNS with are bound from ``main`` afterwards (see :func:`_bind_defaults`);
    the copied default expressions only stay in the signature so the lowering
    passes see the same ``input`` calls in it that ``main`` has.

    Only the KEPT statements are copied, and the copy stops at the module:
    ``ModulePropertyTransformer`` leaves a ``parent`` back-reference on every
    node, so a copy that walked out of the slice would drag the whole module
    behind it — including the clones emitted before this one, which would make
    the emission quadratic. Seeding the memo with the module maps that edge
    onto the module itself, and sharing one memo across the statements of a
    clone copies the objects hanging off them once.
    """
    memo: dict[int, object] = {id(module): module}
    clone = copy.copy(main)
    clone.name = name
    clone.decorator_list = []
    clone.args = copy.deepcopy(main.args, memo)
    clone.returns = copy.deepcopy(main.returns, memo)
    # PEP 695 type parameters exist from Python 3.12 on; a Pyne script never
    # has any, but the field must not stay shared with ``main`` where it does.
    type_params = getattr(main, 'type_params', None)
    if type_params is not None:
        clone.type_params = copy.deepcopy(type_params, memo)
    body = [copy.deepcopy(main.body[index], memo) for index in kept]
    clone.body = body or [ast.Pass()]
    ast.copy_location(clone, main)
    ast.fix_missing_locations(clone)
    return clone


def _bind_defaults(main_name: str, clone_name: str) -> list[ast.stmt]:
    """Statements rebinding the clone's parameter defaults to ``main``'s.

    The clone's signature carries a COPY of ``main``'s default expressions, and
    those are ``input.*()`` calls: re-evaluating them at the clone's ``def``
    yields the source defaults, not the configured values. ``script``'s
    decorator runs at ``main``'s ``def`` and clears the loaded overrides
    (``core/script.py``), so by the time the clone is defined they are gone. The
    values ``main`` was bound with are the configured ones, and handing the very
    same tuple to the clone is what makes the child compute what the chart does.
    """
    def place(name: str, attr: str, ctx: ast.expr_context) -> ast.expr:
        # The type pass is behind us, so every emitted expression carries its
        # own stamp; a function object's dunder is opaque runtime state.
        return stamp_lowering(
            ast.Attribute(value=ast.Name(id=name, ctx=ast.Load()), attr=attr, ctx=ctx),
            OBJECT)

    def assign(attr: str) -> ast.stmt:
        return ast.Assign(targets=[place(clone_name, attr, ast.Store())],
                          value=place(main_name, attr, ast.Load()))

    stmts = [assign('__defaults__'), assign('__kwdefaults__')]
    for stmt in stmts:
        ast.fix_missing_locations(stmt)
    return stmts
