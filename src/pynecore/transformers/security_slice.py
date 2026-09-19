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
``PYNE_NO_SECURITY_SLICE=1`` switches the emission off; the flag is mixed into
the transform pipeline digest, so chart and child can never load bytecode built
under the other setting.
"""
import ast
import copy
from collections.abc import Callable

from ..core.import_hook import security_slice_disabled
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
    :ivar skipped: per sid, why that context got no clone
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
        :return: per sid, the ``main()`` statements its clone kept (absent for a
            context that got no clone — its child runs the whole ``main()``)
        """
        if security_slice_disabled():
            self.module_skip = 'disabled'
            return {}
        contexts = _find_contexts(node)
        main = _find_main(node)
        if contexts is None or main is None:
            self.module_skip = 'no_contexts' if main is not None else 'no_main'
            return {}

        scopes = _SliceScopes(node)
        main_key = scopes.scope_key_of(main)
        if main_key is None:
            self.module_skip = 'no_main_scope'
            return {}
        reachable = _reachable_scopes(scopes, main_key)
        for key in reachable:
            found = _unmodelled_kind(scopes.scope_node(key))
            if found is not None:
                self.module_skip = f'unmodelled:{found}:{key or "<module>"}'
                return {}
            if _unresolved_collection(scopes.scope_node(key)):
                self.module_skip = f'unresolved_collection:{key or "<module>"}'
                return {}

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
            return {}
        written_sids: set[str] = set()
        for entry in index_of_stmt:
            written_sids |= entry.write_sids

        entries = list(enumerate(_iter_contexts(contexts)))
        units = _slice_units(entries, written_sids, self.skipped)

        clones: list[ast.stmt] = []
        sliced: dict[str, list[ast.stmt]] = {}
        for unit in units:
            keep_reads: set[str] = set()
            for _index, sid, ctx in unit:
                keep_reads |= {sid} | _sid_list(ctx, 'depends') | _sid_list(ctx, 'late_reads')
            write_sids = {sid for _index, sid, _ctx in unit}
            kept = _slice(index_of_stmt, write_sids, keep_reads)
            if kept is None:
                for _index, sid, _ctx in unit:
                    self.skipped[sid] = 'nothing_dropped'
                continue
            # The unit's first member names the clone: its index is its own,
            # so no other unit can pick the same name.
            name = f'{CLONE_PREFIX}{unit[0][0]}{CLONE_SUFFIX}'
            clones.append(_build_clone(node, main, kept, name))
            clones.extend(_bind_defaults(main.name, name))
            for _index, _sid, ctx in unit:
                sliced[_sid] = [main.body[index] for index in kept]
                # The type pass has already run, so the two new literals are
                # stamped here — an unstamped node inside a stamped subtree
                # loses the type of everything above it.
                ctx.keys.append(stamp_lowering(ast.Constant(value='slice_main'), STR))
                ctx.values.append(stamp_lowering(ast.Constant(value=name), STR))

        if clones:
            insert_at = node.body.index(main) + 1
            node.body[insert_at:insert_at] = clones
        return sliced


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
    ``BoolOp``: a target in either of them is evaluated only for some values of
    the test, so it is guarded. A target standing in a call argument or in an
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
    if isinstance(node, (ast.BoolOp, ast.Lambda, ast.ListComp, ast.SetComp,
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
    if not isinstance(stmt, (ast.Expr, ast.Assign, ast.AnnAssign, ast.AugAssign)):
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
