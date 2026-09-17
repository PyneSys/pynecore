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

from ..core.import_hook import security_slice_disabled
from .dynamic_default import is_script_entry
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

#: The protocol calls the security split injects. None of them touches a name
#: of the script, so they carry no data dependency of their own.
_PROTOCOL_CALLS = frozenset({
    '__sec_read__', '__sec_write__', '__sec_signal__', '__sec_wait__',
    '__ltf_unzip__',
})

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
        if security_slice_disabled():
            self.module_skip = 'disabled'
            return node
        contexts = _find_contexts(node)
        main = _find_main(node)
        if contexts is None or main is None:
            self.module_skip = 'no_contexts' if main is not None else 'no_main'
            return node

        scopes = _SliceScopes(node)
        main_key = scopes.scope_key_of(main)
        if main_key is None:
            self.module_skip = 'no_main_scope'
            return node
        reachable = _reachable_scopes(scopes, main_key)
        for key in reachable:
            found = _unmodelled_kind(scopes.scope_node(key))
            if found is not None:
                self.module_skip = f'unmodelled:{found}:{key or "<module>"}'
                return node
            if _unresolved_collection(scopes.scope_node(key)):
                self.module_skip = f'unresolved_collection:{key or "<module>"}'
                return node

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
            return node
        written_sids: set[str] = set()
        for entry in index_of_stmt:
            written_sids |= entry.write_sids

        entries = list(enumerate(_iter_contexts(contexts)))
        units = _slice_units(entries, written_sids, self.skipped)

        clones: list[ast.stmt] = []
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
                # The type pass has already run, so the two new literals are
                # stamped here — an unstamped node inside a stamped subtree
                # loses the type of everything above it.
                ctx.keys.append(stamp_lowering(ast.Constant(value='slice_main'), STR))
                ctx.values.append(stamp_lowering(ast.Constant(value=name), STR))

        if clones:
            insert_at = node.body.index(main) + 1
            node.body[insert_at:insert_at] = clones
        return node


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
