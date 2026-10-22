"""
Which security contexts' results flow into which other context's write: the
taint analysis the security pass runs over the module it has lowered (see
:class:`_DependencyAnalyzer`), and whose lexical half the security slice pass
reuses.
"""
import ast
from collections.abc import Iterator
from typing import Any

from . import ast_walk
from .emission_settings import PYNE_RESERVED_NAME_CHAR
from .factory_fields import FactoryFields


# Statement / expression node types the dependency analyzer does not model.
# Any scope containing one of them falls back to "depends on every other sid"
# (a safe over-approximation: an unmodelled construct can only add flow).
_UNMODELLED_NODES: tuple[type[ast.AST], ...] = tuple(
    n for n in (
        getattr(ast, name, None) for name in (
            'Try', 'TryStar', 'With', 'AsyncWith', 'AsyncFor', 'Raise', 'Delete',
            'ClassDef', 'Lambda', 'ListComp', 'SetComp',
            'DictComp', 'GeneratorExp', 'Yield', 'YieldFrom', 'Await', 'Assert',
        )
    ) if n is not None
)

# Expression node types whose taint is simply the union of their children's.
_UNION_EXPRS: tuple[type[ast.AST], ...] = (
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Tuple,
    ast.List, ast.Set, ast.Dict, ast.Slice, ast.Starred, ast.JoinedStr,
    ast.FormattedValue, ast.Subscript,
)

# Names that can never carry taint: ``lib`` is the library namespace root, so
# ``lib.plot(tainted)`` must not make every later ``lib.close`` tainted.
_UNTAINTABLE_ROOTS = frozenset({'lib'})

# ``lib`` namespaces holding the collection APIs (``array.push(id, v)``,
# ``matrix.set(id, r, c, v)``, ``map.put(id, k, v)``, ...). Only a call in one
# of these can write taint back into the collection passed as its first
# argument — doing it for every call would make an ordinary
# ``ta.sma(series, length)`` taint its length input.
_COLLECTION_NAMESPACES = frozenset({'array', 'matrix', 'map'})

# Collection functions that only READ their collection. They derive a value
# from it and leave it unchanged, so an index, key or search argument of theirs
# must not become part of the collection's taint: ``array.get(store, i)`` does
# not put ``i`` into ``store``. The index still reaches the call's RESULT, so
# a value read out under a tainted index stays tainted.
_COLLECTION_READERS: dict[str, frozenset[str]] = {
    'array': frozenset({
        'abs', 'avg', 'binary_search', 'binary_search_leftmost',
        'binary_search_rightmost', 'copy', 'covariance', 'every', 'first',
        'from_items', 'get', 'includes', 'indexof', 'join', 'last',
        'lastindexof', 'max', 'median', 'min', 'mode', 'new', 'new_bool',
        'new_box', 'new_color', 'new_float', 'new_int', 'new_label',
        'new_line', 'new_linefill', 'new_string', 'new_table',
        'percentile_linear_interpolation', 'percentile_nearest_rank',
        'percentrank', 'range', 'size', 'some', 'sort_indices', 'standardize',
        'stdev', 'sum', 'variance',
    }),
    'matrix': frozenset({
        'all', 'avg', 'col', 'columns', 'copy', 'det', 'diff', 'eigenvalues',
        'eigenvectors', 'elements_count', 'get', 'inv', 'is_antidiagonal',
        'is_antisymmetric', 'is_binary', 'is_diagonal', 'is_identity',
        'is_square', 'is_stochastic', 'is_symmetric', 'is_triangular',
        'is_zero', 'kron', 'max', 'median', 'min', 'mode', 'mult', 'new',
        'pinv', 'pow', 'rank', 'row', 'rows', 'submatrix', 'sum', 'trace',
        'transpose',
    }),
    'map': frozenset({'contains', 'copy', 'get', 'keys', 'new', 'size', 'values'}),
}

# Nodes whose children may be skipped on a pass through the code around them:
# branches, handlers, short-circuit operands, comprehensions and nested loops
# (which can run zero times).
_CONDITIONAL_NODES: tuple[type[ast.AST], ...] = (
    ast.If, ast.IfExp, ast.BoolOp, ast.Try, ast.TryStar, ast.Match,
    ast.For, ast.AsyncFor, ast.While,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
)

# Fixpoint safety net — the lattice is finite and monotone, so the loop always
# converges; the cap only guards against an unforeseen non-monotone edit.
_MAX_FIXPOINT_ROUNDS = 200

# Prefix marking a symbolic parameter token inside a taint set. A call site
# replaces the callee's own tokens with that call's argument taints, so two
# calls of one function never share their arguments' taint.
_PARAM_PREFIX = '\x00param\x00'

# Pseudo-parameter name carrying the control taint a call site runs under —
# the caller's ``if`` condition decides whether the callee's writes and
# mutations happen at all, so it is substituted like any other argument.
_CTRL_PARAM = '<ctrl>'

# Change-tracking cells of a function's summaries (see
# ``_DependencyAnalyzer._visit_scope_body``), followed by its scope key. A
# name's taint class is tracked under its root name, a call site binding under
# its parameter token; neither can start with these.
_RET_CELL = '\x01ret\x00'
_OUT_CELL = '\x01out\x00'
_ALIAS_CELL = '\x01alias\x00'


# What :meth:`_DependencyAnalyzer._collect_loop_names` does with a node, by its
# class
_LOOP_NAMES_OTHER = 0  # nothing of its own: visit its children
_LOOP_NAMES_LEAF = 1  # a node that never has a child
_LOOP_NAMES_NAME = 2
_LOOP_NAMES_IMPORT = 3
_LOOP_NAMES_DECLARATION = 4
_LOOP_NAMES_CAPTURE = 5
_LOOP_NAMES_MAPPING = 6
_LOOP_NAMES_FUNCTION = 7
_LOOP_NAMES_ASSIGN = 8
_LOOP_NAMES_ANN_ASSIGN = 9
_LOOP_NAMES_AUG_ASSIGN = 10
_LOOP_NAMES_CONDITIONAL = 11
_LOOP_NAMES_LOOP = 12

_loop_name_kinds: dict[type, int] = {}


def _loop_name_kind(cls: type) -> int:
    """The ``_collect_loop_names`` kind of a node class, decided once per class."""
    kind = _loop_name_kinds.get(cls)
    if kind is not None:
        return kind
    if issubclass(cls, ast.Name):
        kind = _LOOP_NAMES_NAME
    elif issubclass(cls, (ast.Import, ast.ImportFrom)):
        kind = _LOOP_NAMES_IMPORT
    elif issubclass(cls, (ast.Global, ast.Nonlocal)):
        kind = _LOOP_NAMES_DECLARATION
    elif issubclass(cls, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
        kind = _LOOP_NAMES_CAPTURE
    elif issubclass(cls, ast.MatchMapping):
        kind = _LOOP_NAMES_MAPPING
    elif issubclass(cls, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        kind = _LOOP_NAMES_FUNCTION
    elif issubclass(cls, ast.Assign):
        kind = _LOOP_NAMES_ASSIGN
    elif issubclass(cls, ast.AnnAssign):
        kind = _LOOP_NAMES_ANN_ASSIGN
    elif issubclass(cls, ast.AugAssign):
        kind = _LOOP_NAMES_AUG_ASSIGN
    elif issubclass(cls, (ast.For, ast.AsyncFor, ast.While)):
        kind = _LOOP_NAMES_LOOP
    elif issubclass(cls, _CONDITIONAL_NODES):
        kind = _LOOP_NAMES_CONDITIONAL
    elif issubclass(cls, (ast.expr_context, ast.operator, ast.unaryop, ast.cmpop, ast.boolop,
                          ast.Constant)):
        kind = _LOOP_NAMES_LEAF
    else:
        kind = _LOOP_NAMES_OTHER
    _loop_name_kinds[cls] = kind
    return kind


# What :meth:`_DependencyAnalyzer._scan_scope` does with a node, by its class
_SCAN_ENTER = 0  # nothing of its own: walk its children
_SCAN_LEAF = 1  # a node that never has a child
_SCAN_NAME = 2
_SCAN_IMPORT = 3
_SCAN_GLOBAL = 4
_SCAN_NONLOCAL = 5
_SCAN_DEF = 6
_SCAN_CALL = 7
_SCAN_LOOP = 8
_SCAN_UNMODELLED = 9
_SCAN_UNMODELLED_LOOP = 10

_scan_kinds: dict[type, int] = {}


def _scan_kind(cls: type) -> int:
    """The scan kind of a node class, decided once per class."""
    kind = _scan_kinds.get(cls)
    if kind is not None:
        return kind
    if issubclass(cls, ast.Name):
        kind = _SCAN_NAME
    elif issubclass(cls, (ast.Import, ast.ImportFrom)):
        kind = _SCAN_IMPORT
    elif issubclass(cls, ast.Global):
        kind = _SCAN_GLOBAL
    elif issubclass(cls, ast.Nonlocal):
        kind = _SCAN_NONLOCAL
    elif issubclass(cls, (ast.FunctionDef, ast.AsyncFunctionDef)):
        kind = _SCAN_DEF
    elif issubclass(cls, (ast.For, ast.AsyncFor, ast.While)):
        kind = _SCAN_UNMODELLED_LOOP if issubclass(cls, _UNMODELLED_NODES) else _SCAN_LOOP
    elif issubclass(cls, _UNMODELLED_NODES):
        kind = _SCAN_UNMODELLED
    elif issubclass(cls, ast.Call):
        kind = _SCAN_CALL
    elif issubclass(cls, (ast.expr_context, ast.operator, ast.unaryop, ast.cmpop, ast.boolop,
                          ast.Constant)):
        kind = _SCAN_LEAF
    else:
        kind = _SCAN_ENTER
    _scan_kinds[cls] = kind
    return kind


class _Scope:
    """One lexical scope of the analysed module (the module or a function).

    :ivar key: unique dotted path of the scope (``''`` for the module)
    :ivar node: the ``Module`` / ``FunctionDef`` the scope belongs to
    :ivar parent: key of the enclosing scope, ``None`` for the module
    :ivar locals: names bound inside the scope (parameters, assignment
        targets, nested ``def`` names) that are not declared ``global`` /
        ``nonlocal``
    :ivar globals: names the scope declares ``global``
    :ivar funcs: nested function name to scope key
    :ivar body: the scope's statement list
    :ivar calls: the direct-``Name`` calls of the scope's own code (nested
        function definitions excluded) in evaluation order — the arguments
        before the call — as ``(callee name, call, inside a loop)``. The whole
        ``for`` / ``while`` subtree counts as inside the loop: its iterator
        expression runs once, but counting it in only over-approximates the
        ``in_loop`` marking
    """

    __slots__ = ('key', 'node', 'parent', 'locals', 'globals', 'funcs', 'body', 'calls')

    def __init__(self, key: str, node: ast.AST, parent: str | None,
                 body: list[ast.stmt]):
        self.key = key
        self.node = node
        self.parent = parent
        self.locals: set[str] = set()
        self.globals: set[str] = set()
        self.funcs: dict[str, str] = {}
        self.body = body
        self.calls: list[tuple[str, ast.Call, bool]] = []


class _DependencyAnalyzer:
    """Forward taint analysis over an already lowered security AST.

    After :class:`~pynecore.transformers.security.SecurityTransformer` has
    rewritten the module, every ``request.security()`` result is read through
    ``__sec_read__("<sid>", ...)`` and every security expression is written
    through ``__sec_write__("<sid>", <expr>)``. This pass answers one question:
    which sids' results flow into which other sid's write expression.

    What the analysis models:

    - **Flow-insensitive on data**: a name's taint is the union over every
      assignment to it anywhere in its scope (so ``var`` backward flow and
      loop-carried values are covered without ordering rules). The exception
      is a pure loop variable — a name every occurrence of which sits inside a
      loop that writes it before reading it: it holds nothing between those
      loops, so each of them gets a cell of its own instead of one counter
      tying every loop of the scope together.
    - **Lexically scoped names**: a name belongs to the scope that binds it —
      parameters and assignment targets are local, everything else resolves
      outwards, so a nested ``def`` reading a ``main`` local (the usual closure
      in a Pyne script) shares that local's taint while two same-named locals
      of unrelated functions do not. A ``global`` / ``nonlocal`` declaration
      rebinds the name to the module / the enclosing scope that owns it.
    - **Flow-sensitive on control**: a control-taint set accumulates the taint
      of enclosing ``if`` / ``for`` / ``while`` / ``match`` conditions, and a
      tainted ``return`` / ``break`` / ``continue`` taints what runs AFTER it —
      the rest of the block it escapes from, and, through the enclosing
      statement, everything following that. Statements lexically before the
      escape are untouched: leaving a block cannot hand a value backwards
      inside one bar, and a value a later iteration or a later bar carries back
      through a ``var`` is settled by the back-edge filter below.
    - **Collections**: a call in the ``array`` / ``matrix`` / ``map``
      namespace writes back into the collection passed as its first argument.
      A reader writes nothing — an index only reaches the value it returns —
      and a mutator writes back what it stores, not where it stores it. A
      function neither table knows stays conservatively mutating.
    - **Alias classes**: ``a = b`` merges the two names into one taint class,
      so mutating a container through either alias is visible on both. A
      call whose callee returns one of its parameters (or an alias of it)
      merges the receiving name with that argument the same way.
    - **User functions, per call site**: each parameter carries a symbolic
      token instead of a concrete sid. A call's result is the callee's return
      taint with its own tokens replaced by this call's argument taints, so an
      argument only reaches the result when the callee's return actually draws
      on it, and two calls of one function do not mix. Taint the callee adds to
      a parameter (a mutated array, matrix or map) flows back into the caller's
      argument variable the same way. The control taint of the call site is a
      pseudo-parameter: a mutation or write the callee performs happens only if
      the caller reached the call, so it carries the caller's condition too. A
      call whose callee is not a modelled
      function keeps the conservative rule: every argument flows into the
      result. Tokens that survive into a write inside a callee are resolved
      against the union of that parameter's call sites, and a function that is
      called somewhere the analysis could not model resolves to every sid.
    - **Fallback**: a scope containing an unmodelled construct makes every
      write in it — and every call of it — depend on all other sids.

    Over-approximation is safe only towards EARLIER-sited producers, and the
    result is filtered to those: for them ``depends`` is just a wait filter, so
    a too-large set costs extra synchronisation and nothing else. A back-edge
    onto a later-sited producer is NOT safe — the child would block a write
    that the chart is waiting for before it ever reaches the producer's site —
    so every such edge is dropped from ``depends`` and kept in ``late_reads``
    (see :meth:`run`).
    """

    def __init__(self, module: ast.Module, all_sids: list[str]):
        self._module = module
        self._all: frozenset[str] = frozenset(all_sids)
        self.depends: dict[str, set[str]] = {sid: set() for sid in all_sids}
        self.in_loop: set[str] = set()
        self.late_reads: dict[str, set[str]] = {}
        self._funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        self._scopes: dict[str, _Scope] = {}
        self._key_of: dict[int, str] = {}
        self._called_names: set[str] = set()
        self._bound: set[str] = set()
        self._fallback_scopes: set[str] = set()
        self.fallback_reasons: dict[str, str] = {}
        # Per scope key, the unmodelled constructs of the scope's own code in
        # walk order, and the keys of the scopes whose own code has a loop,
        # both gathered by :meth:`_build_scopes`
        self._unmodelled: dict[str, list[ast.AST]] = {}
        self._loop_scopes: set[str] = set()
        self._ret: dict[str, set[str]] = {}
        self._ret_alias: dict[str, set[str]] = {}
        self._param_out: dict[str, dict[str, set[str]]] = {}
        self._param_in: dict[str, set[str]] = {}
        self._escaped: set[str] = set()
        self._broke: set[str] = set()
        self._parent: dict[str, str] = {}
        self._taint: dict[str, set[str]] = {}
        self._loop_cells: dict[int, dict[str, str]] = {}
        # Memo of :meth:`_foreign_uses`, by function node id
        self._foreign_memo: dict[int, dict[str, int]] = {}
        self._name_cells: dict[str, str] = {}
        self._ctrl: set[str] = set()
        self._order: dict[str, int] = {}
        self._scope: str = ''
        self._changed = False
        self._final = False
        # Change tracking of the fixpoint rounds (see :meth:`_visit_scope_body`):
        # a counter bumped on every change, the counter value of each cell's
        # last change, the cells the statement being visited has read so far
        # (``None`` outside a scope's body), and per statement id ``(counter
        # before its last visit, ctrl in, ctrl out, ``_ctrl`` left behind,
        # cells read)``
        self._tick = 0
        self._stamps: dict[str, int] = {}
        self._reads: set[str] | None = None
        self._memo: dict[int, tuple[int, set[str], set[str], set[str], set[str]]] = {}
        # Number of ``return`` / ``break`` / ``continue`` statements visited
        self._escapes = 0

    # --- scopes ---

    @staticmethod
    def _params(fn: 'ast.FunctionDef | ast.AsyncFunctionDef') -> list[ast.arg]:
        spec = fn.args
        params = list(spec.posonlyargs) + list(spec.args) + list(spec.kwonlyargs)
        if spec.vararg is not None:
            params.append(spec.vararg)
        if spec.kwarg is not None:
            params.append(spec.kwarg)
        return params

    def _build_scopes(self) -> None:
        """Build the lexical scope tree and each scope's set of bound names."""
        root = _Scope('', self._module, None, self._module.body)
        self._scopes[''] = root
        self._key_of[id(self._module)] = ''
        work: list[_Scope] = [root]
        used: set[str] = {''}
        while work:
            scope = work.pop()
            if isinstance(scope.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scope.locals.update(p.arg for p in self._params(scope.node))
            funcs, declared = self._scan_scope(scope)
            for fn in funcs:
                scope.locals.add(fn.name)
                key = (scope.key + '.' if scope.key else '') + fn.name
                while key in used:
                    key += '~'
                used.add(key)
                child = _Scope(key, fn, scope.key, fn.body)
                self._scopes[key] = child
                self._key_of[id(fn)] = key
                scope.funcs[fn.name] = key
                work.append(child)
            scope.locals -= declared

    def _scan_scope(self, scope: _Scope) -> tuple[
            list[ast.FunctionDef | ast.AsyncFunctionDef], set[str]]:
        """Walk the scope's own code once, gathering everything kept per scope.

        The walk covers every descendant of the scope's node except the inside
        of a nested function definition, which is a scope of its own; together
        the scopes walk every node of the module exactly once. On the way it
        records the scope's bound names, ``global`` names, loops, unmodelled
        constructs (in walk order) and direct-``Name`` calls (see
        :attr:`_Scope.calls`).

        :param scope: the scope to walk
        :return: the function definitions the scope owns, in walk order, and
            the names it declares ``global`` / ``nonlocal``
        """
        funcs: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        declared: set[str] = set()
        unmodelled: list[ast.AST] = []
        bound = scope.locals
        calls = scope.calls
        kinds = _scan_kinds
        children_of = ast_walk.iter_child_nodes
        loops = 0
        # Per node being walked: its remaining children, the node itself when
        # it is a direct-``Name`` call — recorded once its children are done,
        # which puts the arguments first — and whether it is a loop. The kind
        # check stands in for the type narrowing, hence ``Any``
        stack: list[tuple[Iterator[Any], Any, bool]] = [
            (children_of(scope.node), None, False)]
        while stack:
            children, call, loop = stack[-1]
            for sub in children:
                kind = kinds.get(sub.__class__)
                if kind is None:
                    kind = _scan_kind(sub.__class__)
                if kind == _SCAN_ENTER:
                    stack.append((children_of(sub), None, False))
                    break
                if kind == _SCAN_LEAF:
                    continue
                if kind == _SCAN_NAME:
                    # Its only child is the context
                    if isinstance(sub.ctx, ast.Store):
                        bound.add(sub.id)
                    continue
                if kind == _SCAN_CALL:
                    named = isinstance(sub.func, ast.Name)
                    stack.append((children_of(sub), sub if named else None, False))
                    break
                if kind == _SCAN_IMPORT:
                    for alias in sub.names:
                        bound.add(alias.asname or alias.name.split('.')[0])
                    continue
                if kind == _SCAN_GLOBAL:
                    scope.globals.update(sub.names)
                    declared.update(sub.names)
                    continue
                if kind == _SCAN_NONLOCAL:
                    declared.update(sub.names)
                    continue
                if kind == _SCAN_DEF:
                    funcs.append(sub)
                    continue
                if kind != _SCAN_LOOP:
                    unmodelled.append(sub)
                if kind != _SCAN_UNMODELLED:
                    self._loop_scopes.add(scope.key)
                    loops += 1
                    stack.append((children_of(sub), None, True))
                else:
                    stack.append((children_of(sub), None, False))
                break
            else:
                stack.pop()
                if call is not None:
                    calls.append((call.func.id, call, loops > 0))
                if loop:
                    loops -= 1
        if unmodelled:
            self._unmodelled[scope.key] = unmodelled
        return funcs, declared

    def _q(self, name: str, scope_key: str | None = None) -> str:
        """Qualify a bare name with the key of the scope that binds it."""
        if scope_key is None:
            cell = self._name_cells.get(name)
            if cell is not None:
                return cell
        key = self._scope if scope_key is None else scope_key
        while True:
            scope = self._scopes.get(key)
            if scope is None:
                return '\x00' + name
            if name in scope.globals:
                return '\x00' + name
            if name in scope.locals or scope.parent is None:
                return key + '\x00' + name
            key = scope.parent

    def _scope_of(self, fn: ast.FunctionDef | ast.AsyncFunctionDef) -> _Scope:
        """The scope of a function definition of the module."""
        return self._scopes[self._key_of[id(fn)]]

    def _lookup_func(self, name: str) -> str | None:
        """Scope key of the user function ``name`` resolves to, if any."""
        key: str | None = self._scope
        while key is not None:
            scope = self._scopes.get(key)
            if scope is None:
                break
            if name in scope.funcs:
                return scope.funcs[name]
            key = scope.parent
        fn = self._funcs.get(name)
        return self._key_of.get(id(fn)) if fn is not None else None

    def _touch(self, cell: str) -> None:
        """Record that ``cell`` of the analysis state has just grown."""
        self._tick += 1
        self._stamps[cell] = self._tick
        self._changed = True

    # --- alias classes ---

    def _find(self, name: str) -> str:
        root = name
        while self._parent.get(root, root) != root:
            root = self._parent[root]
        # Path compression
        while self._parent.get(name, name) != name:
            self._parent[name], name = root, self._parent[name]
        return root

    def _union(self, a: str, b: str) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            return
        self._parent[rb] = ra
        self._taint[ra] = self._taint.get(ra, set()) | self._taint.pop(rb, set())
        # A merge is a change even when neither class gains taint: a later add
        # to one alias now reaches the other, and ``_record_ret_alias`` compares
        # classes, not taints. Each pair of classes merges once, so this cannot
        # keep the fixpoint loop from converging. A statement that read either
        # root has to look again.
        self._touch(ra)
        self._stamps[rb] = self._tick

    @staticmethod
    def _bare(name: str) -> str:
        """The bare name of a scope-qualified name."""
        return name.rpartition('\x00')[2]

    def _get_taint(self, name: str) -> set[str]:
        if self._bare(name) in _UNTAINTABLE_ROOTS:
            return set()
        root = self._find(name)
        if self._reads is not None:
            self._reads.add(root)
        return set(self._taint.get(root, ()))

    def _add_taint(self, name: str, taint: set[str]) -> None:
        if self._bare(name) in _UNTAINTABLE_ROOTS:
            return
        if not taint:
            self._taint.setdefault(self._find(name), set())
            return
        root = self._find(name)
        cur = self._taint.setdefault(root, set())
        if not taint <= cur:
            cur |= taint
            self._touch(root)

    # --- parameter tokens ---

    @staticmethod
    def _token(qualified_param: str) -> str:
        return _PARAM_PREFIX + qualified_param

    @classmethod
    def _ctrl_token(cls, scope_key: str) -> str:
        """Token standing for the control taint a call of ``scope_key`` runs under."""
        return cls._token(scope_key + '\x00' + _CTRL_PARAM)

    @staticmethod
    def _subst(taint: set[str], token_map: dict[str, set[str]]) -> set[str]:
        """Replace this call's parameter tokens with the argument taints.

        Tokens the map does not name belong to an enclosing function (or to a
        parameter this call left at its default) and are carried through
        unchanged — :meth:`_resolve_tokens` settles them later.
        """
        out: set[str] = set()
        for item in taint:
            if item.startswith(_PARAM_PREFIX):
                mapped = token_map.get(item)
                if mapped is None:
                    out.add(item)
                else:
                    out |= mapped
            else:
                out.add(item)
        return out

    def _resolve_tokens(self, taint: set[str]) -> set[str]:
        """Turn a taint set into plain sids by expanding parameter tokens.

        A token expands to the union of the argument taints seen at the
        callee's call sites. A function the module calls somewhere the analysis
        never reached has no recorded call site, so its tokens expand to every
        sid instead of to nothing — but only in the final rounds
        (``_final``): during the first round a callee visited ahead of its
        call site merely has not been bound YET, and expanding it to every sid
        then would pin a spurious dependency that no later round can retract.
        """
        out: set[str] = set()
        seen: set[str] = set()
        work = list(taint)
        while work:
            item = work.pop()
            if not item.startswith(_PARAM_PREFIX):
                out.add(item)
                continue
            if item in seen:
                continue
            seen.add(item)
            if self._reads is not None:
                self._reads.add(item)
            if self._final and self._of_unreached_call(item):
                out |= self._all
                continue
            work.extend(self._param_in.get(item, ()))
        return out

    def _of_unreached_call(self, token: str) -> bool:
        """Whether ``token`` belongs to a function the module calls only where
        the analysis never reached (see :meth:`_resolve_tokens`)."""
        key = token[len(_PARAM_PREFIX):].rpartition('\x00')[0]
        scope = self._scopes.get(key)
        return (key not in self._bound and scope is not None
                and isinstance(scope.node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and scope.node.name in self._called_names)

    # --- helpers ---

    @staticmethod
    def _sid_arg(call: ast.Call) -> str | None:
        """Return the sid string of a ``__sec_read__`` / ``__sec_write__`` call."""
        if not call.args:
            return None
        first = call.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
        return None

    @staticmethod
    def _root_name(node: ast.expr) -> str | None:
        """Base Name of a place expression (``a``, ``a.b``, ``a[i].c``)."""
        while isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        return node.id if isinstance(node, ast.Name) else None

    @staticmethod
    def _is_plain_record(node: ast.ClassDef, factories: FactoryFields,
                         class_names: frozenset[str]) -> bool:
        """Whether a class body only declares fields — a Pine ``type``.

        Such a body runs once, when the module is imported, before any security
        value exists, so it carries no taint and needs no fallback. Creating an
        instance is an ordinary call and assigning a field an ordinary attribute
        store, and both are modelled where they happen. Anything else in the
        body — a method above all — keeps the class unmodelled.

        A field default the compiler lowered to a ``default_factory`` lambda
        runs at every construction instead, so it only qualifies while its body
        reads nothing a script can bind: the lowering's reserved-name helpers,
        ``lib``, and the module's own top-level class names — a field typed as
        another record defaults to ``na(<that class>)``, and a class name is an
        import-time constant binding that cannot carry security taint.

        :param node: the class definition
        :param factories: the module's compiler-emitted field factories
        :param class_names: names of the module's top-level classes
        :return: whether the body holds nothing but field declarations
        """
        factory_calls = {id(call) for call in factories.of(node)}
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign):
                lambdas = [sub for sub in ast_walk.walk(stmt) if isinstance(sub, ast.Lambda)]
                if not lambdas:
                    continue
                value = stmt.value
                if (len(lambdas) != 1 or value is None or id(value) not in factory_calls
                        or any(isinstance(sub, ast.Name)
                               and PYNE_RESERVED_NAME_CHAR not in sub.id
                               and sub.id != 'lib'
                               and sub.id not in class_names
                               for sub in ast_walk.walk(lambdas[0].body))):
                    return False
                continue
            if isinstance(stmt, ast.Pass):
                continue
            if (isinstance(stmt, ast.Assign)
                    and all(isinstance(t, ast.Name) for t in stmt.targets)):
                continue
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue
            return False
        return True

    # --- program order of the write sites ---

    def _compute_site_order(self) -> None:
        """Number every sid by the program order of its ``__sec_write__`` site.

        The walk starts at the module body and at every function nothing calls
        (``main`` and friends), and descends into a user function AT ITS CALL
        SITE — so a write inside a nested or cloned function takes the position
        of the call that instantiates it. Recursion is cut with a visiting set.

        Sites the walk never reaches (dead code) are numbered last, in
        registration order.

        A function is walked at its FIRST call site only. A later walk of it
        could not number anything: the first one numbered every site reachable
        from it except through the functions then being walked, and each of
        those either is still being walked (and cut again) or has since
        numbered what it reaches. That keeps the walk linear in the module
        instead of re-walking a helper at every call of it.
        """
        counter = 0
        visiting: set[str] = set()
        walked: set[str] = set()

        def walk(scope: _Scope) -> None:
            nonlocal counter
            # The calls stand in evaluation order: arguments before their call
            for name, call, _in_loop in scope.calls:
                if name == '__sec_write__':
                    sid = self._sid_arg(call)
                    if sid is not None and sid not in self._order:
                        self._order[sid] = counter
                        counter += 1
                elif name in self._funcs and name not in visiting and name not in walked:
                    visiting.add(name)
                    walk(self._scope_of(self._funcs[name]))
                    visiting.discard(name)
                    walked.add(name)

        walk(self._scopes[''])
        for name, fn in self._funcs.items():
            if name not in self._called_names:
                walk(self._scope_of(fn))
        for sid in self._all:
            if sid not in self._order:
                self._order[sid] = counter
                counter += 1

    # --- loop variables ---

    def _mark_private_loop_vars(self) -> None:
        """Give each loop its own taint cell for the variables only it uses.

        A name has ONE taint cell per scope, so the counter of a loop running
        over a tainted collection would carry that taint into every other loop
        of the same scope that reuses the name — ``i`` usually runs every loop
        of a script. A name qualifies as a loop variable of its scope when
        every occurrence of it sits inside a loop that writes it before reading
        it, in EVALUATION order: it then holds nothing between the loops, and
        each of them can keep a cell of its own. A read anywhere else, a
        nested loop rebinding the same name, a read-modify-write
        (``acc = acc + 1``, ``acc += 1``) carrying the previous loop's value,
        or a read from a nested function — which is visited as its own scope
        and cannot see the private cell — keeps the single scope-wide cell.
        """
        for key, scope in self._scopes.items():
            if key not in self._loop_scopes:
                # No loop of its own: nothing to give a cell to
                continue
            uses, loops, parents = self._collect_loop_names(scope)
            binders: dict[str, list[int]] = {}
            for loop_id, names in loops.items():
                for name in names:
                    binders.setdefault(name, []).append(loop_id)
            index = 0
            for name, own in binders.items():
                if not self._is_loop_var(name, uses.get(name, 0), own,
                                          loops, parents):
                    continue
                for loop_id in own:
                    index += 1
                    self._loop_cells.setdefault(loop_id, {})[name] = (
                        '%s\x00loop%d\x00%s' % (key, index, name))

    def _collect_loop_names(self, scope: _Scope) -> tuple[
            dict[str, int], dict[int, dict[str, tuple[int, bool]]],
            dict[int, int | None]]:
        """Walk a scope once, gathering everything the loop-variable rule needs.

        Names are visited in EVALUATION order, which differs from document
        order exactly where the rule cares: an ``Assign`` evaluates its value
        before binding its targets, a ``For`` its iterable before its target,
        an ``AnnAssign`` its value before binding its target, and an
        ``AugAssign`` reads its target before storing it.

        Only those four statements can establish a name, and only through a
        target built from ``Name``, ``Tuple``, ``List`` and ``Starred``. Every
        other binding form (assignment expression, ``with ... as``, ``except
        ... as``, match capture, import, ``global`` / ``nonlocal``, ``del``,
        comprehension target) is counted as an occurrence outside every loop,
        so it disqualifies the name.

        :param scope: the scope to walk
        :return: occurrences of each name that belong to the scope; per
            loop, ``name -> (occurrences, stored first)`` for the names of the
            scope it mentions; and each loop's enclosing loop, ``None`` at the
            top level. An occurrence inside a nested function belongs to the
            scope but to no loop of it, so it can only disqualify the name.
        """
        uses: dict[str, int] = {}
        loops: dict[int, dict[str, tuple[int, bool]]] = {}
        parents: dict[int, int | None] = {}

        def record(name: str, store: bool, stack: list[int],
                   cond: frozenset[int], after: frozenset[int]) -> None:
            uses[name] = uses.get(name, 0) + 1
            for loop_id in stack:
                seen = loops[loop_id]
                # A store counts as establishing the name only where every pass
                # through the loop body executes it.
                count, first_store = seen.get(name, (0, store and loop_id not in cond))
                if loop_id in after and not store:
                    # A read in the loop's ``else`` clause also runs after zero
                    # iterations, when it sees the value from before the loop.
                    first_store = False
                seen[name] = (count + 1, first_store)

        def foreign(name: str, count: int = 1) -> None:
            # An occurrence no loop owns: the name can only be disqualified.
            uses[name] = uses.get(name, 0) + count

        def target(node: ast.AST, stack: list[int], cond: frozenset[int],
                   after: frozenset[int]) -> None:
            if isinstance(node, ast.Name):
                record(node.id, True, stack, cond, after)
            elif isinstance(node, (ast.Tuple, ast.List)):
                for elt in node.elts:
                    target(elt, stack, cond, after)
            elif isinstance(node, ast.Starred):
                target(node.value, stack, cond, after)
            else:
                visit(node, stack, cond, after)

        def visit(node: Any, stack: list[int], cond: frozenset[int],
                  after: frozenset[int]) -> None:
            # ``node`` is narrowed by its kind, hence ``Any``
            kind = kinds.get(node.__class__)
            if kind is None:
                kind = _loop_name_kind(node.__class__)
            if kind == _LOOP_NAMES_LEAF:
                return
            if kind == _LOOP_NAMES_NAME:
                if not isinstance(node.ctx, ast.Load):
                    foreign(node.id)
                else:
                    record(node.id, False, stack, cond, after)
                return
            if kind == _LOOP_NAMES_IMPORT:
                for alias in node.names:
                    foreign((alias.asname or alias.name).split('.')[0])
                return
            if kind == _LOOP_NAMES_DECLARATION:
                for name in node.names:
                    foreign(name)
                return
            if kind == _LOOP_NAMES_CAPTURE:
                # ``except ... as name``, a match capture
                if node.name is not None:
                    foreign(node.name)
            elif kind == _LOOP_NAMES_MAPPING:
                if node.rest is not None:
                    foreign(node.rest)
            elif kind == _LOOP_NAMES_FUNCTION:
                if node is not scope.node:
                    # Everything below a nested function is an occurrence outside
                    # every loop of this scope (see :meth:`_foreign_uses`).
                    for name, count in self._foreign_uses(node).items():
                        foreign(name, count)
                    return
            elif kind == _LOOP_NAMES_ASSIGN:
                visit(node.value, stack, cond, after)
                for item in node.targets:
                    target(item, stack, cond, after)
                return
            elif kind == _LOOP_NAMES_ANN_ASSIGN:
                # ``x: T = v`` evaluates ``v`` before binding ``x``; a bare
                # ``x: T`` binds nothing, so its name only counts as a use.
                if node.value is not None:
                    visit(node.value, stack, cond, after)
                visit(node.annotation, stack, cond, after)
                if node.value is not None or not isinstance(node.target, ast.Name):
                    target(node.target, stack, cond, after)
                else:
                    record(node.target.id, False, stack, cond, after)
                return
            elif kind == _LOOP_NAMES_AUG_ASSIGN:
                # ``acc += x`` reads the target before it stores into it.
                if isinstance(node.target, ast.Name):
                    record(node.target.id, False, stack, cond, after)
                else:
                    visit(node.target, stack, cond, after)
                visit(node.value, stack, cond, after)
                return
            elif kind == _LOOP_NAMES_CONDITIONAL or kind == _LOOP_NAMES_LOOP:
                # Code below may be skipped on some pass through every loop
                # already entered, so nothing it stores establishes a name there.
                cond = cond | frozenset(stack)
                if kind == _LOOP_NAMES_LOOP:
                    loop_id = id(node)
                    parents[loop_id] = stack[-1] if stack else None
                    loops[loop_id] = {}
                    stack = stack + [loop_id]
                    # The ``else`` clause also runs after zero iterations, when
                    # nothing in the body has executed.
                    else_cond = cond | {loop_id}
                    if isinstance(node, (ast.For, ast.AsyncFor)):
                        visit(node.iter, stack, cond, after)
                        target(node.target, stack, cond, after)
                    else:
                        visit(node.test, stack, cond, after)
                    for child in node.body:
                        visit(child, stack, cond, after)
                    for child in node.orelse:
                        visit(child, stack, else_cond, after | {loop_id})
                    return
            for child in ast_walk.iter_child_nodes(node):
                visit(child, stack, cond, after)

        kinds = _loop_name_kinds
        visit(scope.node, [], frozenset(), frozenset())
        written = {loop_id: {name: seen for name, seen in names.items()
                             if name in scope.locals}
                   for loop_id, names in loops.items()}
        return uses, written, parents

    def _foreign_uses(self, func: ast.AST) -> dict[str, int]:
        """Name occurrences inside a nested function, as an enclosing scope counts them.

        For :meth:`_collect_loop_names` every occurrence below a nested
        function belongs to the scope being walked but to no loop of it, unless
        the function — or one nested further down, below it — binds the name
        itself and so owns that occurrence. That does not depend on which
        enclosing scope asks, so each function is counted once instead of once
        per scope above it.

        :param func: a ``def`` or ``lambda`` nested in the scope being walked
        :return: name -> occurrences not owned by a function at or below ``func``
        """
        found = self._foreign_memo.get(id(func))
        if found is not None:
            return found
        inner = self._scopes.get(self._key_of.get(id(func), ''))
        own = inner.locals if inner is not None else frozenset()
        counts: dict[str, int] = {}

        def add(name: str, count: int = 1) -> None:
            counts[name] = counts.get(name, 0) + count

        stack = list(ast_walk.iter_child_nodes(func))
        while stack:
            node = stack.pop()
            if isinstance(node, ast.Name):
                add(node.id)
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    add((alias.asname or alias.name).split('.')[0])
                continue
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                for name in node.names:
                    add(name)
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                for name, count in self._foreign_uses(node).items():
                    add(name, count)
                continue
            if isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
                if node.name is not None:
                    add(node.name)
            elif isinstance(node, ast.MatchMapping) and node.rest is not None:
                add(node.rest)
            stack.extend(ast_walk.iter_child_nodes(node))
        found = {name: count for name, count in counts.items() if name not in own}
        self._foreign_memo[id(func)] = found
        return found

    @staticmethod
    def _is_loop_var(name: str, uses: int, binders: list[int],
                     loops: dict[int, dict[str, tuple[int, bool]]],
                     parents: dict[int, int | None]) -> bool:
        """Whether ``name`` lives only inside the loops that bind it.

        :param name: the candidate name
        :param uses: how many occurrences of the name the scope owns
        :param binders: ids of the loops that write it
        :param loops: per loop, ``name -> (occurrences, stored first)``
        :param parents: each loop's enclosing loop
        :return: whether each binder may keep a private cell for the name
        """
        binder_ids = set(binders)
        found = 0
        for loop_id in binders:
            # A binder nested in another one would carry the inner loop's
            # value over into the outer loop's body.
            parent = parents[loop_id]
            while parent is not None:
                if parent in binder_ids:
                    return False
                parent = parents[parent]
            count, first_store = loops[loop_id][name]
            # The loop must establish the name before reading it — otherwise
            # the value it reads comes from outside the loop.
            if not first_store:
                return False
            found += count
        return found == uses

    # --- in_loop ---

    def _mark_in_loop(self) -> None:
        """Mark sids whose ``__sec_write__`` runs more than once per bar.

        Two sources: a write block standing directly in a ``for`` / ``while``
        body (nesting included), and a write inside a user function that is
        reached — through any number of call levels — from a loop body. The
        instantiation pass clones a security-bearing function per CALL SITE,
        not per iteration, so a single clone called from a loop writes its sids
        several times on one bar.
        """
        own_sids: dict[str, set[str]] = {}
        callees: dict[str, set[str]] = {}
        loop_callees: set[str] = set()

        scopes: list[tuple[str, _Scope]] = [('', self._scopes[''])]
        scopes.extend((name, self._scope_of(fn)) for name, fn in self._funcs.items())

        for scope, own in scopes:
            own_sids.setdefault(scope, set())
            callees.setdefault(scope, set())
            for fname, node, in_loop in own.calls:
                if fname == '__sec_write__':
                    sid = self._sid_arg(node)
                    if sid is not None:
                        own_sids[scope].add(sid)
                        if in_loop:
                            self.in_loop.add(sid)
                elif fname in self._funcs:
                    callees[scope].add(fname)
                    if in_loop:
                        loop_callees.add(fname)

        reached: set[str] = set()
        work = list(loop_callees)
        while work:
            name = work.pop()
            if name in reached:
                continue
            reached.add(name)
            work.extend(callees.get(name, ()))

        for name in reached:
            self.in_loop |= own_sids.get(name, set())

    # --- expression taint ---

    def _expr_taint(self, node: ast.expr | None) -> set[str]:
        if node is None or isinstance(node, ast.Constant):
            return set()
        if isinstance(node, ast.Name):
            return self._get_taint(self._q(node.id))
        if isinstance(node, ast.Attribute):
            return self._expr_taint(node.value)
        if isinstance(node, ast.Call):
            return self._call_taint(node)
        if isinstance(node, ast.NamedExpr):
            # Walrus assignments are injected by the lowering passes that run
            # before this one; they behave exactly like a plain assignment.
            taint = self._expr_taint(node.value)
            self._assign(node.target, taint | self._ctrl, node.value)
            return taint
        if isinstance(node, _UNION_EXPRS):
            taint: set[str] = set()
            for child in ast_walk.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    taint |= self._expr_taint(child)
            return taint
        # Unmodelled expression form: the scope falls back instead of the
        # value carrying every sid around through name taint.
        self._mark_fallback(node)
        return set()

    def _call_taint(self, node: ast.Call) -> set[str]:
        func = node.func
        if isinstance(func, ast.Name):
            if func.id == '__sec_read__':
                sid = self._sid_arg(node)
                return {sid} if sid is not None else set(self._all)
            if func.id == '__sec_write__':
                sid = self._sid_arg(node)
                value = node.args[1] if len(node.args) > 1 else None
                taint = self._resolve_tokens(self._expr_taint(value) | self._ctrl)
                if self._scope in self._fallback_scopes:
                    taint |= self._all
                if sid is not None:
                    target = self.depends.setdefault(sid, set())
                    new = (taint - {sid}) - target
                    if new:
                        target |= new
                        self._changed = True
                return set()
            if func.id in ('__sec_signal__', '__sec_wait__'):
                return set()

        taint = self._expr_taint(func) if not isinstance(func, ast.Name) else set()
        pos_taints = [self._expr_taint(a) for a in node.args]
        kw_taints = [(kw.arg, self._expr_taint(kw.value)) for kw in node.keywords]
        # A ``*iterable`` spread has an unknown length, so from its position on
        # no argument has a known parameter of its own.
        star_from: int | None = None
        for i, arg in enumerate(node.args):
            if isinstance(arg, ast.Starred):
                star_from = i
                break
        callee = self._lookup_func(func.id) if isinstance(func, ast.Name) else None
        if callee is not None:
            taint |= self._call_user_func(callee, node, pos_taints, kw_taints, star_from)
        else:
            # Unknown callee: every argument is assumed to reach the result.
            for arg_taint in pos_taints:
                taint |= arg_taint
            for _kw_name, arg_taint in kw_taints:
                taint |= arg_taint

        # Mutation: a collection API writes into the id passed as its first
        # argument (``array.push(id, v)``), and a method call writes into the
        # object it runs on (``id.push(v)``).
        if isinstance(func, ast.Attribute):
            namespace = self._collection_namespace(func)
            if namespace is not None:
                if node.args:
                    self._mutate_collection(namespace, func.attr, node, taint)
            else:
                mutation = taint | self._ctrl
                root = self._root_name(func.value)
                if mutation and root is not None:
                    self._add_taint(self._q(root), mutation)
        return taint

    def _mutate_collection(self, namespace: str, fname: str, node: ast.Call,
                           call_taint: set[str]) -> None:
        """Write a collection call's effect back into its first argument.

        A pure reader writes nothing at all: it derives a value and leaves the
        collection alone. Everything else writes back what the whole call
        touches plus the control taint — whether the write happens depends on
        the enclosing conditions, so the control taint is part of the
        collection's new state even when the stored value is a constant. An
        index or key is not exempt: a map hands its keys back out through
        ``map.keys``, and a position decides which slot a later ``array.get``
        observes, so both are content once the collection is read again.

        :param namespace: the collection namespace the call belongs to
        :param fname: the called function's name
        :param node: the call site
        :param call_taint: the taint of the call as a whole
        """
        if fname in _COLLECTION_READERS.get(namespace, frozenset()):
            return
        root = self._root_name(node.args[0])
        if root is None:
            return
        back = call_taint | self._ctrl
        if back:
            self._add_taint(self._q(root), back)

    def _call_user_func(self, key: str, node: ast.Call,
                        pos_taints: list[set[str]],
                        kw_taints: list[tuple[str | None, set[str]]],
                        star_from: int | None) -> set[str]:
        """Taint of one call of a modelled user function.

        The call's arguments are matched to the callee's parameters, giving a
        substitution from the callee's parameter tokens to this call's argument
        taints. The result is the callee's return taint under that
        substitution — an argument the return does not draw on contributes
        nothing. The same substitution carries taint the callee added to a
        parameter (a mutated collection) back into the caller's variable.

        :param key: scope key of the callee
        :param node: the call site
        :param pos_taints: taint of each positional argument, in call order
        :param kw_taints: ``(keyword name, taint)`` pairs; the name is ``None``
            for a ``**mapping`` unpacking, whose targets are unknown
        :param star_from: index of the first ``*iterable`` in ``pos_taints``,
            ``None`` when the call spreads nothing
        :return: the taint of the call's result
        """
        fn = self._scopes[key].node
        assert isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        arg_expr = self._arg_exprs(fn, node, star_from)
        bound: dict[str, set[str]] = {}
        # Taint whose target parameter is not known at compile time
        unknown: set[str] = set()
        positional = list(fn.args.posonlyargs) + list(fn.args.args)
        for i, arg_taint in enumerate(pos_taints):
            if star_from is not None and i >= star_from:
                unknown |= arg_taint
            elif i < len(positional):
                bound.setdefault(positional[i].arg, set()).update(arg_taint)
            else:
                unknown |= arg_taint
        by_name = {a.arg for a in positional + list(fn.args.kwonlyargs)}
        for kw_name, arg_taint in kw_taints:
            if kw_name is not None and kw_name in by_name:
                bound.setdefault(kw_name, set()).update(arg_taint)
            else:
                unknown |= arg_taint
        # A spread or a leftover argument can fill any parameter, so it binds
        # all of them (the catch-alls included).
        if unknown:
            for param in self._params(fn):
                bound.setdefault(param.arg, set()).update(unknown)

        token_map = {
            self._token(key + '\x00' + pname): arg_taint
            for pname, arg_taint in bound.items()
        }
        token_map[self._ctrl_token(key)] = set(self._ctrl)
        self._bound.add(key)
        for token, arg_taint in token_map.items():
            cur = self._param_in.setdefault(token, set())
            if not arg_taint <= cur:
                cur |= arg_taint
                self._touch(token)

        if self._reads is not None:
            self._reads.add(_RET_CELL + key)
            self._reads.add(_OUT_CELL + key)
        result = self._subst(self._ret.get(key, set()), token_map)
        if key in self._fallback_scopes:
            # An unmodelled construct in the callee: its return can carry
            # anything, and so can what it writes into its arguments.
            result |= self._all
            for expr in arg_expr.values():
                root = self._root_name(expr)
                if root is not None:
                    self._add_taint(self._q(root), set(self._all))
            return result

        outs = self._param_out.get(key, {})
        for pname, expr in arg_expr.items():
            back = self._subst(outs.get(self._token(key + '\x00' + pname), set()), token_map)
            if not back:
                continue
            root = self._root_name(expr)
            if root is not None:
                self._add_taint(self._q(root), back)
        return result

    @classmethod
    def _arg_exprs(cls, fn: 'ast.FunctionDef | ast.AsyncFunctionDef', node: ast.Call,
                   star_from: int | None) -> dict[str, ast.expr]:
        """Map each parameter of ``fn`` to the argument expression ``node`` passes
        for it, for the arguments whose parameter is known at compile time."""
        positional = list(fn.args.posonlyargs) + list(fn.args.args)
        by_name = {a.arg for a in positional + list(fn.args.kwonlyargs)}
        out: dict[str, ast.expr] = {}
        for i, arg in enumerate(node.args):
            if star_from is not None and i >= star_from:
                break
            if i < len(positional):
                out[positional[i].arg] = arg
        for keyword in node.keywords:
            if keyword.arg is not None and keyword.arg in by_name:
                out[keyword.arg] = keyword.value
        return out

    def _alias_roots(self, expr: ast.expr) -> list[str]:
        """Root names of the collections ``expr`` may evaluate to an alias of:
        the root of a name/attribute/subscript chain, or — for a user-function
        call — the roots of the arguments its summary says it returns. Calls
        compose: a call-valued argument contributes its own alias roots, so
        ``f(g(store))`` reaches ``store`` when both return their argument."""
        if isinstance(expr, (ast.Name, ast.Attribute, ast.Subscript)):
            root = self._root_name(expr)
            return [] if root is None or root == 'lib' else [root]
        if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name):
            return []
        key = self._lookup_func(expr.func.id)
        if key is None:
            return []
        if self._reads is not None:
            self._reads.add(_ALIAS_CELL + key)
        params = self._ret_alias.get(key)
        if not params:
            return []
        fn = self._scopes[key].node
        assert isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        star_from = next(
            (i for i, a in enumerate(expr.args) if isinstance(a, ast.Starred)), None
        )
        exprs = self._arg_exprs(fn, expr, star_from)
        roots: list[str] = []
        for pname in params:
            arg = exprs.get(pname)
            if arg is not None:
                roots.extend(self._alias_roots(arg))
        return roots

    def _record_ret_alias(self, value: ast.expr | None) -> None:
        """Note which parameters the returned value shares an alias class with."""
        scope = self._scopes[self._scope]
        fn = scope.node
        if value is None or not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return
        aliased = self._ret_alias.setdefault(scope.key, set())
        reads = self._reads
        for root in self._alias_roots(value):
            cls_root = self._find(self._q(root))
            if reads is not None:
                reads.add(cls_root)
            for param in self._params(fn):
                if param.arg in aliased:
                    continue
                param_root = self._find(scope.key + '\x00' + param.arg)
                if reads is not None:
                    reads.add(param_root)
                if param_root == cls_root:
                    aliased.add(param.arg)
                    self._touch(_ALIAS_CELL + scope.key)

    @staticmethod
    def _collection_namespace(func: ast.Attribute) -> str | None:
        """The collection namespace of a ``lib.<array|matrix|map>.<fn>`` callee."""
        owner = func.value
        if (isinstance(owner, ast.Attribute)
                and owner.attr in _COLLECTION_NAMESPACES
                and isinstance(owner.value, ast.Name)
                and owner.value.id == 'lib'):
            return owner.attr
        return None

    # --- assignment ---

    def _assign(self, target: ast.expr, taint: set[str], value: ast.expr | None) -> None:
        if self._scope in self._fallback_scopes:
            # The assigned value may come from an unmodelled construct whose
            # taint the analysis could not see.
            taint = taint | self._all
        if isinstance(target, ast.Name):
            self._add_taint(self._q(target.id), taint)
            if value is not None:
                for root in self._alias_roots(value):
                    self._union(self._q(target.id), self._q(root))
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._assign(elt, taint, None)
            return
        if isinstance(target, ast.Starred):
            self._assign(target.value, taint, None)
            return
        root = self._root_name(target)
        if root is not None:
            self._add_taint(self._q(root), taint)

    # --- statements ---

    def _visit_body(self, body: list[ast.stmt], ctrl: set[str], top: bool = False) -> set[str]:
        """Visit a statement list; return the control taint after it.

        While the reads are recorded (see :meth:`_visit_scope_body`), a
        statement of a final round none of whose inputs changed since its last
        visit is skipped and hands on the control taint it handed on then.

        :param body: the statements
        :param ctrl: the control taint the first of them runs under
        :param top: whether ``body`` is a scope's own body, whose escape
            accumulators nothing reads afterwards
        :return: the control taint after the last statement
        """
        if self._reads is None:
            for stmt in body:
                ctrl = self._visit_stmt(stmt, ctrl)
            return ctrl
        memo = self._memo
        stamps = self._stamps
        skip = self._final
        for stmt in body:
            entry = memo.get(id(stmt)) if skip else None
            if entry is not None:
                start, ctrl_in, ctrl_out, ctrl_last, cells = entry
                if ctrl_in == ctrl and all(stamps.get(cell, 0) <= start for cell in cells):
                    # The enclosing statement's visit depends on these too
                    self._reads |= cells
                    # A ``match`` guard after a case body reads ``_ctrl`` as
                    # the body's last statement left it
                    self._ctrl = ctrl_last
                    ctrl = ctrl_out
                    continue
            outer = self._reads
            self._reads = cells = set()
            start = self._tick
            escapes = self._escapes
            ctrl_out = self._visit_stmt(stmt, ctrl)
            outer |= cells
            self._reads = outer
            # A statement holding a ``return`` / ``break`` / ``continue`` also
            # feeds the escape accumulators of its block, which the memo does
            # not replay. Whether it holds one does not depend on the taint, so
            # such a statement is simply never skipped.
            if top or self._escapes == escapes:
                memo[id(stmt)] = (start, ctrl, ctrl_out, self._ctrl, cells)
            ctrl = ctrl_out
        return ctrl

    def _visit_scope_body(self, scope: _Scope, ctrl: set[str]) -> None:
        """Visit a scope's statements, skipping those a visit cannot change.

        A statement's visit is a function of the control taint it starts under
        and of the state cells it reads: the taint classes of names, the
        summaries of the functions it calls, the call-site bindings of the
        tokens it resolves. Everything a visit writes only ever grows, so when
        none of that changed since the statement's last visit, visiting it
        again rewrites what is already there and changes nothing — skipping it
        leaves the state, the change flag and the number of rounds exactly as
        a full visit would. The same holds for a statement nested in a block,
        whose position fixes the loop cells and the scope it is visited under.

        What a visit reads is recorded by the readers themselves into
        ``_reads``, the set of the innermost statement being visited, and
        handed up to every enclosing one; every change stamps its cell with a
        fresh counter value (:meth:`_touch`). A cell stamped after a visit
        started — even by that same visit — makes the statement due again.

        Statements are only skipped in the final rounds. Three facts are left
        out of the recorded reads because they no longer change once the first
        round has run: the fallback scopes and the bound callees, both settled
        by reaching every modelled call and every unmodelled expression, which
        each round does whatever the taint, and the final mode itself. What a
        first-round visit would have done differently under them is settled
        once, when the final mode starts (:meth:`_settle_first_round`).

        :param scope: the scope whose body to visit
        :param ctrl: the control taint the body starts under
        """
        self._reads = set()
        self._visit_body(scope.body, ctrl, top=True)
        self._reads = None

    def _visit_block(self, body: list[ast.stmt], ctrl: set[str]) -> tuple[set[str], set[str]]:
        """Visit a nested block and collect the escapes taken inside it.

        A ``return`` / ``break`` / ``continue`` inside the block decides whether
        what FOLLOWS the enclosing statement runs, so its control taint has to
        cross the block boundary. The escapes stay in the enclosing block's
        accumulators too, so one nested several levels deep still reaches the
        outermost successors — up to the loop that swallows a ``break``.

        :param body: the block's statements
        :param ctrl: the control taint the block runs under
        :return: the taint of the ``return`` escapes and of the ``break`` /
            ``continue`` escapes taken inside the block
        """
        outer_returned, self._escaped = self._escaped, set()
        outer_broke, self._broke = self._broke, set()
        self._visit_body(body, ctrl)
        returned, broke = self._escaped, self._broke
        self._escaped = outer_returned | returned
        self._broke = outer_broke | broke
        return returned, broke

    def _visit_loop(self, body: list[ast.stmt], orelse: list[ast.stmt],
                    inner: set[str]) -> set[str]:
        """Visit a loop's blocks; return the escape taint its successors inherit.

        The loop consumes the ``break`` / ``continue`` escapes taken inside it:
        they decide whether what follows the LOOP runs, and nothing beyond that.
        A ``return`` keeps travelling outwards.

        :param body: the loop body
        :param orelse: the loop's ``else`` block
        :param inner: the control taint the loop runs under
        :return: the escape taint of the statements following the loop
        """
        # The loop swallows its own escapes: a ``break`` of THIS loop must not
        # reach an enclosing one, which is why the accumulator is saved here.
        outer_broke, self._broke = self._broke, set()
        returned, broke = self._visit_block(body, inner)
        if broke - inner:
            # A tainted escape also decides whether the body runs AGAIN, so
            # everything in it — the statements before the escape included —
            # is conditional on that taint. One extra pass over the body under
            # the widened control taint closes the back edge; the lattice is
            # finite, so nothing further can be added by repeating it.
            returned, broke = self._visit_block(body, inner | broke)
        more_returned, more_broke = self._visit_block(orelse, inner)
        broke |= more_broke
        self._broke = outer_broke
        return returned | more_returned | broke

    def _enter_loop_cells(self, stmt: ast.stmt) -> dict[str, str | None]:
        """Activate the private taint cells of a loop's own variables, if it has any.

        :param stmt: the loop statement
        :return: the overrides that were active before, to restore afterwards
        """
        cells = self._loop_cells.get(id(stmt))
        saved: dict[str, str | None] = {}
        if cells:
            for name, cell in cells.items():
                saved[name] = self._name_cells.get(name)
                self._name_cells[name] = cell
        return saved

    def _leave_loop_cells(self, saved: dict[str, str | None]) -> None:
        for name, previous in saved.items():
            if previous is None:
                self._name_cells.pop(name, None)
            else:
                self._name_cells[name] = previous

    def _visit_stmt(self, stmt: ast.stmt, ctrl: set[str]) -> set[str]:
        self._ctrl = ctrl
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return ctrl
        if isinstance(stmt, ast.Assign):
            taint = self._expr_taint(stmt.value) | ctrl
            for target in stmt.targets:
                self._assign(target, taint, stmt.value)
            return ctrl
        if isinstance(stmt, ast.AnnAssign):
            if stmt.value is not None:
                self._assign(stmt.target, self._expr_taint(stmt.value) | ctrl, stmt.value)
            return ctrl
        if isinstance(stmt, ast.AugAssign):
            taint = self._expr_taint(stmt.value) | self._expr_taint(stmt.target) | ctrl
            self._assign(stmt.target, taint, None)
            return ctrl
        if isinstance(stmt, ast.Expr):
            self._expr_taint(stmt.value)
            return ctrl
        if isinstance(stmt, ast.If):
            inner = ctrl | self._expr_taint(stmt.test)
            escaped = set().union(*self._visit_block(stmt.body, inner),
                                  *self._visit_block(stmt.orelse, inner))
            return ctrl | escaped
        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            inner = ctrl | self._expr_taint(stmt.iter)
            saved = self._enter_loop_cells(stmt)
            self._assign(stmt.target, inner, None)
            escaped = self._visit_loop(stmt.body, stmt.orelse, inner)
            self._leave_loop_cells(saved)
            return ctrl | escaped
        if isinstance(stmt, ast.While):
            inner = ctrl | self._expr_taint(stmt.test)
            saved = self._enter_loop_cells(stmt)
            escaped = self._visit_loop(stmt.body, stmt.orelse, inner)
            self._leave_loop_cells(saved)
            return ctrl | escaped
        if isinstance(stmt, ast.Match):
            inner = ctrl | self._expr_taint(stmt.subject)
            escaped: set[str] = set()
            for case in stmt.cases:
                case_ctrl = inner
                if case.guard is not None:
                    case_ctrl = inner | self._expr_taint(case.guard)
                escaped |= set().union(*self._visit_block(case.body, case_ctrl))
            return ctrl | escaped
        if isinstance(stmt, ast.Return):
            taint = self._expr_taint(stmt.value) | ctrl
            self._record_return(taint)
            self._record_ret_alias(stmt.value)
            self._escaped |= taint
            self._escapes += 1
            return ctrl | taint
        if isinstance(stmt, (ast.Break, ast.Continue)):
            # The decision to leave the loop is itself tainted, so everything
            # that runs after the loop is conditional on that taint.
            self._broke |= ctrl
            self._escapes += 1
            return ctrl
        return ctrl

    def _mark_fallback(self, node: ast.AST | None = None) -> None:
        """Mark the current scope as containing an unmodelled construct.

        :param node: the construct that triggered the fallback, kept for
            diagnostics
        """
        if self._scope not in self._fallback_scopes:
            self._fallback_scopes.add(self._scope)
            self.fallback_reasons.setdefault(
                self._scope, type(node).__name__ if node is not None else '?'
            )
            self._changed = True

    def _record_return(self, taint: set[str]) -> None:
        if not taint:
            self._ret.setdefault(self._scope, set())
            return
        cur = self._ret.setdefault(self._scope, set())
        if not taint <= cur:
            cur |= taint
            self._touch(_RET_CELL + self._scope)

    # --- per-scope pre/post steps ---

    def _seed_params(self) -> None:
        """Give every parameter its own symbolic token as a base taint."""
        for key, scope in self._scopes.items():
            if not isinstance(scope.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for param in self._params(scope.node):
                qualified = key + '\x00' + param.arg
                self._add_taint(qualified, {self._token(qualified)})

    def _visit_defaults(self, scope: _Scope) -> None:
        """Fold parameter default expressions into the parameters' taint.

        Defaults are evaluated in the ENCLOSING scope, and a call that omits
        the argument leaves the default in place, so the taint belongs to the
        parameter unconditionally.
        """
        fn = scope.node
        assert isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        spec = fn.args
        outer, self._scope = self._scope, scope.parent or ''
        ctrl, self._ctrl = self._ctrl, set()
        positional = list(spec.posonlyargs) + list(spec.args)
        pairs: list[tuple[ast.arg, ast.expr | None]] = list(
            zip(positional[len(positional) - len(spec.defaults):], spec.defaults)
        )
        pairs += list(zip(spec.kwonlyargs, spec.kw_defaults))
        taints = [
            (param, self._expr_taint(default))
            for param, default in pairs if default is not None
        ]
        self._scope, self._ctrl = outer, ctrl
        for param, taint in taints:
            self._add_taint(scope.key + '\x00' + param.arg, taint)

    def _record_param_out(self, scope: _Scope) -> None:
        """Snapshot every parameter's taint at the end of the callee's body."""
        fn = scope.node
        assert isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        outs = self._param_out.setdefault(scope.key, {})
        for param in self._params(fn):
            qualified = scope.key + '\x00' + param.arg
            taint = self._get_taint(qualified)
            cur = outs.setdefault(self._token(qualified), set())
            if not taint <= cur:
                cur |= taint
                self._touch(_OUT_CELL + scope.key)

    # --- driver ---

    def _settle_first_round(self, fallback: frozenset[str]) -> None:
        """Keep the first round's statement memo only where a final round would
        have visited alike.

        A first-round visit differs from a final-round one in what it saw of
        the facts the first round settles (see :meth:`_visit_scope_body`): a
        fallback scope the round only discovered on the way — then every
        entry is dropped, which is rare — and the tokens of an unreached call,
        which only the final mode expands to every sid.

        :param fallback: the fallback scopes before the first round
        """
        if self._fallback_scopes != fallback:
            self._memo.clear()
            return
        unreached = [
            key for key, entry in self._memo.items()
            if any(cell.startswith(_PARAM_PREFIX) and self._of_unreached_call(cell)
                   for cell in entry[4])
        ]
        for key in unreached:
            del self._memo[key]

    def _saturated(self) -> bool:
        """Whether every write already depends on every other sid.

        The result is read off ``depends`` alone, and ``depends`` only grows, so
        from here on no further round can change it.
        """
        for sid, deps in self.depends.items():
            if not self._all - {sid} <= deps:
                return False
        return True

    def run(self) -> None:
        self._funcs = {
            n.name: n for n in ast_walk.walk_statements(self._module)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self._build_scopes()
        if len(self._all) < 2:
            # ``depends`` and ``late_reads`` only ever name ANOTHER context, so a
            # lone one has neither, whatever the taint says. ``in_loop`` needs
            # no taint at all.
            self._mark_in_loop()
            return
        # The scopes' calls together are every call of the module
        self._called_names = {
            name for scope in self._scopes.values() for name, _call, _in_loop in scope.calls
            if name in self._funcs
        }
        self._mark_private_loop_vars()
        self._mark_in_loop()
        self._compute_site_order()

        factories = FactoryFields(self._module)
        class_names = frozenset(
            node.name for node in self._module.body if isinstance(node, ast.ClassDef)
        )
        record_nodes: set[int] = set()
        # Only a module-level class body runs at import time; one defined inside
        # a function runs with the function and stays unmodelled.
        for node in self._module.body:
            if (isinstance(node, ast.ClassDef)
                    and self._is_plain_record(node, factories, class_names)):
                record_nodes.update(id(sub) for sub in ast_walk.walk(node))
        for key in self._scopes:
            for sub in self._unmodelled.get(key, ()):
                if id(sub) in record_nodes:
                    continue
                self._fallback_scopes.add(key)
                self.fallback_reasons.setdefault(key, type(sub).__name__)
                break

        self._seed_params()
        ordered = list(self._scopes.values())
        fallback = frozenset(self._fallback_scopes)
        for _ in range(_MAX_FIXPOINT_ROUNDS):
            self._changed = False
            for scope in ordered:
                self._scope = scope.key
                self._escaped = set()
                self._broke = set()
                self._name_cells = {}
                ctrl: set[str] = set()
                if scope.parent is not None:
                    self._visit_defaults(scope)
                    fn = scope.node
                    assert isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
                    if fn.name in self._called_names:
                        # Everything in a called function runs under the
                        # conditions of its call sites (see _CTRL_PARAM).
                        ctrl.add(self._ctrl_token(scope.key))
                self._visit_scope_body(scope, ctrl)
                if scope.parent is not None:
                    self._record_param_out(scope)
            if self._final and not self._changed:
                break
            if self._saturated():
                break
            if not self._final:
                self._settle_first_round(fallback)
            # Every round visits the same calls, whatever the taint, so after
            # the first one ``_bound`` holds every call site the analysis
            # reaches: from here a token still without a call site belongs to
            # an unreached call. Where the final mode starts does not change
            # the result — every round only adds, and the iteration stops at
            # the least fixpoint of the final rounds either way.
            self._final = True

        # Only a producer whose site PRECEDES this write can be waited for.
        # A later-sited producer would deadlock the historical warmup batch:
        # the child replays many bars in one round, so blocking on a peer whose
        # site the chart has not reached yet stops the very write that would
        # release the round. Such an edge can only be a previous-bar carry
        # anyway; the child reads the default for it.
        # The dropped edges are still reads of this write: a peer whose feed is
        # missing must fail such a read instead of answering it with the
        # default, so they are kept apart in ``late_reads``.
        for sid, deps in self.depends.items():
            own = self._order.get(sid, 0)
            valid = {d for d in deps if d != sid and d in self._all}
            self.depends[sid] = {d for d in valid if self._order.get(d, own) < own}
            late = valid - self.depends[sid]
            if late:
                self.late_reads[sid] = late
