"""
Shared slot-layout allocator for the slot-based instance state scheme.

Every transformer that contributes per-instance state (persistent variables,
series, isolated call sites) allocates slots from a single
:class:`ModuleLayout` while walking the module.
:func:`~pynecore.transformers.layout_emit.apply_layout` then materializes the
result into the module AST.

Scope identifiers join the function-name path with the middle-dot separator
(``main``, ``main·helper``) — the same convention the legacy global-name
mangling used. Repeated definitions of one name within a scope (``overload``
implementations) are disambiguated with an ordinal suffix (``highest·2``);
the mapping comes from :func:`collect_scope_segments` and every pipeline
stage reads it through :meth:`ModuleLayout.scope_segment`.

The hidden parameter is named ``__state__`` by default. Functions that
contain nested function definitions get a scope-qualified name
(``__state·main__``) instead, so a nested function can reach the parent's
state vector through a plain closure reference without its own hidden
parameter shadowing it.

The runtime side of the contract (layout dict format, ``_make_state``,
``__resolve_slot__``, ...) lives in :mod:`pynecore.core.instance_state`.
"""
import ast
from dataclasses import dataclass, field

from . import ast_walk

__all__ = ['ModuleLayout', 'ScopeLayout', 'scope_for_function', 'collect_scope_segments',
           'DEFAULT_STATE_PARAM']

DEFAULT_STATE_PARAM = '__state__'


def _scope_defs(scope_node: ast.AST):
    """Yield the function definitions belonging to one scope in source order
    (descends through statements and class bodies, but not into nested
    function definitions — those belong to the inner scope)."""
    # An explicit stack of child iterators: a recursive generator pays one
    # ``yield from`` hop per tree level for every node it passes through. A
    # definition only ever stands in a statement list, so no expression is
    # entered (and no lambda body with it)
    stack = [ast_walk.iter_child_statements(scope_node)]
    while stack:
        for child in stack[-1]:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield child
            else:
                stack.append(ast_walk.iter_child_statements(child))
                break
        else:
            stack.pop()


def collect_scope_segments(tree: ast.Module) -> dict[int, str]:
    """Compute the scope segment of every function definition in a module.

    Repeated definitions of the same name within one scope (typically
    ``overload`` implementations) get an ordinal suffix (``highest``,
    ``highest·2``) so each keeps its own slot layout. The result is a pure
    function of the tree structure, so every pipeline stage that recomputes
    it gets identical segments.

    :param tree: The module AST.
    :return: ``id(def node) -> segment`` mapping.
    """
    return _scan_scopes(tree)[0]


def _scan_scopes(tree: ast.Module) -> tuple[dict[int, str], dict[int, bool]]:
    """The scope segment of every function definition, and whether it nests one.

    :param tree: The module AST.
    :return: ``id(def node) -> segment`` and ``id(def node) -> whether a
        ``FunctionDef`` stands anywhere inside it``.
    """
    segments: dict[int, str] = {}
    nests: dict[int, bool] = {}

    def assign(scope_node: ast.AST) -> bool:
        counts: dict[str, int] = {}
        nested = False
        for func in _scope_defs(scope_node):
            n = counts.get(func.name, 0) + 1
            counts[func.name] = n
            segments[id(func)] = func.name if n == 1 else f'{func.name}·{n}'
            inner = assign(func)
            nests[id(func)] = inner
            nested = nested or inner or isinstance(func, ast.FunctionDef)
        return nested

    assign(tree)
    return segments, nests


@dataclass
class _Slot:
    """One slot of a scope's state vector."""
    index: int
    kind: str  # 'var' | 'flag' | 'series' | 'child' | 'anchor' | 'pin'
    name: str  # debug name for the layout 'names' tuple
    init: ast.expr  # template expression for the layout 'init' tuple
    max_bars_back: ast.expr | None = None  # series slots only
    series_elem: str | None = None  # series slots only: element type name ('float'/'int') or None
    var_elem: str | None = None  # var slots only: the declared scalar type name or None
    call_id: str | None = None  # child/anchor slots only
    in_loop: bool = False  # child slots only
    varip: bool = False  # var and flag slots only


@dataclass
class ScopeLayout:
    """Slot table of one scope (one function definition)."""
    scope: str
    slots: list[_Slot] = field(default_factory=list)
    state_param: str = DEFAULT_STATE_PARAM
    #: Slot holding the instance vector, for a scope whose body resolves
    #: something per instance; None for every other scope, which is nearly
    #: all of them (see :meth:`add_pin`)
    pin_slot: int | None = None

    def _add(self, slot: _Slot) -> int:
        self.slots.append(slot)
        return slot.index

    def add_var(self, name: str, init: ast.expr, *, varip: bool = False,
                elem: str | None = None) -> int:
        """Allocate a slot for a persistent variable.

        :param name: Source-level variable name (debug only).
        :param init: Template expression for the init tuple (literal or ``na``;
            lazy-initialized variables pass ``Constant(None)`` and pair this
            slot with :meth:`add_flag`).
        :param varip: Whether the variable is ``varip`` (excluded from var rollback).
        :param elem: The declared scalar type (``int`` for ``Persistent[int]``),
            which the init literal alone does not tell: a Pine int is a double
            at runtime, so an int-typed persistent starts at a FLOAT literal
            (``PersistentTransformer._numeric_init``), and a consumer that types
            the slot statically reads the declaration here.
        :return: The allocated slot index.
        """
        return self._add(_Slot(len(self.slots), 'var', name, init, varip=varip, var_elem=elem))

    def add_flag(self, name: str, *, varip: bool = False) -> int:
        """Allocate a lazy-init flag slot (init ``False``).

        :param name: Name of the variable the flag belongs to.
        :param varip: Whether the guarded variable is ``varip``. The flag then
            has to survive the var rollback with it: rolled back to ``False`` on
            the first bar, a discarded re-execution re-runs the initializer and
            the ``varip`` loses what the earlier pass of that bar put there.
        :return: The allocated slot index.
        """
        return self._add(_Slot(len(self.slots), 'flag', f'{name}·flag',
                               ast.Constant(value=False), varip=varip))

    def add_series(self, name: str, max_bars_back: ast.expr, elem: str | None = None) -> int:
        """Allocate a series slot (``_make_state`` puts a fresh ``SeriesImpl`` here).

        :param name: Source-level variable name (debug only).
        :param max_bars_back: Expression for the series' ``max_bars_back`` argument.
        :param elem: Element type name from the ``Series[T]`` annotation when it
            is statically known (``'float'`` and ``'int'`` matter: those series
            get the native nan as their out-of-range na value, and an int series
            stores every value as a float).
        :return: The allocated slot index.
        """
        return self._add(_Slot(len(self.slots), 'series', name, ast.Constant(value=None),
                               max_bars_back=max_bars_back, series_elem=elem))

    def add_pin(self, count: int) -> int:
        """Allocate the scope's instance-vector slot.

        Reserved only for a body whose inner sites do not resolve to the same
        thing in every context it is instantiated in (the type pass stamps
        those on the definition). The slot holds one entry per such site; the
        init template is the all-None vector, which means "configure nothing"
        and is what an instance created with no vector -- a cross-module call,
        a call the type pass could not resolve -- keeps.

        :param count: Number of instance-varying sites in the body.
        :return: The allocated slot index.
        """
        index = self._add(_Slot(
            len(self.slots), 'pin', '·pins',
            ast.Tuple(elts=[ast.Constant(value=None) for _ in range(count)], ctx=ast.Load())))
        self.pin_slot = index
        return index

    def add_child(self, call_id: str, *, in_loop: bool) -> int:
        """Allocate a child slot for an isolated call site.

        :param call_id: Call-site identifier (``main·ema·0`` style).
        :param in_loop: Whether the call site sits in a loop (slot holds a
            child list instead of a single child state).
        :return: The allocated slot index.
        """
        return self._add(_Slot(len(self.slots), 'child', call_id, ast.Constant(value=None),
                               call_id=call_id, in_loop=in_loop))

    def add_anchor(self, call_id: str, *, in_loop: bool = False) -> int:
        """Allocate an anchor slot for a uniform-path call site.

        Anchors are emitted into the layout's ``children`` tuple, so
        ``reset()`` clears them and the next call rebinds; the bind helper
        creates whatever it caches. Loop-shaped anchors hold a list of
        ``(callee, bound)`` pairs indexed by the per-invocation counter.

        :param call_id: Call-site identifier.
        :param in_loop: Whether the call site sits in a loop.
        :return: The allocated slot index.
        """
        return self._add(_Slot(len(self.slots), 'anchor', call_id, ast.Constant(value=None),
                               call_id=call_id, in_loop=in_loop))


class ModuleLayout:
    """Slot layouts of every scope in one module, shared by the transformers."""

    def __init__(self, *, compacted_series: bool = False):
        """
        :param compacted_series: True for ``@pyne lib`` modules. Their series are
            the rolling windows of the builtin machines, which leave out na bars
            on purpose so ``src[k]`` is the k-th most recent non-na value. Script
            series instead advance one slot per BAR and repeat the last value on
            a bar their branch never ran, so the flag turns that forward fill off
            for the lib windows alone.
        """
        self.scopes: dict[str, ScopeLayout] = {}
        self._segments: dict[int, str] = {}
        #: id(def node) -> whether a ``FunctionDef`` stands anywhere inside it
        self._nests: dict[int, bool] = {}
        #: The module the two maps above were computed for
        self._tree: ast.Module | None = None
        self.compacted_series = compacted_series

    def assign_scope_ids(self, tree: ast.Module) -> None:
        """Build the definition -> scope-segment mapping for a module.

        Every transformer entry point (and ``apply_layout``) calls this.
        The mapping is computed by the first call for a module and kept: no
        pass of the lowering adds, removes, moves or renames a definition, so
        every later stage would compute the same one.

        :param tree: The module AST.
        """
        if tree is self._tree:
            return
        self._tree = tree
        self._segments, self._nests = _scan_scopes(tree)

    def scanned(self, tree: ast.Module) -> bool:
        """Whether the definition mapping in use was computed for this module.

        :param tree: The module AST.
        :return: True when :meth:`assign_scope_ids` keeps the mapping for it.
        """
        return tree is self._tree

    def scope_segment(self, node: ast.FunctionDef) -> str:
        """Disambiguated scope segment of a function definition.

        :param node: The function definition.
        :return: ``name`` or ``name·N`` for repeated names in one scope.
        """
        return self._segments.get(id(node), node.name)

    def nests_definition(self, node: ast.FunctionDef) -> bool:
        """Whether a ``FunctionDef`` stands anywhere inside a definition.

        :param node: The function definition.
        :return: True if a nested definition exists, at any depth.
        """
        try:
            return self._nests[id(node)]
        except KeyError:
            return any(isinstance(child, ast.FunctionDef)
                       for child in ast_walk.walk_statements(node) if child is not node)

    def scope(self, scope_id: str) -> ScopeLayout:
        """Return (creating on demand) the layout of a scope.

        :param scope_id: Middle-dot joined function-name path.
        :return: The scope's layout.
        """
        try:
            return self.scopes[scope_id]
        except KeyError:
            scope = self.scopes[scope_id] = ScopeLayout(scope_id)
            return scope

    def state_carrying(self, scope_id: str) -> bool:
        """Whether a scope has any state slot (and thus a hidden state parameter).

        :param scope_id: Middle-dot joined function-name path.
        :return: True if the scope carries state.
        """
        scope = self.scopes.get(scope_id)
        return scope is not None and bool(scope.slots)

    def state_param(self, scope_id: str) -> str:
        """Name of a scope's hidden state parameter.

        :param scope_id: Middle-dot joined function-name path.
        :return: The parameter name (``__state__`` or scope-qualified).
        """
        scope = self.scopes.get(scope_id)
        return scope.state_param if scope is not None else DEFAULT_STATE_PARAM


def scope_for_function(layout: ModuleLayout, scope_id: str, node: ast.FunctionDef) -> ScopeLayout:
    """Return the scope layout of a function definition, qualifying the state
    parameter name when the function contains nested definitions.

    Every state-contributing transformer must enter scopes through this
    helper so they agree on the parameter name (``__state__`` vs the
    scope-qualified ``__state·{scope}__`` that nested definitions reach
    through a closure).

    :param layout: The module's shared allocator.
    :param scope_id: Middle-dot joined function-name path.
    :param node: The function definition being entered.
    :return: The scope's layout.
    """
    scope = layout.scope(scope_id)
    if layout.nests_definition(node):
        scope.state_param = f'__state·{scope_id}__'
    return scope
