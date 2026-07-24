"""
Shared slot-layout allocator for the slot-based instance state scheme.

Every transformer that contributes per-instance state (persistent variables,
series, isolated call sites) allocates slots from a single
:class:`ModuleLayout` while walking the module. :func:`apply_layout` then
materializes the result into the module AST:

- the module-level ``__pyne_slot_layout__`` dict (one entry per
  state-carrying scope),
- a ``func.__pyne_layout__ = __pyne_slot_layout__['<scope>']`` attach
  statement after every state-carrying function definition (inside the parent
  body for nested definitions); decorated definitions get an innermost
  ``@__attach_layout__(...)`` decorator instead, so the layout lands on the
  raw function and not on the decorator's return value,
- the hidden state parameter injected as the FIRST parameter of every
  state-carrying function.

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

__all__ = ['ModuleLayout', 'ScopeLayout', 'apply_layout', 'scope_for_function',
           'collect_scope_segments']

DEFAULT_STATE_PARAM = '__state__'


def _scope_defs(scope_node: ast.AST):
    """Yield the function definitions belonging to one scope in source order
    (descends through statements and class bodies, but not into nested
    function definitions — those belong to the inner scope)."""
    for child in ast.iter_child_nodes(scope_node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield child
        elif not isinstance(child, ast.Lambda):
            yield from _scope_defs(child)


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
    segments: dict[int, str] = {}

    def assign(scope_node: ast.AST) -> None:
        counts: dict[str, int] = {}
        for func in _scope_defs(scope_node):
            n = counts.get(func.name, 0) + 1
            counts[func.name] = n
            segments[id(func)] = func.name if n == 1 else f'{func.name}·{n}'
            assign(func)

    assign(tree)
    return segments


@dataclass
class _Slot:
    """One slot of a scope's state vector."""
    index: int
    kind: str  # 'var' | 'flag' | 'kahan' | 'series' | 'child' | 'anchor'
    name: str  # debug name for the layout 'names' tuple
    init: ast.expr  # template expression for the layout 'init' tuple
    max_bars_back: ast.expr | None = None  # series slots only
    series_elem: str | None = None  # series slots only: element type name ('float') or None
    call_id: str | None = None  # child/anchor slots only
    in_loop: bool = False  # child slots only
    varip: bool = False  # var slots and their kahan companions only


@dataclass
class ScopeLayout:
    """Slot table of one scope (one function definition)."""
    scope: str
    slots: list[_Slot] = field(default_factory=list)
    state_param: str = DEFAULT_STATE_PARAM

    def _add(self, slot: _Slot) -> int:
        self.slots.append(slot)
        return slot.index

    def add_var(self, name: str, init: ast.expr, *, varip: bool = False) -> int:
        """Allocate a slot for a persistent variable.

        :param name: Source-level variable name (debug only).
        :param init: Template expression for the init tuple (literal or ``na``;
            lazy-initialized variables pass ``Constant(None)`` and pair this
            slot with :meth:`add_flag`).
        :param varip: Whether the variable is ``varip`` (excluded from var rollback).
        :return: The allocated slot index.
        """
        return self._add(_Slot(len(self.slots), 'var', name, init, varip=varip))

    def add_flag(self, name: str) -> int:
        """Allocate a lazy-init flag slot (init ``False``).

        :param name: Name of the variable the flag belongs to.
        :return: The allocated slot index.
        """
        return self._add(_Slot(len(self.slots), 'flag', f'{name}·flag', ast.Constant(value=False)))

    def add_kahan(self, name: str, *, varip: bool = False) -> int:
        """Allocate a Kahan compensation slot (init ``0.0``).

        :param name: Name of the variable the compensation belongs to.
        :param varip: Whether the compensated variable is ``varip`` — the
            compensation must follow it out of the var rollback, otherwise a
            rollback would desynchronize the pair.
        :return: The allocated slot index.
        """
        return self._add(_Slot(len(self.slots), 'kahan', f'{name}·kahan', ast.Constant(value=0.0),
                               varip=varip))

    def add_series(self, name: str, max_bars_back: ast.expr, elem: str | None = None) -> int:
        """Allocate a series slot (``_make_state`` puts a fresh ``SeriesImpl`` here).

        :param name: Source-level variable name (debug only).
        :param max_bars_back: Expression for the series' ``max_bars_back`` argument.
        :param elem: Element type name from the ``Series[T]`` annotation when it
            is statically known (currently only ``'float'`` matters: those
            series get the native nan as their out-of-range na value).
        :return: The allocated slot index.
        """
        return self._add(_Slot(len(self.slots), 'series', name, ast.Constant(value=None),
                               max_bars_back=max_bars_back, series_elem=elem))

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

    def __init__(self):
        self.scopes: dict[str, ScopeLayout] = {}
        self._segments: dict[int, str] = {}

    def assign_scope_ids(self, tree: ast.Module) -> None:
        """(Re)build the definition -> scope-segment mapping for a module.

        Every transformer entry point (and :func:`apply_layout`) calls this;
        the rebuild is deterministic, so repeated calls on the same tree
        agree even after body mutations.

        :param tree: The module AST.
        """
        self._segments = collect_scope_segments(tree)

    def scope_segment(self, node: ast.FunctionDef) -> str:
        """Disambiguated scope segment of a function definition.

        :param node: The function definition.
        :return: ``name`` or ``name·N`` for repeated names in one scope.
        """
        return self._segments.get(id(node), node.name)

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
    if any(isinstance(child, ast.FunctionDef)
           for child in ast.walk(node) if child is not node):
        scope.state_param = f'__state·{scope_id}__'
    return scope


def _scope_entry_ast(scope: ScopeLayout) -> ast.Dict:
    """Build the layout dict literal of one scope."""
    init = ast.Tuple(elts=[slot.init for slot in scope.slots], ctx=ast.Load())
    series = ast.Tuple(
        elts=[ast.Tuple(elts=[ast.Constant(value=slot.index),
                              slot.max_bars_back if slot.max_bars_back is not None
                              else ast.Constant(value=None),
                              ast.Constant(value=slot.series_elem)],
                        ctx=ast.Load())
              for slot in scope.slots if slot.kind == 'series'],
        ctx=ast.Load())
    varip = ast.Tuple(
        elts=[ast.Constant(value=slot.index) for slot in scope.slots if slot.varip],
        ctx=ast.Load())
    children = ast.Tuple(
        elts=[ast.Tuple(elts=[ast.Constant(value=slot.index),
                              ast.Constant(value=slot.call_id),
                              ast.Constant(value=slot.in_loop)],
                        ctx=ast.Load())
              for slot in scope.slots if slot.kind in ('child', 'anchor')],
        ctx=ast.Load())
    names = ast.Tuple(
        elts=[ast.Constant(value=slot.name) for slot in scope.slots],
        ctx=ast.Load())
    return ast.Dict(
        keys=[ast.Constant(value=key) for key in ('init', 'series', 'varip', 'children', 'names')],
        values=[init, series, varip, children, names],
    )


def _layout_assign_ast(layout: ModuleLayout) -> ast.Assign:
    """Build the module-level ``__pyne_slot_layout__`` assignment."""
    carrying = [scope for scope in layout.scopes.values() if scope.slots]
    return ast.Assign(
        targets=[ast.Name(id='__pyne_slot_layout__', ctx=ast.Store())],
        value=ast.Dict(
            keys=[ast.Constant(value=scope.scope) for scope in carrying],
            values=[_scope_entry_ast(scope) for scope in carrying],
        ),
    )


def _attach_ast(func_name: str, scope_id: str) -> ast.Assign:
    """Build the ``func.__pyne_layout__ = __pyne_slot_layout__['scope']`` attach."""
    return ast.Assign(
        targets=[ast.Attribute(value=ast.Name(id=func_name, ctx=ast.Load()),
                               attr='__pyne_layout__', ctx=ast.Store())],
        value=ast.Subscript(value=ast.Name(id='__pyne_slot_layout__', ctx=ast.Load()),
                            slice=ast.Constant(value=scope_id), ctx=ast.Load()),
    )


def _attach_decorator_ast(scope_id: str) -> ast.Call:
    """Build the ``__attach_layout__(__pyne_slot_layout__['scope'])`` decorator.

    Used for decorated definitions: appended as the INNERMOST decorator it
    tags the raw function, while a post-definition attribute assignment
    would tag whatever the other decorators returned (e.g. an ``overload``
    dispatcher).
    """
    return ast.Call(
        func=ast.Name(id='__attach_layout__', ctx=ast.Load()),
        args=[ast.Subscript(value=ast.Name(id='__pyne_slot_layout__', ctx=ast.Load()),
                            slice=ast.Constant(value=scope_id), ctx=ast.Load())],
        keywords=[])


def _process_defs(body: list[ast.stmt], scope_prefix: str, layout: ModuleLayout,
                  used_imports: set[str]) -> list[ast.stmt]:
    """Inject hidden state parameters and layout attaches into a statement list."""
    new_body: list[ast.stmt] = []
    for stmt in body:
        new_body.append(stmt)
        if not isinstance(stmt, ast.FunctionDef):
            continue
        segment = layout.scope_segment(stmt)
        scope_id = f'{scope_prefix}·{segment}' if scope_prefix else segment
        stmt.body = _process_defs(stmt.body, scope_id, layout, used_imports)
        if layout.state_carrying(scope_id):
            stmt.args.args.insert(0, ast.arg(arg=layout.state_param(scope_id)))
            if stmt.decorator_list:
                stmt.decorator_list.append(_attach_decorator_ast(scope_id))
                used_imports.add('__attach_layout__')
            else:
                new_body.append(_attach_ast(stmt.name, scope_id))
    return new_body


def _insert_index(body: list[ast.stmt]) -> int:
    """Index right after the module docstring and the leading import block."""
    index = 0
    for i, stmt in enumerate(body):
        if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)) or isinstance(stmt, (ast.Import,
                                                                            ast.ImportFrom)):
            index = i + 1
            continue
        break
    return index


def apply_layout(tree: ast.Module, layout: ModuleLayout) -> ast.Module:
    """Materialize the collected layout into the module AST.

    Inserts the ``__pyne_slot_layout__`` dict after the import block, injects
    the hidden state parameter into every state-carrying function definition
    and appends the ``__pyne_layout__`` attach statement after each of them.

    :param tree: The module AST (already processed by the slot transformers).
    :param layout: The shared allocator the transformers filled.
    :return: The same module object, updated in place.
    """
    # The dict is emitted even when empty: its presence marks the module as
    # transformed, which the cross-module call-site classification relies on
    # ("transformed module + no layout attribute -> provably stateless").
    layout.assign_scope_ids(tree)
    used_imports: set[str] = set()
    tree.body = _process_defs(tree.body, '', layout, used_imports)
    index = _insert_index(tree.body)
    tree.body.insert(index, _layout_assign_ast(layout))
    if used_imports:
        tree.body.insert(index, ast.ImportFrom(
            module='pynecore.core.instance_state',
            names=[ast.alias(name=name, asname=None) for name in sorted(used_imports)],
            level=0))
    return tree
