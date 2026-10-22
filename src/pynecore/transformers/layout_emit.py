"""
Materialization of a module's slot layout into its AST.

The state-contributing transformers fill one
:class:`~pynecore.transformers.slot_layout.ModuleLayout` while they walk the
module; :func:`apply_layout` then writes the result into the tree:

- the module-level ``__pyne_slot_layout__`` dict (one entry per
  state-carrying scope),
- a ``func.__pyne_layout__ = __pyne_slot_layout__['<scope>']`` attach
  statement after every state-carrying function definition (inside the parent
  body for nested definitions); decorated definitions get an innermost
  ``@__attach_layout__(...)`` decorator instead, so the layout lands on the
  raw function and not on the decorator's return value,
- the hidden state parameter injected as the FIRST parameter of every
  state-carrying function.

The runtime side of the contract (layout dict format, ``_make_state``,
``__resolve_slot__``, ...) lives in :mod:`pynecore.core.instance_state`.
"""
import ast

from .pine_type_rules import OBJECT, stamp_lowering
from .slot_layout import ModuleLayout, ScopeLayout

__all__ = ['apply_layout']


def _scope_entry_ast(scope: ScopeLayout, compacted: bool = False) -> ast.Dict:
    """Build the layout dict literal of one scope.

    :param scope: The scope's slot table.
    :param compacted: Whether the module's series are na-compacted windows
        (see :attr:`ModuleLayout.compacted_series`).
    """
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
    keys = ['init', 'series', 'varip', 'children', 'names']
    values: list[ast.expr] = [init, series, varip, children, names]
    if scope.pin_slot is not None:
        keys.append('pin')
        values.append(ast.Constant(value=scope.pin_slot))
    if compacted:
        keys.append('compacted')
        values.append(ast.Constant(value=True))
    return ast.Dict(
        keys=[ast.Constant(value=key) for key in keys],
        values=values,
    )


def _layout_assign_ast(layout: ModuleLayout) -> ast.Assign:
    """Build the module-level ``__pyne_slot_layout__`` assignment.

    The literal swallows the declarations' init expressions, which carry Pine
    types; the table around them is machinery, so the whole literal is typed
    as one object and the inits keep the stamps they came with.
    """
    carrying = [scope for scope in layout.scopes.values() if scope.slots]
    return ast.Assign(
        targets=[ast.Name(id='__pyne_slot_layout__', ctx=ast.Store())],
        value=stamp_lowering(ast.Dict(
            keys=[ast.Constant(value=scope.scope) for scope in carrying],
            values=[_scope_entry_ast(scope, layout.compacted_series) for scope in carrying],
        ), OBJECT),
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


def _process_nested_bodies(node: ast.AST, scope_prefix: str, layout: ModuleLayout,
                           used_imports: set[str]) -> None:
    """Run :func:`_process_defs` over every statement list hanging off a node.

    A definition nested in a compound statement belongs to the ENCLOSING
    scope (:func:`_scope_defs` collects it there), so it must get the same
    hidden parameter and layout attach as a top-level one.
    """
    for name in node._fields:
        value = getattr(node, name, None)
        if not isinstance(value, list) or not value:
            continue
        if isinstance(value[0], ast.stmt):
            setattr(node, name, _process_defs(value, scope_prefix, layout, used_imports))
            continue
        for item in value:
            if isinstance(item, (ast.ExceptHandler, ast.match_case)):
                _process_nested_bodies(item, scope_prefix, layout, used_imports)


def _process_defs(body: list[ast.stmt], scope_prefix: str, layout: ModuleLayout,
                  used_imports: set[str]) -> list[ast.stmt]:
    """Inject hidden state parameters and layout attaches into a statement list."""
    new_body: list[ast.stmt] = []
    for stmt in body:
        new_body.append(stmt)
        if not isinstance(stmt, ast.FunctionDef):
            _process_nested_bodies(stmt, scope_prefix, layout, used_imports)
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
