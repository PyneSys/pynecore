"""
Exact verification of the ``closed_shift`` flag on the LOWERED tree.

``closed_shift`` says a ``request.security`` context's value cannot change
inside one HTF period, so the chart may skip the period's developing rounds
(see ``core/security.py``). :class:`~pynecore.transformers.security.
SecurityTransformer` decides it on the SOURCE shape of the expression, which is
necessarily a re-derivation of what the lowering half later establishes
exactly: it also decides the merge grouping, so it has to run there.

This pass closes the gap from the other end. After
:class:`~pynecore.transformers.series.SeriesTransformer` a runtime history
reference is EXACTLY one of two forms:

- ``<state param>[slot][k]`` with ``k`` an int constant >= 1 — the series buffer
  of a slot, the only thing that indexes recorded bars, and
- ``inline_series(expr, k)`` with the same ``k`` — PyneCore's per-anchor history
  of an arbitrary expression, still a plain call at this point (the isolation
  pass anchors it afterwards). Only while the module has not bound that spelling
  itself: a script's own ``def inline_series`` is an ordinary function.

Nothing else indexes history: every other subscript is element access on a
collection or a tuple selection, whose value follows the developing bar. So a
context whose lowered ``__sec_write__`` expression is not built from those forms
(or a tuple/list of them) cannot be ``closed_shift``, whatever the source looked
like, and the flag is cleared — for the context AND for every member of its
merge group, because one child serves the whole group and its round protocol
must be uniform.

A history form is also reached through a NAME: the inline-series hoist lifts
``inline_series(...)`` out of the expression into a scope-local temporary
(``__hist_0__ = inline_series(lib.ta.sma(lib.close, len), 1)``) and leaves the
bare name where the call stood, which is what the compiler's own output looks
like. Such a name is resolved, but only when the binding is beyond doubt
(:func:`_name_binding`): a single plain assignment standing in the SAME
statement list as the write, in front of it, with no other binding of that
spelling anywhere in the scope. Reaching the write then proves the binding ran,
so the name holds exactly the history form it was assigned. Everything less
definite — a rebound name, a parameter, a binding in a branch, a name resolved
across a nested definition — stays rejected.

The pass NEVER sets the flag to True: the source-level classifier stays the
candidate, this is the exact filter behind it.
"""
import ast

from .security_slice import _ctx_get, _iter_contexts
from .slot_layout import DEFAULT_STATE_PARAM, ModuleLayout

__all__ = ['verify_closed_shift']


def _state_params(layout: ModuleLayout) -> frozenset[str]:
    """Every name a lowered state-vector reference can be rooted at.

    :param layout: The module's slot layout, as the series pass filled it.
    :return: The hidden state parameter names of all scopes.
    """
    names = {DEFAULT_STATE_PARAM}
    for scope in layout.scopes.values():
        names.add(scope.state_param)
    return frozenset(names)


def _is_history_index(node: ast.expr) -> bool:
    """Whether ``node`` is an int constant >= 1 (a history offset)."""
    return (isinstance(node, ast.Constant) and isinstance(node.value, int)
            and not isinstance(node.value, bool) and node.value >= 1)


def _is_slot_ref(node: ast.expr, state_params: frozenset[str]) -> bool:
    """Whether ``node`` is a ``<state param>[slot]`` series-buffer reference."""
    return (isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in state_params
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, int)
            and not isinstance(node.slice.value, bool))


def _is_inline_series(node: ast.expr, primitive: bool) -> bool:
    """Whether ``node`` is the ``inline_series(expr, k>=1)`` history call.

    :param node: Expression node.
    :param primitive: Whether the spelling still means PyneCore's own
        ``inline_series`` in this module (:func:`_binds_name`). A module that
        binds the name itself has an ordinary function there, and calling it
        reads no history at all.
    """
    if not primitive:
        return False
    if not (isinstance(node, ast.Call) and len(node.args) == 2 and not node.keywords):
        return False
    func = node.func
    name = (func.id if isinstance(func, ast.Name)
            else func.attr if isinstance(func, ast.Attribute) else None)
    return name == 'inline_series' and _is_history_index(node.args[1])


#: The modules the pipeline imports the history primitives FROM: an import of
#: ``inline_series`` out of one of them is the primitive, not a shadow of it.
_PRIMITIVE_MODULES = frozenset({'pynecore.core.series'})


def _binds_name(module: ast.Module, name: str) -> bool:
    """Whether the module binds ``name`` itself instead of inheriting it.

    A ``def``, a lambda parameter, any assignment / loop / ``with`` target, a
    parameter or an import of that spelling shadows the injected primitive. The
    import lifter's OWN ``from pynecore.core.series import inline_series`` does
    not: that names the primitive itself, under its own spelling.

    :param module: The lowered module.
    :param name: The spelling to look for.
    :return: True when the module binds it.
    """
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            args = node.args
            for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs,
                        args.vararg, args.kwarg]:
                if arg is not None and arg.arg == name:
                    return True
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if (alias.asname or alias.name.split('.')[0]) != name:
                    continue
                if isinstance(node, ast.ImportFrom) and not node.level \
                        and node.module in _PRIMITIVE_MODULES \
                        and alias.asname is None:
                    continue
                return True
            continue
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            targets = [node.target]
        elif isinstance(node, ast.withitem):
            targets = ([node.optional_vars]
                       if node.optional_vars is not None else [])
        for target in targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Name) and sub.id == name:
                    return True
    return False


#: A scope whose statement list a write can stand in.
_Scope = ast.Module | ast.FunctionDef | ast.AsyncFunctionDef


def _own_nodes(node: ast.AST) -> list[ast.AST]:
    """``node`` and everything under it that belongs to the SAME scope.

    A definition opens a scope of its own, whose statements the enclosing
    statement list does not run: the walk stops at one, ``node`` itself
    included.

    :param node: The node to walk.
    :return: The node and its same-scope descendants.
    """
    out: list[ast.AST] = []
    pending: list[ast.AST] = [node]
    while pending:
        current = pending.pop()
        out.append(current)
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef,
                                ast.ClassDef, ast.Lambda)):
            continue
        pending.extend(ast.iter_child_nodes(current))
    return out


def _name_binding(name: str, scope: _Scope, target: ast.AST) -> ast.expr | None:
    """The value ``name`` certainly holds where ``target`` stands, or None.

    The one shape this answers for is the hoist's temporary: a single plain
    ``name = <expr>`` statement in the scope's own statement list, standing in
    front of the statement that holds ``target``. Reaching ``target`` then proves
    the assignment ran and nothing since has rebound the name, so the value is
    exactly that expression.

    Anything else gives None: a second binding of the spelling anywhere in the
    scope (an augmented assignment, a loop target, a tuple unpacking, a nested
    ``def`` of that name), a parameter, a ``global`` / ``nonlocal`` declaration,
    a binding that stands in a branch rather than in the statement list itself,
    or a ``target`` the statement list does not reach without entering a nested
    definition.

    :param name: The spelling to resolve.
    :param scope: The scope whose statement list holds both.
    :param target: The node the resolved value is read at.
    :return: The bound expression, or None when the binding is not certain.
    """
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        spec = scope.args
        for arg in [*spec.posonlyargs, *spec.args, *spec.kwonlyargs,
                    spec.vararg, spec.kwarg]:
            if arg is not None and arg.arg == name:
                return None
    bound: ast.expr | None = None
    bound_index = -1
    for index, stmt in enumerate(scope.body):
        plain = (stmt.targets[0] if isinstance(stmt, ast.Assign)
                 and len(stmt.targets) == 1
                 and isinstance(stmt.targets[0], ast.Name)
                 and stmt.targets[0].id == name else None)
        for sub in ast.walk(stmt):
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                    and sub.name == name:
                return None
            if isinstance(sub, (ast.Global, ast.Nonlocal)) and name in sub.names:
                return None
            if isinstance(sub, ast.Name) and sub.id == name \
                    and isinstance(sub.ctx, (ast.Store, ast.Del)) and sub is not plain:
                return None
            if isinstance(sub, ast.arg) and sub.arg == name:
                return None
        if plain is not None:
            if bound is not None:
                return None
            bound = stmt.value  # type: ignore[union-attr]
            bound_index = index
    if bound is None:
        return None
    for index, stmt in enumerate(scope.body):
        if index <= bound_index:
            continue
        if any(sub is target for sub in _own_nodes(stmt)):
            return bound
    return None


def _is_history_form(node: ast.expr, state_params: frozenset[str],
                     primitive: bool, scope: _Scope | None = None,
                     read: ast.AST | None = None) -> bool:
    """Whether the lowered expression reads closed bars ONLY.

    :param node: The ``__sec_write__`` expression argument.
    :param state_params: The module's state parameter names.
    :param primitive: Whether ``inline_series`` is still PyneCore's own.
    :param scope: The scope the write stands in, for resolving a hoisted
        temporary; None disables the name resolution.
    :param read: The node the value is read at; defaults to ``node`` itself.
    :return: True for a slot history read, an ``inline_series`` call, a
        non-empty tuple/list of such, or a name certainly bound to one
        (:func:`_name_binding`).
    """
    if isinstance(node, (ast.Tuple, ast.List)):
        return bool(node.elts) and all(
            _is_history_form(elt, state_params, primitive, scope, read or node)
            for elt in node.elts)
    if _is_inline_series(node, primitive):
        return True
    if isinstance(node, ast.Subscript) and _is_history_index(node.slice) \
            and _is_slot_ref(node.value, state_params):
        return True
    if isinstance(node, ast.Name) and scope is not None:
        bound = _name_binding(node.id, scope, read or node)
        # The binding stands in front of the read, so resolving its value
        # cannot come back to this name and the recursion terminates.
        if bound is not None:
            return _is_history_form(bound, state_params, primitive, scope,
                                    read or node)
    return False


def _find_contexts(module: ast.Module) -> ast.Dict | None:
    """The ``__security_contexts__`` dict literal of the module, if it has one."""
    for stmt in module.body:
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
            continue
        target = stmt.targets[0]
        if (isinstance(target, ast.Name) and target.id == '__security_contexts__'
                and isinstance(stmt.value, ast.Dict)):
            return stmt.value
    return None


def _collect_writes(module: ast.Module) \
        -> dict[str, list[tuple[ast.expr, _Scope]]]:
    """Every ``__sec_write__(sid, expr)`` expression of the module, by sid.

    The write block stands in ``main()`` and, for a sliced context, in its
    clone as well; both are checked, so a clone that somehow differs cannot
    keep the flag on the strength of the original.

    :param module: The lowered module.
    :return: Per sid, the expression of each of its write calls with the scope
        whose statement list the call stands in (a hoisted temporary in the
        write's value is resolved there, see :func:`_name_binding`).
    """
    scopes: list[_Scope] = [module]
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scopes.append(node)
    writes: dict[str, list[tuple[ast.expr, _Scope]]] = {}
    for scope in scopes:
        for stmt in scope.body:
            for node in _own_nodes(stmt):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == '__sec_write__' and len(node.args) == 2):
                    continue
                sid = node.args[0]
                if isinstance(sid, ast.Constant) and isinstance(sid.value, str):
                    writes.setdefault(sid.value, []).append((node.args[1], scope))
    return writes


def _group_members(contexts: ast.Dict) -> tuple[dict[str, int], dict[int, set[str]]]:
    """The merge groups of the module's contexts.

    :param contexts: The ``__security_contexts__`` dict literal.
    :return: The group ordinal of each grouped sid, and the members per ordinal.
    """
    group_of: dict[str, int] = {}
    members: dict[int, set[str]] = {}
    for sid, ctx in _iter_contexts(contexts):
        value = _ctx_get(ctx, 'group')
        if (isinstance(value, ast.Constant) and isinstance(value.value, int)
                and not isinstance(value.value, bool)):
            group_of[sid] = value.value
            members.setdefault(value.value, set()).add(sid)
    return group_of, members


def verify_closed_shift(module: ast.Module, layout: ModuleLayout) -> ast.Module:
    """Clear ``closed_shift`` wherever the lowered write is not a history read.

    Must run directly after
    :class:`~pynecore.transformers.series.SeriesTransformer`: that pass emits
    the slot references this check recognises, and the isolation pass after it
    rewrites the ``inline_series`` call sites.

    :param module: The lowered module; the flag literals are mutated in place.
    :param layout: The module's slot layout.
    :return: The same module.
    """
    contexts = _find_contexts(module)
    if contexts is None:
        return module
    flagged = {sid for sid, ctx in _iter_contexts(contexts)
               if _is_flag_true(ctx)}
    if not flagged:
        return module

    state_params = _state_params(layout)
    primitive = not _binds_name(module, 'inline_series')
    writes = _collect_writes(module)
    failed: set[str] = set()
    for sid in flagged:
        exprs = writes.get(sid)
        if not exprs or not all(_is_history_form(expr, state_params, primitive, scope)
                                for expr, scope in exprs):
            failed.add(sid)
    if not failed:
        return module

    group_of, members = _group_members(contexts)
    cleared = set(failed)
    for sid in failed:
        group = group_of.get(sid)
        if group is not None:
            cleared |= members[group]

    for sid, ctx in _iter_contexts(contexts):
        if sid not in cleared:
            continue
        value = _ctx_get(ctx, 'closed_shift')
        if isinstance(value, ast.Constant) and value.value is True:
            value.value = False
    return module


def _is_flag_true(ctx: ast.Dict) -> bool:
    """Whether the context's ``closed_shift`` entry says True."""
    value = _ctx_get(ctx, 'closed_shift')
    return isinstance(value, ast.Constant) and value.value is True
