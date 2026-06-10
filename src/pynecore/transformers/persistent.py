"""
Transform Persistent type annotations and accesses into state-vector slots.

A persistent variable lives in a compile-time-assigned slot of its scope's
state vector (a plain list the function receives as its hidden first
parameter, see :mod:`pynecore.transformers.slot_layout`); every access is a
literal-index subscript:

- ``p: Persistent[float] = 0.0`` -> slot allocated, declaration removed (the
  initial value moves into the layout's init template),
- non-literal initializers keep the lazy pattern: a value slot plus a flag
  slot and an ``if not __state__[flag]: ...`` guard at the declaration site,
- reads/writes become ``__state__[N]``,
- ``+=`` with a non-literal value keeps Kahan summation, emitted as a
  four-statement sequence (a slot cannot be the target of a walrus, so the
  legacy single-expression form does not carry over — statement position is
  guaranteed because ``+=`` is always a statement),
- a walrus write to a persistent (expression position) is emitted through
  ``__state__.__setitem__`` inside a tuple expression, preserving the value.

Scope rules mirror the legacy transformer: scopes are the middle-dot joined
function-name path, nested definitions see parent persistents (resolved to
the parent's state vector through a closure reference on the parent's
scope-qualified state parameter), and a plain local assignment shadows a
parent persistent of the same name.
"""
from typing import cast
import ast

from .slot_layout import ModuleLayout, scope_for_function

__all__ = ['PersistentTransformer']

PERSISTENT_TYPES = ('Persistent', 'IBPersistent', 'IBPersistentSeries')
VARIP_TYPES = ('IBPersistent', 'IBPersistentSeries')


class PersistentTransformer(ast.NodeTransformer):
    """Rewrite Persistent declarations and accesses to state-vector slots."""

    def __init__(self, layout: ModuleLayout):
        self.layout = layout
        self.scope_stack: list[str] = []
        self.current_scope: str = ''
        # scope -> var name -> (value slot, flag slot or None, kahan slot or None)
        self.var_slots: dict[str, dict[str, int]] = {}
        self.kahan_slots: dict[str, dict[str, int]] = {}
        self.persistent_declarations: dict[str, set[str]] = {}
        self.local_vars: dict[str, set[str]] = {}

    # --- helpers ---------------------------------------------------------

    def _lookup(self, var_name: str) -> tuple[str, int] | None:
        """Resolve a name to its declaring scope and value slot.

        A name locally declared in the current scope (but not as Persistent)
        shadows any parent persistent of the same name.

        :param var_name: Source-level variable name.
        :return: (declaring scope, slot index) or None.
        """
        if (var_name in self.local_vars.get(self.current_scope, ())
                and var_name not in self.persistent_declarations.get(self.current_scope, ())):
            return None
        slots = self.var_slots.get(self.current_scope)
        if slots is not None and var_name in slots:
            return self.current_scope, slots[var_name]
        for i in range(len(self.scope_stack) - 1, 0, -1):
            scope = '·'.join(self.scope_stack[:i])
            slots = self.var_slots.get(scope)
            if slots is not None and var_name in slots:
                return scope, slots[var_name]
        return None

    def _state_ref(self, scope: str, slot: int, ctx: ast.expr_context) -> ast.Subscript:
        """Build a ``<state param>[slot]`` reference for a scope."""
        return ast.Subscript(
            value=ast.Name(id=self.layout.state_param(scope), ctx=ast.Load()),
            slice=ast.Constant(value=slot), ctx=ctx)

    @staticmethod
    def _is_persistent_type(annotation: ast.expr) -> bool:
        """Check if the annotation is any form of Persistent type."""
        if isinstance(annotation, ast.Name):
            return annotation.id in PERSISTENT_TYPES
        if isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name):
            return annotation.value.id in PERSISTENT_TYPES
        if isinstance(annotation, ast.Attribute):
            return annotation.attr in PERSISTENT_TYPES
        return False

    @staticmethod
    def _is_varip_type(annotation: ast.expr) -> bool:
        """Check if the annotation is a varip (IBPersistent) type."""
        if isinstance(annotation, ast.Name):
            return annotation.id in VARIP_TYPES
        if isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name):
            return annotation.value.id in VARIP_TYPES
        if isinstance(annotation, ast.Attribute):
            return annotation.attr in VARIP_TYPES
        return False

    @staticmethod
    def _is_literal_or_na(node: ast.expr) -> bool:
        """Check if a node is a literal value or ``na``."""
        if isinstance(node, ast.Constant):
            return True
        return isinstance(node, ast.Name) and node.id == 'na'

    # --- visitors --------------------------------------------------------

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom | None:
        """Strip the Persistent name from pynecore imports."""
        if node.module and node.module.startswith('pynecore'):
            new_names = [name for name in node.names if name.name != 'Persistent']
            if not new_names:
                return None
            node.names = new_names
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Track scopes, qualify the state parameter when nested defs exist."""
        self.scope_stack.append(node.name)
        self.current_scope = '·'.join(self.scope_stack)

        scope_for_function(self.layout, self.current_scope, node)

        self.local_vars.setdefault(self.current_scope, set())
        self.persistent_declarations.setdefault(self.current_scope, set())
        for arg in node.args.args:
            self.local_vars[self.current_scope].add(arg.arg)

        node = cast(ast.FunctionDef, self.generic_visit(node))

        # Persistent names are not closure variables anymore — drop them from
        # nonlocal statements (shadowed locals keep theirs).
        ancestors = set()
        for i in range(len(self.scope_stack) - 1, 0, -1):
            scope = '·'.join(self.scope_stack[:i])
            ancestors.update(self.persistent_declarations.get(scope, ()))
        new_body: list[ast.stmt] = []
        for stmt in node.body:
            if isinstance(stmt, ast.Nonlocal):
                stmt.names = [name for name in stmt.names if name not in ancestors]
                if not stmt.names:
                    continue
            new_body.append(stmt)
        node.body = new_body

        self.scope_stack.pop()
        self.current_scope = '·'.join(self.scope_stack)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | None:
        """Convert Persistent declarations into slot allocations."""
        if not (isinstance(node.target, ast.Name) and self._is_persistent_type(node.annotation)):
            if isinstance(node.target, ast.Name) and self.current_scope:
                # An annotated assignment declares a local — it shadows a
                # same-named parent persistent, like a plain assignment does.
                self.local_vars.setdefault(self.current_scope, set()).add(node.target.id)
            if node.value:
                node.value = cast(ast.expr, self.visit(node.value))
            return node

        if not self.current_scope:
            raise SyntaxError("Persistent variables must be declared inside a function")

        var_name = node.target.id
        self.persistent_declarations[self.current_scope].add(var_name)
        self.local_vars[self.current_scope].add(var_name)
        varip = self._is_varip_type(node.annotation)
        scope_layout = self.layout.scope(self.current_scope)

        if node.value is not None and not self._is_literal_or_na(node.value):
            # Lazy pattern: value slot + flag slot, initializer runs on first call
            slot = scope_layout.add_var(var_name, ast.Constant(value=None), varip=varip)
            self.var_slots.setdefault(self.current_scope, {})[var_name] = slot
            flag = scope_layout.add_flag(var_name)
            value = cast(ast.expr, self.visit(node.value))
            return ast.If(
                test=ast.UnaryOp(op=ast.Not(),
                                 operand=self._state_ref(self.current_scope, flag, ast.Load())),
                body=[
                    ast.Assign(targets=[self._state_ref(self.current_scope, slot, ast.Store())],
                               value=value),
                    ast.Assign(targets=[self._state_ref(self.current_scope, flag, ast.Store())],
                               value=ast.Constant(value=True)),
                ],
                orelse=[])

        init = node.value if node.value is not None else ast.Name(id='na', ctx=ast.Load())
        slot = scope_layout.add_var(var_name, init, varip=varip)
        self.var_slots.setdefault(self.current_scope, {})[var_name] = slot
        return None

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        """Convert assignments to persistent variables into slot writes."""
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = cast(ast.Name, node.targets[0]).id
            # The first plain assignment to a not-yet-known name declares a
            # local, shadowing a same-named parent persistent.
            if var_name not in self.local_vars.get(self.current_scope, ()):
                self.local_vars.setdefault(self.current_scope, set()).add(var_name)
            found = self._lookup(var_name)
            if found:
                scope, slot = found
                return ast.Assign(
                    targets=[self._state_ref(scope, slot, ast.Store())],
                    value=cast(ast.expr, self.visit(node.value)))
        node.targets = [cast(ast.expr, self.visit(t)) for t in node.targets]
        node.value = cast(ast.expr, self.visit(node.value))
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST | list[ast.stmt]:
        """Slot-target augmented assignment; Kahan summation for non-literal ``+=``."""
        if isinstance(node.target, ast.Name):
            found = self._lookup(node.target.id)
            if found:
                scope, slot = found
                if isinstance(node.op, ast.Add) and not self._is_literal_or_na(node.value):
                    return self._emit_kahan(scope, slot, node.target.id,
                                            cast(ast.expr, self.visit(node.value)))
                node.target = self._state_ref(scope, slot, ast.Store())
                node.value = cast(ast.expr, self.visit(node.value))
                return node
        return cast(ast.AST, self.generic_visit(node))

    def _emit_kahan(self, scope: str, slot: int, var_name: str,
                    value: ast.expr) -> list[ast.stmt]:
        """Emit the Kahan-summation statement sequence for ``var += value``."""
        scope_kahans = self.kahan_slots.setdefault(scope, {})
        comp = scope_kahans.get(var_name)
        if comp is None:
            comp = scope_kahans[var_name] = self.layout.scope(scope).add_kahan(var_name)

        def var_ref(ctx: ast.expr_context) -> ast.Subscript:
            return self._state_ref(scope, slot, ctx)

        def comp_ref(ctx: ast.expr_context) -> ast.Subscript:
            return self._state_ref(scope, comp, ctx)

        corrected = ast.Name(id='__kahan_corrected__', ctx=ast.Load())
        new_sum = ast.Name(id='__kahan_new_sum__', ctx=ast.Load())
        return [
            # __kahan_corrected__ = <value> - <comp>
            ast.Assign(targets=[ast.Name(id='__kahan_corrected__', ctx=ast.Store())],
                       value=ast.BinOp(left=value, op=ast.Sub(), right=comp_ref(ast.Load()))),
            # __kahan_new_sum__ = <var> + __kahan_corrected__
            ast.Assign(targets=[ast.Name(id='__kahan_new_sum__', ctx=ast.Store())],
                       value=ast.BinOp(left=var_ref(ast.Load()), op=ast.Add(), right=corrected)),
            # <comp> = (__kahan_new_sum__ - <var>) - __kahan_corrected__
            ast.Assign(targets=[comp_ref(ast.Store())],
                       value=ast.BinOp(
                           left=ast.BinOp(left=new_sum, op=ast.Sub(),
                                          right=var_ref(ast.Load())),
                           op=ast.Sub(), right=corrected)),
            # <var> = __kahan_new_sum__
            ast.Assign(targets=[var_ref(ast.Store())], value=new_sum),
        ]

    def visit_NamedExpr(self, node: ast.NamedExpr) -> ast.AST:
        """Walrus write to a persistent: route through ``__setitem__`` so the
        construct stays a valid expression and still yields the value."""
        if isinstance(node.target, ast.Name):
            found = self._lookup(node.target.id)
            if found:
                scope, slot = found
                value = cast(ast.expr, self.visit(node.value))
                param = self.layout.state_param(scope)
                return ast.Subscript(
                    value=ast.Tuple(
                        elts=[
                            ast.NamedExpr(target=ast.Name(id='__pyne_w__', ctx=ast.Store()),
                                          value=value),
                            ast.Call(
                                func=ast.Attribute(value=ast.Name(id=param, ctx=ast.Load()),
                                                   attr='__setitem__', ctx=ast.Load()),
                                args=[ast.Constant(value=slot),
                                      ast.Name(id='__pyne_w__', ctx=ast.Load())],
                                keywords=[]),
                        ],
                        ctx=ast.Load()),
                    slice=ast.Constant(value=0), ctx=ast.Load())
        return cast(ast.AST, self.generic_visit(node))

    def visit_Name(self, node: ast.Name) -> ast.AST:
        """Convert persistent references using scope-aware lookup."""
        found = self._lookup(node.id)
        if found:
            scope, slot = found
            return self._state_ref(scope, slot, node.ctx)
        return node
