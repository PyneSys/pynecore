"""
Transform Series type annotations and accesses into state-vector slots.

A series variable's circular buffer (a
:class:`~pynecore.core.series.SeriesImpl`) lives in a compile-time-assigned
slot of its scope's state vector; ``_make_state`` creates a fresh buffer for
every instance from the layout's ``series`` entries. The emitted code
addresses the buffer with a literal index:

- ``s: Series[float] = value`` -> ``s = __state__[N].add(value)`` (the local
  name keeps tracking the current scalar value, plain reads stay untouched),
- ``s = value``                -> ``s = __state__[N].set(value)``,
- ``s += value``               -> ``s = __state__[N].set(s + value)``,
- ``s[idx]``                   -> ``__state__[N][idx]``,
- ``lib.max_bars_back(s, num)`` in statement position
                               -> ``__state__[N].max_bars_back = num``
  (other positions are left to the ``lib.max_bars_back`` runtime no-op),
- a ``Series``-annotated parameter loses the Series wrapper from its
  annotation and gets ``s = __state__[N].add(s)`` prepended to the body.

Subscript READS resolve through the scope chain: a nested definition reaches
a parent's series buffer through a closure reference on the parent's
scope-qualified state parameter. The library-series declarations that
:class:`~pynecore.transformers.lib_series.LibrarySeriesTransformer` anchors
in ``main`` rely on this. Writes stay same-scope only, like the legacy
transformer: a plain assignment in a nested scope declares a local that
shadows the parent's series.
"""
from typing import cast
import ast

from .slot_layout import ModuleLayout, scope_for_function

__all__ = ['SeriesTransformer']


class SeriesTransformer(ast.NodeTransformer):
    """Rewrite Series declarations and accesses to state-vector slots."""

    def __init__(self, layout: ModuleLayout):
        self.layout = layout
        self.scope_stack: list[str] = []
        self.current_scope: str = ''
        # scope -> var name -> series slot
        self.series_slots: dict[str, dict[str, int]] = {}
        self.series_declarations: dict[str, set[str]] = {}
        self.local_vars: dict[str, set[str]] = {}

    # --- helpers ---------------------------------------------------------

    def _lookup(self, var_name: str) -> tuple[str, int] | None:
        """Resolve a name to its declaring scope and series slot.

        A name locally assigned in the current scope (but not declared as a
        Series there) shadows any parent series of the same name.

        :param var_name: Source-level variable name.
        :return: (declaring scope, slot index) or None.
        """
        if (var_name in self.local_vars.get(self.current_scope, ())
                and var_name not in self.series_declarations.get(self.current_scope, ())):
            return None
        slots = self.series_slots.get(self.current_scope)
        if slots is not None and var_name in slots:
            return self.current_scope, slots[var_name]
        for i in range(len(self.scope_stack) - 1, 0, -1):
            scope = '·'.join(self.scope_stack[:i])
            slots = self.series_slots.get(scope)
            if slots is not None and var_name in slots:
                return scope, slots[var_name]
        return None

    def _state_ref(self, scope: str, slot: int) -> ast.Subscript:
        """Build a ``<state param>[slot]`` reference for a scope."""
        return ast.Subscript(
            value=ast.Name(id=self.layout.state_param(scope), ctx=ast.Load()),
            slice=ast.Constant(value=slot), ctx=ast.Load())

    def _buffer_call(self, scope: str, slot: int, method: str, args: list[ast.expr]) -> ast.Call:
        """Build a ``<state param>[slot].<method>(...)`` call."""
        return ast.Call(
            func=ast.Attribute(value=self._state_ref(scope, slot),
                               attr=method, ctx=ast.Load()),
            args=args, keywords=[])

    def _register(self, var_name: str) -> int:
        """Allocate a series slot for a declaration in the current scope.

        :param var_name: Source-level variable name.
        :return: The allocated slot index.
        """
        slot = self.layout.scope(self.current_scope).add_series(
            var_name, ast.Constant(value=None))
        self.series_slots.setdefault(self.current_scope, {})[var_name] = slot
        self.series_declarations[self.current_scope].add(var_name)
        self.local_vars[self.current_scope].add(var_name)
        return slot

    @staticmethod
    def _is_series_type(annotation: ast.expr) -> bool:
        """Check if a type annotation is Series."""
        if isinstance(annotation, ast.Subscript):
            return (isinstance(annotation.value, ast.Name)
                    and annotation.value.id == 'Series')
        return isinstance(annotation, ast.Name) and annotation.id == 'Series'

    # --- visitors --------------------------------------------------------

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.layout.assign_scope_ids(node)
        return cast(ast.Module, self.generic_visit(node))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom | None:
        """Strip the Series name from pynecore imports."""
        if node.module and node.module.startswith('pynecore'):
            new_names = [name for name in node.names if name.name != 'Series']
            if not new_names:
                return None
            node.names = new_names
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Track scopes and convert Series-annotated parameters."""
        self.scope_stack.append(self.layout.scope_segment(node))
        self.current_scope = '·'.join(self.scope_stack)
        scope_for_function(self.layout, self.current_scope, node)
        self.local_vars.setdefault(self.current_scope, set())
        self.series_declarations.setdefault(self.current_scope, set())

        param_inits: list[ast.stmt] = []
        for arg in node.args.args:
            self.local_vars[self.current_scope].add(arg.arg)
            if arg.annotation is not None and self._is_series_type(arg.annotation):
                slot = self._register(arg.arg)
                arg.annotation = (arg.annotation.slice
                                  if isinstance(arg.annotation, ast.Subscript) else None)
                param_inits.append(ast.Assign(
                    targets=[ast.Name(id=arg.arg, ctx=ast.Store())],
                    value=self._buffer_call(self.current_scope, slot, 'add',
                                            [ast.Name(id=arg.arg, ctx=ast.Load())])))

        node = cast(ast.FunctionDef, self.generic_visit(node))

        if param_inits:
            insert_pos = 0
            first = node.body[0] if node.body else None
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                insert_pos = 1
            node.body[insert_pos:insert_pos] = param_inits

        self.scope_stack.pop()
        self.current_scope = '·'.join(self.scope_stack)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | None:
        """Convert Series declarations into slot allocations with ``add()``."""
        if not (isinstance(node.target, ast.Name) and self._is_series_type(node.annotation)):
            if isinstance(node.target, ast.Name) and self.current_scope:
                # An annotated assignment declares a local — it shadows a
                # same-named parent series, like a plain assignment does.
                self.local_vars.setdefault(self.current_scope, set()).add(node.target.id)
            if node.value:
                node.value = cast(ast.expr, self.visit(node.value))
            return node

        if not self.current_scope:
            raise SyntaxError("Series variables must be declared inside a function")

        slot = self._register(node.target.id)
        if node.value is None:
            return None
        return ast.Assign(
            targets=[ast.Name(id=node.target.id, ctx=ast.Store())],
            value=self._buffer_call(self.current_scope, slot, 'add',
                                    [cast(ast.expr, self.visit(node.value))]))

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        """Convert assignments to same-scope series variables into ``set()``."""
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = cast(ast.Name, node.targets[0]).id
            slots = self.series_slots.get(self.current_scope)
            if slots is not None and var_name in slots:
                return ast.Assign(
                    targets=[ast.Name(id=var_name, ctx=ast.Store())],
                    value=self._buffer_call(self.current_scope, slots[var_name], 'set',
                                            [cast(ast.expr, self.visit(node.value))]))
            if self.current_scope:
                self.local_vars.setdefault(self.current_scope, set()).add(var_name)
        return cast(ast.AST, self.generic_visit(node))

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        """Convert augmented assignments to same-scope series into ``set()``."""
        if isinstance(node.target, ast.Name):
            slots = self.series_slots.get(self.current_scope)
            if slots is not None and node.target.id in slots:
                value = ast.BinOp(left=ast.Name(id=node.target.id, ctx=ast.Load()),
                                  op=node.op,
                                  right=cast(ast.expr, self.visit(node.value)))
                return ast.Assign(
                    targets=[ast.Name(id=node.target.id, ctx=ast.Store())],
                    value=self._buffer_call(self.current_scope, slots[node.target.id],
                                            'set', [value]))
        return cast(ast.AST, self.generic_visit(node))

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        """Rewrite series indexing to address the buffer in its slot."""
        if isinstance(node.value, ast.Name):
            found = self._lookup(node.value.id)
            if found:
                scope, slot = found
                node.value = self._state_ref(scope, slot)
                node.slice = cast(ast.expr, self.visit(node.slice))
                return node
        return cast(ast.AST, self.generic_visit(node))

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        """Convert statement-position ``lib.max_bars_back(s, n)`` calls into
        a ``max_bars_back`` attribute assignment on the buffer."""
        call = node.value
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == 'max_bars_back'
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == 'lib'
                and len(call.args) >= 2 and isinstance(call.args[0], ast.Name)):
            found = self._lookup(call.args[0].id)
            if found:
                scope, slot = found
                return ast.Assign(
                    targets=[ast.Attribute(value=self._state_ref(scope, slot),
                                           attr='max_bars_back', ctx=ast.Store())],
                    value=cast(ast.expr, self.visit(call.args[1])))
        return cast(ast.AST, self.generic_visit(node))
