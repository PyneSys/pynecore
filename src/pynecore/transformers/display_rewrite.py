"""
Display rewrite for transformed-module dumps (debug paths only).

The slot scheme emits literal state indexes (``__state__[3]``), which make a
dump hard to read. This module rebuilds the dump text with named index
constants instead:

- ``__state__[3]`` -> ``__state__[__slot·main·p__]``,
- ``__resolve_slot__(__state__, 5, f)`` / ``__bind_any__(__state__, 7, f)``
  index arguments are renamed the same way,
- ``__state__.__setitem__(2, v)`` (the walrus-write form) likewise,
- the constant definitions are inserted right after the module docstring.

The rewrite works on a CLEAN COPY of the tree (rebuilt from source, so no
stray node attributes travel along — never deepcopy an AST in this
pipeline). The compiled bytecode always stays the literal-index variant;
only what ``PYNE_AST_DEBUG`` / ``PYNE_AST_SAVE`` show is affected, while
``PYNE_AST_DEBUG_RAW`` keeps printing the exact emission (the AST golden
tests compare against that).
"""
from typing import cast
import ast

from .slot_layout import ModuleLayout, DEFAULT_STATE_PARAM, collect_scope_segments

__all__ = ['display_dump']

_INDEXED_HELPERS = ('__resolve_slot__', '__bind_any__')


class _IndexNamer(ast.NodeTransformer):
    """Replace literal state-vector indexes with named constants."""

    def __init__(self, layout: ModuleLayout, segments: dict[int, str]):
        self.layout = layout
        # The display copy is re-parsed, so the layout's node-identity map
        # does not apply — segments are recomputed on the copy (the mapping
        # is a pure function of the tree structure, so they agree).
        self.segments = segments
        self.stack: list[str] = []
        self.used: dict[str, int] = {}  # constant name -> slot index

    # --- helpers ---------------------------------------------------------

    def _scope_of_param(self, param: str) -> str | None:
        """Map a state-parameter name to its scope id."""
        if param == DEFAULT_STATE_PARAM:
            return '·'.join(self.stack) if self.stack else None
        if param.startswith('__state·') and param.endswith('__'):
            return param[len('__state·'):-2]
        return None

    def _name_for(self, param: str, index: int) -> str | None:
        """Named constant for a (state parameter, literal index) pair."""
        scope_id = self._scope_of_param(param)
        if scope_id is None:
            return None
        scope = self.layout.scopes.get(scope_id)
        if scope is None or not 0 <= index < len(scope.slots):
            return None
        label = scope.slots[index].name.replace('.', '·').replace('<', '').replace('>', '')
        if not label.startswith(f'{scope_id}·'):
            label = f'{scope_id}·{label}'
        name = f'__slot·{label}__'
        existing = self.used.get(name)
        if existing is not None and existing != index:
            # duplicate display names in a scope (e.g. PersistentSeries pairs)
            name = f'{name[:-2]}·{index}__'
        self.used[name] = index
        return name

    @staticmethod
    def _literal_index(node: ast.expr) -> int | None:
        if (isinstance(node, ast.Constant) and isinstance(node.value, int)
                and not isinstance(node.value, bool)):
            return node.value
        return None

    # --- visitors --------------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.stack.append(self.segments.get(id(node), node.name))
        self.generic_visit(node)
        self.stack.pop()
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.Subscript:
        self.generic_visit(node)
        if isinstance(node.value, ast.Name):
            index = self._literal_index(node.slice)
            if index is not None:
                name = self._name_for(node.value.id, index)
                if name:
                    node.slice = ast.Name(id=name, ctx=ast.Load())
        return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)
        # __resolve_slot__(P, N, f) / __bind_any__(P, N, f)
        if (isinstance(node.func, ast.Name) and node.func.id in _INDEXED_HELPERS
                and len(node.args) >= 2 and isinstance(node.args[0], ast.Name)):
            index = self._literal_index(node.args[1])
            if index is not None:
                name = self._name_for(node.args[0].id, index)
                if name:
                    node.args[1] = ast.Name(id=name, ctx=ast.Load())
        # P.__setitem__(N, value) — the walrus-write form
        elif (isinstance(node.func, ast.Attribute) and node.func.attr == '__setitem__'
                and isinstance(node.func.value, ast.Name) and node.args):
            index = self._literal_index(node.args[0])
            if index is not None:
                name = self._name_for(node.func.value.id, index)
                if name:
                    node.args[0] = ast.Name(id=name, ctx=ast.Load())
        return node


def display_dump(tree: ast.Module, layout: ModuleLayout) -> str:
    """Readable unparse of a transformed module.

    :param tree: The fully transformed module AST.
    :param layout: The module's shared slot allocator.
    :return: Source text with named index constants.
    """
    clean = ast.parse(ast.unparse(tree))
    namer = _IndexNamer(layout, collect_scope_segments(clean))
    clean = cast(ast.Module, namer.visit(clean))
    if namer.used:
        defs: list[ast.stmt] = [
            ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())],
                       value=ast.Constant(value=index))
            for name, index in sorted(namer.used.items())]
        pos = 0
        first = clean.body[0] if clean.body else None
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            pos = 1
        clean.body[pos:pos] = defs
        ast.fix_missing_locations(clean)
    return ast.unparse(clean)
