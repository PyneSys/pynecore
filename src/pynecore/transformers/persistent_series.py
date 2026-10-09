import ast
from typing import cast

from . import ast_walk


#: Series-carrying persistent annotations mapped to their non-series half.
#: ``IBPersistentSeries`` is the ``varip`` flavour — it must split the same way,
#: otherwise the variable keeps its slot but stays a plain scalar and any
#: history read (``x[1]``) raises ``'float' object is not subscriptable``.
SERIES_PERSISTENT_TYPES = {
    'PersistentSeries': 'Persistent',
    'IBPersistentSeries': 'IBPersistent',
}


def _persistent_half(node: ast.AnnAssign) -> str | None:
    """The persistent half of a series-carrying persistent declaration, else None."""
    if not isinstance(node.target, ast.Name):
        return None
    annotation = node.annotation
    if isinstance(annotation, ast.Subscript):
        annotation = annotation.value
        if not isinstance(annotation, ast.Name):
            return None
    elif not isinstance(annotation, ast.Name):
        return None
    return SERIES_PERSISTENT_TYPES.get(annotation.id)


class PersistentSeriesTransformer(ast_walk.NodeTransformer):
    """
    Transform PersistentSeries and IBPersistentSeries declarations into a
    Persistent (resp. IBPersistent) + Series combination.
    Must be applied before PersistentTransformer and SeriesTransformer.
    """

    def visit_Module(self, node: ast.Module) -> ast.Module:
        # Both the split declarations and the imports of the split types are
        # statements: a module with neither is left as it is, without visiting
        # a single expression
        for stmt in ast_walk.walk_statements(node):
            if isinstance(stmt, ast.AnnAssign):
                if _persistent_half(stmt) is not None:
                    break
            elif isinstance(stmt, ast.ImportFrom) and stmt.module \
                    and stmt.module.startswith('pynecore') \
                    and any(alias.name in SERIES_PERSISTENT_TYPES for alias in stmt.names):
                break
        else:
            return node
        return cast(ast.Module, self.generic_visit(node))

    def visit_ImportFrom(self, node):
        """Handle imports, only remove the split types while keeping the rest"""
        if node.module and node.module.startswith('pynecore'):
            new_names = [name for name in node.names
                         if name.name not in SERIES_PERSISTENT_TYPES]
            if not new_names:
                # If no names left, remove the entire import
                return None
            # Create new import with remaining names
            node.names = new_names
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | list[ast.AnnAssign]:
        """Split a series-carrying persistent annotation into its persistent and Series halves"""
        persistent_type = _persistent_half(node)
        if persistent_type is None:
            return node
        series_type = node.annotation.slice if isinstance(node.annotation, ast.Subscript) else None

        # Create two declarations
        var_name = cast(ast.Name, node.target).id
        value = node.value

        # 1. Persistent declaration
        persistent_decl = ast.AnnAssign(
            target=ast.Name(id=var_name, ctx=ast.Store()),
            annotation=ast.Subscript(
                value=ast.Name(id=persistent_type, ctx=ast.Load()),
                slice=series_type if series_type else ast.Name(id='float', ctx=ast.Load()),
                ctx=ast.Load()
            ) if series_type else ast.Name(id=persistent_type, ctx=ast.Load()),
            value=value,
            simple=1
        )
        # The split stands where the declaration stood: a diagnostic on either
        # half has to point at the line the user wrote
        ast.copy_location(persistent_decl, node)
        ast_walk.fix_missing_locations(persistent_decl)

        # 2. Series declaration
        series_decl = ast.AnnAssign(
            target=ast.Name(id=var_name, ctx=ast.Store()),
            annotation=ast.Subscript(
                value=ast.Name(id='Series', ctx=ast.Load()),
                slice=series_type if series_type else ast.Name(id='float', ctx=ast.Load()),
                ctx=ast.Load()
            ) if series_type else ast.Name(id='Series', ctx=ast.Load()),
            value=ast.Name(id=var_name, ctx=ast.Load()),
            simple=1
        )
        ast.copy_location(series_decl, node)
        ast_walk.fix_missing_locations(series_decl)

        return [persistent_decl, series_decl]
