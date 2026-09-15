import ast


#: Series-carrying persistent annotations mapped to their non-series half.
#: ``IBPersistentSeries`` is the ``varip`` flavour — it must split the same way,
#: otherwise the variable keeps its slot but stays a plain scalar and any
#: history read (``x[1]``) raises ``'float' object is not subscriptable``.
SERIES_PERSISTENT_TYPES = {
    'PersistentSeries': 'Persistent',
    'IBPersistentSeries': 'IBPersistent',
}


class PersistentSeriesTransformer(ast.NodeTransformer):
    """
    Transform PersistentSeries and IBPersistentSeries declarations into a
    Persistent (resp. IBPersistent) + Series combination.
    Must be applied before PersistentTransformer and SeriesTransformer.
    """

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
        if hasattr(node, '_ps_transformed'):
            return node

        if not isinstance(node.target, ast.Name):
            return node

        # Check if it's a series-carrying persistent type
        persistent_type = None
        series_type = None

        if isinstance(node.annotation, ast.Subscript):
            if isinstance(node.annotation.value, ast.Name):
                persistent_type = SERIES_PERSISTENT_TYPES.get(node.annotation.value.id)
                if persistent_type is not None:
                    series_type = node.annotation.slice
        elif isinstance(node.annotation, ast.Name):
            persistent_type = SERIES_PERSISTENT_TYPES.get(node.annotation.id)

        if persistent_type is None:
            return node

        # Create two declarations
        var_name = node.target.id
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
        ast.fix_missing_locations(persistent_decl)
        setattr(persistent_decl, "_ps_transformed", True)

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
        ast.fix_missing_locations(series_decl)
        setattr(series_decl, "_ps_transformed", True)

        return [persistent_decl, series_decl]
