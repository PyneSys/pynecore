import ast


class PersistentSeriesTransformer(ast.NodeTransformer):
    """
    Transform PersistentSeries declarations into Persistent + Series combination.
    Must be applied before PersistentTransformer and SeriesTransformer.
    """

    def visit_ImportFrom(self, node):
        """Handle imports, only remove Series while keeping other imports"""
        if node.module and node.module.startswith('pynecore'):
            # Filter out Persistent from names
            new_names = [name for name in node.names if name.name != 'PersistentSeries']
            if not new_names:
                # If no names left, remove the entire import
                return None
            # Create new import with remaining names
            node.names = new_names
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | list[ast.AnnAssign]:
        """Transform PersistentSeries type annotations into separate Persistent and Series declarations"""
        if hasattr(node, '_ps_transformed'):
            return node

        if not isinstance(node.target, ast.Name):
            return node

        # Check if it's a PersistentSeries type
        is_persistent_series = False
        series_type = None

        if isinstance(node.annotation, ast.Subscript):
            if (isinstance(node.annotation.value, ast.Name) and
                    node.annotation.value.id == 'PersistentSeries'):
                is_persistent_series = True
                series_type = node.annotation.slice
        elif (isinstance(node.annotation, ast.Name) and
              node.annotation.id == 'PersistentSeries'):
            is_persistent_series = True

        if not is_persistent_series:
            return node

        # Create two declarations
        var_name = node.target.id
        value = node.value

        # 1. Persistent declaration
        persistent_decl = ast.AnnAssign(
            target=ast.Name(id=var_name, ctx=ast.Store()),
            annotation=ast.Subscript(
                value=ast.Name(id='Persistent', ctx=ast.Load()),
                slice=series_type if series_type else ast.Name(id='float', ctx=ast.Load()),
                ctx=ast.Load()
            ) if series_type else ast.Name(id='Persistent', ctx=ast.Load()),
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
