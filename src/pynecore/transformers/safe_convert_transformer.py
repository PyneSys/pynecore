from typing import cast
import ast


class SafeConvertTransformer(ast.NodeTransformer):
    """
    Transformer that converts float(na) and int(na) calls to safe alternatives
    that preserve Pine Script semantics.

    This transformer replaces float() and int() function calls with safe_float()
    and safe_int() from pynecore.core.safe_convert module, to ensure proper
    handling of NA values.
    """

    def __init__(self):
        self.has_safe_convert_import = False
        self.has_convert_functions = False  # Track if float()/int() is used

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """
        Visit Call nodes and transform float() and int() calls
        """
        # Continue normal transformation for children
        self.generic_visit(node)

        # Check if it's a built-in float() or int() call
        if (isinstance(node.func, ast.Name) and
                node.func.id in ('float', 'int')):

            # Check for the builtin module
            if hasattr(node.func, 'module') and getattr(node.func, 'module') == 'builtins':
                return node

            # Mark that we need the safe_convert import
            self.has_convert_functions = True

            # Transform to safe_convert.safe_float/safe_int call
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='safe_convert', ctx=ast.Load()),
                    attr=f'safe_{node.func.id}',
                    ctx=ast.Load()
                ),
                args=node.args,
                keywords=node.keywords
            )

        return node

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """
        Add safe_convert import if needed
        """
        # Process the module first
        node = cast(ast.Module, self.generic_visit(node))

        # Only add the import if we actually transformed any functions
        if not self.has_convert_functions:
            return node

        # Check for existing safe_convert import
        for stmt in node.body:
            if isinstance(stmt, ast.ImportFrom) and stmt.module == 'pynecore.core.safe_convert':
                self.has_safe_convert_import = True
                # Check if it's imported as 'safe_convert'
                for alias in stmt.names:
                    if alias.name == 'safe_convert' or alias.asname == 'safe_convert':
                        return node
            elif isinstance(stmt, ast.ImportFrom) and stmt.module == 'pynecore.core':
                for alias in stmt.names:
                    if alias.name == 'safe_convert':
                        self.has_safe_convert_import = True
                        return node

        # Add import if needed
        if not self.has_safe_convert_import:
            import_stmt = ast.ImportFrom(
                module='pynecore.core',
                names=[ast.alias(name='safe_convert', asname=None)],
                level=0
            )

            # Find the right position to insert import - after the docstring if it exists
            insert_pos = 0
            if (node.body and isinstance(node.body[0], ast.Expr) and
                    isinstance(cast(ast.Expr, node.body[0]).value, ast.Constant)):
                insert_pos = 1

            # Insert after any existing imports
            while (insert_pos < len(node.body) and
                   (isinstance(node.body[insert_pos], ast.Import) or
                    isinstance(node.body[insert_pos], ast.ImportFrom))):
                insert_pos += 1

            node.body.insert(insert_pos, import_stmt)

        return node
