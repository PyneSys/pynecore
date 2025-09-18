from _ast import Call, BinOp
from typing import cast
import ast


class SafeDivisionTransformer(ast.NodeTransformer):
    """
    Transformer that converts division operations to safe alternatives
    that preserve Pine Script semantics.

    This transformer replaces division operations (/) with safe_div()
    from pynecore.core.safe_convert module, to ensure proper
    handling of division by zero cases (returns NA instead of raising exception).
    """

    def __init__(self):
        self.has_safe_convert_import = False
        self.has_division_operations = False  # Track if division is used

    def visit_BinOp(self, node: ast.BinOp) -> Call | BinOp:
        """
        Visit BinOp nodes and transform division operations
        """
        # Continue normal transformation for children
        self.generic_visit(node)

        # Check if it's a division operation
        if isinstance(node.op, ast.Div):
            # Only transform if it's not a literal division (e.g., 1/2)
            # Literal divisions are safe and should remain as is for performance
            if not (isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant)):
                # Mark that we need the safe_convert import
                self.has_division_operations = True

                # Transform to safe_convert.safe_div call
                return ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='safe_convert', ctx=ast.Load()),
                        attr='safe_div',
                        ctx=ast.Load()
                    ),
                    args=[node.left, node.right],
                    keywords=[]
                )

        return node

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """
        Add safe_convert import if needed
        """
        # Process the module first
        node = cast(ast.Module, self.generic_visit(node))

        # Only add the import if we actually transformed any divisions
        if not self.has_division_operations:
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
