from typing import List, cast
import ast

from . import ast_walk


class ImportLifterTransformer(ast_walk.NodeTransformer):
    """
    AST transformer that lifts all pynecore.lib related imports to module level.
    Does not transform the imports, just moves them to global scope.
    """

    def __init__(self):
        self.lifted_imports: List[ast.ImportFrom] = []

    @staticmethod
    def _is_lib_import(node: ast.ImportFrom) -> bool:
        """Check if an import is lib-related"""
        return bool(node.module and
                    (node.module == 'pynecore.lib' or
                     node.module.startswith('pynecore.lib.')))

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Process module and add lifted imports at the top"""
        # Process the entire module first to collect all imports
        node = cast(ast.Module, self.generic_visit(node))

        # No imports were lifted, return original
        if not self.lifted_imports:
            return node

        # Insert lifted imports after docstring if exists
        insert_pos = 1 if (node.body and isinstance(node.body[0], ast.Expr) and
                           isinstance(cast(ast.Expr, node.body[0]).value, ast.Constant)) else 0

        node.body[insert_pos:insert_pos] = self.lifted_imports
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Process function definitions and lift lib imports"""
        # Process function body first
        node = cast(ast.FunctionDef, self.generic_visit(node))

        # Collect lib imports and remove them from function body
        new_body = []
        for stmt in node.body:
            if isinstance(stmt, ast.ImportFrom) and self._is_lib_import(stmt):
                # Add to lifted imports if not already there
                if not any(self._imports_equal(stmt, lifted) for lifted in self.lifted_imports):
                    self.lifted_imports.append(stmt)
            else:
                new_body.append(stmt)

        node.body = new_body
        return node

    @staticmethod
    def _imports_equal(import1: ast.ImportFrom, import2: ast.ImportFrom) -> bool:
        """Compare two import statements for equality"""
        return (import1.module == import2.module and
                len(import1.names) == len(import2.names) and
                all(n1.name == n2.name and n1.asname == n2.asname
                    for n1, n2 in zip(import1.names, import2.names)))
