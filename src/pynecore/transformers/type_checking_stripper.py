import ast


class TypeCheckingStripperTransformer(ast.NodeTransformer):
    """
    Remove `if TYPE_CHECKING:` blocks and the TYPE_CHECKING import from @pyne files.
    These blocks contain IDE-only type hints (casts, re-annotations) that are unnecessary at runtime.
    """

    def visit_If(self, node: ast.If) -> ast.AST | list[ast.AST] | None:
        # Match: if TYPE_CHECKING: / if typing.TYPE_CHECKING:
        if ((isinstance(node.test, ast.Name) and node.test.id == 'TYPE_CHECKING')
                or (isinstance(node.test, ast.Attribute) and node.test.attr == 'TYPE_CHECKING'
                    and isinstance(node.test.value, ast.Name)
                    and node.test.value.id == 'typing')):
            # The name is False at runtime: what runs is the else body
            return [self.visit(stmt) for stmt in node.orelse] or None
        return self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom | None:
        if node.module == 'typing':
            node.names = [alias for alias in node.names if alias.name != 'TYPE_CHECKING']
            if not node.names:
                return None
        return node