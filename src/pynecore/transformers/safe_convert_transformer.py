from typing import cast
import ast

from .pine_type_rules import get_ty, stamp_lowering


class SafeConvertTransformer(ast.NodeTransformer):
    """
    Transformer that converts float(na) and int(na) calls to safe alternatives
    that preserve Pine Script semantics.

    This transformer replaces float() and int() function calls with safe_float()
    and safe_int() from pynecore.core.safe_convert module, to ensure proper
    handling of NA values. ``int()`` is Pine's cast: a Pine int is a double at
    runtime, so ``safe_int`` hands back a float. Inside a ``@pyne lib`` module
    ``int()`` is the lib's own truncation for a length, a count or a ring index,
    computed in native int, so there it becomes ``native_int``.

    A ``range()`` argument is a Python-native consumer of a Pine int: the value
    it receives may be a float carrying an integral value (a bar index, an
    ``array.size``, a counter), so every argument that is not an int literal is
    truncated with ``native_int`` -- the same truncation-at-the-consumer that
    the lib applies to its own indexes.
    """

    def __init__(self, lib: bool = False):
        #: The module is a ``@pyne lib``: ``int()`` truncates to a native int
        self.lib = lib
        self.has_safe_convert_import = False
        self.has_convert_functions = False  # Track if float()/int() is used
        #: The module binds its own ``range`` (ta.range, array.range), so a
        #: ``range(...)`` call is not the builtin consumer
        self.range_shadowed = False

    def _native_int(self, arg: ast.expr) -> ast.expr:
        """Wrap a Python-native consumer's argument in the native truncation."""
        self.has_convert_functions = True
        return stamp_lowering(ast.copy_location(ast.Call(
            func=ast.Attribute(value=ast.Name(id='safe_convert', ctx=ast.Load()),
                               attr='native_int', ctx=ast.Load()),
            args=[arg], keywords=[]), arg), 'i')

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """
        Visit Call nodes and transform float() and int() calls
        """
        # Continue normal transformation for children
        self.generic_visit(node)

        if not isinstance(node.func, ast.Name):
            return node

        # Check for the builtin module
        if hasattr(node.func, 'module') and getattr(node.func, 'module') == 'builtins':
            return node

        if node.func.id == 'range' and not node.keywords and not self.range_shadowed:
            node.args = [arg if isinstance(arg, ast.Starred)
                         or isinstance(arg, ast.Constant) and type(arg.value) is int
                         else self._native_int(arg) for arg in node.args]
            return node

        # Check if it's a built-in float() or int() call
        if node.func.id in ('float', 'int'):
            # Mark that we need the safe_convert import
            self.has_convert_functions = True

            # Transform to safe_convert.safe_float/safe_int call, keeping the
            # cast's Pine type: the na-safe form converts what the builtin
            # converts
            if node.func.id == 'int' and self.lib:
                attr = 'native_int'
            else:
                attr = f'safe_{node.func.id}'
            return stamp_lowering(ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='safe_convert', ctx=ast.Load()),
                    attr=attr,
                    ctx=ast.Load()
                ),
                args=node.args,
                keywords=node.keywords
            ), get_ty(node))

        return node

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """
        Add safe_convert import if needed
        """
        self.range_shadowed = any(
            (isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
             and child.name == 'range')
            or (isinstance(child, ast.Name) and child.id == 'range'
                and isinstance(child.ctx, ast.Store))
            or (isinstance(child, ast.alias) and (child.asname or child.name) == 'range')
            for child in ast.walk(node))

        # Process the module first
        node = cast(ast.Module, self.generic_visit(node))

        # Only add the import if we actually transformed any functions
        if not self.has_convert_functions:
            return node

        # An existing binding of the ``safe_convert`` module name is enough
        for stmt in node.body:
            if isinstance(stmt, ast.ImportFrom):
                module = (stmt.module or '').removeprefix('pynecore.')
                if module == 'core.safe_convert':
                    bound = any(alias.asname == 'safe_convert' for alias in stmt.names)
                elif module == 'core':
                    bound = any((alias.asname or alias.name) == 'safe_convert'
                                for alias in stmt.names)
                else:
                    continue
            elif isinstance(stmt, ast.Import):
                bound = any(alias.name == 'pynecore.core.safe_convert'
                            and alias.asname == 'safe_convert' for alias in stmt.names)
            else:
                continue
            if bound:
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
