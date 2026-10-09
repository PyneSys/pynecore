from typing import cast
import ast

from . import ast_walk
from .phase import Rule
from .pine_type_rules import INT, get_ty, stamp_lowering


class _RangeBinding(ast_walk.NodeVisitor):
    """Find whether a module binds the name ``range`` anywhere: a definition,
    a stored name or an import alias."""

    def __init__(self) -> None:
        self.found = False

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef
                          | ast.ClassDef) -> None:
        if node.name == 'range':
            self.found = True
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == 'range' and isinstance(node.ctx, ast.Store):
            self.found = True

    def visit_alias(self, node: ast.alias) -> None:
        if (node.asname or node.name) == 'range':
            self.found = True


class SafeConvertTransformer(Rule):
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

    A subscript index is the other Python-native consumer: ``lst[n]``,
    ``text[n]``, ``lst[a:b]`` refuse a float where a ``Series`` truncates it
    itself. An index (or slice bound) typed as a Pine int is truncated the same
    way; when it is the cast itself, ``safe_int`` simply becomes ``native_int``.
    A series buffer read (``<state>[slot][n]``) is left alone: the buffer
    truncates in its own ``__getitem__``, and it is the hot loop.
    """

    def __init__(self, lib: bool = False):
        #: The module is a ``@pyne lib``: ``int()`` truncates to a native int
        self.lib = lib
        self.has_safe_convert_import = False
        self.has_convert_functions = False  # Track if float()/int() is used
        #: The module being transformed, None when the root is not a module
        self.module: ast.Module | None = None
        #: The module binds its own ``range`` (ta.range, array.range), so a
        #: ``range(...)`` call is not the builtin consumer; None until asked
        self.range_shadowed: bool | None = None
        #: Loop counters bound by an enclosing ``for ... in range(...)``
        self.range_vars: set[str] = set()
        #: Per enclosing ``for``, the counter it added to ``range_vars``, if any
        self._for_counters: list[str | None] = []

    def _is_builtin_range(self) -> bool:
        """
        Whether a ``range`` call is the builtin, i.e. the module binds no ``range``.

        Only a ``range`` call or loop asks, and most modules have none, so the
        whole-module scan waits for the first one. The rewrites done by then
        (this pass's, and those of the rules sharing its traversal) bind no
        ``range`` and drop no binding, so the scan answers as it would have
        before them.
        """
        if self.range_shadowed is None:
            scan = _RangeBinding()
            if self.module is not None:
                scan.visit(self.module)
            self.range_shadowed = scan.found
        return not self.range_shadowed

    def _native_int(self, arg: ast.expr) -> ast.expr:
        """Wrap a Python-native consumer's argument in the native truncation."""
        self.has_convert_functions = True
        return stamp_lowering(ast.copy_location(ast.Call(
            func=ast.Attribute(value=ast.Name(id='safe_convert', ctx=ast.Load()),
                               attr='native_int', ctx=ast.Load()),
            args=[arg], keywords=[]), arg), 'i')

    def _native_index(self, index: ast.expr) -> ast.expr:
        """Truncate a subscript index typed as a Pine int; leave any other alone."""
        if get_ty(index) != INT:
            return index
        if isinstance(index, ast.Constant):
            # A folded Pine int literal is a float (``2.0``): truncate it here
            value = index.value
            if isinstance(value, float) and value.is_integer():
                index.value = int(value)
            return index
        if isinstance(index, ast.Name) and index.id in self.range_vars:
            # The counter of a ``range()`` loop is already a native int
            return index
        if (isinstance(index, ast.Call) and isinstance(index.func, ast.Attribute)
                and index.func.attr == 'safe_int'
                and isinstance(index.func.value, ast.Name)
                and index.func.value.id == 'safe_convert'):
            # The index IS the cast: truncate natively instead of casting to a
            # Pine int and truncating that
            index.func.attr = 'native_int'
            return index
        return self._native_int(index)

    def enter_For(self, node: ast.For) -> None:
        """
        Remember a ``range()`` loop's counter as a native int while its loop is visited
        """
        counter = None
        if (isinstance(node.target, ast.Name) and isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range'
                and node.target.id not in self.range_vars and self._is_builtin_range()):
            counter = node.target.id
            self.range_vars.add(counter)
        self._for_counters.append(counter)

    def leave_For(self, node: ast.For) -> ast.AST:
        counter = self._for_counters.pop()
        if counter is not None:
            self.range_vars.discard(counter)
        return node

    def leave_Subscript(self, node: ast.Subscript) -> ast.AST:
        """
        Truncate a Pine int index for a Python-native container
        """
        # A series buffer read: the SeriesTransformer has rewritten the series
        # to its ``<state param>[slot]`` reference
        base = node.value
        if (isinstance(base, ast.Subscript) and isinstance(base.value, ast.Name)
                and base.value.id.startswith('__state') and base.value.id.endswith('__')):
            return node

        index = node.slice
        if isinstance(index, ast.Slice):
            if index.lower is not None:
                index.lower = self._native_index(index.lower)
            if index.upper is not None:
                index.upper = self._native_index(index.upper)
            if index.step is not None:
                index.step = self._native_index(index.step)
        elif not isinstance(index, ast.Tuple):
            node.slice = self._native_index(index)
        return node

    def leave_Call(self, node: ast.Call) -> ast.AST:
        """
        Transform float() and int() calls, and truncate the arguments of range()
        """
        if not isinstance(node.func, ast.Name):
            return node

        if node.func.id == 'range' and not node.keywords and self._is_builtin_range():
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

    def enter_Module(self, node: ast.Module) -> None:
        self.module = node

    def leave_Module(self, node: ast.Module) -> ast.Module:
        """
        Add safe_convert import if needed
        """
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
