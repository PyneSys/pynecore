"""
Unused Series Detector Transformer

This transformer detects Series[T] annotated variables that are never actually indexed
and removes the Series annotation to optimize performance.

It runs before SeriesTransformer in the AST transformation pipeline.
"""

import ast
from typing import Set, Dict


def _is_in_annotation_context(node: ast.Subscript) -> bool:
    """Check if a subscript is part of a type annotation"""
    # This is a simple heuristic: if the subscript value is "Series",
    # it's likely a type annotation
    if isinstance(node.value, ast.Name) and node.value.id == "Series":
        return True
    return False


class UnusedSeriesDetectorTransformer(ast.NodeTransformer):
    """
    AST transformer that removes unnecessary Series annotations.

    For variables annotated as Series[T] but never indexed with subscript operator [],
    this transformer changes their type to just T, avoiding unnecessary Series overhead.
    """

    def __init__(self):
        # Track variables with their scope
        self.series_vars: Dict[str, Set[str]] = {}  # scope -> set of variable names
        self.indexed_vars: Dict[str, Set[str]] = {}  # scope -> set of variable names
        self.current_scope: str = "__module__"
        self.scope_stack: list[str] = []
        # Track local variables in each scope to handle shadowing
        self.local_vars: Dict[str, Set[str]] = {}  # scope -> set of local variable names

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Track function scope and process function definitions"""
        # Save current scope
        self.scope_stack.append(self.current_scope)
        self.current_scope = f"{self.current_scope}.{node.name}" if self.current_scope != "__module__" else node.name

        # Process the function
        self.generic_visit(node)

        # Restore scope
        self.current_scope = self.scope_stack.pop()

        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign:
        """Detect Series annotations in variable assignments"""
        if isinstance(node.target, ast.Name):
            var_name = node.target.id

            # Check if this is a Series annotation
            if self._is_series_annotation(node.annotation):
                if self.current_scope not in self.series_vars:
                    self.series_vars[self.current_scope] = set()
                self.series_vars[self.current_scope].add(var_name)

                # Track local variables to handle shadowing
                if self.current_scope not in self.local_vars:
                    self.local_vars[self.current_scope] = set()
                self.local_vars[self.current_scope].add(var_name)

        # Continue visiting child nodes
        self.generic_visit(node)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        """Detect Series annotations in function arguments"""
        if node.annotation and self._is_series_annotation(node.annotation):
            if self.current_scope not in self.series_vars:
                self.series_vars[self.current_scope] = set()
            self.series_vars[self.current_scope].add(node.arg)

            # Track local variables to handle shadowing
            if self.current_scope not in self.local_vars:
                self.local_vars[self.current_scope] = set()
            self.local_vars[self.current_scope].add(node.arg)

        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.Subscript:
        """Track which variables are actually indexed"""
        # Only track subscripts that are NOT type annotations
        # Type annotations like Series[float] should not be counted as indexing
        if isinstance(node.value, ast.Name) and not _is_in_annotation_context(node):
            var_name = node.value.id

            # Mark this variable as indexed
            self._mark_variable_as_indexed(var_name)

        # Continue visiting child nodes
        self.generic_visit(node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        """Just visit child nodes - we don't need to track function calls"""
        # The key insight: if a variable is passed to a function, that doesn't mean
        # it needs to be Series in the calling scope. Only direct indexing matters.
        # The ClosureArgumentsTransformer preserves Series annotations on parameters
        # that actually get indexed, so we can rely on that.

        # Continue visiting child nodes
        self.generic_visit(node)
        return node

    def _mark_variable_as_indexed(self, var_name: str):
        """Mark a variable as indexed in the current scope"""
        if self.current_scope not in self.indexed_vars:
            self.indexed_vars[self.current_scope] = set()
        self.indexed_vars[self.current_scope].add(var_name)

    @staticmethod
    def _is_series_annotation(annotation: ast.AST) -> bool:
        """Check if an annotation is Series[T]"""
        if isinstance(annotation, ast.Subscript):
            if isinstance(annotation.value, ast.Name) and annotation.value.id == "Series":
                return True
            elif isinstance(annotation.value, ast.Attribute):
                # Handle cases like types.Series
                if annotation.value.attr == "Series":
                    return True
        return False

    @staticmethod
    def _get_inner_type(annotation: ast.AST) -> ast.expr:
        """Extract T from Series[T]"""
        if isinstance(annotation, ast.Subscript) and isinstance(annotation.slice, ast.Name):
            return annotation.slice
        elif isinstance(annotation, ast.Subscript) and hasattr(annotation.slice, 'value'):
            # Handle ast.Index in older Python versions
            return getattr(annotation.slice, 'value')
        # Default to Any if we can't determine the inner type
        return ast.Name(id='Any', ctx=ast.Load())

    def _resolve_series_owner(self, scope: str, var_name: str) -> str | None:
        """Nearest enclosing scope (including ``scope`` itself) that declares
        ``var_name`` as a Series.

        Mirrors :meth:`SeriesTransformer._lookup`: a subscript resolves to the
        closest series declaration walking outwards through the scope chain.
        """
        parts = scope.split('.')
        for i in range(len(parts), 0, -1):
            candidate = '.'.join(parts[:i])
            if var_name in self.series_vars.get(candidate, set()):
                return candidate
        if var_name in self.series_vars.get("__module__", set()):
            return "__module__"
        return None

    def optimize(self, tree: ast.AST) -> ast.AST:
        """
        Main optimization pass - removes unused Series annotations
        """
        # First pass: collect all Series variables and indexed variables
        self.visit(tree)

        # A parent-scope series indexed only from within a nested function is
        # resolved through the scope chain by SeriesTransformer, so it must not
        # be pruned. Credit the nearest enclosing scope that declares each
        # indexed name as a Series, mirroring that resolution.
        for scope, names in list(self.indexed_vars.items()):
            for var_name in names:
                owner = self._resolve_series_owner(scope, var_name)
                if owner is not None and owner != scope:
                    self.indexed_vars.setdefault(owner, set()).add(var_name)

        # Second pass: remove Series annotations from non-indexed variables
        optimizer = SeriesOptimizer(self.series_vars, self.indexed_vars)
        return optimizer.visit(tree)


class SeriesOptimizer(ast.NodeTransformer):
    """Second pass transformer that actually removes the unused Series annotations"""

    def __init__(self, series_vars: Dict[str, Set[str]], indexed_vars: Dict[str, Set[str]]):
        self.series_vars = series_vars
        self.indexed_vars = indexed_vars
        self.current_scope = "__module__"
        self.scope_stack: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Track function scope"""
        self.scope_stack.append(self.current_scope)
        self.current_scope = f"{self.current_scope}.{node.name}" if self.current_scope != "__module__" else node.name

        # Process the function
        self.generic_visit(node)

        # Restore scope
        self.current_scope = self.scope_stack.pop()

        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign:
        """Remove Series annotation if variable is never indexed"""
        if isinstance(node.target, ast.Name):
            var_name = node.target.id

            # Check if this is an unused Series variable
            if self._is_unused_series(var_name) and self._is_series_annotation(node.annotation):
                # Replace Series[T] with just T
                node.annotation = self._get_inner_type(node.annotation)

        self.generic_visit(node)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        """Remove Series annotation from function arguments if never indexed"""
        if (node.annotation and self._is_unused_series(node.arg) and
                self._is_series_annotation(node.annotation)):
            # Replace Series[T] with just T
            node.annotation = self._get_inner_type(node.annotation)

        return node

    def _is_unused_series(self, var_name: str) -> bool:
        """Check if a variable is Series-annotated but never indexed in the current scope"""
        # Check if variable is Series in current scope
        is_series = (self.current_scope in self.series_vars and
                     var_name in self.series_vars[self.current_scope])

        # Check if variable is indexed in current scope
        is_indexed = (self.current_scope in self.indexed_vars and
                      var_name in self.indexed_vars[self.current_scope])

        # Return True if it's Series but not indexed in this scope
        return is_series and not is_indexed

    def _get_relevant_scopes(self) -> list[str]:
        """Get all scopes that could contain the variable (current + parents)"""
        scopes = ["__module__"]
        if self.current_scope != "__module__":
            parts = self.current_scope.split('.')
            for i in range(len(parts)):
                scopes.append('.'.join(parts[:i + 1]))
        return scopes

    @staticmethod
    def _is_series_annotation(annotation: ast.AST) -> bool:
        """Check if an annotation is Series[T]"""
        if isinstance(annotation, ast.Subscript):
            if isinstance(annotation.value, ast.Name) and annotation.value.id == "Series":
                return True
            elif isinstance(annotation.value, ast.Attribute):
                if annotation.value.attr == "Series":
                    return True
        return False

    @staticmethod
    def _get_inner_type(annotation: ast.AST) -> ast.expr:
        """Extract T from Series[T]"""
        if isinstance(annotation, ast.Subscript):
            if isinstance(annotation.slice, ast.Name):
                return annotation.slice
            elif hasattr(annotation.slice, 'value'):
                # Handle ast.Index in older Python versions
                return getattr(annotation.slice, 'value')
        # Default to float if we can't determine
        return ast.Name(id='float', ctx=ast.Load())
