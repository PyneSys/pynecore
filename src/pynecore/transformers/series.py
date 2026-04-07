from typing import cast, Any
import ast


class SeriesTransformer(ast.NodeTransformer):
    """Transform Series type variables in AST"""

    def __init__(self):
        # Mapping of scopes to variables
        self.series_vars: dict[str, dict[str, str]] = {}  # scope -> var_name -> series_name

        # Global SeriesImpl assignments to be added to module level
        self.series_assigns: list[ast.AST] = []

        # Current function tracking
        self.current_function: str | None = None
        self.parent_functions: list[str] = []

        # Tracking created Series
        self.collected_series: set[str] = set()

        # Import tracking
        self.has_series_import: bool = False

        # Track max_bars_back transformations to handle in visit_Expr
        self.pending_max_bars_back: dict[ast.Call, dict[str, Any]] = {}

    @staticmethod
    def _create_assign_with_lineno(targets, value, lineno=None, col_offset=None):
        """Create an ast.Assign node with proper line number information"""
        assign_node = ast.Assign(targets=targets, value=value)
        if lineno is not None:
            assign_node.lineno = lineno
        if col_offset is not None:
            assign_node.col_offset = col_offset
        return assign_node

    def _register_series(self, var_name: str, scope: str | None = None) -> str:
        """
        Register a Series variable and return its global instance name.

        Args:
            var_name: The variable name to register
            scope: The scope where the variable is defined (default: current scope)

        Returns:
            str: The generated global instance name
        """
        resolved_scope: str = scope if scope is not None else self._get_current_scope()

        series_name = f'__series_{resolved_scope}·{var_name}__'

        if resolved_scope not in self.series_vars:
            self.series_vars[resolved_scope] = {}
        self.series_vars[resolved_scope][var_name] = series_name
        self.collected_series.add(series_name)

        return series_name

    def _get_current_scope(self) -> str:
        """
        Get current scope path.

        Returns:
            str: The current scope path
        """
        if not self.current_function:
            return ""
        return "·".join(self.parent_functions + [self.current_function])

    def _get_series_in_current_scope(self, var_name: str) -> str | None:
        """
        Get series name from current or parent scopes.

        Args:
            var_name: The variable name to look up

        Returns:
            str or None: The series name if found, None otherwise
        """
        current_scope = self._get_current_scope()
        if current_scope in self.series_vars:
            return self.series_vars[current_scope].get(var_name)
        return None

    # noinspection PyShadowingBuiltins
    def visit_Module(self, node: ast.Module) -> ast.Module:
        """
        Create global SeriesImpl instances and register function-variable mapping.

        Args:
            node: The AST module node

        Returns:
            ast.Module: The transformed module
        """
        node = cast(ast.Module, self.generic_visit(node))

        if not self.collected_series:
            return node

        # Create function-to-variables mapping dictionary
        function_vars_dict = {}

        # First collect variables for all functions
        for scope, vars_dict in self.series_vars.items():
            # Convert middle dots to dots in key name only
            function_name = scope.replace('·', '.') if scope else "main"
            # Only add non-empty scopes to the registry
            if scope and vars_dict:
                if function_name not in function_vars_dict:
                    function_vars_dict[function_name] = []

                # Add all series variables from this scope
                for _, global_name in vars_dict.items():
                    function_vars_dict[function_name].append(global_name)

        # Special handling for 'main' function
        main_vars = self.series_vars.get('main', {})
        if main_vars:
            function_vars_dict['main'] = [series_name for _, series_name in main_vars.items()]

        # Create the function variable registry
        function_vars_assign = self._create_assign_with_lineno(
            targets=[ast.Name(id='__series_function_vars__', ctx=ast.Store())],
            value=ast.Dict(
                keys=[ast.Constant(value=k) for k in function_vars_dict],
                values=[
                    ast.Tuple(
                        elts=[ast.Constant(value=var) for var in vars],
                        ctx=ast.Load()
                    )
                    for vars in function_vars_dict.values()
                ]
            ),
            lineno=1
        )

        # Create SeriesImpl import and instances
        imports = [
            ast.ImportFrom(
                module='pynecore.core.series',
                names=[ast.alias(name='SeriesImpl', asname=None)],
                level=0
            )
        ]

        assignments = [
            self._create_assign_with_lineno(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id='SeriesImpl', ctx=ast.Load()),
                    args=[],
                    keywords=[]
                ),
                lineno=1
            )
            for name in sorted(self.collected_series)
        ]

        # Add the function-variable registry
        assignments.append(function_vars_assign)

        # Insert after docstring if exists
        if (node.body and isinstance(node.body[0], ast.Expr) and
                isinstance(cast(ast.Expr, node.body[0]).value, ast.Constant)):
            node.body = [node.body[0]] + imports + node.body[1:]
        else:
            node.body = imports + node.body

        # Find position of first function or assignment
        insert_index = 0
        for i, stmt in enumerate(node.body):
            if isinstance(stmt, (ast.FunctionDef, ast.Assign, ast.AnnAssign)):
                insert_index = i
                break
            # Skip only docstring and imports
            if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and
                    isinstance(stmt.value.value, str) or
                    isinstance(stmt, (ast.Import, ast.ImportFrom))):
                insert_index = i
                break

        # Split body and insert assignments
        pre_body = node.body[:insert_index]
        post_body = node.body[insert_index:]

        node.body = pre_body + assignments + post_body
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """
        Handle function scope and Series parameters.

        Args:
            node: The function definition node

        Returns:
            ast.FunctionDef: The transformed function
        """
        # Store old state
        old_function = self.current_function

        # Special handling for main function
        if node.name == "main":
            self.current_function = "main"
        else:
            if self.current_function:
                self.parent_functions.append(self.current_function)
            self.current_function = node.name

        # Create parameter initializations
        series_initializations = []
        for arg in node.args.args:
            if arg.annotation and self._is_series_type(arg.annotation):
                series_name = self._register_series(arg.arg)
                # Extract inner type from Series[T]
                if isinstance(arg.annotation, ast.Subscript):
                    arg.annotation = arg.annotation.slice
                else:
                    # If no type parameter, remove annotation
                    arg.annotation = None
                # Create initialization statement
                series_initializations.append(
                    self._create_assign_with_lineno(
                        targets=[ast.Name(id=arg.arg, ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id=series_name, ctx=ast.Load()),
                                attr='add',
                                ctx=ast.Load()
                            ),
                            args=[ast.Name(id=arg.arg, ctx=ast.Load())],
                            keywords=[]
                        ),
                        lineno=node.lineno if hasattr(node, 'lineno') else 1
                    )
                )

        # Process function body with correct context
        node = cast(ast.FunctionDef, self.generic_visit(node))

        # Find the right position to insert initializations after docstring if exists
        insert_pos = 0
        first_stmt = node.body[0] if node.body else None
        if (isinstance(first_stmt, ast.Expr) and
                isinstance(first_stmt.value, ast.Constant) and
                isinstance(first_stmt.value.value, str)):
            insert_pos = 1

        # Insert series initializations after docstring
        if series_initializations:
            node.body[insert_pos:insert_pos] = series_initializations

        # Restore function context
        if self.parent_functions:
            self.current_function = self.parent_functions.pop()
        else:
            self.current_function = old_function

        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | ast.Assign | None:
        """
        Handle Series type annotations and first value assignment.

        Args:
            node: The annotated assignment node

        Returns:
            AST node: The transformed node
        """
        if not isinstance(node.target, ast.Name):
            return node

        if self._is_series_type(node.annotation):
            var_name = node.target.id
            series_name = self._register_series(var_name)

            if node.value is None:
                return None

            # First assignment uses add()
            return self._create_assign_with_lineno(
                targets=[ast.Name(id=var_name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=series_name, ctx=ast.Load()),
                        attr='add',
                        ctx=ast.Load()
                    ),
                    args=[self.visit(cast(ast.AST, node.value))],
                    keywords=[]
                ),
                lineno=node.lineno if hasattr(node, 'lineno') else 1,
                col_offset=node.col_offset if hasattr(node, 'col_offset') else 0
            )

        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> ast.AST | ast.Assign:
        """
        Handle Series value assignments using set().

        Args:
            node: The assignment node

        Returns:
            AST node: The transformed node
        """
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return self.generic_visit(node)

        var_name = cast(ast.Name, node.targets[0]).id
        series_name = self._get_series_in_current_scope(var_name)

        if series_name:
            # Regular assignment uses set()
            return self._create_assign_with_lineno(
                targets=[ast.Name(id=var_name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=series_name, ctx=ast.Load()),
                        attr='set',
                        ctx=ast.Load()
                    ),
                    args=[self.visit(cast(ast.AST, node.value))],
                    keywords=[]
                ),
                lineno=node.lineno if hasattr(node, 'lineno') else 1,
                col_offset=node.col_offset if hasattr(node, 'col_offset') else 0
            )

        return self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST | ast.Assign:
        """
        Handle augmented assignments (+=, -=, etc).

        Args:
            node: The augmented assignment node

        Returns:
            AST node: The transformed node
        """
        if not isinstance(node.target, ast.Name):
            return self.generic_visit(node)

        var_name = node.target.id
        series_name = self._get_series_in_current_scope(var_name)

        if series_name:
            # Convert augmented assignment to set() with operation
            return self._create_assign_with_lineno(
                targets=[ast.Name(id=var_name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=series_name, ctx=ast.Load()),
                        attr='set',
                        ctx=ast.Load()
                    ),
                    args=[
                        ast.BinOp(
                            left=ast.Name(id=var_name, ctx=ast.Load()),
                            op=node.op,
                            right=self.visit(cast(ast.AST, node.value))
                        )
                    ],
                    keywords=[]
                ),
                lineno=node.lineno if hasattr(node, 'lineno') else 1,
                col_offset=node.col_offset if hasattr(node, 'col_offset') else 0
            )

        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST | ast.Name:
        """
        Transform Series references - ONLY when parent is indexing.

        Args:
            node: The name node

        Returns:
            AST node: The transformed node
        """
        if isinstance(node.ctx, ast.Load):
            series_name = self._get_series_in_current_scope(node.id)
            if series_name:
                parent = getattr(node, 'parent', None)
                if isinstance(parent, ast.Subscript) and parent.value == node:
                    return ast.Name(id=series_name, ctx=node.ctx)
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        """
        Set parent for indexing operations.

        Args:
            node: The subscript node

        Returns:
            AST node: The transformed node
        """
        if isinstance(node.value, ast.AST):
            setattr(node.value, 'parent', node)
        node.value = self.visit(cast(ast.AST, node.value))
        node.slice = self.visit(cast(ast.AST, node.slice))
        return node

    def generic_visit(self, node: ast.AST):
        """
        Override generic_visit to ensure all Subscript nodes are visited.
        """
        # For all nodes, make sure we visit their children
        # This ensures Subscript nodes in conditional expressions are processed
        return super().generic_visit(node)

    @staticmethod
    def _is_series_type(annotation: ast.expr) -> bool:
        """
        Check if type annotation is Series.

        Args:
            annotation: The type annotation expression

        Returns:
            bool: True if the type is Series, False otherwise
        """
        if isinstance(annotation, ast.Subscript):
            return (isinstance(annotation.value, ast.Name) and
                    annotation.value.id == 'Series')
        elif isinstance(annotation, ast.Name):
            return annotation.id == 'Series'
        return False

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """
        Handle lib.max_bars_back() calls and transform them to Series method calls.
        Also processes all child nodes to ensure Subscript nodes are visited.

        Args:
            node: The call node

        Returns:
            AST node: The transformed node
        """
        # First, visit all child nodes to ensure Subscript nodes in arguments are processed
        node = cast(ast.Call, self.generic_visit(node))

        # Check if this is a lib.max_bars_back call
        if (isinstance(node.func, ast.Attribute) and
                node.func.attr == 'max_bars_back' and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'lib' and
                len(node.args) >= 2):

            # Get the source variable name from first argument
            if isinstance(node.args[0], ast.Name):
                var_name = cast(ast.Name, node.args[0]).id
                series_name = self._get_series_in_current_scope(var_name)

                if series_name:
                    # Mark this call for transformation in visit_Expr
                    self.pending_max_bars_back[node] = {
                        'series_name': series_name,
                        'value': node.args[1],
                        'lineno': getattr(node, 'lineno', 1),
                        'col_offset': getattr(node, 'col_offset', 0)
                    }
                    # Return a placeholder None (max_bars_back returns None)
                    placeholder = ast.Constant(value=None)
                    if hasattr(node, 'lineno'):
                        placeholder.lineno = node.lineno
                    if hasattr(node, 'col_offset'):
                        placeholder.col_offset = node.col_offset
                    return cast(ast.AST, placeholder)

        return node

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        """
        Handle expression statements, particularly for max_bars_back transformations.
        """
        # Visit the expression first
        node = cast(ast.Expr, self.generic_visit(node))

        # Check if this expression is a pending max_bars_back transformation
        if isinstance(node.value, ast.Constant) and node.value.value is None:
            # Look for the original call in pending transformations
            for original_call, transform_info in self.pending_max_bars_back.items():
                # This is a bit hacky, but we need to match the transformed placeholder
                # Since we replaced the call with Constant(None), we check line numbers
                if (hasattr(node.value, 'lineno') and hasattr(original_call, 'lineno') and
                        node.value.lineno == original_call.lineno):
                    # Replace the Expr with an Assign
                    assign_node = self._create_assign_with_lineno(
                        targets=[
                            ast.Attribute(
                                value=ast.Name(id=transform_info['series_name'], ctx=ast.Load()),
                                attr='max_bars_back',
                                ctx=ast.Store()
                            )
                        ],
                        value=transform_info['value'],
                        lineno=transform_info['lineno'],
                        col_offset=transform_info['col_offset']
                    )
                    return cast(ast.AST, assign_node)

        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST | None:
        """
        Handle imports, remove Series import.

        Args:
            node: The import statement

        Returns:
            AST node or None: The transformed node or None if removed
        """
        if node.module and node.module.startswith('pynecore'):
            # Filter out Series from names
            new_names = [name for name in node.names if name.name != 'Series']
            if not new_names:
                # If no names left, remove the entire import
                return None
            # Create new import with remaining names
            node.names = new_names
            return node
        return node
