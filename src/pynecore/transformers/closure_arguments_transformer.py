"""
Closure Arguments Transformer

This transformer runs before function_isolation and converts closure variables in
inner functions to function arguments. It only processes functions in the main
function that is decorated with @lib.script.indicator or @lib.script.strategy.

Example:
    # Before transformation:
    @lib.script.indicator("Test", overlay=True)
    def main():
        length = lib.input.int(14)
        multiplier = 2.0

        def calculate(offset=0):
            return lib.ta.sma(lib.close, length) * multiplier + offset

        return calculate() + calculate(10)

    # After transformation:
    @lib.script.indicator("Test", overlay=True)
    def main():
        length = lib.input.int(14)
        multiplier = 2.0

        def calculate(length, multiplier, offset=0):  # closure vars added at beginning
            return lib.ta.sma(lib.close, length) * multiplier + offset

        return calculate(length, multiplier) + calculate(length, multiplier, 10)
"""

import ast
from typing import Set, Dict, List, Optional, cast, Any


class ClosureArgumentsTransformer(ast.NodeTransformer):
    """Transform closure variables in inner functions to function arguments."""

    def __init__(self):
        # Track if we're in a decorated main function
        self.in_main_function = False
        # Track current function scope
        self.current_function: Optional[str] = None
        # Stack of function scopes for nested functions
        self.function_stack: List[str] = []
        # Variables defined in each scope
        self.scope_variables: Dict[str, Set[str]] = {}
        # Track function definitions to update calls
        self.inner_functions: Dict[str, ast.FunctionDef] = {}
        # Track closure variables for each inner function
        self.closure_vars: Dict[str, Set[str]] = {}
        # Track type annotations for closure variables
        self.closure_var_types: Dict[str, ast.AST] = {}

    def visit_Module(self, node: ast.Module) -> ast.Module:
        # First pass: collect all function definitions and their closure variables
        collector = ClosureVariableCollector()
        collector.visit(node)
        self.scope_variables = collector.scope_variables
        self.closure_vars = collector.closure_vars
        self.closure_var_types = collector.closure_var_types

        # Second pass: transform the functions
        return cast(ast.Module, self.generic_visit(node))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        # Check if this is the main function with required decorators
        is_main_decorated = False
        if node.name == 'main':
            for decorator in node.decorator_list:
                decorator_name = self._get_decorator_name(decorator)
                if decorator_name in ('lib.script.indicator', 'lib.script.strategy',
                                      'script.indicator', 'script.strategy'):
                    is_main_decorated = True
                    break

        # Store old state
        old_in_main = self.in_main_function
        old_current = self.current_function

        # Update state
        if is_main_decorated:
            self.in_main_function = True
        self.current_function = node.name
        if self.current_function:
            self.function_stack.append(self.current_function)

        # Process inner functions if we're in main
        if self.in_main_function and old_current == 'main' and node.name != 'main':
            # This is an inner function in main
            func_key = self._get_function_key(node.name)

            # Check if function has closure variables
            if func_key in self.closure_vars and self.closure_vars[func_key]:
                # Add closure variables as parameters at the beginning
                closure_vars = sorted(self.closure_vars[func_key])
                new_args = []
                for var in closure_vars:
                    # Get type annotation for this closure variable
                    parent_scope = 'main'  # closure vars come from main scope
                    var_key = f"{parent_scope}.{var}"
                    annotation = self.closure_var_types.get(var_key, None)

                    # If annotation is Persistent[T], extract the inner type T
                    if annotation and self._is_persistent_annotation(annotation):
                        annotation = self._extract_inner_type(annotation)

                    # Add closure vars at the beginning with processed annotation
                    new_args.append(ast.arg(arg=var, annotation=cast(ast.expr, annotation)))
                # Add original args after closure vars
                new_args.extend(node.args.args)
                node.args.args = new_args

                # Mark this function as having closure arguments transformed
                # This will be used by function_isolation transformer
                setattr(node, '_has_closure_arguments', True)
                setattr(node, '_closure_vars_count', len(closure_vars))

                # Also store the closure vars list for reference
                setattr(node, '_closure_vars', closure_vars)
            else:
                # No closure variables, but still mark it
                setattr(node, '_has_closure_arguments', True)
                setattr(node, '_closure_vars_count', 0)
                setattr(node, '_closure_vars', [])

            # Store the function definition
            self.inner_functions[node.name] = node

        # Visit the function body
        node = cast(ast.FunctionDef, self.generic_visit(node))

        # Restore state
        self.in_main_function = old_in_main
        self.current_function = old_current
        if self.function_stack:
            self.function_stack.pop()

        return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        # First visit children
        node = cast(ast.Call, self.generic_visit(node))

        # Check if we're calling an inner function that needs closure arguments
        if self.in_main_function and isinstance(node.func, ast.Name):
            func_name = node.func.id

            # Handle regular function calls
            if func_name in self.inner_functions:
                # Get the closure variables for this function
                func_key = self._get_function_key(func_name)
                if func_key in self.closure_vars and self.closure_vars[func_key]:
                    # Add closure variables as arguments at the beginning
                    closure_vars = sorted(self.closure_vars[func_key])
                    new_args = []
                    for var in closure_vars:
                        new_args.append(ast.Name(id=var, ctx=ast.Load()))
                    # Add original args after closure vars
                    new_args.extend(node.args)
                    node.args = new_args

                    # Mark this call as having closure arguments
                    # This will be used by function_isolation transformer
                    setattr(node, '_has_closure_arguments', True)
                    setattr(node, '_closure_vars_count', len(closure_vars))
                else:
                    # No closure variables for this function
                    setattr(node, '_has_closure_arguments', True)
                    setattr(node, '_closure_vars_count', 0)

            # Handle method_call() calls - these need special handling
            elif func_name == 'method_call' and len(node.args) >= 2:
                method_name = None

                # First argument can be either a string literal or a function reference
                first_arg = node.args[0]
                if (isinstance(first_arg, ast.Constant) and
                        isinstance(first_arg.value, str)):
                    # method_call('method_name', this_object, ...) format
                    method_name = first_arg.value
                elif isinstance(first_arg, ast.Name):
                    # method_call(method_function, this_object, ...) format
                    method_name = first_arg.id

                if method_name:
                    # Check if this method name corresponds to an inner function
                    if method_name in self.inner_functions:
                        # Get the closure variables for this function
                        func_key = self._get_function_key(method_name)
                        if func_key in self.closure_vars and self.closure_vars[func_key]:
                            # Add closure variables as arguments
                            # For method_call: method_call(method_ref, closure_vars..., this_obj, original_args...)
                            closure_vars = sorted(self.closure_vars[func_key])
                            new_args: List[ast.expr] = [node.args[0]]

                            # Add closure variables after method name
                            for var in closure_vars:
                                new_args.append(ast.Name(id=var, ctx=ast.Load()))

                            # Add this object after closure vars
                            new_args.append(node.args[1])

                            # Add original args after this object (skip first 2 which are method name and this)
                            new_args.extend(node.args[2:])

                            node.args = new_args

                            # Mark this call as having closure arguments
                            setattr(node, '_has_closure_arguments', True)
                            setattr(node, '_closure_vars_count', len(closure_vars))
                        else:
                            # No closure variables for this method
                            setattr(node, '_has_closure_arguments', True)
                            setattr(node, '_closure_vars_count', 0)

        return node

    def _get_decorator_name(self, decorator: Any) -> Optional[str]:
        """Get the full name of a decorator."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            parts = []
            current = decorator
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return '.'.join(reversed(parts))
        elif isinstance(decorator, ast.Call):
            return self._get_decorator_name(decorator.func)
        return None

    @staticmethod
    def _get_function_key(func_name: str) -> str:
        """Get unique key for a function based on its scope."""
        return 'main.' + func_name

    @staticmethod
    def _is_persistent_annotation(annotation: ast.AST) -> bool:
        """Check if annotation is Persistent[T] or just Persistent."""
        if isinstance(annotation, ast.Name):
            return annotation.id == 'Persistent'
        elif isinstance(annotation, ast.Subscript):
            if isinstance(annotation.value, ast.Name):
                return annotation.value.id == 'Persistent'
        return False

    @staticmethod
    def _extract_inner_type(annotation: ast.AST) -> Optional[ast.AST]:
        """Extract inner type T from Persistent[T] annotation."""
        if isinstance(annotation, ast.Subscript):
            # Persistent[T] -> return T
            return annotation.slice
        elif isinstance(annotation, ast.Name) and annotation.id == 'Persistent':
            # Just Persistent -> return None (no specific type)
            return None
        return annotation


class ClosureVariableCollector(ast.NodeVisitor):
    """Collect closure variables for inner functions."""

    def __init__(self):
        # Current function being analyzed
        self.current_function: Optional[str] = None
        # Stack of function scopes
        self.function_stack: List[str] = []
        # Variables defined in each scope
        self.scope_variables: Dict[str, Set[str]] = {}
        # Variables used in each scope
        self.scope_uses: Dict[str, Set[str]] = {}
        # Closure variables for each inner function
        self.closure_vars: Dict[str, Set[str]] = {}
        # Type annotations for closure variables
        self.closure_var_types: Dict[str, ast.AST] = {}
        # Track if we're in the main function
        self.in_main_function = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Check if this is the main function
        is_main = node.name == 'main' and self._has_required_decorator(node)

        # Store old state
        old_function = self.current_function
        old_in_main = self.in_main_function

        # Update state
        if is_main:
            self.in_main_function = True
        self.current_function = node.name
        if self.current_function:
            self.function_stack.append(self.current_function)
            scope_key = self._get_scope_key()
            self.scope_variables[scope_key] = set()
            self.scope_uses[scope_key] = set()

            # Add function parameters to scope variables
            for arg in node.args.args:
                self.scope_variables[scope_key].add(arg.arg)

        # Visit function body
        self.generic_visit(node)

        # Calculate closure variables if this is an inner function in main
        if self.in_main_function and len(self.function_stack) > 1 and self.function_stack[0] == 'main':
            scope_key = self._get_scope_key()
            parent_scope = '.'.join(self.function_stack[:-1])

            # Find variables used in this scope but defined in parent scope
            found_vars: Set[str] = set()
            for var in self.scope_uses.get(scope_key, set()):
                if var not in self.scope_variables.get(scope_key, set()):
                    # Check if variable is defined in parent scope
                    if var in self.scope_variables.get(parent_scope, set()):
                        found_vars.add(var)

            if found_vars:
                self.closure_vars[scope_key] = found_vars

        # Restore state
        self.current_function = old_function
        self.in_main_function = old_in_main
        if self.function_stack:
            self.function_stack.pop()

    def visit_Name(self, node: ast.Name) -> None:
        if self.current_function:
            scope_key = self._get_scope_key()
            if isinstance(node.ctx, ast.Store):
                # Variable assignment
                self.scope_variables[scope_key].add(node.id)
            elif isinstance(node.ctx, ast.Load):
                # Variable use
                self.scope_uses[scope_key].add(node.id)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        # Handle assignments
        if self.current_function:
            scope_key = self._get_scope_key()
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.scope_variables[scope_key].add(target.id)
                elif isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            self.scope_variables[scope_key].add(elt.id)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # Handle annotated assignments
        if self.current_function and isinstance(node.target, ast.Name):
            scope_key = self._get_scope_key()
            var_name = node.target.id
            self.scope_variables[scope_key].add(var_name)

            # Store type annotation for potential closure variable
            var_key = f"{scope_key}.{var_name}"
            self.closure_var_types[var_key] = node.annotation
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        # Handle for loop variables
        if self.current_function:
            scope_key = self._get_scope_key()
            if isinstance(node.target, ast.Name):
                self.scope_variables[scope_key].add(node.target.id)
            elif isinstance(node.target, ast.Tuple):
                for elt in node.target.elts:
                    if isinstance(elt, ast.Name):
                        self.scope_variables[scope_key].add(elt.id)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        # Handle exception variables
        if self.current_function and node.name:
            scope_key = self._get_scope_key()
            self.scope_variables[scope_key].add(node.name)
        self.generic_visit(node)

    def _has_required_decorator(self, node: ast.FunctionDef) -> bool:
        """Check if function has required decorator."""
        for decorator in node.decorator_list:
            decorator_name = self._get_decorator_name(decorator)
            if decorator_name in ('lib.script.indicator', 'lib.script.strategy', 'script.indicator', 'script.strategy'):
                return True
        return False

    def _get_decorator_name(self, decorator: Any) -> Optional[str]:
        """Get the full name of a decorator."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            parts = []
            current = decorator
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return '.'.join(reversed(parts))
        elif isinstance(decorator, ast.Call):
            return self._get_decorator_name(decorator.func)
        return None

    def _get_scope_key(self) -> str:
        """Get unique key for current scope."""
        return '.'.join(self.function_stack)
