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

A METHOD takes its closure variables right AFTER its receiver instead
(``def m(this, length, x)``, ``method_call(m, obj, length, x)``): the receiver
must stay the first argument, because ``method_call`` selects a builtin method
by the receiver's type before it calls the function, and the type pass reads
the receiver from the same position. A builtin selected that way takes none of
the captures: the method publishes their names as ``__pyne_closure__`` (an
assignment right after its definition) for the runtime to drop them, and every
``method_call`` node passing them carries the same names as
``pyne_closure_args`` for the type pass. A ``method_call`` whose receiver is
unpacked has no position to follow, so it hands the captures over as
``__pyne_captures__=(length, x)``, which ``method_call`` passes to the method
alone, after the receiver.
"""

import ast
from typing import Set, Dict, List, Optional, cast, Any

from . import ast_walk


def _is_persistent_annotation(annotation: ast.AST) -> bool:
    """Check if annotation is Persistent[T] or just Persistent."""
    if isinstance(annotation, ast.Name):
        return annotation.id == 'Persistent'
    elif isinstance(annotation, ast.Subscript):
        if isinstance(annotation.value, ast.Name):
            return annotation.value.id == 'Persistent'
    return False


#: The decorators that make an inner function a Pine method
_METHOD_DECORATORS = ('method', 'pine_method.method')

#: The decorators that make a ``main`` the script entry this pass works in
_MAIN_DECORATORS = ('lib.script.indicator', 'lib.script.strategy',
                    'script.indicator', 'script.strategy')


def _get_decorator_name(decorator: Any) -> Optional[str]:
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
        return _get_decorator_name(decorator.func)
    return None


def _has_nested_function(func: ast.FunctionDef) -> bool:
    """Whether a ``def`` stands anywhere inside a function."""
    return any(isinstance(stmt, ast.FunctionDef) and stmt is not func
               for stmt in ast_walk.walk_statements(func))


class ClosureArgumentsTransformer(ast_walk.NodeTransformer):
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
        # ``scope.var`` keys declared Persistent (a PersistentSeries is both)
        self.persistent_vars: Set[str] = set()
        # Inner functions called as methods: a ``@method`` definition, or a
        # function handed to ``method_call``. Their closure variables follow
        # the receiver instead of preceding it
        self.method_functions: Set[str] = set()

    def visit_Module(self, node: ast.Module) -> ast.Module:
        # Only a function defined inside the decorated main can have closure
        # variables threaded: without one, neither pass would change anything
        if not any(isinstance(stmt, ast.FunctionDef) and stmt.name == 'main'
                   and any(_get_decorator_name(decorator) in _MAIN_DECORATORS
                           for decorator in stmt.decorator_list)
                   and _has_nested_function(stmt)
                   for stmt in ast_walk.walk_statements(node)):
            return node

        # First pass: collect all function definitions and their closure variables
        collector = ClosureVariableCollector()
        collector.visit(node)
        self.scope_variables = collector.scope_variables
        self.closure_vars = collector.closure_vars
        self.closure_var_types = collector.closure_var_types
        self.persistent_vars = collector.persistent_vars
        self.method_functions = set()
        for call in ast_walk.walk(node):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) \
                    and call.func.id == 'method_call' and call.args:
                handed = call.args[0]
                if isinstance(handed, ast.Name):
                    self.method_functions.add(handed.id)

        # A plain-Series free variable (a parent-scope ``s: Series`` or a
        # builtin price series anchored in main by LibrarySeriesTransformer) is
        # resolved through the scope chain — the SeriesTransformer rewrites its
        # reads to the parent's state slot, reached via the parent state param
        # captured as a Python closure. Value-passing it as an argument instead
        # would give the inner function its own per-call history buffer
        # (advancing once per call, not once per bar), so a function invoked a
        # bar-varying number of times reads the wrong history (issue #67). Drop
        # them here so neither the parameter list nor the call sites thread them.
        self._drop_series_closures()

        # Second pass: transform the functions
        return cast(ast.Module, self.generic_visit(node))

    def _drop_series_closures(self) -> None:
        """Remove plain-Series closure variables — they resolve through the
        scope chain, not through argument passing.

        A PersistentSeries is left threaded: its Series slot in the parent is
        pruned by the UnusedSeriesDetector (the only history read sits in the
        inner function), so the scope-chain path has no slot to reach and the
        value argument is what carries the per-bar history into the call."""
        for scope_key, closure_vars in self.closure_vars.items():
            parent_scope = scope_key.rsplit('.', 1)[0]
            kept = {var for var in closure_vars
                    if f"{parent_scope}.{var}" in self.persistent_vars
                    or not self._is_series_annotation(
                        self.closure_var_types.get(f"{parent_scope}.{var}"))}
            closure_vars.intersection_update(kept)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef | list[ast.stmt]:
        # Check if this is the main function with required decorators
        is_main_decorated = False
        if node.name == 'main':
            for decorator in node.decorator_list:
                if _get_decorator_name(decorator) in _MAIN_DECORATORS:
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

        # The ``__pyne_closure__`` publication of a method's captures
        marker: ast.stmt | None = None

        # Process inner functions if we're in main
        if self.in_main_function and old_current == 'main' and node.name != 'main':
            # This is an inner function in main
            func_key = self._get_function_key(node.name)

            if any(_get_decorator_name(decorator) in _METHOD_DECORATORS
                   for decorator in node.decorator_list):
                self.method_functions.add(node.name)

            # Check if function has closure variables
            if func_key in self.closure_vars and self.closure_vars[func_key]:
                # Add closure variables as parameters at the beginning -- after
                # the receiver for a method
                closure_vars = sorted(self.closure_vars[func_key])
                new_args = []
                for var in closure_vars:
                    # Get type annotation for this closure variable
                    parent_scope = 'main'  # closure vars come from main scope
                    var_key = f"{parent_scope}.{var}"
                    annotation = self.closure_var_types.get(var_key, None)

                    # If annotation is Persistent[T], extract the inner type T
                    if annotation and _is_persistent_annotation(annotation):
                        annotation = self._extract_inner_type(annotation)

                    # Add closure vars at the beginning with processed annotation
                    new_args.append(ast.arg(arg=var, annotation=cast(ast.expr, annotation)))
                # Add original args after closure vars
                if node.name in self.method_functions and node.args.args:
                    node.args.args = node.args.args[:1] + new_args + node.args.args[1:]
                    marker = ast.copy_location(ast.Assign(
                        targets=[ast.Attribute(value=ast.Name(id=node.name, ctx=ast.Load()),
                                               attr='__pyne_closure__', ctx=ast.Store())],
                        value=ast.Tuple(elts=[ast.Constant(value=var) for var in closure_vars],
                                        ctx=ast.Load())), node)
                    ast.fix_missing_locations(marker)
                else:
                    new_args.extend(node.args.args)
                    node.args.args = new_args

            # Store the function definition
            self.inner_functions[node.name] = node

        # Visit the function body
        node = cast(ast.FunctionDef, self.generic_visit(node))

        # Restore state
        self.in_main_function = old_in_main
        self.current_function = old_current
        if self.function_stack:
            self.function_stack.pop()

        return node if marker is None else [node, marker]

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
                    closure_vars = sorted(self.closure_vars[func_key])
                    if func_name in self.method_functions:
                        # After the receiver, where the method declares them
                        self._thread_after_receiver(node, closure_vars)
                    else:
                        # Add closure variables as arguments at the beginning
                        new_args = []
                        for var in closure_vars:
                            new_args.append(ast.Name(id=var, ctx=ast.Load()))
                        # Add original args after closure vars
                        new_args.extend(node.args)
                        node.args = new_args

            # Handle method_call() calls - these need special handling
            elif func_name == 'method_call' and len(node.args) >= 2:
                method_name = None

                # Only the reference form ``method_call(method_function, this, ...)``
                # names a function of this module. The string form asks the runtime
                # to resolve the name against the receiver's type and the imported
                # libraries, and that lookup only ever yields an ``Exported`` — never
                # an inner function. Injecting closure arguments there would push a
                # local function's captures in front of a LIBRARY method's receiver
                # whenever the two happen to share a name (`zigZag.update()` next to
                # a local `update()`), so the string form is left untouched.
                first_arg = node.args[0]
                if isinstance(first_arg, ast.Name):
                    method_name = first_arg.id

                if method_name:
                    # Check if this method name corresponds to an inner function
                    if method_name in self.inner_functions:
                        # Get the closure variables for this function
                        func_key = self._get_function_key(method_name)
                        if func_key in self.closure_vars and self.closure_vars[func_key]:
                            # method_call(method_ref, this_obj, closure_vars..., original_args...)
                            # — the converted method declares its closure
                            # parameters right after the receiver
                            closure_vars = sorted(self.closure_vars[func_key])
                            if isinstance(node.args[1], ast.Starred):
                                node.keywords.append(ast.keyword(
                                    arg='__pyne_captures__',
                                    value=ast.Tuple(elts=[ast.Name(id=var, ctx=ast.Load())
                                                          for var in closure_vars],
                                                    ctx=ast.Load())))
                            else:
                                node.args[2:2] = [ast.Name(id=var, ctx=ast.Load())
                                                  for var in closure_vars]
                                setattr(node, 'pyne_closure_args', tuple(closure_vars))

        return node

    @staticmethod
    def _thread_after_receiver(node: ast.Call, closure_vars: List[str]) -> None:
        """Pass a method's closure variables right after its receiver argument
        in a direct call of the method.

        A receiver passed by keyword (or an unpacked one) leaves no position to
        follow, so the closure variables go by keyword there: the parameters
        carry their names.

        :param node: The call node
        :param closure_vars: The method's closure variables, in parameter order
        """
        if node.args and not isinstance(node.args[0], ast.Starred):
            node.args[1:1] = [ast.Name(id=var, ctx=ast.Load()) for var in closure_vars]
        else:
            node.keywords.extend(ast.keyword(arg=var, value=ast.Name(id=var, ctx=ast.Load()))
                                 for var in closure_vars)

    @staticmethod
    def _get_function_key(func_name: str) -> str:
        """Get unique key for a function based on its scope."""
        return 'main.' + func_name

    @staticmethod
    def _is_series_annotation(annotation: ast.AST | None) -> bool:
        """Check if annotation is Series or Series[T]."""
        if isinstance(annotation, ast.Subscript):
            return (isinstance(annotation.value, ast.Name)
                    and annotation.value.id == 'Series')
        return isinstance(annotation, ast.Name) and annotation.id == 'Series'

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


class ClosureVariableCollector(ast_walk.NodeVisitor):
    """Collect closure variables for inner functions."""

    def __init__(self):
        # Current function being analyzed
        self.current_function: Optional[str] = None
        # Stack of function scopes
        self.function_stack: List[str] = []
        # The stack joined: the key of the current scope
        self._scope_key = ''
        # Variables defined in each scope
        self.scope_variables: Dict[str, Set[str]] = {}
        # Variables used in each scope
        self.scope_uses: Dict[str, Set[str]] = {}
        # Closure variables for each inner function
        self.closure_vars: Dict[str, Set[str]] = {}
        # Type annotations for closure variables
        self.closure_var_types: Dict[str, ast.AST] = {}
        # ``scope.var`` keys declared with a Persistent annotation. A
        # PersistentSeries splits into a Persistent + a Series declaration (see
        # PersistentSeriesTransformer), so such a var carries BOTH annotations;
        # this set distinguishes it from a plain Series.
        self.persistent_vars: Set[str] = set()
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
            self._scope_key = '.'.join(self.function_stack)
            scope_key = self._get_scope_key()
            self.scope_variables[scope_key] = set()
            self.scope_uses[scope_key] = set()

            # Add function parameters to scope variables. An annotated
            # parameter carries the same type information as an annotated
            # assignment does — a ``Series[T]`` input of ``main`` is a series
            # exactly like a ``s: Series[T]`` declaration in its body — so it
            # is recorded the same way, or ``_drop_series_closures`` cannot see
            # it and the inner function gets the series value-passed.
            for arg in node.args.args:
                self.scope_variables[scope_key].add(arg.arg)
                if arg.annotation is None:
                    continue
                arg_key = f"{scope_key}.{arg.arg}"
                self.closure_var_types[arg_key] = arg.annotation
                if _is_persistent_annotation(arg.annotation):
                    self.persistent_vars.add(arg_key)

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
            self._scope_key = '.'.join(self.function_stack)

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
            if _is_persistent_annotation(node.annotation):
                self.persistent_vars.add(var_key)
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

    @staticmethod
    def _has_required_decorator(node: ast.FunctionDef) -> bool:
        """Check if function has required decorator."""
        for decorator in node.decorator_list:
            if _get_decorator_name(decorator) in _MAIN_DECORATORS:
                return True
        return False

    def _get_scope_key(self) -> str:
        """Get unique key for current scope."""
        return self._scope_key
