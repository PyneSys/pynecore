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
alone, after the receiver. A DIRECT call of such a method whose receiver is unpacked
(``m(*args)``) is rejected: the receiver has to be passed explicitly.

A call inside another inner function passes a capture by the CALLER's name for
it, so a caller binding the same name itself (a parameter or a local) would hand
its own value over instead of ``main``'s. Such a caller takes ``main``'s value as
a capture of its own, under ``__capture·<name>__``, and passes that on::

    def reader(g):                     # was: return g
        return g

    def f(__capture·g__, g):           # was: def f(g): return reader()
        return reader(__capture·g__)

    f(g, 7.0)
"""

import ast
from typing import Set, Dict, List, Optional, cast, Any

from . import ast_walk
from .emission_settings import PYNE_RESERVED_NAME_CHAR
from .locations import anchor_at, fix_locations


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


def _capture_param(var: str) -> str:
    """The parameter that carries ``main``'s ``var`` into a function that binds
    ``var`` itself."""
    return f'__capture{PYNE_RESERVED_NAME_CHAR}{var}__'


def _called_function(call: ast.Call) -> Optional[str]:
    """The name of the function a call reaches by name: its callee, or the method
    handed to ``method_call`` by reference."""
    if not isinstance(call.func, ast.Name):
        return None
    if call.func.id == 'method_call' and call.args:
        handed = call.args[0]
        return handed.id if isinstance(handed, ast.Name) else None
    return call.func.id


def _thread_captures(closure_vars: Dict[str, Set[str]],
                     scope_variables: Dict[str, Set[str]],
                     scope_calls: Dict[str, Set[str]]) -> Dict[str, Dict[str, str]]:
    """The captures of every inner function of ``main``: ``main``'s variable ->
    the parameter that carries it.

    A function takes its free variables under their own names. A function that
    calls one taking ``var`` while binding ``var`` itself cannot pass ``main``'s
    value by that name, so it takes ``var`` as a capture of its own under
    ``__capture·var__``; that in turn reaches every function calling it.

    :param closure_vars: Scope key -> the free variables it reads from ``main``
    :param scope_variables: Scope key -> the names it binds
    :param scope_calls: Scope key -> the names of the functions it calls
    :return: Scope key -> captures, sorted by ``main``'s variable name
    """
    captures = {key: {var: var for var in vars_} for key, vars_ in closure_vars.items() if vars_}
    changed = True
    while changed:
        changed = False
        for caller, callees in scope_calls.items():
            if caller.count('.') != 1 or not caller.startswith('main.'):
                continue
            bound = scope_variables.get(caller, set())
            own = captures.setdefault(caller, {})
            for callee in callees:
                for var in captures.get('main.' + callee, {}):
                    if var in bound and var not in own:
                        own[var] = _capture_param(var)
                        changed = True
    return {key: dict(sorted(own.items())) for key, own in captures.items() if own}


def _parameters(arguments: ast.arguments) -> List[ast.arg]:
    """Every parameter of a function, of every kind."""
    params = arguments.posonlyargs + arguments.args + arguments.kwonlyargs
    return params + [arg for arg in (arguments.vararg, arguments.kwarg) if arg is not None]


def _insert_positional(arguments: ast.arguments, index: int, new_args: List[ast.arg]) -> None:
    """Insert parameters at a position of the positional parameters: among the
    positional-only ones when the position falls before their end.

    :param arguments: The parameters of the function, changed in place
    :param index: The position among all positional parameters
    :param new_args: The parameters to insert
    """
    if index < len(arguments.posonlyargs):
        arguments.posonlyargs[index:index] = new_args
    else:
        index -= len(arguments.posonlyargs)
        arguments.args[index:index] = new_args


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
        # The inner function of ``main`` the visit is in
        self.caller: Optional[str] = None
        # Variables defined in each scope
        self.scope_variables: Dict[str, Set[str]] = {}
        # The inner functions of ``main``, by name
        self.inner_functions: Set[str] = set()
        # Each inner function's captures: ``main``'s variable -> its parameter
        self.captures: Dict[str, Dict[str, str]] = {}
        # Track type annotations for closure variables
        self.closure_var_types: Dict[str, ast.AST] = {}
        # ``scope.var`` keys declared Persistent (a PersistentSeries is both)
        self.persistent_vars: Set[str] = set()
        # Inner functions called as methods: a ``@method`` definition, or a
        # function handed to ``method_call``. Their closure variables follow
        # the receiver instead of preceding it
        self.method_functions: Set[str] = set()
        # The script path, for located errors
        self._module_file = '<script>'

    def visit_Module(self, node: ast.Module) -> ast.Module:
        # Only a function defined inside the decorated main can have closure
        # variables threaded: without one, neither pass would change anything
        if not any(isinstance(stmt, ast.FunctionDef) and stmt.name == 'main'
                   and any(_get_decorator_name(decorator) in _MAIN_DECORATORS
                           for decorator in stmt.decorator_list)
                   and _has_nested_function(stmt)
                   for stmt in ast_walk.walk_statements(node)):
            return node

        self._module_file = getattr(node, '_module_file_path', '<script>')

        # First pass: collect all function definitions and their closure variables
        collector = ClosureVariableCollector()
        collector.visit(node)
        self.scope_variables = collector.scope_variables
        self.closure_var_types = collector.closure_var_types
        self.persistent_vars = collector.persistent_vars
        self.inner_functions = collector.inner_functions
        self.method_functions = set(collector.method_definitions)
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
        self._drop_series_closures(collector.closure_vars)
        self.captures = _thread_captures(collector.closure_vars, collector.scope_variables,
                                         collector.scope_calls)

        # Second pass: transform the functions
        return cast(ast.Module, self.generic_visit(node))

    def _drop_series_closures(self, closure_vars_of: Dict[str, Set[str]]) -> None:
        """Remove plain-Series closure variables — they resolve through the
        scope chain, not through argument passing.

        A PersistentSeries is left threaded: its Series slot in the parent is
        pruned by the UnusedSeriesDetector (the only history read sits in the
        inner function), so the scope-chain path has no slot to reach and the
        value argument is what carries the per-bar history into the call.

        :param closure_vars_of: Scope key -> its closure variables, changed in place
        """
        for scope_key, closure_vars in closure_vars_of.items():
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

        # The ``__pyne_closure__`` publication of a method's captures
        marker: ast.stmt | None = None

        # Process inner functions if we're in main
        old_caller = self.caller
        is_inner = self.in_main_function and old_current == 'main' and node.name != 'main'
        if is_inner:
            # This is an inner function in main
            captures = self.captures.get(self._get_function_key(node.name))

            # Check if function has closure variables
            if captures:
                # Add closure variables as parameters at the beginning -- after
                # the receiver for a method
                params = list(captures.values())
                new_args = []
                for var, param in captures.items():
                    # Get type annotation for this closure variable
                    parent_scope = 'main'  # closure vars come from main scope
                    var_key = f"{parent_scope}.{var}"
                    annotation = self.closure_var_types.get(var_key, None)

                    # If annotation is Persistent[T], extract the inner type T
                    if annotation and _is_persistent_annotation(annotation):
                        annotation = self._extract_inner_type(annotation)

                    # Add closure vars at the beginning with processed annotation
                    new_args.append(ast.arg(arg=param, annotation=cast(ast.expr, annotation)))
                # Add original args after closure vars
                if node.name in self.method_functions \
                        and (node.args.posonlyargs or node.args.args):
                    _insert_positional(node.args, 1, new_args)
                    marker = anchor_at(ast.Assign(
                        targets=[ast.Attribute(value=ast.Name(id=node.name, ctx=ast.Load()),
                                               attr='__pyne_closure__', ctx=ast.Store())],
                        value=ast.Tuple(elts=[ast.Constant(value=param) for param in params],
                                        ctx=ast.Load())), node)
                    fix_locations(marker)
                else:
                    _insert_positional(node.args, 0, new_args)

        # The defaults, decorators and annotations run in the enclosing scope when
        # the ``def`` executes; only the body runs as the function itself
        body = node.body
        node.body = []
        node = cast(ast.FunctionDef, self.generic_visit(node))

        # Update state
        if is_main_decorated:
            self.in_main_function = True
        self.current_function = node.name
        self.function_stack.append(node.name)
        if is_inner:
            self.caller = node.name
        node.body = cast(ast.Module, self.generic_visit(ast.Module(body=body, type_ignores=[]))).body

        # Restore state
        self.in_main_function = old_in_main
        self.current_function = old_current
        self.caller = old_caller
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
                captures = self.captures.get(self._get_function_key(func_name))
                if captures:
                    if func_name in self.method_functions:
                        if node.args and isinstance(node.args[0], ast.Starred):
                            self._reject_unpacked_receiver(node, func_name)
                        # After the receiver, where the method declares them
                        self._thread_after_receiver(node, captures, self._passed(captures))
                    else:
                        # Add closure variables as arguments at the beginning
                        node.args = self._passed(captures) + node.args

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

                # Check if this method name corresponds to an inner function
                if method_name and method_name in self.inner_functions:
                    captures = self.captures.get(self._get_function_key(method_name))
                    if captures:
                        # method_call(method_ref, this_obj, closure_vars..., original_args...)
                        # — the converted method declares its closure
                        # parameters right after the receiver
                        if isinstance(node.args[1], ast.Starred):
                            node.keywords.append(ast.keyword(
                                arg='__pyne_captures__',
                                value=ast.Tuple(elts=self._passed(captures), ctx=ast.Load())))
                        else:
                            node.args[2:2] = self._passed(captures)
                            setattr(node, 'pyne_closure_args', tuple(captures.values()))

        return node

    def _passed(self, captures: Dict[str, str]) -> List[ast.expr]:
        """The arguments that pass ``main``'s variables to a function taking them
        as captures, in its parameter order.

        A call in ``main`` reads ``main``'s own variable; a call inside an inner
        function reads it through the caller's capture parameter for it, which
        carries ``main``'s value even where the caller binds the name itself.

        :param captures: The callee's captures: ``main``'s variable -> its parameter
        :return: One ``Name`` load per capture
        """
        own = self.captures.get(self._get_function_key(self.caller), {}) \
            if self.caller is not None else {}
        return [ast.Name(id=own.get(var, var), ctx=ast.Load()) for var in captures]

    def _reject_unpacked_receiver(self, node: ast.Call, name: str) -> None:
        """Raise the located error for a direct method call whose receiver
        comes from an unpacked sequence.

        The closure variables must follow the receiver, and the position after
        the first element of an unpacked sequence is only known at run time.

        :param node: The call node
        :param name: The called method
        """
        raise SyntaxError(
            f"{name}() uses variables of main(), so its receiver cannot come from an "
            f"unpacked argument; pass the receiver explicitly: {name}(receiver, *args)",
            (self._module_file, getattr(node, 'lineno', 0),
             getattr(node, 'col_offset', 0) + 1, None))

    @staticmethod
    def _thread_after_receiver(node: ast.Call, captures: Dict[str, str],
                               passed: List[ast.expr]) -> None:
        """Pass a method's closure variables right after its receiver argument
        in a direct call of the method.

        A receiver passed by keyword leaves no position to follow, so the
        closure variables go by keyword there, under the parameter names.

        :param node: The call node
        :param captures: The method's captures: ``main``'s variable -> its parameter
        :param passed: The values passed for them, in parameter order
        """
        if node.args:
            node.args[1:1] = passed
        else:
            node.keywords.extend(ast.keyword(arg=param, value=value)
                                 for param, value in zip(captures.values(), passed))

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
        # The functions each scope calls by name
        self.scope_calls: Dict[str, Set[str]] = {}
        # The inner functions of ``main``, by name, and those declared methods
        self.inner_functions: Set[str] = set()
        self.method_definitions: Set[str] = set()
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

        if self.in_main_function and old_function == 'main' and node.name != 'main':
            self.inner_functions.add(node.name)
            if any(_get_decorator_name(decorator) in _METHOD_DECORATORS
                   for decorator in node.decorator_list):
                self.method_definitions.add(node.name)

        # The defaults, decorators and annotations belong to the enclosing scope:
        # they run there when the ``def`` executes
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.visit(node.args)
        if node.returns is not None:
            self.visit(node.returns)

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
            self.scope_calls[scope_key] = set()

            # Add function parameters to scope variables. An annotated
            # parameter carries the same type information as an annotated
            # assignment does — a ``Series[T]`` input of ``main`` is a series
            # exactly like a ``s: Series[T]`` declaration in its body — so it
            # is recorded the same way, or ``_drop_series_closures`` cannot see
            # it and the inner function gets the series value-passed.
            for arg in _parameters(node.args):
                self.scope_variables[scope_key].add(arg.arg)
                if arg.annotation is None:
                    continue
                arg_key = f"{scope_key}.{arg.arg}"
                self.closure_var_types[arg_key] = arg.annotation
                if _is_persistent_annotation(arg.annotation):
                    self.persistent_vars.add(arg_key)

        # Visit function body
        for stmt in node.body:
            self.visit(stmt)

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

    def visit_Call(self, node: ast.Call) -> None:
        callee = _called_function(node)
        if self.current_function and callee is not None:
            self.scope_calls[self._get_scope_key()].add(callee)
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
