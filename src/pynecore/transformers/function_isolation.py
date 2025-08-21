from typing import cast
import ast
import builtins
import types
import hashlib
from pathlib import Path

from ..utils.stdlib_checker import stdlib_checker

# Functions that should not be transformed because they:
# - don't return anything (plotting, display)
# - can't have Series values
# - are purely for output/display purposes
# This makes code run little bit faster
NON_TRANSFORMABLE_FUNCTIONS = {
    # Plot and display related
    'lib.plot', 'lib.plotchar', 'lib.plotshape', 'lib.plotarrow',
    'lib.label', 'lib.table', 'lib.box', 'lib.line', 'lib.hline',
    'lib.fill', 'lib.bgcolor', 'lib.barcolor', 'lib.plotcandle',
    'lib.alert', 'lib.alertcondition', 'lib.na',

    # Other builtin functions
    'lib.timestamp', 'lib.dayofmonth', 'lib.dayofweek', 'lib.hour', 'lib.minute', 'lib.month', 'lib.second',
    'lib.weekofyear', 'lib.year', 'lib.time', 'lib.is_na', 'lib.nz', 'lib.timestamp',

    # Strategy orders
    'lib.strategy.entry', 'lib.strategy.order', 'lib.strategy.exit', 'lib.strategy.close',
    'lib.strategy.cancel', 'lib.strategy.cancel_all',

    # Input functions
    'lib.input', 'lib.input.int', 'lib.input.float', 'lib.input.bool', 'lib.input.string',
    'lib.input.source', 'lib.input.color',

    # Timeframe functions
    'lib.timeframe.in_seconds', 'lib.timeframe.from_seconds',

    # Logging
    'lib.log.info', 'lib.log.error', 'lib.log.warning',

    # Math functions
    'lib.math.abs', 'lib.math.acos', 'lib.math.asin', 'lib.math.atan', 'lib.math.avg', 'lib.math.ceil', 'lib.math.cos',
    'lib.math.exp', 'lib.math.floor', 'lib.math.log', 'lib.math.log10', 'lib.math.max', 'lib.math.min', 'lib.math.pow',
    'lib.math.round', 'lib.math.round_to_mintick', 'lib.math.sign', 'lib.math.sin', 'lib.math.sqrt',
    'lib.math.tan', 'lib.math.todegrees', 'lib.math.toradians',

    # String functions
    'lib.string.contains', 'lib.string.endswith', 'lib.string.format', 'lib.string.format_time', 'lib.string.length',
    'lib.string.lower', 'lib.string.match', 'lib.string.pos', 'lib.string.repeat', 'lib.string.replace',
    'lib.string.replace_all', 'lib.string.split', 'lib.string.startswith', 'lib.string.substring',
    'lib.string.tonumber', 'lib.string.tostring', 'lib.string.trim', 'lib.string.upper',

    # Array functions
    'lib.array.abs', 'lib.array.avg', 'lib.array.binary_search', 'lib.array.binary_search_leftmost',
    'lib.array.binary_search_rightmost', 'lib.array.clear', 'lib.array.concat', 'lib.array.copy',
    'lib.array.covariance', 'lib.array.every', 'lib.array.fill', 'lib.array.first', 'lib.array.from_items',
    'lib.array.get', 'lib.array.includes', 'lib.array.indexof', 'lib.array.insert', 'lib.array.join',
    'lib.array.last', 'lib.array.lastindexof', 'lib.array.max', 'lib.array.median', 'lib.array.min',
    'lib.array.mode', 'lib.array.percentrank', 'lib.array.percentile_linear_interpolation',
    'percentile_nearest_rank', 'percentile_nearest_rank', 'lib.array.pop', 'lib.array.push', 'lib.array.range',
    'lib.array.remove', 'lib.array.reverse', 'lib.array.set', 'lib.array.shift', 'lib.array.size', 'lib.array.slice',
    'lib.array.some', 'lib.array.sort', 'lib.array.sort_indices', 'lib.array.standardize', 'lib.array.stdev',
    'lib.array.sum', 'lib.array.unshift', 'lib.array.variance', 'lib.array.new',
    'lib.array.new_bool', 'lib.array.new_color', 'lib.array.new_float', 'lib.array.new_int', 'lib.array.new_string',

    # Map functions
    'lib.map.clear', 'lib.map.contains', 'lib.map.copy', 'lib.map.get', 'lib.map.keys', 'lib.map.new',
    'lib.map.put', 'lib.map.put_all', 'lib.map.remove', 'lib.map.size', 'lib.map.values',

    # Color functions
    'lib.color.new', 'lib.color.r', 'lib.color.g', 'lib.color.b', 'lib.color.a',
    'lib.color.rgb', 'lib.color.from_gradient',

    # Strategy functions
    "lib.strategy.fixed", "lib.strategy.cash", "lib.strategy.percent_of_equity", "lib.strategy.long",
    "lib.strategy.short", 'lib.strategy.direction', "lib.strategy.cancel", "lib.strategy.cancel_all",
    "lib.strategy.close", "lib.strategy.close_all", "lib.strategy.entry", "lib.strategy.exit",
    "lib.strategy.closedtrades", "lib.strategy.opentrades",

    # Other
    'lib.max_bars_back',

    'copy', 'dataclass', 'dccopy',
    'pytest.raises',

    'method_call', 'pine_range'
}


class FunctionIsolationTransformer(ast.NodeTransformer):
    """
    Transform function calls to use the isolate_function() wrapper.
    Every function call (except builtins and non-transformable functions)
    is wrapped with isolate_function.
    Also manages scope chain through __scope_id__ variable.
    """

    def __init__(self):
        # Track if isolation was used at all
        self.has_call_usage = False
        # Counter for unique call IDs
        self._call_id_counter = 0
        # Track function context for better call IDs
        self.current_function = None
        # Track parent functions to build the full scope path
        self.parent_functions = []
        # Track context to avoid wrappers in certain places
        self.in_decorator = False
        self.in_default_arg = False
        # Track dataclass classes
        self.dataclass_classes: set[str] = set()
        # Track inner functions that shadow builtins per scope
        # Maps scope -> set of function names that shadow builtins
        self.shadowed_builtins_by_scope: dict[str, set[str]] = {}
        # Track counter variables that need to be initialized per function
        self.function_counters: dict[str, set[str]] = {}

    def _is_dataclass_constructor(self, func: ast.Name | ast.Attribute) -> bool:
        """
        Check if the function is a dataclass constructor by checking if it's a class name
        that's been marked as a dataclass
        """
        func_path = self._get_func_path(func)
        if not func_path:
            return False

        # Check if it's a known dataclass
        if func_path in self.dataclass_classes:
            return True

        # Check if it's directly the dataclass function itself
        if func_path == 'dataclass':
            return True

        return False

    def _is_stdlib_function(self, func: ast.Name | ast.Attribute) -> bool:
        """
        Check if function is from standard library or is a builtin function/method
        that cannot be wrapped (no __globals__ attribute)
        """
        # Get function path
        func_path = self._get_func_path(func)
        if not func_path:
            return False

        # Skip dataclass constructors
        if self._is_dataclass_constructor(func):
            return True

        # Also skip if it's in NON_TRANSFORMABLE_FUNCTIONS
        if func_path in NON_TRANSFORMABLE_FUNCTIONS:
            return True

        try:
            # Try to evaluate the function path to get the actual object
            obj = eval(func_path)
            # Check if it's a builtin function/method
            if isinstance(obj, (types.BuiltinFunctionType, types.BuiltinMethodType)):
                return True
        except:  # noqa
            pass

        # Handle direct builtin functions
        if '.' not in func_path:
            # Check if this is an inner function that shadows a builtin
            if func_path in vars(builtins):
                # Check in current scope if this builtin is shadowed
                if self.current_function and self.current_function in self.shadowed_builtins_by_scope:
                    if func_path in self.shadowed_builtins_by_scope[self.current_function]:
                        # This is a shadowed builtin, so we should isolate it
                        return False
                # It's a real builtin
                return True
            # Not a builtin at all
            return False

        # Get module path
        module_path = func_path.split('.')[0]
        return stdlib_checker.is_stdlib(module_path)

    @staticmethod
    def _get_func_path(func: ast.Attribute | ast.Expr) -> str | None:
        """Get the full path of a function as a string"""
        if isinstance(func, ast.Name):
            return func.id
        elif isinstance(func, ast.Attribute):
            parts = []
            current = func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return '.'.join(reversed(parts))
        return None

    def _create_call_id(self, func: ast.Expr | ast.Attribute) -> str | None:
        """Create a unique call ID for a function with full scope path"""
        func_path = self._get_func_path(func)
        if not func_path:
            return None

        # Build parts for the full call path ID
        parts = []

        # Include all parent functions in the path
        if self.parent_functions:
            parts.extend(self.parent_functions)

        # Include current function
        if self.current_function:
            parts.append(self.current_function)

        # Add the function name
        parts.append(func_path)

        # Add counter for uniqueness
        parts.append(str(self._call_id_counter))
        self._call_id_counter += 1

        # Join all parts with unicode middle dot separator
        # This makes the ID directly usable as a Python variable name
        return '·'.join(parts)

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Process module and add isolation import and scope_id"""
        # Find all dataclass declarations
        self._find_dataclass_declarations(node)

        # Process the module normally
        node = cast(ast.Module, self.generic_visit(node))

        # Only add imports if we actually used isolation
        if not self.has_call_usage:
            return node

        # Get file path from node (it's stored in _module_file_path by the import hook)
        file_path = getattr(node, '_module_file_path', '')

        # This will be the default scope ID for the module
        scope_id = f"{hashlib.sha1(file_path.encode()).hexdigest()[:8]}_{Path(file_path).name}"

        # Add scope_id initialization and import
        scope_id_assign = ast.Assign(
            targets=[ast.Name(id='__scope_id__', ctx=ast.Store())],
            value=ast.Constant(value=scope_id)
        )
        import_stmt = ast.ImportFrom(
            module='pynecore.core.function_isolation',
            names=[ast.alias(name='isolate_function', asname=None)],
            level=0
        )

        # Find the right position to insert new nodes
        # First, check if there's a docstring
        insert_pos = 0
        if (node.body and isinstance(node.body[0], ast.Expr) and
                isinstance(cast(ast.Expr, node.body[0]).value, ast.Constant) and
                isinstance(cast(ast.Constant, cast(ast.Expr, node.body[0]).value).value, str)):
            insert_pos = 1

        # Then find the last import statement after docstring
        for i in range(insert_pos, len(node.body)):
            if isinstance(node.body[i], (ast.Import, ast.ImportFrom)):
                insert_pos = i + 1
            elif not isinstance(node.body[i], ast.Expr):  # Skip other string literals
                break

        node.body.insert(insert_pos, scope_id_assign)
        node.body.insert(insert_pos, import_stmt)

        return node

    def _find_dataclass_declarations(self, node: ast.AST) -> None:
        """
        Recursively search through the entire AST to find all classes decorated with @dataclass
        at any level - module, function, or class scope
        """
        for n in ast.walk(node):
            if not isinstance(n, ast.ClassDef):
                continue

            for decorator in n.decorator_list:
                # Check for both 'dataclass' and 'dataclass(...)' forms
                if (isinstance(decorator, ast.Name) and decorator.id == 'dataclass') or \
                        (isinstance(decorator, ast.Call) and isinstance(decorator.func,
                                                                        ast.Name) and decorator.func.id == 'dataclass'):
                    # Add class name to dataclass classes set
                    self.dataclass_classes.add(n.name)
                    break

    def _has_isolate_function_calls(self, node: ast.FunctionDef) -> bool:
        """Check if function contains any isolate_function calls"""
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            if not isinstance(child.func, (ast.Name, ast.Attribute)):
                continue

            # Skip if we're in decorator or default arg context
            if self.in_decorator or self.in_default_arg:
                continue

            # Skip stdlib and non-transformable functions
            if self._is_stdlib_function(child.func):
                continue

            func_path = self._get_func_path(child.func)
            if func_path in NON_TRANSFORMABLE_FUNCTIONS:
                continue

            # Found a potential call that would be transformed
            return True
        return False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Handle function scope and visit decorators with context"""
        # Store old state
        old_function = self.current_function

        # Reset counter for each function to ensure synchronization with collector
        self._call_id_counter = 0

        # Update function hierarchy - add current function to parent list if not None
        if self.current_function:
            self.parent_functions.append(self.current_function)

        # Set current function to this function's name
        self.current_function = node.name

        # Initialize counter collection for this function
        func_key = '.'.join(self.parent_functions + [node.name]) if self.parent_functions else node.name
        self.function_counters[func_key] = set()

        # Check if this is an inner function that shadows a builtin
        # Inner functions should be isolated if they shadow builtins
        if self.parent_functions and node.name in vars(builtins):
            # Track this function as shadowing a builtin in its parent scope
            parent_scope = self.parent_functions[-1] if self.parent_functions else None
            if parent_scope:
                if parent_scope not in self.shadowed_builtins_by_scope:
                    self.shadowed_builtins_by_scope[parent_scope] = set()
                self.shadowed_builtins_by_scope[parent_scope].add(node.name)

        # Process decorators in decorator context
        old_decorator = self.in_decorator
        self.in_decorator = True
        if node.decorator_list:
            node.decorator_list = [self.visit(cast(ast.AST, d)) for d in node.decorator_list]
        self.in_decorator = old_decorator

        # Process arguments in default arg context
        old_default = self.in_default_arg
        self.in_default_arg = True
        if node.args.defaults:
            node.args.defaults = [self.visit(cast(ast.AST, d)) for d in node.args.defaults]
        self.in_default_arg = old_default

        # Reset context flags for function body processing
        self.in_decorator = False
        self.in_default_arg = False

        # Check if function needs isolation before processing body
        needs_isolation = self._has_isolate_function_calls(node)

        # Process the body
        if node.body:
            node.body = [self.visit(cast(ast.AST, stmt)) for stmt in node.body]

        # Add global scope_id declaration and counter initializations only if function uses isolation
        if needs_isolation:
            # Find the right position after docstring if exists
            insert_pos = 0

            # Check for docstring
            if (node.body and isinstance(node.body[0], ast.Expr) and
                    isinstance(cast(ast.Expr, node.body[0]).value, ast.Constant) and
                    isinstance(cast(ast.Constant, cast(ast.Expr, node.body[0]).value).value, str)):
                insert_pos = 1

            # Add global scope_id declaration
            global_stmt = ast.Global(names=['__scope_id__'])
            node.body.insert(insert_pos, global_stmt)

            # Add counter variable initializations
            func_key = '.'.join(self.parent_functions + [node.name]) if self.parent_functions else node.name
            if func_key in self.function_counters:
                for counter_name in sorted(self.function_counters[func_key]):
                    counter_init = ast.Assign(
                        targets=[ast.Name(id=counter_name, ctx=ast.Store())],
                        value=ast.Constant(value=0)
                    )
                    node.body.insert(insert_pos + 1, counter_init)

            self.has_call_usage = True

        # Restore function context
        if self.parent_functions:
            self.parent_functions.pop()
        self.current_function = old_function

        return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        """Transform function calls to use isolate_function wrapper"""
        # First visit all arguments and keywords to handle nested calls
        node.args = [self.visit(cast(ast.AST, arg)) for arg in node.args]
        node.keywords = [self.visit(cast(ast.AST, kw)) for kw in node.keywords]

        # Skip if already processed or not a relevant call
        if (getattr(node, '_call_processed', False) or
                not isinstance(node.func, (ast.Name, ast.Attribute))):
            return node

        # Skip if starts with underscore
        if isinstance(node.func, ast.Name) and node.func.id.startswith('_'):
            return node

        # Skip if we're in decorator, default arg context or module level
        if self.in_decorator or self.in_default_arg or self.current_function is None:
            return node

        # Skip stdlib and non-transformable functions
        if self._is_stdlib_function(node.func):
            return node

        # Create call ID based on full scope path and function name
        call_id = self._create_call_id(node.func)
        if not call_id:
            return node

        # Check if this call has closure arguments (marked by ClosureArgumentsTransformer)
        closure_vars_count = getattr(node, '_closure_vars_count', -1)

        # Use the exact same call_id as the variable name (it's already a valid Python identifier)
        counter_var_name = f"__call_counter·{call_id}__"

        # Track this counter for initialization
        func_key = ('.'.join(self.parent_functions + [self.current_function])
                    if self.parent_functions else self.current_function)
        if func_key in self.function_counters:
            self.function_counters[func_key].add(counter_var_name)

        # Create a walrus expression to increment the counter
        # counter_var := counter_var + 1
        counter_increment = ast.NamedExpr(
            target=ast.Name(id=counter_var_name, ctx=ast.Store()),
            value=ast.BinOp(
                left=ast.Name(id=counter_var_name, ctx=ast.Load()),
                op=ast.Add(),
                right=ast.Constant(value=1)
            )
        )

        # Create wrapper with scope_id, closure_argument_count, and the incremented counter value
        wrapped = ast.Call(
            func=ast.Call(
                func=ast.Name(id='isolate_function', ctx=ast.Load()),
                args=[
                    node.func,
                    ast.Constant(value=call_id),
                    ast.Name(id='__scope_id__', ctx=ast.Load()),
                    ast.Constant(value=closure_vars_count),  # closure_argument_count (4th param)
                    counter_increment  # Pass the incremented counter value (5th param)
                ],
                keywords=[]
            ),
            args=node.args,
            keywords=node.keywords
        )
        setattr(wrapped, '_call_processed', True)
        return wrapped
