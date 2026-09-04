import ast
from typing import Dict, Set, List, Optional, cast

NON_MODULE_ATTRS = {
    'input',  # class
    'script',  # class
}



def _dotted(path: list[str], node: ast.AST) -> ast.expr:
    """
    Build ``a.b.c`` for a path, at the position of the name it replaces.

    Every node of the chain takes the original name's location, so a
    diagnostic about the rewritten spelling still points at the source.

    :param path: The dotted path, split
    :param node: The name being rewritten
    :return: The attribute chain
    """
    result: ast.expr = ast.copy_location(ast.Name(id=path[0], ctx=ast.Load()), node)
    for part in path[1:]:
        result = ast.copy_location(ast.Attribute(value=result, attr=part, ctx=ast.Load()), node)
    return result

class ImportNormalizerTransformer(ast.NodeTransformer):
    """
    AST transformer that normalizes pynecore.lib imports.
    - Converts all lib-related imports to 'from pynecore import lib'
    - Transforms all references to use fully qualified names (lib.xxx.yyy)
    - Moves function-level imports to module level
    - Supports wildcard imports by using module's __all__
    """

    def __init__(self):
        # Track import mappings: imported_name -> full_path
        self.import_map: Dict[str, List[str]] = {}
        # Track which names to replace with lib.xxx
        self.names_to_replace: Set[str] = set()
        # Track if we need to add 'from pynecore import lib'
        self.needs_lib_import = False
        #: A lib-module alias's bindings in source order: (top-level statement
        #: index, lib path) for each lib import of the name, (index, None) for a
        #: statement rebinding the name to something else
        self.alias_bindings: dict[str, list[tuple[int, str | None]]] = {}
        self._stmt_index = 0
        # Track function level imports to move up
        self.function_imports: List[ast.ImportFrom] = []
        # Current function being processed
        self.current_function: Optional[str] = None
        # Track direct module imports: alias -> module_path
        self.module_imports: Dict[str, str] = {}
        # Track wildcard imports: module_path -> set of exposed names
        self.wildcard_imports: Dict[str, Set[str]] = {}
        # Track required submodules
        self.required_submodules: Set[str] = set()
        # Track function parameters to avoid replacing them
        self.function_parameters: Set[str] = set()

    @staticmethod
    def _is_lib_import(node: ast.ImportFrom) -> bool:
        """Check if an import is lib-related"""
        return bool(node.module and (node.module == 'pynecore.lib' or
                                     node.module.startswith('pynecore.lib.')))

    @staticmethod
    def _is_lib_module_import(node: ast.Import) -> bool:
        """Check if it's a direct pynecore.lib module import"""
        for alias in node.names:
            if alias.name == 'pynecore.lib' or alias.name.startswith('pynecore.lib.'):
                return True
        return False

    @staticmethod
    def _get_full_path(module: str, name: str) -> List[str]:
        """Convert module path to list of components"""
        if module == 'pynecore.lib':
            return ['lib', name]
        # Handle submodules: pynecore.lib.xxx -> ['lib', 'xxx', name]
        parts = module.split('.')
        return ['lib'] + parts[2:] + [name]

    @staticmethod
    def _get_module_all(module: str) -> Set[str]:
        """Get the __all__ list from a module by importing it."""
        try:
            # Import the module
            imported = __import__(module, fromlist=['__all__'])
            # Get its __all__ list
            if hasattr(imported, '__all__'):
                return set(imported.__all__)
        except (ImportError, AttributeError):
            pass
        # Return empty set if anything goes wrong
        return set()

    def _handle_wildcard_import(self, module: str) -> None:
        """Process a wildcard import by recording all names from module's __all__."""
        # Get the exposed names from the module
        exposed = self._get_module_all(module)

        if not exposed:
            # No __all__ found, this is probably an error
            raise SyntaxError(
                f"Cannot use wildcard import: module {module} has no __all__ defined"
            )

        self.wildcard_imports[module] = exposed

        # Add all exposed names to our import mapping
        path_parts = ['lib'] + module.split('.')[2:]  # Skip 'pynecore.lib'
        for name in exposed:
            self.import_map[name] = path_parts + [name]
            self.names_to_replace.add(name)

    def _handle_import_from(self, node: ast.ImportFrom) -> None:
        """Process a lib-related 'from x import y' statement"""
        if not self._is_lib_import(node):
            return

        self.needs_lib_import = True
        module = node.module or ''

        # Extract submodule if it exists
        if module.startswith('pynecore.lib.'):
            submodule = module.split('.')[2]  # Get first part after pynecore.lib
            self.required_submodules.add(submodule)

        # Handle wildcard imports
        for alias in node.names:
            if alias.name == '*':
                self._handle_wildcard_import(module)
                return

        # Handle regular imports
        for alias in node.names:
            name = alias.name
            asname = alias.asname or name
            path = self._get_full_path(module, name)

            self.import_map[asname] = path
            self.names_to_replace.add(asname)

    def _handle_import(self, node: ast.Import) -> None:
        """Process direct module imports"""
        if not self._is_lib_module_import(node):
            return

        self.needs_lib_import = True

        for alias in node.names:
            if not (alias.name == 'pynecore.lib' or
                    alias.name.startswith('pynecore.lib.')):
                continue

            parts = alias.name.split('.')
            if len(parts) <= 2:  # pynecore.lib
                asname = alias.asname or 'lib'
                self.module_imports[asname] = 'lib'
            else:  # pynecore.lib.xxx
                submodule = parts[2]  # First part after pynecore.lib
                self.required_submodules.add(submodule)
                asname = alias.asname or parts[-1]
                self.module_imports[asname] = 'lib.' + '.'.join(parts[2:])

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Handle module level transformations"""
        # First collect all imports
        has_lib_import = False
        for stmt in node.body:
            if isinstance(stmt, ast.ImportFrom):
                if stmt.module == 'pynecore' and any(n.name == 'lib' for n in stmt.names):
                    has_lib_import = True

            if isinstance(stmt, (ast.ImportFrom, ast.Import)):
                self._validate_import(stmt)
                if isinstance(stmt, ast.ImportFrom):
                    self._handle_import_from(stmt)
                else:
                    self._handle_import(stmt)

        # A lib-module alias rebound later in the module (another lib module,
        # an import of something else, a definition, an assignment) denotes
        # that binding from there on: every reference is rewritten to the lib
        # path bound where it stands, or left alone past a foreign rebinding
        for index, stmt in enumerate(node.body):
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    if alias.name == 'pynecore.lib' or alias.name.startswith('pynecore.lib.'):
                        parts = alias.name.split('.')
                        target = 'lib' if len(parts) <= 2 else 'lib.' + '.'.join(parts[2:])
                        bound = alias.asname or parts[-1]
                        self.alias_bindings.setdefault(bound, []).append((index, target))
            for name in self.module_imports:
                if _rebinds(stmt, name):
                    self.alias_bindings.setdefault(name, []).append((index, None))

        # Process the rest of the module to collect attribute usages
        for index, stmt in enumerate(node.body):
            self._stmt_index = index
            node.body[index] = cast(ast.stmt, self.visit(stmt))

        # Filter out old lib imports
        new_body = []
        for stmt in node.body:
            if isinstance(stmt, ast.ImportFrom):
                if not self._is_lib_import(stmt):
                    new_body.append(stmt)
                # Keep original lib import if exists
                elif stmt.module == 'pynecore' and any(n.name == 'lib' for n in stmt.names):
                    new_body.append(stmt)
            elif isinstance(stmt, ast.Import):
                if not self._is_lib_module_import(stmt):
                    new_body.append(stmt)
            else:
                new_body.append(stmt)

        # Add imports if needed
        imports = []

        # Add base lib import if needed and not present
        if self.needs_lib_import and not has_lib_import:
            imports.append(
                ast.ImportFrom(
                    module='pynecore',
                    names=[ast.alias(name='lib', asname=None)],
                    level=0
                )
            )

        # Add required submodule imports
        for submodule in sorted(self.required_submodules):
            imports.append(
                ast.Import(
                    names=[ast.alias(name=f'pynecore.lib.{submodule}', asname=None)]
                )
            )

        # Function level imports moved up
        if self.function_imports:
            imports.extend(self.function_imports)

        # Insert imports after the docstring and the ``__future__`` block
        insert_pos = 1 if (new_body and isinstance(new_body[0], ast.Expr) and
                           isinstance(cast(ast.Expr, new_body[0]).value, ast.Constant)) else 0
        while insert_pos < len(new_body) and isinstance(new_body[insert_pos], ast.ImportFrom) \
                and cast(ast.ImportFrom, new_body[insert_pos]).module == '__future__':
            insert_pos += 1
        new_body[insert_pos:insert_pos] = imports

        node.body = new_body
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        """Track lib.xxx usage to detect required submodules"""
        # Process children first
        node = cast(ast.Attribute, self.generic_visit(node))

        # Extract the full chain
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value

        # Only process if it starts with 'lib'
        if isinstance(current, ast.Name) and current.id == 'lib' and parts:
            self.needs_lib_import = True

            if len(parts) >= 2:  # lib.x.y pattern
                module_name = parts[-1]  # Get the first part after lib
                if module_name not in NON_MODULE_ATTRS:
                    self.required_submodules.add(module_name)

        return node

    @staticmethod
    def _validate_import(node: ast.ImportFrom | ast.Import) -> None:
        """Validate import statements for unsupported patterns."""
        if isinstance(node, ast.ImportFrom):
            if node.module == 'pynecore':
                for alias in node.names:
                    if alias.name == 'lib' and alias.asname:
                        raise SyntaxError(
                            "'lib' must be imported as itself, not as an alias")

    def visit_Name(self, node: ast.Name) -> ast.AST:
        """Transform variable references"""
        if isinstance(node.ctx, ast.Load):
            # Don't replace function parameters
            if node.id in self.function_parameters:
                return node

            # Handle regular imports
            if node.id in self.names_to_replace:
                path = self.import_map[node.id]
                return _dotted(path, node)
            # Handle module imports
            elif node.id in self.module_imports:
                target: str | None = self.module_imports[node.id]
                for index, bound_target in self.alias_bindings.get(node.id, ()):
                    if index > self._stmt_index:
                        break
                    target = bound_target
                if target is None:
                    return node
                return _dotted(target.split('.'), node)
            # Handle names from wildcard imports
            else:
                # Check each wildcard imported module
                for module, exposed in self.wildcard_imports.items():
                    if node.id in exposed:
                        path = ['lib'] + module.split('.')[2:] + [node.id]
                        return _dotted(path, node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Process function definitions and handle imports"""
        old_function = self.current_function
        old_parameters = self.function_parameters.copy()

        self.current_function = node.name

        # Parameter default values are evaluated in the ENCLOSING scope at def-execution
        # time, not inside the function body. Visit them before the parameter names enter
        # `function_parameters`, otherwise a default that references a lib module shadowed
        # by a same-named parameter (e.g. a param `position` defaulting to
        # `position.top_center`) is left unrewritten and the lib import is dropped.
        node.args.defaults = [cast(ast.expr, self.visit(d)) for d in node.args.defaults]
        node.args.kw_defaults = [
            cast(ast.expr, self.visit(d)) if d is not None else None
            for d in node.args.kw_defaults
        ]

        # Collect function parameters
        for arg in node.args.args:
            self.function_parameters.add(arg.arg)

        # Also handle keyword-only args, positional-only args, vararg, and kwarg
        for arg in node.args.posonlyargs:
            self.function_parameters.add(arg.arg)
        for arg in node.args.kwonlyargs:
            self.function_parameters.add(arg.arg)
        if node.args.vararg:
            self.function_parameters.add(node.args.vararg.arg)
        if node.args.kwarg:
            self.function_parameters.add(node.args.kwarg.arg)

        # Process function
        node = cast(ast.FunctionDef, self.generic_visit(node))

        # Reset function context
        self.current_function = old_function
        self.function_parameters = old_parameters
        return node


def _rebinds(stmt: ast.stmt, name: str) -> bool:
    """Whether a top-level statement binds ``name`` to something other than a lib module."""
    if isinstance(stmt, ast.ImportFrom):
        return any((alias.asname or alias.name) == name for alias in stmt.names)
    if isinstance(stmt, ast.Import):
        return any((alias.asname or alias.name.split('.')[0]) == name
                   and not alias.name.startswith('pynecore.lib') for alias in stmt.names)
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return stmt.name == name
    return any(isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Store)
               for node in ast.walk(stmt))
