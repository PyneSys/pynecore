import ast
from typing import Dict, Set, cast


class LibrarySeriesTransformer(ast.NodeTransformer):
    """
    AST transformer that prepares library Series variables for the SeriesTransformer.
    When a library variable is used with indexing, it creates a local Series variable
    declaration that SeriesTransformer can then process.
    """

    def __init__(self):
        self.lib_series_vars: Dict[
            str, Dict[str, tuple[str, ast.AST]]] = {}  # module -> {attr -> (local_name, type_annotation)}
        self.used_series: Set[tuple[str, str, str]] = set()  # (module, attr, function) tuples that are used as Series
        self.declarations_to_insert: Dict[str, list[ast.AnnAssign]] = {}  # function_name -> declarations
        self.current_function: str | None = None
        self.module_level_declarations: list[ast.AnnAssign] = []  # Declarations for module level (main function)

    @staticmethod
    def _make_series_name(attr: str) -> str:
        """Generate unique name for the local Series variable"""
        # Use unicode middle dot to avoid name collisions between module hierarchies
        # Example: mylib.bar vs mylib_bar would both become __lib·mylib·bar vs __lib·mylib_bar
        attr = attr.replace('.', '·')
        return f'__lib·{attr}'

    @staticmethod
    def _create_attribute_chain(chain: list[str]) -> ast.expr:
        """Create an attribute chain from a list of names"""
        result: ast.expr = cast(ast.expr, ast.Name(id=chain[0], ctx=ast.Load()))
        for name in chain[1:]:
            result = cast(ast.expr, ast.Attribute(
                value=result,
                attr=name,
                ctx=ast.Load()
            ))
        return result

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Process module and insert module-level Series declarations"""
        # Process the module first
        node = cast(ast.Module, self.generic_visit(node))
        
        # Insert module-level declarations at the beginning of the module
        if self.module_level_declarations:
            # Find the position after imports and before the main function
            insert_pos = 0
            for i, stmt in enumerate(node.body):
                # Skip imports, from imports, and __all__ assignments
                if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                    insert_pos = i + 1
                elif (isinstance(stmt, ast.Assign) and 
                      len(stmt.targets) == 1 and 
                      isinstance(stmt.targets[0], ast.Name) and 
                      cast(ast.Name, stmt.targets[0]).id == '__all__'):
                    insert_pos = i + 1
                else:
                    break
            
            # Insert the declarations
            node.body[insert_pos:insert_pos] = self.module_level_declarations
        
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Track current function and insert Series declarations"""
        old_function = self.current_function
        self.current_function = node.name

        # Process function body first
        new_node = cast(ast.FunctionDef, self.generic_visit(node))

        # Insert declarations for used Series at the start of the function
        if self.current_function in self.declarations_to_insert:
            new_node.body = self.declarations_to_insert[self.current_function] + new_node.body

        self.current_function = old_function
        return new_node

    def process_series_usage(self, module: str, attr_chain: list[str], type_annotation: ast.AST | None = None) -> str:
        """
        Process a Series usage from a library, creating declaration if needed.
        Returns the local variable name to use.
        """
        if module not in self.lib_series_vars:
            self.lib_series_vars[module] = {}

        # Create unicode middle dot version for the variable name to avoid collisions
        attr_key = '·'.join(attr_chain)
        if attr_key not in self.lib_series_vars[module]:
            local_name = self._make_series_name(attr_key)
            self.lib_series_vars[module][attr_key] = (local_name, type_annotation)

        local_name = self.lib_series_vars[module][attr_key][0]

        # If this Series hasn't been used in current function or module level yet
        function_key = self.current_function or "__module__"
        if (module, attr_key, function_key) not in self.used_series:
            self.used_series.add((module, attr_key, function_key))

            # Create Series declaration with proper attribute chain
            decl = ast.AnnAssign(
                target=ast.Name(id=local_name, ctx=ast.Store()),
                annotation=cast(ast.expr, type_annotation or ast.Name(id='Series', ctx=ast.Load())),
                value=self._create_attribute_chain([module] + attr_chain),
                simple=1
            )

            # Store declaration to be inserted
            if self.current_function:
                # Inside a function
                if self.current_function not in self.declarations_to_insert:
                    self.declarations_to_insert[self.current_function] = []
                self.declarations_to_insert[self.current_function].append(decl)
            else:
                # At module level (main function)
                self.module_level_declarations.append(decl)

        return local_name

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        """Convert library Series access when used with indexing"""

        # Get the full attribute chain
        def get_attribute_chain(_node):
            if isinstance(_node, ast.Name):
                return [_node.id]
            elif isinstance(_node, ast.Attribute):
                return get_attribute_chain(_node.value) + [_node.attr]
            return []

        if isinstance(node.value, ast.Attribute):
            attr_chain = get_attribute_chain(node.value)
            if attr_chain and attr_chain[0] == 'lib':
                # Skip extra_fields since it's a dictionary, not a Series
                # This handles both lib.extra_fields["key"] and lib.extra_fields["key"][0] cases
                if len(attr_chain) >= 2 and attr_chain[1] == 'extra_fields':
                    return self.generic_visit(node)
                # Use the complete chain after 'lib'
                local_name = self.process_series_usage('lib', attr_chain[1:])
                # Replace lib.xxx.yyy[idx] with local_name[idx]
                node.value = cast(ast.expr, ast.Name(id=local_name, ctx=ast.Load()))

        return self.generic_visit(node)
