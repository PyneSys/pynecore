from typing import cast
import ast


class PersistentTransformer(ast.NodeTransformer):
    """
    Transform Persistent type annotations and assignments to global variables
    """

    def __init__(self):
        self.persistent_vars: dict[str, dict[str, str]] = {}  # scope -> var_name -> global_name
        self.varip_vars: dict[str, dict[str, str]] = {}  # scope -> var_name -> global_name (varip only)
        self.scope_stack: list[str] = []
        self.current_scope: str = ""
        self.module_level_assigns: list[ast.Assign] = []
        self.scope_vars: dict[str, set[str]] = {}  # Track all referenced vars per scope
        self.modified_vars: dict[str, set[str]] = {}  # Track modified vars per scope
        self.initialized_flags: dict[str, set[str]] = {}  # scope -> set(init_flags)
        # Track locally declared variables in each scope
        self.local_vars: dict[str, set[str]] = {}  # scope -> set of local var names
        # Track variables defined as Persistent in each scope
        self.persistent_declarations: dict[str, set[str]] = {}  # scope -> set(persistent_var_names)

        self.all_persistent_vars: dict[tuple[str, str], str] = {}
        self.all_local_vars: dict[str, set[str]] = {}
        self.current_verifying_scope = None  # Track scope during verification

    def _get_scope_persistents(self, var_name: str) -> str | None:
        """Get persistent variable name from current scope or parent scopes"""
        # Check if variable is locally declared in current scope but NOT as Persistent
        if (self.current_scope in self.local_vars and
                var_name in self.local_vars[self.current_scope] and
                (self.current_scope not in self.persistent_declarations or
                 var_name not in self.persistent_declarations[self.current_scope])):
            return None

        # Check current scope first
        if self.current_scope in self.persistent_vars:
            if var_name in self.persistent_vars[self.current_scope]:
                global_name = self.persistent_vars[self.current_scope][var_name]
                return global_name

        # Then check parent scopes
        for i in range(len(self.scope_stack) - 1, -1, -1):
            scope = "·".join(self.scope_stack[:i + 1])  # Use middle dot as separator

            if scope in self.persistent_vars:
                if var_name in self.persistent_vars[scope]:
                    global_name = self.persistent_vars[scope][var_name]
                    return global_name

        return None

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom | None:
        """Handle imports, only remove Persistent while keeping other imports"""
        if node.module and node.module.startswith('pynecore'):
            # Filter out Persistent from names
            new_names = [name for name in node.names if name.name != 'Persistent']
            if not new_names:
                # If no names left, remove the entire import
                return None
            # Create new import with remaining names
            node.names = new_names
        return node

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Add module level assignments before first function or assignment"""
        # Process the entire module first
        node = cast(ast.Module, self.generic_visit(node))

        # Save persistent variable mappings for final verification
        # Format: {(scope, var_name): global_name}
        for scope, vars_map in self.persistent_vars.items():
            for var_name, global_name in vars_map.items():
                self.all_persistent_vars[(scope, var_name)] = global_name

        # Save all local variables per scope for lookup during verification
        for scope, local_vars in self.local_vars.items():
            self.all_local_vars[scope] = set(local_vars)

        # Final check pass: verify all calls have their persistent arguments properly transformed
        node = cast(ast.Module, self._verify_all_call_args(node))

        if not self.module_level_assigns:
            return node

        # Create function variables dictionary
        function_vars_dict: dict[str, list[str]] = {}

        # Collect all variables for each scope
        for scope, vars_dict in self.persistent_vars.items():
            # Register every scope with its variables - convert '·' to '.' in key name only
            function_name = scope.replace('·', '.') if scope else "main"
            function_vars: list[str] = []

            for var_name, global_name in vars_dict.items():
                function_vars.append(global_name)

                # Add Kahan compensation variable if it exists
                kahan_compensation = f"{global_name}_kahan_c__"
                if kahan_compensation in self.modified_vars.get(scope, set()):
                    function_vars.append(kahan_compensation)

            # Add initialization flags that actually exist
            if scope in self.initialized_flags:
                for init_flag in self.initialized_flags[scope]:
                    function_vars.append(init_flag)

            if function_vars:  # Only add if there are variables
                function_vars_dict[function_name] = function_vars

        # Create the registration dictionary if there are function variables
        if function_vars_dict:
            # noinspection PyShadowingBuiltins
            function_vars_assign = ast.Assign(
                targets=[ast.Name(id='__persistent_function_vars__', ctx=ast.Store())],
                value=ast.Dict(
                    keys=[ast.Constant(value=k) for k in function_vars_dict],
                    values=[
                        ast.Tuple(
                            elts=[ast.Constant(value=var) for var in vars],
                            ctx=ast.Load()
                        )
                        for vars in function_vars_dict.values()
                    ]
                )
            )
            self.module_level_assigns.append(function_vars_assign)

        # Create varip function variables dictionary (subset of persistent vars excluded from rollback)
        if self.varip_vars:
            varip_vars_dict: dict[str, list[str]] = {}
            for scope, vars_dict in self.varip_vars.items():
                function_name = scope.replace('·', '.') if scope else "main"
                varip_func_vars: list[str] = []
                for var_name, global_name in vars_dict.items():
                    varip_func_vars.append(global_name)
                    kahan_compensation = f"{global_name}_kahan_c__"
                    if kahan_compensation in self.modified_vars.get(scope, set()):
                        varip_func_vars.append(kahan_compensation)
                if scope in self.initialized_flags:
                    for init_flag in self.initialized_flags[scope]:
                        varip_global = init_flag.replace('_initialized__', '__')
                        if any(varip_global.startswith(g) for g in
                               [gn for gn in vars_dict.values()]):
                            varip_func_vars.append(init_flag)
                if varip_func_vars:
                    varip_vars_dict[function_name] = varip_func_vars
            if varip_vars_dict:
                varip_vars_assign = ast.Assign(
                    targets=[ast.Name(id='__varip_function_vars__', ctx=ast.Store())],
                    value=ast.Dict(
                        keys=[ast.Constant(value=k) for k in varip_vars_dict],
                        values=[
                            ast.Tuple(
                                elts=[ast.Constant(value=var) for var in var_names],
                                ctx=ast.Load()
                            )
                            for var_names in varip_vars_dict.values()
                        ]
                    )
                )
                self.module_level_assigns.append(varip_vars_assign)

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

        return ast.Module(
            body=pre_body + self.module_level_assigns + post_body,
            type_ignores=node.type_ignores
        )

    def _verify_all_call_args(self, node: ast.AST) -> ast.AST:
        """Recursively verify and fix all Call nodes in the AST"""
        # Track current scope when verifying function definitions
        if isinstance(node, ast.FunctionDef):
            old_scope = self.current_verifying_scope
            if not self.current_verifying_scope:
                self.current_verifying_scope = node.name
            else:
                self.current_verifying_scope = f"{self.current_verifying_scope}·{node.name}"

            # Process function and update scope back when done
            result = self._process_verify_node(node)
            self.current_verifying_scope = old_scope
            return result

        return self._process_verify_node(node)

    def _process_verify_node(self, node: ast.AST) -> ast.AST:
        """Process a single node during verification"""
        if isinstance(node, ast.Call):
            # Process all arguments to ensure they are transformed
            for i, arg in enumerate(node.args):
                if isinstance(arg, ast.Name):
                    var_name = arg.id
                    # Skip variables that are already transformed
                    if var_name.startswith('__persistent_'):
                        continue

                    # Check if this is a local variable in current scope
                    if (self.current_verifying_scope and
                            self.current_verifying_scope in self.all_local_vars and
                            var_name in self.all_local_vars[self.current_verifying_scope]):
                        # It's a local variable, don't transform it
                        continue

                    # Look for this variable in persistent vars, checking current scope first
                    global_name = None
                    if self.current_verifying_scope:
                        scope_key = (self.current_verifying_scope, var_name)
                        if scope_key in self.all_persistent_vars:
                            global_name = self.all_persistent_vars[scope_key]

                    # Then check parent scopes
                    if not global_name and self.current_verifying_scope:
                        # Try parent scopes
                        scope_parts = self.current_verifying_scope.split('·')
                        for _ in range(len(scope_parts) - 1, 0, -1):
                            parent_scope = '·'.join(scope_parts[:i])
                            parent_key = (parent_scope, var_name)
                            if parent_key in self.all_persistent_vars:
                                global_name = self.all_persistent_vars[parent_key]
                                break

                    # If found, transform
                    if global_name:
                        node.args[i] = ast.Name(id=global_name, ctx=ast.Load())

            # Process keyword arguments too
            for kw in node.keywords:
                if isinstance(kw.value, ast.Name):
                    var_name = kw.value.id
                    # Skip variables that are already transformed
                    if var_name.startswith('__persistent_'):
                        continue

                    # Check if this is a local variable in current scope
                    if (self.current_verifying_scope and
                            self.current_verifying_scope in self.all_local_vars and
                            var_name in self.all_local_vars[self.current_verifying_scope]):
                        # It's a local variable, don't transform it
                        continue

                    # Look for this variable in persistent vars, checking current scope first
                    global_name = None
                    if self.current_verifying_scope:
                        scope_key = (self.current_verifying_scope, var_name)
                        if scope_key in self.all_persistent_vars:
                            global_name = self.all_persistent_vars[scope_key]

                    # Then check parent scopes
                    if not global_name and self.current_verifying_scope:
                        # Try parent scopes
                        scope_parts = self.current_verifying_scope.split('·')
                        for i in range(len(scope_parts) - 1, 0, -1):
                            parent_scope = '·'.join(scope_parts[:i])
                            parent_key = (parent_scope, var_name)
                            if parent_key in self.all_persistent_vars:
                                global_name = self.all_persistent_vars[parent_key]
                                break

                    # If found, transform
                    if global_name:
                        kw.value = ast.Name(id=global_name, ctx=ast.Load())

        # Continue with recursion for all child nodes
        for field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast.AST):
                        value = self._verify_all_call_args(value)
                        new_values.append(value)
                    else:
                        new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                new_node = self._verify_all_call_args(old_value)
                setattr(node, field, new_node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Process function definitions"""
        if node.name == "main":
            # Handle main function specially
            self.scope_stack.append("main")
        else:
            self.scope_stack.append(node.name)

        self.current_scope = "·".join(self.scope_stack)  # Use middle dot as separator to avoid conflicts

        # Initialize tracking sets for this scope
        self.scope_vars.setdefault(self.current_scope, set())
        self.modified_vars.setdefault(self.current_scope, set())
        self.initialized_flags.setdefault(self.current_scope, set())
        self.local_vars.setdefault(self.current_scope, set())
        self.persistent_declarations.setdefault(self.current_scope, set())

        # Add function arguments to local variables
        for arg in node.args.args:
            self.local_vars[self.current_scope].add(arg.arg)

        # Process function body
        node = cast(ast.FunctionDef, self.generic_visit(node))

        # *** FŐ MÓDOSÍTÁS: Csak azokat a változókat adjuk hozzá a global utasításhoz,
        # amelyeket ténylegesen módosítunk (írunk) a függvényben ***
        globals_to_declare = set()

        # A módosított változók közül csak a perzisztens változókat és inicializálási flageket adjuk hozzá
        if self.current_scope in self.modified_vars:
            for global_name in self.modified_vars[self.current_scope]:
                if global_name.startswith('__persistent_'):
                    globals_to_declare.add(global_name)

                    # Az inicializálási flageket is hozzá kell adnunk, ha kapcsolódnak
                    # módosított perzisztens változóhoz
                    init_flag = f"{global_name}_initialized__"
                    if init_flag in self.initialized_flags.get(self.current_scope, set()):
                        globals_to_declare.add(init_flag)

        # Only add global declaration if there are globals to declare
        if globals_to_declare:
            insert_pos = 0
            first_stmt = node.body[0] if node.body else None
            if (isinstance(first_stmt, ast.Expr) and
                    isinstance(first_stmt.value, ast.Constant) and
                    isinstance(first_stmt.value.value, str)):
                insert_pos = 1

            node.body.insert(insert_pos, ast.Global(names=sorted(globals_to_declare)))

        # Fix nonlocal statements - remove persistent variables
        persistent_globals = set()
        for i in range(len(self.scope_stack)):
            scope = "·".join(self.scope_stack[:i + 1])
            if scope in self.persistent_vars:
                for var_name, global_name in self.persistent_vars[scope].items():
                    if global_name in self.modified_vars.get(self.current_scope, set()):
                        persistent_globals.add(var_name)

        new_body = []
        for stmt in node.body:
            if isinstance(stmt, ast.Nonlocal):
                # Filter out variables that are now globals
                new_names = [name for name in stmt.names if name not in persistent_globals]
                if new_names:
                    stmt.names = new_names
                    new_body.append(stmt)
                # Skip empty nonlocal statements
            else:
                new_body.append(stmt)
        node.body = new_body

        self.scope_stack.pop()
        self.current_scope = "·".join(self.scope_stack)  # Use middle dot as separator to avoid conflicts
        return node

    @staticmethod
    def _is_persistent_type(annotation: ast.expr) -> bool:
        """Check if the annotation is any form of Persistent or IBPersistent type"""
        if isinstance(annotation, ast.Name):
            return annotation.id in ('Persistent', 'IBPersistent', 'IBPersistentSeries')
        elif isinstance(annotation, ast.Subscript):
            if isinstance(annotation.value, ast.Name):
                return annotation.value.id in ('Persistent', 'IBPersistent', 'IBPersistentSeries')
        elif isinstance(annotation, ast.Attribute):
            return annotation.attr in ('Persistent', 'IBPersistent', 'IBPersistentSeries')
        return False

    @staticmethod
    def _is_varip_type(annotation: ast.expr) -> bool:
        """Check if the annotation is an IBPersistent type (not regular Persistent)"""
        if isinstance(annotation, ast.Name):
            return annotation.id in ('IBPersistent', 'IBPersistentSeries')
        elif isinstance(annotation, ast.Subscript):
            if isinstance(annotation.value, ast.Name):
                return annotation.value.id in ('IBPersistent', 'IBPersistentSeries')
        elif isinstance(annotation, ast.Attribute):
            return annotation.attr in ('IBPersistent', 'IBPersistentSeries')
        return False

    @staticmethod
    def _is_literal_or_na(node: ast.expr) -> bool:
        """Check if a node represents a literal value or na"""
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, ast.Name):
            return node.id == 'na'
        return False

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Handle function calls and ensure arguments get proper transformation"""
        # Visit children first (function, args, keywords) to process nested calls
        # IMPORTANT: store the result, or nested transformations might be lost
        visited_node = cast(ast.Call, self.generic_visit(node))

        # Now transform args
        for i, arg in enumerate(visited_node.args):
            if isinstance(arg, ast.Name):
                var_name = arg.id
                global_name = self._get_scope_persistents(var_name)

                if global_name:
                    # Replace argument with global name
                    visited_node.args[i] = ast.Name(id=global_name, ctx=ast.Load())

                    # Track usage
                    if self.current_scope:
                        self.scope_vars.setdefault(self.current_scope, set())
                        self.scope_vars[self.current_scope].add(global_name)

        # Also handle keyword arguments
        for kw in visited_node.keywords:
            if isinstance(kw.value, ast.Name):
                var_name = kw.value.id
                global_name = self._get_scope_persistents(var_name)

                if global_name:
                    # Replace keyword value with global name
                    kw.value = ast.Name(id=global_name, ctx=ast.Load())

                    # Track usage
                    if self.current_scope:
                        self.scope_vars.setdefault(self.current_scope, set())
                        self.scope_vars[self.current_scope].add(global_name)

        return visited_node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | None:
        """Convert any Persistent type annotated assignments"""
        if not isinstance(node.target, ast.Name):
            return node

        if self._is_persistent_type(node.annotation):
            var_name = node.target.id

            # Mark this variable as a Persistent declaration in this scope
            self.persistent_declarations.setdefault(self.current_scope, set())
            self.persistent_declarations[self.current_scope].add(var_name)

            # Add to local vars to track the variable in this scope
            self.local_vars.setdefault(self.current_scope, set())
            self.local_vars[self.current_scope].add(var_name)

            # Generate global name using current scope
            global_name = f"__persistent_{self.current_scope}·{var_name}__"

            # Initialize scope dict if needed
            if self.current_scope not in self.persistent_vars:
                self.persistent_vars[self.current_scope] = {}

            # Store the mapping
            self.persistent_vars[self.current_scope][var_name] = global_name

            # Track varip variables separately
            if self._is_varip_type(node.annotation):
                if self.current_scope not in self.varip_vars:
                    self.varip_vars[self.current_scope] = {}
                self.varip_vars[self.current_scope][var_name] = global_name

            # Track this variable in current scope
            if self.current_scope:
                if self.current_scope not in self.scope_vars:
                    self.scope_vars[self.current_scope] = set()
                self.scope_vars[self.current_scope].add(global_name)

                # *** JAVÍTÁS: Csak akkor jelöljük módosítottként, ha nem literál az érték ***
                if node.value and not self._is_literal_or_na(node.value):
                    if self.current_scope not in self.modified_vars:
                        self.modified_vars[self.current_scope] = set()
                    self.modified_vars[self.current_scope].add(global_name)

            # Handle module level assignments and initialization
            if node.value:
                if self._is_literal_or_na(node.value):
                    # For literals, just add module level assignment
                    self.module_level_assigns.append(
                        ast.Assign(
                            targets=[ast.Name(id=global_name, ctx=ast.Store())],
                            value=node.value
                        )
                    )
                else:
                    # For non-literal values:
                    # 1. Initialize with None at module level
                    self.module_level_assigns.append(
                        ast.Assign(
                            targets=[ast.Name(id=global_name, ctx=ast.Store())],
                            value=ast.Constant(value=None)
                        )
                    )
                    # 2. Add initialization flag
                    init_flag = f"{global_name}_initialized__"
                    self.module_level_assigns.append(
                        ast.Assign(
                            targets=[ast.Name(id=init_flag, ctx=ast.Store())],
                            value=ast.Constant(value=False)
                        )
                    )
                    # 3. Register the initialization flag
                    if self.current_scope:
                        if self.current_scope not in self.scope_vars:
                            self.scope_vars[self.current_scope] = set()
                        self.scope_vars[self.current_scope].add(init_flag)

                        # Mark initialization flag as modified since we'll be writing to it
                        if self.current_scope not in self.modified_vars:
                            self.modified_vars[self.current_scope] = set()
                        self.modified_vars[self.current_scope].add(init_flag)

                        if self.current_scope not in self.initialized_flags:
                            self.initialized_flags[self.current_scope] = set()
                        self.initialized_flags[self.current_scope].add(init_flag)

                    return cast(ast.AST, ast.If(
                        test=ast.UnaryOp(
                            op=ast.Not(),
                            operand=ast.Name(id=init_flag, ctx=ast.Load())
                        ),
                        body=[
                            ast.Assign(
                                targets=[ast.Name(id=global_name, ctx=ast.Store())],
                                value=self.visit(cast(ast.AST, node.value))
                            ),
                            ast.Assign(
                                targets=[ast.Name(id=init_flag, ctx=ast.Store())],
                                value=ast.Constant(value=True)
                            )
                        ],
                        orelse=[]
                    ))
            else:
                # No initial value, initialize with na
                self.module_level_assigns.append(
                    ast.Assign(
                        targets=[ast.Name(id=global_name, ctx=ast.Store())],
                        value=ast.Name(id='na', ctx=ast.Load())
                    )
                )
            return None

        # For non-Persistent annotated assignments, we still need to visit the value
        # to transform any persistent variable references
        if node.value:
            node.value = self.visit(cast(ast.AST, node.value))
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        """Convert normal assignments to persistent variables"""
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = cast(ast.Name, node.targets[0])
            var_name = target.id

            # First, check if there are any Persistent declarations in the current scope
            is_first_assignment = (var_name not in self.local_vars.get(self.current_scope, set()))

            # If this is the first assignment in this scope, mark it as a local variable
            if is_first_assignment:
                self.local_vars.setdefault(self.current_scope, set())
                self.local_vars[self.current_scope].add(var_name)

            # Now check if it's a persistent variable reference
            global_name = self._get_scope_persistents(var_name)

            if global_name:
                # Track this variable in current scope
                if self.current_scope:
                    self.scope_vars.setdefault(self.current_scope, set())
                    self.scope_vars[self.current_scope].add(global_name)

                    # Mark variable as modified because we're assigning to it
                    self.modified_vars.setdefault(self.current_scope, set())
                    self.modified_vars[self.current_scope].add(global_name)

                # Visit the value part first to transform any references in it
                transformed_value = self.visit(cast(ast.AST, node.value))

                return ast.Assign(
                    targets=[ast.Name(id=global_name, ctx=ast.Store())],
                    value=transformed_value
                )

        # If not a persistent assignment, still visit the value part
        node.value = self.visit(cast(ast.AST, node.value))
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AugAssign | ast.AST:
        """Handle augmented assignments (+=, *=, etc.) with Kahan summation for += on floats"""
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            global_name = self._get_scope_persistents(var_name)

            if global_name and self.current_scope:
                # Mark as modified since we're augmenting it
                self.modified_vars.setdefault(self.current_scope, set())
                self.modified_vars[self.current_scope].add(global_name)

                # Check if this is += operation and not with a literal
                if isinstance(node.op, ast.Add) and not self._is_literal_or_na(node.value):
                    # Generate compensation variable name
                    compensation_var = f"{global_name}_kahan_c__"

                    # Add compensation variable to module level if not already there
                    if compensation_var not in [t.id
                                                for assign in self.module_level_assigns
                                                if isinstance(assign, ast.Assign) and len(assign.targets) == 1
                                                for t in assign.targets if isinstance(t, ast.Name)]:
                        self.module_level_assigns.append(
                            ast.Assign(
                                targets=[ast.Name(id=compensation_var, ctx=ast.Store())],
                                value=ast.Constant(value=0.0)
                            )
                        )

                    # Mark compensation variable as modified
                    self.modified_vars[self.current_scope].add(compensation_var)

                    # Transform the value
                    transformed_value = self.visit(cast(ast.AST, node.value))

                    # Create Kahan summation sequence using walrus operator
                    # We'll use a single expression with tuple unpacking
                    # (corrected := value - compensation,
                    #  new_sum := var + corrected,
                    #  compensation := (new_sum - var) - corrected,
                    #  var := new_sum)[-1]

                    # Create the Kahan summation expression
                    kahan_expr = ast.Subscript(
                        value=ast.Tuple(
                            elts=[
                                # First: corrected := value - compensation
                                ast.NamedExpr(
                                    target=ast.Name(id='__kahan_corrected__', ctx=ast.Store()),
                                    value=ast.BinOp(
                                        left=transformed_value,
                                        op=ast.Sub(),
                                        right=ast.Name(id=compensation_var, ctx=ast.Load())
                                    )
                                ),
                                # Second: new_sum := var + corrected
                                ast.NamedExpr(
                                    target=ast.Name(id='__kahan_new_sum__', ctx=ast.Store()),
                                    value=ast.BinOp(
                                        left=ast.Name(id=global_name, ctx=ast.Load()),
                                        op=ast.Add(),
                                        right=ast.Name(id='__kahan_corrected__', ctx=ast.Load())
                                    )
                                ),
                                # Third: compensation := (new_sum - var) - corrected
                                ast.NamedExpr(
                                    target=ast.Name(id=compensation_var, ctx=ast.Store()),
                                    value=ast.BinOp(
                                        left=ast.BinOp(
                                            left=ast.Name(id='__kahan_new_sum__', ctx=ast.Load()),
                                            op=ast.Sub(),
                                            right=ast.Name(id=global_name, ctx=ast.Load())
                                        ),
                                        op=ast.Sub(),
                                        right=ast.Name(id='__kahan_corrected__', ctx=ast.Load())
                                    )
                                ),
                                # Fourth: var := new_sum
                                ast.NamedExpr(
                                    target=ast.Name(id=global_name, ctx=ast.Store()),
                                    value=ast.Name(id='__kahan_new_sum__', ctx=ast.Load())
                                )
                            ],
                            ctx=ast.Load()
                        ),
                        slice=ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=1)),
                        ctx=ast.Load()
                    )

                    # Return as an expression statement
                    return ast.Expr(value=kahan_expr)

                # For other augmented assignments or += with literals, keep the original behavior
                node.target = ast.Name(id=global_name, ctx=ast.Store())
                node.value = self.visit(cast(ast.AST, node.value))
                return node

        return cast(ast.AugAssign, self.generic_visit(node))

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Convert variable references using scope-aware lookup"""
        var_name = node.id
        global_name = self._get_scope_persistents(var_name)

        if global_name:
            # Track this variable in current scope if it's being used
            if self.current_scope:
                self.scope_vars.setdefault(self.current_scope, set())
                self.scope_vars[self.current_scope].add(global_name)

                # Only mark as modified if it's in a Store context
                if isinstance(node.ctx, ast.Store):
                    self.modified_vars.setdefault(self.current_scope, set())
                    self.modified_vars[self.current_scope].add(global_name)

            return ast.Name(id=global_name, ctx=node.ctx)
        return node
