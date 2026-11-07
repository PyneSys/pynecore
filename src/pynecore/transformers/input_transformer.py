import ast
import linecache

from . import ast_walk
from .pipeline_names import is_pipeline_name
from .price_series import BUILTIN_PRICE_SERIES


class InputTransformer(ast_walk.NodeVisitor):
    """
    Transform input function calls:
    1. Add _id parameter to input calls
    2. Add getattr for source inputs at the start of functions
    3. Add required imports (lib, na) if not present
    4. Reject every input call that is not a parameter default of module-level ``main()``
    Must be applied after SeriesTransformer.

    Every edit is made in place and no node is replaced, so the walk is a plain
    visit; the module visit returns the module.
    """

    def __init__(self):
        self.has_source_inputs = False
        self.imported_names: set[str] = set()  # Track what's already imported
        # The input calls that are parameter defaults of an entry
        self._declared: set[ast.Call] = set()
        self._module: ast.Module | None = None
        self._module_level: set[int] = set()  # ids of the module's own statements

    @staticmethod
    def _is_input_call(node: ast.Call) -> bool:
        """Check if node is ``lib.input()`` or ``lib.input.xxx()``

        The import normalizer has already rewritten every name imported from
        ``pynecore.lib`` to its ``lib.`` path, so a bare ``input`` here is a name the
        script binds itself (or Python's builtin), never PyneCore's input.
        """
        # Handle lib.input(...) pattern
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'input':
            return isinstance(node.func.value, ast.Name) and node.func.value.id == 'lib'

        # Handle lib.input.xxx(...) pattern
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Attribute):
            return (isinstance(node.func.value.value, ast.Name) and
                    node.func.value.value.id == 'lib' and
                    node.func.value.attr == 'input')

        return False

    def _declare_inputs(self, node: ast.FunctionDef) -> list[str]:
        """Add ``_id`` to the input calls among an entry's defaults and collect its source vars

        :param node: Module-level ``main()`` or one of its generated clones
        :return: The source-input parameters of ``node``, to rebind to the series they name
        """
        source_vars: list[str] = []
        spec = node.args
        positional = spec.posonlyargs + spec.args
        pairs = list(zip(positional[len(positional) - len(spec.defaults):], spec.defaults))
        pairs.extend((arg, default) for arg, default in zip(spec.kwonlyargs, spec.kw_defaults)
                     if default is not None)
        for arg, default in pairs:
            if default and isinstance(default, ast.Call):
                if self._is_input_call(default):
                    self._declared.add(default)
                    # Add id keyword argument with argument name
                    default.keywords.append(
                        ast.keyword(
                            arg='_id',
                            value=ast.Constant(value=arg.arg)
                        )
                    )

                    # Check if it's a source input
                    is_source_call = False
                    is_input_call = False

                    if isinstance(default.func, ast.Attribute) and default.func.attr == 'source':
                        # This is lib.input.source call
                        is_source_call = True
                    elif isinstance(default.func, ast.Attribute) and default.func.attr == 'input':
                        # This is lib.input call
                        is_input_call = True

                    # Find the defval parameter value (either positional or keyword)
                    defval_node = None
                    if default.args:
                        defval_node = default.args[0]
                    else:
                        # Look for defval keyword argument
                        for kw in default.keywords:
                            if kw.arg == 'defval':
                                defval_node = kw.value
                                break

                    # Only proceed if it's a source call or input call with a defval
                    # ``input.source()`` always returns the source NAME, so its defval
                    # is resolved whatever shape it has. The generic ``input()`` only
                    # returns a name when the defval is a builtin price series; for any
                    # other module-level constant (a color, number or string) it returns
                    # the value itself, which must never be looked up on ``lib``.
                    if (is_source_call or is_input_call) and defval_node:
                        source_name = None

                        if isinstance(defval_node, ast.Constant) and is_source_call:
                            # Handle string constant in lib.input.source
                            source_name = defval_node.value
                        elif isinstance(defval_node, ast.Attribute):
                            # Handle attribute reference (e.g., lib.close or close)
                            attr = defval_node
                            if (isinstance(attr.value, ast.Name) and attr.value.id == 'lib'
                                    and (is_source_call or attr.attr in BUILTIN_PRICE_SERIES)):
                                # For lib.xxx pattern, store the attribute name
                                source_name = attr.attr
                        elif isinstance(defval_node, ast.Name):
                            # Handle direct name reference (e.g., close)
                            if is_source_call or defval_node.id in BUILTIN_PRICE_SERIES:
                                source_name = defval_node.id

                        if source_name:
                            source_vars.append(arg.arg)
                            self.has_source_inputs = True
        return source_vars

    def visit_Call(self, node: ast.Call) -> ast.Call:
        """Reject an input call that is not a parameter default of ``main()``"""
        if node not in self._declared and self._is_input_call(node):
            self._reject(node)
        self.generic_visit(node)
        return node

    def _reject(self, node: ast.Call) -> None:
        """Raise a source-located error before the module can execute.

        An input outside the parameter defaults gets no ``_id``: it never reaches the
        settings file, no override can reach it, and it silently runs with its default.
        """
        name = ast.unparse(node.func).removeprefix('lib.')
        filename = getattr(self._module, '_module_file_path', '<script>')
        line = getattr(node, 'lineno', 0)
        text = (linecache.getline(filename, line) or None) if line > 0 else None
        raise SyntaxError(f'{name}() is not a main() parameter default; inputs must be declared '
                          f'as main() parameter defaults: def main(length=input.int(14, ...))',
                          (filename, line, getattr(node, 'col_offset', 0) + 1, text))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Insert getattr calls at the start of functions for source inputs"""
        # The runner reads inputs only from the parameter defaults of module-level
        # main(). The security slices are generated module-level clones of main()
        # that copy its parameters; their names are pipeline names no script can bind
        source_vars: list[str] = []
        if id(node) in self._module_level and (node.name == 'main' or is_pipeline_name(node.name)):
            source_vars = self._declare_inputs(node)

        # Process function arguments and body
        self.generic_visit(node)

        # Add getattr for each source input of this function
        for var_name in source_vars:
            # Create: var_name = __pyne_getattr__(lib, var_name, lib.na)
            # A reserved module-level alias of the builtin is used instead of a
            # bare ``getattr``: a script variable named ``getattr`` would shadow
            # the builtin in the function scope and break this resolution (same
            # class of bug as a ``len`` input shadowing the loop-guard length).
            assign = ast.Assign(
                targets=[ast.Name(id=var_name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id='__pyne_getattr__', ctx=ast.Load()),
                    args=[
                        ast.Name(id='lib', ctx=ast.Load()),
                        ast.Name(id=var_name, ctx=ast.Load()),
                        ast.Attribute(
                            value=ast.Name(id='lib', ctx=ast.Load()),
                            attr='na',
                            ctx=ast.Load()
                        )
                    ],
                    keywords=[]
                )
            )
            node.body.insert(ast_walk.docstring_end(node.body), assign)

        return node

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Add required imports if not present"""
        self._module = node
        self._module_level = {id(stmt) for stmt in node.body}
        # Process the module first to collect existing imports
        self.generic_visit(node)

        if not self.has_source_inputs:
            return node

        # Source-input resolution emits ``__pyne_getattr__(lib, name, lib.na)``;
        # bind that reserved alias to the builtin at module scope so a
        # function-local ``getattr`` variable cannot shadow it.
        alias_import = ast.ImportFrom(
            module='builtins',
            names=[ast.alias(name='getattr', asname='__pyne_getattr__')],
            level=0)
        node.body.insert(ast_walk.future_end(node.body), alias_import)

        return node
