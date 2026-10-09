import ast
from collections.abc import Iterable, Sequence
from typing import cast

from . import ast_walk

#: The comprehension nodes: each is a scope of its own for its loop targets
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)

#: The module name a qualified ``typing.TYPE_CHECKING`` guard reads the flag through
_TYPING = 'typing'


def _binds_typing_module(alias: ast.alias) -> bool:
    """An ``import`` alias that binds the name ``typing`` to the ``typing`` module itself."""
    if alias.asname is None:
        return alias.name.split('.')[0] == _TYPING
    return alias.asname == _TYPING and alias.name == _TYPING


def _argument_names(args: ast.arguments) -> set[str]:
    """The names the parameters of a function or lambda bind."""
    names = {arg.arg for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
    if args.vararg is not None:
        names.add(args.vararg.arg)
    if args.kwarg is not None:
        names.add(args.kwarg.arg)
    return names


def _argument_defaults(args: ast.arguments) -> list[ast.expr]:
    """The default values of a parameter list: they are evaluated in the enclosing scope."""
    return [*args.defaults, *(default for default in args.kw_defaults if default is not None)]


def _flag_imports(nodes: Iterable[ast.AST]) -> set[str]:
    """
    The names a scope's own ``from typing import TYPE_CHECKING`` imports bind.

    :param nodes: The statements of the scope
    :return: The names the flag imports of the scope bind
    """
    names: set[str] = set()
    stack = list(nodes)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
                             *_COMPREHENSIONS)):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == 'typing':
            names.update(alias.asname or alias.name for alias in node.names
                         if alias.name == 'TYPE_CHECKING')
        elif not isinstance(node, ast.expr):
            stack.extend(ast_walk.iter_child_nodes(node))
    return names


def _nonlocal_names(nodes: Iterable[ast.AST]) -> set[str]:
    """
    The names any ``nonlocal`` declaration in the nodes or below them declares.

    :param nodes: The nodes to search
    :return: The declared names
    """
    names: set[str] = set()
    for node in nodes:
        for child in ast_walk.walk(node):
            if isinstance(child, ast.Nonlocal):
                names.update(child.names)
    return names


def _scope_bindings(nodes: Iterable[ast.AST]) -> tuple[set[str], set[str], set[str]]:
    """
    The names one scope binds, and the names it declares ``global`` / ``nonlocal``.

    ``nodes`` are the scope's own nodes (a module or class body, a function body,
    a lambda body). The walk does not enter a nested scope, only the parts of it
    the enclosing scope evaluates (decorators, defaults, bases, the first
    iterable of a comprehension). An import of the flag itself is not a binding:
    the pass removes it, and neither is an import that binds ``typing`` to the
    ``typing`` module.

    :param nodes: The nodes of the scope
    :return: The bound names, the ``global`` names and the ``nonlocal`` names
    """
    bound: set[str] = set()
    declared_global: set[str] = set()
    declared_nonlocal: set[str] = set()
    stack = list(nodes)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound.add(node.name)
            stack.extend(node.decorator_list)
            stack.extend(_argument_defaults(node.args))
            continue
        if isinstance(node, ast.ClassDef):
            bound.add(node.name)
            stack.extend(node.decorator_list)
            stack.extend(node.bases)
            stack.extend(keyword.value for keyword in node.keywords)
            continue
        if isinstance(node, ast.Lambda):
            stack.extend(_argument_defaults(node.args))
            continue
        if isinstance(node, _COMPREHENSIONS):
            stack.append(node.generators[0].iter)
            # An assignment expression in a comprehension binds in the enclosing scope
            for child in ast_walk.walk(node):
                if isinstance(child, ast.NamedExpr) and isinstance(child.target, ast.Name):
                    bound.add(child.target.id)
            continue
        if isinstance(node, ast.Name):
            if not isinstance(node.ctx, ast.Load):
                bound.add(node.id)
        elif isinstance(node, ast.Import):
            bound.update(alias.asname or alias.name.split('.')[0] for alias in node.names
                         if not _binds_typing_module(alias))
        elif isinstance(node, ast.ImportFrom):
            bound.update(alias.asname or alias.name for alias in node.names
                         if alias.name != 'TYPE_CHECKING')
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                bound.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)):
            if node.name:
                bound.add(node.name)
        elif isinstance(node, ast.MatchMapping):
            if node.rest:
                bound.add(node.rest)
        elif isinstance(node, ast.Global):
            declared_global.update(node.names)
        elif isinstance(node, ast.Nonlocal):
            declared_nonlocal.update(node.names)
        stack.extend(ast_walk.iter_child_nodes(node))
    return bound, declared_global, declared_nonlocal


class TypeCheckingStripperTransformer(ast_walk.NodeTransformer):
    """
    Remove `if TYPE_CHECKING:` blocks and the TYPE_CHECKING import from @pyne files.
    These blocks contain IDE-only type hints (casts, re-annotations) that are unnecessary at runtime.
    Any other read of the removed flag becomes the constant ``False``, its runtime value.

    A name means the flag only where it resolves to the flag's binding: a scope
    that binds the name itself (a parameter, an assignment, a loop target, ...)
    keeps its ``if`` and its reads, and a name the module rebinds at module level
    keeps its import as well. The same holds for ``typing`` in a qualified
    ``typing.TYPE_CHECKING`` guard: it is the flag only where ``typing`` resolves
    to an import of the ``typing`` module.
    """

    def __init__(self):
        #: The names the flag goes by: ``TYPE_CHECKING`` and every alias a
        #: ``from typing import TYPE_CHECKING as ...`` binds. The import is
        #: removed, so an ``if`` on an alias has to go with it
        self._names = {'TYPE_CHECKING'}
        #: The names a scope can shadow: the flag names and ``typing``
        self._tracked = {'TYPE_CHECKING', _TYPING}
        #: The names the removed ``typing`` import bound: any other read of one
        #: of them, where it resolves to the import, is the constant False
        self._imported: set[str] = set()
        #: The flag names the module binds at module level besides the flag
        #: import: they are not the flag anywhere they resolve to the module
        self._unflagged: set[str] = set()
        #: The flag names that do not mean the flag in the scope being visited
        self._shadow: set[str] = set()
        #: The same for a function defined in the scope being visited: a class
        #: body's bindings do not reach the functions defined in it
        self._function_shadow: set[str] = set()

    def visit_Module(self, node: ast.Module) -> ast.Module:
        # Both forms this pass rewrites are statements: a module without either
        # is left as it is, without visiting a single expression
        relevant = False
        for stmt in ast_walk.walk_statements(node):
            if isinstance(stmt, ast.ImportFrom) and stmt.module == 'typing':
                for alias in stmt.names:
                    if alias.name == 'TYPE_CHECKING':
                        relevant = True
                        if alias.asname:
                            self._names.add(alias.asname)
                        self._imported.add(alias.asname or alias.name)
            elif isinstance(stmt, ast.If) and self._is_type_checking(stmt.test):
                relevant = True
        if not relevant:
            return node
        self._tracked = self._names | {_TYPING}
        # A name the module binds itself is not (only) the flag; reading it as
        # False, stripping an ``if`` on it or removing its import would change
        # what the code means
        bound, _, _ = _scope_bindings(node.body)
        for stmt in ast_walk.walk_statements(node):
            if isinstance(stmt, ast.Global):
                bound.update(stmt.names)
        self._unflagged = bound & self._tracked
        self._shadow = self._function_shadow = set(self._unflagged)
        return cast(ast.Module, self.generic_visit(node))

    def _scope_shadow(self, outer: set[str], nodes: Sequence[ast.AST],
                      params: set[str] | None = None, function: bool = False) -> set[str]:
        """
        The flag names that do not mean the flag in a nested scope.

        :param outer: The shadowed names the scope inherits
        :param nodes: The scope's own nodes
        :param params: The names the scope's parameters bind
        :param function: The scope is a function body: a ``nonlocal`` declaration of
                         a nested function can resolve to a binding it holds
        :return: The shadowed names inside the scope
        """
        bound, declared_global, declared_nonlocal = _scope_bindings(nodes)
        if params:
            bound |= params
        if function:
            # A nested ``nonlocal`` needs the binding to survive: the import stays.
            # A nested function that rebinds ``typing`` makes it unknown here
            bound |= (_flag_imports(nodes) | {_TYPING}) & _nonlocal_names(nodes)
        local = (bound | declared_nonlocal) & self._tracked
        return ((outer | local) - declared_global) | (declared_global & self._unflagged)

    def _visit_block(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        """Visit a statement list, splicing what each visit returns as ``generic_visit`` does."""
        result: list[ast.stmt] = []
        for stmt in stmts:
            new = self.visit(stmt)
            if new is None:
                continue
            if isinstance(new, ast.AST):
                result.append(cast(ast.stmt, new))
            else:
                result.extend(new)
        return result

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.AST:
        # Decorators, defaults and annotations are evaluated in the enclosing scope
        node.decorator_list = [self.visit(expr) for expr in node.decorator_list]
        node.args = self.visit(node.args)
        if node.returns is not None:
            node.returns = self.visit(node.returns)
        saved = self._shadow, self._function_shadow
        self._shadow = self._function_shadow = self._scope_shadow(
            self._function_shadow, node.body, _argument_names(node.args), function=True)
        node.body = self._visit_block(node.body)
        self._shadow, self._function_shadow = saved
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._visit_function(node)

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        node.args = self.visit(node.args)
        saved = self._shadow, self._function_shadow
        self._shadow = self._function_shadow = self._scope_shadow(
            self._function_shadow, [node.body], _argument_names(node.args))
        node.body = self.visit(node.body)
        self._shadow, self._function_shadow = saved
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.decorator_list = [self.visit(expr) for expr in node.decorator_list]
        node.bases = [self.visit(expr) for expr in node.bases]
        node.keywords = [self.visit(keyword) for keyword in node.keywords]
        saved = self._shadow
        self._shadow = self._scope_shadow(self._shadow, node.body)
        node.body = self._visit_block(node.body)
        self._shadow = saved
        return node

    def _visit_comprehension(self, node: ast.ListComp | ast.SetComp | ast.GeneratorExp
                             | ast.DictComp) -> ast.AST:
        generators = node.generators
        # The first iterable is evaluated in the enclosing scope
        generators[0].iter = self.visit(generators[0].iter)
        targets: set[str] = set()
        for generator in generators:
            targets.update(child.id for child in ast_walk.walk(generator.target)
                           if isinstance(child, ast.Name))
        # A comprehension is a function scope: the class body around it does not
        # reach it, and its loop targets reach the lambdas defined in it
        saved = self._shadow, self._function_shadow
        self._shadow = self._function_shadow = self._function_shadow | (targets & self._tracked)
        for index, generator in enumerate(generators):
            generator.target = self.visit(generator.target)
            if index:
                generator.iter = self.visit(generator.iter)
            generator.ifs = [self.visit(expr) for expr in generator.ifs]
        if isinstance(node, ast.DictComp):
            node.key = self.visit(node.key)
            node.value = self.visit(node.value)
        else:
            node.elt = self.visit(node.elt)
        self._shadow, self._function_shadow = saved
        return node

    def visit_ListComp(self, node: ast.ListComp) -> ast.AST:
        return self._visit_comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> ast.AST:
        return self._visit_comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.AST:
        return self._visit_comprehension(node)

    def visit_DictComp(self, node: ast.DictComp) -> ast.AST:
        return self._visit_comprehension(node)

    def visit_Name(self, node: ast.Name) -> ast.expr:
        # Every other read of the flag (``if not TYPE_CHECKING:``, ``flag =
        # TYPE_CHECKING``) outlives the removed import, so it is the runtime value
        if (node.id in self._imported and isinstance(node.ctx, ast.Load)
                and node.id not in self._shadow):
            return ast.copy_location(ast.Constant(value=False), node)
        return node

    def _is_type_checking(self, test: ast.expr) -> bool:
        """Match: ``TYPE_CHECKING`` (or an alias of it) / ``typing.TYPE_CHECKING``"""
        return ((isinstance(test, ast.Name) and test.id in self._names
                 and test.id not in self._shadow)
                or (isinstance(test, ast.Attribute) and test.attr == 'TYPE_CHECKING'
                    and isinstance(test.value, ast.Name)
                    and test.value.id == _TYPING and _TYPING not in self._shadow))

    def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt] | None:
        if self._is_type_checking(node.test):
            # The name is False at runtime: what runs is the else body
            return self._visit_block(node.orelse) or None
        return self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom | None:
        if node.module == 'typing':
            node.names = [alias for alias in node.names
                          if alias.name != 'TYPE_CHECKING'
                          or (alias.asname or alias.name) in self._shadow]
            if not node.names:
                return None
        return node
