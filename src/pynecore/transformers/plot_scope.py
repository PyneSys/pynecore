"""Require plot declarations to execute directly in the script entry.

The check runs after import normalization and before lowering creates control
flow of its own. Plot function references may be bound to a single static alias,
but cannot escape as callbacks or container values: their eventual call scope
would no longer be verifiable. Plot values and color arguments may still be
conditional; the declaration itself must execute on every completed bar.
"""
import ast
from dataclasses import dataclass, field

from . import ast_walk

__all__ = ['PlotScopeTransformer']

_PLOT_NAMES = frozenset({
    'plot', 'plotshape', 'plotchar', 'plotarrow', 'plotcandle', 'plotbar',
    'hline', 'fill', 'bgcolor', 'barcolor', 'alertcondition',
})
_PLOT_PATHS = {f'lib.{name}': name for name in _PLOT_NAMES}
_PLOT_PATHS.update({'lib.plot.plot': 'plot', 'lib.hline.hline': 'hline'})
_LOCAL_CONTEXTS = (
    ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.TryStar,
    ast.With, ast.AsyncWith, ast.Match, ast.BoolOp, ast.IfExp,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
)

#: Child nodes that hold nothing this pass looks at: no binding, call or reference
_INERT = (ast.expr_context, ast.operator, ast.unaryop, ast.cmpop, ast.boolop)


@dataclass(eq=False)
class _Scope:
    parent: '_Scope | None'
    bindings: dict[str, list[ast.AST]] = field(default_factory=dict)

    def resolve(self, name: str) -> '_Scope | None':
        """Find the lexical scope that binds a name."""
        scope = self
        while scope is not None:
            if name in scope.bindings:
                return scope
            scope = scope.parent
        return None


class PlotScopeTransformer(ast_walk.NodeTransformer):
    """Reject plot calls outside the unconditional body of module-level main."""

    def __init__(self, source: str = '') -> None:
        self._lines = source.splitlines()

    def visit_Module(self, node: ast.Module) -> ast.Module:
        root = _Scope(None)
        owners: dict[ast.AST, _Scope] = {}
        parents: dict[ast.AST, ast.AST] = {}
        allowed: set[ast.Call] = set()
        # A single static alias of a plot function: the aliased reference -> the name it binds
        aliases: dict[ast.AST, ast.Name] = {}
        calls: list[ast.Call] = []
        references: list[ast.Name | ast.Attribute] = []

        def collect(current: ast.AST, scope: _Scope, direct: bool = False) -> None:
            owners[current] = scope
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                if not isinstance(current, ast.Lambda):
                    if current.name.startswith('__test_'):
                        return
                    scope.bindings.setdefault(current.name, []).append(current)
                child = _Scope(scope)
                spec = current.args
                for arg in [*spec.posonlyargs, *spec.args, *spec.kwonlyargs,
                            spec.vararg, spec.kwarg]:
                    if arg is not None:
                        child.bindings[arg.arg] = []
                entry = (isinstance(current, ast.FunctionDef) and current.name == 'main'
                         and isinstance(parents.get(current), ast.Module))
                body = {id(current.body)} if isinstance(current, ast.Lambda) \
                    else {id(stmt) for stmt in current.body}
                for part in ast_walk.iter_child_nodes(current):
                    if isinstance(current, ast.Lambda) and part is current.body:
                        parents[part] = current
                        collect(part, child)
                    elif id(part) not in body:
                        parents[part] = current
                        collect(part, scope)
                if isinstance(current, ast.Lambda):
                    return
                # Only the entry's direct body has to know whether an earlier
                # statement can return, and only until one can
                may_return = not entry
                for stmt in current.body:
                    parents[stmt] = current
                    collect(stmt, child, not may_return)
                    if not may_return:
                        may_return = self._has_return(stmt)
                return
            if isinstance(current, ast.ClassDef):
                scope.bindings.setdefault(current.name, []).append(current)
                child = _Scope(scope)
                for part in ast_walk.iter_child_nodes(current):
                    parents[part] = current
                    collect(part, child)
                return
            if isinstance(current, (ast.Assign, ast.AnnAssign, ast.NamedExpr, ast.AugAssign)):
                targets = current.targets if isinstance(current, ast.Assign) else [current.target]
                for target in targets:
                    for name in ast_walk.walk(target):
                        if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store):
                            values = scope.bindings.setdefault(name.id, [])
                            values.append(current.value or current)
                if (isinstance(current, (ast.Assign, ast.AnnAssign))
                        and len(targets) == 1 and isinstance(targets[0], ast.Name)
                        and isinstance(current.value, (ast.Name, ast.Attribute))):
                    aliases[current.value] = targets[0]
            if isinstance(current, ast.Call):
                calls.append(current)
                if direct:
                    allowed.add(current)
            if isinstance(current, (ast.Name, ast.Attribute)) and isinstance(current.ctx, ast.Load):
                references.append(current)
            if isinstance(current, _LOCAL_CONTEXTS):
                direct = False
            for part in ast_walk.iter_child_nodes(current):
                if isinstance(part, _INERT):
                    continue
                parents[part] = current
                collect(part, scope, direct)

        collect(node, root)

        def path(value: ast.AST, seen: frozenset[tuple[_Scope, str]] = frozenset()) -> str:
            if isinstance(value, ast.Attribute):
                base = path(value.value, seen)
                return base + '.' + value.attr if base else ''
            if not isinstance(value, ast.Name):
                return ''
            scope = owners[value].resolve(value.id)
            if scope is None:
                return value.id
            key = (scope, value.id)
            if key in seen:
                return ''
            bindings = scope.bindings[value.id]
            if len(bindings) == 1 and isinstance(bindings[0], (ast.Name, ast.Attribute)):
                return path(bindings[0], seen | {key})
            return ''

        for call in calls:
            name = _PLOT_PATHS.get(path(call.func))
            if name is not None and call not in allowed:
                self._reject(node, call, name, 'must be called unconditionally in the '
                             'direct body of module-level main, before any possible early return')

        for ref in references:
            name = _PLOT_PATHS.get(path(ref))
            if name is None:
                continue
            parent = parents[ref]
            if isinstance(parent, ast.Call) and parent.func is ref:
                continue
            if isinstance(parent, ast.Attribute) and parent.value is ref:
                if parent.attr in {'plot', 'hline'} or parent.attr.startswith(('style_', 'linestyle_')):
                    continue
            target = aliases.get(ref)
            if target is not None:
                scope = owners[target].resolve(target.id)
                if scope is not None and len(scope.bindings[target.id]) == 1:
                    continue
            self._reject(node, ref, name, 'cannot be passed or stored as a function value; '
                         'use a direct call or a single static alias so its scope can be checked')
        return node

    @staticmethod
    def _has_return(node: ast.AST) -> bool:
        """Whether a statement can return from its own enclosing function.

        A ``return`` is a statement, so only the statements nested in this one
        are searched, never an expression, and not the scopes opened inside it.
        """
        pending = [node]
        while pending:
            current = pending.pop()
            if isinstance(current, ast.Return):
                return True
            if not isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                pending.extend(ast_walk.iter_child_statements(current))
        return False

    def _reject(self, module: ast.Module, node: ast.AST, name: str, reason: str) -> None:
        """Raise a source-located error before the module can execute."""
        line = getattr(node, 'lineno', 0)
        text = self._lines[line - 1] if 0 < line <= len(self._lines) else None
        raise SyntaxError(f'{name}() {reason}.',
                          (getattr(module, '_module_file_path', '<script>'), line,
                           getattr(node, 'col_offset', 0) + 1, text))
