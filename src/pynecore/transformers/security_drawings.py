"""Reject drawing creation in security expressions before request lowering."""
import ast
from dataclasses import dataclass, field

from . import ast_walk

_DRAWING_CALLS = frozenset(
    f'lib.{namespace}.{method}'
    for namespace, methods in (
        ('label', ('new', 'copy')), ('line', ('new', 'copy')), ('box', ('new', 'copy')),
        ('table', ('new',)), ('polyline', ('new',)), ('linefill', ('new',)),
    )
    for method in methods
)
_REQUESTS = {'lib.request.security', 'lib.request.security_lower_tf'}


@dataclass(eq=False)
class _Scope:
    parent: '_Scope | None'
    bindings: dict[str, list[ast.AST]] = field(default_factory=dict)

    def resolve(self, name: str) -> '_Scope | None':
        scope = self
        while scope is not None:
            if name in scope.bindings:
                return scope
            scope = scope.parent
        return None


class SecurityDrawingsTransformer(ast_walk.NodeTransformer):
    """Check expression dependencies without forbidding empty drawing fields."""

    def visit_Module(self, node: ast.Module) -> ast.Module:
        root = _Scope(None)
        owners = {}
        bodies = {}
        requests = []
        annotations = {}
        method_names = {'method_call'}

        def collect(current, scope):
            owners[current] = scope
            if isinstance(current, ast.ImportFrom) and current.module == 'pynecore.core.pine_method':
                method_names.update(alias.asname or alias.name for alias in current.names
                                    if alias.name == 'method_call')
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                if not isinstance(current, ast.Lambda):
                    scope.bindings.setdefault(current.name, []).append(current)
                child = _Scope(scope)
                bodies[current] = child
                args = current.args
                for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
                    if arg is not None:
                        child.bindings[arg.arg] = []
                        if arg.annotation is not None:
                            annotations[child, arg.arg] = arg.annotation
                for default in [*args.defaults, *args.kw_defaults]:
                    if default is not None:
                        collect(default, scope)
                body = [current.body] if isinstance(current, ast.Lambda) else current.body
                for statement in body:
                    collect(statement, child)
                return
            if isinstance(current, ast.ClassDef):
                scope.bindings.setdefault(current.name, []).append(current)
                child = _Scope(scope)
                for statement in ast_walk.iter_child_nodes(current):
                    collect(statement, child)
                return
            if isinstance(current, (ast.Assign, ast.AnnAssign, ast.NamedExpr, ast.AugAssign)):
                targets = current.targets if isinstance(current, ast.Assign) else [current.target]
                for target in targets:
                    for name in ast_walk.walk(target):
                        if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store):
                            values = scope.bindings.setdefault(name.id, [])
                            if current.value is not None:
                                values.append(current.value)
                            if isinstance(current, ast.AnnAssign):
                                annotations[scope, name.id] = current.annotation
            for item in ast_walk.iter_child_nodes(current):
                collect(item, scope)
            if isinstance(current, ast.Call):
                requests.append(current)

        for statement in node.body:
            collect(statement, root)

        def path(value, seen=None):
            if isinstance(value, ast.Attribute):
                return path(value.value, seen) + '.' + value.attr
            if not isinstance(value, ast.Name):
                return ''
            scope = owners[value].resolve(value.id)
            if scope is None:
                return value.id
            key = (scope, value.id)
            seen = set() if seen is None else seen
            if key in seen:
                return ''
            bindings = scope.bindings[value.id]
            if len(bindings) == 1 and isinstance(bindings[0], (ast.Name, ast.Attribute)):
                return path(bindings[0], seen | {key})
            return ''

        def drawing_name(call):
            name = path(call.func)
            if name in _DRAWING_CALLS:
                return name[4:]
            if (isinstance(call.func, ast.Name) and call.func.id in method_names
                    and len(call.args) >= 2 and isinstance(call.args[0], ast.Constant)
                    and call.args[0].value == 'copy' and isinstance(call.args[1], ast.Name)):
                receiver = call.args[1]
                scope = owners[receiver].resolve(receiver.id)
                annotation = annotations.get((scope, receiver.id))
                if isinstance(annotation, ast.Subscript):
                    annotation = annotation.slice
                type_name = (annotation.id if isinstance(annotation, ast.Name)
                             else annotation.attr if isinstance(annotation, ast.Attribute) else '')
                if type_name in ('Label', 'Line', 'Box'):
                    return type_name.lower() + '.copy'
            return None

        request_scopes = {owners[call] for call in requests if path(call.func) in _REQUESTS}
        for call in requests:
            if not isinstance(call.func, ast.Name):
                continue
            owner = owners[call.func].resolve(call.func.id)
            if owner is None:
                continue
            for function in owner.bindings[call.func.id]:
                scope = bodies.get(function)
                if scope not in request_scopes:
                    continue
                args = function.args
                for argument, value in zip([*args.posonlyargs, *args.args], call.args):
                    scope.bindings[argument.arg].append(value)
                for keyword in call.keywords:
                    if keyword.arg in scope.bindings:
                        scope.bindings[keyword.arg].append(keyword.value)

        for request in requests:
            request_name = path(request.func)
            if request_name not in _REQUESTS:
                continue
            expression = next((kw.value for kw in request.keywords if kw.arg == 'expression'),
                              request.args[2] if len(request.args) > 2 else None)
            if expression is None:
                continue
            pending = [expression]
            seen = set()
            while pending:
                current = pending.pop()
                if current in seen:
                    continue
                seen.add(current)
                builtin = drawing_name(current) if isinstance(current, ast.Call) else None
                if builtin is not None:
                    filename = getattr(node, '_module_file_path', '<pyne>')
                    raise SyntaxError(
                        f'{builtin}() cannot be used in the expression of {request_name[4:]}().',
                        (filename, current.lineno, current.col_offset + 1, None),
                    )
                if isinstance(current, ast.Name) and isinstance(current.ctx, ast.Load):
                    scope = owners[current].resolve(current.id)
                    if scope is not None:
                        for binding in scope.bindings[current.id]:
                            if binding in bodies:
                                body = binding.body
                                pending.extend(body if isinstance(body, list) else [body])
                                pending.extend(default for default in
                                               [*binding.args.defaults, *binding.args.kw_defaults]
                                               if default is not None)
                            else:
                                pending.append(binding)
                if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    continue
                pending.extend(ast_walk.iter_child_nodes(current))
        return node
