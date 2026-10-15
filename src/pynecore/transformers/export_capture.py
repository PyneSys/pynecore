"""Validate library exports before closure and state lowering.

An exported function may read immutable scalar bindings from library main scope,
including per-bar values. Objects, inputs and reassigned bindings remain private.
Captured per-bar scalars use the library's state slots so exports defined on the
first bar keep reading current values on subsequent bars.
"""
import ast
from importlib.machinery import PathFinder
from dataclasses import dataclass, field
from pathlib import Path

from . import ast_walk
from .pine_qualifier import CONST, lib_call_qualifier, lib_value_qualifier, builtin_call_qualifier
from .pine_type_infer import lib_types
from .pine_type_rules import BUILTIN_CALL_TYPES, SCALARS, UNKNOWN, annotation_type

__all__ = ['ExportCaptureTransformer']


@dataclass(eq=False)
class _Scope:
    node: ast.AST
    parent: '_Scope | None'
    bindings: dict[str, list[ast.AST]] = field(default_factory=dict)
    reads: list[ast.Name] = field(default_factory=list)
    children: list['_Scope'] = field(default_factory=list)
    defaults: list[ast.Name] = field(default_factory=list)
    globals: set[str] = field(default_factory=set)
    nonlocals: set[str] = field(default_factory=set)

    def resolve(self, name: str) -> '_Scope | None':
        """Resolve a name using lexical bindings, including explicit scope declarations."""
        scope: _Scope | None = self
        if name in self.globals:
            while scope.parent is not None:
                scope = scope.parent
        elif name in self.nonlocals:
            scope = self.parent
        while scope is not None:
            if name in scope.bindings:
                return scope
            scope = scope.parent
        return None


def _path(node: ast.AST) -> str:
    """Return a dotted identifier path, or an empty string for other expressions."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _path(node.value) + '.' + node.attr
    return ''


class _EnumResolver:
    """Resolve enum declarations from source without executing imported modules."""

    def __init__(self):
        self.modules: dict[tuple[str, str, int], _Scope | None] = {}
        self.sources: dict[str, tuple[int, int]] = {}

    def module(self, name: str, owner: _Scope, level: int = 0) -> _Scope | None:
        while owner.parent is not None:
            owner = owner.parent
        source = getattr(owner.node, '_module_file_path', '')
        key = (name, source if level else '', level)
        if key in self.modules:
            return self.modules[key]
        self.modules[key] = None
        search = None
        package_init: Path | None = None
        if level:
            if not source:
                return None
            base = Path(source).parent
            for _ in range(level - 1):
                base = base.parent
            search = [str(base)]
            if not name:
                package_init = base / '__init__.py'
        spec = None
        try:
            if package_init is not None:
                path = package_init
            else:
                segments = name.split('.')
                for index in range(len(segments)):
                    # Resolve one component at a time: namespace packages need
                    # not exist in sys.modules for their source to be inspected.
                    spec = PathFinder.find_spec(segments[index], search)
                    if spec is None:
                        return None
                    locations = spec.submodule_search_locations
                    search = list(locations) if locations is not None else None
                    if index < len(segments) - 1 and search is None:
                        return None
                if spec is None or not spec.origin or not spec.origin.endswith('.py'):
                    return None
                path = Path(spec.origin)
            from .module_interface import stable_source
            loaded = stable_source(path)
            if loaded is None:
                return None
            data, fingerprint = loaded
            tree = ast.parse(data, filename=str(path))
        except (ImportError, OSError, SyntaxError, ValueError):
            return None
        module_path = str(path.resolve())
        setattr(tree, '_module_file_path', module_path)
        self.sources[module_path] = fingerprint
        scope = _Scope(tree, None)

        def bind(node: ast.AST) -> None:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                scope.bindings.setdefault(node.name, []).append(node)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    scope.bindings.setdefault(alias.asname or alias.name.split('.')[0], []).append(node)
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    for item in ast_walk.walk(target):
                        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store):
                            scope.bindings.setdefault(item.id, []).append(node)
            else:
                for child in ast_walk.iter_child_nodes(node):
                    bind(child)

        for statement in tree.body:
            bind(statement)
        self.modules[key] = scope
        return scope

    def resolve(self, parts: list[str], scope: _Scope, seen: set[tuple[_Scope, str]]) \
            -> tuple[ast.ClassDef | str, _Scope] | None:
        if not parts:
            return None
        owner = scope.resolve(parts[0])
        if owner is None:
            return None
        key = (owner, '.'.join(parts))
        bindings = owner.bindings[parts[0]]
        if key in seen or len(bindings) != 1:
            return None
        seen = seen | {key}
        binding = bindings[0]
        if isinstance(binding, ast.ClassDef):
            return (binding, owner) if len(parts) == 1 else None
        if isinstance(binding, (ast.Assign, ast.AnnAssign)) and binding.value is not None:
            alias = _path(binding.value)
            return self.resolve(alias.split('.') + parts[1:], owner, seen) if alias else None
        if isinstance(binding, ast.Import):
            alias = next((alias for alias in binding.names
                          if (alias.asname or alias.name.split('.')[0]) == parts[0]), None)
            if alias is None:
                return None
            prefix = [parts[0]] if alias.asname else alias.name.split('.')
            if parts[:len(prefix)] != prefix:
                return None
            module, tail, level = alias.name, parts[len(prefix):], 0
        elif isinstance(binding, ast.ImportFrom):
            alias = next((alias for alias in binding.names
                          if (alias.asname or alias.name) == parts[0]), None)
            if alias is None:
                return None
            module, tail, level = binding.module or '', [alias.name, *parts[1:]], binding.level
        else:
            return None
        if module == 'enum' and not level and len(tail) == 1 \
                and tail[0] in ('Enum', 'StrEnum', 'IntEnum'):
            return tail[0], owner
        imported = self.module(module, owner, level)
        return self.resolve(tail, imported, seen) if imported is not None else None

    def member(self, path: str, scope: _Scope) -> bool:
        parts = path.split('.')
        if len(parts) < 2:
            return False
        resolved = self.resolve(parts[:-1], scope, set())
        if resolved is None:
            return False
        definition, owner = resolved
        if not isinstance(definition, ast.ClassDef):
            return False
        if definition.decorator_list or definition.keywords:
            return False
        bases = [self.resolve(_path(base).split('.'), owner, set()) for base in definition.bases]
        if not any(base is not None and isinstance(base[0], str) for base in bases):
            return False
        bindings = [statement for statement in definition.body
                    if isinstance(statement, (ast.Assign, ast.AnnAssign))
                    and any(isinstance(target, ast.Name) and target.id == parts[-1]
                            for target in (statement.targets if isinstance(statement, ast.Assign)
                                           else [statement.target]))]
        return (len(bindings) == 1 and not parts[-1].startswith('_')
                and isinstance(bindings[0].value, ast.Constant))


class ExportCaptureTransformer(ast_walk.NodeTransformer):
    """Validate export captures and retain per-bar scalar captures in library state."""

    def visit_Module(self, node: ast.Module) -> ast.Module:
        main_nodes = [statement for statement in node.body
                      if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                      and statement.name == 'main'
                      and any(isinstance(dec, ast.Call) and _path(dec.func) in
                              ('lib.script.library', 'script.library')
                              for dec in statement.decorator_list)]
        if not main_nodes:
            return node
        root = _Scope(node, None)
        scopes: dict[ast.AST, _Scope] = {node: root}
        export_names = {'export'}
        namespace_names: set[str] = set()
        for statement in node.body:
            if isinstance(statement, ast.ImportFrom) and statement.module == 'pynecore.core.pine_export':
                export_names.update(alias.asname or alias.name for alias in statement.names
                                    if alias.name == 'export')

            if isinstance(statement, ast.ImportFrom) and statement.module == 'pynecore.core.pine_import':
                namespace_names.update(alias.asname or alias.name for alias in statement.names
                                       if alias.name == 'shadowed_namespace')

        def collect(current: ast.AST, scope: _Scope) -> None:
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                if not isinstance(current, ast.Lambda):
                    scope.bindings.setdefault(current.name, []).append(current)
                    for decorator in current.decorator_list:
                        collect(decorator, scope)
                child = _Scope(current, scope)
                scopes[current] = child
                scope.children.append(child)
                args = current.args
                for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
                    if arg is not None:
                        child.bindings[arg.arg] = [arg]
                # Defaults can capture state even if the body only reads a parameter.
                for default in [*args.defaults, *args.kw_defaults]:
                    if default is not None:
                        child.defaults.extend(item for item in ast_walk.walk(default)
                                              if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load))
                body = [current.body] if isinstance(current, ast.Lambda) else current.body
                for statement in body:
                    collect(statement, child)
                return
            if isinstance(current, ast.ClassDef):
                scope.bindings.setdefault(current.name, []).append(current)
                return
            if isinstance(current, (ast.Import, ast.ImportFrom)):
                for alias in current.names:
                    scope.bindings.setdefault(alias.asname or alias.name.split('.')[0], []).append(current)
                return
            if isinstance(current, ast.Global):
                scope.globals.update(current.names)
                return
            if isinstance(current, ast.Nonlocal):
                scope.nonlocals.update(current.names)
                return
            if isinstance(current, (ast.Assign, ast.AnnAssign, ast.NamedExpr, ast.AugAssign)):
                targets = current.targets if isinstance(current, ast.Assign) else [current.target]
                for target in targets:
                    for item in ast_walk.walk(target):
                        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store):
                            scope.bindings.setdefault(item.id, []).append(current)
                if current.value is not None:
                    collect(current.value, scope)
                for target in targets:
                    if not isinstance(target, ast.Name):
                        collect(target, scope)
                return
            if isinstance(current, ast.Name):
                if isinstance(current.ctx, ast.Load):
                    scope.reads.append(current)
                elif current.id not in scope.bindings:
                    scope.bindings[current.id] = [current]
                return
            for child in ast_walk.iter_child_nodes(current):
                collect(child, scope)

        for statement in node.body:
            collect(statement, root)
        mains = [scopes[main] for main in main_nodes]
        global_scopes = {root, *mains}
        enums = _EnumResolver()
        # Explicit outer writes invalidate an otherwise literal binding too.
        for scope in scopes.values():
            for name in scope.globals | scope.nonlocals:
                owner = scope.resolve(name)
                if owner is not None and owner is not scope:
                    owner.bindings[name].extend(scope.bindings.get(name, ()))

        def constant(value: ast.AST, scope: _Scope, visiting: set[tuple[_Scope, str]],
                     allow_series: bool = False) -> bool:
            if isinstance(value, ast.Constant):
                return True
            if isinstance(value, ast.Name):
                owner = scope.resolve(value.id)
                return owner is not None and constant_binding(owner, value.id, visiting, allow_series)
            if isinstance(value, ast.Attribute):
                path = _path(value)
                if path.startswith('lib.'):
                    key = path[4:]
                    entry = lib_types().get(key)
                    if entry is not None:
                        return (entry['kind'] == 'value' and not entry.get('callable', False)
                                and (lib_value_qualifier(key, entry['ty']) == CONST
                                     or allow_series and entry['ty'] in SCALARS))
                owner = scope.resolve(path.split('.')[0])
                if owner is not None and any(
                    isinstance(binding, ast.ClassDef)
                    and any(_path(base).split('.')[-1] in ('Enum', 'StrEnum', 'IntEnum')
                            for base in binding.bases)
                    for binding in owner.bindings[path.split('.')[0]]):
                    return True
                return enums.member(path, scope)
            if isinstance(value, (ast.UnaryOp, ast.BinOp, ast.BoolOp, ast.Compare, ast.IfExp)):
                return all(constant(child, scope, visiting, allow_series)
                           for child in ast_walk.iter_child_nodes(value)
                           if isinstance(child, ast.expr))
            if allow_series and isinstance(value, ast.Subscript):
                return (constant(value.value, scope, visiting, True)
                        and constant(value.slice, scope, visiting, True))
            if isinstance(value, ast.Call):
                path = _path(value.func)
                arguments = [*value.args, *(kw.value for kw in value.keywords)]
                if path in namespace_names:
                    return all(isinstance(arg, (ast.Name, ast.Attribute)) for arg in arguments)
                if not all(constant(arg, scope, visiting, allow_series) for arg in arguments):
                    return False
                if path.startswith('lib.'):
                    key = path[4:]
                    if lib_call_qualifier(key, [CONST] * len(arguments), len(arguments)) == CONST:
                        return True
                    entry = lib_types().get(key, {})
                    returns = ([impl.get('ret') for impl in entry['impls']]
                               if entry.get('kind') == 'overloads' else [entry.get('ret')])
                    return allow_series and all(isinstance(ret, str) and ret in SCALARS for ret in returns)
                owner = scope.resolve(path)
                result = BUILTIN_CALL_TYPES.get(path)
                if allow_series and isinstance(result, str) and result in SCALARS:
                    if owner is None or all(
                            isinstance(binding, ast.ImportFrom)
                            and binding.module == 'pynecore.core.pine_cast'
                            for binding in owner.bindings[path]):
                        return True
                return (owner is None
                        and builtin_call_qualifier(path, [CONST] * len(arguments)) == CONST)
            return False

        def constant_binding(scope: _Scope, name: str, visiting: set[tuple[_Scope, str]],
                             allow_series: bool = False) -> bool:
            allow_series = allow_series and scope in mains
            key = (scope, name)
            if key in visiting:
                return False
            bindings = scope.bindings[name]
            if len(bindings) != 1:
                return False
            binding = bindings[0]
            if isinstance(binding, (ast.Import, ast.ImportFrom, ast.ClassDef)):
                return True
            if not isinstance(binding, (ast.Assign, ast.AnnAssign)) or binding.value is None:
                return False
            if allow_series and isinstance(binding, ast.Assign) and (
                    len(binding.targets) != 1 or not isinstance(binding.targets[0], ast.Name)):
                return False
            if isinstance(binding, ast.AnnAssign):
                type_ = annotation_type(binding.annotation)
                if allow_series and type_ != UNKNOWN and type_ not in SCALARS:
                    return False
                annotation = binding.annotation
                if isinstance(annotation, ast.Subscript):
                    annotation = annotation.value
                if not allow_series and _path(annotation).split('.')[-1] in (
                        'Series', 'PersistentSeries', 'IBPersistentSeries'):
                    return False
            return constant(binding.value, scope, visiting | {key}, allow_series)

        default_reads = {id(read) for scope in scopes.values() for read in scope.defaults}
        captures: dict[tuple[_Scope, str], ast.Assign | ast.AnnAssign] = {}
        for scope in scopes.values():
            function = scope.node
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not any((_path(dec.func if isinstance(dec, ast.Call) else dec) in export_names
                        or _path(dec.func if isinstance(dec, ast.Call) else dec).endswith('.export'))
                       for dec in function.decorator_list):
                continue
            forbidden_scopes = set(global_scopes)
            ancestor = scope.parent
            while ancestor is not None:
                forbidden_scopes.add(ancestor)
                ancestor = ancestor.parent
            pending = [scope]
            seen: set[_Scope] = set()
            while pending:
                current = pending.pop()
                if current in seen:
                    continue
                seen.add(current)
                pending.extend(current.children)
                reads = [(read, current) for read in current.reads]
                if current.parent is not None:
                    reads.extend((read, current.parent) for read in current.defaults)
                for read, read_scope in reads:
                    owner = read_scope.resolve(read.id)
                    if owner is None:
                        continue
                    bindings = owner.bindings[read.id]
                    for binding in bindings:
                        if binding in scopes:
                            pending.append(scopes[binding])
                    if owner not in forbidden_scopes or all(binding in scopes for binding in bindings):
                        continue
                    if constant_binding(owner, read.id, set()):
                        continue
                    if (id(read) not in default_reads
                            and constant_binding(owner, read.id, set(), allow_series=True)):
                        captures[owner, read.id] = bindings[0]
                        continue
                    path = getattr(node, '_module_file_path', '<pyne>')
                    raise SyntaxError(
                        f'Exported function "{function.name}" cannot capture '
                        f'non-constant library global "{read.id}".',
                        (path, read.lineno, read.col_offset + 1, None),
                    )
        # Export definitions survive across bars. A scalar held only in a Python
        # closure would freeze at its first-bar value, so read its library slot.
        declarations: dict[int, ast.AnnAssign] = {}
        retained: set[tuple[_Scope, str]] = set()
        for key, binding in captures.items():
            if isinstance(binding, ast.AnnAssign):
                wrapper = binding.annotation
                if isinstance(wrapper, ast.Subscript):
                    wrapper = wrapper.value
                if _path(wrapper).split('.')[-1] in (
                        'Persistent', 'IBPersistent', 'PersistentSeries', 'IBPersistentSeries'):
                    continue
                annotation = (binding.annotation if _path(wrapper) == 'Series' else ast.Subscript(
                    value=ast.Name(id='Series', ctx=ast.Load()),
                    slice=binding.annotation, ctx=ast.Load()))
                target = binding.target
            else:
                annotation = ast.Name(id='Series', ctx=ast.Load())
                target, = binding.targets
            declarations[id(binding)] = ast.copy_location(ast.AnnAssign(
                target=target, annotation=annotation, value=binding.value, simple=1), binding)
            retained.add(key)
        reads = {id(read) for scope in scopes.values() for read in scope.reads
                 if (scope.resolve(read.id), read.id) in retained
                 and scope is not scope.resolve(read.id)}

        class RetainCaptures(ast_walk.NodeTransformer):
            def visit_Name(self, read: ast.Name) -> ast.AST:
                if id(read) in reads:
                    return ast.copy_location(ast.Subscript(
                        value=read, slice=ast.Constant(value=0), ctx=ast.Load()), read)
                return read

            def visit_Subscript(self, read: ast.Subscript) -> ast.AST:
                if id(read.value) in reads:
                    read.slice = self.visit(read.slice)
                    return read
                return self.generic_visit(read)

            def visit_Assign(self, binding: ast.Assign) -> ast.AST:
                return self.generic_visit(declarations.get(id(binding), binding))

            def visit_AnnAssign(self, binding: ast.AnnAssign) -> ast.AST:
                return self.generic_visit(declarations.get(id(binding), binding))

        node = RetainCaptures().visit(node)
        setattr(node, '_export_capture_deps', tuple((path, *fingerprint)
                                                   for path, fingerprint in sorted(enums.sources.items())))
        return node
