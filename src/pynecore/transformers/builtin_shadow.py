from typing import Any, cast
import ast
import importlib
import json
from pathlib import Path

# Pine namespaces whose pynecore module has a different name
_NAMESPACE_RENAMES = {
    'str': 'string',
}


class BuiltinShadowTransformer(ast.NodeTransformer):
    """
    Resolve workdir library imports whose alias shadows a built-in namespace.

    Pine resolves ``ta.x`` after ``import user/somelib/1 as ta`` against the
    library's exports first and falls back to the built-in ``ta.*`` namespace
    for everything else, so e.g. ``ta.valuewhen`` keeps working when the
    library does not export it. The compiled ``import lib.user.somelib.v1 as
    ta`` would route every access to the library module, so attribute reads
    the library cannot serve are rewritten here to the canonical built-in form
    (``lib.ta.valuewhen``), which the downstream transformers (import
    normalizer, module properties, isolation, series) handle natively.

    Which members the library serves is runtime knowledge: the library module
    is imported at transform time (the same pattern callee resolution uses in
    function isolation). Its ``__all__`` is the membership test — matching
    TV, where non-exported library names are not reachable through the alias —
    with a ``hasattr`` fallback for hand-written libraries without ``__all__``.
    If the library cannot be imported, the alias is left untouched and the
    script fails at its own import statement exactly as before.
    """

    def __init__(self):
        # Structure: module -> name -> {"type": "property"|"variable"}
        try:
            with open(Path(__file__).parent / "module_properties.json") as f:
                self.registry: dict[str, dict[str, Any]] = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Failed to load module properties config: {e}")

        # alias -> (library module, exported names or None, builtin namespace)
        self.aliases: dict[str, tuple[Any, frozenset[str] | None, str]] = {}

    def visit_Module(self, node: ast.Module) -> ast.Module:
        for stmt in node.body:
            if not isinstance(stmt, ast.Import):
                continue
            for alias in stmt.names:
                if not alias.asname or not alias.name.startswith('lib.'):
                    continue
                name = alias.asname
                # --strict compilation suffixes every global binding
                if name.endswith('__global__'):
                    name = name[:-len('__global__')]
                namespace = _NAMESPACE_RENAMES.get(name, name)
                if f'lib.{namespace}' not in self.registry:
                    continue
                try:
                    lib_module = importlib.import_module(alias.name)
                except Exception:  # noqa: BLE001 - the script's own import will report it
                    continue
                exported = getattr(lib_module, '__all__', None)
                members = frozenset(exported) if exported is not None else None
                self.aliases[alias.asname] = (lib_module, members, namespace)

        if not self.aliases:
            return node
        return cast(ast.Module, self.generic_visit(node))

    def _visit_scope(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> ast.AST:
        """Visit a function-like scope, masking aliases shadowed by parameters."""
        args = node.args
        params = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
        if args.vararg:
            params.add(args.vararg.arg)
        if args.kwarg:
            params.add(args.kwarg.arg)
        masked = {name: self.aliases.pop(name) for name in params & self.aliases.keys()}
        try:
            return self.generic_visit(node)
        finally:
            self.aliases.update(masked)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._visit_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._visit_scope(node)

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        return self._visit_scope(node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        node = cast(ast.Attribute, self.generic_visit(node))
        if not (isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Name)):
            return node
        entry = self.aliases.get(node.value.id)
        if entry is None:
            return node
        lib_module, members, namespace = entry

        # The library serves the name -> the access stays on the library
        if node.attr in members if members is not None else hasattr(lib_module, node.attr):
            return node

        # Built-in namespace fallback -> canonical lib.<namespace>.<attr> form;
        # the nested-key check covers sub-namespaces (e.g. strategy.commission)
        if (node.attr in self.registry[f'lib.{namespace}']
                or f'lib.{namespace}.{node.attr}' in self.registry):
            return ast.copy_location(
                ast.Attribute(
                    value=ast.Attribute(
                        value=ast.Name(id='lib', ctx=ast.Load()),
                        attr=namespace, ctx=ast.Load()),
                    attr=node.attr, ctx=node.ctx),
                node)

        # In neither -> leave it to fail at runtime, as before
        return node
