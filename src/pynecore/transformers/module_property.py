from typing import cast, Any
import ast
import json
from pathlib import Path

from . import ast_walk

__all__ = ['ModulePropertyTransformer', 'module_properties']

_module_properties: dict[str, dict[str, dict[str, Any]]] | None = None


def module_properties() -> dict[str, dict[str, dict[str, Any]]]:
    """
    The ``module_properties.json`` registry, read once per process.

    The document is shared by every caller and must be treated as read-only.

    :return: module -> name -> {"type": "property"|"variable"}
    :raises RuntimeError: When the registry cannot be read
    """
    global _module_properties
    loaded = _module_properties
    if loaded is None:
        try:
            with open(Path(__file__).parent / "module_properties.json") as f:
                loaded = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Failed to load module properties config: {e}")
        _module_properties = loaded
    return loaded


class ModulePropertyTransformer(ast_walk.NodeTransformer):
    """
    Transform lib.xxx references based on the generated module_properties.json registry.

    - ``property`` entries (Pine names that are values): bare reads become calls
      (``ta.tr`` -> ``ta.tr()``); explicit calls are left untouched.
    - ``variable`` entries: left as plain attribute reads.
    - Function-and-namespace modules (``plot``, ``dayofweek``, ...): calls and promoted
      bare reads are routed to the module's self-named function
      (``plot(x)`` -> ``plot.plot(x)``, bare ``dayofweek`` -> ``dayofweek.dayofweek()``).
    - Unknown names on known pynecore.lib modules raise at transform time — the
      registry is exhaustive, so this catches typos and a stale registry early.
    - Unknown module paths (user ``lib.*`` workdir libraries) and ``_``-prefixed
      names are plain attribute reads.
    """

    def __init__(self):
        # Structure: module -> name -> {"type": "property"|"variable"}
        self.module_info: dict[str, dict[str, dict[str, Any]]] = module_properties()

        # The callee of every call met so far and every node of every type
        # annotation, both by identity: what the visit of an attribute needs to
        # know about the place it stands in, without a parent link per node
        self._callees: set[ast.AST] = set()
        self._annotation_nodes: set[ast.AST] = set()

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self._callees.add(node.func)
        return self.generic_visit(node)

    def _note_annotation(self, annotation: ast.AST | None) -> None:
        if annotation is not None:
            self._annotation_nodes.update(ast_walk.walk(annotation))

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        self._note_annotation(node.annotation)
        return self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        self._note_annotation(node.annotation)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.AST:
        self._note_annotation(node.returns)
        return self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        """Process attribute access, but skip if inside type annotations."""
        # Intermediate modules of the chain (attributes whose parent is also an
        # Attribute) are not the topmost attribute: only what hangs below the
        # chain is visited, and they stay as they are
        inner = node
        while isinstance(inner.value, ast.Attribute):
            inner = inner.value
        inner.value = self.visit(inner.value)

        # If this node has already been processed, or the chain does not start with lib..., skip
        if hasattr(node, '_processed') or not self._is_lib_reference(node):
            return node

        # Skip if inside type annotations
        if node in self._annotation_nodes:
            return node

        # Now it's the topmost attribute (e.g., ...data_window)
        # Check the full module path and the final attribute
        module_path, name = self._get_module_info(node)
        if not module_path or not name:
            return node

        full_path = f"{module_path}.{name}"

        # Call site — explicit calls stay as they are, except when the callee is a
        # function-and-namespace module (its registry entry contains a self-named
        # function): ``lib.plot(...)`` routes to ``lib.plot.plot(...)``
        if node in self._callees:
            inner_attrs = self.module_info.get(full_path)
            if inner_attrs is not None and name in inner_attrs:
                result: ast.expr = ast.Attribute(value=node, attr=name, ctx=ast.Load())
                setattr(result, "_processed", True)
                return result
            return node

        module_attrs = self.module_info.get(module_path)
        if module_attrs is None:
            # Unknown module path: a user workdir library or a class-attribute chain —
            # plain Python attribute access
            return node

        attr_info = module_attrs.get(name)
        if attr_info is not None:
            if attr_info["type"] == "property":
                if full_path == "lib.na":
                    # Bare ``na`` is a constant value (the interned typeless NA):
                    # load it directly instead of emitting a per-bar ``lib.na()``
                    # call. Explicit ``na(x)`` predicate calls are untouched above.
                    result = ast.Attribute(value=ast.Name(id='lib', ctx=ast.Load()),
                                           attr='_na_none', ctx=ast.Load())
                    setattr(result, "_processed", True)
                    return result
                inner_attrs = self.module_info.get(full_path)
                if inner_attrs is not None and name in inner_attrs:
                    # Promoted self-named property of a function-and-namespace module:
                    # bare ``dayofweek`` -> ``lib.dayofweek.dayofweek()``
                    func: ast.expr = ast.Attribute(value=self._copy_node(node), attr=name,
                                                   ctx=ast.Load())
                else:
                    func = self._copy_node(node)
                result = ast.Call(func=func, args=[], keywords=[])
            else:
                result = node

        # Submodule reference — leave as is
        elif full_path in self.module_info:
            result = node

        # Internal names are never module properties — plain attribute access
        elif name.startswith('_'):
            result = node

        else:
            raise SyntaxError(
                f"unknown attribute '{name}' on module '{module_path}' (line {node.lineno}); "
                f"if this is a new pynecore.lib name, regenerate module_properties.json "
                f"with scripts/module_property_collector.py"
            )

        setattr(result, "_processed", True)
        return result

    @staticmethod
    def _is_lib_reference(node: ast.Attribute) -> bool:
        """Check if the attribute chain starts with 'lib'."""
        current = node
        while isinstance(current, ast.Attribute):
            current = current.value
        return isinstance(current, ast.Name) and current.id == 'lib'

    @staticmethod
    def _get_module_info(node: ast.Attribute) -> tuple[str | None, str | None]:
        """
        Gather the full chain of attributes until we reach 'lib',
        then split into (module_path, final_attribute).
        Example: lib.display.data_window -> (lib.display, data_window)
        """
        attrs = []
        current = node
        while isinstance(current, ast.Attribute):
            attrs.append(current.attr)
            current = current.value

        if isinstance(current, ast.Name) and current.id == 'lib':
            attrs.append('lib')
            attrs.reverse()
            # Example: ['lib', 'display', 'data_window']
            if len(attrs) < 2:
                return None, None
            module_path = '.'.join(attrs[:-1])  # 'lib.display'
            final_attr = attrs[-1]  # 'data_window'
            return module_path, final_attr
        return None, None

    @staticmethod
    def _copy_node(node: ast.AST) -> ast.expr:
        """Create a shallow copy of an AST node (Attribute or Name)."""
        if isinstance(node, ast.Name):
            return cast(ast.expr, ast.Name(id=node.id, ctx=node.ctx))
        elif isinstance(node, ast.Attribute):
            value = ModulePropertyTransformer._copy_node(cast(ast.AST, node.value))
            return cast(ast.expr, ast.Attribute(value=value, attr=node.attr, ctx=node.ctx))
        return cast(ast.expr, node)
