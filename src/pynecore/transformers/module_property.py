from typing import cast, Any
import ast
import json
from pathlib import Path


class ModulePropertyTransformer(ast.NodeTransformer):
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
        self.module_info: dict[str, dict[str, dict[str, Any]]] = {}

        # Load config
        try:
            with open(Path(__file__).parent / "module_properties.json") as f:
                self.module_info = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Failed to load module properties config: {e}")

    def visit(self, node: ast.AST) -> ast.AST:
        """
        Override the generic visit method to set .parent on each child node
        for chain detection.
        """
        # Set parent on children
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                setattr(value, "parent", node)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        setattr(item, "parent", node)

        return super().visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        """Process attribute access, but skip if inside type annotations."""
        node = cast(ast.Attribute, self.generic_visit(node))

        # Skip if inside type annotations
        if self._is_in_type_annotation(node):
            return node

        # Retrieve the AST parent node
        parent = getattr(node, 'parent', None)

        # Intermediate module - if the parent is also an Attribute, this is not the topmost attribute
        if isinstance(parent, ast.Attribute):
            return node

        # If this node has already been processed, or the chain does not start with lib..., skip
        if hasattr(node, '_processed') or not self._is_lib_reference(node):
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
        if isinstance(parent, ast.Call) and parent.func == node:
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

    def _is_in_type_annotation(self, node: ast.Attribute) -> bool:
        """Check if the node is inside a type annotation."""
        current = node
        while hasattr(current, 'parent'):
            parent = getattr(current, 'parent', None)

            # Check if we're in an annotated assignment's annotation
            if (isinstance(parent, ast.AnnAssign) and parent.annotation and
                    self._is_node_in_subtree(node, cast(ast.AST, parent.annotation))):
                return True

            # Check if we're in a function argument's annotation
            if (isinstance(parent, ast.arg) and parent.annotation and
                    self._is_node_in_subtree(node, cast(ast.AST, parent.annotation))):
                return True

            # Check if we're in a function return annotation
            if (isinstance(parent, ast.FunctionDef) and parent.returns and
                    self._is_node_in_subtree(node, parent.returns)):
                return True

            # Check if we're in an async function return annotation
            if (isinstance(parent, ast.AsyncFunctionDef) and parent.returns and
                    self._is_node_in_subtree(node, parent.returns)):
                return True

            current = parent

        return False

    @staticmethod
    def _is_node_in_subtree(node: ast.AST, subtree: ast.AST | None) -> bool:
        """Check if a node is contained within a subtree."""
        if subtree is None:
            return False

        if node is subtree:
            return True

        # Recursively check all child nodes
        for child in ast.walk(subtree):
            if child is node:
                return True

        return False

    @staticmethod
    def _copy_node(node: ast.AST) -> ast.expr:
        """Create a shallow copy of an AST node (Attribute or Name)."""
        if isinstance(node, ast.Name):
            return cast(ast.expr, ast.Name(id=node.id, ctx=node.ctx))
        elif isinstance(node, ast.Attribute):
            value = ModulePropertyTransformer._copy_node(cast(ast.AST, node.value))
            return cast(ast.expr, ast.Attribute(value=value, attr=node.attr, ctx=node.ctx))
        return cast(ast.expr, node)
