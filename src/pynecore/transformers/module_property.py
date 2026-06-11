from typing import cast, Any
import ast
import json
from pathlib import Path


class ModulePropertyTransformer(ast.NodeTransformer):
    """
    Transform lib.xxx references based on JSON configuration.
    If module+name exists in config:
        - Transform properties to function calls
        - Leave variables as is
    If not in config:
        - Use callable() check
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

        # Function call - if the parent is a Call, and this is being called
        if isinstance(parent, ast.Call) and parent.func == node:
            return node

        # If this node has already been processed, or the chain does not start with lib..., skip
        if hasattr(node, '_processed') or not self._is_lib_reference(node):
            return node

        # Now it's the topmost attribute (e.g., ...data_window)
        # Check the full module path and the final attribute
        module_path, name = self._get_module_info(node)
        if not module_path or not name:
            return node

        # Check the config for this attribute
        module_attrs = self.module_info.get(module_path, {})
        attr_info = module_attrs.get(name)

        # If the attribute is in the config: property -> () call, or variable -> leave as is
        if attr_info:
            if attr_info["type"] == "property":
                result = ast.Call(
                    func=self._copy_node(node),
                    args=[],
                    keywords=[]
                )
            else:
                result = node

        # Fall back to __module_property__ check
        elif f"{module_path}.{name}" not in self.module_info:  # Skip if it is a module
            # Check for __module_property__ attribute
            result = ast.IfExp(
                test=ast.Call(
                    func=ast.Name(id='hasattr', ctx=ast.Load()),
                    args=[
                        self._copy_node(node),
                        ast.Constant(value='__module_property__')
                    ],
                    keywords=[]
                ),
                body=ast.Call(
                    func=self._copy_node(node),
                    args=[],
                    keywords=[]
                ),
                orelse=self._copy_node(node)
            )
        # Leave module attributes as is
        else:
            result = node

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
