#!/usr/bin/env python3
from typing import cast, Any
import ast
import json
from pathlib import Path


class ModulePropertyCollector:
    """
    Collect module properties and variables from all files under pynecore/lib
    """

    def __init__(self):
        self.project_root = self._find_project_root()
        self.lib_path = self.project_root / 'pynecore' / 'lib'
        self.json_path = self.project_root / 'pynecore' / 'transformers' / 'module_properties.json'
        self.module_info: dict[str, dict[str, dict[str, Any]]] = {}

    @staticmethod
    def _find_project_root() -> Path:
        """Find project root by looking for pyproject.toml"""
        current = Path.cwd()
        while current != current.parent:
            if (current / 'pyproject.toml').exists():
                return current / 'src'
            current = current.parent
        raise FileNotFoundError("Could not find project root (pyproject.toml)")

    def process_file(self, file_path: Path) -> None:
        """Process a single Python file"""
        # Get module path
        rel_path = file_path.relative_to(self.project_root / 'pynecore')
        # Convert path to module path, removing .py extension from the last part
        parts = []
        for part in rel_path.parts[:-1]:  # Process directories
            parts.append(part)
        parts.append(rel_path.stem)  # Last part without .py extension
        module_path = '.'.join(parts)

        # Parse file
        with open(file_path) as f:
            try:
                tree = ast.parse(f.read(), filename=str(file_path))
            except SyntaxError as e:
                print(f"Syntax error in {file_path}: {e}")
                return

        # Process AST
        transformer = ModulePropertyCollectorTransformer(module_path)
        transformer.visit(tree)

        # Update module info
        if transformer.current_module_info:
            self.module_info[module_path.replace('.__init__', '')] = transformer.current_module_info
        else:
            self.module_info[module_path.replace('.__init__', '')] = {}

    def process_all_files(self) -> None:
        """Process all Python files under lib directory"""
        for file_path in self.lib_path.rglob('*.py'):
            if file_path.name.startswith('_') and file_path.name != '__init__.py':
                continue
            print(f"Processing {file_path}")
            self.process_file(file_path)

        # na is a special case force it to be a property
        self.module_info['lib']['na'] = {
            "type": "property",
        }

        # Promote submodule self-named properties to parent module.
        # E.g., lib.strategy.opentrades has property "opentrades" →
        #   add "opentrades" as property under lib.strategy too,
        #   so that strategy.opentrades is transformed to strategy.opentrades().
        # Also handles lib.dayofweek → lib for single-level submodules.
        for module_path, attrs in list(self.module_info.items()):
            parts = module_path.split('.')
            if len(parts) >= 2:  # lib.X or deeper
                submodule_name = parts[-1]
                parent_path = '.'.join(parts[:-1])
                if (submodule_name in attrs
                        and attrs[submodule_name].get("type") == "property"
                        and parent_path in self.module_info):
                    self.module_info[parent_path][submodule_name] = attrs[submodule_name]

        # Save results
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.json_path, 'w') as f:
            json.dump(self.module_info, f, indent=2, sort_keys=True)  # noqa


class ModulePropertyCollectorTransformer(ast.NodeTransformer):
    """AST transformer to collect module properties and variables"""

    def __init__(self, module_path: str):
        self.module_path = module_path
        self.current_module_info: dict[str, dict[str, Any]] = {}
        self.in_class = False

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        """Skip class definitions"""
        old_in_class = self.in_class
        self.in_class = True
        self.generic_visit(node)
        self.in_class = old_in_class
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """Process function definitions for @module_property"""
        if self.in_class:
            return node

        # Check if it has @module_property
        has_module_property = any(
            isinstance(d, ast.Name) and d.id == 'module_property'
            for d in node.decorator_list
        )

        if has_module_property:
            self.current_module_info[node.name] = {
                "type": "property",
            }

        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        """Process any module level assignments"""
        if self.in_class:
            return node

        # Only handle single target assigns at module level
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return node

        # Check if we're at module level
        parent = getattr(node, 'parent', None)
        if not isinstance(parent, ast.Module):
            return node

        name = cast(ast.Name, node.targets[0]).id
        # Skip dunders and uppercase constants
        if not name.startswith('_') and not name.isupper():
            self.current_module_info[name] = {
                "type": "variable",
            }

        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        """Process annotated assignments at module level"""
        if self.in_class:
            return node

        # Only handle Name targets at module level
        if not isinstance(node.target, ast.Name):
            return node

        parent = getattr(node, 'parent', None)
        if not isinstance(parent, ast.Module):
            return node

        name = node.target.id
        # Skip dunders and uppercase constants
        if not name.startswith('_') and not name.isupper() and not name[0].isupper():
            self.current_module_info[name] = {
                "type": "variable",
            }

        return self.generic_visit(node)

    def visit_Module(self, node: ast.Module) -> ast.AST:
        """Set parent for all nodes to track module level"""
        for child in ast.iter_child_nodes(node):
            setattr(child, 'parent', node)
        return self.generic_visit(node)


if __name__ == '__main__':
    collector = ModulePropertyCollector()
    collector.process_all_files()
    print(f"Results saved to {collector.json_path}")
