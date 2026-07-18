#!/usr/bin/env python3
"""
Generate ``pynecore/transformers/module_properties.json`` from the lib source tree.

The registry is the single source of truth for the ModulePropertyTransformer: it must
list EVERY public module-level name under ``pynecore/lib`` — ``@module_property``
functions as ``property``, everything else (plain functions, classes, constants,
imported names) as ``variable``. Names missing from the registry make the transformer
raise at transform time, so rerun this script whenever lib gains or loses a name
(the test suite verifies the committed JSON is current).
"""
from typing import Any
import ast
import json
from pathlib import Path


class ModulePropertyCollector:
    """
    Collect module properties and variables from all files under pynecore/lib
    """

    def __init__(self, project_src: Path | None = None):
        self.project_root = project_src if project_src is not None else self._find_project_root()
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

        # Collect every public module-level name
        self.module_info[module_path.replace('.__init__', '')] = collect_module_names(tree)

    def collect(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Collect the full registry from the lib source tree (no file output)."""
        self.module_info = {}
        for file_path in sorted(self.lib_path.rglob('*.py')):
            if file_path.name.startswith('_') and file_path.name != '__init__.py':
                continue
            self.process_file(file_path)

        # na is a special case force it to be a property
        self.module_info['lib']['na'] = {
            "type": "property",
        }

        # Drop plain-variable entries that shadow a submodule of the same name
        # (e.g. ``from . import plot`` in lib/__init__.py) — the transformer treats
        # those paths as module references.
        for module_path, attrs in self.module_info.items():
            for name in [n for n, info in attrs.items()
                         if info["type"] == "variable"
                         and f"{module_path}.{n}" in self.module_info]:
                del attrs[name]

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

        return self.module_info

    def process_all_files(self) -> None:
        """Collect the registry and write it next to the transformers"""
        self.collect()

        # Save results
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.json_path, 'w') as f:
            json.dump(self.module_info, f, indent=2, sort_keys=True)  # noqa
            f.write('\n')


def _is_type_checking(test: ast.expr) -> bool:
    """Whether an ``if`` test is the TYPE_CHECKING guard."""
    return ((isinstance(test, ast.Name) and test.id == 'TYPE_CHECKING')
            or (isinstance(test, ast.Attribute) and test.attr == 'TYPE_CHECKING'))


# Typing machinery must not become accepted public lib names: imports from these
# modules (and TypeVar assignments) are implementation detail, not Pine API.
_SKIP_IMPORT_MODULES = {'typing', 'typing_extensions'}
_SKIP_IMPORT_MODULE_SUFFIXES = ('core.module_property', 'core.overload')


def _is_typevar_call(value: ast.expr) -> bool:
    """Whether an assignment value is a ``TypeVar(...)`` call."""
    return (isinstance(value, ast.Call)
            and ((isinstance(value.func, ast.Name) and value.func.id == 'TypeVar')
                 or (isinstance(value.func, ast.Attribute) and value.func.attr == 'TypeVar')))


def collect_module_names(tree: ast.Module) -> dict[str, dict[str, Any]]:
    """Collect every public module-level name of a parsed lib module.

    ``@module_property`` functions are recorded as ``property``; plain functions,
    classes, assignments (including conditional ones) and imported names as
    ``variable``. ``_``-prefixed names, ``TYPE_CHECKING`` blocks and typing
    machinery (``typing`` imports, ``TypeVar`` assignments, the decorator helper
    modules) are skipped.

    :param tree: Parsed module AST.
    :return: name -> {"type": "property"|"variable"} mapping.
    """
    info: dict[str, dict[str, Any]] = {}

    def record(name: str, kind: str) -> None:
        if name.startswith('_'):
            return
        # A property entry always wins over a variable one
        if kind == 'variable' and info.get(name, {}).get('type') == 'property':
            return
        info[name] = {"type": kind}

    def record_target(target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            record(target.id, 'variable')
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                record_target(elt)

    def walk(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_property = any(
                    isinstance(d, ast.Name) and d.id in ('module_property', 'module_function_property')
                    for d in node.decorator_list
                )
                record(node.name, 'property' if is_property else 'variable')
            elif isinstance(node, ast.ClassDef):
                record(node.name, 'variable')
            elif isinstance(node, ast.Assign):
                if _is_typevar_call(node.value):
                    continue
                for target in node.targets:
                    record_target(target)
            elif isinstance(node, ast.AnnAssign):
                # Annotation-only declarations count too: the runner injects
                # their values at run time (e.g. syminfo.opening_hours)
                record_target(node.target)
            elif isinstance(node, ast.AugAssign):
                record_target(node.target)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    record(alias.asname or alias.name.split('.')[0], 'variable')
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                if (module in _SKIP_IMPORT_MODULES
                        or module.endswith(_SKIP_IMPORT_MODULE_SUFFIXES)):
                    continue
                for alias in node.names:
                    if alias.name == '*':
                        continue
                    record(alias.asname or alias.name, 'variable')
            elif isinstance(node, ast.If):
                if not _is_type_checking(node.test):
                    walk(node.body)
                    walk(node.orelse)
            elif isinstance(node, ast.Try):
                walk(node.body)
                for handler in node.handlers:
                    walk(handler.body)
                walk(node.orelse)
                walk(node.finalbody)
            elif isinstance(node, ast.With):
                walk(node.body)

    walk(tree.body)
    return info


if __name__ == '__main__':
    collector = ModulePropertyCollector()
    collector.process_all_files()
    print(f"Results saved to {collector.json_path}")
