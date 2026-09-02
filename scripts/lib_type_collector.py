#!/usr/bin/env python3
"""
Generate ``pynecore/transformers/lib_types.json`` from the lib source tree.

The Pine type inference needs to know what every lib name evaluates to before
the program runs: ``ta.sma(...)`` is a float, ``input.int(...)`` an int,
``bar_index`` an int. That is exactly what the return annotations under
``pynecore/lib`` already say, so the registry is EXTRACTED rather than
maintained by hand -- and, being generated, it is diffable and testable.

The registry is deliberately mechanical. Where TradingView's measured type
differs from what the Python annotation says (``math.round`` is int-typed with
one argument but the lib annotates it ``PyneFloat``), the measurement lives in
``transformers/pine_type_rules.py::LIB_TYPE_OVERRIDES`` and wins at inference
time. Keeping the two apart is what lets this script stay a pure extractor.

The JSON is read at transform time instead of importing the lib: importing it
while the lib itself is being transformed would be circular (see the lazy
import pattern in ``transformers/const_fold.py``).

Usage:
    python3 scripts/lib_type_collector.py
"""
import ast
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from pynecore.transformers.pine_type_rules import (  # noqa: E402
    annotation_takes_none, annotation_type, constant_type, NONE_DEFAULT, TYPELESS,
    UNKNOWN, VOID,
)

#: Registry format version. Bump whenever the shape below changes; the
#: consumers (the inference engine, and the PyneAOT front end) pin it.
SCHEMA_VERSION = 4


class LibTypeCollector:
    """
    Collect the Pine result type of every public name under ``pynecore/lib``.
    """

    def __init__(self, project_src: Path | None = None):
        self.project_root = project_src if project_src is not None else self._find_project_root()
        self.lib_path = self.project_root / 'pynecore' / 'lib'
        self.json_path = self.project_root / 'pynecore' / 'transformers' / 'lib_types.json'

    @staticmethod
    def _find_project_root() -> Path:
        """Find the project root by looking for pyproject.toml"""
        current = Path.cwd()
        while current != current.parent:
            if (current / 'pyproject.toml').exists():
                return current / 'src'
            current = current.parent
        raise FileNotFoundError("Could not find project root (pyproject.toml)")

    def collect(self) -> dict[str, Any]:
        """
        Build the registry from the lib source tree (no file output).

        Private modules are parsed too, but only reachable through the public
        module that re-exports their names (``lib/math.py`` gets ``sum`` and
        ``random`` from ``lib/_math_stateful.py``), so the registry lists the
        names as a SCRIPT spells them.

        :return: The registry mapping, ready to be serialized
        """
        per_module: dict[str, dict[str, Any]] = {}
        reexports: dict[str, list[tuple[str, str, str]]] = {}
        for file_path in sorted(self.lib_path.rglob('*.py')):
            prefix = self._module_prefix(file_path)
            tree = ast.parse(file_path.read_text(), filename=str(file_path))
            per_module[prefix] = collect_module_types(tree)
            reexports[prefix] = _sibling_reexports(tree, prefix)

        # Names re-exported from a private sibling belong to the public module
        for prefix, imports in reexports.items():
            for module, original, alias in imports:
                entry = per_module.get(module, {}).get(original)
                if entry is not None:
                    per_module[prefix].setdefault(alias, entry)

        names: dict[str, Any] = {}
        for prefix, entries in per_module.items():
            if prefix.rpartition('.')[2].startswith('_'):
                continue
            for name, entry in entries.items():
                names[f'{prefix}.{name}' if prefix else name] = entry

        # A module whose own name it also defines is spelled without the
        # repetition by a script: ``lib/plot.py::plot`` is written ``plot(...)``
        for prefix, entries in per_module.items():
            if not prefix or prefix.rpartition('.')[2].startswith('_'):
                continue
            leaf = prefix.rpartition('.')[2]
            parent = prefix.rpartition('.')[0]
            if leaf in entries:
                names[f'{parent}.{leaf}' if parent else leaf] = entries[leaf]

        return {'v': SCHEMA_VERSION, 'names': names}

    def _module_prefix(self, file_path: Path) -> str:
        """Dotted lib path of a module, as a script spells it (``''`` for lib itself)."""
        rel = file_path.relative_to(self.lib_path)
        parts = list(rel.parts[:-1])
        if rel.stem != '__init__':
            parts.append(rel.stem)
        return '.'.join(parts)

    def write(self) -> None:
        """Collect the registry and write it next to the transformers"""
        registry = self.collect()
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.json_path, 'w') as f:
            json.dump(registry, f, indent=1, sort_keys=True)
            f.write('\n')


def _sibling_reexports(tree: ast.Module, prefix: str) -> list[tuple[str, str, str]]:
    """
    Public names a module pulls in from a sibling lib module.

    Only single-level relative imports count (``from ._math_stateful import
    sum``): those are the re-export idiom lib uses to keep a ``@pyne lib``
    machine out of an untransformed host module.

    :param tree: Parsed module AST
    :param prefix: Dotted lib path of the module being scanned
    :return: ``(source module prefix, original name, exported name)`` triples
    """
    package = prefix.rpartition('.')[0] if prefix else ''
    out: list[tuple[str, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level != 1 or not node.module:
            continue
        source = f'{package}.{node.module}' if package else node.module
        for alias in node.names:
            if alias.name == '*' or (alias.asname or alias.name).startswith('_'):
                continue
            out.append((source, alias.name, alias.asname or alias.name))
    return out


def _is_type_checking(test: ast.expr) -> bool:
    """Whether an ``if`` test is the TYPE_CHECKING guard."""
    return ((isinstance(test, ast.Name) and test.id == 'TYPE_CHECKING')
            or (isinstance(test, ast.Attribute) and test.attr == 'TYPE_CHECKING'))


def _decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Bare names of a function's decorators, however they are spelled."""
    names = set()
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
    return names


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    """
    One implementation's Pine signature.

    The parameter types are what an overload group is resolved on, so they are
    recorded positionally, in declaration order; ``params`` is empty for a
    zero-argument function and for a module property.

    The parameter NAMES ride along in the same order, because a keyword
    spelling must not change a call's Pine type: ``math.abs(number=d)`` is the
    same int-preserving call as ``math.abs(d)``, and the inference can only
    see that by binding the keyword back to its declared position.

    The DEFAULTS' types ride along too, aligned to the last parameters the way
    Python aligns the defaults themselves. The runtime selector binds an
    omitted argument to its default and type-checks it with the rest, so an
    implementation a call under-fills is selected on those types as much as on
    the ones the call spells out.

    A literal ``None`` default is the one whose acceptance the Pine type
    character cannot express -- ``int`` and ``int | None`` are both int-typed,
    and only the second takes it -- so a parallel ``default_none_ok`` records
    what the annotation answers, and only where such a default exists.

    :param node: The function definition
    :return: ``{'ret': .., 'params': [..], 'names': [..], 'defaults': <count>,
              'default_ty': [..], 'default_none_ok': [..]}``
    """
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    signature: dict[str, Any] = {
        'ret': annotation_type(node.returns),
        'params': [annotation_type(a.annotation) for a in positional],
        'names': [a.arg for a in positional],
        'defaults': len(args.defaults),
    }
    if args.defaults:
        default_ty = [_default_type(d) for d in args.defaults]
        signature['default_ty'] = default_ty
        if NONE_DEFAULT in default_ty:
            signature['default_none_ok'] = [
                annotation_takes_none(p.annotation)
                for p in positional[len(positional) - len(args.defaults):]]
    if args.vararg is not None:
        # ``math.max(*numbers)`` takes any arity; the element type is what the
        # overload is chosen on
        signature['vararg'] = annotation_type(args.vararg.annotation)
    return signature


def _default_type(node: ast.expr) -> str:
    """
    Pine type of one parameter default, as the overload selector reads it.

    Only what a pure extractor can decide: a literal carries its own type,
    ``na`` carries none at all, and a literal ``None`` carries its own
    character, because whether it fits is a question about the ANNOTATION and
    not about the annotation's Pine type. Anything the lib computes
    (``_color.blue``, ``_xloc.bar_index``) stays UNKNOWN, which is what makes
    the inference decline to pick rather than pick wrong.

    :param node: The default expression
    :return: Its type character, ``NONE_DEFAULT``, ``TYPELESS``, or UNKNOWN
    """
    if isinstance(node, ast.Constant):
        return NONE_DEFAULT if node.value is None else constant_type(node.value)
    if isinstance(node, ast.Name) and node.id == 'na':
        return TYPELESS
    if isinstance(node, ast.Attribute) and node.attr == 'na':
        return TYPELESS
    return UNKNOWN


def collect_module_types(tree: ast.Module) -> dict[str, Any]:
    """
    Collect the public names of one lib module with their Pine result types.

    A ``@module_property`` function is recorded by the type it RETURNS, since a
    script reads it as a value (``bar_index`` is an int, not a callable). A
    plain function is recorded as a callable with its signature; an
    ``@overload`` group keeps every implementation, in source order, which is
    the order the runtime dispatcher tries them in.

    Module-level constants get a type only when they carry an annotation --
    an unannotated ``red = Color(...)`` would need the value's class, which
    this extractor deliberately does not evaluate.

    :param tree: Parsed module AST
    :return: name -> entry mapping
    """
    info: dict[str, Any] = {}

    def record_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if node.name.startswith('_'):
            return
        decorators = _decorator_names(node)
        signature = _signature(node)
        if {'module_property', 'module_function_property'} & decorators:
            # Read as a value: its type IS the return type
            info[node.name] = {'kind': 'value', 'ty': signature['ret']}
            return
        if 'overload' in decorators:
            entry = info.get(node.name)
            if entry is not None and entry.get('kind') == 'overloads':
                entry['impls'].append(signature)
            else:
                info[node.name] = {'kind': 'overloads', 'impls': [signature]}
            return
        # A non-decorated definition after an overload group is the dispatch
        # target the group already covers; do not let it overwrite the group
        if info.get(node.name, {}).get('kind') == 'overloads':
            return
        info[node.name] = {'kind': 'function', **signature}

    def record_annotated(target: ast.expr, annotation: ast.expr) -> None:
        if not isinstance(target, ast.Name) or target.id.startswith('_'):
            return
        ty = annotation_type(annotation)
        if ty not in (UNKNOWN, VOID):
            info[target.id] = {'kind': 'value', 'ty': ty}

    def walk(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                record_function(node)
            elif isinstance(node, ast.AnnAssign):
                record_annotated(node.target, node.annotation)
            elif isinstance(node, ast.If):
                if not _is_type_checking(node.test):
                    walk(node.body)
                    walk(node.orelse)
            elif isinstance(node, ast.Try):
                walk(node.body)
                for handler in node.handlers:
                    walk(handler.body)
                walk(node.orelse)

    walk(tree.body)
    return info


if __name__ == '__main__':
    LibTypeCollector().write()
