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
    annotation_takes_none, annotation_type, builtin_class_id, constant_type, join,
    object_ty, ANNOTATION_TYPES, BOOL, FLOAT, INT, NONE_DEFAULT, OBJECT, STR, TYPELESS,
    UNKNOWN, VOID,
)

#: Registry format version. Bump whenever the shape below changes; the
#: consumers (the inference engine, and the PyneAOT front end) pin it.
SCHEMA_VERSION = 8


class LibTypeCollector:
    """
    Collect the Pine result type of every public name under ``pynecore/lib``.
    """

    def __init__(self, project_src: Path | None = None):
        self.project_root = project_src if project_src is not None else self._find_project_root()
        self.lib_path = self.project_root / 'pynecore' / 'lib'
        #: The classes the lib publishes as Pine objects are declared here, not
        #: under ``lib/`` -- a lib module imports them out of this package --
        #: so their FIELDS have to be read from it
        self.types_path = self.project_root / 'pynecore' / 'types'
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
        packages: dict[str, str] = {}
        for file_path in sorted(self.lib_path.rglob('*.py')):
            prefix = self._module_prefix(file_path)
            tree = ast.parse(file_path.read_text(), filename=str(file_path))
            per_module[prefix] = collect_module_types(tree)
            reexports[prefix] = _sibling_reexports(tree, prefix)
            packages[prefix] = prefix if file_path.stem == '__init__' \
                else prefix.rpartition('.')[0]

        # Names re-exported from a private sibling belong to the public module
        for prefix, imports in reexports.items():
            for module, original, alias in imports:
                entry = per_module.get(module, {}).get(original)
                if entry is not None:
                    per_module[prefix].setdefault(alias, entry)

        names: dict[str, Any] = {}
        aliases: list[tuple[str, str]] = []
        for prefix, entries in per_module.items():
            if prefix.rpartition('.')[2].startswith('_'):
                continue
            for name, entry in entries.items():
                key = f'{prefix}.{name}' if prefix else name
                names[key] = entry
                if entry['kind'] == 'alias':
                    package = packages[prefix]
                    aliases.append(
                        (key, f'{package}.{entry["to"]}' if package else entry['to']))

        # A module whose own name it also defines is spelled without the
        # repetition by a script: ``lib/plot.py::plot`` is written ``plot(...)``
        for prefix, entries in per_module.items():
            if not prefix or prefix.rpartition('.')[2].startswith('_'):
                continue
            leaf = prefix.rpartition('.')[2]
            parent = prefix.rpartition('.')[0]
            if leaf in entries:
                names[f'{parent}.{leaf}' if parent else leaf] = entries[leaf]

        _resolve_aliases(names, aliases)
        return {'v': SCHEMA_VERSION, 'names': names, 'classes': self._collect_fields(),
                'scalar_classes': self._collect_scalars()}

    def _collect_fields(self) -> dict[str, dict[str, str]]:
        """
        The fields of every class the lib publishes as a Pine object.

        A ``chart.point`` KNOWS its class, so ``p.price`` is a float -- but
        only if the class says what it holds, and a builtin class says it in
        the type package rather than in an interface. The annotations are
        already there (``price: PyneFloat``), so they are extracted the same
        way the returns are.

        :return: Class name -> field name -> type
        """
        declared: dict[str, ast.ClassDef] = {}
        for file_path in sorted(self.types_path.rglob('*.py')):
            tree = ast.parse(file_path.read_text(), filename=str(file_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    declared.setdefault(node.name, node)
        classes = {name: builtin_class_id(name) for name in declared}
        out: dict[str, dict[str, str]] = {}
        for name, node in declared.items():
            fields = _class_fields(node, classes)
            if fields:
                out[name] = fields
        return out

    def _collect_scalars(self) -> dict[str, str]:
        """
        The classes of the type package that ARE a scalar.

        ``Format`` derives from ``StrLiteral``, which is a ``str``: a
        ``format.percent`` is a string wherever a string is taken, and the
        registry says so by naming the scalar each such class is.

        :return: Class name -> scalar type character
        """
        bases: dict[str, list[str]] = {}
        for file_path in sorted(self.types_path.rglob('*.py')):
            tree = ast.parse(file_path.read_text(), filename=str(file_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    bases.setdefault(node.name, [
                        base.id if isinstance(base, ast.Name) else base.attr
                        for base in node.bases if isinstance(base, (ast.Name, ast.Attribute))])
        scalars: dict[str, str] = {'str': STR, 'StrLiteral': STR, 'int': INT,
                                   'float': FLOAT, 'bool': BOOL}
        settled = False
        while not settled:
            settled = True
            for name, parents in bases.items():
                if name in scalars:
                    continue
                found = next((scalars[parent] for parent in parents if parent in scalars), None)
                if found is not None:
                    scalars[name] = found
                    settled = False
        return {name: scalar for name, scalar in sorted(scalars.items()) if name in bases}

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


def _resolve_aliases(names: dict[str, Any], aliases: list[tuple[str, str]]) -> None:
    """
    Replace every alias entry by the entry it names, and drop the rest.

    A namespace re-publishes a constant of another one by plain assignment
    (``lib/strategy/__init__.py`` spells ``long = direction.long``), and a
    script reads the alias, not the original -- so the registry has to carry
    the same type under both spellings. The target is resolved against the
    aliasing module's own PACKAGE, which is how the source spells it.

    An alias this cannot resolve is removed rather than left in: the ``alias``
    kind is an internal note between the two halves of the collection, and no
    consumer of the registry knows it.

    :param names: The flattened registry, edited in place
    :param aliases: (full name, resolved target key) per alias entry
    """
    for key, target in aliases:
        entry = names.get(target)
        # A module whose own name matches the target's leaf resolves to
        # itself (``lib/math.py`` spells ``e = math.e`` for the STDLIB math);
        # there is nothing to copy, and the alias is simply dropped
        if entry is not None and entry['kind'] != 'alias' and target != key:
            names[key] = entry
        else:
            del names[key]


def _class_fields(node: ast.ClassDef, classes: dict[str, str]) -> dict[str, str]:
    """
    The annotated fields of one class, in declaration order.

    :param node: The class
    :param classes: Class name -> class id, for resolving a field's own type
    :return: Field name -> type
    """
    return {stmt.target.id: annotation_type(stmt.annotation, classes)
            for stmt in node.body
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            and not stmt.target.id.startswith('_')}


def _is_types_module(node: ast.ImportFrom) -> bool:
    """
    Whether an import statement pulls names out of ``pynecore.types``.

    :param node: The import statement
    :return: True when its names are type classes
    """
    module = node.module or ''
    if node.level:
        return module == 'types' or module.startswith('types.')
    return module == 'pynecore.types' or module.startswith('pynecore.types.')


def _class_names(tree: ast.Module) -> dict[str, str]:
    """
    Every name one lib module may spell a Pine type with, and its class id.

    Two sources, and both are needed to type the constants: the classes the
    module declares itself (``chart.py``'s ``_ChartPoint`` namespace) and the
    ones it imports out of ``pynecore.types`` (``Display``, ``Color``,
    ``XLoc``). Resolving them by WHERE THEY COME FROM rather than by their
    spelling is what keeps ``TypeVar`` and ``logging.getLogger`` out.

    Every one of them is a class the LIB publishes, so they share the reserved
    module key -- a script naming ``Line`` in an annotation lands on the same
    id ``line.new`` returns, which is what makes the two comparable at all.

    :param tree: Parsed module AST
    :return: Type name -> class id
    """
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and _is_types_module(node):
            names.update(alias.asname or alias.name
                         for alias in node.names if alias.name != '*')
    return {name: builtin_class_id(name) for name in names}


def _value_type(node: ast.expr, classes: dict[str, str]) -> str:
    """
    Pine type of a module-level constant, from the expression that builds it.

    A lib namespace publishes its constants as bare constructor calls
    (``data_window = Display()``, ``red = Color('#F23645')``,
    ``islast = False``), so an extractor that only reads annotations left every
    one of them untyped -- and with them every script expression that passes
    one. The class the call names IS the type: ``Color`` is Pine's color, and
    every other type class is a known non-scalar.

    ``NA(<type>)`` is the one call whose ARGUMENT carries the type: a typed
    ``na`` is of the type it names, which is what makes
    ``earnings.future_time = NA(int)`` an int.

    :param node: The value expression
    :param classes: The type names visible in the module
    :return: The type character, UNKNOWN for anything this cannot decide
    """
    if isinstance(node, ast.Constant):
        return constant_type(node.value)
    if isinstance(node, ast.Call):
        func = node.func
        name = func.id if isinstance(func, ast.Name) else \
            (func.attr if isinstance(func, ast.Attribute) else '')
        if name == 'NA':
            return annotation_type(node.args[0], classes) if node.args else UNKNOWN
        if name in classes:
            # ``Color('#F23645')`` is Pine's color scalar; every other type
            # class builds an object OF that class
            return ANNOTATION_TYPES.get(name) or object_ty(classes[name])
    return UNKNOWN


def _own_returns(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.Return]:
    """
    Every ``return`` of a function's own body, nested definitions excluded.

    :param node: The definition to scan
    :return: The return statements, in no particular order
    """
    out: list[ast.Return] = []
    stack: list[ast.AST] = list(node.body)
    while stack:
        current = stack.pop()
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(current, ast.Return):
            out.append(current)
        stack.extend(ast.iter_child_nodes(current))
    return out


def _result_type(node: ast.FunctionDef | ast.AsyncFunctionDef,
                 classes: dict[str, str]) -> str:
    """
    What one lib function evaluates to.

    The annotation answers wherever there is one. Where there is none the BODY
    still answers the case that matters most: a function with no ``return
    <value>`` at all returns nothing, which is a VOID and not a typing failure
    -- ``plot()``, ``strategy.entry()`` and ``line.delete()`` are statements,
    and reading them as unknown put every script that calls one outside the
    typed subset for no reason. Where it does return values, they are typed the
    same way a constant is, so ``plot()``'s ``return Plot(t)`` comes out an
    object.

    :param node: The definition to measure
    :param classes: The type names visible in the module
    :return: The result type character
    """
    if node.returns is not None:
        return annotation_type(node.returns, classes)
    returns = _own_returns(node)
    if not returns:
        return VOID
    result = VOID if any(r.value is None for r in returns) else None
    for statement in returns:
        if statement.value is None:
            continue
        ty = _value_type(statement.value, classes)
        result = ty if result is None else join(result, ty)
    return UNKNOWN if result is None else result


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


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef, classes: dict[str, str],
               bound: bool = False) -> dict[str, Any]:
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
    :param classes: The type names visible in the module
    :param bound: Whether the first parameter is the receiver a namespace
                  instance binds away (``chart.point.new`` is called with the
                  arguments the SCRIPT spells, not with the instance)
    :return: ``{'ret': .., 'params': [..], 'names': [..], 'defaults': <count>,
              'default_ty': [..], 'default_none_ok': [..], 'vararg': .., 'kwarg': True}``
    """
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    if bound and positional:
        positional = positional[1:]
    signature: dict[str, Any] = {
        'ret': _result_type(node, classes),
        'params': [annotation_type(a.annotation, classes) for a in positional],
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
        signature['vararg'] = annotation_type(args.vararg.annotation, classes)
    if args.kwarg is not None:
        # ``fill(*args, **kwargs)`` takes any keyword: the call shape is the
        # function's own business
        signature['kwarg'] = True
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

    A module-level CONSTANT is typed from its annotation, or -- for the bare
    constructor calls every lib namespace publishes its constants as
    (``data_window = Display()``) -- from the class the call names. A constant
    that merely re-publishes another namespace's (``long = direction.long``)
    is recorded as an ``alias`` for :func:`_resolve_aliases` to replace with
    the entry it names.

    A namespace that is an INSTANCE of a module-local class (``point =
    _ChartPoint()``) publishes that class's public methods under the dotted
    spelling a script uses: ``chart.point.new(...)``.

    :param tree: Parsed module AST
    :return: name -> entry mapping
    """
    info: dict[str, Any] = {}
    classes = _class_names(tree)
    declared = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}

    def record_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if node.name.startswith('_'):
            return
        decorators = _decorator_names(node)
        signature = _signature(node, classes)
        if {'module_property', 'module_function_property'} & decorators:
            # Read as a value: its type IS the return type; a script may call
            # it as well, which reads the same
            info[node.name] = {'kind': 'value', 'ty': signature['ret'], 'callable': True}
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

    def record_namespace(name: str, node: ast.ClassDef) -> None:
        for statement in node.body:
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    or statement.name.startswith('_'):
                continue
            bound = 'staticmethod' not in _decorator_names(statement)
            info[f'{name}.{statement.name}'] = {
                'kind': 'function', **_signature(statement, classes, bound)}

    def record_value(target: ast.expr, annotation: ast.expr | None,
                     value: ast.expr | None) -> None:
        if not isinstance(target, ast.Name) or target.id.startswith('_'):
            return
        ty = annotation_type(annotation, classes) if annotation is not None else UNKNOWN
        if ty == UNKNOWN and value is not None:
            spelled = _dotted(value)
            if spelled is not None:
                info[target.id] = {'kind': 'alias', 'to': spelled}
                return
            ty = _value_type(value, classes)
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) \
                    and value.func.id in declared:
                record_namespace(target.id, declared[value.func.id])
        if ty not in (UNKNOWN, VOID):
            info[target.id] = {'kind': 'value', 'ty': ty}

    def walk(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                record_function(node)
            elif isinstance(node, ast.AnnAssign):
                record_value(node.target, node.annotation, node.value)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    record_value(target, None, node.value)
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


def _dotted(node: ast.expr) -> str | None:
    """
    Render a dotted name expression, or None when it is not one.

    :param node: The expression to render
    :return: Its dotted spelling, or None
    """
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return '.'.join(reversed(parts))


if __name__ == '__main__':
    LibTypeCollector().write()
