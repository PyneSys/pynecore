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

Two sections are not extracted from the source but measured on the IMPORTED
lib, because what they record only exists once the import hook has transformed
it: ``routes``, the call-site route ``FunctionIsolationTransformer`` gives each
lib callee (``ta.sma`` carries a slot layout, ``ta.highest`` is an overload
dispatcher), and ``star_exports``, the ``__all__`` a ``from pynecore.lib...
import *`` expands to. The transform reads both from here instead of
inspecting the live lib: the lib sources are not part of the pipeline hash,
this file is, so a route-relevant lib edit invalidates cached script bytecode
only through the regenerated file -- and ``test_104`` fails until it is
regenerated.

Usage:
    python3 scripts/lib_type_collector.py
"""
import ast
import importlib
import inspect
import json
import pkgutil
import sys
import types
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import pynecore.lib  # noqa: E402
from pynecore.core.instance_state import __dyn_default__  # noqa: E402
from pynecore.core.overload import Implementation, _check_type, _PIN_WITNESSES  # noqa: E402
from pynecore.transformers.function_isolation import classify_callee  # noqa: E402
from pynecore.transformers.pine_type_rules import (  # noqa: E402
    annotation_takes_none, annotation_type, builtin_class_id, constant_type, join,
    object_ty, ACCEPTS_SEP, ANNOTATION_TYPES, BOOL, FIT_OMISSIBLE, FIT_REQUIRED, FLOAT, INT,
    NONE_DEFAULT, OBJECT, PIN_WITNESSED, STR, TYPELESS, UNKNOWN, VOID,
)

#: Registry format version. Bump whenever the shape below changes; the
#: consumers (the inference engine and its native port) pin it.
SCHEMA_VERSION = 12

#: The route ``classify_callee`` gives a callee it proves nothing about. The
#: ``routes`` section leaves such paths out, and the transform reads a path it
#: does not find there the same way
_UNIFORM = 'uniform'


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
        imported: dict[str, dict[str, str]] = {}
        for file_path in sorted(self.lib_path.rglob('*.py')):
            prefix = self._module_prefix(file_path)
            tree = ast.parse(file_path.read_text(), filename=str(file_path))
            per_module[prefix] = collect_module_types(tree)
            reexports[prefix] = _sibling_reexports(tree, prefix)
            imported[prefix] = _module_imports(tree, prefix, file_path.stem == '__init__')
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
        _resolve_default_refs(per_module, imported, names)
        modules = self._import_lib()
        dispatchers = collect_dispatchers(names)
        measure_dispatchers(names, dispatchers)
        return {'v': SCHEMA_VERSION, 'names': names, 'classes': self._collect_fields(),
                'scalar_classes': self._collect_scalars(),
                'str_literals': self._collect_str_literals(),
                'bit_sets': self._collect_bit_sets(),
                'dispatchers': dispatchers,
                'routes': collect_routes(), 'star_exports': collect_star_exports(modules)}

    def _import_lib(self) -> list[types.ModuleType]:
        """
        Import every module of the lib package, through the import hook.

        The routes are a property of the TRANSFORMED lib, so they are measured on
        the imported one -- which has to be the tree this collector reads the
        source of, or the two halves of the registry would describe different
        libs.

        :return: The lib package and every module under it, by name
        """
        imported = Path(pynecore.lib.__file__).resolve().parent
        if imported != self.lib_path.resolve():
            raise RuntimeError(f"the imported pynecore.lib is {imported}, not {self.lib_path}: "
                               "put this source tree first on the import path")
        modules = [pynecore.lib]
        for info in pkgutil.walk_packages(pynecore.lib.__path__, 'pynecore.lib.'):
            modules.append(importlib.import_module(info.name))
        return sorted(modules, key=lambda module: module.__name__)

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
        bases = self._type_bases()
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

    def _collect_str_literals(self) -> list[str]:
        """
        The classes of the type package that derive from ``StrLiteral``.

        The overload selector takes any string for a parameter of such a class
        (``core/overload._check_type``), and the static selection reads a
        parameter of one the same way (``pine_type_rules.is_str_literal``); a
        string class of another lineage takes only its own instances.

        :return: The class names, sorted
        """
        bases = self._type_bases()
        found = {'StrLiteral'}
        settled = False
        while not settled:
            settled = True
            for name, parents in bases.items():
                if name not in found and any(parent in found for parent in parents):
                    found.add(name)
                    settled = False
        return sorted(found)

    def _collect_bit_sets(self) -> list[str]:
        """
        The classes of the type package that are a set of flags.

        ``Display`` derives from ``BitSet``, whose ``+`` and ``-`` are the union
        and the difference of two sets and answer the same class:
        ``display.all - display.status_line`` is a ``Display``.

        :return: The class names, sorted
        """
        bases = self._type_bases()
        found = {'BitSet'}
        settled = False
        while not settled:
            settled = True
            for name, parents in bases.items():
                if name not in found and any(parent in found for parent in parents):
                    found.add(name)
                    settled = False
        return sorted(found - {'BitSet'})

    def _type_bases(self) -> dict[str, list[str]]:
        """
        The base classes every class of the type package names.

        :return: Class name -> the names of its bases
        """
        bases: dict[str, list[str]] = {}
        for file_path in sorted(self.types_path.rglob('*.py')):
            tree = ast.parse(file_path.read_text(), filename=str(file_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    bases.setdefault(node.name, [
                        base.id if isinstance(base, ast.Name) else base.attr
                        for base in node.bases if isinstance(base, (ast.Name, ast.Attribute))])
        return bases

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


def collect_routes() -> dict[str, str]:
    """
    The call-site route of every lib callee path that does not route uniform.

    The paths are walked the way ``FunctionIsolationTransformer`` resolves a
    callee, attribute by attribute from ``lib``: through the lib's own
    submodules (a module a lib module merely imports, like ``lib.alert.lib``,
    is not followed), and through the namespace objects that publish callables
    as members (see :func:`_is_namespace`). Each object is classified by the
    transformer's own :func:`classify_callee`, so the decision exists once.

    A class is left out, and the transform routes its path uniform, which is
    correct for any callee: a script constructs no object through a lib path
    (``NA`` and ``Series`` come from ``pynecore.types``), while which classes a
    lib module holds depends on its imports and even on the environment
    (``lib.log`` defines its rich handler only where rich is in use) -- a
    registry recording them would differ between two machines.

    A callable defined outside pynecore is left out for the same reason: a lib
    module's own import (``lib.string.localcontext`` from ``decimal``) is no lib
    API, and its route follows the interpreter's build -- ``decimal`` is the C
    ``_decimal`` (a builtin) where the interpreter was built with libmpdec and the
    pure-Python ``_pydecimal`` where it was not.

    Every lib module must be imported first (``LibTypeCollector._import_lib``):
    a submodule is reachable as an attribute of its package only once loaded.

    :return: Callee path as a script spells it (``lib.ta.sma``) -> route
    """
    routes: dict[str, str] = {}
    _record_routes('lib', pynecore.lib, routes)
    return dict(sorted(routes.items()))


def collect_dispatchers(names: dict[str, dict[str, Any]]) -> list[str]:
    """
    The overload groups of the lib that select an implementation at run time.

    A group declared with ``pynecore.core.overload`` is a dispatcher: its
    implementations are separate functions, and the argument types of a call
    decide which one runs (``ta.highest``). One declared with ``typing.overload``
    only spells the types of a single function (``math.max``, ``line.new``): every
    call runs the same body, so a call site has nothing to select.

    Every lib module must be imported first (``LibTypeCollector._import_lib``).

    :param names: The registry's ``names`` section
    :return: The registry keys of the dispatching groups, sorted
    """
    return sorted(key for key, entry in names.items()
                  if entry['kind'] == 'overloads'
                  and getattr(_lib_object(key), '__pyne_bind__', None) is not None)


def _lib_object(key: str) -> object:
    """
    The imported lib object a registry key names.

    :param key: A registry key, as a script spells it (``ta.highest``)
    :return: The object
    """
    value: object = pynecore.lib
    for part in key.split('.'):
        value = getattr(value, part)
    if isinstance(value, types.ModuleType):
        # ``lib/plot.py::plot`` is spelled ``plot``
        value = getattr(value, key.rpartition('.')[2])
    return value


def measure_dispatchers(names: dict[str, dict[str, Any]], dispatchers: list[str]) -> None:
    """
    Record how the selector checks each implementation of every dispatching group.

    The static resolution of a pin (``pine_type_rules.resolve_pin``) has to
    answer exactly what the runtime selector answers on the pin's witness
    values, and an annotation's Pine type does not always say it: an ``int |
    float`` takes an int in the exact pass, a ``str`` alias is a string, a
    default the lib computes (``_xloc.bar_index``) has a type only once it is
    computed. So it is MEASURED here, with the selector's own check
    (``core/overload._check_type``) on the implementations the import
    registered, and recorded per implementation:

    * ``accepts`` -- per positional parameter, the witness characters the exact
      pass takes, then ``ACCEPTS_SEP``, then the ones the widening pass takes;
    * ``fits`` and ``wide_fits`` -- per parameter, the positional ones then the
      keyword-only ones, whether the default an omitted argument binds passes
      that pass's check (``FIT_OMISSIBLE``) or not (``FIT_REQUIRED``, also for
      no default at all).

    The registered implementations must be the source's, one for one: their
    order is the registration order an indexed pin names.

    :param names: The registry's ``names`` section, its group entries edited in place
    :param dispatchers: The registry keys of the dispatching groups
    :raises RuntimeError: Where the imported group is not the collected one
    """
    if ''.join(_PIN_WITNESSES) != PIN_WITNESSED:
        raise RuntimeError(f"the runtime's witnesses are {''.join(_PIN_WITNESSES)!r}, "
                           f"not {PIN_WITNESSED!r}")
    for key in dispatchers:
        registered: list[Implementation] = getattr(_lib_object(key), '__pyne_impls__')
        collected = names[key]['impls']
        if len(registered) != len(collected):
            raise RuntimeError(f"{key}: {len(registered)} implementations registered, "
                               f"{len(collected)} collected")
        for impl, entry in zip(registered, collected):
            positional = [param for param in impl.sig.parameters.values()
                          if param.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                            inspect.Parameter.POSITIONAL_OR_KEYWORD)]
            if [param.name for param in positional] != entry['names']:
                raise RuntimeError(f"{key}: registered {[p.name for p in positional]}, "
                                   f"collected {entry['names']}")
            entry.update(_measured(impl, positional))


def _measured(impl: Implementation, positional: list[inspect.Parameter]) -> dict[str, Any]:
    """
    How the selector checks one registered implementation (``measure_dispatchers``).

    :param impl: The implementation
    :param positional: Its positional parameters
    :return: Its ``accepts``, ``fits`` and ``wide_fits``
    """
    accepts = []
    for param in positional:
        hint = impl.type_hints.get(param.name, Any)
        accepts.append(ACCEPTS_SEP.join(
            ''.join(char for char, witness in _PIN_WITNESSES.items()
                    if _check_type(witness, hint, strict))
            for strict in (True, False)))
    visible = [param for param in impl.sig.parameters.values()
               if param.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                     inspect.Parameter.VAR_KEYWORD)]
    fits = []
    for strict in (True, False):
        fits.append(''.join(_measured_fit(impl, param, strict) for param in visible))
    return {'accepts': accepts, 'fits': fits[0], 'wide_fits': fits[1]}


def _measured_fit(impl: Implementation, param: inspect.Parameter, strict: bool) -> str:
    """
    Whether the default of one parameter passes one pass of the selector.

    :param impl: The implementation
    :param param: One of its visible parameters
    :param strict: The exact pass; the widening pass otherwise
    :return: ``FIT_OMISSIBLE`` or ``FIT_REQUIRED``
    """
    if param.default is inspect.Parameter.empty:
        return FIT_REQUIRED
    if param.name not in impl.type_hints or param.default is __dyn_default__:
        return FIT_OMISSIBLE
    taken = _check_type(param.default, impl.type_hints[param.name], strict)
    return FIT_OMISSIBLE if taken else FIT_REQUIRED


def _record_routes(path: str, owner: object, routes: dict[str, str]) -> None:
    """
    Classify every member of a lib module or namespace, recursively.

    :param path: The owner's path as a script spells it
    :param owner: The module or namespace object
    :param routes: The route table, filled in place
    """
    for name in dir(owner):
        if name.startswith('__') and name.endswith('__'):
            continue
        try:
            value = getattr(owner, name)
        except Exception:  # noqa: the transform reads a failing lookup as unprovable
            continue
        key = f'{path}.{name}'
        if isinstance(value, types.ModuleType):
            if value.__name__ == f'pynecore.{key}':
                _record_routes(key, value, routes)
            continue
        if isinstance(value, type) or not _defined_in_pynecore(value):
            continue
        route = classify_callee(value)
        if route != _UNIFORM:
            routes[key] = route
        if _is_namespace(value):
            _record_routes(key, value, routes)


def _defined_in_pynecore(value: object) -> bool:
    """
    Whether a lib member is pynecore's own: a function, or an instance of a class,
    defined in a pynecore module.

    :param value: A member of a lib module
    :return: False for what a lib module imported from elsewhere
    """
    module = getattr(value, '__module__', None)
    return isinstance(module, str) and (module == 'pynecore' or module.startswith('pynecore.'))


def _is_namespace(value: object) -> bool:
    """
    Whether a lib object is a namespace a script calls members of.

    The lib publishes such a namespace as the one instance of a private class
    (``chart.point = _ChartPoint()``, ``input = _Input()``), so
    ``chart.point.new`` is a callee path of its own. A value object (a
    ``Color`` constant, a ``Source``) is an instance of a public class, and a
    private enum's member (``strategy._order_type_entry``) is a scalar; neither
    has a member a script calls.

    :param value: A member of a lib module
    :return: True when its members are callee paths too
    """
    cls = type(value)
    return (cls.__name__.startswith('_') and cls.__module__.startswith('pynecore.')
            and not isinstance(value, (int, float, str)))


def collect_star_exports(modules: list[types.ModuleType]) -> dict[str, list[str]]:
    """
    The ``__all__`` of every lib module that defines one.

    :param modules: The imported lib modules
    :return: Module name -> its ``__all__``, sorted
    """
    return {module.__name__: sorted(module.__all__)
            for module in modules if hasattr(module, '__all__')}


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
        refs = [_dotted(d) if ty == UNKNOWN else None for d, ty in zip(args.defaults, default_ty)]
        if any(refs):
            # Resolved against the registry once every module is collected
            # (``_resolve_default_refs``), and never written out
            signature[_DEFAULT_REFS] = refs
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


#: Key of the dotted spellings of the defaults ``_default_type`` cannot type,
#: per default, None for the rest; it lives only until ``_resolve_default_refs``
_DEFAULT_REFS = '_default_refs'


def _module_imports(tree: ast.Module, prefix: str, package: bool) -> dict[str, str]:
    """
    The lib modules one lib module imports, by the name it binds them to.

    ``from ..lib import xloc as _xloc`` binds ``_xloc`` to the ``xloc``
    namespace, which is what a default spelled ``_xloc.bar_index`` reads.

    :param tree: Parsed module AST
    :param prefix: Dotted lib path of the module
    :param package: Whether the module is a package's ``__init__``
    :return: Bound name -> dotted lib path of what it names (``''`` for lib itself)
    """
    parts = prefix.split('.') if prefix else []
    if not package and parts:
        parts = parts[:-1]
    found: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or not node.level:
            continue
        # One level is the module's own package, which is ``lib`` plus ``parts``;
        # every further level climbs one package, out of ``lib`` at the last one
        base: list[str] | None = ['lib'] + parts
        for _ in range(node.level - 1):
            base = base[:-1] if base else None
        if base is None:
            continue
        base = base + (node.module.split('.') if node.module else [])
        if not base or base[0] != 'lib':
            continue
        for alias in node.names:
            path = '.'.join(base[1:] + [alias.name])
            found[alias.asname or alias.name] = path
    return found


def _resolve_default_refs(per_module: dict[str, dict[str, Any]],
                          imported: dict[str, dict[str, str]],
                          names: dict[str, Any]) -> None:
    """
    Type the defaults that name a constant of another lib namespace.

    A default the lib spells as another module's constant (``xloc:
    _xloc.XLoc = _xloc.bar_index``) is typed by the registry entry of that
    constant, through the name the defining module imports its namespace as.
    A spelling that resolves to no value entry stays UNKNOWN. Either way the
    note ``_default_type`` left is removed.

    :param per_module: Every module's collected entries, by dotted lib path
    :param imported: Every module's imported lib modules (``_module_imports``)
    :param names: The registry's ``names`` section
    """
    for prefix, entries in per_module.items():
        aliases = imported.get(prefix, {})
        for entry in entries.values():
            for signature in [entry, *entry.get('impls', ())]:
                refs = signature.pop(_DEFAULT_REFS, None)
                if refs is None:
                    continue
                for index, ref in enumerate(refs):
                    first, _, rest = (ref or '').partition('.')
                    target = aliases.get(first)
                    if target is None or not rest:
                        continue
                    found = names.get(f'{target}.{rest}' if target else rest)
                    if isinstance(found, dict) and found.get('kind') == 'value':
                        signature['default_ty'][index] = found['ty']


def _default_type(node: ast.expr) -> str:
    """
    Pine type of one parameter default, as the overload selector reads it.

    What a pure extractor can decide: a literal carries its own type,
    ``na`` carries none at all, and a literal ``None`` carries its own
    character, because whether it fits is a question about the ANNOTATION and
    not about the annotation's Pine type. Anything the lib computes
    (``_color.blue``, ``_xloc.bar_index``) is UNKNOWN here, which is what makes
    the inference decline to pick rather than pick wrong, until
    ``_resolve_default_refs`` types the constants of another namespace.

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
