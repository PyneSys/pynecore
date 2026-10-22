"""
The native (Rust) backend of the transform pipeline.

The Python steps are the reference. A native step is a port of the Python step of
the same name and must hand back the same tree for the same input; it exists so
the pipeline can also run where there is no Python (the browser, the AOT
compiler) and, step by step, run faster here.

Which backend a step runs on:

- ``PYNE_AST_BACKEND=python`` (the default): every step runs in Python.
- ``PYNE_AST_BACKEND=native``: a step with a CURRENT native port runs natively;
  consecutive native steps run in one native call, so the tree crosses the
  boundary once per run of them.
- ``PYNE_AST_BACKEND=verify``: every step runs in Python, and a step with a
  current native port runs natively on the same input as well; any difference
  in the resulting trees (fields, locations, stamped attributes) raises
  :class:`NativeMismatch`. The parity tests and the emission oracle use this.

A port is CURRENT when the Python sources it was ported from are unchanged. The
extension carries, per step, the digest of those sources at the time of the
port (``rust/crates/pyne-transform/ported.json``, compiled in); the digest of the
sources as they are now is computed here. A Python edit that is not yet ported
therefore takes the native step out of use -- the step runs in Python, with one
warning per process -- instead of letting the two drift apart silently.

A :class:`~pynecore.transformers.pipeline.Phase` runs natively as one unit under
its own name, the rules that apply to the module in one traversal, while each of
its rules is current; in sequential mode each rule runs as a step of its own.

A native step that meets something its port does not handle raises
``Unsupported``; the step then runs in Python. Neither case changes the result,
only where it is computed. A native step that finds the module invalid raises
``Rejected``, and the Python step runs to raise the error the author sees; under
``verify`` the two must agree on that verdict as well.
"""
import ast
import dataclasses
import hashlib
import json
import os
import sys
from collections.abc import Iterable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from . import ast_walk
from .builtin_shadow import Library, import_library, library_serves
from .export_capture import module_origin
from . import module_interface
from .lib_registry import lib_star_exports, lib_types
from .pine_type_infer import module_spec_origin, package_member
from .pine_type_table import Analyser
from .source_read import stable_source
from .module_property import module_properties
from .slot_layout import ModuleLayout
from ..utils.stdlib_checker import is_stdlib

try:
    from . import _native
except ImportError:
    _native = None

__all__ = ['available', 'backend', 'current_steps', 'current_phases', 'layout_steps',
           'source_digest', 'run', 'verify', 'verify_rejection', 'Rejection', 'NativeMismatch']

_TRANSFORMERS = Path(__file__).parent
_PACKAGE = _TRANSFORMERS.parent

_BACKENDS = ('python', 'native', 'verify')


class NativeMismatch(AssertionError):
    """A native step produced a different tree than its Python reference."""


def available() -> bool:
    """Whether the native extension is importable."""
    return _native is not None


def backend() -> str:
    """The selected backend: ``python``, ``native`` or ``verify``."""
    value = os.environ.get('PYNE_AST_BACKEND', 'python')
    if value not in _BACKENDS:
        raise ValueError(f"PYNE_AST_BACKEND={value!r}: expected one of {', '.join(_BACKENDS)}")
    return value


def _imports(path: Path) -> list[Path]:
    """The pynecore modules a source file imports, as files (relative imports included)."""
    try:
        tree = ast.parse(path.read_text(encoding='utf-8'))
    except (OSError, SyntaxError):
        return []
    package = path.parent
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = package
                for _ in range(node.level - 1):
                    base = base.parent
                parts = node.module.split('.') if node.module else []
            elif node.module and (node.module == 'pynecore' or node.module.startswith('pynecore.')):
                base = _PACKAGE
                parts = node.module.split('.')[1:]
            else:
                continue
            target = base.joinpath(*parts)
            candidates = [target.with_suffix('.py'), target / '__init__.py']
            # ``from . import ast_walk`` names a module, not a member
            candidates.extend(target / f'{alias.name}.py' for alias in node.names)
            found.extend(candidate for candidate in candidates if candidate.is_file())
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith('pynecore.'):
                    target = _PACKAGE.joinpath(*alias.name.split('.')[1:])
                    found.extend(c for c in (target.with_suffix('.py'), target / '__init__.py')
                                 if c.is_file())
    return found


def _closure(module: str) -> list[Path]:
    """A step module and what it depends on: every ``transformers`` module it reaches
    through imports, plus the other pynecore modules those import directly, plus
    the package's data files that any of them names."""
    start = _TRANSFORMERS / f'{module}.py'
    seen = {start}
    todo = [start]
    while todo:
        path = todo.pop()
        for target in _imports(path):
            if target in seen:
                continue
            seen.add(target)
            if target.parent == _TRANSFORMERS:
                todo.append(target)
    sources = [path.read_text(encoding='utf-8') for path in seen]
    seen.update(data for data in _TRANSFORMERS.glob('*.json')
                if any(data.name in source for source in sources))
    return sorted(seen)


_digests: dict[str, str] = {}


def source_digest(module: str) -> str:
    """Digest of the Python sources a step module's behaviour depends on."""
    digest = _digests.get(module)
    if digest is None:
        hasher = hashlib.sha256()
        for path in _closure(module):
            hasher.update(path.relative_to(_PACKAGE).as_posix().encode('utf-8'))
            hasher.update(b'\0')
            # A checkout with CRLF line ends (Windows) has the same sources
            hasher.update(path.read_bytes().replace(b'\r\n', b'\n'))
            hasher.update(b'\0')
        digest = _digests[module] = hasher.hexdigest()
    return digest


_current: frozenset[str] | None = None


def current_steps() -> frozenset[str]:
    """The native steps whose port matches the Python sources as they are now."""
    global _current
    if _current is None:
        if _native is None:
            _current = frozenset()
            return _current
        current = set()
        stale = []
        registered = set(_native.steps())
        for name, entry in json.loads(_native.ported_json()).items():
            if name in registered and source_digest(entry['module']) == entry['digest']:
                current.add(name)
            else:
                stale.append(name)
        if stale:
            sys.stderr.write(f"[pyne-ast] native steps behind their Python source, run in "
                             f"Python: {', '.join(sorted(stale))}\n")
        _current = frozenset(current)
    return _current


def layout_steps() -> frozenset[str]:
    """The ported steps that read or fill the pipeline's slot layout (``ctx.slot_layout``)."""
    if _native is None:
        return frozenset()
    return frozenset(_native.layout_steps())


def current_phases(phases: Iterable[Any]) -> frozenset[str]:
    """The phases that run natively as one unit.

    A phase does while the native phase of its name declares the same rules, in the
    same order and with the same ``user_code_only``, and every rule's port is
    current (see :func:`current_steps`). A rule is a native step of its own as
    well: in sequential mode it runs alone.

    :param phases: The pipeline's phases (``pipeline.Phase``).
    :return: The names of those that run natively.
    """
    if _native is None:
        return frozenset()
    declared = _native.phases()
    current = current_steps()
    return frozenset(
        phase.name for phase in phases
        if declared.get(phase.name) == [(rule.name, rule.user_code_only) for rule in phase.rules]
        and all(rule.name in current for rule in phase.rules))


class _Host:
    """What a native step asks the interpreter, answered by the very functions the
    Python step uses, at the moment the Python step would use them."""

    def __init__(self, analyse: Analyser | None = None, pipeline_hash: str = '') -> None:
        self._libraries: dict[str, Library] = {}
        self._analyse = analyse
        self._pipeline_hash = pipeline_hash
        self._analysing: list[AbstractContextManager[None]] = []

    def import_library(self, name: str) -> bool:
        library = import_library(name)
        if library is None:
            return False
        self._libraries[name] = library
        return True

    def library_serves(self, name: str, attr: str) -> bool:
        return library_serves(self._libraries[name], attr)

    @staticmethod
    def has_module_properties(module: str) -> bool:
        return module in module_properties()

    @staticmethod
    def module_property(module: str, name: str) -> bool:
        return name in module_properties()[module]

    @staticmethod
    def module_property_type(module: str, name: str) -> str | None:
        entry = module_properties()[module].get(name)
        return None if entry is None else entry["type"]

    @staticmethod
    def star_exports(module: str) -> list[str] | None:
        return lib_star_exports().get(module)

    @staticmethod
    def lib_name(name: str) -> tuple[str, str | None, bool, str | None, list[str | None]] | None:
        entry = lib_types().get(name)
        if entry is None:
            return None
        return (entry['kind'], entry.get('ty'), bool(entry.get('callable', False)),
                entry.get('ret'), [impl.get('ret') for impl in entry.get('impls', ())])

    @staticmethod
    def module_origin(name: str, importer: str, level: int) -> str | None:
        path = module_origin(name, importer, level)
        return None if path is None else str(path)

    @staticmethod
    def read_source(path: str) -> tuple[str, bytes, int, int] | None:
        source = Path(path)
        loaded = stable_source(source)
        if loaded is None:
            return None
        data, (mtime_ns, size) = loaded
        return str(source.resolve()), data, mtime_ns, size

    def interface(self, path: str) -> str | None:
        """The interface a module publishes and the record a dependency on it keeps, as JSON
        in the dataclasses' own field order."""
        found = module_interface.lookup(path, self._analyse, self._pipeline_hash)
        if found is None:
            return None
        return json.dumps({'interface': dataclasses.asdict(found),
                           'record': dataclasses.asdict(module_interface.dep_record(found))})

    @staticmethod
    def closes_cycle(path: str) -> bool:
        return module_interface.closes_cycle(path)

    def begin_analysis(self, path: str) -> None:
        scope = module_interface.analysing_scope(path)
        scope.__enter__()
        self._analysing.append(scope)

    def end_analysis(self) -> None:
        self._analysing.pop().__exit__(None, None, None)

    @staticmethod
    def module_spec_origin(dotted: str) -> str | None:
        return module_spec_origin(dotted)

    @staticmethod
    def package_member(directory: str, dotted: str) -> str | None:
        return package_member(Path(directory), dotted)

    @staticmethod
    def is_stdlib(module: str) -> bool:
        return is_stdlib(module)


def _context(ctx: Any) -> dict[str, Any]:
    return {'path': str(ctx.path), 'pyne_mode': ctx.pyne_mode, 'source': ctx.source,
            'user_code': ctx.user_code, 'na_bool': ctx.na_bool, 'sequential': ctx.sequential,
            'host': _Host(ctx.analyse, ctx.pipeline_hash),
            'slot_layout': getattr(ctx, 'slot_layout', None)}


class Rejection:
    """A native step found the module invalid: the Python step raises the error the
    author sees."""

    __slots__ = ('message',)

    def __init__(self, message: str) -> None:
        self.message = message


def run(names: list[str], tree: ast.Module, ctx: Any,
        layout: ModuleLayout | None = None) -> ast.Module | Rejection | None:
    """Run native steps in order over a module.

    A step of :func:`layout_steps` reads ``ctx.slot_layout`` as the Python steps
    would have left it, and fills a native copy of it.

    :param names: Step names, each in :func:`current_steps`, and phase names, each in
        :func:`current_phases` (a phase runs the rules that apply to the module).
    :param tree: The module; it is not changed.
    :param ctx: The pipeline context; its slot layout is not changed either.
    :param layout: Where the slot layout the steps left is written when the steps
        answer a module and one of them uses the layout: its state is replaced
        field for field. None leaves the layout the steps left unread.
    :return: The transformed module, a new object; a :class:`Rejection` when a step
        finds the module invalid; None when a step does not handle the module. In
        both latter cases the caller runs the Python steps.
    """
    assert _native is not None
    context = _context(ctx)
    try:
        result = _native.run(tree, names, context)
    except _native.Rejected as exc:
        return Rejection(str(exc))
    except _native.Unsupported:
        return None
    left = context.get('slot_layout_result')
    if layout is not None and left is not None:
        state = vars(layout)
        state.clear()
        state.update(vars(left))
    return result


def _describe(node: Any, path: str, seen: dict[int, str], out: list[str]) -> None:
    """Flatten a tree into ``path: value`` lines; a node reached a second time is
    written as a reference to the path it was first seen at.

    An object the native side hands through without looking into it (the type
    pass's ``PineTypeTable``) is written by VALUE, field by field for a dataclass,
    the way a node is: the bridge gives back the very object it was handed, but a
    native copy of a subtree shares it where ``copy.deepcopy`` copies it, and the
    two are the same result."""
    if isinstance(node, (ast.expr_context, ast.operator, ast.unaryop, ast.cmpop, ast.boolop)):
        # Fieldless: whether two of them are one shared instance means nothing
        out.append(f'{path}: {type(node).__name__}')
    elif isinstance(node, ast.AST):
        key = id(node)
        if key in seen:
            out.append(f'{path} -> {seen[key]}')
            return
        seen[key] = path
        out.append(f'{path}: {type(node).__name__}')
        for name in node._fields:
            _describe(getattr(node, name, None), f'{path}.{name}', seen, out)
        for name in node._attributes:
            out.append(f'{path}.{name} = {getattr(node, name, None)!r}')
        for name, value in sorted(vars(node).items()):
            if name not in node._fields and name not in node._attributes:
                _describe(value, f'{path}.<{name}>', seen, out)
    elif isinstance(node, (list, tuple)):
        out.append(f'{path}: {type(node).__name__}[{len(node)}]')
        for index, item in enumerate(node):
            _describe(item, f'{path}[{index}]', seen, out)
    elif isinstance(node, dict):
        out.append(f'{path}: dict[{len(node)}]')
        for key, item in node.items():
            _describe(item, f'{path}[{key!r}]', seen, out)
    elif isinstance(node, (set, frozenset)):
        # Iteration order follows the insertion history, which equal sets need not share
        out.append(f'{path}: {type(node).__name__}[{len(node)}]')
        out.extend(f'{path}{{}} = {item}' for item in sorted(repr(item) for item in node))
    elif dataclasses.is_dataclass(node) and not isinstance(node, type):
        out.append(f'{path}: {type(node).__name__}')
        for item in dataclasses.fields(node):
            _describe(getattr(node, item.name), f'{path}.{item.name}', seen, out)
    elif isinstance(node, float):
        # ``repr`` tells -0.0 from 0.0; NaN compares by spelling
        out.append(f'{path} = float {node!r}')
    else:
        out.append(f'{path} = {type(node).__name__} {node!r}')


def _describe_layout(tree: ast.Module, layout: ModuleLayout, seen: dict[int, str],
                     out: list[str]) -> None:
    """Flatten a slot layout after the tree it was filled for, with the same node
    references: a template expression that is also a node of the tree is written as
    a reference to it. The definition caches are keyed by object identity, so they
    are written as what the layout answers for each definition of the tree."""
    out.append(f'<layout>.compacted_series = {layout.compacted_series!r}')
    _describe(layout.scopes, '<layout>.scopes', seen, out)
    out.append(f'<layout>.scanned = {layout.scanned(tree)!r}')
    definitions = (node for node in ast_walk.walk(tree)
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
    for index, node in enumerate(definitions):
        out.append(f'<layout>.defs[{index}] = {layout.scope_segment(node)!r} '
                   f'{layout.nests_definition(node)!r}')


def describe(tree: ast.AST, layout: ModuleLayout | None = None) -> list[str]:
    """Everything a step's result consists of, one line per value, in a stable order.

    :param tree: The module a step answered.
    :param layout: The slot layout the step left, for a step that uses it.
    """
    out: list[str] = []
    seen: dict[int, str] = {}
    _describe(tree, '', seen, out)
    if layout is not None:
        assert isinstance(tree, ast.Module)
        _describe_layout(tree, layout, seen, out)
    return out


def verify(name: str, expected: ast.Module, actual: ast.Module | Rejection | None,
           ctx: Any, layout: ModuleLayout | None = None) -> None:
    """Raise :class:`NativeMismatch` unless the native result equals the Python one.

    :param name: The step.
    :param expected: What the Python step produced.
    :param actual: What :func:`run` answered for the same input.
    :param ctx: The pipeline context.
    :param layout: For a step of :func:`layout_steps`, the slot layout the native
        step left (the ``layout`` :func:`run` wrote), compared with the one the
        Python step left in ``ctx.slot_layout``.
    """
    if actual is None:
        return
    if isinstance(actual, Rejection):
        raise NativeMismatch(f'native {name} rejects {ctx.path}, which Python accepts: '
                             f'{actual.message}')
    want = describe(expected, None if layout is None else ctx.slot_layout)
    got = describe(actual, layout)
    if want == got:
        return
    for index, (line_want, line_got) in enumerate(zip(want, got)):
        if line_want != line_got:
            break
    else:
        index = min(len(want), len(got))
    context = '\n'.join(f'    python: {w}\n    native: {g}'
                        for w, g in zip(want[index:index + 3], got[index:index + 3]))
    raise NativeMismatch(f'native {name} differs from Python on {ctx.path} at line {index} '
                         f'of the description:\n{context}')


def verify_rejection(name: str, actual: ast.Module | Rejection | None, error: BaseException,
                     ctx: Any) -> None:
    """Raise :class:`NativeMismatch` when the Python step raised and the native one accepted.

    :param name: The step.
    :param actual: What :func:`run` answered for the same input.
    :param error: What the Python step raised.
    :param ctx: The pipeline context.
    """
    if isinstance(actual, ast.Module):
        raise NativeMismatch(f'native {name} accepts {ctx.path}, which Python rejects: '
                             f'{error!r}') from error
