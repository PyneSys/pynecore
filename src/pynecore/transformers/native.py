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

A native step that meets something its port does not handle raises
``Unsupported``; the step then runs in Python. Neither case changes the result,
only where it is computed.
"""
import ast
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from .builtin_shadow import Library, import_library, library_serves
from .lib_registry import lib_star_exports
from .module_property import module_properties

try:
    from . import _native
except ImportError:
    _native = None

__all__ = ['available', 'backend', 'current_steps', 'source_digest', 'run', 'verify',
           'NativeMismatch']

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


class _Host:
    """What a native step asks the interpreter, answered by the very functions the
    Python step uses, at the moment the Python step would use them."""

    def __init__(self) -> None:
        self._libraries: dict[str, Library] = {}

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
    def star_exports(module: str) -> list[str] | None:
        return lib_star_exports().get(module)


def _context(ctx: Any) -> dict[str, Any]:
    return {'path': str(ctx.path), 'pyne_mode': ctx.pyne_mode, 'source': ctx.source,
            'user_code': ctx.user_code, 'host': _Host()}


def run(names: list[str], tree: ast.Module, ctx: Any) -> ast.Module | None:
    """Run native steps in order over a module.

    :param names: Step names, each in :func:`current_steps`.
    :param tree: The module; it is not changed.
    :param ctx: The pipeline context.
    :return: The transformed module, a new object; None when a step does not
        handle the module (the caller then runs the Python steps).
    """
    assert _native is not None
    try:
        return _native.run(tree, names, _context(ctx))
    except _native.Unsupported:
        return None


def _describe(node: Any, path: str, seen: dict[int, str], out: list[str]) -> None:
    """Flatten a tree into ``path: value`` lines; a node reached a second time is
    written as a reference to the path it was first seen at."""
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
    elif isinstance(node, float):
        # ``repr`` tells -0.0 from 0.0; NaN compares by spelling
        out.append(f'{path} = float {node!r}')
    else:
        out.append(f'{path} = {type(node).__name__} {node!r}')


def describe(tree: ast.AST) -> list[str]:
    """Everything a step's result consists of, one line per value, in a stable order."""
    out: list[str] = []
    _describe(tree, '', {}, out)
    return out


def verify(name: str, expected: ast.Module, actual: ast.Module, ctx: Any) -> None:
    """Raise :class:`NativeMismatch` unless the native result equals the Python one."""
    want = describe(expected)
    got = describe(actual)
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
