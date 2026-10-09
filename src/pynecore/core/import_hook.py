from collections.abc import Callable, Iterator
from dataclasses import replace
from typing import TYPE_CHECKING, TypeVar, cast
import gc
import marshal
import os
import sys
import threading
import hashlib
import importlib.util
import importlib.machinery
import re
import unicodedata
from pathlib import Path

# Leaf modules (stdlib imports only), imported here rather than inside the loader
# on purpose: the loader runs them over pynecore's own modules, so a lazy import
# would have the hook load them through itself
from pynecore.transformers.type_erasure import erase_type_calls, has_erasure_marker
from pynecore.transformers.pipeline_names import PIPELINE_NAME, is_pipeline_name

if TYPE_CHECKING:
    import ast

    from pynecore.transformers.pine_type_table import (
        DepRecord, Diag, ModuleInterface, PineTypeTable,
    )
    from pynecore.transformers.slot_layout import ModuleLayout

__all__ = ['PYNE_RESERVED_NAME_CHAR', 'PIPELINE_DIGEST', 'security_slice_disabled',
           'security_merge_disabled', 'security_dev_skip_disabled',
           'source_starts_with_pyne',
           'compile_interface', 'PyneLoader', 'PyneImportHook']


# Module-level constant the transform pipeline bakes into every transformed module
# (see ``PyneLoader.source_to_code``). Its presence — together with a matching
# pipeline hash in ``co_consts`` — certifies a loaded code object as current
# pipeline output, so foreign or stale bytecode can be told apart and dropped.
_PYNE_SENTINEL = '__pyne_transformed__'

# The same certificate for a PLAIN module of the pynecore package, which gets the
# type erasure pass and nothing else (see ``PyneLoader._compile_plain``). It is
# paired with ``_get_type_erasure_hash``, not with the pipeline hash: what such a
# module's bytecode depends on is that one pass.
_PYNE_ERASED_SENTINEL = '__pyne_type_erased__'

#: Directory of the pynecore package, the scope of the plain-module erasure
_PACKAGE_DIR = str(Path(__file__).resolve().parent.parent) + os.sep

# Name of the constant a transformed module carries its type dependencies in, and
# the first element of that constant's tuple. A ``.pyc`` holds many tuples; this
# marker is what tells the records apart from every other one in ``co_consts``.
_PYNE_DEPS = '__pyne_type_deps__'

# Name of the constant a transformed module carries its OWN interface in, and the
# first element of that constant's tuple: ``(name, mtime_ns, size, payload)``,
# the payload being ``module_interface.interface_payload``. Like the records, it
# is read off a ``.pyc`` without executing the module (see ``compile_interface``).
_PYNE_INTERFACE = '__pyne_interface__'

#: Source fingerprints consulted while validating imported enum captures.
_PYNE_CAPTURE_DEPS = '__pyne_capture_deps__'
#: Alias the loader binds ``set_bool_na`` to in a script module's baked prologue
_PYNE_SET_BOOL_NA = '__pyne_set_bool_na·__'
#: Module-level record of the script's bool na choice, read at the module
#: boundary (``pine_export.Exported``) so a library's exported function runs
#: its own script's bool semantics, not its caller's
_PYNE_NA_BOOL = '__pyne_na_bool__'

# Recursion limit every pipeline stage runs under. The passes need about 3 Python
# frames per level of AST nesting (1516 measured on a 503-deep ``elif`` chain), so
# this covers some 3000 levels; a module nested deeper still gets a limit sized
# to its depth (see ``_with_nesting_headroom``)
_TRANSFORM_RECURSION_LIMIT = 10_000
# Python frames a pipeline pass stacks per level of AST nesting, with margin: a
# visitor's visit -> visit_<Node> -> generic_visit cycle takes 3 to 4, doubled
# for passes that recurse through helpers of their own
_FRAMES_PER_NESTING_LEVEL = 8
#: Serializes the transforms that run under a raised recursion limit
_deep_transform_lock = threading.RLock()

_T = TypeVar('_T')

# A module is Pyne code only when its docstring STARTS with ``@pyne``. Matching the
# raw source head mirrors the strict docstring check in ``source_to_code`` without
# paying for a full parse on every import. Leading comment lines are skipped so a
# PEP 723 ``# /// script`` metadata block before the docstring does not hide it.
_PYNE_HEAD_RE = re.compile(
    rb'^(?:\s*#[^\r\n]*(?:\r?\n|$))*\s*[rRbBuUfF]*("""|\'\'\'|"|\')\s*@pyne(?:\s|\1|$)')


def source_starts_with_pyne(head: bytes) -> bool:
    """Return whether a source head is a Pyne module (docstring begins with ``@pyne``).

    :param head: First bytes of the source file.
    :return: Whether the module should carry the transform sentinel.
    """
    return _PYNE_HEAD_RE.match(head) is not None


# Nearly everything the transformers inject into script scope is named with a
# Unicode middle dot: the scope-qualified state parameters and slot constants
# (``__state·main__``, ``__slot·main·x__``), the generated temporaries
# (``__st·__``, ``__cnt·0__``) and the aliased runtime helper imports
# (``__resolve_slot·__``). The separator is a legal identifier character in
# Python (``Other_ID_Continue``), so the namespace is only collision-free while
# scripts stay out of it — a script name spelled with it would shadow or clobber
# an injected one and break the emission in ways no transformer can detect. The
# few plain double-underscore names the transform emits are listed in
# ``transformers/pipeline_names.py`` and reserved the same way.
PYNE_RESERVED_NAME_CHAR = '·'


def _identifiers(tree: "ast.Module") -> "Iterator[tuple[ast.AST, str]]":
    """Every identifier of a parsed module, with the node that spells it.

    The parsed tree is what has to be checked, not the source spelling: Python
    NFKC-normalizes identifiers while parsing, so a bound name can differ from every
    spelling in the source. The token stream cannot stand in for the tree either —
    before Python 3.12 the tokenizer hands out a whole f-string as a single string
    token, hiding every name its replacement expressions bind
    (``f"{(__st·__ := x)}"``).

    :param tree: Parsed module.
    :return: (node, identifier) pairs, in walk order.
    """
    import ast

    # A template string's interpolation carries its own source text in ``str``
    # (``t"{expr}"``, Python 3.14+); the empty tuple makes the check a no-op on
    # older runtimes, where the node does not exist
    interpolation = getattr(ast, 'Interpolation', ())

    for node in ast.walk(tree):
        # A literal and an interpolation's source text are the only ``str`` payloads in
        # the tree that are not identifiers (``type_comment`` stays ``None`` unless
        # ``ast.parse`` is asked for it), so every other one can be tested blindly. That
        # covers all binding and reference forms at once — names, parameters, attributes,
        # keyword arguments, imports, ``global`` / ``nonlocal``, match captures, type
        # parameters — and keeps identifier fields added by later grammar versions
        # covered for free.
        if isinstance(node, ast.Constant):
            continue
        for field, value in ast.iter_fields(node):
            # The interpolated expression itself is a child node of its own, so the
            # names it binds or reads are still reached by the walk
            if field == 'str' and isinstance(node, interpolation):
                continue
            for name in (value if isinstance(value, list) else (value,)):
                if isinstance(name, str):
                    yield node, name


def _identifier_error(node: "ast.AST", message: str, source: str, path: Path) -> SyntaxError:
    """A ``SyntaxError`` pointing at the identifier a node spells."""
    lineno = getattr(node, 'lineno', 1)
    lines = source.splitlines()
    return SyntaxError(message, (str(path), lineno, getattr(node, 'col_offset', 0) + 1,
                                 lines[lineno - 1] if 0 < lineno <= len(lines) else None))


def _reject_reserved_names(tree: "ast.Module", source: str, path: Path) -> None:
    """Reject Pyne code that spells an identifier in the transformers' namespace.

    :param tree: Parsed module AST of ``source``.
    :param source: Full module source, used for the error location.
    :param path: Source path, used for the error location.
    :raises SyntaxError: If any identifier resolves to a name containing the separator.
    """
    # ASCII is NFKC-invariant and the separator is not ASCII, so a pure-ASCII module —
    # virtually every script — cannot produce a reserved name in any spelling
    if source.isascii():
        return
    # A separator in a bound name can only come from a character whose NFKC form
    # contains one: the separator itself, U+0387 GREEK ANO TELEIA (normalizes to one
    # outright) or U+013F / U+0140 LATIN LETTER L WITH MIDDLE DOT (decompose into
    # one). The separator is a starter that never takes part in composition, so
    # testing each distinct non-ASCII character on its own is exact; a module with
    # none of them cannot bind a reserved name, whatever else it spells outside
    # ASCII (PyneComp's header dash, a non-English comment or string)
    if not any(PYNE_RESERVED_NAME_CHAR in unicodedata.normalize('NFKC', char)
               for char in set(source) if char > '\x7f'):
        return
    for node, name in _identifiers(tree):
        if PYNE_RESERVED_NAME_CHAR in name:
            raise _identifier_error(
                node, f"'{name}' contains '{PYNE_RESERVED_NAME_CHAR}', which is reserved "
                      f"for PyneCore's internal names in Pyne code — rename the identifier",
                source, path)


def _reject_pipeline_names(tree: "ast.Module", source: str, path: Path) -> None:
    """Reject user code that spells one of the plain names the transform emits.

    :param tree: Parsed module AST of ``source``, its ``__test_`` functions removed.
    :param source: Full module source, used for the prefilter and the error location.
    :param path: Source path, used for the error location.
    :raises SyntaxError: If any identifier is a name of :data:`PIPELINE_NAME`.
    """
    # Every such identifier is spelled in the source, after NFKC normalization for
    # a module that leaves ASCII; a module whose text holds no match -- virtually
    # every script -- cannot bind one, and only a match in a string or comment is
    # left to be told apart by the tree
    text = source if source.isascii() else unicodedata.normalize('NFKC', source)
    if PIPELINE_NAME.search(text) is None:
        return
    for node, name in _identifiers(tree):
        if is_pipeline_name(name):
            raise _identifier_error(
                node, f"'{name}' is a name PyneCore's transform generates, reserved in "
                      f"Pyne code — rename the identifier", source, path)


#: Env switch turning the per-context ``main()`` slicing off (see
#: ``transformers/security_slice.py``). It changes the EMITTED tree, so it is
#: mixed into the pipeline digest below — a chart and its security children can
#: never end up on bytecode built under the other setting.
SECURITY_SLICE_ENV = 'PYNE_NO_SECURITY_SLICE'

#: Environment switch turning OFF the merge of one context group into ONE child
#: process (see ``core/script_runner.py``). Runtime-only: the emitted bytecode is
#: the same either way — a group clone is correct for a single member too — so it
#: is deliberately NOT part of :func:`pipeline_hash`, and an A/B run of it reuses
#: the very same ``.pyc``.
SECURITY_MERGE_ENV = 'PYNE_NO_SECURITY_MERGE'

#: Environment switch turning OFF the developing-round skip of a ``closed_shift``
#: context — the one whose whole expression is ``<anything>[k>=1]`` under
#: ``lookahead_on``, so its value cannot change inside an HTF period (see
#: ``core/security.py``). Runtime-only: the compile-time ``closed_shift`` flag is
#: emitted either way and only the chart's step building consults it, so this is
#: deliberately NOT part of :func:`pipeline_hash` and an A/B run of it reuses the
#: very same ``.pyc``.
SECURITY_DEV_SKIP_ENV = 'PYNE_NO_SECURITY_DEV_SKIP'

_TRUTHY = frozenset({'1', 'true', 'yes', 'on'})


def security_slice_disabled() -> bool:
    """Whether ``PYNE_NO_SECURITY_SLICE`` asks for the unsliced child behaviour.

    :return: True when no backward slice of ``main()`` may be emitted.
    """
    return os.environ.get(SECURITY_SLICE_ENV, '').strip().lower() in _TRUTHY


def security_merge_disabled() -> bool:
    """Whether ``PYNE_NO_SECURITY_MERGE`` asks for one child process per context.

    :return: True when the contexts of one group may not share a child.
    """
    return os.environ.get(SECURITY_MERGE_ENV, '').strip().lower() in _TRUTHY


def security_dev_skip_disabled() -> bool:
    """Whether ``PYNE_NO_SECURITY_DEV_SKIP`` asks for a developing round per chart bar.

    :return: True when a ``closed_shift`` context must keep every re-tick round
        inside an HTF period.
    """
    return os.environ.get(SECURITY_DEV_SKIP_ENV, '').strip().lower() in _TRUTHY


def _cache_from_source(source_path: Path) -> Path:
    """Return the cached ``.pyc`` path CPython uses for a given ``.py`` source.

    Delegates to :func:`importlib.util.cache_from_source` instead of hand-building
    ``<dir>/__pycache__/<stem>.<tag>.pyc`` so the result matches CPython exactly:
    it honours ``sys.pycache_prefix`` / ``PYTHONPYCACHEPREFIX`` (which mirrors the
    cache under a separate tree rather than a sibling ``__pycache__``) and the
    active optimization level (``.opt-1`` / ``.opt-2`` under ``-O`` / ``-OO``).
    Stale-bytecode invalidation must target the exact file CPython reads back, so
    a mismatch here would silently leave the cache untouched and the bug unfixed.

    :param source_path: Path to the ``.py`` source file.
    :return: Path to the corresponding cached bytecode file.
    """
    return Path(importlib.util.cache_from_source(str(source_path)))


#: The runtime interface the emission calls into: the helpers the transform
#: imports or the runner injects into a script module, by defining file. Emitted
#: code calls them by name, position and keyword, so their parameter lists (a
#: class: its fields) pin cached bytecode -- their bodies do not, as a cached
#: script calls whatever the current body does
_RUNTIME_ABI: tuple[tuple[str, tuple[str, ...]], ...] = (
    ('core/instance_state.py', ('__resolve_slot__', '__loop_state__', '__slot_state__',
                                '__attach_layout__', '__bind_any__', '__bind_pinned__',
                                '__bind_loop__', '__bind_slot__')),
    ('core/safe_convert.py', ('safe_div', 'safe_float', 'safe_int', 'native_int')),
    ('core/security.py', ('__sec_signal__', '__sec_write__', '__sec_read__', '__sec_wait__',
                          '__ltf_unzip__')),
    ('core/broker/models.py', ('ScriptRequirements',)),
)

_DEFINITION_RE = re.compile(r'^([ \t]*)(?:async[ \t]+def|def|class)[ \t]+(\w+)\b', re.M)


def _interface_of(node: "ast.AST") -> tuple:
    """The calling interface of a definition: parameters, or a class's fields."""
    import ast

    if isinstance(node, ast.ClassDef):
        return ('class', tuple(stmt.target.id for stmt in node.body
                               if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)),
                tuple(_interface_of(stmt) for stmt in node.body
                      if isinstance(stmt, ast.FunctionDef) and stmt.name == '__init__'))
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    args = node.args
    return ('def', tuple(arg.arg for arg in args.posonlyargs), tuple(arg.arg for arg in args.args),
            args.vararg.arg if args.vararg else None, tuple(arg.arg for arg in args.kwonlyargs),
            args.kwarg.arg if args.kwarg else None, len(args.defaults),
            tuple(default is not None for default in args.kw_defaults))


def _runtime_abi(*, whole_files: bool = False,
                 root: Path | None = None) -> dict[str, dict[str, list[tuple]]]:
    """The calling interfaces of the definitions listed in :data:`_RUNTIME_ABI`.

    Only the named definitions are parsed -- a function's header up to the colon
    that closes its signature, a class's block up to the next line indented no
    deeper -- so the cost stays far below parsing the files whole; a definition
    the cut cannot isolate falls back to the whole file.
    Every definition of a name counts (the security protocol defines chart and
    child variants of each function).

    :param whole_files: Parse every file whole instead (the reference the cut is
                        checked against).
    :param root: Package directory to read; this package when omitted.
    :return: File -> name -> the interfaces of its definitions, in file order; an
             empty list for a name that is no longer defined.
    """
    import ast
    import textwrap

    package_dir = root if root is not None else Path(__file__).parent.parent
    result: dict[str, dict[str, list[tuple]]] = {}
    for relative, names in _RUNTIME_ABI:
        try:
            text = (package_dir / relative).read_text(encoding='utf-8')
        except OSError:
            text = ''
        found: dict[str, list[tuple]] = {name: [] for name in names}
        result[relative] = found
        if whole_files:
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                        and node.name in found:
                    found[node.name].append((node.lineno, _interface_of(node)))
            for name in found:
                found[name] = [interface for _, interface in sorted(found[name])]
            continue
        lines = text.splitlines()
        whole: "ast.Module | None" = None
        line_of = 0
        position = 0
        for match in _DEFINITION_RE.finditer(text):
            name = match.group(2)
            if name not in found:
                continue
            line_of += text.count('\n', position, match.start())
            position = match.start()
            first = line_of
            if match.group(0).lstrip().startswith('class'):
                indent = len(match.group(1).expandtabs())
                last = first + 1
                while last < len(lines):
                    line = lines[last]
                    stripped = line.lstrip()
                    if stripped and len(line) - len(stripped) <= indent \
                            and not stripped.startswith((')', ']', '}')):
                        break
                    last += 1
                snippet = '\n'.join(lines[first:last])
            else:
                # The header ends at the first line that closes every bracket the
                # signature opened and ends with the colon
                depth = 0
                last = first
                while last < len(lines):
                    line = lines[last]
                    depth += sum(line.count(char) for char in '([{') \
                        - sum(line.count(char) for char in ')]}')
                    last += 1
                    if depth <= 0 and line.rstrip().endswith(':'):
                        break
                header = textwrap.dedent('\n'.join(lines[first:last]))
                snippet = header + '\n    pass'
            definition: "ast.AST | None" = None
            try:
                definition = ast.parse(textwrap.dedent(snippet)).body[0]
            except (SyntaxError, IndexError):
                pass
            if getattr(definition, 'name', None) != name:
                if whole is None:
                    whole = ast.parse(text)
                definition = next((node for node in ast.walk(whole)
                                   if getattr(node, 'name', None) == name
                                   and getattr(node, 'lineno', 0) == first + 1), None)
                if definition is None:
                    continue
            found[name].append(_interface_of(definition))
    return result


def _runtime_abi_digest() -> str:
    """Digest of :func:`_runtime_abi`, mixed into the pipeline hash."""
    return hashlib.sha256(repr(sorted(_runtime_abi().items())).encode('utf-8')).hexdigest()


_transform_pipeline_hash: str | None = None
_transform_pipeline_flag: bool | None = None
_transform_pipeline_files_hash: str | None = None


def _get_transform_pipeline_hash() -> str:
    """Return a content digest identifying the current AST transform pipeline.

    Transformed bytecode is only valid for the exact pipeline that produced it, yet
    CPython validates a ``.pyc`` solely against its source ``.py`` mtime/size — it
    cannot tell a transformed module from one compiled without the import hook
    (``pip``'s post-install ``compileall``, an IDE, a packaging step) or one left
    over by an older PyneCore. This digest is baked into every transformed module as
    ``__pyne_transformed__`` and re-checked on load; a missing or mismatched value
    forces a retransform.

    Hashing the pipeline *contents* — this module plus every file under
    ``transformers/`` (``module_properties.json`` shapes the output yet has no
    bytecode of its own) — keeps the check deterministic and immune to file mtimes,
    cache markers and read-only install locations. Every file a transformer bakes a
    value from must be hashed as well, or the constant could change while the
    digest stays put; ``core/pine_compare.py`` (the comparison tolerance the
    ``FloatToleranceTransformer`` emits as a literal) is such a file. An env
    switch that changes the emission is mixed in for the same reason
    (``PYNE_NO_SECURITY_SLICE``).

    :return: Hex digest pinning the transform pipeline.
    """
    global _transform_pipeline_hash, _transform_pipeline_flag, _transform_pipeline_files_hash
    flag = security_slice_disabled()
    if _transform_pipeline_hash is not None and _transform_pipeline_flag == flag:
        return _transform_pipeline_hash
    files_hash = _transform_pipeline_files_hash
    if files_hash is None:
        # This module runs the reserved-name check, the edge gate and the two
        # halves (their steps live in ``transformers/pipeline.py``, hashed with the
        # rest of that directory); ``pine_compare`` holds a constant the pipeline
        # bakes into the emitted bytecode
        files = [Path(__file__), Path(__file__).parent / "pine_compare.py"]
        # The call-inlining pass copies these bodies into the emission, so a
        # wrapper edit must invalidate cached script bytecode just like a
        # transformer edit does (see transformers/call_inline.py)
        lib_dir = Path(__file__).parent.parent / "lib"
        files.extend([lib_dir / "math.py", lib_dir / "array.py",
                      Path(__file__).parent / "inline_support.py"])
        # The constant-folding pass evaluates literal math calls through these
        # implementations and bakes the results into the emission; the native
        # twins are bit-exact to them, so the Python sources pin the values
        files.extend([Path(__file__).parent / "fdlibm.py",
                      Path(__file__).parent / "pine_math.py"])
        # ... and they reach it through the Pine number types, ``na`` and the
        # overload dispatch (``math.round``), whose behaviour shapes the folded value
        package_dir = Path(__file__).parent.parent
        files.extend([package_dir / "types" / "pine_types.py", package_dir / "types" / "na.py",
                      Path(__file__).parent / "overload.py"])
        transformers_dir = Path(__file__).parent.parent / "transformers"
        try:
            files.extend(transformers_dir.iterdir())
        except OSError:
            pass
        digest = hashlib.sha256()
        for f in sorted(files, key=lambda p: p.name):
            try:
                if f.is_file():
                    digest.update(f.name.encode('utf-8'))
                    digest.update(f.read_bytes())
            except OSError:
                pass
        digest.update(_runtime_abi_digest().encode('utf-8'))
        files_hash = digest.hexdigest()
        _transform_pipeline_files_hash = files_hash

    # The switch is memoized WITH the digest instead of being folded into the
    # file scan: flipping it mid-process must produce a different hash at once,
    # so bytecode built under the other setting can never be loaded as current.
    mixed = hashlib.sha256(files_hash.encode('utf-8'))
    mixed.update(b'1' if flag else b'0')
    _transform_pipeline_hash = mixed.hexdigest()[:16]
    _transform_pipeline_flag = flag
    return _transform_pipeline_hash


_type_erasure_hash: str | None = None


def _get_type_erasure_hash() -> str:
    """Return a content digest of the type erasure pass.

    Bytecode of a plain pynecore module is current when it was compiled through
    this exact pass; the rest of the pipeline never touches such a module, so its
    cache must not be dropped whenever some other transformer changes.

    :return: Hex digest pinning the pass.
    """
    global _type_erasure_hash
    if _type_erasure_hash is None:
        source = Path(__file__).parent.parent / "transformers" / "type_erasure.py"
        try:
            data = source.read_bytes()
        except OSError:
            data = b''
        _type_erasure_hash = hashlib.sha256(b'type_erasure:' + data).hexdigest()[:16]
    return _type_erasure_hash


def _in_package(source_path: str) -> bool:
    """Whether a source file belongs to the pynecore package.

    :param source_path: Path of the module's source.
    :return: True for a module the loader runs the type erasure over.
    """
    return os.path.realpath(source_path).startswith(_PACKAGE_DIR)


#: The pipeline every transformed module in this process was produced by, taken
#: once at import. It is the public face of ``_get_transform_pipeline_hash``:
#: PyneAOT writes it into its bundle so a bundle built by one pipeline can be
#: told apart from a checkout running another. The function stays the live
#: value the loader itself reads, which is what lets a test simulate a
#: different pipeline without rewriting this constant.
PIPELINE_DIGEST: str = _get_transform_pipeline_hash()


def _module_mode(tree: "ast.Module") -> tuple[bool, str | None]:
    """Read the ``@pyne`` marker and mode word out of a module's docstring.

    Strict on purpose: the docstring must START with the token, so an innocuous
    mention inside a non-script library module's docstring is not a marker.

    :param tree: The parsed module.
    :return: Whether the module is Pyne code, and its mode word when it names one
             (``'lib'`` for the builtin machines, ``'edge'`` for compiler output,
             None for a hand-written script).
    """
    import ast

    first = tree.body[0] if tree.body else None
    if not (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)):
        return False, None
    magic = re.match(r'\s*@pyne(?:[ \t]+(?P<mode>\w+))?(\s|$)', first.value.value)
    if magic is None:
        return False, None
    return True, magic.group('mode')


#: Which typed diagnostics a structural one already covers, by reason
_STRUCTURAL_COVERS: dict[str, frozenset[str]] = {
    'edge-call': frozenset({'unknown-call', 'unknown-lib', 'unknown-return', 'bad-call'}),
    'edge-name': frozenset({'unknown-name', 'function-value', 'unknown-lib-name',
                            'unknown-field', 'unknown-class'}),
    'edge-syntax': frozenset({'not-pine', 'unknown-op', 'unknown-index', 'not-series'}),
    'edge-lambda': frozenset({'not-pine'}),
    'edge-subscript': frozenset({'unknown-index', 'not-pine'}),
}


def _repeats(structural: list['Diag'], diag: 'Diag') -> bool:
    """
    Whether a typed diagnostic only repeats a structural one.

    It does when it stands at the structural diagnostic's position, or inside
    the expression that one rejected, AND says the kind of thing the
    rejection already says: a construct that is not Pine has no type, and
    neither do its parts. Another problem at the same place -- a name nothing
    binds, used inside a rejected operator -- stands.
    """
    if diag.origin is None:
        return False
    at = (diag.line, diag.col)
    for found in structural:
        if found.origin is None \
                or diag.origin.reason not in _STRUCTURAL_COVERS.get(found.origin.reason, ()):
            continue
        if at == (found.line, found.col):
            return True
        if found.end_line and (found.line, found.col) <= at <= (found.end_line, found.end_col):
            return True
    return False


def _script_bool_na(tree: "ast.Module", path: Path) -> bool | None:
    """
    Read the script's bool na choice off its ``script.*`` decorator.

    The choice must hold before the module body runs -- a UDT field default or
    a ``bool b = na`` at module level builds its na at import -- so the loader
    reads the keyword statically and bakes the call into the module prologue.
    The decorator is resolved through the module's own import bindings, so
    ``@script.indicator``, ``@lib.script.indicator``, ``@pynecore.lib.script.indicator``,
    an alias of ``script`` and an ``import pynecore.lib as X`` alias are all recognized.

    :param tree: The parsed module, before analysis.
    :param path: The module's path, for the error location.
    :return: The ``na_bool`` keyword's value (False when the decorator omits it), or
             None for a module without a ``script.indicator/strategy/library`` decorator.
    :raises SyntaxError: When ``na_bool`` is not a literal ``True`` or ``False``, or a
        script decorator stands on a function other than ``main``, or ``main`` is
        decorated twice.
    """
    import ast

    lib_names: set[str] = set()
    script_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            # ``from pynecore import lib as X`` is refused by the import normalizer
            for alias in node.names:
                if node.module == 'pynecore.lib' and alias.name == 'script':
                    script_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'pynecore.lib' and alias.asname is not None:
                    lib_names.add(alias.asname)

    def is_script_decorator(func: ast.expr) -> bool:
        chain: list[str] = []
        while isinstance(func, ast.Attribute):
            chain.append(func.attr)
            func = func.value
        if not isinstance(func, ast.Name) or not chain:
            return False
        chain.append(func.id)
        chain.reverse()
        if chain[-1] not in ('indicator', 'strategy', 'library'):
            return False
        head = chain[:-1]
        return (head == ['script'] or head[0] in script_names and len(head) == 1
                or head == ['pynecore', 'lib', 'script']
                or len(head) == 2 and head[1] == 'script' and (head[0] == 'lib' or head[0] in lib_names))

    # A module has ONE entry point: the script decorator stands on ``main`` --
    # that is the function the runner runs, and an imported library's own
    # entry is the ``main`` of its own module
    entry: ast.Call | None = None
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not is_script_decorator(decorator.func):
                continue
            if node.name != 'main':
                raise SyntaxError(
                    f"the script decorator belongs on 'main', not on '{node.name}'",
                    (str(path), decorator.lineno, decorator.col_offset + 1, None))
            if entry is not None:
                raise SyntaxError(
                    "a module has one script decorator: 'main' is defined twice",
                    (str(path), decorator.lineno, decorator.col_offset + 1, None))
            entry = decorator
            break
    if entry is None:
        return None
    for keyword in entry.keywords:
        if keyword.arg != 'na_bool':
            continue
        value = keyword.value
        if isinstance(value, ast.Constant) and isinstance(value.value, bool):
            return value.value
        # The loader must know the answer before the module runs
        raise SyntaxError(
            "na_bool must be a literal True or False: the script's bool "
            "semantics are fixed before its module body runs",
            (str(path), value.lineno, value.col_offset + 1, None))
    return False


def _nesting_depth(tree: "ast.AST") -> int:
    """The deepest node level of a tree, measured without recursion.

    :param tree: The tree to measure.
    :return: Number of edges on the longest root-to-leaf path.
    """
    # Lazy for the same reason the transformers are: the transformers package is
    # itself loaded through this hook
    from pynecore.transformers.ast_walk import iter_child_nodes

    deepest = 0
    stack: list[tuple[ast.AST, int]] = [(tree, 0)]
    while stack:
        node, depth = stack.pop()
        if depth > deepest:
            deepest = depth
        depth += 1
        stack.extend((child, depth) for child in iter_child_nodes(node))
    return deepest


def _with_nesting_headroom(run: Callable[["ast.Module"], _T], tree: "ast.Module",
                           source: str) -> _T:
    """Run a pipeline stage over a tree, however deeply its source nests.

    The passes walk the tree recursively, so the Python stack they need grows
    with the module's nesting depth: a Pine ``switch`` with hundreds of arms
    compiles to an ``elif`` chain hundreds of levels deep, beyond what the
    interpreter's default recursion limit admits. The stage therefore runs
    under :data:`_TRANSFORM_RECURSION_LIMIT`, which fits any realistic module.
    Measuring every module up front would cost a full extra walk on each
    transform, so only a module that still overflows is measured, re-parsed
    (the failed run left its tree half rewritten) and run again under a limit
    sized to its depth. The raised limits hold for the stage only.

    :param run: The stage; it is handed the tree to transform.
    :param tree: The parsed module.
    :param source: Its source, re-parsed for the second run.
    :return: What the stage returns.
    """
    # The limit is process-wide: the lock keeps a concurrent transform from
    # restoring it under this one. Reentrant, because analysing a dependency
    # from inside a stage lands here again.
    with _deep_transform_lock:
        limit = sys.getrecursionlimit()
        base = max(limit, _TRANSFORM_RECURSION_LIMIT)
        sys.setrecursionlimit(base)
        # A transform allocates hundreds of thousands of short-lived AST nodes,
        # dicts and lists that reference counting frees on its own; the cyclic
        # collector, triggered by the allocation count alone, would rescan the
        # growing young generations over and over. The outermost stage switches
        # it off for its duration and back on only if it was on
        collect = gc.isenabled()
        gc.disable()
        try:
            try:
                return run(tree)
            except RecursionError:
                pass

            import ast

            fresh = ast.parse(source)
            sys.setrecursionlimit(base + _nesting_depth(fresh) * _FRAMES_PER_NESTING_LEVEL)
            return run(fresh)
        finally:
            sys.setrecursionlimit(limit)
            if collect:
                gc.enable()


def _analyse_tree(tree: "ast.Module", source: str, path: Path,
                  pyne_mode: str | None) -> "ast.Module":
    """Run the pipeline up to and including the Pine type pass.

    This half only ANALYSES: it normalizes the tree into the form the type pass
    reads and stamps the types onto it, without emitting any of the state
    plumbing the second half does. A dependency transformed for another module's
    lookup stops here when it reaches back into a module still under analysis
    (see :func:`compile_interface`).

    :param tree: The parsed module; it is transformed in place where the passes do so.
    :param source: Full module source, for the reserved-name error location.
    :param path: Source path; the script / lib profile is picked from it.
    :param pyne_mode: The module's mode word, None for a hand-written script.
    :return: The analysed, type-stamped tree.
    """
    import ast

    # The transformers own the middle-dot namespace; a script that spells a
    # name in it is rejected here, before anything is injected
    _reject_reserved_names(tree, source, path)

    # Remove test cases from the output, because they can coorupt the output
    transformed = tree
    # Source path for the transformers (SecurityTransformer hashes it into
    # the per-module sec ids, so security contexts stay unique across the
    # script and its imported library modules). Resolved so the chart
    # process and its security children derive identical ids.
    transformed._module_file_path = str(path.resolve())  # type: ignore[attr-defined]
    transformed.body = [node for node in transformed.body
                        if not (isinstance(node, ast.FunctionDef)
                                and node.name.startswith('__test_') and node.name.endswith('__'))]
    user_code = not path.is_relative_to(Path(__file__).parent.parent)
    # pynecore's own lib modules ARE the runtime side of these names
    if user_code:
        _reject_pipeline_names(transformed, source, path)

    # The edge gate reads the tree AS WRITTEN: what follows injects plumbing
    # no profile should judge. Its findings join the type diagnostics below
    from pynecore.transformers.pine_edge_gate import (
        diag_dump_enabled, gate_module, gated, render_diags, strict_enabled,
    )
    from pynecore.transformers.pine_type_table import PineTypeError
    from pynecore.transformers.pine_type_transformer import module_table

    edge = gated(pyne_mode)
    structural = gate_module(transformed) if edge else []

    # The steps, their order and the reason for it live in transformers/pipeline.py
    from pynecore.transformers.pipeline import ANALYSIS, PipelineContext, run_steps

    ctx = PipelineContext(path=path, pyne_mode=pyne_mode, source=source, user_code=user_code,
                          analyse=compile_interface,
                          pipeline_hash=_get_transform_pipeline_hash())
    transformed = run_steps(ANALYSIS, transformed, ctx)
    table = module_table(transformed)
    if table is not None:
        # One node, one report: where the structural half names the
        # construct, the typed half would only add that it has no type. Only
        # a typed diagnostic ABOUT that construct is the repeat; another
        # problem at the same position stands
        table.diags = sorted(
            structural + [diag for diag in table.diags if not _repeats(structural, diag)],
            key=lambda diag: (diag.line, diag.col))
        if table.diags and diag_dump_enabled():
            sys.stderr.write(render_diags(table.diags, str(path)) + '\n')
        if table.diags and edge and strict_enabled():
            # One code path, two modes: a hand-written script keeps running
            # and keeps the list; an edge module is a promise, and the first
            # thing that breaks it is the error
            first = table.diags[0]
            lines = source.splitlines()
            text = lines[first.line - 1] if 0 < first.line <= len(lines) else None
            raise PineTypeError.from_diag(first, str(path), text)
    return transformed


def _lower_tree(tree: "ast.Module", path: Path, pyne_mode: str | None,
                *, emit_layout: bool = True,
                na_bool: bool = False) -> "tuple[ast.Module, ModuleLayout]":
    """Run the pipeline from the type pass to the finished emission.

    This half EMITS: it turns the analysed tree into the state-plumbed form the
    runtime executes, allocates the module's slot layout and fixes up the
    synthetic locations.

    The allocated layout is returned beside the tree: it is the typed record of
    what every slot is, and the ``__pyne_slot_layout__`` literal ``apply_layout``
    emits is one consumer's materialization of it, not its definition. A consumer
    that builds the state vector itself takes the object and turns the
    materialization off.

    :param tree: The analysed, type-stamped tree.
    :param path: Source path; the script / lib profile is picked from it.
    :param pyne_mode: The module's mode word, None for a hand-written script.
    :param emit_layout: Whether to run ``apply_layout``, which materializes the
        layout for CPython: the slot dict literal, the hidden state parameters
        and the ``__pyne_layout__`` attachments.
    :param na_bool: Whether the module keeps Pine's three-state bool, which
        decides what a comparison with an na operand answers (see
        ``FloatToleranceTransformer``).
    :return: The tree the compiler is handed, and the module's slot layout.
    """
    import ast

    from pynecore.transformers.pipeline import LOWERING, PipelineContext, run_steps
    from pynecore.transformers.slot_layout import ModuleLayout

    # Shared slot allocator of the module (see slot_layout.py); the
    # state-contributing steps fill it, apply_layout emits it
    slot_layout = ModuleLayout(compacted_series=pyne_mode == 'lib')
    ctx = PipelineContext(path=path, pyne_mode=pyne_mode,
                          user_code=not path.is_relative_to(Path(__file__).parent.parent),
                          analyse=compile_interface,
                          pipeline_hash=_get_transform_pipeline_hash(),
                          na_bool=na_bool, emit_layout=emit_layout, slot_layout=slot_layout)
    transformed = run_steps(LOWERING, tree, ctx)

    # Debug output if requested. The pretty dump and the saved copy go
    # through the display rewrite (named index constants instead of
    # literal slot indexes); the RAW dump stays the exact emission —
    # the AST golden tests compare against it.
    if os.environ.get('PYNE_AST_DEBUG'):
        from pynecore.transformers.display_rewrite import display_dump
        print("-" * 100)
        print(f"Transformed {path}:")
        try:
            from rich.syntax import Syntax  # type: ignore
            from rich import print as rprint  # type: ignore
            rprint(Syntax(display_dump(transformed, slot_layout), "python",
                          word_wrap=True, line_numbers=False))
        except ImportError:
            print(display_dump(transformed, slot_layout))
        print("-" * 100)
    elif raw_filter := os.environ.get('PYNE_AST_DEBUG_RAW'):
        # '1' dumps every transformed module; any other value is a source
        # path filter so a capture is not polluted by modules imported
        # during the transform (callee resolution imports lib submodules)
        if raw_filter == '1' or Path(raw_filter).resolve() == path.resolve():
            print(ast.unparse(transformed))

    if os.environ.get('PYNE_AST_SAVE'):
        from pynecore.transformers.display_rewrite import display_dump
        Path("/tmp/pyne").mkdir(parents=True, exist_ok=True)

        with open(f"/tmp/pyne/{path.stem}.py", "w") as f:
            f.write(display_dump(transformed, slot_layout))

    return transformed, slot_layout


def _transform_module(tree: "ast.Module", source: str, path: Path, pyne_mode: str | None,
                      na_bool: bool, fingerprint: tuple[int, int] | None) \
        -> "tuple[PineTypeTable | None, ModuleInterface | None, ast.Module]":
    """Run both halves of the pipeline, publishing what the module exports on the way.

    :param tree: The parsed module; it is transformed in place where the passes do so.
    :param source: Full module source.
    :param path: Source path.
    :param pyne_mode: The module's mode word, None for a hand-written script.
    :param na_bool: Whether the module keeps Pine's three-state bool.
    :param fingerprint: The ``(mtime_ns, size)`` the source was read under, None
        when no such pairing could be had.
    :return: The type table, the interface with its routes settled (its
        dependency records are not, see :func:`_settle_deps`), and the lowered tree.
    """
    # Lazy for the same reason the transformers are: this module is loaded
    # through the hook itself, so importing it at module level would re-enter a
    # half-initialized package
    from pynecore.transformers.module_interface import reached_back
    from pynecore.transformers.pine_type_transformer import module_table

    analysed = _analyse_tree(tree, source, path, pyne_mode)
    table = module_table(analysed)
    published = None if table is None else _publish(analysed, table, path, fingerprint)
    # A dependency a lookup transforms stops here when its analysis reached back
    # into a module still under analysis (see ``_compile_into_cache``)
    if reached_back(str(path)):
        raise _ReachedBack
    published, lowered = _lower_published(analysed, published, path, pyne_mode, na_bool)
    return table, published, lowered


def _publish(analysed: "ast.Module", table: "PineTypeTable", path: Path,
             fingerprint: tuple[int, int] | None) -> "ModuleInterface":
    """Publish what an analysed module exports, for every module that imports it.

    In this process through the registry, across processes through the constant
    the loader bakes into the .pyc. Read off the ANALYSED tree, before the lowering:
    the isolation pass prepends a state parameter to every script function and the
    series pass rewrites the annotations, so a signature taken afterwards is the
    emission's, not the module's. It is registered right away, so a module the
    lowering imports that imports this one back finds it.

    :param analysed: The analysed tree.
    :param table: Its type table.
    :param path: Source path.
    :param fingerprint: The ``(mtime_ns, size)`` the source was read under, None
        when no such pairing could be had.
    :return: The interface, its routes not settled yet.
    """
    from pynecore.transformers.module_interface import NO_FINGERPRINT, build_interface, register

    published = build_interface(analysed, table, str(path.resolve()),
                                NO_FINGERPRINT if fingerprint is None else fingerprint)
    register(published)
    return published


def _lower_published(analysed: "ast.Module", published: "ModuleInterface | None", path: Path,
                     pyne_mode: str | None, na_bool: bool) \
        -> "tuple[ModuleInterface | None, ast.Module]":
    """Lower an analysed module, and settle the routes of what it published.

    :param analysed: The analysed tree; it is lowered in place where the passes do so.
    :param published: What :func:`_publish` published for it, None when nothing.
    :param path: Source path.
    :param pyne_mode: The module's mode word, None for a hand-written script.
    :param na_bool: Whether the module keeps Pine's three-state bool.
    :return: The interface with its routes, registered again, and the lowered tree.
    """
    from pynecore.transformers.module_interface import register, routes_digest, routes_shape

    # The routes follow the module as written, before the lowering adds plumbing
    shape = '' if published is None else routes_shape(analysed)
    lowered, layout = _lower_tree(analysed, path, pyne_mode, na_bool=na_bool)
    if published is not None:
        published = replace(published, routes=routes_digest(shape, lowered, layout))
        register(published)
    return published, lowered


def _settle_deps(table: "PineTypeTable", interface: "ModuleInterface") -> "ModuleInterface":
    """Settle the routes of a transformed module's dependency records.

    The type pass records each dependency off whatever interface it found, and one
    the analysis alone published has no routes (see ``module_interface.settle``).
    The table's records are what the loader bakes, and the interface's are what a
    dependent of THIS module inherits, so both are replaced.

    :param table: The module's type table; its records are replaced in place.
    :param interface: The interface the transform published.
    :return: The interface with the settled records, registered.
    """
    from pynecore.transformers.module_interface import register, settle

    if not table.deps:
        return interface
    pipeline_hash = _get_transform_pipeline_hash()
    table.deps = {path: settle(record, compile_interface, pipeline_hash)
                  for path, record in table.deps.items()}
    interface = replace(interface, deps=dict(table.deps))
    register(interface)
    return interface


class _ReachedBack(Exception):
    """A dependency's transform stopped after its analysis (``module_interface.reached_back``)."""


#: Resolved paths of the Pyne modules whose transform is running in this process.
#: Such a module published its interface before its lowering started, and a
#: transform of it from inside its own would only race the outer one's ``.pyc``
#: (see ``compile_interface``).
_transforming: set[str] = set()


def compile_interface(path: str, pipeline_hash: str) -> "ModuleInterface | None":
    """One Pyne module's interface for another module's transform, executing nothing of it.

    The ``Analyser`` of every type pass and dependency check: it is asked when no
    transform of this process published the interface. The module's own ``.pyc``
    answers when it is one importing the module would run as it is -- the given
    pipeline's, of the current source, built against dependencies that still say
    the same thing -- without executing anything: the interface is a constant
    baked into it (``_PYNE_INTERFACE``). Otherwise the module is transformed into
    its ``.pyc`` here, by the same transform its import would run, so that import
    finds the work done and reuses it (see :func:`_compile_into_cache`).

    The ``.pyc`` is read under the module's analysing mark, because checking its
    dependency records may reach back here for this very module, and a module
    under analysis answers None -- the end of the cycle. The transform runs
    outside it, the way the loader transforms a module: the analysis marks
    itself, and the lowering imports what the module calls -- one that imports
    the module back has to find the interface it publishes before lowering.

    :param path: Resolved path to the ``.py`` source.
    :param pipeline_hash: Digest of the pipeline the ``.pyc`` must come from.
    :return: The interface, its routes settled unless its transform reached back
             into a module under analysis; None when the file is not readable, not
             Pyne code, does not transform, or is being transformed already.
    """
    # Lazy for the same reason the transformers are: this module is loaded
    # through the hook itself, so importing it at module level would re-enter a
    # half-initialized package
    from pynecore.transformers.module_interface import analysing_scope

    if path in _transforming:
        return None
    try:
        with open(path, 'rb') as f:
            head = f.read(4096)
    except OSError:
        return None
    if not source_starts_with_pyne(head):
        return None
    source_path = Path(path)
    with analysing_scope(path):
        interface = _cached_interface(source_path, pipeline_hash)
    if interface is not None:
        return interface
    return _compile_into_cache(source_path)


def _uint32(value: int) -> bytes:
    """One field of a ``.pyc`` header (PEP 552): little-endian, modulo 2**32."""
    return (value & 0xFFFFFFFF).to_bytes(4, 'little')


def _cached_interface(source_path: Path, pipeline_hash: str) -> "ModuleInterface | None":
    """The interface a module's ``.pyc`` carries, when an import would run that ``.pyc``.

    Every check an import makes is made here, in the same order: CPython's own
    (the timestamp header against the source's stat), the pipeline certificate,
    and the dependency records ``get_code`` re-checks -- so an interface read
    here is never one whose module the import then transforms again. The baked
    fingerprint is checked on top: the header only keeps whole seconds.

    :param source_path: Path to the ``.py`` source.
    :param pipeline_hash: Digest of the pipeline the ``.pyc`` must come from.
    :return: The interface, or None when there is no such ``.pyc``.
    """
    from pynecore.transformers.module_interface import interface_from_payload

    try:
        stat = os.stat(source_path)
        data = _cache_from_source(source_path).read_bytes()
    except OSError:
        return None
    if data[:4] != importlib.util.MAGIC_NUMBER or data[4:8] != _uint32(0) \
            or data[8:12] != _uint32(int(stat.st_mtime)) or data[12:16] != _uint32(stat.st_size):
        return None
    try:
        code = marshal.loads(data[16:])
    except (EOFError, ValueError, TypeError):
        return None
    if _PYNE_SENTINEL not in code.co_names or pipeline_hash not in code.co_consts:
        return None
    baked = next((const for const in code.co_consts
                  if isinstance(const, tuple) and const and const[0] == _PYNE_INTERFACE), None)
    if baked is None or (baked[1], baked[2]) != (stat.st_mtime_ns, stat.st_size):
        return None
    if not (_capture_deps_current(code) and _deps_current(code, pipeline_hash)):
        return None
    return interface_from_payload(str(source_path), baked[3],
                                  {record.path: record for record in _baked_deps(code)},
                                  (baked[1], baked[2]))


def _compile_into_cache(source_path: Path) -> "ModuleInterface | None":
    """Transform a module the way importing it would, and leave its ``.pyc`` for the import.

    The source is transformed and compiled through :class:`PyneLoader`, and the
    bytecode is written where, and how, ``SourceLoader.get_code`` writes it -- a
    timestamp ``.pyc`` with the source's permissions, not at all under
    ``sys.dont_write_bytecode``, silently not into an unwritable cache -- so the
    import that follows accepts it as its own. The header has to describe the
    bytes that were transformed, the ones the transform read as one pair with
    their fingerprint. Where nothing can be written, what the transform
    registered still answers for the rest of the process.

    A transform whose analysis reached back into a module under analysis is not
    what the import will produce (see ``module_interface.Compile``): it stops
    once its analysis is done, and answers with what that published -- the types,
    without routes and without any bytecode.

    :param source_path: Path to the ``.py`` source.
    :return: The interface the transform published, or None when it published
             none or the source does not transform.
    """
    from pynecore.transformers.module_interface import compiling, registered

    loader = PyneLoader(source_path.stem, str(source_path))
    try:
        data = loader.get_data(str(source_path))
    except OSError:
        return None
    with compiling(str(source_path)):
        try:
            code = loader.source_to_code(data, str(source_path))
        except _ReachedBack:
            return registered(str(source_path))
        except (SyntaxError, ValueError, UnicodeDecodeError, RecursionError):
            # An untransformable dependency is not a failure to report here: the
            # module that actually imports it raises the real error, with the
            # real traceback. All this can say is that it publishes nothing.
            return None
    interface = registered(str(source_path))
    if interface is None or sys.dont_write_bytecode:
        return interface
    # The header has to describe the bytes that were transformed, so it is only
    # written while the file is still at their fingerprint -- which no file is
    # at for bytes that had none to give
    try:
        stat = os.stat(source_path)
    except OSError:
        return interface
    if (stat.st_mtime_ns, stat.st_size) != (interface.mtime_ns, interface.size):
        return interface
    loader.set_data(str(_cache_from_source(source_path)),
                    importlib.util.MAGIC_NUMBER + _uint32(0) + _uint32(int(stat.st_mtime))
                    + _uint32(stat.st_size) + marshal.dumps(code),
                    _mode=stat.st_mode | 0o200)
    return interface


def _baked_deps(code) -> "tuple[DepRecord, ...]":
    """Read the dependency records a transformed module carries in its bytecode.

    The records are a folded tuple constant rather than module-level data, so they
    can be read off a ``.pyc`` without importing — which is the point: the modules
    a dependency check consults are typically not loaded yet.

    :param code: The module's code object.
    :return: One record per dependency, empty when the module has none.
    """
    # Lazy for the same reason the transformers are: the transformers package is
    # itself loaded through this hook, so a module-level import would re-enter a
    # half-initialized package
    from pynecore.transformers.pine_type_table import DepRecord

    for const in code.co_consts:
        if isinstance(const, tuple) and const and const[0] == _PYNE_DEPS:
            return tuple(DepRecord(path=record[0], mtime_ns=record[1],
                                   size=record[2], digest=record[3], routes=record[4])
                         for record in const[1:])
    return ()


def _deps_current(code, pipeline_hash: str) -> bool:
    """Whether every module this bytecode was built against still says the same thing.

    The same thing in both senses the emission read it in: the signatures the
    types were derived from, and the routes every call into the module was
    emitted on.

    :param code: The module's code object.
    :param pipeline_hash: Digest of the current transform pipeline.
    :return: True while the cached bytecode is still valid.
    """
    records = _baked_deps(code)
    if not records:
        return True

    # Lazy, as above -- and skipped entirely for a module with no dependencies
    from pynecore.transformers.module_interface import dep_current

    return all(dep_current(record, compile_interface, pipeline_hash, settled=True)
               for record in records)


def _capture_deps_current(code) -> bool:
    """Revalidate source files used to prove imported enum captures constant.

    :param code: The cached module code object.
    :return: Whether all consulted source fingerprints still match.
    """
    for constant in code.co_consts:
        if not isinstance(constant, tuple) or not constant or constant[0] != _PYNE_CAPTURE_DEPS:
            continue
        for path, mtime_ns, size in constant[1:]:
            try:
                stat = Path(path).stat()
            except OSError:
                return False
            if (stat.st_mtime_ns, stat.st_size) != (mtime_ns, size):
                return False
    return True


class PyneLoader(importlib.machinery.SourceFileLoader):
    """Loader that handles AST transformation"""

    def get_code(self, fullname: str):
        """Retransform cached bytecode not produced by the current transform pipeline.

        CPython validates a cached ``.pyc`` only against its source ``.py`` mtime and
        size, so it cannot distinguish a transformed ``@pyne`` module from one compiled
        without the import hook (``pip``'s post-install ``compileall``, an IDE, a
        packaging step) or one left over by an older pipeline — all of them load as
        "valid" and silently run the wrong bytecode. Every transformed module carries a
        ``__pyne_transformed__ = <pipeline hash>`` sentinel baked into its code object;
        if the loaded bytecode lacks it or the hash is stale, the ``.pyc`` is dropped and
        the source is retransformed. The check is content-based, so it holds regardless
        of file mtimes, cache markers or a read-only install location.

        The same blind spot applies across modules: a module's types are derived from
        the INTERFACES it imports, its calls into them are routed by whether the
        callee keeps state, and CPython's check sees none of it. So a module that was
        built against others also carries their state in a ``__pyne_type_deps__``
        constant, re-checked here before the cache is accepted.

        :param fullname: Fully-qualified module name being loaded.
        :return: The compiled code object (retransformed if the cache was foreign,
                 stale, or built against a dependency that has changed).
        """
        source_path = self.get_filename(fullname)
        code = super().get_code(fullname)

        try:
            with open(source_path, 'rb') as f:
                # Large enough to cover a PEP 723 metadata block before the docstring
                head = f.read(4096)
        except OSError:
            head = b''

        # ``get_code`` is typed Optional, but a real source file always yields a code
        # object — the ``None`` guard just narrows the type for the checks below.
        if code is None:
            return code

        if not source_starts_with_pyne(head):
            # A plain module of the pynecore package is compiled through the type
            # erasure pass. Foreign bytecode of one runs correctly, only slower, and
            # pip's post-install ``compileall`` would make that the permanent state
            # of every installation — so it is told apart like a transformed module.
            erasure_hash = _get_type_erasure_hash()
            if _PYNE_ERASED_SENTINEL in code.co_names and erasure_hash in code.co_consts:
                return code
            if not _in_package(source_path):
                return code
            try:
                data = self.get_data(source_path)
            except OSError:
                return code
            if not has_erasure_marker(data):
                return code
            return self._retransform(fullname, source_path, _PYNE_ERASED_SENTINEL,
                                     erasure_hash)

        pipeline_hash = _get_transform_pipeline_hash()
        if _PYNE_SENTINEL not in code.co_names or pipeline_hash not in code.co_consts:
            return self._retransform(fullname, source_path, _PYNE_SENTINEL, pipeline_hash)

        # The pipeline is current, but the types this module was compiled against
        # live in OTHER modules; an edit to one of their interfaces makes this
        # bytecode wrong while CPython still sees a valid cache for it.
        if _capture_deps_current(code) and _deps_current(code, pipeline_hash):
            return code
        return self._retransform(fullname, source_path, _PYNE_SENTINEL, pipeline_hash)

    def _retransform(self, fullname: str, source_path: str, sentinel: str, digest: str):
        """Drop bytecode the checks rejected and produce the current transform.

        :param fullname: Fully-qualified module name being loaded.
        :param source_path: Path to the module's ``.py`` source.
        :param sentinel: Name of the certificate the recompiled code object has to carry.
        :param digest: Digest that certificate has to be paired with.
        :return: The compiled code object.
        """
        # Foreign, stale or dependency-invalidated bytecode slipped past CPython's
        # mtime/size check — drop it and let the loader recompile, refreshing the
        # cache when the dir is writable.
        try:
            _cache_from_source(Path(source_path)).unlink()
        except OSError:
            pass  # no cached bytecode, or a read-only cache dir: nothing to drop
        code = super().get_code(fullname)
        if code is None or (sentinel in code.co_names and digest in code.co_consts):
            return code

        # The stale ``.pyc`` could not be removed (read-only / locked cache) and still
        # masks the source. Compile straight from source so the correct bytecode runs
        # regardless; caching is skipped this load — correctness wins over the cache.
        return self.source_to_code(self.get_data(source_path), source_path)

    # noinspection PyMethodOverriding
    def source_to_code(self, data: bytes | str, path: str, *, _optimize: int = -1):
        """Transform source to code if needed"""
        path: Path = Path(path)

        # Fast prefilter: require @pyne as a standalone token, not just any substring.
        # Compiled Pyne code always has it as the first non-whitespace content of the
        # module docstring, either multi-line (`"""\n@pyne\n…"""`) or single-line
        # (`"""@pyne"""`); the latter puts the closing quote right after the token, so a
        # quote must terminate the match alongside whitespace / end-of-input. A loose
        # check would AST-transform ordinary modules that merely *mention* @pyne in a
        # docstring (e.g. standalone.py); the strict docstring check below still gates it.
        data_str = data.decode('utf-8') if isinstance(data, bytes) else data
        if not re.search(r'@pyne(\s|["\']|$)', data_str):
            return self._compile_plain(data, data_str, path, _optimize)

        import ast

        tree = ast.parse(data_str)

        # Strict check: the module docstring must START with @pyne (whitespace-stripped),
        # followed by whitespace or end of string. Substring matches don't count — they
        # would catch innocuous mentions inside docstrings of non-script library modules.
        # The optional word after the token is the module's mode: 'lib' marks the
        # builtin machines shipped with PyneCore (their series are na-compacted
        # windows), 'edge' the compiler's output, nothing at all a hand-written
        # script.
        is_pyne_module, pyne_mode = _module_mode(tree)

        if is_pyne_module:
            # Lazy for the same reason the transformers are: this module is
            # loaded through the hook itself, so importing it at module level
            # would re-enter a half-initialized package. It is also why the
            # pairing below cannot be read any earlier than here: a module that
            # merely MENTIONS @pyne reaches this method while its own transform
            # is on the stack, and module_interface is one of them.
            from pynecore.transformers.module_interface import (
                interface_payload, stable_source,
            )

            # The fingerprint the interface this transform publishes is derived
            # from, read together with the bytes it describes (see
            # ``stable_source``). The loader read ``data`` before handing it
            # over, so an atomic replace in between leaves the two disagreeing;
            # the file on disk is what the fingerprint belongs to, so that is
            # what gets transformed — and the ``.pyc`` the loader then writes is
            # the newer source's, which is the right outcome. None here means no
            # trustworthy pairing was to be had, and nothing derived from these
            # bytes may be published under one.
            source_bytes = data if isinstance(data, bytes) else data.encode('utf-8')
            fingerprint: tuple[int, int] | None = None
            stable = stable_source(path)
            if stable is not None:
                on_disk, fingerprint = stable
                if on_disk != source_bytes:
                    try:
                        data_str = on_disk.decode('utf-8')
                    except UnicodeDecodeError:
                        # Nothing to transform those bytes as: keep what the
                        # loader gave, and publish it under no fingerprint
                        fingerprint = None
                    else:
                        tree = ast.parse(data_str)
                        # The file that owns the fingerprint owns the verdict
                        # too: what replaced this source need not be Pyne code
                        is_pyne_module, pyne_mode = _module_mode(tree)
                        if not is_pyne_module:
                            return self._compile_plain(tree, data_str, path, _optimize)

            pipeline_hash = _get_transform_pipeline_hash()
            # Read off the source tree: the analysis rewrites the decorator
            bool_na = _script_bool_na(tree, path)

            key = str(path.resolve())
            outermost = key not in _transforming
            _transforming.add(key)
            try:
                table, interface, transformed = _with_nesting_headroom(
                    lambda module: _transform_module(module, data_str, path, pyne_mode,
                                                     bool(bool_na), fingerprint),
                    tree, data_str)
                # The lowering has imported what it routed calls into, so the routes
                # of those dependencies are settled by now -- the records the type
                # pass made before that still have to be given them
                if table is not None and interface is not None:
                    interface = _settle_deps(table, interface)
            finally:
                if outermost:
                    _transforming.discard(key)

            # Bake a pipeline-identity sentinel into the module body so a loaded code
            # object can be distinguished from foreign or stale bytecode (see get_code).
            # It must survive into the .pyc, so it is a plain assignment the compiler
            # marshals like any other constant — no .pyc-format surgery needed. Added
            # after the debug/save dumps above so those keep showing the semantic
            # transform, free of this loader-level bookkeeping.
            baked: list[ast.stmt] = [ast.Assign(
                targets=[ast.Name(id=_PYNE_SENTINEL, ctx=ast.Store())],
                value=ast.Constant(value=pipeline_hash),
            )]
            # The interfaces this module's types were derived from, and the routes
            # its calls into them were emitted on. A tuple of constants folds into
            # ONE code constant, which is what lets get_code find the records
            # without executing the module.
            if table is not None and table.deps:
                baked.append(ast.Assign(
                    targets=[ast.Name(id=_PYNE_DEPS, ctx=ast.Store())],
                    value=ast.Tuple(elts=[ast.Constant(value=_PYNE_DEPS)] + [
                        ast.Tuple(elts=[ast.Constant(value=record.path),
                                        ast.Constant(value=record.mtime_ns),
                                        ast.Constant(value=record.size),
                                        ast.Constant(value=record.digest),
                                        ast.Constant(value=record.routes)],
                                  ctx=ast.Load())
                        for _, record in sorted(table.deps.items())], ctx=ast.Load()),
                ))
            # What the module publishes, for a dependent's transform to read off the
            # .pyc without executing it (see ``compile_interface``). Paired with the
            # fingerprint of the bytes it was derived from; without one there is no
            # pairing a reader could check, so nothing is baked.
            if interface is not None and fingerprint is not None:
                baked.append(ast.Assign(
                    targets=[ast.Name(id=_PYNE_INTERFACE, ctx=ast.Store())],
                    value=ast.Constant(value=(_PYNE_INTERFACE, interface.mtime_ns,
                                              interface.size, interface_payload(interface))),
                ))
            capture_deps = getattr(transformed, '_export_capture_deps', ())
            if capture_deps:
                baked.append(ast.Assign(
                    targets=[ast.Name(id=_PYNE_CAPTURE_DEPS, ctx=ast.Store())],
                    value=ast.Constant(value=(_PYNE_CAPTURE_DEPS, *capture_deps)),
                ))
            # is_pyne_module guarantees body[0] is the module docstring; keep it first,
            # and stay after any ``from __future__`` imports (which must lead the module).
            insert_at = 1
            while (insert_at < len(transformed.body)
                   and isinstance(transformed.body[insert_at], ast.ImportFrom)
                   and cast(ast.ImportFrom, transformed.body[insert_at]).module == '__future__'):
                insert_at += 1
            transformed.body[insert_at:insert_at] = baked
            synthetic: list[ast.stmt] = list(baked)
            # The script's bool na choice, re-applied after EVERY top-level import
            # (an imported library's own prologue must not outlive this module's,
            # wherever the import stands) so that no statement of the module
            # builds a bool na under another module's choice (see _script_bool_na)
            if bool_na is not None:
                insert_at += len(baked)
                prologue: list[ast.stmt] = [
                    ast.ImportFrom(
                        module='pynecore.types.na',
                        names=[ast.alias(name='set_bool_na', asname=_PYNE_SET_BOOL_NA)],
                        level=0),
                    ast.Assign(
                        targets=[ast.Name(id=_PYNE_NA_BOOL, ctx=ast.Store())],
                        value=ast.Constant(value=bool_na))]
                transformed.body[insert_at:insert_at] = prologue
                synthetic += prologue
                body: list[ast.stmt] = []
                for index, stmt in enumerate(transformed.body):
                    body.append(stmt)
                    if index >= insert_at and isinstance(stmt, (ast.Import, ast.ImportFrom)) \
                            and not (index + 1 < len(transformed.body) and isinstance(
                                transformed.body[index + 1], (ast.Import, ast.ImportFrom))):
                        reset = ast.Expr(value=ast.Call(
                            func=ast.Name(id=_PYNE_SET_BOOL_NA, ctx=ast.Load()),
                            args=[ast.Constant(value=bool_na)], keywords=[]))
                        body.append(reset)
                        synthetic.append(reset)
                transformed.body = body
            # Everything the pipeline emitted was located by ``fix_locations``;
            # only the loader's own statements above still lack a position. They
            # stand at module level, so each gets exactly what a whole-module
            # ``ast.fix_missing_locations`` would give it, without the walk
            for stmt in synthetic:
                ast.fix_missing_locations(stmt)

            # Let Python handle bytecode caching
            return compile(transformed, path, 'exec', optimize=_optimize)

        return self._compile_plain(tree, data_str, path, _optimize)

    @staticmethod
    def _compile_plain(source: "bytes | str | ast.Module", text: str, path: Path,
                       optimize: int):
        """Compile a module that is not Pyne code.

        Nothing of the pipeline applies to it. The one exception is a module of the
        pynecore package itself: the builtins a script calls millions of times live
        there, and the narrowing casts in them are erased (see
        ``transformers.type_erasure``). Such a module carries its own certificate,
        which ``get_code`` checks a cached ``.pyc`` against.

        :param source: What to compile: the source as the loader read it, or the
                       tree when the module has been parsed already.
        :param text: The decoded source, for the prefilter.
        :param path: Source path.
        :param optimize: Optimization level handed through to ``compile``.
        :return: The compiled code object.
        """
        if not has_erasure_marker(text) or not _in_package(str(path)):
            return compile(source, path, 'exec', optimize=optimize)

        import ast

        tree = source if isinstance(source, ast.Module) else ast.parse(text)
        tree = erase_type_calls(tree)
        # After the docstring and any ``from __future__`` imports, which must lead
        insert_at = 0 if ast.get_docstring(tree, clean=False) is None else 1
        while (insert_at < len(tree.body)
               and isinstance(tree.body[insert_at], ast.ImportFrom)
               and cast(ast.ImportFrom, tree.body[insert_at]).module == '__future__'):
            insert_at += 1
        tree.body.insert(insert_at, ast.Assign(
            targets=[ast.Name(id=_PYNE_ERASED_SENTINEL, ctx=ast.Store())],
            value=ast.Constant(value=_get_type_erasure_hash())))
        ast.fix_missing_locations(tree)
        return compile(tree, path, 'exec', optimize=optimize)


class PyneImportHook:
    """Import hook that uses PyneLoader"""

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def find_spec(self, fullname: str, path, target=None):
        """Find and create module spec"""
        entries = sys.path if path is None else path

        if "." in fullname:
            *_, name = fullname.split(".")
        else:
            name = fullname

        for entry in entries:
            if entry == "":
                entry = "."

            # Check both module.py and module/__init__.py
            candidates = [
                Path(entry) / f"{name}.py",
                Path(entry) / name / "__init__.py"
            ]

            for py_path in candidates:
                if py_path.exists():
                    # Stale/foreign bytecode is handled content-based in
                    # ``PyneLoader.get_code`` (via the transform sentinel), so there is
                    # no per-path cache bookkeeping to do here.
                    return importlib.util.spec_from_file_location(
                        fullname,
                        py_path,
                        loader=PyneLoader(fullname, str(py_path))
                    )
        return None


# Install the import hook
sys.meta_path.insert(0, PyneImportHook())
