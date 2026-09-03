from typing import TYPE_CHECKING, cast
import os
import sys
import hashlib
import importlib.util
import importlib.machinery
import re
from pathlib import Path

if TYPE_CHECKING:
    import ast

    from pynecore.transformers.pine_type_table import DepRecord, Diag, PineTypeTable

__all__ = ['PYNE_RESERVED_NAME_CHAR', 'PIPELINE_DIGEST', 'source_starts_with_pyne',
           'analyse_source', 'PyneLoader', 'PyneImportHook']


# Module-level constant the transform pipeline bakes into every transformed module
# (see ``PyneLoader.source_to_code``). Its presence — together with a matching
# pipeline hash in ``co_consts`` — certifies a loaded code object as current
# pipeline output, so foreign or stale bytecode can be told apart and dropped.
_PYNE_SENTINEL = '__pyne_transformed__'

# Name of the constant a transformed module carries its type dependencies in, and
# the first element of that constant's tuple. A ``.pyc`` holds many tuples; this
# marker is what tells the records apart from every other one in ``co_consts``.
_PYNE_DEPS = '__pyne_type_deps__'

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


# Everything the transformers inject into script scope is named with a Unicode
# middle dot: the scope-qualified state parameters and slot constants
# (``__state·main__``, ``__slot·main·x__``), the generated temporaries
# (``__st·__``, ``__cnt·0__``) and the aliased runtime helper imports
# (``__resolve_slot·__``). The separator is a legal identifier character in
# Python (``Other_ID_Continue``), so the namespace is only collision-free while
# scripts stay out of it — a script name spelled with it would shadow or clobber
# an injected one and break the emission in ways no transformer can detect.
PYNE_RESERVED_NAME_CHAR = '·'


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
    # The parsed tree is what has to be checked, not the source spelling. Python
    # NFKC-normalizes identifiers while parsing, so the bound name can carry a
    # separator the source never spells: U+0387 GREEK ANO TELEIA normalizes to one
    # outright, U+013F / U+0140 (LATIN LETTER L WITH MIDDLE DOT) decompose into one.
    # The token stream cannot stand in for the tree either — before Python 3.12 the
    # tokenizer hands out a whole f-string as a single string token, hiding every
    # name its replacement expressions bind (``f"{(__st·__ := x)}"``).
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
                if not isinstance(name, str) or PYNE_RESERVED_NAME_CHAR not in name:
                    continue
                lineno = getattr(node, 'lineno', 1)
                lines = source.splitlines()
                raise SyntaxError(
                    f"'{name}' contains '{PYNE_RESERVED_NAME_CHAR}', which is reserved "
                    f"for PyneCore's internal names in Pyne code — rename the identifier",
                    (str(path), lineno, getattr(node, 'col_offset', 0) + 1,
                     lines[lineno - 1] if 0 < lineno <= len(lines) else None),
                )


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


_transform_pipeline_hash: str | None = None


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
    ``FloatToleranceTransformer`` emits as a literal) is such a file.

    :return: Hex digest pinning the transform pipeline.
    """
    global _transform_pipeline_hash
    if _transform_pipeline_hash is not None:
        return _transform_pipeline_hash

    # This module pins the transformer pipeline order; ``pine_compare`` holds
    # a constant the pipeline bakes into the emitted bytecode
    files = [Path(__file__), Path(__file__).parent / "pine_compare.py"]
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
    pipeline_hash = digest.hexdigest()[:16]
    _transform_pipeline_hash = pipeline_hash
    return pipeline_hash


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


def _analyse_tree(tree: "ast.Module", source: str, path: Path,
                  pyne_mode: str | None) -> "ast.Module":
    """Run the pipeline up to and including the Pine type pass.

    This half only ANALYSES: it normalizes the tree into the form the type pass
    reads and stamps the types onto it, without emitting any of the state
    plumbing the second half does. Splitting it out is what lets an imported
    module's signatures be derived without compiling or running anything
    (:func:`analyse_source`).

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

    # The edge gate reads the tree AS WRITTEN: what follows injects plumbing
    # no profile should judge. Its findings join the type diagnostics below
    from pynecore.transformers.pine_edge_gate import (
        diag_dump_enabled, gate_module, gated, render_diags, strict_enabled,
    )
    from pynecore.transformers.pine_type_table import PineTypeError
    from pynecore.transformers.pine_type_transformer import module_table

    edge = gated(pyne_mode)
    structural = gate_module(transformed) if edge else []

    # Transform AST - lazy import transformers only when needed
    from pynecore.transformers.import_lifter import ImportLifterTransformer
    from pynecore.transformers.type_checking_stripper import TypeCheckingStripperTransformer
    from pynecore.transformers.builtin_shadow import BuiltinShadowTransformer
    from pynecore.transformers.import_normalizer import ImportNormalizerTransformer
    from pynecore.transformers.const_fold import ConstFoldTransformer
    from pynecore.transformers.dynamic_default import DynamicDefaultTransformer
    from pynecore.transformers.inline_series_hoist import InlineSeriesHoistTransformer
    from pynecore.transformers.security import SecurityTransformer
    from pynecore.transformers.security_instantiation import (
        SecurityInstantiationTransformer,
    )
    from pynecore.transformers.persistent_series import PersistentSeriesTransformer
    from pynecore.transformers.lib_series import LibrarySeriesTransformer
    from pynecore.transformers.closure_arguments_transformer import ClosureArgumentsTransformer
    from pynecore.transformers.pine_type_transformer import PineTypeTransformer
    from pynecore.transformers.module_property import ModulePropertyTransformer
    from pynecore.transformers.ta_variable_hoist import TaVariableHoistTransformer
    from pynecore.transformers.pine_truthiness import PineTruthinessTransformer

    transformed = ImportLifterTransformer().visit(transformed)
    transformed = TypeCheckingStripperTransformer().visit(transformed)
    # The builtin-namespace fallback must run before import normalization
    # so the lib.<ns>.<name> chains it emits get their imports added there
    transformed = BuiltinShadowTransformer().visit(transformed)
    transformed = ImportNormalizerTransformer().visit(transformed)
    # TradingView folds constant subtrees at parse time with fdlibm
    # transcendentals and a 16-decimal embedding cap, while runtime
    # series-fed calls use the Intel-LIBM intrinsics (lib.math /
    # core.pine_math); the fold pass replays that split. It needs the
    # normalized lib.math.* chains, and only user/compiled scripts get
    # it -- pynecore's own lib modules must keep their raw expressions
    if not path.is_relative_to(Path(__file__).parent.parent):
        transformed = ConstFoldTransformer().visit(transformed)
    # Per-call evaluation of lib.*-referencing parameter defaults; must
    # precede the series/isolation passes so the moved expressions get
    # their series slots and call-site anchors like any body statement
    transformed = DynamicDefaultTransformer().visit(transformed)
    # Lazy-context history hoist must run before call-site anchoring:
    # the hoisted statements are the anchorable call sites
    transformed = InlineSeriesHoistTransformer().visit(transformed)
    # Pine's tolerant float-to-bool conversion, over the script's OWN
    # bool contexts: it runs before the passes that emit control flow of
    # their own (lazy-init flags and friends), whose tests are bools by
    # construction. Only user/compiled scripts get it -- see the
    # comparison rewrite at the end of the pipeline for the same rule
    if not path.is_relative_to(Path(__file__).parent.parent):
        transformed = PineTruthinessTransformer().visit(transformed)
    # Pine instantiation semantics: clone security-bearing functions
    # per call site so each call site gets its own security contexts
    transformed = SecurityInstantiationTransformer().visit(transformed)
    transformed = SecurityTransformer().visit(transformed)
    transformed = PersistentSeriesTransformer().visit(transformed)
    transformed = LibrarySeriesTransformer().visit(transformed)
    transformed = ModulePropertyTransformer().visit(transformed)
    # Stateful ta builtin variables become one unconditional per-bar
    # evaluation at the top of main (TradingView keeps a single engine
    # series per builtin variable, gates notwithstanding); must follow
    # the property transformer (bare reads are calls by now) and precede
    # the series/persistent/isolation passes so the hoisted call site is
    # anchored like any hand-written statement
    transformed = TaVariableHoistTransformer().visit(transformed)
    transformed = ClosureArgumentsTransformer().visit(transformed)
    # Pine's static types, stamped on the nodes. The last point where
    # the tree still looks like Pine: the annotations are intact (the
    # series pass rewrites and consumes them), the `/` is still a
    # BinOp (safe division wraps it into a call), and the
    # security-bearing functions are already instantiated per call
    # site. Analysis only -- it stamps, it does not rewrite.
    transformed = PineTypeTransformer(pyne_mode, analyse=analyse_source,
                                      pipeline_hash=_get_transform_pipeline_hash()
                                      ).visit(transformed)
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


def _lower_tree(tree: "ast.Module", path: Path, pyne_mode: str | None) -> "ast.Module":
    """Run the pipeline from the type pass to the finished emission.

    This half EMITS: it turns the analysed tree into the state-plumbed form the
    runtime executes, allocates the module's slot layout and fixes up the
    synthetic locations.

    :param tree: The analysed, type-stamped tree.
    :param path: Source path; the script / lib profile is picked from it.
    :param pyne_mode: The module's mode word, None for a hand-written script.
    :return: The tree the compiler is handed.
    """
    import ast

    from pynecore.transformers.function_isolation import FunctionIsolationTransformer
    from pynecore.transformers.series import SeriesTransformer
    from pynecore.transformers.script_requirements import ScriptRequirementsTransformer
    from pynecore.transformers.unused_series_detector import UnusedSeriesDetectorTransformer
    from pynecore.transformers.persistent import PersistentTransformer
    from pynecore.transformers.input_transformer import InputTransformer
    from pynecore.transformers.safe_convert_transformer import SafeConvertTransformer
    from pynecore.transformers.safe_division_transformer import SafeDivisionTransformer
    from pynecore.transformers.float_tolerance import FloatToleranceTransformer
    from pynecore.transformers.slot_layout import ModuleLayout, apply_layout
    from pynecore.transformers.locations import fix_locations

    # Shared slot allocator of the module (see slot_layout.py); the
    # state-contributing transformers fill it, apply_layout emits it
    slot_layout = ModuleLayout(compacted_series=pyne_mode == 'lib')

    transformed = UnusedSeriesDetectorTransformer().optimize(tree)
    transformed = SeriesTransformer(slot_layout).visit(transformed)
    transformed = PersistentTransformer(slot_layout).visit(transformed)
    # Call-site classification needs the var/series slots, so the
    # isolation transformer must run after Persistent and Series
    transformed = FunctionIsolationTransformer(slot_layout).visit(transformed)
    transformed = ScriptRequirementsTransformer().visit(transformed)
    transformed = InputTransformer().visit(transformed)
    transformed = SafeConvertTransformer(lib=pyne_mode == 'lib').visit(transformed)
    transformed = SafeDivisionTransformer().visit(transformed)
    # After SafeDivision so wrapped operands (safe_div calls) are bound
    # once by the walrus instead of evaluating twice. Only user/compiled
    # scripts get Pine's tolerant comparison semantics: pynecore's own
    # lib modules implement the natively bit-exact builtins and use the
    # raw ``x != x`` nan idiom, both of which the rewrite would break
    if not path.is_relative_to(Path(__file__).parent.parent):
        transformed = FloatToleranceTransformer().visit(transformed)
    transformed = apply_layout(transformed, slot_layout)

    # Debugger-safe variant of ast.fix_missing_locations: synthetic
    # nodes get point anchors, so no prologue bytecode maps onto the
    # function's last line (see transformers/locations.py)
    fix_locations(transformed)

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

    return transformed


def analyse_source(path: str) -> \
        "tuple[ast.Module, PineTypeTable, tuple[int, int] | None] | None":
    """Derive one module's types from its source, compiling and running nothing.

    This is how a module that is not imported yet still answers for its own
    signatures: a dependent's cached bytecode is checked before either module is
    executed, so the check cannot rely on an import. Only the analysing half of
    the pipeline runs, which emits no state plumbing and has no side effects
    beyond the tree it builds and throws away.

    The fingerprint comes back with the tree because it belongs to it: it is the
    one the parsed bytes were read under, and a caller building an interface from
    the tree has no other honest one to pair with the signatures.

    :param path: Path to the ``.py`` source.
    :return: The analysed tree, its type table and the fingerprint of the bytes
             they were derived from, or None when the file is not readable, not
             parseable, or not Pyne code.
    """
    import ast

    # Lazy for the same reason the transformers are: this module is loaded
    # through the hook itself, so importing it at module level would re-enter a
    # half-initialized package
    from pynecore.transformers.pine_type_artifact import stable_source
    from pynecore.transformers.pine_type_transformer import module_table

    # One stable read: the bytes analysed here and the fingerprint they are
    # published under have to describe the same file
    stable = stable_source(Path(path))
    if stable is None:
        return None
    data, fingerprint = stable
    try:
        source = data.decode('utf-8')
        tree = ast.parse(source)
    except (UnicodeDecodeError, SyntaxError, ValueError):
        return None

    is_pyne_module, pyne_mode = _module_mode(tree)
    if not is_pyne_module:
        return None

    try:
        analysed = _analyse_tree(tree, source, Path(path), pyne_mode)
    except (SyntaxError, RecursionError):
        # An unanalysable dependency is not a failure to report here: the
        # module that actually imports it raises the real error, with the real
        # traceback. All this can say is that no interface could be derived.
        return None
    table = module_table(analysed)
    return None if table is None else (analysed, table, fingerprint)


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
                                   size=record[2], digest=record[3])
                         for record in const[1:])
    return ()


def _deps_current(code, pipeline_hash: str) -> bool:
    """Whether every module this bytecode was typed against still says the same thing.

    :param code: The module's code object.
    :param pipeline_hash: Digest of the current transform pipeline.
    :return: True while the cached bytecode is still valid.
    """
    records = _baked_deps(code)
    if not records:
        return True

    # Lazy, as above -- and skipped entirely for a module with no dependencies
    from pynecore.transformers.pine_type_artifact import dep_current

    return all(dep_current(record, analyse_source, pipeline_hash) for record in records)


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
        the INTERFACES it imports, and CPython's check sees none of them. So a module
        that was typed against others also carries their state in a
        ``__pyne_type_deps__`` constant, re-checked here before the cache is accepted.

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

        # Only transformed modules carry the sentinel; leave everything else untouched.
        # ``get_code`` is typed Optional, but a real source file always yields a code
        # object — the ``None`` guard just narrows the type for the checks below.
        if code is None or not source_starts_with_pyne(head):
            return code

        pipeline_hash = _get_transform_pipeline_hash()
        if _PYNE_SENTINEL not in code.co_names or pipeline_hash not in code.co_consts:
            return self._retransform(fullname, source_path, pipeline_hash)

        # The pipeline is current, but the types this module was compiled against
        # live in OTHER modules; an edit to one of their interfaces makes this
        # bytecode wrong while CPython still sees a valid cache for it.
        if _deps_current(code, pipeline_hash):
            return code
        return self._retransform(fullname, source_path, pipeline_hash)

    def _retransform(self, fullname: str, source_path: str, pipeline_hash: str):
        """Drop bytecode the checks rejected and produce the current transform.

        :param fullname: Fully-qualified module name being loaded.
        :param source_path: Path to the module's ``.py`` source.
        :param pipeline_hash: Digest the recompiled code object has to carry.
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
        if code is None or (_PYNE_SENTINEL in code.co_names and pipeline_hash in code.co_consts):
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
            return compile(data, path, 'exec', optimize=_optimize)

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
            # is on the stack, and pine_type_artifact is one of them.
            from pynecore.transformers.pine_type_artifact import (
                NO_FINGERPRINT, artifact_enabled, build_interface, register,
                stable_source, write_artifact,
            )
            from pynecore.transformers.pine_type_transformer import module_table

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
                        source_bytes = on_disk
                        tree = ast.parse(data_str)
                        # The file that owns the fingerprint owns the verdict
                        # too: what replaced this source need not be Pyne code
                        is_pyne_module, pyne_mode = _module_mode(tree)
                        if not is_pyne_module:
                            return compile(tree, path, 'exec', optimize=_optimize)

            pipeline_hash = _get_transform_pipeline_hash()
            analysed = _analyse_tree(tree, data_str, path, pyne_mode)
            table = module_table(analysed)

            # What this module publishes, for every module that imports it: in
            # this process through the registry, across processes through the
            # artifact beside the .pyc. Read off the ANALYSED tree, before the
            # lowering: the isolation pass prepends a state parameter to every
            # script function and the series pass rewrites the annotations, so a
            # signature taken afterwards is the emission's, not the module's.
            interface = None
            if table is not None:
                interface = build_interface(
                    analysed, table, str(path.resolve()),
                    NO_FINGERPRINT if fingerprint is None else fingerprint)
                register(interface)

            transformed = _lower_tree(analysed, path, pyne_mode)

            # No fingerprint means no artifact: a reader validates one by
            # digesting the source it now finds, and nothing here knows which
            # bytes that will be
            if (table is not None and interface is not None
                    and fingerprint is not None and artifact_enabled(pyne_mode)):
                write_artifact(transformed, table, interface, source_bytes, path,
                               pipeline_hash)

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
            # The interfaces this module's types were derived from. A tuple of
            # constants folds into ONE code constant, which is what lets get_code
            # find the records without executing the module.
            if table is not None and table.deps:
                baked.append(ast.Assign(
                    targets=[ast.Name(id=_PYNE_DEPS, ctx=ast.Store())],
                    value=ast.Tuple(elts=[ast.Constant(value=_PYNE_DEPS)] + [
                        ast.Tuple(elts=[ast.Constant(value=record.path),
                                        ast.Constant(value=record.mtime_ns),
                                        ast.Constant(value=record.size),
                                        ast.Constant(value=record.digest)],
                                  ctx=ast.Load())
                        for _, record in sorted(table.deps.items())], ctx=ast.Load()),
                ))
            # is_pyne_module guarantees body[0] is the module docstring; keep it first,
            # and stay after any ``from __future__`` imports (which must lead the module).
            insert_at = 1
            while (insert_at < len(transformed.body)
                   and isinstance(transformed.body[insert_at], ast.ImportFrom)
                   and cast(ast.ImportFrom, transformed.body[insert_at]).module == '__future__'):
                insert_at += 1
            transformed.body[insert_at:insert_at] = baked
            ast.fix_missing_locations(transformed)

            tree = transformed

        # Let Python handle bytecode caching
        return compile(tree, path, 'exec', optimize=_optimize)


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
