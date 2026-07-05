from typing import cast
import os
import sys
import hashlib
import importlib.util
import importlib.machinery
import re
from pathlib import Path


# Module-level constant the transform pipeline bakes into every transformed module
# (see ``PyneLoader.source_to_code``). Its presence — together with a matching
# pipeline hash in ``co_consts`` — certifies a loaded code object as current
# pipeline output, so foreign or stale bytecode can be told apart and dropped.
_PYNE_SENTINEL = '__pyne_transformed__'

# A module is Pyne code only when its docstring STARTS with ``@pyne``. Matching the
# raw source head mirrors the strict docstring check in ``source_to_code`` without
# paying for a full parse on every import. Leading comment lines are skipped so a
# PEP 723 ``# /// script`` metadata block before the docstring does not hide it.
_PYNE_HEAD_RE = re.compile(
    rb'^(?:\s*#[^\r\n]*(?:\r?\n|$))*\s*[rRbBuUfF]*("""|\'\'\'|"|\')\s*@pyne(?:\s|\1|$)')


def _source_starts_with_pyne(head: bytes) -> bool:
    """Return whether a source head is a Pyne module (docstring begins with ``@pyne``).

    :param head: First bytes of the source file.
    :return: Whether the module should carry the transform sentinel.
    """
    return _PYNE_HEAD_RE.match(head) is not None


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
    cache markers and read-only install locations.

    :return: Hex digest pinning the transform pipeline.
    """
    global _transform_pipeline_hash
    if _transform_pipeline_hash is None:
        files = [Path(__file__)]  # this module pins the transformer pipeline order
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
        _transform_pipeline_hash = digest.hexdigest()[:16]
    return _transform_pipeline_hash


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

        :param fullname: Fully-qualified module name being loaded.
        :return: The compiled code object (retransformed if the cache was foreign or stale).
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
        if code is None or not _source_starts_with_pyne(head):
            return code

        pipeline_hash = _get_transform_pipeline_hash()
        if _PYNE_SENTINEL in code.co_names and pipeline_hash in code.co_consts:
            return code

        # Foreign or stale bytecode slipped past CPython's mtime/size check — drop it
        # and let the loader recompile, refreshing the cache when the dir is writable.
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
        is_pyne_module = False
        if (tree.body and isinstance(tree.body[0], ast.Expr) and
                isinstance(cast(ast.Expr, tree.body[0]).value, ast.Constant) and
                isinstance(cast(ast.Constant, cast(ast.Expr, tree.body[0]).value).value, str)):
            docstring = cast(str, cast(ast.Constant, cast(ast.Expr, tree.body[0]).value).value)
            is_pyne_module = re.match(r'\s*@pyne(\s|$)', docstring) is not None

        if is_pyne_module:

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

            # Transform AST - lazy import transformers only when needed
            from pynecore.transformers.import_lifter import ImportLifterTransformer
            from pynecore.transformers.type_checking_stripper import TypeCheckingStripperTransformer
            from pynecore.transformers.builtin_shadow import BuiltinShadowTransformer
            from pynecore.transformers.import_normalizer import ImportNormalizerTransformer
            from pynecore.transformers.inline_series_hoist import InlineSeriesHoistTransformer
            from pynecore.transformers.security import SecurityTransformer
            from pynecore.transformers.persistent_series import PersistentSeriesTransformer
            from pynecore.transformers.lib_series import LibrarySeriesTransformer
            from pynecore.transformers.closure_arguments_transformer import ClosureArgumentsTransformer
            from pynecore.transformers.function_isolation import FunctionIsolationTransformer
            from pynecore.transformers.module_property import ModulePropertyTransformer
            from pynecore.transformers.series import SeriesTransformer
            from pynecore.transformers.script_requirements import ScriptRequirementsTransformer
            from pynecore.transformers.unused_series_detector import UnusedSeriesDetectorTransformer
            from pynecore.transformers.persistent import PersistentTransformer
            from pynecore.transformers.input_transformer import InputTransformer
            from pynecore.transformers.safe_convert_transformer import SafeConvertTransformer
            from pynecore.transformers.safe_division_transformer import SafeDivisionTransformer
            from pynecore.transformers.slot_layout import ModuleLayout, apply_layout

            # Shared slot allocator of the module (see slot_layout.py); the
            # state-contributing transformers fill it, apply_layout emits it
            slot_layout = ModuleLayout()

            transformed = ImportLifterTransformer().visit(transformed)
            transformed = TypeCheckingStripperTransformer().visit(transformed)
            # The builtin-namespace fallback must run before import normalization
            # so the lib.<ns>.<name> chains it emits get their imports added there
            transformed = BuiltinShadowTransformer().visit(transformed)
            transformed = ImportNormalizerTransformer().visit(transformed)
            # Lazy-context history hoist must run before call-site anchoring:
            # the hoisted statements are the anchorable call sites
            transformed = InlineSeriesHoistTransformer().visit(transformed)
            transformed = SecurityTransformer().visit(transformed)
            transformed = PersistentSeriesTransformer().visit(transformed)
            transformed = LibrarySeriesTransformer().visit(transformed)
            transformed = ModulePropertyTransformer().visit(transformed)
            transformed = ClosureArgumentsTransformer().visit(transformed)
            transformed = UnusedSeriesDetectorTransformer().optimize(transformed)
            transformed = SeriesTransformer(slot_layout).visit(transformed)
            transformed = PersistentTransformer(slot_layout).visit(transformed)
            # Call-site classification needs the var/series slots, so the
            # isolation transformer must run after Persistent and Series
            transformed = FunctionIsolationTransformer(slot_layout).visit(transformed)
            transformed = ScriptRequirementsTransformer().visit(transformed)
            transformed = InputTransformer().visit(transformed)
            transformed = SafeConvertTransformer().visit(transformed)
            transformed = SafeDivisionTransformer().visit(transformed)
            transformed = apply_layout(transformed, slot_layout)

            ast.fix_missing_locations(transformed)

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

            # Bake a pipeline-identity sentinel into the module body so a loaded code
            # object can be distinguished from foreign or stale bytecode (see get_code).
            # It must survive into the .pyc, so it is a plain assignment the compiler
            # marshals like any other constant — no .pyc-format surgery needed. Added
            # after the debug/save dumps above so those keep showing the semantic
            # transform, free of this loader-level bookkeeping.
            sentinel = ast.Assign(
                targets=[ast.Name(id=_PYNE_SENTINEL, ctx=ast.Store())],
                value=ast.Constant(value=_get_transform_pipeline_hash()),
            )
            # is_pyne_module guarantees body[0] is the module docstring; keep it first,
            # and stay after any ``from __future__`` imports (which must lead the module).
            insert_at = 1
            while (insert_at < len(transformed.body)
                   and isinstance(transformed.body[insert_at], ast.ImportFrom)
                   and cast(ast.ImportFrom, transformed.body[insert_at]).module == '__future__'):
                insert_at += 1
            transformed.body.insert(insert_at, sentinel)
            ast.fix_missing_locations(transformed)

            tree = transformed

        # Let Python handle bytecode caching
        return compile(tree, path, 'exec', optimize=_optimize)


class PyneImportHook:
    """Import hook that uses PyneLoader"""

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def find_spec(self, fullname: str, path, target=None):
        """Find and create module spec"""
        if path is None:
            path = sys.path

        if "." in fullname:
            *_, name = fullname.split(".")
        else:
            name = fullname

        for entry in path:
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
