"""
@pyne
"""
import ast
import os
import subprocess
import sys
import types
import py_compile
import unicodedata
from contextlib import contextmanager
from pathlib import Path

import pytest

import pynecore.core.import_hook as import_hook
from pynecore.core.import_hook import (
    PyneLoader,
    PYNE_RESERVED_NAME_CHAR,
    _baked_deps,
    _cache_from_source,
    _get_transform_pipeline_hash,
    _reject_reserved_names,
    _PYNE_SENTINEL,
)
from pynecore.transformers import pine_type_artifact
from pynecore.transformers.pine_type_artifact import dep_record, lookup
from pynecore.transformers.pine_type_transformer import PineTypeTransformer, module_table


def main():
    """Dummy main so this file is a valid Pyne script."""
    pass


@contextmanager
def _bytecode_writing_enabled():
    """Temporarily allow ``.pyc`` writing (the test suite disables it globally)."""
    saved = sys.dont_write_bytecode
    sys.dont_write_bytecode = False
    try:
        yield
    finally:
        sys.dont_write_bytecode = saved


def _write_foreign_pyc(mod: Path, pyc: Path) -> None:
    """Write an untransformed ``.pyc`` — what ``pip``'s compileall / an IDE produces.

    ``py_compile`` uses the plain :class:`SourceFileLoader`, so the bytecode never
    sees the transform pipeline; CPython still accepts it as a valid cache for the
    source.
    """
    pyc.parent.mkdir(parents=True, exist_ok=True)
    with _bytecode_writing_enabled():
        py_compile.compile(str(mod), cfile=str(pyc), doraise=True)
    assert pyc.exists(), "foreign .pyc was not produced"


def _is_current_transform(code: types.CodeType | None) -> bool:
    """Whether a code object carries the current pipeline's transform sentinel."""
    return (code is not None and _PYNE_SENTINEL in code.co_names
            and _get_transform_pipeline_hash() in code.co_consts)


def __test_foreign_pyc_is_retransformed__(tmp_path):
    """Untransformed bytecode (pip compileall / IDE) is dropped and retransformed"""
    mod = tmp_path / "foreign_mod.py"
    mod.write_text('"""\n@pyne\n"""\nx = 1\n')
    pyc = _cache_from_source(mod)
    _write_foreign_pyc(mod, pyc)

    loader = PyneLoader("foreign_mod", str(mod))
    with _bytecode_writing_enabled():
        code = loader.get_code("foreign_mod")

    # The foreign cache lacked the sentinel, so the source was retransformed.
    assert _is_current_transform(code)


def __test_single_line_pyne_marker_is_transformed__(tmp_path):
    """A single-line ``\"\"\"@pyne\"\"\"`` docstring still triggers the transform pipeline"""
    # The closing quote follows @pyne immediately, with no whitespace, so the fast
    # prefilter must accept a quote (not only whitespace / end-of-input) as the token
    # terminator. This is the form used throughout the docs and tests; getting it wrong
    # skipped the transform and surfaced as ``TypeError: 'module' object is not callable``.
    mod = tmp_path / "single_line_mod.py"
    mod.write_text('"""@pyne"""\nx = 1\n')

    loader = PyneLoader("single_line_mod", str(mod))
    code = loader.source_to_code(mod.read_bytes(), str(mod))

    # The sentinel is baked in only when the transform pipeline actually runs.
    assert _is_current_transform(code)


def __test_reserved_name_is_rejected__(tmp_path):
    """A script identifier in the transformers' middle-dot namespace is a SyntaxError"""
    # The separator is a legal Python identifier character, so a script CAN spell the
    # injected names (helper aliases, generated temporaries, scope-qualified state
    # parameters). Left alone, such a name silently shadows or clobbers the emission.
    reserved = "__slot_state" + PYNE_RESERVED_NAME_CHAR + "__"
    mod = tmp_path / "reserved_mod.py"
    mod.write_text(f'"""\n@pyne\n"""\ndef main({reserved}):\n    return {reserved}\n',
                   encoding="utf-8")

    loader = PyneLoader("reserved_mod", str(mod))
    with pytest.raises(SyntaxError) as excinfo:
        loader.source_to_code(mod.read_bytes(), str(mod))

    assert reserved in str(excinfo.value)
    assert excinfo.value.lineno == 4  # points at the offending identifier


def __test_nfkc_equivalent_reserved_name_is_rejected__(tmp_path):
    """An identifier that only becomes reserved after NFKC normalization is rejected too"""
    # Python NFKC-normalizes identifiers while parsing, so the bound name is not the
    # source spelling: U+0387 GREEK ANO TELEIA turns into the separator, which would
    # shadow an injected name while the source contains no literal separator at all.
    spelled = "__bind_slot·__"
    parsed = "__bind_slot" + PYNE_RESERVED_NAME_CHAR + "__"
    assert unicodedata.normalize("NFKC", spelled) == parsed
    assert PYNE_RESERVED_NAME_CHAR not in spelled  # the source-level spelling hides it

    mod = tmp_path / "nfkc_mod.py"
    mod.write_text(f'"""\n@pyne\n"""\ndef main({spelled}):\n    return {spelled}\n',
                   encoding="utf-8")

    loader = PyneLoader("nfkc_mod", str(mod))
    with pytest.raises(SyntaxError) as excinfo:
        loader.source_to_code(mod.read_bytes(), str(mod))

    assert parsed in str(excinfo.value)  # the message shows what the parser would bind
    assert excinfo.value.lineno == 4


def __test_reserved_name_inside_fstring_is_rejected__(tmp_path):
    """A reserved name bound inside an f-string replacement expression is rejected too"""
    # Before Python 3.12 the tokenizer hands out a whole f-string as a single string
    # token, so the names its replacement expressions bind are invisible there. The
    # check runs on the parsed tree, which sees the walrus target on every version.
    reserved = "__bind_slot" + PYNE_RESERVED_NAME_CHAR + "__"
    mod = tmp_path / "fstring_mod.py"
    mod.write_text(
        '"""\n@pyne\n"""\n'
        'def main():\n'
        f'    return f"{{({reserved} := 1)}}"\n',
        encoding="utf-8",
    )

    loader = PyneLoader("fstring_mod", str(mod))
    with pytest.raises(SyntaxError) as excinfo:
        loader.source_to_code(mod.read_bytes(), str(mod))

    assert reserved in str(excinfo.value)


@pytest.mark.skipif(sys.version_info < (3, 14), reason="template strings need Python 3.14")
def __test_reserved_char_inside_template_string_text_is_allowed__(tmp_path):
    """The separator stays legal inside a template string's interpolated literal"""
    # ``ast.Interpolation`` keeps the interpolation's own source text in a ``str``
    # field, so the blanket identifier scan must skip it — the interpolated
    # expression is a child node of its own and gets checked there.
    src = ('"""\n@pyne\n"""\n'
           f'LABEL = t"{{\'scope{PYNE_RESERVED_NAME_CHAR}path\'}}"\n')

    _reject_reserved_names(ast.parse(src), src, tmp_path / "template_text_mod.py")


@pytest.mark.skipif(sys.version_info < (3, 14), reason="template strings need Python 3.14")
def __test_reserved_name_inside_template_string_is_rejected__(tmp_path):
    """A reserved name bound inside a template string interpolation is rejected"""
    reserved = "__bind_slot" + PYNE_RESERVED_NAME_CHAR + "__"
    src = ('"""\n@pyne\n"""\n'
           f'LABEL = t"{{({reserved} := 1)}}"\n')

    with pytest.raises(SyntaxError) as excinfo:
        _reject_reserved_names(ast.parse(src), src, tmp_path / "template_name_mod.py")

    assert reserved in str(excinfo.value)


def __test_reserved_char_outside_identifiers_is_allowed__(tmp_path):
    """The reserved character stays legal in strings and comments"""
    mod = tmp_path / "dotted_text_mod.py"
    mod.write_text(
        '"""\n@pyne\n"""\n'
        f'# separator: {PYNE_RESERVED_NAME_CHAR}\n'
        f'LABEL = "scope{PYNE_RESERVED_NAME_CHAR}path"\n'
        'def main():\n'
        '    return LABEL\n',
        encoding="utf-8",
    )

    loader = PyneLoader("dotted_text_mod", str(mod))
    code = loader.source_to_code(mod.read_bytes(), str(mod))

    assert _is_current_transform(code)


def __test_current_pyc_is_kept__(tmp_path):
    """Bytecode carrying the current pipeline sentinel is reused, not recompiled"""
    mod = tmp_path / "current_mod.py"
    mod.write_text('"""\n@pyne\n"""\nx = 2\n')
    pyc = _cache_from_source(mod)

    loader = PyneLoader("current_mod", str(mod))
    pyc.parent.mkdir(parents=True, exist_ok=True)
    with _bytecode_writing_enabled():
        loader.get_code("current_mod")  # writes a transformed .pyc with the sentinel
    assert pyc.exists()
    before = pyc.stat().st_mtime

    with _bytecode_writing_enabled():
        code = loader.get_code("current_mod")

    assert _is_current_transform(code)
    assert pyc.stat().st_mtime == before, "valid transformed bytecode was needlessly recompiled"


def __test_stale_pipeline_pyc_is_retransformed__(tmp_path, monkeypatch):
    """Bytecode produced by a different transform pipeline is dropped and retransformed"""
    mod = tmp_path / "stalepipe_mod.py"
    mod.write_text('"""\n@pyne\n"""\nx = 3\n')
    pyc = _cache_from_source(mod)

    loader = PyneLoader("stalepipe_mod", str(mod))
    pyc.parent.mkdir(parents=True, exist_ok=True)

    # Compile under a DIFFERENT pipeline hash so the baked-in sentinel is stale.
    monkeypatch.setattr(import_hook, "_transform_pipeline_hash", "0000oldpipeline0")
    with _bytecode_writing_enabled():
        loader.get_code("stalepipe_mod")

    # Restore the real hash (None forces a recompute on next access).
    monkeypatch.setattr(import_hook, "_transform_pipeline_hash", None)
    real_hash = _get_transform_pipeline_hash()
    assert real_hash != "0000oldpipeline0"

    with _bytecode_writing_enabled():
        code = loader.get_code("stalepipe_mod")

    assert code is not None
    assert real_hash in code.co_consts
    assert "0000oldpipeline0" not in code.co_consts


def __test_non_pyne_pyc_is_untouched__(tmp_path):
    """A plain (non-@pyne) module keeps its cache; the sentinel check ignores it"""
    mod = tmp_path / "plain_mod.py"
    mod.write_text("y = 4\n")
    pyc = _cache_from_source(mod)
    _write_foreign_pyc(mod, pyc)

    # Backdate the cache; for a non-Pyne module nothing should invalidate it.
    old = pyc.stat().st_mtime - 100.0
    os.utime(pyc, (old, old))

    loader = PyneLoader("plain_mod", str(mod))
    with _bytecode_writing_enabled():
        loader.get_code("plain_mod")

    assert pyc.exists()
    assert pyc.stat().st_mtime == old, "non-@pyne bytecode must not be invalidated"


def __test_foreign_pyc_in_readonly_cache_still_runs_transformed__(tmp_path):
    """A foreign .pyc that cannot be deleted (read-only cache) still runs transformed"""
    mod = tmp_path / "ro_mod.py"
    mod.write_text('"""\n@pyne\n"""\nx = 5\n')
    pyc = _cache_from_source(mod)
    _write_foreign_pyc(mod, pyc)

    loader = PyneLoader("ro_mod", str(mod))

    cache_dir = pyc.parent
    cache_dir.chmod(0o500)  # r-x: deleting the stale .pyc is impossible
    try:
        with _bytecode_writing_enabled():
            code = loader.get_code("ro_mod")
    finally:
        cache_dir.chmod(0o700)

    # The stale cache could not be removed, yet the loader compiled straight from
    # source so the correct, transformed bytecode runs anyway.
    assert _is_current_transform(code)
    assert pyc.exists(), "the read-only stale cache should still be on disk (just ignored)"


def __test_import_script_installs_hook_from_namespace_package_cwd__(tmp_path):
    """The script boundary installs the hook when cwd shadows the editable package.

    From the monorepo root, the checkout's top-level ``pynecore/`` directory
    can make ``pynecore`` a namespace package, so its ``__init__`` never gets
    the chance to install :class:`PyneImportHook`.  A foreign but timestamp-
    valid ``.pyc`` must still be rejected when :func:`import_script` runs.
    """
    mod = tmp_path / "namespace_cwd_strategy.py"
    mod.write_text(
        '"""@pyne"""\n'
        'from pynecore.lib import script\n'
        '@script.indicator("namespace cwd")\n'
        'def main():\n'
        '    return None\n'
    )
    _write_foreign_pyc(mod, _cache_from_source(mod))

    repo_root = Path(__file__).resolve().parents[4]
    probe = (
        "import sys; "
        "from pathlib import Path; "
        "from pynecore.core.script_runner import import_script; "
        "module = import_script(Path(sys.argv[1])); "
        "print(getattr(module, '__pyne_transformed__', 'missing'))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(mod)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == _get_transform_pipeline_hash()


# --- dependency-aware invalidation ----------------------------------------
#
# A module's types are a function of its own source AND of the interfaces it
# imports, so CPython's mtime/size check on the source alone cannot tell that a
# cached .pyc has gone wrong. Every transformed module therefore carries the
# state of its dependencies in a folded ``__pyne_type_deps__`` constant, which
# ``get_code`` re-checks before it accepts the cache.


@contextmanager
def _typed_against(monkeypatch, dependent: str, dependency: Path):
    """Make one module's transform record a dependency on another.

    The inference does not resolve cross-module calls yet, so nothing fills
    ``table.deps`` on its own; this injects the record the resolution will
    produce. It hooks the type pass rather than the loader because that is
    where the real record will be written — everything downstream (the baked
    constant, the check on load, the invalidation) is the production path.

    :param monkeypatch: The active monkeypatch fixture.
    :param dependent: Stem of the module that gets the dependency record.
    :param dependency: Source path of the module it depends on.
    """
    original = PineTypeTransformer.visit

    def visit(self, tree):
        tree = original(self, tree)
        table = module_table(tree)
        path = getattr(tree, '_module_file_path', '')
        if table is not None and Path(path).stem == dependent:
            interface = lookup(str(dependency), import_hook.analyse_source,
                               _get_transform_pipeline_hash())
            if interface is not None:
                record = dep_record(interface)
                table.deps[record.path] = record
        return tree

    monkeypatch.setattr(PineTypeTransformer, "visit", visit)
    try:
        yield
    finally:
        pine_type_artifact._registry.clear()
        pine_type_artifact._analysing.clear()


def _dependency_pair(tmp_path: Path, ret: str = "int") -> tuple[Path, Path]:
    """Write a dependency and a dependent, and hand back both paths."""
    lib = tmp_path / "dep_lib.py"
    lib.write_text(f'"""\n@pyne\n"""\n\n\ndef area(width: int) -> {ret}:\n    return width * 2\n')
    app = tmp_path / "dep_app.py"
    app.write_text('"""\n@pyne\n"""\nSIZE: int = 3\n')
    return lib, app


def _build_dependent(app: Path, lib: Path, monkeypatch) -> tuple[str, float]:
    """Transform the dependent once and report its baked digest and cache mtime."""
    with _typed_against(monkeypatch, app.stem, lib), _bytecode_writing_enabled():
        code = PyneLoader(app.stem, str(app)).get_code(app.stem)
    records = _baked_deps(code)
    assert len(records) == 1, "the dependency record was not baked into the module"
    assert records[0].path == str(lib.resolve())
    return records[0].digest, _cache_from_source(app).stat().st_mtime_ns


@contextmanager
def _counted(monkeypatch):
    """Count the analyses and the retransforms one load costs."""
    counts = {"analyse": 0, "transform": 0}
    real_analyse = import_hook.analyse_source
    real_transform = PyneLoader.source_to_code

    def analyse(path: str):
        counts["analyse"] += 1
        return real_analyse(path)

    def transform(self, data, path, *, _optimize: int = -1):
        counts["transform"] += 1
        return real_transform(self, data, path, _optimize=_optimize)

    monkeypatch.setattr(import_hook, "analyse_source", analyse)
    monkeypatch.setattr(PyneLoader, "source_to_code", transform)
    yield counts


def _reload(app: Path, lib: Path, monkeypatch) -> tuple[dict, tuple]:
    """Load the dependent again from a cold registry, as a new process would."""
    # The registry answers for this process only; a fresh process starts empty,
    # which is the situation the invalidation exists for.
    pine_type_artifact._registry.clear()
    with _typed_against(monkeypatch, app.stem, lib), _counted(monkeypatch) as counts:
        with _bytecode_writing_enabled():
            code = PyneLoader(app.stem, str(app)).get_code(app.stem)
    return counts, _baked_deps(code)


def __test_untouched_dependency_costs_one_stat__(tmp_path, monkeypatch):
    """A dependency whose file did not move is accepted without any analysis"""
    lib, app = _dependency_pair(tmp_path)
    digest, mtime = _build_dependent(app, lib, monkeypatch)

    counts, records = _reload(app, lib, monkeypatch)

    assert counts["analyse"] == 0, "an untouched dependency must not be re-analysed"
    assert counts["transform"] == 0, "valid bytecode was needlessly retransformed"
    assert _cache_from_source(app).stat().st_mtime_ns == mtime
    assert records[0].digest == digest


def __test_edited_dependency_body_keeps_the_cache__(tmp_path, monkeypatch):
    """A body edit moves the file but not the interface, so the cache stands"""
    lib, app = _dependency_pair(tmp_path)
    digest, mtime = _build_dependent(app, lib, monkeypatch)

    lib.write_text('"""\n@pyne\n"""\n\n\ndef area(width: int) -> int:\n'
                   '    doubled: int = width * 2\n    return doubled\n')

    counts, records = _reload(app, lib, monkeypatch)

    assert counts["analyse"] == 1, "the moved file has to be re-analysed exactly once"
    assert counts["transform"] == 0, "an unchanged interface must not invalidate the cache"
    assert _cache_from_source(app).stat().st_mtime_ns == mtime
    assert records[0].digest == digest


def __test_changed_dependency_signature_drops_the_cache__(tmp_path, monkeypatch):
    """A return-type change is exactly what the dependent has to be rebuilt for"""
    lib, app = _dependency_pair(tmp_path)
    digest, _ = _build_dependent(app, lib, monkeypatch)
    pyc = _cache_from_source(app)

    lib.write_text('"""\n@pyne\n"""\n\n\ndef area(width: int) -> float:\n    return width * 2\n')

    counts, records = _reload(app, lib, monkeypatch)

    assert counts["transform"] >= 1, "the stale bytecode was not retransformed"
    assert records[0].digest != digest, "the rebuilt module kept the old dependency digest"
    assert pyc.exists(), "invalidation must refresh the cache, not delete the cache dir"


def __test_missing_dependency_drops_the_cache__(tmp_path, monkeypatch):
    """A dependency that is gone cannot answer for itself, so nothing is assumed"""
    lib, app = _dependency_pair(tmp_path)
    _build_dependent(app, lib, monkeypatch)
    lib.unlink()

    counts, records = _reload(app, lib, monkeypatch)

    assert counts["transform"] >= 1, "a vanished dependency must force a retransform"
    assert records == (), "there is no dependency left to record"


def _real_pair(tmp_path: Path, ret: str = "int") -> tuple[Path, Path]:
    """Write a dependency and a dependent that really imports it."""
    lib = tmp_path / "xm_real_lib.py"
    lib.write_text(f'"""\n@pyne\n"""\n\n\ndef area(width: int) -> {ret}:\n    return width * 2\n')
    app = tmp_path / "xm_real_app.py"
    app.write_text('"""\n@pyne\n"""\nfrom xm_real_lib import area\n\nSIZE = area(3)\n')
    return lib, app


def _load(app: Path):
    """Load the dependent from a cold registry, as a new process would."""
    pine_type_artifact._registry.clear()
    pine_type_artifact._analysing.clear()
    with _bytecode_writing_enabled():
        return PyneLoader(app.stem, str(app)).get_code(app.stem)


def __test_the_inference_records_its_own_dependencies__(tmp_path, monkeypatch):
    """The cross-module resolution fills ``table.deps`` on its own, end to end"""
    monkeypatch.syspath_prepend(tmp_path)
    lib, app = _real_pair(tmp_path)

    records = _baked_deps(_load(app))
    digest = records[0].digest
    mtime = _cache_from_source(app).stat().st_mtime_ns
    assert [record.path for record in records] == [str(lib.resolve())]

    # A body edit moves the file but not the interface, so the cache stands
    lib.write_text('"""\n@pyne\n"""\n\n\ndef area(width: int) -> int:\n'
                   '    doubled: int = width * 2\n    return doubled\n')
    assert _baked_deps(_load(app))[0].digest == digest
    assert _cache_from_source(app).stat().st_mtime_ns == mtime

    # A return-type change is exactly what the dependent has to be rebuilt for
    lib.write_text('"""\n@pyne\n"""\n\n\ndef area(width: int) -> float:\n'
                   '    return width * 2.0\n')
    assert _baked_deps(_load(app))[0].digest != digest


# --- the transitive closure -----------------------------------------------
#
# An export can be INFERRED from a third module: annotated parameters and no
# return annotation take the return from whatever the body calls. The middle
# module's own source then says nothing about a signature that moved, and its
# stat short-circuit would keep answering "unchanged" forever -- so the
# dependent has to remember the whole closure, not just the modules it names.


def _chain(tmp_path: Path, prefix: str, ret: str = "float",
           tail: str = " + 0.5") -> tuple[Path, Path, Path]:
    """Write an app -> middle -> leaf chain whose middle return is inferred."""
    leaf = tmp_path / f"{prefix}_leaf.py"
    leaf.write_text(f'''"""
@pyne
"""


def cval(x: int) -> {ret}:
    return x{tail}
''')
    middle = tmp_path / f"{prefix}_mid.py"
    middle.write_text(f'''"""
@pyne
"""
from {prefix}_leaf import cval


def bval(x: int):
    return cval(x)
''')
    app = tmp_path / f"{prefix}_app.py"
    app.write_text(f'''"""
@pyne
"""
from {prefix}_mid import bval

SIZE = bval(3)
''')
    return leaf, middle, app


def __test_the_dependent_remembers_the_whole_chain__(tmp_path, monkeypatch):
    """A module two hops away is a dependency like any other"""
    monkeypatch.syspath_prepend(tmp_path)
    leaf, middle, app = _chain(tmp_path, "xm_chain_all")

    records = _baked_deps(_load(app))

    assert sorted(record.path for record in records) == \
        sorted([str(leaf.resolve()), str(middle.resolve())])


def __test_a_changed_leaf_signature_drops_the_cache__(tmp_path, monkeypatch):
    """The middle module's file never moves, and its published return still does"""
    monkeypatch.syspath_prepend(tmp_path)
    leaf, middle, app = _chain(tmp_path, "xm_chain_sig")
    before = _baked_deps(_load(app))
    middle_bytes = middle.read_bytes()
    leaf_digest = {record.path: record.digest for record in before}[str(leaf.resolve())]

    leaf.write_text(leaf.read_text().replace("-> float:", "-> int:").replace(" + 0.5", " * 2"))
    records = _baked_deps(_load(app))

    assert middle.read_bytes() == middle_bytes, "the middle module was not touched"
    digests = {record.path: record.digest for record in records}
    assert digests[str(leaf.resolve())] != leaf_digest, "the stale bytecode was kept"


def __test_an_edited_leaf_body_keeps_the_cache__(tmp_path, monkeypatch):
    """A body edit two hops away moves no signature, so nothing is rebuilt"""
    monkeypatch.syspath_prepend(tmp_path)
    leaf, _middle, app = _chain(tmp_path, "xm_chain_body")
    before = {record.path: record.digest for record in _baked_deps(_load(app))}
    mtime = _cache_from_source(app).stat().st_mtime_ns

    leaf.write_text(leaf.read_text().replace(
        "    return x + 0.5", "    doubled: float = x + 0.5\n    return doubled"))
    records = _baked_deps(_load(app))

    assert {record.path: record.digest for record in records} == before
    assert _cache_from_source(app).stat().st_mtime_ns == mtime
