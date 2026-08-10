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
    _cache_from_source,
    _get_transform_pipeline_hash,
    _reject_reserved_names,
    _PYNE_SENTINEL,
)


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
