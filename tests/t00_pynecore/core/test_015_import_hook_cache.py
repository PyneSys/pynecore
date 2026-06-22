"""
@pyne
"""
import os
import sys
import types
import py_compile
from contextlib import contextmanager
from pathlib import Path

import pynecore.core.import_hook as import_hook
from pynecore.core.import_hook import (
    PyneLoader,
    _cache_from_source,
    _get_transform_pipeline_hash,
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


def _is_current_transform(code: types.CodeType) -> bool:
    """Whether a code object carries the current pipeline's transform sentinel."""
    return _PYNE_SENTINEL in code.co_names and _get_transform_pipeline_hash() in code.co_consts


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
