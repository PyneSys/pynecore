"""
@pyne
"""
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path

from pynecore.core.import_hook import (
    PyneLoader,
    _cache_from_source,
    _get_transform_pipeline_mtime,
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


def _write_valid_pyc(loader: PyneLoader, fullname: str, pyc_path: Path) -> None:
    """Compile the module once so a genuine, CPython-valid ``.pyc`` exists on disk."""
    pyc_path.parent.mkdir(parents=True, exist_ok=True)
    with _bytecode_writing_enabled():
        loader.get_code(fullname)
    assert pyc_path.exists(), "baseline .pyc was not produced"


def __test_stale_pyne_pyc_is_retransformed__(tmp_path):
    """Bytecode older than the transform pipeline is dropped and recompiled"""
    mod = tmp_path / "stale_mod.py"
    mod.write_text('"""\n@pyne\n"""\nx = 1\n')
    pyc = _cache_from_source(mod)

    loader = PyneLoader("stale_mod", str(mod))
    _write_valid_pyc(loader, "stale_mod", pyc)

    # CPython still considers this .pyc valid (source untouched), but its file mtime
    # predates the transform pipeline -> emulates a cache left over from an older
    # PyneCore install.
    old = _get_transform_pipeline_mtime() - 100.0
    os.utime(pyc, (old, old))

    with _bytecode_writing_enabled():
        code = loader.get_code("stale_mod")

    assert isinstance(code, types.CodeType)
    # A fresh .pyc must have been written, so its mtime now follows the pipeline.
    assert pyc.exists()
    assert pyc.stat().st_mtime >= _get_transform_pipeline_mtime()


def __test_fresh_pyne_pyc_is_kept__(tmp_path):
    """Bytecode newer than the transform pipeline is reused, not recompiled"""
    mod = tmp_path / "fresh_mod.py"
    mod.write_text('"""\n@pyne\n"""\nx = 2\n')
    pyc = _cache_from_source(mod)

    loader = PyneLoader("fresh_mod", str(mod))
    _write_valid_pyc(loader, "fresh_mod", pyc)

    # Freshly produced -> mtime is already past the pipeline; nothing to invalidate.
    assert pyc.stat().st_mtime >= _get_transform_pipeline_mtime()
    before = pyc.stat().st_mtime

    with _bytecode_writing_enabled():
        loader.get_code("fresh_mod")

    assert pyc.exists()
    assert pyc.stat().st_mtime == before, "valid bytecode was needlessly recompiled"


def __test_stale_pyne_pyc_under_custom_pycache_prefix_is_retransformed__(tmp_path, monkeypatch):
    """Stale bytecode is dropped even when sys.pycache_prefix relocates the cache"""
    monkeypatch.setattr(sys, "pycache_prefix", str(tmp_path / "cacheprefix"))

    mod = tmp_path / "prefixed_mod.py"
    mod.write_text('"""\n@pyne\n"""\nx = 4\n')
    pyc = _cache_from_source(mod)
    # The cache must live under the custom prefix, not in a sibling __pycache__.
    assert str(tmp_path / "cacheprefix") in str(pyc)

    loader = PyneLoader("prefixed_mod", str(mod))
    _write_valid_pyc(loader, "prefixed_mod", pyc)

    old = _get_transform_pipeline_mtime() - 100.0
    os.utime(pyc, (old, old))

    with _bytecode_writing_enabled():
        code = loader.get_code("prefixed_mod")

    assert isinstance(code, types.CodeType)
    assert pyc.exists()
    assert pyc.stat().st_mtime >= _get_transform_pipeline_mtime()


def __test_non_pyne_pyc_is_untouched__(tmp_path):
    """A plain (non-@pyne) module keeps its cache even when older than the pipeline"""
    mod = tmp_path / "plain_mod.py"
    mod.write_text("y = 3\n")
    pyc = _cache_from_source(mod)

    loader = PyneLoader("plain_mod", str(mod))
    _write_valid_pyc(loader, "plain_mod", pyc)

    old = _get_transform_pipeline_mtime() - 100.0
    os.utime(pyc, (old, old))

    with _bytecode_writing_enabled():
        loader.get_code("plain_mod")

    # Not a transformed module: the pipeline mtime is irrelevant, cache stays as-is.
    assert pyc.exists()
    assert pyc.stat().st_mtime == old, "non-@pyne bytecode must not be invalidated"
