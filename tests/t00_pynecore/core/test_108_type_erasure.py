"""
@pyne
"""
import ast
import os
import py_compile
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pynecore.core.import_hook as import_hook
from pynecore.core.import_hook import (
    PyneLoader,
    _cache_from_source,
    _get_type_erasure_hash,
    _PYNE_ERASED_SENTINEL,
    _PYNE_SENTINEL,
)
from pynecore.transformers.type_checking_stripper import TypeCheckingStripperTransformer
from pynecore.transformers.type_erasure import TypeErasureTransformer


def main():
    """Dummy main so this file is a valid Pyne script."""
    pass


def __test_helper_erase(source: str) -> tuple[str, int]:
    """Run the pass over a source text; return the emission and the erased count."""
    transformer = TypeErasureTransformer()
    tree = transformer.visit(ast.parse(source))
    return ast.unparse(tree), transformer.erased


def __test_helper_has_cast(code: types.CodeType, module: bool = True) -> bool:
    """Whether a function compiled into a module still names ``cast``. The module
    body itself always does: the import stays."""
    if not module and 'cast' in code.co_names:
        return True
    return any(isinstance(const, types.CodeType) and __test_helper_has_cast(const, False)
               for const in code.co_consts)


@contextmanager
def __test_helper_package_dir(monkeypatch, directory: Path):
    """Make a temporary directory count as the pynecore package, with ``.pyc`` writing on."""
    monkeypatch.setattr(import_hook, '_PACKAGE_DIR', str(directory.resolve()) + os.sep)
    saved = sys.dont_write_bytecode
    sys.dont_write_bytecode = False
    try:
        yield
    finally:
        sys.dont_write_bytecode = saved


def __test_imported_cast_is_erased__():
    """``cast(T, x)`` becomes ``x`` for each way typing.cast can be imported"""
    out, erased = __test_helper_erase(
        "from typing import cast\n"
        "def f(a):\n"
        "    return cast(float, a) * cast('list[int] | None', a)\n")
    assert erased == 2
    assert "return a * a" in out

    out, erased = __test_helper_erase(
        "from typing import cast as narrow\nimport typing as t\n"
        "x = narrow(int, t.cast(dict[str, int], y))\n")
    assert erased == 2
    assert "x = y" in out


def __test_rebound_name_is_kept__():
    """A module that binds the name anywhere else keeps every call through it"""
    for rebinding in ("def cast(a, b):\n    return b\n",
                      "cast = lambda a, b: b\n",
                      "def g(cast):\n    return cast\n",
                      "from ctypes import cast\n",
                      "for cast in ():\n    pass\n",
                      "def h():\n    global cast\n"):
        out, erased = __test_helper_erase(
            "from typing import cast\n" + rebinding + "x = cast(int, y)\n")
        assert erased == 0, rebinding
        assert "x = cast(int, y)" in out


def __test_foreign_cast_is_kept__():
    """A ``cast`` that is not typing's, or not a plain two-operand call, is left alone"""
    out, erased = __test_helper_erase(
        "import ctypes\nfrom typing import cast\n"
        "a = ctypes.cast(buf, ptr)\n"
        "b = cast(typ=int, val=y)\n"
        "c = cast(*pair)\n"
        "d = cast(make_type(), y)\n")
    assert erased == 0
    assert "ctypes.cast(buf, ptr)" in out
    assert "cast(make_type(), y)" in out


def __test_type_checking_if_is_resolved__():
    """``if TYPE_CHECKING:`` is replaced by its else body, or dropped without one"""
    out, erased = __test_helper_erase(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    import a\nelse:\n    import b\n"
        "def f():\n    if TYPE_CHECKING:\n        import c\n    return 1\n")
    assert erased == 2
    assert "import a" not in out and "import c" not in out
    assert "import b" in out
    assert "def f():\n    pass\n    return 1" in out

    out, erased = __test_helper_erase(
        "import typing\nif x:\n    y = 1\nelif typing.TYPE_CHECKING:\n    y = 2\nelse:\n    y = 3\n")
    assert erased == 1
    assert "y = 2" not in out and "y = 3" in out

    # A rebound flag, or a test that is more than the flag, is left as written
    for source in ("from typing import TYPE_CHECKING\nTYPE_CHECKING = True\n"
                   "if TYPE_CHECKING:\n    import a\n",
                   "from typing import TYPE_CHECKING\nif TYPE_CHECKING or x:\n    import a\n",
                   "if TYPE_CHECKING:\n    import a\n"):
        out, erased = __test_helper_erase(source)
        assert erased == 0, source
        assert "import a" in out


def __test_pyne_stripper_keeps_else_body__():
    """The ``@pyne`` TYPE_CHECKING stripper keeps what runs: the else body"""
    tree = TypeCheckingStripperTransformer().visit(ast.parse(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    import a\nelse:\n    import b\n"))
    out = ast.unparse(tree)
    assert "import a" not in out
    assert "import b" in out


def __test_package_module_is_compiled_erased__(tmp_path, monkeypatch):
    """A plain module of the package loses its casts and carries the certificate"""
    mod = tmp_path / "plain_mod.py"
    mod.write_text("from typing import cast\n\n\ndef f(a):\n    return cast(float, a)\n")
    loader = PyneLoader("plain_mod", str(mod))

    # Outside the package nothing is touched
    code = loader.source_to_code(mod.read_bytes(), str(mod))
    assert _PYNE_ERASED_SENTINEL not in code.co_names
    assert __test_helper_has_cast(code)

    with __test_helper_package_dir(monkeypatch, tmp_path):
        code = loader.source_to_code(mod.read_bytes(), str(mod))
    assert _PYNE_ERASED_SENTINEL in code.co_names
    assert _get_type_erasure_hash() in code.co_consts
    assert _PYNE_SENTINEL not in code.co_names
    assert not __test_helper_has_cast(code)

    namespace: dict = {}
    exec(code, namespace)
    assert namespace['f'](1.5) == 1.5


def __test_foreign_pyc_of_package_module_is_recompiled__(tmp_path, monkeypatch):
    """Bytecode compiled without the hook (pip compileall) is dropped for the erased form"""
    mod = tmp_path / "foreign_plain_mod.py"
    mod.write_text('"""Doc."""\nfrom __future__ import annotations\nfrom typing import cast\n'
                   "\n\ndef f(a):\n    return cast(float, a)\n")
    pyc = _cache_from_source(mod)
    loader = PyneLoader("foreign_plain_mod", str(mod))

    with __test_helper_package_dir(monkeypatch, tmp_path):
        pyc.parent.mkdir(parents=True, exist_ok=True)
        py_compile.compile(str(mod), cfile=str(pyc), doraise=True)
        code = loader.get_code("foreign_plain_mod")
        assert code is not None
        assert _PYNE_ERASED_SENTINEL in code.co_names
        assert not __test_helper_has_cast(code)

        # The rewritten cache is accepted as it is on the next load
        cached = loader.get_code("foreign_plain_mod")
        assert cached is not None
        assert _PYNE_ERASED_SENTINEL in cached.co_names


def __test_module_without_cast_keeps_foreign_pyc__(tmp_path, monkeypatch):
    """A package module with nothing to erase is never parsed or recompiled"""
    mod = tmp_path / "no_cast_mod.py"
    mod.write_text("x = 1\n")
    pyc = _cache_from_source(mod)
    loader = PyneLoader("no_cast_mod", str(mod))

    with __test_helper_package_dir(monkeypatch, tmp_path):
        pyc.parent.mkdir(parents=True, exist_ok=True)
        py_compile.compile(str(mod), cfile=str(pyc), doraise=True)
        stamp = pyc.stat().st_mtime_ns
        code = loader.get_code("no_cast_mod")
        assert code is not None
        assert _PYNE_ERASED_SENTINEL not in code.co_names
        assert pyc.stat().st_mtime_ns == stamp
