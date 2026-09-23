"""Build hook for PyneCore's one optional compiled extension.

All package metadata lives in ``pyproject.toml``; this file only declares
``pynecore.core._native_math``, the compiled twin of the pure-Python transcendental
functions in ``core/pine_math.py`` and ``core/fdlibm.py``.

The extension is optional in every sense: a platform without a C compiler, or a
compile that fails, still installs a working package that runs the pure-Python
implementations, and ``PYNE_BUILD_PURE=1`` builds the pure wheel on purpose (the one
WebAssembly runtimes and unlisted platforms install).
"""
import os
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# The compiled functions must round every operation exactly as the Python originals
# do. clang and gcc may fuse ``a * b + c`` into one FMA instruction unless told not to
# (clang does by default, and arm64 always has FMA), which changes last bits; fast-math
# would additionally reorder sums. MSVC contracts only under /fp:fast or /fp:contract.
_UNIX_FP_FLAGS = ['-ffp-contract=off', '-fno-fast-math', '-fno-unsafe-math-optimizations']
_MSVC_FP_FLAGS = ['/fp:precise']


class OptionalBuildExt(build_ext):
    """Compile with exact floating-point flags; on failure, fall back to pure Python."""

    def build_extensions(self) -> None:
        flags = _MSVC_FP_FLAGS if self.compiler.compiler_type == 'msvc' else _UNIX_FP_FLAGS
        for ext in self.extensions:
            ext.extra_compile_args = [*ext.extra_compile_args, *flags]
        super().build_extensions()

    def run(self) -> None:
        try:
            super().run()
        except Exception as exc:  # noqa: BLE001 -- any build failure means "pure Python"
            self._warn(exc)

    def build_extension(self, ext: Extension) -> None:
        try:
            super().build_extension(ext)
        except Exception as exc:  # noqa: BLE001 -- any build failure means "pure Python"
            self._warn(exc)

    @staticmethod
    def _warn(exc: Exception) -> None:
        print(f"WARNING: pynecore native math extension not built ({exc}); "
              f"the pure-Python implementation will be used.", file=sys.stderr)


def _extensions() -> list[Extension]:
    if os.environ.get('PYNE_BUILD_PURE'):
        return []
    try:
        from Cython.Build import cythonize
    except ImportError:
        print("WARNING: Cython not available; building without the native math extension.",
              file=sys.stderr)
        return []
    return cythonize(
        [Extension('pynecore.core._native_math', ['src/pynecore/core/_native_math.pyx'])],
        compiler_directives={'language_level': 3},
    )


setup(ext_modules=_extensions(), cmdclass={'build_ext': OptionalBuildExt})
