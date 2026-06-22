"""
Unit tests for ``utils/stdlib_checker.py``.

Regression guard for issue #68: a non-editable install (``pip install`` /
``uv sync`` from PyPI) places PyneCore under ``<prefix>/lib/pythonX.Y/
site-packages`` on Linux. A path-based stdlib check computed the stdlib root
as ``<prefix>/lib`` and excluded site-packages as ``<prefix>/lib/
site-packages`` — which never matched the real, version-nested location — so it
classified PyneCore itself as stdlib. The function-isolation transformer then
skipped state threading for ``lib.math.sum`` / ``lib.math.random`` calls,
crashing ``ta.sma`` (and friends) with ``TypeError: sum() missing 1 required
positional argument``. Windows (``Lib\\site-packages``) and editable installs
(source outside ``lib/``) happened to dodge it, which is why dev and CI never
saw it.
"""
import sys

from pynecore.utils.stdlib_checker import is_stdlib


def __test_stdlib_modules_are_recognized__():
    """ Standard library roots resolve as stdlib, dotted paths included """
    assert is_stdlib('math')
    assert is_stdlib('os')
    assert is_stdlib('os.path')
    assert is_stdlib('collections')
    assert is_stdlib('collections.abc')
    assert is_stdlib('xml.etree.ElementTree')
    assert is_stdlib('sys')


def __test_pynecore_is_not_stdlib__():
    """ PyneCore must never be classified as stdlib, regardless of install layout """
    assert not is_stdlib('pynecore')
    assert not is_stdlib('pynecore.lib')
    assert not is_stdlib('pynecore.transformers.function_isolation')


def __test_third_party_is_not_stdlib__():
    """ Arbitrary third-party package names are not stdlib """
    assert not is_stdlib('numpy')
    assert not is_stdlib('not_a_real_module_xyz')


def __test_detection_is_name_based_and_layout_independent__():
    """ Every name in ``sys.stdlib_module_names`` resolves as stdlib, file paths ignored """
    for name in sys.stdlib_module_names:
        assert is_stdlib(name)
        assert is_stdlib(f'{name}.submodule')
