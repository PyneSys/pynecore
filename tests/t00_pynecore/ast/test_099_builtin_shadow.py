"""
Behavior tests for the builtin-namespace fallback on shadowing library aliases.

Pine resolves ``ta.x`` after ``import user/somelib/1 as ta`` against the
library's exports first and falls back to the built-in ``ta.*`` namespace for
everything else (``ta.valuewhen`` works on TV even though the TradingView/ta
library does not export it). The transformer rewrites the accesses the library
cannot serve to the canonical ``lib.ta.x`` form; the import normalizer then
adds the pynecore imports for them like for any other built-in reference.
"""
import ast
import sys
import types
from contextlib import contextmanager

from pynecore.transformers.builtin_shadow import BuiltinShadowTransformer
from pynecore.transformers.import_normalizer import ImportNormalizerTransformer


@contextmanager
def _fake_lib(module_name: str, exported: list[str] | None = None, **attrs):
    """Register a fake workdir library module for the transform-time import."""
    module = types.ModuleType(module_name)
    if exported is not None:
        module.__all__ = exported
    for name, value in attrs.items():
        setattr(module, name, value)
    sys.modules[module_name] = module
    try:
        yield module
    finally:
        del sys.modules[module_name]


def _shadow(source: str) -> str:
    tree = BuiltinShadowTransformer().visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def __test_missing_member_falls_back_to_builtin__():
    """ A member the library does not export routes to the built-in namespace;
    an exported one stays on the library """
    with _fake_lib('lib.tv.ta.v7', exported=['t3'], t3=lambda: None):
        dump = _shadow(
            'import lib.tv.ta.v7 as ta\n'
            'a = ta.valuewhen(c, t, 0)\n'
            'b = ta.t3(x)\n'
        )
    assert 'a = lib.ta.valuewhen(c, t, 0)' in dump
    assert 'b = ta.t3(x)' in dump


def __test_alias_differs_from_library_name__():
    """ The fallback keys on the alias, not the library's own name """
    with _fake_lib('lib.user.ta_ext.v1', exported=['boost'], boost=lambda: None):
        dump = _shadow(
            'import lib.user.ta_ext.v1 as ta\n'
            'a = ta.rsi(src, 14)\n'
            'b = ta.boost(x)\n'
        )
    assert 'a = lib.ta.rsi(src, 14)' in dump
    assert 'b = ta.boost(x)' in dump


def __test_non_shadowing_alias_untouched__():
    """ An alias that is not a built-in namespace gets no fallback """
    with _fake_lib('lib.tv.ta.v7', exported=['t3'], t3=lambda: None):
        dump = _shadow(
            'import lib.tv.ta.v7 as myta\n'
            'a = myta.valuewhen(c, t, 0)\n'
        )
    assert 'a = myta.valuewhen(c, t, 0)' in dump


def __test_unknown_member_left_alone__():
    """ A member in neither the library nor the built-in namespace is kept
    on the library to fail at runtime, as before """
    with _fake_lib('lib.tv.ta.v7', exported=['t3'], t3=lambda: None):
        dump = _shadow(
            'import lib.tv.ta.v7 as ta\n'
            'a = ta.nosuchthing(x)\n'
        )
    assert 'a = ta.nosuchthing(x)' in dump


def __test_unimportable_library_skipped__():
    """ When the library cannot be imported the alias is left untouched —
    the script's own import statement reports the failure """
    dump = _shadow(
        'import lib.missing.ta.v9 as ta\n'
        'a = ta.valuewhen(c, t, 0)\n'
    )
    assert 'a = ta.valuewhen(c, t, 0)' in dump


def __test_library_without_all_uses_hasattr__():
    """ A hand-written library without ``__all__`` falls back to hasattr """
    with _fake_lib('lib.user.ta.v1', helper=lambda: None):
        dump = _shadow(
            'import lib.user.ta.v1 as ta\n'
            'a = ta.helper(x)\n'
            'b = ta.change(src)\n'
        )
    assert 'a = ta.helper(x)' in dump
    assert 'b = lib.ta.change(src)' in dump


def __test_parameter_shadowing_masks_alias__():
    """ A function parameter named like the alias masks the fallback inside
    that scope only """
    with _fake_lib('lib.tv.ta.v7', exported=['t3'], t3=lambda: None):
        dump = _shadow(
            'import lib.tv.ta.v7 as ta\n'
            'def f(ta):\n'
            '    return ta.change(x)\n'
            'a = ta.change(src)\n'
        )
    assert 'return ta.change(x)' in dump
    assert 'a = lib.ta.change(src)' in dump


def __test_normalizer_adds_builtin_imports__():
    """ The import normalizer picks up the rewritten chains and adds the
    pynecore imports for them """
    with _fake_lib('lib.tv.ta.v7', exported=['t3'], t3=lambda: None):
        tree = BuiltinShadowTransformer().visit(ast.parse(
            'import lib.tv.ta.v7 as ta\n'
            'a = ta.valuewhen(c, t, 0)\n'
        ))
        tree = ImportNormalizerTransformer().visit(tree)
        ast.fix_missing_locations(tree)
        dump = ast.unparse(tree)
    assert 'from pynecore import lib' in dump
    assert 'import pynecore.lib.ta' in dump
    assert 'a = lib.ta.valuewhen(c, t, 0)' in dump
