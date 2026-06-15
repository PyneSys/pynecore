"""
Behavior tests for the InputTransformer's source-input resolution.

A ``src = input.source(close, ...)`` declaration compiles to a body statement
that resolves the chosen source name against ``lib`` with an ``na`` fallback.
The resolution must use a reserved builtin alias (``__pyne_getattr__``) rather
than a bare ``getattr``: a script variable named ``getattr`` would shadow the
builtin in the function scope and break the resolution — the same class of bug
as a ``len`` input shadowing the function-isolation loop-counter guard.
"""
import ast
import types

from pynecore.transformers.input_transformer import InputTransformer


def _fake_lib() -> types.SimpleNamespace:
    """A minimal ``lib`` stub exposing the pieces the emission touches.

    ``input.source`` echoes the chosen source name (the parameter default is
    evaluated at function-definition time, so ``lib`` must already exist).
    """
    class _Input:
        @staticmethod
        def source(_default, *_a, **_k):
            return 'close'

    return types.SimpleNamespace(close=42.0, na=None, input=_Input())


def _transform(source: str) -> tuple[dict, str]:
    """Run the InputTransformer on a source string and exec the result.

    :param source: Pyne-style module source.
    :return: (exec'd module namespace, unparsed transformed source)
    """
    tree = InputTransformer().visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    ns: dict = {'lib': _fake_lib()}  # parameter defaults reference ``lib`` at exec
    exec(compile(tree, '<input-source-test>', 'exec'), ns)  # noqa: S102
    return ns, ast.unparse(tree)


def __test_source_resolution_uses_builtin_alias__():
    """ Source resolution emits the reserved alias, never a shadowable builtin """
    _ns, dump = _transform(
        'def main(src=lib.input.source(lib.close)):\n'
        '    return src\n'
    )
    assert 'from builtins import getattr as __pyne_getattr__' in dump
    assert 'src = __pyne_getattr__(lib, src, lib.na)' in dump
    assert 'getattr(lib, src' not in dump  # no bare, shadowable builtin call


def __test_getattr_variable_does_not_break_source_resolution__():
    """ A script variable named ``getattr`` must not break source resolution """
    ns, dump = _transform(
        'def main(src=lib.input.source(lib.close), getattr=7):\n'
        '    return src\n'
    )
    # The shadowing name is a plain function parameter, so a bare ``getattr(...)``
    # call in the body would resolve to the int 7 and raise 'int not callable'.
    assert ns['main']() == 42.0          # resolved lib.close despite the shadow
    assert ns['main']('na') is None      # unknown source name falls back to lib.na
    assert '__pyne_getattr__' in dump
