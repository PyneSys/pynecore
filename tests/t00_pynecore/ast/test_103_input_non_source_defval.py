"""
Behavior tests for the InputTransformer's non-source ``input()`` defvals.

The generic ``input()`` returns the source NAME only when its defval is a
builtin price series; for any other module-level constant (a color, a number, a
string) it returns the value itself. Emitting the source-resolution statement
for those made the value a ``getattr`` attribute name and crashed the script
before its first bar ("attribute name must be string, not 'Color'" — Heatmap
Volume [xdecow], whose colors are declared as ``chm1 = #ff0000`` and fed to
``input(chm1, ...)``).
"""
import ast
import types

from pynecore.transformers.input_transformer import InputTransformer


def _fake_lib() -> types.SimpleNamespace:
    """A minimal ``lib`` stub: ``input`` echoes its defval, ``input.source`` the name."""
    class _Input:
        @staticmethod
        def __call__(default, *_a, **_k):
            return default

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
    exec(compile(tree, '<input-defval-test>', 'exec'), ns)  # noqa: S102
    return ns, ast.unparse(tree)


def __test_constant_defval_is_not_resolved_as_a_source__():
    """ A module-level constant defval keeps its value, no lib lookup """
    ns, dump = _transform(
        'MY_COLOR = 0xFF0000\n'
        'def main(c=lib.input(MY_COLOR)):\n'
        '    return c\n'
    )
    assert '__pyne_getattr__' not in dump
    assert ns['main']() == 0xFF0000


def __test_builtin_source_defval_is_resolved__():
    """ ``input(lib.close)`` still resolves the returned source name """
    ns, dump = _transform(
        'def main(src=lib.input(lib.close)):\n'
        '    return src\n'
    )
    assert 'src = __pyne_getattr__(lib, src, lib.na)' in dump
    assert ns['main']('close') == 42.0


def __test_source_call_resolves_any_defval__():
    """ ``input.source()`` always returns a name, so it is always resolved """
    _ns, dump = _transform(
        'MY_SRC = 1\n'
        'def main(src=lib.input.source(MY_SRC)):\n'
        '    return src\n'
    )
    assert 'src = __pyne_getattr__(lib, src, lib.na)' in dump
