"""
A function reading a ``main`` variable gets ``main``'s value from every caller.

The closure pass turns ``main``'s variables that an inner function reads into its
parameters and passes them at each call site by name. Inside another inner
function that binds the same name itself (a parameter or a local), that name is
the caller's own value, not ``main``'s. TradingView (v5 and v6, CAPITALCOM:EURUSD
60) reads the global there: ``g = close * 2``, ``reader() => g``, ``f(g) => reader()``
plots ``close * 2`` for ``f(7.0)``, and so does ``h() => g = 7.0, reader()``.
"""
import ast

import pytest

from pynecore.transformers.closure_arguments_transformer import ClosureArgumentsTransformer


def __test_helper_run(source: str) -> tuple[object, str]:
    """Run the closure pass on a source, then ``main``.

    :param source: Pyne-style module source with a ``@script.indicator`` ``main``.
    :return: (``main``'s result, unparsed transformed source)
    """
    tree = ClosureArgumentsTransformer().visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    ns: dict = {}
    exec(compile(tree, '<closure-shadow-test>', 'exec'), ns)  # noqa: S102
    return ns['main'](), ast.unparse(tree)


__test_helper_PRELUDE = '''
class script:
    @staticmethod
    def indicator(*_args):
        return lambda func: func


def method(func):
    return func


def method_call(func, *args):
    return func(*args)

'''


def __test_shadowing_parameter_passes_main_value__():
    """ A caller whose parameter shadows the capture hands ``main``'s value on """
    result, _ = __test_helper_run(__test_helper_PRELUDE + '''
@script.indicator('t')
def main():
    g = 20.0

    def reader():
        return g

    def f(g):
        return reader()

    return f(7.0)
''')
    assert result == 20.0


def __test_shadowing_local_passes_main_value__():
    """ A caller whose local shadows the capture hands ``main``'s value on """
    result, _ = __test_helper_run(__test_helper_PRELUDE + '''
@script.indicator('t')
def main():
    g = 20.0

    def reader():
        return g

    def h():
        g = 7.0
        return reader() + g

    return h()
''')
    assert result == 27.0


def __test_capture_reaches_through_a_chain__():
    """ Every shadowing caller on the way takes ``main``'s value as a capture """
    result, dump = __test_helper_run(__test_helper_PRELUDE + '''
@script.indicator('t')
def main():
    g = 20.0

    def reader():
        return g

    def mid(g):
        return reader() + g

    def outer(g):
        return mid(g + 1)

    return outer(5.0)
''')
    assert result == 26.0
    assert 'def outer(__capture·g__, g)' in dump


def __test_method_capture_follows_receiver__():
    """ A method's own capture goes right after its receiver, by direct and
    ``method_call`` calls alike """
    result, dump = __test_helper_run(__test_helper_PRELUDE + '''
@script.indicator('t')
def main():
    g = 20.0

    def reader():
        return g

    @method
    def m(x):
        g = x
        return reader() + g

    def k(g):
        return method_call(m, 1.0) + m(2.0)

    return k(3.0)
''')
    assert result == 43.0
    assert 'def m(x, __capture·g__)' in dump
    assert "m.__pyne_closure__ = ('__capture·g__',)" in dump


def __test_unshadowed_caller_keeps_plain_names__():
    """ A caller that does not bind the name passes it as before """
    result, dump = __test_helper_run(__test_helper_PRELUDE + '''
@script.indicator('t')
def main():
    g = 20.0

    def reader():
        return g

    def f(x):
        return reader() + x

    return f(1.0)
''')
    assert result == 21.0
    assert '__capture' not in dump


def __test_default_calls_in_enclosing_scope__():
    """ A default of a shadowing parameter runs in ``main``, so it passes ``main``'s
    own variable, not a capture of the function it belongs to """
    result, dump = __test_helper_run(__test_helper_PRELUDE + '''
@script.indicator('t')
def main():
    g = 20.0

    def reader():
        return g

    def f(g=reader()):
        return g

    return f()
''')
    assert result == 20.0
    assert '__capture' not in dump


def __test_main_default_calls_module_function__():
    """ A default of ``main`` runs at module level, so a call there reaches the module's
    function, not an inner function of ``main`` of the same name """
    result, dump = __test_helper_run(__test_helper_PRELUDE + '''
def reader():
    return 99.0


@script.indicator('t')
def main(x=reader()):
    g = 20.0

    def reader():
        return g

    return x + reader()
''')
    assert result == 119.0
    assert 'main(x=reader())' in dump


def __test_later_defined_callee_gets_its_captures__():
    """ A call to an inner function defined after its caller passes its captures """
    result, dump = __test_helper_run(__test_helper_PRELUDE + '''
@script.indicator('t')
def main():
    g = 20.0

    def f(g):
        return reader() + g

    def reader():
        return g

    return f(7.0)
''')
    assert result == 27.0
    assert 'reader(__capture·g__)' in dump


def __test_keyword_only_parameter_shadows__():
    """ A keyword-only parameter binds the name like any other parameter """
    result, dump = __test_helper_run(__test_helper_PRELUDE + '''
@script.indicator('t')
def main():
    g = 20.0

    def reader():
        return g

    def f(*, g):
        return reader() + g

    return f(g=7.0)
''')
    assert result == 27.0
    assert 'def f(__capture·g__, *, g)' in dump


def __test_positional_only_parameter_shadows__():
    """ A positional-only parameter binds the name, and the capture goes before it """
    result, dump = __test_helper_run(__test_helper_PRELUDE + '''
@script.indicator('t')
def main():
    g = 20.0

    def reader():
        return g

    def f(g, /):
        return reader() + g

    return f(7.0)
''')
    assert result == 27.0
    assert 'def f(__capture·g__, g, /)' in dump


def __test_method_explicit_receiver_before_unpacked_rest__():
    """ A direct method call with an explicit receiver passes the capture after it,
    also into a positional-only parameter, when the rest of its arguments are unpacked """
    result, dump = __test_helper_run(__test_helper_PRELUDE + '''
@script.indicator('t')
def main():
    g = 20.0

    @method
    def m(this, x, /):
        return this + x + g

    return m(1.0, *(2.0,))
''')
    assert result == 23.0
    assert 'def m(this, g, x, /)' in dump
    assert 'm(1.0, g, *(2.0,))' in dump


def __test_method_unpacked_receiver_is_rejected__():
    """ A direct method call whose receiver comes from an unpacked argument has no
    position for the capture to follow, so the transform rejects it """
    with pytest.raises(SyntaxError, match=r'pass the receiver explicitly: m\(receiver, \*args\)'):
        __test_helper_run(__test_helper_PRELUDE + '''
@script.indicator('t')
def main():
    g = 20.0

    @method
    def m(this, x, /):
        return this + x + g

    return m(*(1.0, 2.0))
''')
