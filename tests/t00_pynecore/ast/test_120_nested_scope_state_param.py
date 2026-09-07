"""
@pyne

A definition inside a compound statement belongs to the ENCLOSING scope, and
must be plumbed like a top-level one.

``collect_scope_segments`` has always walked into compound statements, so the
slot allocator gave such a definition its own scope and its body was rewritten
to read ``__state__[N]`` — but ``apply_layout`` only looked at the top level of
each body, so the hidden state parameter and the layout attach never landed on
it. The call site then passed the state vector into the first declared argument
and the definition died on its first call.
"""
from pynecore.lib import close, ta


def main():
    if close > 0.0:
        def smoothed(x: float):
            return ta.sma(x, 3)

        print(smoothed(close))


def __test_nested_definition_gets_its_state_parameter__(log, ast_transformed_code):
    """
    Definition inside an ``if`` — hidden state parameter and layout attach
    """
    try:
        assert 'def smoothed(__state__, x: float):' in ast_transformed_code
        assert "smoothed.__pyne_layout__ = __pyne_slot_layout__['main·smoothed']" \
               in ast_transformed_code
    except AssertionError:
        log.error("AST transformed code:\n%s\n", ast_transformed_code)
        raise
