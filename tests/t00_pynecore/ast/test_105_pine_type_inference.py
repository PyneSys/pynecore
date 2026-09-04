"""
The Pine type inference reproduces TradingView's measured type algebra.

Pine's ``int`` is a static type only: an int-typed expression keeps its
fractional value and truncation happens at the consuming slot. Which slots
those are, and which overload a call resolves to, both follow from the TYPE --
so the algebra below is the contract the whole pass rests on.

Every expected type is MEASURED on TradingView (FX:EURUSD@60, ``R =
input.int(14)``, ``z = bar_index >= 0 ? 0 : 1``, ``d = (R + z) / 8`` -- an
int-typed expression whose value is 1.75), recorded in
``work/tv-int-tipus-reverse-engineering.md``.
"""
import ast

import pytest

from pynecore.transformers.const_fold import ConstFoldTransformer
from pynecore.transformers.pine_type_infer import infer_module
from pynecore.transformers.pine_type_rules import get_ty


def _types(source: str, scope: str = '') -> dict[str, str]:
    """Infer a snippet and return its bindings as name -> type."""
    tree = ast.parse(source)
    table = infer_module(tree, 'test')
    return {name: binding.ty for name, binding in table.bindings.get(scope, {}).items()}


def _expr_type(expression: str, preamble: str = '') -> str:
    """Infer one expression in the standard measured setting."""
    source = (
        'from pynecore import lib\n'
        'from pynecore.types import Series\n'
        'R = lib.input.int(14)\n'
        'd = R / 8\n'
        'h: Series[int] = d\n'
        f'{preamble}'
        f'value = {expression}\n'
    )
    return _types(source)['value']


# The measured TradingView table. ``d`` is int-typed with the value 1.75.
@pytest.mark.parametrize("expression,expected", [
    # Arithmetic: int op int stays int for EVERY operator, division included
    ('d', 'i'),
    ('d * 100', 'i'),
    ('d + 1', 'i'),
    ('d / 2', 'i'),
    ('d % 2', 'i'),
    ('-d', 'i'),
    # ... and one float operand widens it
    ('d * 1.0', 'f'),
    ('d + 0.5', 'f'),
    # The all-int math family
    ('lib.math.max(d, 1)', 'i'),
    ('lib.math.max(d, 1.0)', 'f'),
    ('lib.math.abs(d)', 'i'),
    ('lib.math.floor(d)', 'i'),
    # ... against the always-float one
    ('lib.math.sqrt(d)', 'f'),
    # math.round splits on arity, which is why it needed a lib fix too
    ('lib.math.round(d)', 'i'),
    ('lib.math.round(d, 2)', 'f'),
    # The ternary joins its arms rather than widening unconditionally
    ('d if d > 1 else R', 'i'),
    ('d if d > 1 else 1.0', 'f'),
    # nz and the history index are type-preserving (the history is a series')
    ('lib.nz(d)', 'i'),
    ('h[1]', 'i'),
])
def __test_tradingview_type_algebra__(expression: str, expected: str):
    """Each expression carries the type TradingView gives it"""
    assert _expr_type(expression) == expected, expression


def __test_int_division_is_the_load_bearing_case__():
    """
    ``int / int`` is int-typed, which is the divergence the pass exists for.

    Python's ``14 / 8`` is a float, so a runtime-type dispatcher picks the
    float overload where TradingView picked the int one.
    """
    assert _types('x = 14 / 8')['x'] == 'i'
    assert _types('x = 14 / 8.0')['x'] == 'f'
    assert _types('x = 14.0 / 8')['x'] == 'f'


def __test_builtin_series_types__():
    """The builtin sources and counters carry their registry types"""
    types = _types(
        'from pynecore import lib\n'
        'c = lib.close\n'
        'i = lib.bar_index\n'
        't = lib.time\n'
        'p = lib.timeframe.period\n'
    )
    assert types['c'] == 'f'
    assert types['i'] == 'i'
    assert types['t'] == 'i'
    assert types['p'] == 's'


def __test_annotation_declares_rather_than_follows__():
    """
    An explicit annotation is the declaration, and the value has to fit it.

    ``float y = 2`` declares a float in Pine whatever the literal is, and
    ``int x = 2.0`` is rejected outright: an int value is a float, a float
    value is not an int, and a declaration the value contradicts would carry
    the lie into every pin downstream.
    """
    source = 'x: int = 2.0\ny: float = 2\n'
    types = _types(source)
    assert types['x'] == '?'
    assert types['y'] == 'f'
    diags = infer_module(ast.parse(source), 'test').diags
    assert [(d.origin.reason, d.line) for d in diags if d.origin is not None] \
        == [('type-mismatch', 1)]


def __test_series_and_na_wrappers_are_transparent__():
    """``Series[int]`` and ``NA[int]`` are storage, not a different Pine type"""
    types = _types(
        'from pynecore.types import Series, NA, Persistent\n'
        'a: Series[int] = 0\n'
        'b: NA[float] = 0\n'
        'c: Persistent[int] = 0\n'
    )
    assert types['a'] == 'i'
    assert types['b'] == 'f'
    assert types['c'] == 'i'


def __test_branch_join_widens_to_float__():
    """Two branches storing different types leave the variable joined"""
    types = _types(
        'from pynecore import lib\n'
        'x = 1\n'
        'if lib.bar_index > 0:\n'
        '    x = 2.5\n'
    )
    assert types['x'] == 'f'


def __test_loop_carried_variable_reaches_a_fixpoint__():
    """
    A variable the loop body widens is float AFTER the loop, not int.

    A single forward pass reads ``total`` as int at the top of the body and
    never revisits it; the bounded fixpoint is what closes that leak. The
    lattice is two high, so it always converges.
    """
    types = _types(
        'from pynecore import lib\n'
        'total = 0\n'
        'for i in range(1, 10):\n'
        '    total = total + lib.close\n'
    )
    assert types['total'] == 'f'
    assert types['i'] == 'i'


def __test_range_counter_follows_its_bounds__():
    """
    A ``for`` counter is typed by its bounds and is NOT truncated.

    MEASURED on TradingView (BINANCE:BTCUSDT 30m): ``for i = R / 8 to R / 4``
    with R = 14 iterates i = 1.75 and 2.75 -- the counter is an int-TYPED
    variable carrying a fractional value, so the loop is not a consuming slot.
    """
    assert _types('for i in range(1, 10):\n    pass\n')['i'] == 'i'
    assert _types('for i in range(1, 10, 2):\n    pass\n')['i'] == 'i'
    assert _types('for i in range(1.5, 10.0):\n    pass\n')['i'] == 'f'


def __test_comparisons_and_logic_are_bool__():
    """Every comparison and boolean operator yields bool whatever it compares"""
    types = _types(
        'from pynecore import lib\n'
        'a = lib.close > 1\n'
        'b = lib.bar_index == 0\n'
        'c = a and b\n'
        'e = not a\n'
    )
    assert all(types[name] == 'b' for name in ('a', 'b', 'c', 'e')), types


def __test_string_concatenation_stays_string__():
    """``+`` over two strings is the one non-numeric arithmetic Pine has"""
    assert _types('x = "a" + "b"')['x'] == 's'


def __test_unannotated_parameter_is_the_known_leak__():
    """
    An unannotated parameter is UNKNOWN, and says so with provenance.

    A call site closes it -- the parameter is typed per context -- but a
    function nothing calls has no context to be typed by, and then the origin
    has to name the parameter so the diagnostic can point at what to annotate.
    """
    tree = ast.parse(
        'def f(x):\n'
        '    return x + 1\n'
    )
    table = infer_module(tree, 'test')
    binding = table.bindings['f']['x']
    assert binding.ty == '?'
    assert binding.unknown is not None
    assert binding.unknown.reason == 'unannotated-param'
    assert binding.unknown.detail == 'x'


def __test_annotated_function_types_its_callers__():
    """A fully annotated function feeds its return type back to the call site"""
    types = _types(
        'def half(x: int) -> int:\n'
        '    return x\n'
        'y = half(4)\n'
    )
    assert types['y'] == 'i'


def __test_every_expression_node_is_stamped__():
    """
    The pass leaves no typed position blank.

    The type travels on the node, so a later pass that reuses the object keeps
    it for free -- but only if it was stamped in the first place.
    """
    tree = ast.parse(
        'from pynecore import lib\n'
        'x = lib.math.max(lib.bar_index / 2, 1) + lib.close\n'
    )
    infer_module(tree, 'test')
    for node in ast.walk(tree):
        if isinstance(node, (ast.BinOp, ast.Call, ast.Constant, ast.Compare, ast.IfExp)):
            assert hasattr(node, '_pine_ty'), ast.dump(node)


def __test_const_folded_literal_keeps_its_type__():
    """
    A stamp already on a literal outranks the literal's Python type.

    The constant folder replaces ``14 / 8`` with ``1.75``, which is
    indistinguishable from a float literal; it records the Pine type on the
    emitted node, and the inference has to believe it.
    """
    tree = ast.parse('x = 1.75')
    literal = tree.body[0].value  # type: ignore[attr-defined]
    literal._pine_ty = 'i'
    table = infer_module(tree, 'test')
    assert table.bindings['']['x'].ty == 'i'
    assert get_ty(literal) == 'i'


def __test_keyword_arguments_count_towards_arity__():
    """
    A keyword argument is an argument, so it picks the same overload.

    ``math.round`` splits on arity, and counting only the positional ones
    would resolve the named-precision spelling to the int form.
    """
    types = _types(
        'from pynecore import lib\n'
        'a = lib.math.round(1.5, 2)\n'
        'b = lib.math.round(1.5, precision=2)\n'
        'c = lib.math.round(1.5)\n'
        'd = lib.math.max(1, 2)\n'
        'e = lib.math.max(1, 2.0)\n'
    )
    assert types['a'] == 'f'
    assert types['b'] == 'f'
    assert types['c'] == 'i'
    assert types['d'] == 'i'
    assert types['e'] == 'f'


def __test_unpacked_arguments_hide_the_arity__():
    """An unpacking makes the arity unknowable, so no overload is picked"""
    types = _types(
        'from pynecore import lib\n'
        'args = (1.5, 2)\n'
        'x = lib.math.round(*args)\n'
    )
    assert types['x'] == '?'


def __test_forward_helper_call_resolves__():
    """
    A helper calling one defined further down is still typed.

    A single pass reads the callee's return type before its body was walked,
    which would leave the caller -- and everything downstream of it --
    UNKNOWN purely because of definition order.
    """
    types = _types(
        'def first():\n'
        '    return later()\n'
        '\n'
        'def later():\n'
        '    return 1\n'
        '\n'
        'z = first()\n'
    )
    assert types['z'] == 'i'


def __test_keyword_spelling_keeps_a_type_preserving_call__():
    """
    Naming an argument must not change what a call returns.

    ``math.abs`` and ``nz`` copy the type of their FIRST parameter, and reading
    that parameter positionally only worked while the caller spelled it
    positionally: ``math.abs(number=d)`` fell out of the typed subset.
    """
    types = _types(
        'from pynecore import lib\n'
        'a = lib.math.abs(number=1)\n'
        'b = lib.nz(source=1)\n'
        'c = lib.nz(1, replacement=2)\n'
        'd = lib.math.sign(number=1)\n'
        'e = lib.fixnan(source=1)\n'
        'f = lib.math.abs(number=1.0)\n'
        # A name the callee does not declare binds nothing
        'g = lib.math.abs(wrong=1)\n'
    )
    assert types['a'] == 'i'
    assert types['b'] == 'i'
    assert types['c'] == 'i'
    assert types['d'] == 'i'
    assert types['e'] == 'i'
    assert types['f'] == 'f'
    assert types['g'] == '?'


def __test_unpacking_defeats_a_type_preserving_call__():
    """An unpacking hides which position an argument landed on"""
    types = _types(
        'from pynecore import lib\n'
        'args = (1,)\n'
        'kw = {}\n'
        'a = lib.math.abs(*args)\n'
        'b = lib.math.max(*args)\n'
        'c = lib.math.max(**kw)\n'
    )
    assert types['a'] == '?'
    assert types['b'] == '?'
    assert types['c'] == '?'


def __test_long_forward_chain_resolves_completely__():
    """
    Chain depth must not decide whether a module types.

    Each pass resolves one more link, so a fixed pass budget left a long enough
    helper chain UNKNOWN; the budget follows the number of functions instead.
    """
    source = ''.join(
        f'def f{index}():\n    return f{index + 1}()\n\n' for index in range(12)
    ) + 'def f12():\n    return 1\n\nz = f0()\n'
    assert _types(source)['z'] == 'i'


def __test_same_named_nested_helpers_stay_apart__():
    """
    Two nested helpers sharing a name are two functions, not one.

    Keying signatures by the bare name let the first one walked win, so the
    int-returning helper's return type was handed to the float-returning
    helper's caller as well.
    """
    source = (
        'def outer_int():\n'
        '    def helper():\n'
        '        return 1\n'
        '    return helper()\n'
        '\n'
        'def outer_float():\n'
        '    def helper():\n'
        '        return 1.0\n'
        '    return helper()\n'
        '\n'
        'a = outer_int()\n'
        'b = outer_float()\n'
    )
    types = _types(source)
    assert types['a'] == 'i'
    assert types['b'] == 'f'
    # ... and the two signatures really are separate entries
    tree = ast.parse(source)
    table = infer_module(tree, 'test')
    assert table.funcs['outer_int·helper'].ret == 'i'
    assert table.funcs['outer_float·helper'].ret == 'f'


def __test_nested_helper_does_not_shadow_a_module_function__():
    """A call still resolves outward when the current scope declares no helper"""
    types = _types(
        'def shared():\n'
        '    return 1\n'
        '\n'
        'def outer():\n'
        '    return shared()\n'
        '\n'
        'z = outer()\n'
    )
    assert types['z'] == 'i'


def __test_mutual_recursion_stays_unknown_without_looping__():
    """Two helpers returning each other are untypable, and must still settle"""
    types = _types(
        'def ping():\n'
        '    return pong()\n'
        '\n'
        'def pong():\n'
        '    return ping()\n'
        '\n'
        'z = ping()\n'
    )
    assert types['z'] == '?'


def _folded_types(source: str, names: set[str]) -> dict[str, str]:
    """Const-fold a snippet and return the type stamped on each named value."""
    tree = ast.parse(source)
    ConstFoldTransformer().visit(tree)
    return {stmt.targets[0].id: get_ty(stmt.value)
            for stmt in tree.body
            if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id in names}


def __test_folded_types_are_scoped_like_their_values__():
    """
    A nested scope rebinding a name must not retype the outer one.

    The folder carries the Pine type of each constant name beside its value;
    a shared type map would let ``x = 1.0`` inside a function turn the outer
    int-typed ``x`` -- and everything folded from it -- into a float.
    """
    types = _folded_types(
        'x = 1\n'
        '\n'
        'def helper():\n'
        '    x = 1.0\n'
        '    return x\n'
        '\n'
        'y = x / 2\n',
        {'y'},
    )
    assert types['y'] == 'i'


def __test_folded_types_do_not_leak_between_branches__():
    """A name one branch rebinds keeps its outer type in the sibling branch"""
    tree = ast.parse(
        'from pynecore import lib\n'
        'x = 1\n'
        'if lib.bar_index > 0:\n'
        '    x = 1.0\n'
        'else:\n'
        '    y = x / 2\n'
    )
    ConstFoldTransformer().visit(tree)
    orelse = tree.body[2].orelse[0]  # type: ignore[attr-defined]
    assert get_ty(orelse.value) == 'i'


def __test_a_script_entry_parameter_is_typed_by_its_default__():
    """
    The runner calls an entry point with NO arguments.

    So an entry's parameter is not "the default if you omit it", it IS the
    default, on every bar -- which is how a compiled script receives its
    inputs, unannotated, because Pine's ``input.int``'s first parameter is the
    input's default VALUE. The type comes from the input constructor.
    """
    types = _types(
        'from pynecore import lib\n'
        '\n'
        '@lib.script.indicator("t")\n'
        'def main(length=lib.input.int(14), src=lib.input.float(1.0), flag=True):\n'
        '    step = length / 8\n'
        '    return step\n',
        'main',
    )
    assert types['length'] == 'i'
    assert types['src'] == 'f'
    assert types['flag'] == 'b'
    assert types['step'] == 'i'


def __test_an_ordinary_default_is_not_a_declaration__():
    """
    Everywhere else a default is what the caller MAY omit, and says nothing.

    ``helper(1.5)`` is a legal call of ``def helper(x=0)``, so reading ``int``
    off the ``0`` would be a guess -- and a guess that would go on to decide
    an overload. The default is only ever the JOIN partner of an actual
    argument, so with no call site the parameter stays unknown.
    """
    types = _types(
        'def helper(x=0):\n'
        '    return x\n',
        'helper',
    )
    assert types['x'] == '?'


def __test_an_annotation_still_wins_over_the_default__():
    """A spelled-out type is a declaration; an entry's default only fills a gap"""
    types = _types(
        'from pynecore import lib\n'
        '\n'
        '@lib.script.indicator("t")\n'
        'def main(x: float = lib.input.int(1)):\n'
        '    return x\n',
        'main',
    )
    assert types['x'] == 'f'


@pytest.mark.parametrize("call,expected", [
    ('cast_int(close)', 'i'),
    ('cast_float(close)', 'f'),
    ('cast_bool(close)', 'b'),
    ('cast_string(close)', 's'),
    ('cast_label(close)', 'o'),
])
def __test_the_compiled_pine_casts_are_typed__(call: str, expected: str):
    """
    A compiled script spells its Pine casts as bare ``pine_cast`` helpers.

    They are imported by name and survive import normalization untouched, so
    an unlisted one would drop the type of every value passing through it.
    """
    types = _types(
        'from pynecore.core.pine_cast import '
        'cast_int, cast_float, cast_bool, cast_string, cast_label\n'
        'from pynecore.lib import close\n'
        f'value = {call}\n'
    )
    assert types['value'] == expected
