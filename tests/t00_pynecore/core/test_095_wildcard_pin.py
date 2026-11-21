"""
@pyne

A position the runtime has no witness for does not block the pin next to it.

An overload site like ``f(math.ceil(x), someEnum, src)`` carries one int-typed
argument and one the pass can only call an object. Refusing the whole site over
the object left the int argument to the VALUE dispatch, which cannot tell an
int-typed 30.0 from a float 30.0 -- a float overload delegating to the int one
through ``math.ceil`` then recursed until the stack ran out.

The wildcard says "this position has no witness": the type pass reads it from the
argument's static type, and the implementation it names is trusted as it is. An
argument of NO known type -- an instance of a plain class -- decides nothing: that
call carries no pin and dispatches on its values.
"""
from pathlib import Path

from pynecore.core.overload import overload
from pynecore.core.pine_udt import udt

from pynecore.lib import script, math, color


@udt
class Marker:
    """A user type: an argument the pin has no witness value for"""
    name: str = ""


class Plain:
    """A plain class: an argument the type pass knows no type of"""

    def __init__(self, name: str):
        self.name = name


@script.indicator("wildcard pin")
def main():
    @overload
    def pick(length: int, marker: Marker, scale: float):
        return 100.0 + length * scale

    @overload
    def pick(length: float, marker: Marker, scale: float):
        # The float form delegates to the int one, which is the shape that
        # recursed forever when the site could not be pinned
        return pick(math.ceil(length), marker, scale)

    @overload
    def only_object(marker: Marker, length: int):
        return 1.0

    @overload
    def only_object(marker: Marker, length: str):
        return 2.0

    @overload
    def plain_pick(length: int, marker: Plain):
        return 1.0

    @overload
    def plain_pick(length: float, marker: Plain):
        return 2.0

    m = Marker("m")
    p = Plain("p")
    return {
        # int-typed argument (int / int keeps the int type, value 1.75)
        "int_typed": pick(7 / 4, m, 2.0),
        "float_arg": pick(1.75, m, 2.0),
        # The wildcard is not what decides here -- the int position is
        "object_first": only_object(m, 3),
        # The plain class leaves the call to its values: 1.75 runs the float one
        "plain_int_typed": plain_pick(7 / 4, p),
        # An argument the values must decide: no int type, so no pin at all
        "no_pin": color.new(color.red, 50) is not None,
    }


def __test_a_wildcard_position_does_not_block_the_pin__(csv_reader, runner):
    """The int-typed argument reaches the int implementation, no recursion"""
    with csv_reader('series_if_for.csv', subdir="data") as cr:
        for i, (_candle, plot) in enumerate(runner(cr).run_iter()):
            # int-typed 1.75 takes the int implementation: 100 + 1.75 * 2
            assert plot["int_typed"] == 103.5
            # ... a float-typed one goes through ceil and lands on 100 + 2 * 2
            assert plot["float_arg"] == 104.0
            assert plot["object_first"] == 1.0
            assert plot["plain_int_typed"] == 2.0
            assert plot["no_pin"]
            if i >= 1:
                break


def __test_an_argument_of_unknown_type_is_reported_and_not_pinned__():
    """The call with the plain-class argument carries no pin and is reported"""
    import ast
    from pynecore.transformers.pine_type_rules import get_pin
    from ..pine_analysis import analyse_module

    analysed = analyse_module(Path(__file__))
    assert analysed is not None
    tree, table, _ = analysed
    pins = sorted((ast.unparse(node.func), str(get_pin(node))) for node in ast.walk(tree)
                  if isinstance(node, ast.Call)
                  and ast.unparse(node.func) in ('pick', 'plain_pick'))
    # The int-typed and the float argument next to the udt name their
    # implementations, the delegating call inside the float one included
    assert pins == [('pick', 'f*f=1'), ('pick', 'i*f=0'), ('pick', 'i*f=0'),
                    ('plain_pick', 'None')]
    found = [(diag.code, diag.args) for diag in table.diags if diag.code in ('E0071', 'E0072')]
    assert found == [('E0071', ('2', 'plain_pick'))]


def __test_helper_group(*impls):
    """The shapes as one whole group, each with its registration index"""
    return [impl._replace(slot=slot, size=len(impls)) for slot, impl in enumerate(impls)]


def __test_the_selection_ignores_only_the_wildcard_positions__():
    """A pin whose witnessed positions leave two candidates picks neither"""
    from pynecore.transformers.pine_type_rules import PIN_ANY, ImplSig, object_ty, resolve_pin

    marker = object_ty('script#Marker')
    # ``one(a: int, b: str)`` and ``two(a: int, b: Marker)``
    impls = __test_helper_group(ImplSig(('i', 's'), 2, False, 'i', 'nn'),
                                ImplSig(('i', marker), 2, False, 'f', 'nn'))
    # 'i*': the int witness matches both, the wildcard is the position that
    # would decide -- so the values keep the decision
    assert resolve_pin(impls, 'i' + PIN_ANY) is None
    # 'is': fully witnessed, one candidate
    assert resolve_pin(impls, 'is') is impls[0]


def __test_a_wildcard_the_widening_pass_reads_differently_blocks_the_pin__():
    """An exact-pass answer whose wildcard types differ from a widening candidate picks neither"""
    from pynecore.transformers.pine_type_rules import PIN_ANY, ImplSig, object_ty, resolve_pin

    first = object_ty('script#First')
    second = object_ty('script#Second')
    # ``one(a: int, b: First)`` and ``two(a: float, b: Second)``: an int with a
    # Second fails ``one`` and widens into ``two``
    impls = __test_helper_group(ImplSig(('i', first), 2, False, 'f', 'nn'),
                                ImplSig(('f', second), 2, False, 'f', 'nn'))
    assert resolve_pin(impls, 'i' + PIN_ANY) is None


def __test_the_wildcard_selection_binds_omitted_defaults__():
    """An implementation whose optional parameter the call omits is a candidate, its default checked"""
    from pynecore.transformers.pine_type_rules import PIN_ANY, ImplSig, object_ty, resolve_pin

    marker = object_ty('script#Marker')
    # ``as_int(x: int, marker: Marker, n: int = 0)`` and its float twin
    impls = __test_helper_group(
        ImplSig(('i', marker, 'i'), 2, False, 'i', 'nny', wide_fits='nny'),
        ImplSig(('f', marker, 'i'), 2, False, 'f', 'nny', wide_fits='nny'))
    # ``bad_default(x: int, marker: Marker, n: float = True)``: a bool is no
    # float to the selector, widened or not
    bad_default = __test_helper_group(
        ImplSig(('i', marker, 'f'), 2, False, 'i', 'nnn', wide_fits='nnn'))
    # 'i*' on a two-argument call: the int-typed position names the int implementation
    assert resolve_pin(impls, 'i' + PIN_ANY) is impls[0]
    # A default that does not fit its declared type rules the implementation out
    assert resolve_pin(bad_default, 'i' + PIN_ANY) is None
    # More arguments than positional parameters: no candidate
    assert resolve_pin(impls, 'i' + PIN_ANY * 3) is None
