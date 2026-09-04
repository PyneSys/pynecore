"""
@pyne

A position the runtime has no witness for does not block the pin next to it.

An overload site like ``f(math.ceil(x), someEnum, src)`` carries one int-typed
argument and one the pass can only call an object. Refusing the whole site over
the object left the int argument to the VALUE dispatch, which cannot tell an
int-typed 30.0 from a float 30.0 -- a float overload delegating to the int one
through ``math.ceil`` then recursed until the stack ran out.

The wildcard says "this position carries nothing": the implementation is pinned
only where the witnessed positions alone name exactly one, and the first call
checks the wildcard positions against what that implementation declares before
the pin is trusted.
"""
from pynecore.core.overload import overload

from pynecore.lib import script, math, color


class Marker:
    """A user type: an argument the pin has no witness value for"""

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

    m = Marker("m")
    return {
        # int-typed argument (int / int keeps the int type, value 1.75)
        "int_typed": pick(7 / 4, m, 2.0),
        "float_arg": pick(1.75, m, 2.0),
        # The wildcard is not what decides here -- the int position is
        "object_first": only_object(m, 3),
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
            assert plot["no_pin"]
            if i >= 1:
                break


def __test_the_selection_ignores_only_the_wildcard_positions__():
    """A pin whose witnessed positions leave two candidates picks neither"""
    from pynecore.core.overload import _PIN_ANY, _select_pinned, Implementation

    def one(a: int, b: str):
        return 1

    def two(a: int, b: Marker):
        return 2

    impls = [Implementation(one), Implementation(two)]
    # 'i*': the int witness matches both, the wildcard is the position that
    # would decide -- so the values keep the decision
    assert _select_pinned(impls, 'i' + _PIN_ANY) is None
    # 'is': fully witnessed, one candidate
    assert _select_pinned(impls, 'is') is impls[0]
