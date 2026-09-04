"""
@pyne

A body whose inner overload site differs per context becomes state-carrying.

The type pass marks such sites on the definition, and the isolation pass gives
that scope an instance-vector slot -- which makes the definition take the
hidden state parameter. The carrier fixpoint runs BEFORE the slot exists, so it
has to anticipate the mark: without it the definition grew the parameter while
its call sites kept calling it bare, and every such call raised "missing 1
required positional argument".
"""
from pynecore.lib import script, math, array


@script.indicator("pin carrier call sites")
def main():
    # ``math.round`` is an overload group, and the pin its call takes depends on
    # what ``pct`` is in the context the body runs in -- an int in one, a float
    # in the other. The body is shared, so the site reads its pin per instance
    def pick(pct, table_a, table_b, use_a):
        return array.get(table_a if use_a else table_b, math.round(pct))

    a = array.from_items(10.0, 11.0, 12.0)
    b = array.from_items(20.0, 21.0, 22.0)
    return {
        # int-typed argument (int / int keeps the int type)
        "int_ctx": pick(4 / 2, a, b, True),
        # float-typed argument at the same site
        "float_ctx": pick(1.4, a, b, False),
        "int_ctx_b": pick(2, a, b, False),
    }


def __test_a_pin_carrying_body_is_called_with_its_state__(csv_reader, runner):
    """Every call site of the definition passes the state parameter it grew"""
    with csv_reader('series_if_for.csv', subdir="data") as cr:
        for i, (_candle, plot) in enumerate(runner(cr).run_iter()):
            assert plot["int_ctx"] == 12.0
            assert plot["float_ctx"] == 21.0
            assert plot["int_ctx_b"] == 22.0
            if i >= 1:
                break
