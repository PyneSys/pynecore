"""
@pyne

A function holding ``request.security``, called twice on a bar through a value
(``h = f``), writes its one context twice: the per-call-site instantiation sees only
calls by name. That is R0054, not an internal error.
"""
from pynecore.lib import bar_index, close, plot, request, script, syminfo, timeframe


def same_context():
    return request.security(syminfo.tickerid, timeframe.period, close + bar_index)


@script.indicator(title="Security through a value")
def main():
    h = same_context
    a = h()
    b = h()
    plot(a + b, "ab")


def __test_a_second_write_outside_a_loop_is_r0054__(runner):
    """The second write of the bar stops the run with R0054 at that bar"""
    import pytest
    from pynecore import errorcodes
    from pynecore.types.ohlcv import OHLCV

    bars = [OHLCV(timestamp=1_735_689_600_000 + i * 300_000, open=1.0, high=1.0, low=1.0,
                  close=1.0, volume=1.0) for i in range(3)]
    with pytest.raises(errorcodes.SecurityNotRewrittenError) as raised:
        for _ in runner(iter(bars)).run_iter():
            pass
    assert raised.value.params == {'function': 'request.security'}
    assert raised.value.bar_index == 0
