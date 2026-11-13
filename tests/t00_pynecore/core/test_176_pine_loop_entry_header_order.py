"""
@pyne
"""
from pynecore import pine_loop
from pynecore.lib import array, bar_index, barstate, high, label, plot, script, string

#: MEASURED on TradingView: the header is read ``from``, ``to``, ``by``, then ``to``
#: before every pass -- ``for i = a() to b() by c()`` logs ``a b c b body0 b body1 b
#: body2 b`` for ``a() = 0``, ``b() = 2``, ``c() = 1``.
_EXPECTED = "a b c b body0 b body1 b body2 b"


def a(log):
    array.push(log, "a")
    return 0


def b(log):
    array.push(log, "b")
    return 2


def c(log):
    array.push(log, "c")
    return 1


@script.indicator("for header reading order", shorttitle="FHRO")
def main():
    log = array.new_string(0)
    ok = 0.0
    if barstate.islast:
        # The compiled form of the v6 ``for i = a() to b() by c()``: ``pine_loop(a(),
        # entry=True)`` reads ``to`` once on entry, ``enter(c())`` takes ``by``, and
        # every pass re-reads ``to`` in the condition.
        loop = pine_loop(a(log), entry=True)
        while loop.step(b(log)):
            if loop.entering:
                loop.enter(c(log))
                continue
            i = loop.value
            array.push(log, "body" + string.tostring(i))
        label.new(bar_index, high, array.join(log, " "))
        ok = 1.0 if array.join(log, " ") == _EXPECTED else 0.0
    plot(ok, "ok")


def __test_a_v6_for_header_is_read_in_the_measured_order__(runner, log):
    """A ``for`` header whose bounds call functions reads them as TradingView does.

    A v6 loop whose bound calls a function is compiled to ``pine_loop(a(),
    entry=True)`` with ``enter(c())``. Without the entry mode the runtime had no
    ``entry`` argument and no ``enter()``, so such a script failed with
    ``TypeError``; with it the header runs ``a, b, c, b`` and ``b`` again before
    every pass, which the script checks against the measured TradingView log.
    """
    from pynecore.types.ohlcv import OHLCV

    base = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC, in ms
    bars = [OHLCV(timestamp=base + i * 60_000, open=1.0, high=1.0, low=1.0, close=1.0,
                  volume=1.0) for i in range(5)]
    oks = [pv["ok"] for _, pv in runner(iter(bars)).run_iter()]
    assert oks == [0.0, 0.0, 0.0, 0.0, 1.0], f"header order check per bar: {oks}"
    log.info("v6 for header read as %s", _EXPECTED)
