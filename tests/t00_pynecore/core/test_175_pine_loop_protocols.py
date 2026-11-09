"""
The two forms of :func:`~pynecore.core.pine_range.pine_loop`, the counter of a Pine
``for`` loop with a re-read ``to`` bound.

The plain form takes ``by`` up front; compiled code of every version uses it, so
its protocol must not change. The ``entry=True`` form follows the evaluation order
MEASURED on TradingView (``from``, ``to``, ``by``, then ``to`` before every
iteration): its first ``step`` is the entry read of ``to``, and ``enter`` hands
over ``by``.
"""
import pytest

from pynecore import pine_loop


def __test_helper_run_old(from_num, to_reads, by=None):
    """Drive ``pine_loop`` the way older compiled code does."""
    loop = pine_loop(from_num) if by is None else pine_loop(from_num, by)
    reads = iter(to_reads)
    values = []
    while loop.step(next(reads)):
        values.append(loop.value)
    return values


def __test_helper_run_new(from_num, to_fn, by_fn=None, body=None):
    """Drive ``pine_loop(..., entry=True)`` the way the compiler emits it."""
    loop = pine_loop(from_num, entry=True)
    values = []
    while loop.step(to_fn()):
        if loop.entering:
            loop.enter(by_fn() if by_fn is not None else None)
            continue
        values.append(loop.value)
        if body is not None:
            body()
    return values


def __test_pine_loop_keeps_its_protocol__():
    """``pine_loop(from, by)`` + ``while step(to)``: every ``step`` reads ``to``"""
    assert __test_helper_run_old(0, [3, 3, 3, 3, 3]) == [0.0, 1.0, 2.0, 3.0]
    assert __test_helper_run_old(0, [5, 5, 5, 5], by=2) == [0.0, 2.0, 4.0]
    assert __test_helper_run_old(3, [0, 0, 0, 0, 0]) == [3.0, 2.0, 1.0, 0.0]
    # A bound that shrinks while the loop runs ends it early
    assert __test_helper_run_old(0, [5, 1, 1]) == [0.0, 1.0]
    # MEASURED (CAPITALCOM:EURUSD 60, v4-v6): a ``by`` of zero or below is a run
    # time error, a descending loop's too
    with pytest.raises(ValueError):
        __test_helper_run_old(0, [3], by=0)
    with pytest.raises(ValueError):
        __test_helper_run_old(0, [4], by=-2)
    with pytest.raises(ValueError):
        __test_helper_run_old(4, [0], by=-2)


def __test_the_bound_is_checked_with_the_pine_tolerance__():
    """MEASURED: ``for i = 1.0 to 0.1 by 0.3`` runs four times, the counter
    accumulated to 0.09999999999999998; a difference of exactly ``EPSILON`` is in,
    0.1 ulp above it is out"""
    assert __test_helper_run_old(1.0, [0.1] * 6, by=0.3) == [
        1.0, 0.7, 0.39999999999999997, 0.09999999999999998]
    assert __test_helper_run_old(1.0, [0.0 + 1e-10] * 4, by=1.0) == [1.0, 0.0]
    assert __test_helper_run_old(0.0, [1.0 - 1e-10] * 4, by=1.0) == [0.0]
    assert __test_helper_run_old(0.0, [1.0 - 9e-11] * 4, by=1.0) == [0.0, 1.0]


def __test_entry_form_follows_the_measured_order__():
    """``for i = a() to b() by c()`` evaluates ``a, b, c, b, body, b, body, b``"""
    trace = []

    def to_fn():
        trace.append('b')
        return 1

    def by_fn():
        trace.append('c')
        return 1

    trace.append('a')
    values = __test_helper_run_new(0, to_fn, by_fn, body=lambda: trace.append('body'))
    assert values == [0.0, 1.0]
    assert trace == ['a', 'b', 'c', 'b', 'body', 'b', 'body', 'b']


def __test_entry_form_direction_comes_from_the_entry_read__():
    """The entry read of ``to`` fixes the direction; later reads only bound it"""
    reads = iter([3, -1, -1, -1])
    assert __test_helper_run_new(0, lambda: next(reads)) == []
    reads = iter([3, 0, 0, 0])
    assert __test_helper_run_new(0, lambda: next(reads)) == [0.0]
    reads = iter([0, 3, 3, 3, 3])
    assert __test_helper_run_new(3, lambda: next(reads)) == [3.0]
    assert __test_helper_run_new(3, lambda: 0, lambda: 2) == [3.0, 1.0]
    with pytest.raises(ValueError):
        __test_helper_run_new(0, lambda: 3, lambda: 0)
