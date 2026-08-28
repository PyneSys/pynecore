from typing import Generator, overload

from pynecore.types.pine_types import PyneFloat, PyneInt


@overload
def pine_range(from_num: int, to_num: int, step_num: int | None = None) -> range: ...


@overload
def pine_range(from_num: PyneInt | PyneFloat, to_num: PyneInt | PyneFloat,
               step_num: PyneInt | PyneFloat | None = None) -> Generator[float, None, None]: ...


def pine_range(from_num: PyneInt | PyneFloat, to_num: PyneInt | PyneFloat, step_num: PyneInt | PyneFloat | None = None):
    """
    Emulates Pine Script's for loop range behavior.

    :param from_num: Start value (inclusive)
    :param to_num: End value (inclusive)
    :param step_num: Step value (optional, defaults to +1/-1 based on direction)
    :return: A native ``range`` for integer bounds, otherwise a generator that yields
             values from from_num to to_num (inclusive)
    :raises ValueError: If step_num is zero
    """
    # Fast path: pure-integer bounds map exactly onto a native range, which iterates at
    # C speed instead of resuming a Python generator on every step. The vast majority of
    # Pine for loops are integer index ranges, so this is the common case.
    if isinstance(from_num, int) and isinstance(to_num, int) and (step_num is None or isinstance(step_num, int)):
        if from_num <= to_num:
            step = 1 if step_num is None else abs(step_num)
            if step == 0:
                raise ValueError("Step cannot be zero in pine_range")
            # +1 makes the upper bound inclusive, matching Pine's `to`
            return range(from_num, to_num + 1, step)
        step = -1 if step_num is None else -abs(step_num)
        if step == 0:
            raise ValueError("Step cannot be zero in pine_range")
        # -1 makes the lower bound inclusive for the descending direction
        return range(from_num, to_num - 1, step)

    return _pine_range_float(from_num, to_num, step_num)


def _pine_range_float(from_num: PyneFloat, to_num: PyneFloat, step_num: PyneFloat | None = None):
    """
    Generator fallback for Pine for-loop ranges with non-integer bounds.

    :param from_num: Start value (inclusive)
    :param to_num: End value (inclusive)
    :param step_num: Step value (optional, defaults to +1/-1 based on direction)
    :return: A generator that yields values from from_num to to_num (inclusive)
    :raises ValueError: If step_num is zero
    """
    # Determine direction based on from_num and to_num
    direction = 1 if from_num <= to_num else -1

    # Use default step if none provided
    step_val = step_num if step_num is not None else direction

    # Prevent infinite loops
    if step_val == 0:
        raise ValueError("Step cannot be zero in pine_range")

    # Ensure step direction matches the from->to direction
    if (direction > 0 > step_val) or (direction < 0 < step_val):
        step_val = -step_val

    # Generate values
    current = from_num
    if direction > 0:
        # Ascending loop
        while current <= to_num:
            yield current
            current += step_val
            # Safety check to prevent infinite loops due to floating point precision
            if step_val > 0 and current > to_num + abs(step_val):
                break
    else:
        # Descending loop
        while current >= to_num:
            yield current
            current += step_val
            # Safety check to prevent infinite loops due to floating point precision
            if step_val < 0 and current < to_num - abs(step_val):
                break


class PineLoop:
    """
    Counter of a Pine ``for`` loop whose ``to`` bound is not loop-invariant.

    MEASURED on TradingView (BINANCE:BTCUSDT@30): the ``to`` expression is
    re-evaluated before EVERY iteration, while ``from`` and ``by`` are evaluated
    once, when the loop is entered. A body that shrinks the collection the bound
    is read from therefore ends the loop early instead of running off the end,
    and a body that grows it keeps iterating.

    The bound cannot travel as a value, so the compiler emits the loop as a
    ``while`` whose condition re-evaluates the Pine expression at the call site::

        loop__1 = pine_loop(0)
        while loop__1.step(array.size(levels) - 1):
            i = loop__1.value
            ...
    """

    __slots__ = ('value', '_by', '_step', '_ascending', '_started')

    def __init__(self, from_num: PyneInt | PyneFloat, step_num: PyneInt | PyneFloat | None = None):
        self.value = from_num
        # The ``by`` expression as written, absent when the loop has none; the
        # signed step it resolves to needs the direction, known only on entry.
        self._by = step_num
        self._step: PyneInt | PyneFloat = 1
        self._ascending = True
        self._started = False

    def step(self, to_num: PyneInt | PyneFloat) -> bool:
        """
        Advance the counter and report whether the body must run again.

        :param to_num: The loop's ``to`` bound, freshly evaluated by the caller
        :return: True while the counter is still within the bound
        :raises ValueError: If the step is zero
        """
        if self._started:
            self.value += self._step
        else:
            self._started = True
            # Direction is fixed when the loop is entered: it is what gives the
            # default step its sign, and Pine never flips it mid-loop.
            self._ascending = self.value <= to_num
            if self._by is None:
                self._step = 1 if self._ascending else -1
            elif self._by == 0:
                raise ValueError("Step cannot be zero in pine_loop")
            else:
                self._step = -self._by if (self._by < 0) == self._ascending else self._by
        return self.value <= to_num if self._ascending else self.value >= to_num


def pine_loop(from_num: PyneInt | PyneFloat,
              step_num: PyneInt | PyneFloat | None = None) -> PineLoop:
    """
    Start a Pine ``for`` loop with a bound that has to be re-read each iteration.

    :param from_num: Start value (inclusive), evaluated once
    :param step_num: Step value (optional), evaluated once
    :return: The loop counter the ``while`` condition drives
    """
    return PineLoop(from_num, step_num)
