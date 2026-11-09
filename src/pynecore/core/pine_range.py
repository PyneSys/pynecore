from typing import Iterable

from pynecore.core.pine_compare import EPSILON
from pynecore.types.pine_types import PyneFloat, PyneInt

#: MEASURED on TradingView (CAPITALCOM:EURUSD@60, v4-v6): a ``by`` of zero or
#: below stops the script at run time, also on a descending loop -- ``by`` is
#: the step's size, the direction comes from ``from`` and ``to``. A na ``by``
#: is no error: the counter turns na after the first iteration and the loop ends.
_STEP_ERROR = "The step of a for loop must be greater than zero"


def pine_range(from_num: PyneInt | PyneFloat, to_num: PyneInt | PyneFloat,
               step_num: PyneInt | PyneFloat | None = None) -> Iterable[float]:
    """
    Emulates Pine Script's for loop range behavior.

    The counter is a Pine number, so every value it takes is a float: a Pine int
    is a double at runtime, and a ``for`` loop does not truncate its bounds
    (MEASURED: a fractional ``from`` makes the counter fractional). The counter
    is accumulated step by step and checked against ``to`` with Pine's tolerant
    comparison (MEASURED: ``for i = 1.0 to 0.1 by 0.3`` runs four times, the last
    one at 0.09999999999999998; a difference of exactly ``EPSILON`` is still in).

    :param from_num: Start value (inclusive)
    :param to_num: End value (inclusive)
    :param step_num: Step size (optional, defaults to 1); the direction comes from
                     ``from_num`` and ``to_num``
    :return: The counter values from from_num to to_num (inclusive)
    :raises ValueError: If step_num is zero or negative
    """
    if step_num is not None and step_num <= 0:
        raise ValueError(_STEP_ERROR)
    # Fast path: integral bounds map exactly onto a native range, which iterates at C
    # speed instead of resuming a Python generator on every step; ``map(float, ...)``
    # hands the values over as floats without a Python-level frame per step. The vast
    # majority of Pine for loops are integer index ranges, so this is the common case.
    # ``x % 1 == 0`` is the value test: it is False for nan, inf and the NA object, so
    # every na bound takes the generator, which runs zero iterations for it.
    if from_num % 1 == 0 and to_num % 1 == 0 and (step_num is None or step_num % 1 == 0):
        start = int(from_num)
        stop = int(to_num)
        step = 1 if step_num is None else int(step_num)
        if start <= stop:
            # +1 makes the upper bound inclusive, matching Pine's `to`
            return map(float, range(start, stop + 1, step))
        # -1 makes the lower bound inclusive for the descending direction
        return map(float, range(start, stop - 1, -step))

    return _pine_range_float(from_num, to_num, step_num)


def _pine_range_float(from_num: PyneFloat, to_num: PyneFloat, step_num: PyneFloat | None = None):
    """
    Generator fallback for Pine for-loop ranges with non-integer bounds.

    :param from_num: Start value (inclusive)
    :param to_num: End value (inclusive)
    :param step_num: Step size (optional, defaults to 1), already checked positive
    :return: A generator that yields values from from_num to to_num (inclusive)
    """
    # Determine direction based on from_num and to_num
    direction = 1 if from_num <= to_num else -1

    # The step's size, signed by the direction
    step_val = direction if step_num is None else step_num * direction

    # Generate values: the counter is a Pine number, a float from the first value on
    current = from_num + 0.0
    if direction > 0:
        # Ascending loop: Pine's tolerant ``<=``
        while current - to_num <= EPSILON:
            yield current
            current += step_val
            # Safety check to prevent infinite loops due to floating point precision
            if step_val > 0 and current > to_num + abs(step_val):
                break
    else:
        # Descending loop: Pine's tolerant ``>=``
        while to_num - current <= EPSILON:
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

    MEASURED on TradingView (CAPITALCOM:EURUSD@60): the exact order is ``from``,
    ``to``, ``by``, then ``to`` before every iteration -- ``for i = a() to b() by
    c()`` runs ``a, b, c, b, body, b, body, b`` -- and every read of ``to`` is one
    call site (a ``var`` counter in the bound function advances on each). The
    plain form above reads ``by`` first and has no separate entry read of ``to``,
    which only shows when the header calls something with an effect or state. For
    such a header the compiler passes ``entry=True``: the first :meth:`step` is
    the entry read that fixes the direction, the body hands over ``by`` once
    through :meth:`enter` and goes back to the condition for the first check::

        loop__1 = pine_loop(a(), entry=True)
        while loop__1.step(b()):
            if loop__1.entering:
                loop__1.enter(c())
                continue
            i = loop__1.value
            ...
    """

    __slots__ = ('value', 'entering', '_by', '_step', '_ascending', '_entry', '_phase')

    def __init__(self, from_num: PyneInt | PyneFloat, step_num: PyneInt | PyneFloat | None = None,
                 *, entry: bool = False):
        # The counter is a Pine number: a float from the first value on (an NA
        # object passes through the addition unchanged)
        self.value = from_num + 0.0
        # The ``by`` expression as written, absent when the loop has none; the
        # signed step it resolves to needs the direction, known only on entry.
        self._by = step_num
        self._step: PyneInt | PyneFloat = 1
        self._ascending = True
        self._entry = entry
        # True between the entry read and :meth:`enter`
        self.entering = False
        # 0: not entered, 1: entered, ``by`` pending, 2: next step() is the first
        # bound check, 3: running
        self._phase = 0

    def _signed_step(self, by: PyneInt | PyneFloat | None) -> PyneInt | PyneFloat:
        """
        The step the ``by`` value gives in the direction fixed on entry.

        :param by: The loop's ``by`` value, or None when the loop has none
        :return: The signed step
        :raises ValueError: If the step is zero or negative
        """
        if by is None:
            return 1 if self._ascending else -1
        if by <= 0:
            raise ValueError(_STEP_ERROR)
        return by if self._ascending else -by

    def step(self, to_num: PyneInt | PyneFloat) -> bool:
        """
        Advance the counter and report whether the body must run again.

        In ``entry`` mode the first call is the entry read of ``to``: it fixes the
        direction and returns True, and the body then calls :meth:`enter`.

        :param to_num: The loop's ``to`` bound, freshly evaluated by the caller
        :return: True while the counter is still within the bound
        :raises ValueError: If the step is zero or negative
        """
        phase = self._phase
        if phase == 3:
            self.value += self._step
        elif phase == 0:
            # Direction is fixed when the loop is entered: it is what gives the
            # default step its sign, and Pine never flips it mid-loop.
            self._ascending = self.value <= to_num
            if self._entry:
                self._phase = 1
                self.entering = True
                return True
            self._step = self._signed_step(self._by)
            self._phase = 3
        else:
            self._phase = 3
        # Pine's tolerant ``<=`` / ``>=``
        if self._ascending:
            return self.value - to_num <= EPSILON
        return to_num - self.value <= EPSILON

    def enter(self, step_num: PyneInt | PyneFloat | None = None) -> None:
        """
        Take the ``by`` value after the entry read of an ``entry`` mode loop.

        :param step_num: The loop's ``by`` value, or None when the loop has none
        :raises ValueError: If the step is zero or negative
        """
        self._step = self._signed_step(step_num)
        self.entering = False
        self._phase = 2


def pine_loop(from_num: PyneInt | PyneFloat,
              step_num: PyneInt | PyneFloat | None = None,
              *, entry: bool = False) -> PineLoop:
    """
    Start a Pine ``for`` loop with a bound that has to be re-read each iteration.

    :param from_num: Start value (inclusive), evaluated once
    :param step_num: Step value (optional), evaluated once; an ``entry`` mode loop
                     takes it through :meth:`PineLoop.enter` instead
    :param entry: Read ``to`` once on entry before ``by`` and the first check, as
                  TradingView does (see :class:`PineLoop`)
    :return: The loop counter the ``while`` condition drives
    """
    return PineLoop(from_num, step_num, entry=entry)
