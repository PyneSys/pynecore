"""
Pine-typed static view of the for-loop counters.

At runtime every counter value is a float (a Pine int is a double). Pine types
the counter from the bounds instead: an int loop when ``from``, ``to`` and ``by``
are all int, a float loop as soon as one of them is a float. The constrained
TypeVar says exactly that, and a bound whose type is unknown keeps the counter
an int, the type every numeric slot takes.
"""
from typing import Generic, Iterable, TypeVar

_TNum = TypeVar('_TNum', int, float)


def pine_range(from_num: _TNum, to_num: _TNum, step_num: _TNum | None = None) -> Iterable[_TNum]:
    """
    Emulates Pine Script's for loop range behavior.

    :param from_num: Start value (inclusive)
    :param to_num: End value (inclusive)
    :param step_num: Step value (optional, defaults to +1/-1 based on direction)
    :return: The counter values from from_num to to_num (inclusive)
    :raises ValueError: If step_num is zero
    """


class PineLoop(Generic[_TNum]):
    """
    Counter of a Pine ``for`` loop whose ``to`` bound is not loop-invariant.
    """
    value: _TNum

    def __init__(self, from_num: _TNum, step_num: _TNum | None = None) -> None: ...

    def step(self, to_num: int | float) -> bool:
        """
        Advance the counter and report whether the body must run again.

        :param to_num: The loop's ``to`` bound, freshly evaluated by the caller
        :return: True while the counter is still within the bound
        :raises ValueError: If the step is zero
        """


def pine_loop(from_num: _TNum, step_num: _TNum | None = None) -> PineLoop[_TNum]:
    """
    Start a Pine ``for`` loop with a bound that has to be re-read each iteration.

    :param from_num: Start value (inclusive), evaluated once
    :param step_num: Step value (optional), evaluated once
    :return: The loop counter the ``while`` condition drives
    """
