"""
@pyne lib

Stateful implementations of ``lib.math.random`` and ``lib.math.sum``. They
live in their own small module because the ``@pyne`` marker is module-level
and the host module (``lib/math.py``) must stay untransformed; the host
re-exports the functions, and the layouts travel on the function objects.
"""
# Absolute imports on purpose: the call-site classifier resolves absolute
# imports at transform time, so NA() calls stay direct instead of anchored
from typing import TypeVar

from pynecore.types import NA, Persistent, PyneFloat, PyneInt
from pynecore.core.random import PineRandom as _PineRandom
from pynecore.core.rolling_sum import SumMachine as _SumMachine, rolling_sum_step

TFI = TypeVar('TFI', float, int)

__all__ = ['random', 'sum']


# The lazy-init narrowing of ``prng`` is invisible to the IDE: ``Persistent`` is a
# marker the AST transformer rewrites, so flow analysis keeps the ``| None`` arm.
# noinspection PyShadowingBuiltins,PyShadowingNames,PyUnresolvedReferences
def random(min: TFI | NA[TFI] = 0, max: TFI | NA[TFI] = 1, seed: PyneInt = NA(int)) -> PyneFloat:
    """
    Returns a random number between two numbers.

    :param min: The minimum number.
    :param max: The maximum number.
    :param seed: The seed for the random number generator.
    :return: A random number between the minimum and maximum numbers.
    """
    prng: Persistent[_PineRandom | None] = None
    if prng is None:  # Lazy init: the PRNG must not be created before the seed is known
        # The seed defaults to na, which means "unseeded": the generator then
        # starts from the clock. Handing the na to the PRNG would XOR it into the
        # state and every single draw would come back na.
        prng = _PineRandom(seed if seed == seed else None)  # is_na_arg
    res = prng.random(min, max)
    return res


# The PRNG advances once per CALL by nature: rolling it back to bar-start
# state on the same-bar re-executions of a shared loop call site would hand
# every iteration the same draw. The layout flag shields it from the
# ``__loop_state__`` rollback (see ``_collect_builtins``).
getattr(random, '__pyne_layout__')['per_call'] = True


# The lazy-init narrowing of ``machine`` is invisible to the IDE, like ``prng`` above.
# noinspection PyShadowingBuiltins,PyUnresolvedReferences
def sum(source: TFI | NA[TFI], length: int) -> PyneFloat:
    """
    Returns the sum of a series over a specified length.

    The window is na-compacted: an na bar returns na and is not stored, so the sum
    always covers the last ``length`` non-na values.

    :param source: Source series
    :param length: Length of the sum
    :return: The sliding sum of the series
    """
    # The whole machine, its value window included, is one object: the loop-site
    # rollback restores it in O(1) through its own snapshot protocol instead of
    # copying slots and a series window back on every iteration. See
    # ``core.rolling_sum`` for the machine and the measured laws behind it.
    machine: Persistent[_SumMachine | None] = None
    if machine is None:
        machine = _SumMachine()
    return rolling_sum_step(machine, source, length)
