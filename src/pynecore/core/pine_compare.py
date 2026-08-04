"""
Pine's tolerant float comparison.

TradingView does not compare floats bit-exactly: two values whose difference is
at most ``EPSILON`` are equal, at every magnitude — the tolerance is absolute,
not relative, and the boundary belongs to equality (a difference of exactly
``EPSILON`` still compares equal). Measured directly on TradingView across four
decades of operand magnitude; it is the same constant the ``ta.rci`` rank-tie
clustering uses, i.e. one shared language rule.

Two consumers share this definition:

* ``transformers.float_tolerance`` rewrites the comparison operators of user
  and compiled scripts into an inline arithmetic form over the difference.
* the builtins in ``lib`` that were *measured* to compare tolerantly
  (``ta.rising``, ``ta.falling``, ``ta.cmo``, ``ta.mfi``, ``array.percentrank``,
  ``array.indexof`` and friends). The rest of the builtins are bit-exact on
  TradingView — ``ta.highest``/``lowest``, ``ta.crossover``, ``ta.pivothigh``,
  ``math.max``/``min``/``sign``, ``array.binary_search`` among them — so the
  tolerance must never be applied blindly to a builtin that has not been
  measured.

Hot per-bar paths inline the arithmetic instead of calling ``equal``: the call
would cost more than the comparison it wraps.
"""
__all__ = ['EPSILON', 'equal', 'lower_bound', 'upper_bound']

EPSILON = 1e-10


def equal(a, b) -> bool:
    """
    Pine's tolerant equality, for cold paths (array searches).

    The tolerance is only reachable for real floats: Pine also allows ``==`` on
    strings, colors and object references, which cannot be subtracted, and for
    ints an exact comparison is already the tolerant one (an int difference is
    either 0 or at least 1).

    The raw comparison comes first, exactly as in the operator rewrite: two
    equal infinities have a nan difference, which the tolerance band would
    reject even though they are equal under IEEE-754.

    :param a: Left operand
    :param b: Right operand
    :return: True if the operands are equal under Pine's comparison rule
    """
    if a == b:
        return True
    if a.__class__ is float or b.__class__ is float:
        difference = a - b
        return -EPSILON <= difference <= EPSILON
    return False


def lower_bound(values: list[float], x: float) -> int:
    """
    First slot of an ordered list that is not below ``x`` under Pine's comparison.

    :param values: Tolerantly ordered values
    :param x: The value to locate
    :return: The insertion point in front of the values ``x`` ties with
    """
    # Neither bound may be expressed as a plain ``bisect`` over a shifted key
    # (``x - EPSILON``): the shift is itself rounded, which moves the decision on
    # operands sitting on the tolerance boundary, and a list ordered tolerantly is
    # not necessarily ordered exactly, so the shifted key would rely on a
    # precondition that does not hold.
    lo = 0
    hi = len(values)
    while lo < hi:
        mid = (lo + hi) // 2
        if values[mid] - x < -EPSILON:
            lo = mid + 1
        else:
            hi = mid
    return lo


def upper_bound(values: list[float], x: float) -> int:
    """
    First slot of an ordered list that is above ``x`` under Pine's comparison.

    :param values: Tolerantly ordered values
    :param x: The value to locate
    :return: The insertion point behind the values ``x`` ties with
    """
    lo = 0
    hi = len(values)
    while lo < hi:
        mid = (lo + hi) // 2
        if x - values[mid] < -EPSILON:
            hi = mid
        else:
            lo = mid + 1
    return lo
