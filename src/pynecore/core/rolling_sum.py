"""
Bit-exact fire detector for the Pine rolling-sum state machine.

Pine's ``math.sum`` (and through it ``ta.sma``) maintains a rolling
compensated sum: each bar evicts the entry stored ``length`` bars ago and
adds the new value in one fused two-round step. On specific bars the
engine abandons the accumulated state and re-baselines: it recomputes the
window as a plain newest-first linear sum, clears the compensation
register and stores the raw incoming value instead of the compensated one.

``sum_fires`` reproduces the re-baseline condition exactly. With ``c`` the
compensation register carried into the bar and ``x`` the incoming value,
the engine shifts the value upward by the compensation's magnitude and
fires exactly when the fixed-order Fast2Sum residue of that addition is
positive: ``e = fl(|c| - fl(fl(x + |c|) - x))``, fire ⟺ ``e > 0``.
``c == 0`` or ``x == 0`` never fire. Two details are load-bearing. The
magnitude ``|c|`` — not the signed ``c`` — makes the test exact at binade
edges: a signed probe walks the finer downward grid below a power of two
and mislabels those bars. And the Fast2Sum runs in this fixed operand
order WITHOUT the usual magnitude swap: when ``|c| > |x|`` the residue is
no longer the exact rounding error, and that inexact value is what the
engine tests (an exact 2Sum fires on bars the engine does not).
"""
__all__ = ['sum_fires']


def sum_fires(compensation: float, value: float) -> bool:
    """
    Decide whether the rolling-sum re-baseline fires on this bar.

    For ``|x| >= |c|`` the residue ``e = |c| - (fl(x + |c|) - x)`` is by
    Fast2Sum exactly ``(x + |c|) - fl(x + |c|)``, i.e. the negated rounding
    error, so ``e > 0`` means "the add rounded down". For ``|c| > |x|`` the
    same expression is evaluated anyway — deliberately: the engine performs
    no magnitude swap, and the then-inexact residue is the tested quantity.
    ``r - r == 0.0`` rejects both nan and infinite ``r``, covering
    non-finite operands as well.

    ``lib._math_stateful.sum`` inlines this same expression in its per-bar
    path to avoid the call; the two must stay in sync.

    :param compensation: The compensation register carried into this bar
    :param value: The incoming source value of this bar
    :return: True if the accumulator must be re-baselined from a plain
             newest-first linear sum over the raw window
    """
    x = value
    if compensation == 0.0 or x == 0.0:
        return False
    b = compensation if compensation > 0.0 else -compensation
    r = x + b
    if r == 0.0 or r - r != 0.0:
        return False
    return b - (r - x) > 0.0
