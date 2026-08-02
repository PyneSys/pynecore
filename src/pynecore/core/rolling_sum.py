"""
Bit-exact fire detector for the Pine rolling-sum state machine.

Pine's ``math.sum`` (and through it ``ta.sma``) maintains a rolling
compensated sum: each bar evicts the entry stored ``length`` bars ago and
adds the new value in one fused two-round step. On specific bars the
engine abandons the accumulated state and re-baselines: it recomputes the
window as a plain newest-first linear sum, clears the compensation
register and stores the raw incoming value instead of the compensated one.

``sum_fires`` reproduces the observed re-baseline condition. With ``c``
the compensation register carried into the bar and ``x`` the incoming
value, the engine fires exactly when the realized rounding error of
``fl(x + c)`` goes against the compensation: with ``e = fl(x + c) -
(x + c)`` (exact, ties already resolved by round-half-even inside ``fl``),
fire ⟺ ``sign(e) == -sign(c)``. Exact adds (``e == 0``) and ``c == 0`` or
``x == 0`` never fire.

Validated on forced-label corpora extracted from TV output: dense chaotic
probes at lengths 2..14 (~330k bars, 100.00%) and a real ta.rci feed
(27751 labels, 99.77% — the residual lives in a still-open degenerate
exact/tie sub-regime).
"""
__all__ = ['sum_fires']


def sum_fires(compensation: float, value: float) -> bool:
    """
    Decide whether the rolling-sum re-baseline fires on this bar.

    The realized rounding error is obtained exactly in plain float
    arithmetic instead of on an integer mantissa grid: by Fast2Sum, for
    ``|x| >= |c|`` the value ``e = c - (fl(x + c) - x)`` is exactly
    ``(x + c) - fl(x + c)``, i.e. the negated rounding error, so the
    condition ``sign(err) == -sign(c)`` becomes ``sign(e) == sign(c)``.
    The operands are swapped when ``|c| > |x|``, which is Fast2Sum's
    precondition. ``r - r == 0.0`` rejects both nan and infinite ``r``,
    covering non-finite operands as well.

    ``lib._math_stateful.sum`` inlines this same expression in its per-bar
    path to avoid the call; the two must stay in sync.

    :param compensation: The compensation register carried into this bar
    :param value: The incoming source value of this bar
    :return: True if the accumulator must be re-baselined from a plain
             newest-first linear sum over the raw window
    """
    c = compensation
    x = value
    if c == 0.0 or x == 0.0:
        return False
    r = x + c
    if r == 0.0 or r - r != 0.0:
        return False
    e = (c - (r - x)) if (x if x > 0.0 else -x) >= (c if c > 0.0 else -c) else (x - (r - c))
    if e == 0.0:
        return False
    return (e > 0.0) if c > 0.0 else (e < 0.0)
