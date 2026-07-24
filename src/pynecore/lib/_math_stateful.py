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

from pynecore.types import NA, Persistent, PyneFloat, PyneInt, Series, na_float
from pynecore.core.random import PineRandom as _PineRandom
# lib import (normalized to ``from pynecore import lib``) so the statement-position
# ``max_bars_back`` call below is anchored and converted to a buffer resize.
from pynecore.lib import max_bars_back

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
        prng = _PineRandom(seed)
    res = prng.random(min, max)
    return res


# Three groups of IDE findings here are artifacts of the ``@pyne`` transform, not real
# defects: ``Persistent`` assignments look dead because their value is read on the NEXT
# bar, ``src`` looks possibly-unbound because it is a series whose storage outlives the
# ``if`` that feeds it, and ``src[i]`` looks like subscripting a float because
# ``Series[T]`` erases to ``T`` for the IDE.
# noinspection PyShadowingBuiltins,PyUnusedLocal,PyUnboundLocalVariable,PyUnresolvedReferences
def sum(source: TFI | NA[TFI], length: int) -> PyneFloat | TFI | NA[TFI]:
    """
    Returns the sum of a series over a specified length using Kahan summation.

    :param source: Source series
    :param length: Length of the sum
    :return: The sliding sum of the series
    """
    summ: Persistent[float] = 0.0
    count: Persistent[int] = 0
    compensation: Persistent[float] = 0.0
    prev_length: Persistent[int] = 0
    removals: Persistent[int] = 0

    # Representation-agnostic na test: an na source is either an NA object or a
    # native nan (OHLCV gaps can already deliver a bare nan). Both must be
    # excluded from the na-compacted buffer, or ``src[k]`` would poison ``summ``.
    source_na = isinstance(source, NA) or source != source

    if not source_na:
        # Record every non-na bar's value into the sliding buffer BEFORE any
        # early return (shortcut / warmup), so the positional recompute below
        # sees a complete history with no holes. NA values are intentionally
        # not stored: the buffer stays na-compacted, so ``src[k]`` is the k-th
        # most recent non-na value — exactly the "last N non-na" window Pine's
        # sum/sma use.
        src: Series[float] = source
        # The sliding window drops the value leaving it via ``src[length]`` (``length``
        # non-na bars back). Grow the na-compacted buffer so that index stays addressable
        # for lengths beyond the per-series default ``max_bars_back`` (500); otherwise the
        # removal reads na and poisons ``summ`` permanently, collapsing any ``ta.sma`` /
        # ``ta.sum`` with length > 500 to na right after warmup. Capacity persists across
        # bars, so setting it on each non-na bar (from the first on) suffices — the window
        # never removes before ``length`` non-na values have accumulated.
        max_bars_back(src, int(length))

    if length == 1:  # Shortcut
        # The sliding accumulator is left untouched here; record length == 1 so a
        # following bar with a different length recomputes instead of trusting the
        # now-stale state.
        prev_length = 1
        return source
    assert length > 0, "Invalid length, length must be greater than 0!"
    length = int(length)

    # The Kahan sliding window below is only valid while ``length`` stays
    # constant. Pine allows a series ``length`` (e.g. ``ta.sma(src, barssince(...))``);
    # when it changes bar-to-bar the accumulator no longer describes the requested
    # trailing window, so recompute the sum directly from the source series over the
    # current length (mirroring ``ta.highest``'s positional fallback) and re-seed the
    # accumulator so a subsequently stable length resumes the O(1) fast path.
    changed = prev_length != 0 and length != prev_length
    prev_length = length
    if changed:
        removals = 0
        if source_na:
            # Length changed on an na bar: the old window is stale and cannot be
            # rebuilt from a missing current value; restart the warmup cleanly.
            summ = 0.0
            count = 0
            compensation = 0.0
        else:
            recomputed = 0.0
            comp = 0.0
            found = 0
            for i in range(length):
                v = src[i]
                if isinstance(v, NA) or v != v:
                    # The buffer is na-compacted, so the first na marks the end of
                    # available history — every deeper index is na too. Stop here
                    # instead of scanning the rest (keeps the recompute O(available),
                    # never O(length) when length outruns the stored history).
                    break
                found += 1
                corrected = float(v) - comp
                new_recomputed = recomputed + corrected
                comp = (new_recomputed - recomputed) - corrected
                recomputed = new_recomputed
            if found < length:  # Not enough non-na history for this window yet
                summ = 0.0
                count = 0
                compensation = 0.0
                return na_float
            summ = recomputed
            compensation = comp
            count = length
            return recomputed

    if count < length - 1:
        if not source_na:
            count += 1
            # Kahan summation for adding new value
            corrected_value = float(source) - compensation
            new_sum = summ + corrected_value
            compensation = (new_sum - summ) - corrected_value
            summ = new_sum
        return na_float
    elif count == length - 1:
        if source_na:
            return na_float
        count += 1
    else:
        if source_na:
            return summ
        # Exact resync: the incremental remove+add path below carries residual
        # rounding error from bars long outside the window (Kahan bounds it but
        # never clears it). That residue breaks identities TV preserves — e.g.
        # a window of equal values must sum to exactly n*v (a run of zeros must
        # sum to 0.0, not -3e-15), or ``sma(sma(x))`` of a flat series flips
        # strict comparisons like the Technical Ratings
        # ``kStochRsi < dStochRsi`` on last-bit noise. Small windows (the
        # precision-sensitive ``sma(x, 3)`` chains) recompute from the
        # na-compacted buffer on EVERY bar — for those lengths the fresh Kahan
        # pass costs the same as the remove+add pair it replaces, and the
        # result depends on the window alone. Longer windows resync once per
        # ``length`` removals, capping any drift's lifetime at one window
        # turnover at an amortized O(1) cost per bar.
        removals += 1
        if removals >= length or length <= 8:
            removals = 0
            recomputed = 0.0
            comp = 0.0
            found = 0
            for i in range(length):
                v = src[i]
                if isinstance(v, NA) or v != v:
                    break
                found += 1
                corrected = float(v) - comp
                new_recomputed = recomputed + corrected
                comp = (new_recomputed - recomputed) - corrected
                recomputed = new_recomputed
            if found == length:
                summ = recomputed
                compensation = comp
                return summ
        # Kahan summation for removing old value (float() compiles to
        # safe_convert.safe_float, returning NA instead of raising on NA)
        old_value = float(src[length])
        corrected_old = -old_value - compensation
        new_sum = summ + corrected_old
        compensation = (new_sum - summ) - corrected_old
        summ = new_sum

    # Kahan summation for adding new value
    corrected_value = float(source) - compensation
    new_sum = summ + corrected_value
    compensation = (new_sum - summ) - corrected_value
    summ = new_sum

    return summ
