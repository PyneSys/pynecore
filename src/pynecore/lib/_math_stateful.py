"""
@pyne lib

Stateful implementations of ``lib.math.random`` and ``lib.math.sum``. They
live in their own small module because the ``@pyne`` marker is module-level
and the host module (``lib/math.py``) must stay untransformed; the host
re-exports the functions, and the layouts travel on the function objects.
"""
# Absolute imports on purpose: the call-site classifier resolves absolute
# imports at transform time, so NA() calls stay direct instead of anchored
import builtins
from typing import TypeVar

from pynecore.types import NA, Persistent, PyneFloat, PyneInt, Series, na_float
from pynecore.core.random import PineRandom as _PineRandom
from pynecore.core.series import SeriesImpl as _SeriesImpl
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
    Returns the sum of a series over a specified length, bit-exact with Pine.

    The window is na-compacted: an na bar returns na and is not stored, so the sum
    always covers the last ``length`` non-na values.

    :param source: Source series
    :param length: Length of the sum
    :return: The sliding sum of the series
    """
    # Pine's engine keeps a rolling compensated sum: each bar evicts the entry stored
    # ``length`` bars ago and adds the new value in one fused two-round step
    # (``y1 = fl(-d0 - c)``; ``t = fl(s + y1)``; ``e1 = fl(fl(t - s) - y1)``;
    # ``y2 = fl(x - e1)``; ``s = fl(t + y2)``; ``c = fl(fl(s - t) - y2)``), storing the
    # realized ``y2`` for the future eviction. On bars where ``sum_fires`` signals it,
    # the engine re-baselines instead: the display and accumulator become the plain
    # newest-first linear sum of the raw window, the compensation clears, and the raw
    # value is stored. The same machine runs during warmup with ``d0 = 0`` and the
    # re-baseline summing the whole available prefix. Validated bit-for-bit against TV
    # output on dense probes for lengths 2..14 (100.00% of ~330k displayed bars).
    summ: Persistent[float] = 0.0
    count: Persistent[int] = 0
    compensation: Persistent[float] = 0.0
    prev_length: Persistent[int] = 0
    entries: Persistent[list | None] = None
    head: Persistent[int] = 0
    capacity: Persistent[int] = _SeriesImpl.DEFAULT_MAX_BARS_BACK

    # Representation-agnostic na test: an na source is either an NA object or a
    # native nan (OHLCV gaps can already deliver a bare nan). Both must be
    # excluded from the na-compacted buffer, or ``src[k]`` would poison ``summ``.
    source_na = isinstance(source, NA) or source != source

    # One conversion up front so every later use is a plain int compare. Bare
    # ``int()``/``float()`` become na-guarded wrapper calls under the transform,
    # so the already-int fast path skips the call and the ``builtins.*`` forms
    # below are used where the na cases are provably handled already.
    if builtins.type(length) is not int:
        length = int(length)

    if not source_na:
        # Record every non-na bar's value into the sliding buffer BEFORE any
        # early return (shortcut / warmup), so the positional reads below see
        # a complete history with no holes. NA values are intentionally not
        # stored: the buffer stays na-compacted, so ``src[k]`` is the k-th
        # most recent non-na value — exactly the "last N non-na" window Pine's
        # sum/sma use.
        src: Series[float] = source
        # The re-baseline reads the raw window via ``src[length - 1]``. Grow the
        # na-compacted buffer so that index stays addressable for lengths beyond
        # the per-series default ``max_bars_back``; otherwise the rebuild reads
        # na and poisons ``summ``, collapsing any ``ta.sma`` / ``ta.sum`` with a
        # length above the default to na right after warmup. The resize is
        # monotonic and floored at the series' own default: a series ``length``
        # that dips low must not shrink the buffer, or the history a later
        # increase needs would already have been thrown away.
        if length > capacity:
            capacity = length
            max_bars_back(src, capacity)

    if length == 1:  # Shortcut
        # The sliding accumulator is left untouched here; record length == 1 so a
        # following bar with a different length recomputes instead of trusting the
        # now-stale state.
        prev_length = 1
        return source
    assert length > 0, "Invalid length, length must be greater than 0!"

    # The rolling machine below is only valid while ``length`` stays constant.
    # Pine allows a series ``length`` (e.g. ``ta.sma(src, barssince(...))``);
    # when it changes bar-to-bar the accumulator no longer describes the
    # requested trailing window, so re-baseline from the raw buffer (the same
    # newest-first linear sum a fire produces) and re-seed the realized queue
    # with the raw window so a subsequently stable length resumes the O(1) path.
    # Reading the slot once and only writing it on an actual change keeps the
    # steady-state path (a constant length) down to a single load.
    prev = prev_length
    if prev != length:
        prev_length = length
    if prev != 0 and prev != length:
        # ``src`` is na-compacted, so ``src[0 .. length - 1]`` is the requested
        # trailing window whether or not the current bar is na: on an na bar the
        # buffer simply was not advanced, so ``src[0]`` still is the newest
        # non-na value — the very window the stable-length na branch preserves.
        newest = src[0]
        if isinstance(newest, NA) or newest != newest:  # No non-na history at all yet
            summ = 0.0
            count = 0
            compensation = 0.0
            entries = None
            return na_float
        rebuilt = builtins.float(newest)
        found = 1
        for i in builtins.range(1, length):
            v = src[i]
            if isinstance(v, NA) or v != v:
                # The buffer is na-compacted, so the first na marks the end
                # of available history — every deeper index is na too.
                break
            found += 1
            rebuilt = builtins.float(v) + rebuilt
        # A short history is kept as a warmup prefix instead of being dropped:
        # the following bars then only need ``length - found`` more values, so a
        # grown length no longer blanks the output for a whole fresh window.
        # Oldest first, so ``head`` addresses the next entry to be evicted.
        entries = [0.0] * length
        for i in builtins.range(found):
            entries[i] = builtins.float(src[found - 1 - i])
        head = found
        if head == length:
            head = 0
        summ = rebuilt
        compensation = 0.0
        count = found
        return rebuilt if found == length else na_float

    # Bind the per-bar state into locals: each ``Persistent`` read/write is a
    # slot index under the transform, and the steady-state step touches the
    # buffer, the head, the accumulator and the compensation several times.
    ent = entries
    if ent is None:
        ent = [0.0] * length
        entries = ent

    n = count
    if source_na:
        return na_float if n < length else summ

    value = builtins.float(source)
    c = compensation
    s = summ
    h = head

    # ``core.rolling_sum.sum_fires`` inlined: a call here would cost more than
    # the whole compensated step it guards, and the transform wraps every call
    # in an isolation binding on top. Keep the two in sync — Fast2Sum gives the
    # realized rounding error of ``fl(value + c)`` exactly, so the fire test is
    # ``sign(e) == sign(c)`` (see that module's docstring for the derivation).
    fires = False
    if c != 0.0 and value != 0.0:
        r = value + c
        if r != 0.0 and r - r == 0.0:  # rejects nan and +-inf without a call
            if (value if value > 0.0 else -value) >= (c if c > 0.0 else -c):
                e = c - (r - value)
            else:
                e = value - (r - c)
            if e != 0.0:
                fires = (e > 0.0) if c > 0.0 else (e < 0.0)

    if n < length:  # Warmup: accumulate with d0 = 0, fires sum the prefix
        n += 1
        count = n
        if fires:
            rebuilt = value
            for i in builtins.range(1, n):
                rebuilt = builtins.float(src[i]) + rebuilt
            s = rebuilt
            compensation = 0.0
            ent[h] = value
        else:
            y1 = -c
            t = s + y1
            e1 = (t - s) - y1
            y2 = value - e1
            new_sum = t + y2
            compensation = (new_sum - t) - y2
            s = new_sum
            ent[h] = y2
        summ = s
        h += 1
        head = 0 if h == length else h
        return s if n == length else na_float

    # ``h`` is both the oldest entry and the slot the new one takes
    old_value = ent[h]
    if fires:
        # Re-baseline: newest-first linear sum of the raw window, raw store
        rebuilt = value
        for i in builtins.range(1, length):
            rebuilt = builtins.float(src[i]) + rebuilt
        s = rebuilt
        compensation = 0.0
        ent[h] = value
    else:
        # Fused two-round evict-and-add, realized store
        y1 = -old_value - c
        t = s + y1
        e1 = (t - s) - y1
        y2 = value - e1
        new_sum = t + y2
        compensation = (new_sum - t) - y2
        s = new_sum
        ent[h] = y2
    summ = s
    h += 1
    head = 0 if h == length else h

    return s
