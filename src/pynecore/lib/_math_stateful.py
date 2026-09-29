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
from functools import reduce as _reduce
from operator import add as _add
from typing import TypeVar

from pynecore.types import (NA, IBPersistent, Persistent, PyneFloat, PyneInt,
                            Series, na_float)
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


# The IDE findings here are artifacts of the ``@pyne`` transform, not real defects:
# ``Persistent`` assignments look dead because their value is read on the NEXT bar,
# ``src`` looks possibly-unbound because it is a series whose storage outlives the
# ``if`` that feeds it, ``src[i]`` looks like subscripting a float because
# ``Series[T]`` erases to ``T`` for the IDE, and assigning ``source`` to the series
# looks type-unsafe because the ``source_na`` guard above it is a value test the
# IDE cannot narrow by.
# noinspection PyShadowingBuiltins,PyUnusedLocal,PyUnboundLocalVariable,PyUnresolvedReferences,PyTypeChecker
def sum(source: TFI | NA[TFI], length: int) -> PyneFloat:
    """
    Returns the sum of a series over a specified length.

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
    # value is stored. While the window fills, a bar evicts nothing and is a plain
    # one-round compensated add instead (``y = fl(x - c)``; ``s = fl(s + y)``;
    # ``c = fl(fl(s' - s) - y)``), storing ``y``; the re-baseline then sums the whole
    # available prefix. Validated bit-for-bit against TV
    # output on dense probes for lengths 2..14 (~330k displayed bars), on zero-gap
    # block probes m562 (5599 independent blocks, lengths 3/4/5/8, every branch
    # decision forced), and on real 22k-bar rsi/stoch/sma chains (probes m561/m562).
    summ: Persistent[float] = 0.0
    compensation: Persistent[float] = 0.0
    # The entry ring grows with the stored history (up to ``capacity``), so it is
    # ``varip``: a var rollback would copy the whole ring back on every same-bar
    # re-execution, and a shared loop call site re-executes once per iteration.
    # A bar writes at most ONE ring slot (or swaps in a grown ring), so
    # ``ring_undo`` records exactly that write, keyed by the bar-start ``seen``:
    # a call that starts from the same ``seen`` is a re-execution of that bar
    # (loop iteration, realtime tick, security re-tick) and undoes it first.
    entries: IBPersistent[list[float]] = []
    ring_undo: IBPersistent[tuple | None] = None
    ring: Persistent[int] = 0
    slot: Persistent[int] = 0
    seen: Persistent[int] = 0
    window: Persistent[int] = 0
    capacity: Persistent[int] = _SeriesImpl.DEFAULT_MAX_BARS_BACK
    # Memo of the eviction walk below, deliberately ``varip`` so the loop-site
    # rollback leaves it alone: it is not machine state, it is a replay of the
    # walk the rolled-back state produces (issue #80). See there for the key.
    # One ``None``-initialized slot holding the whole ``(key, s, c)`` triple:
    # separate list variables would need a lazy-init flag each, and those are
    # checked on EVERY call, including the overwhelming majority that never
    # reaches the walk.
    evict_memo: IBPersistent[tuple | None] = None

    # Representation-agnostic na test: an na source is either an NA object or a
    # native nan (OHLCV gaps can already deliver a bare nan). Both must be
    # excluded from the na-compacted buffer, or ``src[k]`` would poison ``summ``.
    source_na = not (source == source)

    # One conversion up front so every later use is a plain int compare. Bare
    # ``int()``/``float()`` become na-guarded wrapper calls under the transform,
    # so the already-int fast path skips the call and the ``builtins.*`` forms
    # below are used where the na cases are provably handled already.
    if builtins.type(length) is not int:
        length = int(length)

    assert length > 0, "Invalid length, length must be greater than 0!"

    n = seen
    if not source_na:
        # Record every non-na bar's value into the sliding buffer BEFORE any
        # positional read, so ``src[k]`` sees a complete history with no holes.
        # NA values are intentionally not stored: the buffer stays na-compacted,
        # so ``src[k]`` is the k-th most recent non-na value — exactly the "last
        # N non-na" window Pine's sum/sma use. An na bar leaves the buffer where
        # it is, so ``src[k]`` keeps addressing the same stored values.
        src: Series[float] = source
        n += 1
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

    prev_w = window
    new_w = length if n >= length else n
    if source_na and prev_w == new_w:
        # Nothing entered or left: an na bar that does not move the window is a
        # no-op and reports the standing sum (MEASURED, probe sumlen4: a length-1
        # sum on an na bar echoes the last NON-NA value instead of returning na).
        return summ if new_w >= length else na_float

    # The realized-entry ring is addressed by position RELATIVE to the newest
    # entry. Pine's machine does not restart when the length moves (MEASURED, probe
    # sumlen2: a length grown 1..610 reproduces a constant 610 bit-for-bit on all
    # 28746 bars), so the history already stored has to keep its identity — a ring
    # re-based on the new length would lose it. Every stored bar keeps the entry it
    # was stored with for as long as ``src`` can still address it: a grown window
    # re-admits bars that left long ago, and when they leave again it is that
    # ORIGINAL entry that is evicted (see the admission walk below). The ring grows
    # with the stored history up to ``capacity`` like the series buffer itself,
    # doubling so the copy below stays rare.
    undo = ring_undo
    if undo is not None and undo[0] == seen:
        start = undo[1]
        if undo[2] >= 0:
            start[undo[2]] = undo[3]
        entries = start
    else:
        start = entries
    ent = start
    cap = ring
    at = slot
    need = n if n < capacity else capacity
    if need > cap:
        size = cap + cap
        if size < need:
            size = need
        elif size > capacity:
            size = capacity
        grown = [0.0] * size
        j = 0
        if cap:
            kept = cap if seen > cap else seen
            i = at - kept
            if i < 0:
                i += cap
            j = size - kept
            for _ in builtins.range(kept):
                grown[j] = ent[i]
                i += 1
                if i == cap:
                    i = 0
                j += 1
                if j == size:
                    j = 0
        ent = grown
        cap = size
        at = j
        entries = ent
        ring = cap
        slot = at

    c = compensation
    s = summ

    # The window is the last ``length`` non-na values, so a moved length both
    # DROPS and ADMITS entries around it, and TradingView walks that change as a
    # SEQUENCE of ordinary machine steps rather than one fused one (MEASURED,
    # probes sumlen6/sumlen7: a 5->6, 6->7, 4->10 or 6->5 step is bit-exact this
    # way and 1-3 ulp off when the whole change is folded into a single ``d0``).
    # ``shift`` is 1 on a stored bar (every older offset moves up by one) and 0
    # on an na bar, so relative to this bar's offset 0 the previous window
    # covered ``shift``..``prev_w - 1 + shift``. Offsets ``new_w``..
    # ``prev_w - 1 + shift`` LEAVE oldest first, each an eviction-only step, and
    # the newest of them is the one fused with this bar's own value — in the
    # steady state it is the only one, which is exactly the proven single-evict
    # step. A grown window ADMITS offsets ``prev_w + shift``..``new_w`` instead —
    # one PAST the new window — and the fused step then evicts offset ``new_w``
    # again, so every bar whose offset ``new_w`` exists ends on the same fused
    # step, and only a window reaching back to the first stored bar ends on a
    # plain compensated add.
    # Every change measured this way is bit-exact over the full following tail,
    # isolated or on consecutive bars (probes sumlen3/6/7: 8->1, 100->1,
    # 300->150->50, 610->100, 5->3, 6->5; probes wg/wh/wk: 131 grows; probe wr:
    # sawtooth, ramp, alternating and pseudo-random lengths over 21.7k bars on
    # six sources).
    if source_na:
        shift = 0
        base = at - 1
        if base < 0:
            base += cap
        value = 0.0
    else:
        shift = 1
        base = at
        value = builtins.float(source)

    d0 = 0.0
    if prev_w + shift > new_w:
        # Oldest first with the RAW source values, each one compensated add of the
        # negated value (MEASURED, probes sumlen3/6: the isolated 8->1, 100->1,
        # 300->150, 610->100 and 5->3 events; probe wr: shrinks on consecutive
        # bars, where the two-round form drifts 1-16 ulp), the newest leaving
        # entry — the realized one, exactly the steady-state eviction — left for
        # the fused step below.
        # The offsets that leave are ``new_w + 1``..``top``, and an offset is
        # its own ``src`` index in BOTH regimes: on a stored bar the buffer
        # moved with the window, on an na bar neither moved. Reading one lower
        # on an na bar would evict offset ``new_w`` twice (the fused step below
        # already takes it) and leave ``top`` in the sum forever.
        # A shared loop call site re-derives this walk from the SAME bar-start
        # state on every iteration, so a length that moves per iteration
        # replays one deep prefix of it over and over (issue #80: 20 lengths
        # 20..400 each walked 400 -> length on every bar). The walk is a pure
        # function of the state it starts from, so the state after every step
        # is memoized and an iteration that needs fewer steps reads its answer
        # straight out; the leaving offsets are contiguous, so the raw values
        # of the steps that DO run come over as one native list in exactly the
        # walk order (``oldest`` is oldest first) instead of one
        # ``Series.__getitem__`` call per step.
        # The key pins the starting state exactly: ``n`` fixes the stored
        # history (a bar that stores nothing cannot change an offset above 0,
        # and the walk never reads offset 0), ``prev_w``/``shift`` fix which
        # offsets leave, and the accumulator pair fixes the arithmetic.
        # Anything else — a new bar, a re-run with different data — misses and
        # rebuilds, so a hit can only reproduce what the walk would compute.
        top = prev_w - 1 + shift
        if top > new_w:
            need = top - new_w
            key = (n, prev_w, shift, s, c)
            memo = evict_memo
            if memo is None or memo[0] != key:
                ws = [s]
                wc = [c]
                evict_memo = (key, ws, wc)
            else:
                ws = memo[1]
                wc = memo[2]
            walked = builtins.len(ws) - 1
            if need <= walked:
                s = ws[need]
                c = wc[need]
            else:
                s = ws[walked]
                c = wc[walked]
                # Grown by slice assignment and filled by index: an ``append``
                # here is a method call, and the transform turns every call in
                # a loop body into an isolated call site — the binding then
                # costs more than the whole step it records.
                i = walked + 1
                tail = [0.0] * (need - walked)
                ws[i:] = tail
                wc[i:] = tail
                for v in builtins.map(builtins.float,
                                      src[new_w + 1:top - walked + 1].oldest):
                    y = -v - c
                    new_sum = s + y
                    c = (new_sum - s) - y
                    s = new_sum
                    ws[i] = s
                    wc[i] = c
                    i += 1
    elif new_w > prev_w + shift or new_w < n:
        # Newest first with the RAW source values, each one compensated add, down
        # to offset ``new_w`` itself when it exists (a +1 grow of a full window
        # admits just that one), and the admitted bars keep the entries they were
        # ORIGINALLY stored with: the fused step below evicts offset ``new_w``
        # with its original entry, and so does every later eviction of an
        # admitted bar (MEASURED, probes wg/wh/wk: 131 isolated grows of
        # volume-scaled, ratio and raw volume sources, 1..40 wide, every one
        # bit-exact on every historical bar after it this way; admitting only up to
        # ``new_w - 1``, oldest first, with the residues stored, matched 9 of them).
        # Same native window read as the eviction walk above. Unlike that one
        # this walk cannot be memoized across the iterations of a shared loop
        # call site: two lengths share the head of the fold but not its input
        # range, and every length has to be folded from the bar-start state.
        top = new_w + 1 if new_w < n else n
        for admitted in builtins.reversed(
                builtins.list(builtins.map(builtins.float, src[prev_w + shift:top].oldest))):
            y = admitted - c
            new_sum = s + y
            c = (new_sum - s) - y
            s = new_sum
    warm = new_w == n
    if not warm:
        e = base - new_w
        if e < 0:
            e += cap
        d0 = ent[e]

    # ``core.rolling_sum.sum_fires`` inlined: a call here would cost more than
    # the whole compensated step it guards, and the transform wraps every call
    # in an isolation binding on top. Keep the two in sync — the fixed-order
    # Fast2Sum residue of ``fl(value + |c|)`` is tested WITHOUT a magnitude
    # swap, and the machine fires when it is positive (see that module's
    # docstring for the derivation, the binade-edge reason the magnitude and
    # not the signed ``c`` is shifted, and why the ``|c| > |x|`` residue stays
    # deliberately inexact).
    if not source_na:
        # Every branch below stores this bar's entry at ``at``: into the
        # bar-start ring in place, or into a ring grown above, which leaves the
        # bar-start one untouched and only has to be swapped back
        ring_undo = (seen, start, at, start[at]) if ent is start else (seen, start, -1, 0.0)

    fires = False
    if c != 0.0 and value != 0.0:
        b = c if c > 0.0 else -c
        r = value + b
        if r != 0.0 and r - r == 0.0:  # rejects nan and +-inf without a call
            fires = b - (r - value) > 0.0

    if fires:
        # Re-baseline: newest-first linear sum of the raw window, raw store
        # The window travels as ONE native list (oldest first), folded natively
        # and seeded with this bar's own raw value: ``win`` is oldest first, so dropping its
        # last element drops ``src[0]`` and the reversed rest is ``src[1]``..
        # ``src[new_w - 1]``.
        win = builtins.list(builtins.map(builtins.float, src[0:new_w].oldest))
        s = _reduce(_add, builtins.reversed(win[:-1]), value)
        compensation = 0.0
        if not source_na:
            ent[at] = value
    elif warm:
        # Warmup: nothing leaves and nothing else enters, so the bar is a single
        # compensated add. The fused step below with ``d0 = 0`` rounds ``s - c`` on its
        # own first, which differs whenever that rounds away from ``s`` (``|c|`` of half
        # an ulp of ``s`` or more): the stored entry then comes out an ulp off and every
        # later eviction of it carries the error (MEASURED, probe ws: 42 warmups of
        # volume-scaled sources, 841k displayed bars bit-exact this way, none with the
        # fused form). A grown window that reaches back to the first stored bar has
        # no offset ``new_w`` to evict either and ends on the same add.
        y = value - c
        new_sum = s + y
        compensation = (new_sum - s) - y
        s = new_sum
        if not source_na:
            ent[at] = y
    else:
        # Fused two-round evict-and-add, realized store
        y1 = -d0 - c
        t = s + y1
        e1 = (t - s) - y1
        y2 = value - e1
        new_sum = t + y2
        compensation = (new_sum - t) - y2
        s = new_sum
        if not source_na:
            ent[at] = y2
    summ = s
    seen = n
    window = new_w
    if not source_na:
        at += 1
        slot = 0 if at == cap else at

    return s if new_w >= length else na_float
