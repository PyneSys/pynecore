"""
Pine's rolling-sum state machine: ``math.sum`` and everything built on it.

Pine's ``math.sum`` (and through it ``ta.sma``) maintains a rolling
compensated sum: each bar evicts the entry stored ``length`` bars ago and
adds the new value in one fused two-round step. On specific bars the
engine abandons the accumulated state and re-baselines: it recomputes the
window as a plain newest-first linear sum, clears the compensation
register and stores the raw incoming value instead of the compensated one.

:class:`SumMachine` is the whole state of one ``math.sum`` call site, and
``lib._math_stateful.sum`` only creates it and steps it. It ships twice: this
module is the reference implementation and the fallback wherever the compiled
extension is not built, and ``_native_rolling_sum`` is its compiled twin,
operation for operation, which replaces :class:`SumMachine` and
:data:`rolling_sum_step` here at import unless ``PYNE_NO_NATIVE_MATH`` is set.

The machine is one object instead of a set of ``Persistent`` slots so that
rolling it back is O(1): a shared loop call site restores its builtin machines
to their bar-start state before every iteration (see
``instance_state.__loop_state__``), and a generic rollback of the slots and of
a series window costs more than the step itself. The machine takes part in
every rollback through the ``__pyne_snapshot__`` / ``__pyne_restore__`` pair:
the snapshot is a token of the current state, and a restore undoes the steps
taken since, newest first, from the journal the machine keeps for it. A rollback
that re-runs the bar of its snapshot undoes one step; a live
``request.security_lower_tf`` baseline is restored across several intrabars, and
undoes one step per intrabar run since (see :meth:`SumMachine.__pyne_restore__`).

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
import os as _os
from functools import reduce as _reduce
from operator import add as _add
from typing import Any

from .safe_convert import native_int
from .series import SeriesImpl
from ..types.na import na_float

__all__ = ['sum_fires', 'SumMachine', 'rolling_sum_step']

_STALE_TOKEN_ERROR = "math.sum rollback target is not a state its machine can return to"


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

    :meth:`SumMachine.step` inlines this same expression in its per-bar path
    to avoid the call; the two must stay in sync.

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


def _bar_source() -> Any:
    """The module whose ``bar_index`` keys the window writes, loaded the way
    :class:`SeriesImpl` loads it (the import is shared with it on purpose)."""
    lib = SeriesImpl._lib  # noqa: cooperating core internals
    if not lib:
        from .. import lib  # noqa: circular at module import time only
        SeriesImpl._lib = lib
    return lib


class _Journal:
    """
    What the steps taken since one snapshot overwrote in place, oldest first.

    A record is ``(ring_gen, ring_at, ring_old, values_gen, win_at, win_old)``: the
    ring slot and the window slot one step overwrote, each with its old value and
    the generation of the list it was written to (an index of -1 marks "nothing
    overwritten"). ``following`` is the journal of the next snapshot taken after
    this one, so a token reaches every step taken since it, and only a token keeps
    the journals behind the machine's current one alive.
    """

    __slots__ = ('records', 'following')

    def __init__(self) -> None:
        self.records: list[tuple] = []
        self.following: _Journal | None = None


class SumMachine:
    """
    The rolling-sum machine of one ``math.sum`` call site.

    The window is na-compacted: an na bar returns na and is not stored, so the sum
    always covers the last ``length`` non-na values.

    State, in two groups. The machine state proper is the accumulator pair
    (``summ``, ``compensation``), the stored-bar count ``seen``, the window width
    ``window`` of the previous call, the history ``capacity``, the realized-entry
    ring (``ring``, ``ring_cap``, ``slot``) and the raw value window (``values``,
    a circular buffer that behaves exactly like the compacted ``SeriesImpl`` the
    machine used to read: ``values_cap`` is its ``max_bars_back + 1``,
    ``values_size`` / ``values_pos`` its size and write position, ``values_bar``
    the ``bar_index`` of its last write). ``memo`` is not state: it is a replay of
    the eviction walk (see :meth:`step`), and a rollback only drops it when it
    discards stored bars the walk was computed over.

    The rollback bookkeeping is ``journal`` (the :class:`_Journal` of the latest
    snapshot, None until the first one: a machine nobody snapshots records
    nothing), ``ring_gen`` / ``values_gen`` (the generation of the current ring
    and window list, fresh whenever a step replaces one instead of writing into
    it) and ``counter`` (the last generation handed out).
    """

    __slots__ = ('summ', 'compensation', 'seen', 'window', 'capacity',
                 'ring', 'ring_cap', 'slot',
                 'values', 'values_cap', 'values_size', 'values_pos', 'values_bar',
                 'memo', 'journal', 'ring_gen', 'values_gen', 'counter')

    def __init__(self) -> None:
        _bar_source()
        self.summ = 0.0
        self.compensation = 0.0
        self.seen = 0
        self.window = 0
        self.capacity = SeriesImpl.DEFAULT_MAX_BARS_BACK
        self.ring: list[float] = []
        self.ring_cap = 0
        self.slot = 0
        self.values: list[float] = []
        self.values_cap = SeriesImpl.DEFAULT_MAX_BARS_BACK + 1
        self.values_size = 0
        self.values_pos = 0
        self.values_bar: Any = -1
        self.memo: tuple | None = None
        self.journal: _Journal | None = None
        self.ring_gen = 0
        self.values_gen = 0
        self.counter = 0

    def step(self, source: Any, length: Any) -> float:
        """
        One ``math.sum`` call: store ``source`` (unless it is na) and return the
        sum of the last ``length`` non-na values, or na while fewer are stored.

        :param source: The source value of this bar
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

        # Representation-agnostic na test: an na source is either an NA object or a
        # native nan (OHLCV gaps can already deliver a bare nan). Both must be
        # excluded from the na-compacted window, or a window read would poison ``summ``.
        source_na = not (source == source)

        # One conversion up front so every later use is a plain int compare.
        if type(length) is not int:
            length = native_int(length)

        assert length > 0, "Invalid length, length must be greater than 0!"

        n = self.seen
        if not source_na:
            n += 1
        prev_w = self.window
        new_w = length if n >= length else n
        if source_na and prev_w == new_w:
            # Nothing entered or left: an na bar that does not move the window is a
            # no-op and reports the standing sum (MEASURED, probe sumlen4: a length-1
            # sum on an na bar echoes the last NON-NA value instead of returning na).
            return self.summ if new_w >= length else na_float

        # Every call past this point changes the machine, and journals what a rollback
        # has to put back beyond the fields a snapshot holds anyway: the slots it
        # overwrites in the ring and the window list it started with.
        ring_gen = self.ring_gen
        values_gen = self.values_gen
        ring_at = -1
        ring_old = 0.0
        win_at = -1
        win_old = 0.0

        value = 0.0
        if not source_na:
            value = float(source)
            # Record every non-na bar's value into the window BEFORE any positional
            # read, so the window reads see a complete history with no holes. NA
            # values are intentionally not stored: the window stays na-compacted, so
            # offset ``k`` is the k-th most recent non-na value — exactly the "last
            # N non-na" window Pine's sum/sma use. An na bar leaves the window where
            # it is, so offset ``k`` keeps addressing the same stored values.
            # The write is keyed by ``bar_index`` like ``SeriesImpl.add``: a second
            # store on the same bar rewrites the newest value instead of pushing.
            bar = SeriesImpl._lib.bar_index  # noqa: cooperating core internals
            values = self.values
            size = self.values_size
            if self.values_bar == bar:
                if size:
                    pos = self.values_pos - 1
                    if pos < 0:
                        pos += self.values_cap
                    win_at = pos
                    win_old = values[pos]
                    values[pos] = value
            else:
                if size < self.values_cap:
                    values.append(value)
                    self.values_size = size + 1
                    self.values_pos += 1
                else:
                    pos = self.values_pos
                    if pos >= self.values_cap:
                        pos = 0
                        self.values_pos = 1
                    else:
                        self.values_pos = pos + 1
                    win_at = pos
                    win_old = values[pos]
                    values[pos] = value
                self.values_bar = bar
            # The re-baseline reads the raw window down to offset ``length - 1``. Grow
            # the window so that offset stays addressable for lengths beyond the
            # default ``max_bars_back``; otherwise the rebuild reads na and poisons
            # ``summ``, collapsing any ``ta.sma`` / ``ta.sum`` with a length above the
            # default to na right after warmup. The resize is monotonic and floored at
            # the window's own default: a series ``length`` that dips low must not
            # shrink the window, or the history a later increase needs would already
            # have been thrown away.
            if length > self.capacity:
                self.capacity = length
                self._resize_values(length)

        # The realized-entry ring is addressed by position RELATIVE to the newest
        # entry. Pine's machine does not restart when the length moves (MEASURED, probe
        # sumlen2: a length grown 1..610 reproduces a constant 610 bit-for-bit on all
        # 28746 bars), so the history already stored has to keep its identity — a ring
        # re-based on the new length would lose it. Every stored bar keeps the entry it
        # was stored with for as long as the window can still address it: a grown window
        # re-admits bars that left long ago, and when they leave again it is that
        # ORIGINAL entry that is evicted (see the admission walk below). The ring grows
        # with the stored history up to ``capacity`` like the window itself,
        # doubling so the copy below stays rare.
        ent = self.ring
        cap = self.ring_cap
        at = self.slot
        capacity = self.capacity
        need = n if n < capacity else capacity
        grown_ring = False
        if need > cap:
            size = cap + cap
            if size < need:
                size = need
            elif size > capacity:
                size = capacity
            grown = [0.0] * size
            j = 0
            if cap:
                seen = self.seen
                kept = cap if seen > cap else seen
                i = at - kept
                if i < 0:
                    i += cap
                j = size - kept
                for _ in range(kept):
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
            self.ring = ent
            self.ring_cap = cap
            self.slot = at
            self.counter += 1
            self.ring_gen = self.counter
            grown_ring = True

        c = self.compensation
        s = self.summ

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
        else:
            shift = 1
            base = at

        d0 = 0.0
        if prev_w + shift > new_w:
            # Oldest first with the RAW source values, each one compensated add of the
            # negated value (MEASURED, probes sumlen3/6: the isolated 8->1, 100->1,
            # 300->150, 610->100 and 5->3 events; probe wr: shrinks on consecutive
            # bars, where the two-round form drifts 1-16 ulp), the newest leaving
            # entry — the realized one, exactly the steady-state eviction — left for
            # the fused step below.
            # The offsets that leave are ``new_w + 1``..``top``, and an offset is
            # its own window offset in BOTH regimes: on a stored bar the window
            # moved with the machine, on an na bar neither moved. Reading one lower
            # on an na bar would evict offset ``new_w`` twice (the fused step below
            # already takes it) and leave ``top`` in the sum forever.
            # A shared loop call site re-derives this walk from the SAME bar-start
            # state on every iteration, so a length that moves per iteration
            # replays one deep prefix of it over and over (issue #80: 20 lengths
            # 20..400 each walked 400 -> length on every bar). The walk is a pure
            # function of the state it starts from, so the state after every step
            # is memoized and an iteration that needs fewer steps reads its answer
            # straight out; the leaving offsets are contiguous, so the raw values
            # of the steps that DO run come over as one list in exactly the walk
            # order (oldest first).
            # The key pins the starting state exactly: ``n`` fixes the stored
            # history (a bar that stores nothing cannot change an offset above 0,
            # and the walk never reads offset 0), ``prev_w``/``shift`` fix which
            # offsets leave, and the accumulator pair fixes the arithmetic.
            # Anything else — a new bar, a re-run with different data — misses and
            # rebuilds, so a hit can only reproduce what the walk would compute.
            # ``n`` names one history only while the bars below it stay stored: a
            # rollback that discards any of them drops the memo (see
            # :meth:`__pyne_restore__`), because the bars replayed in their place
            # can reach the same key with other values under the walk.
            top = prev_w - 1 + shift
            if top > new_w:
                need = top - new_w
                key = (n, prev_w, shift, s, c)
                memo = self.memo
                if memo is None or memo[0] != key:
                    ws = [s]
                    wc = [c]
                    self.memo = (key, ws, wc)
                else:
                    ws = memo[1]
                    wc = memo[2]
                walked = len(ws) - 1
                if need <= walked:
                    s = ws[need]
                    c = wc[need]
                else:
                    s = ws[walked]
                    c = wc[walked]
                    i = walked + 1
                    tail = [0.0] * (need - walked)
                    ws[i:] = tail
                    wc[i:] = tail
                    for v in self._oldest(new_w + 1, top - walked + 1):
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
            # Unlike the eviction walk above this walk cannot be memoized across the
            # iterations of a shared loop call site: two lengths share the head of the
            # fold but not its input range, and every length has to be folded from the
            # bar-start state.
            top = new_w + 1 if new_w < n else n
            for admitted in reversed(self._oldest(prev_w + shift, top)):
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

        # Every branch below stores this bar's entry at ``at``: into the ring in
        # place, which the journal record keeps the old value of, or into a ring grown
        # above, which leaves the one a snapshot holds untouched
        if not source_na and not grown_ring:
            ring_at = at
            ring_old = ent[at]

        # ``sum_fires`` inlined: a call here would cost more than the whole
        # compensated step it guards. Keep the two in sync — the fixed-order
        # Fast2Sum residue of ``fl(value + |c|)`` is tested WITHOUT a magnitude
        # swap, and the machine fires when it is positive (see the module
        # docstring for the derivation, the binade-edge reason the magnitude and
        # not the signed ``c`` is shifted, and why the ``|c| > |x|`` residue stays
        # deliberately inexact).
        fires = False
        if c != 0.0 and value != 0.0:
            b = c if c > 0.0 else -c
            r = value + b
            if r != 0.0 and r - r == 0.0:  # rejects nan and +-inf without a call
                fires = b - (r - value) > 0.0

        if fires:
            # Re-baseline: newest-first linear sum of the raw window, raw store.
            # The window comes over as one list, oldest first, folded seeded with
            # this bar's own raw value: dropping its last element drops offset 0,
            # and the reversed rest is offsets 1..``new_w - 1``.
            win = self._oldest(0, new_w)
            s = _reduce(_add, reversed(win[:-1]), value)
            self.compensation = 0.0
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
            self.compensation = (new_sum - s) - y
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
            self.compensation = (new_sum - t) - y2
            s = new_sum
            if not source_na:
                ent[at] = y2
        self.summ = s
        self.seen = n
        self.window = new_w
        if not source_na:
            at += 1
            self.slot = 0 if at == cap else at
        journal = self.journal
        if journal is not None:
            journal.records.append((ring_gen, ring_at, ring_old, values_gen, win_at, win_old))

        return s if new_w >= length else na_float

    def _resize_values(self, max_bars_back: int) -> None:
        """Set the window's ``max_bars_back`` exactly like the ``SeriesImpl``
        setter: the history kept is the newest ``max_bars_back + 1`` values.

        :param max_bars_back: The new ``max_bars_back``.
        :raises ValueError: Above :attr:`SeriesImpl.MAXIMUM_MAX_BARS_BACK`.
        """
        if max_bars_back <= 0:
            raise ValueError("The max_bars_back must be a positive integer!")
        if max_bars_back > SeriesImpl.MAXIMUM_MAX_BARS_BACK:
            raise ValueError(f"The max_bars_back cannot exceed {SeriesImpl.MAXIMUM_MAX_BARS_BACK}!")
        new_cap = max_bars_back + 1
        if new_cap == self.values_cap:
            return
        size = self.values_size
        pos = self.values_pos
        if new_cap > size == pos:
            # Not wrapped yet: the buffer is linear and simply keeps growing
            self.values_cap = new_cap
            return
        # A new list, never a rewrite of the old one: a snapshot may still hold it
        old = self.values
        old_cap = self.values_cap
        keep = size if size < new_cap else new_cap
        start = (pos - keep) % old_cap
        self.values = [old[(start + i) % old_cap] for i in range(keep)]
        self.counter += 1
        self.values_gen = self.counter
        self.values_cap = new_cap
        self.values_size = keep
        self.values_pos = keep

    def _oldest(self, start: int, stop: int) -> list[float]:
        """The window offsets ``start``..``stop - 1`` as one list, oldest first —
        the ``series[start:stop].oldest`` read of a ``SeriesImpl``.

        :raises IndexError: When ``stop`` is past the stored history.
        """
        if stop > self.values_size:
            raise IndexError("Slice stop index out of range!")
        if start > stop:
            start = stop
        cap = self.values_cap
        lo = self.values_pos - stop
        hi = self.values_pos - start
        values = self.values
        if lo >= 0:
            return values[lo:hi]
        if hi <= 0:
            return values[lo + cap:hi + cap]
        return values[lo + cap:cap] + values[:hi]

    def __pyne_snapshot__(self) -> tuple:
        """Rollback token of the current state (see :meth:`__pyne_restore__`).

        :return: An opaque token.
        """
        journal = self.journal
        if journal is None or journal.records:
            # The machine moved since the last snapshot (or has none yet): the steps
            # from here on belong to a journal of their own, chained behind the last
            # one so that an older token still reaches them
            fresh = _Journal()
            if journal is not None:
                journal.following = fresh
            self.journal = journal = fresh
        return (journal, self.summ, self.compensation, self.seen, self.window,
                self.capacity, self.ring, self.ring_cap, self.slot, self.values,
                self.values_cap, self.values_size, self.values_pos, self.values_bar,
                self.ring_gen, self.values_gen)

    def __pyne_restore__(self, token: tuple) -> None:
        """Roll the machine back to a :meth:`__pyne_snapshot__` token.

        Most rollbacks of the runtime re-run the bar their snapshot was taken on
        (a shared loop call site's next iteration, a calc_on_order_fills or
        intra-bar tick re-execution, a ``request.security`` re-tick) and find the
        machine at the token's state or exactly one step past it. The live
        ``request.security_lower_tf`` baseline is the state after the latest
        confirmed intrabar, and every provisional and developing intrabar run
        since is a step of its own (see ``LiveLtfCollector``), so the machine can
        be any number of steps past the token. Every step taken since the token is
        undone, newest first: its in-place ring and window writes are put back
        from the journal when they went into the lists the token holds, and a
        grown ring or window is replaced by the one the token holds, which the
        steps left untouched.
        ``memo`` stays when the restored state still holds every stored bar below
        the one its walk ran on — the bar-start rollback of a shared loop call site,
        whose iterations replay that walk — and is dropped otherwise: the bars
        replayed behind a deeper rollback are other bars, and its key cannot tell
        them from the discarded ones.

        The machine cannot move forward again: a token taken after the one
        restored here names a state that no longer exists.

        :param token: A token of this machine.
        :raises RuntimeError: When the token names a state the machine was rolled
            back past, or is not a token of this machine.
        """
        journal = token[0]
        if journal is self.journal:
            if not journal.records:
                return
            nodes = (journal,)
        else:
            nodes = [journal]
            node = journal.following
            while node is not None:
                nodes.append(node)
                node = node.following
            if nodes[-1] is not self.journal:
                raise RuntimeError(_STALE_TOKEN_ERROR)
        (_journal, self.summ, self.compensation, self.seen, self.window, self.capacity,
         ring, self.ring_cap, self.slot, values, self.values_cap, size, self.values_pos,
         self.values_bar, ring_gen, values_gen) = token
        memo = self.memo
        if memo is not None and memo[0][0] > self.seen + 1:
            self.memo = None
        for node in reversed(nodes):
            for record in reversed(node.records):
                if record[1] >= 0 and record[0] == ring_gen:
                    ring[record[1]] = record[2]
                if record[4] >= 0 and record[3] == values_gen:
                    values[record[4]] = record[5]
        if len(values) > size:
            del values[size:]
        self.ring = ring
        self.values = values
        self.values_size = size
        self.ring_gen = ring_gen
        self.values_gen = values_gen
        journal.records.clear()
        journal.following = None
        self.journal = journal


#: The pure-Python machine. ``_native_rolling_sum``, when built, rebinds
#: :class:`SumMachine` and :data:`rolling_sum_step` to its compiled twin
#: (bit-identical, see that module); this keeps the original reachable for the tests
#: that hold the two against each other.
PYTHON_SUM_MACHINE = SumMachine

#: One machine step as a plain function of the machine. ``lib._math_stateful.sum``
#: calls this instead of the method: a method call on a state slot is a callee the
#: slot transform cannot resolve, and it would bind it at an anchored call site on
#: every call.
rolling_sum_step = SumMachine.step

if not _os.environ.get('PYNE_NO_NATIVE_MATH'):
    try:
        from . import _native_rolling_sum  # noqa: F401 -- importing it installs the twin
    except ImportError:
        pass
