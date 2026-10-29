"""
Pine's rolling-sum state machine: ``math.sum`` and everything built on it.

Pine's ``math.sum`` (and through it ``ta.sma``) keeps a Neumaier-compensated
rolling sum: an accumulator ``s`` and a compensation register ``c`` that
collects the exact rounding error of every add, and the reported sum is
``fl(s + c)``. Each bar evicts the raw source value stored ``length`` bars ago
and adds the new one, each a compensated add in that order, and the pair is
never renormalized: after a large value has passed through the window, ``c``
keeps carrying the low-order bits the accumulator rounded away, and so does
TradingView (MEASURED 2026-10-29, CAPITALCOM:BTCUSD 5m, 20.9k bars per probe:
full-mantissa sources in [0.25, 0.75] and around 62000 at lengths 2..14, real
prices at length 3, 1e15 spikes through length 5 and 300 windows, ±1e8
alternating terms, every bar bit-exact; a readout of ``s`` alone, a
renormalized pair, a Kahan pair or an add-first order each miss thousands of
bars). TradingView switched its engine to this between 2026-07-31 and
2026-10-29; the previous fused-Kahan machine with its re-baseline rule is the
engine those earlier probes measured and is wrong on today's output.

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
"""
import os as _os
from typing import Any

from .safe_convert import native_int
from .series import SeriesImpl
from ..types.na import na_float

__all__ = ['SumMachine', 'rolling_sum_step']

_STALE_TOKEN_ERROR = "math.sum rollback target is not a state its machine can return to"


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

    A record is ``(values_gen, win_at, win_old)``: the window slot one step
    overwrote, with its old value and the generation of the list it was written to
    (an index of -1 marks "nothing overwritten"). ``following`` is the journal of
    the next snapshot taken after this one, so a token reaches every step taken
    since it, and only a token keeps the journals behind the machine's current one
    alive.
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
    ``window`` of the previous call, the history ``capacity`` and the raw value
    window (``values``, a circular buffer that behaves exactly like the compacted
    ``SeriesImpl`` the machine used to read: ``values_cap`` is its
    ``max_bars_back + 1``, ``values_size`` / ``values_pos`` its size and write
    position, ``values_bar`` the ``bar_index`` of its last write). ``memo`` is not
    state: it is a replay of the eviction walk (see :meth:`step`), and a rollback
    only drops it when it discards stored bars the walk was computed over.

    The rollback bookkeeping is ``journal`` (the :class:`_Journal` of the latest
    snapshot, None until the first one: a machine nobody snapshots records
    nothing), ``values_gen`` (the generation of the current window list, fresh
    whenever a step replaces it instead of writing into it) and ``counter`` (the
    last generation handed out).
    """

    __slots__ = ('summ', 'compensation', 'seen', 'window', 'capacity',
                 'values', 'values_cap', 'values_size', 'values_pos', 'values_bar',
                 'memo', 'journal', 'values_gen', 'counter')

    def __init__(self) -> None:
        _bar_source()
        self.summ = 0.0
        self.compensation = 0.0
        self.seen = 0
        self.window = 0
        self.capacity = SeriesImpl.DEFAULT_MAX_BARS_BACK
        self.values: list[float] = []
        self.values_cap = SeriesImpl.DEFAULT_MAX_BARS_BACK + 1
        self.values_size = 0
        self.values_pos = 0
        self.values_bar: Any = -1
        self.memo: tuple | None = None
        self.journal: _Journal | None = None
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
        # Pine's engine keeps a Neumaier-compensated rolling sum: every add of a value
        # ``v`` is ``t = fl(s + v)``, ``c += (s - t) + v`` when ``|s| >= |v|`` and
        # ``c += (v - t) + s`` otherwise (the exact rounding error either way), then
        # ``s = t``; the reported sum is ``fl(s + c)``. Each bar evicts the raw value
        # stored ``length`` bars ago and adds the new value, in that order, and while
        # the window fills a bar evicts nothing. The pair is never renormalized
        # (MEASURED, see the module docstring).

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
            return self.summ + self.compensation if new_w >= length else na_float

        # Every call past this point changes the machine, and journals what a rollback
        # has to put back beyond the fields a snapshot holds anyway: the slot it
        # overwrites in the window list it started with.
        values_gen = self.values_gen
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
            # The eviction reads the raw window at offset ``length``. Grow the window
            # so that offset stays addressable for lengths beyond the default
            # ``max_bars_back``; otherwise the read raises and any ``ta.sma`` /
            # ``ta.sum`` with a length above the default fails right after warmup. The
            # resize is monotonic and floored at the window's own default: a series
            # ``length`` that dips low must not shrink the window, or the history a
            # later increase needs would already have been thrown away.
            if length > self.capacity:
                self.capacity = length
                self._resize_values(length)

        c = self.compensation
        s = self.summ

        # The window is the last ``length`` non-na values, so a moved length both
        # DROPS and ADMITS values around it, and TradingView walks that change as a
        # SEQUENCE of ordinary compensated adds rather than one fused one.
        # ``shift`` is 1 on a stored bar (every older offset moves up by one) and 0
        # on an na bar, so relative to this bar's offset 0 the previous window
        # covered ``shift``..``prev_w - 1 + shift``. Offsets ``new_w``..
        # ``prev_w - 1 + shift`` LEAVE oldest first, each an eviction-only add, and
        # the newest of them is the one evicted by this bar's own step — in the
        # steady state it is the only one, which is exactly the proven single-evict
        # step. A grown window ADMITS offsets ``prev_w + shift``..``new_w`` instead —
        # one PAST the new window, newest first — and the step then evicts offset
        # ``new_w`` again, so every bar whose offset ``new_w`` exists ends on the same
        # evict-and-add, and only a window reaching back to the first stored bar ends
        # on a plain add. The walk is the one TradingView's previous engine was
        # measured with (probes sumlen3/6/7, wg/wh/wk, wr: isolated and consecutive
        # grows and shrinks over 21.7k bars on six sources) and the one its current
        # engine keeps (MEASURED 2026-10-29: ±5 jumps, sawtooth and pseudo-random
        # lengths 2..14 over 1e15 spikes, 20.9k bars each, every bar bit-exact; the
        # admitted values re-added as one pre-summed delta, or admitted only down to
        # the window's own edge, drift off within a few hundred bars).
        if source_na:
            shift = 0
        else:
            shift = 1

        if prev_w + shift > new_w:
            # Oldest first with the RAW source values, each one compensated add of the
            # negated value, the newest leaving value left for the step below.
            # The offsets that leave are ``new_w + 1``..``top``, and an offset is
            # its own window offset in BOTH regimes: on a stored bar the window
            # moved with the machine, on an na bar neither moved. Reading one lower
            # on an na bar would evict offset ``new_w`` twice (the step below
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
                        v = -v
                        t = s + v
                        if (s if s >= 0.0 else -s) >= (v if v >= 0.0 else -v):
                            c += (s - t) + v
                        else:
                            c += (v - t) + s
                        s = t
                        ws[i] = s
                        wc[i] = c
                        i += 1
        elif new_w > prev_w + shift or (new_w < n and not source_na):
            # Newest first with the RAW source values, each one compensated add, down
            # to offset ``new_w`` itself when it exists (a +1 grow of a full window
            # admits just that one). On an na bar the window does not move, so the
            # walk stops at the window's own edge and nothing is evicted below
            # (MEASURED 2026-10-29, probe newsum2: sawtooth lengths with every 11th
            # bar na over 1e15 spikes, 20.9k bars bit-exact; admitting one past the
            # edge there drifts off at the first such grow). Unlike the eviction walk
            # above this walk cannot be memoized across the iterations of a shared
            # loop call site: two lengths share the head of the fold but not its
            # input range, and every length has to be folded from the bar-start state.
            if source_na:
                top = new_w
            else:
                top = new_w + 1 if new_w < n else n
            for v in reversed(self._oldest(prev_w + shift, top)):
                t = s + v
                if (s if s >= 0.0 else -s) >= (v if v >= 0.0 else -v):
                    c += (s - t) + v
                else:
                    c += (v - t) + s
                s = t

        if new_w < n and (shift or prev_w > new_w):
            # Evict offset ``new_w`` — the raw value that leaves the window this bar:
            # on a stored bar always, on an na bar only when the window shrank onto it
            v = -self._oldest(new_w, new_w + 1)[0]
            t = s + v
            if (s if s >= 0.0 else -s) >= (v if v >= 0.0 else -v):
                c += (s - t) + v
            else:
                c += (v - t) + s
            s = t
        if not source_na:
            t = s + value
            if (s if s >= 0.0 else -s) >= (value if value >= 0.0 else -value):
                c += (s - t) + value
            else:
                c += (value - t) + s
            s = t
        self.summ = s
        self.compensation = c
        self.seen = n
        self.window = new_w
        journal = self.journal
        if journal is not None:
            journal.records.append((values_gen, win_at, win_old))

        return s + c if new_w >= length else na_float

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
                self.capacity, self.values, self.values_cap, self.values_size,
                self.values_pos, self.values_bar, self.values_gen)

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
        undone, newest first: its in-place window write is put back from the
        journal when it went into the list the token holds, and a grown window is
        replaced by the one the token holds, which the steps left untouched.
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
         values, self.values_cap, size, self.values_pos, self.values_bar,
         values_gen) = token
        memo = self.memo
        if memo is not None and memo[0][0] > self.seen + 1:
            self.memo = None
        for node in reversed(nodes):
            for record in reversed(node.records):
                if record[1] >= 0 and record[0] == values_gen:
                    values[record[1]] = record[2]
        if len(values) > size:
            del values[size:]
        self.values = values
        self.values_size = size
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
