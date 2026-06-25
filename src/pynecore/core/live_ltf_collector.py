"""
Live state machine for ``request.security_lower_tf``.

Drives one chart period's lower-timeframe (LTF) intrabar executions and
accumulates their expression values into a :class:`LiveLtfWindow`. This is the
orchestration layer extracted from the security subprocess so the hard parts —
monotonic ``bar_index``, snapshot save/restore around the developing intrabar,
finalize-on-close, provisional rollover under a reordered feed, and
per-chart-period window reset — can be exercised in-process, with no subprocess,
by injecting the side effects as callbacks. The collector only *sequences* those
callbacks; it performs no I/O and touches no shared memory.

Model (TradingView, verified by a realtime probe on BINANCE:BTCUSDT 30m/3m):
while a chart bar develops, the LTF array is the closed intrabars of
``[period_start, period_end)`` plus the currently-developing intrabar as the
live *last* element; at chart close it converges to the full closed period. The
``bar_index`` is the continuous LTF series index — it never rewinds at a
chart-period boundary (only the window resets), so expression history such as a
``ta.sma`` inside the LTF expression stays intact across periods.

Reordered feed (provisional rollover): a live feed can deliver the forming tick
of intrabar ``N+1`` ahead of the late close of intrabar ``N`` (e.g. cTrader
closes a slot on the next slot's open, or plain stream reordering). The array
must still grow when ``N+1`` starts, so ``N`` is *promoted* to a provisional
element holding its last developing value at its original ``bar_index``; when
``N``'s close finally arrives it *replaces* that value at the same index (it is
never appended at a new index). Because the logical evaluation order is
timestamp order regardless of arrival order, a late close that shifts the
confirmed baseline triggers a *replay* of the still-pending provisional chain
and the developing tail on the corrected baseline — so a ``var``/``ta.*`` LTF
expression is never poisoned by a provisional intrabar whose close later
differs.

Baseline invariant: the saved baseline is the var state *after the latest
confirmed (closed) intrabar and before any provisional/developing execution*. It
is saved after every confirmed intrabar run and restored before every re-run at
an already-used ``bar_index`` — a developing re-tick, a provisional-chain
replay, or a developing/provisional intrabar's own confirmed close. A brand-new
intrabar at a fresh ``bar_index`` runs against the current baseline and never
saves afterwards (until it is confirmed). ``restore`` also resets the per-bar
add/set tracking so a ``Series.add`` at the reused index degrades to a ``set``
(overwrite), matching the proven live-HTF developing mechanism; a single restore
then lets a sequence of intrabars at distinct indices each append exactly once.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .live_ltf_window import LiveLtfWindow

__all__ = ["LiveLtfCollector", "IntrabarLike"]


class IntrabarLike(Protocol):
    """Minimal shape the collector needs from an intrabar (an ``OHLCV``).

    Only the ``timestamp`` (seconds) is read here; the value an intrabar
    contributes to the array comes from ``run_intrabar``, not from the OHLCV.
    ``OHLCV`` is an immutable ``NamedTuple``, so a reference kept for replay
    cannot be mutated out from under the collector.
    """

    timestamp: int


@dataclass
class _Intrabar:
    """A replayable in-period intrabar slot.

    Carries the last-known OHLCV alongside the assigned ``bar_index`` so a
    provisional intrabar can be re-run on a corrected baseline after a late close
    (replay needs the original input, not just the timestamp/index).

    ``is_new`` is the pending ``barstate.isnew`` flag: it is True only until the
    slot has executed once. A bar promoted from the developing tail already fired
    ``is_new=True`` while it was developing, so its slot starts False; a bar that
    is deferred straight into the chain without ever being a developing tail (a
    newer closed bar held behind an unresolved gap) starts True so its very first
    replay still reports a new bar, exactly like the normal closed-bar path. The
    flag is consumed (cleared) by the first run, so later replays report False.
    """

    ts_ms: int
    bar_index: int
    ohlcv: Any
    is_new: bool = False


class LiveLtfCollector:
    """Sequences live LTF intrabar runs into a per-chart-period array.

    :param run_intrabar: ``(ohlcv, bar_index, confirmed, is_new, islast) ->
        value`` — runs the LTF script for one intrabar (sets lib properties +
        barstate, executes ``main``) and returns the expression value it wrote.
        ``islast`` is the live last element flag (``barstate.islast``): true
        only for the single developing tail, false for every closed and
        provisional (rolled-over) intrabar — a provisional is no longer the
        last element once a newer intrabar has started, even though it is still
        unconfirmed.
    :param save_baseline: Persist the current var state as the rollback
        baseline (no-op acceptable when the script has no ``var`` slots).
    :param restore_baseline: Restore the var state to the saved baseline and
        reset per-bar add/set tracking, in preparation for a re-run at an
        already-used ``bar_index``.
    :param start_bar_index: The first ``bar_index`` to assign (default 0).
    """

    def __init__(
        self,
        run_intrabar: Callable[[Any, int, bool, bool, bool], Any],
        save_baseline: Callable[[], None],
        restore_baseline: Callable[[], None],
        *,
        start_bar_index: int = 0,
    ) -> None:
        self._run_intrabar = run_intrabar
        self._save_baseline = save_baseline
        self._restore_baseline = restore_baseline

        self._window = LiveLtfWindow()
        self._bar_index = start_bar_index
        self._seen_any = False
        self._has_baseline = False

        self._period_start: int | None = None
        self._developing: _Intrabar | None = None
        self._pending: list[_Intrabar] = []
        self._late_closes: dict[int, IntrabarLike] = {}

    @property
    def bar_index(self) -> int:
        """The most recently assigned LTF series index (monotonic)."""
        return self._bar_index

    @property
    def window(self) -> LiveLtfWindow:
        """The underlying per-chart-period accumulator (for inspection)."""
        return self._window

    def _advance_index(self) -> int:
        """Assign the next ``bar_index``; the very first bar reuses the start."""
        if self._seen_any:
            self._bar_index += 1
        self._seen_any = True
        return self._bar_index

    def _ensure_baseline(self) -> None:
        """Save an initial baseline if none exists (process opened on a dev bar)."""
        if not self._has_baseline:
            self._save_baseline()
            self._has_baseline = True

    def _is_pending_ts(self, ts_ms: int) -> bool:
        """Whether ``ts_ms`` belongs to a rolled-over provisional awaiting close."""
        return any(p.ts_ms == ts_ms for p in self._pending)

    def _run_pending(self, slot: _Intrabar, confirmed: bool) -> Any:
        """Run one pending provisional slot, consuming its first-run ``is_new``.

        A deferred-into-the-chain slot must report ``barstate.isnew`` on its very
        first execution (the normal closed-bar path does); the flag is cleared
        here so every later replay of the same slot reports ``is_new=False``.
        Pending slots are never the live last element, so ``islast`` is False.
        """
        is_new = slot.is_new
        slot.is_new = False
        return self._run_intrabar(slot.ohlcv, slot.bar_index, confirmed, is_new, False)

    def _drain_pending_closes(self) -> bool:
        """Finalize buffered late closes for the provisional chain, oldest first.

        A provisional is finalized only once its own close is buffered AND it is
        at the front of the chain (closes apply front-to-back so the confirmed
        baseline advances contiguously). Each finalize restores the baseline,
        re-runs the intrabar as confirmed at its ORIGINAL ``bar_index``, replaces
        the provisional value with the confirmed one, and saves the new baseline.

        :return: Whether at least one provisional was finalized.
        """
        drained = False
        while self._pending and self._pending[0].ts_ms in self._late_closes:
            slot = self._pending.pop(0)
            bar = self._late_closes.pop(slot.ts_ms)
            slot.ohlcv = bar
            self._restore_baseline()
            value = self._run_pending(slot, True)
            self._window.finalize_first_provisional(value)
            self._save_baseline()
            self._has_baseline = True
            drained = True
        return drained

    def _rebuild_tail(self, *, run_developing: bool) -> None:
        """Replay the pending provisional chain (and optionally the developing
        tail) on the current confirmed baseline.

        Restores once to the post-confirmed baseline, then re-runs every pending
        provisional in timestamp order (each at its own index, so each
        ``Series.add`` appends exactly once) and writes its value back in place;
        finally re-runs the developing tail on top when requested. None of these
        runs save a baseline — they are all unconfirmed.
        """
        self._restore_baseline()
        for i, slot in enumerate(self._pending):
            value = self._run_pending(slot, False)
            self._window.replace_provisional(i, value)
        if run_developing and self._developing is not None:
            value = self._run_intrabar(
                self._developing.ohlcv, self._developing.bar_index, False, False, True
            )
            self._window.set_developing(value)

    def process_round(
        self,
        period_start: int,
        period_end_exclusive: int,
        closed_bars: Sequence[IntrabarLike],
        developing_bar: IntrabarLike | None,
        chart_confirmed: bool,
    ) -> list[Any]:
        """Advance one chart tick and return the published LTF array (a copy).

        ``closed_bars`` are the newly available closed intrabars (ordered, and
        already bounded above by ``period_end_exclusive`` by the caller — a bar
        belonging to a future chart period must be withheld until that period is
        current). Every closed intrabar is *run* to maintain expression history;
        only those at or after ``period_start`` join the array. ``developing_bar``
        is the currently-forming intrabar (``None`` when none is available, e.g.
        right at period open); it is processed only while the chart bar itself is
        still developing (``chart_confirmed`` is False).

        :param period_start: The chart bar's open instant (ms).
        :param period_end_exclusive: The chart bar's period end (ms, exclusive).
        :param closed_bars: Newly available closed intrabars, time-ordered.
        :param developing_bar: The currently-developing intrabar, or ``None``.
        :param chart_confirmed: Whether the chart bar has closed (period complete).
        :return: A fresh list — closed + provisional values plus the developing
            tail, if any.
        """
        # ── Chart-period roll ──
        # The closing period's confirmed round always arrives (with its old
        # ``period_start``) before the first round of the next period, so the
        # window is already finalized here; the reset just clears it. Stale
        # rollover tracking is dropped defensively.
        if period_start != self._period_start:
            self._window.reset()
            self._period_start = period_start
            self._developing = None
            self._pending = []
            self._late_closes = {}

        # ── Closed intrabars (run all for history; window membership in-period) ──
        # Tracks whether the confirmed baseline advanced inside this loop (an
        # in-loop drain finalized a provisional, or a brand-new closed bar saved).
        # A still-tracked developing tail computed against the OLD baseline is then
        # stale and must be rebuilt below even if the post-loop drain finalizes
        # nothing (it already drained in the loop).
        baseline_advanced = False
        for bar in closed_bars:
            ts_ms = int(bar.timestamp * 1000)
            if (self._developing is not None and ts_ms == self._developing.ts_ms
                    and not self._pending):
                # In-order close of the current developing intrabar (no reorder):
                # re-run as confirmed at the SAME index (its dev ticks restored
                # away) and finalize the tail. The confirmed close may differ
                # from the last developing tick.
                self._restore_baseline()
                value = self._run_intrabar(bar, self._developing.bar_index, True, False, False)
                self._window.finalize_developing(value)
                self._save_baseline()
                self._has_baseline = True
                self._developing = None
            elif self._developing is not None and ts_ms == self._developing.ts_ms:
                # Late close of the current developing intrabar arriving while a
                # provisional chain still leads it (its close was batched behind
                # the heads' closes). Promote it to the back of the chain — it has
                # the largest timestamp — and buffer its close, so the drain below
                # finalizes the whole chain front-to-back in timestamp order, each
                # at its ORIGINAL bar_index, instead of appending a new index.
                self._window.promote_developing_to_provisional()
                self._pending.append(self._developing)
                self._developing = None
                self._late_closes[ts_ms] = bar
            elif self._is_pending_ts(ts_ms):
                # Late close of a rolled-over provisional: buffer it and finalize
                # the provisional chain front-to-back below, so the close
                # replaces the provisional value at its ORIGINAL bar_index rather
                # than appending at a new one.
                self._late_closes[ts_ms] = bar
            else:
                # A brand-new closed intrabar. The streamer can batch a pending
                # provisional's late close ahead of (or behind) this newer bar in
                # the same round; finalize every drainable provisional FIRST so the
                # baseline advances and the chain becomes closed in timestamp order
                # before this newer bar runs and appends. Without this drain the
                # newer bar would run on the stale pre-close baseline, save it, and
                # land in the closed segment ahead of the still-provisional chain —
                # corrupting both the published order and the persisted var state.
                if self._drain_pending_closes():
                    baseline_advanced = True
                # If an older developing intrabar is still unconfirmed AND gets no
                # fresh forming tick this round (its close was skipped — a gap),
                # promote it to the BACK of the provisional chain so this newer bar
                # is deferred behind it below instead of falling through to the
                # closed segment. The developing tail always carries the largest
                # timestamp of the tracked intrabars (it started after every pending
                # provisional), so the back of the chain keeps timestamp order even
                # when an earlier provisional gap left ``_pending`` non-empty.
                # Without this the newer bar would publish ahead of the older
                # developing tail (out of timestamp order) and advance the saved
                # baseline past the gap. A developing that IS still forming this round
                # (its tick is in ``developing_bar``) stays the live tail and the
                # newer closed bar legitimately appends ahead of it.
                if (self._developing is not None
                        and self._developing.ts_ms < ts_ms
                        and not (developing_bar is not None
                                 and int(developing_bar.timestamp * 1000)
                                 == self._developing.ts_ms)):
                    self._window.promote_developing_to_provisional()
                    self._pending.append(self._developing)
                    self._developing = None
                idx = self._advance_index()
                in_period = period_start <= ts_ms < period_end_exclusive
                if self._pending and in_period:
                    # A provider gap left an older provisional's close missing, so
                    # the drain could not empty the chain. This newer bar's value
                    # cannot join the closed segment (it sits ahead of the still-
                    # pending chain in ``publish()``) nor advance the saved baseline
                    # past the unresolved earlier intrabar. Defer it into the
                    # provisional chain behind that gap: hold it at its own index,
                    # buffer its (already-arrived) close so it finalizes the moment
                    # the chain ahead drains, and replay the whole chain in place.
                    # It enters the chain as a brand-new bar, so its first replay
                    # must still report ``barstate.isnew`` like the normal closed
                    # path below — unlike a slot promoted from the developing tail.
                    self._pending.append(
                        _Intrabar(ts_ms=ts_ms, bar_index=idx, ohlcv=bar, is_new=True))
                    self._late_closes[ts_ms] = bar
                    self._window.append_provisional(None)
                    self._rebuild_tail(run_developing=True)
                else:
                    value = self._run_intrabar(bar, idx, True, True, False)
                    if in_period:
                        self._window.append_closed(value)
                    self._save_baseline()
                    self._has_baseline = True
                    baseline_advanced = True

        # ── Finalize buffered late closes; replay the rest of the chain ──
        # After the baseline advances, the still-pending provisionals — and a
        # developing tail computed against the pre-advance baseline — are stale.
        # Rebuild them now unless the developing block below will rebuild the
        # whole tail this round (a fresh developing tick is present), which avoids
        # replaying the provisional chain twice. The post-loop drain alone is not
        # enough to detect staleness: if the only baseline advance happened INSIDE
        # the closed-bar loop (an in-loop drain, or a brand-new bar's save) this
        # final drain finalizes nothing and returns False, yet a tail left from a
        # prior round still sits on the old baseline — so ``baseline_advanced``
        # forces the rebuild too. Skip entirely when nothing remains to replay (the
        # whole chain closed and no developing tail) — a bare ``_rebuild_tail``
        # would only restore the baseline, and that restore also resets
        # function-instance state without a re-run to rebuild it, poisoning the
        # next intrabar's child (``ta.*``-in-function) history.
        if ((self._drain_pending_closes() or baseline_advanced) and developing_bar is None
                and (self._pending or self._developing is not None)):
            self._rebuild_tail(run_developing=True)

        # ── Confirmed-round convergence ──
        # A confirmed chart bar's array is exactly its closed intrabars. Any
        # unconfirmed tail left from the last forming tick — a developing tail or
        # a rolled-over provisional whose close never arrived (a provider gap or
        # grace timeout; the caller publishes the round anyway rather than
        # stalling the bot) — must not appear in a confirmed array. The discarded
        # runs also mutated the script's var state without saving a baseline, so
        # restore the post-confirmed baseline before dropping — otherwise the next
        # period's intrabar would compound from the abandoned ticks. Then drop the
        # tails and clear the rollover tracking so a late close in a later period
        # is treated as its own intrabar, not a re-finalization at a stale index.
        if chart_confirmed and (self._window.has_developing or self._window.has_provisional):
            if self._has_baseline:
                self._restore_baseline()
            self._window.drop_developing()
            self._window.drop_provisionals()
            self._developing = None
            self._pending = []
            self._late_closes = {}

        # ── Developing intrabar (live last element) ──
        if not chart_confirmed and developing_bar is not None:
            ts_ms = int(developing_bar.timestamp * 1000)
            if period_start <= ts_ms < period_end_exclusive:
                cur = self._developing
                if cur is not None and ts_ms == cur.ts_ms:
                    # Another tick of the same developing intrabar: rewind to the
                    # baseline and re-run in place (replaying any provisional
                    # chain ahead of it so it accumulates from the confirmed
                    # prefix, not an abandoned tick).
                    cur.ohlcv = developing_bar
                    self._rebuild_tail(run_developing=True)
                elif self._is_pending_ts(ts_ms):
                    # A stale forming update for an intrabar that already rolled
                    # over into the provisional chain (the feed reordered N's late
                    # tick behind N+1's open). It must NOT allocate a new index —
                    # refresh the provisional's stored OHLCV so its eventual close
                    # / replay uses the latest forming value, then rebuild the
                    # chain (and developing tail) in place at the same indices.
                    for slot in self._pending:
                        if slot.ts_ms == ts_ms:
                            slot.ohlcv = developing_bar
                            break
                    self._rebuild_tail(run_developing=True)
                else:
                    # A fresh developing intrabar at a new timestamp.
                    if cur is not None:
                        # RACE: the previous developing intrabar never closed in
                        # order; promote it to a provisional element (preserve its
                        # slot + index) before advancing, so its late close can
                        # replace it in place rather than append at a new index.
                        self._window.promote_developing_to_provisional()
                        self._pending.append(cur)
                    self._ensure_baseline()
                    idx = self._advance_index()
                    self._developing = _Intrabar(ts_ms=ts_ms, bar_index=idx,
                                                 ohlcv=developing_bar)
                    if self._pending:
                        # Rebuild the provisional chain on the post-confirmed
                        # baseline, then run the NEW developing on top (first run
                        # at a fresh index -> is_new=True), without saving.
                        self._restore_baseline()
                        for i, slot in enumerate(self._pending):
                            v = self._run_pending(slot, False)
                            self._window.replace_provisional(i, v)
                        value = self._run_intrabar(developing_bar, idx, False, True, True)
                        self._window.set_developing(value)
                    else:
                        # No reorder in flight: run once at a new index against
                        # the post-confirmed baseline; do NOT save after.
                        value = self._run_intrabar(developing_bar, idx, False, True, True)
                        self._window.set_developing(value)

        return self._window.publish()
