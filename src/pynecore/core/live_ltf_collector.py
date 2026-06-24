"""
Live state machine for ``request.security_lower_tf``.

Drives one chart period's lower-timeframe (LTF) intrabar executions and
accumulates their expression values into a :class:`LiveLtfWindow`. This is the
orchestration layer extracted from the security subprocess so the hard parts —
monotonic ``bar_index``, snapshot save/restore around the developing intrabar,
finalize-on-close, and per-chart-period window reset — can be exercised
in-process, with no subprocess, by injecting the side effects as callbacks. The
collector only *sequences* those callbacks; it performs no I/O and touches no
shared memory.

Model (TradingView, verified by a realtime probe on BINANCE:BTCUSDT 30m/3m):
while a chart bar develops, the LTF array is the closed intrabars of
``[period_start, period_end)`` plus the currently-developing intrabar as the
live *last* element; at chart close it converges to the full closed period. The
``bar_index`` is the continuous LTF series index — it never rewinds at a
chart-period boundary (only the window resets), so expression history such as a
``ta.sma`` inside the LTF expression stays intact across periods.

Baseline invariant: the saved baseline is the var state *after the latest
confirmed (closed) intrabar and before any developing intrabar execution*. It is
saved after every confirmed intrabar run and restored before every re-run at an
already-used ``bar_index`` — a developing re-tick or the developing intrabar's
own confirmed close. A brand-new intrabar at a fresh ``bar_index`` runs against
the current baseline and never saves afterwards. ``restore`` also resets the
per-bar add/set tracking so a ``Series.add`` at the reused index degrades to a
``set`` (overwrite), matching the proven live-HTF developing mechanism.
"""

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from .live_ltf_window import LiveLtfWindow

__all__ = ["LiveLtfCollector", "IntrabarLike"]


class IntrabarLike(Protocol):
    """Minimal shape the collector needs from an intrabar (an ``OHLCV``).

    Only the ``timestamp`` (seconds) is read here; the value an intrabar
    contributes to the array comes from ``run_intrabar``, not from the OHLCV.
    """

    timestamp: int


class LiveLtfCollector:
    """Sequences live LTF intrabar runs into a per-chart-period array.

    :param run_intrabar: ``(ohlcv, bar_index, confirmed, is_new) -> value`` —
        runs the LTF script for one intrabar (sets lib properties + barstate,
        executes ``main``) and returns the expression value it wrote.
    :param save_baseline: Persist the current var state as the rollback
        baseline (no-op acceptable when the script has no ``var`` slots).
    :param restore_baseline: Restore the var state to the saved baseline and
        reset per-bar add/set tracking, in preparation for a re-run at an
        already-used ``bar_index``.
    :param start_bar_index: The first ``bar_index`` to assign (default 0).
    """

    def __init__(
        self,
        run_intrabar: Callable[[Any, int, bool, bool], Any],
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
        self._active_dev_bar_index: int | None = None
        self._dev_intrabar_ms: int | None = None

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
        :return: A fresh list — closed values plus the developing tail, if any.
        """
        # ── Chart-period roll ──
        # The closing period's confirmed round always arrives (with its old
        # ``period_start``) before the first round of the next period, so the
        # window is already finalized here; the reset just clears it. Stale
        # developing tracking is dropped defensively.
        if period_start != self._period_start:
            self._window.reset()
            self._period_start = period_start
            self._dev_intrabar_ms = None
            self._active_dev_bar_index = None

        # ── Closed intrabars (run all for history; append those in-period) ──
        for bar in closed_bars:
            ts_ms = int(bar.timestamp * 1000)

            dev_ms = self._dev_intrabar_ms
            dev_idx = self._active_dev_bar_index
            if dev_ms is not None and dev_idx is not None and ts_ms == dev_ms:
                # The developing intrabar has closed: re-run as confirmed at the
                # SAME index (its dev ticks restored away) and finalize the tail.
                # The confirmed close may differ from the last developing tick.
                self._restore_baseline()
                value = self._run_intrabar(bar, dev_idx, True, False)
                self._window.finalize_developing(value)
                self._save_baseline()
                self._has_baseline = True
                self._dev_intrabar_ms = None
                self._active_dev_bar_index = None
            else:
                idx = self._advance_index()
                value = self._run_intrabar(bar, idx, True, True)
                if period_start <= ts_ms < period_end_exclusive:
                    self._window.append_closed(value)
                self._save_baseline()
                self._has_baseline = True

        # ── Confirmed-round convergence ──
        # A confirmed chart bar's array is exactly its closed intrabars. If the
        # final intrabar's close never arrived (a provider gap or grace timeout —
        # the caller publishes the round anyway rather than stalling the bot), the
        # developing tail left from the last forming tick is unconfirmed and must
        # not appear in a confirmed array. The discarded developing run also
        # mutated the script's var state without saving a baseline, so restore the
        # post-confirmed baseline before dropping — otherwise the next period's
        # intrabar would compound from the abandoned tick instead of the saved
        # state. Then drop the tail and clear the developing tracking so a late
        # close in a later period is treated as its own intrabar, not a
        # re-finalization at a stale index.
        if chart_confirmed and self._window.has_developing:
            if self._has_baseline:
                self._restore_baseline()
            self._window.drop_developing()
            self._dev_intrabar_ms = None
            self._active_dev_bar_index = None

        # ── Developing intrabar (live last element) ──
        if not chart_confirmed and developing_bar is not None:
            ts_ms = int(developing_bar.timestamp * 1000)
            if period_start <= ts_ms < period_end_exclusive:
                if ts_ms != self._dev_intrabar_ms:
                    # A fresh developing intrabar: run once at a new index
                    # against the post-confirmed baseline; do NOT save after.
                    self._ensure_baseline()
                    idx = self._advance_index()
                    self._active_dev_bar_index = idx
                    self._dev_intrabar_ms = ts_ms
                    value = self._run_intrabar(developing_bar, idx, False, True)
                    self._window.set_developing(value)
                elif self._active_dev_bar_index is not None:
                    # Another tick of the same developing intrabar: rewind to the
                    # baseline and re-run in place at the same index.
                    self._restore_baseline()
                    value = self._run_intrabar(
                        developing_bar, self._active_dev_bar_index, False, False
                    )
                    self._window.set_developing(value)

        return self._window.publish()
