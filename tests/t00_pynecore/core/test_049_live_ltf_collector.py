"""
In-process correctness tests for ``LiveLtfCollector`` — the live
``request.security_lower_tf`` state machine.

These drive the collector directly with fake intrabars and injected side-effect
callbacks (no subprocess, no shared memory), asserting the published array
against TradingView's live developing-bar oracle (verified by a realtime probe
on BINANCE:BTCUSDT 30m/3m): while the chart bar develops the array is the closed
intrabars of ``[period_start, period_end)`` plus the developing intrabar as the
live last element; at chart close it converges to the full closed period; the
``bar_index`` is monotonic across chart-period boundaries.
"""
from dataclasses import dataclass

from pynecore.core.live_ltf_collector import LiveLtfCollector

# A 30m chart period split into 3m intrabars (ms).
T = 1_700_000_000_000
LTF = 3 * 60 * 1000        # 3m intrabar span
TF = 30 * 60 * 1000        # 30m chart span
PERIOD_END = T + TF


@dataclass
class _Bar:
    """Minimal intrabar: an open time (seconds) and the value its run yields."""
    timestamp: int          # seconds
    value: float

    @staticmethod
    def at(ms: int, value: float) -> "_Bar":
        return _Bar(timestamp=ms // 1000, value=value)


class _Harness:
    """Records run/save/restore calls; each run returns ``bar.value``."""

    def __init__(self):
        self.runs: list[tuple[int, bool, bool, float]] = []
        self.islasts: list[tuple[int, bool]] = []
        self.ops: list[str] = []
        self.saves = 0
        self.restores = 0

    def run_intrabar(self, bar, bar_index, confirmed, is_new, islast):
        self.ops.append("run")
        self.runs.append((bar_index, confirmed, is_new, bar.value))
        self.islasts.append((bar_index, islast))
        return bar.value

    def save_baseline(self):
        self.ops.append("save")
        self.saves += 1

    def restore_baseline(self):
        self.ops.append("restore")
        self.restores += 1

    def collector(self, start_bar_index: int = 0) -> LiveLtfCollector:
        return LiveLtfCollector(
            self.run_intrabar, self.save_baseline, self.restore_baseline,
            start_bar_index=start_bar_index,
        )


class _AccHarness:
    """Models a script ``var`` accumulator so snapshot discipline is testable.

    ``run`` does ``acc += bar.value; return acc`` — the returned value depends on
    accumulated state, so a missing or wrong restore is observable as a
    compounded value rather than the post-confirmed baseline value.
    """

    def __init__(self):
        self.acc = 0.0
        self._saved = 0.0
        self.saves = 0
        self.restores = 0

    def run_intrabar(self, bar, _bar_index, _confirmed, _is_new, _islast):
        self.acc += bar.value
        return self.acc

    def save_baseline(self):
        self._saved = self.acc
        self.saves += 1

    def restore_baseline(self):
        self.acc = self._saved
        self.restores += 1

    @property
    def saved(self) -> float:
        """The persisted post-confirmed baseline value (for assertions)."""
        return self._saved

    def collector(self) -> LiveLtfCollector:
        return LiveLtfCollector(
            self.run_intrabar, self.save_baseline, self.restore_baseline,
        )


def __test_warmup_runs_for_history_not_appended__():
    """Closed bars before ``period_start`` run (history) but stay out of the array."""
    h = _Harness()
    c = h.collector()
    warmup = [_Bar.at(T - 3 * LTF, 1.0), _Bar.at(T - 2 * LTF, 2.0), _Bar.at(T - LTF, 3.0)]
    in_period = [_Bar.at(T, 10.0), _Bar.at(T + LTF, 11.0)]

    out = c.process_round(T, PERIOD_END, warmup + in_period, None, chart_confirmed=False)

    # All five ran (history maintained); only the two in-period values are in the array.
    assert [r[0] for r in h.runs] == [0, 1, 2, 3, 4]   # monotonic bar_index
    assert out == [10.0, 11.0]
    assert c.bar_index == 4


def __test_developing_tail_is_live_last_element__():
    """A closed intrabar plus the developing intrabar -> [closed, developing]."""
    h = _Harness()
    c = h.collector()
    out = c.process_round(
        T, PERIOD_END,
        [_Bar.at(T, 100.0)],            # closed intrabar 0
        _Bar.at(T + LTF, 101.5),        # developing intrabar 1
        chart_confirmed=False,
    )
    assert out == [100.0, 101.5]
    # closed at idx 0, developing at idx 1
    assert h.runs == [(0, True, True, 100.0), (1, False, True, 101.5)]


def __test_developing_retick_revises_in_place_with_restore__():
    """Each developing tick replaces the tail in place and restores the baseline."""
    h = _Harness()
    c = h.collector()
    dev_ms = T + LTF

    c.process_round(T, PERIOD_END, [_Bar.at(T, 100.0)], _Bar.at(dev_ms, 101.0),
                    chart_confirmed=False)
    out2 = c.process_round(T, PERIOD_END, [], _Bar.at(dev_ms, 102.0),
                           chart_confirmed=False)
    out3 = c.process_round(T, PERIOD_END, [], _Bar.at(dev_ms, 101.5),
                           chart_confirmed=False)

    assert out2 == [100.0, 102.0]
    assert out3 == [100.0, 101.5]      # only the latest tick survives
    assert c.bar_index == 1            # developing never grows the index
    # Two re-ticks -> two restores; the first developing run did not restore.
    assert h.restores == 2


def __test_probe_boundary_2_to_3__():
    """Reproduce the probe's n: 2 -> 3 step when an intrabar closes."""
    h = _Harness()
    c = h.collector()
    dev0_ms = T + LTF       # intrabar 1 developing
    dev1_ms = T + 2 * LTF   # intrabar 2 developing

    # Mid-period: 1 closed + 1 developing -> size 2.
    out = c.process_round(T, PERIOD_END, [_Bar.at(T, 1.0)], _Bar.at(dev0_ms, 2.0),
                          chart_confirmed=False)
    assert out == [1.0, 2.0]

    # The developing intrabar closes (confirmed 2.5) and the next opens (3.0).
    out = c.process_round(
        T, PERIOD_END,
        [_Bar.at(dev0_ms, 2.5)],        # finalizes the developing tail
        _Bar.at(dev1_ms, 3.0),          # new developing intrabar
        chart_confirmed=False,
    )
    assert out == [1.0, 2.5, 3.0]
    assert c.bar_index == 2


def __test_chart_close_converges_to_full_period__():
    """A confirmed round finalizes the last intrabar -> full period, no tail."""
    h = _Harness()
    c = h.collector()
    last_ms = T + (TF - LTF)            # the final intrabar of the period

    # Last developing tick before close.
    c.process_round(T, PERIOD_END, [_Bar.at(T, 50.0)], _Bar.at(last_ms, 59.9),
                    chart_confirmed=False)
    # Chart confirms: the final intrabar's close (60.0, differs from 59.9) arrives.
    out = c.process_round(T, PERIOD_END, [_Bar.at(last_ms, 60.0)], None,
                          chart_confirmed=True)

    assert out == [50.0, 60.0]         # confirmed close, not the 59.9 tick
    assert not c.window.has_developing


def __test_period_roll_resets_window_keeps_bar_index__():
    """A new chart period clears the array but the LTF index keeps climbing."""
    h = _Harness()
    c = h.collector()

    # Period 1: one closed + developing, then confirm/finalize.
    c.process_round(T, PERIOD_END, [_Bar.at(T, 1.0)], _Bar.at(T + LTF, 2.0),
                    chart_confirmed=False)
    c.process_round(T, PERIOD_END, [_Bar.at(T + LTF, 2.5)], None, chart_confirmed=True)
    idx_after_p1 = c.bar_index

    # Period 2 opens: new period_start; the array starts fresh.
    out = c.process_round(PERIOD_END, PERIOD_END + TF,
                          [_Bar.at(PERIOD_END, 9.0)], _Bar.at(PERIOD_END + LTF, 9.1),
                          chart_confirmed=False)
    assert out == [9.0, 9.1]
    assert c.bar_index > idx_after_p1   # monotonic, never reset


def __test_snapshot_contamination_restored_to_post_confirmed_baseline__():
    """Re-ticks restore to the post-confirmed baseline, never compounding dev runs."""
    h = _AccHarness()
    c = h.collector()
    dev_ms = T + LTF

    # Closed intrabar (+10) -> acc 10, baseline saved at 10.
    out = c.process_round(T, PERIOD_END, [_Bar.at(T, 10.0)], None, chart_confirmed=False)
    assert out == [10.0]
    assert h.acc == 10.0

    # Developing tick (+5) -> acc 15; tail value 15, no save.
    out = c.process_round(T, PERIOD_END, [], _Bar.at(dev_ms, 5.0), chart_confirmed=False)
    assert out == [10.0, 15.0]

    # Re-tick (+3): must restore to baseline 10 first -> 13, NOT 15+3=18.
    out = c.process_round(T, PERIOD_END, [], _Bar.at(dev_ms, 3.0), chart_confirmed=False)
    assert out == [10.0, 13.0]

    # Finalize (+4 confirmed): restore to 10 -> 14, tail becomes confirmed 14.
    out = c.process_round(T, PERIOD_END, [_Bar.at(dev_ms, 4.0)], None, chart_confirmed=False)
    assert out == [10.0, 14.0]
    assert not c.window.has_developing
    assert h.acc == 14.0                # new baseline reflects the confirmed close


def __test_new_developing_run_does_not_save_baseline__():
    """A fresh developing intrabar runs against the prior baseline; no save after."""
    h = _Harness()
    c = h.collector()

    # One closed intrabar: run + save.
    c.process_round(T, PERIOD_END, [_Bar.at(T, 1.0)], None, chart_confirmed=False)
    saves_after_closed = h.saves
    # Developing intrabar: run only, no extra save.
    c.process_round(T, PERIOD_END, [], _Bar.at(T + LTF, 2.0), chart_confirmed=False)
    assert h.saves == saves_after_closed


def __test_empty_developing_round_publishes_empty__():
    """A developing chart bar with no intrabars yet publishes an empty array."""
    h = _Harness()
    c = h.collector()
    out = c.process_round(T, PERIOD_END, [], None, chart_confirmed=False)
    assert out == []
    assert c.bar_index == 0
    assert h.runs == []


def __test_published_array_is_independent_copy__():
    """Mutating the returned array must not corrupt the collector's window."""
    h = _Harness()
    c = h.collector()
    out = c.process_round(T, PERIOD_END, [_Bar.at(T, 1.0)], _Bar.at(T + LTF, 2.0),
                          chart_confirmed=False)
    out.append(999.0)
    out[0] = -1.0
    assert c.window.publish() == [1.0, 2.0]


# ── Reordered feed: provisional rollover + late close ─────────────────────────
# A live feed can deliver intrabar N+1's forming tick ahead of N's late close
# (cTrader closes a slot on the next slot's open, or plain stream reordering).
# The array must still grow when N+1 starts (N held provisionally at its last
# developing value, same bar_index) and N's late close must REPLACE that value at
# the same index — never append a new one. A late close that shifts the baseline
# replays the still-pending provisional chain + developing tail so var/ta state
# is never poisoned by a provisional whose confirmed close later differs.


def __test_race_promotes_developing_to_provisional__():
    """N+1 starts before N closes: the array grows and N is held provisionally."""
    h = _Harness()
    c = h.collector()
    n_ms = T + LTF          # intrabar N (idx 1)
    n1_ms = T + 2 * LTF     # intrabar N+1 (idx 2)

    # idx0 closed, N developing.
    out1 = c.process_round(T, PERIOD_END, [_Bar.at(T, 1.0)], _Bar.at(n_ms, 2.0),
                           chart_confirmed=False)
    assert out1 == [1.0, 2.0]

    # RACE: N+1 forms before N's close arrives -> N promoted to provisional, the
    # array grows to 3, N's last developing value (2.0) preserved at its slot.
    out2 = c.process_round(T, PERIOD_END, [], _Bar.at(n1_ms, 3.0),
                           chart_confirmed=False)
    assert out2 == [1.0, 2.0, 3.0]

    # N's late close (2.5, differs from its 2.0 dev tick) finalizes N in place.
    out3 = c.process_round(T, PERIOD_END, [_Bar.at(n_ms, 2.5)], _Bar.at(n1_ms, 3.5),
                           chart_confirmed=False)
    assert out3 == [1.0, 2.5, 3.5]
    # The late close reused N's original index (1); no new index was allocated.
    assert c.bar_index == 2
    assert max(r[0] for r in h.runs) == 2


def __test_provisional_replay_is_not_islast__():
    """A rolled-over provisional is replayed with ``islast=False``; only the live
    developing tail is ``islast=True`` (``barstate.islast`` must not flag a
    middle-of-array provisional after a newer intrabar has started)."""
    h = _Harness()
    c = h.collector()
    n_ms = T + LTF          # intrabar N (idx 1)
    n1_ms = T + 2 * LTF     # intrabar N+1 (idx 2)

    c.process_round(T, PERIOD_END, [_Bar.at(T, 1.0)], _Bar.at(n_ms, 2.0),
                    chart_confirmed=False)
    # RACE: N+1 starts before N closes -> N is promoted to provisional and
    # replayed at idx 1, N+1 runs as the live developing tail at idx 2.
    h.islasts.clear()
    out = c.process_round(T, PERIOD_END, [], _Bar.at(n1_ms, 3.0),
                          chart_confirmed=False)
    assert out == [1.0, 2.0, 3.0]
    flags = dict(h.islasts)
    assert flags[1] is False           # provisional N is NOT the live last bar
    assert flags[2] is True            # only the developing tail N+1 is islast

    # A re-tick of the developing tail still replays the provisional as non-last.
    h.islasts.clear()
    c.process_round(T, PERIOD_END, [], _Bar.at(n1_ms, 3.5), chart_confirmed=False)
    flags = dict(h.islasts)
    assert flags[1] is False
    assert flags[2] is True


def __test_race_chain_replay_recomputes_var_state__():
    """A late close that differs from the dev tick propagates to later intrabars."""
    h = _AccHarness()                  # run does acc += value; value depends on state
    c = h.collector()
    n_ms = T + LTF
    n1_ms = T + 2 * LTF

    # idx0 (+10) -> acc 10, baseline 10. N developing (+5) -> tail 15.
    out1 = c.process_round(T, PERIOD_END, [_Bar.at(T, 10.0)], _Bar.at(n_ms, 5.0),
                           chart_confirmed=False)
    assert out1 == [10.0, 15.0]

    # RACE: N+1 (+7) runs on top of N's provisional state -> 15 + 7 = 22.
    out2 = c.process_round(T, PERIOD_END, [], _Bar.at(n1_ms, 7.0),
                           chart_confirmed=False)
    assert out2 == [10.0, 15.0, 22.0]

    # N's close is +6 (differs from the +5 dev tick): N becomes 10 + 6 = 16 and
    # the replay recomputes N+1 on the corrected baseline -> 16 + 7 = 23 (NOT 22).
    out3 = c.process_round(T, PERIOD_END, [_Bar.at(n_ms, 6.0)], _Bar.at(n1_ms, 7.0),
                           chart_confirmed=False)
    assert out3 == [10.0, 16.0, 23.0]


def __test_two_provisionals_then_closes_in_order__():
    """Two intrabars roll over before either closes; both finalize in order."""
    h = _Harness()
    c = h.collector()
    n_ms = T + LTF          # idx 1
    n1_ms = T + 2 * LTF     # idx 2
    n2_ms = T + 3 * LTF     # idx 3

    c.process_round(T, PERIOD_END, [_Bar.at(T, 1.0)], _Bar.at(n_ms, 2.0),
                    chart_confirmed=False)
    c.process_round(T, PERIOD_END, [], _Bar.at(n1_ms, 3.0), chart_confirmed=False)
    # Both N and N+1 are now provisional while N+2 develops.
    out = c.process_round(T, PERIOD_END, [], _Bar.at(n2_ms, 4.0), chart_confirmed=False)
    assert out == [1.0, 2.0, 3.0, 4.0]

    # N's late close finalizes the front of the chain.
    out = c.process_round(T, PERIOD_END, [_Bar.at(n_ms, 2.5)], _Bar.at(n2_ms, 4.5),
                          chart_confirmed=False)
    assert out == [1.0, 2.5, 3.0, 4.5]

    # N+1's late close finalizes the next one.
    out = c.process_round(T, PERIOD_END, [_Bar.at(n1_ms, 3.5)], _Bar.at(n2_ms, 4.7),
                          chart_confirmed=False)
    assert out == [1.0, 2.5, 3.5, 4.7]
    assert c.bar_index == 3            # four distinct indices; no extras for closes


def __test_provisional_close_out_of_order_is_buffered__():
    """A close for a later provisional waits until the chain head's close lands."""
    h = _Harness()
    c = h.collector()
    n_ms = T + LTF          # idx 1
    n1_ms = T + 2 * LTF     # idx 2
    n2_ms = T + 3 * LTF     # idx 3

    c.process_round(T, PERIOD_END, [_Bar.at(T, 1.0)], _Bar.at(n_ms, 2.0),
                    chart_confirmed=False)
    c.process_round(T, PERIOD_END, [], _Bar.at(n1_ms, 3.0), chart_confirmed=False)
    c.process_round(T, PERIOD_END, [], _Bar.at(n2_ms, 4.0), chart_confirmed=False)

    # N+1's close arrives before N's (out of order): it must be buffered, not
    # finalized — N+1 stays provisional at 3.0 because the head (N) is still open.
    out = c.process_round(T, PERIOD_END, [_Bar.at(n1_ms, 3.5)], _Bar.at(n2_ms, 4.5),
                          chart_confirmed=False)
    assert out == [1.0, 2.0, 3.0, 4.5]

    # N's close lands: the chain drains front-to-back, applying both buffered
    # closes in timestamp order.
    out = c.process_round(T, PERIOD_END, [_Bar.at(n_ms, 2.5)], _Bar.at(n2_ms, 4.7),
                          chart_confirmed=False)
    assert out == [1.0, 2.5, 3.5, 4.7]
    assert c.bar_index == 3


def __test_confirmed_round_drops_pending_and_developing__():
    """A grace timeout at chart close drops unconfirmed provisional + developing."""
    h = _AccHarness()
    c = h.collector()
    n_ms = T + LTF
    n1_ms = T + 2 * LTF

    c.process_round(T, PERIOD_END, [_Bar.at(T, 10.0)], _Bar.at(n_ms, 5.0),
                    chart_confirmed=False)
    # RACE: N provisional, N+1 developing; neither close has arrived.
    out = c.process_round(T, PERIOD_END, [], _Bar.at(n1_ms, 7.0), chart_confirmed=False)
    assert out == [10.0, 15.0, 22.0]

    # Chart confirms with both tails unconfirmed: the confirmed array is the
    # closed prefix only, and the baseline is restored so the next period does
    # not compound from the abandoned ticks.
    out = c.process_round(T, PERIOD_END, [], None, chart_confirmed=True)
    assert out == [10.0]
    assert not c.window.has_developing
    assert not c.window.has_provisional
    assert h.acc == 10.0               # restored to the post-confirmed baseline


def __test_full_drain_no_developing_does_not_trailing_restore__():
    """Draining the whole chain with no developing tail leaves no spurious restore.

    When buffered late closes finalize the *entire* provisional chain and no
    developing tick is present, there is nothing left to replay. A bare rebuild
    here would only ``restore`` — and in production that restore also calls
    ``instance_state.reset()``, wiping the function-instance (``ta.*``-in-function)
    state the last confirmed run just rebuilt, with nothing re-running to restore
    it. The drained round must therefore end on the confirmed ``save``, not a
    trailing ``restore``.
    """
    h = _Harness()
    c = h.collector()
    n_ms = T + LTF          # idx 1
    n1_ms = T + 2 * LTF     # idx 2
    n2_ms = T + 3 * LTF     # idx 3

    c.process_round(T, PERIOD_END, [_Bar.at(T, 1.0)], _Bar.at(n_ms, 2.0),
                    chart_confirmed=False)
    c.process_round(T, PERIOD_END, [], _Bar.at(n1_ms, 3.0), chart_confirmed=False)
    # N and N+1 are provisional while N+2 develops.
    c.process_round(T, PERIOD_END, [], _Bar.at(n2_ms, 4.0), chart_confirmed=False)

    # All three closes arrive together and no developing tick is present: the
    # whole chain finalizes and nothing remains to rebuild.
    h.ops.clear()
    out = c.process_round(
        T, PERIOD_END,
        [_Bar.at(n_ms, 2.5), _Bar.at(n1_ms, 3.5), _Bar.at(n2_ms, 4.5)],
        None, chart_confirmed=False,
    )
    assert out == [1.0, 2.5, 3.5, 4.5]
    assert h.ops[-1] == "save"          # ends on the confirmed save, not a restore
    # No restore is left dangling after the last confirmed run rebuilt the state.
    last_run = max(i for i, op in enumerate(h.ops) if op == "run")
    assert "restore" not in h.ops[last_run + 1:]
    assert not c.window.has_provisional
    assert not c.window.has_developing


def __test_late_close_batched_before_newer_closed_bar_drains_first__():
    """A pending late close batched ahead of a newer closed bar drains first.

    The streamer can batch a pending provisional's late close together with a
    newer, never-developing closed intrabar in the SAME (time-ordered) round.
    The newer bar must NOT run-and-save on the stale pre-close baseline and land
    in the closed segment ahead of the still-provisional chain; the buffered late
    close has to finalize first so the baseline advances and ordering stays in
    timestamp order. Modeled with the accumulator so the baseline corruption is
    observable as a wrong running value.
    """
    h = _AccHarness()
    c = h.collector()
    n_ms = T + LTF          # idx 1 (N)
    n1_ms = T + 2 * LTF     # idx 2 (N+1, developing)
    b_ms = T + 3 * LTF      # idx 3 (B, brand-new closed bar, newer than N)

    c.process_round(T, PERIOD_END, [_Bar.at(T, 1.0)], _Bar.at(n_ms, 2.0),
                    chart_confirmed=False)
    # N+1 develops -> N rolls over to provisional.
    c.process_round(T, PERIOD_END, [], _Bar.at(n1_ms, 3.0), chart_confirmed=False)
    assert c.window.publish() == [1.0, 3.0, 6.0]  # [closed, N prov, N+1 dev]

    # One round batches N's late close (2.5) ahead of a brand-new closed bar B
    # (4.0); N+1 is refreshed as the developing tail. N must finalize first.
    out = c.process_round(
        T, PERIOD_END,
        [_Bar.at(n_ms, 2.5), _Bar.at(b_ms, 4.0)],
        _Bar.at(n1_ms, 3.0),
        chart_confirmed=False,
    )
    # Confirmed prefix is 1.0 + 2.5 (N) + 4.0 (B) = 7.5; developing tail adds 3.0.
    assert out == [1.0, 3.5, 7.5, 10.5]
    assert h.saved == 7.5             # baseline holds the confirmed prefix only
    assert not c.window.has_provisional


def __test_newer_closed_bar_deferred_while_provisional_gap_open__():
    """A newer closed bar must not jump ahead of an unresolved older provisional.

    A provider gap leaves an older provisional's close missing while a brand-new,
    newer closed bar arrives in the same period AND the developing tail between
    them gets no fresh tick (stalled). The newer bar cannot join the closed
    segment (it sits ahead of the still-pending chain in ``publish()``) nor
    advance the saved baseline past the unresolved earlier intrabar; both the
    stalled developing tail and the newer bar are held in the provisional chain,
    in timestamp order, until the closes ahead of them drain front-to-back.
    Modeled with the accumulator so a baseline that wrongly advanced past the gap,
    or an array reordered past the stalled tail, would surface as a wrong value.
    """
    h = _AccHarness()
    c = h.collector()
    n_ms = T + LTF          # idx 1 (N, the gap)
    n1_ms = T + 2 * LTF     # idx 2 (N+1, developing, then stalled)
    b_ms = T + 3 * LTF      # idx 3 (B, brand-new closed bar, newer than N+1)

    # idx0 (+1) -> acc 1, baseline 1. N developing (+2) -> tail 3.
    c.process_round(T, PERIOD_END, [_Bar.at(T, 1.0)], _Bar.at(n_ms, 2.0),
                    chart_confirmed=False)
    # N+1 develops -> N rolls over to provisional. acc: 1+2(N prov)+3(N+1)=6.
    out = c.process_round(T, PERIOD_END, [], _Bar.at(n1_ms, 3.0), chart_confirmed=False)
    assert out == [1.0, 3.0, 6.0]      # [closed, N prov, N+1 dev]

    # GAP: N's close never arrives; a brand-new closed bar B (+4) shows up with no
    # developing tick (N+1 is stalled). B must be deferred behind the older N+1,
    # NOT appended to the closed segment. The stalled N+1 is promoted to the chain
    # so the array stays in timestamp order: closed[1.0], prov[N=3.0, N+1=6.0,
    # B=6+4=10.0]; the baseline holds only the confirmed prefix (1.0).
    out = c.process_round(T, PERIOD_END, [_Bar.at(b_ms, 4.0)], None,
                          chart_confirmed=False)
    assert out == [1.0, 3.0, 6.0, 10.0]
    assert c.window.has_provisional
    assert h.saved == 1.0             # baseline did NOT advance past the gap

    # N's late close (+2 confirmed, same as its dev tick) lands: only the chain
    # head N finalizes (N+1's and B's closes are still missing), so N+1 and B stay
    # provisional and the array keeps timestamp order on the new baseline (3.0).
    out = c.process_round(T, PERIOD_END, [_Bar.at(n_ms, 2.0)], None,
                          chart_confirmed=False)
    assert out == [1.0, 3.0, 6.0, 10.0]   # N confirmed; N+1, B still provisional
    assert c.window.has_provisional
    assert h.saved == 3.0             # baseline advanced only past N

    # N+1's close (+3) then B's close (+4) arrive together: the chain drains
    # front-to-back, both become closed in timestamp order, and the baseline
    # reaches the full confirmed prefix.
    out = c.process_round(T, PERIOD_END, [_Bar.at(n1_ms, 3.0), _Bar.at(b_ms, 4.0)],
                          None, chart_confirmed=False)
    assert out == [1.0, 3.0, 6.0, 10.0]
    assert not c.window.has_provisional
    assert h.saved == 10.0            # baseline reflects the full confirmed prefix


def __test_newer_closed_bar_deferred_gap_dropped_at_chart_close__():
    """An unresolved gap plus its deferred newer bar drop at chart confirm."""
    h = _AccHarness()
    c = h.collector()
    n_ms = T + LTF
    n1_ms = T + 2 * LTF
    b_ms = T + 3 * LTF

    c.process_round(T, PERIOD_END, [_Bar.at(T, 1.0)], _Bar.at(n_ms, 2.0),
                    chart_confirmed=False)
    c.process_round(T, PERIOD_END, [], _Bar.at(n1_ms, 3.0), chart_confirmed=False)
    # GAP: N never closes; newer bar B is deferred behind it.
    c.process_round(T, PERIOD_END, [_Bar.at(b_ms, 4.0)], None, chart_confirmed=False)
    assert c.window.has_provisional

    # Chart confirms with the gap still open: the confirmed array is the closed
    # prefix only, and the baseline is restored to the post-confirmed state.
    out = c.process_round(T, PERIOD_END, [], None, chart_confirmed=True)
    assert out == [1.0]
    assert not c.window.has_provisional
    assert not c.window.has_developing
    assert h.acc == 1.0


def __test_newer_close_deferred_behind_stalled_developing_gap__():
    """A newer close must not jump ahead of a stalled (un-refreshed) developing.

    One round delivers the pending head's late close together with a brand-new,
    newer closed bar while the currently-developing intrabar gets NO fresh forming
    tick (``developing_bar`` is None) and never closes — a gap on the developing
    slot itself. Draining the head empties ``_pending``, so the existing
    provisional-gap guard no longer fires; without promoting the stalled developing
    first, the newer bar would fall through to the closed segment ahead of the
    older developing tail, publishing ``[..., N, N+2, N+1]`` out of timestamp order
    and advancing the saved baseline past the gap. The developing must instead be
    promoted to the provisional chain and the newer bar deferred behind it. Modeled
    with the accumulator so the out-of-order baseline surfaces as a wrong value.
    """
    h = _AccHarness()
    c = h.collector()
    n_ms = T + LTF          # idx 1 (N)
    n1_ms = T + 2 * LTF     # idx 2 (N+1, developing, then stalled)
    n2_ms = T + 3 * LTF     # idx 3 (N+2, brand-new closed bar, newer than N+1)

    # idx0 (+1) -> acc 1, baseline 1. N developing (+2) -> tail 3.
    c.process_round(T, PERIOD_END, [_Bar.at(T, 1.0)], _Bar.at(n_ms, 2.0),
                    chart_confirmed=False)
    # N+1 develops -> N rolls over to provisional. acc: 1+2(N prov)+3(N+1)=6.
    out = c.process_round(T, PERIOD_END, [], _Bar.at(n1_ms, 3.0), chart_confirmed=False)
    assert out == [1.0, 3.0, 6.0]      # [closed, N prov, N+1 dev]

    # One round batches N's late close (2.5) ahead of a brand-new closed bar N+2
    # (+4); N+1 gets NO forming tick and no close (stalled). After N drains and
    # ``_pending`` empties, N+1 (older, unconfirmed) is promoted to provisional and
    # N+2 deferred behind it -> closed[1.0+2.5=3.5], prov[N+1=3.5+3=6.5,
    # N+2=6.5+4=10.5]. The baseline holds only the confirmed N prefix (3.5), NOT
    # past the N+1 gap.
    out = c.process_round(
        T, PERIOD_END,
        [_Bar.at(n_ms, 2.5), _Bar.at(n2_ms, 4.0)],
        None, chart_confirmed=False,
    )
    assert out == [1.0, 3.5, 6.5, 10.5]
    assert c.window.has_provisional
    assert not c.window.has_developing
    assert h.saved == 3.5             # baseline did NOT advance past the N+1 gap

    # N+1's late close (+3) finally lands: the chain drains front-to-back -> N+1
    # then N+2 both become closed in timestamp order on the corrected baseline.
    out = c.process_round(T, PERIOD_END, [_Bar.at(n1_ms, 3.0)], None,
                          chart_confirmed=False)
    assert out == [1.0, 3.5, 6.5, 10.5]
    assert not c.window.has_provisional
    assert h.saved == 10.5            # full confirmed prefix persisted


# ── Real Series rollback under backward replay ────────────────────────────────
# The fake harnesses above model var state but not a SeriesImpl: a backward
# replay (re-running an earlier bar_index after a later one ran) would APPEND to
# a real series rather than overwrite, growing the circular buffer and corrupting
# ``close[1]``/``ta.*`` history. ``RootVarSnapshot`` excludes series slots, so the
# live LTF baseline pairs it with ``RootSeriesSnapshot``. This harness drives the
# collector with a REAL SeriesImpl in a root vector and the REAL series rollback,
# so the corruption (without rollback) and the fix (with it) are both observable.


class _RealSeriesHarness:
    """Collector callbacks backed by a real ``SeriesImpl`` in a root vector.

    ``run_intrabar`` adds the bar value to the series at ``lib.bar_index`` and
    returns the series' running sum, so a stray appended element (the corruption)
    changes both the published value and ``series._size``. ``roll_series=False``
    mirrors the pre-fix var-only baseline (proving the corruption); ``True`` adds
    the ``RootSeriesSnapshot`` rollback (the fix).
    """

    _ROOT = "test_ltf_series_rollback_root"
    _LAYOUT = {'init': (None,), 'series': [(0, 500)], 'varip': (),
               'children': (), 'names': ('hist',)}

    def __init__(self, *, roll_series: bool):
        from pynecore.core import instance_state
        self._is = instance_state
        self._roll_series = roll_series
        self._state = instance_state.create_root(self._ROOT, dict(self._LAYOUT))
        self._series = self._state[0]
        self._snap = instance_state.RootSeriesSnapshot([self._ROOT])

    @property
    def series(self):
        return self._series

    def _series_sum(self) -> float:
        return sum(float(self._series[k]) for k in range(len(self._series)))

    def run_intrabar(self, bar, bar_index, _confirmed, _is_new, _islast):
        from pynecore import lib
        lib.bar_index = bar_index
        self._series.add(float(bar.value))
        return self._series_sum()

    def save_baseline(self):
        self._snap.save()

    def restore_baseline(self):
        if self._roll_series:
            self._snap.restore()
        self._is.reset()

    def collector(self) -> LiveLtfCollector:
        return LiveLtfCollector(self.run_intrabar, self.save_baseline,
                                self.restore_baseline)

    def cleanup(self):
        self._is.discard_root(self._ROOT)


def _drive_series_rollback_race(h: _RealSeriesHarness) -> list[list[float]]:
    """Run the canonical late-close race and return the published array per round."""
    c = h.collector()
    n_ms = T + LTF          # intrabar N (idx 1)
    n1_ms = T + 2 * LTF     # intrabar N+1 (idx 2)
    rows = [
        c.process_round(T, PERIOD_END, [_Bar.at(T, 1.0)], _Bar.at(n_ms, 2.0),
                        chart_confirmed=False),
        # RACE: N+1 forms before N's close -> N promoted, chain replayed.
        c.process_round(T, PERIOD_END, [], _Bar.at(n1_ms, 3.0),
                        chart_confirmed=False),
        # N's late close (2.5) finalizes N in place; N+1 re-tick on the new baseline.
        c.process_round(T, PERIOD_END, [_Bar.at(n_ms, 2.5)], _Bar.at(n1_ms, 3.0),
                        chart_confirmed=False),
    ]
    return rows


def __test_real_series_rollback_keeps_buffer_intact__():
    """With series rollback the backward replay rebuilds the buffer cleanly."""
    h = _RealSeriesHarness(roll_series=True)
    try:
        rows = _drive_series_rollback_race(h)
        # Sums: idx0=[1]; N dev=[1,2]; N+1 dev rebuilt=[1,2,3]; N close=[1,2.5];
        # N+1 re-tick=[1,2.5,3].
        assert rows[0] == [1.0, 3.0]
        assert rows[1] == [1.0, 3.0, 6.0]
        assert rows[2] == [1.0, 3.5, 6.5]
        # Three distinct intrabars -> exactly three buffer elements, no growth.
        assert len(h.series) == 3
    finally:
        h.cleanup()


def __test_without_series_rollback_buffer_corrupts__():
    """Guard: var-only restore (the pre-fix path) grows the buffer on replay.

    This proves the rollback is load-bearing — drop it and the same race leaves a
    stray appended element behind, which is exactly what the production fix
    (``RootSeriesSnapshot`` in the LTF baseline) prevents.
    """
    h = _RealSeriesHarness(roll_series=False)
    try:
        _drive_series_rollback_race(h)
        # The late close re-ran idx 1 while the buffer's last_bar_index was 2, so
        # add() appended instead of overwriting: four elements for three intrabars.
        assert len(h.series) > 3
    finally:
        h.cleanup()
