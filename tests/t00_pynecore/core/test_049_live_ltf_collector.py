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
        self.ops: list[str] = []
        self.saves = 0
        self.restores = 0

    def run_intrabar(self, bar, bar_index, confirmed, is_new):
        self.ops.append("run")
        self.runs.append((bar_index, confirmed, is_new, bar.value))
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

    def run_intrabar(self, bar, _bar_index, _confirmed, _is_new):
        self.acc += bar.value
        return self.acc

    def save_baseline(self):
        self._saved = self.acc
        self.saves += 1

    def restore_baseline(self):
        self.acc = self._saved
        self.restores += 1

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
