"""
Tests for ``pyne run`` provider-mode bar-count coverage reporting.

When a ``-N bars`` download cannot reach the requested count, the retry loop
must localize the shortfall precisely: distinguish a venue history-horizon
limit from in-session holes (incomplete pagination / omitted ticks) and list
the exact missing intervals, instead of a vague sparse-coverage count.
"""
from datetime import datetime, time, UTC
from types import SimpleNamespace

from pynecore.cli.commands.run import (
    _missing_slots,
    _classify_missing_slots,
    _merge_intervals,
    _format_missing_report,
)
from pynecore.core.syminfo import SymInfoInterval


TF = 60  # one-minute grid


def _ts(y, mo, d, h, mi) -> int:
    return int(datetime(y, mo, d, h, mi, tzinfo=UTC).timestamp())


def __test_missing_slots_finds_interior_hole__():
    """A single absent slot on the real-bar grid is reported, aligned to phase."""
    anchor = _ts(2025, 1, 6, 10, 4)
    real = [anchor - 4 * TF, anchor - 3 * TF, anchor - 1 * TF, anchor]
    missing = _missing_slots(real, anchor - 4 * TF, anchor + TF, TF)
    assert missing == [anchor - 2 * TF]


def __test_missing_slots_empty_without_real_bars__():
    """No real bars → grid phase unknown → nothing reported."""
    assert _missing_slots([], 0, 1000, TF) == []


def __test_classify_24x7_all_in_session__():
    """A symbol with no opening_hours treats every hole as an anomaly."""
    syminfo = SimpleNamespace(opening_hours=[], timezone="UTC")
    slots = [_ts(2025, 1, 6, 3, 0), _ts(2025, 1, 6, 10, 0)]
    in_session, closed = _classify_missing_slots(slots, syminfo, TF)
    assert in_session == slots
    assert closed == []


def __test_classify_splits_by_session__():
    """Holes inside a session are anomalies; out-of-hours holes are expected."""
    # Monday 09:00-17:00 UTC session.
    syminfo = SimpleNamespace(
        opening_hours=[SymInfoInterval(day=0, start=time(9, 0), end=time(17, 0))],
        timezone="UTC",
    )
    in_slot = _ts(2025, 1, 6, 10, 0)   # Monday 10:00 — open
    out_slot = _ts(2025, 1, 6, 3, 0)   # Monday 03:00 — closed
    in_session, closed = _classify_missing_slots([out_slot, in_slot], syminfo, TF)
    assert in_session == [in_slot]
    assert closed == [out_slot]


def __test_merge_intervals_coalesces_runs__():
    """Consecutive grid slots collapse into half-open ``[start, end)`` runs."""
    base = _ts(2025, 1, 6, 10, 0)
    slots = [base, base + TF, base + 5 * TF]
    assert _merge_intervals(slots, TF) == [
        (base, base + 2 * TF),
        (base + 5 * TF, base + 6 * TF),
    ]


def __test_format_reports_horizon__():
    """The horizon case names the oldest available bar and the count deficit."""
    oldest = _ts(2025, 1, 6, 9, 0)
    msg = _format_missing_report(
        bar_count=1500, real_bars=1473, oldest_ts=oldest,
        in_session=[], closed=[], tf_seconds=TF, horizon_reached=True,
    )
    assert "1500" in msg and "1473" in msg
    assert "horizon" in msg.lower()
    assert "2025-01-06 09:00" in msg


def __test_format_reports_in_session_gap__():
    """In-session holes surface as a distinct, exact-interval anomaly line."""
    slot = _ts(2025, 1, 6, 10, 0)
    msg = _format_missing_report(
        bar_count=1500, real_bars=1499, oldest_ts=slot,
        in_session=[slot], closed=[], tf_seconds=TF, horizon_reached=False,
    )
    assert "in-session" in msg.lower()
    assert "2025-01-06 10:00" in msg
