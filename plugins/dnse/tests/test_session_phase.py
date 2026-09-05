"""#70 repro — the session-phase classifier is holiday-blind.

OBSERVED live 2026-08-31 (exchange holiday, a Monday): ``venue.py status`` /
the L0 gate reported ``phase: continuous`` — the classifier is time-of-day
only. Fails safe at the venue (a closed exchange rejects writes) but
misinforms the gate every live tool trusts.

The module lives in the testing tree (pytest never imports testing/), so it
is loaded by path here.
"""
import importlib.util
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

_ICT = timezone(timedelta(hours=7))
_L0 = (Path(__file__).resolve().parents[1] / "testing" / "live_test"
       / "level0_venue_semantics" / "l0_order_semantics.py")


def _load_l0():
    spec = importlib.util.spec_from_file_location("_l0_under_test", _L0)
    module = importlib.util.module_from_spec(spec)
    # The module imports its live_test siblings at import time only if it
    # reaches into them lazily; add its dirs to the path defensively.
    sys.path.insert(0, str(_L0.parent))
    sys.path.insert(0, str(_L0.parent.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(_L0.parent))
        sys.path.remove(str(_L0.parent.parent))
    return module


def __test_measured_holiday_is_closed__():
    """RED on the unmodified tree: 2026-08-31 10:00 ICT (a Monday inside
    continuous hours, measured exchange holiday) must classify 'closed'."""
    l0 = _load_l0()
    when = datetime(2026, 8, 31, 10, 0, tzinfo=_ICT)
    assert l0.session_phase(when) == "closed", (
        "2026-08-31 was a measured exchange holiday and the classifier "
        "reported the venue open (#70)")


def __test_statutory_fixed_holidays_are_closed__():
    """VN Labour Code fixed-date holidays, on weekdays, inside trading hours."""
    l0 = _load_l0()
    for when in (datetime(2026, 1, 1, 10, 0, tzinfo=_ICT),    # Thu, New Year
                 datetime(2026, 4, 30, 10, 0, tzinfo=_ICT),   # Thu, Reunification
                 datetime(2026, 5, 1, 10, 0, tzinfo=_ICT),    # Fri, Labour Day
                 datetime(2026, 9, 2, 10, 0, tzinfo=_ICT)):   # Wed, National Day
        assert l0.session_phase(when) == "closed", f"{when:%Y-%m-%d} must be closed"


def __test_ordinary_weekday_phases_unchanged__():
    """Control: the four measured phases on a plain trading day survive."""
    l0 = _load_l0()
    day = dict(year=2026, month=9, day=4, tzinfo=_ICT)        # Fri, no holiday
    assert l0.session_phase(datetime(hour=10, minute=0, **day)) == "continuous"
    assert l0.session_phase(datetime(hour=12, minute=0, **day)) == "lunch"
    assert l0.session_phase(datetime(hour=14, minute=35, **day)) == "atc"
    assert l0.session_phase(datetime(hour=20, minute=0, **day)) == "closed"
    assert l0.session_phase(datetime(2026, 9, 6, 10, 0, tzinfo=_ICT)) == "closed"  # Sunday
