"""
Unit tests for the dayofweek() function and its named constants.

Regression coverage for Issue #56: the named constants
(dayofweek.sunday ... dayofweek.saturday) must match the values returned by
dayofweek(time), so that `dayofweek(time) == dayofweek.friday` evaluates as
expected. Pine Script spec: sunday=1, monday=2, tuesday=3, wednesday=4,
thursday=5, friday=6, saturday=7.
"""

from datetime import datetime, timezone

from pynecore.lib import dayofweek


def __test_dayofweek_constants_match_pine_spec__():
    """Pine Script numbers weekdays sunday=1 ... saturday=7."""
    assert int(dayofweek.sunday) == 1
    assert int(dayofweek.monday) == 2
    assert int(dayofweek.tuesday) == 3
    assert int(dayofweek.wednesday) == 4
    assert int(dayofweek.thursday) == 5
    assert int(dayofweek.friday) == 6
    assert int(dayofweek.saturday) == 7


def __test_dayofweek_function_matches_named_constant_for_every_weekday__():
    """dayofweek(time) must compare equal to the matching dayofweek.<day> constant."""
    cases = [
        ("2024-06-09", dayofweek.sunday, "Sun"),
        ("2024-06-10", dayofweek.monday, "Mon"),
        ("2024-06-11", dayofweek.tuesday, "Tue"),
        ("2024-06-12", dayofweek.wednesday, "Wed"),
        ("2024-06-13", dayofweek.thursday, "Thu"),
        ("2024-06-14", dayofweek.friday, "Fri"),
        ("2024-06-15", dayofweek.saturday, "Sat"),
    ]
    for iso_date, expected_const, label in cases:
        ts_ms = int(datetime.fromisoformat(iso_date).replace(tzinfo=timezone.utc).timestamp() * 1000)
        assert dayofweek.dayofweek(ts_ms) == expected_const, f"{label} ({iso_date}) failed"


def __test_issue_56_friday_and_saturday_not_swapped__():
    """Reporter's exact bars: Friday must match dayofweek.friday, Saturday must match dayofweek.saturday."""
    ts_fri = int(datetime(2024, 6, 14, 23, 30, tzinfo=timezone.utc).timestamp() * 1000)
    assert dayofweek.dayofweek(ts_fri) == dayofweek.friday
    assert dayofweek.dayofweek(ts_fri) != dayofweek.saturday

    ts_sat = int(datetime(2024, 9, 14, 23, 30, tzinfo=timezone.utc).timestamp() * 1000)
    assert dayofweek.dayofweek(ts_sat) == dayofweek.saturday
    assert dayofweek.dayofweek(ts_sat) != dayofweek.friday
