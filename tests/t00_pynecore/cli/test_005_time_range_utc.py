"""
Regression tests for :func:`pynecore.cli.commands.run._parse_time_value`.

A ``--from`` / ``--to`` date without a UTC offset is UTC, as the CLI help and
docs state. A naive datetime would be read in the host's local time zone when
its timestamp is taken, so the same command selected a different bar window on
machines in different time zones.
"""
import os
import time
from datetime import UTC, datetime, timedelta

import pytest

from pynecore.cli.commands.run import _parse_time_value


@pytest.fixture
def __test_helper_non_utc_host__(monkeypatch):
    """Run the test with a host time zone that is not UTC."""
    monkeypatch.setenv('TZ', 'America/New_York')
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def __test_date_without_offset_is_utc__(__test_helper_non_utc_host__):
    """A bare date is midnight UTC, not midnight in the host's zone."""
    assert os.environ['TZ'] == 'America/New_York'
    parsed = _parse_time_value('2024-01-01')
    assert isinstance(parsed, datetime)
    assert parsed.tzinfo is UTC
    assert parsed.timestamp() == datetime(2024, 1, 1, tzinfo=UTC).timestamp()


def __test_datetime_without_offset_is_utc__(__test_helper_non_utc_host__):
    """A date with a time of day is UTC as well."""
    parsed = _parse_time_value('2024-01-01 12:30:00')
    assert isinstance(parsed, datetime)
    assert parsed.timestamp() == datetime(2024, 1, 1, 12, 30, tzinfo=UTC).timestamp()


def __test_explicit_offset_is_kept__(__test_helper_non_utc_host__):
    """An explicit offset keeps its own meaning."""
    parsed = _parse_time_value('2024-01-01T00:00:00+02:00')
    assert isinstance(parsed, datetime)
    assert parsed.utcoffset() == timedelta(hours=2)
    assert parsed.timestamp() == datetime(2023, 12, 31, 22, 0, tzinfo=UTC).timestamp()


def __test_days_back_is_utc__(__test_helper_non_utc_host__):
    """A days-back value counts back from the current UTC time."""
    before = datetime.now(UTC)
    parsed = _parse_time_value('3')
    assert isinstance(parsed, datetime)
    assert parsed.tzinfo is UTC
    delta = before - parsed
    assert timedelta(days=3) - timedelta(minutes=1) <= delta <= timedelta(days=3, minutes=1)
