"""
Timezone resolution and missing-IANA-database handling (issue #59).

Covers ``core.datetime.parse_timezone`` diagnostics and the behaviour of
``lib.time`` / ``lib.time_close`` when an IANA timezone cannot be resolved
(e.g. Windows without the ``tzdata`` package): the failure must surface as an
actionable error instead of every bar silently becoming ``na`` (which made
``time(timeframe.period, session, "America/Chicago")`` produce 0 trades).
"""
from contextlib import contextmanager
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pytest

import pynecore.lib as lib
from pynecore.core import datetime as pcd
from pynecore.core.datetime import parse_timezone, TimezoneNotFoundError
from pynecore.types.na import NA

_REAL_ZONEINFO = ZoneInfo


@contextmanager
def _missing_tz_db(monkeypatch):
    """Simulate a system with no IANA timezone database (Windows w/o tzdata)."""

    def _no_db(key=None):
        raise ZoneInfoNotFoundError(f"No time zone found with key {key}")

    monkeypatch.setattr(pcd, "ZoneInfo", _no_db)
    pcd._timezone_db_available.cache_clear()
    parse_timezone.cache_clear()
    try:
        yield
    finally:
        monkeypatch.undo()
        pcd._timezone_db_available.cache_clear()
        parse_timezone.cache_clear()


def __test_valid_iana_and_offsets_resolve__():
    """Valid IANA names and UTC/GMT offset forms still resolve"""
    assert parse_timezone("America/Chicago").key == "America/Chicago"
    assert parse_timezone("UTC") is not None
    assert parse_timezone("UTC-5") is not None
    assert parse_timezone("GMT+0530") is not None
    assert parse_timezone("+05:30") is not None


def __test_unknown_name_raises_clear_error__():
    """An unresolvable name (DB present) reports 'Unknown timezone', not a tzdata hint"""
    with pytest.raises(TimezoneNotFoundError, match="Unknown timezone"):
        parse_timezone("Not/AZone")
    # PST/CST are not IANA zone names; previously these wrongly hit the tzdata hint
    with pytest.raises(TimezoneNotFoundError, match="Unknown timezone"):
        parse_timezone("PST")


def __test_missing_db_raises_install_hint__(monkeypatch):
    """A missing IANA database yields an actionable 'pip install tzdata' message"""
    with _missing_tz_db(monkeypatch):
        with pytest.raises(TimezoneNotFoundError) as exc:
            parse_timezone("America/Chicago")
    msg = str(exc.value)
    assert "pip install tzdata" in msg
    assert "America/Chicago" in msg


def __test_time_session_surfaces_missing_tz_db__(monkeypatch):
    """issue #59: time(session, tz) must raise, not silently return na, when tz unresolved"""
    chi = _REAL_ZONEINFO("America/Chicago")
    lib.syminfo.timezone = "America/New_York"
    lib._time = int(datetime(2025, 7, 1, 10, 0, tzinfo=chi).timestamp() * 1000)
    with _missing_tz_db(monkeypatch):
        with pytest.raises(TimezoneNotFoundError):
            lib.time("15", "0900-1600", "America/Chicago")


def __test_time_close_session_surfaces_missing_tz_db__(monkeypatch):
    """time_close(session, tz) surfaces the same configuration error as time()"""
    chi = _REAL_ZONEINFO("America/Chicago")
    lib.syminfo.timezone = "America/New_York"
    lib._time = int(datetime(2025, 7, 1, 10, 0, tzinfo=chi).timestamp() * 1000)
    with _missing_tz_db(monkeypatch):
        with pytest.raises(TimezoneNotFoundError):
            lib.time_close("15", "0900-1600", "America/Chicago")


def __test_time_session_in_and_out_unchanged__():
    """With a working DB, in-session returns a time and out-of-session returns na"""
    chi = _REAL_ZONEINFO("America/Chicago")
    lib.syminfo.timezone = "America/New_York"
    lib._time = int(datetime(2025, 7, 1, 10, 0, tzinfo=chi).timestamp() * 1000)  # Tue RTH
    assert not isinstance(lib.time("15", "0900-1600", "America/Chicago"), NA)
    lib._time = int(datetime(2025, 7, 1, 3, 0, tzinfo=chi).timestamp() * 1000)  # pre-open
    assert isinstance(lib.time("15", "0900-1600", "America/Chicago"), NA)
