"""
``SymInfo`` keeps a symbol's extended trading hours and which hours its bars follow.
"""
from datetime import date, time
from pathlib import Path

from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession

WEEKDAYS = (0, 1, 2, 3, 4)
EARLY_CLOSE = date(2024, 11, 29)


def __test_helper_aapl(session: str) -> SymInfo:
    """NASDAQ:AAPL with regular and extended hours and one early close"""
    return SymInfo(
        prefix="NASDAQ", description="Apple", ticker="AAPL", currency="USD", period="60",
        type="stock", mintick=0.01, pricescale=100, pointvalue=1.0, mincontract=1.0,
        timezone="America/New_York", session=session,
        opening_hours=[SymInfoInterval(day=d, start=time(9, 30), end=time(16)) for d in WEEKDAYS],
        session_starts=[SymInfoSession(day=d, time=time(9, 30)) for d in WEEKDAYS],
        session_ends=[SymInfoSession(day=d, time=time(16)) for d in WEEKDAYS],
        session_corrections={EARLY_CLOSE: (SymInfoInterval(day=4, start=time(9, 30), end=time(13)),)},
        extended_hours=[SymInfoInterval(day=d, start=time(4), end=time(20)) for d in WEEKDAYS],
        extended_session_corrections={EARLY_CLOSE: (SymInfoInterval(day=4, start=time(4), end=time(17)),)},
    )


def __test_extended_hours_round_trip__(tmp_path: Path):
    """ The extended template and its exceptions survive save and load """
    path = tmp_path / "aapl.toml"
    __test_helper_aapl("regular").save_toml(path)
    text = path.read_text()
    assert 'session = "regular"' in text
    assert "[[extended_hours]]" in text and "[[extended_session_corrections]]" in text

    loaded = SymInfo.load_toml(path)
    assert loaded.session == "regular"
    assert loaded.regular_hours is None
    assert loaded.opening_hours[0] == SymInfoInterval(day=0, start=time(9, 30), end=time(16))
    assert loaded.extended_hours == [SymInfoInterval(day=d, start=time(4), end=time(20)) for d in WEEKDAYS]
    assert loaded.extended_session_corrections == {
        EARLY_CLOSE: (SymInfoInterval(day=4, start=time(4), end=time(17)),)}
    assert loaded.session_corrections == {
        EARLY_CLOSE: (SymInfoInterval(day=4, start=time(9, 30), end=time(13)),)}


def __test_extended_bars_follow_the_extended_hours__(tmp_path: Path):
    """ Loaded as "extended", the own session is the extended template, the regular one
    is kept aside, and a save writes both back unchanged """
    path = tmp_path / "aapl.toml"
    __test_helper_aapl("extended").save_toml(path)
    loaded = SymInfo.load_toml(path)
    assert loaded.session == "extended"
    assert loaded.opening_hours == [SymInfoInterval(day=d, start=time(4), end=time(20)) for d in WEEKDAYS]
    assert loaded.session_starts == [SymInfoSession(day=d, time=time(4)) for d in WEEKDAYS]
    assert loaded.session_ends == [SymInfoSession(day=d, time=time(20)) for d in WEEKDAYS]
    assert loaded.session_corrections == {
        EARLY_CLOSE: (SymInfoInterval(day=4, start=time(4), end=time(17)),)}
    assert loaded.regular_hours == [SymInfoInterval(day=d, start=time(9, 30), end=time(16)) for d in WEEKDAYS]
    assert loaded.regular_session_corrections == {
        EARLY_CLOSE: (SymInfoInterval(day=4, start=time(9, 30), end=time(13)),)}

    again = tmp_path / "again.toml"
    loaded.save_toml(again)
    assert again.read_text() == path.read_text()
    assert SymInfo.load_toml(again) == loaded


def __test_a_symbol_without_extended_hours_stays_regular__(tmp_path: Path):
    """ "extended" without an extended template changes nothing """
    si = __test_helper_aapl("extended")
    si.extended_hours = []
    si.extended_session_corrections = {}
    path = tmp_path / "aapl.toml"
    si.save_toml(path)
    loaded = SymInfo.load_toml(path)
    assert loaded.regular_hours is None
    assert loaded.opening_hours[0] == SymInfoInterval(day=0, start=time(9, 30), end=time(16))
