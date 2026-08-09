"""Single-day calendar exceptions (``SymInfo.session_corrections``).

Exchanges shorten a handful of sessions every year — US equities close at 13:00
the day after Thanksgiving, on Christmas Eve and around Independence Day. A bar
opening in the final hour of such a day closes at that early close, so
``request.security`` publishes it to the chart one bar EARLIER than the regular
schedule would. Nothing in the bar data marks the day (its last bar is an
ordinary-looking stub), so the exception has to come from the exchange calendar.

Like ``test_013_dated_session_schedule``, these tests drive the public
``load_htf_bar_opens`` entry point and assert on the resulting
``state.bar_closes``; the per-bar confirmation walk over it is unchanged and
covered by ``test_012_gappy_intraday_htf``.
"""
from datetime import time, date, datetime
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")

# 2023-11-22 (Wed), 2023-11-24 (Fri, early close 13:00) and 2023-11-27 (Mon).
_REGULAR_HOURS = [9, 10, 11, 12, 13, 14, 15]
_EARLY_HOURS = [9, 10, 11, 12]
_FEED = {date(2023, 11, 22): _REGULAR_HOURS,
         date(2023, 11, 24): _EARLY_HOURS,
         date(2023, 11, 27): _REGULAR_HOURS}


def _ms(day, hour, minute=30):
    """Epoch ms of a 2023-11 New York wall-clock instant."""
    return int(datetime(2023, 11, day, hour, minute, tzinfo=NY).timestamp() * 1000)


def _syminfo(corrections):
    """A NASDAQ-like 09:30->16:00 equity ``SymInfo`` with the given corrections."""
    from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
    return SymInfo(
        prefix="NASDAQ", description="Equity", ticker="EQ", currency="USD", period="60",
        type="stock", mintick=0.01, pricescale=100, minmove=1, pointvalue=1.0,
        mincontract=1.0, timezone="America/New_York", volumetype="base",
        opening_hours=[SymInfoInterval(day=d, start=time(9, 30), end=time(16, 0))
                       for d in range(5)],
        session_starts=[SymInfoSession(day=d, time=time(9, 30)) for d in range(5)],
        session_ends=[SymInfoSession(day=d, time=time(16, 0)) for d in range(5)],
        session_corrections=corrections,
    )


def _load(tmp_dir, name, syminfo):
    """Write the 60-minute equity feed (+ its ``.toml``), run ``load_htf_bar_opens``."""
    from pynecore.core.ohlcv import OHLCVWriter
    from pynecore.core.resampler import Resampler
    from pynecore.core.security import SecurityState, load_htf_bar_opens
    from pynecore.types.ohlcv import OHLCV

    path = tmp_dir / f"{name}.ohlcv"
    with OHLCVWriter(path, "60") as w:
        for day, hours in _FEED.items():
            for hour in hours:
                w.write(OHLCV(timestamp=_ms(day.day, hour),
                              open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0))
    syminfo.save_toml(path.with_suffix(".toml"))
    state = SecurityState(sec_id="s", timeframe="60", gaps_on=False, same_timeframe=False,
                          resampler=Resampler.get_resampler("60"), tz=NY, is_ltf=False)
    load_htf_bar_opens(state, str(path))
    return state


def _expected(last_close_hour, last_close_minute):
    """Close instants of the feed, with 11-24's last bar ending where given."""
    closes = []
    for day, hours in _FEED.items():
        for hour in hours:
            if hour == hours[-1]:
                closes.append(_ms(day.day, 16, 0) if hours is _REGULAR_HOURS
                              else _ms(day.day, last_close_hour, last_close_minute))
            else:
                closes.append(_ms(day.day, hour + 1))
    return closes


def __test_session_correction_shortens_the_early_close_day__(log):
    """An early-close date confirms its last bar at the corrected session end.

    Without the correction the 12:30 bar of 2023-11-24 closes at 13:30 (its own
    period end, still inside the regular 16:00 session) and reaches the chart a
    bar late. With it the session ends at 13:00, and so does the bar.
    """
    import tempfile
    from pathlib import Path
    from pynecore.core.syminfo import SymInfoInterval

    early = {date(2023, 11, 24): (
        SymInfoInterval(day=4, start=time(9, 30), end=time(13, 0)),)}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        corrected = _load(tmp, "corrected", _syminfo(early))
        plain = _load(tmp, "plain", _syminfo({}))

    assert corrected.bar_closes == _expected(13, 0), \
        f"bar_closes={corrected.bar_closes}\nexpected={_expected(13, 0)}"
    assert plain.bar_closes == _expected(13, 30), \
        "an uncorrected symbol must keep its arithmetic period end"
    log.info("session correction: 2023-11-24 12:30 bar closes 13:00 instead of 13:30")


def __test_session_corrections_toml_round_trip__(log):
    """The calendar survives ``save_toml`` -> ``load_toml``, and rejects duplicates."""
    import tempfile
    from pathlib import Path
    from pynecore.core.syminfo import SymInfo, SymInfoInterval

    corrections = {
        date(2023, 11, 24): (SymInfoInterval(day=4, start=time(9, 30), end=time(13, 0)),),
        date(2023, 12, 25): (),  # closed all day
        date(2024, 7, 3): (SymInfoInterval(day=2, start=time(9, 30), end=time(13, 0)),),
    }
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "eq.toml"
        _syminfo(corrections).save_toml(path)
        assert SymInfo.load_toml(path).session_corrections == corrections

        # Dates sharing the same hours are written as ONE block, so a decade of
        # early closes stays a few lines.
        text = path.read_text(encoding="utf-8")
        assert text.count("[[session_corrections]]") == 2, text

        duplicated = text.replace("dates = [2023-12-25]", "dates = [2023-12-25, 2024-07-03]")
        path.write_text(duplicated, encoding="utf-8")
        try:
            SymInfo.load_toml(path)
        except ValueError as e:
            assert "2024-07-03" in str(e)
        else:
            raise AssertionError("a date listed twice must be rejected")
    log.info("session corrections round-trip: grouped, reloaded, duplicates rejected")
