"""Effective-dated session schedules (``SymInfo.session_schedules``).

These tests cover the date-aware session-schedule capability that lets a backtest
spanning an exchange's trading-hours change confirm each side with its own
schedule, instead of one static schedule mis-confirming the part of the range on
the other side of the change.

The behaviour under test is how ``state.bar_closes`` is COMPUTED. The per-bar
confirmation walk over it (``_get_confirmed_time``) is unchanged and covered by
``test_012_gappy_intraday_htf``, so these tests drive everything through the
public ``load_htf_bar_opens`` entry point and the public ``SymInfo`` API, and
assert on the resulting ``state.bar_closes``. They pin:

* Variant selection by exchange-local TRADING day, including the overnight
  off-by-one boundary (a night bar opening the evening before is confirmed against
  the next trading day's schedule).
* The single-consumer invariant (only a variant's ``opening_hours`` affects the
  close calculation) and the graceful fallback when a variant cannot describe its
  bars (``state.bar_closes`` stays ``None`` and the caller keeps the grid clamp).
* The ``SymInfo`` history model + TOML round-trip — sorting, duplicate rejection,
  ``effective_from`` normalization, history-wins resolution, and the commented
  example written when no history is present.
* ``load_htf_bar_opens`` dispatch — a history TOML drives the dated path and a
  no-history TOML reproduces the stale single-schedule behaviour the history fixes.
"""
from datetime import time, date, datetime
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")

# Two trading weeks straddling the 2026-01-12 (Mon) schedule change.
_SUN_FRI = [date(2026, 1, d) for d in (4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16)]  # Sat closed
_MON_FRI = [date(2026, 1, d) for d in (5, 6, 7, 8, 9, 12, 13, 14, 15, 16)]


def _ms(month, day, hour):
    """Epoch ms of a 2026 UTC wall-clock instant."""
    return int(datetime(2026, month, day, hour, 0, tzinfo=UTC).timestamp() * 1000)


def _opens_sec(days):
    """Epoch seconds of a 21:00 UTC night-session open on each given date."""
    return [int(datetime(d.year, d.month, d.day, 21, 0, tzinfo=UTC).timestamp()) for d in days]


def _night_variant(effective_from, end_t):
    """A midnight-crossing night session 21:00 -> ``end_t`` on every weekday."""
    from pynecore.core.syminfo import SymInfoInterval, SymInfoSession, SymInfoScheduleVariant
    return SymInfoScheduleVariant(
        effective_from=effective_from,
        opening_hours=[SymInfoInterval(day=d, start=time(21, 0), end=end_t) for d in range(7)],
        session_starts=[SymInfoSession(day=d, time=time(21, 0)) for d in range(7)],
        session_ends=[SymInfoSession(day=d, time=end_t) for d in range(7)],
    )


def _syminfo(variants):
    """A futures ``SymInfo`` whose flat fields mirror the newest (last) variant."""
    from pynecore.core.syminfo import SymInfo
    newest = variants[-1]
    return SymInfo(
        prefix="CME", description="Night future", ticker="NX", currency="USD", period="720",
        type="futures", mintick=0.25, pricescale=4, minmove=1, pointvalue=50.0, mincontract=1.0,
        timezone="UTC", volumetype="base",
        opening_hours=newest.opening_hours, session_starts=newest.session_starts,
        session_ends=newest.session_ends, session_schedules=list(variants),
    )


def _flat_only(variant):
    """A no-history ``SymInfo`` whose flat fields are ``variant`` (empty history)."""
    from pynecore.core.syminfo import SymInfo
    return SymInfo(
        prefix="CME", description="Night future", ticker="NX", currency="USD", period="720",
        type="futures", mintick=0.25, pricescale=4, minmove=1, pointvalue=50.0, mincontract=1.0,
        timezone="UTC", volumetype="base",
        opening_hours=variant.opening_hours, session_starts=variant.session_starts,
        session_ends=variant.session_ends)


def _state():
    """A non-LTF 720-minute security state, as ``setup_security_states`` would build."""
    from pynecore.core.resampler import Resampler
    from pynecore.core.security import SecurityState
    return SecurityState(
        sec_id="s", timeframe="720", gaps_on=False, same_timeframe=False,
        resampler=Resampler.get_resampler("720"), tz=UTC, is_ltf=False)


def _load(tmp_dir, name, syminfo, days):
    """Write a gappy night-session feed (+ its ``.toml``), run ``load_htf_bar_opens``."""
    from pynecore.core.ohlcv_file import OHLCVWriter
    from pynecore.core.security import load_htf_bar_opens
    from pynecore.types.ohlcv import OHLCV
    path = tmp_dir / f"{name}.ohlcv"
    with OHLCVWriter(path) as w:
        for ts in _opens_sec(days):
            w.write(OHLCV(timestamp=ts, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0))
    syminfo.save_toml(path.with_suffix(".toml"))
    state = _state()
    load_htf_bar_opens(state, str(path))
    return state


def __test_dated_session_schedule_trading_day_boundary__(log):
    """Each night bar is confirmed against its TRADING day's schedule variant.

    A market trading Sun..Fri nights changes its night session END 02:00 -> 01:00
    on trading day 2026-01-12. The night bar opening 2026-01-11 21:00 (Sunday) has
    raw calendar date *before* the change, but overnight it belongs to the 01-12
    trading day and must take the NEW variant. Keying on the raw open date would
    mis-assign exactly this boundary bar by one day -- where a real residual lives.
    The 720-minute period exceeds the night session, so each bar closes at its
    session end.
    """
    import tempfile
    from pathlib import Path
    era_a = _night_variant(date(2025, 6, 1), time(2, 0))    # old: night ends 02:00
    era_b = _night_variant(date(2026, 1, 12), time(1, 0))   # new: night ends 01:00
    with tempfile.TemporaryDirectory() as td:
        state = _load(Path(td), "hist", _syminfo([era_a, era_b]), _SUN_FRI)
    expected = [
        _ms(1, 5, 2), _ms(1, 6, 2), _ms(1, 7, 2), _ms(1, 8, 2), _ms(1, 9, 2), _ms(1, 10, 2),
        _ms(1, 12, 1),   # <- Sun 2026-01-11 21:00 bar, trading day 01-12 -> era B (01:00)
        _ms(1, 13, 1), _ms(1, 14, 1), _ms(1, 15, 1), _ms(1, 16, 1), _ms(1, 17, 1)]
    assert state.bar_closes == expected, f"bar_closes={state.bar_closes}\nexpected={expected}"
    log.info("dated session schedule: the 2026-01-11 21:00 bar confirms on its 01-12 trading day")


def __test_dated_session_schedule_single_consumer_and_fallback__(log):
    """Only ``opening_hours`` drives the close calc; an uncoverable variant falls back.

    * Single consumer: a variant carrying garbage dated ``session_starts`` /
      ``session_ends`` produces the same ``bar_closes`` as a clean one -- proving
      the dated session boundaries never leak into the Core close calculation,
      even though the TOML schema carries them.
    * Graceful fallback: when the oldest variant's ``opening_hours`` cannot cover
      its bars, ``state.bar_closes`` stays ``None`` so the caller keeps the
      arithmetic grid clamp instead of crashing.
    """
    import tempfile
    from pathlib import Path
    from pynecore.core.syminfo import SymInfoInterval, SymInfoSession, SymInfoScheduleVariant
    era_a = _night_variant(date(2025, 6, 1), time(2, 0))
    era_b = _night_variant(date(2026, 1, 12), time(1, 0))

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        clean = _load(tmp, "clean", _syminfo([era_a, era_b]), _MON_FRI)
        assert clean.bar_closes is not None

        # Garbage dated session_starts/ends on the oldest variant: same opening_hours.
        garbage = SymInfoScheduleVariant(
            effective_from=date(2025, 6, 1), opening_hours=era_a.opening_hours,
            session_starts=[SymInfoSession(day=d, time=time(5, 5)) for d in range(7)],
            session_ends=[SymInfoSession(day=d, time=time(5, 5)) for d in range(7)])
        leaked = _load(tmp, "garbage", _syminfo([garbage, era_b]), _MON_FRI)
        assert leaked.bar_closes == clean.bar_closes, \
            "dated session_starts/ends leaked into the close calculation"

        # Oldest variant cannot cover the night opens (day session only) -> None.
        uncoverable = SymInfoScheduleVariant(
            effective_from=date(2025, 6, 1),
            opening_hours=[SymInfoInterval(day=d, start=time(10, 0), end=time(18, 0))
                           for d in range(7)],
            session_starts=era_a.session_starts, session_ends=era_a.session_ends)
        fallback = _load(tmp, "bad", _syminfo([uncoverable, era_b]), _MON_FRI)
        assert fallback.bar_closes is None, \
            "an uncoverable variant must leave bar_closes None so the grid clamp is kept"

    log.info("dated session schedule: single-consumer invariant holds and None fallback is graceful")


def __test_dated_session_schedule_toml_model__(log):
    """``SymInfo`` history model + TOML: sort, dup rejection, normalization, history-wins.

    Covers the data-model contract independent of the close calculation: variants
    sort ascending on load, a duplicate ``effective_from`` is rejected,
    ``effective_from`` accepts a native date / quoted string / datetime (all
    normalized to ``date``), the flat fields are regenerated from the newest
    variant (history wins over an inconsistent flat block), a round-trip is stable,
    and a no-history save embeds the commented example which parses back to empty.
    """
    import tempfile
    from pathlib import Path
    from pynecore.core.syminfo import SymInfo

    era_a = _night_variant(date(2025, 6, 1), time(2, 0))
    era_b = _night_variant(date(2026, 1, 12), time(1, 0))

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # Round-trip with history: flat == newest, both variants preserved & sorted.
        p = tmp / "hist.toml"
        _syminfo([era_b, era_a]).save_toml(p)   # saved out of order on purpose
        reloaded = SymInfo.load_toml(p)
        assert [v.effective_from for v in reloaded.session_schedules] == \
            [date(2025, 6, 1), date(2026, 1, 12)], "variants must sort ascending on load"
        assert reloaded.schedule_index_for(date(2025, 12, 31)) == 0
        assert reloaded.schedule_index_for(date(2026, 1, 12)) == 1
        assert reloaded.schedule_index_for(date(2024, 1, 1)) == 0, "pre-earliest clamps to oldest"
        assert reloaded.opening_hours == era_b.opening_hours, "flat mirrors the newest variant"
        # schedule_for resolves the tuple the same way.
        assert reloaded.schedule_for(date(2025, 12, 31))[0] == era_a.opening_hours
        assert reloaded.schedule_for(date(2026, 2, 1))[0] == era_b.opening_hours

        # No-history save embeds the commented example; it parses to empty history.
        p2 = tmp / "flat.toml"
        _flat_only(era_b).save_toml(p2)
        assert not _flat_only(era_b).has_schedule_history
        assert "# [[session_schedules]]" in p2.read_text(), "no-history save must embed the example"
        assert SymInfo.load_toml(p2).session_schedules == [], "the example must stay commented out"

        # Duplicate effective_from -> ValueError.
        dup = tmp / "dup.toml"
        dup.write_text(_minimal_toml() + (
            "[[session_schedules]]\neffective_from = 2026-01-01\n"
            "[[session_schedules.opening_hours]]\nday=0\nstart=\"10:00:00\"\nend=\"18:00:00\"\n"
            "[[session_schedules]]\neffective_from = 2026-01-01\n"
            "[[session_schedules.opening_hours]]\nday=0\nstart=\"10:00:00\"\nend=\"18:00:00\"\n"))
        try:
            SymInfo.load_toml(dup)
            assert False, "duplicate effective_from must raise ValueError"
        except ValueError as exc:
            assert "Duplicate" in str(exc)

        # effective_from accepts string + datetime, normalized to date; history wins
        # over the inconsistent flat block above it.
        mixed = tmp / "mixed.toml"
        mixed.write_text(_minimal_toml(flat_end="23:00:00") + (
            "[[session_schedules]]\neffective_from = \"2026-06-01\"\n"
            "[[session_schedules.opening_hours]]\nday=0\nstart=\"10:00:00\"\nend=\"19:00:00\"\n"
            "[[session_schedules]]\neffective_from = 2025-01-01 00:00:00\n"
            "[[session_schedules.opening_hours]]\nday=0\nstart=\"09:00:00\"\nend=\"17:00:00\"\n"))
        mi = SymInfo.load_toml(mixed)
        effs = [v.effective_from for v in mi.session_schedules]
        assert effs == [date(2025, 1, 1), date(2026, 6, 1)], f"normalized+sorted dates: {effs}"
        assert all(isinstance(e, date) and not isinstance(e, datetime) for e in effs)
        assert mi.opening_hours[0].end == time(19, 0), "history must override the flat block"

    log.info("dated session schedule: TOML model sorts, rejects dups, normalizes dates, history wins")


def __test_session_bar_closes_overnight_after_midnight__(log):
    """Bars opening after midnight match the previous day's overnight interval.

    A sub-session HTF (here 2h) on a midnight-crossing ``21:00->02:00`` night
    session opens several bars per session; the last one opens *after* midnight
    (01:00). Its calendar weekday is the day AFTER the session-open weekday, while
    the ``opening_hours`` interval is keyed on the session-open day, so a naive
    same-weekday match finds nothing and the whole feed would fall back to the late
    arithmetic grid (the 01:00 bar confirming at 03:00 instead of the 02:00 session
    end). The after-midnight bar must instead be matched to the previous day's
    overnight interval and confirm at the session end.
    """
    from pynecore.core.security import _session_bar_closes
    from pynecore.core.syminfo import SymInfoInterval

    oh = [SymInfoInterval(day=d, start=time(21, 0), end=time(2, 0)) for d in range(7)]
    opens = [_ms(1, 5, 21), _ms(1, 5, 23), _ms(1, 6, 1)]  # Mon 21:00, 23:00, Tue 01:00
    closes = _session_bar_closes(opens, UTC, oh, 2 * 60 * 60 * 1000)
    expected = [_ms(1, 5, 23), _ms(1, 6, 1), _ms(1, 6, 2)]  # last clamps to 02:00 session end
    assert closes == expected, f"closes={closes}\nexpected={expected}"
    log.info("session bar closes: after-midnight night bar confirms at its 02:00 session end")


def _minimal_toml(flat_end="18:00:00"):
    """A minimal valid [symbol] section + one flat opening-hours block, for parse tests."""
    return (
        "[symbol]\nprefix=\"X\"\ndescription=\"d\"\nticker=\"T\"\ncurrency=\"USD\"\n"
        "period=\"720\"\ntype=\"futures\"\nmintick=0.25\npricescale=4\nminmove=1\n"
        "pointvalue=50.0\nmincontract=1.0\ntimezone=\"UTC\"\n"
        f"[[opening_hours]]\nday=0\nstart=\"10:00:00\"\nend=\"{flat_end}\"\n")


def __test_load_htf_bar_opens_dated_vs_flat_history__(log):
    """``load_htf_bar_opens``: a history TOML drives the dated path; no-history is the old result.

    The same gappy session feed is loaded twice. With a 2-era history TOML,
    ``state.bar_closes`` is era-correct -- bars before the 2026-01-12 change confirm
    at the old session end (02:00), bars after at the new one (01:00). With a
    no-history TOML (flat = the new schedule only), every bar confirms at 01:00:
    the stale single-schedule result the history exists to correct on the
    pre-change bars.
    """
    import tempfile
    from pathlib import Path
    era_a = _night_variant(date(2025, 6, 1), time(2, 0))
    era_b = _night_variant(date(2026, 1, 12), time(1, 0))

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        dated = _load(tmp, "hist", _syminfo([era_a, era_b]), _MON_FRI)
        flat = _load(tmp, "flat", _flat_only(era_b), _MON_FRI)

    exp_dated = [_ms(1, 6, 2), _ms(1, 7, 2), _ms(1, 8, 2), _ms(1, 9, 2), _ms(1, 10, 2),
                 _ms(1, 13, 1), _ms(1, 14, 1), _ms(1, 15, 1), _ms(1, 16, 1), _ms(1, 17, 1)]
    exp_flat = [_ms(1, 6, 1), _ms(1, 7, 1), _ms(1, 8, 1), _ms(1, 9, 1), _ms(1, 10, 1),
                _ms(1, 13, 1), _ms(1, 14, 1), _ms(1, 15, 1), _ms(1, 16, 1), _ms(1, 17, 1)]
    assert dated.bar_closes == exp_dated, f"dated bar_closes={dated.bar_closes}"
    assert flat.bar_closes == exp_flat, f"flat bar_closes={flat.bar_closes}"
    assert dated.bar_closes != flat.bar_closes, "history must correct the pre-change bars"

    log.info("load_htf_bar_opens: history TOML drives the dated path, no-history is the stale result")
