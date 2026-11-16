"""
@pyne
"""
from pynecore.lib import barstate, plot, script


@script.indicator("Live last confirmed history")
def main():
    plot(1 if barstate.islastconfirmedhistory else 0, "lch")
    plot(1 if barstate.isconfirmed else 0, "confirmed")


__test_helper_MINUTE = 60_000
__test_helper_N_HIST = 10


def __test_helper_bar(i, is_closed=True):
    from pynecore.types.ohlcv import OHLCV
    price = 100.0 + i
    return OHLCV(timestamp=i * __test_helper_MINUTE, open=price, high=price + 1,
                 low=price - 1, close=price, volume=1000.0, is_closed=is_closed)


def __test_helper_history_flags(script_path, module_key, syminfo, catch_up, history_as_of):
    """``(islastconfirmedhistory, isconfirmed)`` of every history bar of a live run.

    :param catch_up: Bars the catch-up drain delivers before the live transition.
    :param history_as_of: The download instant of the history.
    """
    import sys
    from pynecore.core.script_runner import ScriptRunner, LIVE_TRANSITION

    for key in [module_key, script_path.stem]:
        sys.modules.pop(key, None)
    syminfo.period = "1"
    last = __test_helper_N_HIST - 1
    feed = ([__test_helper_bar(i) for i in range(__test_helper_N_HIST)] + catch_up
            + [LIVE_TRANSITION, __test_helper_bar(last), __test_helper_bar(last + 1)])
    runner = ScriptRunner(script_path, iter(feed), syminfo, live=True,
                          last_bar_index=last, last_bar_time=last * __test_helper_MINUTE,
                          history_as_of=history_as_of)
    rows = [(c.timestamp // __test_helper_MINUTE, pv["lch"], pv["confirmed"])
            for c, pv in runner.run_iter()]
    n_hist = __test_helper_N_HIST + len(catch_up) - sum(
        1 for b in catch_up if b.timestamp == last * __test_helper_MINUTE)
    return rows[:n_hist]


def __test_forming_tail_moves_last_confirmed_history_back__(script_path, module_key, syminfo):
    """With a forming tail, the bar before it is the last confirmed history bar"""
    last = __test_helper_N_HIST - 1
    rows = __test_helper_history_flags(
        script_path, module_key, syminfo, [],
        history_as_of=last * __test_helper_MINUTE + __test_helper_MINUTE // 2)
    flagged = [i for i, lch, _ in rows if lch == 1]
    assert flagged == [last - 1], rows
    assert rows[-1][2] == 0, "the forming tail runs unconfirmed"
    assert all(confirmed == 1 for _, _, confirmed in rows[:-1])


def __test_closed_tail_is_the_last_confirmed_history_bar__(script_path, module_key, syminfo):
    """A tail closed before the download is itself the last confirmed history bar"""
    last = __test_helper_N_HIST - 1
    rows = __test_helper_history_flags(
        script_path, module_key, syminfo, [],
        history_as_of=(last + 1) * __test_helper_MINUTE)
    flagged = [i for i, lch, _ in rows if lch == 1]
    assert flagged == [last], rows
    assert all(confirmed == 1 for _, _, confirmed in rows)


def __test_helper_monthly_flags(script_path, module_key, syminfo, months, history_as_of):
    """``(month, islastconfirmedhistory, isconfirmed)`` of a monthly history of 2026.

    :param months: The months of the history, ascending.
    :param history_as_of: The download instant of the history, a UTC ``datetime``.
    """
    import sys
    from datetime import datetime, UTC
    from pynecore.core.script_runner import ScriptRunner, LIVE_TRANSITION
    from pynecore.types.ohlcv import OHLCV

    for key in [module_key, script_path.stem]:
        sys.modules.pop(key, None)
    syminfo.period = "1M"
    opens = [int(datetime(2026, month, 1, tzinfo=UTC).timestamp() * 1000) for month in months]
    feed = [OHLCV(timestamp=ts, open=100.0, high=101.0, low=99.0, close=100.0, volume=1.0,
                  is_closed=True) for ts in opens]
    runner = ScriptRunner(script_path, iter(feed + [LIVE_TRANSITION]), syminfo, live=True,
                          last_bar_index=len(opens) - 1, last_bar_time=opens[-1],
                          history_as_of=int(history_as_of.timestamp() * 1000))
    return [(datetime.fromtimestamp(c.timestamp / 1000, UTC).month, pv["lch"], pv["confirmed"])
            for c, pv in runner.run_iter()][:len(opens)]


def __test_monthly_tail_closes_at_the_calendar_month_end__(script_path, module_key, syminfo):
    """A month ends at its calendar end, not after the nominal ~30.4 days

    Downloaded on March 31, past the nominal length but before April 1, March is
    still forming. Downloaded right after March 1, before the nominal length ran
    out, February is closed.
    """
    from datetime import datetime, UTC
    rows = __test_helper_monthly_flags(script_path, module_key, syminfo, (1, 2, 3),
                                       datetime(2026, 3, 31, 12, 0, tzinfo=UTC))
    assert [m for m, lch, _ in rows if lch == 1] == [2], rows
    assert rows[-1][2] == 0, "March runs unconfirmed"

    rows = __test_helper_monthly_flags(script_path, module_key, syminfo, (1, 2),
                                       datetime(2026, 3, 1, 0, 5, tzinfo=UTC))
    assert [m for m, lch, _ in rows if lch == 1] == [2], rows
    assert all(confirmed == 1 for _, _, confirmed in rows)


def __test_caught_up_tail_is_the_last_confirmed_history_bar__(script_path, module_key, syminfo):
    """A forming tail the catch-up drain closed is the last confirmed history bar"""
    last = __test_helper_N_HIST - 1
    rows = __test_helper_history_flags(
        script_path, module_key, syminfo, [__test_helper_bar(last)],
        history_as_of=last * __test_helper_MINUTE + __test_helper_MINUTE // 2)
    flagged = [i for i, lch, _ in rows if lch == 1]
    assert flagged == [last], rows
    assert all(confirmed == 1 for _, _, confirmed in rows)
