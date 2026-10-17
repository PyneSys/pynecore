"""
@pyne lib

Stateful implementation of ``lib.timeframe.change``. It lives in its own
small module because the ``@pyne`` marker is module-level and the host
module (``lib/timeframe.py``) must stay untransformed; the host re-exports
the function, and the layout travels on the function object.

The stateless helpers (session math) stay in the host module so they run
untransformed at full speed.

For daily/weekly/monthly timeframes ``change`` is equivalent to
``ta.change(time(timeframe))`` on the year-reset scheduled grid (see the
``core.resampler`` module docs): it returns ``true`` on the first chart bar
of each new period. Intraday timeframes replay virtual candles against the
session template to find period starts within the session.
"""
from datetime import datetime, timedelta

from pynecore import lib
from pynecore.lib import syminfo as _syminfo
from pynecore.lib import timeframe as _timeframe
from pynecore.types import Persistent

__all__ = ['change']


# The engine-private ``lib._*`` members are the intended interface here, and the
# ``Persistent`` markers are rewritten by the AST transformer, so the IDE sees neither
# the next-bar reads of the state variables nor their lazy-init narrowing.
# noinspection PyProtectedMember,PyUnusedLocal,PyUnresolvedReferences,PyTypeChecker
def change(timeframe: str = "") -> bool:
    """
    Detects changes in the specified timeframe.

    :param timeframe: The timeframe to check; omitted or empty uses the chart timeframe.
    :return: Returns true on the first bar of a `timeframe`, false otherwise.
    """
    last_dt: Persistent[datetime | None] = None
    last_signal: Persistent[datetime | None] = None
    last_period: Persistent[object] = None

    # An empty timeframe string is the chart's own timeframe (measured: over a
    # full history TradingView's timeframe.change("") never differs from
    # timeframe.change(timeframe.period))
    if not timeframe:
        timeframe = str(_syminfo.period)

    tf_sec = _timeframe.in_seconds(timeframe)
    xchg_tf_sec = _timeframe.in_seconds(_syminfo.period)

    # The timeframe to check must be greater (or equal) than the current timeframe
    if tf_sec < xchg_tf_sec:
        return False

    _modifier, _multiplier = _timeframe._process_tf(timeframe)
    assert _modifier != 'T', "Ticks are not (yet) supported!"

    if _modifier == '' or _modifier == 'S':
        # Intraday: replay virtual candles (even those missing from the
        # dataset) and anchor period starts to the session start
        dt: datetime = lib._datetime

        if last_dt is None:
            last_dt = dt - timedelta(seconds=xchg_tf_sec)
            # The anchor point is the session start, even when the data starts mid-session
            session_start = _timeframe._last_session_start(dt)
            if session_start is not None:
                last_signal = session_start.replace(minute=0, second=0, microsecond=0)
            else:
                last_signal = last_dt

        # Every virtual candle up to the current bar is replayed, so a session
        # start inside a gap moves the anchor even if an earlier candle of the
        # gap already starts a new period
        is_change = False
        while last_dt < dt:
            prev_dt = last_dt
            last_dt = last_dt + timedelta(seconds=xchg_tf_sec)

            # The anchor point is the session start
            if _timeframe._is_new_session(last_dt, prev_dt, xchg_tf_sec):
                # We need to round the session start to the nearest hour
                last_signal = last_dt.replace(minute=0, second=0, microsecond=0)

            # A virtual candle starts a new period when a period boundary lies in
            # (prev_dt, last_dt], also when the requested timeframe is not a
            # multiple of the chart's and no candle opens on the boundary itself
            if ((last_dt - last_signal).total_seconds() // tf_sec
                    != (prev_dt - last_signal).total_seconds() // tf_sec):
                is_change = True

        last_dt = dt
        return is_change and lib.bar_index > 0

    # Daily/Weekly/Monthly: true on the first chart bar of a new period of
    # the scheduled grid — the period identity changing is exactly
    # ``ta.change(time(timeframe))`` on the verified grid
    if _multiplier > 1:
        if (_modifier, _multiplier) == _timeframe._process_tf(str(_syminfo.period)):
            # The chart's own bars are the requested grid: every bar starts one
            key = lib._time
        else:
            key = lib._dwm_change_key(timeframe, _modifier, _multiplier)
    elif _modifier == 'D':
        key = lib._dg_trading_day()
    elif _modifier == 'W':
        td = lib._dg_trading_day()
        key = td - timedelta(days=td.weekday())
    else:  # 'M'
        td = lib._dg_trading_day()
        key = (td.year, td.month)

    if key != last_period:
        is_first = last_period is None
        last_period = key
        if not is_first:
            return lib.bar_index > 0
    return False
