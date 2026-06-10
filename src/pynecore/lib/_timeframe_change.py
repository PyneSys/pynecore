"""
@pyne

Stateful implementation of ``lib.timeframe.change``. It lives in its own
small module because the ``@pyne`` marker is module-level and the host
module (``lib/timeframe.py``) must stay untransformed; the host re-exports
the function, and the layout travels on the function object.

The stateless helpers (session math, anchor replay) stay in the host module:
they run untransformed at full speed, and the first-bar anchor replay loop
(``_anchor_init``) can span hundreds of thousands of virtual candles on
intraday charts — anchored per-iteration call sites there would retain a
bound-pair entry per iteration.

The host module is imported as an alias; the import normalizer rewrites the
references to call-time ``lib.timeframe.*`` attribute lookups, which also
keeps the host <-> submodule import cycle benign.
"""
from datetime import datetime, timedelta, UTC

from pynecore import lib
from pynecore.lib import syminfo as _syminfo
from pynecore.lib import timeframe as _timeframe
from pynecore.types import Persistent
from pynecore.core.datetime import parse_timezone as _parse_timezone

__all__ = ['change']


# TODO: make it simpler and better if you could
# I know this function is awful. It was one of the hardest part of the whole Pine library.
# It may be simplified. The problem is that every timefram has different anchor points and slightly different rules. Or
# just not found the general rule for all timeframes.
def change(timeframe: str) -> bool:
    """
    Detects changes in the specified timeframe.

    :param timeframe: The timeframe to check
    :return: Returns true on the first bar of a `timeframe`, false otherwise.
    """
    next_new_year_session: Persistent[datetime | None] = None
    cycle: Persistent[int] = 0
    last_dt: Persistent[datetime | None] = None
    last_signal: Persistent[datetime | None] = None
    next_anchor: Persistent[datetime | None] = None
    next_signal: Persistent[datetime | None] = None

    tf_sec = _timeframe.in_seconds(timeframe)
    xchg_tf_sec = _timeframe.in_seconds(_syminfo.period)

    # The timeframe to check nust be greater (or equal) than the current timeframe
    if tf_sec < xchg_tf_sec:
        return False

    _modifier, _multiplier = _timeframe._process_tf(timeframe)
    assert _modifier != 'T', "Ticks are not (yet) supported!"
    is_intraday = _modifier == '' or _modifier == 'S'

    # Datetime in exchange timezone
    dt: datetime = lib._datetime

    # Initialize persistent variables
    if last_dt is None:
        last_dt = dt - timedelta(seconds=xchg_tf_sec)
        last_signal = last_dt
    assert isinstance(last_dt, datetime)

    # Find anchor point and replay sessions to the 1st bar
    if not is_intraday and next_new_year_session is None:
        _nys, _cycle, _last_signal, _next_anchor, _next_signal = _timeframe._anchor_init(
            dt, _modifier, _multiplier, tf_sec, xchg_tf_sec, last_signal)
        next_new_year_session = _nys
        cycle = _cycle
        last_signal = _last_signal
        next_anchor = _next_anchor
        next_signal = _next_signal

    # We need to check every virtual candles, even if they are missing in the dataset
    while last_dt < dt:
        prev_dt = last_dt
        last_dt = last_dt + timedelta(seconds=xchg_tf_sec)

        if is_intraday:
            # The anchor point is the session start
            if _timeframe._is_new_session(last_dt, prev_dt, xchg_tf_sec):
                # We need to round the session start to the nearest hour
                last_signal = last_dt.replace(minute=0, second=0, microsecond=0)
            assert isinstance(last_signal, datetime)
            seconds_since_last_session = (last_dt - last_signal).total_seconds()

            if seconds_since_last_session % tf_sec == 0:
                last_dt = dt
                return lib.bar_index > 0

        # We need to check only trading days
        elif _modifier == 'D':
            # Check if it is a new year
            assert isinstance(next_new_year_session, datetime)
            if (_timeframe._is_time_in_candle(last_dt, next_new_year_session, xchg_tf_sec) or
                    next_new_year_session < last_dt):
                # Find the next new year session
                next_new_year_session = _timeframe._get_new_year_session(
                    next_new_year_session + timedelta(days=367))
                cycle = 0

            if _timeframe._is_new_session(last_dt, prev_dt, xchg_tf_sec):
                cycle -= 1
                assert isinstance(last_signal, datetime)
                if cycle <= 0:
                    cycle = _multiplier
                    if last_signal < last_dt:
                        last_signal = dt
                        return lib.bar_index > 0

        # We don't need to skip weekends here
        elif _modifier == 'W':
            assert isinstance(next_new_year_session, datetime)
            if (_timeframe._is_time_in_candle(last_dt, next_new_year_session, xchg_tf_sec) or
                    next_new_year_session < last_dt):
                _dt_utc = ((next_new_year_session + timedelta(days=1))
                           .astimezone(UTC).replace(day=1, month=1, hour=0, minute=0, second=0))
                next_new_year_session = _timeframe._get_new_year_session(_dt_utc.replace(year=_dt_utc.year + 1))

                # The anchor point must be Monday
                while True:
                    if _dt_utc.weekday() != 0:
                        _dt_utc += timedelta(days=1)
                        continue
                    break

                # 1st signal of the new year
                next_signal = None
                while next_signal is None:
                    next_signal = _timeframe._get_first_session_start_in_day(_dt_utc)
                    _dt_utc += timedelta(days=1)
                next_anchor = next_signal
                next_signal = next_signal.astimezone(_parse_timezone(_syminfo.timezone))

            assert isinstance(next_signal, datetime)
            if (_timeframe._is_time_in_candle(last_dt, next_signal, xchg_tf_sec)
                    or last_dt >= next_signal):
                assert isinstance(next_anchor, datetime)
                # Increase the anchor date
                next_anchor = next_anchor + timedelta(seconds=tf_sec)
                # Find the next signal which is usually Monday
                _dt_utc = next_anchor
                next_signal = None
                while next_signal is None:
                    next_signal = _timeframe._get_first_session_start_in_day(_dt_utc)
                    _dt_utc += timedelta(days=1)
                next_signal = next_signal.astimezone(_parse_timezone(_syminfo.timezone))
                return lib.bar_index > 0

        elif _modifier == 'M':
            # Is it a new year candle?
            assert isinstance(next_new_year_session, datetime)
            if (_timeframe._is_time_in_candle(last_dt, next_new_year_session, xchg_tf_sec) or
                    next_new_year_session < last_dt):
                _dt_utc = ((next_new_year_session + timedelta(days=1))
                           .astimezone(UTC).replace(day=1, month=1, hour=0, minute=0, second=0))
                next_new_year_session = _timeframe._get_new_year_session(_dt_utc.replace(year=_dt_utc.year + 1))

                # 1st signal of the new year
                next_signal = None
                while next_signal is None:
                    next_signal = _timeframe._get_first_session_start_in_day(_dt_utc)
                    _dt_utc += timedelta(days=1)
                next_signal = next_signal.astimezone(_parse_timezone(_syminfo.timezone))

                cycle = 0

            assert isinstance(next_signal, datetime)
            if (_timeframe._is_time_in_candle(last_dt, next_signal, xchg_tf_sec)
                    or last_dt >= next_signal):
                _dt_utc = next_signal.astimezone(UTC) + timedelta(days=1)
                if _dt_utc.month == 12:
                    next_month_dt = _dt_utc.replace(year=_dt_utc.year + 1, month=1, day=1, hour=0, minute=0, second=0)
                else:
                    next_month_dt = _dt_utc.replace(month=_dt_utc.month + 1, day=1, hour=0, minute=0, second=0)
                next_signal = None
                while next_signal is None:
                    next_signal = _timeframe._get_first_session_start_in_day(next_month_dt)
                    next_month_dt += timedelta(days=1)

                next_signal = next_signal.astimezone(_parse_timezone(_syminfo.timezone))

                m = cycle
                cycle += 1

                if m % _multiplier == 0:
                    return lib.bar_index > 0

    return False
