"""
@pyne
"""
import pytest

from pynecore.lib import barmerge, close, format, plot, request, script, syminfo, timeframe


@script.indicator(title="Lookahead Modes Test", shorttitle="LM",
                  format=format.price, precision=6)
def main():
    sec_off = request.security(syminfo.ticker, timeframe.period, close)
    sec_last_closed = request.security(
        syminfo.ticker, timeframe.period, close,
        lookahead=barmerge.lookahead_last_closed,
    )
    sec_on = request.security(
        syminfo.ticker, timeframe.period, close,
        lookahead=barmerge.lookahead_on,
    )
    plot(sec_off, title="SecOff")
    plot(sec_last_closed, title="SecLastClosed")
    plot(sec_on, title="SecOn")


def __test_lookahead_modes_historical_equivalence__(csv_reader, runner, log):
    """In historical/backtest mode all three lookahead modes are functionally
    equivalent for ``close`` — they all produce the most-recently-closed
    security bar value. ``lookahead_on`` historically falls back to
    ``lookahead_off`` semantics (no developing exposure) because the
    chart-derived developing OHLCV pipeline is live-mode-only.

    Live mode is where ``lookahead_on`` diverges; this historical
    equivalence guards future live-mode work from accidentally breaking the
    backtest path.
    """
    from pynecore import lib

    with csv_reader('advance_decline_ratio.csv', subdir="data") as cr:
        r = runner(
            cr,
            syminfo_override=dict(timezone="US/Eastern"),
        )
        for i, (candle, plot_values) in enumerate(r.run_iter()):
            off_val = plot_values.get('SecOff')
            llc_val = plot_values.get('SecLastClosed')
            on_val = plot_values.get('SecOn')
            off_is_na = off_val is None or isinstance(off_val, type(lib.na))
            llc_is_na = llc_val is None or isinstance(llc_val, type(lib.na))
            on_is_na = on_val is None or isinstance(on_val, type(lib.na))
            assert off_is_na == llc_is_na == on_is_na, (
                f"bar {i}: na-ness mismatch — "
                f"SecOff={off_val!r}, SecLastClosed={llc_val!r}, SecOn={on_val!r}"
            )
            if not off_is_na:
                assert off_val == llc_val == on_val, (
                    f"bar {i}: SecOff={off_val} != "
                    f"SecLastClosed={llc_val} != SecOn={on_val}"
                )

    log.info("lookahead_off / lookahead_last_closed / lookahead_on agree historically")


def __test_get_confirmed_time_on_live_returns_current_period__(log):
    """Live ``Lookahead.ON`` targets the CONTAINING (developing) period so the
    subprocess steps into the open HTF bar. Historical/off/last_closed target
    the previously closed period.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from pynecore.core.security import (
        SecurityState, Lookahead, _get_confirmed_time,
    )
    from pynecore.core.resampler import Resampler
    from pynecore.core.htf_aggregator import HTFAggregator

    utc = ZoneInfo('UTC')
    tf = '60'

    def _ms(year, month, day, hour, minute=0):
        return int(datetime(year, month, day, hour, minute, tzinfo=utc).timestamp() * 1000)

    prev_chart = _ms(2026, 5, 21, 10, 30)
    curr_chart = _ms(2026, 5, 21, 10, 45)  # still in 10:00-11:00 HTF period
    expected_current_period = _ms(2026, 5, 21, 10, 0)
    expected_prev_period = expected_current_period  # same containing period
    # Use a chart_time across the boundary for the "prev_period changes" case
    curr_chart_boundary = _ms(2026, 5, 21, 11, 5)
    expected_period_after_boundary = _ms(2026, 5, 21, 11, 0)

    state_on_live = SecurityState(
        sec_id='s1', timeframe=tf, gaps_on=False, same_timeframe=False,
        resampler=Resampler.get_resampler(tf), tz=utc,
        lookahead=Lookahead.ON,
        htf_aggregator=HTFAggregator(tf, utc),
        is_live=True,
        prev_chart_time=prev_chart, last_confirmed=expected_prev_period,
    )
    assert _get_confirmed_time(state_on_live, curr_chart) == expected_current_period
    assert _get_confirmed_time(state_on_live, curr_chart_boundary) == \
        expected_period_after_boundary

    state_on_historical = SecurityState(
        sec_id='s2', timeframe=tf, gaps_on=False, same_timeframe=False,
        resampler=Resampler.get_resampler(tf), tz=utc,
        lookahead=Lookahead.ON,
        htf_aggregator=HTFAggregator(tf, utc),
        is_live=False,  # historical
        prev_chart_time=prev_chart, last_confirmed=0,
    )
    # Historical lookahead_on falls back to OFF semantics — no new period
    # opens between curr_chart (10:45) and prev_chart (10:30), so the function
    # returns last_confirmed (0 here, no period closed yet).
    assert _get_confirmed_time(state_on_historical, curr_chart) == 0

    state_off_live = SecurityState(
        sec_id='s3', timeframe=tf, gaps_on=False, same_timeframe=False,
        resampler=Resampler.get_resampler(tf), tz=utc,
        lookahead=Lookahead.OFF,
        is_live=True,
        prev_chart_time=prev_chart, last_confirmed=expected_prev_period,
    )
    # Lookahead.OFF never steps into a developing bar.
    assert _get_confirmed_time(state_off_live, curr_chart) == expected_prev_period


def __test_lookahead_on_setup_initializes_aggregator__(log):
    """``lookahead_on`` is now supported: setup_security_states populates a
    Lookahead.ON state with an HTFAggregator for the developing-bar
    transport. Historical mode keeps closed-only semantics until
    ``state.is_live`` flips at LIVE_TRANSITION.
    """
    from pynecore.core.security import setup_security_states, Lookahead
    from pynecore.lib import barmerge as bm
    from zoneinfo import ZoneInfo

    contexts = {
        'sec\xb70': {
            'symbol': 'AAPL',
            'timeframe': '60',
            'gaps': bm.gaps_off,
            'lookahead': bm.lookahead_on,
        }
    }
    states, sync_block, result_blocks = setup_security_states(
        contexts, chart_timeframe='1', tz=ZoneInfo('UTC'),
        chart_symbol='AAPL',
    )
    try:
        st = states['sec\xb70']
        assert st.lookahead is Lookahead.ON
        assert st.htf_aggregator is not None
        assert st.is_live is False  # historical until LIVE_TRANSITION
    finally:
        for rb in result_blocks.values():
            rb.close()
            rb.unlink()
        sync_block.close()
        sync_block.unlink()


def __test_live_htf_transport_covers_all_lookahead_modes__(log):
    """All lookahead modes (OFF / LAST_CLOSED / ON) get an ``HTFAggregator``
    on same-symbol HTF contexts — the live closed-bar transport drives every
    HTF security context, not just ``Lookahead.ON``. Cross-symbol HTF stays
    aggregator-less (chart OHLCV would be the wrong instrument).
    """
    from pynecore.core.security import setup_security_states, Lookahead
    from pynecore.lib import barmerge as bm
    from zoneinfo import ZoneInfo

    contexts = {
        'sec_off': {
            'symbol': 'AAPL', 'timeframe': '60',
            'gaps': bm.gaps_off, 'lookahead': bm.lookahead_off,
        },
        'sec_lc': {
            'symbol': 'AAPL', 'timeframe': '60',
            'gaps': bm.gaps_off, 'lookahead': bm.lookahead_last_closed,
        },
        'sec_on': {
            'symbol': 'AAPL', 'timeframe': '60',
            'gaps': bm.gaps_off, 'lookahead': bm.lookahead_on,
        },
        'sec_cross': {  # cross-symbol HTF — no aggregator
            'symbol': 'MSFT', 'timeframe': '60',
            'gaps': bm.gaps_off, 'lookahead': bm.lookahead_off,
        },
        'sec_same_tf': {  # same-TF cross-symbol — no aggregator (different branch)
            'symbol': 'MSFT', 'timeframe': '1',
            'gaps': bm.gaps_off, 'lookahead': bm.lookahead_off,
        },
    }
    states, sync_block, result_blocks = setup_security_states(
        contexts, chart_timeframe='1', tz=ZoneInfo('UTC'),
        chart_symbol='AAPL',
    )
    try:
        assert states['sec_off'].lookahead is Lookahead.OFF
        assert states['sec_off'].htf_aggregator is not None

        assert states['sec_lc'].lookahead is Lookahead.LAST_CLOSED
        assert states['sec_lc'].htf_aggregator is not None

        assert states['sec_on'].lookahead is Lookahead.ON
        assert states['sec_on'].htf_aggregator is not None

        # Cross-symbol HTF: aggregator-less (would be wrong-instrument OHLCV)
        assert states['sec_cross'].htf_aggregator is None

        # Same-TF (cross-symbol): different code path, never gets aggregator
        assert states['sec_same_tf'].same_timeframe is True
        assert states['sec_same_tf'].htf_aggregator is None
    finally:
        for rb in result_blocks.values():
            rb.close()
            rb.unlink()
        sync_block.close()
        sync_block.unlink()
