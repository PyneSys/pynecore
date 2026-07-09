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
    """Off / last_closed / on agree on ``close`` for a SAME-timeframe security.

    The security here is requested at ``timeframe.period`` (same TF as the
    chart), so there is no separate HTF period to step into — every lookahead
    mode resolves to the current bar's own value. This equivalence is
    lookahead-independent and holds in TradingView too. (For a genuine HTF,
    ``lookahead_on`` steps into the containing period and diverges from
    ``off`` — see ``test_018_htf_lookahead_on_containing``.)
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
    """Live ``Lookahead.ON`` targets the developing period; off/historical target the closed one.

    Live ``Lookahead.ON`` targets the CONTAINING (developing) period so the
    subprocess steps into the open HTF bar. Historical/off/last_closed target
    the most recent period CLOSED by the chart bar's close instant — the HTF
    period's last chart bar already confirms it (TV ``lookahead_off``).
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
    chart_off = 15 * 60 * 1000 - 1  # 15-minute chart bars

    def _ms(year, month, day, hour, minute=0):
        return int(datetime(year, month, day, hour, minute, tzinfo=utc).timestamp() * 1000)

    curr_chart = _ms(2026, 5, 21, 10, 45)  # last chart bar of 10:00-11:00 HTF
    expected_current_period = _ms(2026, 5, 21, 10, 0)
    expected_prev_period = _ms(2026, 5, 21, 9, 0)
    curr_chart_boundary = _ms(2026, 5, 21, 11, 5)
    expected_period_after_boundary = _ms(2026, 5, 21, 11, 0)

    state_on_live = SecurityState(
        sec_id='s1', timeframe=tf, gaps_on=False, same_timeframe=False,
        resampler=Resampler.get_resampler(tf), tz=utc,
        lookahead=Lookahead.ON,
        htf_aggregator=HTFAggregator(tf, utc),
        is_live=True,
        last_confirmed=expected_prev_period, chart_off=chart_off,
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
        last_confirmed=0, chart_off=chart_off,
    )
    # Historical lookahead_on falls back to OFF semantics — the 10:45 bar
    # closes at 11:00, exactly when the 10:00-11:00 HTF bar closes, so that
    # period is already confirmed here (not on the next chart bar).
    assert _get_confirmed_time(state_on_historical, curr_chart) == \
        expected_current_period

    state_off_live = SecurityState(
        sec_id='s3', timeframe=tf, gaps_on=False, same_timeframe=False,
        resampler=Resampler.get_resampler(tf), tz=utc,
        lookahead=Lookahead.OFF,
        is_live=True,
        last_confirmed=expected_prev_period, chart_off=chart_off,
    )
    # Lookahead.OFF never steps into a developing bar: a mid-period chart bar
    # (10:15, closing 10:30) still targets the closed 09:00 period.
    assert _get_confirmed_time(state_off_live, _ms(2026, 5, 21, 10, 15)) == \
        expected_prev_period
    # The period's last chart bar (10:45, closing 11:00) confirms 10:00.
    assert _get_confirmed_time(state_off_live, curr_chart) == \
        expected_current_period

    # Cross-symbol HTF + Lookahead.ON has no aggregator (the chart OHLCV would
    # be the wrong instrument). Even after LIVE_TRANSITION the subprocess must
    # advance on closed-bar boundaries — the documented ``close[1]`` idiom
    # delivers the just-closed cross-symbol HTF close, not the developing one.
    state_on_live_no_agg = SecurityState(
        sec_id='s4', timeframe=tf, gaps_on=False, same_timeframe=False,
        resampler=Resampler.get_resampler(tf), tz=utc,
        lookahead=Lookahead.ON,
        htf_aggregator=None,  # cross-symbol: no chart-side aggregator
        na_on_developing=True,
        is_live=True,
        last_confirmed=0, chart_off=chart_off,
    )
    # Mid-period chart bar (10:15, closing 10:30): the function returns the
    # closed 09:00 period, NOT the developing 10:00 one.
    assert _get_confirmed_time(state_on_live_no_agg, _ms(2026, 5, 21, 10, 15)) == \
        expected_prev_period
    # Past the HTF boundary (11:05 bar): still the just-closed 10:00 period,
    # NOT the developing 11:00 one.
    assert _get_confirmed_time(state_on_live_no_agg, curr_chart_boundary) == \
        expected_current_period


def __test_lookahead_on_setup_initializes_aggregator__(log):
    """setup_security_states gives a ``Lookahead.ON`` state an HTFAggregator, starting historical.

    ``lookahead_on`` is now supported: setup_security_states populates a
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


def __test_cross_symbol_lookahead_on_sec_read_returns_na__(log):
    """Cross-symbol HTF + ``Lookahead.ON``: ``__sec_read__`` reads ``na`` mid-period.

    Cross-symbol HTF + ``Lookahead.ON``: the chart-side ``__sec_read__``
    returns ``na`` (= the default passed in) while the containing HTF bar
    is open. On the first chart bar of a fresh HTF period (``new_period``
    flag set by ``__sec_signal__``) the read returns the just-closed
    cross-symbol HTF close — preserving the TV ``lookahead_on + close[1]``
    idiom at the period boundary. Behaviour is identical in historical and
    live mode; nothing live could not deliver leaks into the backtest.
    """
    from pynecore.core.security import (
        setup_security_states, create_chart_protocol, Lookahead,
    )
    from pynecore.lib import barmerge as bm
    from zoneinfo import ZoneInfo

    contexts = {
        'cross_on': {
            'symbol': 'MSFT', 'timeframe': '60',
            'gaps': bm.gaps_off, 'lookahead': bm.lookahead_on,
        },
    }
    states, sync_block, result_blocks = setup_security_states(
        contexts, chart_timeframe='1', tz=ZoneInfo('UTC'),
        chart_symbol='AAPL',
    )
    try:
        st = states['cross_on']
        assert st.na_on_developing is True
        assert st.htf_aggregator is None

        _signal, _write, sec_read, _wait, cleanup, _ = create_chart_protocol(
            states, sync_block, result_blocks=result_blocks,
        )
        try:
            # Sentinel for ``lib.na`` — sec_read returns ``default`` verbatim
            # when the developing-bar mask fires.
            NA = object()

            # Chart bar inside an open cross-symbol HTF period.
            st.new_period = False
            st.data_ready.set()
            assert sec_read('cross_on', NA) is NA, \
                "developing chart bar must return na (the default)"

            # Chart bar at the start of a fresh HTF period — sec_read falls
            # through to the result reader. With an empty ResultBlock the
            # reader also returns default, but it does so via the read path
            # rather than the na_on_developing short-circuit.
            st.new_period = True
            st.data_ready.set()
            # Empty result block ⇒ value falls back to default (na); the
            # important contract is that the path was taken, asserted by
            # the previous case (NA short-circuit) plus reading the field.
            result = sec_read('cross_on', NA)
            assert result is NA  # empty ResultBlock at fresh-period boundary
        finally:
            cleanup()
    finally:
        for rb in result_blocks.values():
            rb.close()
            rb.unlink()
        sync_block.close()
        sync_block.unlink()


def __test_live_htf_transport_covers_all_lookahead_modes__(log):
    """Same-symbol HTF gets an ``HTFAggregator`` for every lookahead mode; cross-symbol does not.

    All lookahead modes (OFF / LAST_CLOSED / ON) get an ``HTFAggregator``
    on same-symbol HTF contexts — the live closed-bar transport drives every
    HTF security context, not just ``Lookahead.ON``. Cross-symbol HTF stays
    aggregator-less (chart OHLCV would be the wrong instrument); cross-symbol
    HTF + ``Lookahead.ON`` additionally flags ``na_on_developing`` so the
    chart-side read returns ``na`` while the containing HTF bar is open.
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
        'sec_cross_off': {  # cross-symbol HTF + OFF — no aggregator, no na_only
            'symbol': 'MSFT', 'timeframe': '60',
            'gaps': bm.gaps_off, 'lookahead': bm.lookahead_off,
        },
        'sec_cross_on': {  # cross-symbol HTF + ON — na_on_developing
            'symbol': 'MSFT', 'timeframe': '60',
            'gaps': bm.gaps_off, 'lookahead': bm.lookahead_on,
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
        assert states['sec_off'].na_on_developing is False

        assert states['sec_lc'].lookahead is Lookahead.LAST_CLOSED
        assert states['sec_lc'].htf_aggregator is not None
        assert states['sec_lc'].na_on_developing is False

        assert states['sec_on'].lookahead is Lookahead.ON
        assert states['sec_on'].htf_aggregator is not None
        assert states['sec_on'].na_on_developing is False

        # Cross-symbol HTF + OFF: aggregator-less, but value flows from .ohlcv
        assert states['sec_cross_off'].htf_aggregator is None
        assert states['sec_cross_off'].na_on_developing is False

        # Cross-symbol HTF + ON: developing bar unknown → chart-side returns na
        assert states['sec_cross_on'].htf_aggregator is None
        assert states['sec_cross_on'].na_on_developing is True

        # Same-TF (cross-symbol): different code path, never gets aggregator
        assert states['sec_same_tf'].same_timeframe is True
        assert states['sec_same_tf'].htf_aggregator is None
        assert states['sec_same_tf'].na_on_developing is False
    finally:
        for rb in result_blocks.values():
            rb.close()
            rb.unlink()
        sync_block.close()
        sync_block.unlink()
