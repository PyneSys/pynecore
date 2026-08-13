"""
@pyne
"""
from pynecore.lib import close, high, format, plot, request, script, syminfo, timeframe


@script.indicator(title="Empty Security Symbol", shorttitle="ESS", format=format.price, precision=6)
def main():
    # Pine reads an empty symbol as the chart's own instrument, so these are the
    # same request written three ways
    empty_sym = request.security('', timeframe.period, close)
    bare_sym = request.security(syminfo.ticker, timeframe.period, close)
    empty_both = request.security('', '', high)

    plot(empty_sym, title="EmptySym")
    plot(bare_sym, title="BareSym")
    plot(empty_both, title="EmptyBoth")


def __test_empty_symbol_is_the_chart_symbol__(csv_reader, runner, log):
    """An empty security symbol resolves to the chart instrument, not another one"""
    from pynecore import lib
    from pynecore.types.na import NA

    bars = 0
    with csv_reader('advance_decline_ratio.csv', subdir="data") as cr:
        r = runner(
            cr,
            syminfo_override=dict(timezone="US/Eastern"),
        )

        for i, (_candle, plot_values) in enumerate(r.run_iter()):
            bars += 1
            empty_sym = plot_values.get('EmptySym')
            bare_sym = plot_values.get('BareSym')
            empty_both = plot_values.get('EmptyBoth')

            if not isinstance(empty_sym, NA):
                assert empty_sym == lib.close, \
                    f"bar {i}: EmptySym={empty_sym} != close={lib.close}"
            if not isinstance(bare_sym, NA):
                assert empty_sym == bare_sym, \
                    f"bar {i}: EmptySym={empty_sym} != BareSym={bare_sym}"
            if not isinstance(empty_both, NA):
                assert empty_both == lib.high, \
                    f"bar {i}: EmptyBoth={empty_both} != high={lib.high}"

    assert bars > 0, "no bars were run"
    log.info("Empty security symbol resolved to the chart instrument on %d bars", bars)


def __test_empty_and_qualified_symbols_are_same_symbol_htf__(syminfo, log):
    """Every spelling of the chart instrument keeps the same-symbol HTF transport"""
    from zoneinfo import ZoneInfo
    from pynecore.core.security import setup_security_states
    from pynecore.lib import barmerge as bm

    # Chart is ``PYTEST:TEST`` at 5m; every context below asks for the SAME
    # instrument at 60m, written empty, bare and exchange qualified. Only a real
    # cross-symbol request may lose the chart-side aggregator (and get ``na`` on
    # developing bars under ``lookahead_on``).
    contexts = {
        sec_id: {'symbol': sym, 'timeframe': '60',
                 'gaps': bm.gaps_off, 'lookahead': bm.lookahead_on}
        for sec_id, sym in (('empty', ''), ('bare', 'TEST'),
                            ('qualified', 'PYTEST:TEST'), ('cross', 'OTHER'))
    }
    states, sync_block, result_blocks = setup_security_states(
        contexts, chart_timeframe='5', tz=ZoneInfo('UTC'),
        chart_symbol=str(syminfo.ticker), chart_syminfo=syminfo,
    )
    try:
        for sec_id in ('empty', 'bare', 'qualified'):
            state = states[sec_id]
            assert state.htf_aggregator is not None, f"{sec_id}: no HTF aggregator"
            assert not state.na_on_developing, f"{sec_id}: na on developing HTF bars"
        assert states['cross'].htf_aggregator is None, "cross-symbol got an aggregator"
        assert states['cross'].na_on_developing, "cross-symbol lookahead_on is not na-gated"
    finally:
        for rb in result_blocks.values():
            rb.close()
            rb.unlink()
        sync_block.close()
        sync_block.unlink()

    log.info("Empty and qualified chart symbols classified as same-symbol HTF")
