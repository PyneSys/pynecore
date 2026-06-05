"""
@pyne
"""
from pynecore.lib import close, open, format, plot, request, script, syminfo, timeframe
from pynecore.core.series import inline_series


@script.indicator(title="Inline Series With Security", shorttitle="ISS",
                  format=format.price, precision=6)
def main():
    # The security expression has its own inline_series; same-context delivery
    # must equal close[1].
    sec = request.security(syminfo.ticker, timeframe.period, inline_series(close, 1))
    # An UNGUARDED main-body inline_series must not corrupt the security delivery
    # above (issue #61: both shared one buffer, so this open-write clobbered the
    # security expression's close-write and `sec` came out as open[1]).
    m = inline_series(open, 1)
    plot(sec, "SecInline")
    plot(m, "MainInline")
    plot(close, "c")
    plot(open, "o")


def __test_inline_series_with_security__(csv_reader, runner, log):
    """A main-body inline_series must not corrupt a request.security() inline_series (issue #61).

    Issue #61: a main-body inline_series must not corrupt a request.security()
    expression's own inline_series delivery."""
    from pynecore.types.na import NA

    prev_close = None
    prev_open = None
    with csv_reader('advance_decline_ratio.csv', subdir="data") as cr:
        r = runner(cr, syminfo_override=dict(timezone="US/Eastern"))
        for i, (_candle, pv) in enumerate(r.run_iter()):
            sec = pv.get('SecInline')
            m = pv.get('MainInline')
            if prev_close is not None and not isinstance(sec, NA):
                assert sec == prev_close, \
                    f"bar {i}: security inline_series(close,1)={sec} != close[1]={prev_close}"
            if prev_open is not None and not isinstance(m, NA):
                assert m == prev_open, \
                    f"bar {i}: main inline_series(open,1)={m} != open[1]={prev_open}"
            prev_close = pv.get('c')
            prev_open = pv.get('o')

    log.info("security inline_series stays uncorrupted alongside a main-body inline_series")
