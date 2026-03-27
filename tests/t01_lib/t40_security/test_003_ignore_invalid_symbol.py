"""
@pyne
"""
from pynecore.lib import close, format, na, plot, request, script, timeframe


@script.indicator(title="Ignore Invalid Symbol Test", shorttitle="IIS", format=format.price)
def main():
    # Non-existent symbol with ignore_invalid_symbol=True → should return na
    sec_val = request.security("NONEXISTENT:SYMBOL", timeframe.period, close,
                               ignore_invalid_symbol=True)
    plot(sec_val, title="SecVal")


def __test_ignore_invalid_symbol__(csv_reader, runner, log):
    """ignore_invalid_symbol=True returns na for non-existent symbols without error"""
    from pynecore.types.na import NA

    with csv_reader('advance_decline_ratio.csv', subdir="data") as cr:
        r = runner(
            cr,
            syminfo_override=dict(timezone="US/Eastern"),
        )

        for i, (candle, plot_values) in enumerate(r.run_iter()):
            sec_val = plot_values.get('SecVal')
            assert isinstance(sec_val, NA), \
                f"bar {i}: expected na but got {sec_val}"

    log.info("ignore_invalid_symbol test passed — na returned for missing symbol")
