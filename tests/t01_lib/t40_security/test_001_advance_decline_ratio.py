"""
@pyne
"""
from pynecore.lib import close, format, plot, request, script, timeframe


@script.indicator(title="Advance Decline Ratio", shorttitle="ADR", format=format.price, precision=2)
def main():
    def ratio(t1, t2, source):
        return request.security(t1, timeframe.period, source) / request.security(t2, timeframe.period, source)

    plot(ratio('USI:ADVN.NY', 'USI:DECL.NY', close))


def __test_advance_decline_ratio__(csv_reader, runner, dict_comparator, log):
    """ Advance Decline Ratio """
    from pathlib import Path

    data_dir = Path(__file__).parent / "data"
    security_data = {
        "USI:ADVN.NY": str(data_dir / "USI_ADVN_NY"),
        "USI:DECL.NY": str(data_dir / "USI_DECL_NY"),
    }

    # TV forward-fills from previous full trading day on half-day sessions
    # (July 3, Black Friday, Christmas Eve). Our implementation uses actual data.
    # See: docs/pynecore/request_security/research.md §12
    half_day_ts = {
        1562160600, 1575037800, 1577197800, 1637937000, 1669386600,
        1688391000, 1700836200, 1720013400, 1732890600, 1735050600,
        1751549400, 1764340200, 1766586600,
    }

    with csv_reader('advance_decline_ratio.csv', subdir="data") as cr:
        r = runner(
            cr,
            syminfo_override=dict(timezone="US/Eastern"),
            security_data=security_data,
        )
        for i, (candle, plot_values) in enumerate(r.run_iter()):
            if candle.timestamp in half_day_ts:
                continue
            dict_comparator(plot_values, candle.extra_fields)
