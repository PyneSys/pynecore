"""
@pyne
"""
from pynecore.lib import script, syminfo as syminfo_lib, ticker, plot


@script.indicator(title="SymInfo TickerID Test")
def main():
    plot(0, syminfo_lib.tickerid)
    return {
        "ticker": syminfo_lib.ticker,
        "prefix": syminfo_lib.prefix,
        "tickerid": syminfo_lib.tickerid,
        "root": syminfo_lib.root,
        "ticker_of_full": syminfo_lib.ticker("BINANCE:ETHUSDT"),
        "prefix_of_full": syminfo_lib.prefix("BINANCE:ETHUSDT"),
        "ticker_of_bare": syminfo_lib.ticker("ETHUSDT"),
        "prefix_of_bare": syminfo_lib.prefix("ETHUSDT"),
        "ticker_of_heikinashi": syminfo_lib.ticker(ticker.heikinashi("BINANCE:ETHUSDT")),
        "prefix_of_heikinashi": syminfo_lib.prefix(ticker.heikinashi("BINANCE:ETHUSDT")),
        "ticker_of_own": syminfo_lib.ticker(syminfo_lib.tickerid),
        "ticker_of_spread": syminfo_lib.ticker("BINANCE:ETHUSDT/BINANCE:BTCUSDT"),
        "prefix_of_spread": syminfo_lib.prefix("BINANCE:ETHUSDT/BINANCE:BTCUSDT"),
    }


def __test_tickerid_populated__(csv_reader, runner):
    """ syminfo.tickerid is PREFIX:TICKER, syminfo.ticker and syminfo.root the name alone """
    with csv_reader('series_if_for.csv', subdir="data") as cr:
        for candle, values in runner(cr).run_iter():
            assert values["tickerid"] == "PYTEST:TEST"
            assert values["ticker"] == "TEST"
            assert values["prefix"] == "PYTEST"
            assert values["root"] == "TEST"
            break


def __test_tickerid_with_custom_syminfo__(script_path, module_key, syminfo, csv_reader):
    """ syminfo.tickerid updates when syminfo is overridden """
    import sys
    from pynecore.core.script_runner import ScriptRunner

    stem = script_path.stem
    for key in [module_key, stem]:
        sys.modules.pop(key, None)

    syminfo.prefix = "BINANCE"
    syminfo.ticker = "BTCUSDT"

    with csv_reader('series_if_for.csv', subdir="data") as cr:
        r = ScriptRunner(script_path, cr, syminfo)
        for candle, values in r.run_iter():
            assert values["tickerid"] == "BINANCE:BTCUSDT"
            assert values["ticker"] == "BTCUSDT"
            assert values["prefix"] == "BINANCE"
            assert values["root"] == "BTCUSDT"
            break


def __test_ticker_and_prefix_of_another_symbol__(csv_reader, runner):
    """ The function forms split a symbol at its first colon, as TradingView does

    Measured on TradingView (CAPITALCOM:EURUSD chart): a symbol without a colon
    has neither part, a chart-type ticker answers for its base symbol, and a
    spread keeps its later colons in the ticker part.
    """
    with csv_reader('series_if_for.csv', subdir="data") as cr:
        for candle, values in runner(cr).run_iter():
            assert values["ticker_of_full"] == "ETHUSDT"
            assert values["prefix_of_full"] == "BINANCE"
            assert values["ticker_of_bare"] == ""
            assert values["prefix_of_bare"] == ""
            assert values["ticker_of_heikinashi"] == "ETHUSDT"
            assert values["prefix_of_heikinashi"] == "BINANCE"
            assert values["ticker_of_own"] == "TEST"
            assert values["ticker_of_spread"] == "ETHUSDT/BINANCE:BTCUSDT"
            assert values["prefix_of_spread"] == "BINANCE"
            break
