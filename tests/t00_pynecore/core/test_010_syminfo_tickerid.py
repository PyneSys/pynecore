"""
@pyne
"""
from pynecore.lib import script, syminfo as syminfo_lib, plot


@script.indicator(title="SymInfo TickerID Test")
def main():
    plot(0, syminfo_lib.tickerid)


def __test_tickerid_populated__(csv_reader, runner):
    """ syminfo.tickerid is populated as PREFIX:TICKER """
    with csv_reader('series_if_for.csv', subdir="data") as cr:
        for candle, _plot in runner(cr).run_iter():
            assert syminfo_lib.tickerid == "PYTEST:TEST"
            assert syminfo_lib.ticker == syminfo_lib.tickerid
            assert syminfo_lib.root == "TEST"
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
        for candle, _plot in r.run_iter():
            assert syminfo_lib.tickerid == "BINANCE:BTCUSDT"
            assert syminfo_lib.ticker == "BINANCE:BTCUSDT"
            assert syminfo_lib.root == "BTCUSDT"
            break
