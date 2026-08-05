"""
@pyne
"""
from pynecore.lib import script, log, bar_index, array, ta, close


@script.indicator(title="Array NA Index", shorttitle="array_na_index")
def main():
    # An na index the way it arises in practice: a ta.* value that is still na
    # during warmup, converted to int. Every reference value below was measured
    # on TradingView (FX:EURUSD, 240, read at bar_index == 100); the reference
    # log is the verbatim TradingView output of the two probes.
    nai = int(ta.sma(close, 4000))

    if bar_index == 3:
        a1 = array.from_items(10, 20, 30, 40)
        log.info("get: {0} arr: {1}", array.get(a1, nai), array.join(a1, ","))

        a2 = array.from_items(10, 20, 30, 40)
        array.set(a2, nai, 99)
        log.info("set_arr: {0} size: {1}", array.join(a2, ","), array.size(a2))

        a3 = array.from_items(10, 20, 30, 40)
        log.info("remove: {0} arr: {1}", array.remove(a3, nai), array.join(a3, ","))

        a4 = array.from_items(10, 20, 30, 40)
        array.insert(a4, nai, 77)
        log.info("insert_arr: {0}", array.join(a4, ","))

        a5 = array.from_items(10, 20, 30, 40)
        sl1 = array.slice(a5, nai, 2)
        log.info("slice_nafrom: {0} size: {1}", array.join(sl1, ","), array.size(sl1))

        a6 = array.from_items(10, 20, 30, 40)
        sl2 = array.slice(a6, 1, nai)
        log.info("slice_nato: {0} size: {1}", array.join(sl2, ","), array.size(sl2))

        a7 = array.from_items(10, 20, 30, 40)
        sl3 = array.slice(a7, nai, nai)
        log.info("slice_bothna: {0} size: {1}", array.join(sl3, ","), array.size(sl3))

        a8 = array.from_items(10, 20, 30, 40)
        array.fill(a8, 5, nai, 2)
        log.info("fill_nafrom: {0}", array.join(a8, ","))

        a9 = array.from_items(10, 20, 30, 40)
        array.fill(a9, 5, 1, nai)
        log.info("fill_nato: {0}", array.join(a9, ","))

        b1 = array.from_items(10, 20, 30, 40)
        log.info("percentrank: {0} max: {1} min: {2}", array.percentrank(b1, nai),
                 array.max(b1, nai), array.min(b1, nai))

        # This line is the point of the whole test: on TradingView the script
        # keeps running after every na-indexed call, so it must be logged here
        # too. Before the na guards it never fired -- array.get raised
        # "TypeError: list indices must be integers or slices, not NA".
        log.info("still_running: true")

        e1 = array.new_int()
        log.info("empty_get: {0} size: {1}", array.get(e1, nai), array.size(e1))

        e2 = array.new_int()
        array.set(e2, nai, 9)
        log.info("empty_set_size: {0}", array.size(e2))

        e3 = array.new_int()
        log.info("empty_remove: {0} size: {1}", array.remove(e3, nai), array.size(e3))

        e4 = array.new_int()
        array.insert(e4, nai, 77)
        log.info("empty_insert: {0} size: {1}", array.join(e4, ","), array.size(e4))

        log.info("still_running: true")


def __test_array_na_index__(runner, dummy_ohlcv_iter, file_reader, log_comparator):
    """ na index tolerance across the array family """
    tv_log_out = file_reader(subdir="data", suffix=".txt")
    run_iter = runner(dummy_ohlcv_iter).run_iter()
    with log_comparator(tv_log_out):
        for _ in range(6):
            next(run_iter)
