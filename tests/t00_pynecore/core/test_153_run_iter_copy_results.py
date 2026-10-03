"""
@pyne
"""
from pynecore.lib import script, strategy, close, plot, bar_index


@script.strategy("Run iter copy results", initial_capital=1000000)
def main():
    if bar_index % 4 == 0:
        strategy.entry("L", strategy.long, qty=1)
    if bar_index % 4 == 2:
        strategy.close("L")
    plot(close, "close")


def __test_results_can_be_kept__(csv_reader, runner):
    """ By default every bar yields its own plot_data and new_trades, so they can be kept """
    with csv_reader('series_if_for.csv', subdir="data") as cr:
        read_in_loop = [(p["close"], len(t)) for _c, p, t in runner(cr).run_iter(copy_results=False)]
    with csv_reader('series_if_for.csv', subdir="data") as cr:
        kept = list(runner(cr).run_iter())

    assert [(p["close"], len(t)) for _c, p, t in kept] == read_in_loop
    assert sum(n for _close, n in read_in_loop) > 0
    assert len({id(p) for _c, p, _t in kept}) == len(kept)


def __test_copy_results_false_yields_the_runner_containers__(csv_reader, runner):
    """ ``copy_results=False`` yields the runner's own dict and list, refilled every bar """
    with csv_reader('series_if_for.csv', subdir="data") as cr:
        rows = list(runner(cr).run_iter(copy_results=False))

    assert len({id(p) for _c, p, _t in rows}) == 1
    assert len({id(t) for _c, _p, t in rows}) == 1
