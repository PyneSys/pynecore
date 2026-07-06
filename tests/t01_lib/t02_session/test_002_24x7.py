"""
@pyne
"""
from pynecore.lib import script, session, plot


@script.indicator(title="Session 24x7", shorttitle="session_24x7")
def main():
    plot(1 if session.isfirstbar else 0, "isfirstbar")
    plot(1.5 if session.isfirstbar_regular else 0, "isfirstbar_regular")
    plot(2 if session.islastbar else 0, "islastbar")
    plot(2.5 if session.islastbar_regular else 0, "islastbar_regular")


def __test_session_24x7__(csv_reader, runner, dict_comparator, log):
    """ On a 24/7 market the session ends at midnight (00:00 = 24:00), so
    ``islastbar_regular`` must fire on the last bar of the day, not never. """
    from pathlib import Path
    syminfo_path = Path(__file__).parent / "data" / "session_24x7.toml"
    with csv_reader('session_24x7.csv', subdir="data") as cr:
        for candle, _plot in runner(cr, syminfo_path=syminfo_path).run_iter():
            dict_comparator(_plot, candle.extra_fields)
