"""
@pyne
"""
from pynecore.lib import script, session, plot


@script.indicator(title="Session auction close", shorttitle="session_auction")
def main():
    plot(1 if session.isfirstbar else 0, "isfirstbar")
    plot(1.5 if session.isfirstbar_regular else 0, "isfirstbar_regular")
    plot(2 if session.islastbar else 0, "islastbar")
    plot(2.5 if session.islastbar_regular else 0, "islastbar_regular")


def __test_session_auction_close__(csv_reader, runner, dict_comparator, log):
    """ Auction venues stamp the settlement print exactly AT the session end
    (VN30F1M: the 14:45 ATC row; the series jumps 14:29 -> 14:45), so a bar
    STARTING at the session end is the session's LAST bar — it must not be
    swallowed by the 24/7 midnight-rollover branch (rubycell/pynecore#29). """
    from pathlib import Path
    syminfo_path = Path(__file__).parent / "data" / "session_auction.toml"
    with csv_reader('session_auction.csv', subdir="data") as cr:
        for candle, _plot in runner(cr, syminfo_path=syminfo_path).run_iter():
            dict_comparator(_plot, candle.extra_fields)
