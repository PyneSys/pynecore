"""
@pyne
"""
from pynecore.lib import script, plot, bid, ask


@script.indicator(title="Bid/Ask", shorttitle="bidask")
def main():
    plot(bid, "bid")
    plot(ask, "ask")
    plot(bid[1], "bid_prev")
    plot(ask[1], "ask_prev")


def __test_bid_ask__(runner, log):
    """ Bid / Ask builtin source: real on bars that quote the ask side, na otherwise """
    from datetime import datetime, UTC

    from pynecore.types.ohlcv import OHLCV, AskOHLC
    from pynecore.types.na import isna_num

    base = int(datetime.fromisoformat("2025-01-01T00:00:00").replace(tzinfo=UTC).timestamp())
    nan = float("nan")

    # Bar 1 quotes the ask side, so bid is the bar's (bid-side) close and ask its ask close.
    # Bar 0 carries no ask, bar 2 an all-NaN one (an unquoted bar of a file with ask
    # columns): both read na on both sides. Extra fields named bid/ask stay user data.
    candles = [
        OHLCV(timestamp=base, open=1.0, high=2.0, low=0.5, close=1.1, volume=10.0,
              extra_fields={"bid": 9.0, "ask": 9.5}),
        OHLCV(timestamp=base + 60, open=1.1, high=2.5, low=1.0, close=2.2, volume=10.0,
              ask=AskOHLC(1.2, 2.6, 1.1, 2.3)),
        OHLCV(timestamp=base + 120, open=2.2, high=3.5, low=2.0, close=3.3, volume=10.0,
              ask=AskOHLC(nan, nan, nan, nan)),
    ]

    plots = [_plot for _candle, _plot in runner(iter(candles)).run_iter()]

    assert isna_num(plots[0]["bid"]) and isna_num(plots[0]["ask"])
    assert plots[1]["bid"] == 2.2, plots[1]["bid"]
    assert plots[1]["ask"] == 2.3, plots[1]["ask"]
    assert isna_num(plots[1]["bid_prev"]) and isna_num(plots[1]["ask_prev"])
    assert isna_num(plots[2]["bid"]) and isna_num(plots[2]["ask"])
    assert plots[2]["bid_prev"] == 2.2 and plots[2]["ask_prev"] == 2.3
