"""
@pyne
"""
from pynecore.lib import script, plot, bid, ask


@script.indicator(title="Bid/Ask", shorttitle="bidask")
def main():
    plot(bid, "bid")
    plot(ask, "ask")
    plot(bid[1], "bid_prev")


def __test_bid_ask__(runner, log):
    """ Bid / Ask builtin source: always na (PyneCore has no tick data) """
    from datetime import datetime, UTC

    from pynecore.types.ohlcv import OHLCV
    from pynecore.types.na import isna_num

    base = int(datetime.fromisoformat("2025-01-01T00:00:00").replace(tzinfo=UTC).timestamp())

    # Even when the data carries bid/ask columns they are ignored: tick data is
    # unsupported, so bid/ask (and their history) are na on every bar.
    candles = [
        OHLCV(timestamp=base, open=1.0, high=2.0, low=0.5, close=1.1, volume=10.0),
        OHLCV(timestamp=base + 60, open=1.1, high=2.5, low=1.0, close=2.2, volume=10.0,
              extra_fields={"bid": 2.1, "ask": 2.3}),
        OHLCV(timestamp=base + 120, open=2.2, high=3.5, low=2.0, close=3.3, volume=10.0),
    ]

    for _candle, _plot in runner(iter(candles)).run_iter():
        assert isna_num(_plot["bid"]), f"bid should be na, got {_plot['bid']!r}"
        assert isna_num(_plot["ask"]), f"ask should be na, got {_plot['ask']!r}"
        assert isna_num(_plot["bid_prev"]), f"bid[1] should be na, got {_plot['bid_prev']!r}"
