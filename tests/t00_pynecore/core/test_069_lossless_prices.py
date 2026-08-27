"""
Regression tests for the OHLC storage clean-up gate.

``script_runner._round_price`` exists to undo the sub-tick error float32 storage
adds to a price. The v2 ``.ohlcv`` format leaves nothing to undo: an off-grid
price is promoted to an absolute f64 column, and an on-grid one comes back through
the reader's grid snap. Running the clean-up anyway does not clean, it truncates —
its 6-significant-digit grid is coarser than a legitimately off-grid price, so the
split-adjusted NASDAQ:AAPL 30m close ``70.90875`` read back as ``70.9087`` and put
104 of 21642 plotted bars off TradingView on the wild corpus.

``OHLCVReader.lossless_prices`` reports whether the feed needs the clean-up, and
the runner skips it when it does not.
"""
from pathlib import Path

from pynecore.core.ohlcv import OHLCVReader, OHLCVWriter
from pynecore.core.script_runner import _round_price
from pynecore.types.ohlcv import OHLCV

# A real split-adjusted NASDAQ:AAPL 30m price: a half cent off the 0.01 grid, and
# one digit finer than the 6-significant-digit clean-up keeps at this magnitude.
_FEED_PRICE = 70.90875


def _candle(price: float) -> OHLCV:
    return OHLCV(timestamp=1735689600000, open=price, high=price,
                 low=price, close=price, volume=1000.0)


def __test_round_price_truncates_a_full_precision_feed_value__():
    """The clean-up damages a price that never lost anything to float32 storage."""
    assert _round_price(_FEED_PRICE, 2) == 70.9087
    assert _round_price(_FEED_PRICE, 2) != _FEED_PRICE


def __test_v2_feed_with_a_grid_reports_lossless_prices__(tmp_path: Path):
    """An off-grid price is stored as absolute f64 and read back untouched."""
    path = tmp_path / "offgrid.ohlcv"
    with OHLCVWriter(path, "30", minmove=1, pricescale=100, truncate=True) as writer:
        writer.write(_candle(_FEED_PRICE))

    with OHLCVReader(path) as reader:
        assert reader.lossless_prices is True
        stored = next(iter(reader))
    assert stored.close == _FEED_PRICE
    assert stored.open == _FEED_PRICE


def __test_v2_feed_without_a_grid_still_needs_the_clean_up__(tmp_path: Path):
    """No declared grid means unchecked f32 deltas, so the gate stays closed."""
    path = tmp_path / "gridless.ohlcv"
    with OHLCVWriter(path, "30", truncate=True) as writer:
        writer.write(_candle(_FEED_PRICE))

    with OHLCVReader(path) as reader:
        assert reader.lossless_prices is False
