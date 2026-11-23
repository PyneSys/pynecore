"""
Regression tests for the OHLC storage clean-up gate.

``ohlcv.restore_f32_price`` exists to undo the sub-tick error float32 storage
adds to a price. The v2 ``.ohlcv`` format leaves nothing to undo: an off-grid
price is promoted to an absolute f64 column, and an on-grid one comes back through
the reader's grid snap. Running the clean-up anyway does not clean, it truncates —
its 6-significant-digit grid is coarser than a legitimately off-grid price, so the
split-adjusted NASDAQ:AAPL 30m close ``70.90875`` read back as ``70.9087`` and put
104 of 21642 plotted bars off TradingView on the wild corpus.

``OHLCVReader.lossless_prices`` reports whether the feed needs the clean-up, and
the reader runs it only when it does.
"""
from pathlib import Path

from pynecore.core.aggregator import aggregate_ohlcv
from pynecore.core.ohlcv import OHLCVReader, OHLCVWriter, restore_f32_price
from pynecore.types.ohlcv import OHLCV, AskOHLC

# A real split-adjusted NASDAQ:AAPL 30m price: a half cent off the 0.01 grid, and
# one digit finer than the 6-significant-digit clean-up keeps at this magnitude.
_FEED_PRICE = 70.90875


def _candle(price: float) -> OHLCV:
    return OHLCV(timestamp=1735689600000, open=price, high=price,
                 low=price, close=price, volume=1000.0)


def __test_round_price_truncates_a_full_precision_feed_value__():
    """The clean-up damages a price that never lost anything to float32 storage."""
    assert restore_f32_price(_FEED_PRICE, 2) == 70.9087
    assert restore_f32_price(_FEED_PRICE, 2) != _FEED_PRICE


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


def _write_sibling_toml(path: Path, mintick: str, pricescale: int) -> None:
    path.with_suffix(".toml").write_text(
        '[symbol]\nprefix = "NASDAQ"\ndescription = "AAPL"\nticker = "AAPL"\n'
        'currency = "USD"\nperiod = "30"\ntype = "stock"\n'
        f'mintick = {mintick}\npricescale = {pricescale}\npointvalue = 1.0\ntimezone = "UTC"\n'
        '[[opening_hours]]\nday = 1\nstart = "00:00:00"\nend = "23:59:59"\n'
        '[[session_starts]]\nday = 1\ntime = "00:00:00"\n'
        '[[session_ends]]\nday = 1\ntime = "23:59:59"\n'
    )


def __test_gridless_v2_cleans_only_the_f32_delta_prices__(tmp_path: Path):
    """The exact digits of a gridless file survive; only the f32 delta noise is cleaned.

    The open is an absolute f64, and the low equals it: a zero f32 delta carries no
    noise either, so neither may lose the half cent to the clean-up.
    """
    path = tmp_path / "gridless.ohlcv"
    with OHLCVWriter(path, "30", truncate=True) as writer:
        writer.write(OHLCV(timestamp=1735689600000, open=_FEED_PRICE, high=70.95,
                           low=_FEED_PRICE, close=70.93, volume=1000.0))
    _write_sibling_toml(path, "0.01", 100)

    with OHLCVReader(path) as reader:
        assert reader.lossless_prices is False
        stored = next(iter(reader))
    assert stored.open == _FEED_PRICE
    assert stored.high == 70.95
    assert stored.low == _FEED_PRICE
    assert stored.close == 70.93


def __test_gridless_v2_open_one_ulp_above_its_high_is_snapped__(tmp_path: Path):
    """A feed open one f64 ulp above its own high reads back at the high, as on a grid.

    TradingView answered USDTRY@30 with such opens; cleaning the f32 delta high
    while leaving the exact open alone would hand the script a bar above its high.
    """
    path = tmp_path / "usdtry.ohlcv"
    with OHLCVWriter(path, "30", truncate=True) as writer:
        writer.write(OHLCV(timestamp=1735689600000, open=39.796440000000004, high=39.79644,
                           low=39.70011, close=39.75123, volume=1000.0))
    _write_sibling_toml(path, "0.00001", 100000)

    with OHLCVReader(path) as reader:
        stored = next(iter(reader))
    assert stored.open == 39.79644
    assert stored.open <= stored.high
    assert stored.low == 39.70011
    assert stored.close == 39.75123


def __test_gridless_v2_ask_is_cleaned_with_its_bid__(tmp_path: Path):
    """An ask follows the clean-up of its bid field, so a zero spread stays zero.

    Each ask is an f32 delta on the raw bid: cleaning the bid close 100.00999999977648
    to 100.01 while leaving the zero-spread ask as decoded would quote the ask below
    the bid, in the script and in every aggregate built from the file.
    """
    source = tmp_path / "quoted.ohlcv"
    with OHLCVWriter(source, "30", truncate=True) as writer:
        writer.write(OHLCV(timestamp=1735689600000, open=100.0, high=100.37, low=99.61,
                           close=100.01, volume=1.0,
                           ask=AskOHLC(100.02, 100.37, 99.63, 100.01)))
        writer.write(OHLCV(timestamp=1735691400000, open=100.01, high=100.4, low=99.9,
                           close=100.2, volume=1.0,
                           ask=AskOHLC(100.01, 100.41, 99.9, 100.2)))
    _write_sibling_toml(source, "0.01", 100)

    with OHLCVReader(source) as reader:
        first, second = list(reader)
    assert (first.close, first.high) == (100.01, 100.37)
    assert first.ask == AskOHLC(100.02, 100.37, 99.63, 100.01)
    assert second.ask == AskOHLC(100.01, 100.41, 99.9, 100.2)

    target = tmp_path / "quoted_60.ohlcv"
    aggregate_ohlcv(source, target, "60")
    # The aggregate has no symbol info of its own and reads back as stored: the zero
    # close spread it was handed is a zero delta there
    with OHLCVReader(target) as reader:
        merged = next(iter(reader))
    merged_ask = merged.ask
    assert merged_ask is not None
    assert merged_ask.close == merged.close
    assert merged_ask.high >= merged.high
