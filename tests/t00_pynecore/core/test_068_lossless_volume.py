"""
Regression tests for the volume storage clean-up gate.

``ohlcv.restore_f32_volume`` exists to undo the sub-lot error float32 storage
adds to a volume. The v2 ``.ohlcv`` format keeps volume as an absolute f64, so
there is nothing to undo — and running the clean-up anyway does not clean, it
truncates: its 5-decimal floor is finer than the float32 grid above ~1000, so a
Binance 30m volume of ``9362.123462`` came back as ``9362.12346``. A session
``ta.vwap`` built on that volume then diverged from TradingView for the rest of
the day (measured on the wild corpus: 249 divergent bars, all of them on the 9
days that carried a truncated volume, none anywhere else).

``OHLCVReader.lossless_volume`` reports whether the feed needs the clean-up, and
the runner skips it when it does not.
"""
import struct
from pathlib import Path

from pynecore.core.ohlcv import OHLCVReader, OHLCVWriter, restore_f32_volume
from pynecore.types.ohlcv import OHLCV

# A real Binance BTCUSDT 30m volume with six decimals — more than the 5-decimal
# floor keeps, and far finer than the float32 grid at this magnitude.
_FEED_VOLUME = 9362.123462


def _f32(x: float) -> float:
    """Round-trip through float32, mimicking legacy ``.ohlcv`` storage."""
    return struct.unpack('f', struct.pack('f', x))[0]


def __test_round_volume_truncates_a_full_precision_feed_value__():
    """The clean-up damages a volume that never went through float32 storage."""
    assert restore_f32_volume(_FEED_VOLUME) == 9362.12346
    assert restore_f32_volume(_FEED_VOLUME) != _FEED_VOLUME


def __test_round_volume_still_cleans_float32_dust__():
    """On a genuinely float32-stored volume the clean-up recovers the feed decimal."""
    assert restore_f32_volume(_f32(0.56881)) == 0.56881
    assert _f32(0.56881) != 0.56881


def __test_v2_feed_reports_lossless_volume_and_reads_back_exactly__(tmp_path: Path):
    """A v2 file stores volume as absolute f64: no clean-up needed, none applied."""
    path = tmp_path / "lossless.ohlcv"
    candle = OHLCV(timestamp=1735689600000, open=93761.9, high=93800.0,
                   low=93700.0, close=93780.0, volume=_FEED_VOLUME)
    with OHLCVWriter(path, "30", truncate=True) as writer:
        writer.write(candle)

    with OHLCVReader(path) as reader:
        assert reader.lossless_volume is True
        stored = next(iter(reader))
    assert stored.volume == _FEED_VOLUME
