"""
Regression tests for ``script_runner._round_price``.

The float32 ``.ohlcv`` storage adds sub-tick error to OHLC prices, which the
runner cleans before the script sees ``open/high/low/close``. The original
clean-up rounded to 6 significant digits; for high-priced assets (BTC ~94000)
that only reaches 1 decimal (93898.05 -> 93898.1) and discarded the real
mintick-aligned precision, flipping hysteresis-latch indicators by one bar.

The fix rounds to the FINER of 6-sig and the symbol's mintick decimals:
``max(5 - floor(log10|price|), tick_decimals)``. This recovers the exact
mintick value at high prices while keeping the historical sub-mintick behaviour
at low prices (where TradingView itself carries sub-mintick precision).
"""
import struct

from pynecore.core.script_runner import _round_price


def _f32(x: float) -> float:
    """Round-trip through float32, mimicking ``.ohlcv`` storage."""
    return struct.unpack('f', struct.pack('f', x))[0]


def __test_high_price_snaps_to_mintick_not_six_sig__():
    """A BTC-magnitude price keeps mintick decimals instead of collapsing to 6 sig digits."""
    # 6-sig alone gives 1 decimal here (the old bug); mintick (2) is finer and wins.
    assert _round_price(93898.05, 2) == 93898.05
    assert _round_price(93898.05, None) == 93898.1  # documents the old 6-sig behaviour


def __test_float32_storage_recovers_to_mintick__():
    """Rounding float32-stored OHLC to mintick decimals recovers the exact tick-aligned price."""
    # This is the SSL-latch case: 6-sig would round to the integer (109548).
    assert _round_price(_f32(109547.84), 2) == 109547.84
    assert _round_price(_f32(109547.84), None) == 109548.0
    assert _round_price(_f32(93761.9), 2) == 93761.9


def __test_low_price_keeps_sub_mintick_precision__():
    """A small price keeps 6-sig (sub-mintick) precision; the mintick grid must NOT truncate it."""
    # ganzalgo_v2_pro regression: SL 4.38075 must stay 4.38075, not snap to 4.381.
    assert _round_price(4.38075, 3) == 4.38075
    assert round(4.38075, 3) != 4.38075  # what a pure round-to-mintick fix would corrupt


def __test_no_mintick_falls_back_to_six_sig__():
    """With no real mintick (``None``), the magnitude-relative 6-sig clean-up is used."""
    assert _round_price(93898.05, None) == 93898.1
    assert _round_price(4.1199998856, None) == 4.12  # cleans float32 dust at low magnitude


def __test_never_coarser_than_six_sig__():
    """When mintick is coarser than the 6-sig grid, the finer 6-sig grid is kept."""
    # 12.345678 -> 6-sig is 4 decimals; mintick 0.01 (2) is coarser, so max keeps 4.
    assert _round_price(12.345678, 2) == round(12.345678, 4)


def __test_zero_price_returns_zero__():
    """Zero is returned unchanged (``log10`` guard)."""
    assert _round_price(0.0, 2) == 0.0
    assert _round_price(0.0, None) == 0.0
