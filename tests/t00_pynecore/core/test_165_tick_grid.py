"""
Tick-grid derivation: ``mintick == minmove / pricescale`` for every tick shape.

A tick that is not an integer reciprocal (Bybit options trade on a ``5.0`` tick)
needs ``minmove != 1``; deriving ``pricescale = round(1 / mintick)`` alone yields
``0`` or a wrong grid for those ticks.
"""
from pathlib import Path

import ccxt
import pytest

from pynecore.core.ohlcv import OHLCVWriter
from pynecore.core.syminfo import tick_grid
from pynecore.providers.ccxt import CCXTError, CCXTProvider


@pytest.mark.parametrize(("mintick", "expected"), [
    (0.01, (1, 100)),
    (1.0, (1, 1)),
    (0.05, (1, 20)),
    (0.25, (1, 4)),
    (1e-8, (1, 100_000_000)),
    (5.0, (5, 1)),
    (10.0, (10, 1)),
    (2.5, (25, 10)),
    (0.03, (3, 100)),
    (0.0075, (75, 10_000)),
    (0.000075, (75, 1_000_000)),
    (0.00009, (9, 100_000)),
    (7.5e-9, (3, 400_000_000)),
])
def __test_tick_grid_reconstructs_mintick__(mintick: float, expected: tuple[int, int]):
    """Every tick maps to an integer pair whose quotient is the tick itself."""
    minmove, pricescale = tick_grid(mintick)
    assert (minmove, pricescale) == expected
    assert minmove / pricescale == pytest.approx(mintick, rel=1e-12)


@pytest.mark.parametrize("mintick", [0.0, -0.01, 2.0 / 3.0])
def __test_tick_grid_rejects_inexpressible_ticks__(mintick: float):
    """A non-positive tick or one without an exact decimal grid fails loudly."""
    with pytest.raises(ValueError):
        tick_grid(mintick)


class _StubClient:
    """Minimal CCXT client exposing one Bybit-option-shaped market."""

    id = 'bybit'
    precisionMode = ccxt.TICK_SIZE

    def __init__(self, symbol: str, price_tick: float):
        self.markets = {
            symbol: {
                'symbol': symbol,
                'id': 'BTC-30OCT26-102000-C-USDT',
                'base': 'BTC',
                'quote': 'USDT',
                'precision': {'amount': 0.01, 'price': price_tick},
                'limits': {'amount': {'min': 0.01}, 'price': {'min': price_tick}},
                'contractSize': 1.0,
                'taker': 0.0003,
                'maker': 0.0003,
                'info': {'symbol': 'BTC-30OCT26-102000-C-USDT', 'tickSize': str(price_tick)},
            },
        }

    def load_markets(self) -> dict:
        return self.markets


def _provider(symbol: str, price_tick: float) -> CCXTProvider:
    provider = CCXTProvider(symbol=f"bybit:{symbol}", timeframe="60")
    provider._client = _StubClient(symbol, price_tick)  # type: ignore[assignment]
    return provider


def __test_ccxt_option_tick_five_yields_minmove_grid__():
    """A Bybit option (tick 5) gets ``minmove=5, pricescale=1`` instead of a CCXTError."""
    symbol = 'BTC/USDT:USDT-261030-102000-C'
    syminfo = _provider(symbol, 5.0).update_symbol_info()
    assert syminfo.mintick == 5.0
    assert (syminfo.minmove, syminfo.pricescale) == (5, 1)


def __test_ccxt_inexpressible_tick_raises_ccxt_error__():
    """A tick with no exact integer grid still fails as a clean provider error."""
    symbol = 'BTC/USDT:USDT-261030-102000-C'
    with pytest.raises(CCXTError):
        _provider(symbol, 2.0 / 3.0).update_symbol_info()


def __test_ohlcv_analyzer_coarse_tick_gets_valid_grid__(tmp_path: Path):
    """A detected tick above 1 maps to ``minmove = tick``, never to ``pricescale = 0``."""
    with OHLCVWriter(tmp_path / "coarse.ohlcv", "60", truncate=True) as writer:
        # The histogram grid tops out at 1.0 and fits these changes exactly; the
        # frequency estimator's coarser common divisor must still win.
        writer._price_changes = [10.0, 20.0, 30.0, 10.0, 40.0, 50.0, 10.0, 20.0, 60.0, 30.0, 70.0]
        writer._analyze_tick_size()
        assert writer.analyzed_tick_size == 10.0
        assert (writer.analyzed_min_move, writer.analyzed_price_scale) == (10, 1)


def __test_ohlcv_analyzer_rejects_contradicted_coarse_grid__(tmp_path: Path):
    """One change below the frequency estimate's grid keeps the finer histogram tick."""
    with OHLCVWriter(tmp_path / "contradicted.ohlcv", "60", truncate=True) as writer:
        writer._price_changes = [20.0, 40.0, 60.0, 20.0, 80.0, 100.0, 20.0, 40.0, 120.0, 60.0, 140.0, 1.0]
        writer._analyze_tick_size()
        assert writer.analyzed_tick_size == 1.0
        assert (writer.analyzed_min_move, writer.analyzed_price_scale) == (1, 1)


def __test_ohlcv_analyzer_fallback_keeps_consistent_coarse_grid__(tmp_path: Path):
    """Changes too coarse for every histogram candidate still resolve to their common grid."""
    with OHLCVWriter(tmp_path / "fallback.ohlcv", "60", truncate=True) as writer:
        writer._price_changes = [20.0, 40.0, 60.0, 20.0, 80.0, 100.0, 20.0, 40.0, 120.0, 60.0, 140.0, 20.0]
        writer._analyze_tick_size()
        assert writer.analyzed_tick_size == 20.0
        assert (writer.analyzed_min_move, writer.analyzed_price_scale) == (20, 1)


def __test_ohlcv_analyzer_fallback_rejects_contradicted_coarse_grid__(tmp_path: Path):
    """A change off the frequency grid makes the fallback keep the decimal-precision tick."""
    with OHLCVWriter(tmp_path / "fallback_contradicted.ohlcv", "60", truncate=True) as writer:
        writer._price_changes = [20.0, 40.0, 60.0, 20.0, 80.0, 100.0, 20.0, 40.0, 120.0, 60.0, 140.0, 21.0]
        writer._analyze_tick_size()
        assert writer.analyzed_tick_size == 1.0
        assert (writer.analyzed_min_move, writer.analyzed_price_scale) == (1, 1)
