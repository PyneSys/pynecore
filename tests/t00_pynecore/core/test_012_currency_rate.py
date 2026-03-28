"""
@pyne
"""
import struct
import tempfile
from math import isnan
from pathlib import Path

from pynecore.core.currency import CurrencyRateProvider
from pynecore.core.ohlcv_file import RECORD_SIZE


def _create_test_ohlcv(dir_path: Path, name: str, bars: list[tuple[int, float]],
                        currency: str, basecurrency: str, period: str = "1D") -> str:
    """
    Create a test OHLCV binary file + TOML in the given directory.

    :param bars: List of (timestamp, close) tuples
    :return: Path string to the OHLCV file (without extension for security_data key)
    """
    base_path = dir_path / name
    ohlcv_path = base_path.with_suffix('.ohlcv')
    toml_path = base_path.with_suffix('.toml')

    # Write binary OHLCV: Ifffff = timestamp, open, high, low, close, volume
    with open(ohlcv_path, 'wb') as f:
        for ts, close in bars:
            f.write(struct.pack('Ifffff', ts, close, close, close, close, 100.0))

    # Write minimal TOML
    toml_content = f"""[symbol]
prefix = "TEST"
description = "{name}"
ticker = "{name}"
currency = "{currency}"
basecurrency = "{basecurrency}"
period = "{period}"
type = "forex"
mintick = 0.00001000
pricescale = 100000
pointvalue = 1.00000000
timezone = "UTC"

[[opening_hours]]
day = 1
start = "00:00:00"
end = "23:59:59"

[[session_starts]]
day = 1
time = "00:00:00"

[[session_ends]]
day = 1
time = "23:59:59"
"""
    with open(toml_path, 'w') as f:
        f.write(toml_content)

    return str(base_path)


def __test_same_currency_returns_one__(log):
    """currency_rate returns 1.0 when from == to"""
    provider = CurrencyRateProvider({})
    assert provider.get_rate("USD", "USD", 1000000) == 1.0
    assert provider.get_rate("EUR", "EUR", 1000000) == 1.0
    assert provider.get_rate("BTC", "BTC", 1000000) == 1.0


def __test_none_currency_returns_nan__(log):
    """currency_rate returns nan when NONE is involved"""
    provider = CurrencyRateProvider({})
    assert isnan(provider.get_rate("NONE", "USD", 1000000))
    assert isnan(provider.get_rate("EUR", "NONE", 1000000))
    assert isnan(provider.get_rate("NONE", "NONE", 1000000))


def __test_direct_pair_lookup__(log):
    """Direct pair: basecurrency=EUR, currency=USD → close = EUR/USD rate"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        bars = [
            (1000000, 1.0850),
            (1086400, 1.0900),
            (1172800, 1.0875),
        ]
        path = _create_test_ohlcv(tmpdir, "EURUSD", bars, currency="USD", basecurrency="EUR")

        provider = CurrencyRateProvider({"FX": path})

        # Exact timestamp match
        rate = provider.get_rate("EUR", "USD", 1000000)
        assert abs(rate - 1.0850) < 0.001, f"Expected ~1.085, got {rate}"

        # Second bar
        rate = provider.get_rate("EUR", "USD", 1086400)
        assert abs(rate - 1.0900) < 0.001, f"Expected ~1.09, got {rate}"


def __test_inverse_pair_lookup__(log):
    """Inverse: only EUR/USD data exists, request USD/EUR → 1/close"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        bars = [(1000000, 1.0850)]
        path = _create_test_ohlcv(tmpdir, "EURUSD", bars, currency="USD", basecurrency="EUR")

        provider = CurrencyRateProvider({"FX": path})

        rate = provider.get_rate("USD", "EUR", 1000000)
        expected = 1.0 / 1.0850
        assert abs(rate - expected) < 0.001, f"Expected ~{expected:.4f}, got {rate}"


def __test_nearest_prior_timestamp__(log):
    """Binary search returns the closest prior bar, not future"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        bars = [
            (1000000, 1.0800),
            (1086400, 1.0900),
            (1172800, 1.1000),
        ]
        path = _create_test_ohlcv(tmpdir, "EURUSD", bars, currency="USD", basecurrency="EUR")

        provider = CurrencyRateProvider({"FX": path})

        # Between bar 0 and bar 1 → should return bar 0 close
        rate = provider.get_rate("EUR", "USD", 1050000)
        assert abs(rate - 1.0800) < 0.001, f"Expected ~1.08 (bar 0), got {rate}"

        # After last bar → should return last close
        rate = provider.get_rate("EUR", "USD", 9999999)
        assert abs(rate - 1.1000) < 0.001, f"Expected ~1.10 (last bar), got {rate}"

        # Before first bar → should return nan
        rate = provider.get_rate("EUR", "USD", 500000)
        assert isnan(rate), f"Expected nan (before data), got {rate}"


def __test_no_data_returns_nan__(log):
    """Unknown currency pair returns nan"""
    provider = CurrencyRateProvider({})
    assert isnan(provider.get_rate("EUR", "JPY", 1000000))


def __test_multiple_pairs__(log):
    """Multiple OHLCV files provide different currency pairs"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        path_eu = _create_test_ohlcv(
            tmpdir, "EURUSD", [(1000000, 1.0850)],
            currency="USD", basecurrency="EUR",
        )
        path_bt = _create_test_ohlcv(
            tmpdir, "BTCUSDT", [(1000000, 65000.0)],
            currency="USDT", basecurrency="BTC",
        )

        provider = CurrencyRateProvider({"fx": path_eu, "crypto": path_bt})

        rate_eu = provider.get_rate("EUR", "USD", 1000000)
        assert abs(rate_eu - 1.0850) < 0.001

        rate_bt = provider.get_rate("BTC", "USDT", 1000000)
        assert abs(rate_bt - 65000.0) < 100.0


def __test_duplicate_pair_keeps_most_bars__(log):
    """When multiple files provide the same pair, the one with more bars wins"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # File with 2 bars
        path_small = _create_test_ohlcv(
            tmpdir, "EURUSD_small", [(1000000, 1.08), (1086400, 1.09)],
            currency="USD", basecurrency="EUR",
        )
        # File with 4 bars
        path_large = _create_test_ohlcv(
            tmpdir, "EURUSD_large",
            [(1000000, 2.00), (1086400, 2.10), (1172800, 2.20), (1259200, 2.30)],
            currency="USD", basecurrency="EUR",
        )

        # Pass small first, large second — large should win
        provider = CurrencyRateProvider({"a": path_small, "b": path_large})
        rate = provider.get_rate("EUR", "USD", 1000000)
        assert abs(rate - 2.00) < 0.01, f"Expected 2.00 (from large file), got {rate}"


def __test_chart_as_rate_source__(log):
    """Chart's own syminfo provides a currency pair via lib.close"""
    from pynecore.core.syminfo import SymInfo

    chart_syminfo = SymInfo(
        prefix="CAPITALCOM", description="EUR/USD", ticker="EURUSD",
        currency="USD", basecurrency="EUR", period="1D", type="forex",
        mintick=0.00001, pricescale=100000, pointvalue=1.0,
        opening_hours=[], session_starts=[], session_ends=[],
    )

    provider = CurrencyRateProvider({}, chart_syminfo=chart_syminfo)

    # Set lib.close to simulate a bar
    from pynecore import lib
    original_close = lib.close
    try:
        lib.close = 1.0925
        rate = provider.get_rate("EUR", "USD", 1000000)
        assert abs(rate - 1.0925) < 0.0001, f"Expected 1.0925 (lib.close), got {rate}"

        # Inverse
        rate = provider.get_rate("USD", "EUR", 1000000)
        expected = 1.0 / 1.0925
        assert abs(rate - expected) < 0.001, f"Expected ~{expected:.4f}, got {rate}"
    finally:
        lib.close = original_close


def __test_reset_clears_provider__(log):
    """_reset_request_state clears the currency provider"""
    from pynecore.lib import request

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        path = _create_test_ohlcv(
            tmpdir, "EURUSD", [(1000000, 1.085)],
            currency="USD", basecurrency="EUR",
        )
        request._currency_provider = CurrencyRateProvider({"fx": path})
        assert request._currency_provider is not None

        request._reset_request_state()
        assert request._currency_provider is None
