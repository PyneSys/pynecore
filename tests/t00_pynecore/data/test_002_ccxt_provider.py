import pytest
from datetime import time

from pynecore.providers.ccxt import CCXTProvider


def __test_ccxt_timeframe_conversion__():
    """Test timeframe conversion for CCXT provider"""
    # TradingView to CCXT conversion
    assert CCXTProvider.to_exchange_timeframe("1") == "1m"
    assert CCXTProvider.to_exchange_timeframe("5") == "5m"
    assert CCXTProvider.to_exchange_timeframe("15") == "15m"
    assert CCXTProvider.to_exchange_timeframe("30") == "30m"
    assert CCXTProvider.to_exchange_timeframe("60") == "1h"
    assert CCXTProvider.to_exchange_timeframe("120") == "2h"
    assert CCXTProvider.to_exchange_timeframe("240") == "4h"
    assert CCXTProvider.to_exchange_timeframe("1D") == "1d"
    assert CCXTProvider.to_exchange_timeframe("1W") == "1w"
    assert CCXTProvider.to_exchange_timeframe("1M") == "1M"

    # CCXT to TradingView conversion
    assert CCXTProvider.to_tradingview_timeframe("1m") == "1"
    assert CCXTProvider.to_tradingview_timeframe("5m") == "5"
    assert CCXTProvider.to_tradingview_timeframe("15m") == "15"
    assert CCXTProvider.to_tradingview_timeframe("30m") == "30"
    assert CCXTProvider.to_tradingview_timeframe("1h") == "60"
    assert CCXTProvider.to_tradingview_timeframe("2h") == "120"
    assert CCXTProvider.to_tradingview_timeframe("4h") == "240"
    assert CCXTProvider.to_tradingview_timeframe("1d") == "1D"
    assert CCXTProvider.to_tradingview_timeframe("1w") == "1W"
    assert CCXTProvider.to_tradingview_timeframe("1M") == "1M"

    # Test invalid formats
    with pytest.raises(ValueError):
        CCXTProvider.to_exchange_timeframe("invalid")

    with pytest.raises(ValueError):
        CCXTProvider.to_tradingview_timeframe("invalid")


def __test_ccxt_provider_path_handling__(tmp_path):
    """Test path handling in CCXT provider"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Test path construction for different symbol formats
    path1 = CCXTProvider.get_ohlcv_path("BINANCE:BTC/USDT", "1D", data_dir)
    path2 = CCXTProvider.get_ohlcv_path("BYBIT:BTC/USDT:USDT", "1h", data_dir)

    # Verify paths are created correctly
    assert "ccxt_BINANCE_BTC_USDT_1D.ohlcv" in str(path1)
    assert "ccxt_BYBIT_BTC_USDT_USDT_1h.ohlcv" in str(path2)


def __test_ccxt_session_hours__():
    """Test session hours creation for CCXT provider"""
    opening_hours, session_starts, session_ends = CCXTProvider.get_opening_hours_and_sessions()

    # Check that we have entries for all days
    assert len(opening_hours) == 7
    assert len(session_starts) == 7
    assert len(session_ends) == 7

    # Check that crypto markets are open 24/7
    for day in range(7):
        # Opening hours should be 00:00 to 23:59:59
        assert opening_hours[day].day == day
        assert opening_hours[day].start == time(hour=0, minute=0)
        assert opening_hours[day].end == time(hour=23, minute=59, second=59)

        # Session starts at 00:00
        assert session_starts[day].day == day
        assert session_starts[day].time == time(hour=0, minute=0)

        # Session ends at 23:59:59
        assert session_ends[day].day == day
        assert session_ends[day].time == time(hour=23, minute=59, second=59)


def __test_ccxt_real_data_download__(tmp_path):
    """Test CCXT provider real data download"""
    import pytest
    import os
    import json
    import logging
    import tomllib
    import tempfile
    from pathlib import Path
    from datetime import datetime, UTC, timedelta
    from pynecore.providers.ccxt import CCXTProvider
    from pynecore.core.ohlcv_file import OHLCVReader
    from pynecore.cli.app import app_state

    # Disable debug logging to reduce output noise
    logging.getLogger().setLevel(logging.WARNING)

    # Skip this test if the CCXT library is not available
    try:
        import ccxt
    except ImportError:
        pytest.skip("CCXT library not available")

    # Find workdir using AppState._find_workdir to get config
    workdir = app_state._find_workdir()  # noqa

    # Check if config directory exists
    config_dir = workdir / "config"
    if not config_dir.exists() or not config_dir.is_dir():
        pytest.skip("No config directory found in workdir")

    # Check if providers.toml exists
    providers_toml = config_dir / "providers.toml"
    if not providers_toml.exists() or not providers_toml.is_file():
        pytest.skip("No providers.toml found in config directory")

    # Check if ccxt section exists in providers.toml
    try:
        with open(providers_toml, 'rb') as f:
            config = tomllib.load(f)
            if 'ccxt' not in config:
                pytest.skip("No ccxt section found in providers.toml")
    except Exception as e:
        pytest.skip(f"Error reading providers.toml: {str(e)}")

    # Create temporary data directory for the test
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Define test data - use a stable, historical period for reproducibility
    # Note: Using Kraken instead of Binance because Binance Futures API is geo-blocked in EU
    exchange = "kraken"
    symbol = f"{exchange.upper()}:BTC/USD"
    timeframe = "1D"  # Daily timeframe

    # Use recent period - Kraken doesn't reliably support old historical data via CCXT
    # Get data from last 30 days to ensure availability
    time_to = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    time_from = time_to - timedelta(days=30)

    # Check if a reference file with expected data already exists
    test_data_path = Path(__file__).parent / "ccxt_test_data.json"
    expected_data = None

    # Enable saving reference data to a file
    # Note: With dynamic date range, reference data comparison is not meaningful
    force_save_reference = False

    if test_data_path.exists() and not force_save_reference:
        try:
            with open(test_data_path, 'r') as f:
                expected_data = json.load(f)
            print(f"Loaded reference data from {test_data_path}")
        except Exception as e:
            print(f"Could not load reference data: {e}")
            expected_data = None

    # Create provider instance
    provider = CCXTProvider(
        symbol=symbol,
        timeframe=timeframe,
        ohlv_dir=data_dir,
        config_dir=config_dir
    )

    # Define a function to download data
    # noinspection PyShadowingNames
    def download_ohlcv_data():
        with provider:
            provider.download_ohlcv(time_from, time_to, on_progress=None)

        # Check that the OHLCV file was created
        ohlcv_path = provider.get_ohlcv_path(symbol, timeframe, data_dir)
        assert ohlcv_path.exists(), f"OHLCV file was not created at {ohlcv_path}"

        # Verify the content of the file
        with OHLCVReader(str(ohlcv_path)) as reader:
            # Check that we have data
            assert reader.size > 0, "No data was downloaded"

            # Read candles
            candles = list(reader)
            return candles, ohlcv_path

    # Download the data and get the candles
    candles, ohlcv_path = download_ohlcv_data()

    print(f"\nDownloaded {len(candles)} candles from {exchange} API")

    # Save the downloaded data for future reference if needed
    if expected_data is None or force_save_reference:
        # Save candles to temporary file for further analysis
        candle_data = []
        for candle in candles:
            candle_data.append({
                "timestamp": candle.timestamp,
                "datetime": datetime.fromtimestamp(candle.timestamp, UTC).isoformat(),
                "open": float(candle.open),
                "high": float(candle.high),
                "low": float(candle.low),
                "close": float(candle.close),
                "volume": float(candle.volume)
            })

        # Create a temporary file in the system's temp directory
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
            temp_data_file = temp_file.name
            json.dump(candle_data, temp_file, indent=2)

        try:
            # Verify the file was created and has content
            file_size = os.path.getsize(temp_data_file)

            # Force print this information with the highest priority
            print("\n" + "=" * 80)
            print(f"TEST DATA SUCCESSFULLY SAVED TO: {temp_data_file}")
            print(f"File size: {file_size} bytes, Number of candles: {len(candle_data)}")
            print(
                "Please copy this file to the test directory with the name 'ccxt_test_data.json' for future reference.")
            print("=" * 80 + "\n")

        except Exception as e:
            # If there's any error with file operations, print it clearly
            print(f"\nERROR SAVING TEST DATA: {str(e)}")

        # Basic validations if we don't have reference data
        # Basic validation - timestamps should be in ascending order
        for i in range(1, len(candles)):
            assert candles[i].timestamp > candles[i - 1].timestamp, "Timestamps not in ascending order"

        # Check that we have the correct interval for daily timeframe
        if len(candles) >= 2:
            interval = candles[1].timestamp - candles[0].timestamp
            assert 86000 <= interval <= 86800, f"Interval {interval} is not approximately 24 hours (86400 seconds)"

    else:
        # We have reference data, compare the downloaded data with it
        # Compare the actual results with the expected data
        # We'll check the first 5 candles as they should be stable for this historical period
        check_count = min(5, len(candles), len(expected_data))

        print(f"Comparing first {check_count} candles with expected data")
        for i in range(check_count):
            actual = candles[i]
            expected = expected_data[i]

            # Compare timestamp - timestamps should be exactly the same
            assert actual.timestamp == expected["timestamp"], \
                f"Timestamp mismatch at candle {i}: expected {expected['timestamp']}, got {actual.timestamp}"

            # Compare OHLCV values with a small tolerance for floating point differences
            tolerance = 0.01  # Allow some small difference for float values
            assert abs(actual.open - expected["open"]) < tolerance, \
                f"Open price mismatch at candle {i}: expected {expected['open']}, got {actual.open}"
            assert abs(actual.high - expected["high"]) < tolerance, \
                f"High price mismatch at candle {i}: expected {expected['high']}, got {actual.high}"
            assert abs(actual.low - expected["low"]) < tolerance, \
                f"Low price mismatch at candle {i}: expected {expected['low']}, got {actual.low}"
            assert abs(actual.close - expected["close"]) < tolerance, \
                f"Close price mismatch at candle {i}: expected {expected['close']}, got {actual.close}"

            # Volume might have slightly larger variation
            volume_tolerance = 0.1  # Allow 10% difference in volume
            volume_diff = abs(actual.volume - expected["volume"]) / expected["volume"] if expected["volume"] > 0 else 0
            assert volume_diff <= volume_tolerance, \
                f"Volume mismatch at candle {i}: expected {expected['volume']}, got {actual.volume}"

        print("Data validation successful - downloaded data matches expected values")
