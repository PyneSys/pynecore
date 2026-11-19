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
    opening_hours, session_starts, session_ends = CCXTProvider._create_24_7_sessions()

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
    import tempfile
    from pathlib import Path
    from datetime import datetime, UTC, timedelta
    from pynecore.providers.ccxt import CCXTProvider
    from pynecore.core.ohlcv import OHLCVReader
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

    # Check if ccxt config exists
    ccxt_toml = config_dir / "plugins" / "ccxt.toml"
    if not ccxt_toml.exists():
        pytest.skip("No config/plugins/ccxt.toml found")

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

    # Load config via plugin config system
    from pynecore.core.config import ensure_config
    from pynecore.providers.ccxt import CCXTConfig
    config = ensure_config(CCXTConfig, ccxt_toml)

    # Create provider instance
    provider = CCXTProvider(
        symbol=symbol,
        timeframe=timeframe,
        ohlcv_dir=data_dir,
        config=config
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
                "datetime": datetime.fromtimestamp(candle.timestamp / 1000, UTC).isoformat(),
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
            assert 86_000_000 <= interval <= 86_800_000, \
                f"Interval {interval} is not approximately 24 hours (86 400 000 ms)"

    else:
        # We have reference data, compare the downloaded data with it
        # Compare the actual results with the expected data
        # We'll check the first 5 candles as they should be stable for this historical period
        check_count = min(5, len(candles), len(expected_data))  # type: ignore

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


def _market(symbol: str, market_type: str) -> dict:
    """A CCXT market as an exchange's ``fetch_markets`` returns it."""
    base, quote = symbol.split(':')[0].split('/')
    return {
        'id': symbol.replace('/', '').replace(':', ''), 'symbol': symbol, 'base': base,
        'quote': quote, 'type': market_type, 'spot': market_type == 'spot',
        'swap': market_type == 'swap', 'option': market_type == 'option',
        'linear': None, 'inverse': None, 'precision': {'price': 0.1, 'amount': 0.001},
        'limits': {'amount': {'min': 0.001}}, 'info': {},
    }


def _offline_markets(provider: CCXTProvider) -> list[bool]:
    """Serve the provider's market list without a network call; the returned
    list gets, per load, whether the option chains were asked for."""
    loads: list[bool] = []
    client = provider._client

    def load_markets(reload=False, params=None):
        if client.markets and not reload:
            return client.markets
        with_options = 'option' in client.options['fetchMarkets']['types']
        loads.append(with_options)
        markets = [_market('BTC/USDT', 'spot'), _market('BTC/USDT:USDT', 'swap')]
        if with_options:
            markets.append(_market('BTC/USDT:USDT-271231-50000-C', 'option'))
        return client.set_markets(markets)

    client.load_markets = load_markets
    return loads


def __test_ccxt_symbol_provider_skips_option_markets__():
    """A provider of one symbol leaves the option chains out of the market list"""
    pytest.importorskip("ccxt")
    full = CCXTProvider(symbol="BYBIT", timeframe="1", ohlcv_dir=None)
    full_fetch_markets = dict(full._client.options['fetchMarkets'])
    assert 'option' in full_fetch_markets['types']
    # The next provider must get a client of its own, not the symbol-less one's
    del full

    provider = CCXTProvider(symbol="BYBIT:BTC/USDT:USDT", timeframe="1", ohlcv_dir=None)
    assert provider._client.options['fetchMarkets']['types'] == ['spot', 'linear', 'inverse']
    # The exchange's other fetchMarkets settings stay
    assert ({k: v for k, v in provider._client.options['fetchMarkets'].items() if k != 'types'}
            == {k: v for k, v in full_fetch_markets.items() if k != 'types'})

    loads = _offline_markets(provider)
    assert provider.update_symbol_info().ticker
    assert loads == [False]


def __test_ccxt_option_symbol_loads_option_markets__():
    """A symbol missing from the narrowed market list is looked up among the options"""
    pytest.importorskip("ccxt")
    provider = CCXTProvider(symbol="BYBIT:BTC/USDT:USDT-271231-50000-C", timeframe="1",
                            ohlcv_dir=None)
    loads = _offline_markets(provider)
    assert provider.update_symbol_info().ticker
    assert loads == [False, True]
    assert 'option' in provider._client.options['fetchMarkets']['types']


def __test_ccxt_symbol_list_has_option_markets__():
    """The symbol list is complete also on a provider that has a symbol"""
    pytest.importorskip("ccxt")
    provider = CCXTProvider(symbol="BYBIT:BTC/USDT:USDT", timeframe="1", ohlcv_dir=None)
    loads = _offline_markets(provider)
    provider.update_symbol_info()
    assert 'BTC/USDT:USDT-271231-50000-C' in provider.get_list_of_symbols()
    assert loads == [False, True]


def __test_ccxt_failed_option_market_reload_is_retried__():
    """A failed reload of the option chains leaves the list narrowed, and the next
    symbol list loads them"""
    pytest.importorskip("ccxt")
    import ccxt
    from pynecore.providers.ccxt import CCXTError

    provider = CCXTProvider(symbol="BYBIT:BTC/USDT:USDT", timeframe="1", ohlcv_dir=None)
    loads = _offline_markets(provider)
    provider.update_symbol_info()
    client = provider._client
    load_markets = client.load_markets
    failures = [ccxt.NetworkError("offline")]

    def failing_load_markets(reload=False, params=None):
        if reload and failures:
            raise failures.pop()
        return load_markets(reload, params)

    client.load_markets = failing_load_markets
    with pytest.raises(CCXTError):
        provider.get_list_of_symbols()
    assert 'BTC/USDT:USDT-271231-50000-C' in provider.get_list_of_symbols()
    assert loads == [False, True]


def __test_ccxt_configured_fetch_markets_is_kept__(tmp_path):
    """A fetchMarkets option of the exchange's config section is not narrowed"""
    pytest.importorskip("ccxt")
    from pynecore.providers.ccxt import CCXTConfig

    plugins = tmp_path / "config" / "plugins"
    plugins.mkdir(parents=True)
    (plugins / "ccxt.toml").write_text(
        '[bybit]\noptions = { fetchMarkets = { types = ["linear", "option"] } }\n')
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    provider = CCXTProvider(symbol="BYBIT:BTC/USDT:USDT", timeframe="1", ohlcv_dir=data_dir,
                            config=CCXTConfig())
    assert provider._client.options['fetchMarkets']['types'] == ['linear', 'option']


def __test_ccxt_live_client_shares_loaded_markets__():
    """The stream's client uses the markets the history client loaded"""
    pytest.importorskip("ccxt.pro")
    import asyncio

    provider = CCXTProvider(symbol="BYBIT:BTC/USDT:USDT", timeframe="1", ohlcv_dir=None)
    _offline_markets(provider)
    provider.update_symbol_info()

    async def run():
        await provider.connect()
        try:
            client = provider._async_client
            assert client.markets is provider._client.markets
            # Loaded: the stream asks the exchange for no market list of its own
            assert await client.load_markets() is provider._client.markets
        finally:
            await provider.disconnect()

    asyncio.run(run())


def __test_ccxt_providers_of_an_exchange_share_one_client__(tmp_path):
    """Every provider of one exchange and config uses one client and one market list;
    another exchange, or another config of the same one, gets its own"""
    pytest.importorskip("ccxt")
    from pynecore.providers.ccxt import CCXTConfig

    first = CCXTProvider(symbol="BYBIT:BTC/USDT:USDT", timeframe="1", ohlcv_dir=None)
    second = CCXTProvider(symbol="BYBIT:BTC/USDT", timeframe="5", ohlcv_dir=None)
    assert second._client is first._client

    loads = _offline_markets(first)
    assert first.update_symbol_info().ticker
    assert second.update_symbol_info().ticker
    assert loads == [False]

    other_exchange = CCXTProvider(symbol="OKX:BTC/USDT", timeframe="1", ohlcv_dir=None)
    assert other_exchange._client is not first._client

    plugins = tmp_path / "config" / "plugins"
    plugins.mkdir(parents=True)
    (plugins / "ccxt.toml").write_text('[bybit]\noptions = { defaultType = "spot" }\n')
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    configured = CCXTProvider(symbol="BYBIT:BTC/USDT", timeframe="1", ohlcv_dir=data_dir,
                              config=CCXTConfig())
    assert configured._client is not first._client


def __test_ccxt_shared_client_ends_with_its_last_provider__():
    """A client is kept only while a provider uses it"""
    pytest.importorskip("ccxt")
    import gc
    import weakref

    first = CCXTProvider(symbol="BYBIT:BTC/USDT:USDT", timeframe="1", ohlcv_dir=None)
    second = CCXTProvider(symbol="BYBIT:ETH/USDT:USDT", timeframe="1", ohlcv_dir=None)
    shared = weakref.ref(first._shared)
    del first
    gc.collect()
    assert shared() is second._shared
    del second
    gc.collect()
    assert shared() is None


def __test_ccxt_symbol_list_on_a_shared_client_loads_the_options__():
    """A provider that lists the symbols completes the market list of the client it
    shares with a provider of one symbol"""
    pytest.importorskip("ccxt")
    chart = CCXTProvider(symbol="BYBIT:BTC/USDT:USDT", timeframe="1", ohlcv_dir=None)
    loads = _offline_markets(chart)
    chart.update_symbol_info()
    browser = CCXTProvider(symbol="BYBIT", timeframe="1", ohlcv_dir=None)
    assert browser._client is chart._client
    assert 'BTC/USDT:USDT-271231-50000-C' in browser.get_list_of_symbols()
    assert loads == [False, True]
