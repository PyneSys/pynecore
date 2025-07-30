"""Tests for {{plugin_name_pascal}}Provider"""

import pytest
from datetime import datetime, timezone
from pathlib import Path
import tempfile

from {{plugin_name_snake}}_provider import {{plugin_name_pascal}}Provider
from pynecore.types.ohlcv import OHLCV
from pynecore.core.syminfo import SymInfo


class Test{{plugin_name_pascal}}Provider:
    """Test suite for {{plugin_name_pascal}}Provider"""
    
    @pytest.fixture
    def temp_workdir(self):
        """Create temporary working directory for tests"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def provider(self, temp_workdir):
        """Create provider instance for testing"""
        # Create config directory and providers.toml
        config_dir = temp_workdir / "config"
        config_dir.mkdir(exist_ok=True)
        
        # Create a minimal providers.toml file
        providers_toml = config_dir / "providers.toml"
        providers_toml.write_text("[{{plugin_name_snake}}]\n# Configuration for {{plugin_name}} provider\n")
        
        return {{plugin_name_pascal}}Provider(
            symbol="EURUSD",
            timeframe="1h",
            ohlv_dir=temp_workdir / "data",
            config_dir=config_dir
        )
    
    def test_initialization(self, provider):
        """Test provider initialization"""
        assert provider.symbol == "EURUSD"
        assert provider.timeframe == "1h"
        assert provider.timezone == timezone.utc
        assert isinstance(provider.config_keys, dict)
    
    def test_timeframe_conversion_to_tradingview(self, provider):
        """Test timeframe conversion to TradingView format"""
        test_cases = {
            "1m": "1",
            "5m": "5",
            "15m": "15",
            "30m": "30",
            "1h": "60",
            "4h": "240",
            "1d": "1D",
            "1w": "1W",
            "1M": "1M",
        }
        
        for provider_tf, expected_tv_tf in test_cases.items():
            result = {{plugin_name_pascal}}Provider.to_tradingview_timeframe(provider_tf)
            assert result == expected_tv_tf, f"Failed for {provider_tf}: expected {expected_tv_tf}, got {result}"
    
    def test_timeframe_conversion_to_exchange(self, provider):
        """Test timeframe conversion to exchange format"""
        test_cases = {
            "1": "1m",
            "5": "5m",
            "15": "15m",
            "30": "30m",
            "60": "1h",
            "240": "4h",
            "1D": "1d",
            "1W": "1w",
            "1M": "1M",
        }
        
        for tv_tf, expected_provider_tf in test_cases.items():
            result = {{plugin_name_pascal}}Provider.to_exchange_timeframe(tv_tf)
            assert result == expected_provider_tf, f"Failed for {tv_tf}: expected {expected_provider_tf}, got {result}"
    
    def test_get_list_of_symbols(self, provider):
        """Test symbol listing"""
        symbols = provider.get_list_of_symbols()
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert all(isinstance(symbol, str) for symbol in symbols)
    
    def test_update_symbol_info(self, provider):
        """Test symbol information retrieval"""
        symbol_info = provider.update_symbol_info()
        
        assert isinstance(symbol_info, SymInfo)
        assert symbol_info.symbol == "EURUSD"
        assert symbol_info.description is not None
        assert symbol_info.type is not None
        assert symbol_info.currency is not None
        assert symbol_info.exchange is not None
    
    def test_get_opening_hours_and_sessions(self, provider):
        """Test trading hours and session information"""
        opening_hours, sessions, session_ends = provider.get_opening_hours_and_sessions()
        
        assert isinstance(opening_hours, list)
        assert isinstance(sessions, list)
        assert isinstance(session_ends, list)
    
    def test_download_ohlcv_basic(self, provider):
        """Test basic OHLCV data download"""
        start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2023, 1, 2, tzinfo=timezone.utc)
        
        # Track progress calls
        progress_calls = []
        def progress_callback(timestamp):
            progress_calls.append(timestamp)
        
        # Download data (this saves to provider's OHLCV storage)
        provider.download_ohlcv(
            start_date=start_date,
            end_date=end_date,
            on_progress=progress_callback
        )
        
        # Verify progress was called
        assert len(progress_calls) > 0
        assert all(isinstance(ts, datetime) for ts in progress_calls)
        
        # Read back the saved data using provider's reader
        data = provider.read_ohlcv_data(start_date, end_date)
        
        assert isinstance(data, list)
        assert len(data) > 0
        
        # Check first record structure
        first_record = data[0]
        assert isinstance(first_record, OHLCV)
        
        # Check data types and structure
        assert isinstance(first_record.timestamp, datetime)
        assert isinstance(first_record.open, (int, float))
        assert isinstance(first_record.high, (int, float))
        assert isinstance(first_record.low, (int, float))
        assert isinstance(first_record.close, (int, float))
        assert isinstance(first_record.volume, (int, float))
        
        # Check OHLC relationships
        for record in data:
            assert record.high >= record.open
            assert record.high >= record.close
            assert record.high >= record.low
            assert record.low <= record.open
            assert record.low <= record.close
            assert record.volume >= 0
    
    def test_download_ohlcv_with_progress(self, provider):
        """Test OHLCV download with progress callback"""
        start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2023, 1, 2, tzinfo=timezone.utc)
        
        progress_calls = []
        
        def progress_callback(timestamp):
            progress_calls.append(timestamp)
            assert isinstance(timestamp, datetime)
        
        provider.download_ohlcv(
            start_date=start_date,
            end_date=end_date,
            on_progress=progress_callback
        )
        
        assert len(progress_calls) > 0
        # Verify all progress calls are datetime objects
        assert all(isinstance(ts, datetime) for ts in progress_calls)
        
        # Read back the data to verify it was saved
        data = provider.read_ohlcv_data(start_date, end_date)
        assert isinstance(data, list)
        assert len(data) > 0
    
    def test_download_ohlcv_empty_range(self, provider):
        """Test OHLCV download with invalid time range"""
        start_date = datetime(2024, 1, 2, tzinfo=timezone.utc)
        end_date = datetime(2024, 1, 1, tzinfo=timezone.utc)  # Invalid range
        
        provider.download_ohlcv(
            start_date=start_date,
            end_date=end_date
        )
        
        # Read back the data - should be empty
        data = provider.read_ohlcv_data(start_date, end_date)
        assert isinstance(data, list)
        assert len(data) == 0
    
    def test_download_ohlcv_date_range(self, provider):
        """Test OHLCV download with specific date range"""
        start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2023, 1, 3, tzinfo=timezone.utc)
        
        provider.download_ohlcv(
            start_date=start_date,
            end_date=end_date
        )
        
        # Read back the saved data
        data = provider.read_ohlcv_data(start_date, end_date)
        
        assert isinstance(data, list)
        assert len(data) > 0
        
        # Check that all timestamps are within the requested range
        for record in data:
            timestamp = record.timestamp
            assert start_date <= timestamp <= end_date
    
    @pytest.mark.parametrize("timeframe", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"])
    def test_download_ohlcv_different_timeframes(self, provider, timeframe):
        """Test OHLCV download with different timeframes"""
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2024, 1, 1, 6, tzinfo=timezone.utc)  # 6 hours
        
        # Create provider with specific timeframe
        provider_tf = {{plugin_name_pascal}}Provider(
            symbol="EURUSD",
            timeframe=timeframe,
            ohlv_dir=provider.ohlv_dir,
            config_dir=provider.config_dir
        )
        
        provider_tf.download_ohlcv(
            start_date=start_date,
            end_date=end_date
        )
        
        # Read back the data
        data = provider_tf.read_ohlcv_data(start_date, end_date)
        
        assert isinstance(data, list)
        # The exact number of records depends on the timeframe and implementation
        # but we can check that the structure is correct
        if data:
            assert isinstance(data[0], OHLCV)