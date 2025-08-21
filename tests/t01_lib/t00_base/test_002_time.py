"""
Unit tests for time() and time_close() functions.

This test module validates the functionality of:
- time() function with different timeframes, sessions, and timezones
- time_close() function with different timeframes, sessions, and timezones
- Session string parsing and validation
- Proper NA handling for invalid inputs
- Timeframe resampling logic
"""

import pytest
from datetime import datetime
from pynecore.lib import time, time_close
from pynecore.types.na import NA
import pynecore.lib as lib


class TestTimeFunctions:
    """Test class for time() and time_close() functions."""
    
    def setup_method(self):
        """Setup test environment before each test."""
        # Set a known timestamp for consistent testing
        # 2023-01-02 14:30:00 UTC (Monday, market hours)
        lib._time = 1672672200000
        
    def test_time_basic(self):
        """Test basic time() function without parameters."""
        result = time()
        assert result == 1672672200000
        assert isinstance(result, int)
    
    def test_time_close_basic(self):
        """Test basic time_close() function without parameters."""
        result = time_close()
        assert result == 1672672200000  # Same as time() when no timeframe specified
        assert isinstance(result, int)
    
    def test_time_with_timeframe(self):
        """Test time() function with different timeframes."""
        # Test 1-minute timeframe (should return bar start time)
        result_1m = time("1")
        assert isinstance(result_1m, int)
        
        # Test 1-hour timeframe
        result_1h = time("60")
        assert isinstance(result_1h, int)
        
        # Test daily timeframe
        result_1d = time("1D")
        assert isinstance(result_1d, int)
        
        # Test weekly timeframe
        result_1w = time("1W")
        assert isinstance(result_1w, int)
        
        # Test monthly timeframe
        result_1m_month = time("1M")
        assert isinstance(result_1m_month, int)
    
    def test_time_close_with_timeframe(self):
        """Test time_close() function with different timeframes."""
        # Test that time_close returns time + timeframe duration
        time_start = time("60")  # 1-hour bar start
        time_end = time_close("60")  # 1-hour bar close
        
        # Close should be exactly 1 hour (3600 seconds = 3600000 ms) after start
        assert time_end == time_start + 3600000
        
        # Test daily timeframe
        daily_start = time("1D")
        daily_end = time_close("1D")
        
        # Daily close should be 24 hours after start
        assert daily_end == daily_start + 86400000  # 24 * 60 * 60 * 1000
    
    def test_invalid_timeframe(self):
        """Test time() and time_close() with invalid timeframes."""
        # Invalid timeframe should return NA
        result_time = time("invalid")
        result_time_close = time_close("invalid")
        
        assert isinstance(result_time, NA)
        assert isinstance(result_time_close, NA)
    
    def test_time_with_session_weekday(self):
        """Test time() function with session on a weekday."""
        # Monday 14:30 UTC should be within 0930-1600 session for weekdays
        result = time("60", "0930-1600:23456")  # Monday-Friday
        
        # Should return NA because 14:30 UTC (2:30 PM) is outside 9:30-16:00 session
        # unless the session is in a different timezone
        assert isinstance(result, NA)
    
    def test_time_with_session_all_days(self):
        """Test time() function with session for all days."""
        # Test with a session that includes all days
        result = time("60", "0000-2359:1234567")  # All days, almost 24h
        
        # Should return a valid timestamp
        assert isinstance(result, int)
    
    def test_time_with_session_and_timezone(self):
        """Test time() function with session and timezone."""
        # Test with UTC timezone
        result = time("60", "0930-1600:23456", "UTC")
        
        # Should return NA because 14:30 UTC is outside the session
        assert isinstance(result, NA)
        
        # Test with a timezone where 14:30 UTC might be in session
        result2 = time("60", "0000-2359:23456", "UTC")
        
        # Should return a valid timestamp
        assert isinstance(result2, int)
    
    def test_time_close_with_session(self):
        """Test time_close() function with session filtering."""
        # Test with a valid session
        result = time_close("60", "0000-2359:1234567")
        
        # Should return a valid timestamp (bar close time)
        assert isinstance(result, int)
        
        # Test with invalid session
        result_invalid = time_close("60", "0930-1600:23456")
        
        # Should return NA
        assert isinstance(result_invalid, NA)
    
    def test_invalid_session_string(self):
        """Test time() and time_close() with invalid session strings."""
        invalid_sessions = [
            "invalid",
            "0930",  # Missing end time
            "0930-2500",  # Invalid hour
            "0930-1600:8",  # Invalid day
            "",  # Empty string
        ]
        
        for session in invalid_sessions:
            result_time = time("60", session)
            result_time_close = time_close("60", session)
            
            assert isinstance(result_time, NA), f"time() should return NA for session: {session}"
            assert isinstance(result_time_close, NA), f"time_close() should return NA for session: {session}"
    
    def test_session_overnight(self):
        """Test time() and time_close() with overnight sessions."""
        # Test overnight session (22:00-06:00)
        result_time = time("60", "2200-0600:1234567")
        result_time_close = time_close("60", "2200-0600:1234567")
        
        # At 14:30 UTC, should be outside overnight session
        assert isinstance(result_time, NA)
        assert isinstance(result_time_close, NA)
    
    def test_time_consistency(self):
        """Test that time() and time_close() are consistent."""
        timeframes = ["1", "5", "15", "60", "240", "1D"]
        
        for tf in timeframes:
            time_start = time(tf)
            time_end = time_close(tf)
            
            if isinstance(time_start, NA):
                # If time() returns NA, time_close() should also return NA
                assert isinstance(time_end, NA), f"Inconsistency for timeframe {tf}"
            else:
                # If time() returns a valid timestamp, time_close() should be >= time_start
                assert isinstance(time_end, int), f"time_close() should return int for timeframe {tf}"
                assert time_end >= time_start, f"time_close() should be >= time() for timeframe {tf}"
    
    def test_keyword_arguments(self):
        """Test time() and time_close() with keyword arguments."""
        # Test with keyword arguments
        result1 = time(timeframe="60")
        result2 = time(timeframe="60", session="0000-2359")
        result3 = time(timeframe="60", session="0000-2359", timezone="UTC")
        
        assert isinstance(result1, int)
        assert isinstance(result2, int)
        assert isinstance(result3, int)
        
        # Test time_close with keyword arguments
        result4 = time_close(timeframe="60")
        result5 = time_close(timeframe="60", session="0000-2359")
        result6 = time_close(timeframe="60", session="0000-2359", timezone="UTC")
        
        assert isinstance(result4, int)
        assert isinstance(result5, int)
        assert isinstance(result6, int)
    
    def test_session_days_format(self):
        """Test different session day formats."""
        # Test different day combinations
        sessions = [
            "0930-1600:1",      # Sunday only
            "0930-1600:2",      # Monday only
            "0930-1600:23456",  # Monday-Friday
            "0930-1600:67",     # Weekend only
            "0930-1600:1234567" # All days
        ]
        
        for session in sessions:
            result_time = time("60", session)
            result_time_close = time_close("60", session)
            
            # All should return valid results (either int or NA)
            assert isinstance(result_time, (int, NA)), f"Unexpected type for session: {session}"
            assert isinstance(result_time_close, (int, NA)), f"Unexpected type for session: {session}"
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with None values (should work same as no parameters)
        result1 = time(None)
        result2 = time(None, None)
        result3 = time(None, None, None)
        
        basic_result = time()
        
        assert result1 == basic_result
        assert result2 == basic_result
        assert result3 == basic_result
        
        # Test time_close with None values
        result4 = time_close(None)
        result5 = time_close(None, None)
        result6 = time_close(None, None, None)
        
        basic_close_result = time_close()
        
        assert result4 == basic_close_result
        assert result5 == basic_close_result
        assert result6 == basic_close_result


def __test_time__(csv_reader, runner, dict_comparator, log):
    """Integration test with CSV data (following the project's test pattern)."""
    # This function follows the project's testing pattern but for now
    # we'll just run a simple validation
    
    # Test basic functionality
    lib._time = 1672672200000  # Set test timestamp
    
    # Validate basic functionality
    basic_time = time()
    basic_time_close = time_close()
    
    assert isinstance(basic_time, int)
    assert isinstance(basic_time_close, int)
    
    # Test with timeframes
    hourly_time = time("60")
    hourly_time_close = time_close("60")
    
    assert isinstance(hourly_time, int)
    assert isinstance(hourly_time_close, int)
    assert hourly_time_close > hourly_time


if __name__ == "__main__":
    # Allow running the test file directly
    pytest.main([__file__])