"""
@pyne
"""
import os
import struct
import pytest

from pynecore.types.ohlcv import OHLCV
from pynecore.core.ohlcv_file import OHLCVWriter, OHLCVReader


def main():
    """
    Dummy main function to be a valid Pyne script
    """
    pass


def __test_ohlcv_basic_io__(tmp_path):
    """Basic OHLCV file I/O operations"""
    # Create a test file path
    file_path = tmp_path / "test_basic.ohlcv"

    # Sample data
    candle1 = OHLCV(timestamp=1609459200, open=100.0, high=110.0, low=90.0, close=105.0, volume=1000.0)
    candle2 = OHLCV(timestamp=1609459260, open=105.0, high=115.0, low=95.0, close=110.0, volume=1200.0)

    # Write test data
    with OHLCVWriter(file_path) as writer:
        writer.write(candle1)
        writer.write(candle2)

    # Verify file exists and has correct size
    assert file_path.exists()
    assert os.path.getsize(file_path) == 48  # 2 records * 24 bytes

    # Read data back
    with OHLCVReader(file_path) as reader:
        candles = list(reader)

        # Verify record count
        assert len(candles) == 2

        # Verify first candle data
        assert candles[0].timestamp == 1609459200
        assert candles[0].open == 100.0
        assert candles[0].high == 110.0
        assert candles[0].low == 90.0
        assert candles[0].close == 105.0
        assert candles[0].volume == 1000.0

        # Verify second candle data
        assert candles[1].timestamp == 1609459260
        assert candles[1].open == 105.0
        assert candles[1].high == 115.0
        assert candles[1].low == 95.0
        assert candles[1].close == 110.0
        assert candles[1].volume == 1200.0


def __test_ohlcv_interval_detection__(tmp_path):
    """Test interval detection in OHLCV files"""
    file_path = tmp_path / "test_interval.ohlcv"

    # Create candles with consistent 1-minute interval
    with OHLCVWriter(file_path) as writer:
        writer.write(OHLCV(timestamp=1609459200, open=100.0, high=110.0, low=90.0, close=105.0, volume=1000.0))
        writer.write(OHLCV(timestamp=1609459260, open=105.0, high=115.0, low=95.0, close=110.0, volume=1200.0))
        writer.write(OHLCV(timestamp=1609459320, open=110.0, high=120.0, low=100.0, close=115.0, volume=1400.0))

    # Check interval detection
    with OHLCVReader(file_path) as reader:
        assert reader.interval == 60  # 60 seconds interval
        assert reader.start_timestamp == 1609459200
        assert reader.end_timestamp == 1609459320


def __test_ohlcv_gap_handling__(tmp_path):
    """Test gap handling in OHLCV files"""
    file_path = tmp_path / "test_gap.ohlcv"

    # Create candles with a gap (missing 1609459320)
    with OHLCVWriter(file_path) as writer:
        # First we need to establish an interval (write 2 records)
        writer.write(OHLCV(timestamp=1609459200, open=100.0, high=110.0, low=90.0, close=105.0, volume=1000.0))
        writer.write(OHLCV(timestamp=1609459260, open=105.0, high=115.0, low=95.0, close=108.0, volume=1100.0))

        # Now write a record with a gap to trigger gap filling
        writer.write(OHLCV(timestamp=1609459380, open=110.0, high=120.0, low=100.0, close=115.0, volume=1400.0))

    # Check file size - should have 4 records (3 original + 1 gap filled)
    file_size = os.path.getsize(file_path)
    assert file_size == 4 * 24  # 4 records * 24 bytes

    # Read back and check contents
    with OHLCVReader(file_path) as reader:
        candles = list(reader)

        # Should have 4 records (3 original + 1 gap filled)
        assert len(candles) == 4

        # First record
        assert candles[0].timestamp == 1609459200
        assert candles[0].close == 105.0
        assert candles[0].volume == 1000.0

        # Second record
        assert candles[1].timestamp == 1609459260
        assert candles[1].close == 108.0
        assert candles[1].volume == 1100.0

        # Third record - gap filled with previous close and -1 volume
        assert candles[2].timestamp == 1609459320
        assert candles[2].open == 108.0  # Previous close
        assert candles[2].close == 108.0  # Previous close
        assert candles[2].volume == -1.0  # Gap indicator

        # Fourth record - original
        assert candles[3].timestamp == 1609459380
        assert candles[3].close == 115.0
        assert candles[3].volume == 1400.0


def __test_ohlcv_seek_operations__(tmp_path):
    """Test seek operations in OHLCV files"""
    file_path = tmp_path / "test_seek.ohlcv"

    # Create test data
    with OHLCVWriter(file_path) as writer:
        for i in range(10):
            timestamp = 1609459200 + (i * 60)  # 1-minute interval
            writer.write(OHLCV(timestamp=timestamp, open=100.0 + i, high=110.0 + i, low=90.0 + i, close=105.0 + i,
                               volume=1000.0 + i))

    # Test seeking to a specific position and direct write
    # Note: we use low-level file operations to bypass timestamp checks
    with OHLCVWriter(file_path) as writer:
        writer.seek(5)  # Seek to 6th record

        # Use direct byte writing to avoid chronological checks
        # noinspection PyProtectedMember
        assert writer._file is not None
        data = struct.pack(
            'Ifffff',
            1609459500,  # Timestamp - same as the original at position 5
            200.0,  # Open
            210.0,  # High
            190.0,  # Low
            205.0,  # Close
            2000.0  # Volume
        )
        # noinspection PyProtectedMember
        writer._file.write(data)
        # noinspection PyProtectedMember
        writer._file.flush()

    # Verify seek operation
    with OHLCVReader(file_path) as reader:
        candles = list(reader)
        assert len(candles) == 10  # Total records unchanged
        assert candles[5].timestamp == 1609459500  # 6th record overwritten
        assert candles[5].open == 200.0
        assert candles[5].close == 205.0


def __test_ohlcv_truncate__(tmp_path):
    """Test truncate operation in OHLCV files"""
    file_path = tmp_path / "test_truncate.ohlcv"

    # Create test data
    with OHLCVWriter(file_path) as writer:
        for i in range(10):
            timestamp = 1609459200 + (i * 60)
            writer.write(OHLCV(timestamp=timestamp, open=100.0 + i, high=110.0 + i, low=90.0 + i, close=105.0 + i,
                               volume=1000.0 + i))

    # Truncate the file
    with OHLCVWriter(file_path) as writer:
        writer.seek(5)  # Seek to 6th record
        writer.truncate()  # Truncate after 5th record

    # Verify truncate operation
    with OHLCVReader(file_path) as reader:
        candles = list(reader)
        assert len(candles) == 5  # Only 5 records should remain
        assert candles[-1].timestamp == 1609459200 + (4 * 60)  # Last record


def __test_ohlcv_csv_conversion__(tmp_path):
    """Test CSV conversion operations"""
    ohlcv_path = tmp_path / "test_csv.ohlcv"
    csv_path = tmp_path / "test_output.csv"

    # Create test data
    with OHLCVWriter(ohlcv_path) as writer:
        writer.write(OHLCV(timestamp=1609459200, open=100.0, high=110.0, low=90.0, close=105.0, volume=1000.0))
        writer.write(OHLCV(timestamp=1609459260, open=105.0, high=115.0, low=95.0, close=110.0, volume=1200.0))

    # Convert to CSV
    with OHLCVReader(ohlcv_path) as reader:
        reader.save_to_csv(str(csv_path))

    # Verify CSV content
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 3  # Header + 2 data rows
        assert lines[0].strip() == "timestamp,open,high,low,close,volume"
        assert lines[1].strip().startswith("1609459200")
        assert "105" in lines[1]  # Close value
        assert "1000" in lines[1]  # Volume value

    # Test CSV to OHLCV conversion
    new_ohlcv_path = tmp_path / "test_from_csv.ohlcv"
    with OHLCVWriter(new_ohlcv_path) as writer:
        writer.load_from_csv(csv_path)

    # Verify converted data
    with OHLCVReader(new_ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 2
        assert candles[0].timestamp == 1609459200
        assert candles[0].close == 105.0


def __test_ohlcv_json_conversion__(tmp_path):
    """Test JSON conversion operations"""
    ohlcv_path = tmp_path / "test_json.ohlcv"
    json_path = tmp_path / "test_output.json"

    # Create test data
    with OHLCVWriter(ohlcv_path) as writer:
        writer.write(OHLCV(timestamp=1609459200, open=100.0, high=110.0, low=90.0, close=105.0, volume=1000.0))
        writer.write(OHLCV(timestamp=1609459260, open=105.0, high=115.0, low=95.0, close=110.0, volume=1200.0))

    # Convert to JSON
    with OHLCVReader(ohlcv_path) as reader:
        reader.save_to_json(str(json_path))

    # Verify JSON content (simple check)
    with open(json_path, 'r') as f:
        content = f.read()
        assert "1609459200" in content
        assert "105" in content  # Close value
        assert "1000" in content  # Volume value

    # Test JSON to OHLCV conversion
    new_ohlcv_path = tmp_path / "test_from_json.ohlcv"
    with OHLCVWriter(new_ohlcv_path) as writer:
        writer.load_from_json(json_path)

    # Verify converted data
    with OHLCVReader(new_ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 2
        assert candles[0].timestamp == 1609459200
        assert candles[0].close == 105.0


def __test_ohlcv_reader_from_to__(tmp_path):
    """Test reading specific time ranges"""
    file_path = tmp_path / "test_range.ohlcv"

    # Create test data with 10 candles, 1-minute interval
    with OHLCVWriter(file_path) as writer:
        for i in range(10):
            timestamp = 1609459200 + (i * 60)
            writer.write(OHLCV(timestamp=timestamp, open=100.0 + i, high=110.0 + i, low=90.0 + i, close=105.0 + i,
                               volume=1000.0 + i))

    # Read specific range
    with OHLCVReader(file_path) as reader:
        # Get candles from 3rd to 7th (timestamps 1609459320 to 1609459500)
        candles = list(reader.read_from(1609459320, 1609459500))

        # Should have 4 candles (indices 2, 3, 4, 5)
        assert len(candles) == 4
        assert candles[0].timestamp == 1609459320
        assert candles[-1].timestamp == 1609459500


def __test_ohlcv_reader_rejects_text_disguised_as_ohlcv__(tmp_path):
    """OHLCVReader should raise a clear error when opening a text file renamed to .ohlcv."""
    file_path = tmp_path / "fake_text.ohlcv"

    # Arrange: create a plain-text CSV-like file but with .ohlcv extension
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("timestamp,open,high,low,close,volume\n")
        f.write("1609459200,100,110,90,105,1000\n")

    # Act & Assert: opening via OHLCVReader should fail with a helpful message
    with pytest.raises(ValueError) as excinfo:
        with OHLCVReader(file_path) as _:
            pass  # Should not reach here

    assert "Text file detected with .ohlcv extension" in str(excinfo.value)


def __test_chronological_order_validation__(tmp_path):
    """Test validation of chronological order in timestamps"""
    file_path = tmp_path / "test_chronological.ohlcv"

    # Write two records to establish interval
    with OHLCVWriter(file_path) as writer:
        writer.write(OHLCV(timestamp=1609459200, open=100.0, high=110.0, low=90.0, close=105.0, volume=1000.0))
        writer.write(OHLCV(timestamp=1609459260, open=105.0, high=115.0, low=95.0, close=110.0, volume=1200.0))

    # Now try to write a record with earlier timestamp - should raise ValueError
    with pytest.raises(ValueError) as excinfo:
        with OHLCVWriter(file_path) as writer:
            writer.write(OHLCV(timestamp=1609459100, open=90.0, high=100.0, low=80.0, close=95.0, volume=800.0))

    # Verify the error message contains the expected text
    assert "Timestamps must be in chronological order" in str(excinfo.value)

    # Try writing a timestamp equal to the last one - should also raise ValueError
    with pytest.raises(ValueError) as excinfo:
        with OHLCVWriter(file_path) as writer:
            writer.write(OHLCV(timestamp=1609459260, open=105.0, high=115.0, low=95.0, close=110.0, volume=1200.0))

    # Verify the error message
    assert "Timestamps must be in chronological order" in str(excinfo.value)


def __test_ohlcv_gap_filling_and_skipping__(tmp_path):
    """Test the gap filling functionality and gap skipping during reading"""
    file_path = tmp_path / "test_gaps.ohlcv"

    # Create test data with a gap
    with OHLCVWriter(file_path) as writer:
        # Write initial candles to establish interval
        writer.write(OHLCV(timestamp=1609459200, open=100.0, high=110.0, low=90.0, close=105.0, volume=1000.0))
        writer.write(OHLCV(timestamp=1609459260, open=105.0, high=115.0, low=95.0, close=108.0, volume=1100.0))

        # Skip directly to 1609459380, creating a gap at 1609459320
        # The writer should automatically fill this gap
        writer.write(OHLCV(timestamp=1609459380, open=110.0, high=120.0, low=100.0, close=115.0, volume=1400.0))

    # Verify file size - should have 4 records (3 original + 1 gap filled)
    file_size = os.path.getsize(file_path)
    expected_size = 4 * 24  # 4 records * 24 bytes
    assert file_size == expected_size, f"Expected file size: {expected_size}, actual: {file_size}"

    # Read all data and verify gap is filled correctly
    with OHLCVReader(file_path) as reader:
        # Check total size
        assert reader.size == 4, f"Expected 4 records, found {reader.size}"

        # Read all candles
        candles = list(reader)

        # Verify we have all 4 records including the gap
        assert len(candles) == 4, f"Expected 4 candles, found {len(candles)}"

        # Verify first and second records
        assert candles[0].timestamp == 1609459200
        assert candles[0].close == 105.0
        assert candles[0].volume == 1000.0

        assert candles[1].timestamp == 1609459260
        assert candles[1].close == 108.0
        assert candles[1].volume == 1100.0

        # Verify the gap record
        assert candles[2].timestamp == 1609459320, f"Expected gap at 1609459320, found {candles[2].timestamp}"
        assert candles[2].open == 108.0, f"Gap should use previous close as OHLC values, found open={candles[2].open}"
        assert candles[2].high == 108.0
        assert candles[2].low == 108.0
        assert candles[2].close == 108.0
        assert candles[2].volume == -1.0, f"Gap should have volume -1, found {candles[2].volume}"

        # Verify last record
        assert candles[3].timestamp == 1609459380
        assert candles[3].close == 115.0
        assert candles[3].volume == 1400.0

    # Test reading with skip_gaps=True
    with OHLCVReader(file_path) as reader:
        # Read all real candles (skipping gaps)
        candles = list(reader.read_from(1609459200, skip_gaps=True))

        # Should only have the real data (not gaps)
        assert len(candles) == 3, f"Expected 3 real candles, found {len(candles)}"

        # Verify the records
        assert candles[0].timestamp == 1609459200
        assert candles[1].timestamp == 1609459260
        assert candles[2].timestamp == 1609459380

        # Make sure no gaps are included
        for candle in candles:
            assert candle.volume >= 0, f"Found gap candle with volume {candle.volume}"

    # Test reading with skip_gaps=False
    with OHLCVReader(file_path) as reader:
        # Read all candles including gaps
        candles = list(reader.read_from(1609459200, skip_gaps=False))

        # Should include both real data and gaps
        assert len(candles) == 4, f"Expected 4 candles (with gaps), found {len(candles)}"

        # Verify the records including gap
        assert candles[0].timestamp == 1609459200
        assert candles[1].timestamp == 1609459260
        assert candles[2].timestamp == 1609459320  # Gap record
        assert candles[2].volume == -1.0
        assert candles[3].timestamp == 1609459380

    # Test reading a specific range with skip_gaps=True
    with OHLCVReader(file_path) as reader:
        # Read only candles from 1609459260 to 1609459380 with skip_gaps=True
        candles = list(reader.read_from(1609459260, 1609459380, skip_gaps=True))

        # Should only have the real data in range (skipping gaps)
        assert len(candles) == 2, f"Expected 2 real candles in range, found {len(candles)}"

        # Verify the records
        assert candles[0].timestamp == 1609459260
        assert candles[1].timestamp == 1609459380


def __test_opening_hours_detection_intraday__(tmp_path):
    """Test opening hours detection for intraday timeframes"""
    from datetime import datetime
    file_path = tmp_path / "test_opening_hours_intraday.ohlcv"

    with OHLCVWriter(file_path) as writer:
        # Simulate stock market data: Monday-Friday 9:30-16:00
        # Start from a Monday 9:30 AM EST (2024-01-08 09:30:00)
        base_timestamp = int(datetime(2024, 1, 8, 9, 30).timestamp())

        # Write data for multiple days with 1-minute intervals
        for day in range(5):  # Monday to Friday
            day_offset = day * 86400  # Seconds in a day

            # Trading hours: 9:30 AM to 4:00 PM (6.5 hours = 390 minutes)
            for minute in range(390):
                timestamp = base_timestamp + day_offset + (minute * 60)
                price = 100.0 + (minute * 0.01)  # Gradual price increase
                writer.write(OHLCV(
                    timestamp=timestamp,
                    open=price,
                    high=price + 0.5,
                    low=price - 0.5,
                    close=price + 0.1,
                    volume=1000.0 + minute
                ))

    # Check opening hours detection
    with OHLCVWriter(file_path) as writer:
        opening_hours = writer.analyzed_opening_hours

        # Should detect business hours pattern
        assert opening_hours is not None, "Opening hours should be detected"
        assert len(opening_hours) > 0, "Should have detected some opening hours"

        # Check that we have Monday-Friday entries
        days_with_hours = {interval.day for interval in opening_hours}
        assert 1 in days_with_hours  # Monday
        assert 5 in days_with_hours  # Friday
        assert 6 not in days_with_hours  # Saturday should not be present
        assert 7 not in days_with_hours  # Sunday should not be present


def __test_opening_hours_detection_crypto__(tmp_path):
    """Test opening hours detection for crypto (24/7) markets"""
    from datetime import datetime
    file_path = tmp_path / "test_opening_hours_crypto.ohlcv"

    with OHLCVWriter(file_path) as writer:
        # Simulate crypto data: 24/7 trading
        # Start from a Monday 00:00 UTC (2024-01-08 00:00:00)
        base_timestamp = int(datetime(2024, 1, 8, 0, 0).timestamp())

        # Write data for a full week with 5-minute intervals
        for hour in range(168):  # 7 days * 24 hours
            for five_min in range(12):  # 12 five-minute intervals per hour
                timestamp = base_timestamp + (hour * 3600) + (five_min * 300)
                price = 50000.0 + (hour * 10.0)  # BTC-like prices
                writer.write(OHLCV(
                    timestamp=timestamp,
                    open=price,
                    high=price + 50,
                    low=price - 50,
                    close=price + 10,
                    volume=100.0 + five_min
                ))

    # Check opening hours detection
    with OHLCVWriter(file_path) as writer:
        opening_hours = writer.analyzed_opening_hours

        # Should detect 24/7 pattern
        assert opening_hours is not None, "Opening hours should be detected"
        assert len(opening_hours) == 7, "Should have all 7 days for 24/7 trading"

        # Check that all days are 00:00-23:59
        for interval in opening_hours:
            assert interval.start.hour == 0 and interval.start.minute == 0
            assert interval.end.hour == 23 and interval.end.minute == 59


def __test_opening_hours_detection_daily__(tmp_path):
    """Test opening hours detection for daily timeframes"""
    from datetime import datetime
    file_path = tmp_path / "test_opening_hours_daily.ohlcv"

    with OHLCVWriter(file_path) as writer:
        # Simulate daily stock data: Monday-Friday only
        # Start from a Monday (2024-01-08)
        base_timestamp = int(datetime(2024, 1, 8, 16, 0).timestamp())  # Daily close at 4 PM

        # Write data for 3 weeks (15 business days)
        for week in range(3):
            for day in range(5):  # Monday to Friday only
                timestamp = base_timestamp + (week * 7 * 86400) + (day * 86400)
                price = 150.0 + (week * 5) + day
                writer.write(OHLCV(
                    timestamp=timestamp,
                    open=price,
                    high=price + 2,
                    low=price - 2,
                    close=price + 1,
                    volume=1000000.0
                ))

    # Check opening hours detection
    with OHLCVWriter(file_path) as writer:
        opening_hours = writer.analyzed_opening_hours

        # Should detect weekday-only pattern from daily data
        assert opening_hours is not None, "Opening hours should be detected"
        assert len(opening_hours) == 5, "Should have Monday-Friday for daily stock data"

        # Check that we only have weekdays (1-5)
        days = {interval.day for interval in opening_hours}
        assert days == {1, 2, 3, 4, 5}, "Should only have Monday-Friday"


def __test_opening_hours_insufficient_data__(tmp_path):
    """Test opening hours detection with insufficient data"""
    file_path = tmp_path / "test_opening_hours_insufficient.ohlcv"

    with OHLCVWriter(file_path) as writer:
        # Write only a few data points (less than required minimum)
        base_timestamp = 1609459200
        for i in range(5):  # Only 5 minutes of data
            writer.write(OHLCV(
                timestamp=base_timestamp + (i * 60),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000.0
            ))

    # Check opening hours detection
    with OHLCVWriter(file_path) as writer:
        opening_hours = writer.analyzed_opening_hours

        # Should return None for insufficient data
        assert opening_hours is None, "Should return None for insufficient data"


def __test_ohlcv_txt_conversion_tab_delimited__(tmp_path):
    """Test TXT conversion with tab-delimited format"""
    ohlcv_path = tmp_path / "test_txt_tab.ohlcv"
    txt_path = tmp_path / "test_input_tab.txt"

    # Create tab-delimited test file
    with open(txt_path, 'w') as f:
        f.write("timestamp\topen\thigh\tlow\tclose\tvolume\n")
        f.write("1609459200\t100.0\t110.0\t90.0\t105.0\t1000.0\n")
        f.write("1609459260\t105.0\t115.0\t95.0\t110.0\t1200.0\n")
        f.write("1609459320\t110.0\t120.0\t100.0\t115.0\t1400.0\n")

    # Convert TXT to OHLCV
    with OHLCVWriter(ohlcv_path) as writer:
        writer.load_from_txt(txt_path)

    # Verify converted data
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 3
        assert candles[0].timestamp == 1609459200
        assert candles[0].close == 105.0
        assert candles[0].volume == 1000.0
        assert candles[2].timestamp == 1609459320
        assert candles[2].close == 115.0


def __test_ohlcv_txt_conversion_semicolon_delimited__(tmp_path):
    """Test TXT conversion with semicolon-delimited format"""
    ohlcv_path = tmp_path / "test_txt_semicolon.ohlcv"
    txt_path = tmp_path / "test_input_semicolon.txt"

    # Create semicolon-delimited test file
    with open(txt_path, 'w') as f:
        f.write("timestamp;open;high;low;close;volume\n")
        f.write("1609459200;100.0;110.0;90.0;105.0;1000.0\n")
        f.write("1609459260;105.0;115.0;95.0;110.0;1200.0\n")

    # Convert TXT to OHLCV
    with OHLCVWriter(ohlcv_path) as writer:
        writer.load_from_txt(txt_path)

    # Verify converted data
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 2
        assert candles[0].timestamp == 1609459200
        assert candles[0].close == 105.0
        assert candles[1].timestamp == 1609459260
        assert candles[1].close == 110.0


def __test_ohlcv_txt_conversion_pipe_delimited__(tmp_path):
    """Test TXT conversion with pipe-delimited format"""
    ohlcv_path = tmp_path / "test_txt_pipe.ohlcv"
    txt_path = tmp_path / "test_input_pipe.txt"

    # Create pipe-delimited test file
    with open(txt_path, 'w') as f:
        f.write("timestamp|open|high|low|close|volume\n")
        f.write("1609459200|100.0|110.0|90.0|105.0|1000.0\n")
        f.write("1609459260|105.0|115.0|95.0|110.0|1200.0\n")

    # Convert TXT to OHLCV
    with OHLCVWriter(ohlcv_path) as writer:
        writer.load_from_txt(txt_path)

    # Verify converted data
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 2
        assert candles[0].timestamp == 1609459200
        assert candles[0].close == 105.0


def __test_ohlcv_txt_conversion_quoted_fields__(tmp_path):
    """Test TXT conversion with quoted fields"""
    ohlcv_path = tmp_path / "test_txt_quoted.ohlcv"
    txt_path = tmp_path / "test_input_quoted.txt"

    # Create test file with quoted fields
    with open(txt_path, 'w') as f:
        f.write('timestamp\topen\thigh\tlow\tclose\tvolume\n')
        f.write('1609459200\t"100.0"\t"110.0"\t"90.0"\t"105.0"\t"1000.0"\n')
        f.write("1609459260\t'105.0'\t'115.0'\t'95.0'\t'110.0'\t'1200.0'\n")

    # Convert TXT to OHLCV
    with OHLCVWriter(ohlcv_path) as writer:
        writer.load_from_txt(txt_path)

    # Verify converted data
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 2
        assert candles[0].close == 105.0
        assert candles[1].close == 110.0


def __test_ohlcv_txt_conversion_with_timezone__(tmp_path):
    """Test TXT conversion with timezone handling"""
    ohlcv_path = tmp_path / "test_txt_tz.ohlcv"
    txt_path = tmp_path / "test_input_tz.txt"

    # Create test file with datetime strings
    with open(txt_path, 'w') as f:
        f.write("time\topen\thigh\tlow\tclose\tvolume\n")
        f.write("2025-01-01 12:00:00\t100.0\t110.0\t90.0\t105.0\t1000.0\n")
        f.write("2025-01-01 12:01:00\t105.0\t115.0\t95.0\t110.0\t1200.0\n")

    # Convert TXT to OHLCV with UTC timezone
    with OHLCVWriter(ohlcv_path) as writer:
        writer.load_from_txt(txt_path, tz="UTC")

    # Verify converted data
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 2
        assert candles[0].close == 105.0
        assert candles[1].close == 110.0


def __test_ohlcv_txt_conversion_date_time_columns__(tmp_path):
    """Test TXT conversion with separate date and time columns"""
    ohlcv_path = tmp_path / "test_txt_date_time.ohlcv"
    txt_path = tmp_path / "test_input_date_time.txt"

    # Create test file with separate date/time columns
    with open(txt_path, 'w') as f:
        f.write("date\ttime\topen\thigh\tlow\tclose\tvolume\n")
        f.write("2025-01-01\t12:00:00\t100.0\t110.0\t90.0\t105.0\t1000.0\n")
        f.write("2025-01-01\t12:01:00\t105.0\t115.0\t95.0\t110.0\t1200.0\n")

    # Convert TXT to OHLCV with date/time columns
    with OHLCVWriter(ohlcv_path) as writer:
        writer.load_from_txt(txt_path, date_column="date", time_column="time")

    # Verify converted data
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 2
        assert candles[0].close == 105.0


def __test_ohlcv_txt_conversion_custom_timestamp_format__(tmp_path):
    """Test TXT conversion with custom timestamp format"""
    ohlcv_path = tmp_path / "test_txt_custom_fmt.ohlcv"
    txt_path = tmp_path / "test_input_custom_fmt.txt"

    # Create test file with custom timestamp format
    with open(txt_path, 'w') as f:
        f.write("timestamp\topen\thigh\tlow\tclose\tvolume\n")
        f.write("01.01.2025 12:00:00\t100.0\t110.0\t90.0\t105.0\t1000.0\n")
        f.write("01.01.2025 12:01:00\t105.0\t115.0\t95.0\t110.0\t1200.0\n")

    # Convert TXT to OHLCV with custom timestamp format
    with OHLCVWriter(ohlcv_path) as writer:
        writer.load_from_txt(txt_path, timestamp_format="%d.%m.%Y %H:%M:%S")

    # Verify converted data
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 2
        assert candles[0].close == 105.0


def __test_ohlcv_txt_conversion_whitespace_handling__(tmp_path):
    """Test TXT conversion with extra whitespace"""
    ohlcv_path = tmp_path / "test_txt_whitespace.ohlcv"
    txt_path = tmp_path / "test_input_whitespace.txt"

    # Create test file with extra whitespace
    with open(txt_path, 'w') as f:
        f.write("  timestamp  \t  open  \t  high  \t  low  \t  close  \t  volume  \n")
        f.write("  1609459200  \t  100.0  \t  110.0  \t  90.0  \t  105.0  \t  1000.0  \n")
        f.write("  1609459260  \t  105.0  \t  115.0  \t  95.0  \t  110.0  \t  1200.0  \n")

    # Convert TXT to OHLCV
    with OHLCVWriter(ohlcv_path) as writer:
        writer.load_from_txt(txt_path)

    # Verify converted data
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 2
        assert candles[0].close == 105.0


def __test_ohlcv_txt_conversion_empty_lines__(tmp_path):
    """Test TXT conversion with empty lines"""
    ohlcv_path = tmp_path / "test_txt_empty_lines.ohlcv"
    txt_path = tmp_path / "test_input_empty_lines.txt"

    # Create test file with empty lines
    with open(txt_path, 'w') as f:
        f.write("timestamp\topen\thigh\tlow\tclose\tvolume\n")
        f.write("\n")  # Empty line
        f.write("1609459200\t100.0\t110.0\t90.0\t105.0\t1000.0\n")
        f.write("\n")  # Another empty line
        f.write("1609459260\t105.0\t115.0\t95.0\t110.0\t1200.0\n")
        f.write("\n")  # Final empty line

    # Convert TXT to OHLCV
    with OHLCVWriter(ohlcv_path) as writer:
        writer.load_from_txt(txt_path)

    # Verify converted data
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 2
        assert candles[0].close == 105.0


def __test_ohlcv_txt_conversion_error_cases__(tmp_path):
    """Test TXT conversion error handling"""
    ohlcv_path = tmp_path / "test_txt_errors.ohlcv"

    # Test empty file
    empty_txt = tmp_path / "empty.txt"
    with open(empty_txt, 'w') as f:
        f.write("")

    with pytest.raises(ValueError, match="File is empty"):
        with OHLCVWriter(ohlcv_path) as writer:
            writer.load_from_txt(empty_txt)

    # Test file with no delimiter
    no_delim_txt = tmp_path / "no_delim.txt"
    with open(no_delim_txt, 'w') as f:
        f.write("timestamp open high low close volume\n")  # Space delimited (not supported)
        f.write("1609459200 100.0 110.0 90.0 105.0 1000.0\n")

    with pytest.raises(ValueError, match="No supported delimiter found"):
        with OHLCVWriter(ohlcv_path) as writer:
            writer.load_from_txt(no_delim_txt)

    # Test file with missing required columns
    missing_cols_txt = tmp_path / "missing_cols.txt"
    with open(missing_cols_txt, 'w') as f:
        f.write("timestamp\topen\thigh\n")  # Missing low, close, volume
        f.write("1609459200\t100.0\t110.0\n")

    with pytest.raises(ValueError, match="Missing required column"):
        with OHLCVWriter(ohlcv_path) as writer:
            writer.load_from_txt(missing_cols_txt)

    # Test file with mismatched column count
    mismatch_cols_txt = tmp_path / "mismatch_cols.txt"
    with open(mismatch_cols_txt, 'w') as f:
        f.write("timestamp\topen\thigh\tlow\tclose\tvolume\n")
        f.write("1609459200\t100.0\t110.0\t90.0\t105.0\n")  # Missing volume

    with pytest.raises(ValueError, match="Row has 5 columns, expected 6"):
        with OHLCVWriter(ohlcv_path) as writer:
            writer.load_from_txt(mismatch_cols_txt)


def __test_ohlcv_txt_conversion_escape_characters__(tmp_path):
    """Test TXT conversion with escape characters"""
    ohlcv_path = tmp_path / "test_txt_escape.ohlcv"
    txt_path = tmp_path / "test_input_escape.txt"

    # Create test file with escape characters (though not commonly used in OHLCV data)
    with open(txt_path, 'w') as f:
        f.write("timestamp\topen\thigh\tlow\tclose\tvolume\n")
        f.write("1609459200\t100.0\t110.0\t90.0\t105.0\t1000.0\n")
        f.write("1609459260\t105.0\t115.0\t95.0\t110.0\t1200.0\n")

    # Convert TXT to OHLCV
    with OHLCVWriter(ohlcv_path) as writer:
        writer.load_from_txt(txt_path)

    # Verify converted data
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 2
        assert candles[0].close == 105.0


def __test_ohlcv_txt_mixed_quote_types__(tmp_path):
    """Test TXT conversion with mixed quote types"""
    ohlcv_path = tmp_path / "test_txt_mixed_quotes.ohlcv"
    txt_path = tmp_path / "test_input_mixed_quotes.txt"

    # Create test file with mixed quote types
    with open(txt_path, 'w') as f:
        f.write("timestamp\topen\thigh\tlow\tclose\tvolume\n")
        f.write('1609459200\t"100.0"\t110.0\t"90.0"\t105.0\t"1000.0"\n')
        f.write("1609459260\t'105.0'\t115.0\t'95.0'\t110.0\t'1200.0'\n")

    # Convert TXT to OHLCV
    with OHLCVWriter(ohlcv_path) as writer:
        writer.load_from_txt(txt_path)

    # Verify converted data
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 2
        assert candles[0].close == 105.0
        assert candles[1].close == 110.0


def __test_ohlcv_csv_conversion_with_timezone__(tmp_path):
    """Test CSV conversion with timezone handling"""
    ohlcv_path = tmp_path / "test_csv_tz.ohlcv"
    csv_path = tmp_path / "test_input_tz.csv"

    # Create CSV file with datetime strings
    with open(csv_path, 'w') as f:
        f.write("time,open,high,low,close,volume\n")
        f.write("2025-01-01 12:00:00,100.0,110.0,90.0,105.0,1000.0\n")
        f.write("2025-01-01 12:01:00,105.0,115.0,95.0,110.0,1200.0\n")

    # Convert CSV to OHLCV with UTC timezone
    with OHLCVWriter(ohlcv_path) as writer:
        writer.load_from_csv(csv_path, tz="UTC")

    # Verify converted data
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 2
        assert candles[0].close == 105.0
        assert candles[1].close == 110.0


def __test_ohlcv_csv_conversion_date_time_columns__(tmp_path):
    """Test CSV conversion with separate date and time columns"""
    ohlcv_path = tmp_path / "test_csv_date_time.ohlcv"
    csv_path = tmp_path / "test_input_date_time.csv"

    # Create CSV file with separate date/time columns
    with open(csv_path, 'w') as f:
        f.write("date,time,open,high,low,close,volume\n")
        f.write("2025-01-01,12:00:00,100.0,110.0,90.0,105.0,1000.0\n")
        f.write("2025-01-01,12:01:00,105.0,115.0,95.0,110.0,1200.0\n")

    # Convert CSV to OHLCV with date/time columns
    with OHLCVWriter(ohlcv_path) as writer:
        writer.load_from_csv(csv_path, date_column="date", time_column="time")

    # Verify converted data
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 2
        assert candles[0].close == 105.0


def __test_ohlcv_csv_conversion_custom_timestamp_format__(tmp_path):
    """Test CSV conversion with custom timestamp format"""
    ohlcv_path = tmp_path / "test_csv_custom_fmt.ohlcv"
    csv_path = tmp_path / "test_input_custom_fmt.csv"

    # Create CSV file with custom timestamp format
    with open(csv_path, 'w') as f:
        f.write("timestamp,open,high,low,close,volume\n")
        f.write("01.01.2025 12:00:00,100.0,110.0,90.0,105.0,1000.0\n")
        f.write("01.01.2025 12:01:00,105.0,115.0,95.0,110.0,1200.0\n")

    # Convert CSV to OHLCV with custom timestamp format
    with OHLCVWriter(ohlcv_path) as writer:
        writer.load_from_csv(csv_path, timestamp_format="%d.%m.%Y %H:%M:%S")

    # Verify converted data
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 2
        assert candles[0].close == 105.0


def __test_ohlcv_csv_conversion_custom_timestamp_column__(tmp_path):
    """Test CSV conversion with custom timestamp column name"""
    ohlcv_path = tmp_path / "test_csv_custom_ts.ohlcv"
    csv_path = tmp_path / "test_input_custom_ts.csv"

    # Create CSV file with custom timestamp column name
    with open(csv_path, 'w') as f:
        f.write("datetime,open,high,low,close,volume\n")
        f.write("2025-01-01 12:00:00,100.0,110.0,90.0,105.0,1000.0\n")
        f.write("2025-01-01 12:01:00,105.0,115.0,95.0,110.0,1200.0\n")

    # Convert CSV to OHLCV with custom timestamp column
    with OHLCVWriter(ohlcv_path) as writer:
        writer.load_from_csv(csv_path, timestamp_column="datetime")

    # Verify converted data
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 2
        assert candles[0].close == 105.0


def __test_ohlcv_csv_conversion_quoted_fields__(tmp_path):
    """Test CSV conversion with quoted fields"""
    ohlcv_path = tmp_path / "test_csv_quoted.ohlcv"
    csv_path = tmp_path / "test_input_quoted.csv"

    # Create CSV file with quoted fields
    with open(csv_path, 'w') as f:
        f.write("timestamp,open,high,low,close,volume\n")
        f.write('1609459200,"100.0","110.0","90.0","105.0","1000.0"\n')
        f.write('1609459260,"105.0","115.0","95.0","110.0","1200.0"\n')

    # Convert CSV to OHLCV
    with OHLCVWriter(ohlcv_path) as writer:
        writer.load_from_csv(csv_path)

    # Verify converted data
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 2
        assert candles[0].close == 105.0
        assert candles[1].close == 110.0


def __test_ohlcv_csv_conversion_error_cases__(tmp_path):
    """Test CSV conversion error handling"""
    ohlcv_path = tmp_path / "test_csv_errors.ohlcv"

    # Test file with missing required columns
    missing_cols_csv = tmp_path / "missing_cols.csv"
    with open(missing_cols_csv, 'w') as f:
        f.write("timestamp,open,high\n")  # Missing low, close, volume
        f.write("1609459200,100.0,110.0\n")

    with pytest.raises(ValueError, match="Missing required column"):
        with OHLCVWriter(ohlcv_path) as writer:
            writer.load_from_csv(missing_cols_csv)

    # Test file with invalid timestamp
    invalid_ts_csv = tmp_path / "invalid_ts.csv"
    with open(invalid_ts_csv, 'w') as f:
        f.write("timestamp,open,high,low,close,volume\n")
        f.write("invalid-timestamp,100.0,110.0,90.0,105.0,1000.0\n")

    with pytest.raises(ValueError, match="Failed to parse timestamp"):
        with OHLCVWriter(ohlcv_path) as writer:
            writer.load_from_csv(invalid_ts_csv)


def __test_ohlcv_csv_conversion_case_insensitive__(tmp_path):
    """Test CSV conversion with case insensitive headers"""
    ohlcv_path = tmp_path / "test_csv_case.ohlcv"
    csv_path = tmp_path / "test_input_case.csv"

    # Create CSV file with mixed case headers
    with open(csv_path, 'w') as f:
        f.write("TIMESTAMP,Open,HIGH,low,Close,VOLUME\n")
        f.write("1609459200,100.0,110.0,90.0,105.0,1000.0\n")
        f.write("1609459260,105.0,115.0,95.0,110.0,1200.0\n")

    # Convert CSV to OHLCV
    with OHLCVWriter(ohlcv_path) as writer:
        writer.load_from_csv(csv_path)

    # Verify converted data
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 2
        assert candles[0].close == 105.0


def __test_ohlcv_json_conversion_with_timezone__(tmp_path):
    """Test JSON conversion with timezone handling"""
    ohlcv_path = tmp_path / "test_json_tz.ohlcv"
    json_path = tmp_path / "test_input_tz.json"

    # Create JSON file with datetime strings
    import json
    data = [
        {"time": "2025-01-01 12:00:00", "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0, "volume": 1000.0},
        {"time": "2025-01-01 12:01:00", "open": 105.0, "high": 115.0, "low": 95.0, "close": 110.0, "volume": 1200.0}
    ]
    with open(json_path, 'w') as f:
        json.dump(data, f)

    # Convert JSON to OHLCV with UTC timezone
    with OHLCVWriter(ohlcv_path) as writer:
        writer.load_from_json(json_path, timestamp_field="time", tz="UTC")

    # Verify converted data
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 2
        assert candles[0].close == 105.0
        assert candles[1].close == 110.0


def __test_ohlcv_json_conversion_date_time_fields__(tmp_path):
    """Test JSON conversion with separate date and time fields"""
    ohlcv_path = tmp_path / "test_json_date_time.ohlcv"
    json_path = tmp_path / "test_input_date_time.json"

    # Create JSON file with separate date/time fields
    import json
    data = [
        {"date": "2025-01-01", "time": "12:00:00", "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0,
         "volume": 1000.0},
        {"date": "2025-01-01", "time": "12:01:00", "open": 105.0, "high": 115.0, "low": 95.0, "close": 110.0,
         "volume": 1200.0}
    ]
    with open(json_path, 'w') as f:
        json.dump(data, f)

    # Convert JSON to OHLCV with date/time fields
    with OHLCVWriter(ohlcv_path) as writer:
        writer.load_from_json(json_path, date_field="date", time_field="time")

    # Verify converted data
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 2
        assert candles[0].close == 105.0


def __test_ohlcv_json_conversion_custom_timestamp_format__(tmp_path):
    """Test JSON conversion with custom timestamp format"""
    ohlcv_path = tmp_path / "test_json_custom_fmt.ohlcv"
    json_path = tmp_path / "test_input_custom_fmt.json"

    # Create JSON file with custom timestamp format
    import json
    data = [
        {"timestamp": "01.01.2025 12:00:00", "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0,
         "volume": 1000.0},
        {"timestamp": "01.01.2025 12:01:00", "open": 105.0, "high": 115.0, "low": 95.0, "close": 110.0,
         "volume": 1200.0}
    ]
    with open(json_path, 'w') as f:
        json.dump(data, f)

    # Convert JSON to OHLCV with custom timestamp format
    with OHLCVWriter(ohlcv_path) as writer:
        writer.load_from_json(json_path, timestamp_format="%d.%m.%Y %H:%M:%S")

    # Verify converted data
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 2
        assert candles[0].close == 105.0


def __test_ohlcv_json_conversion_field_mapping__(tmp_path):
    """Test JSON conversion with field mapping"""
    ohlcv_path = tmp_path / "test_json_mapping.ohlcv"
    json_path = tmp_path / "test_input_mapping.json"

    # Create JSON file with custom field names
    import json
    data = [
        {"t": 1609459200, "o": 100.0, "h": 110.0, "l": 90.0, "c": 105.0, "vol": 1000.0},
        {"t": 1609459260, "o": 105.0, "h": 115.0, "l": 95.0, "c": 110.0, "vol": 1200.0}
    ]
    with open(json_path, 'w') as f:
        json.dump(data, f)

    # Convert JSON to OHLCV with field mapping
    mapping = {"timestamp": "t", "open": "o", "high": "h", "low": "l", "close": "c", "volume": "vol"}
    with OHLCVWriter(ohlcv_path) as writer:
        writer.load_from_json(json_path, mapping=mapping)

    # Verify converted data
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 2
        assert candles[0].close == 105.0
        assert candles[1].close == 110.0


def __test_ohlcv_json_conversion_wrapped_data__(tmp_path):
    """Test JSON conversion with wrapped data arrays"""
    ohlcv_path = tmp_path / "test_json_wrapped.ohlcv"
    json_path = tmp_path / "test_input_wrapped.json"

    # Create JSON file with wrapped data array (common API format)
    import json
    wrapped_data = {
        "data": [
            {"timestamp": 1609459200, "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0, "volume": 1000.0},
            {"timestamp": 1609459260, "open": 105.0, "high": 115.0, "low": 95.0, "close": 110.0, "volume": 1200.0}
        ]
    }
    with open(json_path, 'w') as f:
        json.dump(wrapped_data, f)

    # Convert JSON to OHLCV
    with OHLCVWriter(ohlcv_path) as writer:
        writer.load_from_json(json_path)

    # Verify converted data
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 2
        assert candles[0].close == 105.0


def __test_ohlcv_json_conversion_millisecond_timestamps__(tmp_path):
    """Test JSON conversion with millisecond timestamps"""
    ohlcv_path = tmp_path / "test_json_ms.ohlcv"
    json_path = tmp_path / "test_input_ms.json"

    # Create JSON file with millisecond timestamps
    import json
    data = [
        {"timestamp": 1609459200000, "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0, "volume": 1000.0},
        {"timestamp": 1609459260000, "open": 105.0, "high": 115.0, "low": 95.0, "close": 110.0, "volume": 1200.0}
    ]
    with open(json_path, 'w') as f:
        json.dump(data, f)

    # Convert JSON to OHLCV (should auto-detect and convert milliseconds)
    with OHLCVWriter(ohlcv_path) as writer:
        writer.load_from_json(json_path)

    # Verify converted data
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 2
        assert candles[0].timestamp == 1609459200  # Should be converted from ms to s
        assert candles[1].timestamp == 1609459260
        assert candles[0].close == 105.0


def __test_ohlcv_json_conversion_auto_field_detection__(tmp_path):
    """Test JSON conversion with automatic field detection"""
    ohlcv_path = tmp_path / "test_json_auto.ohlcv"
    json_path = tmp_path / "test_input_auto.json"

    # Create JSON file with 't' timestamp field (should auto-detect)
    import json
    data = [
        {"t": 1609459200, "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0, "volume": 1000.0},
        {"t": 1609459260, "open": 105.0, "high": 115.0, "low": 95.0, "close": 110.0, "volume": 1200.0}
    ]
    with open(json_path, 'w') as f:
        json.dump(data, f)

    # Convert JSON to OHLCV (should auto-detect 't' field)
    with OHLCVWriter(ohlcv_path) as writer:
        writer.load_from_json(json_path)

    # Verify converted data
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 2
        assert candles[0].close == 105.0


def __test_ohlcv_json_conversion_error_cases__(tmp_path):
    """Test JSON conversion error handling"""
    ohlcv_path = tmp_path / "test_json_errors.ohlcv"

    # Test file with missing required fields
    missing_fields_json = tmp_path / "missing_fields.json"
    import json
    data = [
        {"timestamp": 1609459200, "open": 100.0, "high": 110.0}  # Missing low, close, volume
    ]
    with open(missing_fields_json, 'w') as f:
        json.dump(data, f)

    with pytest.raises(ValueError, match="Missing field in record"):
        with OHLCVWriter(ohlcv_path) as writer:
            writer.load_from_json(missing_fields_json)

    # Test file with no timestamp field
    no_timestamp_json = tmp_path / "no_timestamp.json"
    data = [
        {"open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0, "volume": 1000.0}  # No timestamp
    ]
    with open(no_timestamp_json, 'w') as f:
        json.dump(data, f)

    with pytest.raises(ValueError, match="Could not find timestamp field"):
        with OHLCVWriter(ohlcv_path) as writer:
            writer.load_from_json(no_timestamp_json)

    # Test file with no OHLCV data array
    no_data_json = tmp_path / "no_data.json"
    data = {"metadata": "some info", "status": "ok"}  # No data array
    with open(no_data_json, 'w') as f:
        json.dump(data, f)

    with pytest.raises(ValueError, match="Could not find OHLCV data array"):
        with OHLCVWriter(ohlcv_path) as writer:
            writer.load_from_json(no_data_json)


def __test_ohlcv_json_conversion_alternative_wrappers__(tmp_path):
    """Test JSON conversion with alternative wrapper keys"""
    ohlcv_path = tmp_path / "test_json_alt_wrap.ohlcv"
    json_path = tmp_path / "test_input_alt_wrap.json"

    # Create JSON file with 'candles' wrapper (alternative to 'data')
    import json
    wrapped_data = {
        "candles": [
            {"timestamp": 1609459200, "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0, "volume": 1000.0},
            {"timestamp": 1609459260, "open": 105.0, "high": 115.0, "low": 95.0, "close": 110.0, "volume": 1200.0}
        ]
    }
    with open(json_path, 'w') as f:
        json.dump(wrapped_data, f)

    # Convert JSON to OHLCV
    with OHLCVWriter(ohlcv_path) as writer:
        writer.load_from_json(json_path)

    # Verify converted data
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 2
        assert candles[0].close == 105.0
