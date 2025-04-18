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
            writer.write(OHLCV(timestamp=timestamp, open=100.0+i, high=110.0 +
                         i, low=90.0+i, close=105.0+i, volume=1000.0+i))

    # Test seeking to a specific position and direct write
    # Note: we use low-level file operations to bypass timestamp checks
    with OHLCVWriter(file_path) as writer:
        writer.seek(5)  # Seek to 6th record

        # Use direct byte writing to avoid chronological checks
        assert writer._file is not None
        data = struct.pack(
            'Ifffff',
            1609459500,  # Timestamp - same as the original at position 5
            200.0,       # Open
            210.0,       # High
            190.0,       # Low
            205.0,       # Close
            2000.0       # Volume
        )
        writer._file.write(data)
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
            writer.write(OHLCV(timestamp=timestamp, open=100.0+i, high=110.0 +
                         i, low=90.0+i, close=105.0+i, volume=1000.0+i))

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
            writer.write(OHLCV(timestamp=timestamp, open=100.0+i, high=110.0 +
                         i, low=90.0+i, close=105.0+i, volume=1000.0+i))

    # Read specific range
    with OHLCVReader(file_path) as reader:
        # Get candles from 3rd to 7th (timestamps 1609459320 to 1609459500)
        candles = list(reader.read_from(1609459320, 1609459500))

        # Should have 4 candles (indices 2, 3, 4, 5)
        assert len(candles) == 4
        assert candles[0].timestamp == 1609459320
        assert candles[-1].timestamp == 1609459500


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
