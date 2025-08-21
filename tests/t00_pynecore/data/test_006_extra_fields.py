"""
@pyne
"""
import os
import pytest
import json
from pathlib import Path

from pynecore.types.ohlcv import OHLCV
from pynecore.core.ohlcv_file import OHLCVWriter, OHLCVReader


def main():
    """
    Dummy main function to be a valid Pyne script
    """
    pass


def _create_test_csv_with_extra_fields(csv_path: Path, include_booleans: bool = False):
    """
    Create a CSV file with extra fields for testing
    
    :param csv_path: Path to create CSV file
    :param include_booleans: Whether to include boolean extra fields
    """
    with open(csv_path, 'w') as f:
        if include_booleans:
            f.write("timestamp,open,high,low,close,volume,volume_sma,custom_indicator,is_bullish,is_bearish,above_ma\n")
            test_data = [
                "1609459200,29000.0,29500.0,28800.0,29200.0,1000.5,950.3,1.25,true,false,true",
                "1609545600,29200.0,29800.0,29000.0,29600.0,1200.3,975.8,1.31,true,false,true",
                "1609632000,29600.0,30200.0,29400.0,29900.0,1100.7,1000.2,1.28,true,false,false",
                "1609718400,29900.0,30500.0,29700.0,30100.0,1300.2,1025.1,1.35,false,true,false",
                "1609804800,30100.0,30800.0,29900.0,30400.0,1150.9,1050.6,1.42,true,false,true"
            ]
        else:
            f.write("timestamp,open,high,low,close,volume,volume_sma,custom_indicator,rsi_14,bollinger_upper\n")
            test_data = [
                "1609459200,29000.0,29500.0,28800.0,29200.0,1000.5,950.3,1.25,45.2,29800.5",
                "1609545600,29200.0,29800.0,29000.0,29600.0,1200.3,975.8,1.31,48.7,30100.2",
                "1609632000,29600.0,30200.0,29400.0,29900.0,1100.7,1000.2,1.28,52.1,30400.8",
                "1609718400,29900.0,30500.0,29700.0,30100.0,1300.2,1025.1,1.35,55.8,30700.1",
                "1609804800,30100.0,30800.0,29900.0,30400.0,1150.9,1050.6,1.42,58.9,31000.4"
            ]

        for row in test_data:
            f.write(row + "\n")


def _create_test_ohlcv_with_extra_fields(ohlcv_path: Path, include_booleans: bool = False):
    """
    Create an OHLCV file with extra fields for testing
    
    :param ohlcv_path: Path to create OHLCV file
    :param include_booleans: Whether to include boolean extra fields
    """
    with OHLCVWriter(ohlcv_path) as writer:
        if include_booleans:
            test_data = [
                (1609459200, 29000.0, 29500.0, 28800.0, 29200.0, 1000.5,
                 {"volume_sma": 950.3, "custom_indicator": 1.25, "is_bullish": True, "is_bearish": False,
                  "above_ma": True}),
                (1609545600, 29200.0, 29800.0, 29000.0, 29600.0, 1200.3,
                 {"volume_sma": 975.8, "custom_indicator": 1.31, "is_bullish": True, "is_bearish": False,
                  "above_ma": True}),
                (1609632000, 29600.0, 30200.0, 29400.0, 29900.0, 1100.7,
                 {"volume_sma": 1000.2, "custom_indicator": 1.28, "is_bullish": True, "is_bearish": False,
                  "above_ma": False}),
                (1609718400, 29900.0, 30500.0, 29700.0, 30100.0, 1300.2,
                 {"volume_sma": 1025.1, "custom_indicator": 1.35, "is_bullish": False, "is_bearish": True,
                  "above_ma": False}),
                (1609804800, 30100.0, 30800.0, 29900.0, 30400.0, 1150.9,
                 {"volume_sma": 1050.6, "custom_indicator": 1.42, "is_bullish": True, "is_bearish": False,
                  "above_ma": True})
            ]
        else:
            test_data = [
                (1609459200, 29000.0, 29500.0, 28800.0, 29200.0, 1000.5,
                 {"volume_sma": 950.3, "custom_indicator": 1.25, "rsi_14": 45.2, "bollinger_upper": 29800.5}),
                (1609545600, 29200.0, 29800.0, 29000.0, 29600.0, 1200.3,
                 {"volume_sma": 975.8, "custom_indicator": 1.31, "rsi_14": 48.7, "bollinger_upper": 30100.2}),
                (1609632000, 29600.0, 30200.0, 29400.0, 29900.0, 1100.7,
                 {"volume_sma": 1000.2, "custom_indicator": 1.28, "rsi_14": 52.1, "bollinger_upper": 30400.8}),
                (1609718400, 29900.0, 30500.0, 29700.0, 30100.0, 1300.2,
                 {"volume_sma": 1025.1, "custom_indicator": 1.35, "rsi_14": 55.8, "bollinger_upper": 30700.1}),
                (1609804800, 30100.0, 30800.0, 29900.0, 30400.0, 1150.9,
                 {"volume_sma": 1050.6, "custom_indicator": 1.42, "rsi_14": 58.9, "bollinger_upper": 31000.4})
            ]

        for timestamp, open_val, high, low, close_val, volume, extra in test_data:
            writer.write(OHLCV(
                timestamp=timestamp,
                open=open_val,
                high=high,
                low=low,
                close=close_val,
                volume=volume,
                extra_fields=extra
            ))


def __test_extra_fields_csv_conversion__(tmp_path):
    """
    Test CSV to OHLCV conversion with extra fields
    """
    csv_path = tmp_path / "test_extra_fields.csv"
    ohlcv_path = tmp_path / "test_extra_fields.ohlcv"

    # Create CSV with extra fields
    _create_test_csv_with_extra_fields(csv_path)

    # Convert CSV to OHLCV
    with OHLCVWriter(ohlcv_path) as writer:
        writer.load_from_csv(csv_path)

    # Verify OHLCV file was created and has correct size
    assert ohlcv_path.exists()
    assert os.path.getsize(ohlcv_path) == 120  # 5 records * 24 bytes

    # Verify extra_fields JSON file was created
    extra_fields_path = ohlcv_path.with_suffix('.extra_fields.json')
    assert extra_fields_path.exists()

    # Read and verify extra fields data
    with open(extra_fields_path, 'r') as f:
        extra_data = json.load(f)

    # Verify extra fields content structure
    assert len(extra_data) == 5
    assert "1609459200" in extra_data
    assert extra_data["1609459200"]["volume_sma"] == 950.3
    assert extra_data["1609459200"]["custom_indicator"] == 1.25
    assert extra_data["1609459200"]["rsi_14"] == 45.2
    assert extra_data["1609459200"]["bollinger_upper"] == 29800.5

    # Verify OHLCV data can be read with extra fields
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 5

        # Check first candle
        first_candle = candles[0]
        assert first_candle.timestamp == 1609459200
        assert first_candle.open == 29000.0
        assert first_candle.high == 29500.0
        assert first_candle.low == 28800.0
        assert first_candle.close == 29200.0
        assert first_candle.volume == 1000.5
        assert first_candle.extra_fields is not None
        assert first_candle.extra_fields["volume_sma"] == 950.3
        assert first_candle.extra_fields["custom_indicator"] == 1.25

        # Check last candle
        last_candle = candles[-1]
        assert last_candle.timestamp == 1609804800
        assert last_candle.extra_fields["volume_sma"] == 1050.6
        assert last_candle.extra_fields["custom_indicator"] == 1.42


def __test_extra_fields_boolean_types__(tmp_path):
    """
    Test boolean extra fields handling
    """
    csv_path = tmp_path / "test_boolean_fields.csv"
    ohlcv_path = tmp_path / "test_boolean_fields.ohlcv"

    # Create CSV with boolean extra fields
    _create_test_csv_with_extra_fields(csv_path, include_booleans=True)

    # Convert CSV to OHLCV
    with OHLCVWriter(ohlcv_path) as writer:
        writer.load_from_csv(csv_path)

    # Verify extra_fields JSON file was created
    extra_fields_path = ohlcv_path.with_suffix('.extra_fields.json')
    assert extra_fields_path.exists()

    # Read and verify boolean fields
    with open(extra_fields_path, 'r') as f:
        extra_data = json.load(f)

    # Verify boolean fields are correctly parsed
    assert extra_data["1609459200"]["is_bullish"] is True
    assert extra_data["1609459200"]["is_bearish"] is False
    assert extra_data["1609459200"]["above_ma"] is True

    # Verify reading works correctly
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        first_candle = candles[0]
        assert first_candle.extra_fields["is_bullish"] is True
        assert first_candle.extra_fields["is_bearish"] is False


def __test_extra_fields_direct_ohlcv_creation__(tmp_path):
    """
    Test creating OHLCV files directly with extra fields
    """
    ohlcv_path = tmp_path / "test_direct_extra.ohlcv"

    # Create OHLCV with extra fields directly
    _create_test_ohlcv_with_extra_fields(ohlcv_path)

    # Verify extra_fields JSON file was created
    extra_fields_path = ohlcv_path.with_suffix('.extra_fields.json')
    assert extra_fields_path.exists()

    # Read and verify data
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 5

        # Verify extra fields are accessible
        for i, candle in enumerate(candles):
            assert candle.extra_fields is not None
            assert "volume_sma" in candle.extra_fields
            assert "custom_indicator" in candle.extra_fields
            assert "rsi_14" in candle.extra_fields
            assert "bollinger_upper" in candle.extra_fields


def __test_extra_fields_missing_data__(tmp_path):
    """
    Test handling of missing extra fields data
    """
    ohlcv_path = tmp_path / "test_missing_extra.ohlcv"

    # Create OHLCV file without extra fields
    with OHLCVWriter(ohlcv_path) as writer:
        writer.write(OHLCV(
            timestamp=1609459200,
            open=29000.0,
            high=29500.0,
            low=28800.0,
            close=29200.0,
            volume=1000.5
        ))

    # Verify no extra_fields JSON file is created
    extra_fields_path = ohlcv_path.with_suffix('.extra_fields.json')
    assert not extra_fields_path.exists()

    # Verify reading works without extra fields
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 1
        assert candles[0].extra_fields is None or candles[0].extra_fields == {}


def __test_extra_fields_mixed_data__(tmp_path):
    """
    Test handling of mixed data with some records having extra fields
    """
    ohlcv_path = tmp_path / "test_mixed_extra.ohlcv"

    # Create OHLCV with mixed extra fields
    with OHLCVWriter(ohlcv_path) as writer:
        # First record with extra fields
        writer.write(OHLCV(
            timestamp=1609459200,
            open=29000.0,
            high=29500.0,
            low=28800.0,
            close=29200.0,
            volume=1000.5,
            extra_fields={"volume_sma": 950.3, "custom_indicator": 1.25}
        ))

        # Second record without extra fields
        writer.write(OHLCV(
            timestamp=1609459260,
            open=29200.0,
            high=29800.0,
            low=29000.0,
            close=29600.0,
            volume=1200.3
        ))

        # Third record with different extra fields
        writer.write(OHLCV(
            timestamp=1609459320,
            open=29600.0,
            high=30200.0,
            low=29400.0,
            close=29900.0,
            volume=1100.7,
            extra_fields={"rsi_14": 52.1, "bollinger_upper": 30400.8}
        ))

    # Verify extra_fields JSON file was created
    extra_fields_path = ohlcv_path.with_suffix('.extra_fields.json')
    assert extra_fields_path.exists()

    # Read and verify mixed data
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 3

        # First candle has extra fields
        assert candles[0].extra_fields is not None
        assert "volume_sma" in candles[0].extra_fields

        # Second candle has no extra fields
        assert candles[1].extra_fields is None or candles[1].extra_fields == {}

        # Third candle has different extra fields
        assert candles[2].extra_fields is not None
        assert "rsi_14" in candles[2].extra_fields


def __test_extra_fields_string_values__(tmp_path):
    """
    Test handling of string values in extra fields
    """
    csv_path = tmp_path / "test_string_fields.csv"
    ohlcv_path = tmp_path / "test_string_fields.ohlcv"

    # Create CSV with string extra fields
    with open(csv_path, 'w') as f:
        f.write("timestamp,open,high,low,close,volume,trend,signal_type,market_state\n")
        f.write("1609459200,29000.0,29500.0,28800.0,29200.0,1000.5,bullish,buy,active\n")
        f.write("1609545600,29200.0,29800.0,29000.0,29600.0,1200.3,bearish,sell,inactive\n")

    # Convert CSV to OHLCV
    with OHLCVWriter(ohlcv_path) as writer:
        writer.load_from_csv(csv_path)

    # Verify string fields are preserved
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 2

        assert candles[0].extra_fields["trend"] == "bullish"
        assert candles[0].extra_fields["signal_type"] == "buy"
        assert candles[0].extra_fields["market_state"] == "active"

        assert candles[1].extra_fields["trend"] == "bearish"
        assert candles[1].extra_fields["signal_type"] == "sell"
        assert candles[1].extra_fields["market_state"] == "inactive"


def __test_extra_fields_large_dataset__(tmp_path):
    """
    Test extra fields with larger dataset
    """
    ohlcv_path = tmp_path / "test_large_extra.ohlcv"

    # Create larger dataset with extra fields
    with OHLCVWriter(ohlcv_path) as writer:
        base_timestamp = 1609459200
        for i in range(100):
            timestamp = base_timestamp + (i * 60)  # 1-minute intervals
            price = 29000.0 + (i * 10.0)

            writer.write(OHLCV(
                timestamp=timestamp,
                open=price,
                high=price + 20.0,
                low=price - 20.0,
                close=price + 5.0,
                volume=1000.0 + i,
                extra_fields={
                    "volume_sma": 950.0 + i,
                    "custom_indicator": 1.25 + (i * 0.01),
                    "rsi_14": 45.0 + (i * 0.1),
                    "is_bullish": i % 2 == 0,
                    "signal_strength": i / 100.0
                }
            ))

    # Verify all data is correctly stored
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 100

        # Check first and last records
        first_candle = candles[0]
        assert first_candle.extra_fields["volume_sma"] == 950.0
        assert first_candle.extra_fields["is_bullish"] is True

        last_candle = candles[-1]
        assert last_candle.extra_fields["volume_sma"] == 950.0 + 99
        assert last_candle.extra_fields["is_bullish"] is False
        assert abs(last_candle.extra_fields["signal_strength"] - 0.99) < 0.001


def __test_extra_fields_json_file_corruption_handling__(tmp_path):
    """
    Test handling of corrupted extra_fields JSON files
    """
    ohlcv_path = tmp_path / "test_corrupt_extra.ohlcv"

    # Create OHLCV file with extra fields
    _create_test_ohlcv_with_extra_fields(ohlcv_path)

    # Corrupt the JSON file
    extra_fields_path = ohlcv_path.with_suffix('.extra_fields.json')
    with open(extra_fields_path, 'w') as f:
        f.write("{invalid json content")

    # Verify reading still works (should handle corruption gracefully)
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 5

        # Extra fields should be empty due to corruption
        for candle in candles:
            assert candle.extra_fields is None or candle.extra_fields == {}


def __test_extra_fields_json_file_missing__(tmp_path):
    """
    Test handling when extra_fields JSON file is missing
    """
    ohlcv_path = tmp_path / "test_missing_json.ohlcv"

    # Create OHLCV file with extra fields
    _create_test_ohlcv_with_extra_fields(ohlcv_path)

    # Remove the JSON file
    extra_fields_path = ohlcv_path.with_suffix('.extra_fields.json')
    extra_fields_path.unlink()

    # Verify reading still works without extra fields
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 5

        # Extra fields should be empty
        for candle in candles:
            assert candle.extra_fields is None or candle.extra_fields == {}


def __test_extra_fields_save_to_csv_with_extra_fields__(tmp_path):
    """
    Test saving OHLCV data back to CSV with extra fields preserved
    """
    ohlcv_path = tmp_path / "test_save_extra.ohlcv"
    output_csv_path = tmp_path / "output_with_extra.csv"

    # Create OHLCV with extra fields
    _create_test_ohlcv_with_extra_fields(ohlcv_path)

    # Save to CSV
    with OHLCVReader(ohlcv_path) as reader:
        reader.save_to_csv(str(output_csv_path))

    # Verify CSV was created and contains basic OHLCV data
    assert output_csv_path.exists()

    with open(output_csv_path, 'r') as f:
        lines = f.readlines()
        assert len(lines) >= 6  # Header + 5 data rows

        # Check header contains standard OHLCV columns
        header = lines[0].strip()
        assert "timestamp" in header
        assert "open" in header
        assert "high" in header
        assert "low" in header
        assert "close" in header
        assert "volume" in header

        # Verify data rows
        assert "1609459200" in lines[1]
        assert "29200" in lines[1]  # close price


def __test_extra_fields_save_to_json_with_extra_fields__(tmp_path):
    """
    Test saving OHLCV data back to JSON with extra fields preserved
    """
    ohlcv_path = tmp_path / "test_save_json_extra.ohlcv"
    output_json_path = tmp_path / "output_with_extra.json"

    # Create OHLCV with extra fields
    _create_test_ohlcv_with_extra_fields(ohlcv_path)

    # Save to JSON
    with OHLCVReader(ohlcv_path) as reader:
        reader.save_to_json(str(output_json_path))

    # Verify JSON was created and contains basic OHLCV data
    assert output_json_path.exists()

    with open(output_json_path, 'r') as f:
        content = f.read()
        assert "1609459200" in content
        assert "29200" in content  # close price
        assert '"open"' in content
        assert '"close"' in content


def __test_extra_fields_numeric_edge_cases__(tmp_path):
    """
    Test numeric edge cases in extra fields
    """
    ohlcv_path = tmp_path / "test_numeric_edge.ohlcv"

    # Create OHLCV with numeric edge cases
    with OHLCVWriter(ohlcv_path) as writer:
        writer.write(OHLCV(
            timestamp=1609459200,
            open=29000.0,
            high=29500.0,
            low=28800.0,
            close=29200.0,
            volume=1000.5,
            extra_fields={
                "zero_value": 0.0,
                "negative_value": -123.45,
                "very_small": 1e-10,
                "very_large": 1e10,
                "infinity": float('inf'),
                "negative_infinity": float('-inf'),
                "nan_value": float('nan')
            }
        ))

    # Verify reading works with edge cases
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 1

        extra = candles[0].extra_fields
        assert extra["zero_value"] == 0.0
        assert extra["negative_value"] == -123.45
        assert extra["very_small"] == 1e-10
        assert extra["very_large"] == 1e10
        assert extra["infinity"] == float('inf')
        assert extra["negative_infinity"] == float('-inf')
        import math
        assert math.isnan(extra["nan_value"])


def __test_extra_fields_empty_field_names__(tmp_path):
    """
    Test handling of empty or unusual field names
    """
    csv_path = tmp_path / "test_empty_fields.csv"
    ohlcv_path = tmp_path / "test_empty_fields.ohlcv"

    # Create CSV with unusual field names
    with open(csv_path, 'w') as f:
        f.write('timestamp,open,high,low,close,volume,"","  ","field with spaces","123numeric","special!@#$%"\n')
        f.write("1609459200,29000.0,29500.0,28800.0,29200.0,1000.5,1.0,2.0,3.0,4.0,5.0\n")

    # Convert CSV to OHLCV
    with OHLCVWriter(ohlcv_path) as writer:
        writer.load_from_csv(csv_path)

    # Verify unusual field names are handled
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        extra = candles[0].extra_fields

        # Empty field names should be preserved or handled gracefully
        assert "" in extra or len([k for k in extra.keys() if not k.strip()]) > 0
        assert "field with spaces" in extra
        assert "123numeric" in extra
        assert "special!@#$%" in extra


def __test_extra_fields_unicode_and_special_characters__(tmp_path):
    """
    Test Unicode and special characters in extra fields
    """
    ohlcv_path = tmp_path / "test_unicode.ohlcv"

    # Create OHLCV with Unicode field names and values
    with OHLCVWriter(ohlcv_path) as writer:
        writer.write(OHLCV(
            timestamp=1609459200,
            open=29000.0,
            high=29500.0,
            low=28800.0,
            close=29200.0,
            volume=1000.5,
            extra_fields={
                "价格": 1234.56,  # Chinese
                "сигнал": 78.90,  # Russian
                "field_with_émojì": 42.0,  # Accented chars
                "string_with_unicode": "测试数据📈",
                "multiline_string": "line1\nline2\nline3",
                "quotes_string": 'contains "quotes" and \'apostrophes\'',
                "json_like": '{"nested": "value"}'
            }
        ))

    # Verify Unicode handling
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        extra = candles[0].extra_fields

        assert extra["价格"] == 1234.56
        assert extra["сигнал"] == 78.90
        assert extra["field_with_émojì"] == 42.0
        assert extra["string_with_unicode"] == "测试数据📈"
        assert extra["multiline_string"] == "line1\nline2\nline3"


def __test_extra_fields_type_consistency__(tmp_path):
    """
    Test type consistency across reads/writes
    """
    ohlcv_path = tmp_path / "test_types.ohlcv"

    # Original data with mixed types
    original_extra = {
        "int_as_float": 42,
        "float_value": 42.5,
        "bool_true": True,
        "bool_false": False,
        "string_number": "123.45",
        "string_text": "hello world",
        "none_value": None
    }

    # Write data
    with OHLCVWriter(ohlcv_path) as writer:
        writer.write(OHLCV(
            timestamp=1609459200,
            open=29000.0,
            high=29500.0,
            low=28800.0,
            close=29200.0,
            volume=1000.5,
            extra_fields=original_extra
        ))

    # Read back and verify types are preserved or consistently converted
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        extra = candles[0].extra_fields

        # Numeric values should be preserved or converted consistently
        assert isinstance(extra["int_as_float"], (int, float))
        assert isinstance(extra["float_value"], float)
        assert isinstance(extra["bool_true"], bool)
        assert isinstance(extra["bool_false"], bool)
        assert isinstance(extra["string_number"], str)
        assert isinstance(extra["string_text"], str)

        # Check values
        assert extra["bool_true"] is True
        assert extra["bool_false"] is False
        assert extra["string_number"] == "123.45"
        assert extra["string_text"] == "hello world"


def __test_extra_fields_concurrent_access__(tmp_path):
    """
    Test concurrent read/write access to extra fields files
    """
    ohlcv_path = tmp_path / "test_concurrent.ohlcv"

    # Create initial file
    _create_test_ohlcv_with_extra_fields(ohlcv_path)

    # Test multiple readers can access simultaneously
    for i in range(3):
        with OHLCVReader(ohlcv_path) as reader:
            candles = list(reader)
            assert len(candles) == 5
            assert candles[0].extra_fields is not None
            assert "volume_sma" in candles[0].extra_fields


def __test_extra_fields_malformed_csv_handling__(tmp_path):
    """
    Test handling of malformed CSV with extra fields
    """
    csv_path = tmp_path / "test_malformed.csv"
    ohlcv_path = tmp_path / "test_malformed.ohlcv"

    # Create malformed CSV
    with open(csv_path, 'w') as f:
        f.write("timestamp,open,high,low,close,volume,extra1,extra2\n")
        f.write("1609459200,29000.0,29500.0,28800.0,29200.0,1000.5,1.25\n")  # Missing extra2
        f.write("1609459260,29200.0,29800.0,29000.0,29600.0,1200.3,1.31,2.62,extra_value\n")  # Extra value
        f.write("1609459320,,29800.0,29000.0,29600.0,1200.3,1.35,2.70\n")  # Missing open

    # Should handle malformed data gracefully
    with pytest.raises(ValueError):  # Should raise error for malformed data
        with OHLCVWriter(ohlcv_path) as writer:
            writer.load_from_csv(csv_path)


def __test_extra_fields_json_structure_validation__(tmp_path):
    """
    Test JSON structure validation
    """
    ohlcv_path = tmp_path / "test_json_validation.ohlcv"

    # Create OHLCV file
    _create_test_ohlcv_with_extra_fields(ohlcv_path)

    # Manually corrupt the JSON structure
    extra_fields_path = ohlcv_path.with_suffix('.extra_fields.json')
    with open(extra_fields_path, 'w') as f:
        json.dump({
            "invalid_timestamp_format": {"field": "value"},  # Non-numeric timestamp
            1609459200: {"field": "value"},  # Numeric timestamp (should be string)
            "1609459260": {"nested": {"deeply": {"nested": "value"}}},  # Nested structure
            "1609459320": [],  # Array instead of object
            "1609459380": "string_value"  # String instead of object
        }, f)

    # Should handle invalid JSON structure gracefully
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 5

        # Should either parse valid entries or default to no extra fields
        for candle in candles:
            if candle.extra_fields:
                assert isinstance(candle.extra_fields, dict)


def __test_extra_fields_performance_large_fields__(tmp_path):
    """Test performance with large number of extra fields"""
    ohlcv_path = tmp_path / "test_performance.ohlcv"

    # Create large extra fields dictionary
    large_extra_fields = {}
    for i in range(1000):  # 1000 extra fields
        large_extra_fields[f"field_{i:04d}"] = i * 1.5

    # Write with many extra fields
    with OHLCVWriter(ohlcv_path) as writer:
        for j in range(10):  # 10 records
            writer.write(OHLCV(
                timestamp=1609459200 + j * 60,
                open=29000.0 + j,
                high=29500.0 + j,
                low=28800.0 + j,
                close=29200.0 + j,
                volume=1000.5 + j,
                extra_fields=large_extra_fields
            ))

    # Verify all data is accessible
    with OHLCVReader(ohlcv_path) as reader:
        candles = list(reader)
        assert len(candles) == 10

        for candle in candles:
            assert len(candle.extra_fields) == 1000
            assert candle.extra_fields["field_0000"] == 0.0
            assert candle.extra_fields["field_0999"] == 999 * 1.5
