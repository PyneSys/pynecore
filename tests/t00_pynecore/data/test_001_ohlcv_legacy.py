"""
@pyne
"""
import json
import struct
from datetime import UTC, datetime

import pytest

from pynecore.core.ohlcv_legacy import OHLCVReader, RECORD_SIZE, STRUCT_FORMAT


def main():
    """Provide a valid Pyne code entry point."""


def _write_legacy_records(path, records: list[tuple[int, float, float, float, float, float]]) -> None:
    with open(path, "wb") as output_file:
        for record in records:
            output_file.write(struct.pack(STRUCT_FORMAT, *record))


def __test_legacy_reader_uses_millisecond_timestamps__(tmp_path):
    file_path = tmp_path / "legacy.ohlcv"
    records = [
        (1_609_459_200, 100.0, 110.0, 90.0, 105.0, 1_000.0),
        (1_609_459_260, 105.0, 115.0, 95.0, 110.0, 1_200.0),
    ]
    _write_legacy_records(file_path, records)

    assert file_path.stat().st_size == len(records) * RECORD_SIZE
    with OHLCVReader(file_path) as reader:
        candles = list(reader)
        assert reader.size == 2
        assert reader.start_timestamp == 1_609_459_200_000
        assert reader.end_timestamp == 1_609_459_260_000
        assert reader.interval == 60_000
        assert reader.start_datetime == datetime(2021, 1, 1, tzinfo=UTC)
        assert reader.end_datetime == datetime(2021, 1, 1, 0, 1, tzinfo=UTC)

    assert candles[0].timestamp == 1_609_459_200_000
    assert candles[0].close == 105.0
    assert candles[1].timestamp == 1_609_459_260_000
    assert candles[1].volume == 1_200.0


def __test_legacy_reader_handles_single_record__(tmp_path):
    file_path = tmp_path / "single.ohlcv"
    _write_legacy_records(file_path, [(1_609_459_200, 100.0, 101.0, 99.0, 100.5, 10.0)])

    with OHLCVReader(file_path) as reader:
        assert reader.size == 1
        assert reader.start_timestamp == 1_609_459_200_000
        assert reader.end_timestamp == 1_609_459_200_000
        assert reader.interval is None
        assert reader.read(0).timestamp == 1_609_459_200_000

        # One record carries no spacing to derive an interval from, but range
        # reads must still address it — the aggregator reaches the lone bar
        # only through ``read_from``.
        assert reader.get_positions(reader.start_timestamp) == (0, 1)
        assert reader.get_size(reader.start_timestamp) == 1
        assert [c.timestamp for c in reader.read_from(reader.start_timestamp)] == [
            1_609_459_200_000
        ]
        # A window ending before the record still excludes it.
        assert reader.get_positions(0, 1_609_459_199_999) == (0, 0)
        assert list(reader.read_from(0, 1_609_459_199_999)) == []


def __test_legacy_reader_range_and_phantom_filtering__(tmp_path):
    file_path = tmp_path / "phantoms.ohlcv"
    records = [
        (1_609_459_200, 100.0, 110.0, 90.0, 105.0, 1_000.0),
        (1_609_459_260, 105.0, 115.0, 95.0, 108.0, 1_100.0),
        (1_609_459_320, 108.0, 108.0, 108.0, 108.0, -1.0),
        (1_609_459_380, 110.0, 120.0, 100.0, 115.0, 1_400.0),
    ]
    _write_legacy_records(file_path, records)

    with OHLCVReader(file_path) as reader:
        start = 1_609_459_260_000
        end = 1_609_459_380_000
        assert reader.get_positions(start, end) == (1, 4)
        assert reader.get_size(start, end) == 3
        filtered = list(reader.read_from(start, end))
        unfiltered = list(reader.read_from(start, end, skip_gaps=False))

    assert [candle.timestamp for candle in filtered] == [
        1_609_459_260_000,
        1_609_459_380_000,
    ]
    assert len(unfiltered) == 3
    assert unfiltered[1].volume == -1.0


def __test_legacy_reader_locates_irregular_range_bounds__(tmp_path):
    file_path = tmp_path / "irregular.ohlcv"
    # Spacing that shrinks, then one that grows: positions projected from the first
    # interval land on the wrong record in both directions.
    _write_legacy_records(
        file_path,
        [
            (1_000_000, 1.0, 2.0, 0.5, 1.5, 10.0),
            (1_001_000, 1.0, 2.0, 0.5, 1.5, 10.0),
            (1_001_100, 1.0, 2.0, 0.5, 1.5, 10.0),
            (1_009_000, 1.0, 2.0, 0.5, 1.5, 10.0),
        ],
    )

    with OHLCVReader(file_path) as reader:
        assert reader.interval == 1_000_000
        # Starting between two closely spaced records excludes the earlier one.
        assert reader.get_positions(1_001_050_000) == (2, 4)
        assert [c.timestamp for c in reader.read_from(1_001_050_000)] == [
            1_001_100_000,
            1_009_000_000,
        ]
        # A far-away last record stays inside a window that reaches it.
        assert reader.get_positions(1_001_100_000, 1_009_000_000) == (2, 4)
        # Both bounds are inclusive of exact matches.
        assert reader.get_positions(1_000_000_000, 1_001_000_000) == (0, 2)
        assert reader.get_positions(None, 1_005_000_000) == (0, 3)


def __test_legacy_reader_loads_position_aligned_extra_fields__(tmp_path):
    file_path = tmp_path / "extras.ohlcv"
    extra_path = tmp_path / "extras.extra.csv"
    _write_legacy_records(
        file_path,
        [
            (1_609_459_200, 100.0, 101.0, 99.0, 100.5, 10.0),
            (1_609_459_260, 101.0, 102.0, 100.0, 101.5, 11.0),
        ],
    )
    extra_path.write_text("signal,note\n1.5,buy\n2.5,sell\n")

    with OHLCVReader(file_path) as reader:
        candles = list(reader)

    assert candles[0].extra_fields == {"signal": 1.5, "note": "buy"}
    assert candles[1].extra_fields == {"signal": 2.5, "note": "sell"}


def __test_legacy_reader_exports_millisecond_timestamps__(tmp_path):
    file_path = tmp_path / "export.ohlcv"
    csv_path = tmp_path / "export.csv"
    json_path = tmp_path / "export.json"
    _write_legacy_records(file_path, [(1_609_459_200, 100.0, 101.0, 99.0, 100.5, 10.0)])

    with OHLCVReader(file_path) as reader:
        reader.save_to_csv(str(csv_path))
        reader.save_to_json(str(json_path), as_datetime=True)

    assert csv_path.read_text().splitlines()[1].startswith("1609459200000,")
    assert json.loads(json_path.read_text())[0]["time"] == "2021-01-01T00:00:00+00:00"


def __test_legacy_reader_rejects_text_disguised_as_ohlcv__(tmp_path):
    file_path = tmp_path / "fake_text.ohlcv"
    file_path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "1609459200,100,110,90,105,1000\n"
    )

    with pytest.raises(ValueError, match="Text file detected with .ohlcv extension"):
        with OHLCVReader(file_path):
            pass
