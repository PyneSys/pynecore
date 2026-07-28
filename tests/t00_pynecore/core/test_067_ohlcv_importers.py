import json
import os

import pytest

from pynecore.core.data_converter import ConversionError, DataConverter
from pynecore.core.ohlcv import OHLCVReader, OHLCVWriter, _LEGACY_RECORD_SIZE
from pynecore.core.ohlcv_importers import (
    infer_csv_period,
    infer_json_period,
    infer_txt_period,
    load_from_csv,
    load_from_json,
    load_from_txt,
)


def __test_csv_import_keeps_gaps_absent_and_aligns_extra_rows__(tmp_path):
    source_path = tmp_path / "prices.csv"
    output_path = tmp_path / "prices.ohlcv"
    extra_path = tmp_path / "prices.extra.csv"
    source_path.write_text(
        "timestamp,open,high,low,close,volume,rsi,signal\n"
        "2024-01-01T00:00:00Z,100,102,99,101,10,45.2,buy\n"
        "2024-01-01T00:01:00Z,101,103,100,102,11,52.1,hold\n"
        "2024-01-01T00:03:00Z,102,104,101,103,12,38.7,sell\n"
    )

    period = infer_csv_period(source_path, tz="UTC")
    assert period == "1"

    with OHLCVWriter(output_path, period, truncate=True) as writer:
        load_from_csv(writer, source_path, tz="UTC", extra_csv_path=extra_path)

    with OHLCVReader(output_path) as reader:
        candles = list(reader)
        assert reader.period == "1"
        assert reader.dense is False

    assert [candle.timestamp for candle in candles] == [
        1_704_067_200_000,
        1_704_067_260_000,
        1_704_067_380_000,
    ]
    assert [candle.extra_fields for candle in candles] == [
        {"rsi": 45.2, "signal": "buy"},
        {"rsi": 52.1, "signal": "hold"},
        {"rsi": 38.7, "signal": "sell"},
    ]
    assert extra_path.read_text().splitlines() == [
        "rsi,signal",
        "45.2,buy",
        "52.1,hold",
        "38.7,sell",
    ]


def __test_txt_import_preserves_quoted_values_and_skips_blank_lines__(tmp_path):
    source_path = tmp_path / "prices.txt"
    output_path = tmp_path / "prices.ohlcv"
    extra_path = tmp_path / "prices.extra.csv"
    source_path.write_text(
        "timestamp;open;high;low;close;volume;label\n"
        "1704067200;'100';'102';'99';'101';'10';'first'\n"
        "\n"
        "1704067260;101;103;100;102;11;second\n"
    )

    period = infer_txt_period(source_path, tz="UTC")
    assert period == "1"

    with OHLCVWriter(output_path, period, truncate=True) as writer:
        load_from_txt(writer, source_path, tz="UTC", extra_csv_path=extra_path)

    with OHLCVReader(output_path) as reader:
        candles = list(reader)

    assert [candle.timestamp for candle in candles] == [1_704_067_200_000, 1_704_067_260_000]
    assert [candle.close for candle in candles] == pytest.approx([101.0, 102.0])
    assert extra_path.read_text().splitlines() == ["label", "first", "second"]


def __test_json_import_supports_wrappers_and_field_mapping__(tmp_path):
    source_path = tmp_path / "prices.json"
    output_path = tmp_path / "prices.ohlcv"
    source_path.write_text(
        json.dumps(
            {
                "candles": [
                    {"t": 1_704_067_200_000, "o": 100, "h": 102, "l": 99, "c": 101, "v": 10},
                    {"t": 1_704_067_260_000, "o": 101, "h": 103, "l": 100, "c": 102, "v": 11},
                ]
            }
        )
    )
    mapping = {
        "timestamp": "t",
        "open": "o",
        "high": "h",
        "low": "l",
        "close": "c",
        "volume": "v",
    }

    period = infer_json_period(source_path, mapping=mapping, tz="UTC")
    assert period == "1"

    with OHLCVWriter(output_path, period, truncate=True) as writer:
        load_from_json(writer, source_path, mapping=mapping, tz="UTC")

    with OHLCVReader(output_path) as reader:
        candles = list(reader)

    assert [candle.timestamp for candle in candles] == [1_704_067_200_000, 1_704_067_260_000]
    assert [candle.volume for candle in candles] == pytest.approx([10.0, 11.0])


def __test_format_detection_does_not_treat_csv_as_legacy_binary__(tmp_path):
    source_path = tmp_path / "aligned.csv"
    content = (
        "timestamp,open,high,low,close,volume\n"
        "1704067200,100,102,99,101,10\n"
    )
    # Pad to a whole multiple of the legacy v1 record stride: that is exactly the
    # ambiguous size at which a text file could be mistaken for legacy binary.
    padding = (-len(content.encode())) % _LEGACY_RECORD_SIZE
    source_path.write_text(content + " " * padding)
    assert source_path.stat().st_size % _LEGACY_RECORD_SIZE == 0

    assert DataConverter.detect_format(source_path) == "csv"


def __test_data_converter_removes_output_after_invalid_ohlc_row__(tmp_path):
    source_path = tmp_path / "invalid.csv"
    source_path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "1704067200,100,102,99,101,10\n"
        "1704067260,101,100,99,102,11\n"
    )

    with pytest.raises(ConversionError, match=r"row 3: Invalid OHLC relation"):
        DataConverter().convert_to_ohlcv(
            source_path,
            force=True,
            symbol="TEST",
            provider="TEST",
        )

    assert not source_path.with_suffix(".ohlcv").exists()
    assert not source_path.with_suffix(".extra.csv").exists()


def __test_data_converter_restores_previous_pair_when_publication_fails__(tmp_path, monkeypatch):
    source_path = tmp_path / "publish.csv"
    ohlcv_path = source_path.with_suffix(".ohlcv")
    extra_path = source_path.with_suffix(".extra.csv")
    source_path.write_text(
        "timestamp,open,high,low,close,volume,sig\n"
        "1704067200,100,102,99,101,10,7\n"
        "1704067260,101,103,100,102,11,8\n"
    )
    converter = DataConverter()
    converter.convert_to_ohlcv(source_path, force=True, symbol="TEST", provider="TEST")
    with OHLCVReader(ohlcv_path) as reader:
        assert [candle.extra_fields for candle in reader] == [{"sig": 7.0}, {"sig": 8.0}]

    source_path.write_text(
        "timestamp,open,high,low,close,volume,sig\n"
        "1704067200,100,102,99,101,10,7\n"
        "1704067260,101,103,100,102,11,8\n"
        "1704067320,102,104,101,103,12,9\n"
    )
    real_replace = os.replace

    def failing_replace(src, dst, **kwargs):
        # Only the publication of the finished binary fails; the sidecar of the new
        # conversion has been moved into place by then.
        if str(src).endswith(".converting.ohlcv"):
            raise PermissionError("destination is held open")
        return real_replace(src, dst, **kwargs)

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(ConversionError):
        converter.convert_to_ohlcv(source_path, force=True, symbol="TEST", provider="TEST")
    monkeypatch.undo()

    # The previous conversion is intact: binary and sidecar still describe each other.
    with OHLCVReader(ohlcv_path) as reader:
        assert [candle.extra_fields for candle in reader] == [{"sig": 7.0}, {"sig": 8.0}]
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [source_path.name, extra_path.name, ohlcv_path.name, "publish.toml"]
    )


def __test_four_week_period_survives_a_missing_bar__(tmp_path):
    source_path = tmp_path / "fourweek.csv"
    rows = ["timestamp,open,high,low,close,volume"]
    # 2024-01-01 plus 28-day steps, with the fourth bar missing so the source
    # carries both a 28-day and a 56-day gap.
    for index in (0, 1, 2, 4, 5):
        rows.append(f"{1_704_067_200 + index * 28 * 86_400},100,102,99,101,10")
    source_path.write_text("\n".join(rows) + "\n")

    assert infer_csv_period(source_path, tz="UTC") == "4W"


def __test_calendar_month_period_survives_missing_months__(tmp_path):
    source_path = tmp_path / "monthly.csv"
    source_path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-31T00:00:00Z,100,102,99,101,10\n"
        "2024-02-29T00:00:00Z,100,102,99,101,10\n"
        "2024-04-30T00:00:00Z,100,102,99,101,10\n"
        "2024-05-31T00:00:00Z,100,102,99,101,10\n"
    )

    assert infer_csv_period(source_path, tz="UTC") == "1M"


def __test_data_converter_rejects_duplicate_timestamps_before_writing__(tmp_path):
    source_path = tmp_path / "duplicate.csv"
    source_path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "1704067200,100,102,99,101,10\n"
        "1704067200,101,103,100,102,11\n"
    )

    with pytest.raises(ConversionError, match=r"row 3: duplicate timestamp"):
        DataConverter().convert_to_ohlcv(
            source_path,
            force=True,
            symbol="TEST",
            provider="TEST",
        )

    assert not source_path.with_suffix(".ohlcv").exists()
