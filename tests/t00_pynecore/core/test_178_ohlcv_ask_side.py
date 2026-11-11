"""OHLCV v2 ask-side columns: layout, round trip, widening and aggregation."""

import math
import struct
from pathlib import Path

# noinspection PyProtectedMember
from pynecore.core.aggregator import _merge_candles
from pynecore.core.ohlcv import OHLCVReader, OHLCVWriter, copy_ohlcv_tail
from pynecore.types.ohlcv import OHLCV, AskOHLC

_HEADER = struct.Struct("<8sHHIIHHQqqIB3xII")
_DESCRIPTOR = struct.Struct("<BBBxH18s")
_T0 = 1_700_000_000_000


def __test_helper_descriptors(path: Path) -> list[tuple[int, int, int, int, str]]:
    data = path.read_bytes()
    column_count = _HEADER.unpack(data[:64])[5]
    descriptors = []
    for index in range(column_count):
        role, dtype, base, offset, name = _DESCRIPTOR.unpack_from(data, 64 + index * 24)
        descriptors.append((role, dtype, base, offset, name.rstrip(b"\0").decode()))
    return descriptors


def __test_helper_bar(index: int, ask: AskOHLC | None) -> OHLCV:
    return OHLCV(_T0 + index * 60_000, 1.10001, 1.10010, 1.09990, 1.10005, 10.0, ask=ask)


def __test_ask_columns_are_f32_deltas_from_the_bid_side__(tmp_path: Path):
    """The ask side adds four columns after volume, each a delta from its bid field."""
    path = tmp_path / "ask.ohlcv"
    ask = AskOHLC(1.10003, 1.10013, 1.09992, 1.10007)
    with OHLCVWriter(path, "1", minmove=1, pricescale=100_000) as writer:
        writer.write(__test_helper_bar(0, ask))
        assert writer.has_ask

    assert _HEADER.unpack(path.read_bytes()[:64])[4] == 52
    assert __test_helper_descriptors(path)[6:] == [
        (6, 5, 1, 36, "ask_open"),
        (7, 5, 2, 40, "ask_high"),
        (8, 5, 3, 44, "ask_low"),
        (9, 5, 4, 48, "ask_close"),
    ]
    with OHLCVReader(path) as reader:
        assert reader.has_ask
        assert reader.read(0).ask == ask


def __test_file_without_ask_stays_on_the_base_profile__(tmp_path: Path):
    """Bars that never quote the ask side keep the 36-byte record and no ask."""
    path = tmp_path / "plain.ohlcv"
    with OHLCVWriter(path, "1") as writer:
        writer.write(__test_helper_bar(0, None))
        assert not writer.has_ask

    assert _HEADER.unpack(path.read_bytes()[:64])[4] == 36
    with OHLCVReader(path) as reader:
        assert not reader.has_ask
        assert reader.read(0).ask is None


def __test_unquoted_bar_reads_back_an_all_nan_ask__(tmp_path: Path):
    """Once the file has ask columns, a bar without an ask stores NaN there."""
    path = tmp_path / "gap.ohlcv"
    with OHLCVWriter(path, "1", minmove=1, pricescale=100_000) as writer:
        writer.write(__test_helper_bar(0, AskOHLC(1.10003, 1.10013, 1.09992, 1.10007)))
        writer.write(__test_helper_bar(1, None))

    with OHLCVReader(path) as reader:
        ask = reader.read(1).ask
        assert ask is not None
        assert all(math.isnan(value) for value in ask)


def __test_first_ask_on_a_continued_file_widens_it_in_place__(tmp_path: Path):
    """A continued download that starts quoting the ask side keeps the stored bars."""
    path = tmp_path / "continued.ohlcv"
    with OHLCVWriter(path, "1", minmove=1, pricescale=100_000) as writer:
        writer.write(__test_helper_bar(0, None)._replace(extra_fields={"tag": "a"}))
        writer.write(__test_helper_bar(1, None)._replace(extra_fields={"tag": "b"}))

    ask = AskOHLC(1.10003, 1.10013, 1.09992, 1.10007)
    with OHLCVWriter(path, "1") as writer:
        writer.write(__test_helper_bar(2, ask)._replace(extra_fields={"tag": "c"}))
        assert writer.has_ask

    with OHLCVReader(path) as reader:
        bars = list(reader)
    assert [bar.timestamp for bar in bars] == [_T0, _T0 + 60_000, _T0 + 120_000]
    assert [bar.close for bar in bars] == [1.10005] * 3
    assert [bar.extra_fields for bar in bars] == [{"tag": "a"}, {"tag": "b"}, {"tag": "c"}]
    first_ask = bars[0].ask
    assert first_ask is not None and math.isnan(first_ask.close)
    assert bars[2].ask == ask


def __test_off_grid_ask_is_promoted_and_the_file_stays_appendable__(tmp_path: Path):
    """An ask price off the tick grid is stored as absolute f64 and read back exactly."""
    path = tmp_path / "promoted.ohlcv"
    off_grid = AskOHLC(1.105, 1.21, 1.01, 1.16)
    with OHLCVWriter(path, "1", minmove=1, pricescale=100) as writer:
        writer.write(OHLCV(_T0, 1.10, 1.20, 1.00, 1.15, 1.0, ask=off_grid))

    assert __test_helper_descriptors(path)[6] == (6, 6, 255, 36, "ask_open")
    assert [descriptor[1] for descriptor in __test_helper_descriptors(path)[7:]] == [5, 5, 5]

    on_grid = AskOHLC(1.16, 1.22, 1.12, 1.19)
    with OHLCVWriter(path, "1") as writer:
        writer.write(OHLCV(_T0 + 60_000, 1.15, 1.21, 1.11, 1.18, 1.0, ask=on_grid))

    with OHLCVReader(path) as reader:
        assert [bar.ask for bar in reader] == [off_grid, on_grid]


def __test_tail_copy_keeps_the_ask_side__(tmp_path: Path):
    """A byte copy of the record tail carries the ask columns with it."""
    source = tmp_path / "source.ohlcv"
    asks = [AskOHLC(open_, 1.10013, 1.09992, 1.10007) for open_ in (1.10003, 1.10004, 1.10006)]
    with OHLCVWriter(source, "1", minmove=1, pricescale=100_000) as writer:
        for index, ask in enumerate(asks):
            writer.write(__test_helper_bar(index, ask))

    target = tmp_path / "tail.ohlcv"
    assert copy_ohlcv_tail(source, target, 1) == 2
    with OHLCVReader(target) as reader:
        assert [bar.ask for bar in reader] == asks[1:]


def __test_exports_carry_the_ask_columns__(tmp_path: Path):
    """CSV and JSON exports name the ask columns of a file that stores them."""
    path = tmp_path / "export.ohlcv"
    with OHLCVWriter(path, "1", minmove=1, pricescale=100_000) as writer:
        writer.write(__test_helper_bar(0, AskOHLC(1.10003, 1.10013, 1.09992, 1.10007)))

    with OHLCVReader(path) as reader:
        reader.save_to_csv(tmp_path / "export.csv")
        reader.save_to_json(tmp_path / "export.json")
    header, row = (tmp_path / "export.csv").read_text().splitlines()
    assert header.endswith(",volume,ask_open,ask_high,ask_low,ask_close")
    assert row.endswith(",1.10003,1.10013,1.09992,1.10007")
    assert '"ask_close": 1.10007' in (tmp_path / "export.json").read_text()


def __test_aggregation_folds_the_ask_side_only_when_every_bar_quotes_it__():
    """The ask side folds like the bid side; a partly quoted window gets no ask."""
    nan = math.nan
    quoted = [
        OHLCV(0, 1.0, 1.3, 0.9, 1.1, 1.0, ask=AskOHLC(1.01, 1.31, 0.91, 1.11)),
        OHLCV(60_000, 1.1, 1.5, 1.0, 1.2, 1.0, ask=AskOHLC(1.11, 1.52, 1.01, 1.22)),
    ]
    merged = _merge_candles(quoted, 0)
    assert merged.ask == AskOHLC(1.01, 1.52, 0.91, 1.22)

    unquoted = [quoted[0], quoted[1]._replace(ask=AskOHLC(nan, nan, nan, nan))]
    assert _merge_candles(unquoted, 0).ask is None
    assert _merge_candles([quoted[0], quoted[1]._replace(ask=None)], 0).ask is None
