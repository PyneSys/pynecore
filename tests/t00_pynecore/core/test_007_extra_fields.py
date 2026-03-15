"""
@pyne
"""
import math
import tempfile
from pathlib import Path

from pynecore import Series
from pynecore.lib import script, bar_index, close, extra_fields, na
from pynecore.types.na import NA


@script.indicator(title="Extra Fields Test")
def main():
    rsi: Series[float] = extra_fields["rsi"]
    signal: Series[str] = extra_fields["signal"]

    return {
        "rsi": rsi[0],
        "prev_rsi": rsi[1],
        "signal": signal[0],
    }


def __test_extra_fields_basic__(csv_reader, runner):
    """ Extra fields are accessible as Series from CSV data """
    # CSVReader returns '' for empty cells (per-value type detection)
    expected_rsi = [45.2, 52.1, 38.7, '', 60.3]
    expected_signal = ['buy', '', 'sell', 'hold', 'buy']

    with csv_reader('extra_fields.csv', subdir="data") as cr:
        for i, (candle, _plot) in enumerate(runner(cr).run_iter()):
            if expected_rsi[i] == '':
                assert _plot["rsi"] == '', f"Bar {i}: expected empty string, got {_plot['rsi']!r}"
            else:
                assert math.isclose(_plot["rsi"], expected_rsi[i], rel_tol=1e-5), \
                    f"Bar {i}: rsi={_plot['rsi']}, expected={expected_rsi[i]}"

            assert _plot["signal"] == expected_signal[i], \
                f"Bar {i}: signal={_plot['signal']}, expected={expected_signal[i]}"


def __test_extra_fields_series_history__(script_path, module_key, syminfo, csv_reader):
    """ Extra fields Series supports [1] indexing for previous bar values """
    import sys
    from pynecore.core.script_runner import ScriptRunner

    sys.modules.pop(module_key, None)
    sys.modules.pop(script_path.stem, None)

    # CSVReader returns '' for empty cells, so bar 3 rsi='', and bar 4 prev_rsi=''
    expected_prev_rsi = [None, 45.2, 52.1, 38.7, '']

    with csv_reader('extra_fields.csv', subdir="data") as cr:
        r = ScriptRunner(script_path, cr, syminfo)
        for i, (candle, _plot) in enumerate(r.run_iter()):
            prev = _plot["prev_rsi"]
            expected = expected_prev_rsi[i]

            if expected is None:
                assert isinstance(prev, NA), f"Bar {i}: expected na, got {prev}"
            elif expected == '':
                assert prev == '', f"Bar {i}: expected empty string, got {prev!r}"
            else:
                assert math.isclose(prev, expected, rel_tol=1e-5), \
                    f"Bar {i}: prev_rsi={prev}, expected={expected}"


def __test_extra_fields_sidecar_generation__():
    """ DataConverter generates .extra.csv sidecar for CSV with extra columns """
    from pynecore.core.data_converter import DataConverter
    from pynecore.core.ohlcv_file import OHLCVReader

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test.csv"
        csv_path.write_text(
            "timestamp,open,high,low,close,volume,rsi,signal\n"
            "1514764800000,100,105,95,102,1000,45.2,buy\n"
            "1514779200000,102,108,100,106,1200,52.1,sell\n"
            "1514793600000,106,110,104,108,800,38.7,\n"
        )

        converter = DataConverter()
        converter.convert_to_ohlcv(csv_path, force=True, symbol='TEST', provider='TEST')

        extra_csv_path = csv_path.with_suffix('.extra.csv')
        assert extra_csv_path.exists(), "Sidecar .extra.csv was not generated"

        lines = extra_csv_path.read_text().strip().split('\n')
        assert lines[0] == 'rsi,signal', f"Unexpected header: {lines[0]}"
        assert len(lines) == 4, f"Expected 4 lines (header + 3 data), got {len(lines)}"

        with OHLCVReader(csv_path.with_suffix('.ohlcv')) as reader:
            for pos in range(reader.size):
                ohlcv = reader.read(pos)
                if ohlcv.volume >= 0:
                    assert ohlcv.extra_fields, f"Position {pos}: extra_fields is empty"
                    assert 'rsi' in ohlcv.extra_fields
                    assert 'signal' in ohlcv.extra_fields


def __test_extra_fields_no_sidecar_for_plain_ohlcv__():
    """ DataConverter does not generate .extra.csv when CSV has only OHLCV columns """
    from pynecore.core.data_converter import DataConverter

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "plain.csv"
        csv_path.write_text(
            "timestamp,open,high,low,close,volume\n"
            "1514764800000,100,105,95,102,1000\n"
            "1514779200000,102,108,100,106,1200\n"
        )

        converter = DataConverter()
        converter.convert_to_ohlcv(csv_path, force=True, symbol='TEST', provider='TEST')

        extra_csv_path = csv_path.with_suffix('.extra.csv')
        assert not extra_csv_path.exists(), "Sidecar should not be generated for plain OHLCV"
