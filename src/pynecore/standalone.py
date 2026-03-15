"""Standalone runner for PyneComp-compiled @pyne scripts."""
import sys
from pathlib import Path


def run(script_file: str) -> None:
    """
    Run a compiled @pyne script in standalone mode.

    Enables ``python script.py data.csv`` without a workdir or the ``pyne`` CLI.

    :param script_file: The ``__file__`` of the calling script
    """
    if len(sys.argv) < 2:
        script_name = Path(script_file).name
        print(f"Usage: python {script_name} <data_file>", file=sys.stderr)
        print(f"\n  data_file: Path to CSV or OHLCV data file", file=sys.stderr)
        sys.exit(1)

    data_arg = sys.argv[1]
    data_path = Path(data_arg).resolve()
    script_path = Path(script_file).resolve()

    if not data_path.exists():
        print(f"Error: Data file '{data_arg}' not found", file=sys.stderr)
        sys.exit(1)

    import shutil
    import tempfile
    from pynecore.core.data_converter import DataConverter, DataFormatError, ConversionError
    from pynecore.core.ohlcv_file import OHLCVReader
    from pynecore.core.syminfo import SymInfo
    from pynecore.core.script_runner import ScriptRunner

    temp_dir = None
    ohlcv_path = data_path

    try:
        # CSV/TXT/JSON → OHLCV conversion in temp directory
        if data_path.suffix != '.ohlcv':
            try:
                temp_dir = tempfile.mkdtemp(prefix="pyne_")
                # Copy preserves filename for guess_symbol_from_filename heuristics
                temp_copy = Path(temp_dir) / data_path.name
                shutil.copy2(data_path, temp_copy)

                converter = DataConverter()
                detected_symbol, detected_provider = DataConverter.guess_symbol_from_filename(
                    data_path
                )
                if not detected_symbol:
                    detected_symbol = data_path.stem.upper()

                print(f"Converting {data_path.name}...", file=sys.stderr)
                converter.convert_to_ohlcv(
                    temp_copy, provider=detected_provider,
                    symbol=detected_symbol, force=True
                )
                ohlcv_path = temp_copy.with_suffix('.ohlcv')
            except (DataFormatError, ConversionError) as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)

        # Load symbol info
        toml_path = ohlcv_path.with_suffix('.toml')
        try:
            syminfo = SymInfo.load_toml(toml_path)
        except FileNotFoundError:
            print(f"Error: Symbol info '{toml_path.name}' not found", file=sys.stderr)
            sys.exit(1)

        # Output paths next to script
        out_dir = script_path.parent
        plot_path = out_dir / f"{script_path.stem}.csv"
        trade_path = out_dir / f"{script_path.stem}_trades.csv"
        strat_path = out_dir / f"{script_path.stem}_strat.csv"

        # Run using the standard ScriptRunner
        with OHLCVReader(ohlcv_path) as reader:
            size = reader.get_size(reader.start_timestamp, reader.end_timestamp)
            ohlcv_iter = reader.read_from(reader.start_timestamp, reader.end_timestamp)
            print(
                f"Running {script_path.name} on {data_path.stem} ({size} bars)...",
                file=sys.stderr
            )

            runner = ScriptRunner(
                script_path, ohlcv_iter, syminfo, last_bar_index=size - 1,
                plot_path=plot_path, strat_path=strat_path, trade_path=trade_path
            )
            runner.run()

        print(f"Done. Output: {plot_path}", file=sys.stderr)

    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
