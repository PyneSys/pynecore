"""
Parameter optimization command for PyneCore.
Runs a strategy with all combinations of specified parameter values (grid search)
and ranks results by a chosen metric. Supports parallel execution via --workers.
"""

import csv
import gc
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, fields as dataclass_fields
from datetime import datetime
from itertools import product
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

from typer import Option, Argument, secho, Exit
from rich.progress import (Progress, SpinnerColumn, TextColumn, BarColumn,
                           MofNCompleteColumn, ProgressColumn, Task)
from rich.table import Table
from rich.text import Text
from rich.console import Console

from ..app import app, app_state
from ...core.ohlcv import OHLCVReader
from ...core.syminfo import SymInfo
from ...core.script_runner import ScriptRunner
from ...core.strategy_stats import StrategyStatistics

__all__ = []

console = Console()

# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

METRIC_ALIASES: dict[str, str] = {
    "net_profit": "net_profit",
    "net_profit_pct": "net_profit_percent",
    "sharpe": "sharpe_ratio",
    "sortino": "sortino_ratio",
    "profit_factor": "profit_factor",
    "max_drawdown_pct": "max_equity_drawdown_percent",
    "win_rate": "percent_profitable",
    "total_trades": "total_trades",
    "avg_trade_pct": "avg_trade_percent",
    "avg_win_loss": "ratio_avg_win_loss",
}

# Metrics where lower is better
MINIMIZE_METRICS = {"max_drawdown_pct"}


# ---------------------------------------------------------------------------
# Parameter parsing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParamSpec:
    """Immutable specification for a single parameter's sweep values."""
    name: str
    values: tuple[Any, ...]


def parse_param_specs(params_dict: dict[str, Any]) -> list[ParamSpec]:
    """Parse parameter JSON into ParamSpec list.

    Supports:
    - Range dict: {"min": 5, "max": 20, "step": 1}
    - Explicit list: [true, false] or ["ema", "sma"]
    - Keys starting with "_" are skipped (metadata like _meta)
    """
    specs: list[ParamSpec] = []
    for name, spec in params_dict.items():
        if name.startswith("_"):
            continue
        if isinstance(spec, list):
            if len(spec) == 0:
                raise ValueError(f"Parameter '{name}': empty list")
            specs.append(ParamSpec(name=name, values=tuple(spec)))
        elif isinstance(spec, dict):
            required = {"min", "max", "step"}
            missing = required - set(spec.keys())
            if missing:
                raise ValueError(f"Parameter '{name}': missing keys {missing}")

            min_val, max_val, step = spec["min"], spec["max"], spec["step"]

            if step <= 0:
                raise ValueError(f"Parameter '{name}': step must be positive")
            if min_val > max_val:
                raise ValueError(f"Parameter '{name}': min ({min_val}) > max ({max_val})")

            # Integer range — use native range for precision
            if (isinstance(min_val, int) and isinstance(max_val, int)
                    and isinstance(step, int)):
                values = tuple(range(min_val, max_val + 1, step))
            else:
                # Float range — use index-based generation to avoid drift
                steps_count = int((max_val - min_val) / step) + 1
                values = []
                for i in range(steps_count):
                    val = round(min_val + i * step, 10)
                    if val > max_val + 1e-9:
                        break
                    values.append(val)
                values = tuple(values)

            if len(values) == 0:
                raise ValueError(f"Parameter '{name}': range produces no values")
            specs.append(ParamSpec(name=name, values=values))
        else:
            raise ValueError(
                f"Parameter '{name}': must be a list or dict with min/max/step, "
                f"got {type(spec).__name__}"
            )
    return specs


def generate_combinations(specs: list[ParamSpec]) -> list[dict[str, Any]]:
    """Generate all parameter combinations (cartesian product)."""
    if not specs:
        return [{}]
    names = [s.name for s in specs]
    value_lists = [s.values for s in specs]
    return [dict(zip(names, combo)) for combo in product(*value_lists)]


def generate_explicit_combinations(specs: list[ParamSpec]) -> list[dict[str, Any]]:
    """Generate combinations by column-wise zip (explicit mode).

    All multi-value param lists must have the same length N.
    Single-value params (fixed) expand to that value for all N combos.
    Combo i uses index i from each multi-value list.

    Raises ValueError if multi-value lists have inconsistent lengths.
    """
    if not specs:
        return [{}]

    # Separate fixed (single-value) from variable (multi-value) params
    fixed_params: list[tuple[str, Any]] = []
    variable_params: list[tuple[str, tuple[Any, ...]]] = []

    for spec in specs:
        if len(spec.values) == 1:
            fixed_params.append((spec.name, spec.values[0]))
        else:
            variable_params.append((spec.name, spec.values))

    # If no variable params, return single combo with all fixed values
    if not variable_params:
        return [{name: value for name, value in fixed_params}]

    # Validate: all variable param lists must have same length
    combo_count = len(variable_params[0][1])
    for name, values in variable_params[1:]:
        if len(values) != combo_count:
            raise ValueError(
                f"Explicit mode: param '{name}' has {len(values)} values, "
                f"but '{variable_params[0][0]}' has {combo_count}. "
                f"All multi-value param lists must have the same length."
            )

    # Build N combos by zipping columns
    combinations = []
    for i in range(combo_count):
        combo = {name: value for name, value in fixed_params}
        for name, values in variable_params:
            combo[name] = values[i]
        combinations.append(combo)

    return combinations


def parse_chunk(chunk_str: str) -> tuple[int, int]:
    """Parse 'N/M' -> (chunk_num, total_chunks). Validates 1 <= N <= M."""
    parts = chunk_str.split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid chunk format '{chunk_str}', expected 'N/M' (e.g. '2/4')")
    try:
        n, m = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid chunk format '{chunk_str}', N and M must be integers")
    if m < 1:
        raise ValueError(f"Total chunks M must be >= 1, got {m}")
    if n < 1 or n > m:
        raise ValueError(f"Chunk number N must be 1..{m}, got {n}")
    return n, m


# ---------------------------------------------------------------------------
# Single-run helper (used by both sequential and parallel paths)
# ---------------------------------------------------------------------------

def _run_backtest(
    script_path: Path,
    reader: OHLCVReader,
    syminfo: SymInfo,
    start_ts: int,
    end_ts: int,
    size: int,
    params: dict[str, Any],
) -> tuple[StrategyStatistics | None, str | None]:
    """Run one backtest with the given parameter overrides.

    Returns (stats, None) on success or (None, error_message) on failure.
    """
    from pynecore.core import script as script_module

    # 1. Clean input state for this combo, then hand the values to the runner via
    #    the sanctioned inputs= API. Upstream loads inputs= into
    #    _programmatic_inputs, which the script import copies into
    #    _old_input_values (script.py). This replaces the fork's manual
    #    _old_input_values mutation + PYNE_OPTIMIZE_MODE + _var_cache machinery,
    #    and is per-run rather than shared mutable module state.
    script_module._old_input_values.clear()
    script_module._programmatic_inputs.clear()

    # 2. Force fresh module import so each combo runs isolated
    module_name = script_path.stem
    if module_name in sys.modules:
        del sys.modules[module_name]

    # 3. Clear registered libraries from previous run
    script_module._registered_libraries.clear()

    # 4. Fresh OHLCV iterator
    ohlcv_iter = reader.read_from(start_ts, end_ts)

    # 5. Run (no file I/O)
    try:
        runner = ScriptRunner(
            script_path, ohlcv_iter, syminfo,
            last_bar_index=size - 1,
            plot_path=None,
            strat_path=None,
            trade_path=None,
            inputs=params,
        )
        runner.run()
        return (runner.stats, None)
    except Exception as e:
        return (None, f"Run failed for {params}: {e}")


# ---------------------------------------------------------------------------
# Parallel worker functions
# ---------------------------------------------------------------------------

# Per-worker globals (set by _worker_init, used by _worker_run)
_w_script_path: Path | None = None
_w_reader: OHLCVReader | None = None
_w_syminfo: SymInfo | None = None
_w_start_ts: int = 0
_w_end_ts: int = 0
_w_size: int = 0


def _worker_init(
    script_path_str: str,
    data_path_str: str,
    syminfo_toml_str: str,
    start_ts: int,
    end_ts: int,
    size: int,
    lib_dir_str: str | None,
    cache_path: str | None = None,
) -> None:
    """Initialize per-worker state. Called once per worker process."""
    global _w_script_path, _w_reader, _w_syminfo, _w_start_ts, _w_end_ts, _w_size

    _w_script_path = Path(script_path_str)
    _w_reader = OHLCVReader(Path(data_path_str))
    _w_reader.__enter__()
    _w_syminfo = SymInfo.load_toml(Path(syminfo_toml_str))
    _w_start_ts = start_ts
    _w_end_ts = end_ts
    _w_size = size

    os.environ["PYNE_OPTIMIZE_MODE"] = "1"
    os.environ["PYNE_SAVE_SCRIPT_TOML"] = "0"

    if lib_dir_str:
        sys.path.insert(0, lib_dir_str)

    if cache_path:
        import pickle
        from pynecore.core import _var_cache
        with open(cache_path, 'rb') as f:
            _var_cache._data = pickle.load(f)


def _worker_run(
    params: dict[str, Any],
) -> tuple[dict[str, Any], StrategyStatistics | None, str | None]:
    """Run a single backtest in a worker process.

    Returns (params, stats, error_message).
    """
    stats, error = _run_backtest(
        _w_script_path, _w_reader, _w_syminfo,
        _w_start_ts, _w_end_ts, _w_size, params,
    )
    return (params, stats, error)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def get_metric_value(stats: StrategyStatistics, metric_attr: str) -> float:
    """Extract a metric value from StrategyStatistics."""
    return float(getattr(stats, metric_attr, 0.0))


def write_results_csv(
    results: list[tuple[dict[str, Any], StrategyStatistics]],
    param_names: list[str],
    output_path: Path,
) -> None:
    """Write all optimization results to CSV."""
    if not results:
        return

    stat_fields = [f.name for f in dataclass_fields(StrategyStatistics)]
    header = param_names + stat_fields

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for params, stats in results:
            row = [params.get(name, "") for name in param_names]
            row += [getattr(stats, field) for field in stat_fields]
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Incremental CSV + Resume helpers
# ---------------------------------------------------------------------------

def _load_completed_keys(
    csv_path: Path,
    param_names: list[str],
) -> tuple[set[tuple[str, ...]], list[list[str]], list[str]]:
    """Load completed parameter combos from a partial CSV.

    Returns (completed_keys, existing_rows, fieldnames).
    completed_keys: set of param-value tuples already done.
    existing_rows: raw CSV rows (list of values) for later merge.
    fieldnames: header from the CSV file.
    """
    completed: set[tuple[str, ...]] = set()
    existing_rows: list[list[str]] = []
    fieldnames: list[str] = []

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return completed, existing_rows, fieldnames

    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            fieldnames = next(reader, [])
            if not fieldnames:
                return completed, existing_rows, fieldnames
            # Find param column indices
            param_indices = []
            for name in param_names:
                try:
                    param_indices.append(fieldnames.index(name))
                except ValueError:
                    # Header doesn't match — can't resume
                    return set(), [], []
            for row in reader:
                if len(row) < len(fieldnames):
                    continue
                key = tuple(row[i] for i in param_indices)
                completed.add(key)
                existing_rows.append(row)
    except Exception:
        return set(), [], []

    return completed, existing_rows, fieldnames


def _combo_to_key(
    combo: dict[str, Any],
    param_names: list[str],
) -> tuple[str, ...]:
    """Convert a parameter combo to a hashable string tuple for comparison."""
    return tuple(str(combo.get(name, "")) for name in param_names)


def _run_signature(
    script: Path,
    data: Path,
    params_dict: dict[str, Any],
    time_from: datetime | None,
    time_to: datetime | None,
    chunk: str | None,
) -> str:
    """Compute a signature that uniquely identifies an optimization invocation.

    Two invocations share a result CSV only when this signature matches. It
    covers every input that changes the computed numbers — script, data file,
    parameter spec, and date window — so results from an unrelated run are never
    silently reused (see resume logic in ``optimize``). The metric is
    deliberately excluded: it only affects sort order, not the values.

    :param script: Resolved strategy script path.
    :param data: Resolved OHLCV data file path.
    :param params_dict: Full parameter specification (as parsed from JSON).
    :param time_from: Start of the backtest window, if any.
    :param time_to: End of the backtest window, if any.
    :param chunk: Grid chunk selector (e.g. ``"2/4"``), if any.
    :return: Hex signature string.
    """
    payload = json.dumps(
        {
            "script": str(script.resolve()),
            "data": str(data.resolve()),
            "params": params_dict,
            "from": time_from.isoformat() if time_from else "",
            "to": time_to.isoformat() if time_to else "",
            "chunk": chunk or "",
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sort_and_rewrite_csv(
    csv_path: Path,
    metric_attr: str,
    is_minimize: bool,
) -> int:
    """Read CSV, sort by metric column, rewrite in-place. Returns row count."""
    rows: list[list[str]] = []
    header: list[str] = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        if not header:
            return 0
        for row in reader:
            rows.append(row)

    if not rows:
        return 0

    # Find metric column index
    try:
        metric_idx = header.index(metric_attr)
    except ValueError:
        return len(rows)  # Can't sort, leave as-is

    def sort_key(row):
        try:
            return float(row[metric_idx])
        except (ValueError, TypeError, IndexError):
            return 0.0

    rows.sort(key=sort_key, reverse=not is_minimize)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    return len(rows)


def write_best_toml(best_params: dict[str, Any], output_path: Path) -> None:
    """Write the best parameter set as a TOML snippet."""
    lines = [
        "# Best optimization parameters",
        "# Generated by: pyne optimize",
        "",
        "[script]",
        "",
        "# Input Settings",
        "",
    ]
    for name, value in best_params.items():
        lines.append(f"[inputs.{name}]")
        if isinstance(value, bool):
            lines.append(f"value = {str(value).lower()}")
        elif isinstance(value, str):
            lines.append(f'value = "{value}"')
        else:
            lines.append(f"value = {value}")
        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Progress column
# ---------------------------------------------------------------------------

class RateColumn(ProgressColumn):
    """Show runs per second."""

    def render(self, task: Task) -> Text:
        elapsed = task.elapsed
        if elapsed is None or elapsed == 0 or task.completed == 0:
            return Text("-- runs/s", style="cyan")
        rate = task.completed / elapsed
        return Text(f"{rate:.1f} runs/s", style="cyan")


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

@app.command()
def optimize(
    script: Path = Argument(
        ..., dir_okay=False, file_okay=True,
        help="Strategy script to optimize (.py)",
    ),
    data: Path = Argument(
        ..., dir_okay=False, file_okay=True,
        help="Data file to use (*.ohlcv)",
    ),
    params: Path = Argument(
        ..., dir_okay=False, file_okay=True,
        help="Parameter specification JSON file",
    ),
    metric: str = Option(
        "net_profit", "--metric", "-m",
        help="Metric to optimize. Options: "
             + ", ".join(METRIC_ALIASES.keys()),
    ),
    top_n: int = Option(10, "--top", "-n", help="Number of top results to display"),
    output: Path | None = Option(
        None, "--output", "-o",
        help="CSV output path for all results",
    ),
    save_best: bool = Option(False, "--save-best", help="Save best params as .toml"),
    workers: int = Option(
        0, "--workers", "-w",
        help="Parallel workers (0=auto/half cores, 1=sequential)",
    ),
    time_from: datetime | None = Option(
        None, "--from", "-f",
        formats=["%Y-%m-%d", "%Y-%m-%d %H:%M:%S"],
        help="Start date (UTC)",
    ),
    time_to: datetime | None = Option(
        None, "--to", "-t",
        formats=["%Y-%m-%d", "%Y-%m-%d %H:%M:%S"],
        help="End date (UTC)",
    ),
    chunk: str | None = Option(
        None, "--chunk", "-c",
        help="Run chunk N/M of the grid, e.g. '2/4' runs the 2nd quarter",
    ),
):
    """
    Optimize strategy parameters via grid search.

    Runs the strategy with every combination of parameters defined in the JSON
    file and ranks results by the chosen metric.

    \b
    Example JSON (optimize.json):
    {
        "fast_length": {"min": 5, "max": 20, "step": 1},
        "slow_length": {"min": 20, "max": 50, "step": 5},
        "use_filter": [true, false]
    }

    \b
    Parallel execution (use all CPU cores):
        pyne optimize script.py data.ohlcv params.json -w 0
    Sequential execution:
        pyne optimize script.py data.ohlcv params.json -w 1
    """

    # --- Resolve script path ---
    if script.suffix != ".py":
        script = script.with_suffix(".py")
    if len(script.parts) == 1:
        script = app_state.scripts_dir / script
    if not script.exists():
        secho(f"Script file '{script}' not found!", fg="red", err=True)
        raise Exit(1)

    # --- Resolve data path ---
    if data.suffix == "":
        data = data.with_suffix(".ohlcv")
    if len(data.parts) == 1:
        data = app_state.data_dir / data
    if not data.exists():
        secho(f"Data file '{data}' not found!", fg="red", err=True)
        raise Exit(1)

    # --- Resolve params path ---
    if len(params.parts) == 1:
        # Try scripts dir first, then workdir
        candidate = app_state.scripts_dir / params
        if candidate.exists():
            params = candidate
        else:
            candidate = app_state.workdir / params
            if candidate.exists():
                params = candidate
    if not params.exists():
        secho(f"Parameter file '{params}' not found!", fg="red", err=True)
        raise Exit(1)

    # --- Validate metric ---
    if metric not in METRIC_ALIASES:
        secho(
            f"Unknown metric '{metric}'. Available: {', '.join(METRIC_ALIASES.keys())}",
            fg="red", err=True,
        )
        raise Exit(1)
    metric_attr = METRIC_ALIASES[metric]

    # --- Load parameter specs ---
    try:
        with open(params, "r") as f:
            params_dict = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        secho(f"Failed to read parameter file: {e}", fg="red", err=True)
        raise Exit(1)

    try:
        specs = parse_param_specs(params_dict)
    except ValueError as e:
        secho(f"Invalid parameter specification: {e}", fg="red", err=True)
        raise Exit(1)

    is_explicit_mode = params_dict.get("_mode") == "explicit"
    if is_explicit_mode:
        combinations = generate_explicit_combinations(specs)
    else:
        combinations = generate_combinations(specs)
    chunk_num, total_chunks = 0, 0
    if chunk:
        chunk_num, total_chunks = parse_chunk(chunk)
        combinations = combinations[chunk_num - 1::total_chunks]
    param_names = [s.name for s in specs]
    total_combos = len(combinations)

    # --- Resolve worker count ---
    num_workers = workers if workers > 0 else max((cpu_count() or 1) // 2, 1)
    num_workers = min(num_workers, total_combos)

    # --- Load symbol info ---
    syminfo_toml = data.with_suffix(".toml")
    try:
        syminfo = SymInfo.load_toml(syminfo_toml)
    except FileNotFoundError:
        secho(f"Symbol info file '{syminfo_toml}' not found!",
              fg="red", err=True)
        raise Exit(1)

    # --- Pre-compute data parameters ---
    with OHLCVReader(data) as reader:
        # Timestamps are int64 milliseconds since the v2 OHLCV format.
        start_ts = (reader.start_timestamp if not time_from
                    else int(time_from.replace(tzinfo=None).timestamp() * 1000))
        end_ts = (reader.end_timestamp if not time_to
                  else int(time_to.replace(tzinfo=None).timestamp() * 1000))
        size = reader.get_size(start_ts, end_ts)

    # --- Print summary ---
    secho(f"\nOptimizing: {script.name}", fg="cyan")
    secho(f"Data:       {data.name}", fg="cyan")
    secho(f"Metric:     {metric} ({metric_attr})", fg="cyan")
    secho(f"Workers:    {num_workers}", fg="cyan")
    secho(f"Parameters: {len(specs)}", fg="cyan")
    for spec in specs:
        secho(f"  {spec.name}: {len(spec.values)} values "
              f"({spec.values[0]} .. {spec.values[-1]})", fg="cyan")
    if chunk:
        secho(f"Chunk:      {chunk_num}/{total_chunks} ({total_combos} combinations)", fg="cyan")
    secho(f"Total combinations: {total_combos}\n", fg="cyan")

    # --- Lib directory ---
    lib_dir = app_state.scripts_dir / "lib"
    lib_dir_str = str(lib_dir) if lib_dir.exists() and lib_dir.is_dir() else None

    # --- Set optimize-mode environment (for sequential path) ---
    saved_env: dict[str, str | None] = {}
    for key in ("PYNE_OPTIMIZE_MODE", "PYNE_SAVE_SCRIPT_TOML"):
        saved_env[key] = os.environ.get(key)

    # --- Resolve CSV path early (needed for resume check) ---
    if chunk and not output:
        csv_path = app_state.output_dir / f"{script.stem}_optimize_chunk{chunk_num}of{total_chunks}.csv"
    else:
        csv_path = output or (app_state.output_dir / f"{script.stem}_optimize.csv")

    # --- Resume support: check for partial results ---
    # Only resume from a CSV that belongs to THIS exact invocation. The run
    # signature covers script, data, params and date window, so results from an
    # unrelated run against the same script name are never silently reused.
    stat_fields = [f.name for f in dataclass_fields(StrategyStatistics)]
    run_sig = _run_signature(script, data, params_dict, time_from, time_to, chunk)
    sig_path = csv_path.with_name(csv_path.name + ".runmeta")
    can_resume = sig_path.exists() and sig_path.read_text().strip() == run_sig

    if can_resume:
        completed_keys, existing_rows, _ = _load_completed_keys(
            csv_path, param_names,
        )
    else:
        if csv_path.exists() and csv_path.stat().st_size > 0:
            console.print(
                "[yellow]Existing results CSV is from a different run "
                "(script/data/params/date window changed) — starting fresh.[/yellow]"
            )
        completed_keys, existing_rows = set(), []
    resumed_count = len(completed_keys)

    if resumed_count > 0:
        combinations = [
            c for c in combinations
            if _combo_to_key(c, param_names) not in completed_keys
        ]
        console.print(
            f"[green]Resuming: {resumed_count} already completed, "
            f"{len(combinations)} remaining of {total_combos} total[/green]\n"
        )

    remaining_count = len(combinations)
    if remaining_count == 0:
        console.print("[green]All combinations already completed![/green]")
        # Re-sort and display existing results
        _sort_and_rewrite_csv(csv_path, metric_attr, metric in MINIMIZE_METRICS)
        console.print(f"[green]Results at: {csv_path}[/green]")
        return

    # Adjust worker count for remaining work
    num_workers = min(num_workers, remaining_count)

    # --- Open CSV for incremental writing ---
    csv_header = param_names + stat_fields
    write_header = resumed_count == 0
    csv_file = open(
        csv_path,
        "a" if resumed_count > 0 else "w",
        newline="",
    )
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow(csv_header)
        csv_file.flush()
        # Stamp this fresh CSV with its run signature so a later invocation can
        # tell whether the file belongs to it before resuming from it.
        sig_path.write_text(run_sig)

    # --- Run optimization ---
    results: list[tuple[dict[str, Any], StrategyStatistics]] = []
    failed_count = 0

    try:
        with Progress(
            SpinnerColumn(finished_text="[green]OK"),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[cyan]{task.percentage:>3.0f}%"),
            RateColumn(),
        ) as progress:
            desc = (
                f"Optimizing ({resumed_count} resumed)..."
                if resumed_count > 0
                else "Optimizing..."
            )
            task = progress.add_task(desc, total=remaining_count)

            if num_workers > 1:
                # --- Parallel execution ---
                # Each worker runs each combo independently through
                # _run_backtest's ScriptRunner(inputs=...) path. The fork's
                # cross-combo variable cache (fork-only `_var_cache` module,
                # probed once and pickled to workers) is intentionally dropped:
                # it required fork internals absent upstream, and mutating that
                # shared cache across pooled workers was the source of the
                # recorded `_vc` cross-worker contamination bug. Workers now
                # recompute series per combo — slower, but correct and isolated.
                os.environ["PYNE_OPTIMIZE_MODE"] = "1"
                os.environ["PYNE_SAVE_SCRIPT_TOML"] = "0"
                if lib_dir_str:
                    sys.path.insert(0, lib_dir_str)

                cache_path = None
                worker_combos = combinations

                with ProcessPoolExecutor(
                    max_workers=num_workers,
                    initializer=_worker_init,
                    initargs=(
                        str(script), str(data), str(syminfo_toml),
                        start_ts, end_ts, size, lib_dir_str,
                        cache_path,
                    ),
                ) as pool:
                    futures = [
                        pool.submit(_worker_run, combo)
                        for combo in worker_combos
                    ]
                    for future in as_completed(futures):
                        combo_params, stats, error = future.result()
                        if stats is not None:
                            results.append((combo_params, stats))
                            # Incremental CSV write
                            row = [combo_params.get(n, "") for n in param_names]
                            row += [getattr(stats, f) for f in stat_fields]
                            csv_writer.writerow(row)
                            csv_file.flush()
                        else:
                            failed_count += 1
                            if error:
                                console.print(
                                    f"[yellow]Warning: {error}[/yellow]"
                                )
                        progress.update(task, advance=1)

                        if len(results) % 50 == 0:
                            gc.collect()

                # Clean up cache temp file
                if cache_path:
                    try:
                        os.unlink(cache_path)
                    except OSError:
                        pass

            else:
                # --- Sequential execution ---
                os.environ["PYNE_OPTIMIZE_MODE"] = "1"
                os.environ["PYNE_SAVE_SCRIPT_TOML"] = "0"

                if lib_dir_str:
                    sys.path.insert(0, lib_dir_str)

                with OHLCVReader(data) as reader:
                    for combo in combinations:
                        stats, error = _run_backtest(
                            script_path=script,
                            reader=reader,
                            syminfo=syminfo,
                            start_ts=start_ts,
                            end_ts=end_ts,
                            size=size,
                            params=combo,
                        )

                        if stats is not None:
                            results.append((combo, stats))
                            # Incremental CSV write
                            row = [combo.get(n, "") for n in param_names]
                            row += [getattr(stats, f) for f in stat_fields]
                            csv_writer.writerow(row)
                            csv_file.flush()
                        else:
                            failed_count += 1
                            if error:
                                console.print(
                                    f"[yellow]Warning: {error}[/yellow]"
                                )

                        progress.update(task, advance=1)

                        if len(results) % 50 == 0:
                            gc.collect()

    finally:
        csv_file.close()

        # Restore environment
        for key, val in saved_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

        if lib_dir_str and lib_dir_str in sys.path:
            sys.path.remove(lib_dir_str)

    # --- Check results ---
    total_successful = len(results) + resumed_count
    if total_successful == 0:
        secho("\nNo successful runs. Cannot produce results.", fg="red", err=True)
        raise Exit(1)

    # --- Sort the full CSV (resumed + new results) ---
    is_minimize = metric in MINIMIZE_METRICS
    _sort_and_rewrite_csv(csv_path, metric_attr, is_minimize)

    # --- Sort in-memory results for display ---
    results.sort(
        key=lambda r: get_metric_value(r[1], metric_attr),
        reverse=not is_minimize,
    )

    # --- Display top N table ---
    display_count = min(top_n, len(results))
    table = Table(
        title=f"\nTop {display_count} Results (by {metric})",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("#", style="dim", width=4)
    for name in param_names:
        table.add_column(name, style="cyan")
    table.add_column(metric, justify="right", style="green bold")
    table.add_column("Net P %", justify="right", style="green")
    table.add_column("Sharpe", justify="right")
    table.add_column("PF", justify="right")
    table.add_column("Win %", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Max DD %", justify="right", style="red")

    for rank, (combo, stats) in enumerate(results[:display_count], 1):
        row = [str(rank)]
        row += [str(combo.get(name, "")) for name in param_names]
        row.append(f"{get_metric_value(stats, metric_attr):.4f}")
        row.append(f"{stats.net_profit_percent:.2f}")
        row.append(f"{stats.sharpe_ratio:.3f}")
        row.append(f"{stats.profit_factor:.3f}")
        row.append(f"{stats.percent_profitable:.1f}")
        row.append(str(stats.total_trades))
        row.append(f"{stats.max_equity_drawdown_percent:.2f}")
        table.add_row(*row)

    console.print(table)

    if failed_count > 0:
        console.print(f"\n[yellow]{failed_count} combination(s) failed.[/yellow]")

    console.print(
        f"\n[dim]Total: {total_successful} successful / {total_combos} combinations"
        f"{f' ({resumed_count} resumed)' if resumed_count > 0 else ''}[/dim]"
    )

    console.print(f"[green]Results saved to: {csv_path}[/green]")

    # --- Write best TOML ---
    if save_best and results:
        best_params = results[0][0]
        toml_path = script.with_name(script.stem + ".optimized.toml")
        write_best_toml(best_params, toml_path)
        console.print(f"[green]Best parameters saved to: {toml_path}[/green]")
