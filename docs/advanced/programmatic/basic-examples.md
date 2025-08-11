<!--
---
weight: 10003
title: "Basic Examples"
description: "Basic examples for PyneCore's programmatic API"
icon: "code"
date: "2025-08-11"
lastmod: "2025-08-11"
draft: false
toc: true
categories: ["Advanced", "API"]
tags: ["basic-examples", "api", "ohlcv", "syminfo", "scriptrunner"]
---
-->


# Basic Programmatic Examples

These examples mirror real-world usage patterns.

## 1) Run a Strategy Over Full Dataset

```python
from pathlib import Path
from pynecore.core.syminfo import SymInfo
from pynecore.core.ohlcv_file import OHLCVReader
from pynecore.core.script_runner import ScriptRunner

syminfo = SymInfo.load_toml(Path("./workdir/config/syminfo.toml"))
script_path = Path("./workdir/scripts/my_strategy.py")
data_path = Path("./workdir/data/BTCUSD_1D.ohlcv")

# Ensure output directory exists
output_dir = Path("./workdir/output")
output_dir.mkdir(parents=True, exist_ok=True)

with OHLCVReader(data_path) as reader:
    runner = ScriptRunner(script_path, reader, syminfo,
                          plot_path=output_dir / "plots.csv",
                          trade_path=output_dir / "trades.csv",
                          strat_path=output_dir / "strategy.csv",
                          last_bar_index=reader.size - 1)
    runner.run()
```

## 2) Run Only Over a Date Range

```python
from datetime import datetime, UTC

start = datetime(2024, 1, 1, tzinfo=UTC)
end = datetime(2024, 6, 30, tzinfo=UTC)

with OHLCVReader(data_path) as reader:
    ohlcv_subset = reader.read_from(int(start.timestamp()), int(end.timestamp()))
    runner = ScriptRunner(script_path, ohlcv_subset, syminfo)
    runner.run()
```

## 3) Progress Feedback

```python
def on_progress(current_dt):
    print("Processing:", current_dt)

with OHLCVReader(data_path) as reader:
    runner = ScriptRunner(script_path, reader, syminfo, last_bar_index=reader.size - 1)
    runner.run(on_progress=on_progress)
```

## 4) Saving and Inspecting Outputs

```python
from csv import DictReader

with OHLCVReader(data_path) as reader:
    runner = ScriptRunner(script_path, reader, syminfo,
                          plot_path=Path("plots.csv"),
                          trade_path=Path("trades.csv"),
                          strat_path=Path("strategy.csv"))
    runner.run()

# Read trade results
with open("trades.csv") as f:
    for row in DictReader(f):
        print(row)
```

## 5) Error Handling Template

```python
try:
    with OHLCVReader(data_path) as reader:
        runner = ScriptRunner(script_path, reader, syminfo)
        runner.run()
except FileNotFoundError:
    print("File not found. Verify paths under workdir/.")
except ImportError as e:
    print(f"Script import error. Ensure main() is decorated: {e}")
except ValueError as e:
    print(f"OHLCV data format error: {e}")
```