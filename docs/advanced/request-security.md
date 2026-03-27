<!--
---
weight: 1006
title: "request.security()"
description: "Using request.security() for multi-symbol and multi-timeframe data in PyneCore"
icon: "security"
date: "2026-03-27"
lastmod: "2026-03-27"
draft: false
toc: true
categories: ["Advanced", "Technical Implementation"]
tags: ["request-security", "multiprocessing", "shared-memory", "ipc", "ast"]
---
-->

# request.security()

`request.security()` lets you access data from other symbols or timeframes within your script.
PyneCore runs each security context as a separate OS process with its own Series history,
enabling true multi-symbol and multi-timeframe analysis.

## Quick Start

### 1. Prepare OHLCV Data

Each security context needs its own OHLCV data file. Convert your data sources to PyneCore's
binary format using the CLI:

```bash
# Chart data (5-minute EURUSD)
pyne data convert csv EURUSD_5m.csv

# Security data (daily EURUSD for HTF analysis)
pyne data convert csv EURUSD_1D.csv

# Or aggregate from existing data
pyne data aggregate EURUSD_5m -tf 1D
```

This creates `.ohlcv` + `.toml` file pairs in `workdir/data/`.

### 2. Write Your Script

Use `request.security()` like you would in Pine Script:

```python
"""@pyne"""

from pynecore.lib import *
from pynecore.types import *


@script.indicator("Multi-Timeframe SMA")
def main():
    # Fetch daily SMA(20) while running on 5-minute chart
    daily_sma: Series[float] = lib.request.security(
        lib.syminfo.tickerid, "1D", lib.ta.sma(lib.close, 20)
    )

    lib.plot.plot(daily_sma, "Daily SMA", color=lib.color.blue)
    lib.plot.plot(lib.close, "Close")
```

### 3. Run with Security Data

#### Programmatic API (ScriptRunner)

Pass OHLCV paths via the `security_data` parameter:

```python
from pathlib import Path
from pynecore.core.script_runner import ScriptRunner
from pynecore.core.ohlcv_file import OHLCVReader
from pynecore.core.syminfo import SymInfo

syminfo = SymInfo.load_toml("workdir/data/EURUSD_5m.toml")
reader = OHLCVReader("workdir/data/EURUSD_5m")
reader.open()

runner = ScriptRunner(
    script_path=Path("workdir/scripts/multi_tf_sma.py"),
    ohlcv_iter=reader,
    syminfo=syminfo,
    security_data={
        "1D": "workdir/data/EURUSD_1D",  # timeframe-only key
    },
)

for candle, plot_data in runner.run_iter():
    daily_sma = plot_data.get("Daily SMA")
    print(f"Bar {candle.timestamp}: Daily SMA = {daily_sma}")

reader.close()
```

#### CLI (pyne run)

Use the `--security` flag to provide OHLCV data for each security context. The flag can be
repeated for multiple contexts:

```bash
# Single security context (daily data for same symbol)
pyne run multi_tf_sma.py EURUSD_5m --security "1D=EURUSD_1D"

# Multiple security contexts (different symbols)
pyne run advance_decline.py SPX_1D \
  --security "USI:ADVN.NY=USI_ADVN_NY" \
  --security "USI:DECL.NY=USI_DECL_NY"
```

The format is `"KEY=DATA_NAME"` where:

- **KEY** is `"TIMEFRAME"` or `"SYMBOL:TIMEFRAME"` (matching the `request.security()` call)
- **DATA_NAME** is the OHLCV data name in `workdir/data/` (without extension)

### Key Matching Rules

The `security_data` dict keys are matched against each `request.security()` call's symbol and
timeframe:

| Key format           | Example         | Matches                                  |
|----------------------|-----------------|------------------------------------------|
| `"TIMEFRAME"`        | `"1D"`          | Any security call with timeframe `"1D"`  |
| `"SYMBOL:TIMEFRAME"` | `"AAPL:1H"`     | Exact match on both symbol and timeframe |
| `"SYMBOL"`           | `"USI:ADVN.NY"` | Any security call with that symbol       |

Timeframe-only keys are convenient when all security calls use the same symbol (the chart symbol).

## Examples

### Multi-Timeframe Indicator

```python
"""@pyne"""

from pynecore.lib import *
from pynecore.types import *


@script.indicator("MTF RSI")
def main():
    rsi_5m: Series[float] = lib.ta.rsi(lib.close, 14)

    # Get RSI from higher timeframes
    rsi_1h: Series[float] = lib.request.security(
        lib.syminfo.tickerid, "60", lib.ta.rsi(lib.close, 14)
    )
    rsi_daily: Series[float] = lib.request.security(
        lib.syminfo.tickerid, "1D", lib.ta.rsi(lib.close, 14)
    )

    lib.plot.plot(rsi_5m, "RSI 5m")
    lib.plot.plot(rsi_1h, "RSI 1H")
    lib.plot.plot(rsi_daily, "RSI Daily")
```

```python
security_data = {
    "60": "workdir/data/EURUSD_60",  # 1-hour bars
    "1D": "workdir/data/EURUSD_1D",  # daily bars
}
```

### Multi-Symbol Analysis (Advance/Decline Ratio)

```python
"""@pyne"""

from pynecore.lib import *
from pynecore.types import *


@script.indicator("Advance/Decline Ratio")
def main():
    advancing: Series[float] = lib.request.security("USI:ADVN.NY", "", lib.close)
    declining: Series[float] = lib.request.security("USI:DECL.NY", "", lib.close)

    ratio: Series[float] = lib.nz(advancing) / lib.nz(declining, 1.0)
    lib.plot.plot(ratio, "A/D Ratio")
```

```python
security_data = {
    "USI:ADVN.NY": "workdir/data/USI_ADVN_NY",
    "USI:DECL.NY": "workdir/data/USI_DECL_NY",
}
```

> **Note:** When the timeframe argument is `""` (empty string), the chart's own timeframe is used.

## Supported Features

| Feature                 | Status        | Notes                                      |
|-------------------------|---------------|--------------------------------------------|
| Different timeframe     | supported     | HTF (1D, 1W, 1M, etc.) from lower TF chart |
| Different symbol        | supported     | Any symbol with available OHLCV data       |
| Multiple security calls | supported     | Each gets its own process                  |
| Conditional calls       | supported     | Inside `if`/`for`/`while` blocks           |
| Nested security calls   | supported     | `security(... security(...) ...)`          |
| `barmerge.gaps_off`     | supported     | Forward-fills last value (default)         |
| `barmerge.gaps_on`      | supported     | Returns `na` between periods               |
| `lookahead_off`         | supported     | Confirmed previous period (default)        |
| `lookahead_on`          | not supported | Deliberate safety-first decision           |
| Lower timeframe         | not supported | Only same or higher TF                     |

## How It Works

Under the hood, each `request.security()` call spawns a separate OS process:

1. **AST transformation** rewrites the call into signal/write/read/wait protocol functions
2. **ScriptRunner** detects the transformed code, creates shared memory, and spawns processes
3. Each security **process** loads its own OHLCV data and runs the script independently
4. Processes communicate results via **shared memory** — no serialization overhead for simple types
5. The chart process **waits** for security results only when a new period is confirmed
6. **Pipeline parallelism**: security processes run on separate CPU cores concurrently

Side effects (plots, strategy orders, alerts) are automatically suppressed in security contexts
via the existing `_lib_semaphore` mechanism.

### HTF Period Confirmation

For higher-timeframe data, values are confirmed with **lookahead_off** semantics: a daily value
becomes available only when the next daily bar opens (i.e., when the period boundary is crossed).

```
Chart bars (5m):    10:00  10:05  10:10 ... 23:55  00:00  00:05
                                                    ↑
                                              New daily period starts
                                              Yesterday's daily value is now confirmed
```

For same-timeframe contexts (different symbol), values are confirmed on every bar.

### gaps_on vs gaps_off

| Mode       | New period confirmed | Between periods                  |
|------------|----------------------|----------------------------------|
| `gaps_off` | Return new value     | Return last value (forward-fill) |
| `gaps_on`  | Return new value     | Return `na`                      |

## Technical Deep-Dive

### Architecture

```
                           Chart Process
                          +----------------------------+
                          |  ScriptRunner.run_iter()   |
                          |    for each bar:           |
                          |      __sec_signal__(id)    |-----> advance_event.set()
                          |      ...                   |
                          |      __sec_read__(id, na)  |<----- data_ready.wait()
                          |      ...                   |
                          |      __sec_wait__(id)      |<----- done_event.wait()
                          +----------------------------+
                                    |    ^
                              SharedMemory (SyncBlock + ResultBlocks)
                                    v    |
                          +----------------------------+
                          |  Security Process (per ID) |
                          |    advance_event.wait()    |
                          |    run bars to target_time |
                          |      __sec_write__(id, v)  |-----> write to ResultBlock
                          |    data_ready.set()        |
                          |    done_event.set()        |
                          +----------------------------+
```

### AST Transformation

The `SecurityTransformer` rewrites each `lib.request.security()` call into four protocol
functions. For example:

**Before:**

```python
def main():
    daily = lib.request.security(lib.syminfo.tickerid, "1D", lib.ta.sma(lib.close, 20))
```

**After (conceptual):**

```python
def main():
    if __active_security__ is None:
        __sec_signal__("sec-abc-0")  # chart: signal security process

    if __active_security__ == "sec-abc-0":
        __sec_write__("sec-abc-0", lib.ta.sma(lib.close, 20))  # security: write result
    daily = __sec_read__("sec-abc-0", lib.na)  # both: read result

    if __active_security__ is None:
        __sec_wait__("sec-abc-0")  # chart: wait for completion
```

The `__active_security__` variable is `None` in the chart process and set to the security ID
in each security process. This makes the same compiled code run correctly in all contexts.

Write/read blocks stay at the **same scope level** as the original call, preventing `NameError`
for conditionally-scoped variables.

### Shared Memory Layout

**SyncBlock** — one per script run, contains metadata for all security slots:

| Field            | Type   | Offset | Description                              |
|------------------|--------|--------|------------------------------------------|
| `last_timestamp` | int64  | 0      | Last bar timestamp from security process |
| `version`        | uint32 | 8      | ResultBlock version (reallocation count) |
| `result_size`    | uint32 | 12     | Current pickle data size in bytes        |
| `target_time`    | int64  | 16     | Time the process should advance to       |
| `flags`          | uint8  | 24     | State flags                              |

**ResultBlock** — one per security context, holds pickled result. Auto-reallocates with doubled
size when data outgrows the block.

### Security Process Lifecycle

1. **Spawn** — `multiprocessing.Process` with `spawn` mode
2. **Init** — re-register import hooks, open shared memory by name, load OHLCV + syminfo
3. **Import** — re-import script module (AST transforms run again, fresh Series state)
4. **Configure** — `lib._lib_semaphore = True`, `__active_security__ = sec_id`
5. **Loop** — `advance_event.wait()` → run bars to `target_time` → `data_ready.set()` + `done_event.set()`
6. **Shutdown** — `stop_event` detected → cleanup and exit

### Cross-Context Reads

Security processes reading other contexts' values (nested dependencies) get **immediate** reads
without waiting. This prevents deadlocks between security processes.

### File Reference

| File                       | Purpose                                       |
|----------------------------|-----------------------------------------------|
| `transformers/security.py` | AST transformation (SecurityTransformer)      |
| `core/security_shm.py`     | Shared memory: SyncBlock, ResultBlock, Reader |
| `core/security.py`         | Runtime protocol, SecurityState, event logic  |
| `core/security_process.py` | Security process entry point and bar loop     |
| `core/script_runner.py`    | Integration: spawn, inject, cleanup           |

## Limitations

- **lookahead_on** — not supported. Only `lookahead_off` (confirmed previous period). Deliberate
  safety-first decision.
- **Lower timeframe** — fetching data from a lower timeframe than the chart is not supported.
- **Early return** — if a function has an early `return`, the `__sec_wait__` block at function end
  is skipped. The security process still completes, but results may not be read on that bar.
- **Same symbol + same TF** — still spawns a separate process. Optimization planned.
- **Standalone mode** — `python script.py data.csv` does not support `--security` yet.
  Use `pyne run` or the ScriptRunner API.
