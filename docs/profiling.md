<!--
---
weight: 1310
title: "Profiling"
description: "Finding the slow lines of a script with pyne run --profile"
icon: "speed"
date: "2026-10-24"
lastmod: "2026-10-24"
draft: false
toc: true
categories: ["Support"]
tags: ["profiling", "performance", "cli"]
---
-->

# Profiling

The profiler shows where a backtest spends its time: how long each line of the script ran,
how many times it ran, and how the whole run splits between the script, the strategy engine,
reading the data and writing the outputs.

## Running a Profile

Add `--profile` to a backtest:

```bash
pyne run bollinger_bands_strategy capitalcom_EURUSD_5 --profile
```

After the run, the CLI prints a summary and writes the full profile to
`output/<script_name>_profile.json`:

```
Profile: 74,088 bars, 1.32 s; 198 ms in the script's code (the base of % script)
  Script code         198 ms    14.9%
  Strategy engine     491 ms    37.2%
  Output             82.0 ms     6.2%
  Data feed           398 ms    30.1%
  Runner              153 ms    11.5%

 Line     Time  % script   Execs     Avg  Code
   18  42.7 ms     21.6%  74,088  577 ns  dev = mult * ta.stdev(source, length)
   17  29.3 ms     14.8%  74,088  395 ns  basis = ta.sma(source, length)
   31  21.0 ms     10.6%  71,806  293 ns  strategy.cancel(id='BBandLE')
   ...

Full profile: /path/to/workdir/output/bollinger_bands_strategy_profile.json
```

`--profile-top N` sets how many lines the summary lists (15 by default). For a Pine Script
file the lines are those of the Pyne code PyneComp compiled from it.

## Reading the Numbers

**Line time** runs from the moment a line starts to the moment the next line of the same
function starts, so it includes everything the line calls. A `ta.sma(...)` line carries the
indicator's own work, a call of one of your functions carries the whole call, and a
`strategy.entry(...)` line carries the order bookkeeping it does right away.

**% script** is the line's share of the time spent in the script's code, which is the time
from entering the script (or one of its libraries) to leaving it, everything it calls
included.

**Execs** is how many times the line ran. A line inside a loop runs once per iteration; a
loop header runs once more than its body, because its last check ends the loop. A strategy
that re-runs the script within a bar (`calc_on_order_fills`, `calc_on_every_tick`) counts
those runs too, so a line can run more times than there are bars.

**per-call setup** marks a function's `def` line. Before the first statement of a function,
PyneCore prepares its series and persistent state for the call; that time lands on the `def`
line.

The **categories** split the run's time without overlap:

| Category         | What it covers                                                                      |
|------------------|-------------------------------------------------------------------------------------|
| Script code      | The script's own code and the PyneCore functions it calls                           |
| Libraries        | The code of the Pyne libraries the script imports                                   |
| Strategy engine  | Filling orders between bars: stops, limits, margin calls, `process_orders_on_close` |
| Output           | Writing the plot, trade, statistics and visual output files                         |
| Data feed        | Reading the OHLCV data and building the chart's bars                                |
| request.security | Starting the `request.security` contexts and waiting for their results              |
| Runner           | Everything else the runner does per bar                                             |

## request.security Contexts

Each `request.security` context runs the script, or the part of it the context needs, in a
process of its own, on the context's own bars. That process is profiled too, and the
summary lists each context with its data, timeframe, bars and the time its script code
took:

```
request.security contexts (their own processes)
  bybit_BTCUSDT_1D.ohlcv    D    352 bars    14.0 ms in the script's code
```

The script waits for a context's results, and that wait is what the `request.security`
category and the time of a `request.security(...)` line hold. When the context's own script
time is much smaller than the wait, the rest is spent outside the script: starting the
context's process, reading its data and handing its values over.

## The Profile File

`output/<script_name>_profile.json` holds the full measurement:

- `bars`, `wall_ns` (the run's time), `script_ns` (the time in the script's code) and
  `categories` (nanoseconds per category);
- `files`: the script and every library that ran, each with its `path`, the `sha256` of the
  source that was profiled, `lines` (line number → `[nanoseconds, executions]`) and
  `blocks` (every `def`, loop, `if`, `match`, `with` and `try` that ran, with the total time
  of its lines, the header line's own time and the header's executions).

- `contexts`: one entry per `request.security` process that ran script code, with its
  `sec_ids`, `timeframe`, `data` (the data file or symbol it read), `bars`, `script_ns`
  and `files` (`path`, `lines` and `blocks` as above, measured in that process).

The `sha256` tells whether a profile still belongs to the source: once the file is edited,
its line numbers may no longer match.

## Limits

- Profiling needs Python 3.13 or newer and PyneCore's compiled profiler extension, which
  every PyneCore wheel carries. An installation without it (a pure-Python build) refuses
  `--profile` with an error instead of producing a skewed profile.
- Only backtests can be profiled: `--profile` is refused with `--live` and `--broker`.
- Another tool that holds Python's profiler slot in the same process (another profiler)
  makes `--profile` fail with an error.
- The profiler slows a long backtest down by about 15%.
- A loop header spanning several lines whose last line also holds the body
  (`for i in range(\n 3): x = i`) counts the header too many times; the times are still
  exact. A body on its own line is counted exactly.

From Python, the same profile is written by
[`ScriptRunner(profile_path=...)`](./programmatic/script-runner.md).
