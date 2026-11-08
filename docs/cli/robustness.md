<!--
---
weight: 306
title: "Robustness Tests"
description: "Test a strategy's backtest with pyne robustness: Monte Carlo, coin-flip, bootstrap, cost and period tests, written as CSV files. Reference of the command's arguments and options."
icon: "casino"
date: "2026-11-08"
lastmod: "2026-11-08"
draft: false
toc: true
categories: ["CLI"]
tags: ["robustness", "monte-carlo", "coin-flip", "backtesting", "cli"]
---
-->

# Robustness Tests

The `robustness` command runs a strategy's backtest once and tests how much of its result
could be luck. What the tests do and how to read them is in
[Robustness Tests](../robustness.md); this page is the command reference.

## Basic Usage

```bash
pyne robustness SCRIPT DATA [OPTIONS]
pyne robustness SCRIPT --from-trades TRADE_CSV [OPTIONS]
```

Where:
- `SCRIPT`: The strategy, as Pyne code (.py) or as a Pine Script (.pine) file that is
  converted to Pyne code first (a PyneSys API key is needed, see
  [Running Scripts](./run.md#pine-script-support))
- `DATA`: The data file (.ohlcv, .csv, .json, .txt) or a provider string, as for
  [`pyne run`](./run.md)
- `TRADE_CSV`: A trade CSV written by `pyne run`, analysed without running the backtest

```bash
# Test a strategy with the default settings
pyne robustness ema_cross binance_BTCUSDT_15

# Both coin-flip tests, a 2x target and no leverage in the account Monte Carlo
pyne robustness ema_cross binance_BTCUSDT_15 --coinflip both --target 2 --max-leverage 1
```

The command prints a summary table and writes the results as CSV files.

## Command Options

### Data Options

The same as for `pyne run`:

- `--from`, `-f`: Start date (UTC), days back from now, or `-N` bars back
- `--to`, `-t`: End date (UTC) or days back from now
- `--timeframe`, `-tf`: Chart timeframe in TradingView format (e.g. `60`, `1D`)
- `--security`, `-sec`: Security data, `"TIMEFRAME=data_name"` or `"SYMBOL:TIMEFRAME=data_name"`
- `--api-key`, `-a`: PyneSys API key for Pine Script compilation

### Input Options

- `--from-trades`: Analyse a trade CSV written by `pyne run` instead of running the backtest.
  The file has no stops and no point value, so the results are in account currency, the cost
  test is skipped, and the coin-flip tests, which run the strategy, do not run.

### Test Options

- `--unit`: The unit trades are measured in: `auto` (default; R when every trade has a stop,
  account currency otherwise), `r`, `currency` or `percent`
- `--draws`: Resamples of every random test (default: 10000)
- `--seed`: Seed of every random test (default: 0). The same trades, settings and seed give
  the same numbers.
- `--mc-method`: Monte Carlo method: `bootstrap` (default; draws trades with replacement) or
  `shuffle` (reorders the trades)
- `--block`: Bootstrap block: `week` (default), `month` or `trade`
- `--confidence`: Coverage of the intervals, percent (default: 95)
- `--cost-bps`: Extra cost scenarios per side, basis points of the traded value (default:
  `0,2,5,10`)
- `--periods`: Calendar breakdowns, any of `year` (default), `month`, `weekday`, comma separated

### Account Options

- `--capital`: Starting account of the account Monte Carlo (default: the equity the first
  trade opened with)
- `--risk`: Account risked per trade when results in R are compounded, percent (default: 1)
- `--max-leverage`: Largest position of the account Monte Carlo of results in R, as a multiple
  of the account (`1` = no leverage). A trade whose stop is too close to risk `--risk` within
  the limit risks less. Without the option there is no limit.
- `--ruin`: Loss of the starting account that counts as ruin, percent (default: 50)
- `--target`: Multiple of the starting account counted as reaching the target (e.g. `2`)

### Coin Flip Options

- `--coinflip`: `direction` (default; a coin decides long or short on every real entry),
  `timing` (the real trades enter at random moments), `both` or `none`
- `--coinflip-runs`: Backtests of the tests that run one per draw: `timing`, and `direction`
  with `--exact` (default: 200)
- `--exact`: Direction test: a full backtest per draw instead of pairing the trades of one
  flipped backtest. Needed when the strategy's later entries depend on its earlier trades.

### Output Path Options

- `--output`, `-o`: Directory of the result files (default: `workdir/output/`)
- `--json`: Also write the whole result as `<script>_robustness.json`

## Output Files

| File                           | Contents                                                     |
|--------------------------------|--------------------------------------------------------------|
| `<script>_robustness.csv`      | One row per measured figure (columns below)                  |
| `<script>_montecarlo.csv`      | P5–P95 path bands by trade number: running total and account |
| `<script>_costs.csv`           | The statistics under each extra cost                         |
| `<script>_periods.csv`         | The statistics by year, month or weekday                     |
| `<script>_coinflip_<mode>.csv` | Every random total of a coin-flip test, with the real total  |
| `<script>_robustness.json`     | With `--json`: the whole result                              |

The columns of `<script>_robustness.csv`:

| Column       | Contents                                                   |
|--------------|------------------------------------------------------------|
| Test         | Backtest, Hit rate, Bootstrap, Monte Carlo, Coin flip, ... |
| Metric       | The measured figure                                        |
| Unit         | R, currency, %, bps or trades                              |
| Value        | The figure                                                 |
| Low / P5     | Lower end of the interval, or the 5th percentile           |
| Median / P50 | Median of a random distribution                            |
| High / P95   | Upper end of the interval, or the 95th percentile          |
| Draws        | Resamples or backtests behind the figure                   |
| Seed         | Seed of the random test                                    |
| Method       | How the figure was made                                    |
| Verdict      | The figure in plain words, where there is one              |
