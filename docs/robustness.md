<!--
---
weight: 1320
title: "Robustness Tests"
description: "Test how much of a PyneCore strategy's backtest result could be luck: Monte Carlo of the trade sequence, coin-flip tests, bootstrap intervals, extra costs and period breakdowns."
icon: "casino"
date: "2026-11-08"
lastmod: "2026-11-08"
draft: false
toc: true
categories: ["Strategy"]
tags: ["robustness", "monte-carlo", "coin-flip", "bootstrap", "backtesting", "strategies"]
---
-->

# Robustness Tests

A backtest is one path through history. The robustness tests ask how much of its result
depends on luck: on the order the trades came in, on the few best trades, on the trading
costs, on the years in the sample, and on whether the strategy's decisions beat random ones.

The tests run on the strategy's closed trades. The coin-flip tests also run the strategy
again in the same engine, so the real and the random trades fill by the same rules.

```bash
pyne robustness ema_cross binance_BTCUSDT_15
```

The command runs the backtest once, prints a summary table and writes the results as CSV
files. Its options are listed in [pyne robustness](./cli/robustness.md).

## Measuring Trades in R

A trade with a stop-loss is measured in **R**, its profit divided by the loss it would have
made at its initial stop. A trade that hit its stop is −1 R, one that made twice its risk is
+2 R. R removes the position size from the result, so a strategy is judged by its decisions
and not by how much it bet.

The initial stop is the first stop level among the exit orders bound to the entry: the ones
placed together with the entry, or the ones the first script execution after the fill
placed. When an exit has both a `stop` price and a `loss` offset, the nearer level counts.

Without a stop on every trade the results are in account currency (`--unit currency`).
`--unit percent` measures each trade as a percent of the equity it opened with. A trade
closed in parts counts as one trade: its parts are merged into one record.

## The Tests

| Test                | What it shows                                                                            |
|---------------------|------------------------------------------------------------------------------------------|
| Backtest            | Total, mean and median per trade, profit factor, drawdown, longest losing streak         |
| Concentration       | The total without the 3 best trades                                                      |
| Hit rate            | Wilson interval of the share of winning trades                                           |
| Bootstrap           | Interval of the mean trade, resampling calendar weeks (or months, or single trades)      |
| Monte Carlo         | Spread of the final result and of the drawdown when the trades come in another order     |
| Account Monte Carlo | The same as a compounding account: final equity, drawdown, ruin and target chances       |
| Costs               | The result with extra trading costs per side, and the extra cost that brings it to zero  |
| Periods             | The result by year, month or weekday                                                     |
| Coin flip           | The real result against the same trades with a random direction or a random entry moment |

Every random test is seeded (`--seed`, 0 by default): the same trades, settings and seed give
the same numbers.

### Hit Rate and Bootstrap

The **hit rate** comes with a Wilson interval (95% by default, `--confidence`). With 25 trades
and 9 winners the interval is roughly 20% to 55%: a short backtest says little about the
real hit rate.

The **bootstrap** resamples the trades in calendar blocks, a week by default, so trades that
cluster in time are drawn together, and empty weeks are drawn too. Its interval is the range
of the mean trade the resamples gave. When the interval includes zero, the backtest cannot
tell the mean trade from zero. The row below it is the share of resamples whose mean was
zero or below.

### Monte Carlo

The Monte Carlo builds 10,000 paths from the trades (`--draws`):

- `bootstrap` (default) draws as many trades as there were, with replacement, so the total
  varies as well;
- `shuffle` keeps every trade exactly once and only changes the order: the total stays the
  same, the drawdown and the path change.

It reports the P5, P50 and P95 of the final result and of the largest drawdown, and the share
of paths that ended at or below the start. Both forms treat the trades as independent and
describe this backtest, not the future.

The **account Monte Carlo** runs the same paths as an account that compounds:

- Trades in R risk a fixed share of the account per trade (`--risk`, 1% by default), so a
  −1 R trade loses 1% of the current account.
- Trades in other units compound the strategy's own return per trade.

The account starts with the equity of the first trade (or `--capital`). A path that falls to
the **ruin** level stops trading: by default that is the loss of half the starting account
(`--ruin 50`). `--target 2` also counts the paths that doubled the account before any ruin.

Risking 1% with a stop 0.1% away takes a position of 10 times the account. `--max-leverage`
caps the position at a multiple of the account: with `--max-leverage 1` that trade risks
0.1% instead of 1%, and the trades whose stop is far enough keep the full 1%. The report says
how many trades the limit capped. Without the option there is no limit.

### Costs

Every scenario charges each trade an extra cost in basis points of its traded value, one side
on the entry and one on the exit, on top of what the backtest already paid. The default
scenarios are 0, 2, 5 and 10 bps per side (`--cost-bps`). The fills stay as they were, so
a strategy whose signals depend on its own equity could react differently under the cost.

The **break-even** cost is the extra cost per side that brings the total to zero. A strategy
that breaks even at 1 bps per side does not survive a real spread.

### Coin Flip

A coin-flip test compares the real result with the results of random decisions on the same
trades. One choice of the strategy is randomized at a time:

| Mode        | What stays                                                         | What is random           |
|-------------|--------------------------------------------------------------------|--------------------------|
| `direction` | Every entry: its moment, price and size, stop and target distances | Long or short, by a coin |
| `timing`    | Every trade's direction, size, stop distance and holding time      | The moment of the entry  |

**Direction.** A flipped entry fills at the same moment and price: a stop entry becomes a
limit entry at the same level and the other way round. Its stop and target are mirrored
through the fill price, so they keep their distances. The script keeps seeing the side it
asked for: `strategy.position_size`, `strategy.opentrades.size()` and
`strategy.closedtrades.size()` carry the sign of the entry it placed, so a bracket it builds
from `strategy.position_size > 0` after the fill is the long bracket, mirrored like the
others. Profits, equity and prices stay the real ones.

By default the test runs one more backtest with every entry flipped, pairs its trades with
the real ones, and each of the 10,000 draws picks the real or the flipped outcome of every
trade by a coin. This assumes the trades do not depend on each other. A strategy whose later
entries depend on the earlier trades (a flipped trade that closes earlier or later, a
compounding position size) loses pairs: when fewer than 90% of the real trades have a
counterpart in the flipped backtest, the test gives no verdict and says so. Such a strategy
needs `--exact`: a full backtest per draw, 200 by default (`--coinflip-runs`).

**Timing.** The script places no order. The real trades enter in their own order at random
bars between the first real entry and the last real exit, each once the previous one has
closed, with the real trade's direction and size, a stop at its stop distance and a close
after its holding time. The script's own exits are not used, as they belong to its own entry
signals. Every draw is a backtest, 200 by default.

**Backtests per draw.** The exact direction test and the timing test run a backtest per draw.
They run several at the same time, each in its own process: by default as many as the CPU
count minus one (`--jobs`). Every draw takes its random choices from the seed and its own
number, so the results are the same with any number of processes.

They also stop once the verdict is clear (`--early-stop`). After 30 backtests, and after
every 10 more, the test computes a 99% interval of the share of random draws the real rules
beat. The default rule, `edge`, stops when that interval is all below or all above 95%: the
real rules clearly do not, or clearly do, beat 95% of the random draws. `better` asks the same
about half of the random draws and stops sooner. `off` runs every backtest. The interval is
99% rather than 95% because it is checked again and again. A stopped test reports how many
backtests it ran; its percentages are less precise than those of 200 backtests, but the
answer to the rule's question is clear. A strategy whose result sits close to the mark runs
every backtest.

The result reads like this:

```
Coin flip (direction): 28% of the random direction draws did better; the real rules beat 72% of them
```

after an early stop:

```
Coin flip (direction): 87% of the random direction draws did better; the real rules beat 13% of them (stopped after 30 of 200 backtests: the real rules clearly beat fewer than 95% of the random draws)
```

or, when the pairing fails:

```
Coin flip (direction): not valid for this script: only 927 of 1122 trades had a counterpart in the flipped backtest (at least 90% needed)
```

A strategy whose rules carry information should beat most of the random draws. When the
real result sits in the middle of the random ones, the backtest is what random decisions on
the same trades give. The test names what it could not randomize: stop-limit entries have no
mirror image and keep their direction, `strategy.order` calls are not flipped, and a timing
draw can run out of flat bars before the end of the data.

`--coinflip` selects `direction` (default), `timing`, `both` or `none`.

## Example

A 15-minute EMA crossover on BTCUSDT with a stop at the recent low and a target at twice the
risk, 996 trades:

```bash
pyne robustness ema_cross binance_BTCUSDT_15 --coinflip both --target 2 --max-leverage 1
```

| Test                  | Result                                                                    |
|-----------------------|---------------------------------------------------------------------------|
| Backtest              | +39.0 R over 996 trades, 0.039 R per trade, hit rate 33%                  |
| Concentration         | +33.0 R without the 3 best trades                                         |
| Bootstrap             | Mean per trade −0.024 R to 0.108 R: the interval includes zero            |
| Monte Carlo           | Final result P5 −24.7 R, P50 38.4 R; 15% of the paths ended at or below 0 |
| Account Monte Carlo   | 1% per R, at most 1x leverage (668 trades capped): P50 13,242 from 10,000 |
| Costs                 | Break-even at 1.1 bps per side; −31 R at 2 bps                            |
| Coin flip (direction) | The real rules beat 72% of the random directions                          |
| Coin flip (timing)    | The real rules beat 83% of the random entry moments                       |

The backtest is profitable before costs, but its mean trade cannot be told from zero, and
1.1 bps of extra cost per side erases it.

## Output Files

The results go next to the other outputs (`workdir/output/`, or `--output`):

| File                           | Contents                                                     |
|--------------------------------|--------------------------------------------------------------|
| `<script>_robustness.csv`      | One row per measured figure, with its interval and method    |
| `<script>_montecarlo.csv`      | P5–P95 path bands by trade number: running total and account |
| `<script>_costs.csv`           | The statistics under each extra cost                         |
| `<script>_periods.csv`         | The statistics by year, month or weekday                     |
| `<script>_coinflip_<mode>.csv` | Every random total of a coin-flip test, with the real total  |
| `<script>_robustness.json`     | With `--json`: the whole result                              |

The CSV files open in any spreadsheet program. Their columns are listed in
[pyne robustness](./cli/robustness.md#output-files).

## Analysing a Trade CSV

`--from-trades` analyses the trade CSV of an earlier `pyne run` without running the
backtest again:

```bash
pyne robustness ema_cross --from-trades workdir/output/ema_cross_trade.csv
```

The trade CSV has no stops, so the results are in account currency, and it has no point
value, so the cost test is skipped. The coin-flip tests need the strategy itself and do not
run.

## From Python

The tests are in the `pynecore.analysis` package. A `TradeRecorder` collects the closed
trades with their initial stops while a [`ScriptRunner`](./programmatic/script-runner.md)
runs, and `analyze()` tests them:

```python
from pathlib import Path

from pynecore.analysis import RobustnessConfig, TradeRecorder, analyze
from pynecore.analysis.coinflip import coinflip_test
from pynecore.core.ohlcv import OHLCVReader
from pynecore.core.script_runner import ScriptRunner
from pynecore.core.syminfo import SymInfo

script = Path("workdir/scripts/ema_cross.py")
data = Path("workdir/data/binance_BTCUSDT_15.ohlcv")
syminfo = SymInfo.load_toml(data.with_suffix(".toml"))
with OHLCVReader(data) as reader:
    candles = list(reader.read_from(reader.start_timestamp, reader.end_timestamp))


def make_runner() -> ScriptRunner:
    """A fresh runner of the strategy; the coin-flip tests build one per backtest."""
    return ScriptRunner(script, iter(candles), syminfo, last_bar_index=len(candles) - 1,
                        last_bar_time=candles[-1].timestamp)


runner = make_runner()
recorder = TradeRecorder(runner.script.position, runner.syminfo)
for _ in runner.run_iter(copy_results=False):
    recorder.on_bar()

report = analyze(recorder.trades, RobustnessConfig(draws=2000, max_leverage=1.0))
report.coinflips.append(coinflip_test("direction", make_runner, syminfo, recorder.trades,
                                      report.unit))
report.write_csv(Path("workdir/output"), "ema_cross")
print(report.coinflips[0].verdict)
```

`coinflip_test` runs its backtests one at a time and to the end unless it gets `jobs` and
`early_stop`. With `jobs` above 1, `make_runner` goes to the worker processes, so it must be
picklable: a module-level function or class instance, not a closure like the one above.

`report.to_dict()` gives the whole result as JSON-ready data, `report.summary_rows()` the
rows of the summary table. The engine's closed-trade list is capped, so a long backtest must
be recorded while it runs, as above, rather than read back at the end.
