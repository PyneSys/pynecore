<!--
---
weight: 507
title: "request.currency_rate() & Corporate Events"
description: "Currency conversion rates and corporate event data (dividends, splits, earnings) in PyneCore"
icon: "currency_exchange"
date: "2026-03-28"
lastmod: "2026-10-25"
draft: false
toc: true
categories: ["Reference"]
tags: ["request", "currency", "dividends", "splits", "earnings", "corporate-events"]
---
-->

# request.currency_rate() & Corporate Events

PyneCore supports `request.currency_rate()` for currency conversion, and provides graceful
fallback behavior for `request.dividends()`, `request.splits()`, and `request.earnings()`.

## request.currency_rate()

Returns the exchange rate between two currencies at the current bar's timestamp.

### Function Signature

```python
request.currency_rate(from_currency: str, to_currency: str) -> float
```

| Parameter       | Type   | Description                                        |
|-----------------|--------|----------------------------------------------------|
| `from_currency` | string | Source currency code (`"EUR"`, `currency.EUR`, etc) |
| `to_currency`   | string | Target currency code (`"USD"`, `currency.USD`, etc) |

**Returns:** Exchange rate as `float`, or `na` if no data is available.

### How It Works

`request.currency_rate()` resolves exchange rates from OHLCV data files by reading their TOML
metadata. Every OHLCV file has an accompanying `.toml` with `currency` and `basecurrency` fields:

```toml
# capitalcom_EURUSD_1D.toml
[symbol]
currency = "USD"
basecurrency = "EUR"
```

When you call `request.currency_rate("EUR", "USD")`, PyneCore:

1. Looks for a source whose metadata has `basecurrency = "EUR"` and `currency = "USD"`, in this
   order of priority:
   1. a rate-source feed that a data provider plugin supplies for this run (it runs as a hidden
      security context) — the latest close it has published is used; ordinary
      `request.security()` calls of the script are never read as rates;
   2. an OHLCV file from `security_data` whose `.toml` matches (if several match, the one with
      more bars wins);
   3. the same lookups for the inverse pair (`USD`→`EUR`), returning `1.0 / rate`;
   4. the chart itself, if it is the pair (or its inverse) — PyneCore reads `lib.close` directly.
2. For a file source, lazy-loads the OHLCV file on first access
3. For a file source, takes the close of the last rate bar that has **already closed** at the
   current chart bar's open time. A rate bar opening at `t` closes at `t + period`, so on a chart
   bar opening at `c` only rate bars with `t + period <= c` count. With an hourly rate file, the
   bar that opened at 10:00 is first usable on a chart bar opening at 11:00; a chart bar opening at
   10:30 gets the 09:00 bar's close. Reading the still-open bar would be lookahead.
4. Returns that close as the exchange rate

### Special Cases

| Condition                    | Result             |
|------------------------------|--------------------|
| `from == to`                 | `1.0`              |
| `from` or `to` is empty `""` | `1.0`              |
| `from` or `to` is `NONE`     | `na`               |
| No matching data             | `na`               |
| Inverse pair available       | `1.0 / close`      |
| Chart is the currency pair   | `lib.close` direct |

### Data Preparation

Provide FX or crypto OHLCV data via the `security_data` parameter. The key names don't matter —
currency pairs are detected from TOML metadata:

```python
from pathlib import Path
from pynecore.core.script_runner import ScriptRunner
from pynecore.core.ohlcv import OHLCVReader
from pynecore.core.syminfo import SymInfo

syminfo = SymInfo.load_toml("workdir/data/BTCUSDT_4h.toml")
reader = OHLCVReader("workdir/data/BTCUSDT_4h.ohlcv")
reader.open()

runner = ScriptRunner(
    script_path=Path("workdir/scripts/portfolio.py"),
    ohlcv_iter=reader,
    syminfo=syminfo,
    security_data={
        # Key names are arbitrary — currency pairs are detected from TOML
        "eurusd": "workdir/data/capitalcom_EURUSD_1D",
        "btcusd": "workdir/data/ccxt_BINANCE_BTC_USDT_1D",
    },
)
```

CLI:

```bash
pyne run portfolio.py BTCUSDT_4h \
  --security "eurusd=capitalcom_EURUSD_1D" \
  --security "btcusd=ccxt_BINANCE_BTC_USDT_1D"
```

### Example: Multi-Currency Comparison

```python
"""@pyne"""

from pynecore.lib import *
from pynecore.types import *


@script.indicator("BTC in Multiple Currencies")
def main():
    btc_usd: Series[float] = close  # Chart is BTC/USD

    # Convert BTC price to EUR and GBP
    eur_rate: float = request.currency_rate("USD", "EUR")
    gbp_rate: float = request.currency_rate("USD", "GBP")

    btc_eur: Series[float] = btc_usd * eur_rate
    btc_gbp: Series[float] = btc_usd * gbp_rate

    plot.plot(btc_usd, "BTC/USD", color=color.blue)
    plot.plot(btc_eur, "BTC/EUR", color=color.green)
    plot.plot(btc_gbp, "BTC/GBP", color=color.red)
```

### Pair Resolution

When multiple OHLCV files match the same currency pair (e.g., two EUR/USD sources at different
timeframes), the file with the most bars is selected. This ensures the best timestamp coverage.

Inverse lookups are automatic: if you only have EUR/USD data but request
`currency_rate("USD", "EUR")`, PyneCore returns `1.0 / close`.

### Relationship to request.security() currency Parameter

The `currency` parameter in `request.security()` auto-converts results to a target currency using
the same `CurrencyRateProvider`. When a `currency` argument is provided, the security runtime
reads the source currency from the security context's TOML metadata, fetches the exchange rate
via `request.currency_rate()`, and multiplies the result automatically.

### File Reference

| File                  | Purpose                                           |
|-----------------------|---------------------------------------------------|
| `core/currency.py`   | CurrencyRateProvider: TOML scan, OHLCV cache      |
| `lib/request.py`     | `currency_rate()` function stub + provider bridge  |
| `core/script_runner.py` | Provider creation, injection, and cleanup       |

## Corporate Event Functions

### request.dividends(), request.splits(), request.earnings()

These functions request corporate event data (dividends, stock splits, earnings reports).
In TradingView, they return values on bars where events occur and `na` on other bars.

**Current status:** These functions are **not yet fully implemented** in PyneCore. However, they
handle `ignore_invalid_symbol=True` gracefully by returning `na` instead of raising an error. They
also return `na` when they request the current crypto chart's own ticker (a crypto asset has no
corporate events).
This allows scripts like the TradingView VWAP indicator (which uses all three) to run without
modification.

```python
# This works — returns na silently, script continues
new_earnings = request.earnings(
    syminfo.tickerid, earnings.actual,
    barmerge.gaps_on, barmerge.lookahead_on,
    ignore_invalid_symbol=True  # Required for graceful fallback
)

# This still raises NotImplementedError (unless the chart is a crypto symbol)
new_earnings = request.earnings(syminfo.tickerid, earnings.actual)
```

### Planned Implementation

Full corporate event support is planned using the existing `.extra.csv` sidecar mechanism, where
event data is provided as named columns alongside OHLCV bars.

### request.footprint()

`request.footprint()` returns TradingView's per-bar volume footprint (buy/sell aggressor
split, delta, and the POC / VAH / VAL price rows). TradingView derives it from tick-level
bid/ask order-flow data; PyneCore only has OHLCV bars, so that data source is unavailable.

**Current status:** `request.footprint()` returns `na` rather than raising. Well-written
footprint scripts already guard every read with `na()` (TradingView itself returns `na`
when footprint data is unavailable), so they keep running and fall back to their
price-action paths instead of aborting the whole script.

```python
# The required Pine pattern — guard the footprint, degrade gracefully:
fp = request.footprint(50, 70)
buy = footprint.buy_volume(fp) if not na(fp) else 0.0  # na fp -> price-action fallback
```

## Other request.* Functions

The following functions are not implemented and are intended as plugin extension points:

| Function              | Data Source           | Status              |
|-----------------------|-----------------------|---------------------|
| `request.financial()` | FactSet (proprietary) | Plugin extension    |
| `request.economic()`  | Macro data            | Plugin extension    |
| `request.quandl()`    | Nasdaq Data Link      | Plugin extension    |
| `request.seed()`      | GitHub repositories   | Plugin extension    |
