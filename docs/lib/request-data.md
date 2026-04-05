<!--
---
weight: 507
title: "request.currency_rate() & Corporate Events"
description: "Currency conversion rates and corporate event data (dividends, splits, earnings) in PyneCore"
icon: "currency_exchange"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Library", "API Reference"]
tags: ["request", "currency", "dividends", "splits", "earnings"]
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

1. Scans all TOML files in `security_data` for `basecurrency = "EUR"` and `currency = "USD"`
2. Lazy-loads the matching OHLCV file on first access
3. Uses binary search to find the close price at or before the current bar's timestamp
4. Returns the close price as the exchange rate

If the chart itself is a currency pair (e.g., EURUSD chart), it automatically serves as a rate
source — no separate OHLCV file needed. PyneCore reads `lib.close` directly.

### Special Cases

| Condition                  | Result             |
|----------------------------|--------------------|
| `from == to`               | `1.0`              |
| `from` or `to` is `NONE`  | `na`               |
| No matching data           | `na`               |
| Inverse pair available     | `1.0 / close`      |
| Chart is the currency pair | `lib.close` direct |

### Data Preparation

Provide FX or crypto OHLCV data via the `security_data` parameter. The key names don't matter —
currency pairs are detected from TOML metadata:

```python
from pathlib import Path
from pynecore.core.script_runner import ScriptRunner
from pynecore.core.ohlcv_file import OHLCVReader
from pynecore.core.syminfo import SymInfo

syminfo = SymInfo.load_toml("workdir/data/BTCUSDT_4h.toml")
reader = OHLCVReader("workdir/data/BTCUSDT_4h")
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
handle `ignore_invalid_symbol=True` gracefully by returning `na` instead of raising an error.
This allows scripts like the TradingView VWAP indicator (which uses all three) to run without
modification.

```python
# This works — returns na silently, script continues
new_earnings = request.earnings(
    syminfo.tickerid, earnings.actual,
    barmerge.gaps_on, barmerge.lookahead_on,
    ignore_invalid_symbol=True  # Required for graceful fallback
)

# This still raises NotImplementedError
new_earnings = request.earnings(syminfo.tickerid, earnings.actual)
```

### Planned Implementation

Full corporate event support is planned using the existing `.extra.csv` sidecar mechanism, where
event data is provided as named columns alongside OHLCV bars.

## Other request.* Functions

The following functions are not implemented and are intended as plugin extension points:

| Function              | Data Source           | Status              |
|-----------------------|-----------------------|---------------------|
| `request.financial()` | FactSet (proprietary) | Plugin extension    |
| `request.economic()`  | Macro data            | Plugin extension    |
| `request.quandl()`    | Nasdaq Data Link      | Plugin extension    |
| `request.seed()`      | GitHub repositories   | Plugin extension    |
| `request.footprint()` | Tick-level volume     | Plugin extension    |
