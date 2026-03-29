<!--
---
weight: 428
title: "request"
description: "Data requests from other symbols and timeframes"
icon: "cloud_download"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["request", "library", "reference"]
---
-->

# request

Request data from other symbols, timeframes, and economic sources. The most commonly used function is `request.security()`, which allows you to evaluate expressions from different symbols and timeframes. PyneCore also provides currency conversion rates and stubs for dividend/earnings data.

## Quick Example

```python
from pynecore.lib import (
    close, high, low, open, bar_index, ta, script, request, label
)

@script.indicator(title="Multi-Symbol SMA", overlay=True)
def main():
    # Get 20-bar SMA from a different symbol at 1-hour timeframe
    btc_sma: float = request.security("BTCUSD", "60", ta.sma(close, 20))
    
    # Convert a value from EUR to USD
    rate: float = request.currency_rate("EUR", "USD")
    converted: float = close * rate
    
    # Compare current symbol's close with Bitcoin
    if close > btc_sma:
        label.new(bar_index, high, "Price above BTC SMA")
```

## Functions

### security()

Evaluates an expression from a specified symbol and timeframe.

| Parameter | Type | Description |
|-----------|------|-------------|
| symbol | str | Symbol to request (e.g., "BTCUSD", "SPY") |
| timeframe | str | Timeframe as string (e.g., "60", "D", "W") |
| expression | any | Expression to evaluate in the target context |
| gaps | barmerge | Gap handling mode (only `barmerge.gaps_off` supported) |
| lookahead | barmerge | Lookahead mode (only `barmerge.lookahead_off` supported) |
| ignore_invalid_symbol | bool | Return `na` for invalid symbols instead of raising an error |
| currency | str | Currency for conversion (not yet used) |
| calc_bars_count | int | Number of bars to calculate (not yet used) |

**Returns:** The result of the expression evaluated in the target context. Type matches the expression type.

**Example:**
```python
sma_value: float = request.security("EURUSD", "D", ta.sma(close, 50))  # Daily 50-bar SMA
upper_band: float = request.security("SPY", "240", ta.highest(high, 14))  # Highest of last 14 bars
```

**Note:** Fully implemented via the `SecurityTransformer`, which rewrites all calls into a multiprocessing protocol at compile time. The function itself is never called at runtime. Only `barmerge.lookahead_off` is supported (intentional safety constraint). Conditional calls and nested security requests are supported.

### security_lower_tf()

Requests intrabar values from a lower timeframe, returning an array of values per chart bar.

| Parameter | Type | Description |
|-----------|------|-------------|
| symbol | str | Symbol to request |
| timeframe | str | Lower timeframe (must be ≤ chart timeframe) |
| expression | any | Expression to evaluate per intrabar |
| ignore_invalid_symbol | bool | Return empty array for invalid symbols |
| currency | str | Currency for conversion (not yet used) |
| ignore_invalid_timeframe | bool | Ignore invalid timeframe errors |
| calc_bars_count | int | Number of bars to calculate (not yet used) |

**Returns:** Array of values, one per intrabar within each chart bar. Empty array if no data.

**Example:**
```python
ltf_closes: list[float] = request.security_lower_tf("EURUSD", "5", close)  # All 5-min closes per chart bar
ltf_volumes: list[float] = request.security_lower_tf("SPY", "15", volume)  # All 15-min volumes
```

**Note:** Fully implemented with multiprocessing support. Returns an array of intrabar values per chart bar. If the chart timeframe is lower than the requested timeframe, returns empty arrays.

### currency_rate()

Gets the exchange rate between two currencies at the current bar's timestamp.

| Parameter | Type | Description |
|-----------|------|-------------|
| from_currency | str | Source currency code (e.g., "EUR", "GBP") |
| to_currency | str | Target currency code (e.g., "USD") |

**Returns:** Exchange rate as float, or `na` if no data is available.

**Example:**
```python
eur_to_usd: float = request.currency_rate("EUR", "USD")  # 1.095
gbp_to_eur: float = request.currency_rate("GBP", "EUR")  # 1.168
```

**Note:** Looks up rates from OHLCV data whose metadata matches the requested currency pair. Automatically uses inverse pairs (1.0 / rate) if only the reverse pair is available.

### dividends()

Requests dividend data for a symbol.

| Parameter | Type | Description |
|-----------|------|-------------|
| ticker | str | Symbol ticker |
| field | str | Dividend field (not yet supported) |
| gaps | barmerge | Gap handling mode |
| lookahead | barmerge | Lookahead mode |
| ignore_invalid_symbol | bool | Return `na` instead of raising error |

**Returns:** Dividend value or `na`.

**Note:** Returns `na` when `ignore_invalid_symbol=True`. Otherwise raises `NotImplementedError`. No actual dividend data support yet. Used by indicators that reference `request.dividends()` but do not require real data (e.g., VWAP).

### splits()

Requests stock split data for a symbol.

| Parameter | Type | Description |
|-----------|------|-------------|
| ticker | str | Symbol ticker |
| field | str | Split field (numerator, denominator) |
| gaps | barmerge | Gap handling mode |
| lookahead | barmerge | Lookahead mode |
| ignore_invalid_symbol | bool | Return `na` instead of raising error |

**Returns:** Split value or `na`.

**Note:** Returns `na` when `ignore_invalid_symbol=True`. Otherwise raises `NotImplementedError`. No actual split data support yet.

### earnings()

Requests earnings data for a symbol.

| Parameter | Type | Description |
|-----------|------|-------------|
| ticker | str | Symbol ticker |
| field | str | Earnings field (actual, estimate, standardized) |
| gaps | barmerge | Gap handling mode |
| lookahead | barmerge | Lookahead mode |
| ignore_invalid_symbol | bool | Return `na` instead of raising error |

**Returns:** Earnings value or `na`.

**Note:** Returns `na` when `ignore_invalid_symbol=True`. Otherwise raises `NotImplementedError`. No actual earnings data support yet.

### financial()

Requests financial data from FactSet.

**Returns:** Financial value as float.

**Note:** Not yet implemented in PyneCore. Requires FactSet data feed (TradingView-only feature).

### economic()

Requests economic data such as GDP, inflation rate, or employment statistics.

**Returns:** Economic indicator value as float.

**Note:** Not yet implemented in PyneCore. Requires TradingView economic data feed.

### quandl()

Requests data from Nasdaq Data Link (formerly Quandl).

**Note:** Deprecated. Nasdaq Data Link no longer accepts QUANDL requests. This function is not supported.

### seed()

Requests data from user-maintained GitHub repositories (Pine Seeds).

**Note:** Not supported. Pine Seeds feature was discontinued by TradingView.

### footprint()

Requests volume footprint data for the current bar.

| Parameter | Type | Description |
|-----------|------|-------------|
| ticks_per_row | int | Number of ticks per footprint row |
| va_percent | int | Value Area percentage |

**Returns:** Footprint object with volume data.

**Note:** Not yet implemented in PyneCore. Raises `NotImplementedError`.

## Compatibility Notes

- **Fully implemented**: `request.security()`, `request.security_lower_tf()`, `request.currency_rate()`
- **Partial support**: `request.dividends()`, `request.earnings()`, `request.splits()` — return `na` when `ignore_invalid_symbol=True`, raise `NotImplementedError` otherwise
- **Not available**: `request.economic()`, `request.financial()`, `request.footprint()`, `request.quandl()`, `request.seed()`
- **Safety constraint**: Only `barmerge.lookahead_off` is supported for `request.security()` — lookahead is deliberately disabled for production safety
- **Data sources**: `request.security()` and `request.security_lower_tf()` require separate OHLCV data files per symbol/timeframe. `request.currency_rate()` uses OHLCV metadata to auto-detect currency pairs.