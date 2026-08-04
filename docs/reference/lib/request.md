<!--
---
weight: 428
title: "request"
description: "Data requests from other symbols and timeframes"
icon: "cloud_download"
date: "2026-03-28"
lastmod: "2026-08-04"
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
| gaps | barmerge | Gap handling mode (`barmerge.gaps_off` or `barmerge.gaps_on`) |
| lookahead | barmerge | Alignment mode: `barmerge.lookahead_off`, `barmerge.lookahead_on`, or `barmerge.lookahead_last_closed` |
| ignore_invalid_symbol | bool | Return `na` for invalid symbols instead of raising an error |
| currency | str | Target currency — auto-converts result using `CurrencyRateProvider` |
| calc_bars_count | int | Number of bars to calculate (not yet used) |

**Returns:** The result of the expression evaluated in the target context. Type matches the expression type.

**Example:**
```python
sma_value: float = request.security("EURUSD", "D", ta.sma(close, 50))  # Daily 50-bar SMA
upper_band: float = request.security("SPY", "240", ta.highest(high, 14))  # Highest of last 14 bars
```

**Lookahead behavior:** `barmerge.lookahead_off` (the default) and the PyneSys-native
`barmerge.lookahead_last_closed` return closed security bars and are repaint-free. With
`barmerge.lookahead_on`, same-symbol higher-timeframe requests use the containing higher-timeframe
bar. In historical data, a bare value can therefore expose that completed bar's final value; use
`close[1]` inside the expression when the intent is the prior closed bar. In live mode, the
same-symbol higher-timeframe bar is aggregated from the chart data and is unconfirmed. A
cross-symbol higher-timeframe request cannot build that developing bar, so it yields `na` until the
period closes.

**Implementation:** `SecurityTransformer` rewrites calls into the security execution protocol at
compile time; the Python stub itself is not called by a transformed script. Conditional calls and
nested security requests are supported.

### security_lower_tf()

Requests intrabar values from a lower timeframe, returning an array of values per chart bar.

| Parameter | Type | Description |
|-----------|------|-------------|
| symbol | str | Symbol to request |
| timeframe | str | Lower timeframe (must be ≤ chart timeframe) |
| expression | any | Expression to evaluate per intrabar |
| ignore_invalid_symbol | bool | Return empty array for invalid symbols |
| currency | str | Target currency — auto-converts result using `CurrencyRateProvider` |
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

**Note:** Not implemented. The function raises `NotImplementedError`.

### seed()

Requests data from user-maintained GitHub repositories (Pine Seeds).

**Note:** Seed repositories are unavailable to PyneCore. The function returns `na`, allowing scripts
that guard the result with `na()` to take their fallback path.

### footprint()

Requests volume footprint data for the current bar.

| Parameter | Type | Description |
|-----------|------|-------------|
| ticks_per_row | int | Number of ticks per footprint row |
| va_percent | int | Value Area percentage |

**Returns:** `Footprint | na`.

**Note:** Footprint data requires Level 2 / tick-by-tick market data, which standard OHLCV files do
not contain. PyneCore returns `na` so scripts can use their normal unavailable-data fallback.

## Compatibility Notes

- **Implemented**: `request.security()`, `request.security_lower_tf()`, `request.currency_rate()`
- **Partial support**: `request.dividends()`, `request.earnings()`, `request.splits()` — return `na` when `ignore_invalid_symbol=True`, raise `NotImplementedError` otherwise
- **Unavailable data fallbacks**: `request.seed()` and `request.footprint()` return `na`.
  `request.financial()` returns `na` only when `ignore_invalid_symbol=True`; otherwise it, along
  with `request.economic()` and `request.quandl()`, raises `NotImplementedError`.
- **Gap handling**: Both `barmerge.gaps_off` (forward-fill, default) and `barmerge.gaps_on` (return `na` between periods) are supported
- **Currency conversion**: The `currency` parameter auto-converts results using `CurrencyRateProvider` when OHLCV metadata for the currency pair is available
- **Lookahead modes**: `barmerge.lookahead_off`, `barmerge.lookahead_on`, and
  `barmerge.lookahead_last_closed` are supported; their differing historical, live, and
  cross-symbol behavior is described above.
- **Data sources**: `request.security()` and `request.security_lower_tf()` require separate OHLCV data files per symbol/timeframe. `request.currency_rate()` uses OHLCV metadata to auto-detect currency pairs.
