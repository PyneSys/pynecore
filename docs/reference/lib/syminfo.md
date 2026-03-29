<!--
---
weight: 440
title: "syminfo"
description: "Symbol information — ticker, exchange, currency, session, etc."
icon: "info"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["syminfo", "library", "reference"]
---
-->

# syminfo

The `syminfo` namespace provides information about the current trading symbol, including its name, exchange prefix, currency, trading characteristics, and analyst price targets.

## Quick Example

```python
from pynecore.lib import (
    syminfo, bar_index, label, close, script, time, string
)

@script.indicator(title="Symbol Info Display", overlay=True)
def main():
    # Build symbol identifier
    sym_id: str = f"{syminfo.prefix}:{syminfo.ticker}"
    currency: str = syminfo.currency or "NA"
    market_type: str = syminfo.type or "NA"
    
    # Display symbol details at first bar
    if bar_index == 0:
        info_text: str = f"{sym_id} | {market_type} | {currency}"
        label.new(bar_index, close, info_text)
```

## Variables

### Basic Symbol Information

#### prefix
Type: `str | NA[str]`

The exchange prefix of the symbol. For example, "NASDAQ" for "NASDAQ:AAPL" or "BINANCE" for "BINANCE:BTCUSDT".

```python
exch: str | NA[str] = syminfo.prefix  # "NASDAQ"
```

#### ticker
Type: `str | NA[str]`

The symbol name without the exchange prefix. For example, "AAPL" for "NASDAQ:AAPL" or "BTCUSDT" for "BINANCE:BTCUSDT".

```python
sym: str | NA[str] = syminfo.ticker  # "AAPL"
```

#### tickerid
Type: `str | NA[str]`

A full ticker identifier representing the symbol, including the exchange prefix and name separated by a colon.

```python
full_id: str | NA[str] = syminfo.tickerid  # "NASDAQ:AAPL"
```

#### root
Type: `str | NA[str]`

The root contract name for derivatives like futures. For non-derivative symbols, returns the same value as `ticker`.

```python
root_name: str | NA[str] = syminfo.root  # "AAPL" or "ES" for futures
```

### Currency and Location

#### currency
Type: `str | NA[str]`

The currency code of the symbol's prices. For example, "USD" for "NASDAQ:AAPL" or "USDT" for "BINANCE:BTCUSDT".

```python
ccy: str | NA[str] = syminfo.currency  # "USD"
```

#### basecurrency
Type: `str | NA[str]`

The base currency code for Forex or cryptocurrency pairs. For example, "EUR" for "FX:EURUSD" or "BTC" for "BINANCE:BTCUSDT".

```python
base: str | NA[str] = syminfo.basecurrency  # "EUR" or "BTC"
```

#### country
Type: `str | NA[str]`

The two-letter country code (ISO 3166-1 alpha-2 format) where the symbol is traded, or `na` if not available.

```python
nation: str | NA[str] = syminfo.country  # "US", "GB", etc.
```

#### timezone
Type: `str | NA[str]`

The timezone of the exchange where the symbol is traded. Uses IANA timezone identifiers (e.g., "America/New_York", "Europe/London").

```python
tz: str | NA[str] = syminfo.timezone  # "America/New_York"
```

### Market Characteristics

#### type
Type: `Literal['stock', 'future', 'option', 'forex', 'index', 'fund', 'bond', 'crypto'] | NA[str]`

The market type of the symbol. Possible values include "stock", "future", "option", "forex", "index", "fund", "bond", or "crypto".

```python
market: str = syminfo.type  # "stock", "future", etc.
```

#### session
Type: `Session`

The session type of the chart's main series. Possible values are `session.regular` or `session.extended`.

```python
sess: Session = syminfo.session  # session.regular
```

#### volumetype
Type: `Literal["base", "quote", "tick", "n/a"] | NA[str]`

The volume type of the symbol. Possible values are "base" (base currency), "quote" (quote currency), "tick" (number of transactions), or "n/a" if not applicable.

```python
vol_type: str = syminfo.volumetype  # "base", "quote", etc.
```

### Price Scaling

#### mintick
Type: `float | NA[float]`

The minimum tick value for the symbol — the smallest increment between price movements. For stocks, typically 0.01; for forex, often 0.0001.

```python
tick: float | NA[float] = syminfo.mintick  # 0.01
```

#### pricescale
Type: `int | NA[int]`

The denominator used to calculate the minimum tick. The minimum tick is calculated as `minmove / pricescale`.

```python
scale: int | NA[int] = syminfo.pricescale  # 100
```

#### minmove
Type: `int`

The numerator used to calculate the minimum tick. The minimum tick is calculated as `minmove / pricescale`. Default value is 1.

```python
move: int = syminfo.minmove  # 1
```

#### pointvalue
Type: `float | NA[float]`

The point value of the symbol — the currency amount per "point" of price movement. For stocks, typically 1.0; for futures, varies by contract.

```python
pv: float | NA[float] = syminfo.pointvalue  # 1.0
```

### Classification

#### description
Type: `str | NA[str]`

A descriptive text for the symbol provided by the exchange or data source.

```python
desc: str | NA[str] = syminfo.description  # "Apple Inc."
```

#### sector
Type: `str | NA[str]`

The sector classification for stocks, or `na` if not available. Examples: "Electronic Technology", "Technology Services", "Energy Minerals".

```python
sec: str | NA[str] = syminfo.sector  # "Electronic Technology"
```

#### industry
Type: `str | NA[str]`

The industry classification for stocks, or `na` if not available. Examples: "Internet Software/Services", "Integrated Oil", "Packaged Software".

```python
ind: str | NA[str] = syminfo.industry  # "Internet Software/Services"
```

#### period
Type: `str | NA[str]`

The period or resolution of the chart data. Examples: "1", "60" for minute bars; "D" for daily; "W" for weekly; "M" for monthly.

```python
p: str | NA[str] = syminfo.period  # "60", "D", "W"
```

### Analyst Price Targets

#### target_price_average
Type: `float | NA[float]`

The average price target for the symbol predicted by analysts.

```python
avg_target: float | NA[float] = syminfo.target_price_average  # 150.0
```

#### target_price_high
Type: `float | NA[float]`

The highest price target for the symbol predicted by analysts.

```python
high_target: float | NA[float] = syminfo.target_price_high  # 175.0
```

#### target_price_low
Type: `float | NA[float]`

The lowest price target for the symbol predicted by analysts.

```python
low_target: float | NA[float] = syminfo.target_price_low  # 125.0
```

#### target_price_date
Type: `int | NA[int]`

A UNIX timestamp representing the date of the last analyst price target prediction for the symbol.

```python
target_date: int | NA[int] = syminfo.target_price_date  # 1704067200
```

## Compatibility Notes

The following `syminfo` fields are **not available** in PyneCore:

- **current_contract**: The ticker identifier of the underlying contract for continuous futures
- **employees**: Number of employees
- **expiration_date**: UNIX timestamp for the last trading day of a futures contract
- **main_tickerid**: Alternate full identifier for the chart symbol
- **mincontract**: Smallest amount of the symbol that can be traded
- **recommendations_buy**, **recommendations_buy_strong**, **recommendations_hold**, **recommendations_sell**, **recommendations_sell_strong**: Analyst recommendation counts by rating
- **recommendations_total**, **recommendations_date**: Total recommendation count and recommendation date
- **shareholders**: Number of shareholders
- **shares_outstanding_float**, **shares_outstanding_total**: Outstanding share counts
- **target_price_estimates**, **target_price_median**: Median price target and estimate count

These fields depend on fundamental and analyst data that is not available in historical OHLCV data sources used by PyneCore.