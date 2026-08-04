<!--
---
weight: 442
title: "ticker"
description: "Ticker identifiers and supported synthetic chart types"
icon: "sell"
date: "2026-08-04"
lastmod: "2026-08-04"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["ticker", "request.security", "heikinashi", "library", "reference"]
---
-->

# ticker

The `ticker` namespace creates identifiers for use with `request.security()`. Standard ticker
identifiers use `EXCHANGE:SYMBOL` form. PyneCore supports Heikin Ashi requests; other synthetic
chart types require tick-derived data and are not available.

## Quick Example

```python
from pynecore.lib import close, request, script, ticker

@script.indicator("Heikin Ashi close")
def main():
    ha_symbol = ticker.heikinashi("BINANCE:BTCUSDT")
    ha_close = request.security(ha_symbol, "60", close)
```

## Identifier Functions

### new()

Creates an exchange-qualified identifier.

```python
symbol = ticker.new("BINANCE", "BTCUSDT")  # "BINANCE:BTCUSDT"
```

If `ticker` already contains `:`, it is returned unchanged. The optional `session`, `adjustment`,
`backadjustment`, and `settlement_as_close` arguments are accepted for Pine compatibility but do
not alter PyneCore identifiers.

### modify()

Returns the supplied identifier. Its optional session and adjustment arguments are accepted for
Pine compatibility, but PyneCore data feeds do not expose per-request variants.

### standard()

Returns the standard-chart identifier. With no argument it returns `syminfo.tickerid`; otherwise
it returns the provided identifier.

### inherit()

Builds an identifier for `symbol` using the exchange prefix from `from_tickerid` when `symbol` is
not already qualified.

```python
eth = ticker.inherit("BINANCE:BTCUSDT", "ETHUSDT")  # "BINANCE:ETHUSDT"
```

## Chart Types

### heikinashi()

Creates an identifier that `request.security()` evaluates as Heikin Ashi bars. The transform runs
in the security context, after higher-timeframe aggregation. It works in historical and live mode.
`request.security_lower_tf()` does not support synthetic chart types.

### Unsupported synthetic chart types

`ticker.renko()`, `ticker.pointfigure()`, `ticker.kagi()`, and `ticker.linebreak()` raise
`NotImplementedError`. These chart types require tick data that standard OHLCV inputs do not
contain.

## Compatibility

`new()`, `modify()`, `standard()`, `inherit()`, and `heikinashi()` are supported. Session and
adjustment modifiers are intentionally no-ops; PyneCore has one data variant per supplied symbol
and timeframe.
