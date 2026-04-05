<!--
---
weight: 484
title: "adjustment"
description: "Dividend adjustment constants"
icon: "tune"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["adjustment", "library", "reference"]
---
-->

# adjustment

The `adjustment` namespace provides constants that specify how historical price data is adjusted for corporate actions. These constants are used with `request.security()` and similar data-fetching functions to control whether dividend or split adjustments are applied to OHLCV values.

## Quick Example

```python
from pynecore.lib import script, request, close, adjustment

@script.indicator(title="Adjusted vs Raw Close", overlay=False)
def main():
    adj_close: float = request.security("NASDAQ:AAPL", "D", close, adjustment=adjustment.dividends)
    raw_close: float = request.security("NASDAQ:AAPL", "D", close, adjustment=adjustment.none)
```

---

## Constants

| Constant               | Value | Description                                              |
|------------------------|-------|----------------------------------------------------------|
| `adjustment.none`      | `0`   | No adjustment applied — raw historical prices.           |
| `adjustment.dividends` | `1`   | Dividend adjustment applied to historical prices.        |
| `adjustment.splits`    | `2`   | Split adjustment applied to historical prices.           |