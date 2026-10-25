<!--
---
weight: 473
title: "splits"
description: "Stock split data field constants"
icon: "call_split"
date: "2026-08-04"
lastmod: "2026-10-25"
draft: false
toc: true
categories: ["Reference"]
tags: ["splits", "request", "corporate-events"]
---
-->

# splits

The `splits` namespace identifies the value returned by `request.splits()`.

| Constant | Description |
|----------|-------------|
| `splits.numerator` | The numerator of a stock split ratio |
| `splits.denominator` | The denominator of a stock split ratio |

```python
from pynecore.lib import request, splits

numerator = request.splits("NASDAQ:AAPL", splits.numerator, ignore_invalid_symbol=True)
denominator = request.splits("NASDAQ:AAPL", splits.denominator, ignore_invalid_symbol=True)
```

## Compatibility

The constants are available for Pine-compatible source code. PyneCore does not currently provide
split data: `request.splits()` returns `na` with `ignore_invalid_symbol=True` or when it requests
the current crypto chart's own ticker, and otherwise raises `NotImplementedError`.
