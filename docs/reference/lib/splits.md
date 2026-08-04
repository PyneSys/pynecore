<!--
---
weight: 473
title: "splits"
description: "Stock split data field constants"
icon: "call_split"
date: "2026-08-04"
lastmod: "2026-08-04"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["splits", "request", "library", "reference"]
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
split data: `request.splits()` returns `na` with `ignore_invalid_symbol=True` and otherwise raises
`NotImplementedError`.
