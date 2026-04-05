<!--
---
weight: 470
title: "barmerge"
description: "Bar merge constants for request.security() gaps and lookahead"
icon: "merge"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["barmerge", "library", "reference"]
---
-->

# barmerge

The `barmerge` namespace provides constants that control how data requested via `request.security()` is merged with the current chart's bar data. Two independent concerns are covered: gap-filling behavior (`gaps_off` / `gaps_on`) and bar alignment by open vs. close time (`lookahead_off` / `lookahead_on`).

## Quick Example

```python
from pynecore.lib import close, high, low, open, bar_index, script, request, barmerge

@script.indicator(title="HTF Close", overlay=True)
def main():
    htf_close: float = request.security(
        "NASDAQ:AAPL",
        "D",
        close,
        gaps=barmerge.gaps_off,
        lookahead=barmerge.lookahead_off
    )
```

---

## Constants

| Constant                  | Description                                                                                              |
|---------------------------|----------------------------------------------------------------------------------------------------------|
| `barmerge.gaps_off`       | Continuous merge — gaps are filled with the most recent available value. No `na` values are introduced. |
| `barmerge.gaps_on`        | Merge with gaps — missing bars are left as `na`.                                                         |
| `barmerge.lookahead_off`  | Bars are aligned by **close time**. The requested value becomes available only after the bar closes.     |
| `barmerge.lookahead_on`   | Bars are aligned by **open time**. The requested value is available at the start of the current bar.    |

---

## Compatibility

- `barmerge.gaps_off` and `barmerge.gaps_on` are fully supported in `request.security()`.
- `barmerge.lookahead_off` is the only supported lookahead mode. `barmerge.lookahead_on` is defined as a constant but **not supported at runtime** — PyneCore deliberately disables lookahead to prevent future-leak in backtests.