<!--
---
weight: 470
title: "barmerge"
description: "Bar merge constants for request.security() gaps and lookahead"
icon: "merge"
date: "2026-03-28"
lastmod: "2026-08-04"
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
| `barmerge.lookahead_on`   | Uses the containing higher-timeframe bar as a developing, unconfirmed bar built from chart data up to the current bar — in historical and live mode alike. Never its completed final value (unlike TradingView). |
| `barmerge.lookahead_last_closed` | PyneSys-native explicit last-closed mode. It is equivalent to `lookahead_off` in PyneCore and remains repaint-free. |

---

## Compatibility

- `barmerge.gaps_off` and `barmerge.gaps_on` are supported in `request.security()`.
- `barmerge.lookahead_off` is the default closed-bar, repaint-free mode.
- `barmerge.lookahead_on` is supported. On same-symbol higher-timeframe requests it follows the
  containing bar, which is developing and unconfirmed in historical and live mode alike: a bare
  value reads the period as built so far, never the completed final value TradingView would
  return. The `close[1]` idiom is TV-exact and unaffected. Cross-symbol requests return `na`
  while their current higher-timeframe bar is still open.
- `barmerge.lookahead_last_closed` is a PyneSys-native alias for explicit closed-bar intent; use it
  when that intent should be clear without relying on the TradingView `close[1]` idiom.
