<!--
---
weight: 460
title: "alert"
description: "Alert triggering functions"
icon: "notifications"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["alert", "library", "reference"]
---
-->

# alert

The `alert` namespace provides a callable function for triggering alerts during script execution. In PyneCore, alerts print a highlighted message to the terminal rather than sending notifications — the `freq` parameter is accepted for Pine Script compatibility but has no effect at runtime.

## Quick Example

```python
from pynecore.lib import script, alert, ta, close

@script.indicator(title="Alert Demo", overlay=True)
def main():
    rsi = ta.rsi(close, 14)
    if rsi > 70:
        alert("RSI overbought!", freq=alert.freq_once_per_bar)
```

---

## Functions

### alert()

Prints an alert message to the terminal. Outputs with color and formatting if `typer` is installed; falls back to a plain `print` otherwise.

| Parameter | Type        | Description                                                     |
|-----------|-------------|------------------------------------------------------------------|
| `message` | `str`       | The alert message to display.                                    |
| `freq`    | `AlertEnum` | Alert frequency. Optional, defaults to `alert.freq_once_per_bar`. Currently ignored. |

**Returns:** `None`

```python
alert("Price crossed above SMA!", freq=alert.freq_once_per_bar)
```

---

## Constants

| Constant                       | Description                                                                    |
|-------------------------------|--------------------------------------------------------------------------------|
| `alert.freq_all`              | Every call to `alert()` triggers the alert.                                   |
| `alert.freq_once_per_bar`     | Only the first `alert()` call during a bar triggers the alert. *(default)*    |
| `alert.freq_once_per_bar_close` | Triggers only when `alert()` is called on the bar's closing execution.      |

---

## Compatibility Notes

- Calls made while an imported library's `main()` runs, including its helper calls, are suppressed.
  Calls from the importing script into an exported library function can produce output normally.
- Calls in `request.security` workers are suppressed to avoid duplicate output when they replay the script.
- The `freq` parameter is accepted for Pine Script compatibility but is **not enforced**.
  Outside the suppressed contexts, calls print on historical as well as realtime bars.
- There is no support for alert conditions or webhook-based alerts. `alert()` is terminal output only.
  A library run directly as the main script can also print these messages. TradingView instead
  supports actual alerts only on realtime bars and does not allow creating an alert directly
  from a library; an indicator or strategy must consume its exported function.
  See [TradingView's library constraints](https://www.tradingview.com/pine-script-docs/concepts/libraries/#library-functions).
