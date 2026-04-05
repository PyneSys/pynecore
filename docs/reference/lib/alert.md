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

- The `freq` parameter is accepted for Pine Script compatibility but is **not enforced** — all `alert()` calls produce output regardless of the frequency constant used.
- There is no support for alert conditions or webhook-based alerts. `alert()` is terminal output only.