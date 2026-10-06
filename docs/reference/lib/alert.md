<!--
---
weight: 460
title: "alert"
description: "Alert triggering functions"
icon: "notifications"
date: "2026-03-28"
lastmod: "2026-10-06"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["alert", "library", "reference"]
---
-->

# alert

The `alert` namespace provides a callable function for triggering alerts during script execution. In PyneCore, `alert()` prints a highlighted message to the terminal, so converted Pine scripts that call it run unchanged. The `freq` parameter is accepted for Pine Script compatibility but has no effect at runtime.

On TradingView, an alert is the only way a Pine script can reach the outside world: the script cannot send orders to a broker or make network calls, so trading bots receive its alerts through a TradingView webhook. A PyneCore script needs no such detour. It trades directly through a broker plugin (`pyne run --broker`, see [Live Mode](../../advanced/live-mode.md)), and as Python code it can send any notification itself — a Telegram or Discord message, an email, an HTTP request — with the library of your choice.

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
- `alertcondition()` is accepted and does nothing, and TradingView's webhook delivery has no
  counterpart: a script that needs to notify an external service calls it directly from Python.
  Guard such calls with `barstate.isrealtime` in live mode, since the script also runs on the
  historical bars.
- A library run directly as the main script can also print `alert()` messages. TradingView instead
  supports actual alerts only on realtime bars and does not allow creating an alert directly
  from a library; an indicator or strategy must consume its exported function.
  See [TradingView's library constraints](https://www.tradingview.com/pine-script-docs/concepts/libraries/#library-functions).
