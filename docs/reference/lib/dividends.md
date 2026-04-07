<!--
---
weight: 485
title: "dividends"
description: "Dividend data field constants"
icon: "account_balance"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["dividends", "library", "reference"]
---
-->

# dividends

The `dividends` namespace provides access to dividend-related data for the current instrument. It exposes three read-only variables for upcoming dividend payment details and two constants used with `request.dividends` to specify gross or net return types.

## Quick Example

```python
from pynecore.lib import script, dividends, label, bar_index, high, barstate, na

@script.indicator(title="Dividend Info", overlay=True)
def main():
    if not na(dividends.future_ex_date) and barstate.islast:
        msg: str = f"Next ex-date: {dividends.future_ex_date}, Amount: {dividends.future_amount}"
        label.new(bar_index, high, msg)
```

## Variables

### dividends.future_amount

The payment amount of the upcoming dividend in the instrument's currency. Returns `na` if the data is unavailable.

| Property | Detail      |
|----------|-------------|
| Type     | `float`     |
| Returns  | `na` if unavailable |

### dividends.future_ex_date

The ex-dividend date of the instrument's next dividend payment, expressed as a UNIX timestamp in milliseconds. Returns `na` if the data is unavailable.

The ex-dividend date is the cutoff date — you must own the stock before this date to receive the dividend.

| Property | Detail                          |
|----------|---------------------------------|
| Type     | `float`                         |
| Returns  | UNIX timestamp (ms), or `na`    |

### dividends.future_pay_date

The payment date of the instrument's next dividend, expressed as a UNIX timestamp in milliseconds. Returns `na` if the data is unavailable.

The pay date is when the dividend is actually distributed to eligible shareholders.

| Property | Detail                          |
|----------|---------------------------------|
| Type     | `float`                         |
| Returns  | UNIX timestamp (ms), or `na`    |

## Constants

These constants are passed to `request.dividends` to specify which type of dividend return to retrieve.

| Constant            | Description                                              |
|---------------------|----------------------------------------------------------|
| `dividends.gross`   | Request dividend return before tax deductions            |
| `dividends.net`     | Request dividend return after tax deductions             |

**Example:**

```python
from pynecore.lib import script, dividends, request, close

@script.indicator(title="Dividend Yield", overlay=False)
def main():
    div: float = request.dividends("AAPL", dividends.gross)
```