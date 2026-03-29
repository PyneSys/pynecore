<!--
---
weight: 432
title: "strategy.risk"
description: "Strategy risk management rules"
icon: "shield"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["strategy.risk", "library", "reference"]
---
-->

# strategy.risk

The `strategy.risk` namespace provides functions to define risk management rules for your trading strategy. These rules allow you to automatically close positions, cancel pending orders, and stop placing new orders when specific risk conditions are met.

## Quick Example

Here's a complete PyneCore script demonstrating the `strategy.risk` namespace:

```python
from pynecore.lib import (
    close, strategy, ta, script
)

@script.strategy(title="Risk Management Strategy", overlay=True)
def main():
    # Configure risk rules
    strategy.risk.max_position_size(10)
    strategy.risk.max_drawdown(10, strategy.percent_of_equity)
    strategy.risk.max_intraday_loss(5000, strategy.cash)
    strategy.risk.max_cons_loss_days(3)
    strategy.risk.max_intraday_filled_orders(5)
    strategy.risk.allow_entry_in(strategy.direction.long)
    
    # Trading logic
    sma20: float = ta.sma(close, 20)
    if close > sma20:
        strategy.entry("long", strategy.long, qty=1)
```

## Functions

### allow_entry_in()

Restricts strategy entries to only occur in a specified market direction.

| Parameter | Type | Description |
|-----------|------|-------------|
| `value` | `strategy.direction.Direction` | The allowed direction: `strategy.direction.long`, `strategy.direction.short`, or `strategy.direction.both` |

**Returns:** `None`

**Example:**
```python
strategy.risk.allow_entry_in(strategy.direction.long)  # Only allow long entries
```

### max_cons_loss_days()

Closes all positions and cancels pending orders after a specified number of consecutive losing days.

| Parameter | Type | Description |
|-----------|------|-------------|
| `count` | `int` | Maximum number of consecutive losing days allowed |
| `alert_message` | `str` *(optional)* | Alert message to display when the rule is triggered |

**Returns:** `None`

**Example:**
```python
strategy.risk.max_cons_loss_days(3)  # Stop trading after 3 consecutive losing days
```

### max_drawdown()

Closes all positions and cancels pending orders when the strategy reaches a maximum drawdown limit.

| Parameter | Type | Description |
|-----------|------|-------------|
| `value` | `float \| int` | Maximum drawdown threshold |
| `type` | `strategy.QtyType` *(optional)* | Unit type: `strategy.percent_of_equity` (default) or `strategy.cash` |
| `alert_message` | `str` *(optional)* | Alert message to display when the rule is triggered |

**Returns:** `None`

**Example:**
```python
strategy.risk.max_drawdown(20, strategy.percent_of_equity)  # Stop when 20% drawdown is reached
```

### max_intraday_filled_orders()

Limits the maximum number of orders that can be filled during a single day.

| Parameter | Type | Description |
|-----------|------|-------------|
| `count` | `int` | Maximum number of filled orders allowed per day |
| `alert_message` | `str` *(optional)* | Alert message to display when the rule is triggered |

**Returns:** `None`

**Example:**
```python
strategy.risk.max_intraday_filled_orders(5)  # Limit to 5 filled orders per day
```

### max_intraday_loss()

Closes all positions and cancels pending orders when intraday losses reach a specified limit.

| Parameter | Type | Description |
|-----------|------|-------------|
| `value` | `float \| int` | Maximum intraday loss threshold |
| `type` | `strategy.QtyType` *(optional)* | Unit type: `strategy.percent_of_equity` (default) or `strategy.cash` |
| `alert_message` | `str` *(optional)* | Alert message to display when the rule is triggered |

**Returns:** `None`

**Example:**
```python
strategy.risk.max_intraday_loss(5000, strategy.cash)  # Stop when $5000 intraday loss is reached
```

### max_position_size()

Limits the maximum size of any individual market position opened by the strategy.

| Parameter | Type | Description |
|-----------|------|-------------|
| `contracts` | `int \| float` | Maximum position size (contracts or shares) |

**Returns:** `None`

**Example:**
```python
strategy.risk.max_position_size(10)  # Limit positions to 10 contracts maximum
```

## Compatibility

All six functions in the `strategy.risk` namespace are fully implemented in PyneCore and work exactly as documented.