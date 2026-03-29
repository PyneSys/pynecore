<!--
---
weight: 431
title: "strategy.opentrades"
description: "Information about open trades in a strategy"
icon: "assignment"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["strategy.opentrades", "library", "reference"]
---
-->

# strategy.opentrades

Access information about open trades in a strategy. This namespace provides functions to query entry details, profit/loss metrics, drawdown and runup statistics, and commission information for each open trade by its trade number (zero-indexed).

## Quick Example

```python
from pynecore.lib import close, strategy, label, bar_index, script

@script.strategy(title="Monitor Open Trades")
def main():
    # Enter a trade on every close above 100
    if close > 100:
        strategy.entry("long", strategy.long, qty=1)
    
    # Monitor first open trade
    if strategy.opentrades.size(0) != 0:
        entry_price: float = strategy.opentrades.entry_price(0)
        current_profit: float = strategy.opentrades.profit(0)
        profit_pct: float = strategy.opentrades.profit_percent(0)
        max_dd: float = strategy.opentrades.max_drawdown(0)
        
        # Label showing trade metrics
        label.new(bar_index, close, 
                  f"P&L: {current_profit:.2f} ({profit_pct:.1f}%) | DD: {max_dd:.2f}")
        
        # Exit if drawdown exceeds 5%
        if strategy.opentrades.max_drawdown_percent(0) < -5.0:
            strategy.close("long")
```

## Functions

### commission(trade_num)

Returns the sum of entry and exit fees paid in the open trade, expressed in the strategy's account currency.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int | Trade number (zero-indexed) |

**Returns:** `float` — Total fees, or 0.0 if trade doesn't exist

```python
fees: float = strategy.opentrades.commission(0)  # 12.50
```

### entry_price(trade_num)

Returns the entry price of the open trade.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int | Trade number (zero-indexed) |

**Returns:** `float` — Entry price, or NA if trade doesn't exist

```python
price: float = strategy.opentrades.entry_price(0)  # 150.25
```

### entry_bar_index(trade_num)

Returns the bar index where the trade was entered.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int | Trade number (zero-indexed) |

**Returns:** `int` — Bar index of entry, or NA if trade doesn't exist

```python
bar: int = strategy.opentrades.entry_bar_index(0)  # 42
```

### entry_time(trade_num)

Returns the UNIX timestamp (in milliseconds) of the trade entry.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int | Trade number (zero-indexed) |

**Returns:** `int` — UNIX time in milliseconds, or NA if trade doesn't exist

```python
entry_ts: int = strategy.opentrades.entry_time(0)  # 1674123456000
```

### entry_id(trade_num)

Returns the entry ID string of the open trade.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int | Trade number (zero-indexed) |

**Returns:** `str` — Entry ID, or NA if trade doesn't exist

```python
tid: str = strategy.opentrades.entry_id(0)  # "long"
```

### entry_comment(trade_num)

Returns the comment message associated with the trade entry.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int | Trade number (zero-indexed) |

**Returns:** `str` — Entry comment, or NA if trade doesn't exist

```python
comment: str = strategy.opentrades.entry_comment(0)  # "Breakout signal"
```

### size(trade_num)

Returns the position size and direction of the open trade. Positive values indicate long positions, negative values indicate short positions.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int | Trade number (zero-indexed) |

**Returns:** `float` — Position size (positive for long, negative for short), or 0 if trade doesn't exist

```python
pos_size: float = strategy.opentrades.size(0)  # 2.5 (long) or -1.0 (short)
```

### profit(trade_num)

Returns the unrealized profit or loss of the open trade in the strategy's account currency. Losses are negative values.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int | Trade number (zero-indexed) |

**Returns:** `float` — Profit/loss, or 0 if trade doesn't exist

```python
pnl: float = strategy.opentrades.profit(0)  # 125.50 or -50.25
```

### profit_percent(trade_num)

Returns the unrealized profit or loss of the open trade as a percentage. Losses are negative values.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int | Trade number (zero-indexed) |

**Returns:** `float` — Profit/loss percentage, or 0 if trade doesn't exist

```python
pnl_pct: float = strategy.opentrades.profit_percent(0)  # 5.25 or -2.10
```

### max_runup(trade_num)

Returns the maximum unrealized profit during the open trade, expressed in the strategy's account currency.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int | Trade number (zero-indexed) |

**Returns:** `float` — Maximum runup, or 0 if trade doesn't exist

```python
max_profit: float = strategy.opentrades.max_runup(0)  # 250.75
```

### max_runup_percent(trade_num)

Returns the maximum unrealized profit during the open trade as a percentage of entry value.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int | Trade number (zero-indexed) |

**Returns:** `float` — Maximum runup percentage, or 0 if trade doesn't exist

```python
max_profit_pct: float = strategy.opentrades.max_runup_percent(0)  # 8.5
```

### max_drawdown(trade_num)

Returns the maximum unrealized loss during the open trade, expressed in the strategy's account currency.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int | Trade number (zero-indexed) |

**Returns:** `float` — Maximum drawdown, or 0 if trade doesn't exist

```python
max_loss: float = strategy.opentrades.max_drawdown(0)  # -75.50
```

### max_drawdown_percent(trade_num)

Returns the maximum unrealized loss during the open trade as a percentage of entry value.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int | Trade number (zero-indexed) |

**Returns:** `float` — Maximum drawdown percentage, or 0 if trade doesn't exist

```python
max_loss_pct: float = strategy.opentrades.max_drawdown_percent(0)  # -3.25
```

## Compatibility

**Not available in PyneCore:**
- `strategy.opentrades.capital_held` — Not yet implemented