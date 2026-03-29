<!--
---
weight: 430
title: "strategy.closedtrades"
description: "Information about closed trades in a strategy"
icon: "assignment_turned_in"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["strategy.closedtrades", "library", "reference"]
---
-->

# strategy.closedtrades

Access detailed information about closed trades in a strategy. Each closed trade stores entry/exit details, timing information, commissions, and profit metrics. Trades are numbered starting from zero.

## Quick Example

```python
from pynecore.lib import close, strategy, bar_index, script

@script.strategy(title="Trade Analysis")
def main():
    # Entry logic
    if bar_index == 10:
        strategy.entry("buy", strategy.long, 1.0)
    
    # Exit logic
    if bar_index == 20:
        strategy.close("buy")
    
    # Access the first closed trade (if it exists)
    if strategy.closedtrades.size(0) != 0:
        entry_px: float = strategy.closedtrades.entry_price(0)  # Entry price
        exit_px: float = strategy.closedtrades.exit_price(0)    # Exit price
        profit: float = strategy.closedtrades.profit(0)         # P/L in currency
        pct: float = strategy.closedtrades.profit_percent(0)    # P/L as %
```

## Entry Information

### entry_bar_index()

Returns the bar index at which the trade was entered.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int  | Trade number (first trade is 0) |

**Returns:** int

```python
idx: int = strategy.closedtrades.entry_bar_index(0)  # 10
```

### entry_time()

Returns the UNIX time (in milliseconds) when the trade was entered.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int  | Trade number (first trade is 0) |

**Returns:** int

```python
ts: int = strategy.closedtrades.entry_time(0)  # 1672531200000
```

### entry_price()

Returns the price at which the trade was entered.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int  | Trade number (first trade is 0) |

**Returns:** float

```python
px: float = strategy.closedtrades.entry_price(0)  # 100.5
```

### entry_id()

Returns the ID string of the entry order.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int  | Trade number (first trade is 0) |

**Returns:** str

```python
eid: str = strategy.closedtrades.entry_id(0)  # "buy_signal_1"
```

### entry_comment()

Returns the comment message attached to the entry order, or NA if no comment was provided.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int  | Trade number (first trade is 0) |

**Returns:** str

```python
msg: str = strategy.closedtrades.entry_comment(0)  # "Long signal detected"
```

## Exit Information

### exit_bar_index()

Returns the bar index at which the trade was exited.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int  | Trade number (first trade is 0) |

**Returns:** int

```python
idx: int = strategy.closedtrades.exit_bar_index(0)  # 20
```

### exit_time()

Returns the UNIX time (in milliseconds) when the trade was exited.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int  | Trade number (first trade is 0) |

**Returns:** int

```python
ts: int = strategy.closedtrades.exit_time(0)  # 1672617600000
```

### exit_price()

Returns the price at which the trade was exited.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int  | Trade number (first trade is 0) |

**Returns:** float

```python
px: float = strategy.closedtrades.exit_price(0)  # 102.3
```

### exit_id()

Returns the ID string of the exit order.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int  | Trade number (first trade is 0) |

**Returns:** str

```python
eid: str = strategy.closedtrades.exit_id(0)  # "close_buy_1"
```

### exit_comment()

Returns the comment message attached to the exit order, or NA if no comment was provided.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int  | Trade number (first trade is 0) |

**Returns:** str

```python
msg: str = strategy.closedtrades.exit_comment(0)  # "Stop loss hit"
```

## Trade Metrics

### profit()

Returns the profit or loss of the closed trade, expressed in the strategy's account currency. Commissions are deducted from the result.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int  | Trade number (first trade is 0) |

**Returns:** float

```python
pl: float = strategy.closedtrades.profit(0)  # 1.8 (positive for gain, negative for loss)
```

### profit_percent()

Returns the profit or loss of the closed trade as a percentage of the entry price.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int  | Trade number (first trade is 0) |

**Returns:** float

```python
pct: float = strategy.closedtrades.profit_percent(0)  # 1.79
```

### commission()

Returns the total commissions (entry + exit fees) paid for the trade, expressed in the account currency.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int  | Trade number (first trade is 0) |

**Returns:** float

```python
fee: float = strategy.closedtrades.commission(0)  # 0.10
```

### size()

Returns the trade size. Positive values indicate a long position, negative values indicate a short position.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int  | Trade number (first trade is 0) |

**Returns:** float

```python
sz: float = strategy.closedtrades.size(0)  # 1.0 (long) or -1.0 (short)
```

## Drawdown & Runup

### max_drawdown()

Returns the maximum drawdown during the trade, expressed in account currency. This is the largest loss that occurred between entry and exit.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int  | Trade number (first trade is 0) |

**Returns:** float

```python
dd: float = strategy.closedtrades.max_drawdown(0)  # 1.5
```

### max_drawdown_percent()

Returns the maximum drawdown as a percentage of the entry price.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int  | Trade number (first trade is 0) |

**Returns:** float

```python
ddpct: float = strategy.closedtrades.max_drawdown_percent(0)  # 1.49
```

### max_runup()

Returns the maximum runup during the trade, expressed in account currency. This is the largest unrealized profit that occurred between entry and exit.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int  | Trade number (first trade is 0) |

**Returns:** float

```python
ru: float = strategy.closedtrades.max_runup(0)  # 3.2
```

### max_runup_percent()

Returns the maximum runup as a percentage of the entry price.

| Parameter | Type | Description |
|-----------|------|-------------|
| trade_num | int  | Trade number (first trade is 0) |

**Returns:** float

```python
rupct: float = strategy.closedtrades.max_runup_percent(0)  # 3.18
```

## Compatibility

- `strategy.closedtrades.first_index` — Not available in PyneCore. Access closed trades by iterating from trade number 0 onwards until a function returns NA.