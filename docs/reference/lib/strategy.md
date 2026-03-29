<!--
---
weight: 429
title: "strategy"
description: "Strategy order management — entry, exit, close, and order functions"
icon: "trending_up"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["strategy", "library", "reference"]
---
-->

# strategy

Manage entries, exits, and track position metrics for backtesting strategies. The strategy namespace provides order creation, cancellation, and real-time P&L tracking. Use with `@script.strategy()` decorator to enable position management.

## Quick Example

```python
from pynecore.lib import (
    close, high, low, strategy, ta, bar_index, script
)
from pynecore.types import Persistent

@script.strategy(title="Simple Strategy", initial_capital=10000)
def main():
    sma20: Persistent[float] = ta.sma(close, 20)
    
    if bar_index == 20:
        strategy.entry("long", strategy.long, qty=1)
    
    if ta.crossunder(close, sma20):
        strategy.close("long", comment="Exit on cross below")
    
    # Check performance
    pnl: float = strategy.netprofit
    position: float = strategy.position_size
```

## Functions

### strategy.entry()

Create a new order to open or add to a position. Modifies existing unfilled orders with the same id.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | str | Order identifier |
| direction | int | Trade direction: `strategy.long` or `strategy.short` |
| qty | float \| None | Quantity in units (optional, uses strategy default if None) |
| limit | float \| None | Limit price for entry (optional) |
| stop | float \| None | Stop price for entry (optional) |
| oca_name | str \| None | One-Cancels-All group identifier (optional) |
| oca_type | int | OCA behavior type (optional) |
| comment | str \| None | Order comment (optional) |
| alert_message | str \| None | Alert message text (optional) |

Returns: `None`

```python
strategy.entry("long_1", strategy.long, qty=2.5)
strategy.entry("entry_limit", strategy.long, qty=1, limit=100.5)
```

### strategy.exit()

Create price-based exit orders (take-profit, stop-loss, or trailing stop). Modifies existing unfilled orders with the same id.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | str | Exit order identifier |
| from_entry | str \| None | Entry id to exit (optional, exits from any entry if None) |
| qty | float \| None | Exit quantity (optional) |
| qty_percent | float \| None | Exit as % of position (optional) |
| profit | float \| None | Take-profit in currency units (optional) |
| limit | float \| None | Limit price for take-profit (optional) |
| loss | float \| None | Stop-loss in currency units (optional) |
| stop | float \| None | Stop price for stop-loss (optional) |
| trail_price | float \| None | Trailing stop price offset (optional) |
| trail_points | float \| None | Trailing stop in points (optional) |
| trail_offset | float \| None | Trailing offset (optional) |
| oca_name | str \| None | OCA group identifier (optional) |
| comment | str \| None | Order comment (optional) |
| comment_profit | str \| None | TP comment (optional) |
| comment_loss | str \| None | SL comment (optional) |
| comment_trailing | str \| None | Trailing stop comment (optional) |
| alert_message | str \| None | Alert text (optional) |
| alert_profit | str \| None | TP alert (optional) |
| alert_loss | str \| None | SL alert (optional) |
| alert_trailing | str \| None | Trailing alert (optional) |
| disable_alert | bool | Suppress alerts if True (optional) |

Returns: `None`

```python
strategy.exit("tp_sl", qty_percent=100, profit=500, loss=200)
strategy.exit("trail", trail_points=50, comment="Trailing stop")
```

### strategy.close()

Exit a position opened by entries with a specific id. Closes the position immediately at market price.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | str | Entry id to close |
| comment | str \| None | Order comment (optional) |
| qty | float \| None | Partial close quantity (optional) |
| qty_percent | float \| None | Partial close as % of position (optional) |
| alert_message | str \| None | Alert text (optional) |
| immediately | bool | Close at market immediately (optional) |

Returns: `None`

```python
strategy.close("long_1", comment="Exit signal")
strategy.close("entry_a", qty_percent=50)
```

### strategy.close_all()

Close the entire open position immediately at market price, regardless of entry ids.

| Parameter | Type | Description |
|-----------|------|-------------|
| comment | str \| None | Order comment (optional) |
| alert_message | str \| None | Alert text (optional) |
| immediately | bool | Close immediately (optional) |

Returns: `None`

```python
strategy.close_all(comment="Exit all positions")
```

### strategy.order()

Create a new order to open, add to, or exit a position. Modifies existing unfilled orders with the same id.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | str | Order identifier |
| direction | int | Trade direction: `strategy.long` or `strategy.short` |
| qty | float \| None | Quantity in units (optional) |
| limit | float \| None | Limit price (optional) |
| stop | float \| None | Stop price (optional) |
| oca_name | str \| None | OCA group identifier (optional) |
| oca_type | int | OCA behavior type (optional) |
| comment | str \| None | Order comment (optional) |
| alert_message | str \| None | Alert text (optional) |
| disable_alert | bool | Suppress alerts if True (optional) |

Returns: `None`

```python
strategy.order("hedge", strategy.short, qty=1, limit=99.5)
```

### strategy.cancel()

Cancel a pending or unfilled order by id. Cancels all orders sharing the same id.

| Parameter | Type | Description |
|-----------|------|-------------|
| id | str | Order identifier to cancel |

Returns: `None`

```python
strategy.cancel("limit_order")
```

### strategy.cancel_all()

Cancel all pending or unfilled orders regardless of id.

Returns: `None`

```python
strategy.cancel_all()
```

## Variables

| Name | Type | Description |
|------|------|-------------|
| position_size | float | Current position size (> 0 = long, < 0 = short, 0 = flat). Returns `NaN` if no position. |
| position_avg_price | float | Average entry price of current position. Returns `NaN` if flat. |
| opentrades | int | Count of open (unfilled entry) positions. |
| openprofit | float | Current unrealized P&L for all open positions in currency units. |
| closedtrades | int | Total count of closed trades for the entire trading range. |
| wintrades | int | Count of winning trades. |
| losstrades | int | Count of losing trades. |
| eventrades | int | Count of breakeven trades. |
| netprofit | float | Total realized P&L for all closed trades in currency units. |
| grossprofit | float | Total P&L from winning trades in currency units. |
| grossloss | float | Total P&L from losing trades in currency units. |
| equity | float | Current equity = initial_capital + netprofit + openprofit. |
| max_drawdown | float | Maximum equity drawdown from peak in currency units. |
| max_drawdown_percent | float | Maximum drawdown as % of initial capital. |
| max_runup | float | Maximum equity run-up from entry in currency units. |
| max_runup_percent | float | Maximum run-up as % of initial capital. |
| initial_capital | float | Initial capital set in strategy properties. |

## Constants

| Name | Type | Description |
|------|------|-------------|
| long | int | Direction constant for `strategy.entry()` and `strategy.order()`. Creates a buy/long position. |
| short | int | Direction constant for `strategy.entry()` and `strategy.order()`. Creates a sell/short position. |
| fixed | QtyType | Quantity type for strategy properties. Fixed number of units per entry. |
| cash | QtyType | Quantity type for strategy properties. Fixed currency amount per entry. |
| percent_of_equity | QtyType | Quantity type for strategy properties. Percentage of equity per entry. |

## Compatibility

**Not yet implemented:**
- `strategy.account_currency` — Account currency from properties
- `strategy.convert_to_account()` — Currency conversion to account currency
- `strategy.convert_to_symbol()` — Currency conversion to symbol currency
- `strategy.default_entry_qty()` — Calculate default order quantity
- `strategy.margin_liquidation_price` — Margin call liquidation price
- `strategy.max_contracts_held_*` — Maximum contract tracking variables
- All percentage variables: `strategy.netprofit_percent`, `strategy.grossprofit_percent`, `strategy.grossloss_percent`, `strategy.openprofit_percent`, `strategy.avg_trade_percent`, `strategy.avg_winning_trade_percent`, `strategy.avg_losing_trade_percent`
- `strategy.position_entry_name` — Entry order name for current position