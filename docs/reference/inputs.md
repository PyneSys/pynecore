<!--
---
weight: 420
title: "Input Functions"
description: "User-configurable script parameters in PyneCore"
icon: "tune"
date: "2026-03-28"
lastmod: "2026-04-05"
draft: false
toc: true
categories: ["Reference"]
tags: ["input", "parameters", "configuration", "settings", "toml"]
---
-->

# Input Functions

Input functions define configurable parameters for your script. They are used as default values
for `main()` function parameters. At runtime, values can be overridden via the auto-generated
TOML configuration file.

## How Inputs Work

Inputs are declared as `main()` parameters with `input.*()` as default values:

```python
from pynecore.lib import script, input, close, ta, strategy

@script.strategy("MA Crossover", overlay=True)
def main(
    fast_length: int = input.int(9, title="Fast MA Length", minval=1),
    slow_length: int = input.int(21, title="Slow MA Length", minval=1),
    ma_type: str = input.string("SMA", title="MA Type", options=("SMA", "EMA", "WMA")),
):
    fast_ma = ta.sma(close, fast_length) if ma_type == "SMA" else ta.ema(close, fast_length)
    slow_ma = ta.sma(close, slow_length) if ma_type == "SMA" else ta.ema(close, slow_length)
    # ...
```

When you run a script, PyneCore generates a `.toml` file with the same base name. Edit this file
to change parameter values without modifying code. See [Strategy Development](../strategy.md) for
details on the TOML configuration workflow.

## Available Functions

### input()

Generic input — infers the type from the default value.

```python
length: int = input(14, "RSI Length")
```

### input.int()

Integer input with optional bounds and step.

```python
length: int = input.int(14, title="Length", minval=1, maxval=200, step=1)
```

**Parameters:** `defval`, `title`, `minval`, `maxval`, `step`, `tooltip`, `inline`, `group`,
`confirm`, `options`, `display`

### input.float()

Float input with optional bounds and step.

```python
mult: float = input.float(2.0, title="Multiplier", minval=0.1, maxval=10.0, step=0.1)
```

**Parameters:** `defval`, `title`, `minval`, `maxval`, `step`, `tooltip`, `inline`, `group`,
`confirm`, `options`, `display`

### input.bool()

Boolean input.

```python
show_signals: bool = input.bool(True, title="Show Signals")
```

**Parameters:** `defval`, `title`, `tooltip`, `inline`, `group`, `confirm`, `display`

### input.string()

String input with optional dropdown options.

```python
ma_type: str = input.string("SMA", title="MA Type", options=("SMA", "EMA", "WMA"))
```

**Parameters:** `defval`, `title`, `tooltip`, `inline`, `group`, `confirm`, `options`, `display`

### input.color()

Color input.

```python
from pynecore.lib import color

bull_color: Color = input.color(color.green, title="Bullish Color")
```

**Parameters:** `defval`, `title`, `tooltip`, `inline`, `group`, `confirm`, `display`

### input.enum()

Enum value selector.

```python
from enum import StrEnum as Enum

class Direction(Enum):
    Long = "Long"
    Short = "Short"

dir: Direction = input.enum(Direction.Long, title="Direction")
```

**Parameters:** `defval`, `title`, `tooltip`, `inline`, `group`, `confirm`, `options`, `display`

### input.source()

Source value selector (open, high, low, close, hl2, hlc3, ohlc4, etc.).

```python
src = input.source(close, title="Source")
```

**Parameters:** `defval`, `title`, `tooltip`, `inline`, `group`, `confirm`, `display`

### input.timeframe()

Timeframe selector.

```python
tf: str = input.timeframe("D", title="Resolution")
```

### input.symbol()

Symbol/ticker selector.

```python
sym: str = input.symbol("AAPL", title="Symbol")
```

### input.session()

Session time range selector.

```python
sess: str = input.session("0930-1600", title="Trading Session")
```

### input.time()

Timestamp input.

```python
start_date: int = input.time(timestamp("2020-01-01"), title="Start Date")
```

### input.text_area()

Multi-line text input.

```python
notes: str = input.text_area("", title="Notes")
```

### input.price()

Price level input.

```python
level: float = input.price(0.0, title="Entry Price", confirm=True)
```

## Common Parameters

All input functions accept these optional parameters:

| Parameter | Type | Description                                    |
|-----------|------|------------------------------------------------|
| `title`   | str  | Display name for the input                     |
| `tooltip` | str  | Hover text                                     |
| `inline`  | str  | Groups inputs horizontally when sharing a value |
| `group`   | str  | Section name in settings                       |
| `confirm` | bool | Requires user confirmation                     |
| `display` | int  | Controls where the input info is displayed     |
