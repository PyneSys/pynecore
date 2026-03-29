<!--
---
weight: 404
title: "Input Functions"
description: "User-configurable script parameters in PyneCore"
icon: "tune"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Language"]
tags: ["input", "parameters", "configuration", "settings"]
---
-->

# Input Functions

Input functions allow Pine Script users to configure script parameters through a settings
dialog. In PyneCore, the PyneComp compiler transforms these into parameter declarations with
default values.

## How Inputs Work in PyneCore

In TradingView, `input.*()` functions create interactive UI controls. Since PyneCore runs
offline without a UI, input values are resolved at compile time to their default values.
Future versions may support external configuration files for input overrides.

## Available Input Functions

### input()

Generic input that infers the type from the default value.

```
length = input(14, "RSI Length")
```

**Parameters:**
- `defval` — Default value (determines the input type)
- `title` — Display name
- `tooltip` — Tooltip text
- `inline` — Group identifier for horizontal layout
- `group` — Settings group name

### input.int()

Integer input with optional min/max bounds and step.

```
length = input.int(14, "Length", minval=1, maxval=200, step=1)
```

### input.float()

Float input with optional min/max bounds and step.

```
mult = input.float(2.0, "Multiplier", minval=0.1, maxval=10.0, step=0.1)
```

### input.bool()

Boolean toggle input.

```
showSignals = input.bool(true, "Show Signals")
```

### input.string()

String input with optional dropdown options.

```
maType = input.string("SMA", "MA Type", options=["SMA", "EMA", "WMA"])
```

### input.color()

Color picker input.

```
bullColor = input.color(color.green, "Bullish Color")
```

### input.enum()

Enum value selector. Provides a dropdown of the enum's values.

```
enum Direction
    Long
    Short

dir = input.enum(Direction.Long, "Direction")
```

### input.timeframe()

Timeframe selector.

```
tf = input.timeframe("D", "Resolution")
```

### input.symbol()

Symbol/ticker selector.

```
sym = input.symbol("AAPL", "Symbol")
```

### input.session()

Session time range selector.

```
sess = input.session("0930-1600", "Trading Session")
```

### input.source()

Source value selector (open, high, low, close, hl2, etc.).

```
src = input.source(close, "Source")
```

### input.time()

Timestamp input with calendar picker.

```
startDate = input.time(timestamp("2020-01-01"), "Start Date")
```

### input.text_area()

Multi-line text input.

```
notes = input.text_area("", "Notes")
```

### input.price()

Price level input with chart interaction.

```
level = input.price(0.0, "Entry Price", confirm=true)
```

## Common Parameters

All input functions share these optional parameters:

| Parameter  | Type   | Description                          |
|-----------|--------|--------------------------------------|
| `title`   | string | Display name in the settings dialog  |
| `tooltip` | string | Hover text for the input             |
| `inline`  | string | Groups inputs horizontally           |
| `group`   | string | Section name in settings             |
| `confirm` | bool   | Requires user confirmation on chart  |
