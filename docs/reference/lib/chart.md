<!--
---
weight: 443
title: "chart"
description: "Chart properties — type, colors, visible range"
icon: "candlestick_chart"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["chart", "library", "reference"]
---
-->

# chart

The `chart` namespace provides information about the current chart's appearance, type, and visible range. Use these properties to detect which chart type is active, access the chart's color scheme, and determine which bars are visible in the chart viewport.

## Quick Example

```python
from pynecore.lib import chart, label, bar_index, high, script

@script.indicator(title="Chart Info", overlay=True)
def main():
    # Detect chart type
    is_standard: bool = chart.is_standard
    is_renko: bool = chart.is_renko
    is_kagi: bool = chart.is_kagi
    
    # Get the time range currently visible on the chart
    left_time: int = chart.left_visible_bar_time
    right_time: int = chart.right_visible_bar_time
    
    if is_standard:
        label.new(bar_index, high, "Standard Chart")
    elif is_renko:
        label.new(bar_index, high, "Renko Chart")
```

## Variables

### Colors

**chart.bg_color**
- Type: `Color`
- The background color of the chart from Chart settings → Appearance. When a gradient is selected, returns the color at the middle point of the gradient.

**chart.fg_color**
- Type: `Color`
- A color that contrasts well with the background, suitable for drawing text and UI elements that remain visible on any background.

### Chart Type

**chart.is_standard**
- Type: `bool`
- `True` if the chart displays standard OHLC candles, `False` if an alternative chart type is active.

**chart.is_renko**
- Type: `bool`
- `True` if the chart type is Renko, `False` otherwise.

**chart.is_kagi**
- Type: `bool`
- `True` if the chart type is Kagi, `False` otherwise.

**chart.is_linebreak**
- Type: `bool`
- `True` if the chart type is Line break, `False` otherwise.

**chart.is_pnf**
- Type: `bool`
- `True` if the chart type is Point & figure, `False` otherwise.

**chart.is_range**
- Type: `bool`
- `True` if the chart type is Range, `False` otherwise.

**chart.is_heikinashi**
- Type: `bool`
- `True` if the chart type is Heikin Ashi, `False` otherwise.

### Visible Range

**chart.left_visible_bar_time**
- Type: `int`
- The timestamp (in milliseconds since epoch) of the leftmost "visible" bar. In PyneCore this is a heuristic: `current_bar_time - 20 × timeframe_seconds`. There is no real viewport — the value approximates a 20-bar visible window ending at the current bar.

**chart.right_visible_bar_time**
- Type: `int`
- The timestamp (in milliseconds since epoch) of the rightmost "visible" bar. In PyneCore this is always the current bar's timestamp.

## Compatibility

Chart type properties (`is_standard`, `is_renko`, etc.) are fully supported. Visible range properties (`left_visible_bar_time`, `right_visible_bar_time`) use a static heuristic (20-bar window at the current bar) since PyneCore has no graphical viewport. Scripts that rely on precise visible range detection may behave differently than on TradingView.