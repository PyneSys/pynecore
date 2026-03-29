<!--
---
weight: 442
title: "barstate"
description: "Bar state flags — first bar, last bar, new bar, etc."
icon: "toggle_on"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["barstate", "library", "reference"]
---
-->

# barstate

The `barstate` namespace provides boolean flags indicating the current bar's position in the historical data and real-time status. Use these to identify first/last bars, detect new bars, or check whether the script is running on confirmed, historical, or real-time data.

## Quick Example

```python
from pynecore.lib import (
    close, high, label, bar_index, barstate, script
)

@script.indicator(title="Bar State Detector", overlay=True)
def main():
    # Mark the first and last bars
    if barstate.isfirst:
        label.new(bar_index, high, "First")
    
    if barstate.islast:
        label.new(bar_index, high, "Last")
    
    # Log bar state information
    if barstate.isconfirmed:
        # This is the final update for the current bar
        pass
    
    if barstate.ishistory:
        # We are on a historical bar
        pass
```

## Variables

### barstate.isfirst

Returns `true` if the current bar is the first bar in the dataset, `false` otherwise.

**Type**: `bool`

```python
if barstate.isfirst:
    label.new(bar_index, high, "Start")
```

### barstate.islast

Returns `true` if the current bar is the last bar in the dataset, `false` otherwise. For real-time bars, this is always `true`.

**Type**: `bool`

```python
if barstate.islast:
    label.new(bar_index, high, "End")
```

### barstate.isconfirmed

Returns `true` if the script is calculating the final update of the current bar (at bar close). The next script execution will be on a new bar.

**Type**: `bool`

```python
if barstate.isconfirmed:
    # Execute only once per bar, at close
    pass
```

### barstate.ishistory

Returns `true` if the current bar is a historical bar (not real-time), `false` otherwise.

**Type**: `bool`

```python
if barstate.ishistory:
    # Processing historical data
    pass
```

### barstate.islastconfirmedhistory

Returns `true` if the script is executing on the dataset's last bar when the market is closed, or on the bar immediately before the real-time bar when the market is open.

**Type**: `bool`

```python
if barstate.islastconfirmedhistory:
    # Last historical bar before real-time data begins
    pass
```

### barstate.isnew

Returns `true` if the script is currently calculating on a new bar, `false` otherwise.

**Type**: `bool`

```python
if barstate.isnew:
    # New bar detected
    pass
```

### barstate.isrealtime

Returns `true` if the current bar is a real-time bar (live market data), `false` otherwise.

**Type**: `bool`

```python
if barstate.isrealtime:
    # Real-time bar processing
    pass
```

## Compatibility Notes

**Current Limitations:**

- `barstate.isconfirmed` — Always returns `true`; bar magnifier support not yet implemented.
- `barstate.ishistory` — Always returns `true`; live trading not yet supported.
- `barstate.islastconfirmedhistory` — Always returns `false`; live trading not yet supported.
- `barstate.isnew` — Always returns `false`; bar magnifier support not yet implemented.
- `barstate.isrealtime` — Always returns `false`; live trading not yet supported.

For backtesting on historical datasets, `isfirst`, `islast`, and `isconfirmed` work as documented. Live market support is planned for a future release.