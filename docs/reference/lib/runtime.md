<!--
---
weight: 451
title: "runtime"
description: "Runtime error handling"
icon: "error"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["runtime", "library", "reference"]
---
-->

# runtime

The `runtime` namespace provides error handling capabilities for PyneCore scripts. Use these functions to stop script execution and report error conditions.

## Quick Example

```python
from pynecore.lib import close, runtime, ta, script

@script.indicator(title="Runtime Error Example")
def main():
    sma: float = ta.sma(close, 20)
    
    if sma == 0:
        runtime.error("Invalid moving average calculation")
    
    if close < 0:
        runtime.error("Price data is corrupted")
```

## Functions

### error()

Terminates script execution with an error message.

| Parameter | Type | Description |
|-----------|------|-------------|
| message | str | Error message to display |

**Return type:** void (raises RuntimeError)

**Example:**
```python
if bar_index == 0:
    runtime.error("Script requires historical data")
```

## Compatibility

All functions in the `runtime` namespace are fully implemented.