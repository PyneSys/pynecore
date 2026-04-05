<!--
---
weight: 480
title: "scale"
description: "Scale type constants for indicators"
icon: "straighten"
date: "2026-03-28"
lastmod: "2026-03-28"
draft: false
toc: true
categories: ["Reference", "Library"]
tags: ["scale", "library", "reference"]
---
-->

# scale

The `scale` namespace provides constants that control which price scale an indicator is attached to. These constants are passed to the `scale` parameter of `@script.indicator()`.

## Quick Example

```python
from pynecore.lib import script, close, ta, scale

@script.indicator(title="Volume Oscillator", scale=scale.left)
def main():
    fast: float = ta.ema(close, 5)
    slow: float = ta.ema(close, 20)
    osc: float = fast - slow
```

## Constants

| Constant      | Description                                                                                        |
|---------------|----------------------------------------------------------------------------------------------------|
| `scale.left`  | Attaches the indicator to the left price scale.                                                    |
| `scale.right` | Attaches the indicator to the right price scale.                                                   |
| `scale.none`  | Displays the indicator without a price scale ("No Scale" mode). Requires `overlay=True`.           |

### Usage

Pass a `scale` constant to `@script.indicator()`:

```python
@script.indicator(title="My Overlay", overlay=True, scale=scale.none)
```